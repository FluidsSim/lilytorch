"""
Unified BDIM handler for FARMS <-> lilytorch coupling.

Supports both 2-D and 3-D simulations.  Dimensionality is auto-detected
from the presence of ``Nz`` in ``pars['solver']``.

All simulation-specific hyperparameters are read from the ``bdim_yaml``
config dict, so a single class covers every animat (1guilla, pleurodeles,
zebrafish, salamander, ...).

New config keys (add to ``bdim_yaml``)::

    solver.dtype                 : "float32" | "float64"   (default "float32")
    solver.zero_pressure_inside  : bool                    (default False)
    solver.force_method          : "eulerian" | "lagrangian"   (default "eulerian";
                                   legacy "method1"/"method2" accepted with a DeprecationWarning)
    body.force_scaling           : "auto" | float          (default "auto")
    body.contour_mask            : bool                    (default False)
    physics.solref               : [float, float] | null   (default null)
    physics.solimp               : [float, float, float, float, float] | null
                                   (default null)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from lilytorch.src.solver import FluidSolver
from lilytorch.src.two_phase_solver import TwoPhaseSolver
from lilytorch.src.free_surface_solver import FreeSurfaceSolver
from lilytorch.src.body import (rotate_grid_2d, rotate_grid_3d,
                                _rotate_grid_3d_compiled)

import torch


_FS_FREE_AFTER_FORCES_3D = {
    'xstress_tensor': None, 'ystress_tensor': None, 'zstress_tensor': None,
    'pforce_x': None, 'pforce_y': None, 'pforce_z': None,
}


class _MujocoCheckpoint:
    """Save/restore the full MuJoCo integration state for sub-iteration.

    ``mjSTATE_INTEGRATION`` bundles qpos, qvel, act, time, qacc_warmstart,
    ctrl, qfrc_applied, xfrc_applied, mocap and eq_active — everything
    needed to reproduce an integration step deterministically across a
    checkpoint/restore.  Used by the implicit (strongly-coupled) step to
    run throwaway prediction integrations that are each undone before the
    next coupling sweep.  See STRONG_COUPLING_FARMS_DESIGN.md §5.
    """

    def __init__(self, physics):
        import mujoco
        self._mj = mujoco
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.m = physics.model.ptr
        self.d = physics.data.ptr
        self.n = mujoco.mj_stateSize(self.m, self.spec)
        self._buf = np.zeros(self.n, dtype=np.float64)

    def save(self):
        self._mj.mj_getState(self.m, self.d, self._buf, self.spec)
        return self._buf.copy()

    def restore(self, state):
        self._mj.mj_setState(self.m, self.d, state, self.spec)
        self._mj.mj_forward(self.m, self.d)

    def integrate(self, nstep=1):
        """Advance ``nstep`` MuJoCo steps in place, then refresh derived
        quantities (xpos/xipos/xquat/sensordata) for the new state so the
        fluid's ``physics`` pose source reads the predicted pose."""
        self._mj.mj_step(self.m, self.d, nstep)
        self._mj.mj_forward(self.m, self.d)


class BDIMhandler:
    """Unified boundary-data immersion method handler (2-D and 3-D)."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, yaml_file, data, physics, dtype=None):

        self.pars = yaml_file

        # ---- auto-detect dimensionality ----
        self.ndim = 3 if "Nz" in self.pars["solver"] else 2

        # ---- dtype ----
        # Resolution rule (single source of truth: solver.py):
        #   explicit ``dtype=`` kwarg  >  ``solver.dtype`` from YAML  >  float32.
        # We intentionally delegate parsing so that strings like "double"
        # / "single" / "float64" / "float32" all behave the same here as
        # they do in :class:`FluidSolver`.

        # ---- create fluid solver ----
        # Auto-select the two-phase (water + real air) solver when the config
        # carries a ``solver.two_phase`` block, or the one-fluid free-surface
        # solver when ``solver.free_surface`` is present (which also requires
        # ``solver.two_phase`` for the VOF interface tracking).
        self._two_phase = self.pars["solver"].get("two_phase") is not None
        self._free_surface = self.pars["solver"].get("free_surface") is not None
        if self._free_surface:
            _SolverCls = FreeSurfaceSolver
        elif self._two_phase:
            _SolverCls = TwoPhaseSolver
        else:
            _SolverCls = FluidSolver
        self.fluid_solver = _SolverCls(
            self.pars,
            dtype=dtype,
            custom_update=True,
            compute_forces=True,
        )
        self.device = self.fluid_solver.device
        self.dtype = self.fluid_solver.dtype
        self.dtype_np = np.float64 if self.dtype == torch.float64 else np.float32

        # used for 2D contour-mask neighbor only
        self._prev_body_index = ()
        self._next_body_index = ()

        # ---- bookkeeping ----
        self.data = data          # list[AnimatData] from FARMS
        self.iteration = 0

        # ---- pose source for gather_data (strong-coupling support) ----
        # "sensors": read link poses/velocities from the FARMS AnimatData
        #   sensor buffers at ``iteration`` (default; explicit coupling).
        # "physics": read them live from ``physics.data`` (xpos/xipos/xquat
        #   + framelinvel/frameangvel sensordata), mirroring
        #   ``farms_mujoco...physics.physics2data``.  Used by the implicit
        #   (strongly-coupled) loop so that an internal ``physics.step``
        #   prediction is immediately visible to the fluid without a
        #   ``task.update_sensors`` round-trip.  See STRONG_COUPLING_FARMS_DESIGN.md §14.
        self._pose_source = "sensors"
        self._physics = physics    # same Physics instance reused each step
        self._task = None          # set by the implicit step / equivalence test




        # ---- initialise u0 from inlet BC ----
        u_inlet = self.fluid_solver.adv_diff_solver.BC_values_u[1]
        if self.ndim == 3:
            gs = self.fluid_solver.grid_shape
            self.fluid_solver.u0 = u_inlet * torch.ones(
                gs, device=self.device, dtype=self.dtype
            )
        else:
            self.fluid_solver.u0 = u_inlet * torch.ones(
                (self.fluid_solver.nx, self.fluid_solver.ny),
                device=self.device, dtype=self.dtype,
            )

        # ---- axes (derived from ndim and 2d_plane) ----
        # Simulated velocity components (u,v[,w]) — indexes the unified
        # _update_python per-axis field lists (Step 6 unification).
        self._sim_axes = list(range(self.ndim))
        if self.ndim == 3:
            self.lin_axes = [0, 1, 2]
            self.ang_axes = [0, 1, 2]
            self._2d_plane        = None
            self._2d_ang_ax       = None
            self._2d_force_axes   = None
            self._2d_has_buoyancy = False
        else:
            _2d_plane = self.pars.get("body", {}).get("2d_plane", "xy")
            self._2d_plane = _2d_plane
            if _2d_plane == "xz":
                # MuJoCo (x, z) → fluid (x, y): sphere falling under gravity in z
                self.lin_axes         = [0, 2]
                self.ang_axes         = [1]
                self._2d_ang_ax       = 1          # rotation around MuJoCo y-axis
                self._2d_force_axes   = (0, 2, 4)  # xfrc: fx, fz, ty
                self._2d_has_buoyancy = True
            else:  # "xy" — default
                self.lin_axes         = [0, 1]
                self.ang_axes         = [2]
                self._2d_ang_ax       = 2          # rotation around MuJoCo z-axis
                self._2d_force_axes   = (0, 1, 5)  # xfrc: fx, fy, tz
                self._2d_has_buoyancy = False

        # ---- force scaling (config or auto) ----
        fs_cfg = self.pars.get("body", {}).get("force_scaling", "auto")
        if fs_cfg == "auto":
            if self.ndim == 2:
                comp = self.fluid_solver.composite_body
                try:
                    self.force_scaling = torch.tensor(
                        [np.diff(body.bb[2])[0] for body in comp.bodies],
                        device=self.device, dtype=self.dtype
                    )
                except Exception:
                    self.force_scaling = 1.0
            else:
                self.force_scaling = 1.0
        else:
            self.force_scaling = float(fs_cfg)

        # ---- temporal under-relaxation of force feedback (FSI stability) ----
        # F_applied^{n+1} = β · F_lag^{n+1} + (1-β) · F_applied^{n}
        # β=1.0 (default): no filtering, raw force applied each step.
        # β<1.0: low-pass filter; damps the explicit-coupling oscillation
        # (e.g. the 20Hz / 5-iter coupling-lag mode seen in salamander
        # gamepad with Lagrangian forces) while preserving DC / time-
        # average physical force. Lives at the coupling boundary; the
        # raw per-step forces in FluidSolver are unchanged for analysis.
        self.force_relaxation = float(
            self.pars.get("body", {}).get("force_relaxation", 1.0))
        self._fr_lin_prev = None   # numpy (D, B), set on first call
        self._fr_ang_prev = None   # numpy (Nt, B)

        # ---- strong (implicit) coupling configuration ----
        # body.coupling:
        #   scheme: "explicit" (default) | "implicit"
        #   accelerator: "iqn-ils" (default) | "aitken" | "constant"
        #   reuse: int (IQN-ILS time-window reuse, default 2)
        #   tol: relative interface-residual tolerance (default 1e-4)
        #   max_iter: max coupling sweeps per step (default 30)
        #   predict_substeps: MuJoCo steps per prediction (default 1; must
        #     match the runtime integration cb_sub_steps*num_sub_steps).
        # When scheme == "implicit", force_relaxation is ignored (the
        # quasi-Newton fixed point replaces the constant low-pass).  See
        # STRONG_COUPLING_FARMS_DESIGN.md.
        _cpl = self.pars.get("body", {}).get("coupling", {}) or {}
        self._coupling_scheme = str(_cpl.get("scheme", "explicit")).lower()
        self.tol = float(_cpl.get("tol", 1e-4))
        self.max_iter = int(_cpl.get("max_iter", 30))
        self._predict_nstep = int(_cpl.get("predict_substeps", 1))
        # Robustness: reject a candidate coupling-force whose norm exceeds
        # ``force_bound`` x (running max load magnitude).  Prevents a runaway
        # quasi-Newton iterate from being applied to MuJoCo and blowing up.
        self._impl_force_bound = float(_cpl.get("force_bound", 100.0))
        self.last_iters = 0
        self.last_residual = 0.0
        self._mj_ckpt = None
        self._implicit_prev = None
        self.accelerator = None
        if self._coupling_scheme == "implicit":
            from lilytorch.integration.fsi_coupling import make_accelerator
            self.accelerator = make_accelerator(
                _cpl.get("accelerator", "iqn-ils"),
                **({"reuse": int(_cpl.get("reuse", 2))}
                   if str(_cpl.get("accelerator", "iqn-ils")).lower().startswith("iqn")
                   or str(_cpl.get("accelerator", "iqn-ils")).lower() in ("qn", "quasi-newton")
                   else {}),
            )
            print(f"[BDIMhandler] strong (implicit) coupling: "
                  f"accelerator={_cpl.get('accelerator', 'iqn-ils')}, "
                  f"tol={self.tol}, max_iter={self.max_iter}")

        # ---- smooth body-velocity blend in the overlap band ----
        # With convexify (or otherwise overlapping links), the running-min
        # SDF union hard-switches the imposed solid velocity at the seam
        # between two links, injecting a grid-scale divergence -> pressure
        # spike -> explicit-coupling blow-up.  When ``body_velocity_blend``
        # is on, the imposed face velocity in the band becomes an
        # SDF-weighted average  v = Σ w_i v_i / Σ w_i ,  w_i = σ(-φ_i/ε_w),
        # which is continuous across the seam (equals (v_A+v_B)/2 where
        # φ_A=φ_B) and reduces to v_i exactly for a single (non-overlapping)
        # body.  ``ε_w`` is given in cells via ``body_velocity_blend_eps_cells``.
        # None / 0  -> legacy hard running-min winner-take-all.
        _blend_cells = self.pars.get("body", {}).get(
            "body_velocity_blend_eps_cells",
            self.pars["solver"].get("body_velocity_blend_eps_cells", None))
        self._blend_eps_cells = (
            float(_blend_cells) if _blend_cells else None)
        self._blend_eps = None      # set per-step from grid spacing h
        self._blend_den = None      # lazily-allocated denominator buffers

        # ---- densities ----
        self.rho_fluid = self.pars["solver"]["rho"]

        # ---- toggles ----
        self.zero_pressure_inside = self.pars["solver"].get(
            "zero_pressure_inside", False
        )
        self.contour_mask = self.pars.get("body", {}).get("contour_mask", False)
        self.force_method = self.pars["solver"].get("force_method", "eulerian")

        # Resolve legacy aliases ("method1" → "lagrangian", "method2" → "eulerian").
        # Actual dispatch is handled by FluidSolver.step_() via FluidSolver.force_method.
        _fm_aliases = {"method1": "lagrangian", "method2": "eulerian"}
        if self.force_method in _fm_aliases:
            import warnings
            warnings.warn(
                f"force_method={self.force_method!r} is deprecated; use "
                f"{_fm_aliases[self.force_method]!r} instead.",
                DeprecationWarning, stacklevel=2,
            )
            self.force_method = _fm_aliases[self.force_method]

        # ---- buoyancy parameters ----
        self.gravity_z = float(physics.model.opt.gravity[2])  # e.g. -9.81
        # water_surface: height at which buoyancy becomes full.
        # 3-D:        MuJoCo z ↔ fluid z  → use solver.zmax
        # 2-D xz:     MuJoCo z ↔ fluid y  → use solver.ymax
        # 2-D xy:     no buoyancy needed   → 0.0
        if self.ndim == 3:
            self.water_surface = float(self.pars["solver"]["zmax"]) if "zmax" in self.pars["solver"] else 0.0
        elif self._2d_plane == "xz":
            self.water_surface = float(self.pars["solver"]["ymax"]) if "ymax" in self.pars["solver"] else 0.0
        else:
            self.water_surface = 0.0
        self._buoyancy_initialized = False

        # ---- optional physics contact tweaks ----
        solref = self.pars.get("physics", {}).get("solref", None)
        if solref is not None:
            physics.model.geom_solref[:, 0] = solref[0]
            physics.model.geom_solref[:, 1] = solref[1]
        solimp = self.pars.get("physics", {}).get("solimp", None)
        if solimp is not None:
            physics.model.geom_solimp[:, 0] = solimp[0]
            physics.model.geom_solimp[:, 1] = solimp[1]
            physics.model.geom_solimp[:, 2] = solimp[2]
            physics.model.geom_solimp[:, 3] = solimp[3]
            physics.model.geom_solimp[:, 4] = solimp[4]



        # ---- allocate per-body SDF / velocity arrays if missing ----
        if self.fluid_solver._solver_method == "kernel":
            self._init_interp()
        self._init_static_body_metadata()
        self._init_update()
        self._init_apply_forces()
        # Two-phase: buoyancy is EMERGENT from the fluid pressure (it is already
        # inside the loads returned by ``get_loads()`` / applied via xfrc), so
        # the external FARMS-style buoyancy term must be turned OFF here to avoid
        # double-counting. The ``_buoyancy_*`` indices then simply go unused and
        # ``_init_buoyancy_params`` is never invoked.
        if self._two_phase:
            self._has_buoyancy = False
        # override composite-body update with our FARMS-driven version
        self.fluid_solver.composite_body.update = self.update
        # back-pointer so downstream code (e.g. forces.py diagnostics) can
        # reach FARMS kinematics through the composite without importing
        # BDIMhandler directly.
        self.fluid_solver.composite_body._bdim_handler = self

    def _init_update(self):
        kernel_mode = self.fluid_solver._solver_method == "kernel"
        if kernel_mode:
            # Unified streaming path (Step 6 unification): the former
            # _update_2d/3d_streaming_multi are retained as the parity oracle
            # (test_update_streaming_parity.py) but no longer dispatched.
            self.update = self._update_streaming_multi
        else:
            # Unified Python path (Step 6 unification): the former
            # _update_2d / _update_3d are retained as the parity oracle
            # (see test_update_python_parity.py) but no longer dispatched.
            self.update = self._update_python

    def _init_apply_forces(self):
        """Bind ``self.apply_forces`` and precompute per-D force-axis maps.

        The 2-D and 3-D paths are now collapsed into a single
        :meth:`_apply_forces` (Step 6 of the unification refactor).  The
        per-D ``xfrc_applied`` index mapping, the buoyancy axis, and the
        FluidSolver field names to gather are precomputed here so the hot
        path is a single attribute read + list iteration with zero
        per-step Python branching on ``self.ndim`` or ``_2d_plane``.
        """
        self.apply_forces = self._apply_forces

        if self.ndim == 3:
            self._lin_xfrc_idx      = (0, 1, 2)
            self._ang_xfrc_idx      = (3, 4, 5)
            self._buoyancy_xfrc_idx = 2          # fz
            self._buoyancy_pos_idx  = 2          # com_pos[i][2]
            self._has_buoyancy      = True
            self._lin_visc_attrs = ('friction_force_lin_x',
                                    'friction_force_lin_y',
                                    'friction_force_lin_z')
            self._ang_visc_attrs = ('friction_force_ang_x',
                                    'friction_force_ang_y',
                                    'friction_force_ang_z')
            self._lin_pres_attrs = ('pressure_force_x',
                                    'pressure_force_y',
                                    'pressure_force_z')
            self._ang_pres_attrs = ('pressure_force_ang_x',
                                    'pressure_force_ang_y',
                                    'pressure_force_ang_z')
        else:
            # 2-D xz plane: MuJoCo (x, z) → fluid (x, y); buoyancy on z.
            # 2-D xy plane: MuJoCo (x, y) → fluid (x, y); no buoyancy.
            self._lin_xfrc_idx      = self._2d_force_axes[:2]
            self._ang_xfrc_idx      = (self._2d_force_axes[2],)
            self._has_buoyancy      = self._2d_has_buoyancy
            if self._2d_has_buoyancy:                # xz plane
                # Buoyancy is added to the fluid-y xfrc index, which is
                # the second linear xfrc index.  com_pos uses fluid-y
                # (= MuJoCo z) for the surface comparison.
                self._buoyancy_xfrc_idx = self._2d_force_axes[1]
                self._buoyancy_pos_idx  = 1
            else:                                    # xy plane
                self._buoyancy_xfrc_idx = None
                self._buoyancy_pos_idx  = None
            self._lin_visc_attrs = ('friction_force_lin_x',
                                    'friction_force_lin_y')
            self._ang_visc_attrs = ('friction_force_ang_z',)
            self._lin_pres_attrs = ('pressure_force_x',
                                    'pressure_force_y')
            self._ang_pres_attrs = ('pressure_force_ang_z',)

    # ------------------------------------------------------------------
    #  Kernel-path per-body SDF metadata
    # ------------------------------------------------------------------
    def _init_static_body_metadata(self):
        """Precompute immutable grid axes and body-local AABB bounds."""
        comp = self.fluid_solver.composite_body
        comp.gx_1d = comp.x.contiguous()
        comp.gy_1d = comp.y.contiguous()
        if self.ndim == 3:
            comp.gz_1d = comp.z.contiguous()
        comp._body_aabbs = [None] * len(comp.bodies)
        for body in comp.bodies:
            if self.ndim == 3:
                body._bdim_local_aabb = self._body_local_aabb_3d(body)
            else:
                body._bdim_local_aabb = self._body_local_aabb_2d(body)

    @staticmethod
    def _make_stream_meta(F, axes):
        inv = []
        for axis in axes:
            step = float(axis[1].item() - axis[0].item()) if axis.numel() > 1 else 1.0
            inv.append(1.0 / step)
        meta = {
            'F': F.contiguous(),
            'bx': axes[0].contiguous(),
            'by': axes[1].contiguous(),
            'bx0': float(axes[0][0].item()),
            'by0': float(axes[1][0].item()),
            'bx_last': float(axes[0][-1].item()),
            'by_last': float(axes[1][-1].item()),
            'inv_dx': inv[0],
            'inv_dy': inv[1],
        }
        if len(axes) == 3:
            meta.update({
                'bz': axes[2].contiguous(),
                'bz0': float(axes[2][0].item()),
                'bz_last': float(axes[2][-1].item()),
                'inv_dz': inv[2],
                'inv_vol': inv[0] * inv[1] * inv[2],
            })
        else:
            meta['inv_vol'] = inv[0] * inv[1]
        return meta

    def gather_data(self, iteration):
        """Gather FARMS link poses/velocities once per update path.

        Reads from the ``AnimatData`` sensor buffers at ``iteration`` when
        ``self._pose_source == "sensors"`` (default), or live from
        ``physics.data`` when ``"physics"`` (strong coupling).  Both
        sources return the same SI-scaled quantities — the physics path
        mirrors ``physics2data`` field-for-field — so they agree at the
        start-of-step pose (see :meth:`_gather_data_physics` and the
        equivalence test).
        """
        if self._pose_source == "physics":
            return self._gather_data_physics()

        com_poses = []
        urdf_poses = []
        Rs = []
        lin_vels = []
        ang_vels = []
        for exp_data in self.data:
            sen = exp_data.sensors.links

            com = np.asarray(sen.com_positions()[iteration, :], dtype=self.dtype_np)
            urdf = np.asarray(sen.urdf_positions()[iteration, :], dtype=self.dtype_np)
            R = Rotation.from_quat(sen.urdf_orientations()[iteration, :]).as_matrix().astype(self.dtype_np)
            lin = np.asarray(sen.com_lin_velocities()[iteration, :], dtype=self.dtype_np)
            nlinks = len(sen.names)
            if self.ndim == 2:
                com = com[:, self.lin_axes]
                urdf = urdf[:, self.lin_axes]
                R = R[:, self.lin_axes, :][:, :, self.lin_axes]
                lin = lin[:, self.lin_axes]
                ang = np.asarray(
                    [sen.com_ang_velocity(iteration, lk)[self._2d_ang_ax]
                        for lk in range(nlinks)],
                    dtype=self.dtype_np,
                )
            else:
                ang = np.stack([
                    np.asarray(sen.com_ang_velocity(iteration, lk), dtype=self.dtype_np)
                    for lk in range(nlinks)
                ])
            com_poses.append(com)
            urdf_poses.append(urdf)
            Rs.append(R)
            lin_vels.append(lin)
            ang_vels.append(ang)
        return com_poses, urdf_poses, Rs, lin_vels, ang_vels

    def _gather_data_physics(self):
        """``gather_data`` variant reading live ``physics.data`` (strong coupling).

        Mirrors ``farms_mujoco.simulation.physics.physics2data`` exactly so
        the returned SI quantities match the ``"sensors"`` path at the same
        physics state:

        * ``urdf_pos`` ← ``data.xpos[xpos2data]   / units.meters``
        * ``com_pos``  ← ``data.xipos[xipos2data] / units.meters``
        * ``R``        ← ``Rotation.from_quat(data.xquat[xquat2data][:, [1,2,3,0]])``
        * ``lin_vel``  ← ``data.sensordata[framelinvel2data] / units.velocity``
        * ``ang_vel``  ← ``data.sensordata[frameangvel2data] / units.angular_velocity``

        Because it reads ``physics.data`` directly, an internal
        ``physics.step`` prediction in the implicit loop is seen here without
        a ``task.update_sensors`` round-trip.  Requires ``self._task`` and
        ``self._physics`` to be set (the implicit step / the test set them).
        """
        physics = self._physics
        task    = self._task
        units   = task.units
        d       = physics.data

        com_poses, urdf_poses, Rs, lin_vels, ang_vels = [], [], [], [], []
        for animat_i, _exp in enumerate(self.data):
            sm = task.maps[animat_i]["sensors"]

            urdf  = np.asarray(d.xpos[sm["xpos2data"]],  dtype=self.dtype_np) / units.meters
            com   = np.asarray(d.xipos[sm["xipos2data"]], dtype=self.dtype_np) / units.meters
            quat  = np.asarray(d.xquat[sm["xquat2data"]], dtype=self.dtype_np)[:, [1, 2, 3, 0]]
            R     = Rotation.from_quat(quat).as_matrix().astype(self.dtype_np)
            lin   = np.asarray(d.sensordata[sm["framelinvel2data"]], dtype=self.dtype_np) / units.velocity
            ang   = np.asarray(d.sensordata[sm["frameangvel2data"]], dtype=self.dtype_np) / units.angular_velocity

            if self.ndim == 2:
                com  = com[:, self.lin_axes]
                urdf = urdf[:, self.lin_axes]
                R    = R[:, self.lin_axes, :][:, :, self.lin_axes]
                lin  = lin[:, self.lin_axes]
                ang  = ang[:, self._2d_ang_ax]

            com_poses.append(com)
            urdf_poses.append(urdf)
            Rs.append(R)
            lin_vels.append(lin)
            ang_vels.append(ang)
        return com_poses, urdf_poses, Rs, lin_vels, ang_vels

    def _init_interp(self):
        """Build per-body regular-grid streaming metadata for the kernel path.

        Mesh bodies reuse their precomputed ``body.sdf`` tables.
        Analytical bodies are pre-sampled onto a regular local-frame grid
        derived from ``body.local_aabb``.

        Raises ``RuntimeError`` if any body has neither a regular-grid SDF
        table nor a callable SDF with ``local_aabb``. Silent fallback to
        the Python path is intentionally forbidden when ``use_kernels=True``.
        """
        comp = self.fluid_solver.composite_body
        h = float(self.fluid_solver.h)
        axis_names = ('x', 'y', 'z') if self.ndim == 3 else ('x', 'y')
        n_tabulated = 0
        n_sampled = 0

        for body in comp.bodies:
            sdf = getattr(body, 'sdf', None)
            # Mesh bodies with precomputed SDF tables
            if hasattr(body, 'mesh_file'):
                F = sdf.F.contiguous()
                axes = tuple(
                    getattr(sdf, axis_name).contiguous()
                    for axis_name in axis_names
                )
                body._stream_meta = self._make_stream_meta(F, axes)
                # The precomputed SDF table axes may be padded much larger than
                # the BDIM band (legacy tables sometimes have hundreds of mm of
                # padding).  Store a tight contour-based AABB so that
                # _update_*_streaming_multi uses it for world-space cell selection
                # rather than the full padded table bounds.
                cnt = getattr(body, 'cnt', None)
                if cnt is not None and cnt.numel() > 2:
                    _bm = float(getattr(body, 'eps', 0.05)) + 4.0 * h
                    body._stream_meta['local_aabb_lo'] = (
                        cnt.min(dim=1).values - _bm
                    )
                    body._stream_meta['local_aabb_hi'] = (
                        cnt.max(dim=1).values + _bm
                    )
                n_tabulated += 1
                continue
            # Analytical bodies with callable SDFs
            else:
                local_aabb = getattr(body, 'local_aabb', None)
                sdf_callable = getattr(body, 'sdf', None)
                lo = local_aabb[0].cpu()
                hi = local_aabb[1].cpu()
                sizes = [
                    max(2, int(round(float(hi[idx] - lo[idx]) / h)) + 1)
                    for idx in range(self.ndim)
                ]
                axes = tuple(
                    torch.linspace(
                        float(lo[idx]), float(hi[idx]), sizes[idx],
                        dtype=self.dtype, device=self.device,
                    )
                    for idx in range(self.ndim)
                )
                mesh = torch.meshgrid(*axes, indexing='ij')
                with torch.no_grad():
                    F = sdf_callable(*mesh).contiguous()
                body._stream_meta = self._make_stream_meta(F, axes)
                n_sampled += 1
                continue


        print(
            f"  [stream-meta-{self.ndim}D] built {len(comp.bodies)}/{len(comp.bodies)} "
            f"per-body {self.ndim}-D streaming-SDF metadata records "
            f"({n_tabulated} tabulated, {n_sampled} sampled from analytical SDF)"
        )


    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _init_body_neighbors_2d(self):
        """Cache previous/next body indices within each animat for 2-D."""
        comp = self.fluid_solver.composite_body

        prev_body_index = [None] * len(comp.bodies)
        next_body_index = [None] * len(comp.bodies)
        last_body_index_by_animat = {}
        for built_body_i, (built_animat_id, _) in enumerate(comp.body_ids):
            prev_i = last_body_index_by_animat.get(built_animat_id)
            if prev_i is not None:
                prev_body_index[built_body_i] = prev_i
                next_body_index[prev_i] = built_body_i
            last_body_index_by_animat[built_animat_id] = built_body_i

        self._prev_body_index = tuple(prev_body_index)
        self._next_body_index = tuple(next_body_index)

    def _init_buoyancy_params(self, task, physics):
        """Precompute per-body mass & half-height for FARMS-style buoyancy.

        Called lazily on the first ``_apply_forces_3d`` invocation because
        ``task`` (needed for the body-index mapping) is not available at
        ``__init__`` time.

        FARMS formula (from drag.pyx ``compute_buoyancy``):
            if pos_z - height < surface:
                F_z = -rho_water * mass * gravity / density
                      * min((surface + height - pos_z) / (2*height), 1)
        where
            mass   = MuJoCo body_mass            [kg]
            density = morphology link density     [kg/m³]
            height  = 0.5 * geom bounding radius  [m]
            surface = water surface height (zmax)  [m]
            gravity = -9.81                        [m/s²]
        """
        comp = self.fluid_solver.composite_body
        n = comp.nbodies
        experiment_options = self.pars.get("body", {}).get("experiment_options", None)

        # Body density for the displaced-volume term (V = mass/density) comes
        # from the ANIMAT config (morphology link density) below; this is the
        # ONLY place the body density enters the coupling.  The fallback is the
        # FLUID density (neutral buoyancy when an animat density is missing) --
        # the solver no longer carries a ``rho_body``.
        self._buoy_mass   = np.zeros(n)
        self._buoy_density = np.full(n, float(self.fluid_solver.rho))
        self._buoy_height = np.zeros(n)

        for body_i in range(n):
            (animat_id, link_id) = comp.body_ids[body_i]
            ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            self._buoy_mass[body_i] = float(physics.model.body_mass[ind])
            if experiment_options is not None:
                try:
                    density = float(
                        experiment_options.animats[animat_id]
                        .morphology.links[link_id].density
                    )
                except Exception:
                    density = float(self.fluid_solver.rho)
                if density > 0.0:
                    self._buoy_density[body_i] = density

            # Find the maximum bounding-sphere radius among geoms attached
            # to this MuJoCo body (same logic as FARMS SwimmingHandler).
            max_rbound = 0.0
            for gi in range(physics.model.ngeom):
                if int(physics.model.geom_bodyid[gi]) == ind:
                    rb = float(physics.model.geom_rbound[gi])
                    if rb > max_rbound:
                        max_rbound = rb
            self._buoy_height[body_i] = 0.5 * max_rbound

        self._buoyancy_initialized = True

    # ==================================================================
    #  update: FARMS kinematics  ->  SDF fields + body velocities
    # ==================================================================


    @staticmethod
    def _body_local_aabb_2d(body):
        """Return ``(sdf_lo, sdf_hi)`` 1-D tensors describing the
        body's local-frame AABB, or ``None`` if no AABB info is
        available.  Source priority:

        1. ``body.local_aabb`` — an explicit ``(2, 2)`` tensor set by
           analytical bodies (auto-derived from the contour with a
           safety margin in :meth:`BodyAnalytical._initialize_2d`,
           or user-provided).
        2. ``body.cnt`` — contour-based tight AABB for mesh bodies.
           The precomputed SDF table (``body.sdf.x/y``) may be padded
           far beyond the BDIM band; the contour gives the actual body
           outline and yields a much tighter bound.
        3. ``body.sdf.x`` / ``body.sdf.y`` — fallback to full SDF
           table bounds when no contour is available.

        Returns ``None`` when neither source is available; callers
        should then use the full-grid evaluation path.
        """
        if hasattr(body, '_bdim_local_aabb'):
            return body._bdim_local_aabb
        local_aabb = getattr(body, 'local_aabb', None)
        if local_aabb is not None:
            return local_aabb[0], local_aabb[1]
        # Contour-based tight AABB: use body surface + eps+4h margin.
        cnt = getattr(body, 'cnt', None)
        if cnt is not None and cnt.numel() > 2:
            eps_m = float(getattr(body, 'eps', 0.05))
            h_m   = float(getattr(body, 'h',   0.001))
            band  = eps_m + 4.0 * h_m
            lo = cnt.min(dim=1).values - band
            hi = cnt.max(dim=1).values + band
            return lo, hi
        # Last resort: full SDF-table bounds (may be very large for old
        # precomputed mesh SDF tables with excessive padding).
        if (hasattr(body, 'sdf')
                and hasattr(body.sdf, 'x')
                and hasattr(body.sdf, 'y')):
            return (
                torch.stack([body.sdf.x[0],  body.sdf.y[0]]),
                torch.stack([body.sdf.x[-1], body.sdf.y[-1]]),
            )
        return None

    @staticmethod
    def _body_aabb_local_2d(body, R, urdf_pos, comp_x, comp_y,
                              h, gs, pad=3):
        """2-D analogue of :meth:`_body_aabb_indices`.

        Computes fluid-grid index ranges ``(i0, i1, j0, j1)`` for the
        AABB of a body's local-frame SDF support, transformed into
        world space.

        Supports both mesh bodies (``body.sdf`` exposes ``x``/``y``
        axis tensors) and analytical bodies that provide an explicit
        ``body.local_aabb`` covering the SDF band of width ``~2*eps``
        (auto-derived from ``body.cnt`` for :class:`BodyAnalytical`
        instances; user-supplied otherwise).  When neither AABB
        descriptor is available, returns ``None`` and the caller falls
        back to the full-grid path.

        Returns ``None`` when the AABB covers >90% of the grid (full-
        grid evaluation is then cheaper).  Returns half-open Python
        slice indices otherwise.
        """
        bounds = BDIMhandler._body_local_aabb_2d(body)
        if bounds is None:
            return None
        sdf_lo, sdf_hi = bounds

        local_center = 0.5 * (sdf_lo + sdf_hi)
        local_half   = 0.5 * (sdf_hi - sdf_lo)

        # AABB of the oriented local box in world space:
        #   world_half[i] = Σ_j |R[i,j]| · local_half[j]
        world_half   = R.abs() @ local_half
        world_center = R @ local_center + urdf_pos

        w_min = world_center - world_half
        w_max = world_center + world_half

        x0, y0 = comp_x[0], comp_y[0]
        inv_h = 1.0 / float(h)

        i0 = max(0,     int(((w_min[0] - x0) * inv_h).item()) - pad)
        i1 = min(gs[0], int(((w_max[0] - x0) * inv_h).item()) + 1 + pad)
        j0 = max(0,     int(((w_min[1] - y0) * inv_h).item()) - pad)
        j1 = min(gs[1], int(((w_max[1] - y0) * inv_h).item()) + 1 + pad)

        sub_vol  = (i1 - i0) * (j1 - j0)
        full_vol = gs[0] * gs[1]
        if sub_vol > 0.9 * full_vol:
            return None

        return (i0, i1, j0, j1)

    # ---- 3-D update --------------------------------------------------
    # ------------------------------------------------------------------
    #  AABB narrow-band helpers (3-D)
    # ------------------------------------------------------------------
    @staticmethod
    def _body_local_aabb_3d(body):
        """Return ``(sdf_lo, sdf_hi)`` 1-D tensors describing the
        body's local-frame AABB, or ``None`` if no AABB info is
        available.  Source priority:

        1. ``body.local_aabb`` — an explicit ``(2, 3)`` tensor for
           analytical or otherwise tabulated bodies that provide an
           AABB covering their SDF band of width ``~2*eps``.  For
           ``BodyAnalytical`` this is auto-derived during
           ``_initialize_3d`` via marching cubes on the local SDF
           plus a ``eps + 4*h`` Lipschitz-safe margin (mirrors the
           2-D contour-based path); users may still override it via
           the constructor when they want a tighter / looser bound.
        2. ``body.sdf.x`` / ``body.sdf.y`` / ``body.sdf.z`` — for mesh
           bodies whose local SDF table carries axis tensors and is
           padded with the ``_FAR`` sentinel outside the mesh.

        Returns ``None`` when neither source is available.
        """
        if hasattr(body, '_bdim_local_aabb'):
            return body._bdim_local_aabb
        local_aabb = getattr(body, 'local_aabb', None)
        if local_aabb is not None:
            return local_aabb[0], local_aabb[1]
        if (hasattr(body, 'sdf')
                and hasattr(body.sdf, 'x')
                and hasattr(body.sdf, 'y')
                and hasattr(body.sdf, 'z')):
            return (
                torch.stack([body.sdf.x[0],  body.sdf.y[0],  body.sdf.z[0]]),
                torch.stack([body.sdf.x[-1], body.sdf.y[-1], body.sdf.z[-1]]),
            )
        return None

    @staticmethod
    def _body_aabb_indices(body, R, urdf_pos, comp_x, comp_y, comp_z,
                           h, gs, pad=3):
        """Compute fluid-grid index ranges for a body's SDF domain.

        Transforms the body-local SDF AABB (mesh: from
        ``body.sdf.x/y/z`` axes; analytical: from ``body.local_aabb``)
        into world coordinates accounting for rotation, then finds
        the axis-aligned sub-block of the fluid grid that covers this
        region plus ``pad`` cells of margin on each side.

        Returns (i0, i1, j0, j1, k0, k1) — Python-style half-open
        indices suitable for slicing ``comp.X[i0:i1, j0:j1, k0:k1]``.
        If the sub-block covers >90 % of the grid, returns ``None``
        to signal that full-grid evaluation is cheaper (avoids the
        fill + scatter overhead).  Also returns ``None`` when the
        body provides no AABB descriptor.
        """
        bounds = BDIMhandler._body_local_aabb_3d(body)
        if bounds is None:
            return None
        sdf_lo, sdf_hi = bounds

        local_center = 0.5 * (sdf_lo + sdf_hi)
        local_half   = 0.5 * (sdf_hi - sdf_lo)

        # AABB of the oriented bounding box in world space
        #   world_half[i] = Σ_j |R[i,j]| · local_half[j]
        world_half   = R.abs() @ local_half
        world_center = R @ local_center + urdf_pos

        w_min = world_center - world_half
        w_max = world_center + world_half

        # Convert to grid indices (uniform grid: idx = (coord - x0) / h)
        x0, y0, z0 = comp_x[0], comp_y[0], comp_z[0]
        inv_h = 1.0 / h

        i0 = max(0,     int(((w_min[0] - x0) * inv_h).item()) - pad)
        i1 = min(gs[0], int(((w_max[0] - x0) * inv_h).item()) + 1 + pad)
        j0 = max(0,     int(((w_min[1] - y0) * inv_h).item()) - pad)
        j1 = min(gs[1], int(((w_max[1] - y0) * inv_h).item()) + 1 + pad)
        k0 = max(0,     int(((w_min[2] - z0) * inv_h).item()) - pad)
        k1 = min(gs[2], int(((w_max[2] - z0) * inv_h).item()) + 1 + pad)

        sub_vol = (i1 - i0) * (j1 - j0) * (k1 - k0)
        full_vol = gs[0] * gs[1] * gs[2]

        if sub_vol > 0.9 * full_vol:
            return None                       # fall back to full grid

        return (i0, i1, j0, j1, k0, k1)

    def _compose_body_frame_3d(self, body, urdf_pos, R):
        """Compose the MuJoCo link pose with the body's local collision pose."""
        local_pose = getattr(body, "local_pose", None)
        if local_pose is None:
            return urdf_pos, R

        local_translation = getattr(body, "_local_translation_t", None)
        local_rotation = getattr(body, "_local_rotation_t", None)
        if local_translation is None or local_rotation is None:
            local_pose_np = np.asarray(local_pose, dtype=self.dtype_np)
            local_translation = torch.tensor(
                local_pose_np[:3], device=self.device, dtype=self.dtype,
            )
            local_rotation = torch.tensor(
                Rotation.from_euler("xyz", local_pose_np[3:]).as_matrix().astype(self.dtype_np),
                device=self.device, dtype=self.dtype,
            )
            body._local_translation_t = local_translation
            body._local_rotation_t = local_rotation

        body_pos = urdf_pos + R @ local_translation
        body_rot = R @ local_rotation
        return body_pos, body_rot

    @staticmethod
    def _slice_from_aabb(aabb, ndim):
        """Return a region slice tuple and whether it spans the full grid."""
        if aabb is None:
            return (slice(None),) * ndim, True
        return tuple(
            slice(aabb[2 * axis], aabb[2 * axis + 1])
            for axis in range(ndim)
        ), False

    @staticmethod
    def _store_sparse_sdf(comp, body_i, aabb, sdf_region):
        comp._sdf_sparse[body_i] = (
            (None, sdf_region) if aabb is None else (aabb, sdf_region)
        )

    @staticmethod
    def _merge_region_sdf(target_sdf, sdf_region, sl, full_region):
        if full_region:
            mask = sdf_region < target_sdf
            target_sdf.copy_(torch.where(mask, sdf_region, target_sdf))
            return

        old_sdf = target_sdf[sl].contiguous()
        mask = sdf_region < old_sdf
        target_sdf[sl] = torch.where(mask, sdf_region, old_sdf)

    @staticmethod
    def _merge_region_sdf_and_velocity(target_sdf, target_vel, sdf_region,
                                       vel_region, sl, full_region):
        if full_region:
            mask = sdf_region < target_sdf
            target_sdf.copy_(torch.where(mask, sdf_region, target_sdf))
            target_vel.copy_(torch.where(mask, vel_region, target_vel))
            return

        old_sdf = target_sdf[sl].contiguous()
        mask = sdf_region < old_sdf
        target_sdf[sl] = torch.where(mask, sdf_region, old_sdf)
        target_vel[sl] = torch.where(
            mask, vel_region, target_vel[sl].contiguous())

    def _accum_region_velocity_blend(self, num, den, sdf_region, vel_region,
                                     sl, full_region):
        """Accumulate the SDF-weighted velocity blend for one body/stagger.

        ``num`` holds Σ w_i v_i and ``den`` holds Σ w_i (both per-face,
        zeroed at the start of the step).  ``w_i = σ(-φ_i/ε_w)`` is a smooth
        per-body weight: ~1 deep inside body i, 0.5 on its surface, →0 a
        band ε_w outside.  Finalised by :meth:`_finalize_velocity_blend`.
        """
        w = torch.sigmoid(-sdf_region / self._blend_eps)
        wv = w * vel_region
        if full_region:
            num.add_(wv)
            den.add_(w)
        else:
            num[sl] += wv
            den[sl] += w

    @staticmethod
    def _finalize_velocity_blend(target_vel, num, den):
        """Write body velocity = num/den (guarded where den≈0, i.e. fluid)."""
        target_vel.copy_(num / den.clamp_min(1e-12))

    # ==================================================================
    #  Unified Python-path SDF/velocity update (2-D + 3-D)
    # ==================================================================
    def _update_python(self, t, iteration, dt=1):
        """Dimension-agnostic FARMS-driven SDF + body-velocity update.

        Merges the former ``_update_2d`` / ``_update_3d`` Python paths
        (Step 6 of the 2D/3D unification).  ``self.ndim`` (2 or 3)
        selects the per-axis field lists, the body-frame composition,
        the rotation helper, the rigid-body velocity formula, and the
        AABB descriptor; the per-body running-min union loop, the
        smooth velocity blend, and the sparse-SDF bookkeeping are shared.

        Behaviour is identical to the two former per-dim methods (no
        semantics change).  ``self._sim_axes = range(ndim)`` indexes the
        simulated velocity components (u,v[,w]).
        """
        D    = self.ndim
        fs   = self.fluid_solver
        comp = fs.composite_body
        gs   = fs.grid_shape
        B    = len(comp.bodies)

        # Far-field SDF value (>> eps) so mu0 = 1 (pure fluid) and the
        # union-min always prefers the closest real body value.
        _FAR = 1e4

        com_poses, urdf_poses, Rs, lin_vels, ang_vels = self.gather_data(iteration)

        # Convert per-animat kinematics to device tensors for torch ops.
        com_poses_t  = [torch.from_numpy(a).to(self.device) for a in com_poses]
        urdf_poses_t = [torch.from_numpy(a).to(self.device) for a in urdf_poses]
        Rs_t         = [torch.from_numpy(a).to(self.device) for a in Rs]
        lin_vels_t   = [torch.from_numpy(a).to(self.device) for a in lin_vels]
        ang_vels_t   = [torch.from_numpy(a).to(self.device) for a in ang_vels]

        h_grid = comp.h          # uniform grid spacing

        # Per-body sparse SDF storage for forces (reset each step).
        comp._sdf_sparse = [None] * B

        # Initialise union fields to _FAR / zero in-place (once per step).
        comp.sdf_val.fill_(_FAR)
        sdf_stag_fields = [comp.sdf_val_u, comp.sdf_val_v]
        body_vel_fields = [comp.body_u, comp.body_v]
        if D == 3:
            sdf_stag_fields.append(comp.sdf_val_w)
            body_vel_fields.append(comp.body_w)
        for f in sdf_stag_fields:
            f.fill_(_FAR)
        for f in body_vel_fields:
            f.zero_()

        # CC grid coords (3-D uses Z_grid for the cell-centred z axis).
        cc_coords = (comp.X, comp.Y) if D == 2 else (comp.X, comp.Y, comp.Z_grid)
        # Per-axis staggered face coords (u,v[,w]).
        if D == 2:
            stag = [(comp.Xu_stag, comp.Yu_stag),
                    (comp.Xv_stag, comp.Yv_stag)]
        else:
            stag = [(comp.Xu_stag, comp.Yu_stag, comp.Zu_stag),
                    (comp.Xv_stag, comp.Yv_stag, comp.Zv_stag),
                    (comp.Xw_stag, comp.Yw_stag, comp.Zw_stag)]

        # Smooth velocity-blend bookkeeping: body_* hold the Σ w_i v_i
        # numerator (zeroed above), den_* the Σ w_i denominator.
        blend = self._blend_eps_cells is not None
        if blend:
            self._blend_eps = h_grid * self._blend_eps_cells
            if (self._blend_den is None
                    or self._blend_den[0].shape != comp.body_u.shape):
                self._blend_den = [torch.zeros_like(comp.body_u)
                                   for _ in range(D)]
            den = self._blend_den
            for d in den:
                d.zero_()

        # Cache per-body AABBs for downstream use (e.g. narrow-band forces).
        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]

            com_pos  = com_poses_t[animat_id][link_id]
            urdf_pos = urdf_poses_t[animat_id][link_id]
            R        = Rs_t[animat_id][link_id]
            lin_vel  = lin_vels_t[animat_id][link_id]
            ang_vel  = ang_vels_t[animat_id][link_id]

            # ── Body frame + AABB descriptor ────────────────────────
            # 2-D rigid-body links collapse to identity local frames, so
            # body_pos = urdf_pos, body_rot = R.  3-D composes the MuJoCo
            # link pose with the body's local collision pose.
            if D == 2:
                body_pos, body_rot = urdf_pos, R
                aabb = self._body_aabb_local_2d(
                    body, body_rot, body_pos,
                    comp.x, comp.y, h_grid, gs, pad=3,
                )
            else:
                body_pos, body_rot = self._compose_body_frame_3d(
                    body, urdf_pos, R)
                aabb = self._body_aabb_indices(
                    body, body_rot, body_pos,
                    comp.x, comp.y, comp.z, h_grid, gs, pad=3,
                )
            comp._body_aabbs[body_i] = aabb
            sl, full_region = self._slice_from_aabb(aabb, D)

            R_T = body_rot.T
            sdf_eval = body.sdf

            def _rotate(coords):
                if D == 2:
                    return rotate_grid_2d(coords[0], coords[1], R_T, body_pos)
                return _rotate_grid_3d_compiled(
                    coords[0], coords[1], coords[2], R_T, body_pos)

            # Cell-centred SDF (running-min geometry union).
            p_cc = _rotate(tuple(c[sl] for c in cc_coords))
            sdf_cc = sdf_eval(*p_cc)
            self._store_sparse_sdf(comp, body_i, aabb, sdf_cc)
            self._merge_region_sdf(comp.sdf_val, sdf_cc, sl, full_region)

            # Staggered-face SDF + rigid-body velocity per simulated axis.
            # v = lin + ω × (r − com); 2-D ω is the single out-of-plane
            # component (the z-component of that cross product), 3-D is the
            # full cross product evaluated at each face's stagger location.
            if D == 2:
                vels = [
                    lin_vel[0] - ang_vel * (stag[0][1][sl] - com_pos[1]),
                    lin_vel[1] + ang_vel * (stag[1][0][sl] - com_pos[0]),
                ]
            else:
                vels = [
                    (lin_vel[0]
                     + ang_vel[1] * (stag[0][2][sl] - com_pos[2])
                     - ang_vel[2] * (stag[0][1][sl] - com_pos[1])),
                    (lin_vel[1]
                     + ang_vel[2] * (stag[1][0][sl] - com_pos[0])
                     - ang_vel[0] * (stag[1][2][sl] - com_pos[2])),
                    (lin_vel[2]
                     + ang_vel[0] * (stag[2][1][sl] - com_pos[1])
                     - ang_vel[1] * (stag[2][0][sl] - com_pos[0])),
                ]

            for a in self._sim_axes:
                p_a = _rotate(tuple(c[sl] for c in stag[a]))
                sdf_a = sdf_eval(*p_a)
                if blend:
                    # Geometry stays running-min; velocity is the smooth
                    # SDF-weighted blend (continuous across link seams).
                    self._merge_region_sdf(
                        sdf_stag_fields[a], sdf_a, sl, full_region)
                    self._accum_region_velocity_blend(
                        body_vel_fields[a], den[a], sdf_a, vels[a],
                        sl, full_region)
                else:
                    self._merge_region_sdf_and_velocity(
                        sdf_stag_fields[a], body_vel_fields[a],
                        sdf_a, vels[a], sl, full_region)

            comp.com_pos[body_i] = com_pos
            body.com_pos = com_pos

            # ── Lagrangian world-frame surface-marker refresh ───────
            if self.force_method == "lagrangian":
                if D == 2:
                    self._refresh_lagrangian_contour_2d(
                        body, R, urdf_pos)
                else:
                    self._refresh_lagrangian_tris_3d(
                        comp, body_i, body, body_rot, body_pos)

            # ── 2-D contour mask + per-link r_com (2-D only) ────────
            if D == 2:
                if self.contour_mask:
                    self._apply_contour_mask_2d(
                        comp, body_i, body, R, urdf_pos,
                        Rs_t, urdf_poses_t, animat_id)
                body.r_com = body.cnt_update - com_pos[:, None]

        if blend:
            for a in self._sim_axes:
                self._finalize_velocity_blend(
                    body_vel_fields[a], body_vel_fields[a], den[a])

    def _refresh_lagrangian_contour_2d(self, body, R, urdf_pos):
        """World-frame refresh of a 2-D body contour (``body.cnt_update``).

        Lagrangian forces sample at ``body.cnt_update`` — must be in world
        coords.  The body contour is oriented CCW once (kernel/force
        convention: outward normal via ``nx = ty, ny = -tx``).
        """
        if body.cnt is None or body.cnt.numel() == 0:
            return
        if not hasattr(body, '_cnt_ccw_oriented'):
            cl = body.cnt
            dx = torch.roll(cl[0], -1) - cl[0]
            dy = torch.roll(cl[1], -1) - cl[1]
            cx_mid = 0.5 * (cl[0] + torch.roll(cl[0], -1))
            cy_mid = 0.5 * (cl[1] + torch.roll(cl[1], -1))
            signed_area = (0.5 * (cx_mid * dy - cy_mid * dx).sum()).item()
            if signed_area < 0:
                body.cnt = body.cnt.flip(dims=[1]).contiguous()
            body._cnt_ccw_oriented = True
        body.cnt_update = R @ body.cnt.to(self.dtype) + urdf_pos[:, None]

    def _refresh_lagrangian_tris_3d(self, comp, body_i, body, body_rot, body_pos):
        """World-frame refresh of a 3-D body's surface triangulation.

        Mirrors the 2-D ``cnt_update`` refresh: rotate+translate the
        body-local triangle centroids/normals into world coords so the
        3-D Lagrangian force samples the right fluid locations.  The
        bbox-centred mesh markers are shifted into the SDF-local frame
        via :meth:`_lagr_marker_offset`.
        """
        tcl = getattr(body, 'tri_centroid_local', None)
        tnl = getattr(body, 'tri_normal_local', None)
        if tcl is None or tnl is None:
            return
        # NOTE: capture the marker offset BEFORE overwriting
        # tri_centroid_world (it is read from the body's init value).
        off = self._lagr_marker_offset(comp, body_i, body)
        tcl_d = tcl.to(dtype=self.dtype, device=self.device)
        tnl_d = tnl.to(dtype=self.dtype, device=self.device)
        body.tri_centroid_world = body_rot @ (tcl_d + off) + body_pos[:, None]
        body.tri_normal_world   = body_rot @ tnl_d

    def _apply_contour_mask_2d(self, comp, body_i, body, R, urdf_pos,
                               Rs_t, urdf_poses_t, animat_id):
        """Optional contour mask for overlapping 2-D links (verbatim from
        the former ``_update_2d``)."""
        body.cnt_update = R @ body.cnt + urdf_pos[:, None]
        x_cnt = body.cnt_update[0]
        y_cnt = body.cnt_update[1]
        prev_body_i = self._prev_body_index[body_i]
        next_body_i = self._next_body_index[body_i]

        if prev_body_i is None and next_body_i is None:
            mask = torch.ones_like(x_cnt, dtype=torch.bool)
        elif prev_body_i is None:
            (_, next_link_id) = comp.body_ids[next_body_i]
            body_p = comp.bodies[next_body_i]
            pt = Rs_t[animat_id][next_link_id].T @ (
                torch.stack((x_cnt, y_cnt))
                - urdf_poses_t[animat_id][next_link_id][:, None]
            )
            mask = (body_p.sdf(pt[0], pt[1]) - body.h) >= 0
        elif next_body_i is None:
            (_, prev_link_id) = comp.body_ids[prev_body_i]
            body_m = comp.bodies[prev_body_i]
            pt = Rs_t[animat_id][prev_link_id].T @ (
                torch.stack((x_cnt, y_cnt))
                - urdf_poses_t[animat_id][prev_link_id][:, None]
            )
            mask = (body_m.sdf(pt[0], pt[1]) - body.h) >= 0
        else:
            (_, prev_link_id) = comp.body_ids[prev_body_i]
            body_m = comp.bodies[prev_body_i]
            pt_m = Rs_t[animat_id][prev_link_id].T @ (
                torch.stack((x_cnt, y_cnt))
                - urdf_poses_t[animat_id][prev_link_id][:, None]
            )
            (_, next_link_id) = comp.body_ids[next_body_i]
            body_p = comp.bodies[next_body_i]
            pt_p = Rs_t[animat_id][next_link_id].T @ (
                torch.stack((x_cnt, y_cnt))
                - urdf_poses_t[animat_id][next_link_id][:, None]
            )
            sdf_m = body_m.sdf(pt_m[0], pt_m[1]) - body.h
            sdf_p = body_p.sdf(pt_p[0], pt_p[1]) - body.h
            mask = (sdf_m >= 0) & (sdf_p >= 0)
        body.mask = mask

    # ---- 2-D update --------------------------------------------------
    def _update_2d(self, t, iteration, dt=1):

        fs   = self.fluid_solver
        comp = fs.composite_body
        gs   = fs.grid_shape
        B    = comp.nbodies

        _FAR = 1e4   # far-field SDF sentinel (same as 3-D path)

        com_poses, urdf_poses, Rs, lin_vels, ang_vels = self.gather_data(iteration)

        # Convert per-animat kinematics to device tensors for torch operations.
        com_poses_t  = [torch.from_numpy(a).to(self.device) for a in com_poses]
        urdf_poses_t = [torch.from_numpy(a).to(self.device) for a in urdf_poses]
        Rs_t         = [torch.from_numpy(a).to(self.device) for a in Rs]
        lin_vels_t   = [torch.from_numpy(a).to(self.device) for a in lin_vels]
        ang_vels_t   = [torch.from_numpy(a).to(self.device) for a in ang_vels]

        h_grid = comp.h          # uniform grid spacing

        # Per-body sparse SDF storage (mirrors the 3-D path).  Replaces
        # the legacy dense (B, Nx, Ny) ``comp.sdf_vals`` stack.
        comp._sdf_sparse = [None] * B

        # Initialise union fields to _FAR / zero (once per step).
        comp.sdf_val.fill_(_FAR)
        comp.sdf_val_u.fill_(_FAR)
        comp.sdf_val_v.fill_(_FAR)
        comp.body_u.zero_()
        comp.body_v.zero_()

        # Smooth velocity-blend bookkeeping (see _update_3d): body_* hold the
        # Σ w_i v_i numerator, den_* the Σ w_i denominator.
        blend = self._blend_eps_cells is not None
        if blend:
            self._blend_eps = h_grid * self._blend_eps_cells
            if (self._blend_den is None
                    or self._blend_den[0].shape != comp.body_u.shape):
                self._blend_den = [torch.zeros_like(comp.body_u)
                                   for _ in range(2)]
            den_u, den_v = self._blend_den
            den_u.zero_(); den_v.zero_()

        # Cache per-body AABBs for downstream use (e.g. narrow-band forces).
        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]

            com_pos  = com_poses_t[animat_id][link_id]
            urdf_pos = urdf_poses_t[animat_id][link_id]
            R        = Rs_t[animat_id][link_id]
            lin_vel  = lin_vels_t[animat_id][link_id]
            ang_vel  = ang_vels_t[animat_id][link_id]

            R_T = R.T  # (2, 2) transposed rotation

            # ── AABB-clipped SDF evaluation ─────────────────────────
            # Mirror the 3-D pattern: AABB-clip whenever the body
            # provides a local-frame bounding-box descriptor (mesh
            # bodies via ``body.sdf.x/y`` axes; analytical bodies via
            # ``body.local_aabb`` derived from the contour with a
            # band-radius safety margin).  Bodies without either
            # descriptor fall through to the full-grid path.
            aabb = self._body_aabb_local_2d(
                body, R, urdf_pos,
                comp.x, comp.y,
                h_grid, gs, pad=3,
            )
            comp._body_aabbs[body_i] = aabb
            sl, full_region = self._slice_from_aabb(aabb, self.ndim)

            sdf_eval = body.sdf

            px, py = rotate_grid_2d(comp.X[sl], comp.Y[sl], R_T, urdf_pos)
            sdf_cc = sdf_eval(px, py)
            self._store_sparse_sdf(comp, body_i, aabb, sdf_cc)

            # Evaluate SDF directly at staggered face locations and merge the
            # resulting region, whether it is a narrow AABB or the full grid.
            px_u, py_u = rotate_grid_2d(
                comp.Xu_stag[sl], comp.Yu_stag[sl], R_T, urdf_pos)
            sdf_u = sdf_eval(px_u, py_u)

            px_v, py_v = rotate_grid_2d(
                comp.Xv_stag[sl], comp.Yv_stag[sl], R_T, urdf_pos)
            sdf_v = sdf_eval(px_v, py_v)

            vel_u = lin_vel[0] - ang_vel * (comp.Yu_stag[sl] - com_pos[1])
            vel_v = lin_vel[1] + ang_vel * (comp.Xv_stag[sl] - com_pos[0])

            self._merge_region_sdf(comp.sdf_val, sdf_cc, sl, full_region)
            if blend:
                self._merge_region_sdf(comp.sdf_val_u, sdf_u, sl, full_region)
                self._merge_region_sdf(comp.sdf_val_v, sdf_v, sl, full_region)
                self._accum_region_velocity_blend(
                    comp.body_u, den_u, sdf_u, vel_u, sl, full_region)
                self._accum_region_velocity_blend(
                    comp.body_v, den_v, sdf_v, vel_v, sl, full_region)
            else:
                self._merge_region_sdf_and_velocity(
                    comp.sdf_val_u, comp.body_u, sdf_u, vel_u, sl, full_region)
                self._merge_region_sdf_and_velocity(
                    comp.sdf_val_v, comp.body_v, sdf_v, vel_v, sl, full_region)

            comp.com_pos[body_i] = com_pos
            body.com_pos = com_pos

            # Lagrangian forces sample at ``body.cnt_update`` — must be in
            # world coords.  Historically only set inside the contour-mask
            # branch, which left it as the body-local init for the common
            # contour_mask=False case → wrong sample positions for Lagrangian
            # forces.  Refresh unconditionally when Lagrangian is enabled.
            if self.force_method == "lagrangian":
                if body.cnt is not None and body.cnt.numel() > 0:
                    # See _update_2d_streaming_multi for CCW orientation rationale.
                    if not hasattr(body, '_cnt_ccw_oriented'):
                        cl = body.cnt
                        dx = torch.roll(cl[0], -1) - cl[0]
                        dy = torch.roll(cl[1], -1) - cl[1]
                        cx_mid = 0.5 * (cl[0] + torch.roll(cl[0], -1))
                        cy_mid = 0.5 * (cl[1] + torch.roll(cl[1], -1))
                        signed_area = (0.5 * (cx_mid * dy - cy_mid * dx).sum()).item()
                        if signed_area < 0:
                            body.cnt = body.cnt.flip(dims=[1]).contiguous()
                        body._cnt_ccw_oriented = True
                    body.cnt_update = R @ body.cnt.to(self.dtype) + urdf_pos[:, None]

            # optional contour mask for overlapping links
            if self.contour_mask:
                body.cnt_update = R @ body.cnt + urdf_pos[:, None]
                x_cnt = body.cnt_update[0]
                y_cnt = body.cnt_update[1]
                prev_body_i = self._prev_body_index[body_i]
                next_body_i = self._next_body_index[body_i]

                if prev_body_i is None and next_body_i is None:
                    mask = torch.ones_like(x_cnt, dtype=torch.bool)
                elif prev_body_i is None:
                    (_, next_link_id) = comp.body_ids[next_body_i]
                    body_p = comp.bodies[next_body_i]
                    pt = Rs_t[animat_id][next_link_id].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses_t[animat_id][next_link_id][:, None]
                    )
                    mask = (body_p.sdf(pt[0], pt[1]) - body.h) >= 0
                elif next_body_i is None:
                    (_, prev_link_id) = comp.body_ids[prev_body_i]
                    body_m = comp.bodies[prev_body_i]
                    pt = Rs_t[animat_id][prev_link_id].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses_t[animat_id][prev_link_id][:, None]
                    )
                    mask = (body_m.sdf(pt[0], pt[1]) - body.h) >= 0
                else:
                    (_, prev_link_id) = comp.body_ids[prev_body_i]
                    body_m = comp.bodies[prev_body_i]
                    pt_m = Rs_t[animat_id][prev_link_id].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses_t[animat_id][prev_link_id][:, None]
                    )
                    (_, next_link_id) = comp.body_ids[next_body_i]
                    body_p = comp.bodies[next_body_i]
                    pt_p = Rs_t[animat_id][next_link_id].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses_t[animat_id][next_link_id][:, None]
                    )
                    sdf_m = body_m.sdf(pt_m[0], pt_m[1]) - body.h
                    sdf_p = body_p.sdf(pt_p[0], pt_p[1]) - body.h
                    mask = (sdf_m >= 0) & (sdf_p >= 0)
                body.mask = mask

            body.r_com = body.cnt_update - com_pos[:, None]

        if blend:
            self._finalize_velocity_blend(comp.body_u, comp.body_u, den_u)
            self._finalize_velocity_blend(comp.body_v, comp.body_v, den_v)

    # ---- 3-D update --------------------------------------------------
    def _update_3d(self, t, iteration, dt=1):

        fs   = self.fluid_solver
        comp = fs.composite_body
        gs   = fs.grid_shape

        # Far-field SDF value (>> eps) so mu0 = 1 (pure fluid) and
        # union-min always prefers the closest real body value.
        _FAR = 1e4

        com_poses, urdf_poses, Rs, lin_vels, ang_vels = self.gather_data(iteration)

        # Convert per-animat kinematics to device tensors for torch operations.
        com_poses_t  = [torch.from_numpy(a).to(self.device) for a in com_poses]
        urdf_poses_t = [torch.from_numpy(a).to(self.device) for a in urdf_poses]
        Rs_t         = [torch.from_numpy(a).to(self.device) for a in Rs]
        lin_vels_t   = [torch.from_numpy(a).to(self.device) for a in lin_vels]
        ang_vels_t   = [torch.from_numpy(a).to(self.device) for a in ang_vels]

        h_grid = comp.h          # uniform grid spacing

        # Per-body SDF storage for forces (reset sparse list each step)
        comp._sdf_sparse = [None] * len(comp.bodies)

        # Initialise union fields to _FAR / zero (once per step).
        # Reuse existing tensors in-place to avoid re-allocation.
        comp.sdf_val.fill_(_FAR)
        comp.sdf_val_u.fill_(_FAR)
        comp.sdf_val_v.fill_(_FAR)
        comp.sdf_val_w.fill_(_FAR)
        comp.body_u.zero_()
        comp.body_v.zero_()
        comp.body_w.zero_()

        # Smooth velocity-blend bookkeeping: use comp.body_* as the Σw_i v_i
        # numerator (already zeroed above) and a matching denominator Σw_i.
        blend = self._blend_eps_cells is not None
        if blend:
            self._blend_eps = h_grid * self._blend_eps_cells
            if (self._blend_den is None
                    or self._blend_den[0].shape != comp.body_u.shape):
                self._blend_den = [torch.zeros_like(comp.body_u)
                                   for _ in range(3)]
            den_u, den_v, den_w = self._blend_den
            den_u.zero_(); den_v.zero_(); den_w.zero_()

        # Cache per-body AABBs for downstream use (e.g. narrow-band forces)
        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]

            com_pos  = com_poses_t[animat_id][link_id]
            urdf_pos = urdf_poses_t[animat_id][link_id]
            R        = Rs_t[animat_id][link_id]
            lin_vel  = lin_vels_t[animat_id][link_id]
            ang_vel  = ang_vels_t[animat_id][link_id]

            body_pos, body_rot = self._compose_body_frame_3d(body, urdf_pos, R)

            R_T = body_rot.T  # (3, 3) — transposed rotation

            # ── AABB-clipped SDF evaluation ─────────────────────────
            # AABB-clip whenever the body provides a local-frame
            # bounding-box descriptor: mesh bodies via the
            # ``body.sdf.x/y/z`` axis tensors of their padded SDF
            # table, analytical bodies via an explicit
            # ``body.local_aabb`` (sized to cover the BDIM band).
            # Bodies without either fall through to the full-grid
            # path returned as ``aabb=None``.
            aabb = self._body_aabb_indices(
                body, body_rot, body_pos,
                comp.x, comp.y, comp.z,
                h_grid, gs, pad=3,
            )
            comp._body_aabbs[body_i] = aabb   # cache for force integration
            sl, full_region = self._slice_from_aabb(aabb, self.ndim)

            # Default `body.sdf` (grid_sample-backed) per-body SDF evaluator.
            # The kernel-path regular-grid sampler is only used by the
            # streaming `_update_3d_streaming` path, which short-circuits
            # this function at the top.
            sdf_eval = body.sdf

            px, py, pz = _rotate_grid_3d_compiled(
                comp.X[sl], comp.Y[sl], comp.Z_grid[sl],
                R_T, body_pos,
            )
            sdf_cc = sdf_eval(px, py, pz)
            self._store_sparse_sdf(comp, body_i, aabb, sdf_cc)

            # Evaluate SDF directly at staggered face locations
            # (exact interpolation instead of CC averaging).
            px_u, py_u, pz_u = _rotate_grid_3d_compiled(
                comp.Xu_stag[sl], comp.Yu_stag[sl], comp.Zu_stag[sl],
                R_T, body_pos)
            sdf_u = sdf_eval(px_u, py_u, pz_u)

            px_v, py_v, pz_v = _rotate_grid_3d_compiled(
                comp.Xv_stag[sl], comp.Yv_stag[sl], comp.Zv_stag[sl],
                R_T, body_pos)
            sdf_v = sdf_eval(px_v, py_v, pz_v)

            px_w, py_w, pz_w = _rotate_grid_3d_compiled(
                comp.Xw_stag[sl], comp.Yw_stag[sl], comp.Zw_stag[sl],
                R_T, body_pos)
            sdf_w = sdf_eval(px_w, py_w, pz_w)

            vel_u = (
                lin_vel[0]
                + ang_vel[1] * (comp.Zu_stag[sl] - com_pos[2])
                - ang_vel[2] * (comp.Yu_stag[sl] - com_pos[1]))
            vel_v = (
                lin_vel[1]
                + ang_vel[2] * (comp.Xv_stag[sl] - com_pos[0])
                - ang_vel[0] * (comp.Zv_stag[sl] - com_pos[2]))
            vel_w = (
                lin_vel[2]
                + ang_vel[0] * (comp.Yw_stag[sl] - com_pos[1])
                - ang_vel[1] * (comp.Xw_stag[sl] - com_pos[0]))

            self._merge_region_sdf(comp.sdf_val, sdf_cc, sl, full_region)
            if blend:
                # Geometry (sdf_val_*) stays running-min; velocity is the
                # smooth SDF-weighted blend (continuous across seams).
                self._merge_region_sdf(comp.sdf_val_u, sdf_u, sl, full_region)
                self._merge_region_sdf(comp.sdf_val_v, sdf_v, sl, full_region)
                self._merge_region_sdf(comp.sdf_val_w, sdf_w, sl, full_region)
                self._accum_region_velocity_blend(
                    comp.body_u, den_u, sdf_u, vel_u, sl, full_region)
                self._accum_region_velocity_blend(
                    comp.body_v, den_v, sdf_v, vel_v, sl, full_region)
                self._accum_region_velocity_blend(
                    comp.body_w, den_w, sdf_w, vel_w, sl, full_region)
            else:
                self._merge_region_sdf_and_velocity(
                    comp.sdf_val_u, comp.body_u, sdf_u, vel_u, sl, full_region)
                self._merge_region_sdf_and_velocity(
                    comp.sdf_val_v, comp.body_v, sdf_v, vel_v, sl, full_region)
                self._merge_region_sdf_and_velocity(
                    comp.sdf_val_w, comp.body_w, sdf_w, vel_w, sl, full_region)

            comp.com_pos[body_i] = com_pos
            body.com_pos = com_pos

            # Lagrangian 3-D forces sample fluid fields at the world-frame
            # surface triangulation; refresh it here so the Python path
            # matches the kernel path (which updates it in
            # _update_3d_streaming_multi). Without this the markers stay at
            # their body-local init and the 3-D Lagrangian force is ~zero.
            if self.force_method == "lagrangian":
                tcl = getattr(body, 'tri_centroid_local', None)
                tnl = getattr(body, 'tri_normal_local', None)
                if tcl is not None and tnl is not None:
                    # NOTE: capture the marker offset BEFORE overwriting
                    # tri_centroid_world (it is read from the body's init value).
                    off = self._lagr_marker_offset(comp, body_i, body)
                    tcl_d = tcl.to(dtype=self.dtype, device=self.device)
                    tnl_d = tnl.to(dtype=self.dtype, device=self.device)
                    body.tri_centroid_world = body_rot @ (tcl_d + off) + body_pos[:, None]
                    body.tri_normal_world   = body_rot @ tnl_d

        if blend:
            self._finalize_velocity_blend(comp.body_u, comp.body_u, den_u)
            self._finalize_velocity_blend(comp.body_v, comp.body_v, den_v)
            self._finalize_velocity_blend(comp.body_w, comp.body_w, den_w)

    # ------------------------------------------------------------------
    #  Lagrangian 3-D surface-marker frame offset
    # ------------------------------------------------------------------
    def _lagr_marker_offset(self, comp, b, body):
        """Body-local offset between the surface-triangulation frame and the
        SDF-local frame, returned as a ``(3, 1)`` tensor (cached per body).

        ``BodyMesh`` builds ``tri_centroid_local`` re-centred on the SDF
        bounding-box centre, whereas the SDF interpolator / streaming kernel
        anchor the body frame at the SDF-grid origin.  The world-marker
        transform ``R @ tri_centroid_local + body_pos`` therefore lands
        ``R @ local_center`` short of the real surface — for the boat hull/keel
        that is ~2.2 m, placing every Lagrangian sample point well off the body
        (→ spurious buoyancy + a large pitch torque).  The offset is captured
        ONCE from the body's INIT ``tri_centroid_world`` (set in the body
        constructor, before this handler first overwrites it): it equals
        ``local_center`` for a bbox-centred mesh and is exactly zero for an
        origin-centred analytical body (so the corrected transform is a no-op
        there — analytical bodies, e.g. the validated drop-sphere, are
        unchanged).
        """
        cache = comp.__dict__.setdefault('_lagr_marker_off_cache', {})
        off = cache.get(b)
        if off is None:
            tcl = getattr(body, 'tri_centroid_local', None)
            tcw = getattr(body, 'tri_centroid_world', None)
            if (tcl is not None and tcw is not None
                    and tuple(tcw.shape) == tuple(tcl.shape)):
                off = (tcw.to(dtype=self.dtype, device=self.device)
                       - tcl.to(dtype=self.dtype, device=self.device)
                       ).mean(dim=1, keepdim=True)
            else:
                off = torch.zeros(3, 1, dtype=self.dtype, device=self.device)
            cache[b] = off
        return off

    # ==================================================================
    #  Unified streaming (kernel-path) SDF/velocity update (2-D + 3-D)
    # ==================================================================
    def _stream_kin_static(self, comp, B, gs, D):
        """Build (once) + return the cached static per-body kinematics
        descriptor for the streaming kernel path.

        Stored on ``comp._stream_kin_static`` (3-D) / ``_stream_kin_static_2d``
        (2-D).  Holds the per-body local pose (``local_lt``/``local_lr``), the
        local-frame SDF-box centre/half extent, the grid origin/spacing, and
        numpy mirrors used by the per-step host assembly.  Merges the former
        per-dim builders; the only dim-specific bit is the local-pose source
        (3-D reads ``body.local_pose``; 2-D links collapse to identity) and
        the SDF-box extent source (2-D prefers a tight ``local_aabb``).
        """
        attr = '_stream_kin_static' if D == 3 else '_stream_kin_static_2d'
        kin_static = getattr(comp, attr, None)
        if kin_static is not None:
            return kin_static

        anames = ('x', 'y', 'z')[:D]
        body_ids = torch.tensor(
            [(int(a), int(l)) for (a, l) in comp.body_ids], dtype=torch.int64,
        )
        local_lt = torch.zeros((B, D), dtype=self.dtype, device=self.device)
        local_lr = torch.eye(
            D, dtype=self.dtype, device=self.device,
        ).unsqueeze(0).repeat(B, 1, 1)
        sdf_lo = torch.empty((B, D), dtype=self.dtype, device=self.device)
        sdf_hi = torch.empty((B, D), dtype=self.dtype, device=self.device)

        for b, body in enumerate(comp.bodies):
            m = body._stream_meta
            if D == 3:
                # Compose with the body's local collision pose (mesh links).
                lp = getattr(body, 'local_pose', None)
                if lp is not None:
                    lp_t = torch.as_tensor(
                        lp, dtype=self.dtype, device=self.device)
                    local_lt[b] = lp_t[:3]
                    local_lr[b] = torch.as_tensor(
                        Rotation.from_euler(
                            'xyz', lp_t[3:].detach().cpu().numpy(),
                        ).as_matrix(),
                        dtype=self.dtype, device=self.device,
                    )
                sx, sy, sz = m['bx'], m['by'], m['bz']
                sdf_lo[b] = torch.stack((sx[0], sy[0], sz[0]))
                sdf_hi[b] = torch.stack((sx[-1], sy[-1], sz[-1]))
            else:
                # 2-D rigid-body links have identity local frames (lt=0, lr=I,
                # set above).  Prefer the tight contour-based AABB for mesh
                # bodies whose SDF table is padded far beyond the BDIM band.
                if 'local_aabb_lo' in m:
                    sdf_lo[b] = m['local_aabb_lo'].to(
                        dtype=self.dtype, device=self.device)
                    sdf_hi[b] = m['local_aabb_hi'].to(
                        dtype=self.dtype, device=self.device)
                else:
                    sdf_lo[b] = torch.stack((
                        torch.as_tensor(m['bx0'], dtype=self.dtype, device=self.device),
                        torch.as_tensor(m['by0'], dtype=self.dtype, device=self.device),
                    ))
                    sdf_hi[b] = torch.stack((
                        torch.as_tensor(m['bx_last'], dtype=self.dtype, device=self.device),
                        torch.as_tensor(m['by_last'], dtype=self.dtype, device=self.device),
                    ))

        grid_origin = [float(getattr(comp, a)[0].item()) for a in anames]
        kin_static = {
            'body_ids':     body_ids,
            'local_lt':     local_lt,
            'local_lr':     local_lr,
            'local_center': 0.5 * (sdf_lo + sdf_hi),
            'local_half':   0.5 * (sdf_hi - sdf_lo),
            'grid_origin':  torch.tensor(
                grid_origin, dtype=self.dtype, device=self.device),
            'inv_h':        1.0 / float(comp.h),
            'gs':           torch.tensor(gs, dtype=torch.int64, device=self.device),
            'pad':          3,
            # numpy mirrors used by the host-side per-step assembly.  2-D
            # stores zeros/identity for local_lt/lr so the unified compose
            # einsum is a bit-exact no-op there (R@I = R, urdf + R@0 = urdf).
            'body_ids_np':     body_ids.numpy(),
            'local_lt_np':     local_lt.detach().cpu().numpy().copy(),
            'local_lr_np':     local_lr.detach().cpu().numpy().copy(),
            'local_center_np': (0.5 * (sdf_lo + sdf_hi)).detach().cpu().numpy().copy(),
            'local_half_np':   (0.5 * (sdf_hi - sdf_lo)).detach().cpu().numpy().copy(),
            'grid_origin_np':  np.array(grid_origin, dtype=self.dtype_np),
            'gs_np':           np.array(gs, dtype=np.int64),
        }
        setattr(comp, attr, kin_static)
        return kin_static

    def _stream_static_pack(self, comp, D):
        """Build (once) the packed static per-body device tensors consumed by
        the streaming kernels.  Stored under the exact names ``_kernel_static_2d``
        / ``_kernel_static_3d`` (read by ``forces.py`` and ``solver.fluid_step``).

        ``body_meta`` row layout (generic over the D simulated axes):
        ``[b{a}0 …] + [b{a}_last …] + [inv_d{a} …] + [inv_vol]`` — exactly the
        former per-dim layouts (10 entries in 3-D, 7 in 2-D).
        """
        attr = '_kernel_static_3d' if D == 3 else '_kernel_static_2d'
        if getattr(comp, attr, None) is not None:
            return
        anames = ('x', 'y', 'z')[:D]
        _use_combined_cache = self.fluid_solver._use_kernels

        F_chunks = []
        F_off    = [0]
        shapes   = []
        meta     = []
        if not _use_combined_cache:
            b_chunks = {a: [] for a in anames}
            b_off    = {a: [0] for a in anames}
        for body in comp.bodies:
            m = body._stream_meta
            F_chunks.append(m['F'].flatten())
            F_off.append(F_off[-1] + m['F'].numel())
            shapes.append([int(s) for s in m['F'].shape])
            meta.append(
                [m[f'b{a}0'] for a in anames]
                + [m[f'b{a}_last'] for a in anames]
                + [m[f'inv_d{a}'] for a in anames]
                + [m['inv_vol']]
            )
            if not _use_combined_cache:
                for a in anames:
                    b_chunks[a].append(m[f'b{a}'])
                    b_off[a].append(b_off[a][-1] + m[f'b{a}'].numel())
        F_flat = torch.cat(F_chunks).contiguous()
        sm = {
            'F_flat':      F_flat,
            'F_offsets':   torch.tensor(F_off,  dtype=torch.int64, device=self.device),
            'body_shapes': torch.tensor(shapes, dtype=torch.int64, device=self.device),
            'body_meta':   torch.tensor(meta,   dtype=self.dtype,  device=self.device),
        }
        if not _use_combined_cache:
            for a in anames:
                sm[f'b{a}_flat']    = torch.cat(b_chunks[a]).contiguous()
                sm[f'b{a}_offsets'] = torch.tensor(
                    b_off[a], dtype=torch.int64, device=self.device)
        # De-duplicate per-body body-template SDFs into the packed F_flat buffer.
        for b, body in enumerate(comp.bodies):
            m = body._stream_meta
            m['F'] = F_flat[F_off[b]:F_off[b + 1]].view(*shapes[b])
        del F_chunks
        if not _use_combined_cache:
            del b_chunks
        setattr(comp, attr, sm)

    def _stream_lagrangian_refresh(self, comp, D, body_R_np, body_pos_np):
        """World-frame refresh of the Lagrangian surface markers (kernel path).

        Batched ``bmm`` over a cached flat pack: 3-D rotates the triangle
        centroids/normals (``tri_centroid/normal_world``), 2-D rotates the
        contour (``cnt_update``).  Skipped entirely for the Eulerian path.
        """
        if D == 3:
            R_b_t   = torch.from_numpy(np.ascontiguousarray(body_R_np)).to(self.device)
            pos_b_t = torch.from_numpy(np.ascontiguousarray(body_pos_np)).to(self.device)
            pack = getattr(comp, '_lagr_tri_pack', None)
            if pack is None:
                cs, ns, idxs, offs, valid = [], [], [], [0], []
                for b, body in enumerate(comp.bodies):
                    tcl = getattr(body, 'tri_centroid_local', None)
                    tnl = getattr(body, 'tri_normal_local', None)
                    if tcl is None or tnl is None:
                        offs.append(offs[-1]); valid.append(False); continue
                    tcl_d = tcl.to(dtype=self.dtype, device=self.device)
                    tnl_d = tnl.to(dtype=self.dtype, device=self.device)
                    cs.append(tcl_d + self._lagr_marker_offset(comp, b, body))
                    ns.append(tnl_d)
                    idxs.append(torch.full((tcl_d.shape[1],), b, dtype=torch.long, device=self.device))
                    offs.append(offs[-1] + tcl_d.shape[1]); valid.append(True)
                pack = {
                    'cen':      torch.cat(cs, dim=1) if cs
                                else torch.empty(3, 0, dtype=self.dtype, device=self.device),
                    'nrm':      torch.cat(ns, dim=1) if ns
                                else torch.empty(3, 0, dtype=self.dtype, device=self.device),
                    'body_idx': torch.cat(idxs) if idxs
                                else torch.empty(0, dtype=torch.long, device=self.device),
                    'offs': offs, 'valid': valid,
                }
                comp._lagr_tri_pack = pack
            cen = pack['cen']
            if cen.shape[1] > 0:
                nrm = pack['nrm']; body_idx = pack['body_idx']
                offs = pack['offs']; valid = pack['valid']
                Rm     = R_b_t.index_select(0, body_idx)
                cw     = torch.bmm(Rm, cen.transpose(0, 1).unsqueeze(-1)).squeeze(-1) \
                         + pos_b_t.index_select(0, body_idx)
                nw     = torch.bmm(Rm, nrm.transpose(0, 1).unsqueeze(-1)).squeeze(-1)
                cw_flat = cw.transpose(0, 1).contiguous()
                nw_flat = nw.transpose(0, 1).contiguous()
                for b, body in enumerate(comp.bodies):
                    if valid[b]:
                        body.tri_centroid_world = cw_flat[:, offs[b]:offs[b + 1]]
                        body.tri_normal_world   = nw_flat[:, offs[b]:offs[b + 1]]
        else:
            # 2-D: body_R_np == R_link_np and body_pos_np == urdf_pos_np
            # (identity local frame), so these are the same world transforms.
            R_t    = torch.from_numpy(np.ascontiguousarray(body_R_np)).to(self.device)
            urdf_t = torch.from_numpy(np.ascontiguousarray(body_pos_np)).to(self.device)
            pack = getattr(comp, '_lagr_cnt_pack', None)
            if pack is None:
                locals_, idxs, offs, valid = [], [], [0], []
                for b, body in enumerate(comp.bodies):
                    if body.cnt is None or body.cnt.numel() == 0:
                        offs.append(offs[-1]); valid.append(False); continue
                    cl = body.cnt
                    dx = torch.roll(cl[0], -1) - cl[0]
                    dy = torch.roll(cl[1], -1) - cl[1]
                    cx_mid = 0.5 * (cl[0] + torch.roll(cl[0], -1))
                    cy_mid = 0.5 * (cl[1] + torch.roll(cl[1], -1))
                    signed_area = (0.5 * (cx_mid * dy - cy_mid * dx).sum()).item()
                    if signed_area < 0:
                        body.cnt = body.cnt.flip(dims=[1]).contiguous()
                    body._cnt_ccw_oriented = True
                    cl = body.cnt.to(dtype=self.dtype, device=self.device)
                    locals_.append(cl)
                    idxs.append(torch.full((cl.shape[1],), b, dtype=torch.long, device=self.device))
                    offs.append(offs[-1] + cl.shape[1]); valid.append(True)
                pack = {
                    'local_flat': torch.cat(locals_, dim=1) if locals_
                                  else torch.empty(2, 0, dtype=self.dtype, device=self.device),
                    'body_idx':   torch.cat(idxs) if idxs
                                  else torch.empty(0, dtype=torch.long, device=self.device),
                    'offs': offs, 'valid': valid,
                }
                comp._lagr_cnt_pack = pack
            local_flat = pack['local_flat']
            if local_flat.shape[1] > 0:
                body_idx = pack['body_idx']; offs = pack['offs']; valid = pack['valid']
                Rm    = R_t.index_select(0, body_idx)
                clq   = local_flat.transpose(0, 1).unsqueeze(-1)
                world = torch.bmm(Rm, clq).squeeze(-1) + urdf_t.index_select(0, body_idx)
                world_flat = world.transpose(0, 1).contiguous()
                for b, body in enumerate(comp.bodies):
                    if valid[b]:
                        body.cnt_update = world_flat[:, offs[b]:offs[b + 1]]

    def _update_streaming_multi(self, t, iteration, dt=1):
        """Unified batched multi-body streaming SDF update (2-D + 3-D).

        Merges the former ``_update_2d_streaming_multi`` /
        ``_update_3d_streaming_multi`` (Step 6 unification).  Assembles the
        per-step packed kinematics + AABB tensors and stashes them on
        ``comp._kernel_step`` / ``_kernel_static_{2,3}d`` for the fused CUDA
        SDF/coefficient kernels launched in ``FluidSolver.fluid_step``.  The
        per-dim packed layouts (kin row, body_meta, dirty-AABB keys) are
        reproduced exactly via ``self._sim_axes``; behaviour is unchanged.
        """
        D    = self.ndim
        fs   = self.fluid_solver
        comp = fs.composite_body
        gs   = fs.grid_shape
        _FAR = 1e4
        B    = len(comp.bodies)
        anames = ('x', 'y', 'z')[:D]

        com_poses, urdf_poses, Rs, lin_vels, ang_vels = self.gather_data(iteration)

        g_1d = tuple(getattr(comp, f'g{a}_1d') for a in anames)

        if not fs._use_kernels:
            raise RuntimeError(
                f"The legacy sparse {D}-D kernel path has been removed; "
                f"use solver_method='kernel' or 'python'.")

        # ---- static per-body descriptors (cached) ----
        kin_static = self._stream_kin_static(comp, B, gs, D)
        self._stream_static_pack(comp, D)
        body_ids_np = kin_static['body_ids_np']

        # ---- gather per-body kinematics on the host (numpy) ----
        urdf_pos_np = np.empty((B, D), dtype=self.dtype_np)
        com_pos_np  = np.empty((B, D), dtype=self.dtype_np)
        R_link_np   = np.empty((B, D, D), dtype=self.dtype_np)
        lin_vel_np  = np.empty((B, D), dtype=self.dtype_np)
        ang_vel_np  = np.empty((B, 3) if D == 3 else (B,), dtype=self.dtype_np)
        for b in range(B):
            a_id = int(body_ids_np[b, 0]); l_id = int(body_ids_np[b, 1])
            urdf_pos_np[b] = urdf_poses[a_id][l_id]
            com_pos_np[b]  = com_poses[a_id][l_id]
            R_link_np[b]   = Rs[a_id][l_id]
            lin_vel_np[b]  = lin_vels[a_id][l_id]
            ang_vel_np[b]  = ang_vels[a_id][l_id]

        # ---- compose body frame: body_pos = urdf + R@lt; body_R = R@lr ----
        local_lt_np = kin_static['local_lt_np']
        local_lr_np = kin_static['local_lr_np']
        body_pos_np = urdf_pos_np + np.einsum('bij,bj->bi', R_link_np, local_lt_np)
        body_R_np   = np.einsum('bij,bjk->bik', R_link_np, local_lr_np)

        # ---- Lagrangian world-frame marker refresh (per-dim) ----
        if self.force_method == "lagrangian":
            self._stream_lagrangian_refresh(comp, D, body_R_np, body_pos_np)

        # ---- vectorised world-space AABB of the oriented body-SDF box ----
        local_center_np = kin_static['local_center_np']
        local_half_np   = kin_static['local_half_np']
        g0_np  = kin_static['grid_origin_np']
        inv_h  = kin_static['inv_h']
        gs_np  = kin_static['gs_np']
        pad    = kin_static['pad']

        abs_R_np     = np.abs(body_R_np)
        world_half   = np.einsum('bij,bj->bi', abs_R_np, local_half_np)
        world_center = np.einsum('bij,bj->bi', body_R_np, local_center_np) + body_pos_np
        w_min = world_center - world_half
        w_max = world_center + world_half

        i_lo = np.floor((w_min - g0_np) * inv_h).astype(np.int64) - pad
        i_hi = np.floor((w_max - g0_np) * inv_h).astype(np.int64) + 1 + pad
        np.clip(i_lo, 0, None, out=i_lo)
        np.minimum(i_hi, gs_np[None, :], out=i_hi)
        if D == 2:
            # Clamp partially/entirely out-of-bounds bodies to a zero-size AABB.
            np.maximum(i_hi, i_lo, out=i_hi)

        dims     = i_hi - i_lo
        sub_vol  = dims.prod(axis=1)
        full_vol = int(np.prod(gs_np))
        fallback = sub_vol > int(0.9 * full_vol)   # >90 % grid -> full grid
        if fallback.any():
            i_lo[fallback, :] = 0
            i_hi[fallback, :] = gs_np
            dims = i_hi - i_lo

        max_vol = int(dims.prod(axis=1).max()) if B > 0 else 0

        # ---- per-body AABB metadata + current/dirty union AABB ----
        for b in range(B):
            lo = [int(i_lo[b, ax]) for ax in range(D)]
            dm = [int(dims[b, ax]) for ax in range(D)]
            comp._body_aabbs[b] = tuple(
                v for ax in range(D) for v in (lo[ax], lo[ax] + dm[ax]))

        if B > 0:
            curr_lo = [int(i_lo[:, ax].min()) for ax in range(D)]
            curr_hi = [int(i_hi[:, ax].max()) for ax in range(D)]
        else:
            curr_lo = [0] * D
            curr_hi = [int(gs[ax]) for ax in range(D)]

        prev = getattr(comp, '_combined_union_aabb', None)
        if prev is not None:
            d_lo = [min(prev[2 * ax],     curr_lo[ax]) for ax in range(D)]
            d_hi = [max(prev[2 * ax + 1], curr_hi[ax]) for ax in range(D)]
        else:
            # First step: comp.sdf_val starts at _FAR everywhere, so the
            # current union AABB alone is a safe dirty region.
            d_lo = list(curr_lo)
            d_hi = list(curr_hi)

        # Reset running-min CC SDF in the dirty sub-block (O(dirty_vol)).
        comp._sdf_sparse = [None] * B
        comp.sdf_val[tuple(slice(d_lo[ax], d_hi[ax]) for ax in range(D))].fill_(_FAR)

        # ---- assemble per-step kin on the host ----
        # Layout: [R^T (D*D) | body_pos (D) | com_pos (D) | lin_vel (D) | ang (3|1)].
        wRT   = D * D
        ang_w = 3 if D == 3 else 1
        kin_np = np.empty((B, wRT + 3 * D + ang_w), dtype=self.dtype_np)
        kin_np[:, 0:wRT] = body_R_np.transpose(0, 2, 1).reshape(B, wRT)
        o = wRT
        kin_np[:, o:o + D] = body_pos_np; o += D
        kin_np[:, o:o + D] = com_pos_np;  o += D
        kin_np[:, o:o + D] = lin_vel_np;  o += D
        if D == 3:
            kin_np[:, o:o + 3] = ang_vel_np
        else:
            kin_np[:, o] = ang_vel_np

        # ---- single H2D transfer for the packed per-step tensors ----
        kin       = torch.from_numpy(np.ascontiguousarray(kin_np)).to(self.device)
        aabb_lo   = torch.from_numpy(np.ascontiguousarray(i_lo)).to(self.device)
        aabb_dim  = torch.from_numpy(np.ascontiguousarray(dims)).to(self.device)
        com_pos_t = torch.from_numpy(np.ascontiguousarray(com_pos_np)).to(self.device)

        # ---- maintain comp.com_pos / body.com_pos views for downstream code ----
        for b, body in enumerate(comp.bodies):
            comp.com_pos[b] = com_pos_t[b]
            body.com_pos = comp.com_pos[b]

        # Kernel path does not populate per-body CC-SDF slabs.
        for b in range(B):
            comp._sdf_sparse[b] = None
        # Cache the union AABB so _compute_union_aabb can use the cheap
        # sub-block mu/normals path without reading _sdf_sparse.
        comp._combined_union_aabb = tuple(
            v for ax in range(D) for v in (curr_lo[ax], curr_hi[ax]))
        # Kernel forces are evaluated later from post-fluid-step fields.
        comp._combined_forces_out = None

        # Stash per-step metadata for FluidSolver.fluid_step (Kernel A streaming
        # SDF + Kernel B fused BDIM2/Poisson coefficients over the dirty block).
        kstep = {
            'kin':      kin,
            'aabb_lo':  aabb_lo,
            'aabb_dim': aabb_dim,
            'max_vol':  max_vol,
        }
        for ax, a in enumerate(anames):
            kstep[f'g{a}'] = g_1d[ax]
        kstep['dirty_i0'] = int(d_lo[0])
        kstep['dirty_j0'] = int(d_lo[1])
        kstep['dirty_Ai'] = int(d_hi[0] - d_lo[0])
        kstep['dirty_Aj'] = int(d_hi[1] - d_lo[1])
        if D == 3:
            kstep['dirty_k0'] = int(d_lo[2])
            kstep['dirty_Ak'] = int(d_hi[2] - d_lo[2])
        comp._kernel_step = kstep

    # ------------------------------------------------------------------
    #  Streaming combined-CUDA 3-D SDF update (Phase B)
    # ------------------------------------------------------------------
    def _update_3d_streaming_multi(self, t, iteration, dt=1):
        """Phase C: batched multi-body streaming SDF update.

        Single Python op call dispatches B per-body kernels in C++,
        eliminating B torch.ops dispatches/step.  Requires that ALL
        bodies expose `_stream_meta` (regular-grid SDF).
        """

        fs   = self.fluid_solver
        comp = fs.composite_body
        gs   = fs.grid_shape
        _FAR = 1e4
        B    = len(comp.bodies)

        com_poses, urdf_poses, Rs, lin_vels, ang_vels = self.gather_data(
            iteration
        )

        h_grid = float(comp.h)

        # Phase I removed the persistent CC normal buffers from the
        # kernel-mode 3-D path: ``streaming_sdf_forces_post_3d`` derives
        # surface normals internally from the body SDF table at query
        # time, so the solver no longer needs ``fs.normal_x/y/z``.  No
        # defensive init is required here.

        # ------------------------------------------------------------------
        # Compute per-body AABBs on CPU (all numpy, no GPU I/O) BEFORE any
        # GPU fills.  This lets us restrict the fills and the CUDA
        # init/decode passes to the dirty sub-block = prev∪curr union-AABB,
        # reducing them from O(Nx*Ny*Nz) to O(dirty_vol).
        # ------------------------------------------------------------------
        kin_static = getattr(comp, '_stream_kin_static', None)
        if kin_static is None or 'local_lt_np' not in kin_static:
            body_ids = torch.tensor(
                [(int(a), int(l)) for (a, l) in comp.body_ids],
                dtype=torch.int64,
            )
            local_lt = torch.zeros((B, 3), dtype=self.dtype, device=self.device)
            local_lr = torch.eye(
                3, dtype=self.dtype, device=self.device,
            ).repeat(B, 1, 1)
            sdf_lo = torch.empty((B, 3), dtype=self.dtype, device=self.device)
            sdf_hi = torch.empty((B, 3), dtype=self.dtype, device=self.device)
            for b, body in enumerate(comp.bodies):
                lp = getattr(body, 'local_pose', None)
                if lp is not None:
                    lp_t = torch.as_tensor(
                        lp, dtype=self.dtype, device=self.device,
                    )
                    local_lt[b] = lp_t[:3]
                    local_lr[b] = torch.as_tensor(
                        Rotation.from_euler(
                            'xyz', lp_t[3:].detach().cpu().numpy(),
                        ).as_matrix(),
                        dtype=self.dtype,
                        device=self.device,
                    )
                sx, sy, sz = (
                    body._stream_meta['bx'],
                    body._stream_meta['by'],
                    body._stream_meta['bz'],
                )
                sdf_lo[b] = torch.stack(
                    (sx[0], sy[0], sz[0])
                )
                sdf_hi[b] = torch.stack(
                    (sx[-1], sy[-1], sz[-1])
                )
            kin_static = {
                'body_ids':     body_ids,
                'local_lt':     local_lt,
                'local_lr':     local_lr,
                'local_center': 0.5 * (sdf_lo + sdf_hi),
                'local_half':   0.5 * (sdf_hi - sdf_lo),
                'grid_origin':  torch.tensor([
                    float(comp.x[0].item()),
                    float(comp.y[0].item()),
                    float(comp.z[0].item()),
                ], dtype=self.dtype, device=self.device),
                'inv_h':        1.0 / float(comp.h),
                'gs':           torch.tensor(
                    gs, dtype=torch.int64, device=self.device,
                ),
                'pad':          3,
                # numpy mirrors used by the host-side kinematics assembly
                'body_ids_np':     body_ids.numpy(),
                'local_lt_np':     local_lt.detach().cpu().numpy().copy(),
                'local_lr_np':     local_lr.detach().cpu().numpy().copy(),
                'local_center_np': (0.5 * (sdf_lo + sdf_hi)).detach().cpu().numpy().copy(),
                'local_half_np':   (0.5 * (sdf_hi - sdf_lo)).detach().cpu().numpy().copy(),
                'grid_origin_np':  np.array([
                    float(comp.x[0].item()),
                    float(comp.y[0].item()),
                    float(comp.z[0].item()),
                ], dtype=self.dtype_np),
                'gs_np':           np.array(gs, dtype=np.int64),
            }
            comp._stream_kin_static = kin_static

        body_ids_np = kin_static['body_ids_np']

        # Gather per-body kinematics on the host (numpy).
        urdf_pos_np = np.empty((B, 3), dtype=self.dtype_np)
        com_pos_np  = np.empty((B, 3), dtype=self.dtype_np)
        R_link_np   = np.empty((B, 3, 3), dtype=self.dtype_np)
        lin_vel_np  = np.empty((B, 3), dtype=self.dtype_np)
        ang_vel_np  = np.empty((B, 3), dtype=self.dtype_np)
        for b in range(B):
            a_id = int(body_ids_np[b, 0])
            l_id = int(body_ids_np[b, 1])
            urdf_pos_np[b] = urdf_poses[a_id][l_id]
            com_pos_np[b]  = com_poses[a_id][l_id]
            R_link_np[b]   = Rs[a_id][l_id]
            lin_vel_np[b]  = lin_vels[a_id][l_id]
            ang_vel_np[b]  = ang_vels[a_id][l_id]

        # Compose with per-body local pose: body_pos = urdf + R @ lt; body_R = R @ lr
        local_lt_np     = kin_static['local_lt_np']
        local_lr_np     = kin_static['local_lr_np']
        local_center_np = kin_static['local_center_np']
        local_half_np   = kin_static['local_half_np']
        g0_np           = kin_static['grid_origin_np']
        inv_h           = kin_static['inv_h']
        gs_np           = kin_static['gs_np']
        pad             = kin_static['pad']

        body_pos_np = urdf_pos_np + np.einsum('bij,bj->bi', R_link_np, local_lt_np)
        body_R_np   = np.einsum('bij,bjk->bik', R_link_np, local_lr_np)  # (B, 3, 3)

        # Lagrangian 3-D surface-integral forces sample fluid fields at
        # ``body.tri_centroid_world`` with ``body.tri_normal_world``; these
        # MUST be in world coords each step.  Direct analogue of the 2-D
        # ``cnt_update`` refresh below — the kernel-mode streaming update
        # historically left them at the body-local frame from init, which
        # made the 3-D Lagrangian forces ~zero (samples were at wrong
        # world locations).  Cheap: one matmul + add per body, skipped
        # entirely for the default Eulerian path.
        if self.force_method == "lagrangian":
            R_b_t   = torch.from_numpy(np.ascontiguousarray(body_R_np)).to(self.device)
            pos_b_t = torch.from_numpy(np.ascontiguousarray(body_pos_np)).to(self.device)
            # Batched world refresh of every triangle centroid + normal in a
            # single ``bmm`` (see the 2-D ``cnt`` refresh).  Pack the static
            # body-local triangulations once into flat (3, T_total) tensors
            # with a per-triangle body index, then rotate(+translate) once.
            pack = getattr(comp, '_lagr_tri_pack', None)
            if pack is None:
                cs, ns, idxs, offs, valid = [], [], [], [0], []
                for b, body in enumerate(comp.bodies):
                    tcl = getattr(body, 'tri_centroid_local', None)
                    tnl = getattr(body, 'tri_normal_local', None)
                    if tcl is None or tnl is None:
                        offs.append(offs[-1]); valid.append(False); continue
                    tcl_d = tcl.to(dtype=self.dtype, device=self.device)
                    tnl_d = tnl.to(dtype=self.dtype, device=self.device)
                    # Shift bbox-centred mesh markers into the SDF-local frame
                    # so ``R @ cen + body_pos`` lands ON the surface (the raw
                    # ``tcl`` is short by ``R @ local_center``; see
                    # ``_lagr_marker_offset``).  Baked into the cached pack once.
                    cs.append(tcl_d + self._lagr_marker_offset(comp, b, body))
                    ns.append(tnl_d)
                    idxs.append(torch.full((tcl_d.shape[1],), b, dtype=torch.long, device=self.device))
                    offs.append(offs[-1] + tcl_d.shape[1]); valid.append(True)
                pack = {
                    'cen':      torch.cat(cs, dim=1) if cs
                                else torch.empty(3, 0, dtype=self.dtype, device=self.device),
                    'nrm':      torch.cat(ns, dim=1) if ns
                                else torch.empty(3, 0, dtype=self.dtype, device=self.device),
                    'body_idx': torch.cat(idxs) if idxs
                                else torch.empty(0, dtype=torch.long, device=self.device),
                    'offs': offs, 'valid': valid,
                }
                comp._lagr_tri_pack = pack
            cen = pack['cen']
            if cen.shape[1] > 0:
                nrm = pack['nrm']; body_idx = pack['body_idx']
                offs = pack['offs']; valid = pack['valid']
                Rm     = R_b_t.index_select(0, body_idx)                    # (T, 3, 3)
                cw     = torch.bmm(Rm, cen.transpose(0, 1).unsqueeze(-1)).squeeze(-1) \
                         + pos_b_t.index_select(0, body_idx)
                nw     = torch.bmm(Rm, nrm.transpose(0, 1).unsqueeze(-1)).squeeze(-1)
                cw_flat = cw.transpose(0, 1).contiguous()                   # (3, T)
                nw_flat = nw.transpose(0, 1).contiguous()
                for b, body in enumerate(comp.bodies):
                    if valid[b]:
                        body.tri_centroid_world = cw_flat[:, offs[b]:offs[b + 1]]
                        body.tri_normal_world   = nw_flat[:, offs[b]:offs[b + 1]]

        # Vectorised AABB of the oriented body-SDF box in world space.
        abs_R_np     = np.abs(body_R_np)
        world_half   = np.einsum('bij,bj->bi', abs_R_np, local_half_np)
        world_center = np.einsum('bij,bj->bi', body_R_np, local_center_np) + body_pos_np
        w_min = world_center - world_half
        w_max = world_center + world_half

        i_lo = np.floor((w_min - g0_np) * inv_h).astype(np.int64) - pad
        i_hi = np.floor((w_max - g0_np) * inv_h).astype(np.int64) + 1 + pad
        np.clip(i_lo, 0, None, out=i_lo)
        np.minimum(i_hi, gs_np[None, :], out=i_hi)

        dims    = i_hi - i_lo
        sub_vol = dims.prod(axis=1)
        full_vol = int(np.prod(gs_np))
        # Bodies covering >90 % of the grid: fall back to full grid.
        fallback = sub_vol > int(0.9 * full_vol)
        if fallback.any():
            i_lo[fallback, :] = 0
            i_hi[fallback, :] = gs_np
            dims = i_hi - i_lo

        # Dirty region = union(prev_union_aabb, curr_union_aabb).
        # Restricting fills and the CUDA init/decode passes to this sub-block
        # makes body update O(dirty_vol) not O(Nx*Ny*Nz).
        _curr_ui0 = int(i_lo[:, 0].min()) if B > 0 else 0
        _curr_uj0 = int(i_lo[:, 1].min()) if B > 0 else 0
        _curr_uk0 = int(i_lo[:, 2].min()) if B > 0 else 0
        _curr_ui1 = int(i_hi[:, 0].max()) if B > 0 else gs[0]
        _curr_uj1 = int(i_hi[:, 1].max()) if B > 0 else gs[1]
        _curr_uk1 = int(i_hi[:, 2].max()) if B > 0 else gs[2]
        _prev = getattr(comp, '_combined_union_aabb', None)
        if _prev is not None:
            _p_i0, _p_i1, _p_j0, _p_j1, _p_k0, _p_k1 = _prev
            d_i0 = min(_p_i0, _curr_ui0); d_i1 = max(_p_i1, _curr_ui1)
            d_j0 = min(_p_j0, _curr_uj0); d_j1 = max(_p_j1, _curr_uj1)
            d_k0 = min(_p_k0, _curr_uk0); d_k1 = max(_p_k1, _curr_uk1)
        else:
            # First step: there is no previous body footprint to union with.
            # Use the current union AABB only — safe because comp.sdf_val is
            # initialised to _FAR=1e4 ("outside body everywhere") in
            # CompositeBody.__init__, so cells outside this AABB already hold
            # the correct "no body here" value.  Using the current AABB
            # (instead of the full grid) shrinks the dirty_vol-sized int64
            # key buffers in Kernel A by ~1000× for a single-body scene at
            # 512³, eliminating the ~4 GiB first-step memory spike.
            d_i0, d_i1 = _curr_ui0, _curr_ui1
            d_j0, d_j1 = _curr_uj0, _curr_uj1
            d_k0, d_k1 = _curr_uk0, _curr_uk1

        # Reset running-min CC SDF in the dirty sub-block (O(dirty_vol) not
        # O(N)).  Phase I removed the persistent staggered SDF and body
        # velocity buffers: those are now per-step temporaries owned by
        # FluidSolver.fluid_step and Kernel A fills them from _FAR each
        # time the kernel runs, so no Python-side reset is needed here.
        comp._sdf_sparse = [None] * B
        comp.sdf_val[d_i0:d_i1, d_j0:d_j1, d_k0:d_k1].fill_(_FAR)

        gx_1d, gy_1d, gz_1d = comp.gx_1d, comp.gy_1d, comp.gz_1d

        # ------------------------------------------------------------------
        # Build / refresh the static per-body packed device tensors once.
        # ------------------------------------------------------------------
        fs_for_cache = self.fluid_solver
        _use_combined_cache = fs_for_cache._use_kernels

        sm = getattr(comp, '_kernel_static_3d', None)
        if sm is None:
            F_chunks  = []
            F_off  = [0]
            shapes = []
            meta   = []
            if not _use_combined_cache:
                bx_chunks = []; by_chunks = []; bz_chunks = []
                bx_off = [0]; by_off = [0]; bz_off = [0]
            for body in comp.bodies:
                m = body._stream_meta
                F_chunks.append(m['F'].flatten())
                F_off.append(F_off[-1]   + m['F'].numel())
                shapes.append([m['F'].shape[0], m['F'].shape[1], m['F'].shape[2]])
                meta.append([
                    m['bx0'], m['by0'], m['bz0'],
                    m['bx_last'], m['by_last'], m['bz_last'],
                    m['inv_dx'], m['inv_dy'], m['inv_dz'], m['inv_vol'],
                ])
                if not _use_combined_cache:
                    bx_chunks.append(m['bx']); by_chunks.append(m['by']); bz_chunks.append(m['bz'])
                    bx_off.append(bx_off[-1] + m['bx'].numel())
                    by_off.append(by_off[-1] + m['by'].numel())
                    bz_off.append(bz_off[-1] + m['bz'].numel())
            F_flat = torch.cat(F_chunks).contiguous()
            sm = {
                'F_flat':       F_flat,
                'F_offsets':    torch.tensor(F_off,  dtype=torch.int64, device=self.device),
                'body_shapes':  torch.tensor(shapes, dtype=torch.int64, device=self.device),
                'body_meta':    torch.tensor(meta,   dtype=self.dtype,  device=self.device),
            }
            if not _use_combined_cache:
                sm['bx_flat']    = torch.cat(bx_chunks).contiguous()
                sm['bx_offsets'] = torch.tensor(bx_off, dtype=torch.int64, device=self.device)
                sm['by_flat']    = torch.cat(by_chunks).contiguous()
                sm['by_offsets'] = torch.tensor(by_off, dtype=torch.int64, device=self.device)
                sm['bz_flat']    = torch.cat(bz_chunks).contiguous()
                sm['bz_offsets'] = torch.tensor(bz_off, dtype=torch.int64, device=self.device)
            # De-duplicate per-body body-template SDFs: replace each
            # body's `_stream_meta['F']` with a view into the packed
            # `F_flat` buffer.  After this the `torch.cat` copy is the
            # only owner of body-template storage, so when no other
            # references exist the per-body originals can be released
            # by the allocator.  This saves up to Σ_b Mx·My·Mz floats
            # of duplicated body-template memory and therefore scales
            # linearly with the number of bodies.
            for b, body in enumerate(comp.bodies):
                m = body._stream_meta
                Mx, My, Mz = shapes[b]
                m['F'] = F_flat[F_off[b]:F_off[b + 1]].view(Mx, My, Mz)
            # Free the temporary chunk lists eagerly so the original
            # `m['F'].flatten()` views don't pin storage longer than
            # needed.
            del F_chunks
            if not _use_combined_cache:
                del bx_chunks, by_chunks, bz_chunks
            comp._kernel_static_3d = sm

        # ------------------------------------------------------------------
        # Per-step: kin_static, per-body kin, AABB, dirty region, and sub-
        # block fills were all computed above (before the sm cache block).
        # Continue from max_vol and kin assembly.
        # ------------------------------------------------------------------
        max_vol = int(dims.prod(axis=1).max()) if B > 0 else 0

        # Assemble kin on the host.
        # Layout: [R^T (9) | body_pos (3) | com_pos (3) | lin_vel (3) | ang_vel (3)] = 21
        # Row-major flatten of R^T = body_R.transpose(0,2,1).reshape(B,9).
        kin_np = np.empty((B, 21), dtype=self.dtype_np)
        kin_np[:, 0:9]   = body_R_np.transpose(0, 2, 1).reshape(B, 9)
        kin_np[:, 9:12]  = body_pos_np
        kin_np[:, 12:15] = com_pos_np
        kin_np[:, 15:18] = lin_vel_np
        kin_np[:, 18:21] = ang_vel_np

        aabb_lo_np  = np.ascontiguousarray(i_lo)
        aabb_dim_np = np.ascontiguousarray(dims)

        # Single H2D transfer for all packed per-step device tensors.
        kin      = torch.from_numpy(np.ascontiguousarray(kin_np)).to(self.device)
        aabb_lo  = torch.from_numpy(aabb_lo_np).to(self.device)
        aabb_dim = torch.from_numpy(aabb_dim_np).to(self.device)
        com_pos_t = torch.from_numpy(np.ascontiguousarray(com_pos_np)).to(self.device)

        # Update Python-side AABB metadata used downstream (slab split, forces).
        aabbs_for_split = []
        for b in range(B):
            i0 = int(i_lo[b, 0]); j0 = int(i_lo[b, 1]); k0 = int(i_lo[b, 2])
            Ai = int(dims[b, 0]);  Aj = int(dims[b, 1]);  Ak = int(dims[b, 2])
            aabb = (i0, i0 + Ai, j0, j0 + Aj, k0, k0 + Ak)
            comp._body_aabbs[b] = aabb
            aabbs_for_split.append(aabb)

        # Maintain `comp.com_pos[b]` and `body.com_pos` views for downstream code.
        for b, body in enumerate(comp.bodies):
            comp.com_pos[b] = com_pos_t[b]
            body.com_pos = comp.com_pos[b]

        fs = self.fluid_solver
        _use_combined = fs._use_kernels

        if _use_combined:
            # Kernel path does not populate per-body CC-SDF slabs.
            for body_i in range(B):
                comp._sdf_sparse[body_i] = None
            # Cache the union AABB directly so _compute_union_aabb can
            # activate the cheap sub-block mu/normals path without reading
            # _sdf_sparse.  Without this, _compute_union_aabb returns
            # None → _recompute_mu_normals falls into the full-grid
            # CUDA-graph (reduce-overhead) path, which statically holds
            # ~2-3 GB of output + intermediate buffers for the full grid.
            _u_i0 = _u_j0 = _u_k0 = 1 << 30
            _u_i1 = _u_j1 = _u_k1 = -1
            for _i0, _i1, _j0, _j1, _k0, _k1 in aabbs_for_split:
                if _i0 < _u_i0: _u_i0 = _i0
                if _j0 < _u_j0: _u_j0 = _j0
                if _k0 < _u_k0: _u_k0 = _k0
                if _i1 > _u_i1: _u_i1 = _i1
                if _j1 > _u_j1: _u_j1 = _j1
                if _k1 > _u_k1: _u_k1 = _k1
            comp._combined_union_aabb = (_u_i0, _u_i1, _u_j0, _u_j1, _u_k0, _u_k1)

            # Kernel forces are evaluated later from post-fluid-step fields.
            comp._combined_forces_out  = None

            # Stash per-step metadata.  Phase I added the dirty-AABB
            # bounds so FluidSolver.fluid_step can launch Kernel A
            # (streaming SDF + body velocity update) and Kernel B (fused
            # BDIM2 + variable-density Poisson coefficients) over the
            # same sub-block this update prepared.  The SDF kernel call
            # itself has moved into fluid_step.
            comp._kernel_step = {
                'kin':     kin,
                'aabb_lo': aabb_lo,
                'aabb_dim': aabb_dim,
                'max_vol': max_vol,
                'gx':      gx_1d,
                'gy':      gy_1d,
                'gz':      gz_1d,
                'dirty_i0': int(d_i0),
                'dirty_j0': int(d_j0),
                'dirty_k0': int(d_k0),
                'dirty_Ai': int(d_i1 - d_i0),
                'dirty_Aj': int(d_j1 - d_j0),
                'dirty_Ak': int(d_k1 - d_k0),
            }
        else:
            raise RuntimeError("The legacy sparse 3-D kernel path has been removed; use solver_method='kernel' or 'python'.")


    def _update_2d_streaming_multi(self, t, iteration, dt=1):
        """2-D analogue of :meth:`_update_3d_streaming_multi`.

        Single Python op call dispatches B per-body 2-D streaming-SDF
        kernels in C++/CUDA, eliminating B torch.ops dispatches/step.
        Requires that ALL bodies expose ``_stream_meta`` (set by
        ``_init_interp`` for both mesh bodies and analytical
        bodies whose ``local_aabb`` is available).

        Side-effects per call:
            * fills ``comp.sdf_val``, ``comp.sdf_val_u``, ``comp.sdf_val_v``,
              ``comp.body_u``, ``comp.body_v`` (union over all bodies);
            * stashes per-step packed tensors on ``comp._kernel_step``
              for the post-fluid-step 2-D force kernel;
            * splits the sparse cc-SDF into per-body slabs on
              ``comp._sdf_sparse[b] = (aabb, slab)`` for downstream code;
            * maintains ``comp.com_pos[b]``, ``body.com_pos``,
              ``body.cnt_update``, ``body.r_com`` and the optional
              contour mask, exactly like the legacy 2-D path.
        """

        fs   = self.fluid_solver
        comp = fs.composite_body
        gs   = fs.grid_shape
        _FAR = 1e4
        B    = len(comp.bodies)

        com_poses, urdf_poses, Rs, lin_vels, ang_vels = (
            self.gather_data(iteration)
        )

        h_grid = float(comp.h)

        # Phase I removed the persistent CC normal buffers from the
        # kernel-mode 2-D path: ``streaming_sdf_forces_post_2d``
        # derives surface normals internally from the body SDF table at
        # query time, so the solver no longer needs ``fs.normal_x/y``.
        # No defensive init is required here.

        # Reset per-body sparse storage
        comp._sdf_sparse = [None] * B

        gx_1d, gy_1d = comp.gx_1d, comp.gy_1d

        # ──────────────────────────────────────────────────────────
        # Build / refresh the static per-body packed device tensors once.
        # ──────────────────────────────────────────────────────────
        _use_combined_cache = fs._use_kernels

        sm = getattr(comp, '_kernel_static_2d', None)
        if sm is None:
            F_chunks = []
            F_off    = [0]
            shapes   = []
            meta     = []
            if not _use_combined_cache:
                bx_chunks = []; by_chunks = []
                bx_off = [0]; by_off = [0]
            for body in comp.bodies:
                m = body._stream_meta
                F_chunks.append(m['F'].flatten())
                F_off.append(F_off[-1] + m['F'].numel())
                shapes.append([m['F'].shape[0], m['F'].shape[1]])
                meta.append([
                    m['bx0'], m['by0'],
                    m['bx_last'], m['by_last'],
                    m['inv_dx'], m['inv_dy'], m['inv_vol'],
                ])
                if not _use_combined_cache:
                    bx_chunks.append(m['bx']); by_chunks.append(m['by'])
                    bx_off.append(bx_off[-1] + m['bx'].numel())
                    by_off.append(by_off[-1] + m['by'].numel())
            F_flat = torch.cat(F_chunks).contiguous()
            sm = {
                'F_flat':      F_flat,
                'F_offsets':   torch.tensor(F_off,  dtype=torch.int64, device=self.device),
                'body_shapes': torch.tensor(shapes, dtype=torch.int64, device=self.device),
                'body_meta':   torch.tensor(meta,   dtype=self.dtype,  device=self.device),
            }
            if not _use_combined_cache:
                sm['bx_flat']    = torch.cat(bx_chunks).contiguous()
                sm['bx_offsets'] = torch.tensor(bx_off, dtype=torch.int64, device=self.device)
                sm['by_flat']    = torch.cat(by_chunks).contiguous()
                sm['by_offsets'] = torch.tensor(by_off, dtype=torch.int64, device=self.device)
                del bx_chunks, by_chunks
            # De-duplicate per-body body-template SDFs into the packed F_flat buffer.
            for b, body in enumerate(comp.bodies):
                m = body._stream_meta
                Mx, My = shapes[b]
                m['F'] = F_flat[F_off[b]:F_off[b + 1]].view(Mx, My)
            del F_chunks
            comp._kernel_static_2d = sm

        # ──────────────────────────────────────────────────────────
        # Per-step: compose body frames, AABBs, kinematics (numpy host).
        # ──────────────────────────────────────────────────────────
        kin_static = getattr(comp, '_stream_kin_static_2d', None)
        if kin_static is None or 'local_center_np' not in kin_static:
            body_ids = torch.tensor(
                [(int(a), int(l)) for (a, l) in comp.body_ids],
                dtype=torch.int64,
            )
            # 2-D bodies have no `local_pose` concept (rigid-body links
            # in the 2-D projection collapse to identity local frames),
            # so the local translation / rotation are trivial.
            local_lt = torch.zeros((B, 2), dtype=self.dtype, device=self.device)
            local_lr = torch.eye(2, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(B, 1, 1)
            sdf_lo = torch.empty((B, 2), dtype=self.dtype, device=self.device)
            sdf_hi = torch.empty((B, 2), dtype=self.dtype, device=self.device)
            for b, body in enumerate(comp.bodies):
                m = body._stream_meta  # set for both mesh and analytical bodies
                # Prefer the tight contour-based AABB (set by _init_interp for
                # mesh bodies whose SDF table is padded far beyond the BDIM band).
                if 'local_aabb_lo' in m:
                    sdf_lo[b] = m['local_aabb_lo'].to(
                        dtype=self.dtype, device=self.device)
                    sdf_hi[b] = m['local_aabb_hi'].to(
                        dtype=self.dtype, device=self.device)
                else:
                    sdf_lo[b] = torch.stack((
                        torch.as_tensor(m['bx0'], dtype=self.dtype, device=self.device),
                        torch.as_tensor(m['by0'], dtype=self.dtype, device=self.device),
                    ))
                    sdf_hi[b] = torch.stack((
                        torch.as_tensor(m['bx_last'], dtype=self.dtype, device=self.device),
                        torch.as_tensor(m['by_last'], dtype=self.dtype, device=self.device),
                    ))
            kin_static = {
                'body_ids':     body_ids,
                'local_lt':     local_lt,
                'local_lr':     local_lr,
                'local_center': 0.5 * (sdf_lo + sdf_hi),
                'local_half':   0.5 * (sdf_hi - sdf_lo),
                'grid_origin':  torch.tensor([
                    float(comp.x[0].item()),
                    float(comp.y[0].item()),
                ], dtype=self.dtype, device=self.device),
                'inv_h':        1.0 / float(comp.h),
                'gs':           torch.tensor(gs, dtype=torch.int64, device=self.device),
                'pad':          3,
                # numpy mirrors used by the host-side kinematics assembly
                'body_ids_np':     body_ids.numpy(),
                'local_center_np': (0.5 * (sdf_lo + sdf_hi)).detach().cpu().numpy().copy(),
                'local_half_np':   (0.5 * (sdf_hi - sdf_lo)).detach().cpu().numpy().copy(),
                'grid_origin_np':  np.array([
                    float(comp.x[0].item()),
                    float(comp.y[0].item()),
                ], dtype=self.dtype_np),
                'gs_np':           np.array(gs, dtype=np.int64),
            }
            comp._stream_kin_static_2d = kin_static

        body_ids_np = kin_static['body_ids_np']

        # Gather per-body kinematics on the host (numpy) – avoids B×5 small
        # GPU indexed-tensor writes and defers the H2D to a single packed copy.
        # In 2-D, local_lt = 0 and local_lr = I by construction, so
        # body_pos = urdf_pos and body_R = R_link directly.
        urdf_pos_np = np.empty((B, 2), dtype=self.dtype_np)
        com_pos_np  = np.empty((B, 2), dtype=self.dtype_np)
        R_link_np   = np.empty((B, 2, 2), dtype=self.dtype_np)
        lin_vel_np  = np.empty((B, 2), dtype=self.dtype_np)
        ang_vel_np  = np.empty((B,),    dtype=self.dtype_np)
        for b in range(B):
            a_id = int(body_ids_np[b, 0])
            l_id = int(body_ids_np[b, 1])
            urdf_pos_np[b] = urdf_poses[a_id][l_id]
            com_pos_np[b]  = com_poses[a_id][l_id]
            R_link_np[b]   = Rs[a_id][l_id]
            lin_vel_np[b]  = lin_vels[a_id][l_id]
            ang_vel_np[b]  = ang_vels[a_id][l_id]

        # Lagrangian surface-integral forces sample fluid fields at
        # ``body.cnt_update``; that tensor MUST be in world coords for
        # the kernel to read the right fields.  This streaming path (and
        # the contour-mask=False ``_update_2d`` Python path) historically
        # left ``cnt_update`` at the body-local frame from init, which
        # made ``force_method="lagrangian"`` produce finite-but-bogus
        # forces and eventually crashed MuJoCo (mjWARN_BADQACC).  Refresh
        # it here when Lagrangian is enabled — cheap (one matmul + add
        # per body) and skipped entirely for the default Eulerian path.
        if self.force_method == "lagrangian":
            R_t       = torch.from_numpy(np.ascontiguousarray(R_link_np)).to(self.device)
            urdf_t    = torch.from_numpy(np.ascontiguousarray(urdf_pos_np)).to(self.device)
            # Refresh every contour to world coords with a single batched
            # ``bmm`` instead of B small per-body matmuls.  The body-local
            # markers (CCW-oriented, dtype/device-cast) are packed once into
            # a flat (2, M_total) tensor with a per-marker body index; per
            # step we gather R per marker, rotate+translate in one launch,
            # then hand each body a (cheap) view of its slice.
            pack = getattr(comp, '_lagr_cnt_pack', None)
            if pack is None:
                locals_, idxs, offs, valid = [], [], [0], []
                for b, body in enumerate(comp.bodies):
                    if body.cnt is None or body.cnt.numel() == 0:
                        offs.append(offs[-1]); valid.append(False); continue
                    # Ensure CCW orientation (kernel assumes CCW outward
                    # normal via ``nx = ty, ny = -tx``).  Mesh bodies' contour
                    # ordering depends on the source mesh and is often CW;
                    # analytical bodies are built CCW.  Check signed area once.
                    cl = body.cnt
                    dx = torch.roll(cl[0], -1) - cl[0]
                    dy = torch.roll(cl[1], -1) - cl[1]
                    cx_mid = 0.5 * (cl[0] + torch.roll(cl[0], -1))
                    cy_mid = 0.5 * (cl[1] + torch.roll(cl[1], -1))
                    signed_area = (0.5 * (cx_mid * dy - cy_mid * dx).sum()).item()
                    if signed_area < 0:
                        body.cnt = body.cnt.flip(dims=[1]).contiguous()
                    body._cnt_ccw_oriented = True
                    cl = body.cnt.to(dtype=self.dtype, device=self.device)
                    locals_.append(cl)
                    idxs.append(torch.full((cl.shape[1],), b, dtype=torch.long, device=self.device))
                    offs.append(offs[-1] + cl.shape[1]); valid.append(True)
                pack = {
                    'local_flat': torch.cat(locals_, dim=1) if locals_
                                  else torch.empty(2, 0, dtype=self.dtype, device=self.device),
                    'body_idx':   torch.cat(idxs) if idxs
                                  else torch.empty(0, dtype=torch.long, device=self.device),
                    'offs': offs, 'valid': valid,
                }
                comp._lagr_cnt_pack = pack
            local_flat = pack['local_flat']
            if local_flat.shape[1] > 0:
                body_idx = pack['body_idx']; offs = pack['offs']; valid = pack['valid']
                Rm    = R_t.index_select(0, body_idx)                       # (M, 2, 2)
                clq   = local_flat.transpose(0, 1).unsqueeze(-1)            # (M, 2, 1)
                world = torch.bmm(Rm, clq).squeeze(-1) + urdf_t.index_select(0, body_idx)
                world_flat = world.transpose(0, 1).contiguous()            # (2, M)
                for b, body in enumerate(comp.bodies):
                    if valid[b]:
                        body.cnt_update = world_flat[:, offs[b]:offs[b + 1]]

        # body_pos = urdf_pos (local_lt = 0); body_R = R_link (local_lr = I)
        body_pos_np = urdf_pos_np
        body_R_np   = R_link_np

        # Vectorised AABB of the oriented body-SDF box in world space.
        local_center_np = kin_static['local_center_np']
        local_half_np   = kin_static['local_half_np']
        g0_np           = kin_static['grid_origin_np']
        inv_h           = kin_static['inv_h']
        gs_np           = kin_static['gs_np']
        pad             = kin_static['pad']

        abs_R_np     = np.abs(body_R_np)
        world_half   = np.einsum('bij,bj->bi', abs_R_np, local_half_np)
        world_center = np.einsum('bij,bj->bi', body_R_np, local_center_np) + body_pos_np
        w_min = world_center - world_half
        w_max = world_center + world_half

        i_lo = np.floor((w_min - g0_np) * inv_h).astype(np.int64) - pad
        i_hi = np.floor((w_max - g0_np) * inv_h).astype(np.int64) + 1 + pad
        np.clip(i_lo, 0, None, out=i_lo)
        np.minimum(i_hi, gs_np[None, :], out=i_hi)
        # Bodies partially or entirely outside the grid produce i_hi < i_lo.
        # Clamp to zero-size AABB for out-of-bounds bodies.
        np.maximum(i_hi, i_lo, out=i_hi)

        dims    = i_hi - i_lo
        sub_vol = dims.prod(axis=1)
        full_vol = int(np.prod(gs_np))
        # Bodies covering >90 % of the grid: fall back to full grid.
        fallback = sub_vol > int(0.9 * full_vol)
        if fallback.any():
            i_lo[fallback, :] = 0
            i_hi[fallback, :] = gs_np
            dims = i_hi - i_lo

        max_vol = int(dims.prod(axis=1).max()) if B > 0 else 0

        # Assemble kin on the host.
        # 2-D kin row layout: [R^T (4) | bp (2) | cm (2) | lv (2) | omega (1)] = 11
        kin_np = np.empty((B, 11), dtype=self.dtype_np)
        kin_np[:, 0:4]  = body_R_np.transpose(0, 2, 1).reshape(B, 4)
        kin_np[:, 4:6]  = body_pos_np
        kin_np[:, 6:8]  = com_pos_np
        kin_np[:, 8:10] = lin_vel_np
        kin_np[:, 10]   = ang_vel_np

        aabb_lo_np  = np.ascontiguousarray(i_lo)
        aabb_dim_np = np.ascontiguousarray(dims)

        # Single H2D transfer for all packed per-step device tensors.
        kin       = torch.from_numpy(np.ascontiguousarray(kin_np)).to(self.device)
        aabb_lo   = torch.from_numpy(aabb_lo_np).to(self.device)
        aabb_dim  = torch.from_numpy(aabb_dim_np).to(self.device)
        com_pos_t = torch.from_numpy(np.ascontiguousarray(com_pos_np)).to(self.device)

        # Update Python-side AABB metadata used downstream.
        aabbs_for_split = []
        for b in range(B):
            i0 = int(i_lo[b, 0]); j0 = int(i_lo[b, 1])
            Ai = int(dims[b, 0]);  Aj = int(dims[b, 1])
            aabb = (i0, i0 + Ai, j0, j0 + Aj)
            comp._body_aabbs[b] = aabb
            aabbs_for_split.append(aabb)

        # Compute current union AABB (i0, i1, j0, j1).
        _curr_ui0 = _curr_uj0 = 1 << 30
        _curr_ui1 = _curr_uj1 = -1
        for _i0, _i1, _j0, _j1 in aabbs_for_split:
            if _i0 < _curr_ui0: _curr_ui0 = _i0
            if _j0 < _curr_uj0: _curr_uj0 = _j0
            if _i1 > _curr_ui1: _curr_ui1 = _i1
            if _j1 > _curr_uj1: _curr_uj1 = _j1

        # Dirty region = prev union AABB ∪ curr union AABB.
        _prev_union = getattr(comp, '_combined_union_aabb', None)
        if _prev_union is not None:
            _p_i0, _p_i1, _p_j0, _p_j1 = _prev_union
            d_i0 = min(_p_i0, _curr_ui0)
            d_i1 = max(_p_i1, _curr_ui1)
            d_j0 = min(_p_j0, _curr_uj0)
            d_j1 = max(_p_j1, _curr_uj1)
        else:
            # First step: use current union AABB only.  Safe because
            # comp.sdf_val starts at the _FAR sentinel (see CompositeBody),
            # so cells outside this AABB already encode "no body here".
            # Matches the 3-D path's first-step optimisation.
            d_i0, d_i1 = _curr_ui0, _curr_ui1
            d_j0, d_j1 = _curr_uj0, _curr_uj1

        # Phase I: in kernel mode the staggered SDF and body-velocity
        # tensors are per-step temporaries owned by
        # FluidSolver.fluid_step (Kernel A fills them from _FAR each
        # call), so no Python-side reset is needed here.  Only the CC
        # SDF dirty sub-block is wiped — Kernel A still writes into the
        # persistent ``comp.sdf_val`` for the force kernel to consume.
        comp.sdf_val[d_i0:d_i1, d_j0:d_j1].fill_(_FAR)

        # Maintain `comp.com_pos[b]` views for downstream code.
        for b, body in enumerate(comp.bodies):
            comp.com_pos[b] = com_pos_t[b]
            body.com_pos = comp.com_pos[b]

        fs = self.fluid_solver

        if fs._use_kernels:
            for b in range(B):
                comp._sdf_sparse[b] = None

            # Cache the union AABB so _compute_union_aabb can activate
            # the sub-block path without reading _sdf_sparse.
            _u_i0 = _curr_ui0
            _u_j0 = _curr_uj0
            _u_i1 = _curr_ui1
            _u_j1 = _curr_uj1
            comp._combined_union_aabb = (_u_i0, _u_i1, _u_j0, _u_j1)

            # 2-D force integration is computed in the later force stage by
            # the native Phase-D-only op, so the update stage only maintains
            # the memory-saving geometry state here.
            comp._combined_forces_out = None

            # Stash per-step metadata.  Phase I added the dirty-AABB
            # bounds so FluidSolver.fluid_step can launch Kernel A
            # (streaming SDF + body velocity update) and Kernel B (fused
            # BDIM2 + variable-density Poisson coefficients) over the
            # same sub-block this update prepared.  The SDF kernel call
            # itself has moved into fluid_step.
            comp._kernel_step = {
                'kin':      kin,
                'aabb_lo':  aabb_lo,
                'aabb_dim': aabb_dim,
                'max_vol':  max_vol,
                'gx':       gx_1d,
                'gy':       gy_1d,
                'dirty_i0': int(d_i0),
                'dirty_j0': int(d_j0),
                'dirty_Ai': int(d_i1 - d_i0),
                'dirty_Aj': int(d_j1 - d_j0),
            }

        else:
            raise RuntimeError("The legacy sparse 2-D kernel path has been removed; use solver_method='kernel' or 'python'.")


    # ==================================================================
    #  apply_forces: fluid -> body forces via MuJoCo xfrc_applied
    # ==================================================================
    def _assemble_loads_scaled(self):
        """Per-body force-scaled hydrodynamic loads ``(lin_total, ang_total)``.

        Returns numpy ``(D, B)`` / ``(Nt, B)`` arrays = ``force_scaling`` *
        (viscous + pressure), in solver load units (before ``units.newtons``).
        Shared by the explicit apply path and the implicit coupling vector.
        """
        fs = self.fluid_solver
        s  = self.force_scaling
        D  = self.ndim
        Nt = len(self._ang_xfrc_idx)

        # Single GPU→CPU transfer (4 groups: lin_visc, ang_visc, lin_pres, ang_pres).
        attrs = (self._lin_visc_attrs + self._ang_visc_attrs
                 + self._lin_pres_attrs + self._ang_pres_attrs)
        forces_gpu = torch.stack([getattr(fs, a) for a in attrs])  # (2D + 2Nt, B)
        forces_cpu = (s * forces_gpu).cpu().numpy()                # single sync

        lin_total = forces_cpu[:D] + forces_cpu[D + Nt: 2 * D + Nt]   # (D, B)
        ang_total = forces_cpu[D: D + Nt] + forces_cpu[2 * D + Nt:]   # (Nt, B)
        return lin_total, ang_total

    def _apply_forces(self, task, physics, loads=None):
        """Dim-agnostic fluid → body force application via MuJoCo xfrc.

        Per-D xfrc index map, buoyancy axis, and FluidSolver field names
        are precomputed in :meth:`_init_apply_forces`, so this method
        has zero per-step Python branching on ``ndim`` or ``_2d_plane``.

        FARMS-style buoyancy is applied additively to a single linear
        xfrc index (``_buoyancy_xfrc_idx``); for the 2-D xy plane, where
        no buoyancy is needed, that index is ``None`` and the buoyancy
        block is skipped entirely.

        When ``loads is None`` (explicit path) the per-body loads are read
        from the FluidSolver and the ``force_relaxation`` low-pass is
        applied; the method only *reads* cached force tensors, so it is safe
        to call once per MuJoCo substep to keep ``xfrc_applied`` fresh.
        When ``loads=(lin_total, ang_total)`` is given (implicit coupling)
        those scaled loads are written verbatim (no low-pass) — the
        strongly-coupled fixed point has already converged the force.
        """
        D  = self.ndim
        Nt = len(self._ang_xfrc_idx)
        fs = self.fluid_solver

        if loads is None:
            lin_total, ang_total = self._assemble_loads_scaled()

            # Temporal under-relaxation: low-pass at the coupling boundary
            # to damp the explicit-coupling oscillation while preserving
            # the DC / time-averaged physical force.
            beta = self.force_relaxation
            if beta < 1.0:
                if self._fr_lin_prev is None or self._fr_lin_prev.shape != lin_total.shape:
                    self._fr_lin_prev = lin_total.copy()
                    self._fr_ang_prev = ang_total.copy()
                else:
                    lin_total = beta * lin_total + (1.0 - beta) * self._fr_lin_prev
                    ang_total = beta * ang_total + (1.0 - beta) * self._fr_ang_prev
                    self._fr_lin_prev = lin_total.copy()
                    self._fr_ang_prev = ang_total.copy()
        else:
            lin_total, ang_total = loads

        # FARMS-identical buoyancy (drag.pyx ``compute_buoyancy``).
        if self._has_buoyancy and not self._buoyancy_initialized:
            self._init_buoyancy_params(task, physics)

        comp    = fs.composite_body
        surface = self.water_surface
        g_z     = self.gravity_z
        units_N = task.units.newtons
        buoy_xidx = self._buoyancy_xfrc_idx
        buoy_pidx = self._buoyancy_pos_idx
        has_buoy  = self._has_buoyancy

        for body_i in range(len(comp.bodies)):
            (animat_id, link_id) = comp.body_ids[body_i]
            ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            buoyancy = 0.0
            if has_buoy:
                mass    = self._buoy_mass[body_i]
                density = self._buoy_density[body_i]
                height  = self._buoy_height[body_i]
                pos_z   = float(comp.com_pos[body_i][buoy_pidx])
                if mass > 0 and height > 0 and pos_z - height < surface:
                    frac = min((surface + height - pos_z) / (2.0 * height), 1.0)
                    buoyancy = -self.rho_fluid * mass * g_z / density * frac

            for d, xidx in enumerate(self._lin_xfrc_idx):
                val = lin_total[d][body_i] * units_N
                if xidx == buoy_xidx:
                    val += buoyancy * units_N
                physics.data.xfrc_applied[ind, xidx] = val

            for d, xidx in enumerate(self._ang_xfrc_idx):
                physics.data.xfrc_applied[ind, xidx] = ang_total[d][body_i] * units_N

    # ==================================================================
    #  step: one full coupled step (called by FluidExtension.before_step)
    # ==================================================================
    def step(self, task, physics):
        """Dispatch to the explicit or implicit (strong) coupling step."""
        if self._coupling_scheme == "implicit":
            self._step_implicit(task, physics)
        else:
            self._step_explicit(task, physics)

    def _step_explicit(self, task, physics):
        """Weakly-coupled (explicit) step: advance fluid once, push loads."""
        iteration = self.iteration
        timestep  = self.pars["solver"]["dt"]
        if iteration >= self.pars["solver"]["nt"]:
            return

        t  = iteration * timestep
        fs = self.fluid_solver

        if not fs.terminate:
            if self.ndim == 3:
                (fs.u0, fs.v0, fs.p0, fs.w0, fs.terminate) = fs.step_(
                    fs.u0, fs.v0, fs.p0, iteration, t, w_vel=fs.w0
                )
            else:
                (fs.u0, fs.v0, fs.p0, fs.terminate) = fs.step_(
                    fs.u0, fs.v0, fs.p0, iteration, t
                )

            fs.__dict__.update(_FS_FREE_AFTER_FORCES_3D)
            self.apply_forces(task, physics)

        self.iteration += 1

    # ==================================================================
    #  implicit (strongly-coupled) step  — see STRONG_COUPLING_FARMS_DESIGN.md
    # ==================================================================
    def _fluid_snapshot(self):
        fs = self.fluid_solver
        snap = {"u0": fs.u0.clone(), "v0": fs.v0.clone(), "p0": fs.p0.clone()}
        if self.ndim == 3:
            snap["w0"] = fs.w0.clone()
        return snap

    def _fluid_restore(self, snap):
        fs = self.fluid_solver
        fs.u0 = snap["u0"].clone()
        fs.v0 = snap["v0"].clone()
        fs.p0 = snap["p0"].clone()
        if self.ndim == 3:
            fs.w0 = snap["w0"].clone()

    def _loads_to_vector(self):
        """Flatten the current scaled solver loads into the coupling vector
        ``[lin_total(D,B).ravel(), ang_total(Nt,B).ravel()]``."""
        lin_total, ang_total = self._assemble_loads_scaled()
        return np.concatenate([lin_total.ravel(), ang_total.ravel()]), \
            (lin_total.shape, ang_total.shape)

    def _vector_to_loads(self, x, shapes):
        (lin_shape, ang_shape) = shapes
        nlin = int(np.prod(lin_shape))
        lin_total = np.asarray(x[:nlin], dtype=np.float64).reshape(lin_shape)
        ang_total = np.asarray(x[nlin:], dtype=np.float64).reshape(ang_shape)
        return lin_total, ang_total

    def _advance_fluid_at_current_pose(self, iteration, t):
        """One fluid solve at the body pose currently in ``physics.data``
        (read via the ``physics`` pose source).  Sets the solver loads."""
        fs = self.fluid_solver
        if self.ndim == 3:
            fs.advance_and_compute_loads(fs.u0, fs.v0, fs.p0, iteration, t, w_vel=fs.w0)
        else:
            fs.advance_and_compute_loads(fs.u0, fs.v0, fs.p0, iteration, t)

    def _step_implicit(self, task, physics):
        """Strongly-coupled step (preCICE-style quasi-Newton on the force).

        Runs the fixed-point loop entirely inside ``before_step`` using
        throwaway MuJoCo predictions (each undone by checkpoint restore),
        then leaves MuJoCo at the start-of-step state with ``xfrc_applied``
        holding the converged force — so the runtime integrates once
        (Option A).  Couples on the per-body scaled hydrodynamic load.
        """
        iteration = self.iteration
        timestep  = self.pars["solver"]["dt"]
        if iteration >= self.pars["solver"]["nt"]:
            return
        t  = iteration * timestep
        fs = self.fluid_solver
        self._task, self._physics = task, physics

        if fs.terminate:
            self.iteration += 1
            return

        if self._has_buoyancy and not self._buoyancy_initialized:
            self._init_buoyancy_params(task, physics)

        if self._mj_ckpt is None:
            self._mj_ckpt = _MujocoCheckpoint(physics)
        ckpt = self._mj_ckpt
        acc  = self.accelerator
        nsub = self._predict_nstep

        # ---- checkpoints at start-of-step state sⁿ ----
        mj_n    = ckpt.save()
        fluid_n = self._fluid_snapshot()

        # coupling-vector shape: (lin (D, B), ang (Nt, B))
        B = len(fs.composite_body.bodies)
        shapes = ((self.ndim, B), (len(self._ang_xfrc_idx), B))

        # Read poses live from physics so each prediction step is visible to
        # the fluid without a task.update_sensors round-trip.
        prev_src = self._pose_source
        self._pose_source = "physics"

        x = self._implicit_prev            # warm-start (None on the first step)
        x_tilde = None
        converged = False
        # Robustness bookkeeping: track the best (min-residual, finite) load
        # so a non-converging / oscillating step commits a sane force rather
        # than a blown-up quasi-Newton iterate.  ``load_scale`` bounds the
        # candidate force so a runaway step can never be applied to MuJoCo
        # (which would teleport the body and crash the kernel AABB).
        best_x, best_res = None, np.inf
        load_scale = 1.0
        diverged = False

        for k in range(1, self.max_iter + 1):
            # ---- guard: never apply a non-finite or runaway candidate ----
            if x is not None and (
                    not np.all(np.isfinite(x))
                    or np.linalg.norm(x) > self._impl_force_bound * load_scale):
                diverged = True
                break

            ckpt.restore(mj_n)              # MuJoCo -> sⁿ  (also mj_forward)
            self._fluid_restore(fluid_n)    # fluid   -> fluidⁿ

            # ---- structure: integrate sⁿ under candidate load x ----
            if x is not None:
                self._apply_forces(task, physics,
                                   loads=self._vector_to_loads(x, shapes))
            ckpt.integrate(nsub)            # mj_step×nsub + mj_forward -> s̃

            # ---- fluid: solve at predicted pose s̃ -> new load x̃ ----
            self._advance_fluid_at_current_pose(iteration, t)
            fs.__dict__.update(_FS_FREE_AFTER_FORCES_3D)
            x_tilde, _ = self._loads_to_vector()

            if x is None:                   # very first sweep: no guess yet
                x = np.zeros_like(x_tilde)

            if np.all(np.isfinite(x_tilde)):
                load_scale = max(load_scale, float(np.linalg.norm(x_tilde)))

            res = acc.residual_norm(x, x_tilde)
            self.last_iters, self.last_residual = k, float(res)
            if np.isfinite(res) and res < best_res:
                best_res, best_x = res, x_tilde.copy()
            if res < self.tol * (1.0 + np.linalg.norm(x)):
                converged = True
                break
            if not np.isfinite(res):
                diverged = True
                break
            x = np.asarray(acc.relax(x, x_tilde), dtype=np.float64)

        if converged:
            acc.finalize_timestep()
        else:
            # A diverged / non-converged step collected garbage secant
            # pairs; finalize_timestep() would push them into the IQN-ILS
            # reuse store, where they poison every subsequent step: the
            # first quasi-Newton candidate built from them trips the force
            # bound at sweep 1, a 1-sweep step appends no fresh columns to
            # rotate the bad ones out, and the coupling degenerates
            # permanently to explicit commits (added-mass unstable).  Drop
            # ALL accelerator history instead — failures are rare and a
            # cold secant restart costs only a few extra sweeps.
            acc.reset()
        self._pose_source = prev_src

        # ---- choose the committed load ----
        # Converged: the last x̃.  Otherwise: the best-effort (min-residual)
        # iterate seen — never a blown-up one.  Do NOT warm-start the next
        # step from a non-converged guess.
        x_commit = x_tilde if converged else best_x
        if x_commit is None:               # not even one finite sweep
            x_commit = np.zeros(int(np.prod(shapes[0])) + int(np.prod(shapes[1])))
        self._implicit_prev = x_commit if converged else None

        if not converged:
            print(f"[BDIMhandler implicit] step {iteration}: NOT converged in "
                  f"{self.last_iters} sweeps (res={self.last_residual:.3e}"
                  f"{', diverged' if diverged else ''}); committing best-effort "
                  f"load (res={best_res:.3e}). Strong coupling can be "
                  f"ill-conditioned for articulated / position-controlled "
                  f"swimmers (especially force_method='lagrangian'); if this "
                  f"recurs, use body.coupling.scheme='explicit'.", flush=True)
            # Re-solve once at the committed load so the fluid state matches
            # the force we are about to leave on xfrc (the last sweep may have
            # been a rejected/oscillating iterate).
            ckpt.restore(mj_n)
            self._fluid_restore(fluid_n)
            self._apply_forces(task, physics, loads=self._vector_to_loads(x_commit, shapes))
            ckpt.integrate(nsub)
            self._advance_fluid_at_current_pose(iteration, t)
            fs.__dict__.update(_FS_FREE_AFTER_FORCES_3D)

        # ---- commit (Option A): leave sⁿ + the committed force ----
        # The fluid is at the solve for the committed pose; committing the
        # same force makes the runtime's single integration reproduce it.
        ckpt.restore(mj_n)
        self._apply_forces(task, physics, loads=self._vector_to_loads(x_commit, shapes))
        # NOTE: implicit mode requires cb_sub_steps == 1 (design §7.1), so the
        # non-full-substep apply_forces(loads=None) path is never invoked
        # between full steps; xfrc_applied set here is what the runtime
        # integrates.

        # ---- once-per-step fluid tail (plot / free-surface / release) ----
        if self.ndim == 3:
            fs.terminate = fs.finalize_step(fs.u0, fs.v0, fs.p0, iteration, w_vel=fs.w0)
        else:
            fs.terminate = fs.finalize_step(fs.u0, fs.v0, fs.p0, iteration)

        self.iteration += 1
