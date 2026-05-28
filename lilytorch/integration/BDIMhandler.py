"""
Unified BDIM handler for FARMS <-> lilytorch coupling.

Supports both 2-D and 3-D simulations.  Dimensionality is auto-detected
from the presence of ``Nz`` in ``pars['solver']``.

All simulation-specific hyperparameters are read from the ``bdim_yaml``
config dict, so a single class covers every animat (1guilla, pleurodeles,
zebrafish, salamander, ...).

New config keys (add to ``bdim_yaml``):
    solver.dtype                 : "float32" | "float64"   (default "float32")
    solver.rho_body              : float                   (default 800.0)
    solver.zero_pressure_inside  : bool                    (default False)
    solver.force_method          : "method1" | "method2"   (default "method2";
                                   3-D always uses method2_3d)
    body.force_scaling           : "auto" | float          (default "auto")
    body.contour_mask            : bool                    (default False)
    physics.solref               : [float, float] | null   (default null)
    physics.solimp               : [float, float, float, float, float] | null
                                   (default null)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from lilytorch.src.solver import FluidSolver
from lilytorch.src.body import (rotate_grid_2d, rotate_grid_3d,
                                _rotate_grid_3d_compiled)

import torch


_FS_FREE_AFTER_FORCES_3D = {
    'xstress_tensor': None, 'ystress_tensor': None, 'zstress_tensor': None,
    'pforce_x': None, 'pforce_y': None, 'pforce_z': None,
}


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
        self.fluid_solver = FluidSolver(
            self.pars,
            dtype=dtype,
            custom_update=True,
            compute_forces=True,
        )
        self.device = self.fluid_solver.device
        self.dtype = self.fluid_solver.dtype
        self.dtype_np = np.float64 if self.dtype == torch.float64 else np.float32

        # Stash a callable on the composite so the fluid step (in
        # solver.py) can invoke the softmin body-velocity blend after
        # the streaming SDF kernel writes the winning fields, without
        # importing BDIMhandler.  The callable returns early when
        # ``fluid_solver.sigma_softmin`` is None / non-positive.
        comp = self.fluid_solver.composite_body
        comp._softmin_blend_callable = (
            self._softmin_blend_body_velocity_3d if self.ndim == 3
            else self._softmin_blend_body_velocity_2d
        )

        # used for 2D contour-mask neighbor only
        self._prev_body_index = ()
        self._next_body_index = ()

        # ---- bookkeeping ----
        self.data = data          # list[AnimatData] from FARMS
        self.iteration = 0




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

        # ---- densities ----
        self.rho_fluid = self.pars["solver"]["rho"]

        # ---- toggles ----
        self.zero_pressure_inside = self.pars["solver"].get(
            "zero_pressure_inside", False
        )
        self.contour_mask = self.pars.get("body", {}).get("contour_mask", False)
        self.force_method = self.pars["solver"].get("force_method", "method2")

        # ``force_method == "method1"`` is a 2-D-only contour-integral
        # variant that lives entirely in pure-Python (``forces_method1``
        # in ``forces.py``).  It does not consume the per-body cc-SDF
        # produced by the streaming/combined C++/CUDA kernels (it samples
        # CC stress / pressure-force tensors at the body contour via
        # ``interp_utility``), so combining it with ``use_kernels=True``
        # would silently bypass the kernel-mode optimisations on the
        # forces stage.  Raise here so users get a clear error instead
        # of a confusing performance regression.  Method 1 has no 3-D
        # analogue (only ``forces_method2_3d`` exists), and the 3-D
        # dispatch in :meth:`step` already short-circuits to method 2
        # regardless of ``self.force_method`` — no extra check needed
        # for 3-D.

        if (self.ndim == 2
                and self.force_method == "method1"
                and self.fluid_solver._solver_method=="kernel"):
            raise ValueError(
                "force_method='method1' is incompatible with "
                "solver_method='kernel': forces_method1 "
                "is a contour-integral implementation that does not "
                "consume the per-body cc-SDF produced by the streaming/"
                "kernel path. Either set solver.solver_method='python' "
                "(pure-Python path) or switch to force_method='method2' "
                "(default), which integrates the smoothed delta with "
                "full kernel-mode acceleration."
            )

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
        # override composite-body update with our FARMS-driven version
        self.fluid_solver.composite_body.update = self.update

    def _init_update(self):
        if self.ndim == 3:
            if self.fluid_solver._solver_method == "kernel":
                self.update = self._update_3d_streaming_multi
            else:
                self.update = self._update_3d
        else:
            if self.fluid_solver._solver_method == "kernel":
                self.update = self._update_2d_streaming_multi
            else:
                self.update = self._update_2d

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
        """Gather FARMS link poses/velocities once per update path."""
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

        self._buoy_mass   = np.zeros(n)
        self._buoy_density = np.full(n, float(self.fluid_solver.rho_body))
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
                    density = float(self.fluid_solver.rho_body)
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

    # ──────────────────────────────────────────────────────────────────
    # Softmin SDF blending of multibody face velocities.
    #
    # The default per-link winning-body assignment makes ``body_u/v/w``
    # discontinuous on the SDF-intersection surface between two
    # interpenetrating links (different rigid velocities meet at the
    # surface where the winner switches).  In the band that
    # discontinuity becomes a huge ``div(u*)`` in low-mu0 cells, which
    # the BDIM2 mu0-weighted projection cannot remove → blow-up.
    #
    # Softmin blending replaces the winner with a smooth weighted
    # average:
    #     u(x)  =  Σ_k  w_k(x) · u_k(x)  /  Σ_k  w_k(x)
    #     w_k   =  exp( -(sdf_k(x) - sdf_min(x)) / σ )
    # so band cells receive a continuous velocity across intersection
    # surfaces.  The winner (smallest ``sdf_k``) gets ``w = 1``; the
    # next-closest body gets ``exp(-Δ/σ) ≤ 1``.  Far bodies contribute
    # negligibly.  σ has units of length; a reasonable choice is
    # ``~comp.eps`` (BDIM band half-width), giving a smooth blend
    # comparable to the BDIM transition.
    #
    # Requires that ``sdf_face`` already holds the per-cell *min* over
    # all bodies (both the streaming kernel and ``_update_*_full``'s
    # winning merge produce this).  Re-evaluates each body's SDF on its
    # AABB-clipped staggered grids and overwrites ``body_face`` with
    # the blended velocity.  Called from both the Python (full) and
    # kernel update paths when ``self.fluid_solver.sigma_softmin`` is
    # a positive float.
    # ──────────────────────────────────────────────────────────────────
    def _softmin_blend_body_velocity_3d(self, sdf_u, sdf_v, sdf_w,
                                        body_u, body_v, body_w):
        """Softmin-blend per-link body velocities into body_u/v/w (in place).

        Reads sdf_u/v/w as the per-cell *min* over all bodies (already
        populated by the caller).  Reads per-body kinematics stashed on
        the composite body as ``comp._latest_link_kin_3d``.  Overwrites
        body_u/v/w with the softmin-weighted average.
        """
        sigma = getattr(self.fluid_solver, 'sigma_softmin', None)
        if sigma is None or sigma <= 0.0:
            return  # off → caller's winning-body fields untouched
        comp = self.fluid_solver.composite_body
        kin  = getattr(comp, '_latest_link_kin_3d', None)
        if kin is None:
            return  # no kinematics stashed → silently skip

        inv_sigma = 1.0 / float(sigma)
        eps_floor = 1e-30
        h_grid = float(comp.h)
        gs     = sdf_u.shape

        # Allocate per-step weight accumulators (same shape as the face
        # tensors).  body_u/v/w become Σ w·v during the loop, then we
        # divide by Σ w at the end.
        sumw_u = torch.zeros_like(sdf_u)
        sumw_v = torch.zeros_like(sdf_v)
        sumw_w = torch.zeros_like(sdf_w)
        body_u.zero_()
        body_v.zero_()
        body_w.zero_()

        com_pos_t  = kin['com_pos']    # (B, 3)
        urdf_pos_t = kin['urdf_pos']   # (B, 3)
        R_t        = kin['R']          # (B, 3, 3)
        lin_vel_t  = kin['lin_vel']    # (B, 3)
        ang_vel_t  = kin['ang_vel']    # (B, 3)

        for body_i, body in enumerate(comp.bodies):
            com_pos  = com_pos_t[body_i]
            urdf_pos = urdf_pos_t[body_i]
            R        = R_t[body_i]
            lin_vel  = lin_vel_t[body_i]
            ang_vel  = ang_vel_t[body_i]

            body_pos, body_rot = self._compose_body_frame_3d(
                body, urdf_pos, R)
            R_T = body_rot.T

            aabb = self._body_aabb_indices(
                body, body_rot, body_pos,
                comp.x, comp.y, comp.z, h_grid, gs, pad=3,
            )
            sl, _full = self._slice_from_aabb(aabb, self.ndim)

            # Per-body SDF at staggered faces.
            px_u, py_u, pz_u = _rotate_grid_3d_compiled(
                comp.Xu_stag[sl], comp.Yu_stag[sl], comp.Zu_stag[sl],
                R_T, body_pos)
            sdf_u_k = body.sdf(px_u, py_u, pz_u)

            px_v, py_v, pz_v = _rotate_grid_3d_compiled(
                comp.Xv_stag[sl], comp.Yv_stag[sl], comp.Zv_stag[sl],
                R_T, body_pos)
            sdf_v_k = body.sdf(px_v, py_v, pz_v)

            px_w, py_w, pz_w = _rotate_grid_3d_compiled(
                comp.Xw_stag[sl], comp.Yw_stag[sl], comp.Zw_stag[sl],
                R_T, body_pos)
            sdf_w_k = body.sdf(px_w, py_w, pz_w)

            # Per-body rigid velocity at staggered faces (identical
            # formula to the winning-merge path so the limiting case
            # σ→0 recovers winning-body up to FP noise).
            vel_u_k = (lin_vel[0]
                       + ang_vel[1] * (comp.Zu_stag[sl] - com_pos[2])
                       - ang_vel[2] * (comp.Yu_stag[sl] - com_pos[1]))
            vel_v_k = (lin_vel[1]
                       + ang_vel[2] * (comp.Xv_stag[sl] - com_pos[0])
                       - ang_vel[0] * (comp.Zv_stag[sl] - com_pos[2]))
            vel_w_k = (lin_vel[2]
                       + ang_vel[0] * (comp.Yw_stag[sl] - com_pos[1])
                       - ang_vel[1] * (comp.Xw_stag[sl] - com_pos[0]))

            # Weights relative to the per-cell minimum SDF.  dphi ≥ 0
            # by construction (sdf_face = min over all bodies), so the
            # exponent is non-positive and weights are in (0, 1].
            w_u = torch.exp(-(sdf_u_k - sdf_u[sl]) * inv_sigma)
            w_v = torch.exp(-(sdf_v_k - sdf_v[sl]) * inv_sigma)
            w_w = torch.exp(-(sdf_w_k - sdf_w[sl]) * inv_sigma)

            sumw_u[sl] += w_u
            sumw_v[sl] += w_v
            sumw_w[sl] += w_w
            body_u[sl] += w_u * vel_u_k
            body_v[sl] += w_v * vel_v_k
            body_w[sl] += w_w * vel_w_k

        # Divide.  Cells outside every body's AABB keep sumw=0 and
        # body=0 — clamp_min prevents division by zero; the numerator
        # is also 0 there so the cell stays 0.
        body_u.div_(sumw_u.clamp_min_(eps_floor))
        body_v.div_(sumw_v.clamp_min_(eps_floor))
        body_w.div_(sumw_w.clamp_min_(eps_floor))

    def _softmin_blend_body_velocity_2d(self, sdf_u, sdf_v, body_u, body_v):
        """2-D analogue of :meth:`_softmin_blend_body_velocity_3d`.

        Uses the configured 2-D plane (xy or xz) for the rigid-body
        velocity formula; mirrors the winning-merge logic in
        :meth:`_update_2d`.
        """
        sigma = getattr(self.fluid_solver, 'sigma_softmin', None)
        if sigma is None or sigma <= 0.0:
            return
        comp = self.fluid_solver.composite_body
        kin  = getattr(comp, '_latest_link_kin_2d', None)
        if kin is None:
            return

        inv_sigma = 1.0 / float(sigma)
        eps_floor = 1e-30
        h_grid    = float(comp.h)
        gs        = sdf_u.shape

        sumw_u = torch.zeros_like(sdf_u)
        sumw_v = torch.zeros_like(sdf_v)
        body_u.zero_()
        body_v.zero_()

        # Per-body kinematics for the 2D plane (already projected).
        com_pos_t  = kin['com_pos']    # (B, 2) — in-plane coords
        urdf_pos_t = kin['urdf_pos']   # (B, 2)
        R_t        = kin['R']          # (B, 2, 2) or (B, 3, 3) depending
        lin_vel_t  = kin['lin_vel']    # (B, 2)
        ang_vel_t  = kin['ang_vel']    # (B,)   — scalar out-of-plane

        for body_i, body in enumerate(comp.bodies):
            com_pos  = com_pos_t[body_i]
            urdf_pos = urdf_pos_t[body_i]
            R        = R_t[body_i]
            lin_vel  = lin_vel_t[body_i]
            ang_vel  = ang_vel_t[body_i]

            # Compose body frame (use the existing 2-D helper if any,
            # else build from R + urdf_pos directly).  We piggyback on
            # the same primitives used by ``_update_2d``.
            body_pos, body_rot = self._compose_body_frame_2d(
                body, urdf_pos, R)
            R_T = body_rot.T

            aabb = self._body_aabb_indices(
                body, body_rot, body_pos,
                comp.x, comp.y, None, h_grid, gs, pad=3,
            )
            sl, _full = self._slice_from_aabb(aabb, self.ndim)

            from lilytorch.src.body import _rotate_grid_2d_compiled
            px_u, py_u = _rotate_grid_2d_compiled(
                comp.Xu_stag[sl], comp.Yu_stag[sl], R_T, body_pos)
            sdf_u_k = body.sdf(px_u, py_u)

            px_v, py_v = _rotate_grid_2d_compiled(
                comp.Xv_stag[sl], comp.Yv_stag[sl], R_T, body_pos)
            sdf_v_k = body.sdf(px_v, py_v)

            # Rigid-body velocity in 2-D: u = lin + ω × (r - com).
            vel_u_k = lin_vel[0] - ang_vel * (comp.Yu_stag[sl] - com_pos[1])
            vel_v_k = lin_vel[1] + ang_vel * (comp.Xv_stag[sl] - com_pos[0])

            w_u = torch.exp(-(sdf_u_k - sdf_u[sl]) * inv_sigma)
            w_v = torch.exp(-(sdf_v_k - sdf_v[sl]) * inv_sigma)

            sumw_u[sl] += w_u
            sumw_v[sl] += w_v
            body_u[sl] += w_u * vel_u_k
            body_v[sl] += w_v * vel_v_k

        body_u.div_(sumw_u.clamp_min_(eps_floor))
        body_v.div_(sumw_v.clamp_min_(eps_floor))

    # ─── per-body kinematics stash (consumed by the softmin blend) ───
    def _stash_per_body_kin_3d(self, comp,
                               com_poses_t, urdf_poses_t, Rs_t,
                               lin_vels_t, ang_vels_t):
        """Gather per-(animat, link) FARMS kinematics into per-body (B, …)
        tensors and stash on ``comp._latest_link_kin_3d``.

        The softmin blend reads this stash; both the Python update and
        the kernel-mode fluid step call us when ``sigma_softmin > 0``.
        """
        B = len(comp.bodies)
        dev, dt_ = self.device, self.dtype
        com_pos  = torch.empty((B, 3),    dtype=dt_, device=dev)
        urdf_pos = torch.empty((B, 3),    dtype=dt_, device=dev)
        R_       = torch.empty((B, 3, 3), dtype=dt_, device=dev)
        lin_vel  = torch.empty((B, 3),    dtype=dt_, device=dev)
        ang_vel  = torch.empty((B, 3),    dtype=dt_, device=dev)
        _as = lambda x: torch.as_tensor(x, dtype=dt_, device=dev)
        for b, (a_id, l_id) in enumerate(comp.body_ids):
            a_id = int(a_id); l_id = int(l_id)
            com_pos[b]  = _as(com_poses_t[a_id][l_id])
            urdf_pos[b] = _as(urdf_poses_t[a_id][l_id])
            R_[b]       = _as(Rs_t[a_id][l_id])
            lin_vel[b]  = _as(lin_vels_t[a_id][l_id])
            ang_vel[b]  = _as(ang_vels_t[a_id][l_id])
        comp._latest_link_kin_3d = {
            'com_pos': com_pos, 'urdf_pos': urdf_pos, 'R': R_,
            'lin_vel': lin_vel, 'ang_vel': ang_vel,
        }

    def _stash_per_body_kin_2d(self, comp,
                               com_poses_t, urdf_poses_t, Rs_t,
                               lin_vels_t, ang_vels_t):
        """2-D analogue of :meth:`_stash_per_body_kin_3d`.

        Stores in-plane coordinates / linear velocity (size 2) and a
        scalar out-of-plane angular velocity, matching how
        :meth:`_update_2d` indexes its kinematic arrays.
        """
        B = len(comp.bodies)
        dev, dt_ = self.device, self.dtype
        com_pos  = torch.empty((B, 2),    dtype=dt_, device=dev)
        urdf_pos = torch.empty((B, 2),    dtype=dt_, device=dev)
        R_       = torch.empty((B, 3, 3), dtype=dt_, device=dev)
        lin_vel  = torch.empty((B, 2),    dtype=dt_, device=dev)
        ang_vel  = torch.empty((B,),      dtype=dt_, device=dev)
        ang_ax   = self._2d_ang_ax
        ax0, ax1 = self.lin_axes
        _as = lambda x: torch.as_tensor(x, dtype=dt_, device=dev)
        for b, (a_id, l_id) in enumerate(comp.body_ids):
            a_id = int(a_id); l_id = int(l_id)
            cp = _as(com_poses_t[a_id][l_id])
            up = _as(urdf_poses_t[a_id][l_id])
            lv = _as(lin_vels_t[a_id][l_id])
            av = _as(ang_vels_t[a_id][l_id])
            com_pos[b]  = torch.stack((cp[ax0], cp[ax1]))
            urdf_pos[b] = torch.stack((up[ax0], up[ax1]))
            R_[b]       = _as(Rs_t[a_id][l_id])
            lin_vel[b]  = torch.stack((lv[ax0], lv[ax1]))
            ang_vel[b]  = av[ang_ax]
        comp._latest_link_kin_2d = {
            'com_pos': com_pos, 'urdf_pos': urdf_pos, 'R': R_,
            'lin_vel': lin_vel, 'ang_vel': ang_vel,
        }

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
            self._merge_region_sdf_and_velocity(
                comp.sdf_val_u, comp.body_u, sdf_u, vel_u, sl, full_region)
            self._merge_region_sdf_and_velocity(
                comp.sdf_val_v, comp.body_v, sdf_v, vel_v, sl, full_region)

            comp.com_pos[body_i] = com_pos
            body.com_pos = com_pos

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

        # Softmin SDF blending (2-D) — see _update_3d for rationale.
        sigma = getattr(self.fluid_solver, 'sigma_softmin', None)
        if sigma is not None and sigma > 0.0:
            self._stash_per_body_kin_2d(
                comp, com_poses_t, urdf_poses_t, Rs_t, lin_vels_t, ang_vels_t,
            )
            self._softmin_blend_body_velocity_2d(
                comp.sdf_val_u, comp.sdf_val_v,
                comp.body_u,    comp.body_v,
            )

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
            self._merge_region_sdf_and_velocity(
                comp.sdf_val_u, comp.body_u, sdf_u, vel_u, sl, full_region)
            self._merge_region_sdf_and_velocity(
                comp.sdf_val_v, comp.body_v, sdf_v, vel_v, sl, full_region)
            self._merge_region_sdf_and_velocity(
                comp.sdf_val_w, comp.body_w, sdf_w, vel_w, sl, full_region)

            comp.com_pos[body_i] = com_pos
            body.com_pos = com_pos

        # Softmin SDF blending of multibody face velocities (no-op when
        # sigma_softmin is None / non-positive).  Replaces the winning
        # per-link velocity stored above with a smooth weighted average,
        # eliminating the velocity discontinuity at SDF-intersection
        # surfaces of overlapping links — the root cause of multibody
        # blow-up in the BDIM2 mu0-weighted projection.
        sigma = getattr(self.fluid_solver, 'sigma_softmin', None)
        if sigma is not None and sigma > 0.0:
            self._stash_per_body_kin_3d(
                comp, com_poses_t, urdf_poses_t, Rs_t, lin_vels_t, ang_vels_t,
            )
            self._softmin_blend_body_velocity_3d(
                comp.sdf_val_u, comp.sdf_val_v, comp.sdf_val_w,
                comp.body_u,    comp.body_v,    comp.body_w,
            )

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

        # Stash per-body kinematics for the softmin body-velocity blend
        # (consumed in solver.py after streaming_sdf_stag_3d_multi).  No-op
        # when sigma_softmin is off, so no wasted GPU traffic in the default
        # path.
        if (getattr(self.fluid_solver, 'sigma_softmin', None) is not None
                and self.fluid_solver.sigma_softmin > 0.0):
            self._stash_per_body_kin_3d(
                comp, com_poses, urdf_poses, Rs, lin_vels, ang_vels,
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

        # Stash per-body kinematics for the softmin body-velocity blend
        # consumed in solver.py after streaming_sdf_stag_2d_multi.
        if (getattr(self.fluid_solver, 'sigma_softmin', None) is not None
                and self.fluid_solver.sigma_softmin > 0.0):
            self._stash_per_body_kin_2d(
                comp, com_poses, urdf_poses, Rs, lin_vels, ang_vels,
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
    def _apply_forces(self, task, physics):
        """Dim-agnostic fluid → body force application via MuJoCo xfrc.

        Per-D xfrc index map, buoyancy axis, and FluidSolver field names
        are precomputed in :meth:`_init_apply_forces`, so this method
        has zero per-step Python branching on ``ndim`` or ``_2d_plane``.

        FARMS-style buoyancy is applied additively to a single linear
        xfrc index (``_buoyancy_xfrc_idx``); for the 2-D xy plane, where
        no buoyancy is needed, that index is ``None`` and the buoyancy
        block is skipped entirely.

        This method only reads cached force tensors on the FluidSolver
        (``friction_force_*``, ``pressure_force_*``) — it does not
        advance the fluid solver, so it is safe to call once per MuJoCo
        substep to keep ``xfrc_applied`` fresh between full BDIM steps
        (mirroring ``SwimmingExtension`` with ``substep=True``).
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

        # Total per-axis (viscous + pressure) at CPU level.
        lin_total = forces_cpu[:D] + forces_cpu[D + Nt: 2 * D + Nt]   # (D, B)
        ang_total = forces_cpu[D: D + Nt] + forces_cpu[2 * D + Nt:]   # (Nt, B)

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
