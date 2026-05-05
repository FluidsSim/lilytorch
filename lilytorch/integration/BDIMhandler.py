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
"""

import numpy as np
from scipy.spatial.transform import Rotation
from lilytorch.src.solver import FluidSolver
from lilytorch.src.body import (rotate_grid_2d, rotate_grid_3d,
                                _rotate_grid_3d_compiled)
from lilytorch.src.kernels import streaming_sdf_forces_fused_3d_multi
from lilytorch.src.kernels import streaming_sdf_forces_fused_2d_multi
from lilytorch.src.kernels import RegularGridInterpolator3D
from lilytorch.src.kernels import streaming_sdf_min_2d_multi
from lilytorch.src.kernels import streaming_sdf_min_3d
from lilytorch.src.kernels import streaming_sdf_min_3d_multi

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
        if dtype is not None:
            self.dtype = dtype
        else:
            dtype_str = self.pars["solver"].get("dtype", "float32")
            if isinstance(dtype_str, torch.dtype):
                self.dtype = dtype_str
            else:
                _dtype_map = {"float32": torch.float32, "float64": torch.float64,
                              "double":  torch.float64, "single":  torch.float32}
                if dtype_str not in _dtype_map:
                    raise ValueError(
                        f"Unknown solver.dtype '{dtype_str}'. "
                        f"Expected one of {sorted(_dtype_map)}."
                    )
                self.dtype = _dtype_map[dtype_str]
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                f"BDIMhandler dtype must be torch.float32 or torch.float64, "
                f"got {self.dtype}."
            )
        self.dtype_np = np.float32 if self.dtype == torch.float32 else np.float64
        self._prev_body_index = () # used for 2D contour-mask neighbor only
        self._next_body_index = ()

        # ---- bookkeeping ----
        self.data = data          # list[AnimatData] from FARMS
        self.iteration = 0

        # ---- create fluid solver ----
        self.fluid_solver = FluidSolver(
            self.pars,
            dtype=self.dtype,
            custom_update=True,
            compute_forces=True,
        )
        self.device = self.fluid_solver.device

        # override composite-body update with our FARMS-driven version
        self.fluid_solver.composite_body.update = self.update

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
        # produced by the streaming/fused C++/CUDA kernels (it samples
        # CC stress / pressure-force tensors at the body contour via
        # ``interp_utility``), so combining it with ``use_kernels=True``
        # would silently bypass the kernel-mode optimisations on the
        # forces stage.  Raise here so users get a clear error instead
        # of a confusing performance regression.  Method 1 has no 3-D
        # analogue (only ``forces_method2_3d`` exists), and the 3-D
        # dispatch in :meth:`step` already short-circuits to method 2
        # regardless of ``self.force_method`` — no extra check needed
        # for 3-D.
        _solver_cfg = self.pars["solver"]
        _kernel_path_active = (
            _solver_cfg.get("solver_method", None) in ("kernels", "fused")
            or (_solver_cfg.get("solver_method", None) is None
                and bool(_solver_cfg.get("use_kernels", True)))
        )
        if (self.ndim == 2
                and self.force_method == "method1"
                and _kernel_path_active):
            raise ValueError(
                "force_method='method1' is incompatible with "
                "solver_method ∈ {'kernels', 'fused'}: forces_method1 "
                "is a contour-integral implementation that does not "
                "consume the per-body cc-SDF produced by the streaming/"
                "fused kernels. Either set solver.solver_method='python' "
                "(pure-Python path) or switch to force_method='method2' "
                "(default), which integrates the smoothed delta with "
                "full kernel-mode acceleration."
            )

        # ---- FARMS-style buoyancy parameters ----
        # With all-Neumann BCs the BDIM pressure field is purely dynamic;
        # no hydrostatic gradient builds up, so buoyancy must be added
        # explicitly.  We replicate FARMS' compute_buoyancy() formula from
        # drag.pyx which uses MuJoCo body mass, bounding-sphere half-height,
        # and a linear submersion fraction.
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

        # ---- optional physics solref tweak ----
        solref = self.pars.get("physics", {}).get("solref", None)
        if solref is not None:
            physics.model.geom_solref[:, 0] = solref[0]
            physics.model.geom_solref[:, 1] = solref[1]

        # ---- allocate per-body SDF / velocity arrays if missing ----
        comp = self.fluid_solver.composite_body
        gs = self.fluid_solver.grid_shape
        if self.ndim == 3:
            # 3-D never uses comp.sdf_vals — all update paths write comp._sdf_sparse
            # and forces_method2_3d reads _sdf_sparse or _fused_forces_out directly.
            # body.py skips the 3-D allocation; nothing to reassign here.
            # Streaming fused-CUDA path requires the per-body C++/CUDA
            # trilinear samplers (re-uses body._stream_meta cached there).
            # When ``streaming_sdf_3d`` is on, ``custom_trilinear_3d`` is
            # auto-set in ``solver.py``.
            if getattr(self.fluid_solver, '_custom_trilinear_3d', False):
                self._init_custom_trilinear_3d()
            if getattr(self.fluid_solver, '_streaming_sdf_3d', False):
                print("  [streaming-sdf-3D] fused per-body C++/CUDA SDF "
                      "min-update enabled (replaces _update_3d streaming "
                      "Python loop)")
        else:
            # 2-D update uses per-body SDF stacks for argmin-based union
            # and force integration.  body.py pre-allocates these for 2-D;
            # the guards below are safety nets for unusual construction paths.
            _streaming_2d = getattr(self.fluid_solver, '_streaming_sdf_2d', False)

            # sdf_vals is always kept: _update_2d falls back to the batched
            # grid_sample path when any body lacks _stream_meta, and that
            # path writes/reads comp.sdf_vals.
            if not hasattr(comp, 'sdf_vals'):
                comp.sdf_vals = torch.zeros(
                    (comp.nbodies, *gs), device=self.device, dtype=self.dtype)

            if not hasattr(comp, 'sdf_vals_u'):
                comp.sdf_vals_u = torch.zeros(
                    (comp.nbodies, *gs), device=self.device, dtype=self.dtype)
            if not hasattr(comp, 'sdf_vals_v'):
                comp.sdf_vals_v = torch.zeros(
                    (comp.nbodies, *gs), device=self.device, dtype=self.dtype)
            if not hasattr(comp, 'u_vals'):
                comp.u_vals = torch.zeros(
                    (comp.nbodies, *gs), device=self.device, dtype=self.dtype)
            if not hasattr(comp, 'v_vals'):
                comp.v_vals = torch.zeros(
                    (comp.nbodies, *gs), device=self.device, dtype=self.dtype)

            # ``comp.body_ids`` is built once with the composite body, so the
            # per-animat previous/next body links are static as well.
            self._init_body_neighbors_2d()

            # NOTE: the 2-D Python path no longer pre-samples / pads / stacks
            # bodies for grid_sample.  It now mirrors the 3-D Python path
            # (per-body AABB-cropped sub-grid + ``torch.where`` running-min
            # union); analytical bodies are evaluated directly on the
            # sub-grid via ``body.sdf(X, Y)``.

            # ── 2-D streaming fused-CUDA path (Phase C analogue of 3-D) ──
            # Opt-in via solver.streaming_sdf_2d.  Builds per-body
            # `_stream_meta` from each body's own `sdf.{F, x, y}` so the
            # `streaming_sdf_min_2d_multi` kernel can run.
            if _streaming_2d:
                self._init_custom_trilinear_2d()
                print("  [streaming-sdf-2D] fused per-body C++/CUDA SDF "
                      "min-update enabled (replaces _update_2d batched "
                      "grid_sample loop)")

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

    def cython2numpy(self, array):
        return torch.from_numpy(
            np.array(array).astype(self.dtype_np)
        ).to(self.device)

    # ==================================================================
    #  update: FARMS kinematics  ->  SDF fields + body velocities
    # ==================================================================
    def update(self, t, iteration, dt=1):
        if self.ndim == 3:
            self._update_3d(t, iteration, dt)
        else:
            self._update_2d(t, iteration, dt)

    # ---- 2-D update --------------------------------------------------
    def _update_2d(self, t, iteration, dt=1):
        # Streaming fused-CUDA path — opt-in via solver.streaming_sdf_2d.
        # The streaming dispatch (``_update_2d_streaming_multi``) handles
        # composites where every body exposes ``_stream_meta`` (regular-
        # grid SDF table).  When any body is analytical, fall through to
        # the Python path below — which itself supports mixed composites
        # (it evaluates ``body.sdf`` per body, with mesh bodies using
        # their grid_sample-backed interpolator and analytical bodies
        # using their callable SDF).  This mirrors the 3-D streaming
        # dispatch (see ``_update_3d_streaming``); both use the same
        # "all-mesh → batched kernel; mixed → per-body fallback" rule.
        if getattr(self.fluid_solver, '_streaming_sdf_2d', False):
            comp_check = self.fluid_solver.composite_body
            if all(getattr(b, '_stream_meta', None) is not None
                   for b in comp_check.bodies):
                return self._update_2d_streaming_multi(t, iteration, dt)

        fs   = self.fluid_solver
        comp = fs.composite_body
        gs   = fs.grid_shape
        B    = comp.nbodies

        _FAR = 1e4   # far-field SDF sentinel (same as 3-D path)

        # gather per-animat kinematics (kept as device tensors for the
        # downstream contour-mask path which queries other bodies' SDFs).
        com_poses  = []
        urdf_poses = []
        Rs         = []
        lin_vels   = []
        ang_vels   = []
        for exp_data in self.data:
            sen = exp_data.sensors.links
            com_poses.append(
                self.cython2numpy(sen.com_positions()[iteration, :])[:, self.lin_axes]
            )
            urdf_poses.append(
                self.cython2numpy(sen.urdf_positions()[iteration, :])[:, self.lin_axes]
            )
            Rs.append(
                self.cython2numpy(
                    Rotation.from_quat(
                        sen.urdf_orientations()[iteration, :]
                    ).as_matrix().astype(self.dtype_np)
                )[:, self.lin_axes, :][:, :, self.lin_axes]
            )
            lin_vels.append(
                self.cython2numpy(sen.com_lin_velocities()[iteration, :])[:, self.lin_axes]
            )
            ang_vels.append(
                self.cython2numpy(
                    [sen.com_ang_velocity(iteration, lk)[self._2d_ang_ax]
                     for lk in range(len(sen.names))]
                )
            )

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
        if not hasattr(comp, '_body_aabbs'):
            comp._body_aabbs = [None] * B

        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]

            com_pos  = com_poses[animat_id][link_id]
            urdf_pos = urdf_poses[animat_id][link_id]
            R        = Rs[animat_id][link_id]
            lin_vel  = lin_vels[animat_id][link_id]
            ang_vel  = ang_vels[animat_id][link_id]

            R_T = R.T  # (2, 2) transposed rotation

            # ── AABB-clipped SDF evaluation ─────────────────────────
            # Mirror the 3-D pattern: AABB-clip whenever the body
            # provides a local-frame bounding-box descriptor (mesh
            # bodies via ``body.sdf.x/y`` axes; analytical bodies via
            # ``body.local_aabb`` derived from the contour with a
            # band-radius safety margin).  Bodies without either
            # descriptor fall through to the full-grid path.
            aabb = self._body_aabb_indices_2d(
                body, R, urdf_pos,
                comp.x, comp.y,
                h_grid, gs, pad=3,
            )
            comp._body_aabbs[body_i] = aabb

            sdf_eval = body.sdf

            if aabb is not None:
                # ── Sub-block path (main saving) ────────────────────
                (i0, i1, j0, j1) = aabb
                sl = (slice(i0, i1), slice(j0, j1))

                px, py = rotate_grid_2d(
                    comp.X[sl], comp.Y[sl], R_T, urdf_pos)
                sdf_sub = sdf_eval(px, py)

                # Per-body SDF (sparse: store sub-block + AABB for forces)
                comp._sdf_sparse[body_i] = (aabb, sdf_sub)

                # Evaluate SDF directly at staggered face locations
                px_u, py_u = rotate_grid_2d(
                    comp.Xu_stag[sl], comp.Yu_stag[sl], R_T, urdf_pos)
                sdf_sub_u = sdf_eval(px_u, py_u)

                px_v, py_v = rotate_grid_2d(
                    comp.Xv_stag[sl], comp.Yv_stag[sl], R_T, urdf_pos)
                sdf_sub_v = sdf_eval(px_v, py_v)

                # Sub-block body velocity (rigid-body: lv - ω × r in 2-D
                # collapses to ω being a scalar out-of-plane).
                vel_sub_u = lin_vel[0] - ang_vel * (
                    comp.Yu_stag[sl] - com_pos[1])
                vel_sub_v = lin_vel[1] + ang_vel * (
                    comp.Xv_stag[sl] - com_pos[0])

                # Sub-block union min ─ contiguous read avoids slow
                # strided reads inside torch.where.
                old_cc = comp.sdf_val[sl].contiguous()
                comp.sdf_val[sl] = torch.where(sdf_sub < old_cc, sdf_sub, old_cc)

                old_u = comp.sdf_val_u[sl].contiguous()
                mask_u = sdf_sub_u < old_u
                comp.sdf_val_u[sl] = torch.where(mask_u, sdf_sub_u, old_u)
                comp.body_u[sl] = torch.where(
                    mask_u, vel_sub_u, comp.body_u[sl].contiguous())

                old_v = comp.sdf_val_v[sl].contiguous()
                mask_v = sdf_sub_v < old_v
                comp.sdf_val_v[sl] = torch.where(mask_v, sdf_sub_v, old_v)
                comp.body_v[sl] = torch.where(
                    mask_v, vel_sub_v, comp.body_v[sl].contiguous())

            else:
                # ── Full-grid path (body covers >90 % of grid) ─────
                px, py = rotate_grid_2d(comp.X, comp.Y, R_T, urdf_pos)
                sdf_cc = sdf_eval(px, py)
                comp._sdf_sparse[body_i] = (None, sdf_cc)

                px_u, py_u = rotate_grid_2d(
                    comp.Xu_stag, comp.Yu_stag, R_T, urdf_pos)
                sdf_u = sdf_eval(px_u, py_u)
                px_v, py_v = rotate_grid_2d(
                    comp.Xv_stag, comp.Yv_stag, R_T, urdf_pos)
                sdf_v = sdf_eval(px_v, py_v)

                vel_u = lin_vel[0] - ang_vel * (comp.Yu_stag - com_pos[1])
                vel_v = lin_vel[1] + ang_vel * (comp.Xv_stag - com_pos[0])

                mask_cc = sdf_cc < comp.sdf_val
                comp.sdf_val = torch.where(mask_cc, sdf_cc, comp.sdf_val)
                mask_u = sdf_u < comp.sdf_val_u
                comp.sdf_val_u = torch.where(mask_u, sdf_u, comp.sdf_val_u)
                comp.body_u    = torch.where(mask_u, vel_u, comp.body_u)
                mask_v = sdf_v < comp.sdf_val_v
                comp.sdf_val_v = torch.where(mask_v, sdf_v, comp.sdf_val_v)
                comp.body_v    = torch.where(mask_v, vel_v, comp.body_v)

            comp.com_pos[body_i] = com_pos
            body.com_pos = com_pos

            # contour update (world frame)
            body.cnt_update = R @ body.cnt + urdf_pos[:, None]

            # optional contour mask for overlapping links
            if self.contour_mask:
                x_cnt = body.cnt_update[0]
                y_cnt = body.cnt_update[1]
                prev_body_i = self._prev_body_index[body_i]
                next_body_i = self._next_body_index[body_i]

                if prev_body_i is None and next_body_i is None:
                    mask = torch.ones_like(x_cnt, dtype=torch.bool)
                elif prev_body_i is None:
                    (_, next_link_id) = comp.body_ids[next_body_i]
                    body_p = comp.bodies[next_body_i]
                    pt = Rs[animat_id][next_link_id].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses[animat_id][next_link_id][:, None]
                    )
                    mask = (body_p.sdf(pt[0], pt[1]) - body.h) >= 0
                elif next_body_i is None:
                    (_, prev_link_id) = comp.body_ids[prev_body_i]
                    body_m = comp.bodies[prev_body_i]
                    pt = Rs[animat_id][prev_link_id].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses[animat_id][prev_link_id][:, None]
                    )
                    mask = (body_m.sdf(pt[0], pt[1]) - body.h) >= 0
                else:
                    (_, prev_link_id) = comp.body_ids[prev_body_i]
                    body_m = comp.bodies[prev_body_i]
                    pt_m = Rs[animat_id][prev_link_id].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses[animat_id][prev_link_id][:, None]
                    )
                    (_, next_link_id) = comp.body_ids[next_body_i]
                    body_p = comp.bodies[next_body_i]
                    pt_p = Rs[animat_id][next_link_id].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses[animat_id][next_link_id][:, None]
                    )
                    sdf_m = body_m.sdf(pt_m[0], pt_m[1]) - body.h
                    sdf_p = body_p.sdf(pt_p[0], pt_p[1]) - body.h
                    mask = (sdf_m >= 0) & (sdf_p >= 0)
                body.mask = mask

            body.r_com = body.cnt_update - com_pos[:, None]

    @staticmethod
    def _body_local_aabb_2d(body):
        """Return ``(sdf_lo, sdf_hi)`` 1-D tensors describing the
        body's local-frame AABB, or ``None`` if no AABB info is
        available.  Source priority:

        1. ``body.local_aabb`` — an explicit ``(2, 2)`` tensor set by
           analytical bodies (auto-derived from the contour with a
           safety margin in :meth:`BodyAnalytical._initialize_2d`,
           or user-provided).
        2. ``body.sdf.x`` / ``body.sdf.y`` — for mesh bodies whose
           local SDF table carries axis tensors.  These tables are
           padded with a ``_FAR`` sentinel outside the mesh, so
           ``grid_sample(padding_mode='border')`` returns far-field
           values for queries beyond the AABB.

        Returns ``None`` when neither source is available; callers
        should then use the full-grid evaluation path.
        """
        local_aabb = getattr(body, 'local_aabb', None)
        if local_aabb is not None:
            return local_aabb[0], local_aabb[1]
        if (hasattr(body, 'sdf')
                and hasattr(body.sdf, 'x')
                and hasattr(body.sdf, 'y')):
            return (
                torch.stack([body.sdf.x[0],  body.sdf.y[0]]),
                torch.stack([body.sdf.x[-1], body.sdf.y[-1]]),
            )
        return None

    @staticmethod
    def _body_aabb_indices_2d(body, R, urdf_pos, comp_x, comp_y,
                              h, gs, pad=3):
        """2-D analogue of :meth:`_body_aabb_indices`.

        Computes fluid-grid index ranges ``(i0, i1, j0, j1)`` for the
        AABB of a body's local-frame SDF support, transformed into
        world space.

        Supports both mesh bodies (``body.sdf`` exposes ``x``/``y``
        axis tensors) and analytical bodies that provide an explicit
        ``body.local_aabb`` covering the SDF band of width ``~4*eps``
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
           AABB covering their SDF band of width ``~4*eps``.  For
           ``BodyAnalytical`` this is auto-derived during
           ``_initialize_3d`` via marching cubes on the local SDF
           plus a ``4*eps + 4*h`` Lipschitz-safe margin (mirrors the
           2-D contour-based path); users may still override it via
           the constructor when they want a tighter / looser bound.
        2. ``body.sdf.x`` / ``body.sdf.y`` / ``body.sdf.z`` — for mesh
           bodies whose local SDF table carries axis tensors and is
           padded with the ``_FAR`` sentinel outside the mesh.

        Returns ``None`` when neither source is available.
        """
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


    # ---- 3-D update --------------------------------------------------
    def _update_3d(self, t, iteration, dt=1):
        # Streaming fused-CUDA path (Phase B) — opt-in via solver.streaming_sdf_3d
        if getattr(self.fluid_solver, '_streaming_sdf_3d', False):
            return self._update_3d_streaming(t, iteration, dt)
        fs   = self.fluid_solver
        comp = fs.composite_body
        gs   = fs.grid_shape

        # Far-field SDF value (>> eps) so mu0 = 1 (pure fluid) and
        # union-min always prefers the closest real body value.
        _FAR = 1e4

        com_poses  = []
        urdf_poses = []
        Rs         = []
        lin_vels   = []
        ang_vels   = []

        for exp_data in self.data:
            sen = exp_data.sensors.links
            com_poses.append(self.cython2numpy(sen.com_positions()[iteration, :]))
            urdf_poses.append(self.cython2numpy(sen.urdf_positions()[iteration, :]))
            Rs.append(
                self.cython2numpy(
                    Rotation.from_quat(
                        sen.urdf_orientations()[iteration, :]
                    ).as_matrix().astype(self.dtype_np)
                )
            )
            lin_vels.append(self.cython2numpy(sen.com_lin_velocities()[iteration, :]))
            nlinks = len(sen.names)
            ang_vels.append(
                self.cython2numpy(
                    np.stack([sen.com_ang_velocity(iteration, lk)
                              for lk in range(nlinks)])
                )
            )

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
        if not hasattr(comp, '_body_aabbs'):
            comp._body_aabbs = [None] * len(comp.bodies)

        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]

            com_pos  = com_poses[animat_id][link_id]
            urdf_pos = urdf_poses[animat_id][link_id]
            R        = Rs[animat_id][link_id]
            lin_vel  = lin_vels[animat_id][link_id]
            ang_vel  = ang_vels[animat_id][link_id]

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

            # Default `body.sdf` (grid_sample-backed) per-body SDF evaluator.
            # The custom C++/CUDA trilinear sampler is only used by the
            # streaming `_update_3d_streaming` path, which short-circuits
            # this function at the top.
            sdf_eval = body.sdf

            if aabb is not None:
                # ── Sub-block path (main saving) ────────────────────
                # Only rotate + interpolate the AABB sub-block of the
                # grid, then evaluate SDF directly at staggered
                # locations / velocity / union-min ALL on the sub-block
                # with contiguous intermediates.  Strided reads of
                # union fields are small (~3-5 MB per body) and fit in
                # L2 cache.
                (i0, i1, j0, j1, k0, k1) = aabb
                sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))

                px, py, pz = rotate_grid_3d(
                    comp.X[sl], comp.Y[sl], comp.Z_grid[sl],
                    R_T, body_pos,
                )
                sdf_sub = sdf_eval(px, py, pz)       # contiguous

                # Per-body SDF (sparse: store sub-block + AABB for forces)
                comp._sdf_sparse[body_i] = (aabb, sdf_sub)

                # Evaluate SDF directly at staggered face locations
                # (exact interpolation instead of CC averaging)
                px_u, py_u, pz_u = rotate_grid_3d(
                    comp.Xu_stag[sl], comp.Yu_stag[sl], comp.Zu_stag[sl],
                    R_T, body_pos)
                sdf_sub_u = sdf_eval(px_u, py_u, pz_u)

                px_v, py_v, pz_v = rotate_grid_3d(
                    comp.Xv_stag[sl], comp.Yv_stag[sl], comp.Zv_stag[sl],
                    R_T, body_pos)
                sdf_sub_v = sdf_eval(px_v, py_v, pz_v)

                px_w, py_w, pz_w = rotate_grid_3d(
                    comp.Xw_stag[sl], comp.Yw_stag[sl], comp.Zw_stag[sl],
                    R_T, body_pos)
                sdf_sub_w = sdf_eval(px_w, py_w, pz_w)

                # Sub-block body velocity (strided coord reads →
                # contiguous output from element-wise arithmetic)
                vel_sub_u = (
                    lin_vel[0]
                    + ang_vel[1] * (comp.Zu_stag[sl] - com_pos[2])
                    - ang_vel[2] * (comp.Yu_stag[sl] - com_pos[1]))
                vel_sub_v = (
                    lin_vel[1]
                    + ang_vel[2] * (comp.Xv_stag[sl] - com_pos[0])
                    - ang_vel[0] * (comp.Zv_stag[sl] - com_pos[2]))
                vel_sub_w = (
                    lin_vel[2]
                    + ang_vel[0] * (comp.Yw_stag[sl] - com_pos[1])
                    - ang_vel[1] * (comp.Xw_stag[sl] - com_pos[0]))

                # Sub-block union min  ─  read old union values
                # (contiguous copy avoids slow strided reads inside
                # torch.where), then write result back (strided write).
                old_cc = comp.sdf_val[sl].contiguous()
                mask_cc = sdf_sub < old_cc
                comp.sdf_val[sl] = torch.where(
                    mask_cc, sdf_sub, old_cc)

                old_u = comp.sdf_val_u[sl].contiguous()
                mask_u = sdf_sub_u < old_u
                comp.sdf_val_u[sl] = torch.where(
                    mask_u, sdf_sub_u, old_u)
                comp.body_u[sl] = torch.where(
                    mask_u, vel_sub_u,
                    comp.body_u[sl].contiguous())

                old_v = comp.sdf_val_v[sl].contiguous()
                mask_v = sdf_sub_v < old_v
                comp.sdf_val_v[sl] = torch.where(
                    mask_v, sdf_sub_v, old_v)
                comp.body_v[sl] = torch.where(
                    mask_v, vel_sub_v,
                    comp.body_v[sl].contiguous())

                old_w = comp.sdf_val_w[sl].contiguous()
                mask_w = sdf_sub_w < old_w
                comp.sdf_val_w[sl] = torch.where(
                    mask_w, sdf_sub_w, old_w)
                comp.body_w[sl] = torch.where(
                    mask_w, vel_sub_w,
                    comp.body_w[sl].contiguous())

            else:
                # ── Full-grid path (body covers >90 % of grid) ─────
                _rotate = (_rotate_grid_3d_compiled
                           if fs._compile_sdf else rotate_grid_3d)
                px, py, pz = _rotate(comp.X, comp.Y, comp.Z_grid,
                                     R_T, body_pos)
                sdf_cc = sdf_eval(px, py, pz)
                comp._sdf_sparse[body_i] = (None, sdf_cc)  # None = full grid

                # Evaluate SDF directly at staggered face locations
                # (exact interpolation instead of CC averaging)
                px_u, py_u, pz_u = _rotate(
                    comp.Xu_stag, comp.Yu_stag, comp.Zu_stag,
                    R_T, body_pos)
                sdf_u = sdf_eval(px_u, py_u, pz_u)

                px_v, py_v, pz_v = _rotate(
                    comp.Xv_stag, comp.Yv_stag, comp.Zv_stag,
                    R_T, body_pos)
                sdf_v = sdf_eval(px_v, py_v, pz_v)

                px_w, py_w, pz_w = _rotate(
                    comp.Xw_stag, comp.Yw_stag, comp.Zw_stag,
                    R_T, body_pos)
                sdf_w = sdf_eval(px_w, py_w, pz_w)

                vel_u = (lin_vel[0]
                         + ang_vel[1] * (comp.Zu_stag - com_pos[2])
                         - ang_vel[2] * (comp.Yu_stag - com_pos[1]))
                vel_v = (lin_vel[1]
                         + ang_vel[2] * (comp.Xv_stag - com_pos[0])
                         - ang_vel[0] * (comp.Zv_stag - com_pos[2]))
                vel_w = (lin_vel[2]
                         + ang_vel[0] * (comp.Yw_stag - com_pos[1])
                         - ang_vel[1] * (comp.Xw_stag - com_pos[0]))

                mask_cc = sdf_cc < comp.sdf_val
                comp.sdf_val = torch.where(mask_cc, sdf_cc, comp.sdf_val)

                mask_u = sdf_u < comp.sdf_val_u
                comp.sdf_val_u = torch.where(mask_u, sdf_u, comp.sdf_val_u)
                comp.body_u    = torch.where(mask_u, vel_u, comp.body_u)

                mask_v = sdf_v < comp.sdf_val_v
                comp.sdf_val_v = torch.where(mask_v, sdf_v, comp.sdf_val_v)
                comp.body_v    = torch.where(mask_v, vel_v, comp.body_v)

                mask_w = sdf_w < comp.sdf_val_w
                comp.sdf_val_w = torch.where(mask_w, sdf_w, comp.sdf_val_w)
                comp.body_w    = torch.where(mask_w, vel_w, comp.body_w)

            comp.com_pos[body_i] = com_pos
            body.com_pos = com_pos

    # ------------------------------------------------------------------
    #  Streaming fused-CUDA 3-D SDF update (Phase B)
    # ------------------------------------------------------------------
    #
    #  Per body, a single C++/CUDA kernel call replaces:
    #      * AABB sub-block rotate_grid_3d (× 4: cc + 3 faces)
    #      * 4 trilinear samples
    #      * 3 strided body-velocity expressions
    #      * 4 strided where()-based union-min writes
    #      * 4 strided body-velocity where() writes
    #  ...and writes the per-body sparse SDF (cc) into a pre-allocated
    #  AABB-shaped tensor used by the force integration.
    #
    #  Bodies without a regular SDF table (analytical) fall through to
    #  the streaming Python path used by `_update_3d`.
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

        # Kinematics (host)
        com_poses_np  = []
        urdf_poses_np = []
        Rs_np         = []
        lin_vels_np   = []
        ang_vels_np   = []
        for exp_data in self.data:
            sen = exp_data.sensors.links
            com_poses_np.append(np.asarray(sen.com_positions()[iteration, :], dtype=self.dtype_np))
            urdf_poses_np.append(np.asarray(sen.urdf_positions()[iteration, :], dtype=self.dtype_np))
            Rs_np.append(
                Rotation.from_quat(sen.urdf_orientations()[iteration, :])
                .as_matrix().astype(self.dtype_np)
            )
            lin_vels_np.append(np.asarray(sen.com_lin_velocities()[iteration, :], dtype=self.dtype_np))
            nlinks = len(sen.names)
            ang_vels_np.append(
                np.stack([
                    np.asarray(sen.com_ang_velocity(iteration, lk), dtype=self.dtype_np)
                    for lk in range(nlinks)
                ])
            )

        h_grid = float(comp.h)

        # The fused force kernel computes SDF samples and optional |∇φ|
        # correction on the fly, but force stress still needs lagged CC
        # normals.  On the first fused step these attributes may not exist
        # yet because normal recomputation happens after the SDF update in
        # BDIMhandler.step().  Initialize them once from the pre-update
        # union SDF before resetting fields below.
        if (
            getattr(fs, '_fused_sdf_forces_3d', False)
            and (
                getattr(fs, 'normal_x', None) is None
                or getattr(fs, 'normal_y', None) is None
                or getattr(fs, 'normal_z', None) is None
            )
        ):
            fs.normal_x, fs.normal_y, fs.normal_z = comp.compute_normals(
                comp.sdf_val
            )

        # Reset running-min fields
        comp._sdf_sparse = [None] * B
        comp.sdf_val.fill_(_FAR)
        comp.sdf_val_u.fill_(_FAR)
        comp.sdf_val_v.fill_(_FAR)
        comp.sdf_val_w.fill_(_FAR)
        comp.body_u.zero_()
        comp.body_v.zero_()
        comp.body_w.zero_()

        if not hasattr(comp, '_body_aabbs'):
            comp._body_aabbs = [None] * B

        if getattr(comp, '_grid_axes_1d', None) is None:
            comp._grid_axes_1d = (
                comp.x.contiguous(),
                comp.y.contiguous(),
                comp.z.contiguous(),
            )
        gx_1d, gy_1d, gz_1d = comp._grid_axes_1d

        # ------------------------------------------------------------------
        # Build / refresh the static per-body packed device tensors once.
        #
        # Memory note: the fused SDF+force path does NOT consume the
        # per-axis bx/by/bz_flat tables (it computes body-frame coords
        # analytically inside the kernel via the rotation matrix in
        # ``kin``).  Skip them in fused mode to avoid Σ_b (Mx+My+Mz)
        # device-side bytes per static cache.
        # ------------------------------------------------------------------
        fs_for_cache = self.fluid_solver
        _use_fused_cache = getattr(fs_for_cache, '_fused_sdf_forces_3d', False)

        sm = getattr(comp, '_stream_multi_static', None)
        if sm is None:
            F_chunks  = []
            F_off  = [0]
            shapes = []
            meta   = []
            if not _use_fused_cache:
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
                if not _use_fused_cache:
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
            if not _use_fused_cache:
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
            if not _use_fused_cache:
                del bx_chunks, by_chunks, bz_chunks
            comp._stream_multi_static = sm

        # ------------------------------------------------------------------
        # Per-step: compose body frames, AABBs, kinematics, sparse buffers.
        #
        # Vectorised path: avoids per-body torch ops (Rotation, AABB.item()
        # syncs, B individual H2D copies). All host-side work is done with
        # numpy in a single batched pass; tensors are uploaded once.
        # ------------------------------------------------------------------
        kin_static = getattr(comp, '_stream_kin_static', None)
        if kin_static is None:
            body_ids_np = np.asarray(
                [(int(a), int(l)) for (a, l) in comp.body_ids],
                dtype=np.int64,
            )
            local_lt_np = np.zeros((B, 3), dtype=self.dtype_np)
            local_lr_np = np.tile(np.eye(3, dtype=self.dtype_np), (B, 1, 1))
            sdf_lo_np   = np.empty((B, 3), dtype=self.dtype_np)
            sdf_hi_np   = np.empty((B, 3), dtype=self.dtype_np)
            for b, body in enumerate(comp.bodies):
                lp = getattr(body, 'local_pose', None)
                if lp is not None:
                    lp_np = np.asarray(lp, dtype=self.dtype_np)
                    local_lt_np[b] = lp_np[:3]
                    local_lr_np[b] = (
                        Rotation.from_euler('xyz', lp_np[3:])
                        .as_matrix().astype(self.dtype_np)
                    )
                sx, sy, sz = body.sdf.x, body.sdf.y, body.sdf.z
                sdf_lo_np[b] = (
                    float(sx[0].item()),  float(sy[0].item()),  float(sz[0].item()),
                )
                sdf_hi_np[b] = (
                    float(sx[-1].item()), float(sy[-1].item()), float(sz[-1].item()),
                )
            kin_static = {
                'body_ids':     body_ids_np,
                'local_lt':     local_lt_np,
                'local_lr':     local_lr_np,
                'local_center': 0.5 * (sdf_lo_np + sdf_hi_np),
                'local_half':   0.5 * (sdf_hi_np - sdf_lo_np),
                'grid_origin':  np.array([
                    float(comp.x[0].item()),
                    float(comp.y[0].item()),
                    float(comp.z[0].item()),
                ], dtype=self.dtype_np),
                'inv_h':        1.0 / float(comp.h),
                'gs':           np.asarray(gs, dtype=np.int64),
                'pad':          3,
            }
            comp._stream_kin_static = kin_static

        body_ids_np = kin_static['body_ids']

        # Gather per-body kinematics from the per-animat numpy snapshots above.
        urdf_pos = np.empty((B, 3), dtype=self.dtype_np)
        com_pos  = np.empty((B, 3), dtype=self.dtype_np)
        R_link   = np.empty((B, 3, 3), dtype=self.dtype_np)
        lin_vel  = np.empty((B, 3), dtype=self.dtype_np)
        ang_vel  = np.empty((B, 3), dtype=self.dtype_np)
        for b in range(B):
            a_id = int(body_ids_np[b, 0])
            l_id = int(body_ids_np[b, 1])
            urdf_pos[b] = urdf_poses_np[a_id][l_id]
            com_pos[b]  = com_poses_np[a_id][l_id]
            R_link[b]   = Rs_np[a_id][l_id]
            lin_vel[b]  = lin_vels_np[a_id][l_id]
            ang_vel[b]  = ang_vels_np[a_id][l_id]

        # Compose with per-body local pose: body_pos = urdf + R @ lt; body_R = R @ lr
        body_pos = urdf_pos + np.einsum(
            'bij,bj->bi', R_link, kin_static['local_lt'],
        )
        body_R = np.einsum(
            'bij,bjk->bik', R_link, kin_static['local_lr'],
        )  # (B,3,3)

        # Vectorised AABB of the oriented body-SDF box in world space.
        abs_R        = np.abs(body_R)
        world_half   = np.einsum('bij,bj->bi', abs_R, kin_static['local_half'])
        world_center = (
            np.einsum('bij,bj->bi', body_R, kin_static['local_center']) + body_pos
        )
        w_min = world_center - world_half
        w_max = world_center + world_half

        g0    = kin_static['grid_origin']
        inv_h = kin_static['inv_h']
        gs_np = kin_static['gs']
        pad   = kin_static['pad']

        i_lo = np.floor((w_min - g0) * inv_h).astype(np.int64) - pad
        i_hi = np.floor((w_max - g0) * inv_h).astype(np.int64) + 1 + pad
        np.maximum(i_lo, 0, out=i_lo)
        np.minimum(i_hi, gs_np[None, :], out=i_hi)

        dims     = i_hi - i_lo
        sub_vol  = dims.prod(axis=1)
        full_vol = int(gs_np.prod())
        # Bodies covering >90 % of the grid: fall back to full grid (matches
        # the scalar `_body_aabb_indices` heuristic).
        fallback = sub_vol > int(0.9 * full_vol)
        if fallback.any():
            i_lo[fallback] = 0
            i_hi[fallback] = gs_np[None, :]
            dims = i_hi - i_lo

        aabb_lo_h_np  = np.ascontiguousarray(i_lo, dtype=np.int64)
        aabb_dim_h_np = np.ascontiguousarray(dims, dtype=np.int64)
        cell_vols     = dims.prod(axis=1)
        cell_off_h_np = np.empty(B + 1, dtype=np.int64)
        cell_off_h_np[0] = 0
        np.cumsum(cell_vols, out=cell_off_h_np[1:])
        max_vol = int(cell_vols.max()) if B > 0 else 0

        # kin row layout (matches the original scalar path):
        #   [R^T (9) | body_pos (3) | com_pos (3) | lin_vel (3) | ang_vel (3)]  = 21
        # Row-major flatten of R^T equals body_R.transpose(0,2,1).reshape(B,9).
        kin_h_np = np.empty((B, 21), dtype=self.dtype_np)
        kin_h_np[:, 0:9]   = np.ascontiguousarray(
            body_R.transpose(0, 2, 1)
        ).reshape(B, 9)
        kin_h_np[:, 9:12]  = body_pos
        kin_h_np[:, 12:15] = com_pos
        kin_h_np[:, 15:18] = lin_vel
        kin_h_np[:, 18:21] = ang_vel

        # Update Python-side AABB metadata used downstream (slab split, forces).
        aabbs_for_split = []
        for b in range(B):
            i0 = int(aabb_lo_h_np[b, 0])
            j0 = int(aabb_lo_h_np[b, 1])
            k0 = int(aabb_lo_h_np[b, 2])
            Ai = int(aabb_dim_h_np[b, 0])
            Aj = int(aabb_dim_h_np[b, 1])
            Ak = int(aabb_dim_h_np[b, 2])
            aabb = (i0, i0 + Ai, j0, j0 + Aj, k0, k0 + Ak)
            comp._body_aabbs[b] = aabb
            aabbs_for_split.append(aabb)

        # Single H2D per packed tensor (instead of B individual transfers).
        kin       = torch.from_numpy(kin_h_np).to(
            self.device, dtype=self.dtype, non_blocking=True,
        )
        aabb_lo   = torch.from_numpy(aabb_lo_h_np).to(
            self.device, non_blocking=True,
        )
        aabb_dim  = torch.from_numpy(aabb_dim_h_np).to(
            self.device, non_blocking=True,
        )
        cell_off  = torch.from_numpy(cell_off_h_np).to(
            self.device, non_blocking=True,
        )
        com_pos_t = torch.from_numpy(com_pos).to(
            self.device, dtype=self.dtype, non_blocking=True,
        )

        # Maintain `comp.com_pos[b]` and `body.com_pos` views for downstream code.
        for b, body in enumerate(comp.bodies):
            comp.com_pos[b] = com_pos_t[b]
            body.com_pos = comp.com_pos[b]

        # Python-list copy of cell offsets for the per-body slab split below.
        cell_off_h = cell_off_h_np.tolist()

        fs = self.fluid_solver
        _use_fused = getattr(fs, '_fused_sdf_forces_3d', False)

        if _use_fused:
            # Per-body densities: use rho_body per body (same for all for now;
            # per-link densities can be wired here in future).
            # Persistent buffer: re-allocate only when B changes.
            _rho_buf = getattr(comp, '_fused_rho_bodies', None)
            if _rho_buf is None or _rho_buf.numel() != B \
                    or _rho_buf.dtype != self.dtype \
                    or _rho_buf.device != self.device:
                _rho_buf = torch.empty(
                    (B,), device=self.device, dtype=self.dtype,
                )
                comp._fused_rho_bodies = _rho_buf
            _rho_buf.fill_(float(fs.rho_body))
            rho_bodies = _rho_buf
            # winning_rho_cc: pre-filled with rho_fluid; the fused kernel
            # stamps each cell with the winning body's density.
            # Reuse the persistent buffer to avoid a full-grid allocation
            # every step (saves ~56 MB on a 900×300×52 grid).
            _wrcc = getattr(comp, '_winning_rho_cc', None)
            if _wrcc is None or _wrcc.shape != comp.sdf_val.shape:
                _wrcc = torch.empty(
                    comp.sdf_val.shape, device=self.device, dtype=self.dtype,
                )
            winning_rho_cc = _wrcc
            winning_rho_cc.fill_(float(fs.rho))
            # (B, 12) force accumulator, pre-zeroed.
            # Persistent buffer: re-allocate only when B changes.
            _out_buf = getattr(comp, '_fused_out_buf', None)
            if _out_buf is None or _out_buf.shape != (B, 12) \
                    or _out_buf.device != self.device:
                _out_buf = torch.empty(
                    (B, 12), dtype=torch.float64, device=self.device,
                )
                comp._fused_out_buf = _out_buf
            _out_buf.zero_()
            fused_out = _out_buf
            # nu_rho_field: size=1 for constant viscosity, full-grid for variable.
            if fs.use_variable_viscosity:
                # Persistent full-grid buffer for variable viscosity ν·ρ.
                # Reused every step; reallocates only when grid shape changes.
                _nu_rho_buf = getattr(comp, '_fused_nu_rho_buf', None)
                if _nu_rho_buf is None \
                        or _nu_rho_buf.shape != comp.sdf_val.shape \
                        or _nu_rho_buf.dtype != self.dtype \
                        or _nu_rho_buf.device != self.device:
                    _nu_rho_buf = torch.empty(
                        comp.sdf_val.shape,
                        device=self.device, dtype=self.dtype,
                    )
                    comp._fused_nu_rho_buf = _nu_rho_buf
                nu_rho_field = fs._compute_nu_rho_for_forces(
                    fs.u0, fs.v0, fs.w0, out=_nu_rho_buf,
                )
            else:
                # Tiny scalar buffer (size=1); persist to avoid even a
                # 1-element allocation per step.
                _nu_rho_scalar = getattr(comp, '_fused_nu_rho_scalar', None)
                if _nu_rho_scalar is None \
                        or _nu_rho_scalar.dtype != self.dtype \
                        or _nu_rho_scalar.device != self.device:
                    _nu_rho_scalar = torch.empty(
                        (1,), device=self.device, dtype=self.dtype,
                    )
                    comp._fused_nu_rho_scalar = _nu_rho_scalar
                _nu_rho_scalar.fill_(float(fs.nu) * float(fs.rho))
                nu_rho_field = _nu_rho_scalar
            delta_order = int(getattr(fs, 'force_delta_order', 1))
            eps_body   = float(comp.bodies[0].eps)
            eps_solver = float(fs.eps)
            h3         = float(fs.h ** 3)

            # Loud, one-time contiguity check on the full-grid inputs so
            # any non-contiguous tensor is caught here instead of being
            # silently duplicated by a `.contiguous()` copy at every
            # call site.  Fluid solver fields are constructed contiguous
            # (torch.zeros / torch.ones / torch.full), and normals come
            # from element-wise arithmetic on contiguous gradients, so
            # this assertion should always hold.
            if not getattr(comp, '_fused_contig_checked', False):
                _required_contig = {
                    'fs.u0': fs.u0, 'fs.v0': fs.v0,
                    'fs.w0': fs.w0, 'fs.p0': fs.p0,
                    'fs.normal_x': fs.normal_x,
                    'fs.normal_y': fs.normal_y,
                    'fs.normal_z': fs.normal_z,
                }
                for _name, _t in _required_contig.items():
                    if not _t.is_contiguous():
                        raise RuntimeError(
                            f"streaming_sdf_forces_fused_3d_multi requires "
                            f"contiguous full-grid inputs, but {_name} is "
                            f"non-contiguous (shape={tuple(_t.shape)}, "
                            f"strides={_t.stride()}). Calling .contiguous() "
                            f"here would silently allocate a full-grid copy "
                            f"every step. Fix the upstream construction of "
                            f"{_name} to return a contiguous tensor."
                        )
                comp._fused_contig_checked = True

            streaming_sdf_forces_fused_3d_multi(
                sm['F_flat'], sm['F_offsets'],
                sm['body_shapes'], sm['body_meta'], kin,
                aabb_lo, aabb_dim,
                gx_1d, gy_1d, gz_1d, h_grid, max_vol,
                comp.sdf_val, comp.sdf_val_u, comp.sdf_val_v, comp.sdf_val_w,
                comp.body_u,  comp.body_v,    comp.body_w,
                getattr(fs, '_sdf_interp_method', 0),
                rho_bodies, winning_rho_cc,
                fs.u0, fs.v0, fs.w0, fs.p0,
                fs.normal_x, fs.normal_y, fs.normal_z,
                nu_rho_field, eps_body, eps_solver, h3,
                delta_order,
                fused_out,
            )

            # Fused path does not populate per-body CC-SDF slabs.
            for body_i in range(B):
                comp._sdf_sparse[body_i] = None
            # Cache the union AABB directly so _compute_union_aabb_3d can
            # activate the cheap sub-block mu/normals path without reading
            # _sdf_sparse.  Without this, _compute_union_aabb_3d returns
            # None → _recompute_mu_normals_3d falls into the full-grid
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
            comp._fused_union_aabb = (_u_i0, _u_i1, _u_j0, _u_j1, _u_k0, _u_k1)

            # Store fused outputs on the composite body for downstream use.
            comp._fused_forces_out  = fused_out
            comp._winning_rho_cc    = winning_rho_cc

            # Stash per-step metadata (kin/aabb without sparse_cc_flat).
            comp._stream_multi_step = {
                'kin':     kin,
                'aabb_lo': aabb_lo,
                'aabb_dim': aabb_dim,
                'max_vol': max_vol,
                'gx':      gx_1d,
                'gy':      gy_1d,
                'gz':      gz_1d,
            }
        else:
            # ---- Two-phase path (original) ----
            sparse_flat = torch.zeros(
                int(cell_off_h_np[-1]), device=self.device, dtype=self.dtype,
            )
            streaming_sdf_min_3d_multi(
                sm['F_flat'],  sm['F_offsets'],
                sm['bx_flat'], sm['bx_offsets'],
                sm['by_flat'], sm['by_offsets'],
                sm['bz_flat'], sm['bz_offsets'],
                sm['body_shapes'], sm['body_meta'], kin,
                aabb_lo, aabb_dim, cell_off,
                gx_1d, gy_1d, gz_1d, h_grid, max_vol,
                comp.sdf_val, comp.sdf_val_u, comp.sdf_val_v, comp.sdf_val_w,
                comp.body_u,  comp.body_v,    comp.body_w,
                sparse_flat,
                getattr(fs, '_sdf_interp_method', 0),
            )

            # Split sparse_flat into per-body slabs and store
            for body_i, aabb in enumerate(aabbs_for_split):
                i0, i1, j0, j1, k0, k1 = aabb
                Ai, Aj, Ak = i1 - i0, j1 - j0, k1 - k0
                lo = cell_off_h[body_i]
                hi = cell_off_h[body_i + 1]
                slab = sparse_flat[lo:hi].view(Ai, Aj, Ak)
                comp._sdf_sparse[body_i] = (aabb, slab)

            comp._fused_forces_out = None
            comp._winning_rho_cc   = None
            comp._fused_union_aabb = None   # clear stale fused-path cache

            comp._stream_multi_step = {
                'kin':            kin,
                'aabb_lo':        aabb_lo,
                'aabb_dim':       aabb_dim,
                'max_vol':        max_vol,
                'gx':             gx_1d,
                'gy':             gy_1d,
                'gz':             gz_1d,
                'sparse_cc_flat': sparse_flat,
                'cell_offsets':   cell_off,
            }

    # ------------------------------------------------------------------
    def _update_3d_streaming(self, t, iteration, dt=1):

        # Multi-body batched fast path (Phase C): one Python op call
        # handles all bodies, eliminating ~B torch.ops dispatches/step
        # and the per-body launch+sync overhead.  Falls back to the
        # per-body sequential loop below if any body lacks meta.
        #
        # Mixed analytical + mesh composites
        # ----------------------------------
        # When ``_stream_meta`` is missing on at least one body (i.e. an
        # analytical body without a regular-grid SDF table) this gate
        # falls through to the per-body sequential loop below.  That
        # loop correctly handles mixed composites:
        #   * mesh bodies                    → ``streaming_sdf_min_3d``
        #     (single-body C++/CUDA kernel)
        #   * analytical / non-meta bodies   → ``_fallback_update_one_body_3d``
        #     (per-body PyTorch AABB + ``torch.where`` running-min).
        # Both write into the same union fields with running-min
        # semantics, so order is irrelevant and the result is correct.
        # The only thing lost in mixed mode is the multi-body batching
        # of the fast path (and the fused SDF+forces kernel, which is
        # SDF-table-only by design).
        comp_check = self.fluid_solver.composite_body
        if all(getattr(b, '_stream_meta', None) is not None
               for b in comp_check.bodies):
            return self._update_3d_streaming_multi(t, iteration, dt)

        fs   = self.fluid_solver
        comp = fs.composite_body
        gs   = fs.grid_shape
        _FAR = 1e4

        # Gather kinematics on the host (numpy).  Avoids the per-body
        # device->host syncs that .tolist() on a CUDA tensor would cost.
        com_poses_np  = []
        urdf_poses_np = []
        Rs_np         = []
        lin_vels_np   = []
        ang_vels_np   = []
        for exp_data in self.data:
            sen = exp_data.sensors.links
            com_poses_np.append(np.asarray(sen.com_positions()[iteration, :], dtype=self.dtype_np))
            urdf_poses_np.append(np.asarray(sen.urdf_positions()[iteration, :], dtype=self.dtype_np))
            Rs_np.append(
                Rotation.from_quat(sen.urdf_orientations()[iteration, :])
                .as_matrix().astype(self.dtype_np)
            )
            lin_vels_np.append(np.asarray(sen.com_lin_velocities()[iteration, :], dtype=self.dtype_np))
            nlinks = len(sen.names)
            ang_vels_np.append(
                np.stack([
                    np.asarray(sen.com_ang_velocity(iteration, lk), dtype=self.dtype_np)
                    for lk in range(nlinks)
                ])
            )

        h_grid = float(comp.h)

        # Reset running-min fields and per-body sparse storage
        comp._sdf_sparse = [None] * len(comp.bodies)
        comp.sdf_val.fill_(_FAR)
        comp.sdf_val_u.fill_(_FAR)
        comp.sdf_val_v.fill_(_FAR)
        comp.sdf_val_w.fill_(_FAR)
        comp.body_u.zero_()
        comp.body_v.zero_()
        comp.body_w.zero_()

        if not hasattr(comp, '_body_aabbs'):
            comp._body_aabbs = [None] * len(comp.bodies)

        # 1-D fluid grid axes (created lazily once)
        if getattr(comp, '_grid_axes_1d', None) is None:
            comp._grid_axes_1d = (
                comp.x.contiguous(),
                comp.y.contiguous(),
                comp.z.contiguous(),
            )
        gx_1d, gy_1d, gz_1d = comp._grid_axes_1d

        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]

            com_pos_np  = com_poses_np[animat_id][link_id]
            urdf_pos_np = urdf_poses_np[animat_id][link_id]
            R_np        = Rs_np[animat_id][link_id]
            lin_vel_np  = lin_vels_np[animat_id][link_id]
            ang_vel_np  = ang_vels_np[animat_id][link_id]

            meta = getattr(body, '_stream_meta', None)
            if meta is None:
                # Analytical / non-mesh body: fall back to Python path
                # by using the cached `_sdf_tri3d` (custom trilinear) or
                # plain `body.sdf` at full grid.  Re-emit kinematics as
                # device tensors for that path.
                com_pos  = torch.as_tensor(com_pos_np,  device=self.device, dtype=self.dtype)
                urdf_pos = torch.as_tensor(urdf_pos_np, device=self.device, dtype=self.dtype)
                R        = torch.as_tensor(R_np,        device=self.device, dtype=self.dtype)
                lin_vel  = torch.as_tensor(lin_vel_np,  device=self.device, dtype=self.dtype)
                ang_vel  = torch.as_tensor(ang_vel_np,  device=self.device, dtype=self.dtype)
                self._fallback_update_one_body_3d(
                    body, body_i, com_pos, urdf_pos, R, lin_vel, ang_vel,
                    h_grid, gs, comp,
                )
                continue

            # Compose body frame in numpy (small, cheap)
            local_translation = getattr(body, "_local_translation_t", None)
            local_rotation    = getattr(body, "_local_rotation_t",    None)
            local_pose = getattr(body, "local_pose", None)
            if local_pose is not None:
                if local_translation is None or local_rotation is None:
                    local_pose_np = np.asarray(local_pose, dtype=self.dtype_np)
                    body._local_translation_t = torch.tensor(
                        local_pose_np[:3], device=self.device, dtype=self.dtype)
                    body._local_rotation_t = torch.tensor(
                        Rotation.from_euler("xyz", local_pose_np[3:])
                        .as_matrix().astype(self.dtype_np),
                        device=self.device, dtype=self.dtype)
                local_pose_np = np.asarray(local_pose, dtype=self.dtype_np)
                lt_np = local_pose_np[:3]
                lr_np = Rotation.from_euler("xyz", local_pose_np[3:]) \
                    .as_matrix().astype(self.dtype_np)
                body_pos_np = urdf_pos_np + R_np @ lt_np
                body_rot_np = R_np @ lr_np
            else:
                body_pos_np = urdf_pos_np
                body_rot_np = R_np

            # Compute AABB on grid (need device tensors for the existing helper)
            R_t        = torch.as_tensor(body_rot_np, device=self.device, dtype=self.dtype)
            body_pos_t = torch.as_tensor(body_pos_np, device=self.device, dtype=self.dtype)

            aabb = self._body_aabb_indices(
                body, R_t, body_pos_t,
                comp.x, comp.y, comp.z,
                h_grid, gs, pad=3,
            )
            comp._body_aabbs[body_i] = aabb

            if aabb is None:
                # Body covers ~entire grid: clamp AABB to full grid.
                aabb = (0, gs[0], 0, gs[1], 0, gs[2])
                comp._body_aabbs[body_i] = aabb

            i0, i1, j0, j1, k0, k1 = aabb
            Ai, Aj, Ak = i1 - i0, j1 - j0, k1 - k0
            sparse_cc = torch.empty(
                (Ai, Aj, Ak), device=self.device, dtype=self.dtype)

            R_T_np = body_rot_np.T  # (3,3) row-major

            streaming_sdf_min_3d(
                meta['F'], meta['bx'], meta['by'], meta['bz'],
                meta['bx0'], meta['by0'], meta['bz0'],
                meta['bx_last'], meta['by_last'], meta['bz_last'],
                meta['inv_dx'], meta['inv_dy'], meta['inv_dz'], meta['inv_vol'],
                R_T_np.flatten().tolist(),
                body_pos_np.tolist(),
                com_pos_np.tolist(),
                lin_vel_np.tolist(),
                ang_vel_np.tolist(),
                gx_1d, gy_1d, gz_1d, h_grid,
                i0, i1, j0, j1, k0, k1,
                comp.sdf_val, comp.sdf_val_u, comp.sdf_val_v, comp.sdf_val_w,
                comp.body_u,  comp.body_v,    comp.body_w,
                sparse_cc,
                getattr(self.fluid_solver, '_sdf_interp_method', 0),
            )

            comp._sdf_sparse[body_i] = (aabb, sparse_cc)

            comp.com_pos[body_i] = torch.as_tensor(
                com_pos_np, device=self.device, dtype=self.dtype)
            body.com_pos = comp.com_pos[body_i]

    def _fallback_update_one_body_3d(self, body, body_i,
                                     com_pos, urdf_pos, R,
                                     lin_vel, ang_vel,
                                     h_grid, gs, comp):
        """Single-body PyTorch fallback used by `_update_3d_streaming`
        for bodies without a regular SDF table (analytical / etc.).

        Mirrors the per-body block of `_update_3d` for one body.
        """
        body_pos, body_rot = self._compose_body_frame_3d(body, urdf_pos, R)
        R_T = body_rot.T

        # AABB-clip via either the mesh axis tensors or analytical
        # bodies' ``local_aabb`` descriptor (handled inside the
        # helper).  Returns ``None`` when no descriptor is available
        # — the body then takes the full-grid fallback below.
        aabb = self._body_aabb_indices(
            body, body_rot, body_pos,
            comp.x, comp.y, comp.z,
            h_grid, gs, pad=3,
        )
        comp._body_aabbs[body_i] = aabb
        sdf_eval = getattr(body, '_sdf_tri3d', None) or body.sdf

        if aabb is not None:
            (i0, i1, j0, j1, k0, k1) = aabb
            sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))

            px, py, pz = rotate_grid_3d(
                comp.X[sl], comp.Y[sl], comp.Z_grid[sl], R_T, body_pos)
            sdf_sub = sdf_eval(px, py, pz)
            comp._sdf_sparse[body_i] = (aabb, sdf_sub)

            px_u, py_u, pz_u = rotate_grid_3d(
                comp.Xu_stag[sl], comp.Yu_stag[sl], comp.Zu_stag[sl],
                R_T, body_pos)
            sdf_sub_u = sdf_eval(px_u, py_u, pz_u)
            px_v, py_v, pz_v = rotate_grid_3d(
                comp.Xv_stag[sl], comp.Yv_stag[sl], comp.Zv_stag[sl],
                R_T, body_pos)
            sdf_sub_v = sdf_eval(px_v, py_v, pz_v)
            px_w, py_w, pz_w = rotate_grid_3d(
                comp.Xw_stag[sl], comp.Yw_stag[sl], comp.Zw_stag[sl],
                R_T, body_pos)
            sdf_sub_w = sdf_eval(px_w, py_w, pz_w)

            vel_sub_u = (lin_vel[0]
                         + ang_vel[1] * (comp.Zu_stag[sl] - com_pos[2])
                         - ang_vel[2] * (comp.Yu_stag[sl] - com_pos[1]))
            vel_sub_v = (lin_vel[1]
                         + ang_vel[2] * (comp.Xv_stag[sl] - com_pos[0])
                         - ang_vel[0] * (comp.Zv_stag[sl] - com_pos[2]))
            vel_sub_w = (lin_vel[2]
                         + ang_vel[0] * (comp.Yw_stag[sl] - com_pos[1])
                         - ang_vel[1] * (comp.Xw_stag[sl] - com_pos[0]))

            old_cc = comp.sdf_val[sl].contiguous()
            comp.sdf_val[sl] = torch.where(sdf_sub < old_cc, sdf_sub, old_cc)

            old_u = comp.sdf_val_u[sl].contiguous()
            mask_u = sdf_sub_u < old_u
            comp.sdf_val_u[sl] = torch.where(mask_u, sdf_sub_u, old_u)
            comp.body_u[sl] = torch.where(
                mask_u, vel_sub_u, comp.body_u[sl].contiguous())

            old_v = comp.sdf_val_v[sl].contiguous()
            mask_v = sdf_sub_v < old_v
            comp.sdf_val_v[sl] = torch.where(mask_v, sdf_sub_v, old_v)
            comp.body_v[sl] = torch.where(
                mask_v, vel_sub_v, comp.body_v[sl].contiguous())

            old_w = comp.sdf_val_w[sl].contiguous()
            mask_w = sdf_sub_w < old_w
            comp.sdf_val_w[sl] = torch.where(mask_w, sdf_sub_w, old_w)
            comp.body_w[sl] = torch.where(
                mask_w, vel_sub_w, comp.body_w[sl].contiguous())
        else:
            px, py, pz = rotate_grid_3d(
                comp.X, comp.Y, comp.Z_grid, R_T, body_pos)
            sdf_cc = sdf_eval(px, py, pz)
            comp._sdf_sparse[body_i] = (None, sdf_cc)

            px_u, py_u, pz_u = rotate_grid_3d(
                comp.Xu_stag, comp.Yu_stag, comp.Zu_stag, R_T, body_pos)
            sdf_u = sdf_eval(px_u, py_u, pz_u)
            px_v, py_v, pz_v = rotate_grid_3d(
                comp.Xv_stag, comp.Yv_stag, comp.Zv_stag, R_T, body_pos)
            sdf_v = sdf_eval(px_v, py_v, pz_v)
            px_w, py_w, pz_w = rotate_grid_3d(
                comp.Xw_stag, comp.Yw_stag, comp.Zw_stag, R_T, body_pos)
            sdf_w = sdf_eval(px_w, py_w, pz_w)

            vel_u = (lin_vel[0]
                     + ang_vel[1] * (comp.Zu_stag - com_pos[2])
                     - ang_vel[2] * (comp.Yu_stag - com_pos[1]))
            vel_v = (lin_vel[1]
                     + ang_vel[2] * (comp.Xv_stag - com_pos[0])
                     - ang_vel[0] * (comp.Zv_stag - com_pos[2]))
            vel_w = (lin_vel[2]
                     + ang_vel[0] * (comp.Yw_stag - com_pos[1])
                     - ang_vel[1] * (comp.Xw_stag - com_pos[0]))

            mask_cc = sdf_cc < comp.sdf_val
            comp.sdf_val = torch.where(mask_cc, sdf_cc, comp.sdf_val)
            mask_u = sdf_u < comp.sdf_val_u
            comp.sdf_val_u = torch.where(mask_u, sdf_u, comp.sdf_val_u)
            comp.body_u    = torch.where(mask_u, vel_u, comp.body_u)
            mask_v = sdf_v < comp.sdf_val_v
            comp.sdf_val_v = torch.where(mask_v, sdf_v, comp.sdf_val_v)
            comp.body_v    = torch.where(mask_v, vel_v, comp.body_v)
            mask_w = sdf_w < comp.sdf_val_w
            comp.sdf_val_w = torch.where(mask_w, sdf_w, comp.sdf_val_w)
            comp.body_w    = torch.where(mask_w, vel_w, comp.body_w)

        comp.com_pos[body_i] = com_pos
        body.com_pos = com_pos

    # ------------------------------------------------------------------
    #  Custom-trilinear (C++/CUDA) per-body SDF samplers
    # ------------------------------------------------------------------
    def _init_custom_trilinear_2d(self):
        """2-D analogue of :meth:`_init_custom_trilinear_3d`.

        For each 2-D body, stash the metadata needed by the
        ``streaming_sdf_min_2d_multi`` kernel directly on
        ``body._stream_meta``.  The streaming kernel does its own
        bilinear / biquadratic sampling from ``F``, ``bx``, ``by``
        and the cached scalars below.

        Mesh bodies use the precomputed ``body.sdf.{F, x, y}`` table.

        Analytical bodies (callable SDF, no precomputed table) are
        handled by pre-sampling the callable onto a regular local-frame
        grid derived from ``body.local_aabb`` (auto-derived from the
        contour in :meth:`BodyAnalytical._initialize_2d`).

        Raises ``RuntimeError`` if any body has neither a precomputed SDF
        grid nor a ``local_aabb`` descriptor.  Silent fallback to the
        Python path is intentionally forbidden when ``use_kernels=True``.
        """
        comp = self.fluid_solver.composite_body
        h    = float(self.fluid_solver.h)
        n_built = 0
        n_sampled = 0
        for body in comp.bodies:
            # ── Mesh body: has a precomputed regular-grid SDF ──────────
            if (hasattr(body, 'sdf')
                    and hasattr(body.sdf, 'F')
                    and hasattr(body.sdf, 'x')
                    and hasattr(body.sdf, 'y')):
                F  = body.sdf.F.contiguous()
                bx = body.sdf.x.contiguous()
                by = body.sdf.y.contiguous()
                dx_b = float(bx[1].item() - bx[0].item())
                dy_b = float(by[1].item() - by[0].item())
                body._stream_meta = {
                    'F':       F,
                    'bx':      bx,
                    'by':      by,
                    'bx0':     float(bx[0].item()),
                    'by0':     float(by[0].item()),
                    'bx_last': float(bx[-1].item()),
                    'by_last': float(by[-1].item()),
                    'inv_dx':  1.0 / dx_b,
                    'inv_dy':  1.0 / dy_b,
                    'inv_vol': 1.0 / (dx_b * dy_b),
                }
                n_built += 1
                continue

            # ── Analytical body: sample SDF onto a grid from local_aabb ─
            # local_aabb is in body-local frame relative to the URDF origin,
            # derived from the zero-level-set contour during _initialize_2d.
            # We sample at fluid-grid resolution so the streaming kernel's
            # bilinear interpolation is as accurate as the mesh-body path.
            local_aabb  = getattr(body, 'local_aabb', None)
            sdf_callable = getattr(body, 'sdf', None)
            if local_aabb is None or not callable(sdf_callable):
                body._stream_meta = None
                continue

            lo = local_aabb[0].cpu()
            hi = local_aabb[1].cpu()
            Mx = max(2, int(round(float(hi[0] - lo[0]) / h)) + 1)
            My = max(2, int(round(float(hi[1] - lo[1]) / h)) + 1)
            bx = torch.linspace(float(lo[0]), float(hi[0]), Mx,
                                dtype=self.dtype, device=self.device)
            by = torch.linspace(float(lo[1]), float(hi[1]), My,
                                dtype=self.dtype, device=self.device)
            X, Y = torch.meshgrid(bx, by, indexing='ij')
            with torch.no_grad():
                F = sdf_callable(X, Y).contiguous()
            dx_b = float(bx[1].item() - bx[0].item()) if Mx > 1 else h
            dy_b = float(by[1].item() - by[0].item()) if My > 1 else h
            body._stream_meta = {
                'F':       F,
                'bx':      bx,
                'by':      by,
                'bx0':     float(bx[0].item()),
                'by0':     float(by[0].item()),
                'bx_last': float(bx[-1].item()),
                'by_last': float(by[-1].item()),
                'inv_dx':  1.0 / dx_b,
                'inv_dy':  1.0 / dy_b,
                'inv_vol': 1.0 / (dx_b * dy_b),
            }
            n_sampled += 1

        print(f"  [stream-meta-2D] built {n_built}/{len(comp.bodies)} "
              f"per-body 2-D streaming-SDF metadata records "
              f"({n_sampled} sampled from analytical SDF)")

    def _update_2d_streaming_multi(self, t, iteration, dt=1):
        """2-D analogue of :meth:`_update_3d_streaming_multi`.

        Single Python op call dispatches B per-body 2-D streaming-SDF
        kernels in C++/CUDA, eliminating B torch.ops dispatches/step.
        Requires that ALL bodies expose ``_stream_meta`` (set by
        ``_init_custom_trilinear_2d`` for both mesh bodies and analytical
        bodies whose ``local_aabb`` is available).

        Side-effects per call:
            * fills ``comp.sdf_val``, ``comp.sdf_val_u``, ``comp.sdf_val_v``,
              ``comp.body_u``, ``comp.body_v`` (union over all bodies);
            * stashes per-step packed tensors on ``comp._stream_multi_step``
              for the fused 2-D forces kernel (``bdim_forces_2d_multi``);
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

        # ── Kinematics (host) ─────────────────────────────────────
        # Mirror the legacy ``_update_2d`` axis-projection: only the
        # in-plane axes (``self.lin_axes``) and the out-of-plane angular
        # axis (``self._2d_ang_ax``) are pulled from the FARMS sensors.
        com_poses_np  = []
        urdf_poses_np = []
        Rs_np         = []
        lin_vels_np   = []
        ang_vels_np   = []
        for exp_data in self.data:
            sen = exp_data.sensors.links
            com_poses_np.append(
                self.cython2numpy(
                    sen.com_positions()[iteration, :]
                ).cpu().numpy()[:, self.lin_axes].astype(self.dtype_np)
            )
            urdf_poses_np.append(
                self.cython2numpy(
                    sen.urdf_positions()[iteration, :]
                ).cpu().numpy()[:, self.lin_axes].astype(self.dtype_np)
            )
            R3 = (
                Rotation.from_quat(sen.urdf_orientations()[iteration, :])
                .as_matrix().astype(self.dtype_np)
            )
            # Project to the in-plane 2x2 rotation block.
            Rs_np.append(R3[:, self.lin_axes, :][:, :, self.lin_axes])
            lin_vels_np.append(
                self.cython2numpy(
                    sen.com_lin_velocities()[iteration, :]
                ).cpu().numpy()[:, self.lin_axes].astype(self.dtype_np)
            )
            nlinks = len(sen.names)
            ang_vels_np.append(
                np.asarray(
                    [sen.com_ang_velocity(iteration, lk)[self._2d_ang_ax]
                     for lk in range(nlinks)],
                    dtype=self.dtype_np,
                )
            )

        h_grid = float(comp.h)

        # Reset running-min fields and per-body sparse storage
        comp._sdf_sparse = [None] * B
        comp.sdf_val.fill_(_FAR)
        comp.sdf_val_u.fill_(_FAR)
        comp.sdf_val_v.fill_(_FAR)
        comp.body_u.zero_()
        comp.body_v.zero_()

        if not hasattr(comp, '_body_aabbs'):
            comp._body_aabbs = [None] * B

        if getattr(comp, '_grid_axes_1d', None) is None:
            comp._grid_axes_1d = (
                comp.x.contiguous(),
                comp.y.contiguous(),
            )
        gx_1d, gy_1d = comp._grid_axes_1d

        # ──────────────────────────────────────────────────────────
        # Build / refresh the static per-body packed device tensors once.
        # ──────────────────────────────────────────────────────────
        sm = getattr(comp, '_stream_multi_static_2d', None)
        if sm is None:
            F_chunks  = []
            bx_chunks = []; by_chunks = []
            F_off  = [0]; bx_off = [0]; by_off = [0]
            shapes = []
            meta   = []
            for body in comp.bodies:
                m = body._stream_meta
                F_chunks.append(m['F'].flatten())
                bx_chunks.append(m['bx']); by_chunks.append(m['by'])
                F_off.append(F_off[-1]   + m['F'].numel())
                bx_off.append(bx_off[-1] + m['bx'].numel())
                by_off.append(by_off[-1] + m['by'].numel())
                shapes.append([m['F'].shape[0], m['F'].shape[1]])
                meta.append([
                    m['bx0'], m['by0'],
                    m['bx_last'], m['by_last'],
                    m['inv_dx'], m['inv_dy'], m['inv_vol'],
                ])
            sm = {
                'F_flat':       torch.cat(F_chunks).contiguous(),
                'F_offsets':    torch.tensor(F_off,  dtype=torch.int64, device=self.device),
                'bx_flat':      torch.cat(bx_chunks).contiguous(),
                'bx_offsets':   torch.tensor(bx_off, dtype=torch.int64, device=self.device),
                'by_flat':      torch.cat(by_chunks).contiguous(),
                'by_offsets':   torch.tensor(by_off, dtype=torch.int64, device=self.device),
                'body_shapes':  torch.tensor(shapes, dtype=torch.int64, device=self.device),
                'body_meta':    torch.tensor(meta,   dtype=self.dtype,  device=self.device),
            }
            comp._stream_multi_static_2d = sm

        # ──────────────────────────────────────────────────────────
        # Per-step: compose body frames, AABBs, kinematics (numpy host).
        # ──────────────────────────────────────────────────────────
        kin_static = getattr(comp, '_stream_kin_static_2d', None)
        if kin_static is None:
            body_ids_np = np.asarray(
                [(int(a), int(l)) for (a, l) in comp.body_ids],
                dtype=np.int64,
            )
            # 2-D bodies have no `local_pose` concept (rigid-body links
            # in the 2-D projection collapse to identity local frames),
            # so the local translation / rotation are trivial.
            local_lt_np = np.zeros((B, 2), dtype=self.dtype_np)
            local_lr_np = np.tile(np.eye(2, dtype=self.dtype_np), (B, 1, 1))
            sdf_lo_np   = np.empty((B, 2), dtype=self.dtype_np)
            sdf_hi_np   = np.empty((B, 2), dtype=self.dtype_np)
            for b, body in enumerate(comp.bodies):
                m = body._stream_meta  # set for both mesh and analytical bodies
                sdf_lo_np[b] = (m['bx0'],     m['by0'])
                sdf_hi_np[b] = (m['bx_last'], m['by_last'])
            kin_static = {
                'body_ids':     body_ids_np,
                'local_lt':     local_lt_np,
                'local_lr':     local_lr_np,
                'local_center': 0.5 * (sdf_lo_np + sdf_hi_np),
                'local_half':   0.5 * (sdf_hi_np - sdf_lo_np),
                'grid_origin':  np.array([
                    float(comp.x[0].item()),
                    float(comp.y[0].item()),
                ], dtype=self.dtype_np),
                'inv_h':        1.0 / float(comp.h),
                'gs':           np.asarray(gs, dtype=np.int64),
                'pad':          3,
            }
            comp._stream_kin_static_2d = kin_static

        body_ids_np = kin_static['body_ids']

        # Gather per-body kinematics from the per-animat numpy snapshots.
        urdf_pos = np.empty((B, 2), dtype=self.dtype_np)
        com_pos  = np.empty((B, 2), dtype=self.dtype_np)
        R_link   = np.empty((B, 2, 2), dtype=self.dtype_np)
        lin_vel  = np.empty((B, 2), dtype=self.dtype_np)
        ang_vel  = np.empty((B,),    dtype=self.dtype_np)  # scalar in 2-D
        for b in range(B):
            a_id = int(body_ids_np[b, 0])
            l_id = int(body_ids_np[b, 1])
            urdf_pos[b] = urdf_poses_np[a_id][l_id]
            com_pos[b]  = com_poses_np[a_id][l_id]
            R_link[b]   = Rs_np[a_id][l_id]
            lin_vel[b]  = lin_vels_np[a_id][l_id]
            ang_vel[b]  = ang_vels_np[a_id][l_id]

        # Compose with per-body local pose (identity in 2-D, kept for
        # symmetry with the 3-D path).
        body_pos = urdf_pos + np.einsum(
            'bij,bj->bi', R_link, kin_static['local_lt'],
        )
        body_R = np.einsum(
            'bij,bjk->bik', R_link, kin_static['local_lr'],
        )  # (B, 2, 2)

        # Vectorised AABB of the oriented body-SDF box in world space.
        abs_R        = np.abs(body_R)
        world_half   = np.einsum('bij,bj->bi', abs_R, kin_static['local_half'])
        world_center = (
            np.einsum('bij,bj->bi', body_R, kin_static['local_center']) + body_pos
        )
        w_min = world_center - world_half
        w_max = world_center + world_half

        g0    = kin_static['grid_origin']
        inv_h = kin_static['inv_h']
        gs_np = kin_static['gs']
        pad   = kin_static['pad']

        i_lo = np.floor((w_min - g0) * inv_h).astype(np.int64) - pad
        i_hi = np.floor((w_max - g0) * inv_h).astype(np.int64) + 1 + pad
        np.maximum(i_lo, 0, out=i_lo)
        np.minimum(i_hi, gs_np[None, :], out=i_hi)
        # Bodies partially or entirely outside the grid produce i_hi < i_lo
        # (i_hi is clamped above by gs_np but not from below by i_lo).
        # Clamp to zero-size AABB for out-of-bounds bodies.
        np.maximum(i_hi, i_lo, out=i_hi)

        dims     = i_hi - i_lo
        sub_vol  = dims.prod(axis=1)
        full_vol = int(gs_np.prod())
        # Bodies covering >90 % of the grid: fall back to full grid (matches
        # the 3-D heuristic).
        fallback = sub_vol > int(0.9 * full_vol)
        if fallback.any():
            i_lo[fallback] = 0
            i_hi[fallback] = gs_np[None, :]
            dims = i_hi - i_lo

        aabb_lo_h_np  = np.ascontiguousarray(i_lo, dtype=np.int64)
        aabb_dim_h_np = np.ascontiguousarray(dims, dtype=np.int64)
        cell_vols     = dims.prod(axis=1)
        cell_off_h_np = np.empty(B + 1, dtype=np.int64)
        cell_off_h_np[0] = 0
        np.cumsum(cell_vols, out=cell_off_h_np[1:])
        max_vol = int(cell_vols.max()) if B > 0 else 0

        # 2-D kin row layout (matches the kernel signature):
        #   [R^T (4) | bp (2) | cm (2) | lv (2) | omega (1)]  = 11
        # Row-major flatten of R^T equals body_R.transpose(0,2,1).reshape(B,4).
        kin_h_np = np.empty((B, 11), dtype=self.dtype_np)
        kin_h_np[:, 0:4]   = np.ascontiguousarray(
            body_R.transpose(0, 2, 1)
        ).reshape(B, 4)
        kin_h_np[:, 4:6]   = body_pos
        kin_h_np[:, 6:8]   = com_pos
        kin_h_np[:, 8:10]  = lin_vel
        kin_h_np[:, 10]    = ang_vel

        # Update Python-side AABB metadata used downstream.
        aabbs_for_split = []
        for b in range(B):
            i0 = int(aabb_lo_h_np[b, 0])
            j0 = int(aabb_lo_h_np[b, 1])
            Ai = int(aabb_dim_h_np[b, 0])
            Aj = int(aabb_dim_h_np[b, 1])
            aabb = (i0, i0 + Ai, j0, j0 + Aj)
            comp._body_aabbs[b] = aabb
            aabbs_for_split.append(aabb)

        # Single H2D per packed tensor.
        kin       = torch.from_numpy(kin_h_np).to(
            self.device, dtype=self.dtype, non_blocking=True,
        )
        aabb_lo   = torch.from_numpy(aabb_lo_h_np).to(
            self.device, non_blocking=True,
        )
        aabb_dim  = torch.from_numpy(aabb_dim_h_np).to(
            self.device, non_blocking=True,
        )
        cell_off  = torch.from_numpy(cell_off_h_np).to(
            self.device, non_blocking=True,
        )
        com_pos_t = torch.from_numpy(com_pos).to(
            self.device, dtype=self.dtype, non_blocking=True,
        )
        urdf_pos_t = torch.from_numpy(urdf_pos).to(
            self.device, dtype=self.dtype, non_blocking=True,
        )
        R_t        = torch.from_numpy(body_R).to(
            self.device, dtype=self.dtype, non_blocking=True,
        )
        # Maintain `comp.com_pos[b]` views for downstream code.
        for b, body in enumerate(comp.bodies):
            comp.com_pos[b] = com_pos_t[b]
            body.com_pos = comp.com_pos[b]

        cell_off_h = cell_off_h_np.tolist()

        comp._fused_forces_out = None

        fs = self.fluid_solver
        interp_method = getattr(fs, '_sdf_interp_method', 0)

        if fs._fused_sdf_forces_2d:
            rho_bodies_buf = getattr(self, '_fused_rho_bodies_2d', None)
            if rho_bodies_buf is None or rho_bodies_buf.shape[0] != B:
                rho_bodies_buf = torch.full(
                    (B,), fs.rho_body, device=self.device, dtype=self.dtype,
                )
                self._fused_rho_bodies_2d = rho_bodies_buf

            winning_rho_cc = getattr(self, '_winning_rho_cc_2d', None)
            if winning_rho_cc is None or winning_rho_cc.shape != comp.sdf_val.shape:
                winning_rho_cc = torch.empty_like(comp.sdf_val)
                self._winning_rho_cc_2d = winning_rho_cc
            winning_rho_cc.fill_(fs.rho)

            fused_out = getattr(self, '_fused_out_buf_2d', None)
            if fused_out is None or fused_out.shape[0] != B:
                fused_out = torch.zeros((B, 6), dtype=torch.float64, device=self.device)
                self._fused_out_buf_2d = fused_out
            else:
                fused_out.zero_()

            if getattr(fs, 'use_variable_viscosity', False):
                nu_rho_buf = getattr(self, '_fused_nu_rho_buf_2d', None)
                if nu_rho_buf is None or nu_rho_buf.shape != comp.sdf_val.shape:
                    nu_rho_buf = torch.empty_like(comp.sdf_val)
                    self._fused_nu_rho_buf_2d = nu_rho_buf
                nu_rho_field = fs._compute_nu_rho_for_forces(fs.u0, fs.v0, out=nu_rho_buf)
            else:
                nu_rho_scalar_buf = getattr(self, '_fused_nu_rho_scalar_2d', None)
                if nu_rho_scalar_buf is None:
                    nu_rho_scalar_buf = torch.empty(
                        (1,), device=self.device, dtype=self.dtype,
                    )
                    self._fused_nu_rho_scalar_2d = nu_rho_scalar_buf
                nu_rho_scalar_buf.fill_(fs.nu * fs.rho)
                nu_rho_field = nu_rho_scalar_buf

            eps_body_val = float(comp.bodies[0].eps)
            eps_solver_val = float(fs.eps)

            nx_cc_t = getattr(fs, 'normal_x', None)
            ny_cc_t = getattr(fs, 'normal_y', None)
            if nx_cc_t is None or ny_cc_t is None:
                (nx_cc_t, ny_cc_t) = comp.compute_normals(comp.sdf_val)

            if not getattr(self, '_fused_contig_checked_2d', False):
                for _t, _name in [
                    (fs.u0, 'u0'), (fs.v0, 'v0'), (fs.p0, 'p0'),
                    (nx_cc_t, 'normal_x'), (ny_cc_t, 'normal_y'),
                ]:
                    if not _t.is_contiguous():
                        import warnings
                        warnings.warn(
                            f"streaming_sdf_forces_fused_2d_multi: {_name} is not "
                            "contiguous; forcing contiguous copy.",
                            stacklevel=2,
                        )
                self._fused_contig_checked_2d = True

            streaming_sdf_forces_fused_2d_multi(
                sm['F_flat'], sm['F_offsets'],
                sm['body_shapes'], sm['body_meta'], kin,
                aabb_lo, aabb_dim,
                gx_1d, gy_1d, h_grid, max_vol,
                comp.sdf_val, comp.sdf_val_u, comp.sdf_val_v,
                comp.body_u, comp.body_v,
                interp_method,
                rho_bodies_buf, winning_rho_cc,
                fs.u0.contiguous(), fs.v0.contiguous(), fs.p0.contiguous(),
                nx_cc_t.contiguous(), ny_cc_t.contiguous(),
                nu_rho_field,
                eps_body_val, eps_solver_val,
                fs.h2,
                getattr(fs, 'force_delta_order', 1),
                fused_out,
            )

            for b in range(B):
                comp._sdf_sparse[b] = None

            comp._fused_forces_out = fused_out
            comp._winning_rho_cc   = winning_rho_cc

            comp._stream_multi_step = {
                'kin':      kin,
                'aabb_lo':  aabb_lo,
                'aabb_dim': aabb_dim,
                'max_vol':  max_vol,
                'gx':       gx_1d,
                'gy':       gy_1d,
            }

        else:
            sparse_n = int(cell_off_h_np[-1])
            sparse_flat = getattr(self, '_stream_sparse_flat_2d', None)
            if (
                sparse_flat is None
                or sparse_flat.numel() < sparse_n
                or sparse_flat.device != self.device
                or sparse_flat.dtype != self.dtype
            ):
                sparse_flat = torch.empty(
                    sparse_n, device=self.device, dtype=self.dtype,
                )
                self._stream_sparse_flat_2d = sparse_flat
            else:
                sparse_flat = sparse_flat[:sparse_n]

            streaming_sdf_min_2d_multi(
                sm['F_flat'],  sm['F_offsets'],
                sm['bx_flat'], sm['bx_offsets'],
                sm['by_flat'], sm['by_offsets'],
                sm['body_shapes'], sm['body_meta'], kin,
                aabb_lo, aabb_dim, cell_off,
                gx_1d, gy_1d, h_grid, max_vol,
                comp.sdf_val, comp.sdf_val_u, comp.sdf_val_v,
                comp.body_u,  comp.body_v,
                sparse_flat,
                interp_method,
            )

            for body_i, aabb in enumerate(aabbs_for_split):
                i0, i1, j0, j1 = aabb
                Ai, Aj = i1 - i0, j1 - j0
                lo = cell_off_h[body_i]
                hi = cell_off_h[body_i + 1]
                slab = sparse_flat[lo:hi].view(Ai, Aj)
                comp._sdf_sparse[body_i] = (aabb, slab)

            comp._stream_multi_step = {
                'kin':             kin,
                'aabb_lo':         aabb_lo,
                'aabb_dim':        aabb_dim,
                'max_vol':         max_vol,
                'gx':              gx_1d,
                'gy':              gy_1d,
                'sparse_cc_flat':  sparse_flat,
                'cell_offsets':    cell_off,
            }

        # ── Per-body: contour update + (optional) contour mask ──
        # 2-D bodies expose 1-D contours; the legacy ``_update_2d`` keeps
        # them consistent with the per-step rotation/translation.  We
        # mirror that loop here so downstream code (force projection,
        # plotting) sees the same per-body tensors.
        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]

            # contour update (world frame)
            body.cnt_update = R_t[body_i] @ body.cnt + urdf_pos_t[body_i, :, None]

            # optional contour mask for overlapping links
            if self.contour_mask:
                x_cnt = body.cnt_update[0]
                y_cnt = body.cnt_update[1]
                prev_body_i = self._prev_body_index[body_i]
                next_body_i = self._next_body_index[body_i]

                if prev_body_i is None and next_body_i is None:
                    mask = torch.ones_like(x_cnt, dtype=torch.bool)
                else:
                    # Per-animat numpy snapshots are stored above; rebuild
                    # the device tensors needed for the SDF query.
                    Rs_a = torch.as_tensor(
                        Rs_np[animat_id], device=self.device, dtype=self.dtype,
                    )
                    urdf_a = torch.as_tensor(
                        urdf_poses_np[animat_id], device=self.device, dtype=self.dtype,
                    )
                    if prev_body_i is None:
                        (_, next_link_id) = comp.body_ids[next_body_i]
                        body_p = comp.bodies[next_body_i]
                        pt = Rs_a[next_link_id].T @ (
                            torch.stack((x_cnt, y_cnt))
                            - urdf_a[next_link_id][:, None]
                        )
                        mask = (body_p.sdf(pt[0], pt[1]) - body.h) >= 0
                    elif next_body_i is None:
                        (_, prev_link_id) = comp.body_ids[prev_body_i]
                        body_m = comp.bodies[prev_body_i]
                        pt = Rs_a[prev_link_id].T @ (
                            torch.stack((x_cnt, y_cnt))
                            - urdf_a[prev_link_id][:, None]
                        )
                        mask = (body_m.sdf(pt[0], pt[1]) - body.h) >= 0
                    else:
                        (_, prev_link_id) = comp.body_ids[prev_body_i]
                        body_m = comp.bodies[prev_body_i]
                        pt_m = Rs_a[prev_link_id].T @ (
                            torch.stack((x_cnt, y_cnt))
                            - urdf_a[prev_link_id][:, None]
                        )
                        (_, next_link_id) = comp.body_ids[next_body_i]
                        body_p = comp.bodies[next_body_i]
                        pt_p = Rs_a[next_link_id].T @ (
                            torch.stack((x_cnt, y_cnt))
                            - urdf_a[next_link_id][:, None]
                        )
                        sdf_m = body_m.sdf(pt_m[0], pt_m[1]) - body.h
                        sdf_p = body_p.sdf(pt_p[0], pt_p[1]) - body.h
                        mask = (sdf_m >= 0) & (sdf_p >= 0)
                body.mask = mask

            body.r_com   = body.cnt_update - com_pos_t[body_i, :, None]
            # body.com_pos already set above.

    # ------------------------------------------------------------------
    #  Custom-trilinear (C++/CUDA) per-body SDF samplers
    # ------------------------------------------------------------------
    def _init_custom_trilinear_3d(self):
        """Build a per-body ``RegularGridInterpolator3D`` (C++/CUDA
        trilinear) and stash it on ``body._sdf_tri3d``.

        Used by the streaming ``_update_3d`` path to bypass
        ``grid_sample`` (the default backend of
        ``RegularGridInterpolatorAutomatic`` for 3-D, which the bodies
        construct in ``_initialize_3d_mesh``).  The custom kernel skips
        coordinate normalisation and the 5-D reshape, and matches
        ``grid_sample(padding_mode='border')`` semantics via
        ``fill_method=3``.
        """

        comp = self.fluid_solver.composite_body
        n_built = 0
        for body in comp.bodies:
            if not (hasattr(body, 'sdf')
                    and hasattr(body.sdf, 'F')
                    and hasattr(body.sdf, 'x')
                    and hasattr(body.sdf, 'y')
                    and hasattr(body.sdf, 'z')):
                # Analytical / non-mesh body — keep the existing sampler.
                body._sdf_tri3d = None
                continue
            body._sdf_tri3d = RegularGridInterpolator3D(
                (body.sdf.x.contiguous(),
                 body.sdf.y.contiguous(),
                 body.sdf.z.contiguous()),
                body.sdf.F.contiguous(),
                fill_value="nearest",
            )
            # Cache scalars for the streaming_sdf_min_3d kernel
            bx_t = body._sdf_tri3d.x
            by_t = body._sdf_tri3d.y
            bz_t = body._sdf_tri3d.z
            dx_b = float(body._sdf_tri3d.dx)
            dy_b = float(body._sdf_tri3d.dy)
            dz_b = float(body._sdf_tri3d.dz)
            body._stream_meta = {
                'F':       body._sdf_tri3d.F,
                'bx':      bx_t,
                'by':      by_t,
                'bz':      bz_t,
                'bx0':     float(bx_t[0].item()),
                'by0':     float(by_t[0].item()),
                'bz0':     float(bz_t[0].item()),
                'bx_last': float(bx_t[-1].item()),
                'by_last': float(by_t[-1].item()),
                'bz_last': float(bz_t[-1].item()),
                'inv_dx':  1.0 / dx_b,
                'inv_dy':  1.0 / dy_b,
                'inv_dz':  1.0 / dz_b,
                'inv_vol': 1.0 / (dx_b * dy_b * dz_b),
            }
            n_built += 1
        print(f"  [custom-trilinear-3D] built {n_built}/{len(comp.bodies)} "
              f"per-body C++/CUDA trilinear samplers (replacing grid_sample)")

    # ==================================================================
    #  apply_forces: fluid -> body forces via MuJoCo xfrc_applied
    # ==================================================================
    def apply_forces(self, task, physics):
        if self.ndim == 3:
            self._apply_forces_3d(task, physics)
        else:
            self._apply_forces_2d(task, physics)

    def _apply_forces_2d(self, task, physics):
        fs = self.fluid_solver
        s  = self.force_scaling
        fx_idx, fy_idx, torque_idx = self._2d_force_axes

        # Single GPU→CPU transfer instead of 6 separate .cpu().numpy() calls
        forces_gpu = torch.stack([
            fs.friction_force_lin_x,
            fs.friction_force_lin_y,
            fs.friction_force_ang_z,
            fs.pressure_force_x,
            fs.pressure_force_y,
            fs.pressure_force_ang_z,
        ])                                          # (6, B)
        forces_cpu = (s * forces_gpu).cpu().numpy() # single sync
        friction_force_lin_x = forces_cpu[0]
        friction_force_lin_y = forces_cpu[1]
        friction_force_ang_z = forces_cpu[2]
        pressure_force_x     = forces_cpu[3]
        pressure_force_y     = forces_cpu[4]
        pressure_force_ang_z = forces_cpu[5]

        # Lazy-init buoyancy parameters (xz plane only)
        if self._2d_has_buoyancy and not self._buoyancy_initialized:
            self._init_buoyancy_params(task, physics)

        comp    = fs.composite_body
        surface = self.water_surface
        g_z     = self.gravity_z

        for body_i in range(len(comp.bodies)):
            (animat_id, link_id) = comp.body_ids[body_i]
            ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            # FARMS-style buoyancy: only active for xz plane (fluid y = MuJoCo z)
            buoyancy_y = 0.0
            if self._2d_has_buoyancy:
                mass   = self._buoy_mass[body_i]
                density = self._buoy_density[body_i]
                height = self._buoy_height[body_i]
                # comp.com_pos[body_i][1] = fluid y = MuJoCo z (vertical)
                pos_z  = float(comp.com_pos[body_i][1])
                if mass > 0 and height > 0 and pos_z - height < surface:
                    frac = min((surface + height - pos_z) / (2.0 * height), 1.0)
                    buoyancy_y = -self.rho_fluid * mass * g_z / density * frac

            physics.data.xfrc_applied[ind, fx_idx] = (
                friction_force_lin_x[body_i] + pressure_force_x[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, fy_idx] = (
                friction_force_lin_y[body_i] + pressure_force_y[body_i] + buoyancy_y
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, torque_idx] = (
                friction_force_ang_z[body_i] + pressure_force_ang_z[body_i]
            ) * task.units.newtons

    def _apply_forces_3d(self, task, physics):
        fs = self.fluid_solver
        s  = self.force_scaling

        # Single GPU→CPU transfer instead of 12 separate .cpu().numpy()
        # calls (each one was an implicit sync + DtoH copy and the single
        # biggest contributor to "Other (residual)" at low N).
        # Mirrors the 2-D batched transfer in _apply_forces_2d.
        forces_gpu = torch.stack([
            fs.friction_force_lin_x,
            fs.friction_force_lin_y,
            fs.friction_force_lin_z,
            fs.friction_force_ang_x,
            fs.friction_force_ang_y,
            fs.friction_force_ang_z,
            fs.pressure_force_x,
            fs.pressure_force_y,
            fs.pressure_force_z,
            fs.pressure_force_ang_x,
            fs.pressure_force_ang_y,
            fs.pressure_force_ang_z,
        ])                                          # (12, B)
        forces_cpu = (s * forces_gpu).cpu().numpy() # single sync
        friction_force_lin_x = forces_cpu[0]
        friction_force_lin_y = forces_cpu[1]
        friction_force_lin_z = forces_cpu[2]
        friction_force_ang_x = forces_cpu[3]
        friction_force_ang_y = forces_cpu[4]
        friction_force_ang_z = forces_cpu[5]
        pressure_force_x     = forces_cpu[6]
        pressure_force_y     = forces_cpu[7]
        pressure_force_z     = forces_cpu[8]
        pressure_force_ang_x = forces_cpu[9]
        pressure_force_ang_y = forces_cpu[10]
        pressure_force_ang_z = forces_cpu[11]

        # ---- FARMS-identical buoyancy (drag.pyx  compute_buoyancy) ----
        # Lazy-init per-body mass & half-height on first call
        if not self._buoyancy_initialized:
            self._init_buoyancy_params(task, physics)

        comp    = fs.composite_body
        surface = self.water_surface
        g_z     = self.gravity_z          # e.g. -9.81

        for body_i in range(len(comp.bodies)):
            (animat_id, link_id) = comp.body_ids[body_i]
            ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            # FARMS buoyancy per link
            mass   = self._buoy_mass[body_i]
            density = self._buoy_density[body_i]
            height = self._buoy_height[body_i]
            pos_z  = float(comp.com_pos[body_i][2])

            buoyancy_z = 0.0
            if mass > 0 and height > 0 and pos_z - height < surface:
                frac = min((surface + height - pos_z) / (2.0 * height), 1.0)
                # -rho_water * (mass/density) * gravity * frac
                # = -rho_water * V_link * gravity * frac  (upward when g<0)
                buoyancy_z = (
                    -self.rho_fluid * mass * g_z / density * frac
                )

            physics.data.xfrc_applied[ind, 0] = (
                friction_force_lin_x[body_i] + pressure_force_x[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 1] = (
                friction_force_lin_y[body_i] + pressure_force_y[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 2] = (
                friction_force_lin_z[body_i] + pressure_force_z[body_i]
                + buoyancy_z
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 3] = (
                friction_force_ang_x[body_i] + pressure_force_ang_x[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 4] = (
                friction_force_ang_y[body_i] + pressure_force_ang_y[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 5] = (
                friction_force_ang_z[body_i] + pressure_force_ang_z[body_i]
            ) * task.units.newtons

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

            # 1. update SDF + body velocities from FARMS kinematics
            self.update(t, iteration, dt=timestep)

            # 2. recompute mu / normal fields
            if self.ndim == 3:
                fs._recompute_mu_normals_3d()
            else:
                fs._recompute_mu_normals_2d()

            # 3. BDIM fluid step
            if self.ndim == 3:
                (u, v, w, p) = fs.fluid_step(
                    fs.u0, fs.v0, fs.w0, fs.p0, timestep
                )
                if self.zero_pressure_inside:
                    p = torch.where(fs.composite_body.sdf_val < 0, 0, p)
                (fs.u0, fs.v0, fs.w0, fs.p0) = (u, v, w, p)
            else:
                (u, v, p) = fs.fluid_step(
                    fs.u0, fs.v0, fs.p0, timestep
                )
                if self.zero_pressure_inside:
                    p = torch.where(fs.composite_body.sdf_val < 0, 0, p)
                (fs.u0, fs.v0, fs.p0) = (u, v, p)

            # 3b. bail out immediately if the fluid blew up
            fs.check_explosion(iteration)

            # 4. compute fluid forces on each body
            if self.ndim == 3:
                fs.forces_method2_3d(fs.u0, fs.v0, fs.w0, fs.p0, iteration)
            elif self.force_method == "method1":
                fs.forces_method1(fs.u0, fs.v0, fs.p0, iteration)
            else:
                fs.forces_method2(fs.u0, fs.v0, fs.p0, iteration)

            fs.__dict__.update(_FS_FREE_AFTER_FORCES_3D)

            # 5. plotting / saving
            if self.ndim == 3:
                fs.terminate = fs.plotting_and_saving(
                    fs.u0, fs.v0, fs.p0, iteration, w_vel=fs.w0, check_termination=False
                )
            else:
                fs.terminate = fs.plotting_and_saving(
                    fs.u0, fs.v0, fs.p0, iteration, check_termination=False
                )

            # 6. apply forces to MuJoCo bodies
            self.apply_forces(task, physics)

            # 7. free BDIM intermediates to reclaim GPU memory
            fs._release_bdim_fields()

        self.iteration += 1
