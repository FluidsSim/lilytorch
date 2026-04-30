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
import torch


# ---------------------------------------------------------------------
# Pre-built attribute-free dicts used inside the hot path of
# `_fluid_step_3d` to release BDIM intermediates with one
# ``__dict__.update`` call instead of N × (hasattr + setattr) Python
# lookups.  Building these dicts at import time keeps the per-step cost
# at one C-level dict merge.
# ---------------------------------------------------------------------
_FS_FREE_AFTER_BDIM_3D = {
    'mu1_all_u': None, 'mu1_all_v': None, 'mu1_all_w': None,
    'normal_x_u': None, 'normal_y_u': None, 'normal_z_u': None,
    'normal_x_v': None, 'normal_y_v': None, 'normal_z_v': None,
    'normal_x_w': None, 'normal_y_w': None, 'normal_z_w': None,
    # CC mu1 / m_m0 are also not needed by project
    'mu1_all': None, 'm_m0_all': None,
}
_FS_FREE_AFTER_VAR_DENS_3D = {
    'mu0_all_u': None, 'mu0_all_v': None, 'mu0_all_w': None,
    'mu0_all': None,
}
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
        if dtype is not None:
            self.dtype = dtype
        else:
            dtype_str = self.pars["solver"].get("dtype", "float32")
            self.dtype = torch.float32 if dtype_str == "float32" else torch.float64
        self.dtype_np = np.float32 if self.dtype == torch.float32 else np.float64
        self._prev_body_index = ()
        self._next_body_index = ()

        # ---- bookkeeping ----
        self.data = data          # list[AnimatData] from FARMS
        self.iteration = 0
        self.terminate = False

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
        self.rho_body  = self.pars["solver"].get("rho_body", 1000.0)

        # ---- toggles ----
        self.zero_pressure_inside = self.pars["solver"].get(
            "zero_pressure_inside", False
        )
        self.contour_mask = self.pars.get("body", {}).get("contour_mask", False)
        self.force_method = self.pars["solver"].get("force_method", "method2")
        # Heun (RK2) for the full BDIM-projection cycle (2-D only).
        # Off by default (Euler) for backwards compatibility; enable for
        # benchmarks that require RK2 accuracy (e.g. Gazzola sedimentation).
        self._use_heun = self.pars["solver"].get("heun", False) if self.ndim == 2 else False

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

        # ---- explosion detection (cheap per-step NaN + |u|_max guard) ----
        # vmax_abort defaults to a generous multiple of the inlet speed
        # (or 100 m/s if no inlet). Finite-check is always on.
        u_inlet = float(self.fluid_solver.adv_diff_solver.BC_values_u[1])
        self._vmax_abort = float(self.pars["solver"].get(
            "vmax_abort", max(100.0 * abs(u_inlet), 100.0),
        ))

        # ---- optional physics solref tweak ----
        solref = self.pars.get("physics", {}).get("solref", None)
        if solref is not None:
            physics.model.geom_solref[:, 0] = solref[0]
            physics.model.geom_solref[:, 1] = solref[1]

        # ---- allocate per-body SDF / velocity arrays if missing ----
        comp = self.fluid_solver.composite_body
        gs = self.fluid_solver.grid_shape
        if self.ndim == 3:
            comp.sdf_vals = torch.zeros(
                (comp.nbodies, *gs), device=self.device, dtype=self.dtype,
            )
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
            # 2-D update uses per-body stacks for argmin-based union
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

            # Pre-build batched SDF tensor for grid_sample (2-D only)
            self._init_batched_sdf_2d()

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
        self._buoy_density = np.full(n, float(self.rho_body))
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
                    density = float(self.rho_body)
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
        fs   = self.fluid_solver
        comp = fs.composite_body
        B    = comp.nbodies

        _FAR = 1e4   # far-field SDF sentinel (same as 3-D path)

        # gather per-animat kinematics
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

        # ── Stack per-body kinematics into batched GPU tensors ────
        urdf_t = torch.stack([urdf_poses[aid][lid] for aid, lid in comp.body_ids])  # (B, 2)
        com_t  = torch.stack([com_poses[aid][lid]  for aid, lid in comp.body_ids])   # (B, 2)
        R_t    = torch.stack([Rs[aid][lid]         for aid, lid in comp.body_ids])    # (B, 2, 2)
        lv_t   = torch.stack([lin_vels[aid][lid]   for aid, lid in comp.body_ids])   # (B, 2)
        av_t   = torch.stack([ang_vels[aid][lid]   for aid, lid in comp.body_ids])   # (B,)
        R_T_t  = R_t.transpose(1, 2)  # (B, 2, 2)

        # ── Batched rotation + grid_sample for CC grid ────────────
        dx = comp.X.unsqueeze(0) - urdf_t[:, 0, None, None]   # (B, nx, ny)
        dy = comp.Y.unsqueeze(0) - urdf_t[:, 1, None, None]   # (B, nx, ny)
        px = R_T_t[:, 0, 0, None, None] * dx + R_T_t[:, 0, 1, None, None] * dy
        py = R_T_t[:, 1, 0, None, None] * dx + R_T_t[:, 1, 1, None, None] * dy
        x_n = px * self._sdf_x_scale[:, None, None] + self._sdf_x_offset[:, None, None]
        y_n = py * self._sdf_y_scale[:, None, None] + self._sdf_y_offset[:, None, None]
        # grid_sample grid: [..., 0] → W (y axis), [..., 1] → H (x axis)
        sdf_cc_all = torch.nn.functional.grid_sample(
            self._sdf_batch, torch.stack([y_n, x_n], dim=-1),
            mode='bilinear', padding_mode='border', align_corners=True,
        ).squeeze(1)                                              # (B, nx, ny)

        # ── Batched rotation + grid_sample for u-staggered grid ───
        dx = comp.Xu_stag.unsqueeze(0) - urdf_t[:, 0, None, None]
        dy = comp.Yu_stag.unsqueeze(0) - urdf_t[:, 1, None, None]
        px = R_T_t[:, 0, 0, None, None] * dx + R_T_t[:, 0, 1, None, None] * dy
        py = R_T_t[:, 1, 0, None, None] * dx + R_T_t[:, 1, 1, None, None] * dy
        x_n = px * self._sdf_x_scale[:, None, None] + self._sdf_x_offset[:, None, None]
        y_n = py * self._sdf_y_scale[:, None, None] + self._sdf_y_offset[:, None, None]
        sdf_u_all = torch.nn.functional.grid_sample(
            self._sdf_batch, torch.stack([y_n, x_n], dim=-1),
            mode='bilinear', padding_mode='border', align_corners=True,
        ).squeeze(1)                                              # (B, nx, ny)

        # ── Batched rotation + grid_sample for v-staggered grid ───
        dx = comp.Xv_stag.unsqueeze(0) - urdf_t[:, 0, None, None]
        dy = comp.Yv_stag.unsqueeze(0) - urdf_t[:, 1, None, None]
        px = R_T_t[:, 0, 0, None, None] * dx + R_T_t[:, 0, 1, None, None] * dy
        py = R_T_t[:, 1, 0, None, None] * dx + R_T_t[:, 1, 1, None, None] * dy
        x_n = px * self._sdf_x_scale[:, None, None] + self._sdf_x_offset[:, None, None]
        y_n = py * self._sdf_y_scale[:, None, None] + self._sdf_y_offset[:, None, None]
        sdf_v_all = torch.nn.functional.grid_sample(
            self._sdf_batch, torch.stack([y_n, x_n], dim=-1),
            mode='bilinear', padding_mode='border', align_corners=True,
        ).squeeze(1)                                              # (B, nx, ny)

        # ── Batched body velocities ───────────────────────────────
        vel_u_all = lv_t[:, 0, None, None] - av_t[:, None, None] * (
            comp.Yu_stag.unsqueeze(0) - com_t[:, 1, None, None])  # (B, nx, ny)
        vel_v_all = lv_t[:, 1, None, None] + av_t[:, None, None] * (
            comp.Xv_stag.unsqueeze(0) - com_t[:, 0, None, None])  # (B, nx, ny)

        # ── Batched union-min ─────────────────────────────────────
        comp.sdf_vals[:]   = sdf_cc_all
        comp.sdf_vals_u[:] = sdf_u_all
        comp.sdf_vals_v[:] = sdf_v_all
        comp.sdf_val       = sdf_cc_all.min(dim=0).values

        min_u = sdf_u_all.min(dim=0)
        comp.sdf_val_u = min_u.values
        comp.body_u    = vel_u_all.gather(0, min_u.indices.unsqueeze(0)).squeeze(0)

        min_v = sdf_v_all.min(dim=0)
        comp.sdf_val_v = min_v.values
        comp.body_v    = vel_v_all.gather(0, min_v.indices.unsqueeze(0)).squeeze(0)

        # ── Per-body: contour update + contour mask ───────────────
        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]

            comp.com_pos[body_i] = com_t[body_i]

            # contour update
            body.cnt_update = R_t[body_i] @ body.cnt + urdf_t[body_i, :, None]

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

            body.r_com   = body.cnt_update - com_t[body_i, :, None]
            body.com_pos = com_t[body_i]

    @staticmethod
    def _body_sdf_grid(body):
        """Return ``(F, x, y)`` for any 2-D body type.

        * **Mesh / interpolated bodies** store the SDF on a regular
          grid inside a ``RegularGridInterpolator``-like object →
          return ``(sdf.F, sdf.x, sdf.y)`` directly.
        * **Analytical bodies** (circle, box, segment …) only expose
          a callable ``sdf(X, Y)`` with no pre-sampled grid → evaluate
          the SDF on a local grid covering the body contour plus a
          margin of ``4 h`` (enough for the BDIM transition layer +
          bilinear stencil) and return the sampled tensor.
        """
        if hasattr(body.sdf, 'F'):
            # Mesh / interpolated body – grid already available
            return body.sdf.F, body.sdf.x, body.sdf.y

        # ── Analytical body: pre-sample onto a regular grid ──────
        h = body.h
        margin = 4.0 * h

        # Derive local extent from the contour (computed during init
        # in _initialize_2d via skimage.measure.find_contours).
        x_lo = float(body.cnt[0].min()) - margin
        x_hi = float(body.cnt[0].max()) + margin
        y_lo = float(body.cnt[1].min()) - margin
        y_hi = float(body.cnt[1].max()) + margin

        nx = max(int((x_hi - x_lo) / h) + 1, 4)
        ny = max(int((y_hi - y_lo) / h) + 1, 4)

        x_coords = torch.linspace(x_lo, x_hi, nx,
                                  device=body.device, dtype=body.dtype)
        y_coords = torch.linspace(y_lo, y_hi, ny,
                                  device=body.device, dtype=body.dtype)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        F = body.sdf(X, Y)
        return F, x_coords, y_coords

    def _init_batched_sdf_2d(self):
        """Pre-build batched SDF tensor and normalization constants for grid_sample.

        Called lazily on the first ``_update_2d`` invocation.  The SDF
        fields are in body-local coordinates and never change, so this
        only runs once.

        Body SDF grids may have different resolutions (e.g. legs vs
        torso links).  We pad every SDF to the maximum (H, W) with a
        far-field sentinel so they can be stacked into a single tensor
        for ``grid_sample``.  The coordinate normalization maps each
        body's physical extent to the *unpadded* sub-region of the
        padded tensor.
        """
        comp = self.fluid_solver.composite_body
        B = comp.nbodies
        _FAR = 1e4  # sentinel: "far from any body surface"

        # --- extract (F, x, y) for every body (mesh or analytical) ---
        grids = [self._body_sdf_grid(body) for body in comp.bodies]

        # --- gather per-body shapes and find max dims ---
        shapes = [(F.shape[0], F.shape[1]) for F, _x, _y in grids]
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)

        # --- pad each SDF to (max_h, max_w) ---
        padded = []
        for (F, _x, _y), (h_i, w_i) in zip(grids, shapes):
            if h_i < max_h or w_i < max_w:
                # F.pad order: (W_left, W_right, H_top, H_bottom)
                F = torch.nn.functional.pad(
                    F, (0, max_w - w_i, 0, max_h - h_i), value=_FAR)
            padded.append(F)

        self._sdf_batch = torch.stack(padded).unsqueeze(1).contiguous()
        # shape: (B, 1, max_h, max_w)

        # --- axis ranges for coordinate normalization ---
        x_min = torch.tensor([g[1][0].item()  for g in grids],
                             device=self.device, dtype=self.dtype)
        x_max = torch.tensor([g[1][-1].item() for g in grids],
                             device=self.device, dtype=self.dtype)
        y_min = torch.tensor([g[2][0].item()  for g in grids],
                             device=self.device, dtype=self.dtype)
        y_max = torch.tensor([g[2][-1].item() for g in grids],
                             device=self.device, dtype=self.dtype)

        # Original per-body pixel counts (float for division)
        h_orig = torch.tensor([s[0] for s in shapes],
                              device=self.device, dtype=self.dtype)
        w_orig = torch.tensor([s[1] for s in shapes],
                              device=self.device, dtype=self.dtype)

        # --- affine map: physical coord → grid_sample [-1, 1] ---
        # With align_corners=True the grid maps [-1,+1] onto
        # [0, max_dim-1].  The *original* data lives in [0, orig-1],
        # so physical x_min→ grid=-1 and x_max→ grid=-1+2*(orig-1)/(max-1).
        #   grid_x = x * scale_x + offset_x
        # where scale_x = 2*(h_orig-1) / ((x_max-x_min) * (max_h-1))
        self._sdf_x_scale  = 2.0 * (h_orig - 1) / ((x_max - x_min) * (max_h - 1))
        self._sdf_x_offset = -1.0 - x_min * self._sdf_x_scale
        self._sdf_y_scale  = 2.0 * (w_orig - 1) / ((y_max - y_min) * (max_w - 1))
        self._sdf_y_offset = -1.0 - y_min * self._sdf_y_scale

        n_unique = len(set(shapes))
        print(f"  [batched-SDF] built ({B}, 1, {max_h}, {max_w}) "
              f"grid_sample tensor  "
              f"({self._sdf_batch.nelement()*self._sdf_batch.element_size()/1e6:.0f} MB)  "
              f"({n_unique} unique SDF sizes, padded to max)")

    # ---- 3-D update --------------------------------------------------
    # ------------------------------------------------------------------
    #  AABB narrow-band helpers (3-D)
    # ------------------------------------------------------------------
    @staticmethod
    def _body_aabb_indices(body, R, urdf_pos, comp_x, comp_y, comp_z,
                           h, gs, pad=3):
        """Compute fluid-grid index ranges for a body's SDF domain.

        Transforms the body-local SDF grid bounding box into world
        coordinates (accounting for rotation), then finds the axis-
        aligned sub-block of the fluid grid that covers this region
        plus ``pad`` cells of margin on each side.

        Returns (i0, i1, j0, j1, k0, k1) — Python-style half-open
        indices suitable for slicing ``comp.X[i0:i1, j0:j1, k0:k1]``.
        If the sub-block covers >90 % of the grid, returns ``None``
        to signal that full-grid evaluation is cheaper (avoids the
        fill + scatter overhead).
        """
        # Body SDF grid bounds in local coordinates
        sdf_lo = torch.stack([body.sdf.x[0],  body.sdf.y[0],  body.sdf.z[0]])
        sdf_hi = torch.stack([body.sdf.x[-1], body.sdf.y[-1], body.sdf.z[-1]])

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
            aabb = None
            if hasattr(body, 'sdf') and hasattr(body.sdf, 'z'):
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
        from lilytorch.src.kernels import streaming_sdf_min_3d_multi

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
        # ------------------------------------------------------------------
        sm = getattr(comp, '_stream_multi_static', None)
        if sm is None:
            F_chunks  = []
            bx_chunks = []; by_chunks = []; bz_chunks = []
            F_off  = [0]; bx_off = [0]; by_off = [0]; bz_off = [0]
            shapes = []
            meta   = []
            for body in comp.bodies:
                m = body._stream_meta
                F_chunks.append(m['F'].flatten())
                bx_chunks.append(m['bx']); by_chunks.append(m['by']); bz_chunks.append(m['bz'])
                F_off.append(F_off[-1]   + m['F'].numel())
                bx_off.append(bx_off[-1] + m['bx'].numel())
                by_off.append(by_off[-1] + m['by'].numel())
                bz_off.append(bz_off[-1] + m['bz'].numel())
                shapes.append([m['F'].shape[0], m['F'].shape[1], m['F'].shape[2]])
                meta.append([
                    m['bx0'], m['by0'], m['bz0'],
                    m['bx_last'], m['by_last'], m['bz_last'],
                    m['inv_dx'], m['inv_dy'], m['inv_dz'], m['inv_vol'],
                ])
            sm = {
                'F_flat':       torch.cat(F_chunks).contiguous(),
                'F_offsets':    torch.tensor(F_off,  dtype=torch.int64, device=self.device),
                'bx_flat':      torch.cat(bx_chunks).contiguous(),
                'bx_offsets':   torch.tensor(bx_off, dtype=torch.int64, device=self.device),
                'by_flat':      torch.cat(by_chunks).contiguous(),
                'by_offsets':   torch.tensor(by_off, dtype=torch.int64, device=self.device),
                'bz_flat':      torch.cat(bz_chunks).contiguous(),
                'bz_offsets':   torch.tensor(bz_off, dtype=torch.int64, device=self.device),
                'body_shapes':  torch.tensor(shapes, dtype=torch.int64, device=self.device),
                'body_meta':    torch.tensor(meta,   dtype=self.dtype,  device=self.device),
            }
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
        sparse_flat = torch.zeros(
            int(cell_off_h_np[-1]), device=self.device, dtype=self.dtype,
        )

        # Maintain `comp.com_pos[b]` and `body.com_pos` views for downstream code.
        for b, body in enumerate(comp.bodies):
            comp.com_pos[b] = com_pos_t[b]
            body.com_pos = comp.com_pos[b]

        # Python-list copy of cell offsets for the per-body slab split below.
        cell_off_h = cell_off_h_np.tolist()

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
        )

        # Split sparse_flat into per-body slabs and store
        for body_i, aabb in enumerate(aabbs_for_split):
            i0, i1, j0, j1, k0, k1 = aabb
            Ai, Aj, Ak = i1 - i0, j1 - j0, k1 - k0
            lo = cell_off_h[body_i]
            hi = cell_off_h[body_i + 1]
            slab = sparse_flat[lo:hi].view(Ai, Aj, Ak)
            comp._sdf_sparse[body_i] = (aabb, slab)

        # Phase D: stash per-step packed tensors so forces kernel can
        # reuse kin / aabb_lo / aabb_dim without rebuilding. The
        # `sparse_cc_flat` + `cell_offsets` pair is also stashed so
        # bdim_forces_3d_multi can read the cached per-body cc-SDF
        # directly instead of re-sampling it.
        comp._stream_multi_step = {
            'kin':             kin,
            'aabb_lo':         aabb_lo,
            'aabb_dim':        aabb_dim,
            'max_vol':         max_vol,
            'gx':              gx_1d,
            'gy':              gy_1d,
            'gz':              gz_1d,
            'sparse_cc_flat':  sparse_flat,
            'cell_offsets':    cell_off,
        }

    # ------------------------------------------------------------------
    def _update_3d_streaming(self, t, iteration, dt=1):
        from lilytorch.src.kernels import streaming_sdf_min_3d

        # Multi-body batched fast path (Phase C): one Python op call
        # handles all bodies, eliminating ~B torch.ops dispatches/step
        # and the per-body launch+sync overhead.  Falls back to the
        # per-body sequential loop below if any body lacks meta.
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

        aabb = None
        if hasattr(body, 'sdf') and hasattr(body.sdf, 'z'):
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
        try:
            from pytorch_interpolation import RegularGridInterpolator3D
        except Exception as e:
            raise RuntimeError(
                "custom_trilinear_3d requires pytorch_interpolation with "
                "the C++/CUDA build (RegularGridInterpolator3D).  "
                f"Import failed: {e}"
            )

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
    #  fluid_step: one BDIM time-step (advection-diffusion + projection)
    # ==================================================================
    def _compute_variable_density_coefficients(self, timestep):
        """Compute variable-density Poisson coefficients.

        Returns ``(ch, cv, ch_cc)`` for 2-D or ``(ch, cv, cw, ch_cc)``
                for 3-D, where:

                * ``ch, cv, cw`` -- staggered ``dt / rho_eff`` on the respective
                    face grids. Used by both the multigrid correction step and the
                    FFT correction step.
                * ``ch_cc`` -- cell-centred ``dt / rho_eff_cc``. Used as the
                    Poisson RHS coefficient in the FFT path so that the solve
                    accounts for spatially-varying density:
                    ``∇²p = div / ch_cc  ≡  div * rho_eff_cc / dt``.

                Effective density at location x:
                        rho_eff(x) = rho_body + (rho_fluid - rho_body) * mu0(x)

                The density blend is already built from the BDIM face field ``mu0``,
                so the coupled variable-density path does not apply an additional
                ``mu0`` factor in the numerator.

        Narrow-band fast-path (3-D only)
        --------------------------------
        When the solver runs with ``_mu_normals_union = True`` the mu0
        face buffers contain the outside-body default value ``mu0 = 1``
        everywhere except inside the union AABB of all bodies.  Outside
        that AABB ``ch = cv = cw = ch_cc = dt / rho_fluid`` is a runtime
        constant — so we allocate persistent full-grid ``ch/cv/cw/ch_cc``
        buffers once, pre-fill them with ``dt / rho_fluid``, and each
        step only overwrite the union sub-block.  This avoids the four
        full-grid divisions that otherwise dominate the "Other
        (residual)" cost at large grids.
        """
        fs = self.fluid_solver
        _drho = self.rho_fluid - self.rho_body

        # ──────────────────────────────────────────────────────────────
        # Narrow-band path (3-D + mu_normals_union active)
        # ──────────────────────────────────────────────────────────────
        if (self.ndim == 3
            and getattr(fs, '_mu_normals_union', False)
            and fs.mu0_all_u is not None
            and fs.mu0_all_v is not None
            and fs.mu0_all_w is not None
            and fs.mu0_all   is not None):
            u_aabb = fs._compute_union_aabb_3d(halo=2, bucket=16)
            # u_aabb may be None when ``_compute_union_aabb_3d``
            # decides cropping is not worth it (large fraction of grid)
            # or when sparse SDFs aren't available.  Fall through to
            # the full-grid path in that case.
            if u_aabb is None:
                pass
            else:

                # Lazy-allocate persistent buffers on first call (or after
                # grid / dtype / device changes).  The outside-body default
                # dt / rho_fluid is stamped in once and never overwritten by
                # the narrow-band update.
                _dt_over_rhofluid = float(timestep / self.rho_fluid)
                mu0_u = fs.mu0_all_u
                needs_realloc = (
                    getattr(self, '_ch_persist', None) is None
                    or self._ch_persist.shape != mu0_u.shape
                    or self._ch_persist.dtype != mu0_u.dtype
                    or self._ch_persist.device != mu0_u.device
                    or self._ch_outside_val != _dt_over_rhofluid
                )
                if needs_realloc:
                    self._ch_persist    = torch.full_like(fs.mu0_all_u, _dt_over_rhofluid)
                    self._cv_persist    = torch.full_like(fs.mu0_all_v, _dt_over_rhofluid)
                    self._cw_persist    = torch.full_like(fs.mu0_all_w, _dt_over_rhofluid)
                    self._ch_cc_persist = torch.full_like(fs.mu0_all,   _dt_over_rhofluid)
                    self._ch_outside_val = _dt_over_rhofluid

                ui0, ui1, uj0, uj1, uk0, uk1 = u_aabb
                usl = (slice(ui0, ui1), slice(uj0, uj1), slice(uk0, uk1))

                # Only inside the union AABB can mu0 differ from 1 → so only
                # there can ch/cv/cw/ch_cc differ from dt/rho_fluid.  Four
                # small tensor expressions, each on an (Nx_sub, Ny_sub, Nz_sub)
                # sub-block, instead of four full-grid divisions.
                self._ch_persist[usl]    = timestep / (self.rho_body + _drho * fs.mu0_all_u[usl])
                self._cv_persist[usl]    = timestep / (self.rho_body + _drho * fs.mu0_all_v[usl])
                self._cw_persist[usl]    = timestep / (self.rho_body + _drho * fs.mu0_all_w[usl])
                self._ch_cc_persist[usl] = timestep / (self.rho_body + _drho * fs.mu0_all[usl])
                return (self._ch_persist, self._cv_persist,
                        self._cw_persist, self._ch_cc_persist)

        # ──────────────────────────────────────────────────────────────
        # Full-grid reference path (2-D, or 3-D without narrow-band)
        # ──────────────────────────────────────────────────────────────
        ch = timestep / (self.rho_body + _drho * fs.mu0_all_u)
        cv = timestep / (self.rho_body + _drho * fs.mu0_all_v)

        # Cell-centred effective density → CC coefficient for FFT RHS.
        rho_cc = self.rho_body + _drho * fs.mu0_all
        ch_cc = timestep / rho_cc

        if self.ndim == 3:
            cw = timestep / (self.rho_body + _drho * fs.mu0_all_w)
            return ch, cv, cw, ch_cc
        return ch, cv, ch_cc

    def fluid_step(self, *args):
        if self.ndim == 3:
            return self._fluid_step_3d(*args)
        return self._fluid_step_2d(*args)

    # ---- 2-D fluid step ---------------------------------------------
    def _fluid_step_2d(self, u, v, p, timestep):
        fs = self.fluid_solver
        _bdim = fs._bdim_meta_compiled
        _h = fs.h
        comp = fs.composite_body

        # Pre-compute variable-density Poisson coefficients once.
        #   ch/cv   -- staggered dt/rho_eff, used for the correction step
        #   ch_cc   -- cell-centred dt/rho_eff_cc, used for the FFT RHS
        _ch, _cv, _ch_cc = self._compute_variable_density_coefficients(timestep)

        def _advect_bdim(u_in, v_in, nu_t=None, u_rebase=None, v_rebase=None):
            """One advection-diffusion + BDIM pass.

            Returns the BDIM-corrected (uprime, vprime) with BCs applied
            *after* BDIM so that wall BCs override any BDIM corrections
            near the domain boundary.
            """
            (up, vp) = fs.adv_diff_solver.solve(u_in, v_in, nu_t=nu_t)
            # Clone CUDA-graph outputs before in-place ops / set_BCs
            up = up.clone()
            vp = vp.clone()
            # Rebase predictor/corrector increment from (u_in, v_in)
            # to a reference state (u_rebase, v_rebase). This is used
            # in Heun corrector to build u^n + dt*RHS(u_pred).
            if u_rebase is not None:
                up = u_rebase + (up - u_in)
            if v_rebase is not None:
                vp = v_rebase + (vp - v_in)
            up = _bdim(
                up, fs.mu0_all_u,
                comp.body_u, fs.mu1_all_u,
                fs.normal_x_u, fs.normal_y_u, _h, 2,
            ).clone()
            vp = _bdim(
                vp, fs.mu0_all_v,
                comp.body_v, fs.mu1_all_v,
                fs.normal_x_v, fs.normal_y_v, _h, 2,
            ).clone()
            # BCs enforced *after* BDIM so that wall conditions override
            # any BDIM blending that reaches ghost cells near a wall.
            fs.adv_diff_solver.set_BCs(up, vp)
            return up, vp

        nu_t = fs._compute_nu_t(u, v)

        if self._use_heun:
            # ===== Heun (RK2) predictor-corrector =====
            # Matches WaterLily.jl's mom_step! exactly:
            #   1. Predictor: adv-diff → BDIM → project(w=1) → BCs → u_pred
            #   2. Corrector: adv-diff on u_pred → rebase from u^n → BDIM
            #      → average PROJECTED predictor with corrector BDIM
            #      → project with halved coefficients (w=0.5 equivalent)
            #
            # The corrector average is 0.5*(u_pred + BDIM(u^n + dt*RHS(u_pred))).
            # Because u_pred is div-free, div(u_avg) = 0.5*div(BDIM_corr).
            # Halving the Poisson coefficients compensates this, giving the
            # correct physical pressure for force computation.

            # Predictor
            uprime, vprime = _advect_bdim(u, v, nu_t=nu_t)
            u1, v1, p1 = fs.project(uprime, vprime, p, ch=_ch, cv=_cv, ch_cc=_ch_cc)
            # Re-apply BCs after projection (matches WaterLily's BC!
            # after every project! call).
            fs.adv_diff_solver.set_BCs(u1, v1)

            # Corrector
            nu_t = fs._compute_nu_t(u1, v1)
            uprime2, vprime2 = _advect_bdim(
                u1, v1, nu_t=nu_t,
                u_rebase=u, v_rebase=v,
            )

            # Average projected predictor with BDIM-corrected corrector,
            # then project once with half coefficients (w=0.5 equivalent).
            u_avg = 0.5 * (u1 + uprime2)
            v_avg = 0.5 * (v1 + vprime2)
            fs.adv_diff_solver.set_BCs(u_avg, v_avg)

            _ch_half = 0.5 * _ch
            _cv_half = 0.5 * _cv
            _ch_cc_half = 0.5 * _ch_cc
            u_out, v_out, p_out = fs.project(
                u_avg, v_avg, p1,
                ch=_ch_half, cv=_cv_half, ch_cc=_ch_cc_half,
            )


        else:
            # ===== Single Euler step =====
            uprime, vprime = _advect_bdim(u, v, nu_t=nu_t)
            u_out, v_out, p_out = fs.project(uprime, vprime, p,
                                              ch=_ch, cv=_cv, ch_cc=_ch_cc)

        # Sponge damping (2-D)
        if fs.use_sponge:
            (u_out, v_out) = fs.apply_sponge_damping(u_out, v_out)

        # Yield-stress damping (2-D)
        if fs.use_yield_damping:
            (u_out, v_out) = fs.apply_yield_damping(u_out, v_out)

        fs.adv_diff_solver.set_BCs(u_out, v_out)
        return (u_out, v_out, p_out)

    # ---- 3-D fluid step ---------------------------------------------
    def _fluid_step_3d(self, u, v, w, p, timestep):
        fs = self.fluid_solver
        _bdim = fs._bdim_meta_compiled
        _h = fs.h

        nu_t = fs._compute_nu_t(u, v, w)
        (uprime, vprime, wprime) = fs.adv_diff_solver.solve(u, v, w, nu_t=nu_t)
        # Clone CUDA-graph outputs so they can safely be passed to
        # another compiled kernel (_bdim) and modified by set_BCs.
        uprime = uprime.clone()
        vprime = vprime.clone()
        wprime = wprime.clone()
        fs.adv_diff_solver.set_BCs(uprime, vprime, wprime)

        # Cache union AABB on the solver for the BDIM pass.
        fs._bdim_union_aabb = (
            fs._compute_union_aabb_3d(halo=2)
            if getattr(fs, '_bdim_union', False) else None
        )

        # BDIM2 meta-equation  (reuses FluidSolver's compiled kernel,
        # optionally cropped to the union AABB)
        uprime = fs._bdim_apply_3d(
            uprime, fs.mu0_all_u,
            fs.composite_body.body_u, fs.mu1_all_u,
            fs.normal_x_u, fs.normal_y_u, fs.normal_z_u,
        )
        vprime = fs._bdim_apply_3d(
            vprime, fs.mu0_all_v,
            fs.composite_body.body_v, fs.mu1_all_v,
            fs.normal_x_v, fs.normal_y_v, fs.normal_z_v,
        )
        wprime = fs._bdim_apply_3d(
            wprime, fs.mu0_all_w,
            fs.composite_body.body_w, fs.mu1_all_w,
            fs.normal_x_w, fs.normal_y_w, fs.normal_z_w,
        )
        fs._bdim_union_aabb = None

        # ── Free mu1 + staggered normals right after BDIM ────────────
        # project() only uses mu0_{u,v,w}.  forces_method2_3d recomputes
        # CC normals on-the-fly.  Freeing these 12 grid-sized arrays
        # (~12 × 131 MB ≈ 1.6 GB) reduces peak memory during the
        # Poisson solve that follows.
        # Use a single ``__dict__.update`` to avoid 14 ``hasattr`` +
        # 14 ``setattr`` Python lookups per step (≥ 50 µs at low N).
        fs.__dict__.update(_FS_FREE_AFTER_BDIM_3D)

        # Variable-density Poisson coefficients.
        # Computed before freeing mu0 arrays (both FFT and multigrid need
        # ch_cc from mu0_all CC, which is freed below).
        ch, cv, cw, ch_cc = self._compute_variable_density_coefficients(timestep)

        # ── Free mu0 now — ch/cv/cw/ch_cc are independent tensors ────
        fs.__dict__.update(_FS_FREE_AFTER_VAR_DENS_3D)

        poisson_method = getattr(fs, "poisson_method", "multigrid")
        if poisson_method == "fft":
            (u, v, w, p) = fs.project(uprime, vprime, p,
                                      w_vel=wprime, ch=ch, cv=cv, cw=cw,
                                      ch_cc=ch_cc)
        else:
            (u, v, w, p) = fs.project(uprime, vprime, p,
                                      w_vel=wprime, ch=ch, cv=cv, cw=cw)

        # Sponge damping: damp velocity near domain boundaries
        if fs.use_sponge:
            (u, v, w) = fs.apply_sponge_damping(u, v, w)

        # Yield-stress damping (3-D)
        if fs.use_yield_damping:
            (u, v, w) = fs.apply_yield_damping(u, v, w)

        fs.adv_diff_solver.set_BCs(u, v, w)
        return (u, v, w, p)

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

    def _check_explosion(self, iteration):
        """Abort immediately if the fluid fields went non-finite or
        velocities grew past ``vmax_abort``.

        Without this, NaN propagates silently (matplotlib renders NaN
        as transparent) and FARMS happily keeps stepping MuJoCo with
        zero fluid forces.
        """
        fs = self.fluid_solver
        # Batch every reduction the explosion check needs into a *single*
        # GPU→CPU sync.  The previous implementation issued one sync per
        # field for ``isfinite().all()`` (3–4 calls) plus one per velocity
        # component for ``.abs().max()`` (2–3 calls) — i.e. 5–7 separate
        # device-to-host transfers every step.  Each one stalls the
        # pipeline and contributes to the "Other (residual)" plateau at
        # low grid counts.
        if self.ndim == 3:
            field_names  = ("u", "v", "w", "p")
            field_arrs   = (fs.u0, fs.v0, fs.w0, fs.p0)
            n_vel        = 3
        else:
            field_names  = ("u", "v", "p")
            field_arrs   = (fs.u0, fs.v0, fs.p0)
            n_vel        = 2

        finite_flags  = torch.stack([torch.isfinite(a).all() for a in field_arrs])
        vmax_per_vel  = torch.stack([a.abs().amax() for a in field_arrs[:n_vel]])
        # Single fused DtoH transfer + sync.
        diag = torch.cat((finite_flags.to(vmax_per_vel.dtype),
                          vmax_per_vel)).cpu().numpy()
        n_fields = len(field_arrs)
        finite_np = diag[:n_fields]
        vmax_np   = diag[n_fields:]

        for name, ok in zip(field_names, finite_np):
            if not bool(ok):
                self.terminate = True
                raise RuntimeError(
                    f"[BDIM] Fluid explosion at iteration {iteration}: "
                    f"non-finite values in field '{name}'. Likely cause: "
                    f"body intersecting a domain wall (Poisson ill-conditioned) "
                    f"or CFL violation."
                )

        vmax = float(vmax_np.max())
        if vmax > self._vmax_abort:
            self.terminate = True
            raise RuntimeError(
                f"[BDIM] Fluid explosion at iteration {iteration}: "
                f"|u|_max = {vmax:.3e} > vmax_abort = {self._vmax_abort:.3e}."
            )

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

        if not self.terminate:

            # 1. update SDF + body velocities from FARMS kinematics
            self.update(t, iteration, dt=timestep)

            # 2. recompute mu / normal fields
            if self.ndim == 3:
                fs._recompute_mu_normals_3d()
            else:
                fs._recompute_mu_normals_2d()

            # 3. BDIM fluid step
            if self.ndim == 3:
                (u, v, w, p) = self.fluid_step(
                    fs.u0, fs.v0, fs.w0, fs.p0, timestep
                )
                if self.zero_pressure_inside:
                    p = torch.where(fs.composite_body.sdf_val < 0, 0, p)
                (fs.u0, fs.v0, fs.w0, fs.p0) = (u, v, w, p)
            else:
                (u, v, p) = self.fluid_step(
                    fs.u0, fs.v0, fs.p0, timestep
                )
                if self.zero_pressure_inside:
                    p = torch.where(fs.composite_body.sdf_val < 0, 0, p)
                (fs.u0, fs.v0, fs.p0) = (u, v, p)

            # 3b. bail out immediately if the fluid blew up
            self._check_explosion(iteration)

            # 4. compute fluid forces on each body
            if self.ndim == 3:
                fs.forces_method2_3d(fs.u0, fs.v0, fs.w0, fs.p0, iteration)
            elif self.force_method == "method1":
                fs.forces_method1(fs.u0, fs.v0, fs.p0, iteration)
            else:
                fs.forces_method2(fs.u0, fs.v0, fs.p0, iteration)

            # ── Free cached force-density tensors (6 × grid_shape) ───
            # These are only needed for the per-body integration above;
            # plotting_and_saving does not use them.  Use a single
            # ``__dict__.update`` to avoid the 6 × (hasattr + setattr)
            # Python lookups per step.
            fs.__dict__.update(_FS_FREE_AFTER_FORCES_3D)

            # 5. plotting / saving
            if self.ndim == 3:
                self.terminate = fs.plotting_and_saving(
                    fs.u0, fs.v0, fs.p0, iteration, w_vel=fs.w0, check_termination=False
                )
            else:
                self.terminate = fs.plotting_and_saving(
                    fs.u0, fs.v0, fs.p0, iteration, check_termination=False
                )

            # 6. apply forces to MuJoCo bodies
            self.apply_forces(task, physics)

            # 7. free BDIM intermediates to reclaim GPU memory
            fs._release_bdim_fields()

        self.iteration += 1
