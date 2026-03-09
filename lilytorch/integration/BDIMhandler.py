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
                                _rotate_grid_3d_compiled,
                                _stagger_sdf_3d, _stagger_sdf_3d_compiled,
                                _mu_normals_batched_3d,
                                _mu_normals_batched_3d_compiled)
import torch


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

        # ---- axes (derived from ndim) ----
        if self.ndim == 3:
            self.lin_axes = [0, 1, 2]
            self.ang_axes = [0, 1, 2]
        else:
            self.lin_axes = [0, 1]
            self.ang_axes = [2]

        # ---- force scaling (config or auto) ----
        fs_cfg = self.pars.get("body", {}).get("force_scaling", "auto")
        if fs_cfg == "auto":
            if self.ndim == 2:
                comp = self.fluid_solver.composite_body
                try:
                    self.force_scaling = np.array(
                        [np.diff(body.bb[2])[0] for body in comp.bodies]
                    )
                except Exception:
                    self.force_scaling = 1.0
            else:
                self.force_scaling = 1.0
        else:
            self.force_scaling = float(fs_cfg)

        # ---- densities ----
        self.rho_fluid = self.pars["solver"]["rho"]
        self.rho_body  = self.pars["solver"].get("rho_body", 800.0)

        # ---- toggles ----
        self.zero_pressure_inside = self.pars["solver"].get(
            "zero_pressure_inside", False
        )
        self.contour_mask = self.pars.get("body", {}).get("contour_mask", False)
        self.force_method = self.pars["solver"].get("force_method", "method2")

        # ---- FARMS-style buoyancy parameters ----
        # With all-Neumann BCs the BDIM pressure field is purely dynamic;
        # no hydrostatic gradient builds up, so buoyancy must be added
        # explicitly.  We replicate FARMS' compute_buoyancy() formula from
        # drag.pyx which uses MuJoCo body mass, bounding-sphere half-height,
        # and a linear submersion fraction.
        self.gravity_z = float(physics.model.opt.gravity[2])  # e.g. -9.81
        self.water_surface = float(self.pars["solver"]["zmax"])
        self._buoyancy_initialized = False

        # ---- optional physics solref tweak ----
        solref = self.pars.get("physics", {}).get("solref", None)
        if solref is not None:
            physics.model.geom_solref[:, 0] = solref[0]
            physics.model.geom_solref[:, 1] = solref[1]

        # ---- 3-D: allocate per-body cell-centre SDF for force computation ----
        if self.ndim == 3:
            comp = self.fluid_solver.composite_body
            comp.sdf_vals = torch.zeros(
                (comp.nbodies, *self.fluid_solver.grid_shape),
                device=self.device,
                dtype=self.dtype,
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
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
            density = link density from config    [kg/m³]
            height  = 0.5 * geom bounding radius  [m]
            surface = water surface height (zmax)  [m]
            gravity = -9.81                        [m/s²]
        """
        comp = self.fluid_solver.composite_body
        n = comp.nbodies

        self._buoy_mass   = np.zeros(n)
        self._buoy_height = np.zeros(n)

        for body_i in range(n):
            (animat_id, link_id) = comp.body_ids[body_i]
            ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            self._buoy_mass[body_i] = float(physics.model.body_mass[ind])

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
                    [sen.com_ang_velocity(iteration, lk)[2]
                     for lk in range(len(sen.names))]
                )
            )

        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]

            com_pos  = com_poses[animat_id][link_id]
            urdf_pos = urdf_poses[animat_id][link_id]
            R        = Rs[animat_id][link_id]
            lin_vel  = lin_vels[animat_id][link_id]
            ang_vel  = ang_vels[animat_id][link_id]

            R_T = R.T  # (2, 2)

            # SDF at cell centres (meshgrid broadcasting, no flatten)
            px, py = rotate_grid_2d(comp.X, comp.Y, R_T, urdf_pos)
            comp.sdf_vals[body_i] = body.sdf(px, py)

            # SDF at u-staggered
            px, py = rotate_grid_2d(comp.Xu_stag, comp.Yu_stag, R_T, urdf_pos)
            comp.sdf_vals_u[body_i] = body.sdf(px, py)

            # SDF at v-staggered
            px, py = rotate_grid_2d(comp.Xv_stag, comp.Yv_stag, R_T, urdf_pos)
            comp.sdf_vals_v[body_i] = body.sdf(px, py)

            # body velocities: v = v_lin + omega_z x (x - x_com)
            comp.u_vals[body_i] = lin_vel[0] - ang_vel * (comp.Yu_stag - com_pos[1])
            comp.v_vals[body_i] = lin_vel[1] + ang_vel * (comp.Xv_stag - com_pos[0])

            comp.com_pos[body_i] = com_pos

            # contour update
            body.cnt_update = R @ body.cnt + urdf_pos[:, None]

            # optional contour mask for overlapping links
            if self.contour_mask:
                x_cnt = body.cnt_update[0]
                y_cnt = body.cnt_update[1]
                if link_id == 0:
                    body_p = comp.bodies[body_i + 1]
                    pt = Rs[animat_id][link_id + 1].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses[animat_id][link_id + 1][:, None]
                    )
                    mask = (body_p.sdf(pt[0], pt[1]) - body.h) >= 0
                elif link_id == urdf_poses[animat_id].shape[0] - 1:
                    body_m = comp.bodies[body_i - 1]
                    pt = Rs[animat_id][link_id - 1].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses[animat_id][link_id - 1][:, None]
                    )
                    mask = (body_m.sdf(pt[0], pt[1]) - body.h) >= 0
                else:
                    body_m = comp.bodies[body_i - 1]
                    pt_m = Rs[animat_id][link_id - 1].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses[animat_id][link_id - 1][:, None]
                    )
                    body_p = comp.bodies[body_i + 1]
                    pt_p = Rs[animat_id][link_id + 1].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses[animat_id][link_id + 1][:, None]
                    )
                    sdf_m = body_m.sdf(pt_m[0], pt_m[1]) - body.h
                    sdf_p = body_p.sdf(pt_p[0], pt_p[1]) - body.h
                    mask = (sdf_m >= 0) & (sdf_p >= 0)
                body.mask = mask

            body.r_com   = body.cnt_update - com_pos[:, None]
            body.com_pos = com_pos

        # union reduction (gather / argmin)
        idx = comp.sdf_vals.argmin(0).unsqueeze(0).expand(comp.sdf_vals.shape)
        comp.sdf_val = (
            comp.sdf_vals.gather(0, idx)[0]
            .reshape(fs.nx, fs.ny)
            .contiguous()
        )

        idx_u = comp.sdf_vals_u.argmin(0).unsqueeze(0).expand(comp.sdf_vals_u.shape)
        comp.sdf_val_u = (
            comp.sdf_vals_u.gather(0, idx_u)[0]
            .reshape(fs.nx, fs.ny)
            .contiguous()
        )
        comp.body_u = (
            comp.u_vals.gather(0, idx_u)[0]
            .reshape(fs.nx, fs.ny)
            .contiguous()
        )

        idx_v = comp.sdf_vals_v.argmin(0).unsqueeze(0).expand(comp.sdf_vals_v.shape)
        comp.sdf_val_v = (
            comp.sdf_vals_v.gather(0, idx_v)[0]
            .reshape(fs.nx, fs.ny)
            .contiguous()
        )
        comp.body_v = (
            comp.v_vals.gather(0, idx_v)[0]
            .reshape(fs.nx, fs.ny)
            .contiguous()
        )

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

    # ---- 3-D update --------------------------------------------------
    def _update_3d(self, t, iteration, dt=1):
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

        # Per-body SDF storage for forces (fill once per step)
        comp.sdf_vals.fill_(_FAR)

        # Initialise union fields to _FAR / zero (once per step).
        # First body's torch.where / torch.minimum will overwrite
        # with its actual SDF since actual < _FAR.
        comp.sdf_val   = torch.full(gs, _FAR, device=self.device, dtype=self.dtype)
        comp.sdf_val_u = torch.full(gs, _FAR, device=self.device, dtype=self.dtype)
        comp.sdf_val_v = torch.full(gs, _FAR, device=self.device, dtype=self.dtype)
        comp.sdf_val_w = torch.full(gs, _FAR, device=self.device, dtype=self.dtype)
        comp.body_u    = torch.zeros(gs, device=self.device, dtype=self.dtype)
        comp.body_v    = torch.zeros(gs, device=self.device, dtype=self.dtype)
        comp.body_w    = torch.zeros(gs, device=self.device, dtype=self.dtype)

        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]

            com_pos  = com_poses[animat_id][link_id]
            urdf_pos = urdf_poses[animat_id][link_id]
            R        = Rs[animat_id][link_id]
            lin_vel  = lin_vels[animat_id][link_id]
            ang_vel  = ang_vels[animat_id][link_id]

            R_T = R.T  # (3, 3) — transposed rotation

            # ── AABB-clipped SDF evaluation ─────────────────────────
            aabb = None
            if hasattr(body, 'sdf') and hasattr(body.sdf, 'z'):
                aabb = self._body_aabb_indices(
                    body, R, urdf_pos,
                    comp.x, comp.y, comp.z,
                    h_grid, gs, pad=3,
                )

            if aabb is not None:
                # ── Sub-block path (main saving) ────────────────────
                # Only rotate + interpolate the AABB sub-block of the
                # grid, then stagger / velocity / union-min ALL on the
                # sub-block with contiguous intermediates.  Strided
                # reads of union fields are small (~3-5 MB per body)
                # and fit in L2 cache.
                (i0, i1, j0, j1, k0, k1) = aabb
                sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))

                px, py, pz = rotate_grid_3d(
                    comp.X[sl], comp.Y[sl], comp.Z_grid[sl],
                    R_T, urdf_pos,
                )
                sdf_sub = body.sdf(px, py, pz)       # contiguous

                # Per-body SDF (for forces; rest stays _FAR)
                comp.sdf_vals[body_i, i0:i1, j0:j1, k0:k1] = sdf_sub

                # Sub-block stagger (contiguous in → contiguous out)
                sdf_sub_u, sdf_sub_v, sdf_sub_w = _stagger_sdf_3d(
                    sdf_sub)

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
                                     R_T, urdf_pos)
                sdf_cc = body.sdf(px, py, pz)
                comp.sdf_vals[body_i] = sdf_cc

                sdf_u, sdf_v, sdf_w = _stagger_sdf_3d(sdf_cc)

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

    # ==================================================================
    #  fluid_step: one BDIM time-step (advection-diffusion + projection)
    # ==================================================================
    def fluid_step(self, *args):
        if self.ndim == 3:
            return self._fluid_step_3d(*args)
        return self._fluid_step_2d(*args)

    # ---- 2-D fluid step ---------------------------------------------
    def _fluid_step_2d(self, u, v, p, timestep):
        fs = self.fluid_solver

        (uprime, vprime) = fs.adv_diff_solver.solve(u, v)
        fs.adv_diff_solver.set_BCs(uprime, vprime)

        # BDIM2 meta-equation  (uses mu0*(phi-body)+body to avoid m_m0)
        diff_u = uprime - fs.composite_body.body_u
        uprime = (
            fs.mu0_all_u * diff_u + fs.composite_body.body_u
            + fs.mu1_all_u * fs.normal_derivative(
                diff_u,
                fs.normal_x_u, fs.normal_y_u,
            )
        )
        diff_v = vprime - fs.composite_body.body_v
        vprime = (
            fs.mu0_all_v * diff_v + fs.composite_body.body_v
            + fs.mu1_all_v * fs.normal_derivative(
                diff_v,
                fs.normal_x_v, fs.normal_y_v,
            )
        )

        fs.div = fs.divergence(uprime, vprime)

        # ---- Poisson solve ----
        poisson_method = getattr(fs, "poisson_method", "multigrid")
        if poisson_method == "fft":
            coeff = timestep / self.rho_fluid
            p = fs.poisson_solverFFT.solve(fs.div / coeff)
            (p_x, p_y) = fs.gradient(p)
            u = uprime - coeff * p_x
            v = vprime - coeff * p_y
        else:
            # variable-density:  rho = rho_body + (rho_fluid - rho_body)*mu0
            _drho = self.rho_fluid - self.rho_body
            rho_u = self.rho_body + _drho * fs.mu0_all_u
            rho_v = self.rho_body + _drho * fs.mu0_all_v
            ch_full = timestep / rho_u
            cv_full = timestep / rho_v

            _poisson_solve = (fs.poisson_solver.solve_mgcg
                              if poisson_method == "mgcg"
                              else fs.poisson_solver.solve_multigrid)
            p0 = p if getattr(fs, 'poisson_warm_start', False) else torch.zeros_like(p)

            p, _ = _poisson_solve(
                fs.div[1:-1, 1:-1],
                p0,
                (timestep / self.rho_body) * torch.ones_like(fs.div),
                ch=ch_full[1:,  1:-1],
                cv=cv_full[1:-1, 1:],
            )
            (p_x, p_y) = fs.gradient(p)
            u = uprime - ch_full * p_x
            v = vprime - cv_full * p_y

        fs.adv_diff_solver.set_BCs(u, v)
        return (u, v, p)

    # ---- 3-D fluid step ---------------------------------------------
    def _fluid_step_3d(self, u, v, w, p, timestep):
        fs = self.fluid_solver

        (uprime, vprime, wprime) = fs.adv_diff_solver.solve(u, v, w)
        fs.adv_diff_solver.set_BCs(uprime, vprime, wprime)

        # BDIM2 meta-equation  (uses mu0*(phi-body)+body to avoid m_m0)
        diff_u = uprime - fs.composite_body.body_u
        uprime = (
            fs.mu0_all_u * diff_u + fs.composite_body.body_u
            + fs.mu1_all_u * fs.normal_derivative(
                diff_u,
                fs.normal_x_u, fs.normal_y_u, fs.normal_z_u,
            )
        )
        diff_v = vprime - fs.composite_body.body_v
        vprime = (
            fs.mu0_all_v * diff_v + fs.composite_body.body_v
            + fs.mu1_all_v * fs.normal_derivative(
                diff_v,
                fs.normal_x_v, fs.normal_y_v, fs.normal_z_v,
            )
        )
        diff_w = wprime - fs.composite_body.body_w
        wprime = (
            fs.mu0_all_w * diff_w + fs.composite_body.body_w
            + fs.mu1_all_w * fs.normal_derivative(
                diff_w,
                fs.normal_x_w, fs.normal_y_w, fs.normal_z_w,
            )
        )

        fs.div = fs.divergence(uprime, vprime, wprime)

        poisson_method = getattr(fs, "poisson_method", "multigrid")
        if poisson_method == "fft":
            coeff = timestep / self.rho_fluid
            p = fs.poisson_solverFFT.solve(fs.div / coeff)
            (p_x, p_y, p_z) = fs.gradient(p)
            u = uprime - coeff * p_x
            v = vprime - coeff * p_y
            w = wprime - coeff * p_z
        else:
            # variable-density:  rho = rho_body + (rho_fluid - rho_body)*mu0
            _drho = self.rho_fluid - self.rho_body
            rho_u = self.rho_body + _drho * fs.mu0_all_u
            rho_v = self.rho_body + _drho * fs.mu0_all_v
            rho_w = self.rho_body + _drho * fs.mu0_all_w
            ch = timestep / rho_u
            cv = timestep / rho_v
            cw = timestep / rho_w

            _poisson_solve = (fs.poisson_solver.solve_mgcg
                              if poisson_method == "mgcg"
                              else fs.poisson_solver.solve_multigrid)
            p0 = p if getattr(fs, 'poisson_warm_start', False) else torch.zeros_like(p)

            p, _ = _poisson_solve(
                fs.div[1:-1, 1:-1, 1:-1],
                p0,
                (timestep / self.rho_body) * torch.ones_like(fs.div),
                ch=ch[1:,  1:-1, 1:-1],
                cv=cv[1:-1, 1:,  1:-1],
                cw=cw[1:-1, 1:-1, 1:],
            )
            (p_x, p_y, p_z) = fs.gradient(p)
            u = uprime - ch * p_x
            v = vprime - cv * p_y
            w = wprime - cw * p_z

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

        self.friction_force_lin_x = s * fs.friction_force_lin_x.cpu().numpy()
        self.friction_force_lin_y = s * fs.friction_force_lin_y.cpu().numpy()
        self.friction_force_ang_z = s * fs.friction_force_ang_z.cpu().numpy()
        self.pressure_force_x     = s * fs.pressure_force_x.cpu().numpy()
        self.pressure_force_y     = s * fs.pressure_force_y.cpu().numpy()
        self.pressure_force_ang_z = s * fs.pressure_force_ang_z.cpu().numpy()

        for body_i in range(len(fs.composite_body.bodies)):
            (animat_id, link_id) = fs.composite_body.body_ids[body_i]
            ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            physics.data.xfrc_applied[ind, 0] = (
                self.friction_force_lin_x[body_i] + self.pressure_force_x[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 1] = (
                self.friction_force_lin_y[body_i] + self.pressure_force_y[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 5] = (
                self.friction_force_ang_z[body_i] + self.pressure_force_ang_z[body_i]
            ) * task.units.newtons

    def _apply_forces_3d(self, task, physics):
        fs = self.fluid_solver
        s  = self.force_scaling

        self.friction_force_lin_x = s * fs.friction_force_lin_x.cpu().numpy()
        self.friction_force_lin_y = s * fs.friction_force_lin_y.cpu().numpy()
        self.friction_force_lin_z = s * fs.friction_force_lin_z.cpu().numpy()
        self.friction_force_ang_x = s * fs.friction_force_ang_x.cpu().numpy()
        self.friction_force_ang_y = s * fs.friction_force_ang_y.cpu().numpy()
        self.friction_force_ang_z = s * fs.friction_force_ang_z.cpu().numpy()

        self.pressure_force_x     = s * fs.pressure_force_x.cpu().numpy()
        self.pressure_force_y     = s * fs.pressure_force_y.cpu().numpy()
        self.pressure_force_z     = s * fs.pressure_force_z.cpu().numpy()
        self.pressure_force_ang_x = s * fs.pressure_force_ang_x.cpu().numpy()
        self.pressure_force_ang_y = s * fs.pressure_force_ang_y.cpu().numpy()
        self.pressure_force_ang_z = s * fs.pressure_force_ang_z.cpu().numpy()

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
            height = self._buoy_height[body_i]
            pos_z  = float(comp.com_pos[body_i][2])

            buoyancy_z = 0.0
            if mass > 0 and height > 0 and pos_z - height < surface:
                frac = min((surface + height - pos_z) / (2.0 * height), 1.0)
                # -rho_water * (mass/density) * gravity * frac
                # = -rho_water * V_link * gravity * frac  (upward when g<0)
                buoyancy_z = (
                    -self.rho_fluid * mass * g_z / self.rho_body * frac
                )

            physics.data.xfrc_applied[ind, 0] = (
                self.friction_force_lin_x[body_i] + self.pressure_force_x[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 1] = (
                self.friction_force_lin_y[body_i] + self.pressure_force_y[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 2] = (
                self.friction_force_lin_z[body_i] + self.pressure_force_z[body_i]
                + buoyancy_z
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 3] = (
                self.friction_force_ang_x[body_i] + self.pressure_force_ang_x[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 4] = (
                self.friction_force_ang_y[body_i] + self.pressure_force_ang_y[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 5] = (
                self.friction_force_ang_z[body_i] + self.pressure_force_ang_z[body_i]
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

        if not self.terminate:

            # 1. update SDF + body velocities from FARMS kinematics
            self.update(t, iteration, dt=timestep)

            # 2. recompute mu / normal fields
            if self.ndim == 3:
                self._recompute_mu_normals_3d()
            else:
                self._recompute_mu_normals_2d()

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

            # 4. compute fluid forces on each body
            if self.ndim == 3:
                fs.forces_method2_3d(fs.u0, fs.v0, fs.w0, fs.p0, iteration)
            elif self.force_method == "method1":
                fs.forces_method1(fs.u0, fs.v0, fs.p0, iteration)
            else:
                fs.forces_method2(fs.u0, fs.v0, fs.p0, iteration)

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

    # ------------------------------------------------------------------
    #  mu / normal recomputation helpers
    # ------------------------------------------------------------------
    def _recompute_mu_normals_2d(self):
        fs   = self.fluid_solver
        comp = fs.composite_body

        # CC normals are computed on-the-fly inside forces_method1/2;
        # CC mu0/mu1 are not consumed by any live code path.

        (fs.mu0_all_u, fs.mu1_all_u) = comp.mu_funcs(comp.sdf_val_u)
        (fs.normal_x_u, fs.normal_y_u) = comp.compute_normals(comp.sdf_val_u)

        (fs.mu0_all_v, fs.mu1_all_v) = comp.mu_funcs(comp.sdf_val_v)
        (fs.normal_x_v, fs.normal_y_v) = comp.compute_normals(comp.sdf_val_v)

    def _recompute_mu_normals_3d(self):
        fs   = self.fluid_solver
        comp = fs.composite_body

        if fs._compile_sdf:
            # ── Batched + compiled path: all 4 grids in one fused pass ──
            _fn = _mu_normals_batched_3d_compiled
            mu0, mu1, nx, ny, nz = _fn(
                comp.sdf_val_u, comp.sdf_val_v, comp.sdf_val_w,
                comp.sdf_val, comp.h, comp.eps,
            )
            # Clone outputs — CUDA graph buffers are overwritten on
            # subsequent replays, so we must detach before storing.
            mu0, mu1 = mu0.clone(), mu1.clone()
            nx, ny, nz = nx.clone(), ny.clone(), nz.clone()

            # Unstack: order is [u, v, w, cc]
            fs.mu0_all_u, fs.mu1_all_u = mu0[0], mu1[0]
            fs.normal_x_u, fs.normal_y_u, fs.normal_z_u = nx[0], ny[0], nz[0]

            fs.mu0_all_v, fs.mu1_all_v = mu0[1], mu1[1]
            fs.normal_x_v, fs.normal_y_v, fs.normal_z_v = nx[1], ny[1], nz[1]

            fs.mu0_all_w, fs.mu1_all_w = mu0[2], mu1[2]
            fs.normal_x_w, fs.normal_y_w, fs.normal_z_w = nx[2], ny[2], nz[2]

            fs.mu0_all, fs.mu1_all = mu0[3], mu1[3]
            fs.m_m0_all = 1 - fs.mu0_all
            fs.normal_x, fs.normal_y, fs.normal_z = nx[3], ny[3], nz[3]
        else:
            # ── Eager path: 4 × individual mu_funcs + compute_normals ──
            # u-grid
            (fs.mu0_all_u, fs.mu1_all_u) = comp.mu_funcs(comp.sdf_val_u)
            (fs.normal_x_u, fs.normal_y_u, fs.normal_z_u) = comp.compute_normals(comp.sdf_val_u)

            # v-grid
            (fs.mu0_all_v, fs.mu1_all_v) = comp.mu_funcs(comp.sdf_val_v)
            (fs.normal_x_v, fs.normal_y_v, fs.normal_z_v) = comp.compute_normals(comp.sdf_val_v)

            # w-grid
            (fs.mu0_all_w, fs.mu1_all_w) = comp.mu_funcs(comp.sdf_val_w)
            (fs.normal_x_w, fs.normal_y_w, fs.normal_z_w) = comp.compute_normals(comp.sdf_val_w)

            # CC-grid (p) — cached for forces_method2_3d
            (fs.mu0_all, fs.mu1_all) = comp.mu_funcs(comp.sdf_val)
            fs.m_m0_all = 1 - fs.mu0_all
            (fs.normal_x, fs.normal_y, fs.normal_z) = comp.compute_normals(comp.sdf_val)
