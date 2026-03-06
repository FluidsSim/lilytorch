"""
3-D BDIM handler for FARMS ↔ lilytorch coupling.

Extends the 2-D BDIMhandler to work with full 3-D velocity fields (u, v, w),
3-D signed-distance functions, and 6-DOF rigid-body kinematics.

Notes
-----
* Fluid → body **force coupling** is NOT yet implemented in 3-D.
  ``apply_forces`` currently zeros all external forces and prints a warning
  on the first call, so the MuJoCo body moves as if no fluid forces were
  present (muscle / gravity only).  This is still useful for visualising the
  3-D flow field around a swimming body.
* The custom ``fluid_step`` mirrors the 2-D handler's variable-density
  Poisson formulation, extended to three dimensions (ch, cv, cw).
"""

import numpy as np
from scipy.spatial.transform import Rotation
from lilytorch.src.solver import FluidSolver
import torch


class BDIMhandler3D:

    def __init__(self, yaml_file, data, physics, dtype=torch.float32):

        self.dtype = dtype
        if self.dtype == torch.float32:
            self.dtype_np = np.float32
        elif self.dtype == torch.float64:
            self.dtype_np = np.float64

        self.data = data  # list[AnimatData] from FARMS
        self.iteration = 0
        self.terminate = False

        self.pars = yaml_file

        self.fluid_solver = FluidSolver(
            self.pars,
            dtype=self.dtype,
            costum_update=True,
            compute_forces=False,  # 3-D forces not implemented yet
        )
        self.device = self.fluid_solver.device

        # Override the composite-body update with our FARMS-driven version
        self.fluid_solver.composite_body.update = self.update

        # Initial uniform inflow (u = U_inlet everywhere)
        gs = self.fluid_solver.grid_shape  # (nx, ny, nz)
        u_inlet = self.fluid_solver.adv_diff_solver.BC_values_u[1]
        self.fluid_solver.u0 = u_inlet * torch.ones(gs, device=self.device, dtype=self.dtype)

        # All three linear + angular axes
        self.lin_axes = [0, 1, 2]
        self.ang_axes = [0, 1, 2]

        self.force_scaling = 0.04  # carried over from 2-D (TODO: revisit for 3-D)
        self.rho_fluid = self.pars["solver"]["rho"]
        self.rho_body  = 800.0

        self._forces_warned = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def cython2numpy(self, array):
        return torch.from_numpy(np.array(array).astype(self.dtype_np)).to(self.device)

    # ------------------------------------------------------------------
    # update:  FARMS kinematics  →  SDF fields + body velocities
    # ------------------------------------------------------------------
    def update(self, t, iteration, dt=1):

        fs   = self.fluid_solver
        comp = fs.composite_body
        gs   = fs.grid_shape  # (nx, ny, nz)

        # ---- gather per-animat data from FARMS ----
        com_poses  = []
        urdf_poses = []
        Rs         = []
        lin_vels   = []
        ang_vels   = []

        for exp_data in self.data:
            sen = exp_data.sensors.links
            com_poses.append(
                self.cython2numpy(sen.com_positions()[iteration, :])
            )  # (nlinks, 3)
            urdf_poses.append(
                self.cython2numpy(sen.urdf_positions()[iteration, :])
            )  # (nlinks, 3)
            Rs.append(
                self.cython2numpy(
                    Rotation.from_quat(
                        sen.urdf_orientations()[iteration, :]
                    ).as_matrix().astype(self.dtype_np)
                )
            )  # (nlinks, 3, 3)
            lin_vels.append(
                self.cython2numpy(sen.com_lin_velocities()[iteration, :])
            )  # (nlinks, 3)
            # Full 3-D angular velocity vector per link
            nlinks = len(sen.names)
            ang_vels.append(
                self.cython2numpy(
                    np.stack([sen.com_ang_velocity(iteration, lk) for lk in range(nlinks)])
                )
            )  # (nlinks, 3)

        # ---- per-body update ----
        for body_i, body in enumerate(comp.bodies[:]):

            (animat_id, link_id) = comp.body_ids[body_i]

            com_pos  = com_poses[animat_id][link_id]   # (3,)
            urdf_pos = urdf_poses[animat_id][link_id]  # (3,)
            R        = Rs[animat_id][link_id]           # (3,3)
            lin_vel  = lin_vels[animat_id][link_id]     # (3,)
            ang_vel  = ang_vels[animat_id][link_id]     # (3,)

            # Mesh sub-bodies now use RegularGridInterpolatorGridSample3D
            # which accepts (xpt, ypt, zpt).  We transform 3-D grid points
            # into the body frame and evaluate the full 3-D SDF.
            # Use the *composite* staggered grids (which are 3-D) for all
            # coordinate lookups.

            # --- SDF at cell centres ---
            pos_trans = R.T @ (comp.stacked_xy - urdf_pos[:, None])  # (3, N)
            px = pos_trans[0].reshape(gs)
            py = pos_trans[1].reshape(gs)
            pz = pos_trans[2].reshape(gs)
            comp.sdf_vals[body_i] = body.sdf(px, py, pz)

            # --- SDF at u-staggered ---
            pos_trans_u = R.T @ (comp.stacked_xy_u - urdf_pos[:, None])
            comp.sdf_vals_u[body_i] = body.sdf(
                pos_trans_u[0].reshape(gs),
                pos_trans_u[1].reshape(gs),
                pos_trans_u[2].reshape(gs),
            )

            # --- SDF at v-staggered ---
            pos_trans_v = R.T @ (comp.stacked_xy_v - urdf_pos[:, None])
            comp.sdf_vals_v[body_i] = body.sdf(
                pos_trans_v[0].reshape(gs),
                pos_trans_v[1].reshape(gs),
                pos_trans_v[2].reshape(gs),
            )

            # --- SDF at w-staggered ---
            pos_trans_w = R.T @ (comp.stacked_xy_w - urdf_pos[:, None])
            comp.sdf_vals_w[body_i] = body.sdf(
                pos_trans_w[0].reshape(gs),
                pos_trans_w[1].reshape(gs),
                pos_trans_w[2].reshape(gs),
            )

            # --- body velocities: v = v_lin + ω × (x - x_com) ---
            # u-staggered  (only u-component kept)
            rx_u = comp.Xu_stag - com_pos[0]
            ry_u = comp.Yu_stag - com_pos[1]
            rz_u = comp.Zu_stag - com_pos[2]
            comp.u_vals[body_i] = (
                lin_vel[0]
                + ang_vel[1] * rz_u - ang_vel[2] * ry_u
            )

            # v-staggered  (only v-component kept)
            rx_v = comp.Xv_stag - com_pos[0]
            ry_v = comp.Yv_stag - com_pos[1]
            rz_v = comp.Zv_stag - com_pos[2]
            comp.v_vals[body_i] = (
                lin_vel[1]
                + ang_vel[2] * rx_v - ang_vel[0] * rz_v
            )

            # w-staggered  (only w-component kept)
            rx_w = comp.Xw_stag - com_pos[0]
            ry_w = comp.Yw_stag - com_pos[1]
            rz_w = comp.Zw_stag - com_pos[2]
            comp.w_vals[body_i] = (
                lin_vel[2]
                + ang_vel[0] * ry_w - ang_vel[1] * rx_w
            )

            # store com position for (future) force computation
            comp.com_pos[body_i] = com_pos

        # ---- union: pick closest body at every grid point ----
        gs_flat = comp.sdf_vals.shape  # (nbodies, *grid_shape)

        idx = comp.sdf_vals.argmin(0).unsqueeze(0).expand(gs_flat)
        comp.sdf_val = comp.sdf_vals.gather(0, idx)[0].reshape(gs)

        idx_u = comp.sdf_vals_u.argmin(0).unsqueeze(0).expand(gs_flat)
        comp.sdf_val_u = comp.sdf_vals_u.gather(0, idx_u)[0].reshape(gs)
        comp.body_u    = comp.u_vals.gather(0, idx_u)[0].reshape(gs)

        idx_v = comp.sdf_vals_v.argmin(0).unsqueeze(0).expand(gs_flat)
        comp.sdf_val_v = comp.sdf_vals_v.gather(0, idx_v)[0].reshape(gs)
        comp.body_v    = comp.v_vals.gather(0, idx_v)[0].reshape(gs)

        idx_w = comp.sdf_vals_w.argmin(0).unsqueeze(0).expand(gs_flat)
        comp.sdf_val_w = comp.sdf_vals_w.gather(0, idx_w)[0].reshape(gs)
        comp.body_w    = comp.w_vals.gather(0, idx_w)[0].reshape(gs)

    # ------------------------------------------------------------------
    # fluid_step: one BDIM time-step (advection-diffusion + projection)
    # ------------------------------------------------------------------
    def fluid_step(self, u, v, w, p, timestep):

        fs = self.fluid_solver

        # ====== advection + diffusion ======
        (uprime, vprime, wprime) = fs.adv_diff_solver.solve(u, v, w)
        fs.adv_diff_solver.set_BCs(uprime, vprime, wprime)

        # ====== BDIM2 meta-equation ======
        uprime = (
            fs.mu0_all_u * uprime
            + fs.m_m0_all_u * fs.composite_body.body_u
            + fs.mu1_all_u * fs.normal_derivative(
                uprime - fs.composite_body.body_u,
                fs.normal_x_u, fs.normal_y_u, fs.normal_z_u,
            )
        )
        vprime = (
            fs.mu0_all_v * vprime
            + fs.m_m0_all_v * fs.composite_body.body_v
            + fs.mu1_all_v * fs.normal_derivative(
                vprime - fs.composite_body.body_v,
                fs.normal_x_v, fs.normal_y_v, fs.normal_z_v,
            )
        )
        wprime = (
            fs.mu0_all_w * wprime
            + fs.m_m0_all_w * fs.composite_body.body_w
            + fs.mu1_all_w * fs.normal_derivative(
                wprime - fs.composite_body.body_w,
                fs.normal_x_w, fs.normal_y_w, fs.normal_z_w,
            )
        )

        # ====== divergence ======
        fs.div = fs.divergence(uprime, vprime, wprime)

        if fs.poisson_method == "fft":
            # ---- FFT solver (constant-coefficient Poisson) ----
            coeff = timestep / self.rho_fluid
            p = fs.poisson_solverFFT.solve(fs.div / coeff)
            (p_x, p_y, p_z) = fs.gradient(p)
            u = uprime - coeff * p_x
            v = vprime - coeff * p_y
            w = wprime - coeff * p_z
        else:
            # ---- Multigrid solver (variable-coefficient Poisson) ----
            rho_u = self.rho_fluid * fs.mu0_all_u + self.rho_body * fs.m_m0_all_u
            rho_v = self.rho_fluid * fs.mu0_all_v + self.rho_body * fs.m_m0_all_v
            rho_w = self.rho_fluid * fs.mu0_all_w + self.rho_body * fs.m_m0_all_w

            ch = timestep / rho_u
            cv = timestep / rho_v
            cw = timestep / rho_w

            p, _ = fs.poisson_solver.solve_multigrid(
                fs.div[1:-1, 1:-1, 1:-1],
                torch.zeros_like(p),
                (timestep / self.rho_body) * torch.ones_like(fs.div),
                ch=ch[1:,  1:-1, 1:-1],
                cv=cv[1:-1, 1:,  1:-1],
                cw=cw[1:-1, 1:-1, 1:],
            )

            # ====== projection step ======
            (p_x, p_y, p_z) = fs.gradient(p)
            u = uprime - ch * p_x
            v = vprime - cv * p_y
            w = wprime - cw * p_z

        fs.adv_diff_solver.set_BCs(u, v, w)

        return (u, v, w, p)

    # ------------------------------------------------------------------
    # apply_forces:  fluid → body  (NOT YET IMPLEMENTED FOR 3-D)
    # ------------------------------------------------------------------
    def apply_forces(self, task, physics):
        if not self._forces_warned:
            print(
                "[BDIMhandler3D] WARNING: 3-D fluid→body force coupling is not "
                "implemented.  External forces on all links are set to zero."
            )
            self._forces_warned = True

        for body_i, body in enumerate(self.fluid_solver.composite_body.bodies[:]):
            (animat_id, link_id) = self.fluid_solver.composite_body.body_ids[body_i]
            ind_task = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]
            # zero all 6 DOF forces/torques
            physics.data.xfrc_applied[ind_task, :] = 0.0

    # ------------------------------------------------------------------
    # step:  one full coupled step  (called by FluidExtension.before_step)
    # ------------------------------------------------------------------
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

            # 2. mu / normal fields at cell-centres
            (fs.mu0_all, fs.mu1_all) = fs.composite_body.mu_funcs(
                fs.composite_body.sdf_val
            )
            fs.m_m0_all = 1 - fs.mu0_all
            (_, fs.normal_x, fs.normal_y, fs.normal_z, _) = (
                fs.composite_body.compute_sdf_properties(fs.composite_body.sdf_val)
            )

            # 3. mu / normals at u-staggered
            (fs.mu0_all_u, fs.mu1_all_u) = fs.composite_body.mu_funcs(
                fs.composite_body.sdf_val_u
            )
            fs.m_m0_all_u = 1 - fs.mu0_all_u
            (_, fs.normal_x_u, fs.normal_y_u, fs.normal_z_u, _) = (
                fs.composite_body.compute_sdf_properties(fs.composite_body.sdf_val_u)
            )

            # 4. mu / normals at v-staggered
            (fs.mu0_all_v, fs.mu1_all_v) = fs.composite_body.mu_funcs(
                fs.composite_body.sdf_val_v
            )
            fs.m_m0_all_v = 1 - fs.mu0_all_v
            (_, fs.normal_x_v, fs.normal_y_v, fs.normal_z_v, _) = (
                fs.composite_body.compute_sdf_properties(fs.composite_body.sdf_val_v)
            )

            # 5. mu / normals at w-staggered
            (fs.mu0_all_w, fs.mu1_all_w) = fs.composite_body.mu_funcs(
                fs.composite_body.sdf_val_w
            )
            fs.m_m0_all_w = 1 - fs.mu0_all_w
            (_, fs.normal_x_w, fs.normal_y_w, fs.normal_z_w, _) = (
                fs.composite_body.compute_sdf_properties(fs.composite_body.sdf_val_w)
            )

            # 6. BDIM fluid step
            (u, v, w, p) = self.fluid_step(
                fs.u0, fs.v0, fs.w0, fs.p0, timestep
            )

            (fs.u0, fs.v0, fs.w0, fs.p0) = (u, v, w, p)

            # 7. plotting / saving
            self.terminate = fs.plotting_and_saving(
                fs.u0, fs.v0, fs.p0, iteration, w_vel=fs.w0
            )

            # 8. apply (zero) forces back to FARMS
            self.apply_forces(task, physics)

        self.iteration += 1
