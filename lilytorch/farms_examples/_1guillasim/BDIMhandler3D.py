"""
3-D BDIM handler for FARMS ↔ lilytorch coupling.

Extends the 2-D BDIMhandler to work with full 3-D velocity fields (u, v, w),
3-D signed-distance functions, and 6-DOF rigid-body kinematics.

Notes
-----
* Fluid → body **force coupling** uses the same smoothed-delta volume-
  integration approach as the 2-D handler (``forces_method2_3d``).
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
            compute_forces=True,
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

        # Allocate per-body cell-centre SDF storage needed by forces_method2_3d.
        # MultiAnimatBodies deliberately omits this to save memory, but force
        # computation requires per-body SDF values for the smoothed-delta
        # integration.  Only cell-centre SDFs are needed (not staggered).
        comp = self.fluid_solver.composite_body
        comp.sdf_vals = torch.zeros(
            (comp.nbodies, *self.fluid_solver.grid_shape),
            device=self.device, dtype=self.dtype,
        )

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

        # ---- streaming union: process bodies one at a time ----
        # Instead of filling (nbodies, *gs) stacks and then reducing,
        # we keep a running-min SDF and pick-closest velocity, saving
        # 7 × nbodies × field_size of GPU memory (~1.9 GB for 9 bodies).
        for body_i, body in enumerate(comp.bodies):

            (animat_id, link_id) = comp.body_ids[body_i]

            com_pos  = com_poses[animat_id][link_id]   # (3,)
            urdf_pos = urdf_poses[animat_id][link_id]  # (3,)
            R        = Rs[animat_id][link_id]           # (3,3)
            lin_vel  = lin_vels[animat_id][link_id]     # (3,)
            ang_vel  = ang_vels[animat_id][link_id]     # (3,)

            # --- SDF at cell centres ---
            pos_trans = R.T @ (comp.stacked_xy - urdf_pos[:, None])
            sdf_cc = body.sdf(
                pos_trans[0].reshape(gs),
                pos_trans[1].reshape(gs),
                pos_trans[2].reshape(gs),
            )

            # --- SDF + velocity at u-staggered ---
            pos_trans_u = R.T @ (comp.stacked_xy_u - urdf_pos[:, None])
            sdf_u = body.sdf(
                pos_trans_u[0].reshape(gs),
                pos_trans_u[1].reshape(gs),
                pos_trans_u[2].reshape(gs),
            )
            vel_u = (lin_vel[0]
                     + ang_vel[1] * (comp.Zu_stag - com_pos[2])
                     - ang_vel[2] * (comp.Yu_stag - com_pos[1]))

            # --- SDF + velocity at v-staggered ---
            pos_trans_v = R.T @ (comp.stacked_xy_v - urdf_pos[:, None])
            sdf_v = body.sdf(
                pos_trans_v[0].reshape(gs),
                pos_trans_v[1].reshape(gs),
                pos_trans_v[2].reshape(gs),
            )
            vel_v = (lin_vel[1]
                     + ang_vel[2] * (comp.Xv_stag - com_pos[0])
                     - ang_vel[0] * (comp.Zv_stag - com_pos[2]))

            # --- SDF + velocity at w-staggered ---
            pos_trans_w = R.T @ (comp.stacked_xy_w - urdf_pos[:, None])
            sdf_w = body.sdf(
                pos_trans_w[0].reshape(gs),
                pos_trans_w[1].reshape(gs),
                pos_trans_w[2].reshape(gs),
            )
            vel_w = (lin_vel[2]
                     + ang_vel[0] * (comp.Yw_stag - com_pos[1])
                     - ang_vel[1] * (comp.Xw_stag - com_pos[0]))

            # --- store per-body cell-centre SDF for force computation ---
            comp.sdf_vals[body_i] = sdf_cc

            # --- streaming min: update union fields ---
            if body_i == 0:
                comp.sdf_val   = sdf_cc
                comp.sdf_val_u = sdf_u
                comp.body_u    = vel_u
                comp.sdf_val_v = sdf_v
                comp.body_v    = vel_v
                comp.sdf_val_w = sdf_w
                comp.body_w    = vel_w
            else:
                mask = sdf_cc < comp.sdf_val
                comp.sdf_val = torch.where(mask, sdf_cc, comp.sdf_val)

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
            body.com_pos = com_pos  # per-body com_pos for forces_method2_3d

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
    # apply_forces:  fluid → body  (3-D viscous + pressure forces)
    # ------------------------------------------------------------------
    def apply_forces(self, task, physics):

        fs = self.fluid_solver

        self.friction_force_lin_x = fs.friction_force_lin_x.cpu().numpy()
        self.friction_force_lin_y = fs.friction_force_lin_y.cpu().numpy()
        self.friction_force_lin_z = fs.friction_force_lin_z.cpu().numpy()
        self.friction_force_ang_x = fs.friction_force_ang_x.cpu().numpy()
        self.friction_force_ang_y = fs.friction_force_ang_y.cpu().numpy()
        self.friction_force_ang_z = fs.friction_force_ang_z.cpu().numpy()

        self.pressure_force_x     = fs.pressure_force_x.cpu().numpy()
        self.pressure_force_y     = fs.pressure_force_y.cpu().numpy()
        self.pressure_force_z     = fs.pressure_force_z.cpu().numpy()
        self.pressure_force_ang_x = fs.pressure_force_ang_x.cpu().numpy()
        self.pressure_force_ang_y = fs.pressure_force_ang_y.cpu().numpy()
        self.pressure_force_ang_z = fs.pressure_force_ang_z.cpu().numpy()

        for body_i, body in enumerate(fs.composite_body.bodies[:]):
            (animat_id, link_id) = fs.composite_body.body_ids[body_i]
            ind_task = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            # linear forces  (Fx, Fy, Fz)
            physics.data.xfrc_applied[ind_task, 0] = (
                self.friction_force_lin_x[body_i] + self.pressure_force_x[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind_task, 1] = (
                self.friction_force_lin_y[body_i] + self.pressure_force_y[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind_task, 2] = (
                self.friction_force_lin_z[body_i] + self.pressure_force_z[body_i]
            ) * task.units.newtons

            # torques  (Tx, Ty, Tz)
            physics.data.xfrc_applied[ind_task, 3] = (
                self.friction_force_ang_x[body_i] + self.pressure_force_ang_x[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind_task, 4] = (
                self.friction_force_ang_y[body_i] + self.pressure_force_ang_y[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind_task, 5] = (
                self.friction_force_ang_z[body_i] + self.pressure_force_ang_z[body_i]
            ) * task.units.newtons

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

            # 7. compute fluid forces on each body
            fs.forces_method2_3d(fs.u0, fs.v0, fs.w0, fs.p0, iteration)

            # 8. plotting / saving
            self.terminate = fs.plotting_and_saving(
                fs.u0, fs.v0, fs.p0, iteration, w_vel=fs.w0
            )

            # 9. apply forces back to FARMS
            self.apply_forces(task, physics)

        self.iteration += 1
