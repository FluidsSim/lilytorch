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
            costum_update=True,
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

            # SDF at cell centres
            pos_trans = R.T @ (comp.stacked_xy - urdf_pos[:, None])
            comp.sdf_vals[body_i] = body.sdf(
                pos_trans[0].reshape(comp.nx, comp.ny),
                pos_trans[1].reshape(comp.nx, comp.ny),
            )

            # SDF at u-staggered
            pos_trans_u = R.T @ (body.stacked_xy_u - urdf_pos[:, None])
            comp.sdf_vals_u[body_i] = body.sdf(
                pos_trans_u[0].reshape(body.nx, body.ny),
                pos_trans_u[1].reshape(body.nx, body.ny),
            )

            # SDF at v-staggered
            pos_trans_v = R.T @ (body.stacked_xy_v - urdf_pos[:, None])
            comp.sdf_vals_v[body_i] = body.sdf(
                pos_trans_v[0].reshape(body.nx, body.ny),
                pos_trans_v[1].reshape(body.nx, body.ny),
            )

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
    def _update_3d(self, t, iteration, dt=1):
        fs   = self.fluid_solver
        comp = fs.composite_body
        gs   = fs.grid_shape

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

        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]

            com_pos  = com_poses[animat_id][link_id]
            urdf_pos = urdf_poses[animat_id][link_id]
            R        = Rs[animat_id][link_id]
            lin_vel  = lin_vels[animat_id][link_id]
            ang_vel  = ang_vels[animat_id][link_id]

            # SDF at cell centres
            pos_cc = R.T @ (comp.stacked_xy - urdf_pos[:, None])
            sdf_cc = body.sdf(
                pos_cc[0].reshape(gs),
                pos_cc[1].reshape(gs),
                pos_cc[2].reshape(gs),
            )

            # SDF + velocity at u-staggered
            pos_u = R.T @ (comp.stacked_xy_u - urdf_pos[:, None])
            sdf_u = body.sdf(
                pos_u[0].reshape(gs), pos_u[1].reshape(gs), pos_u[2].reshape(gs),
            )
            vel_u = (lin_vel[0]
                     + ang_vel[1] * (comp.Zu_stag - com_pos[2])
                     - ang_vel[2] * (comp.Yu_stag - com_pos[1]))

            # SDF + velocity at v-staggered
            pos_v = R.T @ (comp.stacked_xy_v - urdf_pos[:, None])
            sdf_v = body.sdf(
                pos_v[0].reshape(gs), pos_v[1].reshape(gs), pos_v[2].reshape(gs),
            )
            vel_v = (lin_vel[1]
                     + ang_vel[2] * (comp.Xv_stag - com_pos[0])
                     - ang_vel[0] * (comp.Zv_stag - com_pos[2]))

            # SDF + velocity at w-staggered
            pos_w = R.T @ (comp.stacked_xy_w - urdf_pos[:, None])
            sdf_w = body.sdf(
                pos_w[0].reshape(gs), pos_w[1].reshape(gs), pos_w[2].reshape(gs),
            )
            vel_w = (lin_vel[2]
                     + ang_vel[0] * (comp.Yw_stag - com_pos[1])
                     - ang_vel[1] * (comp.Xw_stag - com_pos[0]))

            # per-body cell-centre SDF for force computation
            comp.sdf_vals[body_i] = sdf_cc

            # streaming min: update union fields
            if body_i == 0:
                comp.sdf_val   = sdf_cc
                comp.sdf_val_u = sdf_u;  comp.body_u = vel_u
                comp.sdf_val_v = sdf_v;  comp.body_v = vel_v
                comp.sdf_val_w = sdf_w;  comp.body_w = vel_w
            else:
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

        # BDIM2 meta-equation
        uprime = (
            fs.mu0_all_u * uprime
            + fs.m_m0_all_u * fs.composite_body.body_u
            + fs.mu1_all_u * fs.normal_derivative(
                uprime - fs.composite_body.body_u,
                fs.normal_x_u, fs.normal_y_u,
            )
        )
        vprime = (
            fs.mu0_all_v * vprime
            + fs.m_m0_all_v * fs.composite_body.body_v
            + fs.mu1_all_v * fs.normal_derivative(
                vprime - fs.composite_body.body_v,
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
            # variable-density multigrid
            rho_u = (self.rho_fluid * fs.mu0_all_u
                     + self.rho_body  * fs.m_m0_all_u)
            rho_v = (self.rho_fluid * fs.mu0_all_v
                     + self.rho_body  * fs.m_m0_all_v)
            ch_full = timestep / rho_u
            cv_full = timestep / rho_v

            p, _ = fs.poisson_solver.solve_multigrid(
                fs.div[1:-1, 1:-1],
                torch.zeros_like(p),
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

        # BDIM2 meta-equation
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

        for body_i in range(len(fs.composite_body.bodies)):
            (animat_id, link_id) = fs.composite_body.body_ids[body_i]
            ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            physics.data.xfrc_applied[ind, 0] = (
                self.friction_force_lin_x[body_i] + self.pressure_force_x[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 1] = (
                self.friction_force_lin_y[body_i] + self.pressure_force_y[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 2] = (
                self.friction_force_lin_z[body_i] + self.pressure_force_z[body_i]
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
                    fs.u0, fs.v0, fs.p0, iteration, w_vel=fs.w0
                )
            else:
                self.terminate = fs.plotting_debug(
                    fs.u0, fs.v0, fs.p0, iteration
                )

            # 6. apply forces to MuJoCo bodies
            self.apply_forces(task, physics)

        self.iteration += 1

    # ------------------------------------------------------------------
    #  mu / normal recomputation helpers
    # ------------------------------------------------------------------
    def _recompute_mu_normals_2d(self):
        fs   = self.fluid_solver
        comp = fs.composite_body

        (fs.mu0_all, fs.mu1_all) = comp.mu_funcs(comp.sdf_val)
        fs.m_m0_all = 1 - fs.mu0_all
        (_, fs.normal_x, fs.normal_y, _) = comp.compute_sdf_properties(comp.sdf_val)

        (fs.mu0_all_u, fs.mu1_all_u) = comp.mu_funcs(comp.sdf_val_u)
        fs.m_m0_all_u = 1 - fs.mu0_all_u
        (_, fs.normal_x_u, fs.normal_y_u, _) = comp.compute_sdf_properties(comp.sdf_val_u)

        (fs.mu0_all_v, fs.mu1_all_v) = comp.mu_funcs(comp.sdf_val_v)
        fs.m_m0_all_v = 1 - fs.mu0_all_v
        (_, fs.normal_x_v, fs.normal_y_v, _) = comp.compute_sdf_properties(comp.sdf_val_v)

    def _recompute_mu_normals_3d(self):
        fs   = self.fluid_solver
        comp = fs.composite_body

        (fs.mu0_all, fs.mu1_all) = comp.mu_funcs(comp.sdf_val)
        fs.m_m0_all = 1 - fs.mu0_all
        (_, fs.normal_x, fs.normal_y, fs.normal_z, _) = (
            comp.compute_sdf_properties(comp.sdf_val)
        )

        (fs.mu0_all_u, fs.mu1_all_u) = comp.mu_funcs(comp.sdf_val_u)
        fs.m_m0_all_u = 1 - fs.mu0_all_u
        (_, fs.normal_x_u, fs.normal_y_u, fs.normal_z_u, _) = (
            comp.compute_sdf_properties(comp.sdf_val_u)
        )

        (fs.mu0_all_v, fs.mu1_all_v) = comp.mu_funcs(comp.sdf_val_v)
        fs.m_m0_all_v = 1 - fs.mu0_all_v
        (_, fs.normal_x_v, fs.normal_y_v, fs.normal_z_v, _) = (
            comp.compute_sdf_properties(comp.sdf_val_v)
        )

        (fs.mu0_all_w, fs.mu1_all_w) = comp.mu_funcs(comp.sdf_val_w)
        fs.m_m0_all_w = 1 - fs.mu0_all_w
        (_, fs.normal_x_w, fs.normal_y_w, fs.normal_z_w, _) = (
            comp.compute_sdf_properties(comp.sdf_val_w)
        )
