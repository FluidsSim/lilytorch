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
            custom_update=True,
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

        # FARMS-style buoyancy parameters (drag.pyx compute_buoyancy).
        # With all-Neumann BCs the pressure field is purely dynamic (no
        # hydrostatic component), so buoyancy must be added explicitly.
        self.gravity_z = float(physics.model.opt.gravity[2])  # e.g. -9.81
        self.water_surface = float(self.pars["solver"]["zmax"])
        self._buoyancy_initialized = False

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
    def _init_buoyancy_params(self, task, physics):
        """Precompute per-body mass & half-height for FARMS-style buoyancy.

        Called lazily on first ``apply_forces`` because ``task`` is not
        available at ``__init__`` time.
        """
        comp = self.fluid_solver.composite_body
        n = comp.nbodies

        self._buoy_mass   = np.zeros(n)
        self._buoy_height = np.zeros(n)

        for body_i in range(n):
            (animat_id, link_id) = comp.body_ids[body_i]
            ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            self._buoy_mass[body_i] = float(physics.model.body_mass[ind])

            max_rbound = 0.0
            for gi in range(physics.model.ngeom):
                if int(physics.model.geom_bodyid[gi]) == ind:
                    rb = float(physics.model.geom_rbound[gi])
                    if rb > max_rbound:
                        max_rbound = rb
            self._buoy_height[body_i] = 0.5 * max_rbound

        self._buoyancy_initialized = True

    def cython2numpy(self, array):
        return torch.from_numpy(np.array(array).astype(self.dtype_np)).to(self.device)

    # ------------------------------------------------------------------
    # Batched SDF: evaluate body SDF at 4 grids in one interpolator call
    # ------------------------------------------------------------------
    def _batched_sdf_4grids(self, body, R_T, urdf_pos, stacked_grids_4, gs):
        """Evaluate body.sdf at 4 staggered grids with one shared rotation.

        Instead of 4 separate rotation transforms + 4 ``body.sdf()`` calls,
        we concatenate all 4 grids, apply R^T once, then call ``body.sdf()``
        once with 4N points.  The interpolator handles flat tensors natively
        so no internal surgery is required.

        Parameters
        ----------
        body : BodyMesh  – must have body.sdf (RegularGridInterpolatorGridSample3D)
        R_T  : (3, 3) – transposed rotation matrix
        urdf_pos : (3,) – translation offset
        stacked_grids_4 : (4, 3, N)  – pre-stacked coordinate grids
                          [cc, u-stag, v-stag, w-stag], each (3, N)
        gs   : tuple – grid shape (nx, ny, nz)

        Returns
        -------
        sdf_cc, sdf_u, sdf_v, sdf_w : each of shape gs
        """
        N = stacked_grids_4.shape[2]  # flat points per grid

        # Batched coordinate transform: R^T @ (grid_pts - urdf_pos)
        # stacked_grids_4: (4, 3, N) → shifted: (4, 3, N)
        shifted = stacked_grids_4 - urdf_pos[None, :, None]
        # Reshape (4, 3, N) → (3, 4N) for a single matmul
        rotated = R_T @ shifted.transpose(0, 1).reshape(3, 4 * N)  # (3, 4N)

        # Call body.sdf once with all 4N points (flat tensors)
        sdf_all = body.sdf(rotated[0], rotated[1], rotated[2])  # (4N,)

        # Split back into 4 grids of shape gs
        sdf_4 = sdf_all.reshape(4, *gs)
        return sdf_4[0], sdf_4[1], sdf_4[2], sdf_4[3]

    # ------------------------------------------------------------------
    # update:  FARMS kinematics  →  SDF fields + body velocities
    # ------------------------------------------------------------------
    def update(self, t, iteration, dt=1):

        fs   = self.fluid_solver
        comp = fs.composite_body
        gs   = fs.grid_shape  # (nx, ny, nz)

        # ---- pre-stack the 4 grids once (reused every body) ----------
        # Each stacked_xy is (3, N); we create (4, 3, N).
        if not hasattr(self, '_stacked_grids_4'):
            self._stacked_grids_4 = torch.stack([
                comp.stacked_xy,
                comp.stacked_xy_u,
                comp.stacked_xy_v,
                comp.stacked_xy_w,
            ])  # (4, 3, N)

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

            # --- batched SDF: all 4 grids in one grid_sample call ------
            sdf_cc, sdf_u, sdf_v, sdf_w = self._batched_sdf_4grids(
                body, R.T, urdf_pos, self._stacked_grids_4, gs,
            )

            # --- body velocities at staggered grids (rigid body) ------
            vel_u = (lin_vel[0]
                     + ang_vel[1] * (comp.Zu_stag - com_pos[2])
                     - ang_vel[2] * (comp.Yu_stag - com_pos[1]))

            vel_v = (lin_vel[1]
                     + ang_vel[2] * (comp.Xv_stag - com_pos[0])
                     - ang_vel[0] * (comp.Zv_stag - com_pos[2]))

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
            # ---- Multigrid / MGCG solver (variable-coefficient Poisson) ----
            rho_u = self.rho_fluid * fs.mu0_all_u + self.rho_body * fs.m_m0_all_u
            rho_v = self.rho_fluid * fs.mu0_all_v + self.rho_body * fs.m_m0_all_v
            rho_w = self.rho_fluid * fs.mu0_all_w + self.rho_body * fs.m_m0_all_w

            ch = timestep / rho_u
            cv = timestep / rho_v
            cw = timestep / rho_w

            # Select solve method: MGCG or standalone multigrid
            _poisson_solve = (fs.poisson_solver.solve_mgcg
                              if fs.poisson_method == "mgcg"
                              else fs.poisson_solver.solve_multigrid)

            # Warm-start: reuse previous pressure as initial guess
            p0 = p if getattr(fs, 'poisson_warm_start', False) else torch.zeros_like(p)

            p, _ = _poisson_solve(
                fs.div[1:-1, 1:-1, 1:-1],
                p0,
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

        # ---- FARMS-identical buoyancy (drag.pyx compute_buoyancy) ----
        if not self._buoyancy_initialized:
            self._init_buoyancy_params(task, physics)

        comp    = fs.composite_body
        surface = self.water_surface
        g_z     = self.gravity_z

        for body_i, body in enumerate(comp.bodies[:]):
            (animat_id, link_id) = comp.body_ids[body_i]
            ind_task = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            # FARMS buoyancy per link
            mass   = self._buoy_mass[body_i]
            height = self._buoy_height[body_i]
            pos_z  = float(comp.com_pos[body_i][2])

            buoyancy_z = 0.0
            if mass > 0 and height > 0 and pos_z - height < surface:
                frac = min((surface + height - pos_z) / (2.0 * height), 1.0)
                buoyancy_z = -self.rho_fluid * mass * g_z / self.rho_body * frac

            # linear forces  (Fx, Fy, Fz)
            physics.data.xfrc_applied[ind_task, 0] = (
                self.friction_force_lin_x[body_i] + self.pressure_force_x[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind_task, 1] = (
                self.friction_force_lin_y[body_i] + self.pressure_force_y[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind_task, 2] = (
                self.friction_force_lin_z[body_i] + self.pressure_force_z[body_i]
                + buoyancy_z
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

            # 2. normals at cell-centres (needed by forces_method2_3d)
            # NOTE: mu0_all / mu1_all / m_m0_all at cell centres are NOT
            # consumed by any live code path — only the staggered variants
            # (_u, _v, _w) are used in the BDIM meta-equation and Poisson
            # solve.  The cell-centre mu_funcs call was therefore removed.
            (fs.normal_x, fs.normal_y, fs.normal_z) = (
                fs.composite_body.compute_normals(fs.composite_body.sdf_val)
            )

            # 3. mu / normals at u-staggered
            (fs.mu0_all_u, fs.mu1_all_u) = fs.composite_body.mu_funcs(
                fs.composite_body.sdf_val_u
            )
            fs.m_m0_all_u = 1 - fs.mu0_all_u
            (fs.normal_x_u, fs.normal_y_u, fs.normal_z_u) = (
                fs.composite_body.compute_normals(fs.composite_body.sdf_val_u)
            )

            # 4. mu / normals at v-staggered
            (fs.mu0_all_v, fs.mu1_all_v) = fs.composite_body.mu_funcs(
                fs.composite_body.sdf_val_v
            )
            fs.m_m0_all_v = 1 - fs.mu0_all_v
            (fs.normal_x_v, fs.normal_y_v, fs.normal_z_v) = (
                fs.composite_body.compute_normals(fs.composite_body.sdf_val_v)
            )

            # 5. mu / normals at w-staggered
            (fs.mu0_all_w, fs.mu1_all_w) = fs.composite_body.mu_funcs(
                fs.composite_body.sdf_val_w
            )
            fs.m_m0_all_w = 1 - fs.mu0_all_w
            (fs.normal_x_w, fs.normal_y_w, fs.normal_z_w) = (
                fs.composite_body.compute_normals(fs.composite_body.sdf_val_w)
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
