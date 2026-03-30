"""Network controller – 2-D sphere sedimentation (Gazzola et al.)

Uses BDIM2 meta-equation with single Euler step.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from lilytorch.src.solver import FluidSolver
import torch


class BDIMhandler:

    def __init__(self, yaml_file, data, physics, dtype=torch.float64):

        self.dtype = dtype
        self.dtype_np = np.float32 if dtype == torch.float32 else np.float64

        self.data = data          # list[AnimatData] from FARMS
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

        # Override composite-body update with FARMS-driven kinematics
        self.fluid_solver.composite_body.update = self.update

        # MuJoCo (x,z) sagittal plane <-> fluid-solver (x,y)
        self.lin_axes = [0, 2]    # MuJoCo x,z -> fluid x,y
        self.ang_axes = [1]       # MuJoCo rotation around y -> 2D angular vel

        self.force_scaling = 1.0

        # Contact solver parameters
        physics.model.geom_solref[:, 0] = 0.001
        physics.model.geom_solref[:, 1] = 0.5

        # Physical parameters (read from config)
        self.rho_fluid = self.pars['solver']['rho']
        self.rho_body  = self.pars['solver'].get('rho_body', self.rho_fluid)
        self.radius    = 0.0025
        self.force_method = self.pars['solver'].get('force_method', 'method2')
        if self.force_method not in ('method1', 'method2'):
            raise ValueError(
                f"Unknown force_method '{self.force_method}'. "
                "Choose 'method1' or 'method2'."
            )

        # ---- FARMS-style buoyancy ----
        # With all-Neumann BCs the BDIM pressure field is purely dynamic;
        # no hydrostatic gradient builds up, so buoyancy must be added
        # explicitly.  Replicates FARMS' compute_buoyancy() from drag.pyx.
        self.gravity_z = float(physics.model.opt.gravity[2])  # e.g. -9.81
        self.water_surface = self.pars['solver']['ymax']
        self._buoyancy_initialized = False

        # ---- allocate per-body stack tensors --------------------------
        comp = self.fluid_solver.composite_body
        gs   = comp.grid_shape
        nb   = comp.nbodies
        comp.sdf_vals   = torch.zeros((nb, *gs), device=self.device, dtype=self.dtype)
        comp.sdf_vals_u = torch.zeros((nb, *gs), device=self.device, dtype=self.dtype)
        comp.sdf_vals_v = torch.zeros((nb, *gs), device=self.device, dtype=self.dtype)
        comp.u_vals     = torch.zeros((nb, *gs), device=self.device, dtype=self.dtype)
        comp.v_vals     = torch.zeros((nb, *gs), device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def cython2numpy(self, array):
        return torch.from_numpy(np.array(array).astype(self.dtype_np)).to(self.device)

    # ==================================================================
    #  update: FARMS kinematics -> SDF + body velocities
    # ==================================================================
    def update(self, t, iteration, dt=1):
        fs   = self.fluid_solver
        comp = fs.composite_body

        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]
            sen = self.data[animat_id].sensors.links

            com_pos  = self.cython2numpy(sen.com_positions()[iteration, link_id])[self.lin_axes]
            urdf_pos = self.cython2numpy(sen.urdf_positions()[iteration, link_id])[self.lin_axes]
            R = self.cython2numpy(
                Rotation.from_quat(sen.urdf_orientations()[iteration, link_id])
                .as_matrix().astype(self.dtype_np)
            )[self.lin_axes, :][:, self.lin_axes]
            lin_vel = self.cython2numpy(sen.com_lin_velocities()[iteration, link_id])[self.lin_axes]
            ang_vel = self.cython2numpy(sen.com_ang_velocity(iteration, link_id))[self.ang_axes]

            R_T = R.T

            # SDF at cell centres
            pos_trans = R_T @ (comp.stacked_xy - urdf_pos[:, None])
            px = pos_trans[0].reshape(comp.nx, comp.ny)
            py = pos_trans[1].reshape(comp.nx, comp.ny)
            comp.sdf_vals[body_i] = body.sdf(px, py)

            # SDF at u-staggered grid
            pos_u = R_T @ (body.stacked_xy_u - urdf_pos[:, None])
            comp.sdf_vals_u[body_i] = body.sdf(
                pos_u[0].reshape(body.nx, body.ny),
                pos_u[1].reshape(body.nx, body.ny),
            )

            # SDF at v-staggered grid
            pos_v = R_T @ (body.stacked_xy_v - urdf_pos[:, None])
            comp.sdf_vals_v[body_i] = body.sdf(
                pos_v[0].reshape(body.nx, body.ny),
                pos_v[1].reshape(body.nx, body.ny),
            )

            # Body velocities: v = v_lin + omega x (x - x_com)
            comp.u_vals[body_i] = lin_vel[0] - ang_vel * (comp.Y - com_pos[1])
            comp.v_vals[body_i] = lin_vel[1] + ang_vel * (comp.X - com_pos[0])

            comp.com_pos[body_i] = com_pos

            # Contour update
            body.cnt_update = R @ body.cnt + urdf_pos[:, None]
            body.r_com   = body.cnt_update - com_pos[:, None]
            body.com_pos = com_pos

        # ---- union reduction (argmin / gather) -----------------------
        idx = comp.sdf_vals.argmin(0).unsqueeze(0).expand(comp.sdf_vals.shape)
        comp.sdf_val = comp.sdf_vals.gather(0, idx)[0].reshape(fs.nx, fs.ny)

        idx_u = comp.sdf_vals_u.argmin(0).unsqueeze(0).expand(comp.sdf_vals_u.shape)
        comp.sdf_val_u = comp.sdf_vals_u.gather(0, idx_u)[0].reshape(fs.nx, fs.ny)
        comp.body_u    = comp.u_vals.gather(0, idx_u)[0].reshape(fs.nx, fs.ny)

        idx_v = comp.sdf_vals_v.argmin(0).unsqueeze(0).expand(comp.sdf_vals_v.shape)
        comp.sdf_val_v = comp.sdf_vals_v.gather(0, idx_v)[0].reshape(fs.nx, fs.ny)
        comp.body_v    = comp.v_vals.gather(0, idx_v)[0].reshape(fs.nx, fs.ny)

        # ---- set per-body SDF arrays needed by forces_method2 --------
        for body_i, body in enumerate(comp.bodies):
            body.sdf_u   = comp.sdf_vals_u[body_i]
            body.sdf_v   = comp.sdf_vals_v[body_i]
            body.sdf_val = comp.sdf_vals[body_i]

    # ------------------------------------------------------------------
    #  buoyancy initialisation
    # ------------------------------------------------------------------
    def _init_buoyancy_params(self, task, physics):
        """Precompute per-body mass & half-height for FARMS-style buoyancy.

        FARMS formula (from drag.pyx compute_buoyancy):
            if pos_z - height < surface:
                F_z = -rho_water * mass * gravity / density
                      * min((surface + height - pos_z) / (2*height), 1)
        """
        comp = self.fluid_solver.composite_body
        n = len(comp.bodies)

        self._buoy_mass   = np.zeros(n)
        self._buoy_height = np.zeros(n)

        for body_i in range(n):
            (animat_id, link_id) = comp.body_ids[body_i]
            ind = task.maps[animat_id]['sensors']['data2xfrc'][link_id]

            self._buoy_mass[body_i] = float(physics.model.body_mass[ind])

            # Maximum bounding-sphere radius among geoms attached to this body
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
    #  apply_forces: fluid -> MuJoCo xfrc_applied
    # ==================================================================
    def apply_forces(self, task, physics):
        fs = self.fluid_solver
        s  = self.force_scaling

        fx_fric  = s * fs.friction_force_lin_x.cpu().numpy()
        fy_fric  = s * fs.friction_force_lin_y.cpu().numpy()
        ang_fric = s * fs.friction_force_ang_z.cpu().numpy()

        fx_pres  = s * fs.pressure_force_x.cpu().numpy()
        fy_pres  = s * fs.pressure_force_y.cpu().numpy()
        ang_pres = s * fs.pressure_force_ang_z.cpu().numpy()

        # Lazy-init buoyancy parameters on first call
        if not self._buoyancy_initialized:
            self._init_buoyancy_params(task, physics)
            m_2d = self.rho_body * np.pi * self.radius**2
            print(f"[DIAG] buoy_mass(MuJoCo)={self._buoy_mass[0]:.6e}  "
                  f"m_2d(rho_b*pi*R²)={m_2d:.6e}  "
                  f"buoy_height={self._buoy_height[0]:.6e}  "
                  f"MuJoCo_weight={self._buoy_mass[0]*9.81:.6e}  "
                  f"2D_net_weight={(self.rho_body-self.rho_fluid)*np.pi*self.radius**2*9.81:.6e}")

        for body_i in range(len(fs.composite_body.bodies)):
            (animat_id, link_id) = fs.composite_body.body_ids[body_i]
            ind  = task.maps[animat_id]['sensors']['data2xfrc'][link_id]
            mass = self.data[animat_id].sensors.links.masses[link_id] * task.units.kilograms

            # ---- FARMS-style buoyancy (drag.pyx compute_buoyancy) ----
            buoy_mass   = self._buoy_mass[body_i]
            buoy_height = self._buoy_height[body_i]
            # com_pos[1] is the fluid y-coordinate = MuJoCo z
            pos_z = float(fs.composite_body.com_pos[body_i][1])

            buoyancy_z = 0.0
            if buoy_mass > 0 and buoy_height > 0 and pos_z - buoy_height < self.water_surface:
                frac = min((self.water_surface + buoy_height - pos_z) / (2.0 * buoy_height), 1.0)
                buoyancy_z = -self.rho_fluid * buoy_mass * self.gravity_z / self.rho_body #* frac


            # MuJoCo xfrc_applied: [fx, fy, fz, tx, ty, tz]
            # Fluid x -> MuJoCo x (index 0)
            # Fluid y -> MuJoCo z (index 2), buoyancy added here
            # 2D torque -> MuJoCo ty (index 4)
            physics.data.xfrc_applied[ind, 0] = (fx_fric[body_i] + fx_pres[body_i]) * task.units.newtons
            physics.data.xfrc_applied[ind, 2] = (fy_fric[body_i] + fy_pres[body_i] + buoyancy_z) * task.units.newtons
            physics.data.xfrc_applied[ind, 4] = (ang_fric[body_i] + ang_pres[body_i]) * task.units.newtons

            if self.iteration % 100 == 0:
                vel = self.cython2numpy(self.data[0].sensors.links.com_lin_velocities()[self.iteration, 0])
                vel_z = vel[2]
                # vel_z = physics.data.qvel[2]
                Re = abs(vel_z) * 2 * self.radius / (self.pars['solver']['nu'])
                print(f"it={self.iteration:6d}  "
                      f"Fvisc_z={fy_fric[body_i]:.4e}  Fpres_z={fy_pres[body_i]:.4e}  "
                      f"Fbuoy={buoyancy_z:.4e}  "
                      f"vel_z={vel_z:.4e}  Re={Re:.1f}")

    # ------------------------------------------------------------------
    #  BDIM2 helper: apply the meta-equation to a velocity component
    # ------------------------------------------------------------------
    def _bdim2(self, phi, mu0, m_m0, body_vel, mu1, nx, ny):
        fs = self.fluid_solver
        return (
            mu0 * phi
            + m_m0 * body_vel
            + mu1 * fs.normal_derivative(phi - body_vel, nx, ny)
        )

    # ==================================================================
    #  fluid_step: Heun (RK2) predictor-corrector with BDIM2
    # ==================================================================
    def fluid_step(self, u, v, p, timestep):
        fs = self.fluid_solver
        comp = fs.composite_body

        # No gravity in the fluid equations — with all-Neumann BCs the
        # Poisson solver cannot build a hydrostatic gradient, so adding
        # g here causes runaway velocity.  Buoyancy is handled explicitly
        # in apply_forces (FARMS style).

        # Variable-density projection coefficients.
        # Bug fix: use SEPARATE rho_blend for u- and v-staggered grids.
        # The u- and v-SDFs differ by up to h/2 near the body surface, so
        # mixing them (old: rho_blend from mu0_all_u applied to cv) gave
        # wrong pressure corrections in the wall-normal direction and caused
        # drag underestimation.
        rho_blend_u = self.rho_fluid * fs.mu0_all_u + self.rho_body * fs.m_m0_all_u
        rho_blend_v = self.rho_fluid * fs.mu0_all_v + self.rho_body * fs.m_m0_all_v
        ch = timestep * fs.mu0_all_u / rho_blend_u
        cv = timestep * fs.mu0_all_v / rho_blend_v

        def _project(up, vp, p_in, ch_, cv_):
            fs.div = fs.divergence(up, vp)
            rho_blend_cc = self.rho_fluid * fs.mu0_all + self.rho_body * fs.m_m0_all
            fft_coeff = timestep/rho_blend_cc
            if fs.poisson_method == "fft":
                p_out = fs.poisson_solverFFT.solve(fs.div / fft_coeff)
            elif fs.poisson_method in ("mgcg", "multigrid"):
                poisson_solve = (
                    fs.poisson_solver.solve_mgcg
                    if fs.poisson_method == "mgcg"
                    else fs.poisson_solver.solve_multigrid
                )
                p_out, _ = poisson_solve(
                    fs.div[1:-1, 1:-1], p_in,
                    ch=ch_[1:, 1:-1], cv=cv_[1:-1, 1:],
                )
            p_x, p_y = fs.gradient(p_out)
            return up - ch_ * p_x, vp - cv_ * p_y, p_out

        # ===== PREDICTOR =====
        (uprime, vprime) = fs.adv_diff_solver.solve(u, v)
        uprime = self._bdim2(
            uprime, fs.mu0_all_u, fs.m_m0_all_u, comp.body_u,
            fs.mu1_all_u, fs.normal_x_u, fs.normal_y_u,
        )
        vprime = self._bdim2(
            vprime, fs.mu0_all_v, fs.m_m0_all_v, comp.body_v,
            fs.mu1_all_v, fs.normal_x_v, fs.normal_y_v,
        )
        # Bug fix: set_BCs must come AFTER BDIM (not before) so that wall
        # boundary conditions are re-enforced on the BDIM-corrected field.
        fs.adv_diff_solver.set_BCs(uprime, vprime)
        uprime_bdim = uprime
        vprime_bdim = vprime
        u1, v1, p1 = _project(uprime, vprime, p, ch, cv)

        # return (u1, v1, p1)

        # ===== CORRECTOR =====
        # Re-enforce BCs on the projected u1 before advecting from it, so
        # that ghost-cell errors from the pressure correction do not propagate.
        fs.adv_diff_solver.set_BCs(u1, v1)
        (uprime2, vprime2) = fs.adv_diff_solver.solve(u1, v1)
        # Rebase increment from u^n (standard Heun rebasing)
        uprime2 = u + (uprime2 - u1)
        vprime2 = v + (vprime2 - v1)
        uprime2 = self._bdim2(
            uprime2, fs.mu0_all_u, fs.m_m0_all_u, comp.body_u,
            fs.mu1_all_u, fs.normal_x_u, fs.normal_y_u,
        )
        vprime2 = self._bdim2(
            vprime2, fs.mu0_all_v, fs.m_m0_all_v, comp.body_v,
            fs.mu1_all_v, fs.normal_x_v, fs.normal_y_v,
        )
        # Heun average of pre-projection BDIM velocities
        u_avg = 0.5 * (uprime_bdim + uprime2)
        v_avg = 0.5 * (vprime_bdim + vprime2)
        fs.adv_diff_solver.set_BCs(u_avg, v_avg)
        # Corrector projection weight = 0.5
        u_out, v_out, p_out = _project(u_avg, v_avg, p1, 0.5 * ch, 0.5 * cv)

        if fs.use_sponge:
            (u_out, v_out) = fs.apply_sponge_damping(u_out, v_out)

        return (u_out, v_out, p_out)

    # ==================================================================
    #  step: one full coupled fluid-body step
    # ==================================================================
    def step(self, task, physics):

        iteration = self.iteration
        timestep  = self.pars['solver']['dt']
        if iteration >= self.pars['solver']['nt']:
            return

        t  = iteration * timestep
        fs = self.fluid_solver
        comp = fs.composite_body

        if not self.terminate:

            # 1. Update SDF + body velocities from FARMS kinematics
            self.update(t, iteration, dt=timestep)

            # 2. Recompute mu / mask fields
            (fs.mu0_all, fs.mu1_all) = comp.mu_funcs(comp.sdf_val)
            fs.m_m0_all = 1 - fs.mu0_all
            (fs.normal_x, fs.normal_y) = comp.compute_normals(comp.sdf_val)

            (fs.mu0_all_u, fs.mu1_all_u) = comp.mu_funcs(comp.sdf_val_u)
            fs.m_m0_all_u = 1 - fs.mu0_all_u
            (fs.normal_x_u, fs.normal_y_u) = comp.compute_normals(comp.sdf_val_u)

            (fs.mu0_all_v, fs.mu1_all_v) = comp.mu_funcs(comp.sdf_val_v)
            fs.m_m0_all_v = 1 - fs.mu0_all_v
            (fs.normal_x_v, fs.normal_y_v) = comp.compute_normals(comp.sdf_val_v)

            # 3. Heun (RK2) step with BDIM2
            (u, v, p) = self.fluid_step(fs.u0, fs.v0, fs.p0, timestep)
            (fs.u0, fs.v0, fs.p0) = (u, v, p)

            # 5. Compute fluid forces
            if self.force_method == 'method1':
                fs.forces_method1(fs.u0, fs.v0, fs.p0, iteration)
            else:
                fs.forces_method2(fs.u0, fs.v0, fs.p0, iteration)

            # 6. Plotting / saving
            self.terminate = fs.plotting_debug(fs.u0, fs.v0, fs.p0, iteration)

            # 7. Apply forces to MuJoCo body
            self.apply_forces(task, physics)

        self.iteration += 1