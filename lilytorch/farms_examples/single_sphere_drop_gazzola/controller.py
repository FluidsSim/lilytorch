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
        self.mass      = np.pi * (self.radius ** 2) * self.rho_body
        self.inertia   = 0.5 * self.mass * (self.radius ** 2)
        print("Body mass: ", self.mass)
        print("Body inertia: ", self.inertia)

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

        for body_i in range(len(fs.composite_body.bodies)):
            (animat_id, link_id) = fs.composite_body.body_ids[body_i]
            ind  = task.maps[animat_id]['sensors']['data2xfrc'][link_id]
            mass = self.data[animat_id].sensors.links.masses[link_id] * task.units.kilograms

            # MuJoCo xfrc_applied: [fx, fy, fz, tx, ty, tz]
            # Fluid x -> MuJoCo x (index 0)
            # Fluid y -> MuJoCo z (index 2)
            # 2D torque -> MuJoCo ty (index 4)
            physics.data.xfrc_applied[ind, 0] = (fx_fric[body_i] + fx_pres[body_i]) * task.units.newtons
            physics.data.xfrc_applied[ind, 2] = (fy_fric[body_i] + fy_pres[body_i]) * task.units.newtons
            physics.data.xfrc_applied[ind, 4] = (ang_fric[body_i] + ang_pres[body_i]) * task.units.newtons

            print(physics.data.xfrc_applied[ind, [0, 2, 4]] / mass, physics.data.qvel[1])

    # ------------------------------------------------------------------
    #  BCs helper
    # ------------------------------------------------------------------
    @staticmethod
    def _set_bc(u, v):
        for i in [1, -1]:
            u[i, :] = 0;  u[:, i] = 0
            v[i, :] = 0;  v[:, i] = 0

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
    #  fluid_step: single Euler step with BDIM2
    # ==================================================================
    def fluid_step(self, u, v, p, timestep):
        fs = self.fluid_solver
        comp = fs.composite_body

        # Gravity body force (g = -9.81 in fluid y-direction)
        v = v - 9.81 * fs.dt

        # Advection-diffusion
        (uprime, vprime) = fs.adv_diff_solver.solve(u, v)
        self._set_bc(uprime, vprime)

        # BDIM2 meta-equation
        uprime = self._bdim2(
            uprime, fs.mu0_all_u, fs.m_m0_all_u, comp.body_u,
            fs.mu1_all_u, fs.normal_x_u, fs.normal_y_u,
        )
        vprime = self._bdim2(
            vprime, fs.mu0_all_v, fs.m_m0_all_v, comp.body_v,
            fs.mu1_all_v, fs.normal_x_v, fs.normal_y_v,
        )

        # Poisson solve (variable-density coefficients)
        # Use local variable — do NOT overwrite fs.rho, which must stay
        # as the scalar fluid density for forces_method1's stress tensor.
        rho_blend = self.rho_fluid * fs.mu0_all_u + self.rho_body * fs.m_m0_all_u
        ch = timestep * fs.mu0_all_u / rho_blend
        cv = timestep * fs.mu0_all_v / rho_blend

        fs.div = fs.divergence(uprime, vprime)
        p, _ = fs.poisson_solver.solve_multigrid(
            fs.div[1:-1, 1:-1],
            p,
            ch=ch[1:, 1:-1],
            cv=cv[1:-1, 1:],
        )

        # Pressure projection
        (p_x, p_y) = fs.gradient(p)
        u = uprime - ch * p_x
        v = vprime - cv * p_y

        return (u, v, p)

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

            # 3. Euler step with Brinkmann penalisation
            (u, v, p) = self.fluid_step(fs.u0, fs.v0, fs.p0, timestep)
            (fs.u0, fs.v0, fs.p0) = (u, v, p)

            # 4. Compute fluid forces
            fs.forces_method2(fs.u0, fs.v0, fs.p0, iteration)

            # 5. Plotting / saving
            self.terminate = fs.plotting_debug(fs.u0, fs.v0, fs.p0, iteration)

            # 6. Apply forces to MuJoCo body
            self.apply_forces(task, physics)

        self.iteration += 1
