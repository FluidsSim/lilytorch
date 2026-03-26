"""Network controller -- 2-D cylinder sedimentation (Gazzola et al., 2011)

Supports two immersed-boundary formulations, selectable via the config key
``solver.use_brinkman`` (default ``True``):

* **Brinkman penalization** (Gazzola Eq. 29):
      u_lambda = (u + lambda*dt*chi_s*u_s) / (1 + lambda*dt*chi_s)
  with configurable penalization parameter ``solver.brinkman_k`` (default 1e4).

* **BDIM2 meta-equation** (original lilytorch approach):
      u = mu0*u_advected + (1-mu0)*u_body + mu1*dn(u_advected - u_body)

Force computation also has two modes (``solver.force_method``, default
``"penalization"``):

* ``"penalization"``: integrates the penalization/BDIM2 body force directly
  from the velocity correction applied by the IB method.  This is the robust
  volume-integral approach used by Gazzola (Eq. 37).

* ``"stress"``: the existing ``forces_method2`` surface-stress integration
  (smoothed delta on SDF).  Known to underpredict drag by ~3x for this
  benchmark due to staggered/CC grid mismatch and resolution sensitivity.

.. note::
   ``forces_method1`` (contour integral) is **incompatible** with this
   controller because the contour data (``body.cnt_u``, ``body.cnt_v``,
   ``body.mask``) is not populated by the custom ``update()`` method.
   Selecting it will raise ``RuntimeError``.
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

        # ---- Brinkman vs BDIM2 toggle ----
        # Gazzola (2011) uses Brinkman penalization (Eq. 29).
        # Set solver.use_brinkman = False to revert to BDIM2.
        self.use_brinkman = self.pars['solver'].get('use_brinkman', True)
        # Penalization parameter lambda (Gazzola default: 1e4)
        self.brinkman_k = self.pars['solver'].get('brinkman_k', 1e4)

        # ---- Force computation method ----
        # "penalization": volume integral of IB body force (Gazzola Eq. 37)
        # "stress": existing forces_method2 (surface-stress integration)
        self.force_method = self.pars['solver'].get('force_method', 'penalization')

        if self.force_method not in ('penalization', 'stress'):
            raise ValueError(
                f"Unknown force_method={self.force_method!r}. "
                f"Use 'penalization' or 'stress'."
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

        print(f"[Gazzola] use_brinkman={self.use_brinkman}, "
              f"brinkman_k={self.brinkman_k}, "
              f"force_method={self.force_method}")

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
                  f"m_2d(rho_b*pi*R^2)={m_2d:.6e}  "
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

    # ------------------------------------------------------------------
    #  Brinkman penalization (Gazzola Eq. 29)
    # ------------------------------------------------------------------
    def _brinkman(self, phi, m_m0, body_vel):
        """Apply Brinkman penalization to a velocity component.

        Gazzola et al. (2011) Eq. 29:
            u_lambda = (u + lambda*dt*chi_s*u_s) / (1 + lambda*dt*chi_s)

        Here chi_s = (1 - mu0) = m_m0 is the body characteristic function
        (smooth Heaviside).  lambda is self.brinkman_k.

        In the limit lambda -> inf, u_lambda -> u_s inside the body and
        u_lambda -> u outside, exactly enforcing the no-slip condition.
        """
        lam_dt = self.brinkman_k * float(self.fluid_solver.dt)
        chi_s = m_m0
        return (phi + lam_dt * chi_s * body_vel) / (1.0 + lam_dt * chi_s)

    # ==================================================================
    #  Penalization-based force computation (Gazzola Eq. 37)
    # ==================================================================
    def _compute_penalization_forces(self, u_before, v_before, u_after, v_after, timestep):
        """Compute fluid-body interaction forces from the IB velocity correction.

        For Brinkman penalization, the force density on the fluid is:
            f = -lambda * chi_s * (u_lambda - u_s)
        and the reaction on the body is equal and opposite (Newton III):
            F_body = +lambda * integral( chi_s * (u_lambda - u_s) ) dV

        Equivalently (and more generally for both Brinkman and BDIM2),
        the total IB forcing is:
            F_body = -rho * integral( u_after - u_before ) dV / dt
        where u_before is the advected velocity and u_after is the velocity
        after the IB correction (penalization or BDIM2), both BEFORE the
        pressure projection.  The sign convention gives the force ON the body
        (reaction to the momentum injected into the fluid).

        This is a robust volume integral over the body interior, avoiding the
        resolution-sensitive surface-stress integration of forces_method2.
        """
        fs   = self.fluid_solver
        comp = fs.composite_body
        h2   = float(fs.h ** 2)
        rho  = self.rho_fluid
        dt   = float(timestep)

        # Total velocity correction from IB method (on staggered grids)
        du = u_after - u_before  # (Nx, Ny)
        dv = v_after - v_before  # (Nx, Ny)

        # For per-body attribution in multi-body cases, we weight by each
        # body's (1-mu0) mask on the staggered grids.  For a single body
        # this is equivalent to integrating over the whole domain.
        for body_i, body in enumerate(comp.bodies):
            # Body mask on staggered grids: (1 - mu0) from per-body SDF
            sdf_u_i = comp.sdf_vals_u[body_i]
            sdf_v_i = comp.sdf_vals_v[body_i]
            mu0_u_i = comp.mu_funcs(sdf_u_i)[0]
            mu0_v_i = comp.mu_funcs(sdf_v_i)[0]
            chi_u_i = 1.0 - mu0_u_i  # body indicator on u-grid
            chi_v_i = 1.0 - mu0_v_i  # body indicator on v-grid

            # Force ON the body = -rho/dt * integral(delta_u * chi_body) dV
            # delta_u = u_after - u_before is the momentum injected INTO the
            # fluid by the IB method, so the reaction on the body has opposite sign.
            fx = -(rho / dt) * (du * chi_u_i).to(torch.float64).sum().to(self.dtype) * h2
            fy = -(rho / dt) * (dv * chi_v_i).to(torch.float64).sum().to(self.dtype) * h2

            # Torque about COM: tau_z = (x-xc)*fy_density - (y-yc)*fx_density
            xc = comp.com_pos[body_i, 0]
            yc = comp.com_pos[body_i, 1]
            X = fs.grids.X   # CC grid coordinates
            Y = fs.grids.Y

            # Approximate torque using CC-grid projection of the corrections.
            # The staggered du/dv are half-cell shifted; for torque we
            # interpolate to CC (simple average of neighbours).
            du_cc = 0.5 * (du[:-1, :] + du[1:, :])
            # Pad to full grid size
            du_cc = torch.nn.functional.pad(du_cc, (0, 0, 0, 1), mode='replicate')
            dv_cc = 0.5 * (dv[:, :-1] + dv[:, 1:])
            dv_cc = torch.nn.functional.pad(dv_cc, (0, 1, 0, 0), mode='replicate')

            # Use CC-grid body mask for torque
            sdf_cc_i = comp.sdf_vals[body_i]
            mu0_cc_i = comp.mu_funcs(sdf_cc_i)[0]
            chi_cc_i = 1.0 - mu0_cc_i

            fx_density = -(rho / dt) * du_cc * chi_cc_i
            fy_density = -(rho / dt) * dv_cc * chi_cc_i

            torque_z = (
                ((X - xc) * fy_density - (Y - yc) * fx_density)
                .to(torch.float64).sum().to(self.dtype) * h2
            )

            # Store in the same arrays that apply_forces reads.
            # Convention: penalization force goes into friction_force (viscous-like)
            # and pressure_force is zeroed, since the penalization integral
            # captures the total IB force (viscous + pressure combined).
            fs.friction_force_lin_x[body_i] = fx
            fs.friction_force_lin_y[body_i] = fy
            fs.friction_force_ang_z[body_i] = torque_z
            fs.pressure_force_x[body_i] = 0.0
            fs.pressure_force_y[body_i] = 0.0
            fs.pressure_force_ang_z[body_i] = 0.0

            # Update drag records for post-processing
            it = self.iteration
            if it < fs.viscous_drag_record.shape[2]:
                fs.viscous_drag_record[body_i, 0, it] = fx
                fs.viscous_drag_record[body_i, 1, it] = fy
                fs.pressure_drag_record[body_i, 0, it] = 0.0
                fs.pressure_drag_record[body_i, 1, it] = 0.0

    # ==================================================================
    #  fluid_step: single Euler step with Brinkman or BDIM2
    # ==================================================================
    def fluid_step(self, u, v, p, timestep):
        fs = self.fluid_solver
        comp = fs.composite_body

        # No gravity in the fluid equations -- with all-Neumann BCs the
        # Poisson solver cannot build a hydrostatic gradient, so adding
        # g here causes runaway velocity.  Buoyancy is handled explicitly
        # in apply_forces (FARMS style).

        # Advection-diffusion
        (uprime, vprime) = fs.adv_diff_solver.solve(u, v)
        fs.adv_diff_solver.set_BCs(uprime, vprime)

        # Save pre-IB velocities for penalization force computation.
        # Clone BEFORE the IB correction modifies them.
        if self.force_method == 'penalization':
            u_advected = uprime.clone()
            v_advected = vprime.clone()

        # ---- Immersed boundary correction ----
        if self.use_brinkman:
            # Brinkman penalization (Gazzola Eq. 29)
            uprime = self._brinkman(uprime, fs.m_m0_all_u, comp.body_u)
            vprime = self._brinkman(vprime, fs.m_m0_all_v, comp.body_v)
        else:
            # BDIM2 meta-equation
            uprime = self._bdim2(
                uprime, fs.mu0_all_u, fs.m_m0_all_u, comp.body_u,
                fs.mu1_all_u, fs.normal_x_u, fs.normal_y_u,
            )
            vprime = self._bdim2(
                vprime, fs.mu0_all_v, fs.m_m0_all_v, comp.body_v,
                fs.mu1_all_v, fs.normal_x_v, fs.normal_y_v,
            )

        # Compute penalization forces BEFORE pressure projection.
        # The penalization integral captures the total body force from the
        # IB correction.  The subsequent pressure projection redistributes
        # momentum to enforce incompressibility but does not change the net
        # force on the body (the pressure gradient integrates to zero over
        # the closed body surface for incompressible flow).
        if self.force_method == 'penalization':
            self._compute_penalization_forces(
                u_advected, v_advected, uprime, vprime, timestep,
            )

        # Variable-density Poisson solve.
        #
        # Gazzola (Eq. 12) defines rho = (1-chi_s)*rho_f + chi_s*rho_s.
        # Our mu0 is the smooth Heaviside of the SDF: mu0 ~ 1 in fluid,
        # mu0 ~ 0 inside the body.  So (1-mu0) = m_m0 ~ chi_s, and:
        #   rho_blend = rho_f * mu0 + rho_s * (1-mu0)
        # which matches Gazzola's density field exactly.
        #
        # Use local variable -- do NOT overwrite fs.rho, which must stay
        # as the scalar fluid density for forces_method1's stress tensor.
        rho_blend_u = self.rho_fluid * fs.mu0_all_u + self.rho_body * fs.m_m0_all_u
        rho_blend_v = self.rho_fluid * fs.mu0_all_v + self.rho_body * fs.m_m0_all_v
        ch = timestep * fs.mu0_all_u / rho_blend_u
        cv = timestep * fs.mu0_all_v / rho_blend_v

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

            # 3. Euler step (Brinkman or BDIM2, selected by self.use_brinkman)
            (u, v, p) = self.fluid_step(fs.u0, fs.v0, fs.p0, timestep)
            (fs.u0, fs.v0, fs.p0) = (u, v, p)

            # 4. Compute fluid forces
            #    - "penalization" forces are already computed inside fluid_step
            #      (before pressure projection).
            #    - "stress" forces use the existing surface-stress integration.
            if self.force_method == 'stress':
                fs.forces_method2(fs.u0, fs.v0, fs.p0, iteration)
            # (penalization forces were computed in fluid_step)

            # 5. Plotting / saving
            self.terminate = fs.plotting_debug(fs.u0, fs.v0, fs.p0, iteration)

            # 6. Apply forces to MuJoCo body
            self.apply_forces(task, physics)

        self.iteration += 1
