"""Network controller"""

import numpy as np
from scipy.spatial.transform import Rotation
from lilytorch.src.solver import FluidSolver
import torch

class BDIMhandler():

    def __init__(self, yaml_file, data, physics, dtype=torch.float64):

        self.dtype = dtype
        if self.dtype == torch.float32:
            self.dtype_np = np.float32
        elif self.dtype == torch.float64:
            self.dtype_np = np.float64

        self.data = data # ExperimentData in FARMS
        self.iteration = 0
        self.terminate = False

        self.pars = yaml_file

        self.fluid_solver = FluidSolver(
            self.pars,
            dtype=self.dtype,
            costum_update=True,
            compute_forces=True
        )
        self.device = self.fluid_solver.device

        self.fluid_solver.composite_body.update = self.update  # modify the update rule

        self.fluid_solver.u0 = self.fluid_solver.adv_diff_solver.BC_values_u[1]*torch.ones((self.fluid_solver.nx,self.fluid_solver.ny),device=self.device,dtype=self.dtype)

        self.lin_axes = [0,1]
        self.ang_axes = [2]

        # self.force_scaling = np.array([np.diff(body.bb[2])[0] for body in self.fluid_solver.composite_body.bodies])

        self.force_scaling = 0.04 # general height of 1guilla border

        self.rho_fluid = self.pars['solver']['rho']
        self.rho_body  = 800.0
        # self.rho_body  = 1000.0

    def cython2numpy(self, array):
        return torch.from_numpy(np.array(array).astype(self.dtype_np)).to(self.device)


    def update(self, t, iteration, dt=1):

        # extract all experimental data used by the fluid solver
        com_poses = []
        urdf_poses = []
        Rs = []
        lin_vels = []
        ang_vels = []
        for data in self.data: # loop over experimental data
            com_poses.append( self.cython2numpy( data.sensors.links.com_positions()[iteration,:] ) [:,self.lin_axes] )
            urdf_poses.append( self.cython2numpy( data.sensors.links.urdf_positions()[iteration,:] ) [:,self.lin_axes] )
            Rs.append( self.cython2numpy(Rotation.from_quat(data.sensors.links.urdf_orientations()[iteration,:]).as_matrix().astype(self.dtype_np))[:,self.lin_axes, :][:, :, self.lin_axes] )
            lin_vels.append( self.cython2numpy( data.sensors.links.com_lin_velocities()[iteration,:] ) [:,self.lin_axes] )
            ang_vels.append( self.cython2numpy( [data.sensors.links.com_ang_velocity(iteration, link)[2] for link in range(len(data.sensors.links.names))]) )


        for body_i, body in enumerate(self.fluid_solver.composite_body.bodies[:]):

            (animat_id, link_id) = self.fluid_solver.composite_body.body_ids[body_i]
            data = self.data[animat_id]

            com_pos = com_poses[animat_id][link_id]
            urdf_pos = urdf_poses[animat_id][link_id]
            R = Rs[animat_id][link_id]
            lin_vel = lin_vels[animat_id][link_id]
            ang_vel = ang_vels[animat_id][link_id]
            # compute sdf values at the body points
            pos_trans = R.T@(self.fluid_solver.composite_body.stacked_xy-urdf_pos[:,None])
            newpos_u = pos_trans[0].reshape(self.fluid_solver.composite_body.nx, self.fluid_solver.composite_body.ny)
            newpos_v = pos_trans[1].reshape(self.fluid_solver.composite_body.nx, self.fluid_solver.composite_body.ny)
            self.fluid_solver.composite_body.sdf_vals[body_i]=body.sdf(newpos_u, newpos_v)

            # compute sdf values at the staggered grid
            pos_trans_u = R.T@(body.stacked_xy_u-urdf_pos[:,None])
            newpos_u = pos_trans_u[0].reshape(body.nx, body.ny)
            newpos_v = pos_trans_u[1].reshape(body.nx, body.ny)
            self.fluid_solver.composite_body.sdf_vals_u[body_i]=body.sdf(newpos_u, newpos_v)

            pos_trans_v = R.T@(body.stacked_xy_v-urdf_pos[:,None])
            newpos_u = pos_trans_v[0].reshape(body.nx, body.ny)
            newpos_v = pos_trans_v[1].reshape(body.nx, body.ny)
            self.fluid_solver.composite_body.sdf_vals_v[body_i]=body.sdf(newpos_u, newpos_v)

            # compute body velocities v = v_lin_com + <v_ang_com, x-x_com>
            self.fluid_solver.composite_body.u_vals[body_i]=lin_vel[0]-ang_vel*(self.fluid_solver.composite_body.Yu_stag-com_pos[1])
            self.fluid_solver.composite_body.v_vals[body_i]=lin_vel[1]+ang_vel*(self.fluid_solver.composite_body.Xv_stag-com_pos[0])

            # store com positions for fluid->body force computation
            self.fluid_solver.composite_body.com_pos[body_i]=com_pos

            # update the contour position
            body.cnt_update = R @ body.cnt+urdf_pos[:,None]

            # # compute the mask for the contour points
            # x_cnt=body.cnt_update[0]
            # y_cnt=body.cnt_update[1]
            # if link_id==0:
            #     body_p=self.fluid_solver.composite_body.bodies[body_i+1]
            #     pos_trans = Rs[animat_id][link_id+1].T@(torch.stack((x_cnt,y_cnt))-urdf_poses[animat_id][link_id+1][:,None])
            #     sdf_p = body_p.sdf(pos_trans[0],pos_trans[1])-body.h
            #     mask=(sdf_p >= 0)
            # elif link_id==urdf_poses[animat_id].shape[0]-1:
            #     body_m=self.fluid_solver.composite_body.bodies[body_i-1]
            #     pos_trans = Rs[animat_id][link_id-1].T@(torch.stack((x_cnt,y_cnt))-urdf_poses[animat_id][link_id-1][:,None])
            #     sdf_m = body_m.sdf(pos_trans[0],pos_trans[1])-body.h
            #     mask=(sdf_m >= 0)
            # else:
            #     body_m=self.fluid_solver.composite_body.bodies[body_i-1]
            #     pos_trans_m = Rs[animat_id][link_id-1].T@(torch.stack((x_cnt,y_cnt))-urdf_poses[animat_id][link_id-1][:,None])
            #     body_p=self.fluid_solver.composite_body.bodies[body_i+1]
            #     pos_trans_p = Rs[animat_id][link_id+1].T@(torch.stack((x_cnt,y_cnt))-urdf_poses[animat_id][link_id+1][:,None])
            #     sdf_m = body_m.sdf(pos_trans_m[0],pos_trans_m[1])-body.h
            #     sdf_p = body_p.sdf(pos_trans_p[0],pos_trans_p[1])-body.h
            #     mask=(sdf_m >= 0) & (sdf_p >= 0)
            # body.mask=mask

            # body.cnt_u = lin_vel[0]-ang_vel*(y_cnt-com_pos[1])
            # body.cnt_v = lin_vel[1]+ang_vel*(x_cnt-com_pos[0])
            body.r_com = body.cnt_update-com_pos[:,None] # moment arm
            body.com_pos=com_pos


        idx=self.fluid_solver.composite_body.sdf_vals.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals.shape)
        self.fluid_solver.composite_body.sdf_val=self.fluid_solver.composite_body.sdf_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny) #-self.fluid_solver.composite_body.suit

        idx_u=self.fluid_solver.composite_body.sdf_vals_u.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals_u.shape)
        self.fluid_solver.composite_body.sdf_val_u=self.fluid_solver.composite_body.sdf_vals_u.gather(0,idx_u)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny) #-self.fluid_solver.composite_body.suit
        self.fluid_solver.composite_body.body_u=self.fluid_solver.composite_body.u_vals.gather(0,idx_u)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)

        idx_v=self.fluid_solver.composite_body.sdf_vals_v.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals_v.shape)
        self.fluid_solver.composite_body.sdf_val_v=self.fluid_solver.composite_body.sdf_vals_v.gather(0,idx_v)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny) #-self.fluid_solver.composite_body.suit
        self.fluid_solver.composite_body.body_v=self.fluid_solver.composite_body.v_vals.gather(0,idx_v)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)


    def apply_forces(self, task, physics):

        self.friction_force_lin_x = self.force_scaling*(self.fluid_solver.friction_force_lin_x).cpu().numpy()
        self.friction_force_lin_y = self.force_scaling*(self.fluid_solver.friction_force_lin_y).cpu().numpy()
        self.friction_force_ang_z = self.force_scaling*(self.fluid_solver.friction_force_ang_z).cpu().numpy()

        self.pressure_force_x     = self.force_scaling*(self.fluid_solver.pressure_force_x).cpu().numpy()
        self.pressure_force_y     = self.force_scaling*(self.fluid_solver.pressure_force_y).cpu().numpy()
        self.pressure_force_ang_z = self.force_scaling*(self.fluid_solver.pressure_force_ang_z).cpu().numpy()

        for body_i, body in enumerate(self.fluid_solver.composite_body.bodies[:]):

            (animat_id, link_id) = self.fluid_solver.composite_body.body_ids[body_i]

            ind_task= task.maps[animat_id]['sensors']['data2xfrc'][link_id]

            physics.data.xfrc_applied[ind_task, 0] = (self.friction_force_lin_x[body_i] + self.pressure_force_x[body_i]) * task.units.newtons
            physics.data.xfrc_applied[ind_task, 1] = (self.friction_force_lin_y[body_i] + self.pressure_force_y[body_i]) * task.units.newtons
            physics.data.xfrc_applied[ind_task, 5] = (self.friction_force_ang_z[body_i] + self.pressure_force_ang_z[body_i]) * task.units.newtons


    def fluid_step(self,u,v,p,timestep):

        (uprime,vprime)  = self.fluid_solver.adv_diff_solver.solve(u,v)

        self.fluid_solver.adv_diff_solver.set_BCs(uprime,vprime)

        # uprime = (uprime+self.fluid_solver.brinkmann_k*timestep*self.fluid_solver.m_m0_all_u*self.fluid_solver.composite_body.body_u)/ \
        #          (1+self.fluid_solver.brinkmann_k*timestep*self.fluid_solver.m_m0_all_u)
        # vprime = (vprime+self.fluid_solver.brinkmann_k*timestep*self.fluid_solver.m_m0_all_v*self.fluid_solver.composite_body.body_v)/ \
        #          (1+self.fluid_solver.brinkmann_k*timestep*self.fluid_solver.m_m0_all_v)


        # ====== STEP 1 =====
        uprime = self.fluid_solver.mu0_all_u*uprime + self.fluid_solver.m_m0_all_u*self.fluid_solver.composite_body.body_u \
            + self.fluid_solver.mu1_all_u*self.fluid_solver.normal_derivative(uprime-self.fluid_solver.composite_body.body_u,self.fluid_solver.normal_x_u,self.fluid_solver.normal_y_u)
        vprime = self.fluid_solver.mu0_all_v*vprime + self.fluid_solver.m_m0_all_v*self.fluid_solver.composite_body.body_v \
            + self.fluid_solver.mu1_all_v*self.fluid_solver.normal_derivative(vprime-self.fluid_solver.composite_body.body_v,self.fluid_solver.normal_x_v,self.fluid_solver.normal_y_v)


        # # for general deforming bodies
        # self.div_body = torch.zeros_like(u)
        # self.div_body[1:-1,1:-1] = self.m_m0_all_u[1:-1,1:-1]*(self.composite_body.body_u[2:, 1:-1] - self.composite_body.body_u[1:-1, 1:-1]) / self.h \
        #                             + self.m_m0_all_v[1:-1,1:-1]*(self.composite_body.body_v[1:-1, 2:] - self.composite_body.body_v[1:-1, 1:-1]) / self.h
        # self.div  = self.divergence(uprime,vprime) - self.div_body

        # for general deforming bodies
        self.fluid_solver.div  = self.fluid_solver.divergence(uprime,vprime)

        # ====== STEP 2: variable-coefficient Poisson solve ======
        # c = dt/rho at each face.  rho is the BDIM mixed density, already
        # computed on the u-face grid (rho_u) and v-face grid (rho_v).
        rho_u = (self.rho_fluid * self.fluid_solver.mu0_all_u
                 + self.rho_body  * self.fluid_solver.m_m0_all_u)   # shape (nx, ny)
        rho_v = (self.rho_fluid * self.fluid_solver.mu0_all_v
                 + self.rho_body  * self.fluid_solver.m_m0_all_v)   # shape (nx, ny)

        # c = dt/rho — variable density but NO μ₀ factor.
        # Adding μ₀ makes J→0 at the body boundary, causing multigrid blow-up.
        # LilyPad's μ₀·dt applies to a normalised single-fluid (rho=1) system
        # and is not equivalent here. Body pressure is zeroed after the solve.
        ch_full = timestep / rho_u
        cv_full = timestep / rho_v

        # The Poisson solver works on the interior block f = div[1:-1,1:-1] of
        # shape (nx-2, ny-2).  It needs:
        #   ch  shape (nx-1, ny-2)  — faces between interior columns
        #   cv  shape (nx-2, ny-1)  — faces between interior rows
        p, _ = self.fluid_solver.poisson_solver.solve_multigrid(
            self.fluid_solver.div[1:-1, 1:-1],
            torch.zeros_like(p),
            (timestep/self.rho_body)*torch.ones_like(self.fluid_solver.div),  # c arg unused when ch/cv given
            ch=ch_full[1:,  1:-1],   # (nx-1, ny-2)
            cv=cv_full[1:-1, 1:],    # (nx-2, ny-1)
        )

        # ====== projection step ======
        # Use the same face coefficients (full grid) for the projection.
        (p_x, p_y) = self.fluid_solver.gradient(p)
        u = uprime - ch_full * p_x
        v = vprime - cv_full * p_y

        self.fluid_solver.adv_diff_solver.set_BCs(u,v)


        return (u,v,p)





    def step(self, task, physics):


        iteration = self.iteration
        timestep  = self.pars['solver']['dt']
        if iteration>=self.pars['solver']['nt']:
            return

        t = iteration*timestep

        # if iteration==0:
        #     (u,v,p) = (self.fluid_solver.u0, self.fluid_solver.v0, self.fluid_solver.p0)

        # === stepping the fluid solver ===
        if not self.terminate:

            # (
            # self.fluid_solver.u0,
            # self.fluid_solver.v0,
            # self.fluid_solver.p0,
            # self.terminate,
            # ) = self.fluid_stepper(
            #     self.fluid_solver.u0,
            #     self.fluid_solver.v0,
            #     self.fluid_solver.p0,
            #     iteration,
            #     t
            # )

            # update sdf_properties
            self.update(t, iteration, dt=timestep)

            (self.fluid_solver.mu0_all, self.fluid_solver.mu1_all)         = self.fluid_solver.composite_body.mu_funcs(self.fluid_solver.composite_body.sdf_val)
            self.fluid_solver.m_m0_all                                     = (1-self.fluid_solver.mu0_all)
            (_, self.fluid_solver.normal_x, self.fluid_solver.normal_y, _) = self.fluid_solver.composite_body.compute_sdf_properties(self.fluid_solver.composite_body.sdf_val)

            (self.fluid_solver.mu0_all_u, self.fluid_solver.mu1_all_u)         = self.fluid_solver.composite_body.mu_funcs(self.fluid_solver.composite_body.sdf_val_u)
            self.fluid_solver.m_m0_all_u                                       = (1-self.fluid_solver.mu0_all_u)
            (_, self.fluid_solver.normal_x_u, self.fluid_solver.normal_y_u, _) = self.fluid_solver.composite_body.compute_sdf_properties(self.fluid_solver.composite_body.sdf_val_u)

            (self.fluid_solver.mu0_all_v, self.fluid_solver.mu1_all_v)         = self.fluid_solver.composite_body.mu_funcs(self.fluid_solver.composite_body.sdf_val_v)
            self.fluid_solver.m_m0_all_v                                       = (1-self.fluid_solver.mu0_all_v)
            (_, self.fluid_solver.normal_x_v, self.fluid_solver.normal_y_v, _) = self.fluid_solver.composite_body.compute_sdf_properties(self.fluid_solver.composite_body.sdf_val_v)

            (u,v,p) = (self.fluid_solver.u0, self.fluid_solver.v0, self.fluid_solver.p0)

            # (u1,v1,p1) = self.fluid_step(u,v,p,timestep)
            # (u2,v2,p2) = self.fluid_step(u1,v1,p1,timestep)
            # u=0.5*(u+u2)
            # v=0.5*(v+v2)
            # p=p2

            (u,v,p) = self.fluid_step(self.fluid_solver.u0, self.fluid_solver.v0, self.fluid_solver.p0,timestep)

            # p = torch.where(self.fluid_solver.composite_body.sdf_val<0,0,p)

            (self.fluid_solver.u0, self.fluid_solver.v0, self.fluid_solver.p0) = (u,v,p)

            # compute fluid forces on the body
            self.fluid_solver.forces_method2(self.fluid_solver.u0, self.fluid_solver.v0, self.fluid_solver.p0, iteration)

            self.terminate = self.fluid_solver.plotting_debug(self.fluid_solver.u0, self.fluid_solver.v0, self.fluid_solver.p0, iteration)

            self.apply_forces(task,physics)

        self.iteration+=1
