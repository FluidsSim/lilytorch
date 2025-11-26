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

        # self.fluid_solver = FluidSolver(
        #     self.pars,
        #     dtype=self.dtype,
        #     costum_update=True,
        #     compute_forces=True
        # )
        # self.device = self.fluid_solver.device

        # self.fluid_solver.composite_body.update = self.update  # modify the update rule
        # self.fluid_stepper = self.fluid_solver.step_

        # self.force_scaling = 1.0

        # # initialize solref
        # physics.model.geom_solref[:,0]= 0.001
        # physics.model.geom_solref[:,1]= 0.5


    def cython2numpy(self, array):
        return torch.from_numpy(np.array(array).astype(self.dtype_np)).to(self.device)


    def update(self, t, iteration, dt=1):

        for body_i, body in enumerate(self.fluid_solver.composite_body.bodies[:]):

            (animat_id, link_id) = self.fluid_solver.composite_body.body_ids[body_i]
            data = self.data[animat_id]

            com_pos = self.cython2numpy( data.sensors.links.com_positions()[iteration,link_id] ) [self.lin_axes]
            urdf_pos = self.cython2numpy( data.sensors.links.urdf_positions()[iteration,link_id] ) [self.lin_axes]
            R = self.cython2numpy(Rotation.from_quat(data.sensors.links.urdf_orientations()[iteration,link_id]).as_matrix().astype(self.dtype_np))[self.lin_axes, :][:, self.lin_axes]
            lin_vel = self.cython2numpy( data.sensors.links.com_lin_velocities()[iteration,link_id] ) [self.lin_axes]
            ang_vel = self.cython2numpy( data.sensors.links.com_ang_velocity(iteration, link_id) )[self.ang_axes]

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
            self.fluid_solver.composite_body.u_vals[body_i]=lin_vel[0]-ang_vel*(self.fluid_solver.composite_body.Y-com_pos[1])
            self.fluid_solver.composite_body.v_vals[body_i]=lin_vel[1]+ang_vel*(self.fluid_solver.composite_body.X-com_pos[0])

            # store com positions for fluid->body force computation
            self.fluid_solver.composite_body.com_pos[body_i]=com_pos

            # update the contour position
            body.cnt_update = R @ body.cnt+urdf_pos[:,None]

            # compute the mask for the contour points
            # x_cnt=body.cnt_update[0]
            # y_cnt=body.cnt_update[1]
            # if i==0:
            #     body_p=self.fluid_solver.composite_body.bodies[i+1]
            #     pos_trans = r[i+1].T@(torch.stack((x_cnt,y_cnt))-urdf_poses[i+1][:,None])
            #     sdf_p = body_p.sdf_interp(pos_trans[0],pos_trans[1])
            #     mask=(sdf_p >= 0)
            # elif i==self.n_bodies-1:
            #     body_m=self.fluid_solver.composite_body.bodies[i-1]
            #     pos_trans = r[i-1].T@(torch.stack((x_cnt,y_cnt))-urdf_poses[i-1][:,None])
            #     sdf_m = body_m.sdf_interp(pos_trans[0],pos_trans[1])
            #     mask=(sdf_m >= 0)
            # else:
            #     body_m=self.fluid_solver.composite_body.bodies[i-1]
            #     pos_trans_m = r[i-1].T@(torch.stack((x_cnt,y_cnt))-urdf_poses[i-1][:,None])
            #     body_p=self.fluid_solver.composite_body.bodies[i+1]
            #     pos_trans_p = r[i+1].T@(torch.stack((x_cnt,y_cnt))-urdf_poses[i+1][:,None])
            #     sdf_m = body_m.sdf_interp(pos_trans_m[0],pos_trans_m[1])
            #     sdf_p = body_p.sdf_interp(pos_trans_p[0],pos_trans_p[1])
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

        # physics.data.qvel[1]=-0.05
        # physics.data.qvel[4]=-0.05
        # physics.data.qvel[7]=-0.05

        self.friction_force_lin_x = self.force_scaling*(self.fluid_solver.friction_force_lin_x).cpu().numpy()
        self.friction_force_lin_y = self.force_scaling*(self.fluid_solver.friction_force_lin_y).cpu().numpy()
        self.friction_force_ang_z = self.force_scaling*(self.fluid_solver.friction_force_ang_z).cpu().numpy()

        # self.friction_force_lin_x = 1000*self.force_scaling*(self.fluid_solver.friction_force_lin_x).cpu().numpy()
        # self.friction_force_lin_y = 1000*self.force_scaling*(self.fluid_solver.friction_force_lin_y).cpu().numpy()
        # self.friction_force_ang_z = 1000*self.force_scaling*(self.fluid_solver.friction_force_ang_z).cpu().numpy()

        self.pressure_force_x     = self.force_scaling*(self.fluid_solver.pressure_force_x).cpu().numpy()
        self.pressure_force_y     = self.force_scaling*(self.fluid_solver.pressure_force_y).cpu().numpy()
        self.pressure_force_ang_z = self.force_scaling*(self.fluid_solver.pressure_force_ang_z).cpu().numpy()




        for body_i, body in enumerate(self.fluid_solver.composite_body.bodies[:]):

            (animat_id, link_id) = self.fluid_solver.composite_body.body_ids[body_i]

            ind_task= task.maps[animat_id]['sensors']['data2xfrc'][link_id]

            # timestep=task.timestep
            # # mass= self.data[animat_id].sensors.links.masses[link_id] * task.units.kilograms # mass
            # # inertia=physics.model.body_inertia[ind_task][1] # rotation inertia around y
            # mass=self.mass
            # inertia=self.inertia

            # fx = (self.friction_force_lin_x[body_i] + self.pressure_force_x[body_i]) * task.units.newtons * task.units.meters
            # fz = (self.friction_force_lin_y[body_i] + self.pressure_force_y[body_i]) * task.units.newtons * task.units.meters
            # g = -9.81
            # torque =(self.friction_force_ang_z[body_i] + self.pressure_force_ang_z[body_i]) * task.units.newtons * task.units.meters

            # physics.data.qvel[0]=self.vel_x0+(fx/mass) * timestep
            # physics.data.qvel[1]=self.vel_z0+(fz/mass+g) * timestep
            # physics.data.qvel[2]=self.ang_y0+(torque/inertia) * timestep

            # self.vel_x0 = physics.data.qvel[0].copy()
            # self.vel_z0 = physics.data.qvel[1].copy()
            # self.ang_y0 = physics.data.qvel[2].copy()

            # print("Friction F_z: {}, Pressure P_z: {}, qvel: {}".format(fz/mass, self.pressure_force_y[body_i]/mass, physics.data.qvel))


            # print(physics.data.qvel[1])
            # physics.data.qvel[0]= (torque/inertia) * timestep


            mass=self.data[animat_id].sensors.links.masses[link_id] * task.units.kilograms

            physics.data.xfrc_applied[ind_task, 0] = (self.friction_force_lin_x[body_i] + self.pressure_force_x[body_i]) * task.units.newtons
            physics.data.xfrc_applied[ind_task, 2] = (self.friction_force_lin_y[body_i] + self.pressure_force_y[body_i]) * task.units.newtons
            physics.data.xfrc_applied[ind_task, 4] = (self.friction_force_ang_z[body_i] + self.pressure_force_ang_z[body_i]) * task.units.newtons


            print(physics.data.xfrc_applied[ind_task, [0,2,4]]/mass,physics.data.qvel[1])

            # physics.data.qvel[1]=-0.05

    def fluid_step(self,u,v,p,timestep):

        v          -= 9.81*self.fluid_solver.dt
        (uprime,vprime)  = self.fluid_solver.adv_diff_solver.solve(u,v)

        self.set_BC(uprime,vprime)

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


        # coeff = timestep/self.fluid_solver.rho
        # p=self.fluid_solver.poisson_solverFFT.solve(self.fluid_solver.div/coeff)
        # (p_x, p_y) = self.fluid_solver.gradient(p)
        # (u,v)=(uprime-coeff*p_x, vprime-coeff*p_y)

        # (c, _) = self.composite_body.mu_funcs(self.composite_body.sdf_val)
        c = torch.ones_like(u)
        # ch = (c[1:,1:-1]+c[:-1,1:-1])/2
        # cv = (c[1:-1,1:]+c[1:-1,:-1])/2
        coeff = timestep/self.fluid_solver.rho
        ch = timestep * self.fluid_solver.mu0_all_u / self.fluid_solver.rho
        cv = timestep * self.fluid_solver.mu0_all_v / self.fluid_solver.rho
        p, _    = self.fluid_solver.poisson_solver.solve_multigrid( # f, u, c
            self.fluid_solver.div[1:-1,1:-1],
            p,
            coeff*c,
            ch = ch[1:,1:-1],
            cv = cv[1:-1,1:],
        )
        # ====== projection step ======
        (p_x, p_y) = self.fluid_solver.gradient(p)
        u          = uprime - ch * p_x
        v          = vprime - cv * p_y

        # self.set_BC(u,v)


        return (u,v,p)


    def set_BC(self, u,v):

        # self.fluid_solver.adv_diff_solver.set_BCs(u,v)
        # u[:,0]  = 0.0
        # v[:,0]  = 0.0

        # u[0,1:-1] = u[1,1:-1]
        # v[0,1:-1] = v[1,1:-1]+0.5*(u[0,2:]-u[1,:-2])

        # u[-1,1:-1] = u[-2,1:-1]
        # v[-1,1:-1] = v[-2,1:-1]+0.5*(u[-1,2:]-u[-1,:-2])

        # u[1:-1,-1] = u[1:-1,-2]
        # v[1:-1,-1] = v[1:-1,-2]+0.5*(u[2:,-1]-u[:-2,-1])

        # self.fluid_solver.adv_diff_solver.set_BCs(u,v)
        # u[:,0]=0
        # v[:,0]=0

        for i in [1,-1]:
            u[i,:]=0
            u[:,i]=0
            v[i,:]=0
            v[:,i]=0

        # u[:,0]  = u[:,2]
        # u[-1,:] = u[-3,:]
        # u[0,:]  = u[2,:]
        # u[:,-1] = u[:,-3]

        # v[:,0]  = v[:,2]
        # v[-1,:] = v[-3,:]
        # v[0,:]  = v[2,:]
        # v[:,-1] = v[:,-3]

    def step(self, task,physics):


        iteration = self.iteration
        timestep  = self.pars['solver']['dt']
        if iteration>=self.pars['solver']['nt']:
            return

        t = iteration*timestep

        # if iteration==0:
        #     (u,v,p) = (self.fluid_solver.u0, self.fluid_solver.v0, self.fluid_solver.p0)

        # # === stepping the fluid solver ===
        # if not self.terminate:

        #     # (
        #     # self.fluid_solver.u0,
        #     # self.fluid_solver.v0,
        #     # self.fluid_solver.p0,
        #     # self.terminate,
        #     # ) = self.fluid_stepper(
        #     #     self.fluid_solver.u0,
        #     #     self.fluid_solver.v0,
        #     #     self.fluid_solver.p0,
        #     #     iteration,
        #     #     t
        #     # )

        #     # update sdf_properties
        #     self.update(t, iteration, dt=timestep)

        #     (self.fluid_solver.mu0_all, self.fluid_solver.mu1_all)         = self.fluid_solver.composite_body.mu_funcs(self.fluid_solver.composite_body.sdf_val)
        #     self.fluid_solver.m_m0_all                                     = (1-self.fluid_solver.mu0_all)
        #     (_, self.fluid_solver.normal_x, self.fluid_solver.normal_y, _) = self.fluid_solver.composite_body.compute_sdf_properties(self.fluid_solver.composite_body.sdf_val)

        #     (self.fluid_solver.mu0_all_u, self.fluid_solver.mu1_all_u)         = self.fluid_solver.composite_body.mu_funcs(self.fluid_solver.composite_body.sdf_val_u)
        #     self.fluid_solver.m_m0_all_u                                       = (1-self.fluid_solver.mu0_all_u)
        #     (_, self.fluid_solver.normal_x_u, self.fluid_solver.normal_y_u, _) = self.fluid_solver.composite_body.compute_sdf_properties(self.fluid_solver.composite_body.sdf_val_u)

        #     (self.fluid_solver.mu0_all_v, self.fluid_solver.mu1_all_v)         = self.fluid_solver.composite_body.mu_funcs(self.fluid_solver.composite_body.sdf_val_v)
        #     self.fluid_solver.m_m0_all_v                                       = (1-self.fluid_solver.mu0_all_v)
        #     (_, self.fluid_solver.normal_x_v, self.fluid_solver.normal_y_v, _) = self.fluid_solver.composite_body.compute_sdf_properties(self.fluid_solver.composite_body.sdf_val_v)

        #     # self.fluid_solver.rho = (self.rho_fluid*self.fluid_solver.mu0_all_u + self.rho_body*self.fluid_solver.m_m0_all_u)
        #     self.fluid_solver.rho = (996.0*self.fluid_solver.mu0_all_u + 1010.0*self.fluid_solver.m_m0_all_u)

        #     (u,v,p) = (self.fluid_solver.u0, self.fluid_solver.v0, self.fluid_solver.p0)

        #     # (u1,v1,p1) = self.fluid_step(u,v,p,timestep)
        #     # (u2,v2,p2) = self.fluid_step(u1,v1,p1,timestep)
        #     # u=0.5*(u+u2)
        #     # v=0.5*(v+v2)
        #     # p=p2

        #     (u,v,p) = self.fluid_step(self.fluid_solver.u0, self.fluid_solver.v0, self.fluid_solver.p0,timestep)


        #     (self.fluid_solver.u0, self.fluid_solver.v0, self.fluid_solver.p0) = (u,v,p)

        #     # compute fluid forces on the body
        #     self.fluid_solver.forces_method2(self.fluid_solver.u0, self.fluid_solver.v0, self.fluid_solver.p0, iteration)

        #     self.terminate = self.fluid_solver.plotting_debug(self.fluid_solver.u0, self.fluid_solver.v0, self.fluid_solver.p0, iteration)

        #     self.apply_forces(task,physics)

        self.iteration+=1
