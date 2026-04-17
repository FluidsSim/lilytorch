"""Network controller"""

import numpy as np
from farms_amphibious.control.network import AnimatNetwork
from scipy.spatial.transform import Rotation
from lilytorch.solver import FluidSolver
import torch
from lilytorch.util.yaml_operations import yaml2pyobject

class DummyController:
    def __init__(self):
        pass

    def step(self, iteration, time, timestep):
        pass

class DragPositionController(AnimatNetwork):

    def __init__(self, animat_data, controller, n_iterations):
        super().__init__(data=animat_data, n_iterations=n_iterations)

    def step(self, iteration, time, timestep):
        pass


class DragMuscleController(AnimatNetwork):

    def __init__(self, animat_data, controller, n_iterations):
        super().__init__(data=animat_data, n_iterations=n_iterations)
        self.njoints=8
        self.offsets=np.zeros(self.njoints) # zero offsets
        self.controller = controller

    def step(self, iteration, time, timestep):
        """Control step"""
        if iteration>=self.n_iterations-1:
            return
        pos = np.array(self.data.sensors.joints.positions(iteration))
        urdf_positions = np.array(self.data.sensors.links.urdf_positions()[iteration])

        self.data.state.array[iteration] = np.concatenate([
            self.controller.step(iteration, time, timestep, pos=pos, urdf_positions=urdf_positions),
            self.offsets
            ])


class BDIMController(AnimatNetwork):

    def __init__(self, animat_data, sim_options, controller, yaml_file, n_iterations, control_type):
        self.n_iterations = n_iterations
        self.iteration=0
        super().__init__(data=animat_data, n_iterations=self.n_iterations)
        self.control_type = control_type
        if control_type == "torque":
            self.offsets                   = np.zeros(controller.n_total_joints) # zero offsets
        self.nlinks                    = len(animat_data.sensors.links.names)
        self.animat_data               = animat_data
        self.controller                = controller
        self.controller.exit_iteration = self.n_iterations
        self.continue_sim              = True

        pars = yaml2pyobject(yaml_file)

        self.fluid_solver = FluidSolver(pars, dtype=torch.float32, custom_update=True, compute_forces=True)

        self.device = self.fluid_solver.device
        self.dtype=self.fluid_solver.X.dtype

        # self.fluid_solver.composite_body.update = self.update_ib # modify the update rule
        # self.fluid_stepper = self.fluid_solver.step_pb_direct_forcing_old

        self.fluid_solver.composite_body.update = self.update # modify the update rule
        self.fluid_stepper = self.fluid_solver.step_

        controller.pars.log_path = self.fluid_solver.save_path

        self.n_bodies=len(self.fluid_solver.composite_body.bodies)
        self.terminate=False

        # enforce fluid timestep by overriding the one in the animat data
        animat_data.timestep = self.fluid_solver.dt
        sim_options["timestep"] = float(self.fluid_solver.dt)

        self.lambdas_u = torch.zeros((self.n_bodies, self.fluid_solver.nx, self.fluid_solver.ny), device=self.device, dtype=self.dtype)
        self.lambdas_v = torch.zeros((self.n_bodies, self.fluid_solver.nx, self.fluid_solver.ny), device=self.device, dtype=self.dtype)

    def update(self,t,iteration,dt=1):
        # iteration = int(t/dt)

        com_poses = torch.from_numpy(np.array(self.data.sensors.links.com_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        urdf_poses = torch.from_numpy(np.array(self.data.sensors.links.urdf_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        lin_vels = torch.from_numpy(np.array(self.data.sensors.links.com_lin_velocities()[iteration,:,:2]))
        ang_vels = torch.from_numpy(np.array([self.data.sensors.links.com_ang_velocity(iteration, link)[2] for link in range(self.nlinks)])) # only the z ang velocity in 2d
        orientation = torch.from_numpy(np.array(self.data.sensors.links.urdf_orientations()[iteration]).astype(np.float32))
        r = torch.from_numpy(Rotation.from_quat(orientation).as_matrix()[:,:2,:2].astype(np.float32)).to(self.device)

        for i, body in enumerate(self.fluid_solver.composite_body.bodies[:]):
            id=body.id # id of the body in FARMS

            R=r[id]
            urdf_pos=urdf_poses[id]
            lin_vel=lin_vels[id]
            ang_vel=ang_vels[id]
            com_pos=com_poses[id]

            # compute sdf values at the body points
            pos_trans = R.T@(body.stacked_xy-urdf_pos[:,None])
            newpos_u = pos_trans[0].reshape(body.nx, body.ny)
            newpos_v = pos_trans[1].reshape(body.nx, body.ny)
            self.fluid_solver.composite_body.sdf_vals[i]=body.sdf_interp(newpos_u, newpos_v)


            # compute sdf values at the body points
            pos_trans_u = R.T@(body.stacked_xy_u-urdf_pos[:,None])
            newpos_u = pos_trans_u[0].reshape(body.nx, body.ny)
            newpos_v = pos_trans_u[1].reshape(body.nx, body.ny)
            self.fluid_solver.composite_body.sdf_vals_u[i]=body.sdf_interp(newpos_u, newpos_v)

            pos_trans_v = R.T@(body.stacked_xy_v-urdf_pos[:,None])
            newpos_u = pos_trans_v[0].reshape(body.nx, body.ny)
            newpos_v = pos_trans_v[1].reshape(body.nx, body.ny)
            self.fluid_solver.composite_body.sdf_vals_v[i]=body.sdf_interp(newpos_u, newpos_v)

            # compute body velocities v = v_lin_com + <v_ang_com, x-x_com>
            self.fluid_solver.composite_body.u_vals[i]=lin_vel[0]-ang_vel*(self.fluid_solver.Y-com_pos[1])
            self.fluid_solver.composite_body.v_vals[i]=lin_vel[1]+ang_vel*(self.fluid_solver.X-com_pos[0])

            # store com positions for fluid->body force computation
            self.fluid_solver.composite_body.com_pos[i]=com_pos

            # update the contour position
            body.cnt_update = R @ body.cnt+urdf_pos[:,None]

            # compute the mask for the contour points
            x_cnt=body.cnt_update[0]
            y_cnt=body.cnt_update[1]
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
        self.fluid_solver.composite_body.sdf_val=self.fluid_solver.composite_body.sdf_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit

        idx_u=self.fluid_solver.composite_body.sdf_vals_u.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals_u.shape)
        self.fluid_solver.composite_body.sdf_val_u=self.fluid_solver.composite_body.sdf_vals_u.gather(0,idx_u)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit
        self.fluid_solver.composite_body.body_u=self.fluid_solver.composite_body.u_vals.gather(0,idx_u)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)

        idx_v=self.fluid_solver.composite_body.sdf_vals_v.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals_v.shape)
        self.fluid_solver.composite_body.sdf_val_v=self.fluid_solver.composite_body.sdf_vals_v.gather(0,idx_v)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit
        self.fluid_solver.composite_body.body_v=self.fluid_solver.composite_body.v_vals.gather(0,idx_v)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)


    def gamma(self, a, b):
        return (0.5*(1+((b-a)/torch.sqrt((b)**2 + (a)**2)).clip(-1,1)))

    def update_overlapping(self,t,iteration,dt=1):
        # iteration = int(t/dt)

        com_poses = torch.from_numpy(np.array(self.data.sensors.links.com_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        urdf_poses = torch.from_numpy(np.array(self.data.sensors.links.urdf_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        lin_vels = torch.from_numpy(np.array(self.data.sensors.links.com_lin_velocities()[iteration,:,:2]))
        ang_vels = torch.from_numpy(np.array([self.data.sensors.links.com_ang_velocity(iteration, link)[2] for link in range(self.nlinks)])) # only the z ang velocity in 2d
        orientation = torch.from_numpy(np.array(self.data.sensors.links.urdf_orientations()[iteration]).astype(np.float32))
        r = torch.from_numpy(Rotation.from_quat(orientation).as_matrix()[:,:2,:2].astype(np.float32)).to(self.device)

        for i, body in enumerate(self.fluid_solver.composite_body.bodies[:]):
            id=body.id # id of the body in FARMS

            R=r[id]
            urdf_pos=urdf_poses[id]
            lin_vel=lin_vels[id]
            ang_vel=ang_vels[id]
            com_pos=com_poses[id]

            # pos_trans = R.T@(body.stacked_xy-urdf_pos[:,None])
            # newpos_u = pos_trans[0].reshape(body.nx, body.ny)
            # newpos_v = pos_trans[1].reshape(body.nx, body.ny)
            # self.fluid_solver.composite_body.sdf_vals[i]=body.sdf_interp(newpos_u, newpos_v)

            # compute sdf values at staggered grid locations
            pos_trans_u = R.T@(body.stacked_xy_u-urdf_pos[:,None])
            newpos_u = pos_trans_u[0].reshape(body.nx, body.ny)
            newpos_v = pos_trans_u[1].reshape(body.nx, body.ny)
            self.fluid_solver.composite_body.sdf_vals_u[i]=body.sdf_interp(newpos_u, newpos_v)

            pos_trans_v = R.T@(body.stacked_xy_v-urdf_pos[:,None])
            newpos_u = pos_trans_v[0].reshape(body.nx, body.ny)
            newpos_v = pos_trans_v[1].reshape(body.nx, body.ny)
            self.fluid_solver.composite_body.sdf_vals_v[i]=body.sdf_interp(newpos_u, newpos_v)

            # # compute body velocities v = v_lin_com + <v_ang_com, x-x_com>
            # self.fluid_solver.composite_body.u_vals[i]=lin_vel[0]-ang_vel*(self.fluid_solver.Y-com_pos[1])
            # self.fluid_solver.composite_body.v_vals[i]=lin_vel[1]+ang_vel*(self.fluid_solver.X-com_pos[0])

            # store com positions for fluid->body force computation
            self.fluid_solver.composite_body.com_pos[i]=com_pos

            # update the contour position
            body.cnt_update = R @ body.cnt+urdf_pos[:,None]



            body.r_com = body.cnt_update-com_pos[:,None] # moment arm
            # body.mask=mask
            body.com_pos=com_pos

        # idx=self.fluid_solver.composite_body.sdf_vals.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals.shape)
        # self.fluid_solver.composite_body.sdf_val=self.fluid_solver.composite_body.sdf_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit

        idx_u=self.fluid_solver.composite_body.sdf_vals_u.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals_u.shape)
        self.fluid_solver.composite_body.sdf_val_u=self.fluid_solver.composite_body.sdf_vals_u.gather(0,idx_u)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit
        # self.fluid_solver.composite_body.body_u=self.fluid_solver.composite_body.u_vals.gather(0,idx_u)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)

        idx_v=self.fluid_solver.composite_body.sdf_vals_v.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals_v.shape)
        self.fluid_solver.composite_body.sdf_val_v=self.fluid_solver.composite_body.sdf_vals_v.gather(0,idx_v)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit
        # self.fluid_solver.composite_body.body_v=self.fluid_solver.composite_body.v_vals.gather(0,idx_v)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)


        # compute the body velocities
        for i, body in enumerate(self.fluid_solver.composite_body.bodies[:]):
            id=body.id # id of the body in FARMS

            R=r[id]
            urdf_pos=urdf_poses[id]
            lin_vel=lin_vels[id]
            ang_vel=ang_vels[id]
            com_pos=com_poses[id]

            # compute lambda functions
            if i==0:
                self.lambdas_u[i] = self.gamma(
                    self.fluid_solver.composite_body.sdf_vals_u[i],
                    self.fluid_solver.composite_body.sdf_vals_u[i+1]
                    )
                self.lambdas_v[i] = self.gamma(
                    self.fluid_solver.composite_body.sdf_vals_v[i],
                    self.fluid_solver.composite_body.sdf_vals_v[i+1]
                    )
            elif i==self.n_bodies-1:
                self.lambdas_u[i] = self.gamma(
                    self.fluid_solver.composite_body.sdf_vals_u[i],
                    self.fluid_solver.composite_body.sdf_vals_u[i-1]
                    )
                self.lambdas_v[i] = self.gamma(
                    self.fluid_solver.composite_body.sdf_vals_v[i],
                    self.fluid_solver.composite_body.sdf_vals_v[i-1]
                    )
            else:
                gamma_u_1 = self.gamma(
                    self.fluid_solver.composite_body.sdf_vals_u[i],
                    self.fluid_solver.composite_body.sdf_vals_u[i-1]
                    )
                gamma_u_2 = self.gamma(
                    self.fluid_solver.composite_body.sdf_vals_u[i],
                    self.fluid_solver.composite_body.sdf_vals_u[i+1]
                    )
                self.lambdas_u[i] = gamma_u_1 * gamma_u_2

                gamma_v_1 = self.gamma(
                    self.fluid_solver.composite_body.sdf_vals_v[i],
                    self.fluid_solver.composite_body.sdf_vals_v[i-1]
                    )
                gamma_v_2 = self.gamma(
                    self.fluid_solver.composite_body.sdf_vals_v[i],
                    self.fluid_solver.composite_body.sdf_vals_v[i+1]
                    )
                self.lambdas_v[i] = gamma_v_1 * gamma_v_2

            # compute body velocity of the rigid body on the staggered grid v = v_lin_com + <v_ang_com, x-x_com>
            vel_u = lin_vel[0]-ang_vel*(self.fluid_solver.Y-com_pos[1])
            vel_v = lin_vel[1]+ang_vel*(self.fluid_solver.X-com_pos[0])

            # compute mmu_0 functions for the bodies
            mu0_u, _ = self.fluid_solver.composite_body.mu_funcs(self.fluid_solver.composite_body.sdf_vals_u[i])
            mm0_u = (1-mu0_u) # this is where the body is located
            self.fluid_solver.composite_body.u_vals[i]=vel_u*mm0_u*self.lambdas_u[i]
            mu0_v, _ = self.fluid_solver.composite_body.mu_funcs(self.fluid_solver.composite_body.sdf_vals_v[i])
            mm0_v = (1-mu0_v) # this is where the body is located
            self.fluid_solver.composite_body.v_vals[i]=vel_v*mm0_v*self.lambdas_v[i]

        self.fluid_solver.composite_body.body_u=self.fluid_solver.composite_body.u_vals.sum(dim=0)
        self.fluid_solver.composite_body.body_v=self.fluid_solver.composite_body.v_vals.sum(dim=0)

    def interp(self, x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:

        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])

        indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indicies = torch.clamp(indicies, 0, len(m) - 1)

        return m[indicies] * x + b[indicies]

    def update_ib(self,t,iteration,dt=1):
        # iteration = int(t/dt)
        com_poses = torch.from_numpy(np.array(self.data.sensors.links.com_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        urdf_poses = torch.from_numpy(np.array(self.data.sensors.links.urdf_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        lin_vels = torch.from_numpy(np.array(self.data.sensors.links.com_lin_velocities()[iteration,:,:2]))
        ang_vels = torch.from_numpy(np.array([self.data.sensors.links.com_ang_velocity(iteration, link)[2] for link in range(self.nlinks)])) # only the z ang velocity in 2d
        orientation = torch.from_numpy(np.array(self.data.sensors.links.urdf_orientations()[iteration]).astype(np.float32))
        r = torch.from_numpy(Rotation.from_quat(orientation).as_matrix()[:,:2,:2].astype(np.float32)).to(self.device)

        for i, body in enumerate(self.fluid_solver.composite_body.bodies):

            id=body.id # id of the body in FARMS

            R=r[id]
            urdf_pos=urdf_poses[id]
            lin_vel=lin_vels[id]
            ang_vel=ang_vels[id]
            com_pos=com_poses[id]

            # update the contour position
            body.cnt_update = R @ body.cnt+urdf_pos[:,None]



            # # compute the mask (points inside the body)
            # x_cnt=body.cnt_update[0]
            # y_cnt=body.cnt_update[1]
            # if id==0:
            #     body_p=self.fluid_solver.composite_body.bodies[i+1]
            #     pos_trans = r[i+1].T@(torch.stack((x_cnt,y_cnt))-urdf_poses[i+1][:,None])
            #     sdf_p = body_p.sdf_interp(pos_trans[0],pos_trans[1])
            #     mask=(sdf_p >= 0)
            # elif id==self.n_bodies-1:
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

            # if i<self.n_bodies-1:
            #     body.first_set = (mask == True) & (body.sign_vec == 1)
            #     body.second_set = (mask == True) & (body.sign_vec == -1)
            # else:
            #     body.first_set = (mask == True)
            #     body.second_set = torch.zeros_like(mask).bool()

            moment_arm = body.cnt_update-com_pos[:,None] # momment arm
            body.cnt_u = lin_vel[0]-ang_vel*(moment_arm[1])
            body.cnt_v = lin_vel[1]+ang_vel*(moment_arm[0])
            body.r_com = moment_arm
            # body.mask=mask
            body.com_pos=com_pos


        self.fluid_solver.composite_body.com_pos = com_poses


        # all_points = [
        #     torch.cat([body.cnt_update[:,body.first_set] for body in self.composite_body.bodies],dim=1),
        #     torch.cat([body.cnt_update[:,body.second_set] for body in self.composite_body.bodies[::-1]],dim=1),
        #     ]
        # ds_all = []


        # cnt_all=torch.cat([
        #     torch.cat([body.cnt_update[:,body.first_set] for body in self.composite_body.bodies],dim=1),
        #     torch.cat([body.cnt_update[:,body.second_set] for body in self.composite_body.bodies[::-1]],dim=1),
        #     ], dim=1)
        # ds_all = (cnt_all.diff(axis=1)**2).sum(axis=0).sqrt()
        # cnt_all_x = (cnt_all[0,1:]+cnt_all[0,:-1])*0.5
        # cnt_all_y = (cnt_all[1,1:]+cnt_all[1,:-1])*0.5



        # for i, body in enumerate(self.fluid_solver.composite_body.bodies):

        #     # compute sdf values at the body points
        #     r_com_p = body.stacked_xy-urdf_pos[i][:,None]
        #     pos_trans = r[i].T@r_com_p
        #     xpos = pos_trans[0].reshape(body.nx, body.ny)
        #     ypos = pos_trans[1].reshape(body.nx, body.ny)
        #     sdf_val = body.sdf_interp(
        #         xpos,
        #         ypos
        #     )
        #     self.fluid_solver.composite_body.sdf_vals[i]=sdf_val
        #     # compute v = v_lin_com + <v_ang_com, x-x_com>
        #     self.fluid_solver.composite_body.u_vals[i]=lin_vel[i][0]-ang_vel[i]*(self.fluid_solver.Y-com_pos[i][1])
        #     self.fluid_solver.composite_body.v_vals[i]=lin_vel[i][1]+ang_vel[i]*(self.fluid_solver.X-com_pos[i][0])

        # idx=self.fluid_solver.composite_body.sdf_vals.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals.shape)
        # self.fluid_solver.composite_body.sdf_val=self.fluid_solver.composite_body.sdf_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit
        # self.fluid_solver.composite_body.body_u=self.fluid_solver.composite_body.u_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)
        # self.fluid_solver.composite_body.body_v=self.fluid_solver.composite_body.v_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)

    def update_ib_cui(self,t,iteration,dt=1):

        com_pos = torch.from_numpy(np.array(self.data.sensors.links.com_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        urdf_pos = torch.from_numpy(np.array(self.data.sensors.links.urdf_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        lin_vel = torch.from_numpy(np.array(self.data.sensors.links.com_lin_velocities()[iteration,:,:2]))
        ang_vel = torch.from_numpy(np.array([self.data.sensors.links.com_ang_velocity(iteration, link)[2] for link in range(self.nlinks)])) # only the z ang velocity in 2d
        orientation = torch.from_numpy(np.array(self.data.sensors.links.urdf_orientations()[iteration]).astype(np.float32))
        r = torch.from_numpy(R.from_quat(orientation).as_matrix()[:,:2,:2].astype(np.float32)).to(self.device)

        for i, body in enumerate(self.fluid_solver.composite_body.bodies):

            # compute sdf values at the body points
            r_com_p = body.stacked_xy-urdf_pos[i][:,None]
            pos_trans = r[i].T@r_com_p
            xpos = pos_trans[0].reshape(body.nx, body.ny)
            ypos = pos_trans[1].reshape(body.nx, body.ny)
            sdf_val = body.sdf_interp(
                xpos,
                ypos
            )
            self.fluid_solver.composite_body.sdf_vals[i]=sdf_val

            # for plotting
            body.com_pos=com_pos[i] # store com positions
            body.cnt_update = r[i] @ body.cnt+urdf_pos[i][:,None] # update the contour position



        idx=self.fluid_solver.composite_body.sdf_vals.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals.shape)
        d = self.fluid_solver.composite_body.sdf_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit
        self.fluid_solver.composite_body.sdf_val=d

        x = self.fluid_solver.x
        y = self.fluid_solver.y

        self.fluid_solver.composite_body.identity = torch.ones_like(d)
        self.fluid_solver.composite_body.identity[1:-1, 1:-1] = torch.where(
            d[1:-1,1:-1]>=0,
            torch.where(
                (d[:-2, 1:-1] >= 0) & (d[2:, 1:-1] >= 0) & (d[1:-1, :-2] >= 0) & (d[1:-1, 2:] >= 0),
                1,
                0
            ),
            -1
        ) # 0 = ib points, 1 = fluid points, -1 = solid points

        (gradx, grady) = torch.gradient(d, spacing=[self.fluid_solver.dx, self.fluid_solver.dy], edge_order=2)
        norm = torch.sqrt(gradx**2+grady**2)
        gradx=torch.where(norm>0, gradx/norm, 0)
        grady=torch.where(norm>0, grady/norm, 0)

        ib_idx = torch.nonzero(self.fluid_solver.composite_body.identity == 0, as_tuple=False)
        self.fluid_solver.composite_body.ib_idx = ib_idx
        # compute points on the contour
        self.fluid_solver.composite_body.xb, self.fluid_solver.composite_body.yb = x[ib_idx[:, 0]]-gradx[ib_idx[:, 0], ib_idx[:, 1]]*d[ib_idx[:, 0], ib_idx[:, 1]], y[ib_idx[:, 1]]-grady[ib_idx[:, 0], ib_idx[:, 1]]*d[ib_idx[:, 0], ib_idx[:, 1]]


        gradx_idx = torch.round(torch.sign(gradx[ib_idx[:, 0], ib_idx[:, 1]])).to(torch.int64)
        grady_idx = torch.round(torch.sign(grady[ib_idx[:, 0], ib_idx[:, 1]])).to(torch.int64)
        # compute horizontal neighbors
        self.fluid_solver.composite_body.nh_idx = torch.stack([ib_idx[:, 0]+gradx_idx, ib_idx[:, 1]], dim=-1)
        self.fluid_solver.composite_body.nh_idx[:, 1] = torch.where(
            self.fluid_solver.composite_body.identity[self.fluid_solver.composite_body.nh_idx[:, 0], self.fluid_solver.composite_body.nh_idx[:, 1]]==0, # if the neighbor is also a ib point
            ib_idx[:, 1]+grady_idx,
            ib_idx[:, 1]
        )
        # compute vertical neighbors
        self.fluid_solver.composite_body.nv_idx = torch.stack([ib_idx[:, 0], ib_idx[:, 1]+grady_idx], dim=-1)
        self.fluid_solver.composite_body.nv_idx[:, 0] = torch.where(
            self.fluid_solver.composite_body.identity[self.fluid_solver.composite_body.nv_idx[:, 0], self.fluid_solver.composite_body.nv_idx[:, 1]]==0,
            ib_idx[:, 0]+gradx_idx,
            ib_idx[:, 0]
        )


        self.fluid_solver.composite_body.ub = lin_vel[i][0]-ang_vel[i]*(self.fluid_solver.composite_body.yb-com_pos[i][1])
        self.fluid_solver.composite_body.vb = lin_vel[i][1]+ang_vel[i]*(self.fluid_solver.composite_body.xb-com_pos[i][0])

    def update_ib_bergmann(self,t,iteration,dt=1):

        com_pos = torch.from_numpy(np.array(self.data.sensors.links.com_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        urdf_pos = torch.from_numpy(np.array(self.data.sensors.links.urdf_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        lin_vel = torch.from_numpy(np.array(self.data.sensors.links.com_lin_velocities()[iteration,:,:2]))
        ang_vel = torch.from_numpy(np.array([self.data.sensors.links.com_ang_velocity(iteration, link)[2] for link in range(self.nlinks)])) # only the z ang velocity in 2d
        orientation = torch.from_numpy(np.array(self.data.sensors.links.urdf_orientations()[iteration]).astype(np.float32))
        r = torch.from_numpy(R.from_quat(orientation).as_matrix()[:,:2,:2].astype(np.float32)).to(self.device)

        for i, body in enumerate(self.fluid_solver.composite_body.bodies):

            # compute sdf values at the body points
            r_com_p = body.stacked_xy-urdf_pos[i][:,None]
            pos_trans = r[i].T@r_com_p
            xpos = pos_trans[0].reshape(body.nx, body.ny)
            ypos = pos_trans[1].reshape(body.nx, body.ny)
            sdf_val = body.sdf_interp(
                xpos,
                ypos
            )
            self.fluid_solver.composite_body.sdf_vals[i]=sdf_val

            # for plotting
            body.com_pos=com_pos[i] # store com positions
            body.cnt_update = r[i] @ body.cnt+urdf_pos[i][:,None] # update the contour position


        idx=self.fluid_solver.composite_body.sdf_vals.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals.shape)
        d = self.fluid_solver.composite_body.sdf_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit
        self.fluid_solver.composite_body.sdf_val=d

        x = self.fluid_solver.x
        y = self.fluid_solver.y

        self.fluid_solver.composite_body.identity = torch.ones_like(d)
        self.fluid_solver.composite_body.identity[1:-1, 1:-1] = torch.where(
            d[1:-1,1:-1]<=0,
            torch.where(
                (d[:-2, 1:-1] > 0) | (d[2:, 1:-1] > 0) | (d[1:-1, :-2] > 0) | (d[1:-1, 2:] > 0),
                0,
                -1
            ),
            1
        ) # 1 fluid, 0 ghost, -1 solid

        (gradx, grady) = torch.gradient(d, spacing=[self.fluid_solver.dx, self.fluid_solver.dy], edge_order=2)
        norm = torch.sqrt(gradx**2+grady**2)
        gradx=torch.where(norm>0, gradx/norm, 0)
        grady=torch.where(norm>0, grady/norm, 0)

        # computed index of the GHOST points
        gp_idx = torch.nonzero(self.fluid_solver.composite_body.identity == 0, as_tuple=False)
        self.fluid_solver.composite_body.gp_idx = gp_idx
        self.fluid_solver.composite_body.x_gp = x[gp_idx[:, 0]]
        self.fluid_solver.composite_body.y_gp = y[gp_idx[:, 1]]

        # compute points on the contour - IB points
        self.fluid_solver.composite_body.x_ib = x[gp_idx[:, 0]]-gradx[gp_idx[:, 0], gp_idx[:, 1]]*d[gp_idx[:, 0], gp_idx[:, 1]]
        self.fluid_solver.composite_body.y_ib = y[gp_idx[:, 1]]-grady[gp_idx[:, 0], gp_idx[:, 1]]*d[gp_idx[:, 0], gp_idx[:, 1]]
        # compute velocity at the IB points
        self.fluid_solver.composite_body.u_ib = lin_vel[i][0]-ang_vel[i]*(self.fluid_solver.composite_body.y_ib-com_pos[i][1])
        self.fluid_solver.composite_body.v_ib = lin_vel[i][1]+ang_vel[i]*(self.fluid_solver.composite_body.x_ib-com_pos[i][0])

        # compute the image points - IP
        self.fluid_solver.composite_body.x_ip = x[gp_idx[:, 0]]-2*gradx[gp_idx[:, 0], gp_idx[:, 1]]*d[gp_idx[:, 0], gp_idx[:, 1]]
        self.fluid_solver.composite_body.y_ip = y[gp_idx[:, 1]]-2*grady[gp_idx[:, 0], gp_idx[:, 1]]*d[gp_idx[:, 0], gp_idx[:, 1]]


        self.fluid_solver.idx_sw = torch.stack([
            ((self.fluid_solver.composite_body.x_ip - self.fluid_solver.x[0]) / self.fluid_solver.dx).to(torch.int64),
            ((self.fluid_solver.composite_body.y_ip - self.fluid_solver.y[0]) / self.fluid_solver.dy).to(torch.int64)
        ], dim=-1)

        self.fluid_solver.idx_nw = self.fluid_solver.idx_sw.clone().detach()
        self.fluid_solver.idx_nw[:, 1]+=1
        self.fluid_solver.idx_ne = self.fluid_solver.idx_sw.clone().detach()
        self.fluid_solver.idx_ne[:, 0]+=1
        self.fluid_solver.idx_ne[:, 1]+=1
        self.fluid_solver.idx_se = self.fluid_solver.idx_sw.clone().detach()
        self.fluid_solver.idx_se[:, 0]+=1


        import matplotlib.pyplot as plt

        # Select a few example image points (ip)
        num_examples = min(10, self.fluid_solver.composite_body.x_ip.shape[0])
        example_indices = torch.linspace(0, self.fluid_solver.composite_body.x_ip.shape[0]-1, num_examples).to(torch.int64)

        x_ip_examples = self.fluid_solver.composite_body.x_ip[example_indices].cpu().numpy()
        y_ip_examples = self.fluid_solver.composite_body.y_ip[example_indices].cpu().numpy()

        # Get neighbor indices for each example
        idx_sw_examples = self.fluid_solver.idx_sw[example_indices]
        idx_nw_examples = self.fluid_solver.idx_nw[example_indices]
        idx_ne_examples = self.fluid_solver.idx_ne[example_indices]
        idx_se_examples = self.fluid_solver.idx_se[example_indices]

        # Get coordinates for each neighbor
        x_sw = x[idx_sw_examples[:, 0]].cpu().numpy()
        y_sw = y[idx_sw_examples[:, 1]].cpu().numpy()
        x_nw = x[idx_nw_examples[:, 0]].cpu().numpy()
        y_nw = y[idx_nw_examples[:, 1]].cpu().numpy()
        x_ne = x[idx_ne_examples[:, 0]].cpu().numpy()
        y_ne = y[idx_ne_examples[:, 1]].cpu().numpy()
        x_se = x[idx_se_examples[:, 0]].cpu().numpy()
        y_se = y[idx_se_examples[:, 1]].cpu().numpy()

        plt.figure(figsize=(8, 8))
        plt.scatter(x_ip_examples, y_ip_examples, color='green', label='Image Points', s=40)
        plt.scatter(x_sw, y_sw, color='purple', label='SW Neighbor', marker='x')
        plt.scatter(x_nw, y_nw, color='blue', label='NW Neighbor', marker='x')
        plt.scatter(x_ne, y_ne, color='red', label='NE Neighbor', marker='x')
        plt.scatter(x_se, y_se, color='orange', label='SE Neighbor', marker='x')

        for i in range(num_examples):
            plt.plot([x_ip_examples[i], x_sw[i]], [y_ip_examples[i], y_sw[i]], 'k--', alpha=0.5)
            plt.plot([x_ip_examples[i], x_nw[i]], [y_ip_examples[i], y_nw[i]], 'k--', alpha=0.5)
            plt.plot([x_ip_examples[i], x_ne[i]], [y_ip_examples[i], y_ne[i]], 'k--', alpha=0.5)
            plt.plot([x_ip_examples[i], x_se[i]], [y_ip_examples[i], y_se[i]], 'k--', alpha=0.5)
        for body in self.fluid_solver.composite_body.bodies:
            plt.scatter(body.cnt_update[0], body.cnt_update[1], color='orange', s=10)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Image Points and Their SW/NW/NE/SE Neighbors')
        plt.show()
        # self.fluid_solver.composite_body.identity



        # import matplotlib.pyplot as plt

        # # Plot IB points (fluid_solver.composite_body.xb, yb)
        # plt.figure(figsize=(8, 8))

        # # Plot ghost points (gp_idx)
        # plt.scatter(x[gp_idx[:, 0]], y[gp_idx[:, 1]], color='red', label='Ghost Points', s=10)

        # # Plot IB points (x_ib, y_ib)
        # plt.scatter(self.fluid_solver.composite_body.x_ib, self.fluid_solver.composite_body.y_ib, color='blue', label='IB Points', s=10)

        # # Plot image points (x_ip, y_ip)
        # plt.scatter(self.fluid_solver.composite_body.x_ip, self.fluid_solver.composite_body.y_ip, color='green', label='Image Points', s=10)

        # # Plot cnt_update points for each body
        # for body in self.fluid_solver.composite_body.bodies:
        #     plt.scatter(body.cnt_update[0], body.cnt_update[1], color='orange', label='cnt_update Points', s=10)

        # # Avoid duplicate legend entries for cnt_update
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys())

        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.legend()
        # plt.title('Ghost, IB, and Image Points')
        # plt.show()

    def step(self, iteration, time, timestep):
        """Control step"""
        pass