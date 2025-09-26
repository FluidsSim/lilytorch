"""Network controller"""

import numpy as np
from farms_amphibious.control.network import AnimatNetwork
from scipy.spatial.transform import Rotation as R
from lilytorch.solver import FluidSolver
import torch
from lilytorch.util.yaml_operations import yaml2pyobject
import matplotlib.pyplot as plt


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

    def __init__(self, animat_data, sim_options, controller, callback, yaml_file):
        self.n_iterations = np.shape(animat_data.state.array)[0]
        super().__init__(data=animat_data, n_iterations=self.n_iterations)
        self.offsets                   = np.zeros(controller.n_total_joints) # zero offsets
        self.nlinks                    = len(animat_data.sensors.links.names)
        self.animat_data               = animat_data
        self.controller                = controller
        self.controller.exit_iteration = self.n_iterations
        self.callback                  = callback
        self.continue_sim              = True

        pars = yaml2pyobject(yaml_file)

        self.fluid_solver = FluidSolver(pars, dtype=torch.float32, costum_update=None)

        self.device = self.fluid_solver.device
        self.fluid_solver.composite_body.update = self.update_ib # modify the update rule
        self.fluid_stepper = self.fluid_solver.step_pb_direct_forcing
        # self.fluid_stepper = self.fluid_solver.step_
        # self.fluid_solver.sdf_properties = self.initialize()

        self.n_bodies=len(self.fluid_solver.composite_body.bodies)

        self.dtype=self.fluid_solver.X.dtype

        # enforce fluid timestep by overriding the one in the animat data
        animat_data.timestep = self.fluid_solver.dt
        sim_options["timestep"] = float(self.fluid_solver.dt)

        _3d_2d_scaling=1
        # scale forces by the z-bounding box size
        # self.callback.force_scaling = 1/_3d_2d_scaling
        # self.callback.force_scaling = _3d_2d_scaling*np.array([np.diff(body.bb[2])[0] for body in self.fluid_solver.composite_body.bodies])
        self.callback.force_scaling=0.0455

        print("Force scaling: ", self.callback.force_scaling)



    def update(self,t,iteration,dt=1):
        # iteration = int(t/dt)
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

            # compute v = v_lin_com + <v_ang_com, x-x_com>
            self.fluid_solver.composite_body.u_vals[i]=lin_vel[i][0]-ang_vel[i]*(self.fluid_solver.X-com_pos[i][0])
            self.fluid_solver.composite_body.v_vals[i]=lin_vel[i][1]+ang_vel[i]*(self.fluid_solver.Y-com_pos[i][1])

            # store com positions for fluid->body force computation
            self.fluid_solver.composite_body.com_pos[i]=com_pos[i]


            # update the contour position
            body.cnt_update = r[i] @ body.cnt+urdf_pos[i][:,None]

            # compute the mask (points inside the body)
            x_cnt=body.cnt_update[0]
            y_cnt=body.cnt_update[1]
            if i==0:
                body_p=self.fluid_solver.composite_body.bodies[i+1]
                pos_trans = r[i+1].T@(torch.stack((x_cnt,y_cnt))-urdf_pos[i+1][:,None])
                sdf_p = body_p.sdf_interp(pos_trans[0],pos_trans[1])
                mask=(sdf_p >= 0)
            elif i==self.n_bodies-1:
                body_m=self.fluid_solver.composite_body.bodies[i-1]
                pos_trans = r[i-1].T@(torch.stack((x_cnt,y_cnt))-urdf_pos[i-1][:,None])
                sdf_m = body_m.sdf_interp(pos_trans[0],pos_trans[1])
                mask=(sdf_m >= 0)
            else:
                body_m=self.fluid_solver.composite_body.bodies[i-1]
                pos_trans_m = r[i-1].T@(torch.stack((x_cnt,y_cnt))-urdf_pos[i-1][:,None])
                body_p=self.fluid_solver.composite_body.bodies[i+1]
                pos_trans_p = r[i+1].T@(torch.stack((x_cnt,y_cnt))-urdf_pos[i+1][:,None])
                sdf_m = body_m.sdf_interp(pos_trans_m[0],pos_trans_m[1])
                sdf_p = body_p.sdf_interp(pos_trans_p[0],pos_trans_p[1])
                mask=(sdf_m >= 0) & (sdf_p >= 0)


            moment_arm = body.cnt_update-com_pos[i][:,None] # momment arm
            body.cnt_u = lin_vel[i][0]-ang_vel[i]*(moment_arm[1])
            body.cnt_v = lin_vel[i][1]+ang_vel[i]*(moment_arm[0])
            body.r_com = moment_arm
            body.mask=mask
            body.com_pos=com_pos[i]

        # print(body.cnt_update.mean(axis=1), pos_global[i])

        idx=self.fluid_solver.composite_body.sdf_vals.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals.shape)

        self.fluid_solver.composite_body.sdf_val=self.fluid_solver.composite_body.sdf_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit

        # (self.mu0_all, self.mu1_all) = self.fluid_solver.composite_body.mu_funcs(self.composite_body.sdf_val)

        self.fluid_solver.composite_body.body_u=self.fluid_solver.composite_body.u_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)
        self.fluid_solver.composite_body.body_v=self.fluid_solver.composite_body.v_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)

    def update_ib(self,t,iteration,dt=1):
        # iteration = int(t/dt)
        com_pos = torch.from_numpy(np.array(self.data.sensors.links.com_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        urdf_pos = torch.from_numpy(np.array(self.data.sensors.links.urdf_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        lin_vel = torch.from_numpy(np.array(self.data.sensors.links.com_lin_velocities()[iteration,:,:2]))
        ang_vel = torch.from_numpy(np.array([self.data.sensors.links.com_ang_velocity(iteration, link)[2] for link in range(self.nlinks)])) # only the z ang velocity in 2d
        orientation = torch.from_numpy(np.array(self.data.sensors.links.urdf_orientations()[iteration]).astype(np.float32))
        r = torch.from_numpy(R.from_quat(orientation).as_matrix()[:,:2,:2].astype(np.float32)).to(self.device)

        for i, body in enumerate(self.fluid_solver.composite_body.bodies):

            # update the contour position
            body.cnt_update = r[i] @ body.cnt+urdf_pos[i][:,None]

            # compute the mask (points inside the body)
            x_cnt=body.cnt_update[0]
            y_cnt=body.cnt_update[1]
            if i==0:
                body_p=self.fluid_solver.composite_body.bodies[i+1]
                pos_trans = r[i+1].T@(torch.stack((x_cnt,y_cnt))-urdf_pos[i+1][:,None])
                sdf_p = body_p.sdf_interp(pos_trans[0],pos_trans[1])
                mask=(sdf_p >= 0)
            elif i==self.n_bodies-1:
                body_m=self.fluid_solver.composite_body.bodies[i-1]
                pos_trans = r[i-1].T@(torch.stack((x_cnt,y_cnt))-urdf_pos[i-1][:,None])
                sdf_m = body_m.sdf_interp(pos_trans[0],pos_trans[1])
                mask=(sdf_m >= 0)
            else:
                body_m=self.fluid_solver.composite_body.bodies[i-1]
                pos_trans_m = r[i-1].T@(torch.stack((x_cnt,y_cnt))-urdf_pos[i-1][:,None])
                body_p=self.fluid_solver.composite_body.bodies[i+1]
                pos_trans_p = r[i+1].T@(torch.stack((x_cnt,y_cnt))-urdf_pos[i+1][:,None])
                sdf_m = body_m.sdf_interp(pos_trans_m[0],pos_trans_m[1])
                sdf_p = body_p.sdf_interp(pos_trans_p[0],pos_trans_p[1])
                mask=(sdf_m >= 0) & (sdf_p >= 0)

            moment_arm = body.cnt_update-com_pos[i][:,None] # momment arm
            body.cnt_u = lin_vel[i][0]-ang_vel[i]*(moment_arm[1])
            body.cnt_v = lin_vel[i][1]+ang_vel[i]*(moment_arm[0])
            body.r_com = moment_arm
            body.mask=mask
            body.com_pos=com_pos[i]

        all_cnt = []
        for i, body in enumerate(self.fluid_solver.composite_body.bodies):
            if i<self.n_bodies-1:
                first_set = (body.mask == True) & (body.sign_vec == 1)
                all_cnt.append(body.cnt_update[:,first_set])
            else:
                all_cnt.append(body.cnt_update[:,body.mask == True])
        for i, body in enumerate(self.fluid_solver.composite_body.bodies):
            if i<self.n_bodies-1:
                second_set = (body.mask == True) & (body.sign_vec == -1)
                all_cnt.append(body.cnt_update[:,second_set])

        self.fluid_solver.composite_body.all_cnt = torch.concatenate(all_cnt, axis=1)
        # plt.figure()
        # plt.scatter(self.fluid_solver.composite_body.all_cnt[0], self.fluid_solver.composite_body.all_cnt[1], c=np.arange(self.fluid_solver.composite_body.all_cnt.shape[1]), cmap='viridis')
        # plt.colorbar(label='Index')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('all_cnt scatter')
        # plt.show()


    def step(self, iteration, time, timestep):
        """Control step"""

        if iteration>=self.n_iterations-1:
            return

        self.pos = np.array(self.data.sensors.joints.positions(iteration)[4:-1])
        self.urdf_positions = np.array(self.data.sensors.links.urdf_positions()[iteration])
        self.urdf_orientations = np.array(self.data.sensors.links.urdf_orientations()[iteration])

        # === stepping the fluid solver ===
        (
        self.fluid_solver.u0,
        self.fluid_solver.v0,
        self.fluid_solver.p0,
        continue_sim,
        ) = self.fluid_stepper(
            self.fluid_solver.u0,
            self.fluid_solver.v0,
            self.fluid_solver.p0,
            iteration,
            time
        )

        # if not continue_sim: # stop sim is the fluid solver return an exit condition
        #     if self.controller.exit_iteration==self.n_iterations:
        #         self.controller.exit_iteration = iteration
        #     return

        self.callback.friction_force_lin_x = self.callback.force_scaling*(self.fluid_solver.friction_force_lin_x).cpu().numpy()
        self.callback.friction_force_lin_y = self.callback.force_scaling*(self.fluid_solver.friction_force_lin_y).cpu().numpy()

        self.callback.friction_force_ang_z = self.callback.force_scaling*(self.fluid_solver.friction_force_ang_z).cpu().numpy()

        self.callback.pressure_force_x = self.callback.force_scaling*(self.fluid_solver.pressure_force_x).cpu().numpy()
        self.callback.pressure_force_y = self.callback.force_scaling*(self.fluid_solver.pressure_force_y).cpu().numpy()

        # === stepping the controller ===
        self.data.state.array[iteration] = np.concatenate([
            self.controller.step(iteration, time, timestep, pos=self.pos, urdf_positions=self.urdf_positions),
            self.offsets
            ])
