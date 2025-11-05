"""Network controller"""

import numpy as np
from scipy.spatial.transform import Rotation
from lilytorch.src.solver import FluidSolver
import torch

class BDIMhandler():

    def __init__(self, yaml_file, data, dtype=torch.float32):

        self.dtype = dtype
        if self.dtype == torch.float32:
            self.dtype_np = np.float32
        elif self.dtype == torch.float64:
            self.dtype_np = np.float64

        self.data = data # ExperimentData in FARMS
        self.iteration = 0

        pars = yaml_file

        self.fluid_solver = FluidSolver(
            pars,
            dtype=self.dtype,
            costum_update=True,
            compute_forces=True
        )
        self.device = self.fluid_solver.device

        self.fluid_solver.composite_body.update = self.update  # modify the update rule
        self.fluid_stepper = self.fluid_solver.step_

        # convert the axes from (x,z) sagittal plane to (x,y) plane
        self.lin_axes = [0,2] # use x and z linear axes for 2D sim
        self.ang_axes = [1]   # use y rotation axis for 2D sim

        self.force_scaling = 1000.0


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

            # # compute sdf values at the body points
            # pos_trans = R.T@(self.fluid_solver.composite_body.stacked_xy-urdf_pos[:,None])
            # newpos_u = pos_trans[0].reshape(self.fluid_solver.composite_body.nx, self.fluid_solver.composite_body.ny)
            # newpos_v = pos_trans[1].reshape(self.fluid_solver.composite_body.nx, self.fluid_solver.composite_body.ny)
            # self.fluid_solver.composite_body.sdf_vals[i]=body.sdf(newpos_u, newpos_v)

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


        # idx=self.fluid_solver.composite_body.sdf_vals.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals.shape)
        # self.fluid_solver.composite_body.sdf_val=self.fluid_solver.composite_body.sdf_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit

        idx_u=self.fluid_solver.composite_body.sdf_vals_u.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals_u.shape)
        self.fluid_solver.composite_body.sdf_val_u=self.fluid_solver.composite_body.sdf_vals_u.gather(0,idx_u)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit
        self.fluid_solver.composite_body.body_u=self.fluid_solver.composite_body.u_vals.gather(0,idx_u)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)

        idx_v=self.fluid_solver.composite_body.sdf_vals_v.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals_v.shape)
        self.fluid_solver.composite_body.sdf_val_v=self.fluid_solver.composite_body.sdf_vals_v.gather(0,idx_v)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit
        self.fluid_solver.composite_body.body_v=self.fluid_solver.composite_body.v_vals.gather(0,idx_v)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)


    def apply_forces(self, task, physics):

        # physics.data.qvel[1]=-0.05
        # physics.data.qvel[4]=-0.05
        # physics.data.qvel[7]=-0.05

        self.friction_force_lin_x = self.force_scaling*(self.fluid_solver.friction_force_lin_x).cpu().numpy()
        self.friction_force_lin_y = self.force_scaling*(self.fluid_solver.friction_force_lin_y).cpu().numpy()
        self.friction_force_ang_z = self.force_scaling*(self.fluid_solver.friction_force_ang_z).cpu().numpy()

        self.pressure_force_x = self.force_scaling*(self.fluid_solver.pressure_force_x).cpu().numpy()
        self.pressure_force_y = self.force_scaling*(self.fluid_solver.pressure_force_y).cpu().numpy()
        self.pressure_force_ang_z = self.force_scaling*(self.fluid_solver.pressure_force_ang_z).cpu().numpy()




        for body_i, body in enumerate(self.fluid_solver.composite_body.bodies[:]):

            (animat_id, link_id) = self.fluid_solver.composite_body.body_ids[body_i]

            ind_task= task.maps[animat_id]['sensors']['data2xfrc'][link_id]


            # mass=self.data[animat_id].sensors.links.masses[link_id] * task.units.kilograms
            # physics.data.xfrc_applied[ind_task, 2] = (9.81) * task.units.newtons*mass


            physics.data.xfrc_applied[ind_task, 0] = (self.friction_force_lin_x[body_i] + 0*self.pressure_force_x[body_i]) * task.units.newtons
            physics.data.xfrc_applied[ind_task, 2] = (self.friction_force_lin_y[body_i] + 0*self.pressure_force_y[body_i]) * task.units.newtons
            physics.data.xfrc_applied[ind_task, 4] = (self.friction_force_ang_z[body_i] + 0*self.pressure_force_ang_z[body_i]) * task.units.newtons


            print("Force z: {}, qvel[1]: {}".format(physics.data.xfrc_applied[ind_task, 2], physics.data.qvel[1]))
            # print(self.pressure_force_y)

            # physics.data.qvel[1]=-0.05
