"""Network controller"""

import numpy as np
from farms_amphibious.control.network import AnimatNetwork
from scipy.spatial.transform import Rotation as R
from lilytorch.solver import FluidSolver
import torch
from lilytorch.util.yaml_operations import yaml2pyobject


class ZebrafishController(AnimatNetwork):
    """zebrafish controller"""

    def __init__(self, animat_data, controller):
        self.n_iterations = np.shape(animat_data.state.array)[0]
        self.njoints = animat_data.sensors.joints.array.shape[1]
        super().__init__(data=animat_data, n_iterations=self.n_iterations)
        self.offsets=np.zeros(self.njoints) # zero offsets
        self.controller = controller

    def step(self, iteration, time, timestep):
        """Control step"""
        if iteration>=self.n_iterations-1:
            return
        pos = np.array(self.data.sensors.joints.positions(iteration)[4:-1])
        urdf_positions = np.array(self.data.sensors.links.urdf_positions()[iteration])

        self.data.state.array[iteration] = np.concatenate([
            self.controller.step(iteration, time, timestep, pos=pos, urdf_positions=urdf_positions),
            self.offsets
            ])


class ZebrafishFluidController(AnimatNetwork):
    """zebrafish controller"""

    def __init__(self, animat_data, controller, callback):
        self.n_iterations = np.shape(animat_data.state.array)[0]
        super().__init__(data=animat_data, n_iterations=self.n_iterations)
        self.offsets                   = np.zeros(controller.n_total_joints) # zero offsets
        self.nlinks                    = len(animat_data.sensors.links.names)
        self.animat_data               = animat_data
        self.controller                = controller
        self.controller.exit_iteration = self.n_iterations
        self.callback                  = callback
        self.continue_sim              = True

        pars = yaml2pyobject("../scripts/zebrafish_fluid.yaml")

        self.fluid_solver = FluidSolver(pars, costum_update=self.bodies_update)
        self.device = self.fluid_solver.device
        self.fluid_solver.composite_body.update = self.update # modify the update rule
        # self.fluid_solver.sdf_properties = self.initialize()

        self.n_bodies=len(self.fluid_solver.composite_body.bodies)

        self.dtype=self.fluid_solver.X.dtype

        # enforce fluid timestep
        animat_data.timestep = self.fluid_solver.dt
        controller.timestep = self.fluid_solver.dt

        _3d_2d_scaling=1
        # scale forces by the z-bounding box size
        self.callback.force_scaling = 1/_3d_2d_scaling
        # self.callback.force_scaling = np.array([np.diff(body.bb[2])[0]/_3d_2d_scaling for body in self.fluid_solver.composite_body.bodies])

    def initialize(self):
        r = torch.tensor([[-1.0,0.0],[0.0,-1.0]]).to(self.device)
        sdf_properties = []
        for body_i, body in enumerate(self.fluid_solver.composite_body.bodies):

            link = self.fluid_solver.composite_body.sdf.links[body_i]
            initial_pose = torch.from_numpy(np.array(link.pose).astype(np.float32))
            trans = torch.stack((initial_pose[0]*body.ones_stacked, initial_pose[1]*body.ones_stacked))
            pos_trans = r@(body.stacked_xy+trans)

            xpos = pos_trans[0].reshape(body.nx, body.ny)
            ypos = pos_trans[1].reshape(body.nx, body.ny)

            sdf_val = body.sdf_interp(xpos, ypos)
            sdf_properties.append(body.compute_sdf_properties(sdf_val))

            body.body_u=torch.zeros_like(self.fluid_solver.X)
            body.body_v=torch.zeros_like(self.fluid_solver.X)

            body.old_points = pos_trans

        return sdf_properties

    def update(self,t,iteration,dt=1):
        # iteration = int(t/dt)
        pos_global = torch.from_numpy(np.array(self.data.sensors.links.urdf_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        lin_vel = torch.from_numpy(np.array(self.data.sensors.links.com_lin_velocities()[iteration,:,:2]))
        ang_vel = torch.from_numpy(np.array([self.data.sensors.links.com_ang_velocity(iteration, link)[2] for link in range(self.nlinks)])) # only the z ang velocity in 2d
        orientation = torch.from_numpy(np.array(self.data.sensors.links.urdf_orientations()[iteration]).astype(np.float32))
        r = torch.from_numpy(R.from_quat(orientation).as_matrix()[:,:2,:2].astype(np.float32)).to(self.device) #R.from_quat(self.urdf_orientations[:,[3,0,1,2]])

        for i, body in enumerate(self.fluid_solver.composite_body.bodies):

            # trans = torch.stack((pos_global[body_i][0]*body.ones_stacked, pos_global[body_i][1]*body.ones_stacked))
            r_com_p = body.stacked_xy-pos_global[i][:,None]
            pos_trans = r[i].T@r_com_p

            # pos_trans = r[body_i].T@pos_trans
            xpos = pos_trans[0].reshape(body.nx, body.ny)
            ypos = pos_trans[1].reshape(body.nx, body.ny)
            sdf_val = body.sdf_interp(
                xpos,
                ypos
            )
            self.fluid_solver.composite_body.sdf_vals[i]=sdf_val

            ###### compute v = v_lin_com + <v_ang_com, x-x_com>
            self.fluid_solver.composite_body.u_vals[i]=lin_vel[i][0]-ang_vel[i]*(r_com_p[1]).reshape(body.nx, body.ny)
            self.fluid_solver.composite_body.v_vals[i]=lin_vel[i][1]+ang_vel[i]*(r_com_p[0]).reshape(body.nx, body.ny)


            # store com positions for fluid->body force computation
            self.fluid_solver.composite_body.com_pos[i]=pos_global[i]


        # self.sdf_val = torch.min(torch.stack([prop for idx, prop in enumerate(sdf_properties)]),axis=0)[0]


        idx=self.fluid_solver.composite_body.sdf_vals.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals.shape)

        self.fluid_solver.composite_body.sdf_val=self.fluid_solver.composite_body.sdf_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx+2,self.fluid_solver.ny+2)-self.fluid_solver.composite_body.suit

        # (self.mu0_all, self.mu1_all) = self.fluid_solver.composite_body.mu_funcs(self.composite_body.sdf_val)

        self.fluid_solver.body_u=self.fluid_solver.composite_body.u_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx+2,self.fluid_solver.ny+2)
        self.fluid_solver.body_v=self.fluid_solver.composite_body.v_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx+2,self.fluid_solver.ny+2)

    def update_old(self,t,iteration,dt=1):
        # iteration = int(t/dt)
        pos_global = torch.from_numpy(np.array(self.data.sensors.links.urdf_positions()[iteration,:, :2]).astype(np.float32)).to(self.device)
        # lin_vel = torch.from_numpy(np.array(self.data.sensors.links.com_lin_velocities()[iteration,:,:2]))
        # ang_vel = torch.from_numpy(np.array([self.data.sensors.links.com_ang_velocity(iteration, link)[2] for link in range(self.nlinks)])) # only the z ang velocity in 2d
        orientation = torch.from_numpy(np.array(self.data.sensors.links.urdf_orientations()[iteration]).astype(np.float32))
        r = torch.from_numpy(R.from_quat(orientation).as_matrix()[:,:2,:2].astype(np.float32)).to(self.device) #R.from_quat(self.urdf_orientations[:,[3,0,1,2]])
        for i, body in enumerate(self.fluid_solver.composite_body.bodies):

            # trans = torch.stack((pos_global[body_i][0]*body.ones_stacked, pos_global[body_i][1]*body.ones_stacked))
            pos_trans = r[i].T@(body.stacked_xy-pos_global[i][:,None])

            # pos_trans = r[body_i].T@pos_trans
            xpos = pos_trans[0].reshape(body.nx, body.ny)
            ypos = pos_trans[1].reshape(body.nx, body.ny)
            sdf_val = body.sdf_interp(
                xpos,
                ypos
            )
            self.fluid_solver.composite_body.sdf_vals[i]=sdf_val


            # sdf_properties.append(body.compute_sdf_properties(sdf_val))

            # pos_rot_glob = r[body_i] @ pos_trans
            # ang_v = ang_vel[body_i]
            # body.body_u = lin_vel[body_i][0]-ang_v*(pos_rot_glob[1]).reshape(body.nx, body.ny)
            # body.body_v = lin_vel[body_i][1]+ang_v*(pos_rot_glob[0]).reshape(body.nx, body.ny)

            # ang_v = -(ang_vel[body_i])
            # body.body_u = vel[body_i][0]-ang_v*(pos_trans[1]).reshape(body.nx, body.ny)
            # body.body_v = vel[body_i][1]+ang_v*(pos_trans[0]).reshape(body.nx, body.ny)


            # body.body_u = (vel[body_i][0]-ang_v*ypos)
            # body.body_v = (vel[body_i][1]+ang_v*xpos)

            velocity = - r[i]@(pos_trans-body.old_points) / dt

            self.fluid_solver.composite_body.u_vals[i]=velocity[0].reshape(body.nx, body.ny)
            self.fluid_solver.composite_body.v_vals[i]=velocity[1].reshape(body.nx, body.ny)

            # idx=sdf_val<0
            # self.fluid_solver.composite_body.u_vals[i]=torch.where(idx,velocity[0].reshape(body.nx, body.ny),0)
            # self.fluid_solver.composite_body.v_vals[i]=torch.where(idx,velocity[1].reshape(body.nx, body.ny),0)

            # body.body_u= velocity[0].reshape(body.nx, body.ny)
            # body.body_v= velocity[1].reshape(body.nx, body.ny)

            body.old_points = pos_trans

            self.fluid_solver.composite_body.com_pos[i]=pos_global[i]


        # self.sdf_val = torch.min(torch.stack([prop for idx, prop in enumerate(sdf_properties)]),axis=0)[0]


        idx=self.fluid_solver.composite_body.sdf_vals.argmin(0).unsqueeze(0).expand(self.fluid_solver.composite_body.sdf_vals.shape)

        self.fluid_solver.composite_body.sdf_val=self.fluid_solver.composite_body.sdf_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)-self.fluid_solver.composite_body.suit

        # (self.mu0_all, self.mu1_all) = self.fluid_solver.composite_body.mu_funcs(self.composite_body.sdf_val)

        self.fluid_solver.body_u=self.fluid_solver.composite_body.u_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)
        self.fluid_solver.body_v=self.fluid_solver.composite_body.v_vals.gather(0,idx)[0].reshape(self.fluid_solver.nx,self.fluid_solver.ny)


        # self.body_u=self.uv[:,0,:].gather(0,idx)[0].reshape(self.body.nx,self.body.ny)
        # self.body_v=self.uv[:,1,:].gather(0,idx)[0].reshape(self.body.nx,self.body.ny)


        # return sdf_properties

    def bodies_update(self, t):
        r = R.from_quat(self.urdf_orientations[:,[3,0,1,2]])
        angles = -r.as_euler("xyz",degrees=True)[:,0].astype(np.float32)
        translations = -self.urdf_positions[:,:2].astype(np.float32)
        return (
            torch.from_numpy(angles),
            torch.from_numpy(translations)
        )

    def step(self, iteration, time, timestep):
        """Control step"""

        if iteration>=self.n_iterations-1:
            return

        self.pos = np.array(self.data.sensors.joints.positions(iteration)[4:-1])
        self.urdf_positions = np.array(self.data.sensors.links.urdf_positions()[iteration])
        self.urdf_orientations = np.array(self.data.sensors.links.urdf_orientations()[iteration])

        # === stepping the controller ===
        self.data.state.array[iteration] = np.concatenate([
            self.controller.step(iteration, time, timestep, pos=self.pos, urdf_positions=self.urdf_positions),
            self.offsets
            ])

        # === stepping the fluid solver ===
        (
        self.fluid_solver.u0,
        self.fluid_solver.v0,
        self.fluid_solver.p0,
        continue_sim,
        ) = self.fluid_solver.step_(
            self.fluid_solver.u0,
            self.fluid_solver.v0,
            self.fluid_solver.p0,
            iteration,
            time
        )

        if not continue_sim: # stop sim is the fluid solver return an exit condition
            if self.controller.exit_iteration==self.n_iterations:
                self.controller.exit_iteration = iteration
            return



        self.callback.friction_force_lin_x = (self.fluid_solver.friction_force_lin_x).cpu().numpy()
        self.callback.friction_force_lin_y = (self.fluid_solver.friction_force_lin_y).cpu().numpy()

        self.callback.friction_force_ang_z = (self.fluid_solver.friction_force_ang_z).cpu().numpy()

        self.callback.pressure_force_x = (self.fluid_solver.pressure_force_x).cpu().numpy()
        self.callback.pressure_force_y = (self.fluid_solver.pressure_force_y).cpu().numpy()




class ZebrafishFluidSegmentController(AnimatNetwork):
    """zebrafish controller"""

    def __init__(self, animat_data, controller, callback):
        self.n_iterations = np.shape(animat_data.state.array)[0]
        super().__init__(data=animat_data, n_iterations=self.n_iterations)
        self.offsets      = np.zeros(controller.n_joints) # zero offsets
        self.nlinks = len(animat_data.sensors.links.names)
        self.animat_data  = animat_data
        self.controller   = controller
        self.callback = callback

        pars = yaml2pyobject("../scripts/zebrafish_fluid.yaml")

        self.fluid_solver = FluidSolver(pars, costum_update=True)
        self.device = self.fluid_solver.device
        self.fluid_solver.composite_body.update = self.update # modify the update rule

        # enforce fluid timestep
        animat_data.timestep = self.fluid_solver.dt
        controller.timestep = self.fluid_solver.dt

    def update(self,t,dt=1):
        iteration = int(t/dt)
        links_pos = torch.from_numpy(np.array(self.data.sensors.links.urdf_positions()[iteration,:, :2])).to(self.device) #16
        lin_vel   = torch.from_numpy(np.array(self.data.sensors.links.com_lin_velocities()[iteration,:,:2])).to(self.device) #16
        ang_vel   = torch.from_numpy(np.array([self.data.sensors.links.com_ang_velocity(iteration, link)[2]*np.pi/180.0 for link in range(self.nlinks)])).to(self.device) #16

        self.fluid_solver.composite_body.compute_sdf_and_velocities(links_pos, lin_vel, ang_vel, dt=dt)


       # sdf_properties = []
        # for body_i, body in enumerate(self.fluid_solver.composite_body.bodies):

        #     trans = torch.stack((pos_global[body_i][0]*body.ones_stacked, pos_global[body_i][1]*body.ones_stacked))
        #     pos_trans = (body.stacked_xy-trans)
        #     pos_trans = r[body_i].T@pos_trans

        #     # pos_trans = r[body_i].T@pos_trans
        #     xpos = pos_trans[0].reshape(body.nx, body.ny)
        #     ypos = pos_trans[1].reshape(body.nx, body.ny)
        #     sdf_val = body.sdf_interp(
        #         xpos,
        #         ypos
        #     )
        #     sdf_properties.append(body.compute_sdf_properties(sdf_val))

        #     # pos_rot_glob = r[body_i] @ pos_trans
        #     # ang_v = ang_vel[body_i]
        #     # body.body_u = lin_vel[body_i][0]-ang_v*(pos_rot_glob[1]).reshape(body.nx, body.ny)
        #     # body.body_v = lin_vel[body_i][1]+ang_v*(pos_rot_glob[0]).reshape(body.nx, body.ny)

        #     # ang_v = -(ang_vel[body_i])
        #     # body.body_u = vel[body_i][0]-ang_v*(pos_trans[1]).reshape(body.nx, body.ny)
        #     # body.body_v = vel[body_i][1]+ang_v*(pos_trans[0]).reshape(body.nx, body.ny)


        #     # body.body_u = (vel[body_i][0]-ang_v*ypos)
        #     # body.body_v = (vel[body_i][1]+ang_v*xpos)

        #     velocity = - r[body_i]@(pos_trans-body.old_points) / dt
        #     body.body_u= velocity[0].reshape(body.nx, body.ny)
        #     body.body_v= velocity[1].reshape(body.nx, body.ny)
        #     body.old_points = pos_trans


        # return sdf_properties


    def bodies_update(self, t):
        return

    def step(self, iteration, time, timestep):
        """Control step"""
        if iteration>=self.n_iterations-1:
            return
        self.pos = np.array(self.data.sensors.joints.positions(iteration)[4:-1])
        self.urdf_positions = np.array(self.data.sensors.links.urdf_positions()[iteration])
        self.urdf_orientations = np.array(self.data.sensors.links.urdf_orientations()[iteration])

        # === stepping the controller ===
        self.data.state.array[iteration] = np.concatenate([
            self.controller.step(iteration, time, timestep, pos=self.pos, urdf_positions=self.urdf_positions),
            self.offsets
            ])

        # === stepping the fluid solver ===
        (
        self.fluid_solver.u0,
        self.fluid_solver.v0,
        self.fluid_solver.p0
        ) = self.fluid_solver.step_(
            self.fluid_solver.u0,
            self.fluid_solver.v0,
            self.fluid_solver.p0,
            iteration,
            time
        )



