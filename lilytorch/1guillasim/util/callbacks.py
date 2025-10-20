"""Callbacks"""

import numpy as np
from imageio import imread
from farms_core.sensors.sensor_convention import sc
from farms_mujoco.simulation.task import TaskCallback
from farms_mujoco.swimming.drag import SwimmingHandler
from farms_amphibious.model.options import AmphibiousOptions, AmphibiousArenaOptions, AnimatOptions

def water_velocity_from_maps(position, water_maps):
    """Water velocity from maps"""
    vel = np.zeros(3)
    if all(
            water_maps['pos_min'][i] < position[i] < water_maps['pos_max'][i]
            for i in range(2)
    ):
        vel[:2] = [
            water_maps[png][tuple(
                (
                    max(0, min(
                        water_maps[png].shape[index]-1,
                        round(water_maps[png].shape[index]*(
                            (
                                position[index]
                                - water_maps['pos_min'][index]
                            ) / (
                                water_maps['pos_max'][index]
                                - water_maps['pos_min'][index]
                            )
                        ))
                    ))
                )
                for index in range(2)
            )]
            for png_i, png in enumerate(['vel_x', 'vel_y'])
        ]
    # vel[1] *= -1
    return vel

class DragCallback(TaskCallback):
    """Swimming callback"""

    def __init__(
            self,
            animat_options: AmphibiousOptions,
            arena_options: AmphibiousArenaOptions,
            substep=True,
    ):
        super().__init__(substep=substep)
        self.animat_options = animat_options
        self.arena_options = arena_options
        self._handler: SwimmingHandler = None

        self.constant_velocity: bool = (
            len(arena_options.water.velocity) == 3
        )
        if not self.constant_velocity:
            water_velocity = arena_options.water.velocity
            water_maps = arena_options.water.maps
            pngs = [np.flipud(imread(water_maps[i])).T for i in range(2)]
            pngs_info = [np.iinfo(png.dtype) for png in pngs]
            vels = [
                (
                    png - info.min  # .astype(np.double)
                ) * (
                    water_velocity[png_i+3] - water_velocity[png_i+0]
                ) / (
                    info.max - info.min
                ) + water_velocity[png_i+0]
                for png_i, (png, info) in enumerate(zip(pngs, pngs_info))
            ]
            self.water_maps = {
                'pos_min': np.array(water_velocity[6:8]),
                'pos_max': np.array(water_velocity[8:10]),
                'vel_x': vels[0],
                'vel_y': vels[1],
            }

    def initialize_episode(self, task, physics):
        """Initialize episode"""
        self._handler = SwimmingHandler(
            data=task.data,
            animat_options=self.animat_options,
            arena_options=self.arena_options,
            units=task.units,
            physics=physics,
        )

    def before_step(self, task, action, physics):
        """Step hydrodynamics"""

        # Water maps
        if not self.constant_velocity:
            water_velocities = np.array(
                [
                    water_velocity_from_maps(
                        position=task.data.sensors.links.urdf_position(
                            iteration=task.iteration,
                            link_i= link_i,
                        ),
                        water_maps=self.water_maps,
                    )
                    for link_i, link in enumerate(self.animat_options.morphology.links)
                    if link.swimming
                ]
            )

            self._handler.set_water_velocities(water_velocities)

        # Compute fluid forces
        self._handler.step(task.iteration)

        # Set fluid forces in physics engine
        indices = task.maps['sensors']['data2xfrc']
        physics.data.xfrc_applied[indices, :] = (
            task.data.sensors.xfrc.array[
                task.iteration, :,
                sc.xfrc_force_x:sc.xfrc_torque_z+1,
            ]
        )
        for force_i, (rotation_mat, force_local) in enumerate(zip(
                physics.data.xmat[indices],
                physics.data.xfrc_applied[indices],
        )):
            physics.data.xfrc_applied[indices[force_i]] = (
                rotation_mat.reshape([3, 3])  # Local to global frame
                @ force_local.reshape([3, 2], order='F')
            ).flatten(order='F')
        physics.data.xfrc_applied[indices, :3] *= task.units.newtons
        physics.data.xfrc_applied[indices, 3:] *= task.units.torques

class FluidCallback(TaskCallback):
    """Fluid callback"""

    def __init__(
            self,
            animat_options: AmphibiousOptions,
            arena_options: AmphibiousArenaOptions,
            bdim_controller, # BDIMController
            substep=True,
    ):
        super().__init__(substep=substep)
        self.animat_options  = animat_options
        self.arena_options   = arena_options
        self.bdim_controller = bdim_controller
        self.nfrc            = len(self.animat_options['control']['sensors']['xfrc'])

        self.friction_force_lin_x = np.zeros(self.nfrc)
        self.friction_force_lin_y = np.zeros(self.nfrc)
        self.friction_force_ang_z = np.zeros(self.nfrc)
        self.pressure_force_x     = np.zeros(self.nfrc)
        self.pressure_force_y     = np.zeros(self.nfrc)
        self.pressure_force_ang_z = np.zeros(self.nfrc)


        # _3d_2d_scaling=1000 # TO CONVERT FROM RHO GIVEN IN NON-DIMENSIONAL FORM OF THE NS EQUATIONS TO ACTUAL RHO (DENSITY) IN MUJOCO
        # # scale forces by the z-bounding box size
        # self.force_scaling =_3d_2d_scaling*np.array([np.diff(body.bb[2])[0] for body in self.bdim_controller.fluid_solver.composite_body.bodies])

        # self.force_scaling = 1/_3d_2d_scaling
        self.force_scaling=1 #0.0455*1000/float(self.bdim_controller.fluid_solver.rho) # scale by

        print("Force scaling: ", self.force_scaling)



    def after_step(self, task, physics):
    # def before_step(self, task, action, physics):
        """Step hydrodynamics"""
        indices = task.maps['sensors']['data2xfrc']


        iteration= self.bdim_controller.iteration
        timestep = task.timestep
        time    = iteration*timestep

        if iteration>=self.bdim_controller.n_iterations-1:
            return


        # === stepping the fluid solver ===
        if not self.bdim_controller.terminate:
            (
            self.bdim_controller.fluid_solver.u0,
            self.bdim_controller.fluid_solver.v0,
            self.bdim_controller.fluid_solver.p0,
            self.bdim_controller.terminate,
            ) = self.bdim_controller.fluid_stepper(
                self.bdim_controller.fluid_solver.u0,
                self.bdim_controller.fluid_solver.v0,
                self.bdim_controller.fluid_solver.p0,
                iteration,
                time
            )

        # if not continue_sim: # stop sim is the fluid solver return an exit condition
        #     if self.controller.exit_iteration==self.n_iterations:
        #         self.controller.exit_iteration = iteration
        #     return

        self.friction_force_lin_x = self.force_scaling*(self.bdim_controller.fluid_solver.friction_force_lin_x).cpu().numpy()
        self.friction_force_lin_y = self.force_scaling*(self.bdim_controller.fluid_solver.friction_force_lin_y).cpu().numpy()
        self.friction_force_ang_z = self.force_scaling*(self.bdim_controller.fluid_solver.friction_force_ang_z).cpu().numpy()

        self.pressure_force_x = self.force_scaling*(self.bdim_controller.fluid_solver.pressure_force_x).cpu().numpy()
        self.pressure_force_y = self.force_scaling*(self.bdim_controller.fluid_solver.pressure_force_y).cpu().numpy()
        self.pressure_force_ang_z = self.force_scaling*(self.bdim_controller.fluid_solver.pressure_force_ang_z).cpu().numpy()

        if self.bdim_controller.control_type == "torque":
            # === stepping the controller ===

            pos = np.array(self.data.sensors.joints.positions(iteration)[4:-1])
            urdf_positions = np.array(self.data.sensors.links.urdf_positions()[iteration])
            urdf_orientations = np.array(self.data.sensors.links.urdf_orientations()[iteration])

            self.data.state.array[iteration] = np.concatenate([
                self.controller.step(iteration, time, timestep, pos=pos, urdf_positions=urdf_positions),
                self.offsets
                ])

        physics.data.xfrc_applied[indices, 0] = (self.friction_force_lin_x + self.pressure_force_x) * task.units.newtons
        physics.data.xfrc_applied[indices, 1] = (self.friction_force_lin_y + self.pressure_force_y) * task.units.newtons
        physics.data.xfrc_applied[indices, 5] = (self.friction_force_ang_z + self.pressure_force_ang_z) * task.units.newtons

        print(physics.data.xfrc_applied[indices,0])


        self.bdim_controller.iteration+=1
