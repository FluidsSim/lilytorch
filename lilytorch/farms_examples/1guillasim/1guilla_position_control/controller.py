
from farms_core.model.control import AnimatController
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from farms_core.sensors.sensor_convention import sc
from farms_core.model.control import ControlType
import numpy as np
from lilytorch.util.rw import Dict2Class
from farms_amphibious.control.kinematics import KinematicsController
import matplotlib.pyplot as plt

class PositionController(KinematicsController):
    def __init__(self, animat_data, animat_options, experiment_options, config, animat_i):

        joints_names          = animat_options.control.joints_names()
        kinematics_sampling   = experiment_options.simulation.physics.timestep
        kinematics_indices    = [1,2,3,4,5,6,7,8]
        kinematics_time_index = 0
        kinematics_invert     = False
        kinematics_degrees    = False
        kinematics_start      = 0.0
        kinematics_end        = experiment_options.simulation.physics.timestep*experiment_options.simulation.runtime.n_iterations
        kinematics            = self.generate_positions(
            tstop=kinematics_end,
            sampling_rate=1/kinematics_sampling,
            wlength=1,
            amp_deg=30.0,
            freq=0.5,
            TWL=12,
            nmotors=8,
            plot=False
        )
        joints_control_types  = {
            motor.joint_name: ControlType.from_string_list(
                motor.control_types,
            )
            for motor in animat_options.control.motors
        }
        joints_names_per_type = AnimatController.joints_from_control_types(
            joints_names=joints_names,
            joints_control_types=joints_control_types,
        )
        max_torques = {
            motor.joint_name: motor.limits_torque[1]
            for motor in animat_options.control.motors
        }
        max_torques_per_type = AnimatController.max_torques_from_control_types(
            joints_names=joints_names,
            max_torques=max_torques,
            joints_control_types=joints_control_types,
        )
        super().__init__(
            joints_names=joints_names_per_type,
            kinematics=kinematics,
            sampling=kinematics_sampling,
            indices=kinematics_indices,
            time_index=kinematics_time_index,
            invert_motors=kinematics_invert,
            degrees=kinematics_degrees,
            timestep=experiment_options.simulation.physics.timestep,
            n_iterations=experiment_options.simulation.runtime.n_iterations,
            animat_data=animat_data,
            max_torques=max_torques_per_type,
            init_time=kinematics_start,
            end_time=kinematics_end,
        )

        self.animat_data = animat_data
        self.animat_options = animat_options
        self.experiment_options = experiment_options
        self.config = Dict2Class(config)
        self.animat_i = animat_i

        self.n_joints = self.animat_data.sensors.joints.array.shape[1]
        self.n_iterations = self.animat_data.sensors.links.array.shape[0]


    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
            animat_i: int,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
        ):

        """From options"""
        return cls(
            animat_data = animat_data,
            animat_options = animat_options,
            experiment_options = experiment_options,
            config = config,
            animat_i = animat_i,
        )

    def generate_positions(
            self,
            tstop=3,
            sampling_rate=1000,
            wlength=1,
            amp_deg=20.0,
            freq=1.0,
            nmotors=8,
            TWL=14,
            plot=True
        ):
        amp = amp_deg * (np.pi / 180.0)
        times = np.expand_dims(np.arange(0, tstop, 1 / sampling_rate), axis=1)
        times_expanded = np.repeat(times, nmotors, axis=1)

        idxs = np.arange(nmotors)
        x = (idxs + 1) / nmotors
        factor = (1 + 0.323 * (x - 1) + 0.31 * (x ** 2 - 1))
        # factor[:-1] *= 0
        # factor[-1] = 4

        thetas = amp * factor * np.sin(
            2 * np.pi * (
                wlength * idxs / TWL - freq * times_expanded
            )
        )

        data = np.column_stack([times, thetas])

        if plot:
            x_plot = data[:, 0]
            y_plot = data[:, 1:]
            colors = plt.cm.jet(np.linspace(0, 1, y_plot.shape[1]))
            for i in range(y_plot.shape[1]):
                plt.plot(x_plot, y_plot[:, i], color=colors[i], label=f'Motor {i+1}')
            plt.legend()
            plt.show()

        return data

    def step(self, iteration, time, timestep):
        """Postions"""
        pass
