
from farms_core.model.control import AnimatController
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from farms_core.sensors.sensor_convention import sc
from farms_core.model.control import ControlType
import numpy as np
from lilytorch.util.rw import Dict2Class
from lilytorch.integration.kinematics import KinematicsController
import os
import matplotlib.pyplot as plt

class PositionController(KinematicsController):
    def __init__(self, animat_data, animat_options, experiment_options, config, animat_i):

        joints_names          = animat_options.control.joints_names()

        kinematics_sampling   = experiment_options.simulation.physics.timestep
        kinematics_indices    = range(1,len(joints_names)+1)
        kinematics_time_index = 0
        kinematics_invert     = False
        kinematics_degrees    = False
        kinematics_start      = 0.0
        kinematics_end        = experiment_options.simulation.physics.timestep*experiment_options.simulation.runtime.n_iterations
        kinematics            = self.generate_positions(
            animat_options=animat_options,
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
            animat_i=animat_i,
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

    def repeat_cycles(self, data, n_cycles):
        """
        Repeat joint position data for n_cycles.
        data: numpy array, first column is time, rest are joint positions (one cycle)
        n_cycles: int, number of cycles to repeat
        Returns: numpy array with repeated cycles, time is continuous
        """
        time = data[:, 0]
        joints = data[:, 1:]
        cycle_duration = time[-1] - time[0]
        result = []

        for i in range(n_cycles):
            new_time = time + i * cycle_duration
            cycle = np.column_stack((new_time, joints))
            result.append(cycle)

        return np.vstack(result)

    def generate_positions(
            self,
            animat_options=None,
            n_cycles=15,
            plot=True
        ):


        base_dir = os.path.dirname(__file__)
        data_file = os.path.join(base_dir, 'salamander_kinematics_2D_x15.csv')
        data = np.loadtxt(data_file, delimiter=',', skiprows=1)

        # amp_deg=40
        # TWL=39
        # freq=3.0
        # nmotors = 39
        # tstop=30
        # amp = amp_deg * (np.pi / 180.0)
        # times = np.expand_dims(np.arange(0, tstop, 0.01), axis=1)
        # times_expanded = np.repeat(times, nmotors, axis=1)

        # idxs   = np.arange(nmotors)
        # x      = (idxs + 1) / nmotors
        # c1     = +0.05,
        # c2     = -0.13,
        # c3     = +0.28
        # factor = c1+c2*x+c3*x**2

        # # factor[:-1] *= 0
        # # factor[-1] = 4

        # data=np.zeros((times.shape[0], nmotors+13))
        # data[:,0]=times[:,0]

        # data[:,1:40] = amp * factor * np.sin(
        #     2 * np.pi * (
        #         idxs / TWL - freq * times_expanded
        #     )
        # )


        # ======================= TEMPORARY FIX =======================
        # Set limb joint columns to their initial positions
        if animat_options is not None:
            joints_names = animat_options.control.joints_names()
            morph_joints = {j.name: j for j in animat_options.morphology.joints}
            for i, name in enumerate(joints_names):
                if 'leg' in name and name in morph_joints:
                    init_pos = morph_joints[name].initial[0]
                    data[:, i + 1] = init_pos  # col 0 is time
        # ======================= TEMPORARY FIX =======================

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
