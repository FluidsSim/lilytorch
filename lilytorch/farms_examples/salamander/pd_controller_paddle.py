
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
        kinematics_indices    = range(1,len(joints_names)+1)
        kinematics_time_index = 0
        kinematics_invert     = False
        kinematics_degrees    = False
        kinematics_start      = 0.0
        kinematics_end        = experiment_options.simulation.physics.timestep*experiment_options.simulation.runtime.n_iterations
        kinematics            = self.generate_positions(
            tstop=kinematics_end,
            sampling_rate=1/kinematics_sampling,
            wlength=1,
            amp_deg=config["amp"],
            freq=config["freq"],
            TWL=config["twl"],
            limb_pose1=config["limb_pose1"],
            limb_pose2=config["limb_pose2"],
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


    def ComputeIK2D(self, elbowID, x, y, d1, d2):
        """
        elbowID=1 - elbow down
        elbowID=-1 - elbow up
        """
        D=((x**2+y**2-d1**2-d2**2)/(2*d1*d2)).clip(-1,1)
        elbow=elbowID*np.acos(D)
        k1=d1+d2*np.cos(elbow)
        k2=d2*np.sin(elbow)
        thigh=np.atan2(y,x)-np.atan2(k2,k1)
        return (thigh, elbow)

    def generate_positions(
            self,
            tstop=3,
            sampling_rate=1000,
            wlength=1,
            amp_deg=20.0,
            freq=1.0,
            TWL=14,
            limb_pose1=-0.4,
            limb_pose2=0.0,
            plot=True
        ):

        nmotors = 8

        times = np.expand_dims(np.arange(0, tstop, 1 / sampling_rate), axis=1)
        thetas_spine = np.zeros((times.shape[0], nmotors))

        radius = 0.005
        center = np.array([0.01, 0.00])
        phase_shift = 0 #np.pi
        x_traj_left = center[0] + radius * np.sin(2*np.pi*freq*times.T[0])
        y_traj_left = center[1] + radius * np.cos(2*np.pi*freq*times.T[0])

        x_traj_right = center[0] + radius * np.sin(2*np.pi*freq*times.T[0] + phase_shift)
        y_traj_right = center[1] + radius * np.cos(2*np.pi*freq*times.T[0] + phase_shift)

        l1 = np.array(0.005)
        l2 = np.array(0.005)

        thigh_left, elbow_left = self.ComputeIK2D(1, x_traj_left, y_traj_left, l1, l2)
        thigh_right, elbow_right = self.ComputeIK2D(1, x_traj_right, y_traj_right, l1, l2)

        limb_angles = np.zeros((times.shape[0], 8))

        limb_angles[:,0] = thigh_left
        limb_angles[:,1] = elbow_left
        limb_angles[:,2] = thigh_right
        limb_angles[:,3] = elbow_right

        # also hindlimbs
        limb_angles[:,4] = thigh_right
        limb_angles[:,5] = elbow_right
        limb_angles[:,6] = thigh_left
        limb_angles[:,7] = elbow_left


        data = np.column_stack([times, thetas_spine, limb_angles])


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
