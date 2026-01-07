
from matplotlib.pyplot import step
from farms_core.model.control import AnimatController
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from farms_core.sensors.sensor_convention import sc
from farms_core.model.control import ControlType
from dm_control.rl.control import Task
from dm_control.mjcf.physics import Physics

import numpy as np
from lilytorch.util.rw import Dict2Class


class WaveController(AnimatController):

    def __init__(self, animat_data, animat_options, experiment_options, config, animat_i, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.animat_data = animat_data
        self.animat_options = animat_options
        self.experiment_options = experiment_options
        self.config = Dict2Class(config)
        self.animat_i = animat_i

        self.n_joints = self.animat_data.sensors.joints.array.shape[1]
        self.n_iterations = self.animat_data.sensors.links.array.shape[0]

        self.state    = np.zeros((self.n_iterations, 2*self.n_joints))

        self.muscle_l = 2*np.arange(0, self.n_joints) # indexes of the left muscle
        self.muscle_r = self.muscle_l+1

        self.config.amplitudes_left = self.config.amp+self.config.bias
        self.config.amplitudes_right = self.config.amp-self.config.bias

        # Define muscle parameters for each joint
        muscles_params = [
            {
            "alpha": 1.0,
            "beta": 0.001,
            "gamma": 1600,
            "delta": 0.1
            } for _ in range(self.n_joints)
        ]

        self.muscle_coeff = {
            "alpha": np.array([joint["alpha"] for joint in muscles_params]),
            "beta": np.array([joint["beta"] for joint in muscles_params]),
            "gamma": np.array([joint["gamma"] for joint in muscles_params]),
            "delta": np.array([joint["delta"] for joint in muscles_params]),
        }
        self.torque = np.zeros(self.n_joints)
        self.offsets = np.zeros(self.n_joints)

        self.muscle_method=config["method"]
        if self.muscle_method == "explicit":
            self.step_muscles = self.step_muscles_explicit
        elif self.muscle_method == "implicit":
            self.step_muscles = self.step_muscles_implicit

        self.log_torques = config.get("log_torques", True)


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
        joints_names = [
            joint.name
            for joint in animat_options.morphology.joints
        ]
        joints_control_types: dict[str, list[ControlType]] = {
            motor.joint_name: ControlType.from_string_list(motor.control_types)
            for motor in animat_options.control.motors
        }
        return cls(
            joints_names=AnimatController.joints_from_control_types(
                joints_names=joints_names,
                joints_control_types=joints_control_types,
            ),
            muscles_names=[],
            max_torques=AnimatController.max_torques_from_control_types(
                joints_names=joints_names,
                max_torques={
                    motor.joint_name: motor.limits_torque[1]
                    for motor in animat_options.control.motors
                },
                joints_control_types=joints_control_types,
            ),
            animat_data = animat_data,
            animat_options = animat_options,
            experiment_options = experiment_options,
            config = config,
            animat_i = animat_i,
        )

    def step_controller(self, iteration, time, timestep):
        """Compute neural activity"""
        time     = iteration * timestep
        aux_sine = np.sin(
            2*np.pi * ( self.config.freq*time - self.config.twl*np.arange(self.n_joints)/self.n_joints )
        )

        self.state[iteration, self.muscle_l]  = self.config.amplitudes_left * (1+aux_sine)/2
        self.state[iteration, self.muscle_r]  = self.config.amplitudes_right * (1-aux_sine)/2



    def step_muscles_explicit(self, iteration, time, timestep):

        """
        integrate the muscles explicitly
        """

        joints_pos = np.array(self.animat_data.sensors.joints.positions(iteration))
        joint_vel  = np.array(self.animat_data.sensors.joints.velocities(iteration))
        M_diff     = (self.state[iteration,self.muscle_l] - self.state[iteration,self.muscle_r])
        M_sum      = (self.state[iteration,self.muscle_l] + self.state[iteration,self.muscle_r])

        m_delta_phi = (self.offsets - joints_pos)

        active_torque          = self.muscle_coeff["alpha"] * M_diff
        stiffness_intermediate = self.muscle_coeff["beta"] * m_delta_phi
        active_stiffness       = M_sum*stiffness_intermediate
        passive_stiffness      = self.muscle_coeff["gamma"] * stiffness_intermediate
        damping                = -self.muscle_coeff["delta"] * joint_vel

        self.torque = active_torque + active_stiffness + passive_stiffness + damping

        self.animat_data.sensors.joints.array[iteration,:,sc.joint_cmd_torque] = self.torque

    def step_muscles_implicit(self, iteration, time, timestep):

        """
        integrate the muscles semi-implicitly
        i.e. all the stiffness and damping terms are treated implicitly, except for the active torque
        """

        M_diff     = (self.state[iteration,self.muscle_l] - self.state[iteration,self.muscle_r])
        M_sum      = (self.state[iteration,self.muscle_l] + self.state[iteration,self.muscle_r])

        self.kp = self.muscle_coeff["beta"] * (M_sum + self.muscle_coeff["gamma"]) # conversion from ekeberg to pd controller
        self.kd = self.muscle_coeff["delta"]
        self.torque = self.muscle_coeff["alpha"] * M_diff

        if self.log_torques:
            self.animat_data.sensors.joints.array[iteration,:,sc.joint_cmd_torque] = self.torque


    def before_step(self, task: Task, action, physics: Physics):
        time = physics.time()
        timestep = physics.timestep()
        index = task.iteration % task.buffer_size
        self.step_controller(iteration=index, time=time, timestep=timestep)
        self.step_muscles(iteration=index, time=time, timestep=timestep)


    def springrefs(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Spring references"""
        output = {}
        if self.muscle_method == "implicit":
            output={
                joint: self.offsets[idx]
                for idx, joint in enumerate(self.joints_names[ControlType.TORQUE])
            }
        return output

    def springcoefs(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Spring coefficients"""
        output = {}
        if self.muscle_method == "implicit":
            output={
                joint: self.kp[idx]
                for idx, joint in enumerate(self.joints_names[ControlType.TORQUE])
            }
        return output

    def dampingcoefs(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Damping coefficients"""
        output = {}
        if self.muscle_method == "implicit":
            output={
                joint: self.kd[idx]
                for idx, joint in enumerate(self.joints_names[ControlType.TORQUE])
            }
        return output

    def torques(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Torques"""
        return {
            joint: self.torque[idx]
            for idx, joint in enumerate(self.joints_names[ControlType.TORQUE])
        }