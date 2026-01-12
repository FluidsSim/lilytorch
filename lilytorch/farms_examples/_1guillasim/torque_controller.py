

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

import importlib

class EkebergMuscleController(AnimatController):

    def __init__(self, animat_data, animat_options, experiment_options, config, animat_i, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.animat_data        = animat_data
        self.animat_options     = animat_options
        self.experiment_options = experiment_options
        self.config             = Dict2Class(config)
        self.animat_i           = animat_i
        self.n_joints           = len(self.animat_options.control.motors)
        self.n_iterations       = experiment_options.simulation.runtime.n_iterations

        # load class from string
        module_name, class_name = self.config.load_controller.rsplit('.', 1)
        module = importlib.import_module(module_name)
        controller_class = getattr(module, class_name)

        self.nn = controller_class(
            animat_data,
            self.n_joints,
            self.n_iterations,
            self.config
        )

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

        self.muscle_method=config.pop("method", "implicit")
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


    def step_muscles_explicit(self, M_diff, M_sum, iteration, time, timestep):

        """
        integrate the muscles explicitly
        """

        joints_pos = np.array(self.animat_data.sensors.joints.positions(iteration))
        joint_vel  = np.array(self.animat_data.sensors.joints.velocities(iteration))

        m_delta_phi = (self.offsets - joints_pos)

        active_torque          = self.muscle_coeff["alpha"] * M_diff
        stiffness_intermediate = self.muscle_coeff["beta"] * m_delta_phi
        active_stiffness       = M_sum*stiffness_intermediate
        passive_stiffness      = self.muscle_coeff["gamma"] * stiffness_intermediate
        damping                = -self.muscle_coeff["delta"] * joint_vel

        self.torque = active_torque + active_stiffness + passive_stiffness + damping

        self.animat_data.sensors.joints.array[iteration,:,sc.joint_cmd_torque] = self.torque

    def step_muscles_implicit(self, M_diff, M_sum, iteration, time, timestep):

        """
        integrate the muscles semi-implicitly
        i.e. all the stiffness and damping terms are treated implicitly, except for the active torque
        """
        self.kp     = self.muscle_coeff["beta"] * (M_sum + self.muscle_coeff["gamma"])  # conversion from ekeberg to pd controller
        self.kd     = self.muscle_coeff["delta"]
        self.torque = self.muscle_coeff["alpha"] * M_diff

        if self.log_torques:
            self.animat_data.sensors.joints.array[iteration,:,sc.joint_cmd_torque] = self.torque


    def before_step(self, task: Task, action, physics: Physics):
        time = physics.time()
        timestep = physics.timestep()
        index = task.iteration % task.buffer_size
        Mdiff, Msum = self.nn.step(iteration=index, time=time, timestep=timestep)
        self.step_muscles(Mdiff, Msum, iteration=index, time=time, timestep=timestep)


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