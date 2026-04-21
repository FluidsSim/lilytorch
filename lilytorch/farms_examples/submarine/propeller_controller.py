"""Torque controller for the submarine propeller joints.

By default this applies the same scalar torque ``tau`` to every
torque-controlled propeller joint. For contra-rotating setups, a
``joint_torques`` mapping can override the torque per joint so paired
rotors spin in opposite directions.
"""

from farms_core.experiment.options import ExperimentOptions
from farms_core.model.control import AnimatController, ControlType
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions


class PropellerController(AnimatController):
    def __init__(self, joints_names, max_torques, tau, joint_torques, animat_i):
        super().__init__(
            animat_i      = animat_i,
            joints_names  = joints_names,
            muscles_names = [],
            max_torques   = max_torques,
        )
        self.tau = float(tau)
        self.joint_torques = {
            str(joint_name): float(joint_tau)
            for joint_name, joint_tau in joint_torques.items()
        }

    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
            animat_i: int,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
    ):
        del experiment_options, animat_data
        joints_names = [
            motor.joint_name for motor in animat_options.control.motors
        ]
        joints_control_types = {
            motor.joint_name: ControlType.from_string_list(motor.control_types)
            for motor in animat_options.control.motors
        }
        max_torques = {
            motor.joint_name: motor.limits_torque[1]
            for motor in animat_options.control.motors
        }
        return cls(
            joints_names = AnimatController.joints_from_control_types(
                joints_names         = joints_names,
                joints_control_types = joints_control_types,
            ),
            max_torques  = AnimatController.max_torques_from_control_types(
                joints_names         = joints_names,
                max_torques          = max_torques,
                joints_control_types = joints_control_types,
            ),
            tau          = config.get("tau", 0.0),
            joint_torques = config.get("joint_torques", {}),
            animat_i     = animat_i,
        )

    def torques(self, iteration, time, timestep):
        del iteration, time, timestep
        return {
            joint: self.joint_torques.get(joint, self.tau)
            for joint in self.joints_names[ControlType.TORQUE]
        }
