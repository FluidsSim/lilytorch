"""PD position controller that replays joint trajectories from xlsx.

Loads slow/fast swimming joint trajectories from
``joints_positions_slow.xlsx`` / ``joints_positions_fast.xlsx`` and
drives the zebrafish through PD position control via
``KinematicsController``.

Patterned after
``lilytorch.examples._1guillasim.experiments.controller.PositionController``
but selects the trajectory file from a ``mode`` config field (``"slow"``
or ``"fast"``) — analogous to ``network.WaveController``.
"""

import os

import numpy as np
import pandas as pd

from farms_core.experiment.options import ExperimentOptions
from farms_core.model.control import AnimatController, ControlType
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions

from lilytorch.integration.kinematics import KinematicsController
from lilytorch.util.rw import Dict2Class


class PositionController(KinematicsController):
    """PD controller following a recorded joint trajectory."""

    def __init__(self, animat_data, animat_options, experiment_options, config, animat_i):

        config_obj = Dict2Class(config)

        data_folder = config["data_folder"]

        # Accept a direct file_path override (e.g. ep223 / ep248 model angles).
        if "file_path" in config:
            file_path = config["file_path"]
            if not os.path.isabs(file_path):
                file_path = os.path.join(data_folder, file_path)
        else:
            mode = config["mode"]
            if mode == "slow":
                file_path = os.path.join(data_folder, "joints_positions_slow_sigmoid.xlsx")
            elif mode == "fast":
                file_path = os.path.join(data_folder, "joints_positions_fast.xlsx")
            else:
                raise ValueError(f"Unknown mode {mode!r}. Expected 'slow' or 'fast'.")

        joints_names = animat_options.control.joints_names()

        kinematics_sampling = config.get(
            "kinematics_sampling",
            experiment_options.simulation.physics.timestep,
        )
        kinematics_invert = True
        kinematics_degrees = False  # xlsx values are already in radians
        kinematics_start = 0.0
        kinematics_end = (
            experiment_options.simulation.physics.timestep
            * experiment_options.simulation.runtime.n_iterations
        )

        kinematics = self.load_positions(file_path)

        # Trim/check columns vs. number of position joints
        n_pos_joints = len(joints_names)
        if kinematics.shape[1] < n_pos_joints:
            raise ValueError(
                f"Trajectory has {kinematics.shape[1]} joint columns but the model "
                f"declares {n_pos_joints} joints."
            )
        kinematics = kinematics[:, :n_pos_joints]

        joints_control_types = {
            motor.joint_name: ControlType.from_string_list(motor.control_types)
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
            indices=None,
            time_index=None,
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
        self.config = config_obj
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
        return cls(
            animat_data=animat_data,
            animat_options=animat_options,
            experiment_options=experiment_options,
            config=config,
            animat_i=animat_i,
        )

    @staticmethod
    def load_positions(file_path):
        """Read joint trajectory xlsx → ``(n_samples, n_joints)`` array."""
        df = pd.read_excel(file_path)
        return df.to_numpy(dtype=float)

    def step(self, iteration, time, timestep):
        """Positions are looked up via KinematicsController.positions()."""
        pass
