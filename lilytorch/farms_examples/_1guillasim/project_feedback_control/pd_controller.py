
from farms_core.model.control import AnimatController
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from farms_core.model.control import ControlType
import numpy as np
from lilytorch.util.rw import Dict2Class


class PositionController(AnimatController):
    def __init__(self, animat_data, animat_options, experiment_options, config, animat_i):

        # --- Joint control setup (same as before) ---
        joints_names = animat_options.control.joints_names()
        joints_control_types = {
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

        # --- Parent init (AnimatController, no pre-computed kinematics) ---
        super().__init__(
            animat_i=animat_i,
            joints_names=joints_names_per_type,
            muscles_names=[],
            max_torques=max_torques_per_type,
        )

        self.animat_data = animat_data
        self.animat_options = animat_options
        self.experiment_options = experiment_options
        self.config = Dict2Class(config)
        self.animat_i = animat_i

        # --- Reference trajectory parameters ---
        self._amp = config["amp"] * (np.pi / 180.0)  # deg → rad
        self._freq = config["freq"]
        self._twl = config["twl"]
        self._nmotors = 8
        self._wlength = 1
        self._tau_rise = 1.0

        # Pre-compute the amplitude envelope factor per motor
        x = (np.arange(self._nmotors) + 1) / self._nmotors
        self._factor = 1.0 + 0.323 * (x - 1.0) + 0.31 * (x ** 2 - 1.0)

        # --- Sensor metadata ---
        self.n_joints = self.animat_data.sensors.joints.array.shape[1]

        # Map position-controlled joint names → sensor array indices
        # self._pos_sensor_indices = self._build_sensor_mapping()



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

    def _build_sensor_mapping(self):
        """Build array of sensor indices for position-controlled joints."""
        pos_joint_names = self.joints_names[ControlType.POSITION]
        sensor_names = list(self.animat_data.sensors.joints.names)
        return np.array([
            sensor_names.index(name) for name in pos_joint_names
        ], dtype=int)

    def _envelope(self, t, joint_pos, joint_vel):
        """Compute open-loop sine-wave reference positions at a single time t.

        Returns a 1D numpy array of length ``self._nmotors`` (radians).
        """
        idxs = np.arange(self._nmotors)
        phase = 2.0 * np.pi * (
            self._wlength * idxs / self._twl - self._freq * t
        )
        envelope = 1.0 - np.exp(-t / self._tau_rise)
        return -self._amp * self._factor * np.sin(phase) * envelope


    def positions(self, iteration, time, timestep):
        """Return reference joint positions (implicit control).

        The sine-wave reference is evaluated on-the-fly at the current *time*.
        MuJoCo's built-in position actuators (kp/kv gains configured in the
        MJCF model) handle the low-level tracking.

        Actual joint state is read from sensors and stored in
        ``self.joint_positions`` / ``self.joint_velocities`` for optional
        feedback use (e.g. in a subclass or external monitor).
        """
        # Read actual joint state for feedback (only position-controlled joints)
        joint_pos = np.array(self.animat_data.sensors.joints.positions(iteration))
        joint_vel = np.array(self.animat_data.sensors.joints.velocities(iteration))
        # Implicit control: return reference only (MuJoCo actuators do the rest)
        ref = self._envelope(time, joint_pos=joint_pos, joint_vel=joint_vel)
        return dict(zip(
            self.joints_names[ControlType.POSITION],
            ref,
        ))
