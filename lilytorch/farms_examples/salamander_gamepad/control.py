"""Control"""

import numpy as np
from dm_control.mjcf.physics import Physics
from farms_mujoco.simulation.task import ExperimentTask

try:
    from pynput import keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    pynput_keyboard = None
    PYNPUT_AVAILABLE = False

import farms_core.pylog as pylog
from farms_core.model.data import AnimatData
from farms_core.array.types import NDARRAY_V1
from farms_core.model.options import AnimatOptions
from farms_core.model.extensions import AnimatExtension
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.control import AnimatController, ControlType
from farms_mujoco.simulation.extensions import create_cylinder
from lilytorch.util.rw import Dict2Class

from ...integration.gamepad import GamepadHandler


class _KeyboardInputState:
    """Process-wide keyboard state shared by local controllers."""

    def __init__(self):
        self._listener = None
        self._pressed: set[str] = set()
        self._started = False
        self._announced = False

    def start(self) -> bool:
        if self._started or not PYNPUT_AVAILABLE:
            return self._started

        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        self._started = True
        return True

    def announce_once(self):
        if self._announced:
            return
        pylog.info(
            "Keyboard controls active: keypad 5 forward, keypad 2 brake, "
            "keypad 1/3 turn, "
            "Q walk preset, E swim preset, R/F fine speed trim."
        )
        self._announced = True

    def snapshot(self) -> dict[str, float | bool]:
        pressed = self._pressed
        return {
            "active": bool(pressed),
            "js_left_x": float("kp_3" in pressed) - float("kp_1" in pressed),
            "js_left_y": float("kp_2" in pressed) - float("kp_5" in pressed),
            "trigger_left": float("f" in pressed),
            "trigger_right": float("r" in pressed),
            "walk": "q" in pressed,
            "swim": "e" in pressed,
        }

    def _on_press(self, key):
        key_name = self._normalize_key(key)
        if key_name is not None:
            self._pressed.add(key_name)

    def _on_release(self, key):
        key_name = self._normalize_key(key)
        if key_name is not None:
            self._pressed.discard(key_name)

    @staticmethod
    def _normalize_key(key):
        try:
            if key.char is not None:
                key_char = key.char.lower()
                key_vk = getattr(key, "vk", None)
                if key_char in {"1", "2", "3", "5"} and key_vk is None:
                    return f"kp_{key_char}"
                if key_char in {"1", "2", "3", "5"}:
                    return None
                return key_char
        except AttributeError:
            return None
        return None


_KEYBOARD_INPUT = _KeyboardInputState()


class PositionController(AnimatController):
    """Gamepad- or keyboard-driven position controller for 2-D swimming."""

    def __init__(
            self,
            animat_data,
            animat_options,
            experiment_options,
            config,
            animat_i,
            joints_names,
            max_torques,
    ):
        super().__init__(
            animat_i=animat_i,
            joints_names=joints_names,
            muscles_names=[],
            max_torques=max_torques,
        )

        self.animat_data = animat_data
        self.animat_options = animat_options
        self.experiment_options = experiment_options
        self.config = Dict2Class(config)
        self.animat_i = animat_i

        position_joints = list(self.joints_names[ControlType.POSITION])
        self.body_joint_names = sorted(
            [
                joint_name for joint_name in position_joints
                if joint_name.startswith("joint_body_")
            ],
            key=self._body_joint_index,
        )
        self.limb_joint_names = [
            joint_name for joint_name in position_joints
            if not joint_name.startswith("joint_body_")
        ]
        if not self.body_joint_names:
            raise ValueError(
                "The salamander gamepad controller requires body joints named "
                "'joint_body_*'."
            )

        self.base_amp = np.deg2rad(float(self.config.amp))
        self.base_freq = float(self.config.freq)
        self.wave_length = float(getattr(self.config, "wlength", 1.0))
        self.twl = float(self.config.twl)
        self.limb_pose1 = float(self.config.limb_pose1)
        self.limb_pose2 = float(self.config.limb_pose2)
        self.turn_gain = float(getattr(self.config, "turn_gain", 0.2))
        self.nominal_drive = float(getattr(self.config, "nominal_drive", 4.0))
        self.max_drive_scale = float(getattr(self.config, "max_drive_scale", 1.25))
        self.walk_drive = float(getattr(self.config, "walk_drive", 2.0))
        self.swim_drive = float(getattr(self.config, "swim_drive", 4.0))
        self.drive = float(getattr(self.config, "drive", self.swim_drive))
        self.max_drive = float(getattr(self.config, "max_drive", 5.0))
        self.body_amp_profile = self._body_amplitude_profile(len(self.body_joint_names))
        self.body_turn_profile = np.linspace(
            0.25,
            1.0,
            len(self.body_joint_names),
            dtype=float,
        )

        self.phase = 0.0
        self.last_iteration = -1
        self.cached_positions = {
            joint_name: 0.0 for joint_name in position_joints
        }

        self.js_left_x = 0.0
        self.js_left_y = 0.0
        self.tr_left = 0.0
        self.tr_right = 0.0
        self.shoulders = (False, False)
        self.dpad_left = False
        self.dpad_right = False
        self.dpad_up = False
        self.dpad_down = False
        self.keyboard = None

        try:
            self.gamepad = GamepadHandler(
                deadzone=float(getattr(self.config, "deadzone", 0.1))
            )
        except RuntimeError as exc:
            pylog.warning(f"Gamepad controller disabled: {exc}")
            self.gamepad = None

        if bool(getattr(self.config, "keyboard_enabled", True)):
            if _KEYBOARD_INPUT.start():
                self.keyboard = _KEYBOARD_INPUT
                self.keyboard.announce_once()
            elif not PYNPUT_AVAILABLE:
                pylog.warning(
                    "Keyboard control disabled: install 'pynput' to use "
                    "keyboard-only control."
                )

    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
            animat_i: int,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
    ):
        """Build the controller from FARMS options."""
        joints_names = [
            motor.joint_name for motor in animat_options.control.motors
        ]
        joints_control_types = {
            motor.joint_name: ControlType.from_string_list(
                motor.control_types,
            )
            for motor in animat_options.control.motors
        }
        max_torques = {
            motor.joint_name: motor.limits_torque[1]
            for motor in animat_options.control.motors
        }
        return cls(
            animat_data=animat_data,
            animat_options=animat_options,
            experiment_options=experiment_options,
            config=config,
            animat_i=animat_i,
            joints_names=AnimatController.joints_from_control_types(
                joints_names=joints_names,
                joints_control_types=joints_control_types,
            ),
            max_torques=AnimatController.max_torques_from_control_types(
                joints_names=joints_names,
                max_torques=max_torques,
                joints_control_types=joints_control_types,
            ),
        )

    @staticmethod
    def _body_joint_index(joint_name: str) -> int:
        return int(joint_name.rsplit("_", maxsplit=1)[-1])

    @staticmethod
    def _body_amplitude_profile(n_body_joints: int) -> np.ndarray:
        indices = np.arange(n_body_joints, dtype=float)
        x_coord = (indices + 1.0) / n_body_joints
        return 0.05 - 0.13*x_coord + 0.28*x_coord**2

    def _update_input_state(self):
        self.js_left_x = 0.0
        self.js_left_y = 0.0
        self.tr_left = 0.0
        self.tr_right = 0.0

        if self.gamepad is not None and self.gamepad.update():
            state = self.gamepad.get_state()
            left_shoulder = bool(state.button_shoulder_left)
            right_shoulder = bool(state.button_shoulder_right)
            if left_shoulder and not self.shoulders[0]:
                self.drive = self.walk_drive
            if right_shoulder and not self.shoulders[1]:
                self.drive = self.swim_drive
            self.shoulders = (left_shoulder, right_shoulder)

            self.dpad_left = bool(state.button_dpad_left)
            self.dpad_right = bool(state.button_dpad_right)
            self.dpad_up = bool(state.button_dpad_up)
            self.dpad_down = bool(state.button_dpad_down)

            dpad_x = float(self.dpad_right) - float(self.dpad_left)
            dpad_y = float(self.dpad_down) - float(self.dpad_up)
            if abs(state.js_left_x) > 1e-6:
                self.js_left_x = float(state.js_left_x)
            else:
                self.js_left_x = dpad_x
            if abs(state.js_left_y) > 1e-6:
                self.js_left_y = float(state.js_left_y)
            else:
                self.js_left_y = dpad_y
            self.tr_left = float(state.trigger_left)
            self.tr_right = float(state.trigger_right)

        if self.keyboard is not None:
            keyboard_state = self.keyboard.snapshot()
            if keyboard_state["walk"]:
                self.drive = self.walk_drive
            if keyboard_state["swim"]:
                self.drive = self.swim_drive
            if keyboard_state["active"]:
                self.js_left_x = float(keyboard_state["js_left_x"])
                self.js_left_y = float(keyboard_state["js_left_y"])
                self.tr_left = float(keyboard_state["trigger_left"])
                self.tr_right = float(keyboard_state["trigger_right"])

    def _mixed_drives(self) -> tuple[float, float]:
        drive_diff = -2.0*self.js_left_x
        drive_mult = np.clip(-self.js_left_y + abs(self.js_left_x), 0.0, 1.0)
        drive = self.drive + 0.5*self.tr_right - 0.5*self.tr_left
        drive_left = (drive + drive_diff)*drive_mult
        drive_right = (drive - drive_diff)*drive_mult
        drive_mean = 0.5*(drive_left + drive_right)

        if 1.0 < drive_mean <= 3.0:
            drive_left = np.clip(drive_left, 1.01, 2.99)
            drive_right = np.clip(drive_right, 1.01, 2.99)
        elif 3.0 < drive_mean <= 5.0:
            drive_left = np.clip(drive_left, 3.01, 4.99)
            drive_right = np.clip(drive_right, 3.01, 4.99)
        else:
            drive_left = np.clip(drive_left, 0.0, self.max_drive)
            drive_right = np.clip(drive_right, 0.0, self.max_drive)

        return float(drive_left), float(drive_right)

    def _body_targets(self, timestep: float) -> np.ndarray:
        drive_left, drive_right = self._mixed_drives()
        mean_drive = 0.5*(drive_left + drive_right)
        if mean_drive <= 0.0:
            return np.zeros(len(self.body_joint_names), dtype=float)

        drive_scale = np.clip(
            mean_drive / self.nominal_drive,
            0.0,
            self.max_drive_scale,
        )
        turn_scale = np.clip(
            0.5*(drive_left - drive_right) / self.nominal_drive,
            -1.0,
            1.0,
        )
        self.phase = (self.phase + 2.0*np.pi*self.base_freq*drive_scale*timestep) % (2.0*np.pi)

        indices = np.arange(len(self.body_joint_names), dtype=float)
        wave = self.base_amp*drive_scale*self.body_amp_profile*np.sin(
            2.0*np.pi*self.wave_length*indices/self.twl - self.phase
        )
        turn_bias = self.turn_gain*turn_scale*self.body_turn_profile
        return wave + turn_bias

    def _limb_target(self, joint_name: str) -> float:
        if joint_name.endswith("_0"):
            return self.limb_pose1
        if joint_name.endswith("_3"):
            return self.limb_pose2
        return 0.0

    def _update_positions(self, timestep: float):
        self._update_input_state()

        joint_positions = {}
        for joint_name, joint_target in zip(
                self.body_joint_names,
                self._body_targets(timestep),
        ):
            joint_positions[joint_name] = float(joint_target)
        for joint_name in self.limb_joint_names:
            joint_positions[joint_name] = self._limb_target(joint_name)

        self.cached_positions = {
            joint_name: joint_positions.get(joint_name, 0.0)
            for joint_name in self.joints_names[ControlType.POSITION]
        }

    def positions(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Return PD joint targets driven by the current gamepad state."""
        del time
        if iteration != self.last_iteration:
            self._update_positions(timestep)
            self.last_iteration = iteration
        return dict(self.cached_positions)


class DriveControl(AnimatExtension):
    """Drive control"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gamepad = GamepadHandler(deadzone=0.1)
        self.n_drives: int = 0
        self.drive = 2
        self.drive_mult = 0
        self.drive_diff = 0
        self.drive_vector = np.array([])
        self.viewer = None
        self.shoulders = (False, False)
        self.azimuth = 0
        self.elevation = 0
        self.distance = 0
        self.distance_diff = 1.0
        self.js_left_x = 0
        self.js_left_y = 0
        self.js_right_x = 0
        self.js_right_y = 0
        self.tr_left = 0
        self.tr_right = 0

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
        return cls(config)

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialize episode"""
        drives = task.data.animats[0].network.drives
        self.n_drives: int = np.shape(drives.array)[1]
        self.drive_vector = np.ones(self.n_drives)
        self.viewer = task.viewer
        if self.viewer is not None:
            self.azimuth = self.viewer.cam.azimuth
            self.distance = self.viewer.cam.distance
            self.elevation = self.viewer.cam.elevation

    def before_step(self, task: ExperimentTask, action, physics: Physics):
        """Before step"""
        del action
        timestep = physics.timestep()
        # Polling-based gamepad update
        # self.gamepad.update()
        # state = self.gamepad.get_state()
        # self.drive_diff = -1*state.js_left_x
        # self.drive_mult = -state.js_left_y + abs(state.js_left_x)
        # self.drive_mult = min(1, self.drive_mult)
        # self.drive_mult = max(0, self.drive_mult)
        # if state.button_shoulder_left and not self.shoulders[0]:
        #     self.drive -= 0.25
        # if state.button_shoulder_right and not self.shoulders[1]:
        #     self.drive += 0.25
        # self.shoulders = (state.button_shoulder_left, state.button_shoulder_right)
        # self.azimuth -= 1.5*state.js_right_x
        # self.elevation -= 1.0*state.js_right_y

        # if self.viewer:
        #     self.viewer.cam.azimuth = self.azimuth
        #     # self.viewer.cam.distance = self.distance
        #     self.viewer.cam.elevation = self.elevation

        # Event-based gamepad update
        for event in self.gamepad.update_events():
            event_name = self.gamepad.state.EVENT_NAMES.get(event.type)
            if event_name == 'button_down':
                button_name = self.gamepad.state.BUTTON_NAMES.get(event.cbutton.button)
                match button_name:
                    case 'button_shoulder_left':
                        self.drive = 2
                    case 'button_shoulder_right':
                        self.drive = 4
                    case 'button_dpad_up':
                        self.distance_diff = 0.99
                    case 'button_dpad_down':
                        self.distance_diff = 1.01

            elif event_name == 'button_up':
                button_name = self.gamepad.state.BUTTON_NAMES.get(event.cbutton.button)
                match button_name:
                    case 'button_dpad_down' | 'button_dpad_up':
                        self.distance_diff = 1.0

            elif event_name == 'axis_motion':
                axis_name = self.gamepad.state.AXIS_NAMES.get(event.caxis.axis)
                match axis_name:
                    case 'trigger_left':
                        self.tr_left = self.gamepad.state.trigger_left
                    case 'trigger_right':
                        self.tr_right = self.gamepad.state.trigger_right
                    case 'js_left_x' | 'js_left_y': # Update drive difference and multiplier
                        self.js_left_x = self.gamepad.state.js_left_x
                        self.js_left_y = self.gamepad.state.js_left_y
                    case 'js_right_x' | 'js_right_y': # Update camera
                        self.js_right_x = self.gamepad.state.js_right_x
                        self.js_right_y = self.gamepad.state.js_right_y

            self.drive_diff = -2*self.js_left_x
            self.drive_mult = -1*self.js_left_y + abs(self.js_left_x)
            self.drive_mult = min(1, self.drive_mult)
            self.drive_mult = max(0, self.drive_mult)

            if self.viewer:
                self.viewer.cam.azimuth = self.azimuth
                self.viewer.cam.elevation = self.elevation

        self.azimuth -= 500*self.js_right_x*timestep
        self.elevation -= 300*self.js_right_y*timestep
        if self.viewer:
            self.viewer.cam.distance *= 1 + 1000*(self.distance_diff-1)*timestep

        # Set drive
        drive = self.drive + 0.5*self.tr_right - 0.5*self.tr_left
        drive_left = (drive + self.drive_diff)*self.drive_mult
        drive_right = (drive - self.drive_diff)*self.drive_mult
        drive_mean = 0.5*(drive_left + drive_right)
        if 1 < drive_mean <= 3:
            drive_left = max(1.01, min(2.99, drive_left))
            drive_right = max(1.01, min(2.99, drive_right))
        elif 3 < drive_mean <= 5:
            drive_left = max(3.01, min(4.99, drive_left))
            drive_right = max(3.01, min(4.99, drive_right))
        drives = task.data.animats[0].network.drives
        iteration = task.iteration
        self.set_left_drives(drives, iteration, drive_left*self.drive_vector)
        self.set_right_drives(drives, iteration, drive_right*self.drive_vector)

    def set_left_drives(
            self,
            drives,
            iteration: int,
            values,
            brain: bool = True,
    ):
        """Set forward drives"""
        for index in drives.spine_left_indices:
            drives.array[iteration, index] = values[index]
        if brain:
            for index in drives.brain_left_indices:
                drives.array[iteration, index] = values[index]

    def set_right_drives(
            self,
            drives,
            iteration: int,
            values,
            brain: bool = True,
    ):
        """Set right drives"""
        for index in drives.spine_right_indices:
            drives.array[iteration, index] = values[index]
        if brain:
            for index in drives.brain_right_indices:
                drives.array[iteration, index] = values[index]

# Similar to DriveControl but with additional feedback from foot contact sensors aside from gamepad control
class DriveControl_Feedback(AnimatExtension):
    """Drive control"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gamepad = GamepadHandler(deadzone=0.1)
        self.n_drives: int = 0
        self.drive = 2
        self.drive_override = False # Whether the drive is overridden by gamepad input. If True, the drive cannot be set by the foot contact feedback until the same gamepad input is received again to release the hold.
        self.drive_mult = 0
        self.drive_diff = 0
        self.drive_vector = np.array([])
        self.viewer = None
        self.shoulders = (False, False)
        self.azimuth = 0
        self.elevation = 0
        self.distance = 0
        self.distance_diff = 1.0
        self.js_left_x = 0
        self.js_left_y = 0
        self.js_right_x = 0
        self.js_right_y = 0
        self.tr_left = 0
        self.tr_right = 0

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
        return cls(config)

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialize episode"""
        drives = task.data.animats[0].network.drives
        self.n_drives: int = np.shape(drives.array)[1]
        self.drive_vector = np.ones(self.n_drives)
        self.viewer = task.viewer
        if self.viewer is not None:
            self.azimuth = self.viewer.cam.azimuth
            self.distance = self.viewer.cam.distance
            self.elevation = self.viewer.cam.elevation
        # Find the indices of the 4 feet in the sensor array, which has patterns 'link_leg_*_*_1' in the contacts.names
        self.id_feet = []
        for i, name in enumerate([task.data.animats[0].sensors.contacts.names[i][0] for i in range(len(task.data.animats[0].sensors.contacts.names))]):
            if 'link_leg' in name and name.endswith('_1'):
                self.id_feet.append(i)

    def before_step(self, task: ExperimentTask, action, physics: Physics):
        """Before step"""
        del action
        timestep = physics.timestep()
        iteration = task.iteration

        # Get foot contact sensor readings and adjust drive accordingly
        if not self.drive_override:
            # Get the contact sensors for the four feet. For the definition of contact sensor array, see farms_core/sensors/sensor_convension
            GRF_TotalZs = task.data.animats[0].sensors.contacts.array.base[
                    iteration,
                    self.id_feet,
                    8
                ]
            # Rule of drive adjustment: if the foot is in contact with the ground (sum of GRF_TotalZs > small threshold), set drive to the walking region (1~3), otherwise set the drive to the swimming region(3~5).
            if np.sum(np.abs(GRF_TotalZs)) > 1:
                if self.drive != 2: # Only log when there is a change of mode
                    pylog.info('Switched to walking mode by foot contact feedback')
                self.drive = 2
            else:
                if self.drive != 4: # Only log when there is a change of mode
                    pylog.info('Switched to swimming mode by foot contact feedback')
                self.drive = 4

        # Event-based gamepad update
        for event in self.gamepad.update_events():
            event_name = self.gamepad.state.EVENT_NAMES.get(event.type)
            if event_name == 'button_down':
                button_name = self.gamepad.state.BUTTON_NAMES.get(event.cbutton.button)
                match button_name:
                    case 'button_shoulder_left':
                        if self.drive != 2: # Force switching of mode, also set the override flag to True
                            self.drive = 2
                            self.drive_override = True
                            pylog.info('Forced walking mode by gamepad input')
                        else: # If the mode is already in walking, release the override flag to allow the foot contact feedback to adjust the drive again
                            self.drive_override = False
                            pylog.info('Released override of walking mode by gamepad input')
                    case 'button_shoulder_right':
                        if self.drive != 4: # Force switching of mode, also set the override flag to True
                            self.drive = 4
                            self.drive_override = True
                            pylog.info('Forced swimming mode by gamepad input')
                        else: # If the mode is already in swimming, release the override flag to allow the foot contact feedback to adjust the drive again
                            self.drive_override = False
                            pylog.info('Released override of swimming mode by gamepad input')
                    case 'button_dpad_up':
                        self.distance_diff = 0.99
                    case 'button_dpad_down':
                        self.distance_diff = 1.01

            elif event_name == 'button_up':
                button_name = self.gamepad.state.BUTTON_NAMES.get(event.cbutton.button)
                match button_name:
                    case 'button_dpad_down' | 'button_dpad_up':
                        self.distance_diff = 1.0

            elif event_name == 'axis_motion':
                axis_name = self.gamepad.state.AXIS_NAMES.get(event.caxis.axis)
                match axis_name:
                    case 'trigger_left':
                        self.tr_left = self.gamepad.state.trigger_left
                    case 'trigger_right':
                        self.tr_right = self.gamepad.state.trigger_right
                    case 'js_left_x' | 'js_left_y': # Update drive difference and multiplier
                        self.js_left_x = self.gamepad.state.js_left_x
                        self.js_left_y = self.gamepad.state.js_left_y
                    case 'js_right_x' | 'js_right_y': # Update camera
                        self.js_right_x = self.gamepad.state.js_right_x
                        self.js_right_y = self.gamepad.state.js_right_y

            self.drive_diff = -2*self.js_left_x
            self.drive_mult = -1*self.js_left_y + abs(self.js_left_x)
            self.drive_mult = min(1, self.drive_mult)
            self.drive_mult = max(0, self.drive_mult)

            if self.viewer:
                self.viewer.cam.azimuth = self.azimuth
                self.viewer.cam.elevation = self.elevation

        self.azimuth -= 500*self.js_right_x*timestep
        self.elevation -= 300*self.js_right_y*timestep
        if self.viewer:
            self.viewer.cam.distance *= 1 + 1000*(self.distance_diff-1)*timestep

        # Set drive
        drive = self.drive + 0.5*self.tr_right - 0.5*self.tr_left
        drive_left = (drive + self.drive_diff)*self.drive_mult
        drive_right = (drive - self.drive_diff)*self.drive_mult
        drive_mean = 0.5*(drive_left + drive_right)
        if 1 < drive_mean <= 3:
            drive_left = max(1.01, min(2.99, drive_left))
            drive_right = max(1.01, min(2.99, drive_right))
        elif 3 < drive_mean <= 5:
            drive_left = max(3.01, min(4.99, drive_left))
            drive_right = max(3.01, min(4.99, drive_right))
        drives = task.data.animats[0].network.drives
        self.set_left_drives(drives, iteration, drive_left*self.drive_vector)
        self.set_right_drives(drives, iteration, drive_right*self.drive_vector)

    def set_left_drives(
            self,
            drives,
            iteration: int,
            values,
            brain: bool = True,
    ):
        """Set forward drives"""
        for index in drives.spine_left_indices:
            drives.array[iteration, index] = values[index]
        if brain:
            for index in drives.brain_left_indices:
                drives.array[iteration, index] = values[index]

    def set_right_drives(
            self,
            drives,
            iteration: int,
            values,
            brain: bool = True,
    ):
        """Set right drives"""
        for index in drives.spine_right_indices:
            drives.array[iteration, index] = values[index]
        if brain:
            for index in drives.brain_right_indices:
                drives.array[iteration, index] = values[index]

class SetupController(AnimatController):
    """Setup controller"""

    def __init__(
            self,
            animat_i: int,
            joints_names: tuple[list[str], ...],
            muscles_names: tuple[str, ...],
            max_torques: tuple[NDARRAY_V1, ...],
            joints_config,
    ):
        super().__init__(
            animat_i=animat_i,
            joints_names=joints_names,
            muscles_names=muscles_names,
            max_torques=max_torques,
        )
        self.joints_config = joints_config

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
            animat_i = animat_i,
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
            joints_config=config,
        )

    def positions(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Positions"""
        assert iteration >= 0
        assert time >= 0
        assert timestep > 0
        return self.joints_config


class Targets2Reach(AnimatExtension):
    """Drive control"""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cylinder = []
        self.animat_id = kwargs.pop('animat_id', 0)
        self.links: LinkSensorArray | None = None

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
            config=config,
            animat_id=animat_i,
        )

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialize episode"""
        del physics
        self.links = task.data.animats[self.animat_id].sensors.links
        self.viewer = task.viewer
        if self.viewer is not None:
            self.azimuth = self.viewer.cam.azimuth
            self.distance = self.viewer.cam.distance
            self.elevation = self.viewer.cam.elevation
            self.cylinders = [
                create_cylinder(
                    self.viewer,
                    pos=target['position'],
                    size=target['size'],
                    rgba=target['rgba1'],
                )
                for target in self.config['targets']
            ]

    def after_step(self, task: ExperimentTask, physics: Physics):
        """After step"""
        del physics
        if self.viewer:
            pos = np.array(self.links.global_com_position(
                iteration=task.iteration-1,
            ))
            for target, cylinder in zip(self.config['targets'], self.cylinders):
                if np.linalg.norm(cylinder.pos[:2] - pos[:2]) < cylinder.size[0]:
                    cylinder.rgba = target['rgba2']
