"""Simple keyboard/gamepad-controlled PD swim controller."""

from __future__ import annotations

import os
import numpy as np

try:
    from pynput import keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    pynput_keyboard = None
    PYNPUT_AVAILABLE = False

import farms_core.pylog as pylog
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.control import AnimatController, ControlType
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from lilytorch.util.rw import Dict2Class

# VK-code → logical key name.
# pynput on X11 sets key.vk to the *base* X11 keysym (group 0, level 0,
# i.e. the Num-Lock-OFF variant) regardless of the current Num Lock state.
# Also covers the Num-Lock-ON keysyms and Windows VK_NUMPAD codes.
#
# X11 base keysyms (Num Lock OFF):  KP_End=0xFF9C, KP_Next=0xFF9B,
#   KP_Left=0xFF96, KP_Begin=0xFF9D, KP_Right=0xFF98
# X11 keysyms (Num Lock ON):        KP_1=0xFFB1 … KP_6=0xFFB6
# Windows VK_NUMPAD codes:          VK_NUMPAD1=0x61 … VK_NUMPAD6=0x66
_KP_VK_MAP: dict[int, str] = {
    # Num Lock OFF (base keysyms pynput typically reports)
    0xFF9C: "kp_1",  # KP_End
    0xFF9B: "kp_3",  # KP_Next / Page Down
    0xFF96: "kp_4",  # KP_Left
    0xFF9D: "kp_5",  # KP_Begin
    0xFF98: "kp_6",  # KP_Right
    # Num Lock ON keysyms (some pynput versions / distros report these)
    0xFFB1: "kp_1",  0xFFB3: "kp_3",  0xFFB4: "kp_4",
    0xFFB5: "kp_5",  0xFFB6: "kp_6",
    # Windows VK_NUMPAD codes
    0x61: "kp_1",    0x63: "kp_3",    0x64: "kp_4",
    0x65: "kp_5",    0x66: "kp_6",
}

# Set LILY_DEBUG_KEYS=1 to print raw pynput key info on every key press.
_DEBUG_KEYS = os.environ.get("LILY_DEBUG_KEYS", "0") == "1"


class _KeyboardInputState:
    """Process-wide keyboard state shared by the local swim controller."""

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
            "Keyboard controls active: 5=swim straight, 5+3=turn right, "
            "5+1=turn left, 4/6=decrease/increase turn strength. "
            "No key pressed = all amplitudes zero."
        )
        self._announced = True

    def snapshot(self) -> dict[str, bool]:
        pressed = self._pressed
        return {
            "move_left": "kp_1" in pressed,
            "move_right": "kp_3" in pressed,
            "move_straight": "kp_5" in pressed,
            "turn_less": "kp_4" in pressed,
            "turn_more": "kp_6" in pressed,
        }

    def _on_press(self, key):
        if _DEBUG_KEYS:
            print(
                f"[key] char={getattr(key,'char',None)!r}  "
                f"vk={getattr(key,'vk',None)!r}  "
                f"str={key!s}"
            )
        key_name = self._normalize_key(key)
        if key_name is not None:
            self._pressed.add(key_name)

    def _on_release(self, key):
        key_name = self._normalize_key(key)
        if key_name is not None:
            self._pressed.discard(key_name)

    @staticmethod
    def _normalize_key(key):
        # Regular number-row keys: pynput sets key.char to the character.
        char = getattr(key, "char", None)
        if char in {"1", "3", "4", "5", "6"}:
            return f"kp_{char}"
        # Numpad keys on X11: key.char is None; the X11 keysym lives in key.vk.
        vk = getattr(key, "vk", None)
        if vk is not None:
            return _KP_VK_MAP.get(vk)
        return None


_KEYBOARD_INPUT = _KeyboardInputState()


class _GamepadInputState:
    """Process-wide gamepad state shared by the local swim controller."""

    def __init__(self):
        self._handler = None
        self._announced = False
        self._forward_threshold = 0.35
        self._turn_threshold = 0.35

    def start(
            self,
            deadzone: float = 0.1,
            forward_threshold: float = 0.35,
            turn_threshold: float = 0.35,
    ) -> bool:
        if self._handler is not None:
            self._forward_threshold = forward_threshold
            self._turn_threshold = turn_threshold
            return True

        self._forward_threshold = forward_threshold
        self._turn_threshold = turn_threshold

        try:
            from lilytorch.integration.gamepad import GamepadHandler
        except Exception as exc:
            pylog.warning(
                "Gamepad control disabled: could not import SDL support (%s).",
                exc,
            )
            return False

        try:
            handler = GamepadHandler(deadzone=deadzone)
        except Exception as exc:
            pylog.warning(
                "Gamepad control disabled: could not initialize controller (%s).",
                exc,
            )
            return False

        if handler.controller is None:
            pylog.warning(
                "Gamepad control enabled but no compatible controller was detected."
            )
            return False

        self._handler = handler
        return True

    def announce_once(self):
        if self._announced:
            return
        pylog.info(
            "Gamepad controls active: left stick up=swim forward, left stick "
            "left/right=turn, D-pad up/left/right mirror the same commands, "
            "Cross/Triangle=forward, Square/L1=left, Circle/R1=right."
        )
        self._announced = True

    def snapshot(self) -> dict[str, float]:
        if self._handler is None or not self._handler.update():
            return {"forward": 0.0, "turn": 0.0}

        state = self._handler.get_state()
        left_x = float(state.js_left_x)
        left_y = float(state.js_left_y)
        right_x = float(state.js_right_x)
        right_y = float(state.js_right_y)

        # Some PS3 controllers on Linux expose all analog axes as -1 until the
        # first valid motion sample arrives. Treat that sentinel pattern as idle
        # so the swimmer does not steer on its own at startup.
        if all(axis <= -0.99 for axis in (left_x, left_y, right_x, right_y)):
            left_x = 0.0
            left_y = 0.0

        forward = max(0.0, -left_y)
        if (
                (state.button_dpad_up and not state.button_dpad_down)
                or state.button_bottom
                or state.button_top
        ):
            forward = 1.0
        elif forward < self._forward_threshold:
            forward = 0.0
        else:
            forward = 1.0

        if (
                (state.button_dpad_left and not state.button_dpad_right)
                or state.button_left
                or state.button_shoulder_left
        ):
            turn = -1.0
        elif (
                (state.button_dpad_right and not state.button_dpad_left)
                or state.button_right
                or state.button_shoulder_right
        ):
            turn = 1.0
        else:
            turn_raw = left_x
            if turn_raw <= -self._turn_threshold:
                turn = -1.0
            elif turn_raw >= self._turn_threshold:
                turn = 1.0
            else:
                turn = 0.0

        return {
            "forward": float(np.clip(forward, 0.0, 1.0)),
            "turn": float(np.clip(turn, -1.0, 1.0)),
        }


_GAMEPAD_INPUT = _GamepadInputState()


class PositionController(AnimatController):
    """Simple oscillatory position controller based on pd_controller_swim."""

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
        self.swim_freq = self.config.freq
        self.turn_strength = float(getattr(self.config, "turn_strength", 0.8))
        self.turn_strength_step = float(
            getattr(self.config, "turn_strength_step", 0.)
        )
        self.min_turn_strength = float(
            getattr(self.config, "min_turn_strength", 0.0)
        )
        self.max_turn_strength = float(
            getattr(self.config, "max_turn_strength", 0.95)
        )
        self.twl = float(self.config.twl)
        self.limb_pose1 = float(self.config.limb_pose1)
        self.limb_pose2 = float(self.config.limb_pose2)
        self.body_amp_profile = self._body_amplitude_profile(
            len(self.body_joint_names)
        )
        self._turn_less_pressed = False
        self._turn_more_pressed = False

        self.last_iteration = -1
        self.cached_positions = {
            joint_name: 0.0 for joint_name in position_joints
        }

        self.keyboard = None
        if bool(getattr(self.config, "keyboard_enabled", True)):
            if _KEYBOARD_INPUT.start():
                self.keyboard = _KEYBOARD_INPUT
                self.keyboard.announce_once()
            elif not PYNPUT_AVAILABLE:
                pylog.warning(
                    "Keyboard control disabled: install 'pynput' to use "
                    "keyboard-only control."
                )

        self.gamepad = None
        if bool(getattr(self.config, "gamepad_enabled", True)):
            gamepad_deadzone = float(
                getattr(self.config, "gamepad_deadzone", 0.1)
            )
            gamepad_forward_threshold = float(
                getattr(self.config, "gamepad_forward_threshold", 0.35)
            )
            gamepad_turn_threshold = float(
                getattr(self.config, "gamepad_turn_threshold", 0.35)
            )
            if _GAMEPAD_INPUT.start(
                    deadzone=gamepad_deadzone,
                    forward_threshold=gamepad_forward_threshold,
                    turn_threshold=gamepad_turn_threshold,
            ):
                self.gamepad = _GAMEPAD_INPUT
                self.gamepad.announce_once()

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

    def _update_turn_strength(
            self,
            turn_less_pressed: bool,
            turn_more_pressed: bool,
    ):
        if turn_less_pressed and not self._turn_less_pressed:
            self.turn_strength = max(
                self.min_turn_strength,
                self.turn_strength - self.turn_strength_step,
            )
        if turn_more_pressed and not self._turn_more_pressed:
            self.turn_strength = min(
                self.max_turn_strength,
                self.turn_strength + self.turn_strength_step,
            )
        self._turn_less_pressed = turn_less_pressed
        self._turn_more_pressed = turn_more_pressed

    def _keyboard_modifiers(
            self,
            keyboard_state: dict[str, bool],
    ) -> tuple[bool, float, float, float] | None:
        straight = keyboard_state["move_straight"]
        left = keyboard_state["move_left"]
        right = keyboard_state["move_right"]

        if not straight:
            return None

        if left and not right:
            return (
                True,
                self.swim_freq,
                1.0 - self.turn_strength,
                1.0 + self.turn_strength,
            )
        if right and not left:
            return (
                True,
                self.swim_freq,
                1.0 + self.turn_strength,
                1.0 - self.turn_strength,
            )
        return True, self.swim_freq, 1.0, 1.0

    def _gamepad_modifiers(
            self,
            gamepad_state: dict[str, float],
    ) -> tuple[bool, float, float, float] | None:
        if (
            gamepad_state["forward"] <= 0.0
            and abs(gamepad_state["turn"]) <= 0.0
        ):
            return None

        turn = float(gamepad_state["turn"])
        turn_amount = self.turn_strength*abs(turn)
        if turn < 0.0:
            return (
                True,
                self.swim_freq,
                1.0 - turn_amount,
                1.0 + turn_amount,
            )
        if turn > 0.0:
            return (
                True,
                self.swim_freq,
                1.0 + turn_amount,
                1.0 - turn_amount,
            )
        return True, self.swim_freq, 1.0, 1.0

    def _input_modifiers(self) -> tuple[bool, float, float, float]:
        if self.keyboard is None and self.gamepad is None:
            return False, 0.0, 0.0, 0.0

        keyboard_state = {
            "move_left": False,
            "move_right": False,
            "move_straight": False,
            "turn_less": False,
            "turn_more": False,
        }
        if self.keyboard is not None:
            keyboard_state = self.keyboard.snapshot()

        self._update_turn_strength(
            keyboard_state["turn_less"],
            keyboard_state["turn_more"],
        )

        keyboard_modifiers = self._keyboard_modifiers(keyboard_state)
        if keyboard_modifiers is not None:
            return keyboard_modifiers

        if self.gamepad is not None:
            gamepad_modifiers = self._gamepad_modifiers(
                self.gamepad.snapshot()
            )
            if gamepad_modifiers is not None:
                return gamepad_modifiers

        return False, 0.0, 0.0, 0.0

    def _body_targets(self, time: float) -> np.ndarray:
        is_swimming, freq, left_scale, right_scale = self._input_modifiers()
        if not is_swimming:
            return np.zeros(len(self.body_joint_names), dtype=float)

        indices = np.arange(len(self.body_joint_names), dtype=float)
        wave = self.base_amp*self.body_amp_profile*np.sin(
            2.0*np.pi*(indices/self.twl - freq*time)
        )
        # Positive angles are treated as leftward swings, negative as rightward.
        return np.where(wave >= 0.0, left_scale*wave, right_scale*wave)

    def _limb_target(self, joint_name: str) -> float:
        if joint_name.endswith("_0"):
            return self.limb_pose1
        if joint_name.endswith("_3"):
            return self.limb_pose2
        return 0.0

    def _update_positions(self, time: float):
        joint_positions = {}
        for joint_name, joint_target in zip(
                self.body_joint_names,
                self._body_targets(time),
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
        """Return PD position targets for the current input state."""
        del timestep
        if iteration != self.last_iteration:
            self._update_positions(time)
            self.last_iteration = iteration
        return dict(self.cached_positions)
