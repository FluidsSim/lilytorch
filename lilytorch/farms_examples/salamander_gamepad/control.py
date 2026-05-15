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
    0xFF95: "kp_7",  # KP_Home
    0xFF9A: "kp_9",  # KP_Prior / Page Up
    # Num Lock ON keysyms (some pynput versions / distros report these)
    0xFFB1: "kp_1",  0xFFB3: "kp_3",  0xFFB4: "kp_4",
    0xFFB5: "kp_5",  0xFFB6: "kp_6",  0xFFB7: "kp_7",  0xFFB9: "kp_9",
    # Windows VK_NUMPAD codes
    0x61: "kp_1",    0x63: "kp_3",    0x64: "kp_4",
    0x65: "kp_5",    0x66: "kp_6",    0x67: "kp_7",    0x69: "kp_9",
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
            "5+1=turn left, 4/6=decrease/increase turn strength, "
            "7=swim mode, 9=paddle mode. "
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
            "mode_swim": "kp_7" in pressed,
            "mode_paddle": "kp_9" in pressed,
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
        if char in {"1", "3", "4", "5", "6", "7", "9"}:
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

        if handler.controller is None and handler.joystick is None:
            pylog.warning(
                "Gamepad control enabled but no compatible controller was detected."
            )
            return False

        if handler.controller is None:
            pylog.warning(
                "Gamepad opened as a generic joystick. Bluetooth/gamepad mappings "
                "may vary, but fallback input is enabled."
            )

        self._handler = handler
        return True

    def announce_once(self):
        if self._announced:
            return
        pylog.info(
            "Gamepad controls active: left stick up=swim forward, left stick "
            "left/right=turn, D-pad up/left/right mirror the same commands, "
            "Cross/Triangle=forward, Square/L1=left, Circle/R1=right. "
            "L2 or Select=swim mode, R2 or Start=paddle mode."
        )
        self._announced = True

    def snapshot(self) -> dict[str, float]:
        if self._handler is None or not self._handler.update():
            return {
                "forward": 0.0, "turn": 0.0,
                "mode_swim": False, "mode_paddle": False,
            }

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
            # L2 analog, L2 digital (PS3 generic joystick btn 6), or Select/Back button
            "mode_swim": bool(
                state.trigger_left > 0.5
                or state.button_trigger_left
                or state.button_middle_left
            ),
            # R2 analog, R2 digital (PS3 generic joystick btn 7), or Start button
            "mode_paddle": bool(
                state.trigger_right > 0.5
                or state.button_trigger_right
                or state.button_middle_right
            ),
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


        self.base_amp = np.deg2rad(float(getattr(self.config, "amp",     300)))
        self.swim_freq = float(getattr(self.config, "freq", 2))
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
        self.twl = float(getattr(self.config, "twl", 10))
        self.limb_pose1 = float(getattr(self.config, "limb_pose1", -0.35 * 3.141592653589793))
        self.limb_pose2 = float(getattr(self.config, "limb_pose2", -0.2 * 3.141592653589793))
        self.body_amp_profile = self._body_amplitude_profile(
            len(self.body_joint_names)
        )
        self._turn_less_pressed = False
        self._turn_more_pressed = False

        # ── Paddle mode ──────────────────────────────────────────────
        self._mode: str = "swim"          # "swim" or "paddle"
        self._prev_mode_swim   = False
        self._prev_mode_paddle = False

        # IK parameters (match pd_controller_paddle.py defaults)
        self.paddle_freq     = float(getattr(self.config, "paddle_freq",     2.0))
        self.paddle_radius   = float(getattr(self.config, "paddle_radius",   0.01))
        self.paddle_center_x = float(getattr(self.config, "paddle_center_x", 0.01))
        self.paddle_center_y = float(getattr(self.config, "paddle_center_y", 0.0))
        self.paddle_l1       = float(getattr(self.config, "paddle_l1",       0.006))
        self.paddle_l2       = float(getattr(self.config, "paddle_l2",       0.006))
        self._precompute_paddle_trajectory()

        # Cached inputs (set each step by _input_modifiers)
        self._last_inputs: tuple = (
            {
                "move_left": False, "move_right": False,
                "move_straight": False, "turn_less": False, "turn_more": False,
                "mode_swim": False, "mode_paddle": False,
            },
            {"forward": 0.0, "turn": 0.0, "mode_swim": False, "mode_paddle": False},
        )

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

    # ── Paddle IK helpers ─────────────────────────────────────────────

    @staticmethod
    def _compute_ik_2d(elbow_id, x, y, d1, d2):
        """2-D inverse kinematics (matching pd_controller_paddle.py)."""
        D = np.clip(
            (x**2 + y**2 - d1**2 - d2**2) / (2.0 * d1 * d2), -1.0, 1.0
        )
        elbow = elbow_id * np.arccos(D)
        k1 = d1 + d2 * np.cos(elbow)
        k2 = d2 * np.sin(elbow)
        thigh = np.arctan2(y, x) - np.arctan2(k2, k1)
        return thigh, elbow

    def _precompute_paddle_trajectory(self, k: float = 0.8, n_samples: int = 1000):
        """
        Precompute one full period of the asymmetric IK paddle trajectory and
        store it as a lookup table for real-time interpolation.
        """
        freq         = self.paddle_freq
        radius       = self.paddle_radius
        cx, cy       = self.paddle_center_x, self.paddle_center_y
        l1, l2       = self.paddle_l1, self.paddle_l2
        phase_shift  = np.pi  # right leg is π out of phase with left

        # Exact period of the autonomous oscillator ds/dt = 2π·f·(1+k·cos(s))
        # Period = 1 / (f · sqrt(1 - k²))
        T  = 1.0 / (freq * np.sqrt(1.0 - k**2))
        dt = T / n_samples

        # Integrate left leg (phase=0) and right leg (phase=π)
        s_left  = np.zeros(n_samples)
        s_right = np.zeros(n_samples)
        for i in range(1, n_samples):
            s_left[i]  = s_left[i-1]  + 2*np.pi*freq*(1 + k*np.cos(s_left[i-1]))*dt
            s_right[i] = s_right[i-1] + 2*np.pi*freq*(1 + k*np.cos(s_right[i-1] + phase_shift))*dt

        x_left  = cx + radius * np.cos(-s_left)
        y_left  = cy + radius * np.sin(-s_left)
        x_right = cx + radius * np.cos(-s_right + phase_shift)
        y_right = cy + radius * np.sin(-s_right + phase_shift)

        tl, el = self._compute_ik_2d(-1, x_left,  y_left,  l1, l2)
        tr, er = self._compute_ik_2d(-1, x_right, y_right, l1, l2)

        self._paddle_period    = T
        self._paddle_n_samples = n_samples
        self._paddle_thigh_left  = tl
        self._paddle_elbow_left  = el
        self._paddle_thigh_right = tr
        self._paddle_elbow_right = er

        # Rest pose: foot at trajectory centre
        rest_thigh, rest_elbow = self._compute_ik_2d(-1, cx, cy, l1, l2)
        self._paddle_rest_thigh = float(rest_thigh)
        self._paddle_rest_elbow = float(rest_elbow)
        pylog.warning(
            "[Paddle IK] precomputed %.0f-sample LUT  "
            "period=%.3f s  rest=(%.3f, %.3f) rad",
            n_samples, T, self._paddle_rest_thigh, self._paddle_rest_elbow,
        )

    def _paddle_ik_sample(self, time: float) -> dict:
        """Look up IK angles for the current simulation time."""
        t_mod = time % self._paddle_period
        idx = int(t_mod / self._paddle_period * self._paddle_n_samples) % self._paddle_n_samples
        return {
            "thigh_left":  float(self._paddle_thigh_left[idx]),
            "elbow_left":  float(self._paddle_elbow_left[idx]),
            "thigh_right": float(self._paddle_thigh_right[idx]),
            "elbow_right": float(self._paddle_elbow_right[idx]),
        }

    def _paddle_limb_target(
            self, joint_name: str, time: float, direction: str | None
    ) -> float:
        """
        IK-based limb target for paddle mode.

        Straight  → both front legs active, synchronous (same phase).
        Left      → only left front leg active, right holds swim pose.
        Right     → only right front leg active, left holds swim pose.
        None/idle → all limbs hold their swim pose (no burst on mode switch).
        Hind limbs always use the fixed swim-rest pose.
        """
        is_front_left  = "leg_0_L" in joint_name
        is_front_right = "leg_0_R" in joint_name

        if not (is_front_left or is_front_right):
            # Hind limbs: same fixed poses as swim mode
            return self._limb_target(joint_name)

        is_active = (
            direction is not None
            and (
                (is_front_left  and direction in ("straight", "left"))
                or (is_front_right and direction in ("straight", "right"))
            )
        )

        if is_active:
            ik = self._paddle_ik_sample(time)
            # Both legs use the same (left / phase-0) trajectory → synchronous
            if joint_name.endswith("_0"):
                return ik["thigh_left"]
            if joint_name.endswith("_3"):
                return ik["elbow_left"]
            return 0.0

        # Inactive / idle → stay at swim pose so mode switch has no burst
        return self._limb_target(joint_name)

    # ── Mode management ───────────────────────────────────────────────

    def _update_mode(self, mode_swim: bool, mode_paddle: bool):
        """Switch between swim/paddle on rising edge of mode inputs."""
        if mode_swim and not self._prev_mode_swim:
            self._mode = "swim"
            pylog.warning("Controller mode → SWIM  (body undulation active)")
        if mode_paddle and not self._prev_mode_paddle:
            self._mode = "paddle"
            pylog.warning("Controller mode → PADDLE  (forelimb IK active)")
        self._prev_mode_swim   = mode_swim
        self._prev_mode_paddle = mode_paddle

    def _paddle_direction(
            self, ks: dict, gs: dict
    ) -> str | None:
        """
        Determine paddle direction from current inputs.

        Returns 'straight', 'left', 'right', or None (idle/stop).
        Keyboard is checked first; falls back to gamepad.
        """
        straight = ks["move_straight"]
        left     = ks["move_left"]
        right    = ks["move_right"]
        if straight or left or right:
            if left and not right:
                return "left"
            if right and not left:
                return "right"
            return "straight"

        forward = gs["forward"]
        turn    = gs["turn"]
        if forward > 0.0 or abs(turn) > 0.0:
            if turn < 0.0:
                return "left"
            if turn > 0.0:
                return "right"
            return "straight"

        return None

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
        keyboard_state = {
            "move_left": False,
            "move_right": False,
            "move_straight": False,
            "turn_less": False,
            "turn_more": False,
            "mode_swim": False,
            "mode_paddle": False,
        }
        if self.keyboard is not None:
            keyboard_state = self.keyboard.snapshot()

        gamepad_state = {
            "forward": 0.0,
            "turn": 0.0,
            "mode_swim": False,
            "mode_paddle": False,
        }
        if self.gamepad is not None:
            gamepad_state = self.gamepad.snapshot()

        # Cache for paddle-mode helpers
        self._last_inputs = (keyboard_state, gamepad_state)

        self._update_turn_strength(
            keyboard_state["turn_less"],
            keyboard_state["turn_more"],
        )

        self._update_mode(
            keyboard_state.get("mode_swim",   False)
            or gamepad_state.get("mode_swim",   False),
            keyboard_state.get("mode_paddle", False)
            or gamepad_state.get("mode_paddle", False),
        )

        if self.keyboard is None and self.gamepad is None:
            return False, 0.0, 0.0, 0.0

        keyboard_modifiers = self._keyboard_modifiers(keyboard_state)
        if keyboard_modifiers is not None:
            return keyboard_modifiers

        gamepad_modifiers = self._gamepad_modifiers(gamepad_state)
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
        # Read all inputs, update turn strength and mode (side-effects inside).
        is_swimming, freq, left_scale, right_scale = self._input_modifiers()
        ks, gs = self._last_inputs

        joint_positions: dict[str, float] = {}

        if self._mode == "paddle":
            # ── Paddle mode ───────────────────────────────────────────
            direction = self._paddle_direction(ks, gs)
            for joint_name in self.body_joint_names:
                joint_positions[joint_name] = 0.0
            for joint_name in self.limb_joint_names:
                joint_positions[joint_name] = self._paddle_limb_target(
                    joint_name, time, direction
                )
        else:
            # ── Swim mode ─────────────────────────────────────────────
            if is_swimming:
                indices = np.arange(len(self.body_joint_names), dtype=float)
                wave = self.base_amp * self.body_amp_profile * np.sin(
                    2.0 * np.pi * (indices / self.twl - freq * time)
                )
                targets = np.where(wave >= 0.0, left_scale * wave, right_scale * wave)
            else:
                targets = np.zeros(len(self.body_joint_names), dtype=float)
            for joint_name, target in zip(self.body_joint_names, targets):
                joint_positions[joint_name] = float(target)
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
