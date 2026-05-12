"""
Platform-independent gamepad handler that can run without ROS2.
This module handles gamepad input and converts it to standard control commands.
Uses PySDL2 for cross-platform gamepad support with better documentation.

Differences between gamepads should have been handled by SDL2.
Below are tables just for referencing.

GAMEPAD BUTTON/AXIS MAPPING REFERENCE (PySDL2):
===============================================

PySDL2 Documentation: https://pysdl2.readthedocs.io/
ROS2 joy (using SDL2) documentation: https://docs.ros.org/en/rolling/p/joy/
SDL gamepad tool (can print mapping): https://generalarcade.com/gamepadtool/
SDL2 GameController API provides standardized mapping across all platforms.
Linux joystick: jstest & jstest-gtk (with GUI)

Axes (SDL_GameController):
-----------------------------------
Component          | Xbox Name | PS4 Name   | SDL2 Axis        | Value (SDL2)                           | Linux js# | Linux event | Value (Linux js)
-------------------|-----------|----------- |------------------|----------------------------------------|-----------|-------------|------
Left Stick X       | LS X      | L3 X       | 0: LEFTX         | -32768 (up/left) to 32767 (down/right) | Axis 0    | ABS_X       | -32767 (left) to 32767 (right)
Left Stick Y       | LS Y      | L3 Y       | 1: LEFTY         | -32768 (up/left) to 32767 (down/right) | Axis 1    | ABS_Y       | -32767 (up)   to 32767 (down)
Right Stick X      | RS X      | R3 X       | 3: RIGHTX        | -32768 (up/left) to 32767 (down/right) | Axis 3    | ABS_RX      | -32767 (left) to 32767 (right)
Right Stick Y      | RS Y      | R3 Y       | 4: RIGHTY        | -32768 (up/left) to 32767 (down/right) | Axis 4    | ABS_RY      | -32767 (up)   to 32767 (down)
Left Trigger       | LT        | L2         | 2: TRIGGERLEFT   | 0 (released) to 32767 (fully pressed)  | Axis 2    | ABS_Z       | -32767 (home) to 32767 (pressed)
Right Trigger      | RT        | R2         | 5: TRIGGERRIGHT  | 0 (released) to 32767 (fully pressed)  | Axis 5    | ABS_RZ      | -32767 (home) to 32767 (pressed)

Buttons (SDL_GameController):
----------------------------------
Component          | Xbox Name | PS4 Name   | SDL2 Button      | Value (SDL2) | Linux js# | Linux event | Value
-------------------|-----------|------------|------------------|--------------|-----------|-------------|------
Bottom Button      | A         | Cross (X)  | 0: A             | 0/1          | Button 0  | BTN_A       | 0/1
Right Button       | B         | Circle (O) | 1: B             | 0/1          | Button 1  | BTN_B       | 0/1
Left Button        | X         | Square     | 3: X             | 0/1          | Button 3  | BTN_Y       | 0/1
Top Button         | Y         | Triangle   | 2: Y             | 0/1          | Button 2  | BTN_X       | 0/1
Left Shoulder      | LB        | L1         | 4: LEFTSHOULDER  | 0/1          | Button 4  | BTN_TL      | 0/1
Right Shoulder     | RB        | R1         | 5: RIGHTSHOULDER | 0/1          | Button 5  | BTN_TR      | 0/1
Back/Select (left) | View      | Share      | 8: BACK          | 0/1          | Button 8  | BTN_SELECT  | 0/1
Start/Menu (right) | Menu      | Options    | 9: START         | 0/1          | Button 9  | BTN_START   | 0/1
Guide (middle)     | XBOX      | PS         | 10: GUIDE        | 0/1          | Button 10 | BTN_MODE    | 0/1
Left Stick Click   | LS        | L3         | 11: LEFTSTICK    | 0/1          | Button 11 | BTN_THUMBL  | 0/1
Right Stick Click  | RS        | R3         | 12: RIGHTSTICK   | 0/1          | Button 12 | BTN_THUMBR  | 0/1
?                  | ?         | N/A        | ?                | 0/1          | Button 6  | BTN_TL2     | 0/1
?                  | ?         | N/A        | ?                | 0/1          | Button 7  | BTN_TR2     | 0/1
D-Pad Up           | D-Up      | D-Up       | h0.1: DPAD_UP    | 0/1          | Axis 7- (PS4) / Button 13   | ABS_HAT0Y (PS4) / BTN_DPAD_*  | 0/-32767 (PS4) / 0/1
D-Pad Down         | D-Down    | D-Down     | h0.4: DPAD_DOWN  | 0/1          | Axis 7+ (PS4) / Button 14   | ABS_HAT0Y (PS4) / BTN_DPAD_*  | 0/ 32767 (PS4) / 0/1
D-Pad Left         | D-Left    | D-Left     | h0.8: DPAD_LEFT  | 0/1          | Axis 6- (PS4) / Button 15   | ABS_HAT0X (PS4) / BTN_DPAD_*  | 0/-32767 (PS4) / 0/1
D-Pad Right        | D-Right   | D-Right    | h0.2: DPAD_RIGHT | 0/1          | Axis 6+ (PS4) / Button 16   | ABS_HAT0X (PS4) / BTN_DPAD_*  | 0/ 32767 (PS4) / 0/1

* PS4 controller mapping checked with jstest and gamepad tool. XBOX and other controllers assumed to follow SDL2 standard.

LINUX TESTING COMMANDS:
-----------------------
- `jstest /dev/input/js0` - Test joystick functionality
"""

try:
    import sdl2
    import sdl2.ext
    SDL2_AVAILABLE = True
except ImportError:
    SDL2_AVAILABLE = False
    print("Warning: 'pysdl2' module not available. Install with: pip install pysdl2")

import time

from enum import IntEnum

# For enums for button and axis indices, use trhe SDL_GameControllerButton and SDL_GameControllerAxis values from sdl2 module. E.g., sdl2.SDL_CONTROLLER_BUTTON_A is 0.

class GamepadState:
    """Simple class to hold the current gamepad state with SDL2 mapping"""
    def __init__(self):
        # Joysticks: normalized between -1.0 and 1.0
        self.js_left_x = 0.0      # Left stick x-axis (left-/right+), can be used to control fore-aft velocity
        self.js_left_y = 0.0      # Left stick y-axis (forward-/backward+), can be used to control lateral velocity
        self.js_right_x = 0.0     # Right stick x-axis (left-/right+), can be used to control yaw angular velocity
        self.js_right_y = 0.0     # Right stick y-axis (forward-/backward+), can be used to control pitch angular velocity

        # Triggers: normalized between 0.0 (home) and 1.0 (fully pressed)
        self.trigger_left = 0.0     # SDL_CONTROLLER_AXIS_TRIGGERLEFT (Xbox: LT, PS4: L2)
        self.trigger_right = 0.0    # SDL_CONTROLLER_AXIS_TRIGGERRIGHT (Xbox: RT, PS4: R2)

        # Right: action pad buttons
        self.button_bottom = False    # SDL_CONTROLLER_BUTTON_A (Xbox: A, PS4: Cross)
        self.button_right  = False    # SDL_CONTROLLER_BUTTON_B (Xbox: B, PS4: Circle)
        self.button_left   = False    # SDL_CONTROLLER_BUTTON_X (Xbox: X, PS4: Square)
        self.button_top    = False    # SDL_CONTROLLER_BUTTON_Y (Xbox: Y, PS4: Triangle)

        # Middle: Menu/System buttons
        self.button_middle_right = False   # SDL_CONTROLLER_BUTTON_START (Xbox: Menu, PS4: Options)
        self.button_middle_left = False  # SDL_CONTROLLER_BUTTON_BACK (Xbox: View, PS4: Share)
        self.button_middle_logo = False  # SDL_CONTROLLER_BUTTON_GUIDE (Xbox: XBOX, PS4: PS)

        # Shoulder buttons
        self.button_shoulder_left = False   # SDL_CONTROLLER_BUTTON_LEFTSHOULDER (Xbox: LB, PS4: L1)
        self.button_shoulder_right = False  # SDL_CONTROLLER_BUTTON_RIGHTSHOULDER (Xbox: RB, PS4: R1)

        # Stick buttons
        self.button_stick_left = False      # SDL_CONTROLLER_BUTTON_LEFTSTICK (Xbox: LS, PS4: L3)
        self.button_stick_right = False     # SDL_CONTROLLER_BUTTON_RIGHTSTICK (Xbox: RS, PS4: R3)

        # D-pad - digital directional buttons
        self.button_dpad_up = False        # SDL_CONTROLLER_BUTTON_DPAD_UP
        self.button_dpad_down = False      # SDL_CONTROLLER_BUTTON_DPAD_DOWN
        self.button_dpad_left = False      # SDL_CONTROLLER_BUTTON_DPAD_LEFT
        self.button_dpad_right = False     # SDL_CONTROLLER_BUTTON_DPAD_RIGHT

        # PS3-style digital trigger buttons (Linux joystick buttons 6 and 7)
        # On PS3 pads that expose as generic joystick, L2/R2 appear as
        # digital buttons (indices 6 and 7) rather than analog axes.
        self.button_trigger_left = False   # Linux joystick button 6 (PS3 L2 digital)
        self.button_trigger_right = False  # Linux joystick button 7 (PS3 R2 digital)

        # Define mappings betweeen SDL2 button/axis names and the fields above
        # Remapped to be more intuitive for non-gamers
        self.BUTTON_NAMES = {
            sdl2.SDL_CONTROLLER_BUTTON_A: 'button_bottom',
            sdl2.SDL_CONTROLLER_BUTTON_B: 'button_right',
            sdl2.SDL_CONTROLLER_BUTTON_X: 'button_left',
            sdl2.SDL_CONTROLLER_BUTTON_Y: 'button_top',
            sdl2.SDL_CONTROLLER_BUTTON_BACK: 'button_middle_left',
            sdl2.SDL_CONTROLLER_BUTTON_GUIDE: 'button_middle_logo',
            sdl2.SDL_CONTROLLER_BUTTON_START: 'button_middle_right',
            sdl2.SDL_CONTROLLER_BUTTON_LEFTSTICK: 'button_stick_left',
            sdl2.SDL_CONTROLLER_BUTTON_RIGHTSTICK: 'button_stick_right',
            sdl2.SDL_CONTROLLER_BUTTON_LEFTSHOULDER: 'button_shoulder_left',
            sdl2.SDL_CONTROLLER_BUTTON_RIGHTSHOULDER: 'button_shoulder_right',
            sdl2.SDL_CONTROLLER_BUTTON_DPAD_UP: 'button_dpad_up',
            sdl2.SDL_CONTROLLER_BUTTON_DPAD_DOWN: 'button_dpad_down',
            sdl2.SDL_CONTROLLER_BUTTON_DPAD_LEFT: 'button_dpad_left',
            sdl2.SDL_CONTROLLER_BUTTON_DPAD_RIGHT: 'button_dpad_right',
        }

        self.AXIS_NAMES = {
            sdl2.SDL_CONTROLLER_AXIS_LEFTX: 'js_left_x',
            sdl2.SDL_CONTROLLER_AXIS_LEFTY: 'js_left_y',
            sdl2.SDL_CONTROLLER_AXIS_RIGHTX: 'js_right_x',
            sdl2.SDL_CONTROLLER_AXIS_RIGHTY: 'js_right_y',
            sdl2.SDL_CONTROLLER_AXIS_TRIGGERLEFT: 'trigger_left',
            sdl2.SDL_CONTROLLER_AXIS_TRIGGERRIGHT: 'trigger_right',
        }

        # Expose event names in SDL2
        self.EVENT_NAMES = {
            sdl2.SDL_CONTROLLERBUTTONDOWN: 'button_down',
            sdl2.SDL_CONTROLLERBUTTONUP: 'button_up',
            sdl2.SDL_CONTROLLERAXISMOTION: 'axis_motion',
        }

    def copy(self):
        """Create a deep copy of the current gamepad state"""
        new_state = GamepadState()

        # Copy joystick axes
        new_state.js_left_x = self.js_left_x
        new_state.js_left_y = self.js_left_y
        new_state.js_right_x = self.js_right_x
        new_state.js_right_y = self.js_right_y

        # Copy triggers
        new_state.trigger_left = self.trigger_left
        new_state.trigger_right = self.trigger_right

        # Copy action pad buttons
        new_state.button_bottom = self.button_bottom
        new_state.button_right = self.button_right
        new_state.button_left = self.button_left
        new_state.button_top = self.button_top

        # Copy menu/system buttons
        new_state.button_middle_right = self.button_middle_right
        new_state.button_middle_left = self.button_middle_left
        new_state.button_middle_logo = self.button_middle_logo

        # Copy shoulder buttons
        new_state.button_shoulder_left = self.button_shoulder_left
        new_state.button_shoulder_right = self.button_shoulder_right

        # Copy stick buttons
        new_state.button_stick_left = self.button_stick_left
        new_state.button_stick_right = self.button_stick_right

        # Copy D-pad buttons
        new_state.button_dpad_up = self.button_dpad_up
        new_state.button_dpad_down = self.button_dpad_down
        new_state.button_dpad_left = self.button_dpad_left
        new_state.button_dpad_right = self.button_dpad_right

        # Copy digital trigger buttons
        new_state.button_trigger_left = self.button_trigger_left
        new_state.button_trigger_right = self.button_trigger_right

        return new_state


class GamepadHandler:

    def __init__(self, deadzone=0.1):
        """
        Initialize the gamepad handler with PySDL2.

        Args:
            deadzone: Minimum stick movement to register (0.0 to 1.0)
        """
        if deadzone > 0.0 and deadzone < 1.0:
            self.deadzone = deadzone
        else:
            print("Warning: Invalid deadzone value, using default 0.1")
            self.deadzone = 0.1  # Default deadzone value

        self.state = GamepadState()
        self._running = False
        self.controller = None
        self.joystick = None
        self.using_game_controller = False

        if not SDL2_AVAILABLE:
            raise RuntimeError("PySDL2 module is required for gamepad functionality. Install with: pip install pysdl2")

        # Initialize SDL2
        if sdl2.SDL_Init(sdl2.SDL_INIT_GAMECONTROLLER | sdl2.SDL_INIT_JOYSTICK) < 0:
            raise RuntimeError(f"Failed to initialize SDL2: {sdl2.SDL_GetError()}")

        # Find and open the first available game controller
        self._initialize_controller()

    def _initialize_controller(self):
        """Find and initialize the first available game controller"""
        num_joysticks = sdl2.SDL_NumJoysticks()

        if num_joysticks == 0:
            print("Warning: No joysticks detected. Check permissions.")
            return

        # Find first game controller
        for i in range(num_joysticks):
            if sdl2.SDL_IsGameController(i):
                self.controller = sdl2.SDL_GameControllerOpen(i)
                if self.controller:
                    self.using_game_controller = True
                    controller_name = sdl2.SDL_GameControllerName(self.controller)
                    if controller_name:
                        print(f"Opened game controller: {controller_name.decode('utf-8')}")
                    else:
                        print(f"Opened game controller #{i}")
                    break

        if not self.controller:
            for i in range(num_joysticks):
                self.joystick = sdl2.SDL_JoystickOpen(i)
                if self.joystick:
                    joystick_name = sdl2.SDL_JoystickName(self.joystick)
                    if joystick_name:
                        print(
                            f"Opened generic joystick: {joystick_name.decode('utf-8')}"
                        )
                    else:
                        print(f"Opened generic joystick #{i}")
                    self.using_game_controller = False
                    break

        if not self.controller and not self.joystick:
            print("Warning: No compatible game controllers or joysticks found")

    def _close_device(self):
        if hasattr(self, 'controller') and self.controller:
            sdl2.SDL_GameControllerClose(self.controller)
            self.controller = None
        if hasattr(self, 'joystick') and self.joystick:
            sdl2.SDL_JoystickClose(self.joystick)
            self.joystick = None
        self.using_game_controller = False

    def __del__(self):
        """Cleanup SDL2 resources"""
        self._close_device()
        if SDL2_AVAILABLE:
            sdl2.SDL_Quit()

    def _joystick_axis(self, axis_index):
        if self.joystick is None:
            return 0
        if sdl2.SDL_JoystickNumAxes(self.joystick) <= axis_index:
            return 0
        return sdl2.SDL_JoystickGetAxis(self.joystick, axis_index)

    def _joystick_button(self, button_index):
        if self.joystick is None:
            return False
        if sdl2.SDL_JoystickNumButtons(self.joystick) <= button_index:
            return False
        return bool(sdl2.SDL_JoystickGetButton(self.joystick, button_index))

    def _update_generic_joystick_state(self):
        # Generic joystick fallback for devices such as some PS3 pads that do
        # not advertise an SDL GameController mapping.
        self.state.js_left_x = self.normalize_stick_input(self._joystick_axis(0))
        self.state.js_left_y = self.normalize_stick_input(self._joystick_axis(1))
        self.state.js_right_x = self.normalize_stick_input(self._joystick_axis(3))
        self.state.js_right_y = self.normalize_stick_input(self._joystick_axis(4))
        self.state.trigger_left = 0.0
        self.state.trigger_right = 0.0

        self.state.button_bottom = self._joystick_button(0)
        self.state.button_right = self._joystick_button(1)
        self.state.button_top = self._joystick_button(2)
        self.state.button_left = self._joystick_button(3)
        self.state.button_shoulder_left = self._joystick_button(4)
        self.state.button_shoulder_right = self._joystick_button(5)
        self.state.button_trigger_left = self._joystick_button(6)
        self.state.button_trigger_right = self._joystick_button(7)
        self.state.button_middle_left = self._joystick_button(8)
        self.state.button_middle_right = self._joystick_button(9)
        self.state.button_middle_logo = self._joystick_button(10)
        self.state.button_stick_left = self._joystick_button(11)
        self.state.button_stick_right = self._joystick_button(12)

        if sdl2.SDL_JoystickNumHats(self.joystick) > 0:
            hat = sdl2.SDL_JoystickGetHat(self.joystick, 0)
            self.state.button_dpad_up = bool(hat & sdl2.SDL_HAT_UP)
            self.state.button_dpad_down = bool(hat & sdl2.SDL_HAT_DOWN)
            self.state.button_dpad_left = bool(hat & sdl2.SDL_HAT_LEFT)
            self.state.button_dpad_right = bool(hat & sdl2.SDL_HAT_RIGHT)
        else:
            self.state.button_dpad_up = self._joystick_button(13)
            self.state.button_dpad_down = self._joystick_button(14)
            self.state.button_dpad_left = self._joystick_button(15)
            self.state.button_dpad_right = self._joystick_button(16)

        return True

    def _update_game_controller_state(self):
        # Update analog sticks
        # Left stick X-axis
        left_x = sdl2.SDL_GameControllerGetAxis(self.controller, sdl2.SDL_CONTROLLER_AXIS_LEFTX)
        self.state.js_left_x = self.normalize_stick_input(left_x)
        # Left stick Y-axis
        left_y = sdl2.SDL_GameControllerGetAxis(self.controller, sdl2.SDL_CONTROLLER_AXIS_LEFTY)
        self.state.js_left_y = self.normalize_stick_input(left_y)

        # Right stick X-axis
        right_x = sdl2.SDL_GameControllerGetAxis(self.controller, sdl2.SDL_CONTROLLER_AXIS_RIGHTX)
        self.state.js_right_x = self.normalize_stick_input(right_x)

        # Right stick Y-axis
        right_y = sdl2.SDL_GameControllerGetAxis(self.controller, sdl2.SDL_CONTROLLER_AXIS_RIGHTY)
        self.state.js_right_y = self.normalize_stick_input(right_y)

        # Left trigger
        left_trigger = sdl2.SDL_GameControllerGetAxis(self.controller, sdl2.SDL_CONTROLLER_AXIS_TRIGGERLEFT)
        self.state.trigger_left = self.normalize_trigger_input(left_trigger)

        # Right trigger
        right_trigger = sdl2.SDL_GameControllerGetAxis(self.controller, sdl2.SDL_CONTROLLER_AXIS_TRIGGERRIGHT)
        self.state.trigger_right = self.normalize_trigger_input(right_trigger)

        # Update action pad buttons
        self.state.button_bottom = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_A))
        self.state.button_right = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_B))
        self.state.button_left = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_X))
        self.state.button_top = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_Y))

        # Update middle panel buttons
        self.state.button_middle_right = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_START))
        self.state.button_middle_left = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_BACK))
        self.state.button_middle_logo = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_GUIDE))

        # Update shoulder buttons
        self.state.button_shoulder_left = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_LEFTSHOULDER))
        self.state.button_shoulder_right = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_RIGHTSHOULDER))

        # Update stick click buttons
        self.state.button_stick_left = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_LEFTSTICK))
        self.state.button_stick_right = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_RIGHTSTICK))

        # Update D-pad
        self.state.button_dpad_up = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_DPAD_UP))
        self.state.button_dpad_down = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_DPAD_DOWN))
        self.state.button_dpad_left = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_DPAD_LEFT))
        self.state.button_dpad_right = bool(sdl2.SDL_GameControllerGetButton(self.controller, sdl2.SDL_CONTROLLER_BUTTON_DPAD_RIGHT))

        return True

    def apply_deadzone(self, value):
        """Apply deadzone to stick input"""
        if abs(value) < self.deadzone:
            return 0.0
        result = value - self.deadzone if value > 0 else value + self.deadzone # Calculate value after deadzone
        result /= (1.0 - self.deadzone) # Re-map from 0.0 to +- 1.0
        # Clamp result to -1.0 to 1.0
        result = result if result < 1.0 else 1.0
        result = result if result > -1.0 else -1.0
        return result

    def normalize_stick_input(self, raw_value):
        """
        Normalize stick input from SDL2 range to desired output range.

        Args:
            raw_value: Raw value from SDL2 (typically -32767 to 32767)

        Returns:
            Normalized value between -1.0 and 1.0
        """
        normalized = raw_value / 32768.0
        normalized = self.apply_deadzone(normalized)
        return normalized

    def normalize_trigger_input(self, raw_value):
        """
        Normalize trigger input from SDL2 range to 0.0 (home) - 1.0 (fully pressed).

        Args:
            raw_value: Raw value from SDL2 (-32767 to 32767)

        Returns:
            Normalized value between 0.0 and 1.0
        """
        normalized = raw_value / 32767.0
        normalized = self.apply_deadzone(normalized)
        return normalized

    def update(self):
        """
        Update gamepad state by reading SDL2 events.

        Returns:
            True if controller is available, False otherwise
        """
        if not self.controller and not self.joystick:
            return False

        # Refresh SDL input state without draining the event queue that other
        # parts of the application may be using.
        sdl2.SDL_PumpEvents()

        # Check if the current device is still connected.
        if self.using_game_controller:
            attached = bool(sdl2.SDL_GameControllerGetAttached(self.controller))
            disconnected_name = "Controller"
        else:
            attached = bool(sdl2.SDL_JoystickGetAttached(self.joystick))
            disconnected_name = "Joystick"

        if not attached:
            print(f"{disconnected_name} disconnected, attempting to reconnect...")
            self._close_device()
            self._initialize_controller()
            return False

        try:
            if self.using_game_controller:
                return self._update_game_controller_state()
            return self._update_generic_joystick_state()

        except Exception as e:
            print(f"Gamepad error: {e}")
            return False

    def update_events(self):
        """
        Process SDL2 controller events (event-driven approach).
        Uses SDL2's native event system - more efficient than polling.

        This method processes the SDL event queue and yields SDL_Event objects
        for controller-related events. The state of the gamepad handler is automatically updated in this process.

        Yields:
            sdl2.SDL_Event: SDL2 events for controller button/axis/device changes
                - SDL_CONTROLLERBUTTONDOWN: event.cbutton has button, state, timestamp
                - SDL_CONTROLLERBUTTONUP: event.cbutton has button, state, timestamp
                - SDL_CONTROLLERAXISMOTION: event.caxis has axis, value, timestamp
                - SDL_CONTROLLERDEVICEADDED/REMOVED: device connection changes

        Example:
            for event in handler.update_events():
                event_name = handler.state.EVENT_NAMES.get(event.type)
                if event_name == 'button_down':
                    print(f"Button {event.cbutton.button} pressed")
                elif event_name == 'axis_motion':
                    print(f"Axis {event.caxis.axis} = {event.caxis.value}")
        """
        if not self.controller:
            return

        event = sdl2.SDL_Event()
        while sdl2.SDL_PollEvent(event) != 0:
            # Filter for controller events from our controller
            if event.type in (sdl2.SDL_CONTROLLERBUTTONDOWN, sdl2.SDL_CONTROLLERBUTTONUP):
                self._update_button_state(button = event.cbutton.button, pressed = (event.type == sdl2.SDL_CONTROLLERBUTTONDOWN))
                yield event

            elif event.type == sdl2.SDL_CONTROLLERAXISMOTION:
                if event.caxis.axis in (sdl2.SDL_CONTROLLER_AXIS_TRIGGERLEFT, sdl2.SDL_CONTROLLER_AXIS_TRIGGERRIGHT): # Trigger axes
                    self._update_axis_state(axis = event.caxis.axis, normalized_value = self.normalize_trigger_input(event.caxis.value))
                else: # Joystick axes
                    self._update_axis_state(axis = event.caxis.axis, normalized_value = self.normalize_stick_input(event.caxis.value))
                yield event

            elif event.type == sdl2.SDL_CONTROLLERDEVICEADDED:
                if not self.controller:
                    print("Controller connected")
                    self._initialize_controller()
                yield event

            elif event.type == sdl2.SDL_CONTROLLERDEVICEREMOVED:
                if event.cdevice.which == sdl2.SDL_JoystickInstanceID(
                    sdl2.SDL_GameControllerGetJoystick(self.controller)
                ):
                    print("Controller disconnected")
                    sdl2.SDL_GameControllerClose(self.controller)
                    self.controller = None
                yield event

    def _update_button_state(self, button, pressed):
        """Update internal state from button event"""
        attr_name = self.state.BUTTON_NAMES.get(button)
        if attr_name:
            setattr(self.state, attr_name, pressed)

    def _update_axis_state(self, axis, normalized_value):
        """Update internal state from axis event"""
        attr_name = self.state.AXIS_NAMES.get(axis)
        if attr_name:
            setattr(self.state, attr_name, normalized_value)

    def get_state(self):
        """Get current gamepad state"""
        return self.state

    def run_loop(self, update_rate_hz=20.0):
        """
        Run continuous update loop.

        Args:
            update_rate_hz: Update frequency in Hz
        """
        self._running = True
        sleep_time = 1.0 / update_rate_hz

        print(f"Starting gamepad loop at {update_rate_hz} Hz")
        try:
            while self._running:
                self.update()
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("Gamepad loop interrupted by user")
        finally:
            self.stop()

    def run_loop_events(self, update_rate_hz=20.0):
        """
        Run continuous event-based update loop.
        Uses SDL2's native event system - more efficient than polling.
        """
        self._running = True
        sleep_time = 1.0 / update_rate_hz

        print(f"Starting event-based gamepad loop at {update_rate_hz} Hz")
        try:
            while self._running:
                for event in self.update_events():
                    pass  # State is updated automatically
                time.sleep(sleep_time)  # Small sleep to prevent busy loop
        except KeyboardInterrupt:
            print("Gamepad event loop interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the gamepad loop"""
        self._running = False
        print("Gamepad handler stopped")


# Example usage for standalone testing
def example_usage():
    """Example of how to use the GamepadHandler standalone"""

    # Create handler
    handler = GamepadHandler(deadzone=0.1)

    print("Gamepad handler example")
    print("Use gamepad sticks and buttons, press Ctrl+C to quit")

    try:
        while True:
            # Print out the states of all the buttons and axes
            if handler.update():
                state = handler.get_state()
                output = (
                    f"\rLeft Stick: ({state.js_left_x:5.4f}, {state.js_left_y:5.4f}; {int(state.button_stick_left)}) | "
                    f"Right Stick: ({state.js_right_x:5.4f}, {state.js_right_y:5.4f}; {int(state.button_stick_right)}) | "
                    f"Triggers: L{state.trigger_left:.4f} R{state.trigger_right:.4f} | "
                    f"Shoulders: LB[{int(state.button_shoulder_left)}] RB[{int(state.button_shoulder_right)}] | "
                    f"Action Buttons: A(cross)[{int(state.button_bottom)}] B(circle)[{int(state.button_right)}] X(square)[{int(state.button_left)}] Y(triangle)[{int(state.button_top)}] | "
                    f"Middle Buttons: View(Share, left)[{int(state.button_middle_left)}] Menu(Options, right)[{int(state.button_middle_right)}] Logo[{int(state.button_middle_logo)}] | "
                    f"D-pad: ↑{int(state.button_dpad_up)}↓{int(state.button_dpad_down)}←{int(state.button_dpad_left)}→{int(state.button_dpad_right)}"
                )
                print(output, end='', flush=True)

            time.sleep(0.05)  # 20 Hz
    except KeyboardInterrupt:
        print("Example stopped")


def example_usage_events():
    """Example using SDL2's native event system (more efficient)"""

    # Create handler
    handler = GamepadHandler(deadzone=0.1)

    print("SDL2 Event-Based Gamepad Example")
    print("This uses SDL2's native event system - only actual changes are shown.")
    print("Use gamepad sticks and buttons, press Ctrl+C to quit")

    try:
        print("Waiting for controller events...\n")
        while True:
            # Process SDL2 events - yields only actual changes!
            for event in handler.update_events():
                event_name = handler.state.EVENT_NAMES.get(event.type)
                if event_name == 'button_down':
                    button_name = handler.state.BUTTON_NAMES.get(event.cbutton.button)
                    if button_name:
                        print(f"[{event.cbutton.timestamp:10d}ms] PRESSED:  {button_name}")

                elif event_name == 'button_up':
                    button_name = handler.state.BUTTON_NAMES.get(event.cbutton.button)
                    if button_name:
                        print(f"[{event.cbutton.timestamp:10d}ms] RELEASED: {button_name}")

                elif event_name == 'axis_motion':
                    axis_name = handler.state.AXIS_NAMES.get(event.caxis.axis)
                    # Method 1: Use the returned event to calculate normalized value for display
                    # if axis_name:
                    #     if axis_name in ['trigger_left', 'trigger_right']:
                    #         normalized = handler.normalize_trigger_input(event.caxis.value)
                    #     else:
                    #         normalized = handler.normalize_stick_input(event.caxis.value)
                    # Method 2: Alternatively, get the current state from the handler
                    normalized = getattr(handler.state, axis_name)

                    # Only print if significant (avoid spam from noise)
                    if abs(normalized) > 0.01:
                        print(f"[{event.caxis.timestamp:10d}ms] AXIS:     {axis_name:10s} = {normalized:6.3f}")

            # Small sleep to prevent busy loop (events are queued by SDL2)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nEvent-based example stopped")


if __name__ == '__main__':
    if SDL2_AVAILABLE:
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == '--event':
            example_usage_events()
        else:
            # Default to polling example for compatibility
            example_usage()
    else:
        print("Please install the 'pysdl2' module to run this example:")
        print("pip install pysdl2")

"""
TROUBLESHOOTING GAMEPAD MAPPING ISSUES WITH PySDL2:
===================================================

PySDL2 provides standardized controller mapping through SDL's GameController API.
This reduces mapping issues compared to raw input libraries.

1. CONTROLLER COMPATIBILITY:
   - SDL2 has built-in database for popular controllers (Xbox, PS4, PS5, Switch Pro, etc.)
   - Automatic mapping to standard button/axis layout
   - Custom controller mappings can be added via SDL_GAMECONTROLLERCONFIG

2. UNSUPPORTED CONTROLLERS:
   - If controller isn't detected, check SDL2 controller database
   - Add custom mapping string for your controller
   - Use SDL2 controller mapping tools online

3. TESTING SDL2 CONTROLLERS:
   Python testing:
   - Run this script to see controller detection and input
   - Check SDL_NumJoysticks() and SDL_IsGameController() results

4. COMMON ISSUES:
   - Controller not detected: May not be in SDL2 database
   - Wrong button mapping: Controller may be detected as joystick, not gamecontroller
   - Wireless issues: Ensure proper Bluetooth pairing

5. DEBUGGING COMMANDS:
   Linux:
   - `lsusb` - List USB devices to identify controller
   - `evtest` - See raw Linux input events
   - `jstest /dev/input/js0` - Test basic joystick functionality

   SDL2-specific:
   - Check SDL_GameControllerMapping() for your controller
   - Use SDL2 controller test programs

6. CONTROLLER MAPPING DATABASE:
   SDL2 maintains a community database of controller mappings.
   If your controller isn't working, check:
   - https://github.com/gabomdq/SDL_GameControllerDB
   - Submit new mappings for unsupported controllers

7. PERMISSIONS:
   - Add user to 'input' group: `sudo usermod -a -G input $USER`
   - Logout/login to apply changes
   - Modern systems usually handle this automatically

8. WIRELESS CONTROLLERS:
   - Ensure proper Bluetooth pairing
   - Check battery level (low battery causes issues)
   - Some controllers auto-sleep, press buttons to wake

9. MULTIPLE CONTROLLERS:
   - SDL2 handles multiple controllers automatically
   - Each controller gets a unique index
   - Use SDL_GameControllerName() to identify controllers
"""