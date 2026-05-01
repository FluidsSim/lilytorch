"""Control"""

import numpy as np
from dm_control.mjcf.physics import Physics
from farms_mujoco.simulation.task import ExperimentTask

import farms_core.pylog as pylog
from farms_core.model.data import AnimatData
from farms_core.array.types import NDARRAY_V1
from farms_core.model.options import AnimatOptions
from farms_core.model.extensions import AnimatExtension
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.control import AnimatController, ControlType
from farms_mujoco.simulation.extensions import create_cylinder

from ...integration.gamepad import GamepadHandler


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
