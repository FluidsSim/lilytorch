
from farms_core.model.control import AnimatController
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from farms_core.model.control import ControlType
import numpy as np
from lilytorch.util.rw import Dict2Class
from farms_amphibious.control.kinematics import KinematicsController


class PositionController(KinematicsController):
    """Traveling-wave position controller for the undulatory submarine.

    A sinusoidal wave of lateral (yaw) joint displacement propagates from
    head to tail at frequency *freq* Hz and traveling wavelength *twl*
    body-lengths, generating forward thrust (carangiform BCF locomotion).

    Config keys
    -----------
    freq : float
        Oscillation frequency [Hz].
    twl  : float
        Travelling wave length expressed as a multiple of body segments
        (analogous to the zebrafish / salamander controllers).
    amp  : float
        Peak joint angle amplitude [degrees].
    """

    def __init__(self, animat_data, animat_options, experiment_options, config, animat_i):

        joints_names = animat_options.control.joints_names()
        timestep = experiment_options.simulation.physics.timestep
        n_iterations = experiment_options.simulation.runtime.n_iterations
        t_end = timestep * n_iterations

        kinematics = self.generate_positions(
            tstop=t_end,
            sampling_rate=1.0 / timestep,
            wlength=1,
            amp_deg=config["amp"],
            freq=config["freq"],
            TWL=config["twl"],
            nmotors=len(joints_names),
            plot=False,
        )

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
            joints_names=joints_names_per_type,
            kinematics=kinematics,
            sampling=timestep,
            indices=range(1, len(joints_names) + 1),
            time_index=0,
            invert_motors=False,
            degrees=False,
            timestep=timestep,
            n_iterations=n_iterations,
            animat_data=animat_data,
            max_torques=max_torques_per_type,
            init_time=0.0,
            end_time=t_end,
        )

        self.animat_data = animat_data
        self.animat_options = animat_options
        self.experiment_options = experiment_options
        self.config = Dict2Class(config)
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
        """From options"""
        return cls(
            animat_data=animat_data,
            animat_options=animat_options,
            experiment_options=experiment_options,
            config=config,
            animat_i=animat_i,
        )

    def generate_positions(
        self,
        tstop=3,
        sampling_rate=100,
        wlength=1,
        amp_deg=20.0,
        freq=1.0,
        nmotors=4,
        TWL=8,
        plot=False,
    ):
        """Return a (N_timesteps, 1 + nmotors) kinematics array.

        The first column is time; remaining columns are joint angles [rad].
        The traveling wave is:
            theta_i(t) = amp * factor_i * sin(2*pi*(wlength*i/TWL - freq*t))
        where *factor_i* increases from head to tail so that the tail
        segment oscillates with larger amplitude (typical of carangiform
        swimmers).
        """
        amp = amp_deg * (np.pi / 180.0)
        times = np.expand_dims(np.arange(0, tstop, 1.0 / sampling_rate), axis=1)
        times_expanded = np.repeat(times, nmotors, axis=1)

        idxs = np.arange(nmotors)
        x = (idxs + 1) / nmotors
        # Amplitude envelope: small at head, larger at tail
        c1, c2, c3 = 0.05, -0.13, 0.28
        factor = c1 + c2 * x + c3 * x ** 2

        thetas = amp * factor * np.sin(
            2 * np.pi * (wlength * idxs / TWL - freq * times_expanded)
        )

        return np.column_stack([times, thetas])

    def step(self, iteration, time, timestep):
        """Positions are precomputed; no additional step logic required."""
        pass
