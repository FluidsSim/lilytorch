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
from scipy.signal import butter, filtfilt

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
        kinematics_invert = config.get("kinematics_invert", True)
        kinematics_degrees = False  # xlsx values are already in radians
        kinematics_start = 0.0
        kinematics_end = (
            experiment_options.simulation.physics.timestep
            * experiment_options.simulation.runtime.n_iterations
        )

        kinematics = self.load_positions(file_path)

        # Trim/check columns vs. number of position joints.
        n_pos_joints = len(joints_names)

        # Model-angle files (ep223 / ep248) pack a leading time column that
        # gives 1 + n_joints columns.  Extract time, derive the true sampling
        # rate, and strip the column so columns 1..N map to the N position-
        # controlled joints.
        if kinematics.shape[1] == n_pos_joints + 1:
            time_col = kinematics[:, 0]
            # Use the median diff to be robust against floating-point jitter.
            kinematics_sampling = float(np.median(np.diff(time_col)))
            kinematics = kinematics[:, 1:]

        if kinematics.shape[1] < n_pos_joints:
            raise ValueError(
                f"Trajectory has {kinematics.shape[1]} joint columns but the model "
                f"declares {n_pos_joints} joints."
            )
        kinematics = kinematics[:, :n_pos_joints]

        # Snapshot raw kinematics for optional plotting (before filtering).
        _kinematics_raw = kinematics.copy()

        # ── Low-pass filter ──────────────────────────────────────────
        # High-acceleration kinematics (e.g. ep223 fast model angles)
        # can trigger mjWARN_BADQACC.  A gentle Butterworth low-pass
        # removes the high-frequency content that the PD controller
        # cannot track.
        lp_cutoff = config.get("lowpass_cutoff", None)
        if lp_cutoff is not None and lp_cutoff > 0:
            fs = 1.0 / kinematics_sampling
            nyq = 0.5 * fs
            order = config.get("lowpass_order", 4)
            b, a = butter(order, lp_cutoff / nyq, btype="low")
            # filtfilt applies forward-backward → zero-phase, order is
            # effectively doubled.
            kinematics = np.column_stack([
                filtfilt(b, a, kinematics[:, j])
                for j in range(kinematics.shape[1])
            ])

        # ── Optional kinematics plot (before zero-row prepend) ───────
        if config.get("plot_kinematics", False):
            self._plot_kinematics(
                _kinematics_raw, kinematics, kinematics_sampling,
                lp_cutoff, os.getcwd(), file_path,
            )

        # If the first frame has non-zero joint targets but the MuJoCo
        # model initialises all joints to zero, the step discontinuity can
        # trigger mjWARN_BADQACC.  Prepend a zero-row so that the first
        # PD error is zero.
        if not np.allclose(kinematics[0], 0.0):
            kinematics = np.vstack([np.zeros_like(kinematics[0]), kinematics])
            kinematics_start += kinematics_sampling

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
    def _plot_kinematics(
        raw: "np.ndarray",
        filt: "np.ndarray",
        fs: float,
        cutoff: float | None,
        output_dir: str,
        file_path: str,
    ):
        """Save a raw-vs-filtered kinematics comparison plot to *output_dir*."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_joints = raw.shape[1]
        n_cols = 3
        n_rows = int(np.ceil(n_joints / n_cols))

        t_raw = np.arange(raw.shape[0]) / fs
        t_filt = np.arange(filt.shape[0]) / fs

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 5, n_rows * 2.5),
            sharex=True,
        )
        axes = axes.flatten()

        for j in range(n_joints):
            ax = axes[j]
            ax.plot(t_raw, raw[:, j], color="silver", linewidth=0.6, label="raw")
            ax.plot(t_filt, filt[:, j], color="steelblue", linewidth=1.0, label="filtered")
            ax.set_ylabel(f"joint_{j}", fontsize=7)
            ax.tick_params(labelsize=6)

        # Hide unused subplots.
        for j in range(n_joints, len(axes)):
            axes[j].set_visible(False)

        title = f"Kinematics — {os.path.basename(file_path)}"
        if cutoff:
            title += f"  |  lowpass {cutoff:.0f} Hz"
        fig.suptitle(title, fontsize=9, y=1.01)
        fig.legend(
            ["raw", "filtered"], loc="lower center",
            ncol=2, fontsize=7, frameon=False,
        )
        fig.tight_layout()

        out_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(file_path))[0] + "_kinematics.png",
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def load_positions(file_path):
        """Read joint trajectory xlsx → ``(n_samples, n_joints)`` array."""
        df = pd.read_excel(file_path)
        return df.to_numpy(dtype=float)

    def step(self, iteration, time, timestep):
        """Positions are looked up via KinematicsController.positions()."""
        pass
