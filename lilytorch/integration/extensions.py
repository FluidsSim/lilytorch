
import time
import collections

import mujoco
import numpy as np
from dm_control.mjcf.physics import Physics
from dm_control.viewer import views as _dm_views
from farms_core.experiment.data import ExperimentData
from farms_core.experiment.options import ExperimentOptions
from farms_core.extensions.extensions import import_item
from farms_core.io.hdf5 import dict_to_hdf5
from farms_core.simulation.extensions import TaskExtension
from farms_mujoco.simulation.task import ExperimentTask

class DummyOptionCallback(TaskExtension):

    def __init__(
            self,
            experiment_options: ExperimentOptions,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.data: ExperimentData | None = None

    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
    ):
        """From options"""
        return cls(
            experiment_options=experiment_options,
        )



class FluidExtension(TaskExtension):

    def __init__(
            self,
            experiment_options: ExperimentOptions,
            handler_path: str,
            bdim_yaml: str,
    ):
        # substep=True so that ``before_step`` runs on every MuJoCo
        # substep (cb_sub_steps).  The fluid solver itself is still
        # advanced only on the full step; the substep calls only
        # re-apply the cached body forces to xfrc_applied, mirroring
        # the way FARMS' SwimmingExtension keeps xfrc fresh against
        # anything that may clear it between substeps (e.g. the
        # interactive viewer's perturbation handling).
        super().__init__(substep=True)
        self.experiment_options = experiment_options
        self.data: ExperimentData | None = None
        self.n_animats = len(self.experiment_options.animats)
        self.BDIMhandler_class = import_item(handler_path)
        self.bdim_yaml = bdim_yaml

        self.initialization_it = 0



    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
    ):
        """From options"""
        return cls(
            experiment_options=experiment_options,
            handler_path=config.get("handler_path", ""),
            bdim_yaml=config.get("bdim_yaml", ""),
        )

    def initialize_forces(self, force: str):
        force_array = []
        for animat in self.experiment_options.animats:
            force_array.append(np.zeros(len(animat.control.sensors.xfrc)))
        setattr(self, force, force_array)


    def initialize_episode(self, task: ExperimentTask, physics: Physics):

        if self.initialization_it == 0:
            """Initialize episode"""
            # self.forces = ["friction_force_lin_x", "friction_force_lin_y", "friction_force_ang_z",
            #         "pressure_force_x", "pressure_force_y", "pressure_force_ang_z"]
            # for force in self.forces:
            #     self.initialize_forces(force)
            runtime_nt = self.experiment_options.simulation.runtime.n_iterations
            physics_dt = self.experiment_options.simulation.physics.timestep

            user_nt = self.bdim_yaml["solver"].get("nt", None)
            user_dt = self.bdim_yaml["solver"].get("dt", None)
            if user_nt is not None and int(user_nt) != int(runtime_nt):
                print(
                    "[FluidExtension] overriding bdim_yaml.solver.nt "
                    f"({user_nt}) with runtime.n_iterations ({runtime_nt})."
                )
            if user_dt is not None and float(user_dt) != float(physics_dt):
                print(
                    "[FluidExtension] overriding bdim_yaml.solver.dt "
                    f"({user_dt}) with physics.timestep ({physics_dt}). "
                    "Set physics.timestep to change the coupled fluid timestep."
                )

            self.bdim_yaml["solver"]["nt"] = runtime_nt
            self.bdim_yaml["solver"]["dt"] = physics_dt  # enforce farms timestep
            self.bdim_yaml["body"]["experiment_options"] = self.experiment_options

            self.BDIMhandler = self.BDIMhandler_class(self.bdim_yaml, task.data.animats, physics)
            self.initialization_it += 1


    # def after_step(self, task: ExperimentTask, physics: Physics):
    def before_step(self, task: ExperimentTask, action, physics: Physics):
        # Full step: advance the fluid solver one timestep, compute new
        # body forces, and write them to xfrc_applied.
        # Sub-step: do not advance the fluid solver — re-apply the
        # previously cached body forces to xfrc_applied so they survive
        # any clearing that may happen between MuJoCo substeps.
        full_step = not task.sim_iteration % task.cb_sub_steps
        if full_step:
            self.BDIMhandler.step(task, physics)
        else:
            self.BDIMhandler.apply_forces(task, physics)

    def end_episode(self, task: ExperimentTask, physics: Physics):
        """Flush per-link force/torque histories to ``<save_path>/drags.h5``.

        FARMS-coupled runs do not call ``FluidSolver.run_sim`` (the step
        loop is driven by MuJoCo), so the drag records would otherwise
        never be written. We also block on pending HDF5 I/O so that the
        file is fully on disk before the task finishes.
        """
        del task, physics
        fs = getattr(self.BDIMhandler, "fluid_solver", None)
        if fs is None:
            return
        if getattr(fs, "compute_forces", False) and getattr(fs, "save_drags", False):
            try:
                fs.save_drags_h5()
                fs.flush_io()
            except Exception as exc:
                print(f"[FluidExtension] save_drags_h5 failed: {exc}")


class _RTColumnModel(_dm_views.ColumnTextModel):
    """Dynamic two-column text model for dm_control viewer RT overlay."""

    def __init__(self):
        self._text = "..."

    def get_columns(self):
        return [("RT", self._text)]


class RealtimeMonitor(TaskExtension):
    """Overlay the realtime factor in the top-right corner of the MuJoCo viewer.

    Works with both viewer backends:
    - Native MuJoCo passive viewer (default): uses ``viewer.set_texts()``
    - dm_control Application viewer: uses ``_viewer_layout`` ColumnTextView

    No-ops silently in headless mode.

    Parameters
    ----------
    window : int
        Rolling window size (steps) for averaging wall-clock step time.
    """

    def __init__(
            self,
            experiment_options: ExperimentOptions,
            window: int = 30,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self._window = window
        self._times: collections.deque = collections.deque(maxlen=window)
        self._last_t: float | None = None
        self._last_display_t: float = 0.0
        self._physics_dt: float = 0.0
        self._rt_model: _RTColumnModel | None = None

    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
    ):
        return cls(
            experiment_options=experiment_options,
            window=config.get("window", 30),
        )

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        self._physics_dt = float(
            self.experiment_options.simulation.physics.timestep
        )
        self._last_t = None
        self._times.clear()
        self._rt_model = None

        # dm_control viewer: register a persistent overlay panel
        app = getattr(task, "_app", None)
        if app is not None:
            layout = getattr(app, "_viewer_layout", None)
            if layout is not None:
                self._rt_model = _RTColumnModel()
                rt_view = _dm_views.ColumnTextView(self._rt_model)
                layout.add(rt_view, _dm_views.PanelLocation.TOP_RIGHT)

    def before_step(self, task: ExperimentTask, action, physics: Physics):
        now = time.perf_counter()
        if self._last_t is not None:
            self._times.append(now - self._last_t)
        self._last_t = now

        # Rate-limit viewer updates to ~5 Hz to avoid starving the render thread.
        if now - self._last_display_t < 0.2:
            return
        self._last_display_t = now

        if self._times:
            mean_dt = sum(self._times) / len(self._times)
            rt = self._physics_dt / mean_dt if mean_dt > 0 else 0.0
        else:
            rt = 0.0

        # dm_control viewer path
        if self._rt_model is not None:
            self._rt_model._text = f"{rt:.2f}x"
            return

        # Native MuJoCo passive viewer path
        viewer = getattr(task, "viewer", None)
        if viewer is None:
            return
        set_texts = getattr(viewer, "set_texts", None)
        if set_texts is None:
            return
        set_texts([(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "RT",
            f"{rt:.2f}x",
        )])


class PhysicsOptionsExtension(TaskExtension):
    """Apply global MuJoCo geom contact parameters at episode start."""

    def __init__(
            self,
            experiment_options: ExperimentOptions,
            physics_options: dict | None,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.physics_options = physics_options or {}

    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
    ):
        """From options"""
        return cls(
            experiment_options=experiment_options,
            physics_options=config.get("physics_options", {}),
        )

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        del task

        solref = self.physics_options.get("solref", None)
        if solref is not None:
            physics.model.geom_solref[:, 0] = solref[0]
            physics.model.geom_solref[:, 1] = solref[1]

        solimp = self.physics_options.get("solimp", None)
        if solimp is not None:
            physics.model.geom_solimp[:, :] = solimp


class DataLogger(TaskExtension):
    """
    Log data attached through external extensions.
    The data is expected to be stored in animat_data.record as a dictionary.
    """

    def __init__(
            self,
            experiment_options: ExperimentOptions,
            log_path: str,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.log_path = log_path

    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
    ):
        """From options"""
        return cls(
            experiment_options=experiment_options,
            log_path=config.get("log_path", ""),
        )

    def initialize_episode(self, task: TaskExtension, physics: Physics):
        """Iteration 0"""
        del physics
        self.data = task.data

    def end_episode(self, task: ExperimentTask, physics: Physics):
        self.experiment_options.animats
        # if data is not set or no animats, do nothing
        if not hasattr(self, "data") or self.data is None:
            return
        records = []
        for idx, animat_data in enumerate(self.data.animats):
            if hasattr(animat_data, "record"):
                records.append(animat_data.record)
        # nothing to log
        if not records:
            return
        data = {"animats": records}
        dict_to_hdf5(filename=self.log_path, data=data)





