

import numpy as np
from farms_core.simulation.extensions import TaskExtension
from farms_core.experiment.options import ExperimentOptions
from farms_core.experiment.data import ExperimentData
from farms_core.extensions.extensions import import_item
from farms_mujoco.simulation.task import ExperimentTask
from farms_core.experiment.data import ExperimentData
from dm_control.mjcf.physics import Physics
import logging
logging.basicConfig(level=logging.ERROR)
from farms_core.io.hdf5 import dict_to_hdf5


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
        super().__init__()
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
            self.forces = ["friction_force_lin_x", "friction_force_lin_y", "friction_force_ang_z",
                    "pressure_force_x", "pressure_force_y", "pressure_force_ang_z"]
            for force in self.forces:
                self.initialize_forces(force)
            self.bdim_yaml["solver"]["nt"] = self.experiment_options.simulation.runtime.n_iterations
            self.bdim_yaml["solver"]["dt"] = self.experiment_options.simulation.physics.timestep  # enforce farms timestep
            self.bdim_yaml["body"]["experiment_options"] = self.experiment_options

            self.BDIMhandler = self.BDIMhandler_class(self.bdim_yaml, task.data.animats, physics)
            self.initialization_it += 1


    # def after_step(self, task: ExperimentTask, physics: Physics):
    def before_step(self, task: ExperimentTask, action, physics: Physics):

        self.BDIMhandler.step(task, physics)



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





