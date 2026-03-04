"""
FARMS TaskExtension for preCICE-OpenFOAM coupling.

Drop-in replacement for FluidExtension when using preCICE for 3D FSI.
Loads the PreCICEHandler instead of the BDIMhandler.
"""

import numpy as np
from farms_core.simulation.extensions import TaskExtension
from farms_core.experiment.options import ExperimentOptions
from farms_core.experiment.data import ExperimentData
from farms_mujoco.simulation.task import ExperimentTask
from dm_control.mjcf.physics import Physics
import logging

logger = logging.getLogger(__name__)


class PreCICEExtension(TaskExtension):
    """
    FARMS extension that couples the MuJoCo simulation with OpenFOAM
    via preCICE for 3D fluid-structure interaction.
    """

    def __init__(
        self,
        experiment_options: ExperimentOptions,
        precice_config: str,
        stl_folder: str,
        rho_fluid: float = 1000.0,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.data: ExperimentData | None = None
        self.precice_config = precice_config
        self.stl_folder = stl_folder
        self.rho_fluid = rho_fluid
        self.n_animats = len(self.experiment_options.animats)
        self._initialized = False

    @classmethod
    def from_options(
        cls,
        config: dict,
        experiment_options: ExperimentOptions,
    ):
        """From options"""
        return cls(
            experiment_options=experiment_options,
            precice_config=config.get("precice_config", "precice-config.xml"),
            stl_folder=config.get("stl_folder", ""),
            rho_fluid=config.get("rho_fluid", 1000.0),
        )

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        """Initialize the preCICE handler on the first episode."""
        if not self._initialized:
            from lilytorch.farms_examples._1guillasim.precice_coupling.precice_handler import PreCICEHandler

            # Gather link names from the first animat
            animat_opts = self.experiment_options.animats[0]
            link_names = [link.name for link in animat_opts.morphology.links]

            config = {
                "precice_config": self.precice_config,
                "stl_folder": self.stl_folder,
                "link_names": link_names,
                "rho_fluid": self.rho_fluid,
            }

            self.handler = PreCICEHandler(
                config,
                task.data.animats,
                physics,
            )
            self._initialized = True
            logger.info(
                f"PreCICE extension initialized: "
                f"{len(link_names)} links, stl_folder={self.stl_folder}"
            )

    def before_step(self, task: ExperimentTask, action, physics: Physics):
        """Called before each MuJoCo step — runs the preCICE coupling."""
        self.handler.step(task, physics)

    def end_episode(self, task: ExperimentTask, physics: Physics):
        """Finalize preCICE when the episode ends."""
        if self._initialized and hasattr(self, 'handler'):
            self.handler.finalize()
