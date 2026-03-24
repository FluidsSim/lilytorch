"""Gen-config for the single sphere drop (Gazzola et al.) benchmark.

Usage::

    python -m lilytorch.farms_examples.single_sphere_drop_gazzola.gen_configs

This uses the custom BDIMhandler in controller.py (Heun time-stepping with
variable density and explicit gravity) rather than the unified BDIMhandler.
"""

import os

from farms_core.io.yaml import pyobject2yaml
from farms_core.model.options import SpawnMode

from lilytorch.farms_examples.base_sim_config import BaseSimConfig
from lilytorch.util.paths import sdfs_path


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        # ── Paths ──────────────────────────────────────────────────────
        self.data_folder = os.path.dirname(os.path.abspath(__file__))
        self.bdim_handler_path = (
            "lilytorch.farms_examples.single_sphere_drop_gazzola.controller.BDIMhandler"
        )

        # ── Hardware ───────────────────────────────────────────────────
        self.use_gpu  = True
        self.nthreads = 16
        self.use_bdim = True

        # ── Simulation flags ───────────────────────────────────────────
        self.headless = False
        self.fast     = False

        # ── Grid (2-D, tall domain for sedimentation) ─────────────────
        self.Nx   = 256
        self.Ny   = 2048
        self.xmin = -0.02
        self.xmax = 0.02
        self.ymin = 0.0
        self.ymax = 0.32

        # ── Fluid / body physics ──────────────────────────────────────
        self.rho       = 996.0
        self.rho_body  = 1010.0
        self.nu        = 8.0e-7
        self.timestep  = 0.0001
        self.n_iterations = 100001

        # ── BDIM solver ───────────────────────────────────────────────
        self.convection_method       = "abdquickest"
        self.bdim_dt                 = 0.0001
        self.bdim_nt                 = 150000
        self.poisson_tol             = 1.0e-7
        self.poisson_max_cycles      = 10
        self.poisson_max_mgcg_cycles = 5
        self.jacobi_weight           = 0.7
        self.poisson_nsmoothing      = 5
        self.poisson_verbose         = True

        # ── Compilation (disabled — custom controller does its own stepping)
        self.compile_adv_diff = False
        self.compile_forces   = False
        self.poisson_compile  = False

        # ── BCs (all Neumann) ─────────────────────────────────────────
        self.bc_type_u   = ["D", "D", "N", "N"]
        self.bc_values_u = [0, 0, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0]

        # ── Body ──────────────────────────────────────────────────────
        self.n_samples  = [1000, 1000]
        self.convexify  = True
        self.suit       = 0.0

        # ── Output ────────────────────────────────────────────────────
        self.save_frames = True
        self.save_every  = 500
        self.save        = True
        self.vmin        = -5.0
        self.vmax        = 5.0

        # ── MuJoCo ───────────────────────────────────────────────────
        self.num_sub_steps = 1
        self.cb_sub_steps  = 1
        self.visual_scale  = 1.0
        self.extent        = 100.0
        self.shadow_size   = 1024

        # Disable drag (BDIM handles fluid forces)
        self.generate_pool = False
        self.use_drag      = False

    # ── Override: sphere has no joints / controller ───────────────────

    def gen_animat_config(self, output_folder, index=0):
        sphere_sdf = os.path.join(sdfs_path, "sphere", "sphere.sdf")
        animat_dict = {
            "spawn": {
                "loader"  : 0,
                "mode"    : SpawnMode.SAGITTAL,
                "pose"    : [0.0, 0.0, 0.3, 0, 0, 0],
                "velocity": [0, 0, 0, 0, 0, 0],
                "extras"  : {},
            },
            "sdf": sphere_sdf,
            "morphology": {
                "links": [
                    {
                        "name"      : "base_link",
                        "collisions": True,
                        "friction"  : [0, 0, 0],
                        "density"   : 1,
                    },
                ],
                "joints"          : [],
                "self_collisions" : [],
            },
            "control": {
                "controller_loader": [],
                "sensors": {
                    "links"    : ["base_link"],
                    "joints"   : [],
                    "contacts" : [],
                    "xfrc"     : ["base_link"],
                    "muscles"  : [],
                    "adhesions": [],
                    "visuals"  : [],
                },
                "motors"      : [],
                "hill_muscles": [],
            },
            "extensions": [],
        }
        pyobject2yaml(
            os.path.join(output_folder, "animat_config_0.yaml"),
            animat_dict,
        )

    def gen_experiment_config(self, output_folder, index=0):
        experiment_dict = {
            "simulation": "simulation_config.yaml",
            "arenas"    : ["arena_config.yaml"],
            "animats"   : ["animat_config_0.yaml"],
            "loaders"   : {
                "simulation_options": "farms_core.simulation.options.SimulationOptions",
                "animats_options"   : ["farms_core.model.options.AnimatOptions"],
                "arenas_options"    : ["farms_core.model.options.ArenaOptions"],
                "experiment_data"   : "farms_core.experiment.data.ExperimentData",
                "animats_data"      : ["farms_core.model.data.AnimatData"],
            },
        }
        pyobject2yaml(
            os.path.join(output_folder, "experiment_config.yaml"),
            experiment_dict,
        )


if __name__ == "__main__":
    SimConfig().run()
