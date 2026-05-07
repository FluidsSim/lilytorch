
import os
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.farms_examples.base_sim_config import BaseSimConfig


class SimConfig(BaseSimConfig):

    def __init__(self):
        self.freqs = [0.5]

        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', '_1guillasim',
        )

        # ── Simulation flags ──────────────────────────────────────────
        self.use_bdim    = True
        self.compute_sdf = True
        self.convexify   = True

        # ── Animats ───────────────────────────────────────────────────
        self.filter_fixed_joints = False

        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                "sdf_name"       : "1guilla.sdf",
                "control_type"   : "position",
                "gains"          : [100.0, 4.0, 0],
                "spawn_mode"     : SpawnMode.TRANSVERSE,
                "pose"           : [0, 0, 0.3, 0, 0, 3.141592653589793],
                "controller_path": "lilytorch.farms_examples._1guillasim.pd_controller.PositionController",
                "control_pars"   : {'freq': 1, 'twl': 12, 'amp': 30.0},
            },
        ]


        # ── 2-D grid ─────────────────────────────────────────────────
        self.Nx   = 1024
        self.Ny   = 512
        self.xmin = -0.9
        self.xmax =  5.1
        self.ymin = -1.5
        self.ymax =  1.5

        # ── Physics ───────────────────────────────────────────────────
        self.timestep          = 0.001
        self.convection_method = "abdquickest"
        self.n_iterations      = 18001
        self.save_every        = 50

        # ── BDIM solver (unused since use_bdim=False) ────────────────
        self.dtype    = "float64"
        self.rho_body = 800.0

        # ── Body ─────────────────────────────────────────────────────
        self.force_scaling = 0.04
        # self.n_samples     = (2000, 2000)

    # ── Hooks ─────────────────────────────────────────────────────────

    def customize_animat(self, animat_i, animat_pars, n_joints, index):
        animat_pars["control_pars"]["freq"] = float(self.freqs[index])

    def run(self):
        for i in range(len(self.freqs)):
            self.single_run(i)


if __name__ == "__main__":
    SimConfig().run()
