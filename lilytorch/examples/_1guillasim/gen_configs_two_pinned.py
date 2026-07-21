
import os
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.examples.base_sim_config import BaseSimConfig


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'examples', '_1guillasim',
        )

        self.freqs = [
            0.4, 0.43, 0.47, 0.5, 0.53, 0.57, 0.6, 0.63, 0.67,
            0.7, 0.73, 0.77, 0.8, 0.83, 0.87, 0.9, 0.93, 0.97, 1.0,
        ]

        # ── Hardware ──────────────────────────────────────────────────
        self.use_gpu  = False
        self.headless = True

        # ── Animats ───────────────────────────────────────────────────
        self.filter_fixed_joints = False

        self.animats_pars = [
            {
                "model_name"   : "1guilla",
                "sdf_name"     : "1guilla.sdf",
                "control_type" : "torque",
                "muscle_loader": "farms_ekeberg.src.ekeberg.EkebergMuscleController",
                "muscle_config": {
                    'load_controller': 'lilytorch.examples._1guillasim.network.PAOscillatorController',
                    'method'         : 'implicit',
                    'muscle_pars'    : os.path.join(self.data_folder, 'muscle_params_od.csv'),
                },
                "gains"     : [0, 0, 0],
                "spawn_mode": SpawnMode.FIXED,
                "pose"      : [1, 0, 0, 0, 0, 3.141592653589793],
            },
            {
                "model_name"   : "1guilla",
                "sdf_name"     : "1guilla.sdf",
                "control_type" : "torque",
                "muscle_loader": "farms_ekeberg.src.ekeberg.EkebergMuscleController",
                "muscle_config": {
                    'load_controller': 'lilytorch.examples._1guillasim.network.PAOscillatorController',
                    'method'         : 'implicit',
                    'muscle_pars'    : os.path.join(self.data_folder, 'muscle_params_od.csv'),
                },
                "gains"     : [0, 0, 0],
                "spawn_mode": SpawnMode.FIXED,
                "pose"      : [2, 0, 0, 0, 0, 3.141592653589793],
            },
        ]

        # ── 2-D grid ─────────────────────────────────────────────────
        self.Nx   = 512
        self.Ny   = 256
        self.xmin = -0.9
        self.xmax =  2.1
        self.ymin = -0.75
        self.ymax =  0.75

        # ── Physics ───────────────────────────────────────────────────
        self.timestep          = 0.001
        self.convection_method = "abdquickest"
        self.n_iterations      = 40001
        self.save_every        = 1000
        self.vmin              = -40
        self.vmax              =  40

        # ── BDIM solver ──────────────────────────────────────────────
        self.dtype    = "float64"
        self.rho_body = 800.0

        # ── Boundary conditions ──────────────────────────────────────
        self.bc_type_u   = ["D", "D", "N", "N"]
        self.bc_values_u = [-0.5, -0.5, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.force_scaling = 0.04
        self.n_samples     = (2000, 2000)

    # ── Hooks ─────────────────────────────────────────────────────────

    def customize_animat(self, animat_i, animat_pars, n_joints, index):
        mc = animat_pars["muscle_config"]
        mc["go_straight"] = False
        if animat_i == 0:
            mc["freq"] = 0.7
        if animat_i == 1:
            mc["freq"] = self.freqs[index]

    def run(self):
        for i, _freq in enumerate(self.freqs):
            self.single_run(i)


if __name__ == "__main__":
    SimConfig().run()
