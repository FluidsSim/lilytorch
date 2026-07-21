
import os
import numpy as np
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.examples.base_sim_config import BaseSimConfig


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'examples', '_1guillasim',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.use_gpu = False

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
                "spawn_mode": SpawnMode.TRANSVERSE,
                "pose"      : [0, 0, 0, 0, 0, 3.141592653589793],
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
                "spawn_mode": SpawnMode.TRANSVERSE,
                "pose"      : [1, 0, 0, 0, 0, 3.141592653589793],
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
        self.n_iterations      = 30001
        self.save_every        = 50

        # ── BDIM solver ──────────────────────────────────────────────
        self.dtype    = "float64"
        self.rho_body = 800.0

        # ── Boundary conditions (2-D, Dirichlet inlet) ───────────────
        self.bc_type_u   = ["D", "D", "N", "N"]
        self.bc_values_u = [-0.1, -0.1, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.force_scaling = 0.04
        self.n_samples     = (2000, 2000)

    # ── Hooks ─────────────────────────────────────────────────────────

    def customize_animat(self, animat_i, animat_pars, n_joints, index):
        mc = animat_pars["muscle_config"]
        if animat_i == 0:
            mc["initial_state"]   = np.roll(
                np.linspace(0, -2 * np.pi, n_joints), index
            )
            mc["go_straight"]     = True
            mc["weight_feedback"] = 20.0
            mc["freq"]            = 0.6
        else:
            mc["go_straight"]     = True
            mc["freq"]            = 0.7
            mc["weight_feedback"] = 0.0

    def run(self):
        for i in range(8):
            self.single_run(i)


if __name__ == "__main__":
    SimConfig().run()
