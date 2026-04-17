
import os
import numpy as np
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root, sdfs_path
from lilytorch.farms_examples.base_sim_config import BaseSimConfig


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', 'salamander',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.use_bdim     = False
        self.use_drag     = True
        self.use_gpu      = False
        self.compute_sdf  = False
        self.wall_height  = 0.02
        self.water_height = 0.015
        self.stack_folder = "salamander"

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"   : "salamander_v5",
                "sdf_name"     : "sdf/salamander3d.sdf",
                "control_type" : "torque",
                "muscle_loader": "farms_ekeberg.src.ekeberg.EkebergMuscleController",
                "muscle_config": {
                    'load_controller': 'lilytorch.farms_examples.salamander.network.PAOscillatorController',
                    'method'         : 'implicit',
                    'muscle_pars'    : os.path.join(self.data_folder, 'muscle_params.csv'),
                },
                "gains"     : [0, 0, 0],
                "spawn_mode": SpawnMode.FREE,
                "pose"      : [0, 0, 0.02, 0, 0, 3.141592653589793],
            },
        ]

        self.bdim_physics = {"solref": [-1000, -400.0]}

        # ── 2-D grid ─────────────────────────────────────────────────
        self.Nx   = 1024
        self.Ny   = 256
        self.xmin = -0.13
        self.xmax =  0.27
        self.ymin = -0.05
        self.ymax =  0.05

        # ── Physics ───────────────────────────────────────────────────
        self.timestep          = 0.001
        self.convection_method = "abdquickest"
        self.n_iterations      = 50001
        self.save_every        = 50
        self.num_sub_steps     = 5

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 10.0
        self.extent       = 10.0

        # ── Arena ────────────────────────────────────────────────────
        self.wall_thickness = 0.003
        self.wall_height    = 0.03
        self.arena_pose     = [0, 0, 0, 0, 0, 0]
        self.water_drag     = True
        self.water_buoyancy = True

        # ── BDIM solver ──────────────────────────────────────────────
        self.bdim_dt                  = self.timestep
        self.bdim_nt                  = self.n_iterations
        self.zero_pressure_inside     = True
        self.rho_body                 = 1050.0

        # ── Boundary conditions ──────────────────────────────────────
        self.bc_type_u   = ["D", "D", "N", "N"]
        self.bc_values_u = [0.0, 0.0, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.contour_mask = True
        self.n_samples    = (2000, 2000)

    # ── Hooks ─────────────────────────────────────────────────────────

    def customize_morphology_links(self, links_list, animat_i, animat_pars, index):
        del animat_i, animat_pars, index

        for link in links_list:
            if "passive" not in link["name"]:
                link["density"] = self.rho_body
                link["friction"] = [0.7, 0.0, 0.0]
                link["fluid_interaction"] = True
                if "foot" not in link["name"] and "leg" not in link["name"]:
                    link["drag_coefficients"] = [[-0.01, -0.06, -0.06], [-1e-6, -1e-6, -1e-6]]
                else:
                    link["drag_coefficients"] = [[-0.005, -0.01, -0.01], [-1e-9, -1e-9, -1e-9]]
            else:
                print(link["name"])
                link["density"] = self.rho_body
                link["friction"] = [0.7, 0.0, 0.0]
                link["fluid_interaction"] = False
                link["drag_coefficients"] = [[0,0,0], [0,0,0]]


if __name__ == "__main__":
    SimConfig().run()
