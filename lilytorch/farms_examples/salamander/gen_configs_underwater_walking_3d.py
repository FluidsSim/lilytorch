
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
        self.use_bdim     = True
        self.use_drag     = True
        self.use_gpu      = True
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
                    "animat_id": 0,
                },
                "gains"     : [0, 0, 0],
                "spawn_mode": SpawnMode.FREE,
                "pose"      : [0, 0, 0.002, 0, 0, 3.141592653589793],
            },
            {
                "model_name"   : "salamander_v5",
                "sdf_name"     : "sdf/salamander3d.sdf",
                "control_type" : "torque",
                "muscle_loader": "farms_ekeberg.src.ekeberg.EkebergMuscleController",
                "muscle_config": {
                    'load_controller': 'lilytorch.farms_examples.salamander.network.PAOscillatorController',
                    'method'         : 'implicit',
                    'muscle_pars'    : os.path.join(self.data_folder, 'muscle_params.csv'),
                    "animat_id": 1,
                },
                "gains"     : [0, 0, 0],
                "spawn_mode": SpawnMode.FREE,
                "pose"      : [0, -0.05, 0.002, 0, 0, 3.141592653589793],
            },
            # {
            #     "model_name"   : "salamander_v5",
            #     "sdf_name"     : "sdf/salamander3d.sdf",
            #     "control_type" : "torque",
            #     "muscle_loader": "farms_ekeberg.src.ekeberg.EkebergMuscleController",
            #     "muscle_config": {
            #         'load_controller': 'lilytorch.farms_examples.salamander.network.PAOscillatorController',
            #         'method'         : 'implicit',
            #         'muscle_pars'    : os.path.join(self.data_folder, 'muscle_params.csv'),
            #         "animat_id": 2,
            #     },
            #     "gains"     : [0, 0, 0],
            #     "spawn_mode": SpawnMode.FREE,
            #     "pose"      : [0, 0.05, 0.002, 0, 0, 3.141592653589793],
            # },
        ]

        self.bdim_physics = {"solref": [-1000, -400.0]}

        # ── 3-D grid ─────────────────────────────────────────────────
        self.Nx   = 640
        self.Ny   = 320
        self.Nz   = 64
        self.xmin = -0.13
        self.xmax =  0.27
        self.ymin = -0.1
        self.ymax =  0.1
        self.zmin = -0.01
        self.zmax =  0.03

        # ── Physics ───────────────────────────────────────────────────
        self.timestep          = 0.001
        self.convection_method = "abdquickest"
        self.n_iterations      = 50001
        self.save_every        = 50
        self.save              = True
        self.num_sub_steps     = 6
        self.poisson_verbose   = True
        self.poisson_method    = "fft"

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

        # Abort the sim loudly if the fluid blows up instead of letting
        # NaN propagate silently (matplotlib renders NaN as transparent).
        self.vmax_abort = 50.0

        # ── Boundary conditions ──────────────────────────────────────
        self.bc_type_u   = ["D", "D", "N", "N", "N", "N"]
        self.bc_values_u = [0.0, 0.0, 0, 0, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0, 0, 0]
        self.bc_type_w   = ["N", "N", "N", "N", "D", "D"]
        self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.contour_mask = True
        self.interp_data_subfolder = "interp_data_3d"

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
                link["density"] = self.rho_body
                link["friction"] = [0.7, 0.0, 0.0]
                link["fluid_interaction"] = False
                link["drag_coefficients"] = [[0,0,0], [0,0,0]]


if __name__ == "__main__":
    SimConfig().run()
