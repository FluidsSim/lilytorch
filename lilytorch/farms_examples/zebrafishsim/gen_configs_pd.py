
import os
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root, sdfs_path
from lilytorch.farms_examples.base_sim_config import BaseSimConfig


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', 'zebrafishsim',
        )

        # ── Hardware ──────────────────────────────────────────────────
        # self.compute_sdf = True
        self.use_gpu = False

        # ── Animats ───────────────────────────────────────────────────
        self.filter_fixed_joints = False

        self.constant_drags = [
            [-0.0, -0.0102864, -0.0080005],
            [0, 0, 0],
        ]

        self.animats_pars = [
            {
                "sdf_file"       : os.path.join(sdfs_path, "zebrafish", "zebrafish_v1_triangulated", "sdf", "zebrafish.sdf"),
                "control_type"   : "position",
                "gains"          : [0.001, .00002, 0],
                "spawn_mode"     : SpawnMode.TRANSVERSE,
                "pose"           : [0, 0, 0.005, 0, 0, 3.141592653589793],
                "controller_path": "lilytorch.farms_examples.zebrafishsim.pd_controller.PositionController",
                "control_pars"   : {'freq': 10.0, 'twl': 20, 'amp': 120},
            },
        ]

        # ── 2-D grid ─────────────────────────────────────────────────
        self.Nx   = 1024
        self.Ny   = 256
        self.xmin = -0.02
        self.xmax =  0.08
        self.ymin = -0.0125
        self.ymax =  0.0125

        # ── Physics ───────────────────────────────────────────────────
        self.timestep          = 0.001
        self.convection_method = "implicit"
        self.n_iterations      = 1501
        self.save_every        = 50
        self.cb_sub_steps      = 2

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 100.0
        self.extent       = 10.0

        # ── Arena ────────────────────────────────────────────────────
        self.wall_thickness = 0.003
        self.wall_height    = 0.01

        # ── BDIM solver ──────────────────────────────────────────────
        self.dtype                    = "float64"
        self.rho_body                 = 800.0
        self.zero_pressure_inside     = True

        # ── Boundary conditions ──────────────────────────────────────
        self.bc_type_u   = ["D", "D", "N", "N"]
        self.bc_values_u = [0., 0., 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.convexify = False
        self.n_samples = (2000, 2000)


if __name__ == "__main__":
    SimConfig().run()
