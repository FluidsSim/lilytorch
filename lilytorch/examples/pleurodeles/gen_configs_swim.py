
import os
import numpy as np
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root, sdfs_path
from lilytorch.examples.base_sim_config import BaseSimConfig


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'examples', 'pleurodeles',
        )
        self.compute_sdf = True

        # ── Hardware ──────────────────────────────────────────────────
        self.use_gpu  = True

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"  : "pleurodeles",
                "sdf_name"    : "salamander_animal_fmsv0.21_2D.sdf",
                "control_type": "position",
                "gains"       : [0.2, .005, 0],
                "controller_config": {
                    'path': "lilytorch.examples.pleurodeles.pd_controller_swim.PositionController",
                },
                "spawn_mode": SpawnMode.TRANSVERSE,
                "pose"      : [0, 0, 0., 0, 0, 3.141592653589793],
            },
        ]

        # ── 2-D grid ─────────────────────────────────────────────────
        self.Nx                  = 2048
        self.Ny                  = 512
        self.xmin                = -0.13 * 2
        self.xmax                = 0.27 * 2
        self.ymin                = -0.05 * 2
        self.ymax                = 0.05 * 2

        # ── Physics ───────────────────────────────────────────────────
        self.timestep          = 0.00025
        self.convection_method = "quick"
        self.n_iterations      = 10001
        self.save_every        = 50
        self.num_sub_steps     = 1

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 10.0
        self.extent       = 10.0

        # ── Arena ────────────────────────────────────────────────────
        self.wall_thickness = 0.003
        self.wall_height    = 0.03
        self.arena_pose     = [0, 0, -0.015, 0, 0, 0]
        self.water_drag     = False
        self.water_buoyancy = False

        # ── BDIM solver ──────────────────────────────────────────────
        self.bdim_dt                  = self.timestep
        self.bdim_nt                  = self.n_iterations
        self.zero_pressure_inside     = True
        self.rho_body                 = 800.0

        # ── Boundary conditions ──────────────────────────────────────
        self.bc_type_u   = ["D", "D", "N", "N"]
        self.bc_values_u = [0.0, 0.0, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.convexify    = False
        self.contour_mask = True
        self.n_samples    = (2000, 2000)

        # ── GPU performance ──────────────────────────────────────────

        # ── BDIM physics ─────────────────────────────────────────────
        self.bdim_physics = {"solref": [0.001, 0.5]}

    # ── Hooks ─────────────────────────────────────────────────────────

    def customize_joint_initials(self, joints_list):
        for joint in joints_list:
            if joint['name'] in ("joint_leg_0_L_0", "joint_leg_0_R_0"):
                joint['initial'] = [-0.3 * 3.141592653589793, -0]
            if joint['name'] in ("joint_leg_0_L_1", "joint_leg_0_R_1"):
                joint['initial'] = [-0.2 * 3.141592653589793, -0]
            if joint['name'] in ("joint_leg_1_L_0", "joint_leg_1_R_0"):
                joint['initial'] = [-0.35 * 3.141592653589793, -0]
            if joint['name'] in ("joint_leg_1_L_1", "joint_leg_1_R_1"):
                joint['initial'] = [-0.2 * 3.141592653589793, -0]

    def extra_simulation_extensions(self, output_folder):
        return [
            {
                "loader": "farms_mujoco.simulation.extensions.CameraFollower",
                "config": {
                    "animat_id"       : 0,
                    "distance"        : 0.8,
                    "azimuth"         : -30,
                    "elevation"       : -20,
                    "angular_velocity": 0,
                },
            },
            {
                "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
                "config": {
                    "path"            : os.path.join(output_folder, "output", "video.mp4"),
                    "animat_id"       : 0,
                    "fps"             : 30,
                    "speed"           : 1.0,
                    "azimuth"         : -30,
                    "elevation"       : -15,
                    "distance"        : 0.2,
                    "angular_velocity": 0,
                    "offset"          : [0, 0, 0.0],
                    "resolution"      : [1280, 720],
                },
            },
        ]


if __name__ == "__main__":
    SimConfig().run()
