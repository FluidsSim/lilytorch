
import os
import numpy as np
from farms_core.model.options import SpawnMode
from lilytorch.integration.camera import top_down_camera_config
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
        self.use_gpu      = True
        self.compute_sdf  = True
        self.convexify    = True
        self.wall_height  = 0.2
        self.water_height = 0.1
        self.stack_folder = "salamander"

        self.solver_method    = "fused"
        # self.poisson_compile  = True
        # self.compile_adv_diff = True
        # self.compile_forces   = True
        # self.compile_sdf      = True

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                "sdf_name"       : "1guilla.sdf",
                "control_type"   : "position",
                "gains"          : [100.0, 1., 0],
                "spawn_mode"     : SpawnMode.TRANSVERSE,
                "pose"           : [0, 0., 0.1, 0, 0, 3.141592653589793],
                # "controller_config": {
                #     'path'      : "lilytorch.farms_examples.salamander_gamepad.control.PositionController",
                #     'freq'      : 1,
                #     'twl'       : 10,
                #     'amp'       : 100,
                #     'limb_pose1': -0.35 * 3.141592653589793,
                #     'limb_pose2': -0.2 * 3.141592653589793,
                # },
                "controller_path": "lilytorch.farms_examples._1guillasim.experiments.controller.PositionController",
                "control_pars"   : {
                    "file_path": os.path.join(
                        self.data_folder, "/data/andreaferrario/1guilla_experiments/swim/log/ms007mpt001log.csv"
                    ),
                },
            },
        ]

        # ── 2-D grid ─────────────────────────────────────────────────
        self.Nx   = 1024
        self.Ny   = 512
        # self.Nx   = 1024
        # self.Ny   = 256
        self.xmin = -0.9
        self.xmax =  5.1
        self.ymin = -1.5
        self.ymax =  1.5

        # ── Physics ───────────────────────────────────────────────────
        self.poisson_method    = "fft"
        self.timestep          = 0.01
        self.convection_method = "implicit"
        self.n_iterations      = 80001
        self.save_frames       = False
        self.num_sub_steps     = 1

        # # ── MuJoCo ───────────────────────────────────────────────────
        # self.visual_scale = 10.0
        # self.extent       = 10.0

        # ── Arena ────────────────────────────────────────────────────
        self.wall_thickness = 0.003
        self.wall_height    = 0.03
        self.arena_pose     = [0, 0, 0, 0, 0, 0]
        self.water_drag     = False
        self.water_buoyancy = False

        # ── BDIM solver ──────────────────────────────────────────────
        self.bdim_dt      = self.timestep
        self.bdim_nt      = self.n_iterations
        self.rho_body = 800.0

        # ── Body ─────────────────────────────────────────────────────
        self.contour_mask = False
        self.zero_pressure_inside = True
        # self.n_samples    = (2000, 2000)
        self.force_scaling = 0.04

    # ── Hooks ─────────────────────────────────────────────────────────

    def customize_joint_initials(self, joints_list):
        for joint in joints_list:
            if joint['name'] in ("joint_leg_0_L_0", "joint_leg_0_R_0"):
                joint['initial'] = [-np.pi / 3, 0.0]
            if joint['name'] in ("joint_leg_0_L_3", "joint_leg_0_R_3"):
                joint['initial'] = [-np.pi / 4, 0.0]
            if joint['name'] in ("joint_leg_1_L_0", "joint_leg_1_R_0"):
                joint['initial'] = [-np.pi / 3, 0.0]
            if joint['name'] in ("joint_leg_1_L_3", "joint_leg_1_R_3"):
                joint['initial'] = [-np.pi / 4, 0.0]

    # ── Extensions ────────────────────────────────────────────────────

    def extra_simulation_extensions(self, output_folder):
        extensions = []

        # FlowViewer2D – overlay 2-D flow field on the MuJoCo viewer
        extensions.append({
            "loader": "lilytorch.integration.flow_viewer_2d_gpu.FlowViewer2D",
            "config": {
                "field"         : "curl",
                "nx_vis"        : 512,
                "ny_vis"        : 128,
                "alpha"         : 1,
                "z_offset"      : 0.015,
                "smooth_sigma"  : 0,
                "crop_boundary" : 0,
                "update_every"  : 5,
                "synchronize_cuda": False,
                "vmin"          : -10,
                "vmax"          : 10,
            },
        })

        extensions.append({
            "loader": "lilytorch.integration.extensions.RealtimeMonitor",
            "config": {
                "window": 30,
            },
        })

        # cam = top_down_camera_config(
        #     self.xmin, self.xmax,
        #     self.ymin, self.ymax,
        # )
        # extensions.append({
        #     "loader": "farms_mujoco.sensors.camera.CameraRecording",
        #     "config": {
        #         "path"            : os.path.join(output_folder, "output", "video.mp4"),
        #         "animat_id"       : None,
        #         "fps"             : 30,
        #         "speed"           : 1.0,
        #         "angular_velocity": 0,
        #         **cam,
        #     },
        # })

        return extensions

if __name__ == "__main__":
    SimConfig().run()
