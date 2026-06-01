
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
        self.use_bdim       = True
        self.water_drag     = False
        self.water_buoyancy = False
        self.use_gpu        = True
        self.compute_sdf    = True
        self.stack_folder   = "salamander"

        self.solver_method    = "kernel"
        # self.compile_adv_diff = True

        self.bdim_physics = {"solref": [-100000.0, -500.0]}

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"  : "salamander_v4",
                "sdf_name"    : "sdf/salamander_no_passive.sdf",
                "control_type": "position",
                "gains"       : [0.0001, .00002, 0],
                "controller_config": {
                    'path'      : "lilytorch.farms_examples.salamander_gamepad.control.PositionController",
                },
                "spawn_mode": SpawnMode.TRANSVERSE,
                "pose"      : [0, 0, 0.015, 0, 0, 3.141592653589793],
            },
        ]

        # ── 2-D grid ─────────────────────────────────────────────────
        self.Nx   = 1024
        self.Ny   = 512
        # self.Nx   = 1024
        # self.Ny   = 256
        self.xmin = -0.4
        self.xmax =  0.4
        self.ymin = -0.2
        self.ymax =  0.2

        # ── Physics ───────────────────────────────────────────────────
        self.poisson_method    = "multigrid"
        self.poisson_bc_type   = "neumann"
        self.timestep          = 0.01
        self.convection_method = "implicit"
        self.n_iterations      = 80001
        self.save_frames       = False
        self.num_sub_steps     = 1


        self.force_method         = "lagrangian"
        # self.force_relaxation     = 0.05
        self.zero_pressure_inside = False
        self.bdim_mu0_projection  = False
        self.bdim_body_div_correction = True
        self.body_velocity_blend_eps_cells = 2
        # self.lagrangian_sample_offset = 2*(self.xmax - self.xmin) / self.Nx

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 10.0
        self.extent       = 10.0
        # top-view trackcom camera distance: pool is 0.8 m x 0.2 m,
        # camera fovy=70° → 0.5 m gives a comfortable fit
        self.camera_dist  = 0.5

        # ── Arena ────────────────────────────────────────────────────
        self.wall_thickness = 0.01
        self.wall_height    = 0.03
        self.water_height   = 0.015
        self.arena_pose     = [0, 0, 0, 0, 0, 0]


        # ── BDIM solver ──────────────────────────────────────────────
        self.bdim_dt      = self.timestep
        self.bdim_nt      = self.n_iterations
        self.rho_body     = 1000.0



        # ── Boundary conditions ──────────────────────────────────────
        self.bc_type_u   = ["D", "D", "D", "D"]
        self.bc_values_u = [0, 0, 0, 0]
        self.bc_type_v   = ["D", "D", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.contour_mask = False

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

    def customize_morphology_links(self, links_list, animat_i, animat_pars, index):
        del animat_i, animat_pars, index

        for link in links_list:
            link["density"] = self.rho_body
            link["friction"] = [0.7, 0.0, 0.0]
            link["fluid_interaction"] = True
            if "foot" not in link["name"] and "leg" not in link["name"]:
                link["drag_coefficients"] = [[-0.001, -0.3, -0.3], [-1e-9, -1e-9, -1e-9]]
            else:
                link["drag_coefficients"] = [[-0.001, -0.01, -0.01], [-1e-9, -1e-9, -1e-9]]

    # ── Extensions ────────────────────────────────────────────────────
    def extra_simulation_extensions(self, output_folder):
        extensions = []

        # FlowViewer2D – overlay 2-D flow field on the MuJoCo viewer
        extensions.append({
            "loader": "lilytorch.integration.flow_viewer_2d_gpu.FlowViewer2D",
            "config": {
                "field"         : "curl",
                "nx_vis"        : 1024,
                "ny_vis"        : 512,
                "alpha"         : 1,
                "z_offset"      : 0.015,
                "smooth_sigma"  : 0,
                "crop_boundary" : 0,
                "update_every"  : 1,
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
