import os
import numpy as np
from farms_core.model.options import SpawnMode
from lilytorch.integration.camera import top_down_camera_config
from lilytorch.util.paths import lilytorch_repo_root, sdfs_path
from lilytorch.farms_examples.base_sim_config import BaseSimConfig


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.compute_sdf = True

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', 'salamander',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.use_bdim = True
        self.use_gpu  = True
        self.headless = False   # FlowViewer works in headless too

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"  : "salamander_v5",
                "sdf_name"    : "sdf/salamander.sdf",
                "control_type": "position",
                "gains"       : [0.001, .0002, 0],
                "controller_config": {
                    'path'      : "lilytorch.farms_examples.salamander.pd_controller_swim.PositionController",
                    'freq'      : 1,
                    'twl'       : 10,
                    'amp'       : 200,
                    'limb_pose1': -0.35 * 3.141592653589793,
                    'limb_pose2': -0.2 * 3.141592653589793,
                },
                "spawn_mode": SpawnMode.TRANSVERSE,
                "pose"      : [-0.1, 0, 0.0, 0, 0, 3.141592653589793],
            },
        ]

        # ── 3-D grid ─────────────────────────────────────────────────
        self.Nx   = 512
        self.Ny   = 256
        self.Nz   = 128
        self.xmin = -0.13
        self.xmax =  0.27
        self.ymin = -0.05
        self.ymax =  0.05
        self.zmin = -0.03
        self.zmax =  0.03

        # ── Physics ───────────────────────────────────────────────────
        self.timestep          = 0.01
        self.convection_method = "quick"
        self.n_iterations      = 1001
        self.save_every        = 50
        self.num_sub_steps     = 1

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 10.0
        self.extent       = 10.0

        # ── Arena ────────────────────────────────────────────────────
        self.wall_thickness = 0.003
        self.arena_pose     = [0, 0, 0, 0, 0, 0]
        self.water_drag     = False
        self.water_buoyancy = False

        # ── BDIM solver ──────────────────────────────────────────────
        self.bdim_dt                 = self.timestep
        self.bdim_nt                 = self.n_iterations
        self.bdim_handler_path       = "lilytorch.integration.BDIMhandler.BDIMhandler"
        self.zero_pressure_inside    = False
        self.rho_body                = 900.0
        self.poisson_method          = "multigrid"
        self.poisson_tol             = 1.0e-4
        self.poisson_max_cycles      = 30
        self.poisson_max_mgcg_cycles = 10
        self.poisson_precond_vcycles = 1
        self.poisson_warm_start      = True
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "free"
        self.compile_adv_diff        = True

        # ── Boundary conditions (3-D, all Neumann) ───────────────────
        self.bc_type_u   = ["D", "D", "N", "N", "N", "N"]
        self.bc_values_u = [0.0, 0.0, 0, 0, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D", "N", "N"]
        self.bc_values_v = [0, 0, 0, 0, 0, 0]
        self.bc_type_w   = ["N", "N", "N", "N", "D", "D"]
        self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.contour_mask          = True
        self.n_samples             = (2000, 2000)
        self.force_scaling         = 1.0
        self.interp_data_subfolder = "interp_data_3d"

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

        # FlowViewer – 3-D iso-surface visualisation in MuJoCo viewer
        extensions.append({
            "loader": "lilytorch.integration.flow_viewer.FlowViewer",
            "config": {
                "field"        : "omega_z",
                "max_spheres"  : 4000,
                "iso_fraction" : 0.15,
                "smooth_sigma" : 2.5,
                "crop_boundary": 3,
                "sphere_size"  : 0.005,
                "update_every" : None,
            },
        })

        # Top-down camera auto-fitted to the pool
        cam = top_down_camera_config(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.zmin, self.zmax,
        )
        extensions.append({
            "loader": "farms_mujoco.sensors.camera.CameraRecording",
            "config": {
                "path"            : os.path.join(output_folder, "output", "video.mp4"),
                "animat_id"       : None,
                "fps"             : 30,
                "speed"           : 1.0,
                "angular_velocity": 0,
                **cam,
            },
        })

        return extensions

if __name__ == "__main__":
    SimConfig().run()
