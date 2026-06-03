
import os
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.farms_examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', '_1guillasim',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.use_gpu                       = True
        self.use_bdim                      = True
        self.compute_sdf                   = True
        self.convexify                     = True
        self.force_method                  = "eulerian"
        self.zero_pressure_inside          = True
        self.body_velocity_blend_eps_cells = None
        self.bdim_mu0_projection           = False
        self.bdim_body_div_correction      = True
        self.poisson_method                = "multigrid"
        self.smagorinsky_cs                = 0.
        self.compile_adv_diff              = True
        self.poisson_warm_start            = True

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                "sdf_name"       : "1guilla.sdf",
                "control_type"   : "position",
                "gains"          : [100.0, 1., 0],
                "spawn_mode"     : SpawnMode.TRANSVERSE,
                "pose"           : [-0.07, 0, 0.0, 0, 0, 3.],
                "controller_path": "lilytorch.farms_examples._1guillasim.experiments.controller.PositionController",
                "control_pars"   : {
                    "file_path": os.path.join(
                        self.data_folder, "experiments/robot_data/robot_data_log_2025-09-01_16_13_04.csv"
                    ),
                },
            },
        ]

        # ── 3-D grid ─────────────────────────────────────────────────
        self.Nx   = 512
        self.Ny   = 256
        self.Nz   = 32
        self.xmin = -0.9
        self.xmax = 1.5
        self.ymin = -0.6
        self.ymax = 0.6
        self.zmin = -0.075
        self.zmax = 0.075

        # # ── 3-D grid ─────────────────────────────────────────────────
        # self.Nx   = 800
        # self.Ny   = 400
        # self.Nz   = 50
        # self.xmin = -0.9
        # self.xmax = 1.5
        # self.ymin = -0.6
        # self.ymax = 0.6
        # self.zmin = -0.075
        # self.zmax = 0.075

        # # ── 3-D grid ─────────────────────────────────────────────────
        # self.Nx   = 400
        # self.Ny   = 200
        # self.Nz   = 25
        # self.xmin = -0.9
        # self.xmax = 1.5
        # self.ymin = -0.6
        # self.ymax = 0.6
        # self.zmin = -0.075
        # self.zmax = 0.075


        # ── Physics ───────────────────────────────────────────────────
        self.rho_body          = 800.0
        self.timestep          = 0.001
        self.convection_method = "quick"
        self.n_iterations      = 15001
        self.save_every        = 200
        self.vmin              = -10.0
        self.vmax              = 10.0
        self.save              = True

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale  = 10.0
        self.extent        = 100.0

        # ── BDIM solver ──────────────────────────────────────────────
        self.bdim_dt                 = self.timestep
        self.bdim_nt                 = self.n_iterations + 1
        self.poisson_tol             = 1.0e-4
        self.poisson_max_cycles      = 30
        self.poisson_max_mgcg_cycles = 10
        self.poisson_precond_vcycles = 1
        self.poisson_warm_start      = False
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "neumann"
        self.zero_pressure_inside    = True

        self.compile_adv_diff        = False

        # u: no-penetration on x-walls, free-slip on y/z-walls
        self.bc_type_u   = ["D", "D", "N", "N", "N", "N"]
        self.bc_values_u = [0, 0, 0, 0, 0, 0]
        # v: free-slip on x/z-walls, no-penetration on y-walls
        self.bc_type_v   = ["N", "N", "D", "D", "N", "N"]
        self.bc_values_v = [0, 0, 0, 0, 0, 0]
        # w: free-slip on x/y-walls, no-penetration on z-walls
        self.bc_type_w   = ["N", "N", "N", "N", "D", "D"]
        self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # # ── Boundary conditions  ───────────────────
        # self.bc_type_u   = ["D", "D", "D", "D", "D", "D"]
        # self.bc_values_u = [0, 0, 0, 0, 0, 0]
        # self.bc_type_v   = ["D", "D", "D", "D", "D", "D"]
        # self.bc_values_v = [0, 0, 0, 0, 0, 0]
        # self.bc_type_w   = ["D", "D", "D", "D", "D", "D"]
        # self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.force_scaling         = 1.0
        self.interp_data_subfolder = "interp_data_3d"

        # ── Visualization ───────────────────────────────────────────────
        self.floor_color       = "#484848"
        self.wall_alpha        = 1.
        self.water_alpha       = 0.05

    # ── Extensions ────────────────────────────────────────────────────

    def extra_simulation_extensions(self, output_folder):
        extensions = []

        # FlowViewer (works headless via CameraRecording)
        # extensions.append({
        #     "loader": "lilytorch.integration.flow_viewer.FlowViewer",
        #     "config": {
        #         "field"        : "omega_z",
        #         "max_spheres"  : 4000,
        #         "iso_fraction" : 0.15,
        #         "smooth_sigma" : 2.5,
        #         "crop_boundary": 3,
        #         "sphere_size"  : 0.01,
        #         "update_every" : None,
        #     },
        # })

        extensions.append({
            "loader": "lilytorch.integration.particle_viewer.ParticleViewer",
            "config": {
                "max_particles"   : 1800000,
                "seed_n_particles": 3,
                "seed_interval"   : 5,
                "turb_diffusivity": 0.0000,
                "sphere_size"     : 0.003,
                "particle_color"  : "#FF00A699",
                "trail_length"    : 0,
                "update_every"    : 1,
                "n_z_layers"      : 3,
                "z_spread_fraction": 0.1,     # fraction of body z-thickness (0.9=default, <0.5=tighter)
                "z_center"        : 0.0,     # None=body midpoint; 0.0=domain mid-plane
                "floor_color"     : "#5B5B5B63",
                "body_color"      : "#D09F23",
                # "light_color"      : [0.05, 0.12, 0.85, 1.0], # blue lamp
                # "emissive_particles": True,                   # glow independent of light
                "save_particles"   : True,
                "save_dir"         : os.path.join(output_folder, "particles"),
                "save_every"       : self.save_every,
            }
        })

        # Top-down camera auto-fitted to the pool
        cam = top_down_camera_config(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.zmin, self.zmax,
            overshoot=1.5,
            max_width=3840, max_height=2160,
        )

        # Soft, reflection-free lighting for tank recordings
        extensions.append({
            "loader": "lilytorch.integration.light_modifier.LightModifier",
            "config": {
                "diffuse": [1, 1, 1],
                "ambient": [0.3, 0.3, 0.3],
            },
        })


        extensions.append({
            "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
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
