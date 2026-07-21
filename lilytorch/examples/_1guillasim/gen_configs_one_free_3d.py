
import os
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'examples', '_1guillasim',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.use_gpu        = True
        self.use_bdim       = False
        self.headless       = False
        self.smagorinsky_cs = 0.2

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                "sdf_name"       : "1guilla.sdf",
                "control_type"   : "position",
                "gains"          : [20.0, 4.0, 0],
                "spawn_mode"     : SpawnMode.FREE,
                "pose"           : [-0.07, 0, 0.0, 0, 0, 3.141592653589793],
                "controller_path": "lilytorch.examples._1guillasim.pd_controller.PositionController",
                "control_pars"   : {'freq': 0.5, 'twl': 12, 'amp': 20.0},
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

        # ── 3-D grid ─────────────────────────────────────────────────
        self.Nx   = 512
        self.Ny   = 256
        self.Nz   = 64
        self.xmin = -0.9
        self.xmax = 1.5
        self.ymin = -0.6
        self.ymax = 0.6
        self.zmin = -0.15
        self.zmax = 0.15


        # ── Physics ───────────────────────────────────────────────────
        self.rho_body          = 1000.0
        self.timestep          = 0.001
        self.convection_method = "abdquickest"
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
        self.poisson_warm_start      = True
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "free"

        # ── Boundary conditions (3-D, all Neumann) ───────────────────
        self.bc_type_u   = ["D", "D", "D", "D", "D", "D"]
        self.bc_values_u = [0, 0, 0, 0, 0, 0]
        self.bc_type_v   = ["D", "D", "D", "D", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0, 0, 0]
        self.bc_type_w   = ["D", "D", "D", "D", "D", "D"]
        self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.force_scaling         = 1.0
        self.interp_data_subfolder = "interp_data_3d"

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
                "max_particles"   : 800000,
                "seed_n_particles": 3,
                "seed_interval"   : 1,
                "turb_diffusivity": 0.00001,
                "sphere_size"     : 0.003,
                "particle_color"  : [255/256, 0.0, 166/256, 0.85],   #FF00A6
                "trail_length"    : 0,
                "update_every"    : None,
                "n_z_layers"      : 1,
                "floor_color"     : "#FFFFFF",                       # dark blue floor
                "body_color"      : "#C0AD1E",                       # near-black robot
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
            overshoot=2
        )
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
