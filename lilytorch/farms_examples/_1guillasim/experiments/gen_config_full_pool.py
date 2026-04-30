
import os
from farms_core.model.options import SpawnMode
from lilytorch.integration.flow_viewer import FlowViewer
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
        self.use_gpu              = True
        self.use_bdim             = False
        self.compute_sdf          = True
        self.convexify            = True
        self.headless             = False
        self.smagorinsky_cs       = 0.2

        self.streaming_sdf_3d = True
        self.streaming_forces_3d = True
        self.force_shared_union = True
        self.mu_normals_union = True
        self.bdim_union = True
        self.force_narrow_batch = True

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                "sdf_name"       : "1guilla.sdf",
                "control_type"   : "position",
                "gains"          : [100.0, 1., 0],
                "spawn_mode"     : SpawnMode.FREE,
                "pose"           : [4.75, 0.1, 0.0, 0, 0, 0.05],
                "controller_path": "lilytorch.farms_examples._1guillasim.experiments.controller.PositionController",
                "control_pars"   : {
                    "file_path": os.path.join(
                        self.data_folder, "/data/andreaferrario/1guilla_experiments/swim/log/ms007mpt001log.csv"
                    ),
                },
            },
        ]

        # ── 3-D grid ─────────────────────────────────────────────────
        self.Nx   = 900
        self.Ny   = 300
        self.Nz   = 52
        self.xmin = 0
        self.xmax = 6
        self.ymin = -1
        self.ymax = 1
        self.zmin = -(2/300*52/2)
        self.zmax = (2/300*52/2)

        # ── Physics ───────────────────────────────────────────────────
        self.rho_body          = 1000.0
        self.timestep          = 0.001
        self.convection_method = "quick"
        self.n_iterations      = 20001
        self.save_every        = 200
        self.vmin              = -10.0
        self.vmax              = 10.0
        self.save              = False

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
        self.poisson_method          = "multigrid"
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "free"
        self.zero_pressure_inside    = True

        self.poisson_compile         = True
        self.compile_adv_diff        = True
        self.compile_forces          = True
        self.compile_sdf             = True

        # # u: no-penetration on x-walls, free-slip on y/z-walls
        # self.bc_type_u   = ["D", "D", "N", "N", "N", "N"]
        # self.bc_values_u = [0, 0, 0, 0, 0, 0]
        # # v: free-slip on x/z-walls, no-penetration on y-walls
        # self.bc_type_v   = ["N", "N", "D", "D", "N", "N"]
        # self.bc_values_v = [0, 0, 0, 0, 0, 0]
        # # w: free-slip on x/y-walls, no-penetration on z-walls
        # self.bc_type_w   = ["N", "N", "N", "N", "D", "D"]
        # self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # ── Boundary conditions (3-D, all Dirichlet / no-slip) ───────
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

        # # FlowViewer (works headless via CameraRecording)
        # extensions.append({
        #     "loader": "lilytorch.integration.flow_viewer.FlowViewer",
        #     "config": {
        #         "field"        : "omega_z",
        #         "max_spheres"  : 800000,
        #         "iso_fraction" : 0.15,
        #         "smooth_sigma" : 2.5,
        #         "crop_boundary": 3,
        #         "sphere_size"  : 0.01,
        #         "update_every" : None,
        #     },
        # })

        # extensions.append({
        #     "loader": "lilytorch.integration.particle_viewer.ParticleViewer",
        #     "config": {
        #         "seed_mode"       : "boundary",
        #         "max_particles"   : 800000,
        #         "seed_n_particles": 3,
        #         "seed_interval"   : 1,
        #         "turb_diffusivity": 0.00001,
        #         "sphere_size"     : 0.003,
        #         "particle_color"  : [255/256, 0.0, 0.0, 0.6],
        #         "trail_length"    : 0,
        #         "update_every"    : None,
        #         "n_z_layers"      : 1,
        #         "floor_color"     : "#5B5B5B63",
        #         "body_color"      : "#B5A425",
        #         # "light_color"      : [0.05, 0.12, 0.85, 1.0], # blue lamp
        #         # "emissive_particles": True,                   # glow independent of light
        #         "save_particles"   : True,
        #         "save_dir"         : os.path.join(output_folder, "particles"),
        #         "save_every"       : self.save_every,
        #     }
        # })

        # Top-down camera auto-fitted to the pool
        cam = top_down_camera_config(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.zmin, self.zmax,
            overshoot=1
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
