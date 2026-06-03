
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
        self.compute_sdf          = True
        self.convexify            = False
        self.use_gpu              = True
        self.use_bdim             = True
        self.headless             = False
        self.smagorinsky_cs       = 0.2
        self.zero_pressure_inside = True

        # self.carreau = {
        #     "nu_0"  : 450.0e-6,
        #     "nu_inf": 450.0e-6,
        #     "lam"   : 0.5,
        #     "n"     : 1,
        #     "tau_y" : 0.0,   # enable yield stress
        # }
        self.yield_damping = {
            "gamma_c" : 20.0,   # shear-thinning onset ≈ 1/λ [s⁻¹]
            "strength": 10,     # damping half-life ≈ 0.03 s [s⁻¹]
        }

        # self.sponge         = {
        #     "width"   : 0.6,        # sponge layer thickness          [m]
        #     "strength": 4.0,       # max damping coefficient σ_max   [1/s]
        # }
        self.nu             = 450.0 * 1e-6
        self.force_scaling  = 1
        self.rho_body       = 1000.0


        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                "sdf_name"       : "1guilla.sdf",
                "control_type"   : "position",
                "gains"          : [100.0, 1., 0],
                "spawn_mode"     : SpawnMode.FREE,
                "pose"           : [0., -0.15, 0., 0, 0, 2.9],
                "controller_path": "lilytorch.farms_examples._1guillasim.experiments.controller.PositionController",
                "control_pars"   : {
                    "file_path": os.path.join(
                        self.data_folder, "experiments/robot_data/robot_data_log_2025-09-01_19_09_17.csv"
                    ),
                },
            },
        ]

        # ── 3-D grid  (≈221×150 cm pool, h = 1.50/224 ≈ 0.006696) ────
        mult      = 1
        h         = 1.50 / (224 * mult)
        self.Nx   = 330 * mult
        self.Ny   = 224 * mult
        self.Nz   = 64 * mult
        self.xmin = -0.9
        self.xmax = self.xmin + self.Nx * h
        self.ymin = -0.75
        self.ymax = 0.75
        self.zmin = -self.Nz * h / 2
        self.zmax =  self.Nz * h / 2


        # ── Physics ───────────────────────────────────────────────────
        self.timestep          = 0.003
        self.convection_method = "quick"
        self.n_iterations      = 5001
        self.save_every        = 100
        self.vmin              = -10.0
        self.vmax              = 10.0
        self.save              = True

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale  = 10.0
        self.extent        = 100.0

        # ── BDIM solver ──────────────────────────────────────────────
        self.bdim_dt                 = self.timestep
        self.bdim_nt                 = self.n_iterations + 1
        self.poisson_method          = "fft"
        self.poisson_tol             = 1.0e-4
        self.poisson_max_cycles      = 30
        self.poisson_max_mgcg_cycles = 10
        self.poisson_precond_vcycles = 1
        self.poisson_warm_start      = True
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "neumann"
        self.compile_adv_diff        = True

        # # ── Boundary conditions (3-D, all no-slip Dirichlet) ────────
        # self.bc_type_u   = ["D", "D", "D", "D", "D", "D"]
        # self.bc_values_u = [0, 0, 0, 0, 0, 0]
        # self.bc_type_v   = ["D", "D", "D", "D", "D", "D"]
        # self.bc_values_v = [0, 0, 0, 0, 0, 0]
        # self.bc_type_w   = ["D", "D", "D", "D", "D", "D"]
        # self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # u: no-penetration on x-walls, free-slip on y/z-walls
        self.bc_type_u   = ["D", "D", "N", "N", "N", "N"]
        self.bc_values_u = [0, 0, 0, 0, 0, 0]
        # v: free-slip on x/z-walls, no-penetration on y-walls
        self.bc_type_v   = ["N", "N", "D", "D", "N", "N"]
        self.bc_values_v = [0, 0, 0, 0, 0, 0]
        # w: free-slip on x/y-walls, no-penetration on z-walls
        self.bc_type_w   = ["N", "N", "N", "N", "D", "D"]
        self.bc_values_w = [0, 0, 0, 0, 0, 0]


        # ── Body ─────────────────────────────────────────────────────
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

        # ParticleViewer – Lagrangian dye particles
        # Fluorescein-under-blue-light setup:
        #   light_color  → blue excitation lamp  (~470 nm)
        #   particle_color → fluorescein emission (~520 nm, bright green-yellow)
        #   floor_color  → deep blue (appears blue under UV/blue light)
        #   body_color   → black (robot absorbs, no fluorescence)
        extensions.append({
            "loader": "lilytorch.integration.particle_viewer.ParticleViewer",
            "config": {
                "max_particles"    : 800000,
                "seed_n_particles" : 3,
                "seed_interval"    : 1,
                "turb_diffusivity" : 0.0,
                "sphere_size"      : 0.003,
                "particle_color"   : [0.545098039, 1.0, 0.0, 0.1],
                "trail_length"     : 0,
                "update_every"     : None,
                "n_z_layers"       : 1,
                "floor_color"      : "#0A1866",               # dark blue floor
                "body_color"       : "#050505",               # near-black robot
                "light_color"      : [0.05, 0.12, 0.85, 1.0], # blue lamp
                "emissive_particles": True,                   # glow independent of light — needed under blue lamp
                "save_particles"   : True,
                "save_dir"         : os.path.join(output_folder, "particles"),
                "save_every"       : self.save_every,
            },
        })

        # Top-down camera auto-fitted to the pool
        cam = top_down_camera_config(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.zmin, self.zmax,
            overshoot=3
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
