
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
        self.compute_sdf    = False
        self.convexify      = True
        self.use_gpu        = True
        self.use_bdim       = True
        self.headless       = False
        self.smagorinsky_cs = 0.
        self.carreau        = None   # Step 1: constant-ν baseline + sponge
        self.sponge         = {
            "width"   : 0.3,        # sponge layer thickness          [m]
            "strength": 20.0,       # max damping coefficient σ_max   [1/s]
        }
        self.nu             = 450.0 * 1e-6
        self.force_scaling  = 1
        self.rho_body       = 1000.0


        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                "sdf_name"       : "1guilla_800.sdf",
                "control_type"   : "position",
                "gains"          : [100.0, 1., 0],
                "spawn_mode"     : SpawnMode.FREE,
                "pose"           : [0.1, -0.2, 0., 0, 0, 2.9],
                "controller_path": "lilytorch.farms_examples._1guillasim.pd_controller.PositionController",
                "control_pars"   : {'freq': 0.5, 'twl': 14, 'amp': 40.0},
            },
        ]

        # ── 3-D grid  (221×150×43 cm pool, uniform h = 0.43/64) ─────
        mult      = 1
        self.Nx   = 336*mult
        self.Ny   = 224*mult
        self.Nz   = 64*mult
        self.xmin = -0.9
        self.xmax = 1.3575
        self.ymin = -0.7525
        self.ymax = 0.7525
        self.zmin = -0.215
        self.zmax = 0.215


        # self.Nx   = 1024
        # self.Ny   = 256
        # self.Nz   = 128

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
        self.compile_forces          = True
        self.compile_sdf             = True
        self.poisson_compile         = True

        # ── Boundary conditions (3-D, all no-slip Dirichlet) ────────
        self.bc_type_u   = ["D", "D", "D", "D", "D", "D"]
        self.bc_values_u = [0, 0, 0, 0, 0, 0]
        self.bc_type_v   = ["D", "D", "D", "D", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0, 0, 0]
        self.bc_type_w   = ["D", "D", "D", "D", "D", "D"]
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
                "particle_color"   : [0.545098039, 1.0, 0.0, 0.85],
                "trail_length"     : 0,
                "update_every"     : None,
                "n_z_layers"       : 1,
                "floor_color"      : "#0A1866",               # dark blue floor
                "body_color"       : "#050505",               # near-black robot
                "light_color"      : [0.05, 0.12, 0.85, 1.0], # blue lamp
                "emissive_particles": False,                   # glow independent of light
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
