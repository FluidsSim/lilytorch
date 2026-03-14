
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
        self.use_gpu  = True
        self.use_bdim = True
        self.headless = True
        self.nu       = 450.0e-6

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                "sdf_name"       : "1guilla.sdf",
                "control_type"   : "position",
                "gains"          : [100.0, 4.0, 0],
                "spawn_mode"     : SpawnMode.FREE,
                "pose"           : [-0.07, 0, 0., 0, 0, 3.141592653589793],
                "controller_path": "lilytorch.farms_examples._1guillasim.pd_controller.PositionController",
                "control_pars"   : {'freq': 0.5, 'twl': 12, 'amp': 40.0},
            },
        ]

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
        self.poisson_compile         = True
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "free"
        self.compile_adv_diff        = True

        # ── Boundary conditions (3-D, all Neumann) ───────────────────
        self.bc_type_u   = ["D", "D", "N", "N", "N", "N"]
        self.bc_values_u = [0, 0, 0, 0, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D", "N", "N"]
        self.bc_values_v = [0, 0, 0, 0, 0, 0]
        self.bc_type_w   = ["N", "N", "N", "N", "D", "D"]
        self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.force_scaling         = 1.0
        self.interp_data_subfolder = "interp_data_3d"

    # ── Extensions ────────────────────────────────────────────────────

    def extra_simulation_extensions(self, output_folder):
        extensions = []

        # FlowViewer (works headless via CameraRecording)
        extensions.append({
            "loader": "lilytorch.integration.flow_viewer.FlowViewer",
            "config": {
                "field"        : "omega_z",
                "max_spheres"  : 4000,
                "iso_fraction" : 0.15,
                "smooth_sigma" : 2.5,
                "crop_boundary": 3,
                "sphere_size"  : 0.01,
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
