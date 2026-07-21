
import os
from farms_core.model.options import SpawnMode
from lilytorch.integration.camera import top_down_camera_config
from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.examples.base_sim_config import BaseSimConfig


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'examples', '_1guillasim',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.compute_sdf    = True
        self.convexify      = True
        self.use_gpu        = True
        self.headless       = True
        self.smagorinsky_cs = 0.25
        # self.n_samples   = (200, 200, 200)

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                "sdf_name"       : "1guilla.sdf",
                "control_type"   : "position",
                "gains"          : [100.0, 1.0, 0],
                "spawn_mode"     : SpawnMode.ROTZ,
                "pose"           : [-0.8, 0.05, 0., 0, 0, 0.0],
                "controller_path": "lilytorch.examples._1guillasim.pd_controller.PositionController",
                "control_pars"   : {'freq': 1, 'twl': 0.571429*14, 'amp': 15.0},
            },
        ]

        u_inlet = 0.215971

        # ── 3-D grid ─────────────────────────────────────────────────
        self.Nx   = 1024
        self.Ny   = 256
        self.Nz   = 128
        self.xmin = -0.9
        self.xmax =  1.14
        self.ymin = -0.255
        self.ymax =  0.255
        self.zmin = -0.1275
        self.zmax =  0.1275

        # ── Physics ───────────────────────────────────────────────────
        self.timestep          = 0.001
        self.convection_method = "implicit"
        self.n_iterations      = 20001
        self.save              = True
        self.save_every        = 200
        self.vmin              = -40
        self.vmax              = 40

        # ── BDIM solver ──────────────────────────────────────────────
        self.poisson_tol             = 1.0e-5
        self.poisson_max_cycles      = 30
        self.poisson_max_mgcg_cycles = 10
        self.poisson_precond_vcycles = 1
        self.poisson_warm_start      = True
        self.poisson_method          = "multigrid"
        self.dtype                   = "float32"
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "neumann"
        self.rho_body                = 1000.0
        self.zero_pressure_inside    = True

        # Disable compilation to speed up startup

        # ── Boundary conditions (3-D, Dirichlet inlet) ───────────────
        self.bc_type_u   = ["D", "D", "N", "N", "N", "N"]
        self.bc_values_u = [u_inlet, u_inlet, 0, 0, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D", "N", "N"]
        self.bc_values_v = [0, 0, 0, 0, 0, 0]
        self.bc_type_w   = ["N", "N", "N", "N", "D", "D"]
        self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.force_scaling         = 1
        self.interp_data_subfolder = "interp_data_3d"

    # ── Extensions ────────────────────────────────────────────────────

    def extra_simulation_extensions(self, output_folder):
        extensions = []

    #     # FlowViewer
    #     extensions.append({
    #         "loader": "lilytorch.integration.flow_viewer.FlowViewer",
    #         "config": {
    #             "field"        : "omega_z",
    #             "max_spheres"  : 4000,
    #             "iso_fraction" : 0.15,
    #             "smooth_sigma" : 0,
    #             "crop_boundary": 3,
    #             "sphere_size"  : 0.02,
    #             "update_every" : None,
    #         },
    #     })

    #     # Top-down camera auto-fitted to the pool
    #     cam = top_down_camera_config(
    #         self.xmin, self.xmax,
    #         self.ymin, self.ymax,
    #         self.zmin, self.zmax,
    #     )
    #     extensions.append({
    #         "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
    #         "config": {
    #             "path"            : os.path.join(output_folder, "output", "video.mp4"),
    #             "animat_id"       : None,
    #             "fps"             : 30,
    #             "speed"           : 1.0,
    #             "angular_velocity": 0,
    #             **cam,
    #         },
    #     })

        return extensions


if __name__ == "__main__":
    SimConfig().run()
