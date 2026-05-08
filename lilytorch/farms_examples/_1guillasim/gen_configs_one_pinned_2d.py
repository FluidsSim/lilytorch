
import os
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.farms_examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.use_bdim       = True
        self.headless       = False
        self.compute_sdf    = True
        self.wall_thickness = 0.3
        self.wall_height    = 0.3

        self.poisson_method = "fft"

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', '_1guillasim',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.use_gpu = True

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                "sdf_name"       : "1guilla_link1_base.sdf",
                "control_type"   : "position",
                "gains"          : [50.0, .4, 0],
                "spawn_mode"     : SpawnMode.ROTZ,
                "pose"           : [-0.65, 0, 0.15, 0, 0, 0],
                "controller_path": "lilytorch.farms_examples._1guillasim.pd_controller_fixed_neck.PositionController",
                "control_pars"   : {'freq': 1, 'twl': 0.571429*14, 'amp': 15.0},
            },
        ]

        u_inlet = 0.215971

        # ── 2-D grid ─────────────────────────────────────────────────
        self.Nx   = 1024
        self.Ny   = 128
        self.xmin = -0.9
        self.xmax =  1.5
        self.ymin = -0.15
        self.ymax =  0.15

        # ── Physics ───────────────────────────────────────────────────
        self.timestep          = 0.0005
        self.convection_method = "abdquickest"
        self.n_iterations      = 10001
        self.save_every        = 200
        self.vmin              = -40 * u_inlet / 0.85
        self.vmax              = 40 * u_inlet / 0.85

        # ── BDIM solver ──────────────────────────────────────────────
        self.bdim_dt  = self.timestep
        self.bdim_nt  = self.n_iterations
        self.dtype    = "float64"
        self.rho_body = 800.0

        # ── Boundary conditions (2-D, Dirichlet inlet) ───────────────
        self.bc_type_u   = ["D", "D", "N", "N"]
        self.bc_values_u = [u_inlet, u_inlet, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.force_scaling         = 0.04
        self.interp_data_subfolder = "interp_data_2d"

    # ── Extensions ────────────────────────────────────────────────────

    def extra_simulation_extensions(self, output_folder):
        extensions = []

        # FlowViewer2D – overlay 2-D flow field on the MuJoCo viewer
        extensions.append({
            "loader": "lilytorch.integration.flow_viewer_2d.FlowViewer2D",
            "config": {
                "field"         : "curl",
                "nx_vis"        : 80,
                "ny_vis"        : 40,
                "alpha"         : 0.65,
                "z_offset"      : 0.005,
                "smooth_sigma"  : 1.5,
                "crop_boundary" : 2,
                "update_every"  : 10,
            },
        })

        cam = top_down_camera_config(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
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
