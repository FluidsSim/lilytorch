
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

        # ── FARMS settings ───────────────────────────────────────────────
        self.headless             = False

        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                "sdf_name"       : "1guilla.sdf",
                "control_type"   : "position",
                "gains"          : [100.0, 1., 0],
                "spawn_mode"     : SpawnMode.TRANSVERSE,
                "pose"           : [4.75, 0.1, 0.15, 0, 0, 0.05],
                "controller_path": "lilytorch.farms_examples._1guillasim.project_feedback_control.pd_controller.PositionController",
                "control_pars"   : {'freq': 0.5, 'twl': 0.571429*14, 'amp': 35.0},
            },
        ]

        # ── 2-D grid ─────────────────────────────────────────────────
        self.Nx   = 900
        self.Ny   = 300
        # self.Nz   = 52
        self.xmin = 0
        self.xmax = 6
        self.ymin = -1
        self.ymax = 1
        self.zmin = -(2/300*52/2)
        self.zmax = (2/300*52/2)
        self.wall_thickness = 0.3
        self.wall_height    = 0.3


        # ── Physics ───────────────────────────────────────────────────
        self.timestep          = 0.001
        self.convection_method = "abdquickest"
        self.n_iterations      = 20001
        self.save_every        = 200
        # self.dtype             = "float64"

        # ── BDIM solver ──────────────────────────────────────────────
        # self.solver_method                 = "python"
        self.use_gpu                       = True
        self.use_bdim                      = True
        self.compute_sdf                   = True
        self.convexify                     = True
        self.force_method                  = None   # Eulerian: stabler than Lagrangian in 2D
        self.zero_pressure_inside          = False
        self.body_velocity_blend_eps_cells = 3
        self.bdim_mu0_projection           = False  # plain dt/rho; mu0-weighted degenerates at inter-link seams
        self.bdim_body_div_correction      = True   # needed: removes seam-velocity divergence from Poisson RHS (convexify=True)
        self.poisson_method                = "multigrid"
        self.compile_adv_diff              = True

        # self.force_scaling         = 0.04
        # self.force_relaxation              = 0.3

        self.coupling = {
            "scheme": "implicit",
            "accelerator": "iqn-ils",  # IQN-ILS: quadratic convergence near fixed point, handles ρ_body=ρ_fluid
            "reuse": 2,
            "tol": 1e-4,
            "max_iter": 100,           # 30 was too few at peak swimming speed; 100 gives Aitken/IQN room
        }


        self.bdim_dt                 = self.timestep
        self.bdim_nt                 = self.n_iterations + 1
        self.poisson_tol             = 1.0e-4
        self.poisson_max_cycles      = 30
        self.poisson_max_mgcg_cycles = 10
        self.poisson_precond_vcycles = 1
        self.poisson_warm_start      = True
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "neumann"

        # ── Boundary conditions (2-D, Dirichlet inlet) ───────────────
        self.bc_type_u   = ["D", "D", "N", "N"]
        self.bc_values_u = [0, 0, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0]

        # ── Data+Viz ───────────────────────────────────────────────
        self.interp_data_subfolder = "interp_data_2d"
        self.visual_scale      = 10.0
        self.extent            = 100.0
        self.sky_color         = [0.02, 0.05, 0.15]
        self.floor_color       = "#E0D4D4"
        self.viewer_body_color = "#D09F23"
        self.wall_alpha        = 1.
        self.water_alpha       = 0.05
        self.grid_spacing      = 0.5*(self.ymax - self.ymin)  # lines on background floor
        self.vmin              = -10.0
        self.vmax              = 10.0
        self.save              = False

    # ── Extensions ────────────────────────────────────────────────────

    def extra_simulation_extensions(self, output_folder):
        extensions = []


        # Soft, reflection-free lighting for tank recordings
        extensions.append({
            "loader": "lilytorch.integration.light_modifier.LightModifier",
            "config": {
                "diffuse": [1, 1, 1],
                "ambient": [0.3, 0.3, 0.3],
            },
        })


        # Top-down camera auto-fitted to the pool
        cam = top_down_camera_config(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.zmin, self.zmax,
            overshoot=1,
            max_width=3840, max_height=2160,
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
