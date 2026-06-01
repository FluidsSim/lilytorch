
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
        self.use_gpu                       = True
        self.use_bdim                      = True
        self.compute_sdf                   = True
        self.convexify                     = True
        self.force_method                  = "eulerian"
        self.zero_pressure_inside          = True
        # self.force_relaxation              = 0.3
        self.body_velocity_blend_eps_cells = None
        self.bdim_mu0_projection           = False
        self.convexify                     = True
        self.bdim_body_div_correction      = True

        self.headless             = False
        self.smagorinsky_cs       = 0.

        # self.solver_method    = "python"
        self.compile_adv_diff = True

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                "sdf_name"       : "1guilla.sdf",
                "control_type"   : "position",
                "gains"          : [100.0, 1., 0],
                "spawn_mode"     : SpawnMode.TRANSVERSE,
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
        self.poisson_method          = "mgcg"
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "free"
        # Multibody swimmer: drop the mu0 factor in the Poisson coefficient so
        # the variable-density operator stays non-degenerate (dt/rho_eff).
        # The mu0-weighted form (default True) creates divergence at the
        # inter-link seams that the degenerate solve cannot remove → blow-up.


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

        # ──
        self.wall_alpha   = 0.
        self.water_alpha  = 0.05
        self.grid_spacing = 0.5*(self.ymax - self.ymin)  # lines on background floor

    # ── Extensions ────────────────────────────────────────────────────

    def extra_simulation_extensions(self, output_folder):
        extensions = []


        # Soft, reflection-free lighting for tank recordings
        extensions.append({
            "loader": "lilytorch.integration.light_modifier.LightModifier",
            "config": {
                "diffuse": [1, 1, 1],
                "ambient": [0.7, 0.7, 0.7],
            },
        })

        extensions.append({
            "loader": "lilytorch.integration.flow_iso_gl_viewer.FlowIsoGLViewer",
            "config": {
                # "field"              : "omega_z",
                "field"              : "omega_mag",
                "alpha"              : 0.2,
                "update_every"       : 1,
                "max_vertices"       : 20 * self.Nx * self.Ny,
                "smooth_sigma"       : 0,
                "crop_boundary"      : 0,
                "exclude_body"       : True,
                "iso_value"          : 3.0,
                "debug_force_visible": False,
                "color_uni"          : "#00FFFF",
                "color_pos"          : "#FF4500",
                "color_neg"          : "#00FFFF",
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

        # SkyModifier must be LAST so all CameraRecording renderers already
        # exist when initialize_episode runs the GPU texture upload.
        extensions.append({
            "loader": "lilytorch.integration.sky_modifier.SkyModifier",
            "config": {"rgb": [0.0, 0.0, 0.0]},
        })


        return extensions

    def _extra_run_patch(self):
        # Replace FARMS starry-night sky with flat black in the subprocess.
        return (
            "_m.night_sky=lambda mjcf_model:mjcf_model.asset.add("
            "'texture',name='skybox',type='skybox',"
            "builtin='flat',rgb1=[0,0,0],rgb2=[0,0,0],width=8,height=8);"
        )

if __name__ == "__main__":
    SimConfig().run()
