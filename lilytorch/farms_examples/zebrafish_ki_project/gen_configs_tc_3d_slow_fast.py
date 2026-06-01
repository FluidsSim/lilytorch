
import csv
import os
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root, sdfs_path
from lilytorch.farms_examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config, side_camera_config


def _load_drags_csv(path):
    """Load per-link drag coefficients from *path*.

    Returns a list of ``[[lin_x, lin_y, lin_z], [quad_x, quad_y, quad_z]]``
    entries, one per link, ordered by the row order in the CSV.
    """
    drags = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            drags.append([
                [float(row["lin_x"]),  float(row["lin_y"]),  float(row["lin_z"])],
                [float(row["quad_x"]), float(row["quad_y"]), float(row["quad_z"])],
            ])
    return drags

class SimConfig(BaseSimConfig):

    def __init__(self):

        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', 'zebrafish_ki_project',
        )

        self.constant_drags = _load_drags_csv(
            os.path.join(self.data_folder, "drag_coefficients.csv")
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.compute_sdf         = True
        self.use_gpu             = True
        self.use_bdim            = True
        self.headless            = False
        self.water_buoyancy      = True
        # self.force_delta_order   = 2
        self.sdf_interp_method   = "triquadratic"
        # self.solver_method       = "python"
        self.force_method        = "lagrangian"

        self.bdim_physics = {
            "solref": [-2e4, -30e1],
            "solimp": [0., 0.95, 0.001, 0.5, 2],
        }

        # ── Animats ───────────────────────────────────────────────────
        self.filter_fixed_joints = False

        self.animats_pars = [

            {
                "sdf_file"       : os.path.join(sdfs_path, "zebrafish", "zebrafish_v1_triangulated", "sdf", "zebrafish.sdf"),
                "control_type" : "torque",
                "muscle_loader": "farms_ekeberg.src.ekeberg.EkebergMuscleController",
                "muscle_config": {
                    'load_controller': 'lilytorch.farms_examples.zebrafish_ki_project.network.WaveController',
                    'method'         : 'implicit',
                    'muscle_pars'    : os.path.join(self.data_folder, 'muscle_params.csv'),
                    'mode'           : 'slow',
                },
                "gains"     : [0, 0, 0],
                "spawn_mode": SpawnMode.TRANSVERSE,
                "pose"      : [0, 0, 0, 0, 0, 3.141592653589793],
            },

            # {
            #     "sdf_file"       : os.path.join(sdfs_path, "zebrafish", "zebrafish_v1_triangulated", "sdf", "zebrafish_old.sdf"),
            #     "control_type"   : "position",
            #     "gains"          : [0.001, .00002, 0],
            #     "spawn_mode"     : SpawnMode.FREE,
            #     "pose"           : [0, 0, 0.0, 0, 0, 3.141592653589793],
            #     "controller_path": "lilytorch.farms_examples.zebrafishsim.pd_controller.PositionController",
            #     "control_pars"   : {
            #         'freq': 5.0, 'twl': 20, 'amp': 120,
            #         'bout_duration': None, 'glide_duration': 1,
            #         'bout_ramp': 0.2,
            #     },
            # },
        ]

        # ── 3-D grid ─────────────────────────────────────────────────
        self.Nx           = 512
        self.Ny           = 128
        self.Nz           = 64
        self.xmin         = -0.02
        self.xmax         = 0.08
        self.ymin         = -0.0125
        self.ymax         = 0.0125
        self.zmin         = -0.00625
        self.zmax         = 0.00625
        self.timestep     = 0.0005
        self.n_iterations = 4001

        # self.Nx           = 1024
        # self.Ny           = 256
        # self.Nz           = 128
        # self.xmin         = -0.02
        # self.xmax         = 0.08
        # self.ymin         = -0.0125
        # self.ymax         = 0.0125
        # self.zmin         = -0.00625
        # self.zmax         = 0.00625
        # self.timestep     = 0.00025
        # self.n_iterations = 8001

        # ── Physics ───────────────────────────────────────────────────
        self.rho_body          = 1000.0
        self.convection_method = "abdquickest"
        self.save_every        = 50
        self.vmin              = -10.0
        self.vmax              = 10.0
        self.save              = False

        self.eps_multiplier = 2.0
        # BDIM-σ correction for thin zebrafish body links (r < eps).
        # self.apply_bdim_sigma = True


        # ── Arena ────────────────────────────────────────────────────
        self.wall_thickness = 0.003
        self.wall_height    = 0.01

        # ── BDIM solver ──────────────────────────────────────────────
        self.dtype                   = "float32"
        self.zero_pressure_inside    = True
        self.bdim_dt                 = self.timestep
        self.bdim_nt                 = self.n_iterations + 1
        self.poisson_tol             = 1.0e-7
        self.poisson_max_cycles      = 30
        self.poisson_max_mgcg_cycles = 10
        self.poisson_precond_vcycles = 1
        self.poisson_warm_start      = False
        self.poisson_method          = "multigrid"
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "neumann"
        self.compile_adv_diff        = True
        # self.force_delta_order       = 2
        # self.sdf_interp_method       = "triquadratic"

        # ── Boundary conditions (3-D, all Neumann / zero-gradient) ──
        self.bc_type_u   = ["D", "D", "N", "N", "N", "N"]
        self.bc_values_u = [0, 0, 0, 0, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D", "N", "N"]
        self.bc_values_v = [0, 0, 0, 0, 0, 0]
        self.bc_type_w   = ["N", "N", "N", "N", "D", "D"]
        self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.convexify             = False
        self.interp_data_subfolder = "interp_data"

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 20.0
        self.extent       = 3.0
        self.camera_dist  = 0.02

        self.iso_3d_specs = [
            {"name": "omega_mag", "iso_value": 80.0},
            {"name": "vel_mag",   "iso_value": 2e-02},
        ]

        self.wall_alpha = 0.
        self.water_alpha = 0.05
        self.grid_spacing = 0.0125  # lines on background floor


    # ── Extensions ────────────────────────────────────────────────────

    def extra_simulation_extensions(self, output_folder):
        extensions = []

        # Soft, reflection-free lighting for tank recordings
        extensions.append({
            "loader": "lilytorch.integration.light_modifier.LightModifier",
            "config": {
                "diffuse": [0.70, 0.70, 0.70],
                "ambient": [0.65, 0.65, 0.65],
            },
        })

        # Interactive viewer: keep the camera locked on the fish CoM
        extensions.append({
            "loader": "farms_mujoco.simulation.extensions.CameraFollower",
            "config": {
                "animat_id"       : 0,
                "azimuth"         : 90,
                "elevation"       : -30,
                "distance"        : 0.04,
                "angular_velocity": 0,
            },
        })

        extensions.append({
            "loader": "lilytorch.integration.flow_iso_gl_viewer.FlowIsoGLViewer",
            "config": {
                # "field"              : "omega_z",
                "field"              : "omega_mag",
                "alpha"              : 0.2,
                "update_every"       : 1,
                "max_vertices"       : 4 * self.Nx * self.Ny,
                "smooth_sigma"       : 0,
                "crop_boundary"      : 0,
                "exclude_body"       : True,
                "iso_value"          : 100.0,
                "debug_force_visible": False,
                "color_uni"          : "#00FFFF",
                "color_pos"          : "#FF4500",
                "color_neg"          : "#00FFFF",
            },
        })

        # Top-down camera auto-fitted to the domain
        cam = top_down_camera_config(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.zmin, self.zmax,
            overshoot=1.0,  # controls CameraRecording distance (not camera_dist)
            max_width=3840, max_height=2160,
        )
        cam["elevation"] = -30   # tilt 20° from straight-down (−90 = top-down)
        extensions.append({
            "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
            "config": {
                "path"            : os.path.join(output_folder, "output", "video.mp4"),
                "animat_id"       : None,
                "fps"             : 30,
                "speed"           : 0.1,
                "angular_velocity": 0,
                **cam,
            },
        })

        # # Side view: looking along Y axis (shows swimming direction vs tank depth)
        # side = side_camera_config(
        #     self.xmin, self.xmax,
        #     self.ymin, self.ymax,
        #     self.zmin, self.zmax,
        #     view_axis="y",
        #     overshoot=1.0,
        # )
        # extensions.append({
        #     "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
        #     "config": {
        #         "path"            : os.path.join(output_folder, "output", "video_side.mp4"),
        #         "animat_id"       : None,
        #         "fps"             : 30,
        #         "speed"           : 0.1,
        #         "angular_velocity": 0,
        #         **side,
        #     },
        # })

        # Following camera: tight view locked on fish CoM
        extensions.append({
            "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
            "config": {
                "path"            : os.path.join(output_folder, "output", "video_follow.mp4"),
                "animat_id"       : 0,
                "fps"             : 30,
                "speed"           : 0.1,
                "angular_velocity": 0,
                "azimuth"         : 90,
                "elevation"       : -30,
                "distance"        : 0.04,
                "offset"          : [0, 0, 0],
                "resolution"      : [3840, 2160],
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
    SimConfig().single_run()



