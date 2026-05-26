
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
        self.compute_sdf    = True
        self.use_gpu        = True
        self.use_bdim       = True
        self.headless       = False
        self.water_buoyancy = True

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
        # Fish body length ~17 mm, cross-section radius ~0.65 mm.
        # BDIM accuracy requires ε/r < 0.3 where ε = 2h.
        # 512×128×64  → h≈0.195 mm, ε/r≈0.60 → ~1.6× speed deficit (current)
        # 1024×256×128 → h≈0.098 mm, ε/r≈0.30 → ~1.3× deficit (recommended)
        # 2048×512×256 → h≈0.049 mm, ε/r≈0.15 → within ~5% (accurate but costly)

        # self.Nx   = 256
        # self.Ny   = 64
        # self.Nz   = 64

        # self.xmin = -0.02
        # self.xmax =  0.03
        # self.ymin = -0.00625
        # self.ymax =  0.00625
        # self.zmin = -0.00625
        # self.zmax =  0.00625


        self.Nx   = 512
        self.Ny   = 128
        self.Nz   = 64

        self.xmin = -0.02
        self.xmax =  0.08
        self.ymin = -0.0125
        self.ymax =  0.0125
        self.zmin = -0.00625
        self.zmax =  0.00625

        # self.Nx   = 1024
        # self.Ny   = 256
        # self.Nz   = 128


        # self.xmin = -0.02
        # self.xmax =  0.18
        # self.ymin = -0.025
        # self.ymax =  0.025
        # self.zmin = -0.0125
        # self.zmax =  0.0125

        # ── Physics ───────────────────────────────────────────────────
        self.rho_body          = 1000.0
        # self.timestep          = 0.0005
        # self.convection_method = "quick"
        self.timestep          = 0.0005
        self.convection_method = "abdquickest"
        self.n_iterations      = 2001
        self.save_every        = 50
        # self.cb_sub_steps      = 2
        self.vmin              = -10.0
        self.vmax              = 10.0
        self.save              = False


        # ── Arena ────────────────────────────────────────────────────
        self.wall_thickness = 0.003
        self.wall_height    = 0.01

        # ── BDIM solver ──────────────────────────────────────────────
        self.dtype                   = "float32"
        self.zero_pressure_inside    = True
        self.bdim_dt                 = self.timestep
        self.bdim_nt                 = self.n_iterations + 1
        self.poisson_tol             = 1.0e-4
        self.poisson_max_cycles      = 30
        self.poisson_max_mgcg_cycles = 10
        self.poisson_precond_vcycles = 1
        self.poisson_warm_start      = False
        self.poisson_method          = "multigrid"
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "neumann"
        self.compile_adv_diff        = True
        # order-2 delta (Towers 2008) corrects |∇SDF|≠1 at mesh joints/corners;
        # recommended for mesh bodies, no-op for analytical SDFs.
        self.force_delta_order       = 2

        # ── Boundary conditions (3-D, all Neumann / zero-gradient) ──
        self.bc_type_u   = ["N", "N", "N", "N", "N", "N"]
        self.bc_values_u = [0, 0, 0, 0, 0, 0]
        self.bc_type_v   = ["N", "N", "N", "N", "N", "N"]
        self.bc_values_v = [0, 0, 0, 0, 0, 0]
        self.bc_type_w   = ["N", "N", "N", "N", "N", "N"]
        self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.convexify             = False
        self.interp_data_subfolder = "interp_data"

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 20.0
        self.extent       = 3.0
        self.camera_dist  = 0.1

        self.iso_3d_specs = [
            {"name": "omega_mag", "iso_value": 80.0},
            {"name": "vel_mag",   "iso_value": 2e-02},
        ]

    # ── Extensions ────────────────────────────────────────────────────

    def extra_simulation_extensions(self, output_folder):
        extensions = []

        # Soft, reflection-free lighting for tank recordings
        extensions.append({
            "loader": "lilytorch.integration.light_modifier.LightModifier",
            "config": {},
        })

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

        # extensions.append({
        #     "loader": "lilytorch.integration.particle_viewer.ParticleViewer",
        #     "config": {
        #         "max_particles"   : 800000,
        #         "seed_n_particles": 3,
        #         "seed_interval"   : 1,
        #         "turb_diffusivity": 0.,
        #         "sphere_size"     : 0.0001,
        #         "particle_color"  : [255/256, 0.0, 166/256, 0.85],   #FF00A6
        #         "trail_length"    : 0,
        #         "update_every"    : None,
        #         "n_z_layers"      : 3,
        #         "floor_color"     : "#FFFFFF",                       # dark blue floor
        #         "save_particles"   : True,
        #         "save_dir"         : os.path.join(output_folder, "particles"),
        #         "save_every"       : self.save_every,
        #     }
        # })

        extensions.append({
            "loader": "lilytorch.integration.flow_iso_gl_viewer.FlowIsoGLViewer",
            "config": {
                "field"              : "omega_z",
                "alpha"              : 0.2,
                "update_every"       : 1,
                "max_vertices"       : 300000,
                "smooth_sigma"       : 0, # heavier smoothing → fewer noisy lobes
                "crop_boundary"      : 0, # drop 8 cells per face → kills wall noise
                "exclude_body"       : True,
                "iso_value"          : 30.0,
                "debug_force_visible": False,
                "record_viewer"      : True,
                "record_path"        : os.path.join(output_folder, "output", "viewer.mp4"),
                "record_fps"         : 30.0,
            },
        })

        # extensions.append({
        #     "loader": "lilytorch.integration.flow_iso_viewer.FlowIsoViewer",
        #     "config": {
        #         "field": "omega_z",
        #         "max_tiles": 8000,
        #         "iso_fraction": 0.15,
        #         "smooth_sigma": 0,
        #         "crop_boundary": 3,
        #         "tile_size": 0.0005,
        #         "tile_thickness": 0.0001,
        #         "tile_alpha": 0.7,
        #     },
        # },
        # )

        # Top-down camera auto-fitted to the domain
        cam = top_down_camera_config(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.zmin, self.zmax,
            overshoot=1.10,  # controls CameraRecording distance (not camera_dist)
        )
        extensions.append({
            "loader": "farms_mujoco.sensors.camera.CameraRecording",
            "config": {
                "path"            : os.path.join(output_folder, "output", "video.mp4"),
                "animat_id"       : None,
                "fps"             : 30,
                "speed"           : 0.1,
                "angular_velocity": 0,
                **cam,
            },
        })

        # Side view: looking along Y axis (shows swimming direction vs tank depth)
        side = side_camera_config(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.zmin, self.zmax,
            view_axis="y",
            overshoot=1.10,
        )
        extensions.append({
            "loader": "farms_mujoco.sensors.camera.CameraRecording",
            "config": {
                "path"            : os.path.join(output_folder, "output", "video_side.mp4"),
                "animat_id"       : None,
                "fps"             : 30,
                "speed"           : 0.1,
                "angular_velocity": 0,
                **side,
            },
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



