import os
import numpy as np

from farms_core.model.options import SpawnMode

from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.farms_examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        # Reuse the existing pleurodeles mesh SDF and generate a separate
        # 3-D interpolation cache so it does not clash with the 2-D data.
        self.compute_sdf    = False

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', 'pleurodeles',
        )

        # Hardware / runtime
        self.use_bdim          = True
        self.use_gpu           = True
        self.headless          = False
        self.poisson_method    = "multigrid"
        # self.sdf_interp_method = "triquadratic"
        # self.force_delta_order = 2

        # self.apply_bdim_sigma = True

        # Animat
        self.animats_pars = [
            {
                "model_name": "pleurodeles",
                "sdf_name": "salamander_animal_fmsv0.21_2D.sdf",
                "control_type": "position",
                "gains": [0.2, 0.005, 0.0],
                "controller_config": {
                    "path": "lilytorch.farms_examples.pleurodeles.pd_controller_swim.PositionController",
                },
                "spawn_mode": SpawnMode.TRANSVERSE,
                "pose": [0.0, 0.0, 0.0, 0.0, 0.0, np.pi],
            },
        ]

        # 3-D grid
        self.Nx = 1024
        self.Ny = 192
        self.Nz = 96
        self.xmin = -0.2
        self.xmax = 0.5

        # Keep the long swimming domain in x and derive the transverse
        # extents from the same cell size so dx = dy = dz. With the FFT
        # Neumann pressure solve we can choose smaller transverse counts for
        # a tighter tank without being constrained by multigrid powers of 2.
        dx = (self.xmax - self.xmin) / self.Nx
        self.ymin = -0.5 * self.Ny * dx
        self.ymax = 0.5 * self.Ny * dx
        self.zmin = -0.5 * self.Nz * dx
        self.zmax = 0.5 * self.Nz * dx

        # Physics
        self.timestep          = 0.0005
        self.convection_method = "abdquickest"
        self.n_iterations      = 8001
        self.save_every        = 50
        self.save              = False

        # MuJoCo
        self.visual_scale = 10.0
        self.extent = 10.0

        # Arena
        self.wall_thickness = 0.003
        self.arena_pose     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.water_drag     = False
        self.water_buoyancy = False

        # BDIM solver
        self.bdim_dt                 = self.timestep
        self.bdim_nt                 = self.n_iterations
        self.zero_pressure_inside    = True
        self.rho_body                = 1000.0
        self.poisson_tol             = 1.0e-4
        self.poisson_max_cycles      = 3
        self.poisson_max_mgcg_cycles = 10
        self.poisson_precond_vcycles = 1
        self.poisson_warm_start      = True
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "neumann"
        self.compile_adv_diff        = True

        # Boundary conditions
        self.bc_type_u = ["D", "D", "N", "N", "N", "N"]
        self.bc_values_u = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.bc_type_v = ["N", "N", "D", "D", "N", "N"]
        self.bc_values_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.bc_type_w = ["N", "N", "N", "N", "D", "D"]
        self.bc_values_w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Body
        self.convexify    = False
        self.contour_mask = False
        self.interp_data_subfolder = "interp_data_3d"

        # BDIM physics
        self.bdim_physics = {"solref": [0.001, 0.5]}

        self.wall_alpha   = 0.
        self.water_alpha  = 0 #0.05
        self.grid_spacing = None #0.5*(self.ymax - self.ymin)  # lines on background floor


    def customize_joint_initials(self, joints_list):
        for joint in joints_list:
            if joint["name"] in ("joint_leg_0_L_0", "joint_leg_0_R_0"):
                joint["initial"] = [-0.3 * np.pi, -0.0]
            if joint["name"] in ("joint_leg_0_L_1", "joint_leg_0_R_1"):
                joint["initial"] = [-0.2 * np.pi, -0.0]
            if joint["name"] in ("joint_leg_1_L_0", "joint_leg_1_R_0"):
                joint["initial"] = [-0.35 * np.pi, -0.0]
            if joint["name"] in ("joint_leg_1_L_1", "joint_leg_1_R_1"):
                joint["initial"] = [-0.2 * np.pi, -0.0]

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
                "iso_value"          : 40.0,
                "debug_force_visible": False,
                "color_uni"          : "#FF4500",
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
        cam["elevation"] = -30   # tilt 20° from straight-down (−90 = top-down)
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


        # Following camera: tight view locked on fish CoM.
        # CameraRecordingFrames == CameraRecording + per-frame PNG export; the
        # PNGs are exactly the frames that make up video_follow.mp4. Swap the
        # loader back to farms_mujoco.sensors.camera.CameraRecording to skip the
        # PNGs, or set "save_video": False to keep only the frames.
        extensions.append({
            "loader": "lilytorch.integration.camera_frame_recorder.CameraRecordingFrames",
            "config": {
                "path"            : os.path.join(output_folder, "output", "video_follow.mp4"),
                "animat_id"       : 0,
                "fps"             : 30,
                "speed"           : 1.0,
                "angular_velocity": 0,
                "azimuth"         : 90,
                "elevation"       : -50,
                "distance"        : 0.4,
                "offset"          : [0, 0, 0],
                "resolution"      : [3840, 2160],
                # # PNG frames -> output/video_follow_frames/frame_%06d.png
                # "frames_dir"      : os.path.join(output_folder, "output", "video_follow_frames"),
                # "save_video"      : True,
            },
        })


        # ForceViewer – draw the fluid force on each body as a scalable arrow.
        # force_scale sets the arrow length (m per N); force_width sets the
        # circular shaft radius (m). Tune both to taste.
        extensions.append({
            "loader": "lilytorch.integration.force_viewer.ForceViewer",
            "config": {
                "force_scale" : 30,     # metres of arrow per Newton
                "force_width" : 0.001,    # shaft (circular) radius in metres
                "color"       : "#0011FF",
                # "force_source": "hydro",  # viscous+pressure (no buoyancy)
                # "max_length"  : 0.3,      # clamp arrow length (m); null = off
                # "min_force"   : 0.0,      # hide arrows below this magnitude (N)
                # "show_torque" : True,     # also draw a torque arrow per body
                # "torque_scale": 30,       # m per N·m (default: force_scale)
                # "torque_width": 0.001,    # shaft radius (default: force_width)
                # "torque_color": "#00B0FF",
                # "min_torque"  : 0.0,      # hide torque arrows below this (N·m)
                "update_every": 1,     # null -> solver.save_every cadence
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