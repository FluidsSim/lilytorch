"""Full-3D free-swimming pleurodeles configuration.

Same solver / grid / controller setup as ``gen_configs_swim_3d.py`` but the
animat is spawned in ``SpawnMode.FREE`` (full 6-DOF base joint: surge, sway,
heave, roll, pitch, yaw) instead of ``TRANSVERSE`` (horizontal-plane
constraint).  The body is therefore free to dive, roll and pitch, propelled
purely by the fluid forces from the actuated swim gait.

Intended for a NEW full-3D pleurodeles mesh SDF that is still being prepared.
Drop the file into ``farms_examples/sdfs/pleurodeles/`` and set
``FULL3D_SDF_NAME`` below to its filename.
"""

import os
import numpy as np

from farms_core.model.options import SpawnMode

from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.farms_examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config


# ── New full-3D mesh SDF (not available yet) ──────────────────────────────
# TODO: replace with the actual filename once the full-3D SDF is ready.
# It must live under ``farms_examples/sdfs/pleurodeles/``.
FULL3D_SDF_NAME = "pleurosim_v0.3.sdf"


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        # Build a fresh 3-D interpolation cache for the new mesh, kept in its
        # own subfolder so it never clashes with the existing 3-D cache built
        # for the 2-D-named SDF.
        self.compute_sdf    = True

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', 'pleurodeles',
        )

        # Hardware / runtime
        self.use_bdim          = True
        self.use_gpu           = True
        self.headless          = False
        self.poisson_method    = "fft"
        self.sdf_interp_method = "triquadratic"
        self.force_delta_order = 2

        # self.apply_bdim_sigma = True

        # Animat — FREE spawn: full 6-DOF base, free to dive/roll/pitch.
        self.animats_pars = [
            {
                "model_name": "pleurodeles",
                "sdf_name": FULL3D_SDF_NAME,
                "control_type": "position",
                "gains": [0.2, 0.005, 0.0],
                "controller_config": {
                    "path": "lilytorch.farms_examples.pleurodeles.pd_controller_swim.PositionController",
                },
                "spawn_mode": SpawnMode.FREE,
                # Spawn submerged at mid-depth (z=0 is the tank centre; the
                # water surface is at zmax), facing -x (yaw = pi).
                "pose": [0.0, 0.0, 0.0, 0.0, 0.0, np.pi],
            },
        ]

        # 3-D grid
        self.Nx = 1024
        self.Ny = 192
        self.Nz = 192
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
        self.n_iterations      = 4001
        self.save_every        = 50
        self.save              = False
        self.save_drags        = True

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
        self.zero_pressure_inside    = False
        self.rho_body                = 1000.0   # neutrally buoyant swimmer
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
        self.interp_data_subfolder = "interp_data_full3d"

        # BDIM physics
        self.bdim_physics = {"solref": [0.001, 0.5]}

        self.wall_alpha   = 0.
        self.water_alpha  = 0.05
        self.grid_spacing = 0.5*(self.ymax - self.ymin)  # lines on background floor


    def customize_joint_initials(self, joints_list):
        for joint in joints_list:
            if joint["name"] in ("joint_leg_0_L_0_z", "joint_leg_0_R_0_z"):
                joint["initial"] = [-0.3 * np.pi, -0.0]
            if joint["name"] in ("joint_leg_1_L_0_z", "joint_leg_1_R_0_z"):
                joint["initial"] = [-0.35 * np.pi, -0.0]

            if joint["name"] in ("joint_leg_1_L_0_y", "joint_leg_1_R_0_y",):
                joint["initial"] = [0.5*np.pi, -0.0]
            if joint["name"] in ("joint_leg_1_L_1_y", "joint_leg_1_R_1_y"):
                joint["initial"] = [0.5 * np.pi, -0.0]
            if joint["name"] in ("joint_leg_0_L_0_y", "joint_leg_0_R_0_y",):
                joint["initial"] = [0.5*np.pi, -0.0]
            if joint["name"] in ("joint_leg_0_L_1_y", "joint_leg_0_R_1_y"):
                joint["initial"] = [0.5 * np.pi, -0.0]

            if joint["name"] in ("joint_leg_0_L_1_x", "joint_leg_0_R_1_x"):
                joint["initial"] = [-0.2 * np.pi, -0.0]
            if joint["name"] in ("joint_leg_1_L_1_x", "joint_leg_1_R_1_x"):
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


        # Following camera: tight view locked on fish CoM
        extensions.append({
            "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
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
