"""PD position-control config using ep223/ep248 model-angle trajectories.

Loads joint-angle trajectories exported from trained network models
(ep223_Cl1_fast_fish13_model_angles.xlsx / ep248_Cl2_slow_fish13_model_angles.xlsx)
and drives the zebrafish through PD position control via
``PositionController``.

Usage
-----
    python gen_configs_pd_3d_ep223_ep248.py              # ep223 fast (default)
    python gen_configs_pd_3d_ep223_ep248.py --mode ep248_slow
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root, sdfs_path
from lilytorch.examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config


_MODE_FILES = {
    "ep223_fast": "ep223_Cl1_fast_fish13_model_angles.xlsx",
    "ep248_slow": "ep248_Cl2_slow_fish13_model_angles.xlsx",
}


def _load_drags_csv(path):
    """Load per-link drag coefficients from *path*."""
    drags = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            drags.append([
                [float(row["lin_x"]),  float(row["lin_y"]),  float(row["lin_z"])],
                [float(row["quad_x"]), float(row["quad_y"]), float(row["quad_z"])],
            ])
    return drags


class SimConfig(BaseSimConfig):

    def __init__(self, mode: str = "ep223_fast"):

        if mode not in _MODE_FILES:
            raise ValueError(
                f"Unknown mode {mode!r}.  Choose from: {list(_MODE_FILES)}."
            )
        self._mode = mode

        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'examples', 'zebrafish_ki_project',
        )

        self.constant_drags = _load_drags_csv(
            os.path.join(self.data_folder, "drag_coefficients.csv")
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.compute_sdf                   = True
        self.use_gpu                       = True
        self.use_bdim                      = True
        self.headless                      = False
        self.water_buoyancy                = True
        self.sdf_interp_method             = "triquadratic"
        self.force_method                  = "lagrangian"
        self.convexify                     = True
        self.zero_pressure_inside          = False
        self.body_velocity_blend_eps_cells = 2
        self.bdim_mu0_projection           = False
        self.bdim_body_div_correction      = True

        # Wall contact (mass-independent solref for stability).
        self.bdim_physics = {
            "solref": [0.002, 1.0],
            "solimp": [0.9, 0.95, 0.001, 0.5, 2],
        }

        # ── Animats ───────────────────────────────────────────────────
        self.filter_fixed_joints = False

        self.animats_pars = [
            {
                "sdf_file"       : os.path.join(sdfs_path, "zebrafish", "zebrafish_v1_triangulated", "sdf", "zebrafish.sdf"),
                "control_type"   : "position",
                "controller_path": "lilytorch.examples.zebrafish_ki_project.pd_controller.PositionController",
                "control_pars"   : {
                    "data_folder"      : self.data_folder,
                    "file_path"        : _MODE_FILES[mode],
                    "kinematics_invert": False,
                    "lowpass_cutoff"   : 30,
                    # Set to True to save a raw-vs-filtered kinematics
                    # comparison plot before the simulation starts.
                    "plot_kinematics"  : True,
                },
                "gains"     : [0.2, 0.001, 0],
                "spawn_mode": SpawnMode.TRANSVERSE,
                "pose"      : [0, 0, 0, 0, 0, 3.141592653589793],
            },
        ]

        self.coupling = {
            "scheme": "explicit",
        }

        # ── 3-D grid ─────────────────────────────────────────────────
        self.Nx           = 512
        self.Ny           = 256          # doubled (with the y extent) to keep h isotropic
        self.Nz           = 64
        self.xmin         = -0.02
        self.xmax         = 0.08
        self.ymin         = -0.025        # doubled lateral tank: fish stays off the walls
        self.ymax         = 0.025
        self.zmin         = -0.00625
        self.zmax         = 0.00625
        self.timestep     = 0.00025
        self.n_iterations = 2001

        # ── Physics ───────────────────────────────────────────────────
        self.rho_body          = 1000.0
        self.convection_method = "abdquickest"
        self.save_every        = 50
        self.vmin              = -10.0
        self.vmax              = 10.0
        self.save              = False
        self.save_frames       = False

        self.eps_multiplier = 1.0

        # ── Arena ────────────────────────────────────────────────────
        self.wall_thickness = 0.003
        self.wall_height    = 0.01

        # ── BDIM solver ──────────────────────────────────────────────
        self.dtype                   = "float32"
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

        # ── Boundary conditions (3-D, all Neumann / zero-gradient) ──
        self.bc_type_u   = ["D", "D", "N", "N", "N", "N"]
        self.bc_values_u = [0, 0, 0, 0, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D", "N", "N"]
        self.bc_values_v = [0, 0, 0, 0, 0, 0]
        self.bc_type_w   = ["N", "N", "N", "N", "D", "D"]
        self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.interp_data_subfolder = "interp_data"

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 20.0
        self.extent       = 3.0
        self.camera_dist  = 0.02

        self.iso_3d_specs = [
            {"name": "omega_mag", "iso_value": 80.0},
            {"name": "vel_mag",   "iso_value": 2e-02},
        ]

        self.wall_alpha   = 0.
        self.water_alpha  = 0.05
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

        extensions.append({
            "loader": "lilytorch.integration.flow_iso_gl_viewer.FlowIsoGLViewer",
            "config": {
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
            overshoot=1.0,
            max_width=3840, max_height=2160,
        )
        cam["elevation"] = -30
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
    parser = argparse.ArgumentParser(
        description="PD position-control simulation with ep223/ep248 model-angle trajectories."
    )
    parser.add_argument(
        "--mode",
        default="ep223_fast",
        choices=list(_MODE_FILES),
        help="Which model-angle trajectory to replay (default: ep223_fast).",
    )
    args = parser.parse_args()
    SimConfig(mode=args.mode).single_run()
