"""Extended pool with ramp animat for amphibious walking→swimming simulations.

Builds a substantially extended rectangular pool (12 m × 3 m × 0.6 m)
and generates a separate **ramp animat** (static box, fixed to world) that
slopes from the x-min wall down to the pool floor.  Because the ramp is an
animat, it participates in the BDIM fluid–rigid coupling — fluid forces
are computed on it and it affects the flow field.

Usage
-----
    python gen_config_amphibious.py              # generate + run
    python gen_config_amphibious.py --no-run     # generate configs only

The generated configs are written under the standard save path
(:data:`lilytorch.util.paths.save_path`).
"""

from __future__ import annotations

import argparse
import os
from math import atan2, sqrt, sin, cos

from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root, sdfs_path
from lilytorch.examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config, side_camera_config
from lilytorch.examples.amphibious_pool.gen_ramp_sdf import create_ramp_sdf


# ═══════════════════════════════════════════════════════════════════════
# Pool geometry
# ═══════════════════════════════════════════════════════════════════════

POOL_XMIN, POOL_XMAX = 0.0, 12.0     # 12 m long
POOL_YMIN, POOL_YMAX = -1.5, 1.5      #  3 m wide
POOL_ZMIN, POOL_ZMAX = -0.3, 0.3      #  0.6 m deep

WATERLINE = 0.2  # water fills bottom half (z < 0), air above — two-phase surface

# ── Ramp (animat) ──────────────────────────────────────────────────────
RAMP_X_START = POOL_XMIN              # against the left wall
RAMP_Z_START = POOL_ZMAX              # flush with pool wall top
RAMP_X_END   = 4.0                    # 4 m into the pool
RAMP_Z_END   = POOL_ZMIN              # bottom at pool floor

# Derived ramp pose
_ramp_dx = RAMP_X_END - RAMP_X_START
_ramp_dz = RAMP_Z_END - RAMP_Z_START
RAMP_LENGTH = sqrt(_ramp_dx * _ramp_dx + _ramp_dz * _ramp_dz)
RAMP_PITCH  = atan2(RAMP_Z_START - RAMP_Z_END, _ramp_dx)  # positive → slopes down

# Passed explicitly to create_ramp_sdf() below, so it need not match that
# function's default; it must stay >= 2-3 cells thick to resolve the SDF band.
_RAMP_THICKNESS = 0.2
_half_t = _RAMP_THICKNESS / 2.0

# The ramp centre-line passes through (RAMP_X_START, RAMP_Z_START) and
# (RAMP_X_END, RAMP_Z_END).  Shift the body origin by -half_t along the
# local +Z (surface normal) so the *top* face of the box sits exactly on
# that line, flush with the pool walls.
RAMP_CX = (RAMP_X_START + RAMP_X_END) / 2 - _half_t * sin(RAMP_PITCH)
RAMP_CZ = (RAMP_Z_START + RAMP_Z_END) / 2 - _half_t * cos(RAMP_PITCH)
RAMP_POSE = [RAMP_CX, 0.0, RAMP_CZ, 0, RAMP_PITCH, 0]


class SimConfig(BaseSimConfig):
    """Amphibious pool — extended tank + ramp animat."""

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'examples', '_1guillasim',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.use_gpu                       = True
        self.use_bdim                      = True    # BDIM for fluid coupling
        self.compute_sdf                   = True
        # Must stay False: convexify=True blows the sim up by ~iter 60 (measured:
        # max|p| 7.6e3 -> 6.8e4, BADQACC) because the convex hulls of adjacent
        # 1guilla links OVERLAP — the known hull-overlap coupling instability.
        # The cost is that the raw link*_collision.stl are non-watertight, so
        # open3d's ray-parity sign test speckles them (link8 fragments into a
        # 2624-voxel core plus ~15 satellites that survive body.py's absolute
        # >=8-voxel island filter).  That fragmentation is a MESH/SDF-cleanup bug,
        # not a config choice — fix it in the SDF tabulation, not by convexifying.
        self.convexify                     = True
        self.force_method                  = "eulerian"
        self.force_submethod               = "deltaH"
        self.zero_pressure_inside          = False
        self.body_velocity_blend_eps_cells = None
        self.bdim_mu0_projection           = False   # μ₀ from the CC union SDF
        self.bdim_body_div_correction      = False   # off: avoids RHS inconsistency near ramp

        # self.ground_height = POOL_ZMAX

        self.headless             = False
        self.smagorinsky_cs       = 0.0

        # ── Water ─────────────────────────────────────────────────────
        self.water_height   = WATERLINE
        self.water_drag     = False   # BDIM handles fluid forces
        self.water_buoyancy = False

        # ── Animats ───────────────────────────────────────────────────
        # 1. Ramp — static body, fixed to world, part of fluid coupling
        # 2. 1guilla — swimming fish, position-controlled
        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                # Ventral-ballast SDF: identical mass/inertia/geometry to
                # 1guilla_800.sdf (still uniform rho=800, same weight + BDIM
                # buoyancy) but each link's COM lowered 3 cm, giving a passive
                # righting moment that keeps the planar gait from rolling the
                # body over (gait roll 24.5deg -> ~5deg). Regenerate/tune with
                # sdfs/1guilla/make_ballast_sdf.py.
                "sdf_name"       : "1guilla_600.sdf",
                "control_type"   : "position",
                "gains"          : [100.0, 1., 0],
                "spawn_mode"     : SpawnMode.FREE,
                # spawn at the true floating equilibrium (centreline ~1.1 cm
                # below the waterline) to avoid the initial heave-overshoot.
                "pose"           : [4.75, 0.1, -0.0, 0, 0, 0.05],
                "controller_path": "lilytorch.examples._1guillasim.experiments.controller.PositionController",
                "control_pars"   : {
                    "file_path": os.path.join(
                        # self.data_folder, "/data/andreaferrario/1guilla_experiments/swim/log/ms004mpt003log.csv"
                        self.data_folder, "/data/andreaferrario/1guilla_experiments/swim/log/ms001mpt001log.csv"
                    ),
                },
            },

            # # -- 1guilla animat --
            # {
            #     "model_name"     : "1guilla",
            #     "sdf_name"       : "1guilla.sdf",
            #     "control_type"   : "position",
            #     "gains"          : [100.0, 1.0, 0],
            #     "spawn_mode"     : SpawnMode.FREE,
            #     # Spawn in open water at the far (deep) end — the ramp occupies
            #     # x in [0, 4], so the fish starts well clear of it.
            #     "pose"           : [11, 0.0, 0.1, 0, 0, 0.0],
            #     "controller_path": "lilytorch.examples._1guillasim.experiments.controller.PositionController",
            #     "control_pars"   : {
            #         "file_path": os.path.join(
            #             self.data_folder,
            #             "/data/andreaferrario/1guilla_experiments/swim/log/ms004mpt003log.csv",
            #         ),
            #     },
            # },
            # -- Ramp animat --
            {
                "sdf_file"       : os.path.join(sdfs_path, "ramp", "sdf", "ramp.sdf"),
                "control_type"   : "position",
                "gains"          : [0, 0, 0],    # no joints → unused
                "spawn_mode"     : SpawnMode.FIXED,
                "pose"           : RAMP_POSE,
            },
        ]

        # ── 3-D grid ─────────────────────────────────────────────────
        self.Nx   = 1200
        self.Ny   = 300
        self.Nz   = 60
        self.xmin = POOL_XMIN
        self.xmax = POOL_XMAX
        self.ymin = POOL_YMIN
        self.ymax = POOL_YMAX
        self.zmin = POOL_ZMIN
        self.zmax = POOL_ZMAX

        # ── Physics ───────────────────────────────────────────────────
        self.rho_body          = 1000.0
        self.rho               = 1000.0
        self.nu                = 1.0e-6
        self.timestep          = 0.001
        self.convection_method = "quick"
        self.n_iterations      = 20001
        self.save_every        = 100
        self.vmin              = -10.0
        self.vmax              = 10.0
        self.save              = False

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale  = 10.0
        self.extent        = 100.0

        # ── BDIM solver ──────────────────────────────────────────────
        # The geometric V-cycle contracts only ~0.99/cycle on this operator
        # (measured), so the pressure stays under-converged at any practical
        # cycle count.  That is tolerable ONLY because rho_air is capped (see
        # the two-phase block): the air's Poisson coefficient is dt/rho_air, so
        # the air turns leftover pressure error into velocity with a gain of
        # rho_water/rho_air.  Do not raise the cycle counts hoping to fix a
        # blow-up here — measured: it does not help.
        self.bdim_dt                 = self.timestep
        self.bdim_nt                 = self.n_iterations + 1
        self.poisson_method          = "multigrid"
        self.poisson_tol             = 1.0e-5
        self.poisson_max_cycles      = 30
        self.poisson_max_mgcg_cycles = 10
        self.poisson_precond_vcycles = 1
        self.poisson_warm_start      = True
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "neumann"
        self.empty_cache_every       = 10**9

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

        # ── Arena ────────────────────────────────────────────────────
        self.wall_thickness = None   # auto
        self.wall_alpha     = 1.0
        self.water_alpha    = 0.05
        self.grid_spacing   = 0.5 * (self.ymax - self.ymin)
        self.floor_color    = "#E0D4D4"
        self.sky_color      = [0.65, 0.65, 0.65]  # light neutral gray

    # ── Arena: standard pool (no ramp in pool SDF) ───────────────────

    def gen_arena_config(self, output_folder, index=0):
        from lilytorch.integration.gen_pool_sdf import (
            create_pool_sdf, create_water_sdf,
        )
        from farms_core.io.yaml import pyobject2yaml
        from farms_core.model.options import SpawnMode

        wt = 5

        create_pool_sdf(
            self.xmin, self.xmax, self.ymin, self.ymax,
            zmin=self.zmin, zmax=self.zmax,
            wall_thickness=wt, plotting=False,
            wall_alpha=self.wall_alpha,
            grid_spacing=self.grid_spacing,
            floor_color=self.floor_color,
            lip=0,            # walls flush with water surface (pool is fully filled)
            include_floor=False,  # no floor — open-bottom pool
        )

        water_sdf = create_water_sdf(
            self.xmin, self.xmax, self.ymin, self.ymax,
            zmin=self.zmin, zmax=WATERLINE,
            water_height=self.water_height,
            water_alpha=self.water_alpha,
        )

        arena_sdf = os.path.join(sdfs_path, "pool", "sdf", "pool.sdf")

        arena_dict = {
            "sdf": arena_sdf,
            "spawn": {
                "loader"  : 0,
                "mode"    : SpawnMode.FREE,
                "pose"    : list(self.arena_pose),
                "velocity": [0, 0, 0, 0, 0, 0],
                "extras"  : {},
            },
            "water": {
                "sdf"      : water_sdf,
                "drag"     : self._water_drag,
                "buoyancy" : self._water_buoyancy,
                "height"   : WATERLINE,
                "velocity" : [0, 0, 0],
                "viscosity": 1.0,
                "density"  : self.rho,
                "maps"     : ["", ""],
            },
            "ground_height": self.ground_height,
        }
        pyobject2yaml(
            os.path.join(output_folder, 'arena_config.yaml'), arena_dict,
        )

    # ── Two-phase BDIM extension ──────────────────────────────────────
    def _bdim_extension(self, output_folder):
        bdim_ext = super()._bdim_extension(output_folder)
        solver = bdim_ext["config"]["bdim_yaml"]["solver"]

        # The pre-Poisson CUDA graph re-captures whenever the step's key changes,
        # and the key contains the data_ptrs of the per-step scratch staggered
        # SDF/body fields (freed + reallocated every step, see
        # solver._fluid_step_fused_3d).  An allocator reshuffle therefore mints a
        # new key mid-run and the re-capture dies with "operation failed due to a
        # previous error during capture" (~iter 200 here).  Run eager until that
        # is fixed; costs speed, not correctness.

        solver["gravity"] = [0, 0, -9.81]
        solver["two_phase"] = {
            "alpha_init"             : f"lambda X, Y, Z: (Z < {WATERLINE}).double()",
            "rho_water"              : 1000.0,
            # Capped air density (80:1, NOT the physical 1.2 → 833:1).  The
            # projection coefficient is dt/rho, so the air converts any
            # leftover Poisson error into velocity with gain rho_water/rho_air.
            # The V-cycle leaves the pressure well short of converged on this
            # grid, and at 833:1 that residual drives the air to metres/second
            # within a step: the air ran away linearly from step 1 and NaN'd at
            # iteration 15 (mjWARN_BADQACC).  At 80:1 the same pressure error
            # is 66x less amplified and the air velocity saturates (~1.2 m/s).
            # Raise this back to 1.2 only with a Poisson that actually converges.
            "rho_air"                : 1.5,
            "nu_water"               : self.nu,
            "nu_air"                 : 1.5e-5,
            "alpha_exclude_body"     : True,
            "alpha_volume_compensate": True,
            "air_transparent_body"   : False,
        }

        return bdim_ext

    # ── Extensions ────────────────────────────────────────────────────

    def extra_simulation_extensions(self, output_folder):
        extensions = []

        # reduce overall scene light: lower diffuse and ambient
        extensions.append({
            "loader": "lilytorch.integration.light_modifier.LightModifier",
            "config": {
                # dim the main light
                "diffuse": [0.6, 0.6, 0.6],
                # reduce ambient/global illumination
                "ambient": [0.1, 0.1, 0.1],
            },
        })

        # # Air/water interface visualisation
        # extensions.append({
        #     "loader": "lilytorch.integration.flow_iso_gl_viewer.FlowIsoGLViewer",
        #     "config": {
        #         "update_every"      : 1,
        #         "max_vertices"      : 20 * self.Nx * self.Ny,
        #         "crop_boundary"     : 0,
        #         "debug_force_visible": False,
        #         "fields": [
        #             {
        #                 "field"     : "interface",
        #                 "iso_value" : 0.5,
        #                 "alpha"     : 0.45,
        #                 "color"     : "#3399FF",
        #                 "smooth_sigma": 0,
        #                 "exclude_body": False,
        #                 "reflective": True,
        #             },
        #         ],
        #     },
        # })

        # # Top-down camera auto-fitted to the pool
        # cam = top_down_camera_config(
        #     self.xmin, self.xmax,
        #     self.ymin, self.ymax,
        #     self.zmin, self.zmax,
        #     overshoot=1,
        #     max_width=3840, max_height=2160,
        # )
        # extensions.append({
        #     "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
        #     "config": {
        #         "path"            : os.path.join(output_folder, "output", "video.mp4"),
        #         "animat_id"       : None,
        #         "fps"             : 30,
        #         "speed"           : 1.0,
        #         "angular_velocity": 0,
        #         **cam,
        #     },
        # })

        # # Side camera auto-fitted to the pool
        # cam_side = side_camera_config(
        #     self.xmin, self.xmax,
        #     self.ymin, self.ymax,
        #     self.zmin, self.zmax,
        #     overshoot=1,
        #     max_width=3840, max_height=2160,
        # )
        # extensions.append({
        #     "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
        #     "config": {
        #         "path"            : os.path.join(output_folder, "output", "video_side.mp4"),
        #         "animat_id"       : None,
        #         "fps"             : 30,
        #         "speed"           : 1.0,
        #         "angular_velocity": 0,
        #         **cam_side,
        #     },
        # })

        return extensions

    def _extra_run_patch(self):
        r, g, b = self.sky_color
        return (
            f"_m.night_sky=lambda mjcf_model:mjcf_model.asset.add("
            f"'texture',name='skybox',type='skybox',"
            f"builtin='flat',rgb1=[{r},{g},{b}],rgb2=[{r},{g},{b}],width=8,height=8);"
        )

    def run(self):
        # Generate ramp SDF before building configs
        create_ramp_sdf(
            length=RAMP_LENGTH,
            width=POOL_YMAX - POOL_YMIN,
            thickness=_RAMP_THICKNESS,
        )
        super().run()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Amphibious pool — extended tank with ramp animat",
    )
    parser.add_argument(
        "--no-run", action="store_true",
        help="Generate config files only, do not launch the simulation.",
    )
    args = parser.parse_args()

    cfg = SimConfig()
    if args.no_run:
        # Generate ramp SDF + configs without launching
        create_ramp_sdf(
            length=RAMP_LENGTH,
            width=POOL_YMAX - POOL_YMIN,
            thickness=_RAMP_THICKNESS,
        )
        output_folder = cfg.stack_folder
        os.makedirs(output_folder, exist_ok=True)
        cfg.gen_simulation_config(output_folder)
        cfg.gen_experiment_config(output_folder)
        cfg.gen_arena_config(output_folder)
        cfg.gen_animat_config(output_folder)
        print(f"Configs written to {output_folder}")
    else:
        cfg.run()
