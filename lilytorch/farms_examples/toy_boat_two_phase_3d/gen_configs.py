#!/usr/bin/env python3
"""Toy motorboat in a small two-phase (water+air) tank with spinning propeller.

A cigar-shaped hull (cylinder L=0.12 m, R=0.028 m, ~350 kg/m³) with a rounded
bow, deck cabin, keel fin, and a two-blade propeller at the stern.  The
propeller is driven by a constant-torque controller (same PropellerController
used by the submarine example), and the BDIM fluid coupling turns blade
rotation into forward thrust.

The two-phase VOF solver models both water and real air, so the boat
experiences emergent buoyancy and dynamic pressure forces.

Grid:  112 × 56 × 72  (h=0.0025 m, ~11 cells across the hull beam).
The effective viscosity is raised (~2e-4 m²/s) to keep the Reynolds number low
(Re ~ U·L/ν ≈ 0.05·0.12/2e-4 ≈ 30) so the flow is laminar and stable.

Run with::

    cd lilytorch/farms_examples/toy_boat_two_phase_3d
    bash run.sh
"""

import os

from farms_core.model.options import SpawnMode

from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.farms_examples.base_sim_config import BaseSimConfig


# ── Toy boat geometry (SI units — reference only; mass comes from SDF) ────
BOAT_LENGTH  = 0.12   # m  (X — long axis, cylinder hull + bow sphere)
BOAT_RADIUS  = 0.028  # m  (hull cylinder radius → beam = 0.056 m)
BOAT_MASS    = 0.12   # kg (total, from toy_boat.sdf <inertial>)

# ── Tank ──────────────────────────────────────────────────────────────────
TANK_LX, TANK_LY, TANK_LZ = 0.28, 0.14, 0.18   # m  (comfortable for 0.12 m boat)
WATERLINE = 0.10                                  # m  (z-coordinate of the surface)


class SimConfig(BaseSimConfig):
    """Two-phase toy motorboat configuration with propeller propulsion."""

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', 'toy_boat_two_phase_3d',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.use_gpu  = True
        self.headless = False

        # ── Simulation flags ──────────────────────────────────────────
        self.use_bdim = True
        self.animat_fluid_interaction = True
        # Keep fixed joints (blade attachments) filtered out so only the
        # revolute propeller joint gets a control motor.
        self.filter_fixed_joints = True

        # ── 3-D grid ──────────────────────────────────────────────────
        self.Nx   = 112
        self.Ny   = 56
        self.Nz   = 72
        self.xmin = 0.0
        self.xmax = TANK_LX
        self.ymin = 0.0
        self.ymax = TANK_LY
        self.zmin = 0.0
        self.zmax = TANK_LZ

        # ── Animats ───────────────────────────────────────────────────
        boat_sdf = os.path.join(self.data_folder, 'toy_boat.sdf')
        controller_config = {
            "path": "lilytorch.farms_examples.submarine."
                    "propeller_controller.PropellerController",
            "tau": 0.0002,   # very gentle torque
        }
        self.animats_pars = [
            {
                "model_name"       : "toy_boat",
                "sdf_file"         : boat_sdf,
                "control_type"     : "torque",
                "gains"            : [0.0, 0.0, 0.0],
                "controller_config": controller_config,
                "spawn_mode"       : SpawnMode.FREE,
                # Spawn above waterline (like sphere drop example), centred in XY.
                "pose"             : [
                    TANK_LX / 2,           # x = 0.14
                    TANK_LY / 2,           # y = 0.07
                    WATERLINE + 0.03,      # z = 0.13 (fully in air, will drop)
                    0.0, 0.0, 0.0,
                ],
            },
        ]

        # ── Physics ───────────────────────────────────────────────────
        self.rho_body     = 1000.0         # Poisson conditioning (not boat density)
        self.rho          = 1000.0         # water density
        self.nu           = 2.0e-4         # effective viscosity → low Re
        self.timestep     = 0.0001
        self.n_iterations = 6000           # 0.6 s of simulation
        self.num_sub_steps     = 1
        self.save_every        = 100

        # ── BDIM solver ──────────────────────────────────────────────
        self.bdim_dt                 = self.timestep
        self.bdim_nt                 = self.n_iterations
        self.convection_method       = "abdquickest"
        self.poisson_method          = "multigrid"
        self.poisson_tol             = 1.0e-6
        self.poisson_max_cycles      = 4
        self.poisson_max_mgcg_cycles = 30
        self.poisson_precond_vcycles = 1
        self.poisson_warm_start      = True
        self.poisson_smoother        = "rbgs"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "neumann"
        self.solver_method           = "kernel"
        self.time_integration        = "euler"
        self.force_method            = "lagrangian"
        self.force_delta_order       = 1
        self.eps_multiplier          = 2
        self.zero_pressure_inside    = False
        self.dtype                   = "float32"

        # Boundary conditions — Neumann on lateral walls (like submarine),
        # Dirichlet on top/bottom to avoid free-slip water loss.
        self.bc_type_u   = ["D", "D", "N", "N", "N", "N"]
        self.bc_values_u = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.bc_type_v   = ["N", "N", "D", "D", "N", "N"]
        self.bc_values_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.bc_type_w   = ["N", "N", "N", "N", "D", "D"]
        self.bc_values_w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # ── Arena ────────────────────────────────────────────────────
        self.wall_thickness = 0.02
        self.arena_pose     = [0, 0, 0, 0, 0, 0]
        self.water_drag     = False   # BDIM handles all hydrodynamics
        self.water_buoyancy = False
        self.water_height   = WATERLINE

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 1.0
        self.extent       = 10.0
        self.shadow_size  = 1024

        # ── Output ───────────────────────────────────────────────────
        self.save_frames = True
        self.vmin        = -5.0
        self.vmax        = 5.0

    # ------------------------------------------------------------------
    # Override: inject the two-phase block into the BDIM YAML
    # ------------------------------------------------------------------
    def _bdim_extension(self, output_folder):
        """Build the BDIM extension dict, adding the two-phase VOF block."""
        bdim_ext = super()._bdim_extension(output_folder)

        # The base _bdim_extension returns a dict with:
        #   loader, config.handler_path, config.bdim_yaml
        solver = bdim_ext["config"]["bdim_yaml"]["solver"]

        # Two-phase: water + real air free surface
        solver["gravity"] = [0, 0, -9.81]
        solver["two_phase"] = {
            "alpha_init": f"lambda X, Y, Z: (Z < {WATERLINE}).double()",
            "rho_water": 1000.0,
            "rho_air": 1.2,
            "nu_water": self.nu,
            "nu_air": 3.0e-3,
            "face_density": "harmonic",
        }

        return bdim_ext

    # ------------------------------------------------------------------
    # Override: water SDF only fills up to the waterline (not tank top)
    # ------------------------------------------------------------------
    def gen_arena_config(self, output_folder, index=0):
        """Generate pool + water SDFs, with cosmetic water stopping at WATERLINE."""
        from lilytorch.integration.gen_pool_sdf import create_pool_sdf, create_water_sdf
        from lilytorch.util.paths import sdfs_path
        from farms_core.io.yaml import pyobject2yaml
        from farms_core.model.options import SpawnMode

        wt = self.wall_thickness
        if wt is None:
            pool_dims = [self.xmax - self.xmin, self.ymax - self.ymin,
                         self.zmax - self.zmin]
            wt = max(round(0.08 * min(pool_dims), 4), 0.01)

        # Pool: walls + floor (open top)
        create_pool_sdf(
            self.xmin, self.xmax, self.ymin, self.ymax,
            zmin=self.zmin, zmax=self.zmax,
            wall_thickness=wt, plotting=False,
            wall_alpha=self.wall_alpha,
            grid_spacing=self.grid_spacing,
            floor_color=self.floor_color,
        )
        # Cosmetic water: only fills up to WATERLINE, not the tank top
        water_sdf = create_water_sdf(
            self.xmin, self.xmax, self.ymin, self.ymax,
            zmin=self.zmin, zmax=WATERLINE,
            water_height=0.0,
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

    # ------------------------------------------------------------------
    # Extensions: FlowIsoGLViewer + video recording
    # ------------------------------------------------------------------
    def extra_simulation_extensions(self, output_folder):
        extensions = []

        # Live air/water INTERFACE in the MuJoCo viewer: the VOF field
        # alpha at iso 0.5, drawn as a translucent blue surface.
        extensions.append({
            "loader": "lilytorch.integration.flow_iso_gl_viewer.FlowIsoGLViewer",
            "config": {
                "field": "interface",
                "iso_value": 0.5,
                "alpha": 0.45,
                "color_uni": "#3399FF",
                "smooth_sigma": 0,
                "exclude_body": False,
                "update_every": 1,
                "max_vertices": 800000,
                "crop_boundary": 0,
                "debug_force_visible": False,
            },
        })

        # Fixed side-on camera recording the tank + boat at the waterline.
        extensions.append({
            "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
            "config": {
                "path"            : os.path.join(output_folder, "output", "video.mp4"),
                "animat_id"       : None,            # fixed camera (not following the boat)
                "fps"             : 30,
                "speed"           : 0.5,
                "angular_velocity": 0,
                "azimuth"         : 90,              # side view
                "elevation"       : -15,             # slightly above horizontal
                "distance"        : 0.35,            # close-up on the small tank
                "offset"          : [TANK_LX / 2, TANK_LY / 2, WATERLINE],  # look at waterline centre
                "resolution"      : [1920, 1080],
            },
        })

        return extensions


if __name__ == "__main__":
    SimConfig().run()
