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


# ── Boat geometry (DSYHS yacht hull from OBJ meshes; mass comes from SDF) ──
# Native mesh frame: X = length (0..6.04), Y = vertical UP, Z = beam (±1.088).
# The model is spawned with roll = +pi/2 so mesh-Y(up) -> world-Z(up).
BOAT_LENGTH  = 6.04    # m  (X — long axis)
BOAT_BEAM    = 2.18    # m  (Z in mesh frame → world Y after spawn)
BOAT_MASS    = 420.0   # kg (total, from toy_boat.sdf <inertial> blocks)

# ── Tank (sized for the ~6 m hull, with fore/aft + lateral margins) ─────────
#   X: -1.5 .. 8.1 (boat 0..6.04)   Y: -2.4 .. 2.4 (beam ±1.09)
#   Z: -1.5 .. 2.1 (keel/rudder hang to ~-0.2 below the waterline)
WATERLINE = 0.40       # m  (world-z of the free surface)
# Spawn the boat AT its floating draft so it starts in near-equilibrium and
# does not free-fall.  Computed from the hull SDF: the boat displaces 0.83 m^3
# (~800 kg) at a draft of 0.24 m, i.e. hull bottom (mesh-y -0.094) at world-z
# 0.16 -> origin (mesh-y 0) at world-z 0.256.  The old 1.10 spawn free-fell
# 0.7 m and slammed in at ~3.3 m/s, plunging far past the float line where the
# violent, asymmetric entry pitched the boat (NOT a mass-balance problem: the
# boat trims level at its design waterline, COM_x 2.215 vs COB_x 2.20).
SPAWN_Z   = 0.256      # m  (world-z of the model origin = floating draft)


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
        self.water_drag     = False      # BDIM handles all hydrodynamics
        self.water_buoyancy = False
        self.water_height   = WATERLINE

        self.animat_fluid_interaction = True
        # Keep fixed joints (keel/rudder welds + blade attachments) filtered
        # out so only the revolute propeller joint gets a control motor.
        self.filter_fixed_joints = True
        # Hull / keel / rudder are OBJ meshes → build their fluid SDFs.
        # Leave n_samples unset: the mesh SDF table is then auto-sized from
        # the bounding box at h/2 spacing (a fixed (2000, 2000) would build a
        # ~2000×2000×k table and OOM for a multi-metre hull).
        self.compute_sdf = True

        # ── 3-D grid (h=0.05 m) ───────────────────────────────────────────
        # NOTE (resolution experiment, 2026-06): refining to h=0.03 + dt=1e-4
        # made the blow-up come SOONER (it~50 vs it~533), not later — so the
        # instability is NOT under-resolution.  It is a μ0/SDF stiffness at the
        # seams of the overlapping convex hulls (convexify=True, mandatory: the
        # raw meshes are broken), which finer cells sharpen.  Keep h=0.05.
        self.Nx   = 192
        self.Ny   = 96
        self.Nz   = 72
        self.xmin = -1.5
        self.xmax = 8.1
        self.ymin = -2.4
        self.ymax = 2.4
        self.zmin = -1.5
        self.zmax = 2.1

        # ── Animats ───────────────────────────────────────────────────
        boat_sdf = os.path.join(self.data_folder, 'toy_boat.sdf')
        controller_config = {
            "path": "lilytorch.farms_examples.submarine."
                    "propeller_controller.PropellerController",
            "tau": 2.0,   # propeller torque (tune for thrust)
        }
        self.animats_pars = [
            {
                "model_name"       : "toy_boat",
                "sdf_file"         : boat_sdf,
                "control_type"     : "torque",
                "gains"            : [0.0, 0.0, 0.0],
                "controller_config": controller_config,
                "spawn_mode"       : SpawnMode.FREE,
                # roll = +pi/2 rotates the mesh (Y-up) into the world (Z-up)
                # frame; the boat floats at SPAWN_Z and settles to its draft.
                "pose"             : [
                    0.0,                    # x: hull spans world-x 0..6.04
                    0.0,                    # y: beam centred on y=0
                    SPAWN_Z,                # z: hull bottom ≈ waterline
                    1.5707963267948966,     # roll = +pi/2 (mesh Y-up → world Z-up)
                    0.0, 0.0,
                ],
            },
        ]

        # ── FSI coupling stability ────────────────────────────────────
        # Explicit coupling with a light force low-pass.  NOTE: the violent
        # rise+pitch on entry was NOT a coupling instability — it was a wrong
        # FORCE (force_method="lagrangian" over-buoyed a surface-straddling body
        # 3x with a huge spurious pitch torque; the fix is the eulerian band
        # integral below).  Implicit coupling only delayed the symptom of that
        # bad force, so it is left off (it re-solves the fluid many times per
        # step → slow in 3-D).  Set scheme:"implicit" to re-enable if a genuine
        # added-mass instability shows up once the residual blow-up is cured.
        self.force_relaxation = 0.5
        self.coupling = {
            "scheme"     : "explicit",
            "accelerator": "iqn-ils",
            "reuse"      : 2,
            "tol"        : 1.0e-4,
            "max_iter"   : 30,
        }
        # convexify=True makes the hull/keel/rudder convex hulls overlap; the
        # running-min SDF union then HARD-SWITCHES the imposed solid velocity at
        # the seam between links, injecting a grid-scale divergence → pressure
        # spike → blow-up (and finer cells make it WORSE, which is what we saw).
        # Blend the imposed band velocity with an SDF-weighted softmin over a
        # few cells to make it continuous across the seam.
        self.body_velocity_blend_eps_cells = None

        # ── Physics (real ~6 m scale) ─────────────────────────────────
        self.rho_body     = 1000.0         # Poisson conditioning (not boat density)
        self.rho          = 1000.0         # water density
        self.nu           = 3.0e-3         # effective viscosity → Re ≈ U·L/ν ~ 1e3
        self.timestep     = 0.0005
        self.n_iterations = 8000           # 4 s of simulation
        self.num_sub_steps     = 1
        self.save_every        = 50

        # ── BDIM solver ──────────────────────────────────────────────
        self.bdim_dt                 = self.timestep
        self.bdim_nt                 = self.n_iterations
        self.convection_method       = "abdquickest"
        self.convexify                = True               # re-express the convection term in a more stable form (BDIM-specific)
        self.poisson_method          = "multigrid"
        self.poisson_tol             = 1.0e-6
        self.poisson_max_cycles      = 4
        self.poisson_max_mgcg_cycles = 30
        self.poisson_precond_vcycles = 1
        # self.poisson_warm_start      = True
        self.poisson_smoother        = "rbgs"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "neumann"
        self.solver_method           = "kernel"
        self.time_integration        = "euler"
        # Eulerian band-integral forces: hull, keel, rudder and propeller are
        # separate (overlapping) fluid bodies, so the force is integrated over
        # the smoothed delta of the *union* SDF rather than each body's closed
        # surface (gauge-invariant; handles the overlaps correctly).
        self.force_method            = "eulerian"
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
        self.wall_thickness = 0.2
        self.arena_pose     = [0, 0, 0, 0, 0, 0]


        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 1.0
        self.extent       = 12.0
        self.shadow_size  = 1024

        # ── Output ───────────────────────────────────────────────────
        self.save_frames = True
        self.vmin        = -2.0
        self.vmax        = 2.0

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
            "nu_air": 5.0e-2,
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
            water_alpha=0.1,
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
                "distance"        : 11.0,            # whole ~6 m boat in frame
                "offset"          : [3.0, 0.0, WATERLINE],  # look at the boat centre
                "resolution"      : [1920, 1080],
            },
        })

        return extensions


if __name__ == "__main__":
    SimConfig().run()
