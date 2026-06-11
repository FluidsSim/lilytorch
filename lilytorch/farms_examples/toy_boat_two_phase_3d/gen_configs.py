#!/usr/bin/env python3
"""Toy motorboat in a two-phase (water+air) tank with spinning propeller.

Full DSYHS yacht hull (L=6.04 m) with keel, rudder, and a 3-blade propeller
at the stern (just in front of the rudder).  The propeller is driven by a
constant-torque controller, and the BDIM fluid coupling turns blade rotation
into forward thrust.

The two-phase VOF solver models both water and real air, so the boat
experiences emergent buoyancy and dynamic pressure forces.  The air-transparent
body fix (on by default) stabilises the waterline triple-point.

Grid:  270 × 64 × 64  (h=0.10 m, ~1.1M cells, ~60 cells along the hull).
Tank:  27 × 6.4 × 6.4 m  (18 m ahead of the bow for forward navigation).

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
BOAT_MASS    = 2000.0  # kg (total: 1631 hull + 315 keel + 20 rudder + ~34 propeller; split chosen
                       # for level trim COM_x=COB_x=2.045 on the convex BDIM envelope; the heavy
                       # prop assembly stabilises the free revolute joint, see toy_boat.sdf)

# ── Tank (sized for the ~6 m hull, with maneuvering room) ──────────────────
#   X: -3 .. 24 (boat 0..6.04, 3 m behind, 18 m ahead for forward navigation)
#   Y: -3.2 .. 3.2 (beam ±1.09, generous lateral clearance)
#   Z: -2.8 .. 3.6 (keel/rudder hang below hull; 3.2 m air gap above)
WATERLINE = 0.40       # m  (world-z of the free surface)
# Spawn the boat AT its floating draft so it starts in near-equilibrium and
# does not free-fall.  At 2000 kg the boat displaces 2.0 m^3 = 30% of the convex
# BDIM envelope -> equilibrium model-origin world-z = +0.081 (hull wetted depth
# 0.41 m = 8.3 cells at h=0.05, well resolved).  See MASS_INERTIA_NOTES.md.
SPAWN_Z   = 0.08     # m  (world-z of the model origin = floating draft)


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
        self.use_bdim       = True
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

        # ── 3-D grid (h=0.10 m, ~1.1M cells) ─────────────────────────────
        # Coarser grid to keep cell count manageable in the longer tank.
        # ~60 cells along the 6 m hull, ~2.4 cells across the 0.24 m draft.
        # With the air-transparent-body fix (on by default), the waterline
        # triple-point is stable; multi-body seams may need blending.
        self.Nx   = 270*1
        self.Ny   = 64*1
        self.Nz   = 32*1
        self.xmin = -3.0
        self.xmax = 24.0
        self.ymin = -3.2
        self.ymax = 3.2
        self.zmin = -2.
        self.zmax = 1.2

        # ── Animats ───────────────────────────────────────────────────
        boat_sdf = os.path.join(self.data_folder, 'toy_boat.sdf')
        controller_config = {
            "path": "lilytorch.farms_examples.submarine."
                    "propeller_controller.PropellerController",
            "tau": 400.0,    # propeller torque [N·m]
            # Ramp torque linearly from 0 → tau over this many steps (0.5 s at
            # dt=0.001).  Prevents the blade spin-up vertical-force transient
            # that pitches the bow down before steady-state flow is established.
            "tau_ramp_steps": 500,
        }
        self.animats_pars = [
            {
                "model_name"       : "toy_boat",
                "sdf_file"         : boat_sdf,
                "control_type"     : "torque",
                "gains"            : [0.0, 0.0, 0.0],
                "controller_config": controller_config,
                "spawn_mode"       : SpawnMode.FREE,
                # roll  = +pi/2 rotates the mesh (Y-up) into the world (Z-up)
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
        # EXPLICIT coupling (implicit DIVERGES here: res->inf — strong coupling is
        # ill-conditioned with force_method='lagrangian' + the articulated
        # propeller joint, per the BDIMhandler warning).  The stern-up pitch is
        # NOT static (masses trim-balanced on the convex envelope: COM_x 2.254
        # over COB_x 2.192, static pitch torque ~-594 N*m ~ 0.3deg) and persists
        # under BOTH schemes -> it is not a coupling-scheme problem.  Prime
        # suspect now: the convex hull floats only ~14.5% submerged (~2.5 cells of
        # draft on h=0.10) while the lagrangian force band is ~2*eps=2 cells wide,
        # so the wetted layer is under-resolved -> fore-aft-asymmetric pressure
        # integration -> steady spurious pitch.  See MASS_INERTIA_NOTES.md.
        # self.force_relaxation = 0.5
        # self.coupling = {
        #     "scheme"     : "explicit",
        #     "accelerator": "iqn-ils",
        #     "reuse"      : 2,
        #     "tol"        : 1.0e-4,
        #     "max_iter"   : 30,
        # }
        # ── Joint damping: prevent passive revolute joints (propeller)
        # from spinning up due to tiny numerical fluid torques.  With
        # Ixx≈0.03 kg·m², even τ≈1e-3 N·m gives α≈0.03 rad/s² and the
        # angular velocity runs away over thousands of steps.
        # damping=1.0 gives τ_damp = -1.0·ω, enough to kill numerical
        # runaway while still letting the torque controller drive the
        # propeller when tau>0.
        self.joint_damping = 5.0
        # convexify=True makes the hull/keel/rudder convex hulls overlap; the
        # running-min SDF union then HARD-SWITCHES the imposed solid velocity at
        # the seam between links, injecting a grid-scale divergence → pressure
        # spike → blow-up (and finer cells make it WORSE, which is what we saw).
        # Blend the imposed band velocity with an SDF-weighted softmin over a
        # few cells to make it continuous across the seam.  This is the
        # documented cure for the convexify-overlap μ0/SDF seam stiffness that
        # drives the two-phase blow-up (sharpened by finer cells); 3 cells of
        # sigmoid SDF weighting smooths body_{u,v,w} across the link seams.

        # self.body_velocity_blend_eps_cells = 3

        # ── Physics (real ~6 m scale) ─────────────────────────────────
        self.rho_body      = 1000.0   # Poisson conditioning (not boat density)
        self.rho           = 1000.0   # water density
        self.nu            = 1.0e-6   # real water kinematic viscosity [m²/s]
        self.timestep      = 0.001
        self.n_iterations  = 8000
        self.num_sub_steps = 1
        self.save_every    = 50

        # ── BDIM solver ──────────────────────────────────────────────
        self.bdim_dt                 = self.timestep
        self.bdim_nt                 = self.n_iterations
        self.convection_method       = "abdquickest"
        self.convexify               = True
        self.poisson_method          = "multigrid"
        self.poisson_tol             = 1.0e-6
        self.poisson_max_cycles      = 4
        self.poisson_max_mgcg_cycles = 30
        self.poisson_precond_vcycles = 1
        self.poisson_smoother        = "rbgs"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "neumann"
        # self.solver_method           = "python"
        self.time_integration        = "euler"
        # Eulerian band-integral forces: hull, keel, rudder and propeller are
        # separate (overlapping) fluid bodies, so the force is integrated over
        # the smoothed delta of the *union* SDF rather than each body's closed
        # surface (gauge-invariant; handles the overlaps correctly).
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
        self.wall_thickness = 0.3       # slightly thicker walls for bigger tank
        self.arena_pose     = [0, 0, 0, 0, 0, 0]


        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 1.0
        self.extent       = 28.0        # scaled for ~27 m tank
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
            "nu_water": self.nu,       # 1.0e-6 — real water
            "nu_air": 1.5e-5,          # was 1.0e-3; stabilises air pressure (67× real, 50× less than old 0.05)
            # rho_solid (~rho_water) includes the body as a finite third density
            # in the BDIM band instead of excluding it (c->0), regularizing the
            # immersed-boundary Poisson at the waterline -> cures the body-band
            # blow-up.  Python path only; optimum near rho_water (4000 was worse).
            # NB: this stabilises the BODY band, NOT thin appendages — keep every
            # fin/blade >=3 cells thick (the 1-cell propeller still blows up).
            "rho_solid": 1000.0,
            "alpha_exclude_body": True,        # carve body interior out of the initial water
            "alpha_volume_compensate": True   # default; restore the displaced volume
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
            wall_alpha=0.02,
            grid_spacing=self.grid_spacing,
            floor_color=self.floor_color,
        )
        # Cosmetic water: only fills up to WATERLINE, not the tank top
        water_sdf = create_water_sdf(
            self.xmin, self.xmax, self.ymin, self.ymax,
            zmin=self.zmin, zmax=WATERLINE,
            water_height=self.water_height,  # must match arena water.height for correct Z placement
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

        # Live two-field overlay in the MuJoCo viewer: the air/water INTERFACE
        # (VOF alpha at iso 0.5, translucent blue) together with the vorticity
        # magnitude shell (omega_mag, more opaque orange) so the wake under the
        # surface is visible. Each layer keeps its own colour + opacity.
        extensions.append({
            "loader": "lilytorch.integration.flow_iso_gl_viewer.FlowIsoGLViewer",
            "config": {
                # Global (shared) knobs.
                "update_every": 1,
                "max_vertices": 800000,
                "crop_boundary": 0,
                "debug_force_visible": False,
                "fields": [
                    {   # air/water interface
                        "field": "interface",
                        "iso_value": 0.5,
                        "alpha": 0.45,
                        "color": "#3399FF",
                        "smooth_sigma": 0,
                        "exclude_body": False,
                    },
                    {   # vorticity-magnitude shell (the wake), water-only
                        "field": "omega_mag",
                        "iso_value": 50,
                        "alpha": 0.3,
                        "color": "#FF8C1A",
                        "smooth_sigma": 0,
                        "exclude_body": True,
                        "phase_mask": "water",    # only show vorticity in the water phase
                    },
                ],
            },
        })

        # CoM trail: draws a fading orange line tracing the boat's centre of mass
        # in the MuJoCo viewer, one segment every `spacing` steps.
        extensions.append({
            "loader": "farms_mujoco.simulation.extensions.TrailCoMViewer",
            "config": {
                "animat_id": 0,                      # track the first (only) animat
                "width"    : 20,                     # line width in pixels
                "rgba"     : [1.0, 0.3, 0.0, 0.6],   # orange, semi-transparent
                "spacing"  : 25,                     # draw a new segment every 25 steps
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
                "distance"        : 20.0,            # whole ~27 m tank in frame
                "offset"          : [10.0, 0.0, WATERLINE],  # look at boat centre
                "resolution"      : [1920, 1080],
            },
        })

        return extensions


if __name__ == "__main__":
    SimConfig().run()
