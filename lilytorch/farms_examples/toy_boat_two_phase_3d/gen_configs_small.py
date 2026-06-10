#!/usr/bin/env python3
"""SCALED-DOWN (S=0.1) toy boat — controlled size/resolution experiment.

Same hull+keel+rudder as the full boat but 10x smaller (toy_boat_small.sdf),
in a proportionally scaled domain with a FINER grid (~7 cells across the draft
vs ~4.8 for the full boat, approaching the working sphere's ~9). No propeller.

Question: the full 6 m boat explodes at its waterline; the well-resolved sphere
demo does not.  If this small, better-resolved boat is STABLE -> the cause is
size/resolution; if it still explodes -> it is the multi-body mesh, not size.

Run::  cd lilytorch/farms_examples/toy_boat_two_phase_3d
       PYTHONPATH=$PWD python3 _verify_run_small.py     # headless z/pitch/Fz
"""
import os

from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root
from gen_configs import SimConfig

S          = 0.1
WATERLINE  = 0.20 * S      # 0.040
SPAWN_Z    = 0.7 * S     # 0.0256


class SmallSimConfig(SimConfig):
    """10x-smaller boat, finer grid, no propeller."""

    def __init__(self):
        super().__init__()

        # ── smaller SDF, no propeller, scaled spawn ───────────────────
        boat_sdf = os.path.join(self.data_folder, 'toy_boat_small.sdf')
        self.animats_pars = [{
            "model_name"       : "toy_boat_small",
            "sdf_file"         : boat_sdf,
            "control_type"     : "torque",
            "gains"            : [0.0, 0.0, 0.0],
            "controller_config": {
                "path": "lilytorch.farms_examples.submarine."
                        "propeller_controller.PropellerController",
                "tau": 0.0,
            },
            "spawn_mode"       : SpawnMode.FREE,
            "pose"             : [0.0, 0.0, SPAWN_Z,
                                  1.5707963267948966, 0.0, 0.0],
        }]

        # ── scaled domain (×0.1) + FINER grid (~7 cells/draft) ─────────
        # draft ≈ 0.024 m, h ≈ 0.00333 -> 7.2 cells/draft (vs 4.8 at full scale)
        self.Nx   = 288    # x: -0.15 .. 0.81  (0.96 / 288 = 0.00333)
        self.Ny   = 144    # y: -0.24 .. 0.24  (0.48 / 144 = 0.00333)
        self.Nz   = 108    # z: -0.15 .. 0.21  (0.36 / 108 = 0.00333)
        self.xmin = -0.15
        self.xmax = 0.81
        self.ymin = -0.24
        self.ymax = 0.24
        self.zmin = -0.15
        self.zmax = 0.21

        # ── Froude-scaled timestep (dt ∝ sqrt(S)) ─────────────────────
        self.timestep          = 1.5e-3
        self.n_iterations      = 20000
        self.bdim_dt           = self.timestep
        self.bdim_nt           = self.n_iterations

        # Two-phase stability: the fused KERNEL two-phase path is less stable at
        # the 833:1 density ratio (the bare hull blows up in the waterline body
        # band at it=69).  The PYTHON path + rho_solid (see _bdim_extension) is
        # the stable combination for a floating body at the waterline.
        self.solver_method     = "python"

        # Wall sponge: in this small tank the bobbing hull drives waves/air motion
        # into the CLOSE lateral walls, where the light air phase piles up at the
        # water/air/wall triple line and blows up (~it=438) even after rho_solid
        # cures the body band.  A quadratic absorbing layer (~6 cells) damps it;
        # the hull then settles to equilibrium (Fz≈weight) and runs indefinitely.
        self.sponge            = {"width": 0.02, "strength": 500.0,
                                  "axes": ["x", "y", "z"]}

        self.water_height = WATERLINE

        # Viewer: the full boat sets extent=12 for the ~10 m scene; at S=0.1 the
        # scene is ~1 m, so without rescaling the camera is zoomed ~40x out and
        # the boat + thin interface surface are an invisible speck. Scale it.
        self.extent = 1.2

    # waterline-dependent overrides (the base uses the module-level WATERLINE
    # of gen_configs.py; here it must be the scaled one)
    def _bdim_extension(self, output_folder):
        from lilytorch.farms_examples.base_sim_config import BaseSimConfig
        bdim_ext = BaseSimConfig._bdim_extension(self, output_folder)
        solver = bdim_ext["config"]["bdim_yaml"]["solver"]
        solver["gravity"] = [0, 0, -9.81]
        solver["two_phase"] = {
            "alpha_init": f"lambda X, Y, Z: (Z < {WATERLINE}).double()",
            "rho_water": 1000.0,
            "rho_air": 1.2,
            "nu_water": self.nu,
            "nu_air": 5.0e-2,
            "face_density": "harmonic",
            # rho_solid (~rho_water) includes the body as a finite third density
            # in the BDIM band INSTEAD of excluding it (c->0), regularizing the
            # immersed-boundary Poisson at the waterline -> cures the body-band
            # blow-up.  Python path only; optimum near rho_water (4000 was WORSE
            # than 1000).  This is the active stabiliser for the hull-only sim.
            "rho_solid": 1000.0,
        }
        return bdim_ext

    def gen_arena_config(self, output_folder, index=0):
        from lilytorch.integration.gen_pool_sdf import create_pool_sdf, create_water_sdf
        from lilytorch.util.paths import sdfs_path
        from farms_core.io.yaml import pyobject2yaml
        from farms_core.model.options import SpawnMode as _SM

        wt = 0.02
        create_pool_sdf(self.xmin, self.xmax, self.ymin, self.ymax,
                        zmin=self.zmin, zmax=self.zmax, wall_thickness=wt,
                        plotting=False, wall_alpha=self.wall_alpha,
                        grid_spacing=self.grid_spacing, floor_color=self.floor_color)
        water_sdf = create_water_sdf(self.xmin, self.xmax, self.ymin, self.ymax,
                                     zmin=self.zmin, zmax=WATERLINE,
                                     water_height=0.0, water_alpha=0.)  # visible water box
        arena_sdf = os.path.join(sdfs_path, "pool", "sdf", "pool.sdf")
        arena_dict = {
            "sdf": arena_sdf,
            "spawn": {"loader": 0, "mode": _SM.FREE,
                      "pose": list(self.arena_pose),
                      "velocity": [0, 0, 0, 0, 0, 0], "extras": {}},
            "water": {"sdf": water_sdf, "drag": self._water_drag,
                      "buoyancy": self._water_buoyancy, "height": WATERLINE,
                      "velocity": [0, 0, 0], "viscosity": 1.0,
                      "density": self.rho, "maps": ["", ""]},
            "ground_height": self.ground_height,
        }
        pyobject2yaml(os.path.join(output_folder, 'arena_config.yaml'), arena_dict)

    def extra_simulation_extensions(self, output_folder):
        # Live air/water INTERFACE in the MuJoCo viewer (the VOF field alpha at
        # iso 0.5). Without this the water is invisible — the cosmetic water SDF
        # is water_alpha=0.0, so the only thing that shows the surface is this
        # iso viewer. (The headless verify runner overrides this with just the
        # z-logger, so this only affects interactive `gen_configs_small.py`.)
        return [{
            "loader": "lilytorch.integration.flow_iso_gl_viewer.FlowIsoGLViewer",
            "config": {
                "field": "interface", "iso_value": 0.5, "alpha": 0.45,
                "color_uni": "#3399FF", "smooth_sigma": 0, "exclude_body": False,
                "update_every": 1, "max_vertices": 800000, "crop_boundary": 0,
                "debug_force_visible": False,
            },
        }]


if __name__ == "__main__":
    SmallSimConfig().run()
