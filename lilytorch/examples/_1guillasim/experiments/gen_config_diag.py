"""Headless diagnostic harness for the surface-pool flip investigation.

Reuses the real surface-pool SimConfig but:
  * runs headless for a short horizon,
  * logs the base-link roll/pitch/yaw every step (OrientationLogger),
  * exposes a few knobs through env vars so we can A/B the hypotheses:

      DIAG_TAG        output label                       (default "baseline")
      DIAG_N          n_iterations                       (default 1200)
      DIAG_CONVEXIFY  "1"/"0"  link convexify            (default inherit=1)
      DIAG_NZ         override Nz                         (default inherit=52)
      DIAG_GAIT       "1"=replay CSV gait, "0"=hold straight (default 1)
      DIAG_OUT        directory for the CSV log          (default /data/.../_flip_diag)
"""

import os

from gen_config_surface_pool import SimConfig as _PoolConfig

_OUT = os.environ.get("DIAG_OUT", "/data/andreaferrario/lilytorch/_flip_diag")
os.makedirs(_OUT, exist_ok=True)
_TAG = os.environ.get("DIAG_TAG", "baseline")


class SimConfig(_PoolConfig):

    def __init__(self):
        super().__init__()
        self.headless     = True
        self.save         = False
        self.n_iterations = int(os.environ.get("DIAG_N", 1200))
        self.bdim_nt      = self.n_iterations + 1

        if "DIAG_CONVEXIFY" in os.environ:
            self.convexify = os.environ["DIAG_CONVEXIFY"] == "1"
        if "DIAG_NZ" in os.environ:
            # refine z-resolution: keep the SAME z-extent (inherited from the
            # pool config) and add cells -> smaller dz, sharper interface.
            self.Nz = int(os.environ["DIAG_NZ"])

        # Pin the diagnostic harness to the original coarse grid (fast, fits
        # 16 GB) regardless of what the production config currently uses, unless
        # a finer grid is explicitly requested.
        if os.environ.get("DIAG_FINE", "0") != "1":
            self.Nx, self.Ny, self.Nz = 900, 300, 52
            self.zmin = -(2 / 300 * 52 / 2)
            self.zmax = (2 / 300 * 52 / 2)

        # Small domain at 2x finer isotropic resolution (dx=dz=3.33 mm) to test
        # whether the surface buoyancy under-resolution is what sinks the head.
        if os.environ.get("DIAG_FINE", "0") == "1":
            self.xmin, self.xmax = 4.2, 5.4
            self.ymin, self.ymax = -0.4, 0.4
            self.zmin, self.zmax = -0.12, 0.12
            self.Nx, self.Ny, self.Nz = 360, 240, 72   # dx=dy=dz=1/300 m
            self.grid_spacing = 0.5 * (self.ymax - self.ymin)

        if "DIAG_SDF" in os.environ:
            self.animats_pars[0]["sdf_name"] = os.environ["DIAG_SDF"]

        if "DIAG_FORCE" in os.environ:           # "eulerian" | "lagrangian"
            self.force_method = os.environ["DIAG_FORCE"]

        if "DIAG_SPAWNZ" in os.environ:
            self.animats_pars[0]["pose"][2] = float(os.environ["DIAG_SPAWNZ"])

        # Re-enable simple vertical water drag to damp the heave overshoot.
        if os.environ.get("DIAG_DRAG", "0") == "1":
            self.water_drag = True

        # Hold the body straight (no undulation) to test gait-driven vs passive.
        if os.environ.get("DIAG_GAIT", "1") == "0":
            self.animats_pars[0]["controller_path"] = (
                "lilytorch.examples._1guillasim.experiments."
                "straight_controller.StraightController"
            )

    def extra_simulation_extensions(self, output_folder):
        exts = super().extra_simulation_extensions(output_folder)
        exts.append({
            "loader": "lilytorch.integration.orientation_logger.OrientationLogger",
            "config": {
                "log_path": os.path.join(_OUT, f"orient_{_TAG}.csv"),
                "base_body_match": "link0",
            },
        })
        return exts


if __name__ == "__main__":
    SimConfig().run()
