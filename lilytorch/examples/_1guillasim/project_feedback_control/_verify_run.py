"""Env-driven headless harness to bisect the 2-D free-swimmer explosion.

Overrides (all optional):
  VERIFY_NITER      int   number of iterations (default 6000)
  VERIFY_CONVEXIFY  bool  convexify
  VERIFY_MU0        bool  bdim_mu0_projection
  VERIFY_DIVCORR    bool  bdim_body_div_correction
  VERIFY_POISSON    str   poisson_method ("fft" | "multigrid")
  VERIFY_FMETHOD    str   force_method ("eulerian" | "lagrangian")
  VERIFY_FSCALE     float force_scaling (0 -> free body, no fluid feedback)
  VERIFY_BLEND      float body_velocity_blend_eps_cells
  VERIFY_IMPLICIT   bool  enable implicit coupling (iqn-ils)
"""
import os
from lilytorch.examples._1guillasim.project_feedback_control.gen_config import (
    SimConfig,
)


def _env_bool(name, cur):
    v = os.environ.get(name)
    if v is None:
        return cur
    return v.strip().lower() in ("1", "true", "yes")


class VerifyConfig(SimConfig):

    def __init__(self):
        super().__init__()
        self.headless = True
        self.fast = True
        self.save = False
        self.n_iterations = int(os.environ.get("VERIFY_NITER", "6000"))
        self.bdim_nt = self.n_iterations + 1

        self.convexify = _env_bool("VERIFY_CONVEXIFY", self.convexify)
        self.bdim_mu0_projection = _env_bool(
            "VERIFY_MU0", self.bdim_mu0_projection)
        self.bdim_body_div_correction = _env_bool(
            "VERIFY_DIVCORR", self.bdim_body_div_correction)
        if os.environ.get("VERIFY_POISSON"):
            self.poisson_method = os.environ["VERIFY_POISSON"]
        if os.environ.get("VERIFY_FMETHOD"):
            self.force_method = os.environ["VERIFY_FMETHOD"]
        if os.environ.get("VERIFY_FSCALE") is not None:
            self.force_scaling = float(os.environ["VERIFY_FSCALE"])
        if os.environ.get("VERIFY_BLEND"):
            self.body_velocity_blend_eps_cells = float(
                os.environ["VERIFY_BLEND"])
        if _env_bool("VERIFY_IMPLICIT", False):
            self.coupling = {
                "scheme": "implicit",
                "accelerator": "iqn-ils",
                "reuse": 2,
                "tol": 1e-4,
                "max_iter": 30,
            }
        if _env_bool("VERIFY_EXPLICIT", False):
            self.coupling = None
        if os.environ.get("VERIFY_ACCEL"):
            if not isinstance(self.coupling, dict):
                self.coupling = {"scheme": "implicit", "tol": 1e-4,
                                 "max_iter": 30}
            self.coupling["accelerator"] = os.environ["VERIFY_ACCEL"]
        if os.environ.get("VERIFY_REUSE") is not None and \
                isinstance(self.coupling, dict):
            self.coupling["reuse"] = int(os.environ["VERIFY_REUSE"])

    def extra_simulation_extensions(self, output_folder):
        # No cameras / lights in headless verification runs.
        return []


if __name__ == "__main__":
    VerifyConfig().run()
