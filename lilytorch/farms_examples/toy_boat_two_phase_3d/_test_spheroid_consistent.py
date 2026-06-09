#!/usr/bin/env python3
"""Decisive test: the elongated SPHEROID spawned AT the waterline (the case that
blows up). Baseline (non-conservative) should explode early; CONSISTENT
conservative momentum should survive. Python path (consistent_momentum needs it).

  FLOAT_BODY=spheroid CONSISTENT=0 NITER=80 python3 _test_spheroid_consistent.py
  FLOAT_BODY=spheroid CONSISTENT=1 NITER=80 python3 _test_spheroid_consistent.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ["PYTHONPATH"] = _HERE + os.pathsep + os.environ.get("PYTHONPATH", "")

from gen_float_demo import FloatDemo, BODY
from gen_configs_small import WATERLINE


class SpheroidConsistentTest(FloatDemo):
    def __init__(self):
        super().__init__()
        self.headless = True
        self.solver_method = "python"          # consistent_momentum needs the python path
        self.poisson_method = "mgcg"           # two-phase variable-density Poisson
        # coarser grid so the python reference path is tractable for the test
        self.Nx, self.Ny, self.Nz = 96, 48, 36
        self.n_iterations = int(os.environ.get("NITER", "80"))
        self.bdim_nt = self.n_iterations
        self.save_frames = False
        self.save = False
        # spawn the body CENTRE at the waterline -> static float -> unstable case
        self.animats_pars[0]["pose"] = [0.30, 0.0, WATERLINE, 0.0, 0.0, 0.0]

    def _bdim_extension(self, output_folder):
        ext = super()._bdim_extension(output_folder)
        tp = ext["config"]["bdim_yaml"]["solver"]["two_phase"]
        if os.environ.get("CONSISTENT") == "1":
            tp["consistent_momentum"] = True
        if os.environ.get("RHO_SOLID"):
            tp["rho_solid"] = float(os.environ["RHO_SOLID"])   # body as 3rd density phase
        if os.environ.get("NCYCLES"):
            tp["consistent_n_cycles"] = int(os.environ["NCYCLES"])  # fixed-point iteration
        return ext

    def extra_simulation_extensions(self, output_folder):
        return []                              # no viewer for the headless test


if __name__ == "__main__":
    mode = "CONSISTENT" if os.environ.get("CONSISTENT") == "1" else "BASELINE"
    print(f"=== SPHEROID @ waterline | {BODY} | {mode} | python ===", flush=True)
    SpheroidConsistentTest().run()
