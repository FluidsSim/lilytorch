"""Jellyfish examples.

Two runtimes are provided:

* :mod:`gen_configs_drag` — legacy MuJoCo + FARMS drag-model setup
  (6-DOF Newton–Euler integrated by MuJoCo, no BDIM fluid coupling).
* :mod:`run_jellyfish_fluid` — fluid-only simulation that prescribes the
  WaterLily-style jellyfish kinematics and integrates Navier–Stokes
  around the moving, deforming body with lilytorch's 3-D BDIM
  ``FluidSolver``.  MuJoCo is not involved, which is what makes a
  deforming SDF actually possible.
"""

from .jellyfish_body import JellyfishBody, JellyfishParams

__all__ = ["JellyfishBody", "JellyfishParams"]
