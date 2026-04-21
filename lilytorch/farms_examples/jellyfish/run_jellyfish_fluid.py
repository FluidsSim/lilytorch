"""Run the 3-D jellyfish fluid simulation (no MuJoCo, no drag model).

This is the ad-hoc driver for the jellyfish example.  It

1. loads a YAML config describing the 3-D fluid solver parameters,
2. constructs :class:`lilytorch.src.solver.FluidSolver` from it,
3. *replaces* ``solver.composite_body`` with a
   :class:`JellyfishBody` — an analytical, time-varying SDF body that
   reproduces the WaterLily ``ThreeD_Jelly.jl`` jellyfish (thin
   spherical-shell bell intersected with a horizontal plane and animated
   with a sinusoidal "pulse"), and
4. runs ``solver.run_sim()``.

MuJoCo cannot integrate the equations of motion of a body whose SDF is
being deformed at every time-step, so the entire pipeline is deliberately
side-stepped here: the kinematics are prescribed analytically and the
BDIM fluid solver integrates Navier–Stokes around the moving, deforming
body — exactly the pattern used by WaterLily.

Usage
-----
::

    python -m lilytorch.farms_examples.jellyfish.run_jellyfish_fluid
    python -m lilytorch.farms_examples.jellyfish.run_jellyfish_fluid path/to/config.yaml

When no argument is passed, ``config_fluid.yaml`` next to this file is
used.
"""

from __future__ import annotations

import os
import sys
import time
import torch

from lilytorch.src.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject

from .jellyfish_body import JellyfishBody, JellyfishParams


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config_fluid.yaml")


def build_solver(config_path: str, dtype=torch.float32) -> FluidSolver:
    """Load the YAML, build the solver, and swap in the jellyfish body."""
    pars = yaml2pyobject(config_path)

    # Basic sanity check – this example is 3-D only.
    if "Nz" not in pars["solver"]:
        raise ValueError(
            f"{config_path}: jellyfish example requires a 3-D config "
            "(missing Nz/zmin/zmax in 'solver' section)."
        )

    solver = FluidSolver(pars, dtype=dtype, compute_forces=True)

    # ------------------------------------------------------------------
    # Replace the placeholder composite_body with the jellyfish body.
    # The solver's step_() only touches composite_body via update() and
    # a handful of well-defined attributes, all of which JellyfishBody
    # provides.  Re-using the already-built staggered grids keeps memory
    # at zero additional cost.
    # ------------------------------------------------------------------
    jelly = JellyfishBody(
        device=solver.device,
        x=solver.x,
        y=solver.y,
        z=solver.z,
        eps=float(solver.eps),
        grids=solver.grids,
        params=JellyfishParams(),
    )
    solver.composite_body = jelly
    solver.n_bodies = len(jelly.bodies)

    # Prime the force / mu / normal fields so the very first step starts
    # with consistent masks.
    jelly.update(solver.starting_time, 0, dt=float(solver.dt))
    solver._recompute_mu_normals()

    return solver


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = argv[0] if argv else DEFAULT_CONFIG

    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        candidate = os.path.join(HERE, config_path)
        if os.path.exists(candidate):
            config_path = candidate

    if not os.path.exists(config_path):
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"  jellyfish 3-D fluid simulation   ({os.path.basename(config_path)})")
    print("=" * 60)

    t0 = time.time()
    solver = build_solver(config_path)
    solver.run_sim()
    elapsed = time.time() - t0

    print(f"\n  Config  : {config_path}")
    print(f"  Elapsed : {elapsed:.2f} s")
    print(f"  Grid    : {solver.nx}x{solver.ny}x{solver.nz}")
    print(f"  Steps   : {solver.nt}")
    if hasattr(solver, "save_path"):
        print(f"  Output  : {solver.save_path}")


if __name__ == "__main__":
    main()
