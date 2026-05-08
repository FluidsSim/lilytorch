"""Run the standalone free-swimming 3-D jellyfish fluid simulation.

This is the ad-hoc driver for the jellyfish example.  It

1. loads a YAML config describing the 3-D fluid solver parameters,
2. constructs :class:`lilytorch.src.solver.FluidSolver` from it,
3. *replaces* ``solver.composite_body`` with a
    :class:`JellyfishBody` — an analytical, pulsing SDF body that now also
    integrates its own rigid 6D motion from the solver's hydrodynamic loads,
    and
4. runs ``solver.run_sim()``.

MuJoCo cannot integrate the equations of motion of a body whose SDF is
being deformed at every time-step, so the entire pipeline is deliberately
side-stepped here: the bell actuation is analytical, while the global pose
is advanced directly in the standalone BDIM loop.

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
from lilytorch.util.paths import gen_new_folder
from lilytorch.util.yaml_operations import yaml2pyobject

from lilytorch.farms_examples.jellyfish.jellyfish_body import JellyfishBody, JellyfishParams


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config_fluid.yaml")


def _resolve_output_folder(pars: dict) -> str:
    """Create a timestamped output folder using the shared repo helper."""
    output = pars.setdefault("output", {})
    stack_folder = output.get("stack_folder")
    if not stack_folder:
        configured = output.get("save_path", "")
        normalized = os.path.normpath(configured) if configured else ""
        basename = os.path.basename(normalized) if normalized else ""
        stack_folder = basename if basename not in ("", ".") else "jellyfish_fluid_output"

    output_folder = gen_new_folder(stack_folder)
    output["existing_folder"] = output_folder
    return output_folder


def build_solver(config_path: str, dtype=torch.float32) -> FluidSolver:
    """Load the YAML, build the solver, and swap in the jellyfish body."""
    pars = yaml2pyobject(config_path)

    # Basic sanity check – this example is 3-D only.
    if "Nz" not in pars["solver"]:
        raise ValueError(
            f"{config_path}: jellyfish example requires a 3-D config "
            "(missing Nz/zmin/zmax in 'solver' section)."
        )

    output_dir = _resolve_output_folder(pars)
    solver = FluidSolver(pars, dtype=dtype, compute_forces=True)
    solver.jellyfish_output_dir = output_dir

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
        params=JellyfishParams.from_solver_config(pars),
    )
    solver.composite_body = jelly
    solver.n_bodies = len(jelly.bodies)

    # Prime the force / mu / normal fields so the very first step starts
    # with consistent masks.
    jelly.update(solver.starting_time, 0, dt=float(solver.dt))
    solver._recompute_normals()
    jelly.clear_history()

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
    if hasattr(solver.composite_body, "save_state_history"):
        output_dir = getattr(solver, "save_path", None)
        if output_dir is None:
            output_dir = getattr(solver, "jellyfish_output_dir", HERE)
            os.makedirs(output_dir, exist_ok=True)
        solver.composite_body.save_state_history(output_dir)
    elapsed = time.time() - t0

    print(f"\n  Config  : {config_path}")
    print(f"  Elapsed : {elapsed:.2f} s")
    print(f"  Grid    : {solver.nx}x{solver.ny}x{solver.nz}")
    print(f"  Steps   : {solver.nt}")
    output_dir = getattr(solver, "save_path", None)
    if output_dir is None:
        output_dir = getattr(solver, "jellyfish_output_dir", None)
    if output_dir is not None:
        print(f"  Output  : {output_dir}")


if __name__ == "__main__":
    main()
