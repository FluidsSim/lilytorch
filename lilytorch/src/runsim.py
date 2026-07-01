"""
Run a FluidSolver simulation from one or more YAML configuration files.

Usage:
    python run_test.py                                           # runs both default configs
    python run_test.py flow_past_circle_2d.yaml                  # 2D only
    python run_test.py flow_past_sphere_3d.yaml                  # 3D only
    python run_test.py flow_past_circle_2d.yaml flow_past_sphere_3d.yaml
"""

import sys, os, time
import torch
import numpy as np

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(yaml_path):
    """Load a YAML config and run the simulation."""
    from lilytorch.src.solver import FluidSolver
    from lilytorch.util.yaml_operations import yaml2pyobject

    pars = yaml2pyobject(yaml_path)
    ndim = 3 if "Nz" in pars["solver"] else 2

    tag = os.path.splitext(os.path.basename(yaml_path))[0]
    print("=" * 60)
    print(f"  {tag}  ({ndim}D)")
    print("=" * 60)

    t0 = time.time()
    # dtype is read from pars['solver']['dtype'] when present (falls back
    # to float32). Pass dtype=... explicitly here only to override the YAML.
    solver = FluidSolver(pars, compute_forces=False)
    solver.run_sim()
    elapsed = time.time() - t0

    # ---- summary ----
    grid_str = (
        f"{solver.nx}x{solver.ny}x{solver.nz}" if ndim == 3
        else f"{solver.nx}x{solver.ny}"
    )
    print(f"\n  Config         : {yaml_path}")
    print(f"  Elapsed        : {elapsed:.2f}s")
    print(f"  Grid           : {grid_str}  (ndim={solver.ndim})")
    print(f"  Steps          : {solver.nt}")

    if hasattr(solver, "save_path"):
        print(f"  Output dir     : {solver.save_path}")

    print()


if __name__ == "__main__":
    configs = sys.argv[1:]

    if not configs:
        # Default: run both examples shipped alongside this script
        configs = [
            # os.path.join(SCRIPT_DIR, "configs", "flow_past_circle_2d.yaml"),
            os.path.join(SCRIPT_DIR, "configs", "flow_past_cylinder_3d.yaml"),
        ]

    for cfg in configs:
        # Allow bare filenames to be resolved from the script directory
        if not os.path.isabs(cfg) and not os.path.exists(cfg):
            candidate = os.path.join(SCRIPT_DIR, "configs", cfg)
            if os.path.exists(candidate):
                cfg = candidate

        if not os.path.exists(cfg):
            print(f"ERROR: config not found: {cfg}")
            sys.exit(1)

        run(cfg)

    print("All simulations finished.")
