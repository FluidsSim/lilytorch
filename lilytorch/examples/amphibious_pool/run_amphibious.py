#!/usr/bin/env python
"""Launch the amphibious pool simulation (extended tank + ramp).

Usage
-----
    python run_amphibious.py                        # generate configs + launch
    python run_amphibious.py --no-run               # generate configs only
    python run_amphibious.py --plot-ramp            # show a top-view plot of the pool/ramp

The pool is 12 m × 3 m × 0.6 m with a sloped ramp from the left wall at
the water surface down to the floor, enabling walking→swimming transitions.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure lilytorch is importable from this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LILYTORCH_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _LILYTORCH_ROOT not in sys.path:
    sys.path.insert(0, _LILYTORCH_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Amphibious pool — extended tank with ramp",
    )
    parser.add_argument(
        "--no-run", action="store_true",
        help="Generate config files only, do not launch the simulation.",
    )
    parser.add_argument(
        "--plot-ramp", action="store_true",
        help="Show a matplotlib top-view of the pool + ramp, then exit.",
    )
    args = parser.parse_args()

    if args.plot_ramp:
        from lilytorch.integration.gen_pool_sdf import create_pool_with_ramp_sdf
        from gen_config_amphibious import (
            POOL_XMIN, POOL_XMAX, POOL_YMIN, POOL_YMAX,
            POOL_ZMIN, POOL_ZMAX,
            RAMP_X_START, RAMP_Z_START, RAMP_X_END, RAMP_Z_END, RAMP_THICK,
        )
        create_pool_with_ramp_sdf(
            POOL_XMIN, POOL_XMAX, POOL_YMIN, POOL_YMAX,
            zmin=POOL_ZMIN, zmax=POOL_ZMAX,
            plotting=True,
            ramp={
                "x_start":  RAMP_X_START,
                "z_start":  RAMP_Z_START,
                "x_end":    RAMP_X_END,
                "z_end":    RAMP_Z_END,
                "thickness": RAMP_THICK,
            },
        )
        return

    from gen_config_amphibious import SimConfig

    cfg = SimConfig()
    if args.no_run:
        output_folder = cfg.stack_folder
        os.makedirs(output_folder, exist_ok=True)
        cfg.gen_simulation_config(output_folder)
        cfg.gen_experiment_config(output_folder)
        cfg.gen_arena_config(output_folder)
        cfg.gen_animat_config(output_folder)
        print(f"Configs written to {output_folder}")
    else:
        cfg.run()


if __name__ == "__main__":
    main()
