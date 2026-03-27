#!/usr/bin/env python3
"""
Run the Gazzola et al. (2011) 2-D cylinder sedimentation experiment using FARMS.

Usage (from this directory):
    python run_gazzola.py [--low-res] [--steps N]

Reference:
    Gazzola et al. (2011). "C-start: optimal start of larval fish."
    J. Fluid Mech. 698, 5–17.
    Terminal velocity: Ut ≈ 2.501 cm/s  (Re ≈ 156)
"""

import argparse
import os
import sys

# --- ensure lilytorch is importable ------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from farms_core.experiment.options import ExperimentOptions
from farms_core.simulation.options import Simulator
from farms_sim.simulation import run_simulation, simulation_post

# Reference values (Gazzola 2011)
UT_REF = 0.02501   # m/s
RE_REF = 156


def parse_args():
    p = argparse.ArgumentParser(description='Gazzola sedimentation benchmark')
    p.add_argument('--low-res', action='store_true',
                   help='Use coarser grid (Nx=64, Ny=512) for quick testing')
    p.add_argument('--steps', type=int, default=None,
                   help='Override number of simulation steps (default: from yaml)')
    p.add_argument('--log-path', default='output',
                   help='Directory for output logs / plots')
    return p.parse_args()


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)   # FARMS resolves relative paths from CWD

    os.makedirs('data', exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    print("=" * 60)
    print("  Gazzola 2D cylinder sedimentation — FARMS + LilyTorch")
    print(f"  Target Ut = {UT_REF*100:.3f} cm/s   (Re = {RE_REF})")
    print("=" * 60)

    # --- load experiment options from YAML ----------------------------------
    experiment_options = ExperimentOptions.load('experiment_config.yaml')

    # --- optionally override resolution for quick testing -------------------
    bdim_cfg = experiment_options.simulation.extensions[0].config['bdim_yaml']
    if args.low_res:
        print("  [low-res] Using Nx=64, Ny=512 (quick test)")
        bdim_cfg['solver']['Nx'] = 64
        bdim_cfg['solver']['Ny'] = 512
    else:
        print(f"  [full-res] Using Nx={bdim_cfg['solver']['Nx']}, "
              f"Ny={bdim_cfg['solver']['Ny']}")

    if args.steps is not None:
        print(f"  [steps] Overriding nt → {args.steps}")
        bdim_cfg['solver']['nt'] = args.steps
        experiment_options.simulation.runtime.n_iterations = args.steps

    print(f"  eps_multiplier = {bdim_cfg['solver'].get('eps_multiplier', 1.0):.4f}")
    print("=" * 60 + "\n")

    # --- run simulation ------------------------------------------------------
    sim = run_simulation(
        experiment_options,
        simulator=Simulator.MUJOCO,
    )

    # --- post-process --------------------------------------------------------
    try:
        simulation_post(sim, log_path=args.log_path)
    except Exception as e:
        print(f"  [post-process skipped: {e}]")

    print("\n" + "=" * 60)
    print("  Simulation complete.")
    print("  Results logged to:", os.path.abspath(args.log_path))
    print("=" * 60)


if __name__ == '__main__':
    main()
