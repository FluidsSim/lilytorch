#!/usr/bin/env python3
"""
Three-way Poisson solver comparison: multigrid vs MGCG vs FFT.

Runs the 3-D BDIM swimming simulation (1guilla pinned, headless, fluid-only)
with each Poisson method and reports per-step timing.

Usage
-----
    source /data/andreaferrario/venv_ns_312/bin/activate
    python run_poisson_comparison_3d.py
    python run_poisson_comparison_3d.py --Nx 256 --Ny 64 --Nz 64
"""

import argparse
import os
import sys
import time
import copy
import shutil
import types
from contextlib import contextmanager

import numpy as np
import torch

# ── CLI args ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="3-D Poisson solver comparison")
parser.add_argument("--Nx",        type=int, default=128)
parser.add_argument("--Ny",        type=int, default=32)
parser.add_argument("--Nz",        type=int, default=32)
parser.add_argument("--n_steps",   type=int, default=25,
                    help="Total steps (first --warmup are discarded)")
parser.add_argument("--warmup",    type=int, default=3)
parser.add_argument("--device",    type=str, default="cuda",
                    choices=["cuda", "cpu"])
args = parser.parse_args()

USE_CUDA = args.device == "cuda" and torch.cuda.is_available()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════
# Lightweight per-step timer
# ═══════════════════════════════════════════════════════════════════════

class StepTimer:
    def __init__(self, use_cuda, warmup):
        self.use_cuda = use_cuda
        self.warmup = warmup
        self.step_times = []
        self._step_count = 0

    def new_step(self):
        self._step_count += 1

    @contextmanager
    def time_step(self):
        self._step_count += 1
        if self._step_count <= self.warmup:
            yield
            return
        if self.use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if self.use_cuda:
            torch.cuda.synchronize()
        self.step_times.append(time.perf_counter() - t0)

    def mean_ms(self):
        if not self.step_times:
            return float('nan')
        return 1e3 * np.mean(self.step_times)

    def std_ms(self):
        if not self.step_times:
            return float('nan')
        return 1e3 * np.std(self.step_times)

    def median_ms(self):
        if not self.step_times:
            return float('nan')
        return 1e3 * np.median(self.step_times)


# ═══════════════════════════════════════════════════════════════════════
# Run one configuration
# ═══════════════════════════════════════════════════════════════════════

def run_one_method(method_name, nx, ny, nz, n_steps, warmup,
                   smoother="jacobi"):
    """Run simulation with given Poisson method, return StepTimer."""

    # IMPORTANT: We need to reimport the config module to get fresh state.
    # Since gen_configs writes to disk, we can modify the YAML after.
    import importlib
    import lilytorch.farms_examples._1guillasim.gen_configs_one_pinned_3d as cfg
    importlib.reload(cfg)

    cfg.Nx           = nx
    cfg.Ny           = ny
    cfg.Nz           = nz
    cfg.n_iterations = n_steps + 1
    cfg.save_every   = 99999
    cfg.save_frames  = False
    cfg.headless     = True
    cfg.save         = False

    # Override to strip non-fluid extensions and inject poisson_method
    _orig_gen_sim = cfg.gen_simulation_config

    def gen_sim_lean(output_folder):
        _orig_gen_sim(output_folder)

        import yaml
        yaml_path = os.path.join(output_folder, 'simulation_config.yaml')
        with open(yaml_path, 'r') as f:
            sim_dict = yaml.safe_load(f)

        # Keep only FluidExtension
        keep = {"lilytorch.integration.extensions.FluidExtension"}
        sim_dict["extensions"] = [
            ext for ext in sim_dict.get("extensions", [])
            if ext.get("loader", "") in keep
        ]
        sim_dict.setdefault("runtime", {})["headless"] = True

        # Inject poisson_method + smoother into solver config
        for ext in sim_dict["extensions"]:
            if ext.get("loader") == "lilytorch.integration.extensions.FluidExtension":
                solver_cfg = ext["config"]["bdim_yaml"]["solver"]
                solver_cfg["poisson_method"] = method_name
                solver_cfg["poisson_smoother"] = smoother
                # For FFT, ensure poisson_folder exists
                if method_name == "fft":
                    pf = solver_cfg.get("poisson_folder", "lilytorch/data/")
                    os.makedirs(pf, exist_ok=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(sim_dict, f, default_flow_style=False, sort_keys=False)

    cfg.gen_simulation_config = gen_sim_lean

    # Create output folder
    from lilytorch.util.paths import gen_new_folder, save_path
    output_folder = gen_new_folder(save_path)
    os.makedirs(output_folder, exist_ok=True)

    cfg.gen_animat_config(output_folder)
    cfg.gen_arena_config(output_folder)
    cfg.gen_simulation_config(output_folder)
    cfg.gen_experiment_config(output_folder)

    saved_cwd = os.getcwd()
    os.chdir(output_folder)

    # Set up timer and patch
    timer = StepTimer(USE_CUDA, warmup)

    from lilytorch.integration.extensions import FluidExtension
    _orig_init_ep = FluidExtension.initialize_episode

    def _patched_init(self_ext, task, physics):
        _orig_init_ep(self_ext, task, physics)
        handler = getattr(self_ext, 'BDIMhandler', None)
        if handler is None:
            return
        _orig_step = handler.step.__func__ if hasattr(handler.step, '__func__') else type(handler).step

        def timed_step(self_h, task, physics):
            with timer.time_step():
                _orig_step(self_h, task, physics)

        handler.step = types.MethodType(timed_step, handler)

    FluidExtension.initialize_episode = _patched_init

    try:
        from farms_core.experiment.options import ExperimentOptions
        from farms_core.extensions.extensions import import_item
        from farms_sim.simulation import run_simulation
        from farms_core.simulation.options import Simulator

        experiment_options = ExperimentOptions.load("experiment_config.yaml")
        experiment_data_loader = import_item(experiment_options.loaders.experiment_data)
        experiment_data = experiment_data_loader.from_options(experiment_options)

        run_simulation(
            experiment_data=experiment_data,
            experiment_options=experiment_options,
            simulator=Simulator.MUJOCO,
        )
    finally:
        FluidExtension.initialize_episode = _orig_init_ep
        os.chdir(saved_cwd)
        # Clean up output folder
        try:
            shutil.rmtree(output_folder)
        except Exception:
            pass

    return timer


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.path.insert(0, SCRIPT_DIR)
    os.chdir(SCRIPT_DIR)

    # Each entry: (label, poisson_method, smoother)
    configs = [
        ("mg-jacobi",   "multigrid", "jacobi"),
        ("mg-rbgs",     "multigrid", "rbgs"),
        ("mgcg-jacobi", "mgcg",      "jacobi"),
        ("fft",         "fft",       "jacobi"),
    ]
    results = {}

    grid_label = f"{args.Nx}x{args.Ny}x{args.Nz}"
    n_cells = args.Nx * args.Ny * args.Nz

    print("=" * 72)
    print(f"  Poisson Solver Comparison  ({grid_label}, {n_cells:,} cells)")
    print(f"  Steps: {args.n_steps} (warmup: {args.warmup})")
    print(f"  Device: {'CUDA' if USE_CUDA else 'CPU'}")
    print("=" * 72)

    for label, method, smoother in configs:
        print(f"\n{'─'*72}")
        print(f"  Running: {label}")
        print(f"{'─'*72}")

        # Clear GPU cache between runs
        if USE_CUDA:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            timer = run_one_method(
                method, args.Nx, args.Ny, args.Nz,
                args.n_steps, args.warmup,
                smoother=smoother,
            )
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if USE_CUDA else 0
            results[label] = {
                "mean": timer.mean_ms(),
                "std":  timer.std_ms(),
                "median": timer.median_ms(),
                "n": len(timer.step_times),
                "peak_mem_gb": peak_mem,
            }
            print(f"  {label}: {timer.mean_ms():.2f} ± {timer.std_ms():.2f} ms/step  "
                  f"(median {timer.median_ms():.2f} ms, n={len(timer.step_times)}, "
                  f"peak GPU {peak_mem:.2f} GB)")
        except Exception as e:
            print(f"  {label}: FAILED — {e}")
            import traceback; traceback.print_exc()
            results[label] = None

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  SUMMARY  ({grid_label} = {n_cells:,} cells, "
          f"{'GPU' if USE_CUDA else 'CPU'})")
    print("=" * 72)
    print(f"  {'Config':<24s} {'Mean (ms)':>10s} {'Std (ms)':>10s} "
          f"{'Median (ms)':>12s} {'Peak GPU':>10s} {'Speedup':>10s}")
    print("-" * 78)

    baseline = results.get("mg-jacobi", {})
    baseline_mean = baseline.get("mean", float('nan')) if baseline else float('nan')

    for label, _, _, _ in configs:
        r = results.get(label)
        if r is None:
            print(f"  {label:<24s}   FAILED")
            continue
        speedup = baseline_mean / r["mean"] if r["mean"] > 0 else float('nan')
        print(f"  {label:<24s} {r['mean']:10.2f} {r['std']:10.2f} "
              f"{r['median']:12.2f} {r['peak_mem_gb']:9.2f}G "
              f"{speedup:10.2f}x")

    print("=" * 78)
