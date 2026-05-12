#!/usr/bin/env python3
"""
Computational-cost analysis for the 2-D pinned 1guilla simulation.

Instruments every major sub-kernel inside BDIMhandler (2-D path) with
CUDA-synchronised timers.  Reports per-step statistics, saves CSV and
paper-quality figures.

Operation groups:
  1. SDF update         – body kinematics → SDF eval on 3 staggered grids (cc, u, v)
  2. mu + normals       – Heaviside (mu) + normal computation
  3. Fluid step         – advection + diffusion + BDIM + BCs + projection
  4. Forces             – forces_method2 (2-D viscous stress + pressure)
  6. FARMS apply_forces – GPU→CPU force transfer + MuJoCo xfrc write

KEY DESIGN CHOICES:
  - Runs the FARMS simulation **in-process** so Python-level monkey-patches
    apply to every hot path.
  - All non-fluid extensions (FlowViewer2D, CameraRecording, DataLogger)
    are stripped for pure solver cost.
  - Headless mode.

Usage
-----
    source /path/to/venv/bin/activate
    python run_cost_analysis.py                          # default 512×128, 50 steps
    python run_cost_analysis.py --Nx 256 --Ny 64
    python run_cost_analysis.py --Nx 1024 --Ny 256 --mode kernel
"""

import argparse
import gc
import logging
import os
import sys
import time
import types
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

# ── torch.compile recompile / graph-break logging ───────────────────
torch._logging.set_logs(recompiles=True, graph_breaks=True)
_TORCH_DYNAMO_LOG = logging.getLogger("torch._dynamo")
_TORCH_DYNAMO_LOG.setLevel(logging.INFO)

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="2-D pinned 1guilla – computational cost analysis")
parser.add_argument("--Nx",        type=int, default=512)
parser.add_argument("--Ny",        type=int, default=128)
# The pinned 2-D benchmark currently becomes unstable beyond roughly
# 60 total simulation steps at the default 512x128 grid, so the default
# timing window stays inside that validated range.
parser.add_argument("--n_steps",   type=int, default=20,
                    help="Measured steps (compilation excluded via pre-compile phase)")
parser.add_argument("--precompile", type=int, default=30,
                    help="Pre-compilation steps (trigger all torch.compile captures)")
parser.add_argument("--settle_steps", type=int, default=5,
                    help="Untimed settle steps after pre-compilation")
parser.add_argument("--discard_first", type=int, default=5,
                    help="Discard first N timed steps from summary stats")
parser.add_argument("--stability_tol", type=float, default=0.05,
                    help="Warn if rolling (std/mean) of last 10 steps > threshold")
parser.add_argument("--save_every", type=int, default=9999)
parser.add_argument("--mode", type=str, default=None,
                choices=["python", "kernel"],
                help="Solver mode to benchmark. Use 'python' for the "
                    "reference path or 'kernel' for the optimised path.")
parser.add_argument("--use_kernels", action="store_true",
                help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--no_kernels", action="store_true",
                help="DEPRECATED: alias for --mode python.")
# Deprecated alias kept for backward compatibility — collapses into --use_kernels.
parser.add_argument("--streaming_sdf_2d", action="store_true",
                help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--device",    type=str, default="cuda",
                    choices=["cuda", "cpu"])
parser.add_argument("--out_dir",   type=str, default=None,
                    help="Output directory (default: figures/ in this folder)")
parser.add_argument("--Lx_fixed", type=float, default=None,
                    help="Override the x-extent of the fluid domain (m). "
                         "When set, Ly is derived from Ny/Nx ratio. "
                         "Overrides the default auto-scaling rule.")
parser.add_argument("--tag_suffix", type=str, default="",
                    help="Suffix appended to the CSV filename tag.")
args = parser.parse_args()


def _resolve_solver_mode(cli_args):
    legacy_kernel_mode = cli_args.use_kernels or cli_args.streaming_sdf_2d

    if cli_args.mode == "python":
        if legacy_kernel_mode:
            raise ValueError("--mode python conflicts with kernel-enabling aliases")
        return "python"
    if cli_args.mode == "kernel":
        if cli_args.no_kernels:
            raise ValueError("--mode kernel conflicts with --no_kernels")
        return "kernel"

    if cli_args.no_kernels and legacy_kernel_mode:
        raise ValueError("--no_kernels conflicts with kernel-enabling aliases")
    if cli_args.no_kernels:
        return "python"
    if legacy_kernel_mode:
        return "kernel"
    return None


try:
    SOLVER_MODE = _resolve_solver_mode(args)
except ValueError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)

USE_CUDA = args.device == "cuda" and torch.cuda.is_available()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if args.out_dir is None:
    args.out_dir = os.path.join(SCRIPT_DIR, "figures")
args.out_dir = os.path.abspath(args.out_dir)


# ═══════════════════════════════════════════════════════════════════════
# Timer infrastructure
# ═══════════════════════════════════════════════════════════════════════

class TimerBank:
    """CUDA-synchronised timer collection.

    An explicit ``_active`` flag gates accumulation: during pre-compile
    and settle the wrapped methods execute the production code paths but
    timings are discarded, so the first entry corresponds to a fully warm
    CUDA-graph-replayed step.
    """

    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda
        self._data: dict[str, list[float]] = defaultdict(list)
        self._step_count = 0
        self._active = False

    @contextmanager
    def __call__(self, label: str):
        if not self._active:
            yield
            return
        if self.use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if self.use_cuda:
            torch.cuda.synchronize()
        self._data[label].append(time.perf_counter() - t0)

    def new_step(self):
        self._step_count += 1

    def summary(self, discard_first: int = 0):
        trimmed = {}
        for label, times in self._data.items():
            trimmed[label] = times[discard_first:] if discard_first < len(times) else times
        outer_total = sum(trimmed.get("TOTAL step", [1.0]))
        rows = {}
        for label, times in sorted(trimmed.items()):
            arr = np.array(times)
            rows[label] = {
                "median": float(np.median(arr)),
                "mean":   arr.mean(),
                "std":    arr.std(),
                "min":    arr.min(),
                "max":    arr.max(),
                "total":  arr.sum(),
                "count":  len(arr),
                "pct":    100.0 * arr.sum() / outer_total if outer_total > 0 else 0,
            }
        return rows

T = TimerBank(USE_CUDA)


# ═══════════════════════════════════════════════════════════════════════
# Project imports
# ═══════════════════════════════════════════════════════════════════════
repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from lilytorch.integration.extensions import FluidExtension
from lilytorch.integration.BDIMhandler import BDIMhandler  # noqa: F401
from lilytorch.src.poisson_mult import PoissonSolver        # noqa: F401


# ═══════════════════════════════════════════════════════════════════════
# Deep instrumentation (instance-level monkey-patching)
# ═══════════════════════════════════════════════════════════════════════

def instrument_handler(handler):
    """Wrap every hot-path method on live 2-D instances with fine-grained timers.

    We NEVER reimplement production code paths; instead we wrap existing
    (possibly compiled) methods with timing context managers so that:
      • torch.compile / CUDA-graph recordings from precompile are REUSED
      • Results are identical to production
      • Sync overhead is minimised (≤ 7 timer blocks per step)
    """
    if getattr(handler, "_profiled", False):
        return
    handler._profiled = True

    fs   = handler.fluid_solver
    comp = fs.composite_body
    adv  = fs.adv_diff_solver

    _orig_step       = type(handler).step
    _orig_update     = handler.update
    _orig_fluid_step = type(fs).fluid_step
    _orig_apply      = type(handler)._apply_forces
    _orig_forces     = type(fs).forces_method2
    _orig_recompute  = type(fs)._recompute_mu_normals

    _precompile_count = [0]
    _precompile_done  = [args.precompile <= 0]
    _settle_count     = [0]
    _settle_done      = [False]
    _deep_patches_installed = [False]

    def detailed_step(self, task, physics):
        # Pre-compilation phase
        if not _precompile_done[0]:
            _precompile_count[0] += 1
            _orig_step(self, task, physics)
            if _precompile_count[0] >= args.precompile:
                _precompile_done[0] = True
                if USE_CUDA:
                    torch.cuda.synchronize()
                _install_deep_patches()
                print(f"\n  [profiler] Pre-compilation complete "
                      f"({_precompile_count[0]} steps).  "
                      f"Settling physics for {args.settle_steps} more steps…\n",
                      flush=True)
            return

        # Settle phase (timer inactive)
        if not _settle_done[0]:
            _settle_count[0] += 1
            _orig_step(self, task, physics)
            if _settle_count[0] >= args.settle_steps:
                _settle_done[0] = True
                if USE_CUDA:
                    torch.cuda.synchronize()
                T._active = True
                print(f"  [profiler] Settling complete "
                      f"({_settle_count[0]} steps).  "
                      f"Measuring {args.n_steps} steps…\n", flush=True)
            return

        # Timed phase
        T.new_step()
        iteration = self.iteration
        timestep  = self.pars["solver"]["dt"]
        if iteration >= self.pars["solver"]["nt"]:
            self.iteration += 1
            return

        t   = iteration * timestep
        ffs = self.fluid_solver

        if not ffs.terminate:

            # ── 1. SDF update ────────────────────────────────────
            with T("1  SDF update (body kinematics + SDF eval)"):
                self.update(t, iteration, dt=timestep)

            # ── 2. mu + normals ───────────────────────────────────
            with T("2  mu+normals (recompute)"):
                _orig_recompute(ffs)

            # ── 3. Fluid step (2-D PDE) ───────────────────────────
            with T("3  fluid_step (total PDE)"):
                (u, v, p) = _orig_fluid_step(ffs,
                    ffs.u0, ffs.v0, ffs.p0, timestep)

            (ffs.u0, ffs.v0, ffs.p0) = (u, v, p)

            # ── 4. Forces ─────────────────────────────────────────
            with T("4  forces_method2"):
                _orig_forces(ffs, ffs.u0, ffs.v0, ffs.p0, iteration)

            # ── Free cached force-density tensors ─────────────────
            from lilytorch.integration.BDIMhandler import _FS_FREE_AFTER_FORCES_3D
            ffs.__dict__.update(_FS_FREE_AFTER_FORCES_3D)

            # ── 5. Plotting / saving — SKIPPED for pure solver cost
            ffs.terminate = False

            # ── 6. Apply forces (FARMS) ───────────────────────────
            with T("6  apply_forces (FARMS)"):
                _orig_apply(self, task, physics)

            # ── 7. Free BDIM intermediates ────────────────────────
            ffs._release_bdim_fields()

        self.iteration += 1

    def outer_step(self_h, task, physics):
        if (not _precompile_done[0]) or (not _settle_done[0]):
            detailed_step(self_h, task, physics)
        else:
            with T("TOTAL step"):
                detailed_step(self_h, task, physics)
    handler.step = types.MethodType(outer_step, handler)

    # References for deep patches (deferred until after pre-compilation)
    _orig_adv_solve_bound = adv.solve
    _orig_bdim_meta    = fs._bdim_meta_compiled
    _orig_project      = type(fs).project
    _orig_vardens      = type(fs)._compute_variable_density_coefficients

    poisson_mg = getattr(fs, "poisson_solver", None)

    def _install_deep_patches():
        if _deep_patches_installed[0]:
            return
        _deep_patches_installed[0] = True

        # ── 1b. Timed SDF eval sub-step ──────────────────────────
        def timed_update_detailed(self_h, t, iteration, dt=1):
            with T("1b   SDF eval (per-body × 3 grids)"):
                return _orig_update(t, iteration, dt)
        handler.update = types.MethodType(timed_update_detailed, handler)

        # Safety: Smagorinsky must be off for a clean cost baseline.
        if getattr(fs, 'use_smagorinsky', False):
            raise RuntimeError(
                "Smagorinsky LES is active but cost analysis is configured "
                "to exclude it.  Set cfg.smagorinsky_cs = 0.0."
            )

        # ── Wrap adv_diff_solver.solve ───────────────────────────
        _saved_adv_solve = adv.solve
        def timed_adv_solve(*a, **kw):
            with T("3a   advection+diffusion"):
                return _saved_adv_solve(*a, **kw)
        adv.solve = timed_adv_solve

        # ── Wrap _bdim_meta_compiled ──────────────────────────────
        _saved_bdim = fs._bdim_meta_compiled
        def timed_bdim(*a, **kw):
            with T("3b   BDIM meta-equation"):
                return _saved_bdim(*a, **kw)
        fs._bdim_meta_compiled = timed_bdim

        if hasattr(fs, '_bdim_meta_dyn_compiled'):
            _saved_bdim_dyn = fs._bdim_meta_dyn_compiled
            def timed_bdim_dyn(*a, **kw):
                with T("3b   BDIM meta-equation"):
                    return _saved_bdim_dyn(*a, **kw)
            fs._bdim_meta_dyn_compiled = timed_bdim_dyn

        # ── Wrap fs.project ───────────────────────────────────────
        def timed_project(self_fs, *a, **kw):
            with T("3c   projection (Poisson+gradient+correction)"):
                return _orig_project(self_fs, *a, **kw)
        fs.project = types.MethodType(timed_project, fs)

        # ── Wrap set_BCs ──────────────────────────────────────────
        _orig_set_bcs_fn = type(adv).set_BCs
        def timed_set_bcs(self_a, *a, **kw):
            with T("3d   set_BCs"):
                return _orig_set_bcs_fn(self_a, *a, **kw)
        adv.set_BCs = types.MethodType(timed_set_bcs, adv)

        # ── Wrap variable-density coefficients ────────────────────
        def timed_vardens(self_fs, *a, **kw):
            with T("3e   var-density coeffs [ch cv ch_cc]"):
                return _orig_vardens(self_fs, *a, **kw)
        fs._compute_variable_density_coefficients = types.MethodType(
            timed_vardens, fs)

        # ── Wrap _release_bdim_fields ─────────────────────────────
        _orig_release = type(fs)._release_bdim_fields
        def timed_release(self_fs, *a, **kw):
            with T("3f   release BDIM fields"):
                return _orig_release(self_fs, *a, **kw)
        fs._release_bdim_fields = types.MethodType(timed_release, fs)

        # ── Instrument Poisson sub-components (large grids only) ──
        _instrument_poisson_internals = (args.Nx * args.Ny) >= 500_000

        if poisson_mg is not None and _instrument_poisson_internals:
            _orig_jacobi_fn = type(poisson_mg).Jacobi
            def timed_jacobi(self_p, *a, **kw):
                with T("3c.i   Jacobi smoothing"):
                    return _orig_jacobi_fn(self_p, *a, **kw)
            poisson_mg.Jacobi = types.MethodType(timed_jacobi, poisson_mg)

            _orig_vcycle_fn = type(poisson_mg)._vcycle
            _in_vcycle = [False]
            def timed_vcycle(self_p, *a, **kw):
                if _in_vcycle[0]:
                    return _orig_vcycle_fn(self_p, *a, **kw)
                _in_vcycle[0] = True
                try:
                    with T("3c.ii  V-cycle (top-level)"):
                        return _orig_vcycle_fn(self_p, *a, **kw)
                finally:
                    _in_vcycle[0] = False
            poisson_mg._vcycle = types.MethodType(timed_vcycle, poisson_mg)

        print(f"  [profiler] Deep patches installed (SDF, advection, BDIM, project"
              f"{', Poisson internals' if _instrument_poisson_internals and poisson_mg else ''})",
              flush=True)

    if _precompile_done[0]:
        _install_deep_patches()
        print(f"  [profiler] Pre-compilation complete (0 steps).  "
              f"Settling physics for {args.settle_steps} more steps…\n",
              flush=True)

    print(f"  [profiler] Instrumented handler (grid {fs.grid_shape}, "
          f"device {fs.device}, poisson={fs.poisson_method})", flush=True)
    print(f"  [profiler] Running {args.precompile} pre-compilation steps "
          f"(untimed)…", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Patch FluidExtension to instrument handler after creation
# ═══════════════════════════════════════════════════════════════════════

_orig_init_episode = FluidExtension.initialize_episode
_handler_ref = [None]

def _patched_init_episode(self, task, physics):
    _orig_init_episode(self, task, physics)
    if hasattr(self, "BDIMhandler") and self.BDIMhandler is not None:
        instrument_handler(self.BDIMhandler)
        _handler_ref[0] = self.BDIMhandler

FluidExtension.initialize_episode = _patched_init_episode


# ═══════════════════════════════════════════════════════════════════════
# Configure: pinned 2-D 1guilla (headless, fluid-only)
# ═══════════════════════════════════════════════════════════════════════

import lilytorch.farms_examples._1guillasim.gen_configs_one_pinned_2d as cfg_mod

cfg = cfg_mod.SimConfig()

cfg.Nx       = args.Nx
cfg.Ny       = args.Ny
cfg.use_bdim = True

# Disable Smagorinsky for a clean cost baseline.
cfg.smagorinsky_cs = 0.0

# Domain scaling: keep fish body well inside the domain.
# Reference: xmin=-0.9, xmax=1.5 (Lx=2.4 m) at Nx=1024 → dx_ref ≈ 0.00234 m.
# Fish body (pinned at x=-0.65, ~0.81 m long) needs Lx ≥ 1.0 m.
_dx_ref      = 2.4 / 1024
_MIN_LX_FISH = 1.0

if args.Lx_fixed is not None:
    _dx = args.Lx_fixed / args.Nx
    _Lx = args.Lx_fixed
    _Ly = args.Ny * _dx
    print(f"  [domain] FIXED-DOMAIN mode: Lx={_Lx:.4f} m  "
          f"dx={_dx:.6f} m  (Nx={args.Nx})")
else:
    _dx = max(_dx_ref, _MIN_LX_FISH / args.Nx)
    _Lx = args.Nx * _dx
    _Ly = args.Ny * _dx

# Centre x-domain on fish body midpoint; y-domain centred on y=0.
_x_body_center = -0.325   # midpoint of pinned fish body (spawn at x=-0.65, tail ~x=0)
cfg.xmin = _x_body_center - 0.5 * _Lx
cfg.xmax = _x_body_center + 0.5 * _Lx
cfg.ymin = -0.5 * _Ly
cfg.ymax =  0.5 * _Ly

if _dx > _dx_ref * 1.01:
    print(f"  [domain] Nx={args.Nx} too small for dx_ref → using dx={_dx:.6f} "
          f"({_dx/_dx_ref:.1f}× coarser) so fish fits in Lx={_Lx:.3f} m")
print(f"  [domain] x=[{cfg.xmin:.3f}, {cfg.xmax:.3f}]  "
      f"y=[{cfg.ymin:.3f}, {cfg.ymax:.3f}]  dx={_dx:.6f}")

cfg.n_iterations = args.precompile + args.settle_steps + args.n_steps + 1
cfg.save_every   = args.save_every
cfg.save_frames  = False
cfg.headless     = True
cfg.save         = False


# Override gen_simulation_config to strip non-fluid extensions
_orig_gen_sim_config = cfg.gen_simulation_config

def gen_simulation_config_lean(output_folder):
    """Keep only FluidExtension — strip FlowViewer2D, CameraRecording, etc."""
    _orig_gen_sim_config(output_folder)
    import yaml
    yaml_path = os.path.join(output_folder, "simulation_config.yaml")
    with open(yaml_path, "r") as f:
        sim_dict = yaml.safe_load(f)

    keep_loaders = {"lilytorch.integration.extensions.FluidExtension"}
    original_exts = sim_dict.get("extensions", [])
    sim_dict["extensions"] = [
        ext for ext in original_exts if ext.get("loader", "") in keep_loaders
    ]
    sim_dict.setdefault("runtime", {})["headless"] = True

    # Enable torch.compile + 2-D streaming path
    for ext in sim_dict["extensions"]:
        bdim_yaml = ext.get("config", {}).get("bdim_yaml", {})
        solver_cfg = bdim_yaml.get("solver", {})
        if solver_cfg:
            solver_cfg["compile_adv_diff"] = True
            solver_cfg["poisson_compile"]  = True
            if SOLVER_MODE is not None:
                solver_cfg["solver_method"] = SOLVER_MODE

    with open(yaml_path, "w") as f:
        yaml.dump(sim_dict, f, default_flow_style=False, sort_keys=False)

    stripped = len(original_exts) - len(sim_dict["extensions"])
    print(f"  [profiler] Stripped {stripped} extensions, kept FluidExtension only")
    print(f"  [profiler] Enabled torch.compile for adv-diff + Poisson + forces + SDF")

cfg.gen_simulation_config = gen_simulation_config_lean

grid_N = args.Nx * args.Ny
print("=" * 72)
print("  2-D Pinned 1guilla – Computational Cost Analysis")
print(f"  Grid:   {args.Nx} × {args.Ny}  ({grid_N:,} cells)")
print(f"  Steps:  {args.n_steps} measured  (+ {args.precompile} pre-compile, "
      f"+ {args.settle_steps} settle)")
print(f"  Device: {'CUDA' if USE_CUDA else 'CPU'}")
print("=" * 72)

# ── torch._dynamo recompile log → per-grid file ─────────────────
os.makedirs(args.out_dir, exist_ok=True)
_recompile_log_path = os.path.join(
    args.out_dir, f"recompiles_{args.Nx}x{args.Ny}.log")
open(_recompile_log_path, "w").close()
_rc_handler = logging.FileHandler(_recompile_log_path, mode="a")
_rc_handler.setLevel(logging.INFO)
_rc_handler.setFormatter(
    logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
_TORCH_DYNAMO_LOG.addHandler(_rc_handler)
print(f"  Recompile log: {_recompile_log_path}")


# ═══════════════════════════════════════════════════════════════════════
# Run FARMS in-process
# ═══════════════════════════════════════════════════════════════════════

from lilytorch.util.paths import gen_new_folder, save_path
output_folder = gen_new_folder(save_path)
os.makedirs(output_folder, exist_ok=True)
print(f"  Config folder: {output_folder}")

cfg.gen_animat_config(output_folder)
cfg.gen_arena_config(output_folder)
cfg.gen_simulation_config(output_folder)
cfg.gen_experiment_config(output_folder)
os.chdir(output_folder)

from farms_core.experiment.options import ExperimentOptions
from farms_core.extensions.extensions import import_item
from farms_sim.simulation import run_simulation
from farms_core.simulation.options import Simulator

experiment_options = ExperimentOptions.load("experiment_config.yaml")
experiment_data_loader = import_item(experiment_options.loaders.experiment_data)
experiment_data = experiment_data_loader.from_options(experiment_options)

print("  Starting simulation…\n")
sim = run_simulation(
    experiment_data=experiment_data,
    experiment_options=experiment_options,
    simulator=Simulator.MUJOCO,
)
del sim
gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════

os.chdir(SCRIPT_DIR)
os.makedirs(args.out_dir, exist_ok=True)

summary = T.summary(discard_first=args.discard_first)
n_meas = T._step_count
n_used = n_meas - args.discard_first
if args.discard_first > 0:
    print(f"\n  [profiler] Discarded first {args.discard_first} timed steps "
          f"→ using {n_used} of {n_meas} measured steps for statistics")

if not summary:
    print(f"\nERROR: No measurements (step_count={T._step_count}, "
          f"precompile={args.precompile})")
    sys.exit(1)

step_median = summary.get("TOTAL step", {}).get("median", 0)

print("\n" + "=" * 108)
print(f"  TIMING RESULTS  ({n_meas} measured steps, "
      f"grid {args.Nx}×{args.Ny} = {grid_N:,} cells, "
      f"{'GPU' if USE_CUDA else 'CPU'})")
print("=" * 108)
hdr = (f"  {'Component':<44s} {'Median':>9s} {'Std':>9s} {'Min':>9s} "
       f"{'Max':>9s} {'Total':>9s} {'%step':>7s} {'Calls':>6s}")
print(hdr)
print(f"  {'':44s} {'(ms)':>9s} {'(ms)':>9s} {'(ms)':>9s} "
      f"{'(ms)':>9s} {'(s)':>9s} {'':>7s}")
print("-" * 108)

for label in sorted(summary.keys(), key=lambda k: -summary[k]["total"]):
    s = summary[label]
    print(f"  {label:<44s} {1e3*s['median']:9.2f} {1e3*s['std']:9.2f} "
          f"{1e3*s['min']:9.2f} {1e3*s['max']:9.2f} "
          f"{s['total']:9.4f} {s['pct']:7.1f} {s['count']:6d}")

print("=" * 108)
if step_median > 0:
    print(f"\n  Median step time: {1e3*step_median:.2f} ms  "
          f"({1.0/step_median:.1f} steps/s)")
    print(f"  Grid cells: {grid_N:,}  "
          f"({1e6*step_median/grid_N:.3f} µs/cell/step)")

# ── Stability diagnostic ─────────────────────────────────────────────
_total_times = np.array(T._data.get("TOTAL step", [])[args.discard_first:])
if len(_total_times) >= 10:
    tail = _total_times[-10:]
    cv = tail.std() / tail.mean() if tail.mean() > 0 else 0.0
    status = "OK" if cv < args.stability_tol else "UNSTABLE"
    print(f"  Stability check (last 10 timed steps): CV = {cv*100:.1f}%  "
          f"[{status}, tol = {args.stability_tol*100:.0f}%]")

# ── Recompile summary ────────────────────────────────────────────────
_rc_handler.flush()
try:
    with open(_recompile_log_path) as _rcf:
        _rc_text = _rcf.read()
    _n_recompiles = sum(1 for ln in _rc_text.splitlines()
                        if "recompiling" in ln.lower()
                        or "Recompiling" in ln
                        or "cache_size_limit" in ln)
    _n_breaks = sum(1 for ln in _rc_text.splitlines()
                    if "graph break" in ln.lower())
    print(f"  torch.compile: {_n_recompiles} recompile event(s), "
          f"{_n_breaks} graph break(s)  → see {_recompile_log_path}")
except OSError:
    pass


# ── Save CSV ─────────────────────────────────────────────────────────
tag = f"{args.Nx}x{args.Ny}{args.tag_suffix}"
csv_path = os.path.join(args.out_dir, f"cost_breakdown_{tag}.csv")
with open(csv_path, "w") as f:
    f.write("component,median_ms,mean_ms,std_ms,min_ms,max_ms,total_s,pct_of_step,calls\n")
    for label in sorted(summary.keys(), key=lambda k: -summary[k]["total"]):
        s = summary[label]
        f.write(f"{label},{1e3*s['median']:.4f},{1e3*s['mean']:.4f},{1e3*s['std']:.4f},"
                f"{1e3*s['min']:.4f},{1e3*s['max']:.4f},"
                f"{s['total']:.6f},{s['pct']:.2f},{s['count']}\n")
print(f"\n  CSV saved → {csv_path}")

# ── Save per-step raw timings ─────────────────────────────────────────
raw_path = os.path.join(args.out_dir, f"cost_perstep_{tag}.csv")
raw_labels = ["TOTAL step"] + sorted(
    k for k in T._data.keys() if k != "TOTAL step")
with open(raw_path, "w") as f:
    f.write("step,used," + ",".join(raw_labels) + "\n")
    n_steps = len(T._data.get("TOTAL step", []))
    for i in range(n_steps):
        used = "yes" if i >= args.discard_first else "discarded"
        vals = []
        for lb in raw_labels:
            times = T._data.get(lb, [])
            n_calls = len(times)
            if n_calls == 0:
                vals.append("")
            elif n_calls == n_steps:
                vals.append(f"{1e3*times[i]:.4f}")
            else:
                start = (i * n_calls) // n_steps
                end   = ((i + 1) * n_calls) // n_steps
                vals.append(f"{1e3 * sum(times[start:end]):.4f}" if start < n_calls else "")
        f.write(f"{i},{used},{','.join(vals)}\n")
print(f"  Per-step CSV saved → {raw_path}")


# ═══════════════════════════════════════════════════════════════════════
# Paper-quality figures
# ═══════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
})

# ── Aggregate into paper categories ──────────────────────────────────
# "3c " (trailing space) matches "3c   projection" but NOT "3c.i Jacobi"
# or "3c.ii V-cycle" sub-timers (avoids double-counting on large grids).
CATEGORIES = {
    "Body update\n(SDF eval)":      ["1b"],
    "mu + normals":                 ["2 "],
    "Convection\n& diffusion":      ["3a  "],
    "BDIM\nmeta-equation":          ["3b"],
    "Projection\n(pressure)":       ["3c "],
    "set_BCs (2d)":                 ["3d"],
    "var-density\ncoeffs (3e)":     ["3e"],
    "release BDIM\nfields (3f)":    ["3f"],
    "Forces\ncomputation":          ["4 "],
    "FARMS\n(apply_forces)":        ["6 "],
}
_OTHER_LABEL = "Other\n(residual)"

CAT_COLOURS = {
    "Body update\n(SDF eval)":      "#26a69a",
    "mu + normals":                 "#66bb6a",
    "Convection\n& diffusion":      "#42a5f5",
    "BDIM\nmeta-equation":          "#ab47bc",
    "Projection\n(pressure)":       "#ef5350",
    "set_BCs (2d)":                 "#fbc02d",
    "var-density\ncoeffs (3e)":     "#7cb342",
    "release BDIM\nfields (3f)":    "#bcaaa4",
    "Forces\ncomputation":          "#ffa726",
    "FARMS\n(apply_forces)":        "#5c6bc0",
    _OTHER_LABEL:                   "#90a4ae",
}

cat_means = {}
cat_pcts  = {}

def _trimmed(lbl):
    times = T._data.get(lbl, [])
    return np.array(times[args.discard_first:]) if args.discard_first < len(times) else np.array(times)

_total_series = _trimmed("TOTAL step")
outer_total_s = float(_total_series.sum())
explicit_per_step = np.zeros_like(_total_series)

for cat_name, prefixes in CATEGORIES.items():
    cat_series = np.zeros_like(_total_series)
    for l in summary:
        if l == "TOTAL step":
            continue
        if any(l.startswith(pfx) for pfx in prefixes):
            s = _trimmed(l)
            if len(s) == len(_total_series):
                cat_series += s
            else:
                cat_series += s.sum() / max(len(_total_series), 1)
    explicit_per_step += cat_series
    if cat_series.sum() > 0:
        med_ms = 1e3 * float(np.median(cat_series))
        pct    = 100.0 * cat_series.sum() / outer_total_s if outer_total_s > 0 else 0.0
        cat_means[cat_name] = med_ms
        cat_pcts[cat_name]  = pct

residual_series = np.clip(_total_series - explicit_per_step, 0.0, None)
if residual_series.sum() > 0:
    cat_means[_OTHER_LABEL] = 1e3 * float(np.median(residual_series))
    cat_pcts[_OTHER_LABEL]  = (
        100.0 * residual_series.sum() / outer_total_s if outer_total_s > 0 else 0.0
    )
explicit_total_s = float(explicit_per_step.sum())
residual_s       = float(residual_series.sum())

print("\n  ── Grouped categories ──")
_ordered_cats = list(CATEGORIES.keys()) + [_OTHER_LABEL]
for cn in _ordered_cats:
    if cn in cat_means:
        label = cn.replace("\n", " ")
        print(f"    {label:<32s}  {cat_means[cn]:8.2f} ms  ({cat_pcts[cn]:5.1f}%)")
if outer_total_s > 0:
    _coverage = 100.0 * (explicit_total_s + residual_s) / outer_total_s
    print(f"    {'(coverage check: Σ categories / TOTAL step)':<32s}  "
          f"{_coverage:7.2f}%")

# ── Figure 1: Horizontal bar chart ──────────────────────────────────
cat_names_sorted = sorted(cat_means.keys(), key=lambda k: cat_means[k])
fig, ax = plt.subplots(figsize=(5.5, 3.0))
bars = ax.barh(
    range(len(cat_names_sorted)),
    [cat_means[c] for c in cat_names_sorted],
    color=[CAT_COLOURS.get(c, "#90a4ae") for c in cat_names_sorted],
    edgecolor="white", linewidth=0.6, height=0.7,
)
ax.set_yticks(range(len(cat_names_sorted)))
ax.set_yticklabels(cat_names_sorted)
ax.set_xlabel("Time per step (ms)")
ax.set_title(f"Cost breakdown – {args.Nx}×{args.Ny} "
             f"({grid_N:,} cells, {'GPU' if USE_CUDA else 'CPU'})")

for bar, cn in zip(bars, cat_names_sorted):
    w = bar.get_width()
    pct = cat_pcts[cn]
    ax.text(w + 0.02 * max(cat_means.values()), bar.get_y() + bar.get_height() / 2,
            f"{w:.1f} ms ({pct:.0f}%)", va="center", fontsize=8)

ax.set_xlim(0, max(cat_means.values()) * 1.35)
fig.tight_layout()
bar_path = os.path.join(args.out_dir, f"cost_barh_{tag}.pdf")
fig.savefig(bar_path)
fig.savefig(bar_path.replace(".pdf", ".png"))
print(f"  Figure saved → {bar_path}")
plt.close(fig)


# ── Figure 2: Pie chart ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(5, 4))
pie_names = list(cat_pcts.keys())
pie_vals  = [cat_pcts[c] for c in pie_names]
pie_cols  = [CAT_COLOURS.get(c, "#90a4ae") for c in pie_names]
pie_display = [c.replace("\n", " ") for c in pie_names]
wedges, texts, autotexts = ax2.pie(
    pie_vals, labels=pie_display, autopct="%1.1f%%",
    startangle=140, pctdistance=0.78, textprops={"fontsize": 8},
    colors=pie_cols,
    wedgeprops=dict(edgecolor="white", linewidth=1.0),
)
for at in autotexts:
    at.set_fontsize(7)
ax2.set_title(f"Step cost distribution – {args.Nx}×{args.Ny}")
fig2.tight_layout()
pie_path = os.path.join(args.out_dir, f"cost_pie_{tag}.pdf")
fig2.savefig(pie_path)
fig2.savefig(pie_path.replace(".pdf", ".png"))
print(f"  Figure saved → {pie_path}")
plt.close(fig2)


# ── Figure 3: Detailed breakdown (all sub-timers) ───────────────────
detail_labels = [l for l in summary if l != "TOTAL step"]
detail_labels.sort(key=lambda k: -summary[k]["pct"])

if detail_labels:
    fig3, ax3 = plt.subplots(figsize=(6, max(3.5, 0.35 * len(detail_labels))))
    detail_pcts  = [summary[l]["pct"] for l in detail_labels]
    detail_means = [1e3 * summary[l]["mean"] for l in detail_labels]

    def _colour(label):
        if "SDF" in label or "sdf" in label:         return "#26a69a"
        if "mu_funcs" in label:                       return "#66bb6a"
        if "normals" in label:                        return "#a5d6a7"
        if "advection" in label:                      return "#42a5f5"
        if "BDIM" in label:                           return "#29b6f6"
        if "Poisson" in label or "poisson" in label:  return "#ef5350"
        if "Jacobi" in label or "V-cycle" in label:   return "#ef9a9a"
        if "divergence" in label:                     return "#ab47bc"
        if "gradient" in label:                       return "#ce93d8"
        if "projection" in label:                     return "#e1bee7"
        if "forces" in label:                         return "#ffa726"
        if "BCs" in label:                            return "#78909c"
        if "apply" in label or "FARMS" in label:      return "#5c6bc0"
        return "#bdbdbd"

    dc = [_colour(l) for l in detail_labels]
    bars3 = ax3.barh(range(len(detail_labels)), detail_pcts, color=dc,
                     edgecolor="white", linewidth=0.4, height=0.7)
    ax3.set_yticks(range(len(detail_labels)))
    ax3.set_yticklabels(detail_labels, fontsize=7)
    ax3.set_xlabel("% of total step time")
    ax3.set_title(f"Detailed breakdown – {args.Nx}×{args.Ny}")
    ax3.invert_yaxis()

    for bar, m, p in zip(bars3, detail_means, detail_pcts):
        ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{m:.2f} ms ({p:.1f}%)", va="center", fontsize=6.5)
    ax3.set_xlim(0, max(detail_pcts) * 1.4)
    fig3.tight_layout()
    detail_path = os.path.join(args.out_dir, f"cost_detailed_{tag}.pdf")
    fig3.savefig(detail_path)
    fig3.savefig(detail_path.replace(".pdf", ".png"))
    print(f"  Figure saved → {detail_path}")
    plt.close(fig3)


# ── Figure 4: Flow field snapshot (pressure, velocity, vorticity) ────
_h = _handler_ref[0]
if _h is not None:
    fs   = _h.fluid_solver
    comp = fs.composite_body

    p   = fs.p0.detach().cpu().float().numpy()[1:-1, 1:-1]
    u   = fs.u0.detach().cpu().float().numpy()[1:-1, 1:-1]
    v   = fs.v0.detach().cpu().float().numpy()[1:-1, 1:-1]
    sdf = comp.sdf_val.detach().cpu().float().numpy()[1:-1, 1:-1]

    vmag = np.sqrt(u**2 + v**2)

    dx = (cfg.xmax - cfg.xmin) / args.Nx
    dvdx = np.gradient(v, dx, axis=0)
    dudy = np.gradient(u, dx, axis=1)
    omega_z = dvdx - dudy

    x_1d = np.linspace(cfg.xmin, cfg.xmax, p.shape[0])
    y_1d = np.linspace(cfg.ymin, cfg.ymax, p.shape[1])

    fig4, axes = plt.subplots(2, 2, figsize=(10, 5), constrained_layout=True)
    fig4.suptitle(
        f"Flow fields – {args.Nx}×{args.Ny}  ({grid_N:,} cells, "
        f"step {args.precompile + args.n_steps})",
        fontsize=12, fontweight="bold",
    )

    def _format_ax(ax, title, data, cmap, symmetric=False):
        if symmetric:
            vmax = max(abs(np.nanpercentile(data, 2)),
                       abs(np.nanpercentile(data, 98)))
            vmin = -vmax
        else:
            vmin = np.nanpercentile(data, 2)
            vmax = np.nanpercentile(data, 98)
        im = ax.imshow(
            data.T, origin="lower", aspect="auto", cmap=cmap,
            extent=[x_1d[0], x_1d[-1], y_1d[0], y_1d[-1]],
            vmin=vmin, vmax=vmax,
        )
        ax.contour(x_1d, y_1d, sdf.T, levels=[0], colors="k",
                   linewidths=0.8, linestyles="-")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    _format_ax(axes[0, 0], "Pressure $p$",             p,       "RdBu_r", symmetric=True)
    _format_ax(axes[0, 1], "Velocity magnitude $|u|$", vmag,    "viridis")
    _format_ax(axes[1, 0], "Vorticity $\\omega_z$",    omega_z, "RdBu_r", symmetric=True)

    _FAR_SENTINEL = 1e3
    # Mask only the _FAR sentinel — show the full union AABB extent.
    # Grey = outside all body AABBs (_FAR); colour = within union AABB.
    sdf_plot = np.ma.masked_where(np.abs(sdf) >= _FAR_SENTINEL, sdf)
    non_far = sdf[np.abs(sdf) < _FAR_SENTINEL]
    _vabs = float(np.percentile(np.abs(non_far), 98)) if non_far.size else 1.0
    cmap_sdf = plt.get_cmap("coolwarm").copy()
    cmap_sdf.set_bad(color="#dddddd")  # grey = _FAR / outside AABB
    im_sdf = axes[1, 1].imshow(
        sdf_plot.T, origin="lower", aspect="auto", cmap=cmap_sdf,
        extent=[x_1d[0], x_1d[-1], y_1d[0], y_1d[-1]],
        vmin=-_vabs, vmax=_vabs,
    )
    sdf_for_contour = np.where(np.abs(sdf) >= _FAR_SENTINEL, np.nan, sdf)
    axes[1, 1].contour(x_1d, y_1d, sdf_for_contour.T, levels=[0], colors="k",
                       linewidths=0.8, linestyles="-")
    axes[1, 1].set_title(f"SDF union (grey=_FAR, ±{_vabs*1e3:.1f} mm p98)", fontsize=10)
    axes[1, 1].set_xlabel("x (m)", fontsize=8)
    axes[1, 1].set_ylabel("y (m)", fontsize=8)
    axes[1, 1].tick_params(labelsize=7)
    plt.colorbar(im_sdf, ax=axes[1, 1], shrink=0.85, pad=0.02)

    field_path = os.path.join(args.out_dir, f"flow_fields_{tag}.pdf")
    fig4.savefig(field_path)
    fig4.savefig(field_path.replace(".pdf", ".png"))
    print(f"  Figure saved → {field_path}")
    plt.close(fig4)
else:
    print("  [warning] Could not access handler — skipping flow field plots.")


print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  Run multi-grid scaling analysis:                                  ║
║                                                                    ║
║  python run_cost_analysis.py --Nx  128 --Ny  32                   ║
║  python run_cost_analysis.py --Nx  256 --Ny  64                   ║
║  python run_cost_analysis.py --Nx  512 --Ny 128                   ║
║  python run_cost_analysis.py --Nx 1024 --Ny 256                   ║
║                                                                    ║
║  Then run:  python plot_scaling.py                                 ║
║  to produce the multi-resolution comparison figure.                ║
╚══════════════════════════════════════════════════════════════════════╝
""")
