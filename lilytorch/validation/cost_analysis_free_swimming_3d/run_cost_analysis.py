#!/usr/bin/env python3
"""
Computational-cost analysis for the 3-D free-swimming 1guilla simulation.

Instruments every major sub-kernel inside BDIMhandler3D with CUDA-synchronised
timers.  Reports per-step statistics, saves CSV and **paper-quality** figures.

Operation groups (user-specified + extras):
  1. FARMS step         – MuJoCo physics + apply_forces (CPU ↔ GPU transfer)
  2. Body update        – SDF evaluation on 4 staggered grids
  3. Body interp. & normals – Heaviside (mu) + normal computation
  4. Convection–diffusion   – advection + diffusion + BCs
  5. Projection         – Poisson solve + gradient + velocity correction
  6. BDIM meta-equation – immersed-boundary forcing (3 components)
  7. Forces             – forces_method2_3d + plotting/saving

KEY DESIGN CHOICES:
  - Runs the FARMS simulation **in-process** so Python-level monkey-patches
    apply to every hot path.
  - All non-fluid extensions (FlowViewer, CameraRecording, ExperimentLogger,
    MjcfSaver, DataLogger) are stripped for pure solver cost.
  - Headless mode.

Usage
-----
    source /path/to/venv/bin/activate
    python run_cost_analysis.py                                      # default 128×32×32, 20 steps
    python run_cost_analysis.py --Nx 256 --Ny 64 --Nz 64 --n_steps 50
    python run_cost_analysis.py --Nx 512 --Ny 128 --Nz 128 --n_steps 30
"""

import argparse
import os
import sys
import time
import types
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="3-D free-swimming BDIM cost analysis (paper figures)")
parser.add_argument("--Nx",        type=int, default=128)
parser.add_argument("--Ny",        type=int, default=32)
parser.add_argument("--Nz",        type=int, default=32)
parser.add_argument("--n_steps",   type=int, default=50,
                    help="Measured steps (all are timed — compilation is "
                         "excluded via a separate pre-compilation phase)")
parser.add_argument("--precompile", type=int, default=30,
                    help="Pre-compilation steps (run before any timing to "
                         "trigger all torch.compile CUDA-graph captures; "
                         "needs ≥ 3× the number of compiled functions)")
parser.add_argument("--discard_first", type=int, default=3,
                    help="Discard the first N timed steps from summary "
                         "stats (avoids CUDA-graph settling overhead after "
                         "deep patches are installed)")
parser.add_argument("--save_every",type=int, default=9999)
parser.add_argument("--device",    type=str, default="cuda",
                    choices=["cuda", "cpu"])
parser.add_argument("--out_dir",   type=str, default=None,
                    help="Output directory (default: figures/ in this folder)")
args = parser.parse_args()

USE_CUDA = args.device == "cuda" and torch.cuda.is_available()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if args.out_dir is None:
    args.out_dir = os.path.join(SCRIPT_DIR, "figures")


# ═══════════════════════════════════════════════════════════════════════
# Timer infrastructure
# ═══════════════════════════════════════════════════════════════════════

class TimerBank:
    """CUDA-synchronised timer collection.  No warmup — all steps timed."""

    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda
        self._data: dict[str, list[float]] = defaultdict(list)
        self._step_count = 0

    @contextmanager
    def __call__(self, label: str):
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
        """Return per-label stats, optionally discarding the first N entries
        (to exclude CUDA-graph settling overhead after deep patches)."""
        trimmed = {}
        for label, times in self._data.items():
            trimmed[label] = times[discard_first:] if discard_first < len(times) else times
        outer_total = sum(trimmed.get("TOTAL step", [1.0]))
        rows = {}
        for label, times in sorted(trimmed.items()):
            arr = np.array(times)
            rows[label] = {
                "mean":  arr.mean(),
                "std":   arr.std(),
                "min":   arr.min(),
                "max":   arr.max(),
                "total": arr.sum(),
                "count": len(arr),
                "pct":   100.0 * arr.sum() / outer_total if outer_total > 0 else 0,
            }
        return rows

T = TimerBank(USE_CUDA)


# ═══════════════════════════════════════════════════════════════════════
# Project imports
# ═══════════════════════════════════════════════════════════════════════
# Ensure the repo root is on the path
repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from lilytorch.integration.extensions import FluidExtension
from lilytorch.integration.BDIMhandler import BDIMhandler  # noqa: F401
from lilytorch.src.poisson_mult import PoissonSolver  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════
# Deep instrumentation (instance-level monkey-patching)
# ═══════════════════════════════════════════════════════════════════════

def instrument_handler(handler):
    """Wrap every hot-path method on live instances with fine-grained timers.

    DESIGN PRINCIPLE: we NEVER reimplement production code paths.  Instead
    we wrap the *existing* compiled methods with timing context managers.
    This ensures:
      • torch.compile / CUDA-graph recordings from precompile are REUSED
        during the timed phase (same code path, just with sync+timer around it).
      • Results are identical to production (no divergent BDIM arithmetic).
      • Sync overhead is minimised (≤ 7 timer blocks per step).
    """
    if getattr(handler, "_profiled", False):
        return
    handler._profiled = True

    fs   = handler.fluid_solver
    comp = fs.composite_body
    adv  = fs.adv_diff_solver

    # Keep originals (unbound class methods)
    _orig_step       = type(handler).step
    _orig_update     = type(handler).update
    _orig_fluid_step = type(handler).fluid_step
    _orig_apply      = type(handler).apply_forces
    _orig_forces     = type(fs).forces_method2_3d
    _orig_plot       = type(fs).plotting_and_saving
    _orig_recompute  = type(fs)._recompute_mu_normals_3d

    # ── Pre-compilation gate ─────────────────────────────────────
    _precompile_count = [0]
    _precompile_done  = [False]

    # ── Replacement for handler.step ─────────────────────────────
    def detailed_step(self, task, physics):
        # ── Pre-compilation phase: run untimed original steps ────
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
                      f"Measuring {args.n_steps} steps…\n", flush=True)
            return

        # ── Timed phase: every step from here is measured ────────
        T.new_step()
        iteration = self.iteration
        timestep  = self.pars["solver"]["dt"]
        if iteration >= self.pars["solver"]["nt"]:
            self.iteration += 1
            return

        t   = iteration * timestep
        ffs = self.fluid_solver

        if not self.terminate:

            # ── 1. SDF update ────────────────────────────────────
            with T("1  SDF update (body kinematics + SDF eval)"):
                self.update(t, iteration, dt=timestep)

            # ── 2. mu + normals (production path) ────────────────
            with T("2  mu+normals (recompute)"):
                _orig_recompute(ffs)

            # ── 3. Fluid step (production _fluid_step_3d) ────────
            # Sub-step timers are installed as wrappers around the
            # production methods (adv.solve, _bdim_meta_compiled,
            # fs.project) — see _install_deep_patches().
            with T("3  fluid_step (total PDE)"):
                (u, v, w, p) = _orig_fluid_step(self,
                    ffs.u0, ffs.v0, ffs.w0, ffs.p0, timestep)

            (ffs.u0, ffs.v0, ffs.w0, ffs.p0) = (u, v, w, p)

            # ── 4. Forces ────────────────────────────────────────
            with T("4  forces_method2_3d"):
                _orig_forces(ffs, ffs.u0, ffs.v0, ffs.w0, ffs.p0, iteration)

            # ── Free cached force-density tensors ────────────────
            for _attr in ('xstress_tensor', 'ystress_tensor', 'zstress_tensor',
                          'pforce_x', 'pforce_y', 'pforce_z'):
                if hasattr(ffs, _attr):
                    setattr(ffs, _attr, None)

            # ── 5. Plotting / saving ─────────────────────────────
            with T("5  plotting_and_saving"):
                self.terminate = _orig_plot(
                    ffs, ffs.u0, ffs.v0, ffs.p0, iteration, w_vel=ffs.w0,
                    check_termination=False)

            # ── 6. Apply forces (FARMS ← GPU) ───────────────────
            with T("6  apply_forces (FARMS)"):
                _orig_apply(self, task, physics)

            # ── 7. Free BDIM intermediates ───────────────────────
            ffs._release_bdim_fields()

        self.iteration += 1

    def outer_step(self_h, task, physics):
        if not _precompile_done[0]:
            detailed_step(self_h, task, physics)   # untimed precompile
        else:
            with T("TOTAL step"):
                detailed_step(self_h, task, physics)
    handler.step = types.MethodType(outer_step, handler)

    # ── Deep patches (deferred until after pre-compilation) ──────
    # These wrap production methods with timers.  Because we wrap
    # rather than reimplement, the compiled CUDA-graph code path
    # from precompile is preserved.* Important: the patches MUST
    # NOT change the function signature or break compiled wrappers.

    _orig_update_3d = type(handler)._update_3d   # unbound method

    # We'll need references to the actual bound/compiled methods
    # for sub-step wrapping inside fluid_step.
    _orig_adv_solve_bound = adv.solve   # possibly compiled
    _orig_set_bcs = type(adv).set_BCs
    _orig_bdim_meta = fs._bdim_meta_compiled  # compiled or eager
    _orig_project = type(fs).project

    poisson_mg  = getattr(fs, "poisson_solver", None)

    def _install_deep_patches():
        """Install timing wrappers around production methods after precompile.

        The wrappers call the ORIGINAL (possibly compiled) methods, so
        CUDA-graph recordings from precompile are replayed correctly.
        """

        # ── 1b. Timed SDF update sub-step ────────────────────────
        def timed_update_detailed(self_h, t, iteration, dt=1):
            with T("1b   SDF eval (per-body × 4 grids)"):
                _orig_update_3d(self_h, t, iteration, dt)
        handler.update = types.MethodType(timed_update_detailed, handler)

        # ── Wrap adv_diff_solver.solve ───────────────────────────
        _saved_adv_solve = adv.solve  # the (compiled) bound method

        def timed_adv_solve(*a, **kw):
            with T("3a   advection+diffusion"):
                return _saved_adv_solve(*a, **kw)
        adv.solve = timed_adv_solve

        # ── Wrap _bdim_meta_compiled ─────────────────────────────
        _saved_bdim = fs._bdim_meta_compiled

        def timed_bdim(*a, **kw):
            with T("3b   BDIM meta-equation"):
                return _saved_bdim(*a, **kw)
        fs._bdim_meta_compiled = timed_bdim

        # ── Wrap fs.project (contains Poisson + gradient + correction) ──
        def timed_project(self_fs, *a, **kw):
            with T("3c   projection (Poisson+gradient+correction)"):
                return _orig_project(self_fs, *a, **kw)
        fs.project = types.MethodType(timed_project, fs)

        # ── Instrument Poisson sub-components (large grids only) ─
        _instrument_poisson_internals = (args.Nx * args.Ny * args.Nz) >= 500_000

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

        print(f"  [profiler] Deep patches installed (SDF, advection, BDIM, "
              f"project{', Poisson internals' if _instrument_poisson_internals and poisson_mg else ''})",
              flush=True)

    print(f"  [profiler] Instrumented handler (grid {fs.grid_shape}, "
          f"device {fs.device}, poisson={fs.poisson_method})", flush=True)
    print(f"  [profiler] Running {args.precompile} pre-compilation steps "
          f"(untimed)…", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Patch FluidExtension to instrument handler after creation
# ═══════════════════════════════════════════════════════════════════════

_orig_init_episode = FluidExtension.initialize_episode

def _patched_init_episode(self, task, physics):
    _orig_init_episode(self, task, physics)
    if hasattr(self, "BDIMhandler") and self.BDIMhandler is not None:
        instrument_handler(self.BDIMhandler)

FluidExtension.initialize_episode = _patched_init_episode


# ═══════════════════════════════════════════════════════════════════════
# Configure: free-swimming 1guilla (headless, fluid-only)
# ═══════════════════════════════════════════════════════════════════════

import lilytorch.farms_examples._1guillasim.gen_configs_one_free_3d as cfg_mod

cfg = cfg_mod.SimConfig()

cfg.Nx           = args.Nx
cfg.Ny           = args.Ny
cfg.Nz           = args.Nz
cfg.use_bdim     = True

# Scale domain extents so that dx = dy = dz regardless of grid shape.
# The baseline domain is centred at (+0.3, 0, 0) with extents 2.4×0.6×0.6.
# For grids larger than production, EXTEND the domain to maintain dx.
# For grids ≤ production (4:1:1), keep the original domain (coarser dx).
_dx_ref = 2.4 / 512                   # reference spacing from production grid
_Lx = max(2.4, args.Nx * _dx_ref)     # extend if grid is wider
_Ly = max(0.6, args.Ny * _dx_ref)
_Lz = max(0.6, args.Nz * _dx_ref)
cfg.xmin = 0.3 - 0.5 * _Lx;  cfg.xmax = 0.3 + 0.5 * _Lx
cfg.ymin = -0.5 * _Ly;        cfg.ymax =  0.5 * _Ly
cfg.zmin = -0.5 * _Lz;        cfg.zmax =  0.5 * _Lz

cfg.n_iterations = args.precompile + args.n_steps + 1
cfg.save_every   = args.save_every
cfg.save_frames  = False
cfg.headless     = True
cfg.save         = False
cfg.save         = False

# Override gen_simulation_config to strip non-fluid extensions
_orig_gen_sim_config = cfg.gen_simulation_config

def gen_simulation_config_lean(output_folder):
    """Keep only FluidExtension — strip viewers, loggers, camera."""
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

    # Enable torch.compile for convection/diffusion + Poisson solver + forces
    # The compile flags live inside the FluidExtension's bdim_yaml.solver dict
    for ext in sim_dict["extensions"]:
        bdim_yaml = ext.get("config", {}).get("bdim_yaml", {})
        solver_cfg = bdim_yaml.get("solver", {})
        if solver_cfg:
            solver_cfg["compile_adv_diff"] = True
            solver_cfg["poisson_compile"]  = True
            solver_cfg["compile_forces"]   = True
            solver_cfg["compile_sdf"]      = True

    with open(yaml_path, "w") as f:
        yaml.dump(sim_dict, f, default_flow_style=False, sort_keys=False)

    stripped = len(original_exts) - len(sim_dict["extensions"])
    print(f"  [profiler] Stripped {stripped} extensions, kept FluidExtension only")
    print(f"  [profiler] Enabled torch.compile for adv-diff + Poisson + forces + SDF")

cfg.gen_simulation_config = gen_simulation_config_lean

grid_N = args.Nx * args.Ny * args.Nz
print("=" * 72)
print("  3-D Free-Swimming 1guilla – Computational Cost Analysis")
print(f"  Grid:   {args.Nx} × {args.Ny} × {args.Nz}  ({grid_N:,} cells)")
print(f"  Steps:  {args.n_steps} measured  (+ {args.precompile} pre-compilation)")
print(f"  Device: {'CUDA' if USE_CUDA else 'CPU'}")
print("=" * 72)


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


# ═══════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════

os.chdir(SCRIPT_DIR)
os.makedirs(args.out_dir, exist_ok=True)

summary = T.summary(discard_first=args.discard_first)
n_meas = T._step_count   # all steps are measured (precompile is separate)
n_used = n_meas - args.discard_first
if args.discard_first > 0:
    print(f"\n  [profiler] Discarded first {args.discard_first} timed steps "
          f"→ using {n_used} of {n_meas} measured steps for statistics")

if not summary:
    print(f"\nERROR: No measurements (step_count={T._step_count}, "
          f"precompile={args.precompile})")
    sys.exit(1)

step_mean = summary.get("TOTAL step", {}).get("mean", 0)

print("\n" + "=" * 108)
print(f"  TIMING RESULTS  ({n_meas} measured steps, "
      f"grid {args.Nx}×{args.Ny}×{args.Nz} = {grid_N:,} cells, "
      f"{'GPU' if USE_CUDA else 'CPU'})")
print("=" * 108)
hdr = (f"  {'Component':<44s} {'Mean':>9s} {'Std':>9s} {'Min':>9s} "
       f"{'Max':>9s} {'Total':>9s} {'%step':>7s} {'Calls':>6s}")
print(hdr)
print(f"  {'':44s} {'(ms)':>9s} {'(ms)':>9s} {'(ms)':>9s} "
      f"{'(ms)':>9s} {'(s)':>9s} {'':>7s}")
print("-" * 108)

for label in sorted(summary.keys(), key=lambda k: -summary[k]["total"]):
    s = summary[label]
    print(f"  {label:<44s} {1e3*s['mean']:9.2f} {1e3*s['std']:9.2f} "
          f"{1e3*s['min']:9.2f} {1e3*s['max']:9.2f} "
          f"{s['total']:9.4f} {s['pct']:7.1f} {s['count']:6d}")

print("=" * 108)
if step_mean > 0:
    print(f"\n  Average step time: {1e3*step_mean:.2f} ms  "
          f"({1.0/step_mean:.1f} steps/s)")
    print(f"  Grid cells: {grid_N:,}  "
          f"({1e6*step_mean/grid_N:.3f} µs/cell/step)")


# ── Save CSV ─────────────────────────────────────────────────────────
tag = f"{args.Nx}x{args.Ny}x{args.Nz}"
csv_path = os.path.join(args.out_dir, f"cost_breakdown_{tag}.csv")
with open(csv_path, "w") as f:
    f.write("component,mean_ms,std_ms,min_ms,max_ms,total_s,pct_of_step,calls\n")
    for label in sorted(summary.keys(), key=lambda k: -summary[k]["total"]):
        s = summary[label]
        f.write(f"{label},{1e3*s['mean']:.4f},{1e3*s['std']:.4f},"
                f"{1e3*s['min']:.4f},{1e3*s['max']:.4f},"
                f"{s['total']:.6f},{s['pct']:.2f},{s['count']}\n")
print(f"\n  CSV saved → {csv_path}")

# ── Save per-step raw timings (for variance diagnosis) ───────────────
raw_path = os.path.join(args.out_dir, f"cost_perstep_{tag}.csv")
# Build columns in consistent order (TOTAL first, then sorted sub-labels)
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
            vals.append(f"{1e3*times[i]:.4f}" if i < len(times) else "")
        f.write(f"{i},{used},{','.join(vals)}\n")
print(f"  Per-step CSV saved → {raw_path}")


# ═══════════════════════════════════════════════════════════════════════
# Paper-quality figures
# ═══════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Publication style ────────────────────────────────────────────────
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
# Maps category display name → list of timer-label prefixes
CATEGORIES = {
    "Body update (SDF)":           ["1b"],
    "Forces\ncomputation":         ["4 "],
    "Projection\n(pressure)":      ["3c"],
    "Convection\n& diffusion":     ["3a"],
    "Other":                       ["1  ", "2 ", "3b", "5 ", "6 "],
}

CAT_COLOURS = {
    "Body update (SDF)":           "#26a69a",
    "Forces\ncomputation":         "#ffa726",
    "Projection\n(pressure)":      "#ef5350",
    "Convection\n& diffusion":     "#42a5f5",
    "Other":                       "#90a4ae",
}

cat_means = {}   # ms
cat_pcts  = {}   # %
for cat_name, prefixes in CATEGORIES.items():
    total_s = 0.0
    for l, s in summary.items():
        if any(l.startswith(pfx) for pfx in prefixes):
            total_s += s["total"]
    mean_ms = 1e3 * total_s / n_meas if n_meas > 0 else 0.0
    pct = 100.0 * total_s / sum(summary.get("TOTAL step", {}).get("total", 1.0) for _ in [1])
    outer_total_s = summary.get("TOTAL step", {}).get("total", 1.0)
    pct = 100.0 * total_s / outer_total_s if outer_total_s > 0 else 0.0
    if mean_ms > 0:
        cat_means[cat_name] = mean_ms
        cat_pcts[cat_name]  = pct

# Print grouped summary
print("\n  ── Grouped categories ──")
for cn in CATEGORIES:
    if cn in cat_means:
        print(f"    {cn:<28s}  {cat_means[cn]:8.2f} ms  ({cat_pcts[cn]:5.1f}%)")

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
ax.set_title(f"Cost breakdown – {args.Nx}×{args.Ny}×{args.Nz} "
             f"({grid_N:,} cells, {'GPU' if USE_CUDA else 'CPU'})")

# Annotate bars
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

# Clean labels (remove \n for pie)
pie_display = [c.replace("\n", " ") for c in pie_names]

wedges, texts, autotexts = ax2.pie(
    pie_vals, labels=pie_display, autopct="%1.1f%%",
    startangle=140, pctdistance=0.78, textprops={"fontsize": 8},
    colors=pie_cols,
    wedgeprops=dict(edgecolor="white", linewidth=1.0),
)
for at in autotexts:
    at.set_fontsize(7)
ax2.set_title(f"Step cost distribution – {args.Nx}×{args.Ny}×{args.Nz}")
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
        if "SDF" in label or "sdf" in label:        return "#26a69a"
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
        if "plotting" in label:                       return "#90a4ae"
        return "#bdbdbd"

    dc = [_colour(l) for l in detail_labels]
    bars3 = ax3.barh(range(len(detail_labels)), detail_pcts, color=dc,
                     edgecolor="white", linewidth=0.4, height=0.7)
    ax3.set_yticks(range(len(detail_labels)))
    ax3.set_yticklabels(detail_labels, fontsize=7)
    ax3.set_xlabel("% of total step time")
    ax3.set_title(f"Detailed breakdown – {args.Nx}×{args.Ny}×{args.Nz}")
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


print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  Run multi-grid scaling analysis:                                  ║
║                                                                    ║
║  python run_cost_analysis.py --Nx  64 --Ny  16 --Nz  16           ║
║  python run_cost_analysis.py --Nx 128 --Ny  32 --Nz  32           ║
║  python run_cost_analysis.py --Nx 256 --Ny  64 --Nz  64           ║
║  python run_cost_analysis.py --Nx 512 --Ny 128 --Nz 128           ║
║                                                                    ║
║  Then run:  python plot_scaling.py                                 ║
║  to produce the multi-resolution comparison figure.                ║
╚══════════════════════════════════════════════════════════════════════╝
""")
