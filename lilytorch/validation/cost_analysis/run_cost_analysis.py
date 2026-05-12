#!/usr/bin/env python3
"""Shared single-grid cost analysis for pinned 1guilla in 2-D or 3-D."""

from __future__ import annotations

import argparse
import gc
import importlib
import logging
import os
import sys
import time
import types
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lilytorch.validation.cost_analysis.common import (
    BODY_CENTER_OFFSET,
    CAT_COLOURS,
    CATEGORY_PREFIXES,
    DEFAULT_DTYPE,
    DEFAULT_POISSON_METHOD,
    DEFAULT_SPAWN_X,
    DEFAULT_TIMESTEP,
    DX_REF,
    MIN_LX_FISH,
    OTHER_LABEL,
    default_results_dir,
    get_dimension_spec,
    grid_cells,
    grid_label,
    grid_tag,
    resolve_grid_tuple,
    resolve_solver_mode,
    timer_colour,
)

from lilytorch.integration.extensions import FluidExtension
from lilytorch.src.poisson_mult import PoissonSolver  # noqa: F401


torch._logging.set_logs(recompiles=True, graph_breaks=True)
_TORCH_DYNAMO_LOG = logging.getLogger("torch._dynamo")
_TORCH_DYNAMO_LOG.setLevel(logging.INFO)


parser = argparse.ArgumentParser(
    description="Pinned 1guilla computational cost analysis with shared 2-D/3-D benchmark code"
)
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--sim", type=str, default="pinned", choices=["pinned"])
parser.add_argument("--Nx", type=int, default=None)
parser.add_argument("--Ny", type=int, default=None)
parser.add_argument("--Nz", type=int, default=None)
parser.add_argument("--n_steps", type=int, default=20,
                    help="Measured steps after warm-up")
parser.add_argument("--precompile", type=int, default=30,
                    help="Untimed pre-compilation steps")
parser.add_argument("--settle_steps", type=int, default=5,
                    help="Untimed settle steps after pre-compilation")
parser.add_argument("--discard_first", type=int, default=5,
                    help="Discard first N measured steps from summary statistics")
parser.add_argument("--stability_tol", type=float, default=0.05)
parser.add_argument("--save_every", type=int, default=9999)
parser.add_argument("--mode", type=str, default=None, choices=["python", "kernel"])
parser.add_argument("--use_kernels", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--no_kernels", action="store_true",
                    help="DEPRECATED: alias for --mode python.")
parser.add_argument("--streaming_sdf_2d", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--force_narrow_batch", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--force_shared_union", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--mu_normals_union", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--bdim_union", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--streaming_sdf_3d", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--streaming_forces_3d", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--Lx_fixed", type=float, default=None,
                    help="Optional fixed x-extent; Ly/Lz follow from the grid aspect ratio.")
parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, choices=["float32", "float64"])
parser.add_argument("--poisson_method", type=str, default=DEFAULT_POISSON_METHOD,
                    choices=["multigrid", "mgcg", "fft"])
parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP)
parser.add_argument("--spawn_x", type=float, default=DEFAULT_SPAWN_X)
parser.add_argument("--freq", type=float, default=1.0)
parser.add_argument("--twl", type=float, default=0.571429 * 14)
parser.add_argument("--amp", type=float, default=15.0)
parser.add_argument("--tag_suffix", type=str, default="")
args = parser.parse_args()

spec = get_dimension_spec(args.dim)
args.grid = resolve_grid_tuple(args, spec)

try:
    SOLVER_MODE = resolve_solver_mode(args)
except ValueError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)

USE_CUDA = args.device == "cuda" and torch.cuda.is_available()
if args.out_dir is None:
    args.out_dir = default_results_dir(SCRIPT_DIR, spec)
args.out_dir = os.path.abspath(args.out_dir)
os.makedirs(args.out_dir, exist_ok=True)


class TimerBank:
    """CUDA-synchronised timer collection gated by an active flag."""

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
                "mean": arr.mean(),
                "std": arr.std(),
                "min": arr.min(),
                "max": arr.max(),
                "total": arr.sum(),
                "count": len(arr),
                "pct": 100.0 * arr.sum() / outer_total if outer_total > 0 else 0.0,
            }
        return rows


T = TimerBank(USE_CUDA)


def instrument_handler(handler):
    if getattr(handler, "_profiled", False):
        return
    handler._profiled = True

    fs = handler.fluid_solver
    adv = fs.adv_diff_solver

    _orig_step = type(handler).step
    _orig_update = handler.update
    _orig_fluid_step = type(fs).fluid_step
    _orig_apply = type(handler)._apply_forces
    _orig_forces = getattr(type(fs), spec.forces_method_name)
    _orig_recompute = type(fs)._recompute_mu_normals

    _precompile_count = [0]
    _precompile_done = [args.precompile <= 0]
    _settle_count = [0]
    _settle_done = [args.settle_steps <= 0]
    _deep_patches_installed = [False]

    def _activate_measurement(steps_completed: int, *, settled: bool):
        if USE_CUDA:
            torch.cuda.synchronize()
        T._active = True
        phase = "Settling complete" if settled else "Pre-compilation complete"
        print(
            f"  [profiler] {phase} ({steps_completed} steps). Measuring {args.n_steps} steps...\n",
            flush=True,
        )

    def _set_state(state):
        for name, value in zip(spec.state_names, state):
            setattr(fs, name, value)

    def _get_state():
        return tuple(getattr(fs, name) for name in spec.state_names)

    def detailed_step(self_h, task, physics):
        if not _precompile_done[0]:
            _precompile_count[0] += 1
            _orig_step(self_h, task, physics)
            if _precompile_count[0] >= args.precompile:
                _precompile_done[0] = True
                if USE_CUDA:
                    torch.cuda.synchronize()
                _install_deep_patches()
                if _settle_done[0]:
                    _activate_measurement(_precompile_count[0], settled=False)
                else:
                    print(
                        f"\n  [profiler] Pre-compilation complete ({_precompile_count[0]} steps). "
                        f"Settling physics for {args.settle_steps} more steps...\n",
                        flush=True,
                    )
            return

        if not _settle_done[0]:
            _settle_count[0] += 1
            _orig_step(self_h, task, physics)
            if _settle_count[0] >= args.settle_steps:
                _settle_done[0] = True
                _activate_measurement(_settle_count[0], settled=True)
            return

        T.new_step()
        iteration = self_h.iteration
        timestep = self_h.pars["solver"]["dt"]
        if iteration >= self_h.pars["solver"]["nt"]:
            self_h.iteration += 1
            return

        t = iteration * timestep
        ffs = self_h.fluid_solver

        if not ffs.terminate:
            with T("1  SDF update (body kinematics + SDF eval)"):
                self_h.update(t, iteration, dt=timestep)

            with T("2  mu+normals (recompute)"):
                _orig_recompute(ffs)

            with T("3  fluid_step (total PDE)"):
                state = _orig_fluid_step(ffs, *_get_state(), timestep)

            _set_state(state)

            with T(spec.forces_timer_label):
                _orig_forces(ffs, *_get_state(), iteration)

            from lilytorch.integration.BDIMhandler import _FS_FREE_AFTER_FORCES_3D
            ffs.__dict__.update(_FS_FREE_AFTER_FORCES_3D)

            if spec.terminate_on_handler:
                self_h.terminate = False
            else:
                ffs.terminate = False

            with T("6  apply_forces (FARMS)"):
                _orig_apply(self_h, task, physics)

            ffs._release_bdim_fields()

        self_h.iteration += 1

    def outer_step(self_h, task, physics):
        if (not _precompile_done[0]) or (not _settle_done[0]):
            detailed_step(self_h, task, physics)
        else:
            with T("TOTAL step"):
                detailed_step(self_h, task, physics)

    handler.step = types.MethodType(outer_step, handler)

    _orig_project = type(fs).project
    _orig_vardens = type(fs)._compute_variable_density_coefficients
    poisson_mg = getattr(fs, "poisson_solver", None)

    def _install_deep_patches():
        if _deep_patches_installed[0]:
            return
        _deep_patches_installed[0] = True

        def timed_update_detailed(self_h, t, iteration, dt=1):
            with T(spec.update_leaf_label):
                return _orig_update(t, iteration, dt)

        handler.update = types.MethodType(timed_update_detailed, handler)

        if getattr(fs, "use_smagorinsky", False):
            raise RuntimeError(
                "Smagorinsky LES is active but this shared cost analysis excludes it. "
                "Set cfg.smagorinsky_cs = 0.0."
            )

        _saved_adv_solve = adv.solve

        def timed_adv_solve(*call_args, **call_kwargs):
            with T("3a   advection+diffusion"):
                return _saved_adv_solve(*call_args, **call_kwargs)

        adv.solve = timed_adv_solve

        _saved_bdim = fs._bdim_meta_dyn_compiled

        def timed_bdim(*call_args, **call_kwargs):
            with T("3b   BDIM meta-equation"):
                return _saved_bdim(*call_args, **call_kwargs)

        fs._bdim_meta_dyn_compiled = timed_bdim

        def timed_project(self_fs, *call_args, **call_kwargs):
            with T("3c   projection (Poisson+gradient+correction)"):
                return _orig_project(self_fs, *call_args, **call_kwargs)

        fs.project = types.MethodType(timed_project, fs)

        _orig_set_bcs_fn = type(adv).set_BCs

        def timed_set_bcs(self_adv, *call_args, **call_kwargs):
            with T("3d   set_BCs"):
                return _orig_set_bcs_fn(self_adv, *call_args, **call_kwargs)

        adv.set_BCs = types.MethodType(timed_set_bcs, adv)

        def timed_vardens(self_fs, *call_args, **call_kwargs):
            with T(spec.vardens_leaf_label):
                return _orig_vardens(self_fs, *call_args, **call_kwargs)

        fs._compute_variable_density_coefficients = types.MethodType(timed_vardens, fs)

        _orig_release = type(fs)._release_bdim_fields

        def timed_release(self_fs, *call_args, **call_kwargs):
            with T("3f   release BDIM fields"):
                return _orig_release(self_fs, *call_args, **call_kwargs)

        fs._release_bdim_fields = types.MethodType(timed_release, fs)

        instrument_poisson_internals = grid_cells(args.grid) >= 500_000
        if poisson_mg is not None and instrument_poisson_internals:
            _orig_jacobi_fn = type(poisson_mg).Jacobi

            def timed_jacobi(self_poisson, *call_args, **call_kwargs):
                with T("3c.i   Jacobi smoothing"):
                    return _orig_jacobi_fn(self_poisson, *call_args, **call_kwargs)

            poisson_mg.Jacobi = types.MethodType(timed_jacobi, poisson_mg)

            _orig_vcycle_fn = type(poisson_mg)._vcycle
            _in_vcycle = [False]

            def timed_vcycle(self_poisson, *call_args, **call_kwargs):
                if _in_vcycle[0]:
                    return _orig_vcycle_fn(self_poisson, *call_args, **call_kwargs)
                _in_vcycle[0] = True
                try:
                    with T("3c.ii  V-cycle (top-level)"):
                        return _orig_vcycle_fn(self_poisson, *call_args, **call_kwargs)
                finally:
                    _in_vcycle[0] = False

            poisson_mg._vcycle = types.MethodType(timed_vcycle, poisson_mg)

        extra = ", Poisson internals" if instrument_poisson_internals and poisson_mg else ""
        print(f"  [profiler] Deep patches installed (SDF, advection, BDIM, project{extra})", flush=True)

    if _precompile_done[0]:
        _install_deep_patches()
        if _settle_done[0]:
            _activate_measurement(0, settled=False)
        else:
            print(
                f"  [profiler] Pre-compilation complete (0 steps). Settling physics for {args.settle_steps} more steps...\n",
                flush=True,
            )

    print(
        f"  [profiler] Instrumented handler (grid {fs.grid_shape}, device {fs.device}, poisson={fs.poisson_method})",
        flush=True,
    )
    print(f"  [profiler] Running {args.precompile} pre-compilation steps (untimed)...", flush=True)


_orig_init_episode = FluidExtension.initialize_episode
_handler_ref = [None]


def _patched_init_episode(self, task, physics):
    _orig_init_episode(self, task, physics)
    if hasattr(self, "BDIMhandler") and self.BDIMhandler is not None:
        instrument_handler(self.BDIMhandler)
        _handler_ref[0] = self.BDIMhandler


FluidExtension.initialize_episode = _patched_init_episode


def _apply_animat_overrides(cfg):
    if not getattr(cfg, "animats_pars", None):
        return
    animat = cfg.animats_pars[0]
    pose = list(animat.get("pose", []))
    if pose:
        pose[0] = args.spawn_x
        animat["pose"] = pose
    animat["gains"] = [50.0, 0.4, 0.0]
    control_pars = dict(animat.get("control_pars", {}))
    control_pars.update({"freq": args.freq, "twl": args.twl, "amp": args.amp})
    animat["control_pars"] = control_pars


def _apply_cfg_overrides(cfg):
    cfg.use_bdim = True
    cfg.Nx = args.Nx
    cfg.Ny = args.Ny
    if spec.dim == 3:
        cfg.Nz = args.Nz

    cfg.smagorinsky_cs = 0.0
    cfg.dtype = args.dtype
    cfg.poisson_method = args.poisson_method
    cfg.timestep = args.timestep
    if hasattr(cfg, "bdim_dt"):
        cfg.bdim_dt = args.timestep
    cfg.n_iterations = args.precompile + args.settle_steps + args.n_steps + 1
    if hasattr(cfg, "bdim_nt"):
        cfg.bdim_nt = cfg.n_iterations
    if hasattr(cfg, "time_integration"):
        cfg.time_integration = "euler"

    # Tuned defaults: warm-start + RBGS with few sweeps + MGCG for BDIM
    # coefficient-jump robustness.  Direct assignment (not getattr) so we
    # override BaseSimConfig.
    #
    # Tolerance: kept at 1e-7 absolute (BaseSimConfig default), NOT loosened.
    # The Poisson residual b = -h²·f scales with h², so an absolute tol
    # near machine-h²·||f|| is the right scale; loosening to 1e-4 makes
    # MGCG init-exit on a zero warm-start every step and the pressure
    # never gets updated (visible as a flat-zero pressure plot).
    cfg.poisson_tol = 1.0e-7
    cfg.poisson_max_cycles = 2
    # With tol=1e-7 and cold-start, MGCG typically needs ~25-35 CG
    # iterations to converge for typical fish-swimming flows
    # (initial residual r0 = h²·f ≈ O(7), 50% reduction per step →
    # ~27 iterations).  Any max_mgcg_cycles < 25 means we always exhaust
    # the budget without converging.  Using 5 instead of 10 gives the
    # same (unconverged) accuracy at half the CG overhead.
    cfg.poisson_max_mgcg_cycles = 5
    cfg.poisson_precond_vcycles = 1
    cfg.poisson_warm_start = True
    cfg.poisson_smoother = "rbgs"
    # nsmoothing=2: each V-cycle does 2 RBGS sweeps per level (pre + post).
    # nsmoothing=4 does 2× more bandwidth work without proportional convergence
    # gain when using warm_start=True (initial guess is already close).
    # At 4096×1024: nsmoothing=4 → ~10 ms, nsmoothing=2 → ~5.7 ms, so FFT
    # (3.9 ms) is approached and crossed with max_vcycles=1 (~2.9 ms).
    cfg.poisson_nsmoothing = 2
    # max_vcycles=1 with warm_start: since the pressure guess from the previous
    # step is close, a single V-cycle (2 sweeps/level) satisfies the tolerance.
    # Use 2 for the first few cold-start steps where the guess may be poor.
    cfg.poisson_max_cycles = 1
    # compile_smoother=True fuses the recursive V-cycle (all O(log N) levels)
    # into a single compiled CUDA graph, eliminating the O(log N) Python
    # dispatch overhead that otherwise makes multigrid scale as O(N log N)
    # instead of O(N).  Set here so it flows through _bdim_extension() →
    # YAML and is visible in the simulation_config.yaml.
    cfg.poisson_compile = True
    cfg.zero_pressure_inside = getattr(cfg, "zero_pressure_inside", False)

    cfg.save_every = args.save_every
    cfg.save_frames = False
    cfg.headless = True
    cfg.save = False

    _apply_animat_overrides(cfg)

    if args.Lx_fixed is not None:
        dx = args.Lx_fixed / args.Nx
        lx = args.Lx_fixed
    else:
        dx = max(DX_REF, MIN_LX_FISH / args.Nx)
        lx = args.Nx * dx
    ly = args.Ny * dx
    body_center = args.spawn_x + BODY_CENTER_OFFSET
    cfg.xmin = body_center - 0.5 * lx
    cfg.xmax = body_center + 0.5 * lx
    cfg.ymin = -0.5 * ly
    cfg.ymax = 0.5 * ly
    if spec.dim == 3:
        lz = args.Nz * dx
        cfg.zmin = -0.5 * lz
        cfg.zmax = 0.5 * lz

    if dx > DX_REF * 1.01:
        print(
            f"  [domain] Nx={args.Nx} too small for dx_ref -> using dx={dx:.6f} ({dx / DX_REF:.1f}x coarser)",
            flush=True,
        )
    domain_bits = [
        f"x=[{cfg.xmin:.3f}, {cfg.xmax:.3f}]",
        f"y=[{cfg.ymin:.3f}, {cfg.ymax:.3f}]",
    ]
    if spec.dim == 3:
        domain_bits.append(f"z=[{cfg.zmin:.3f}, {cfg.zmax:.3f}]")
    print(f"  [domain] {'  '.join(domain_bits)}  dx={dx:.6f}", flush=True)

    original_gen_simulation_config = cfg.gen_simulation_config

    def gen_simulation_config_lean(output_folder):
        original_gen_simulation_config(output_folder)
        import yaml

        yaml_path = os.path.join(output_folder, "simulation_config.yaml")
        with open(yaml_path, "r") as handle:
            raw_yaml = handle.read()
        try:
            sim_dict = yaml.safe_load(raw_yaml)
        except yaml.YAMLError:
            sim_dict = yaml.unsafe_load(raw_yaml)

        keep_loaders = {"lilytorch.integration.extensions.FluidExtension"}
        original_exts = sim_dict.get("extensions", [])
        sim_dict["extensions"] = [
            ext for ext in original_exts if ext.get("loader", "") in keep_loaders
        ]
        sim_dict.setdefault("runtime", {})["headless"] = True

        for ext in sim_dict["extensions"]:
            bdim_yaml = ext.get("config", {}).get("bdim_yaml", {})
            solver_cfg = bdim_yaml.get("solver", {})
            if solver_cfg:
                solver_cfg["compile_adv_diff"] = True
                solver_cfg["poisson_compile"] = True
                # 'reduce-overhead' enables torch.compile CUDA graph capture:
                # the full recursive V-cycle (all log N levels) is replayed
                # as a single CUDA graph call (~5 µs) instead of ~140+
                # individual kernel dispatches (~140 µs with mode='default').
                solver_cfg["poisson_compile_mode"] = "reduce-overhead"
                solver_cfg["dtype"] = args.dtype
                solver_cfg["poisson_method"] = args.poisson_method
                if SOLVER_MODE is not None:
                    solver_cfg["solver_method"] = SOLVER_MODE
        with open(yaml_path, "w") as handle:
            yaml.dump(sim_dict, handle, default_flow_style=False, sort_keys=False)

        stripped = len(original_exts) - len(sim_dict["extensions"])
        print(f"  [profiler] Stripped {stripped} extensions, kept FluidExtension only", flush=True)
        print(
            "  [profiler] Enabled torch.compile for adv-diff, Poisson, forces, and SDF",
            flush=True,
        )

    cfg.gen_simulation_config = gen_simulation_config_lean


cfg_mod = importlib.import_module(spec.config_module)
cfg = cfg_mod.SimConfig()
_apply_cfg_overrides(cfg)

grid_n = grid_cells(args.grid)
print("=" * 72)
print(f"  {spec.label} {args.sim.capitalize()} 1guilla - Computational Cost Analysis")
print(f"  Grid:   {grid_label(args.grid)}  ({grid_n:,} cells)")
print(
    f"  Steps:  {args.n_steps} measured  (+ {args.precompile} pre-compile, + {args.settle_steps} settle)"
)
print(f"  Solver: {args.poisson_method}, dtype={args.dtype}, mode={SOLVER_MODE or 'default'}")
print(f"  Device: {'CUDA' if USE_CUDA else 'CPU'}")
print("=" * 72)

recompile_log_path = os.path.join(args.out_dir, f"recompiles_{grid_tag(args.grid)}.log")
open(recompile_log_path, "w").close()
rc_handler = logging.FileHandler(recompile_log_path, mode="a")
rc_handler.setLevel(logging.INFO)
rc_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
_TORCH_DYNAMO_LOG.addHandler(rc_handler)
print(f"  Recompile log: {recompile_log_path}")


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
from farms_core.simulation.options import Simulator
from farms_sim.simulation import run_simulation

experiment_options = ExperimentOptions.load("experiment_config.yaml")
experiment_data_loader = import_item(experiment_options.loaders.experiment_data)
experiment_data = experiment_data_loader.from_options(experiment_options)

print("  Starting simulation...\n")
sim = run_simulation(
    experiment_data=experiment_data,
    experiment_options=experiment_options,
    simulator=Simulator.MUJOCO,
)
del sim
gc.collect()


os.chdir(SCRIPT_DIR)
summary = T.summary(discard_first=args.discard_first)
n_meas = T._step_count
n_used = max(n_meas - args.discard_first, 0)
if args.discard_first > 0:
    print(
        f"\n  [profiler] Discarded first {args.discard_first} timed steps -> using {n_used} of {n_meas} measured steps for statistics"
    )

if not summary:
    print(f"\nERROR: No measurements were captured (step_count={T._step_count})")
    sys.exit(1)

step_median = summary.get("TOTAL step", {}).get("median", 0.0)

print("\n" + "=" * 108)
print(
    f"  TIMING RESULTS  ({n_meas} measured steps, grid {grid_label(args.grid)} = {grid_n:,} cells, {'GPU' if USE_CUDA else 'CPU'})"
)
print("=" * 108)
hdr = (
    f"  {'Component':<44s} {'Median':>9s} {'Std':>9s} {'Min':>9s} {'Max':>9s} {'Total':>9s} {'%step':>7s} {'Calls':>6s}"
)
print(hdr)
print(f"  {'':44s} {'(ms)':>9s} {'(ms)':>9s} {'(ms)':>9s} {'(ms)':>9s} {'(s)':>9s} {'':>7s}")
print("-" * 108)
for label in sorted(summary.keys(), key=lambda key: -summary[key]["total"]):
    item = summary[label]
    print(
        f"  {label:<44s} {1e3 * item['median']:9.2f} {1e3 * item['std']:9.2f} {1e3 * item['min']:9.2f} "
        f"{1e3 * item['max']:9.2f} {item['total']:9.4f} {item['pct']:7.1f} {item['count']:6d}"
    )
print("=" * 108)
if step_median > 0:
    print(f"\n  Median step time: {1e3 * step_median:.2f} ms  ({1.0 / step_median:.1f} steps/s)")
    print(f"  Grid cells: {grid_n:,}  ({1e6 * step_median / grid_n:.3f} us/cell/step)")

total_times = np.array(T._data.get("TOTAL step", [])[args.discard_first:])
if len(total_times) >= 10:
    tail = total_times[-10:]
    cv = tail.std() / tail.mean() if tail.mean() > 0 else 0.0
    status = "OK" if cv < args.stability_tol else "UNSTABLE"
    print(
        f"  Stability check (last 10 timed steps): CV = {cv * 100:.1f}%  [{status}, tol = {args.stability_tol * 100:.0f}%]"
    )

rc_handler.flush()
try:
    with open(recompile_log_path) as handle:
        rc_text = handle.read()
    n_recompiles = sum(
        1 for line in rc_text.splitlines()
        if "recompiling" in line.lower() or "cache_size_limit" in line
    )
    n_breaks = sum(1 for line in rc_text.splitlines() if "graph break" in line.lower())
    print(
        f"  torch.compile: {n_recompiles} recompile event(s), {n_breaks} graph break(s)  -> see {recompile_log_path}"
    )
except OSError:
    pass


tag = f"{grid_tag(args.grid)}{args.tag_suffix}"
csv_path = os.path.join(args.out_dir, f"cost_breakdown_{tag}.csv")
with open(csv_path, "w") as handle:
    handle.write("component,median_ms,mean_ms,std_ms,min_ms,max_ms,total_s,pct_of_step,calls\n")
    for label in sorted(summary.keys(), key=lambda key: -summary[key]["total"]):
        item = summary[label]
        handle.write(
            f"{label},{1e3 * item['median']:.4f},{1e3 * item['mean']:.4f},{1e3 * item['std']:.4f},"
            f"{1e3 * item['min']:.4f},{1e3 * item['max']:.4f},{item['total']:.6f},{item['pct']:.2f},{item['count']}\n"
        )
print(f"\n  CSV saved -> {csv_path}")

raw_path = os.path.join(args.out_dir, f"cost_perstep_{tag}.csv")
raw_labels = ["TOTAL step"] + sorted(label for label in T._data.keys() if label != "TOTAL step")
with open(raw_path, "w") as handle:
    handle.write("step,used," + ",".join(raw_labels) + "\n")
    n_steps = len(T._data.get("TOTAL step", []))
    for index in range(n_steps):
        used = "yes" if index >= args.discard_first else "discarded"
        values = []
        for label in raw_labels:
            times = T._data.get(label, [])
            n_calls = len(times)
            if n_calls == 0:
                values.append("")
            elif n_calls == n_steps:
                values.append(f"{1e3 * times[index]:.4f}")
            else:
                start = (index * n_calls) // n_steps
                end = ((index + 1) * n_calls) // n_steps
                values.append(f"{1e3 * sum(times[start:end]):.4f}" if start < n_calls else "")
        handle.write(f"{index},{used},{','.join(values)}\n")
print(f"  Per-step CSV saved -> {raw_path}")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


def _trimmed(label):
    times = T._data.get(label, [])
    return np.array(times[args.discard_first:]) if args.discard_first < len(times) else np.array(times)


cat_means = {}
cat_pcts = {}
total_series = _trimmed("TOTAL step")
outer_total_s = float(total_series.sum())
explicit_per_step = np.zeros_like(total_series)
for category, prefixes in CATEGORY_PREFIXES.items():
    category_series = np.zeros_like(total_series)
    for label in summary:
        if label == "TOTAL step":
            continue
        if any(label.startswith(prefix) for prefix in prefixes):
            series = _trimmed(label)
            if len(series) == len(total_series):
                category_series += series
            else:
                category_series += series.sum() / max(len(total_series), 1)
    explicit_per_step += category_series
    if category_series.sum() > 0:
        cat_means[category] = 1e3 * float(np.median(category_series))
        cat_pcts[category] = 100.0 * float(category_series.sum()) / outer_total_s if outer_total_s > 0 else 0.0

residual_series = np.clip(total_series - explicit_per_step, 0.0, None)
if residual_series.sum() > 0:
    cat_means[OTHER_LABEL] = 1e3 * float(np.median(residual_series))
    cat_pcts[OTHER_LABEL] = 100.0 * float(residual_series.sum()) / outer_total_s if outer_total_s > 0 else 0.0

explicit_total_s = float(explicit_per_step.sum())
residual_s = float(residual_series.sum())

print("\n  -- Grouped categories --")
for category in list(CATEGORY_PREFIXES.keys()) + [OTHER_LABEL]:
    if category in cat_means:
        label = category.replace("\n", " ")
        print(f"    {label:<32s}  {cat_means[category]:8.2f} ms  ({cat_pcts[category]:5.1f}%)")
if outer_total_s > 0:
    coverage = 100.0 * (explicit_total_s + residual_s) / outer_total_s
    print(f"    {'(coverage check: sum(categories) / TOTAL step)':<32s}  {coverage:7.2f}%")

cat_names_sorted = sorted(cat_means.keys(), key=lambda key: cat_means[key])
fig, ax = plt.subplots(figsize=(5.8, 3.2))
bars = ax.barh(
    range(len(cat_names_sorted)),
    [cat_means[name] for name in cat_names_sorted],
    color=[CAT_COLOURS.get(name, CAT_COLOURS[OTHER_LABEL]) for name in cat_names_sorted],
    edgecolor="white",
    linewidth=0.6,
    height=0.7,
)
ax.set_yticks(range(len(cat_names_sorted)))
ax.set_yticklabels(cat_names_sorted)
ax.set_xlabel("Time per step (ms)")
ax.set_title(f"Cost breakdown - {grid_label(args.grid)} ({spec.label})")
if cat_names_sorted:
    max_value = max(cat_means.values())
    for bar, category in zip(bars, cat_names_sorted):
        width = bar.get_width()
        pct = cat_pcts[category]
        ax.text(width + 0.02 * max_value, bar.get_y() + bar.get_height() / 2,
                f"{width:.1f} ms ({pct:.0f}%)", va="center", fontsize=8)
    ax.set_xlim(0, max_value * 1.35)
fig.tight_layout()
bar_path = os.path.join(args.out_dir, f"cost_barh_{tag}.pdf")
fig.savefig(bar_path)
fig.savefig(bar_path.replace(".pdf", ".png"))
plt.close(fig)
print(f"  Figure saved -> {bar_path}")

fig2, ax2 = plt.subplots(figsize=(5.0, 4.0))
pie_values = [cat_pcts[name] for name in cat_names_sorted]
pie_colours = [CAT_COLOURS.get(name, CAT_COLOURS[OTHER_LABEL]) for name in cat_names_sorted]
if pie_values and sum(pie_values) > 0:
    ax2.pie(
        pie_values,
        labels=[name.replace("\n", " ") for name in cat_names_sorted],
        autopct=lambda value: f"{value:.0f}%" if value >= 3 else "",
        startangle=100,
        colors=pie_colours,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        textprops={"fontsize": 8},
    )
ax2.set_title(f"Relative cost distribution - {grid_label(args.grid)} ({spec.label})")
fig2.tight_layout()
pie_path = os.path.join(args.out_dir, f"cost_pie_{tag}.pdf")
fig2.savefig(pie_path)
fig2.savefig(pie_path.replace(".pdf", ".png"))
plt.close(fig2)
print(f"  Figure saved -> {pie_path}")

detail_labels = [label for label in summary if label != "TOTAL step"]
detail_labels.sort(key=lambda label: -summary[label]["median"])
fig3, ax3 = plt.subplots(figsize=(7.2, max(3.6, 0.34 * len(detail_labels) + 1.4)))
detail_values = [1e3 * summary[label]["median"] for label in detail_labels]
ax3.barh(
    range(len(detail_labels)),
    detail_values,
    color=[timer_colour(label) for label in detail_labels],
    edgecolor="white",
    linewidth=0.6,
)
ax3.set_yticks(range(len(detail_labels)))
ax3.set_yticklabels(detail_labels)
ax3.invert_yaxis()
ax3.set_xlabel("Median time (ms)")
ax3.set_title(f"Detailed timers - {grid_label(args.grid)} ({spec.label})")
fig3.tight_layout()
detail_path = os.path.join(args.out_dir, f"cost_detail_{tag}.pdf")
fig3.savefig(detail_path)
fig3.savefig(detail_path.replace(".pdf", ".png"))
plt.close(fig3)
print(f"  Figure saved -> {detail_path}")

handler = _handler_ref[0]
if handler is not None:
    fs = handler.fluid_solver
    comp = fs.composite_body
    fig4, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    step_index = args.precompile + args.settle_steps + args.n_steps

    if spec.dim == 2:
        p = fs.p0.detach().cpu().float().numpy()[1:-1, 1:-1]
        u = fs.u0.detach().cpu().float().numpy()[1:-1, 1:-1]
        v = fs.v0.detach().cpu().float().numpy()[1:-1, 1:-1]
        sdf = comp.sdf_val.detach().cpu().float().numpy()[1:-1, 1:-1]
        # eps_val: BDIM transition-zone half-width in metres.
        # Cells with 0 ≤ sdf < eps are in the smearing zone and can carry
        # anomalously high body-motion velocities that corrupt p2/p98 on
        # smaller grids (where the zone is a larger fraction of total cells).
        eps_val = float(fs.eps)

        vmag = np.sqrt(u**2 + v**2)
        dx = (cfg.xmax - cfg.xmin) / args.Nx
        dvdx = np.gradient(v, dx, axis=0)
        dudy = np.gradient(u, dx, axis=1)
        omega_z = dvdx - dudy

        x_1d = np.linspace(cfg.xmin, cfg.xmax, p.shape[0])
        y_1d = np.linspace(cfg.ymin, cfg.ymax, p.shape[1])
        fig4.suptitle(
            f"Flow fields - {grid_label(args.grid)} ({grid_n:,} cells, step {step_index}, dx={dx*1e3:.2f} mm, eps={eps_val*1e3:.2f} mm)",
            fontsize=11,
            fontweight="bold",
        )

        def _format_ax(ax, title, data, cmap, symmetric=False):
            # Use sdf >= eps (far-field fluid, outside BDIM smearing zone) for
            # percentile-based colormap range, but display the full fluid region
            # (sdf >= 0, shown in colour; sdf < 0 body interior shown in grey).
            farfield_mask = sdf >= eps_val
            data_farfield = np.where(farfield_mask, data, np.nan)
            fluid_mask = sdf >= 0
            data_display = np.where(fluid_mask, data, np.nan)
            if symmetric:
                vmax = max(abs(np.nanpercentile(data_farfield, 2)), abs(np.nanpercentile(data_farfield, 98)))
                vmin = -vmax
            else:
                vmin = np.nanpercentile(data_farfield, 2)
                vmax = np.nanpercentile(data_farfield, 98)
            cmap_obj = plt.get_cmap(cmap).copy()
            cmap_obj.set_bad(color="#cccccc")
            im = ax.imshow(
                data_display.T,
                origin="lower",
                aspect="auto",
                cmap=cmap_obj,
                extent=[x_1d[0], x_1d[-1], y_1d[0], y_1d[-1]],
                vmin=vmin,
                vmax=vmax,
            )
            ax.contour(x_1d, y_1d, sdf.T, levels=[0], colors="k", linewidths=0.8, linestyles="-")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x (m)", fontsize=8)
            ax.set_ylabel("y (m)", fontsize=8)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

        _format_ax(axes[0, 0], "Pressure $p$", p, "RdBu_r", symmetric=True)
        _format_ax(axes[0, 1], "Velocity magnitude $|u|$", vmag, "viridis")
        _format_ax(axes[1, 0], "Vorticity $\\omega_z$", omega_z, "RdBu_r", symmetric=True)

        far_sentinel = 1e3
        sdf_plot = np.ma.masked_where(np.abs(sdf) >= far_sentinel, sdf)
        non_far = sdf[np.abs(sdf) < far_sentinel]
        vabs = float(np.percentile(np.abs(non_far), 98)) if non_far.size else 1.0
        cmap_sdf = plt.get_cmap("coolwarm").copy()
        cmap_sdf.set_bad(color="#dddddd")
        im_sdf = axes[1, 1].imshow(
            sdf_plot.T,
            origin="lower",
            aspect="auto",
            cmap=cmap_sdf,
            extent=[x_1d[0], x_1d[-1], y_1d[0], y_1d[-1]],
            vmin=-vabs,
            vmax=vabs,
        )
        sdf_for_contour = np.where(np.abs(sdf) >= far_sentinel, np.nan, sdf)
        axes[1, 1].contour(x_1d, y_1d, sdf_for_contour.T, levels=[0], colors="k", linewidths=0.8, linestyles="-")
        axes[1, 1].set_title(f"SDF union (grey=_FAR, +/-{vabs * 1e3:.1f} mm p98)", fontsize=10)
        axes[1, 1].set_xlabel("x (m)", fontsize=8)
        axes[1, 1].set_ylabel("y (m)", fontsize=8)
        axes[1, 1].tick_params(labelsize=7)
        plt.colorbar(im_sdf, ax=axes[1, 1], shrink=0.85, pad=0.02)
    else:
        p = fs.p0.detach().cpu().float().numpy()
        u = fs.u0.detach().cpu().float().numpy()
        v = fs.v0.detach().cpu().float().numpy()
        sdf = comp.sdf_val.detach().cpu().float().numpy()
        eps_val = float(fs.eps)

        kz = p.shape[2] // 2
        p_slice = p[1:-1, 1:-1, kz]
        u_slice = u[1:-1, 1:-1, kz]
        v_slice = v[1:-1, 1:-1, kz]
        sdf_slice = sdf[1:-1, 1:-1, kz]

        vmag_slice = np.sqrt(u_slice**2 + v_slice**2)
        dx = (cfg.xmax - cfg.xmin) / args.Nx
        dvdx = np.gradient(v_slice, dx, axis=0)
        dudy = np.gradient(u_slice, dx, axis=1)
        omega_z = dvdx - dudy

        x_1d = np.linspace(cfg.xmin, cfg.xmax, p_slice.shape[0])
        y_1d = np.linspace(cfg.ymin, cfg.ymax, p_slice.shape[1])
        fig4.suptitle(
            f"Flow fields (mid-z plane) - {grid_label(args.grid)} ({grid_n:,} cells, step {step_index}, dx={dx*1e3:.2f} mm, eps={eps_val*1e3:.2f} mm)",
            fontsize=11,
            fontweight="bold",
        )

        def _format_ax(ax, title, data, cmap, symmetric=False):
            # Use sdf >= eps (far-field, outside BDIM smearing zone) for
            # percentile-based colormap range; display full fluid sdf >= 0.
            farfield_mask = sdf_slice >= eps_val
            data_farfield = np.where(farfield_mask, data, np.nan)
            fluid_mask = sdf_slice >= 0
            data_display = np.where(fluid_mask, data, np.nan)
            if symmetric:
                vmax = max(abs(np.nanpercentile(data_farfield, 2)), abs(np.nanpercentile(data_farfield, 98)))
                vmin = -vmax
            else:
                vmin = np.nanpercentile(data_farfield, 2)
                vmax = np.nanpercentile(data_farfield, 98)
            cmap_obj = plt.get_cmap(cmap).copy()
            cmap_obj.set_bad(color="#cccccc")
            im = ax.imshow(
                data_display.T,
                origin="lower",
                aspect="equal",
                cmap=cmap_obj,
                extent=[x_1d[0], x_1d[-1], y_1d[0], y_1d[-1]],
                vmin=vmin,
                vmax=vmax,
            )
            ax.contour(x_1d, y_1d, sdf_slice.T, levels=[0], colors="k", linewidths=0.8, linestyles="-")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x (m)", fontsize=8)
            ax.set_ylabel("y (m)", fontsize=8)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

        _format_ax(axes[0, 0], "Pressure $p$", p_slice, "RdBu_r", symmetric=True)
        _format_ax(axes[0, 1], "Velocity magnitude $|u|$", vmag_slice, "viridis")
        _format_ax(axes[1, 0], "Vorticity $\\omega_z$", omega_z, "RdBu_r", symmetric=True)

        far_sentinel = 1e3
        sdf_plot = np.ma.masked_where(np.abs(sdf_slice) >= far_sentinel, sdf_slice)
        non_far = sdf_slice[np.abs(sdf_slice) < far_sentinel]
        vabs = float(np.percentile(np.abs(non_far), 98)) if non_far.size else 1.0
        cmap_sdf = plt.get_cmap("coolwarm").copy()
        cmap_sdf.set_bad(color="#dddddd")
        im_sdf = axes[1, 1].imshow(
            sdf_plot.T,
            origin="lower",
            aspect="equal",
            cmap=cmap_sdf,
            extent=[x_1d[0], x_1d[-1], y_1d[0], y_1d[-1]],
            vmin=-vabs,
            vmax=vabs,
        )
        sdf_for_contour = np.where(np.abs(sdf_slice) >= far_sentinel, np.nan, sdf_slice)
        axes[1, 1].contour(x_1d, y_1d, sdf_for_contour.T, levels=[0], colors="k", linewidths=0.8, linestyles="-")
        axes[1, 1].set_title(f"SDF union (grey=_FAR, +/-{vabs * 1e3:.1f} mm p98)", fontsize=10)
        axes[1, 1].set_xlabel("x (m)", fontsize=8)
        axes[1, 1].set_ylabel("y (m)", fontsize=8)
        axes[1, 1].tick_params(labelsize=7)
        plt.colorbar(im_sdf, ax=axes[1, 1], shrink=0.85, pad=0.02)

    field_path = os.path.join(args.out_dir, f"flow_fields_{tag}.pdf")
    fig4.savefig(field_path)
    fig4.savefig(field_path.replace(".pdf", ".png"))
    plt.close(fig4)
    print(f"  Figure saved -> {field_path}")
else:
    print("  [warning] Could not access handler - skipping flow field plots.")

_TORCH_DYNAMO_LOG.removeHandler(rc_handler)
rc_handler.close()