#!/usr/bin/env python3
"""
GPU memory comparison across solver modes — 2-D and 3-D pinned 1guilla.

This is the *measurement* counterpart to ``docs/memory_analysis.md``.  It
runs the same FARMS-coupled pinned 1guilla scenario used in the cost
analysis (``lilytorch/validation/cost_analysis/``) under the reference
Python path and the optimised kernel path, and reports per-phase /
per-tensor memory usage so the biggest bottlenecks can be identified
directly on hardware.

Using the **pinned** 1guilla benchmark (same as cost_analysis/) means:
  * Memory numbers are directly comparable to the timing numbers produced
    by ``run_cost_analysis.py``.
  * Both 2-D and 3-D dimensions are supported with the same script.
  * The same Poisson settings, domain sizing, and compile flags are used.

Modes
-----
* ``python``  — pure-PyTorch reference path
                (``solver.solver_method = "python"``).
                No batching, no per-body cropping, no streaming kernels.
                Per-body SDF / body-velocity fields are materialised on the
                *full* fluid grid.
* ``kernel``  — native streamed kernel path
                (``solver.solver_method = "kernel"``).
                Uses the final update-only geometry pass plus the
                post-fluid-step native force pass.  Persistent packed
                mu/normals buffers are kept across steps (union AABB mode).

Dimensions
----------
* ``--dim 2`` — 2-D grid, ``gen_configs_one_pinned_2d.SimConfig``
* ``--dim 3`` — 3-D grid, ``gen_configs_one_pinned_3d.SimConfig`` (default)

Driver vs. worker
-----------------
* When called **without** ``--mode``, the script becomes a *driver* and
  spawns one subprocess per mode (so each measurement starts from a
  clean CUDA context / torch.compile cache).  It collects the resulting
  JSON files and prints a side-by-side comparison.

* When called **with** ``--mode <X>``, the script becomes a *worker*: it
  installs deep monkey-patches on ``BDIMhandler`` / ``FluidSolver``,
  runs the in-process FARMS simulation, and writes one JSON file with
  every snapshot.

Typical usage
-------------
::

    # 2-D: run both modes and emit a comparison table
    python run_memory_comparison.py --dim 2 --Nx 512 --Ny 128 --n_steps 80

    # 3-D: same
    python run_memory_comparison.py --dim 3 --Nx 256 --Ny 64 --Nz 64 --n_steps 80

    # Re-run a single mode:
    python run_memory_comparison.py --dim 3 --mode kernel \\
        --Nx 256 --Ny 64 --Nz 64 --n_steps 80

    # Reuse existing JSON files:
    python run_memory_comparison.py --dim 2 --Nx 512 --keep_existing

The script intentionally uses coarsened grids by default so that both
modes fit on a 12 GB GPU (the ``python`` path needs roughly 2× the memory
of the ``kernel`` path on the same grid).
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import subprocess
import sys
import time
import types

# Make cost_analysis/common.py importable by adding the parent dir to sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COST_ANALYSIS_DIR = os.path.join(_SCRIPT_DIR, "..", "cost_analysis")
if _COST_ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _COST_ANALYSIS_DIR)

from common import (  # noqa: E402 — deferred import after path setup
    BODY_CENTER_OFFSET,
    DEFAULT_DTYPE,
    DEFAULT_POISSON_METHOD,
    DEFAULT_SPAWN_X,
    DEFAULT_TIMESTEP,
    DX_REF,
    MIN_LX_FISH,
    get_dimension_spec,
    grid_cells,
    grid_label,
    grid_tag,
)


# ════════════════════════════════════════════════════════════════════════
#  CLI parsing — shared by driver and worker
# ════════════════════════════════════════════════════════════════════════

_MODES = ("python", "kernel")
_MODE_ALIASES = {
    "no_kernels": "python",
    "kernels": "kernel",
}


def _canonical_mode(text: str) -> str:
    mode = _MODE_ALIASES.get(text.strip(), text.strip())
    if mode not in _MODES:
        valid = ", ".join(_MODES)
        raise argparse.ArgumentTypeError(
            f"invalid mode '{text}' (expected one of: {valid})"
        )
    return mode


def _result_path(out_dir: str, mode: str) -> str:
    # Used by the worker to write its result file.  n_bodies is encoded in
    # the filename so that parallel body-sweep workers never collide.
    # The filename pattern matches _worker_result_path() in the driver.
    raise RuntimeError(
        "_result_path() must not be called directly; "
        "use _worker_result_path(out_dir, mode, n_bodies) instead"
    )


def _existing_result_path(out_dir: str, mode: str) -> str:
    # Legacy file names from the old (3-D only) version of this script.
    _LEGACY = {"python": "memory_no_kernels.json", "kernel": "memory_kernels.json"}
    canonical = _worker_result_path(out_dir, mode, n_bodies=1)
    if os.path.exists(canonical):
        return canonical
    legacy_name = _LEGACY.get(mode)
    if legacy_name is not None:
        legacy = os.path.join(out_dir, legacy_name)
        if os.path.exists(legacy):
            return legacy
    return canonical


def _worker_result_path(out_dir: str, mode: str, n_bodies: int) -> str:
    """Canonical JSON path for a (mode, n_bodies) worker result."""
    return os.path.join(out_dir, f"memory_{mode}_b{n_bodies:02d}.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GPU memory comparison across python / kernel solver modes "
            "for 2-D and 3-D pinned 1guilla."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dim", type=int, default=3, choices=[2, 3],
        help="Simulation dimensionality (default: 3).",
    )
    parser.add_argument(
        "--mode", type=_canonical_mode, choices=_MODES, default=None,
        help=(
            "Run a single configuration (worker mode).  Without this flag "
            "the script becomes the driver and spawns one subprocess per "
            "mode."
        ),
    )
    parser.add_argument(
        "--Nx", type=int, default=None,
        help="Grid cells in x.  Defaults to the dimension spec's default_grid.",
    )
    parser.add_argument(
        "--Ny", type=int, default=None,
        help="Grid cells in y.  Defaults to the dimension spec's default_grid.",
    )
    parser.add_argument(
        "--Nz", type=int, default=None,
        help="Grid cells in z (3-D only).  Defaults to the dimension spec's default_grid.",
    )
    parser.add_argument(
        "--n_steps", type=int, default=80,
        help=(
            "Total simulation steps to run inside each worker.  Snapshots "
            "are taken at warm-up, peak (~0.6·n_steps), and end."
        ),
    )
    parser.add_argument(
        "--precompile", type=int, default=30,
        help="Untimed pre-compilation steps before measurement begins.",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=15,
        help=(
            "Number of steps to discard for torch.compile + CUDA-graph "
            "warm-up before the persistent baseline is recorded."
        ),
    )
    parser.add_argument(
        "--peak_step", type=int, default=None,
        help=(
            "Step index at which to capture the per-phase peak transient "
            "snapshots.  Defaults to int(n_steps * 0.6)."
        ),
    )
    parser.add_argument(
        "--poisson_method", type=str, default=DEFAULT_POISSON_METHOD,
        choices=["multigrid", "mgcg", "fft"],
    )
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE,
                        choices=["float32", "float64"])
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP)
    parser.add_argument("--spawn_x", type=float, default=DEFAULT_SPAWN_X)
    parser.add_argument(
        "--out_dir", default=None,
        help=(
            "Directory where worker JSON results land.  Defaults to "
            "``./results/dim<N>/`` next to this script."
        ),
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter used to launch worker subprocesses.",
    )
    parser.add_argument(
        "--n_bodies", type=int, default=1,
        help=(
            "(Worker only) Number of animat bodies to include in this run.  "
            "The driver uses --n_bodies_sweep instead."
        ),
    )
    parser.add_argument(
        "--n_bodies_sweep", type=str, default="1,2,4",
        help=(
            "(Driver only) Comma-separated list of body counts to sweep, "
            "e.g. '1,2,4,8'.  Default: '1,2,4'.  "
            "This is the key knob for studying whether memory scales with "
            "the number of bodies (python) or stays constant (kernel)."
        ),
    )
    parser.add_argument(
        "--keep_existing", action="store_true",
        help=(
            "(Driver only) Skip a mode if its JSON file already exists.  "
            "Useful for incremental sweeps."
        ),
    )
    args = parser.parse_args()

    # Fill in grid defaults from the DimensionSpec when not supplied
    spec = get_dimension_spec(args.dim)
    if args.Nx is None:
        args.Nx = spec.default_grid[0]
    if args.Ny is None:
        args.Ny = spec.default_grid[1]
    if args.dim == 3 and args.Nz is None:
        args.Nz = spec.default_grid[2]

    # Parse the body sweep into a list of ints
    args.n_bodies_sweep = [int(x) for x in args.n_bodies_sweep.split(",") if x.strip()]

    return args


# ════════════════════════════════════════════════════════════════════════
#  Memory-snapshot helpers (shared)
# ════════════════════════════════════════════════════════════════════════

def _mb(b: int) -> float:
    return b / (1024.0 ** 2)


def _snap(label: str) -> dict:
    """One memory snapshot.  CUDA-only by design — CPU footprint is not
    the bottleneck for the two paths under study."""
    import torch
    torch.cuda.synchronize()
    return {
        "label":    label,
        "alloc_mb": _mb(torch.cuda.memory_allocated()),
        "peak_mb":  _mb(torch.cuda.max_memory_allocated()),
        "rsrvd_mb": _mb(torch.cuda.memory_reserved()),
    }


def _reset_peak() -> None:
    import torch
    torch.cuda.reset_peak_memory_stats()


def _tensor_census(top_k: int = 30) -> list[dict]:
    """Walk the Python heap and group every unique CUDA storage by (shape, dtype).

    Deduplicates by untyped_storage().data_ptr() so that views / slices that
    share the same underlying allocation are counted only once.  For each
    unique storage we keep the tensor with the most elements as the
    representative shape (closest to the base allocation).
    """
    import torch
    # storage_ptr → (storage_nbytes, shape, dtype) of the largest tensor seen
    unique: dict[int, tuple] = {}
    for obj in gc.get_objects():
        try:
            if not (torch.is_tensor(obj) and obj.is_cuda):
                continue
            stor  = obj.untyped_storage()
            ptr   = stor.data_ptr()
            nbytes = stor.nbytes()
            if ptr not in unique or obj.nelement() > unique[ptr][1]:
                unique[ptr] = (nbytes, obj.nelement(), list(obj.shape), str(obj.dtype))
        except Exception:
            continue
    grouped: dict[tuple, dict] = {}
    for nbytes, _nel, shape, dtype in unique.values():
        key = (tuple(shape), dtype)
        if key not in grouped:
            grouped[key] = {"shape": shape, "dtype": dtype, "count": 0, "bytes": 0}
        grouped[key]["count"] += 1
        grouped[key]["bytes"] += nbytes
    rows = sorted(grouped.values(), key=lambda r: -r["bytes"])
    return rows[:top_k]


# ════════════════════════════════════════════════════════════════════════
#  WORKER
# ════════════════════════════════════════════════════════════════════════

def _run_worker(args: argparse.Namespace) -> None:
    """In-process FARMS run for a single (mode, n_bodies) configuration.

    Snapshots are captured at three moments — end-of-warmup, the chosen
    peak step, and the last step — so the data is comparable across modes
    without overwhelming JSON output.  A tensor census is taken at the
    peak step to identify WHICH buffers explain the difference:

    * python mode: B full-grid SDF / body-velocity tensors (shape Nx×Ny[×Nz])
      — one per body per staggered grid.  Memory grows linearly with B.
    * kernel mode: ONE union SDF (shape Nx×Ny[×Nz]) regardless of B.
      Per-body SDF tensors are computed only inside the update kernel and
      immediately discarded; the composite SDF is the only thing that
      persists between steps.
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "[memory_comparison worker] CUDA is required — this tool "
            "measures GPU memory.  CPU runs are not informative because "
            "PyTorch on CPU does not honour memory_allocated()/peak."
        )

    mode: str = args.mode
    out_dir: str = args.out_dir
    spec = get_dimension_spec(args.dim)

    # ── 1. Build the SimConfig ────────────────────────────────────────
    cfg = _build_cfg(args, mode)

    grid_str = grid_label(
        (args.Nx, args.Ny) if spec.dim == 2 else (args.Nx, args.Ny, args.Nz)
    )
    print(
        f"\n[memory_comparison worker]  dim={spec.dim}  mode={mode}  "
        f"n_bodies={args.n_bodies}  grid={grid_str}  "
        f"n_steps={args.n_steps}  warmup={args.warmup_steps}",
        flush=True,
    )

    # ── 2. Generate the FARMS configs ────────────────────────────────
    from lilytorch.util.paths import gen_new_folder, save_path
    output_folder = gen_new_folder(save_path)
    os.makedirs(output_folder, exist_ok=True)

    cfg.gen_animat_config(output_folder)
    cfg.gen_arena_config(output_folder)
    cfg.gen_simulation_config(output_folder)
    cfg.gen_experiment_config(output_folder)

    _strip_yaml_extensions(os.path.join(output_folder, "simulation_config.yaml"))

    os.chdir(output_folder)

    # ── 3. Install profiler hooks ────────────────────────────────────
    records: list[dict] = []
    census_at_peak: list[dict] = []

    def _record(label: str) -> dict:
        s = _snap(label)
        records.append(s)
        print(
            f"  [{label:>52s}]  alloc={s['alloc_mb']:8.1f} MB   "
            f"peak={s['peak_mb']:8.1f} MB   reserved={s['rsrvd_mb']:8.1f} MB",
            flush=True,
        )
        return s

    import lilytorch.integration.BDIMhandler as bmod
    from lilytorch.src import solver as smod

    _orig_init_bdim = bmod.BDIMhandler.__init__
    _orig_init_fs   = smod.FluidSolver.__init__
    _orig_step      = bmod.BDIMhandler.step

    def _profiled_init_bdim(self, *a, **kw):
        _record("BDIMhandler.__init__: enter")
        _orig_init_bdim(self, *a, **kw)
        _record("BDIMhandler.__init__: exit")

    def _profiled_init_fs(self, *a, **kw):
        _record("FluidSolver.__init__: enter")
        _orig_init_fs(self, *a, **kw)
        _record("FluidSolver.__init__: exit")

    bmod.BDIMhandler.__init__ = _profiled_init_bdim
    smod.FluidSolver.__init__ = _profiled_init_fs

    step_idx  = [0]
    peak_step = args.peak_step or max(args.warmup_steps + 5,
                                      int(args.n_steps * 0.6))

    def _profiled_step(self, task, physics):
        idx = step_idx[0]
        step_idx[0] = idx + 1

        is_warmup_end = (idx == args.warmup_steps)
        is_peak       = (idx == peak_step)
        is_end        = (idx == args.n_steps - 1)
        is_traced     = is_warmup_end or is_peak or is_end

        if not is_traced:
            return _orig_step(self, task, physics)

        # ── traced step: replicate BDIMhandler.step with phase probes.
        iteration = self.iteration
        timestep  = self.pars["solver"]["dt"]
        if iteration >= self.pars["solver"]["nt"]:
            return

        t  = iteration * timestep
        fs = self.fluid_solver

        if fs.terminate:
            self.iteration += 1
            return

        _reset_peak()
        _record(f"step {idx:03d} [{mode}]: before")

        # 1. update — body kinematics → union SDF + body-velocity grids
        self.update(t, iteration, dt=timestep)
        _record(f"step {idx:03d} [{mode}]: after update (union SDF+body_vel)")

        # 2. mu / normals
        fs._recompute_mu_normals()
        _record(f"step {idx:03d} [{mode}]: after mu/normals")

        # 3. fluid step
        if self.ndim == 3:
            (u, v, w, p) = fs.fluid_step(fs.u0, fs.v0, fs.w0, fs.p0, timestep)
            if self.zero_pressure_inside:
                p = torch.where(fs.composite_body.sdf_val < 0, 0, p)
            (fs.u0, fs.v0, fs.w0, fs.p0) = (u, v, w, p)
        else:
            (u, v, p) = fs.fluid_step(fs.u0, fs.v0, fs.p0, timestep)
            if self.zero_pressure_inside:
                p = torch.where(fs.composite_body.sdf_val < 0, 0, p)
            (fs.u0, fs.v0, fs.p0) = (u, v, p)
        _record(f"step {idx:03d} [{mode}]: after fluid_step")

        fs.check_explosion(iteration)

        # 4. forces
        if self.ndim == 3:
            fs.forces_method2_3d(fs.u0, fs.v0, fs.w0, fs.p0, iteration)
        elif self.force_method == "method1":
            fs.forces_method1(fs.u0, fs.v0, fs.p0, iteration)
        else:
            fs.forces_method2(fs.u0, fs.v0, fs.p0, iteration)
        _record(f"step {idx:03d} [{mode}]: after forces")

        from lilytorch.integration.BDIMhandler import _FS_FREE_AFTER_FORCES_3D
        if self.ndim == 3:
            fs.__dict__.update(_FS_FREE_AFTER_FORCES_3D)
        _record(f"step {idx:03d} [{mode}]: after force-field release")

        # 5. plotting (save=False — near no-op)
        if self.ndim == 3:
            fs.terminate = fs.plotting_and_saving(
                fs.u0, fs.v0, fs.p0, iteration,
                w_vel=fs.w0, check_termination=False,
            )
        else:
            fs.terminate = fs.plotting_and_saving(
                fs.u0, fs.v0, fs.p0, iteration, check_termination=False,
            )

        # 6. apply forces back to MuJoCo
        self.apply_forces(task, physics)
        _record(f"step {idx:03d} [{mode}]: after apply_forces")

        # 7. release BDIM intermediates
        # python mode: frees mu/normals + force fields + div each step.
        # kernel mode: keeps mu/normal packed buffers alive (they hold
        #   outside-body defaults; only the union AABB sub-block is
        #   overwritten next step).
        fs._release_bdim_fields()
        _record(f"step {idx:03d} [{mode}]: after release")

        # Tensor census at peak step — identifies WHICH buffers dominate.
        # Key thing to look for:
        #   python: B × (Nx×Ny×Nz) per-body SDF buffers grow linearly
        #           with the number of bodies.
        #   kernel: only ONE union SDF grid (shape Nx×Ny×Nz) regardless
        #           of how many bodies are in the scene.
        if is_peak:
            gc.collect()
            census_at_peak.extend(_tensor_census(top_k=30))

        self.iteration += 1

    bmod.BDIMhandler.step = _profiled_step

    # ── 4. Run the simulation ────────────────────────────────────────
    _reset_peak()
    _record("baseline (before any allocation)")

    from farms_core.experiment.options import ExperimentOptions
    from farms_core.extensions.extensions import import_item
    from farms_sim.simulation import run_simulation

    experiment_options = ExperimentOptions.load("experiment_config.yaml")
    experiment_data_loader = import_item(
        experiment_options.loaders.experiment_data
    )
    experiment_data = experiment_data_loader.from_options(experiment_options)
    _record("after loading experiment options")

    run_simulation(
        experiment_data=experiment_data,
        experiment_options=experiment_options,
    )
    _record("after simulation complete")

    final_peak_mb = _mb(torch.cuda.max_memory_allocated())
    final_rsrv_mb = _mb(torch.cuda.memory_reserved())

    # ── 5. Persist results ───────────────────────────────────────────
    out = {
        "mode":         mode,
        "dim":          args.dim,
        "n_bodies":     args.n_bodies,
        "Nx":           args.Nx,
        "Ny":           args.Ny,
        "Nz":           args.Nz,
        "n_steps":      args.n_steps,
        "warmup_steps": args.warmup_steps,
        "peak_step":    peak_step,
        "device":       torch.cuda.get_device_name(0),
        "torch":        torch.__version__,
        "records":      records,
        "census_at_peak": census_at_peak,
        "final_peak_mb": final_peak_mb,
        "final_rsrv_mb": final_rsrv_mb,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = _worker_result_path(out_dir, mode, args.n_bodies)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[memory_comparison worker] wrote {out_path}", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  SimConfig builder — shared between worker and driver helper
# ════════════════════════════════════════════════════════════════════════

def _build_cfg(args: argparse.Namespace, mode: str):
    """Instantiate and configure the SimConfig for the given mode.

    Applies the same Poisson / compile / grid / domain overrides as
    ``run_cost_analysis.py`` so memory numbers are directly comparable
    with the cost-analysis timing numbers.

    When ``args.n_bodies > 1``, the single-animat list is replicated with
    staggered y-offsets so that B independent bodies swim in the same
    domain.  This is the key knob for the body-scaling study:

    * **python mode**: each body gets its own full-grid SDF tensor
      (shape ``(Nx, Ny[, Nz])``), so resident memory grows as
      ``B × Nx×Ny[×Nz] × sizeof(dtype)`` — linearly in B.
    * **kernel mode**: all bodies are merged into a single union SDF
      (shape ``(Nx, Ny[, Nz])``), so resident memory is
      **O(1) in the number of bodies** regardless of B.
    """
    spec = get_dimension_spec(args.dim)
    cfg_mod = importlib.import_module(spec.config_module)
    cfg = cfg_mod.SimConfig()

    # ── grid ──────────────────────────────────────────────────────────
    cfg.use_bdim = True
    cfg.Nx = args.Nx
    cfg.Ny = args.Ny
    if spec.dim == 3:
        cfg.Nz = args.Nz

    # ── Poisson / compile (same as run_cost_analysis.py) ─────────────
    cfg.poisson_method          = args.poisson_method
    cfg.poisson_tol             = 1.0e-7
    cfg.poisson_max_cycles      = 1
    cfg.poisson_max_mgcg_cycles = 5
    cfg.poisson_precond_vcycles = 1
    cfg.poisson_warm_start      = True
    cfg.poisson_smoother        = "rbgs"
    cfg.poisson_nsmoothing      = 2
    cfg.poisson_compile         = True
    cfg.dtype                   = args.dtype
    cfg.timestep                = args.timestep
    if hasattr(cfg, "bdim_dt"):
        cfg.bdim_dt = args.timestep

    cfg.smagorinsky_cs = 0.0
    cfg.zero_pressure_inside = getattr(cfg, "zero_pressure_inside", False)

    # ── steps ─────────────────────────────────────────────────────────
    total_steps = args.n_steps + 2
    cfg.n_iterations = total_steps
    if hasattr(cfg, "bdim_nt"):
        cfg.bdim_nt = total_steps
    if hasattr(cfg, "time_integration"):
        cfg.time_integration = "euler"

    cfg.save_every  = total_steps + 999
    cfg.save_frames = False
    cfg.save        = False
    cfg.headless    = True

    # ── mode ──────────────────────────────────────────────────────────
    if mode not in _MODES:
        raise ValueError(f"unknown mode '{mode}'")
    cfg.solver_method = mode

    # ── domain sizing (same formula as run_cost_analysis.py) ─────────
    dx          = max(DX_REF, MIN_LX_FISH / args.Nx)
    lx          = args.Nx * dx
    ly          = args.Ny * dx
    spawn_x     = args.spawn_x
    body_center = spawn_x + BODY_CENTER_OFFSET
    cfg.xmin = body_center - 0.5 * lx
    cfg.xmax = body_center + 0.5 * lx
    cfg.ymin = -0.5 * ly
    cfg.ymax =  0.5 * ly
    if spec.dim == 3:
        lz = args.Nz * dx
        cfg.zmin = -0.5 * lz
        cfg.zmax =  0.5 * lz

    # ── multi-body: replicate the single animat B times ───────────────
    # Stagger bodies in y (and z in 3-D) so they do not overlap.
    # Gap = 3× fish length (1guilla ≈ 0.35 m) to keep interactions weak.
    n_bodies = args.n_bodies
    if n_bodies > 1:
        base_animat = cfg.animats_pars[0].copy()
        base_pose   = list(base_animat.get("pose", [0.0, 0.0, 0.15, 0, 0, 0]))
        body_gap    = 0.35 * 3.0   # 3× body length
        replicated  = []
        for body_idx in range(n_bodies):
            animat = base_animat.copy()
            pose   = base_pose.copy()
            if spec.dim == 2:
                pose[1] = (body_idx - (n_bodies - 1) / 2.0) * body_gap
            else:
                # In 3-D stagger in y; keep z at the base value.
                pose[1] = (body_idx - (n_bodies - 1) / 2.0) * body_gap
            animat["pose"] = pose
            replicated.append(animat)
        cfg.animats_pars = replicated

    # ── gains / control overrides ─────────────────────────────────────
    for animat in cfg.animats_pars:
        animat["gains"] = [50.0, 0.4, 0.0]
        ctrl = dict(animat.get("control_pars", {}))
        ctrl.update({"freq": 1.0, "twl": 0.571429 * 14, "amp": 15.0})
        animat["control_pars"] = ctrl
        pose = list(animat.get("pose", []))
        if pose:
            pose[0] = spawn_x
            animat["pose"] = pose

    # ── strip non-fluid extensions ────────────────────────────────────
    cfg.extra_simulation_extensions = lambda *_a, **_kw: []

    return cfg


def _strip_yaml_extensions(yaml_path: str) -> None:
    """Strip every non-fluid extension from ``simulation_config.yaml``."""
    import yaml
    with open(yaml_path, "r") as f:
        sim_dict = yaml.safe_load(f) or {}
    keep = {"lilytorch.integration.extensions.FluidExtension"}
    sim_dict["extensions"] = [
        ext for ext in sim_dict.get("extensions", [])
        if ext.get("loader", "") in keep
    ]
    sim_dict.setdefault("runtime", {})["headless"] = True
    # Enable the same compile flags used in run_cost_analysis.py so the
    # memory footprint reflects the production kernel path.
    for ext in sim_dict["extensions"]:
        bdim_yaml  = ext.get("config", {}).get("bdim_yaml", {})
        solver_cfg = bdim_yaml.get("solver", {})
        if solver_cfg:
            solver_cfg["compile_adv_diff"]   = True
            solver_cfg["poisson_compile"]    = True
            solver_cfg["poisson_compile_mode"] = "reduce-overhead"
            solver_cfg["compile_forces"]     = True
    with open(yaml_path, "w") as f:
        yaml.dump(sim_dict, f, default_flow_style=False, sort_keys=False)


# ════════════════════════════════════════════════════════════════════════
#  DRIVER
# ════════════════════════════════════════════════════════════════════════

def _run_driver(args: argparse.Namespace) -> None:
    """Spawn one subprocess per (mode × n_bodies) combination, then aggregate.

    The body-scaling sweep is the primary output of this script:

    For each body count in ``--n_bodies_sweep``, both ``python`` and
    ``kernel`` modes are run.  The final summary shows how resident memory
    (persistent baseline + step peak) scales with B for each mode.

    Expected behaviour:
      * **python**: persistent memory grows linearly with B because each
        body allocates its own full-grid SDF / body-velocity tensors.
      * **kernel**: persistent memory is approximately constant in B because
        only the union SDF is stored; per-body SDF buffers are immediately
        discarded after the union is updated.
    """
    spec = get_dimension_spec(args.dim)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 82)
    print(f" GPU memory comparison — {spec.label} pinned 1guilla (python vs kernel)")
    print("=" * 82)
    grid_str = grid_label((args.Nx, args.Ny) if spec.dim == 2
                          else (args.Nx, args.Ny, args.Nz))
    print(f"  Grid:         {grid_str}")
    print(f"  Steps:        {args.n_steps}  "
          f"(warmup={args.warmup_steps}, "
          f"peak step={args.peak_step or '~0.6·n_steps'})")
    print(f"  Body sweep:   {args.n_bodies_sweep}")
    print(f"  Poisson:      {args.poisson_method}  dtype={args.dtype}")
    print(f"  Output dir:   {out_dir}")
    print("=" * 82)

    # ── Run all (mode, n_bodies) workers ─────────────────────────────
    # Results keyed by (mode, n_bodies)
    all_results: dict[tuple[str, int], dict] = {}

    for n_bodies in args.n_bodies_sweep:
        for mode in _MODES:
            out_path = _worker_result_path(out_dir, mode, n_bodies)
            if args.keep_existing and os.path.exists(out_path):
                print(f"  [mode={mode}, B={n_bodies}] reusing {out_path}")
            else:
                print(f"\n  ── mode={mode}  n_bodies={n_bodies} "
                      + "─" * max(1, 40 - len(mode) - len(str(n_bodies))))
                cmd = [
                    args.python, os.path.abspath(__file__),
                    "--dim",          str(args.dim),
                    "--mode",         mode,
                    "--n_bodies",     str(n_bodies),
                    "--Nx",           str(args.Nx),
                    "--Ny",           str(args.Ny),
                    "--n_steps",      str(args.n_steps),
                    "--warmup_steps", str(args.warmup_steps),
                    "--poisson_method", args.poisson_method,
                    "--dtype",        args.dtype,
                    "--timestep",     str(args.timestep),
                    "--spawn_x",      str(args.spawn_x),
                    "--out_dir",      out_dir,
                ]
                if spec.dim == 3:
                    cmd += ["--Nz", str(args.Nz)]
                if args.peak_step is not None:
                    cmd += ["--peak_step", str(args.peak_step)]
                t0  = time.time()
                ret = subprocess.call(cmd)
                dt  = time.time() - t0
                if ret != 0:
                    print(f"  [mode={mode}, B={n_bodies}] subprocess failed "
                          f"(exit {ret}) — skipping")
                    continue
                print(f"  [mode={mode}, B={n_bodies}] completed in {dt:.1f}s")

            if os.path.exists(out_path):
                with open(out_path) as f:
                    all_results[(mode, n_bodies)] = json.load(f)

    if not all_results:
        print("\nNo mode completed successfully — nothing to summarise.")
        return

    _print_comparison(all_results, args)


# ════════════════════════════════════════════════════════════════════════
#  Report helpers
# ════════════════════════════════════════════════════════════════════════

def _persistent_baseline(rec: dict) -> float:
    target_label = f"step {rec['warmup_steps']:03d}"
    for r in rec["records"]:
        if r["label"].startswith(target_label) and "before" in r["label"]:
            return r["alloc_mb"]
    return rec["records"][-1]["alloc_mb"]


def _peak_step_rows(rec: dict) -> list[dict]:
    target_label = f"step {rec['peak_step']:03d}"
    return [r for r in rec["records"] if r["label"].startswith(target_label)]


def _step_peak_mb(rec: dict) -> float:
    return max((r["peak_mb"] for r in _peak_step_rows(rec)), default=float("nan"))


def _phase_deltas(rows: list[dict]) -> list[tuple[str, float, float]]:
    out = []
    for prev, cur in zip(rows, rows[1:]):
        phase = cur["label"].split(":", 1)[1].strip() if ":" in cur["label"] else cur["label"]
        out.append((phase, cur["alloc_mb"] - prev["alloc_mb"], cur["peak_mb"]))
    return out


def _print_comparison(
    all_results: dict[tuple[str, int], dict],
    args: argparse.Namespace,
) -> None:
    spec = get_dimension_spec(args.dim)
    body_counts = sorted({k[1] for k in all_results})
    modes_present = [m for m in _MODES if any(k[0] == m for k in all_results)]

    sep = "=" * 82

    # ── 1. Body-scaling table (THE key result) ────────────────────────
    print(f"\n\n{sep}")
    print(f" BODY-SCALING STUDY — {spec.label}")
    print(sep)
    print(
        "  Kernel mode stores ONE union SDF regardless of body count.\n"
        "  Python mode stores B full-grid SDF tensors → memory grows with B."
    )
    col_w = 15
    hdr = f"  {'N bodies':>9s}  {'Grid cells':>11s}" + "".join(
        f"  {'persistent [' + m + ']':>{col_w}s}  {'step peak [' + m + ']':>{col_w}s}"
        for m in modes_present
    )
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))
    for n_bodies in body_counts:
        cells_n = grid_cells(
            (args.Nx, args.Ny) if spec.dim == 2 else (args.Nx, args.Ny, args.Nz)
        )
        row = f"  {n_bodies:>9d}  {cells_n:>11,d}"
        for mode in modes_present:
            rec = all_results.get((mode, n_bodies))
            if rec is None:
                row += f"  {'—':>{col_w}s}  {'—':>{col_w}s}"
            else:
                p  = _persistent_baseline(rec)
                sp = _step_peak_mb(rec)
                row += f"  {p:>{col_w}.1f}  {sp:>{col_w}.1f}"
        print(row)

    # Print per-mode slope estimate when ≥ 2 body counts are available
    if len(body_counts) >= 2:
        print()
        for mode in modes_present:
            pts = [
                (n, _persistent_baseline(r))
                for n, r in ((nb, all_results.get((mode, nb))) for nb in body_counts)
                if r is not None
            ]
            if len(pts) >= 2:
                slope = (pts[-1][1] - pts[0][1]) / max(pts[-1][0] - pts[0][0], 1)
                print(f"  [{mode}] ΔPersistent per additional body ≈ {slope:+.1f} MB")
        print(
            "\n  Ideal (kernel): slope ≈ 0 MB/body  "
            "(constant — only union SDF stored)\n"
            "  Reference (python): slope ≈ B × grid × sizeof(dtype)  "
            "(per-body full-grid SDF)"
        )

    # ── 2. Per-mode per-phase table at n_bodies=1 (single-body detail) ─
    single_body_results = {
        mode: all_results[(mode, 1)]
        for mode in modes_present
        if (mode, 1) in all_results
    }
    if single_body_results:
        print(f"\n\n{sep}")
        print(f" SINGLE-BODY DETAIL (n_bodies=1) — per-phase snapshots")
        print(sep)

        print(f"\n{'Mode':<22s} {'Persistent MB':>14s} {'Step peak MB':>14s} "
              f"{'Final peak MB':>14s} {'Reserved MB':>14s}")
        print("-" * 78)
        for mode, rec in single_body_results.items():
            p   = _persistent_baseline(rec)
            sp  = _step_peak_mb(rec)
            fp  = rec["final_peak_mb"]
            rsv = rec["final_rsrv_mb"]
            print(f"{mode:<22s} {p:14.1f} {sp:14.1f} {fp:14.1f} {rsv:14.1f}")

        any_rec = next(iter(single_body_results.values()))
        sample_phases = [
            cur["label"].split(":", 1)[1].strip() if ":" in cur["label"] else cur["label"]
            for cur in _peak_step_rows(any_rec)[1:]
        ]
        hdr2 = f"\n{'Phase':<36s}" + "".join(f"{m:>16s}" for m in single_body_results)
        print(f"\nPer-phase PEAK alloc at peak step (MB):{hdr2}")
        print("-" * (36 + 16 * len(single_body_results)))
        for i, phase_name in enumerate(sample_phases):
            cells = []
            for mode, rec in single_body_results.items():
                rows = _peak_step_rows(rec)
                cells.append(f"{rows[i + 1]['peak_mb']:16.1f}" if i + 1 < len(rows) else f"{'—':>16s}")
            print(f"{phase_name:<36s}" + "".join(cells))

        hdr3 = f"\n{'Phase':<36s}" + "".join(f"{m:>16s}" for m in single_body_results)
        print(f"\nPer-phase DELTA alloc at peak step (MB, signed):{hdr3}")
        print("-" * (36 + 16 * len(single_body_results)))
        for i, phase_name in enumerate(sample_phases):
            cells = []
            for mode, rec in single_body_results.items():
                deltas = _phase_deltas(_peak_step_rows(rec))
                cells.append(f"{deltas[i][1]:+16.1f}" if i < len(deltas) else f"{'—':>16s}")
            print(f"{phase_name:<36s}" + "".join(cells))

    # ── 3. Tensor census (what are the largest buffers?) ─────────────
    # Show for each body count so the user can see the per-body SDF
    # tensors appear / grow in python mode but stay flat in kernel mode.
    print(f"\n\n{sep}")
    print(" TENSOR CENSUS AT PEAK STEP (top 10 by size)")
    print(sep)
    print(
        "  Watch for tensors whose 'count' grows with N bodies in python\n"
        "  mode but stays at 1 in kernel mode: those are the per-body SDF\n"
        "  / body-velocity buffers that the union SDF eliminates.\n"
    )
    for n_bodies in body_counts:
        for mode in modes_present:
            rec = all_results.get((mode, n_bodies))
            if rec is None or not rec.get("census_at_peak"):
                continue
            print(f"\n  [mode={mode}  n_bodies={n_bodies}]")
            print(f"  {'Shape':<38s}  {'Dtype':<18s}  {'Count':>5s}  {'Total MB':>10s}")
            print("  " + "-" * 78)
            for row in rec["census_at_peak"][:10]:
                shape = "(" + ", ".join(str(s) for s in row["shape"]) + ")"
                print(f"  {shape:<38s}  {row['dtype']:<18s}  "
                      f"{row['count']:>5d}  {_mb(row['bytes']):>10.1f}")

    # ── 4. python vs kernel savings at n_bodies=1 ────────────────────
    if "python" in single_body_results and "kernel" in single_body_results:
        ref_rec    = single_body_results["python"]
        ref_p      = _persistent_baseline(ref_rec)
        ref_sp     = _step_peak_mb(ref_rec)
        kernel_rec = single_body_results["kernel"]
        k_p        = _persistent_baseline(kernel_rec)
        k_sp       = _step_peak_mb(kernel_rec)
        print(f"\n\n{sep}")
        print(f" KERNEL vs PYTHON SAVINGS (n_bodies=1)")
        print(sep)
        print(f"  ΔPersistent = {k_p - ref_p:+.1f} MB   "
              f"ΔStep peak = {k_sp - ref_sp:+.1f} MB   "
              f"({(k_sp - ref_sp) / max(ref_sp, 1.0) * 100:+.1f}% of python step peak)")


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()
    spec = get_dimension_spec(args.dim)

    if args.out_dir is None:
        args.out_dir = os.path.join(
            _SCRIPT_DIR, "results", spec.short_tag,
        )
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode is not None:
        # worker mode — n_bodies is a single int passed by the driver
        _run_worker(args)
    else:
        # driver mode — expand n_bodies_sweep and run all combinations
        _run_driver(args)


if __name__ == "__main__":
    main()
