#!/usr/bin/env python3
"""
GPU memory comparison across solver modes — 2-D and 3-D pinned 1guilla.

.. note:: PARKED — the python/kernel ``solver_method`` modes were collapsed
   into the single fused path (``solver_method`` is deprecated and ignored),
   so the two "modes" below now run the identical solver and the comparison
   is moot.  The script is kept for its per-phase / per-tensor memory
   instrumentation, which still works on the fused path.

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

# Expandable segments cut allocator-pool fragmentation drastically for workloads
# that repeatedly allocate/free tensors of different shapes within one step
# (advdiff primes, SDF/body-vel temps, multigrid clones, gradient buffers).
# Must be set BEFORE torch initializes CUDA, hence at module top.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
        "--n_bodies_sweep", type=str, default="1",
        # "--n_bodies_sweep", type=str, default="1,2,4",
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
    parser.add_argument(
        "--mem_dbg", action="store_true", default=False,
        help=(
            "Enable LILYTORCH_MEM_DBG=1 inside _fluid_step_kernel_3d so the "
            "per-sub-step alloc/peak is printed.  This is the only way to see "
            "which phase (Kernel A, Kernel B, projection, V-cycle) drives the "
            "transient peak that the post-step gc-based census cannot capture."
        ),
    )
    args = parser.parse_args()
    # Propagate the MEM_DBG flag to the C++/CUDA solver via env var.
    if getattr(args, "mem_dbg", False):
        os.environ["LILYTORCH_MEM_DBG"] = "1"

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


def _build_name_map(*objs) -> dict[int, str]:
    """Map CUDA storage data_ptr → attribute name by inspecting object __dict__s.

    Pass any live Python objects (e.g. a FluidSolver and a BDIMhandler instance)
    and the function returns a dict from storage pointer to the *first* matching
    attribute name found.  When multiple attributes share the same storage
    (views / aliases) the names are joined with '/'.

    Also walks ``composite_body.bodies[i]`` (with prefix ``bodies[i].``) so that
    per-body SDF interpolation caches and similar nested tensors are identified.

    FakeTensors (from ``torch.compile`` tracing) are silently skipped because
    they carry no real GPU allocation.
    """
    import torch
    try:
        from torch._subclasses.fake_tensor import FakeTensor as _FakeTensor
    except ImportError:
        _FakeTensor = None  # older torch — no FakeTensor

    def _is_real_cuda(val) -> bool:
        if not (torch.is_tensor(val) and val.is_cuda):
            return False
        if _FakeTensor is not None and isinstance(val, _FakeTensor):
            return False
        return True

    name_map: dict[int, str] = {}

    def _register(attr_path: str, val) -> None:
        try:
            if not _is_real_cuda(val):
                return
            ptr = val.untyped_storage().data_ptr()
            if ptr in name_map:
                if attr_path not in name_map[ptr].split("/"):
                    name_map[ptr] = name_map[ptr] + "/" + attr_path
            else:
                name_map[ptr] = attr_path
        except Exception:
            pass

    def _walk(obj, prefix: str = "") -> None:
        # Handle plain dicts (e.g. _kernel_static_3d, _kernel_step,
        # _stream_meta) whose tensor values are NOT reachable via vars().
        if isinstance(obj, dict):
            for key, val in obj.items():
                full = (prefix + str(key)) if prefix else str(key)
                _register(full, val)
            return
        # Handle lists/tuples (e.g. _dct_idx, _dct_twiddle in FFT Poisson).
        if isinstance(obj, (list, tuple)):
            for i, val in enumerate(obj):
                _register(f"{prefix}[{i}]" if prefix else f"[{i}]", val)
            return
        try:
            items = list(vars(obj).items())
        except TypeError:
            return
        for attr, val in items:
            full = (prefix + attr) if prefix else attr
            _register(full, val)

    def _walk_sub(obj, prefix: str = "") -> None:
        """Walk one level of sub-objects (both regular objects and dicts)."""
        if isinstance(obj, dict):
            iterator = obj.items()
        else:
            try:
                iterator = vars(obj).items()
            except TypeError:
                return
        for attr, sub in iterator:
            if sub is None or torch.is_tensor(sub):
                continue
            _walk(sub, prefix=f"{prefix}{attr}.")
            # One more level for dict-valued sub-objects (e.g. _fused_bc_cache)
            # and for nested sub-objects of known deep structures.
            if isinstance(sub, dict):
                for k2, v2 in sub.items():
                    if v2 is None or torch.is_tensor(v2):
                        continue
                    _walk(v2, prefix=f"{prefix}{attr}.{k2}.")

    for obj in objs:
        _walk(obj)
        # also walk direct sub-objects of each top-level object, including
        # dict-valued attributes like _kernel_static_3d / _kernel_step
        _walk_sub(obj)
        # one extra level for known deep sub-objects (Poisson / adv_diff
        # solvers carry 3-level-deep caches with small auxiliary tensors)
        _DEEP_ATTRS = ('poisson_solverFFT', 'poisson_solverMG',
                       'adv_diff_solver', 'poisson_solver')
        for deep_attr in _DEEP_ATTRS:
            deep_sub = getattr(obj, deep_attr, None)
            if deep_sub is not None:
                _walk_sub(deep_sub, prefix=f"{deep_attr}.")
        # dive into composite_body.bodies[i] and their .sdf interpolator
        comp = getattr(obj, "composite_body", None) or getattr(
            getattr(obj, "fluid_solver", None), "composite_body", None)
        if comp is not None:
            _walk(comp, prefix="composite_body.")
            _walk_sub(comp, prefix="composite_body.")
            bodies = getattr(comp, "bodies", []) or []
            for i, body in enumerate(bodies):
                _walk(body, prefix=f"bodies[{i}].")
                # one more level: e.g. body.sdf._F (RegularGridInterpolator)
                # and body._stream_meta (dict with per-body kernel data)
                for sub_attr in vars(body):
                    sub_val = getattr(body, sub_attr, None)
                    if sub_val is not None and not torch.is_tensor(sub_val):
                        _walk(sub_val, prefix=f"bodies[{i}].{sub_attr}.")

    # Fallback: for any large CUDA tensor not yet mapped, use gc.get_referrers
    # to find the owner object and attribute name.
    import gc as _gc
    for t in _gc.get_objects():
        try:
            if not _is_real_cuda(t):
                continue
            ptr = t.untyped_storage().data_ptr()
            if ptr in name_map:
                continue  # already identified
            if t.nelement() < 1_000_000:
                continue  # skip small tensors for performance
            found = False
            for ref in _gc.get_referrers(t):
                if isinstance(ref, dict):
                    for rref in _gc.get_referrers(ref):
                        if hasattr(rref, "__dict__") and rref.__dict__ is ref:
                            for attr_name, val in ref.items():
                                if val is t:
                                    name_map[ptr] = f"{type(rref).__name__}.{attr_name}"
                                    found = True
                                    break
                        if found:
                            break
                elif isinstance(ref, (list, tuple)):
                    for rref in _gc.get_referrers(ref):
                        if hasattr(rref, "__dict__"):
                            for attr_name, val in vars(rref).items():
                                if val is ref:
                                    idx = None
                                    try:
                                        idx = list(ref).index(t)
                                    except ValueError:
                                        pass
                                    if idx is not None:
                                        name_map[ptr] = (
                                            f"{type(rref).__name__}.{attr_name}[{idx}]"
                                        )
                                        found = True
                                    break
                        if found:
                            break
                if found:
                    break
        except Exception:
            pass

    # Last-resort: label remaining large tensors by their shape so the
    # census can at least print a useful heuristic name.  CUDA-graph pool
    # tensors (from torch.compile reduce-overhead) are not reachable via
    # normal object graphs; we recognise them by checking whether any
    # referrer is a CompiledFunction / _CudaGraphCache type.
    _compile_type_names = frozenset((
        "CompiledFunction", "_CudaGraphCache", "CUDAGraphCache",
        "_CompiledAutograd", "_InductorGraphCache",
    ))
    for t in _gc.get_objects():
        try:
            if not (torch.is_tensor(t) and t.is_cuda):
                continue
            ptr = t.untyped_storage().data_ptr()
            if ptr in name_map:
                continue
            if t.nelement() < 1_000_000:
                continue
            # Check if any referrer looks like a torch.compile artifact
            for ref in _gc.get_referrers(t):
                ref_type = type(ref).__name__
                if ref_type in _compile_type_names:
                    name_map[ptr] = f"cuda_graph_pool<{ref_type}>"
                    break
                # Also catch lists/tuples that are graph I/O buffers
                if isinstance(ref, (list, tuple)):
                    for rref in _gc.get_referrers(ref):
                        if type(rref).__name__ in _compile_type_names:
                            name_map[ptr] = f"cuda_graph_pool<{type(rref).__name__}>"
                            break
                if ptr in name_map:
                    break
        except Exception:
            pass

    return name_map


def _tensor_census(top_k: int = 30,
                   name_map: dict[int, str] | None = None) -> list[dict]:
    """Walk the Python heap and group every unique CUDA storage by (shape, dtype).

    Deduplicates by untyped_storage().data_ptr() so that views / slices that
    share the same underlying allocation are counted only once.  For each
    unique storage we keep the tensor with the most elements as the
    representative shape (closest to the base allocation).

    If ``name_map`` is provided (storage_ptr → attr_name string), the matching
    attribute names are stored in each census row under the key ``"names"``.
    """
    import torch
    try:
        from torch._subclasses.fake_tensor import FakeTensor as _FakeTensor
    except ImportError:
        _FakeTensor = None
    # storage_ptr → (storage_nbytes, nel, shape, dtype) of the largest tensor seen
    unique: dict[int, tuple] = {}
    for obj in gc.get_objects():
        try:
            if not (torch.is_tensor(obj) and obj.is_cuda):
                continue
            if _FakeTensor is not None and isinstance(obj, _FakeTensor):
                continue  # skip torch.compile tracing artefacts
            stor   = obj.untyped_storage()
            ptr    = stor.data_ptr()
            # Use nelement × element_size, NOT stor.nbytes().
            # stor.nbytes() returns the full parent-storage size even for
            # views/slices, causing massive double-counting when a slice
            # (e.g. normal_x = _mu_pack[0]) outlives the parent tensor.
            # nelement × element_size equals the representative tensor's
            # actual data footprint, so the census total matches
            # torch.cuda.memory_allocated().
            nbytes = obj.nelement() * obj.element_size()
            if ptr not in unique or obj.nelement() > unique[ptr][1]:
                unique[ptr] = (nbytes, obj.nelement(), list(obj.shape), str(obj.dtype))
        except Exception:
            continue
    grouped: dict[tuple, dict] = {}
    for ptr, (nbytes, _nel, shape, dtype) in unique.items():
        key = (tuple(shape), dtype)
        if key not in grouped:
            grouped[key] = {"shape": shape, "dtype": dtype, "count": 0,
                            "bytes": 0, "names": []}
        grouped[key]["count"] += 1
        grouped[key]["bytes"] += nbytes
        if name_map and ptr in name_map:
            name = name_map[ptr]
            if name not in grouped[key]["names"]:
                grouped[key]["names"].append(name)
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
    # Mutable containers so the inner closure can write back to the outer scope
    census_at_peak_alloc: list[float] = [0.0]   # alloc_mb at census time
    census_at_peak_rsrv:  list[float] = [0.0]   # reserved_mb at census time
    census_at_peak_nvml:  list[float] = [None]  # nvidia-smi MiB at census time
    peak_step_nvml:       list[float] = [None]  # nvidia-smi MiB right after fluid_step (before cleanup)
    # TRUE-peak census — taken right after fluid_step (before forces & release),
    # so it includes the transient buffers responsible for the nvidia-smi peak.
    # The pre-existing 'census_at_peak' is taken AFTER release+empty_cache and
    # only captures the PERSISTENT baseline. Compare the two to see exactly
    # which tensors caused the observed peak.
    census_at_true_peak: list[dict] = []
    census_at_true_peak_alloc: list[float] = [0.0]
    census_at_true_peak_rsrv:  list[float] = [0.0]
    census_at_true_peak_nvml:  list[float] = [None]

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
    peak_step = min(
        args.peak_step or max(args.warmup_steps + 5, int(args.n_steps * 0.6)),
        args.n_steps - 1,
    )

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
        # The fused step computes mu0/mu1 and normals in registers inside
        # bdim_apply; the python full-grid pack is only built when the
        # python force readout needs it (mirrors advance_and_compute_loads).
        if fs._needs_python_mu_normals():
            fs._recompute_mu_normals()
            _record(f"step {idx:03d} [{mode}]: after mu/normals")
        else:
            _record(f"step {idx:03d} [{mode}]: mu/normals (computed in fluid_step kernel)")

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

        # Capture nvidia-smi at peak (before forces + cleanup).
        # This is the maximum this process consumes; add system background
        # processes (Xorg, GNOME, etc.) to get what `watch nvidia-smi` shows.
        if is_peak and peak_step_nvml[0] is None:
            try:
                import subprocess as _sp_peak, os as _os_peak
                _r_peak = _sp_peak.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                _pid_str_peak = str(_os_peak.getpid())
                for _ln2 in _r_peak.stdout.strip().split("\n"):
                    _p2 = _ln2.strip().split(",")
                    if len(_p2) == 2 and _p2[0].strip() == _pid_str_peak:
                        peak_step_nvml[0] = float(_p2[1].strip())
                        break
            except Exception:
                pass

        # TRUE-peak tensor census — taken HERE (right after fluid_step, BEFORE
        # release/empty_cache) so transient tensors that survive past the kernel
        # but are still alive (force outputs, pressure-correction temps, divergence
        # buffers, multigrid coarse-level allocations not yet GC'd) are visible.
        # This is the answer to "what occupies the 14 GB the user saw in nvidia-smi".
        if is_peak:
            gc.collect()
            _nm_true = _build_name_map(fs, self)
            census_at_true_peak.extend(_tensor_census(top_k=40, name_map=_nm_true))
            import torch as _t_true
            census_at_true_peak_alloc[0] = _mb(_t_true.cuda.memory_allocated())
            census_at_true_peak_rsrv[0]  = _mb(_t_true.cuda.memory_reserved())
            try:
                import subprocess as _sp_true, os as _os_true
                _r_true = _sp_true.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                _pid_str_true = str(_os_true.getpid())
                for _ln_true in _r_true.stdout.strip().split("\n"):
                    _p_true = _ln_true.strip().split(",")
                    if len(_p_true) == 2 and _p_true[0].strip() == _pid_str_true:
                        census_at_true_peak_nvml[0] = float(_p_true[1].strip())
                        break
            except Exception:
                pass

        fs.check_explosion(iteration)

        # 4. forces
        if self.ndim == 3:
            fs.forces_method2_3d(fs.u0, fs.v0, fs.w0, fs.p0, iteration)
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
        # Flush the CUDA caching allocator (mirrors step_() behaviour).
        # This returns freed blocks from _release_bdim_fields + forces
        # back to the driver so nvidia-smi / memory_reserved() reflects
        # the true working set rather than a high-water-mark cache.
        if fs.device.type == "cuda":
            torch.cuda.empty_cache()
        _record(f"step {idx:03d} [{mode}]: after release")

        # Tensor census at peak step — identifies WHICH buffers dominate.
        # Key thing to look for:
        #   python: B × (Nx×Ny×Nz) per-body SDF buffers grow linearly
        #           with the number of bodies.
        #   kernel: only ONE union SDF grid (shape Nx×Ny×Nz) regardless
        #           of how many bodies are in the scene.
        # Census at 'after fluid_step / before release' — captures the
        # persistent fields PLUS any force-output tensors that are still
        # alive at this point.  Taken before empty_cache so the caching
        # allocator pool is still warm.  Stored separately from the
        # post-cleanup census so the caller can compare the two.
        if is_peak:
            gc.collect()
            nm = _build_name_map(fs, self)
            census_at_peak.extend(_tensor_census(top_k=30, name_map=nm))
            # Also record alloc/reserved at the moment the census is taken
            # (after release + empty_cache, i.e. the persistent baseline)
            import torch as _t_census
            census_at_peak_alloc[0] = _mb(_t_census.cuda.memory_allocated())
            census_at_peak_rsrv[0]  = _mb(_t_census.cuda.memory_reserved())
            # nvidia-smi measured HERE so all 4 layers are at the same step.
            # subprocess is safe between steps (no GPU kernel running).
            try:
                import subprocess as _sp_census, os as _os_census
                _r_nvml = _sp_census.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                _pid_str = str(_os_census.getpid())
                for _ln in _r_nvml.stdout.strip().split("\n"):
                    _p = _ln.strip().split(",")
                    if len(_p) == 2 and _p[0].strip() == _pid_str:
                        census_at_peak_nvml[0] = float(_p[1].strip())
                        break
            except Exception:
                pass

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

    # Flush GC + allocator so final_rsrv_mb reflects live tensors only,
    # not freed-but-cached blocks that accumulated since the last step's
    # empty_cache() call.
    gc.collect()
    torch.cuda.empty_cache()

    final_peak_mb  = _mb(torch.cuda.max_memory_allocated())
    final_alloc_mb = _mb(torch.cuda.memory_allocated())
    final_rsrv_mb  = _mb(torch.cuda.memory_reserved())

    # nvidia-smi "used" for this process — includes PyTorch pool + CUDA
    # runtime + compiled Triton CUBIN code + MuJoCo GPU buffers etc.
    # This is the number the user sees in nvidia-smi.
    _nvml_mb = None
    try:
        import pynvml
        pynvml.nvmlInit()
        _handle = pynvml.nvmlDeviceGetHandleByIndex(
            torch.cuda.current_device()
        )
        _procs = pynvml.nvmlDeviceGetComputeRunningProcesses(_handle)
        _pid = os.getpid()
        for _p in _procs:
            if _p.pid == _pid:
                _nvml_mb = _p.usedGpuMemory / (1024 ** 2)
                break
    except Exception:
        pass  # pynvml not available or failed — not critical

    # Post-simulation tensor census (after cleanup) so we can see
    # exactly what's live at steady state.
    _fs_live = None
    _bh_live = None
    try:
        import lilytorch.integration.BDIMhandler as _bmod2
        import gc as _gc2
        for _o in _gc2.get_objects():
            if isinstance(_o, _bmod2.BDIMhandler):
                _bh_live = _o
                _fs_live = getattr(_o, "fluid_solver", None)
                break
    except Exception:
        pass
    nm_final = _build_name_map(*filter(None, [_fs_live, _bh_live]))
    census_final = _tensor_census(top_k=20, name_map=nm_final)

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
        "census_at_peak_alloc_mb": census_at_peak_alloc[0],
        "census_at_peak_rsrv_mb":  census_at_peak_rsrv[0],
        "census_at_peak_nvml_mb":  census_at_peak_nvml[0],
        "peak_step_nvml_mb":       peak_step_nvml[0],
        # TRUE-peak census (taken at the actual nvidia-smi peak moment,
        # right after fluid_step, BEFORE release+empty_cache).
        "census_at_true_peak": census_at_true_peak,
        "census_at_true_peak_alloc_mb": census_at_true_peak_alloc[0],
        "census_at_true_peak_rsrv_mb":  census_at_true_peak_rsrv[0],
        "census_at_true_peak_nvml_mb":  census_at_true_peak_nvml[0],
        "census_final":  census_final,
        "final_peak_mb":  final_peak_mb,
        "final_alloc_mb": final_alloc_mb,
        "final_rsrv_mb":  final_rsrv_mb,
        "final_nvml_mb":  _nvml_mb,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = _worker_result_path(out_dir, mode, args.n_bodies)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[memory_comparison worker] wrote {out_path}", flush=True)

    # ── 6. Inline summary (mirrors the driver's _print_comparison) ───
    sep = "=" * 82
    persistent_mb = _persistent_baseline(out)
    step_peak     = _step_peak_mb(out)

    # Extract between-step reserved from records (the "before" snapshot of
    # the peak step has alloc=persistent_mb and reflects the allocator pool
    # between steps, including Inductor static buffers when compile=True).
    between_step_rsrv = None
    for rec in records:
        label = rec.get("label", "")
        if "before" in label and abs(rec["alloc_mb"] - persistent_mb) < 10:
            between_step_rsrv = rec["rsrvd_mb"]
            break
    # Also find peak reserved from the after-fluid_step snapshot
    peak_rsrv = None
    for rec in records:
        label = rec.get("label", "")
        if "after fluid_step" in label or "after_fluid_step" in label:
            if peak_rsrv is None or rec["rsrvd_mb"] > peak_rsrv:
                peak_rsrv = rec["rsrvd_mb"]

    # nvidia-smi for this process — use pynvml result if available,
    # else the step-hook subprocess result (measured at the census step,
    # so all 4 layers are at the same point in time).
    _nvml_final = _nvml_mb if _nvml_mb is not None else census_at_peak_nvml[0]
    _nvml_peak  = out.get("peak_step_nvml_mb") or peak_step_nvml[0]

    print(f"\n{sep}")
    print(f" MEMORY SUMMARY — mode={mode}  dim={args.dim}  "
          f"grid={grid_label((args.Nx, args.Ny) if args.dim == 2 else (args.Nx, args.Ny, args.Nz))}"
          f"  n_bodies={args.n_bodies}")
    print(sep)
    print(f"  Persistent baseline (after warmup):  {persistent_mb:8.1f} MB  [torch.cuda.memory_allocated()]")
    print(f"  Step peak (alloc):                   {step_peak:8.1f} MB  [torch.cuda.max_memory_allocated()]")
    if peak_rsrv is not None:
        print(f"  Step peak (reserved):                {peak_rsrv:8.1f} MB  [torch.cuda.memory_reserved() at peak]")
    if between_step_rsrv is not None:
        _inductor_pool = between_step_rsrv - persistent_mb
        print(f"  Between-step reserved:               {between_step_rsrv:8.1f} MB  "
              f"[persistent {persistent_mb:.0f} + pool {_inductor_pool:.0f} MB]")
    print()

    # ── 4-layer memory stack ─────────────────────────────────────────
    # Two checkpoints: BETWEEN STEPS and AT PEAK STEP (after fluid_step, before cleanup).
    # The "watch nvidia-smi" total you see in the terminal includes other GPU processes
    # (Xorg, GNOME compositor, other terminals, etc.) on top of this process.
    print(f"  ── Memory stack — THIS PROCESS ONLY (not system total) ──────────────────")
    _census_total_peak = sum(r["bytes"] for r in census_at_peak) / (1024**2) if census_at_peak else None
    _census_alloc = census_at_peak_alloc[0]
    _census_rsrv  = census_at_peak_rsrv[0]

    print(f"  Checkpoint A: BETWEEN STEPS (after empty_cache) ─────────────────────────")
    if _census_total_peak is not None:
        print(f"  [1] census (gc-visible Python tensors):  {_census_total_peak:8.1f} MB")
        _gap_census_alloc = _census_alloc - _census_total_peak
        if _gap_census_alloc >= 0:
            _gap_census_note = "(C++-owned objects not in gc: MuJoCo interop, torch internals)"
        else:
            _gap_census_note = ("(compile=True: Inductor wrapper tensors in gc but "
                                "storage is in CUDA graph pool → not in alloc)")
        print(f"      └─ gap to alloc = {_gap_census_alloc:+.1f} MB  {_gap_census_note}")
    print(f"  [2] allocated (torch.cuda.memory_allocated()): {_census_alloc:8.1f} MB")
    _gap_alloc_rsrv = _census_rsrv - _census_alloc
    print(f"      └─ gap to reserved = {_gap_alloc_rsrv:+.1f} MB  "
          f"(caching allocator pool; Inductor static bufs when compile=True)")
    print(f"  [3] reserved (torch.cuda.memory_reserved()):   {_census_rsrv:8.1f} MB")
    if _nvml_final is not None:
        _gap_rsrv_nvml = _nvml_final - _census_rsrv
        print(f"      └─ gap to nvidia-smi = {_gap_rsrv_nvml:+.1f} MB  "
              f"(CUDA context + CUBIN code — constant regardless of workload)")
        print(f"  [4] nvidia-smi THIS process (between steps): {_nvml_final:8.1f} MB")
    else:
        print(f"  [4] nvidia-smi (between steps): n/a  "
              f"(grep PID {os.getpid()} in:  nvidia-smi --query-compute-apps=pid,used_memory "
              f"--format=csv,noheader,nounits)")

    print()
    print(f"  Checkpoint B: AT PEAK STEP (right after fluid_step, before cleanup) ─────")
    if peak_rsrv is not None:
        print(f"  [3] reserved (torch.cuda.memory_reserved()):   {peak_rsrv:8.1f} MB")
    if _nvml_peak is not None:
        _ctx_est = _nvml_final - _census_rsrv if _nvml_final is not None else None
        print(f"  [4] nvidia-smi THIS process (at peak step):  {_nvml_peak:8.1f} MB")
        if peak_rsrv is not None:
            _gap_peak = _nvml_peak - peak_rsrv
            print(f"      └─ gap reserved→nvml = {_gap_peak:+.1f} MB  (same CUDA context constant as above)")
        print()
        print(f"  NOTE: 'watch nvidia-smi' or the default nvidia-smi output shows the SYSTEM TOTAL")
        print(f"        (all processes on the GPU).  Subtract background (Xorg, compositor, etc.)")
        print(f"        to get this process's contribution.  Typical background: 500–1500 MB.")
        if _nvml_peak is not None:
            print(f"  => If you see ~{_nvml_peak/1024:.1f} GB in nvidia-smi that is just this process.")
            print(f"     Add desktop GPU processes to get the system total you observe.")
    else:
        print(f"  [4] nvidia-smi (peak step): n/a")
        print()
        print(f"  NOTE: 'watch nvidia-smi' shows the SYSTEM TOTAL (all GPU processes).")
        print(f"        This process peak ≈ reserved({peak_rsrv:.0f} MB) + CUDA context(~286 MB)"
              f" = {(peak_rsrv or 0) + 286:.0f} MB ≈ {((peak_rsrv or 0) + 286)/1024:.1f} GB.")
        print(f"        Add desktop GPU processes (Xorg, compositor, etc.) to get system total.")
    print()

    peak_rows = _peak_step_rows(out)
    if len(peak_rows) > 1:
        print(f"Per-phase breakdown at peak step (step {out['peak_step']:03d}):")
        print(f"  {'Phase':<44s}  {'Alloc MB':>10s}  {'Delta MB':>10s}  {'Peak MB':>10s}  {'Rsvd MB':>10s}")
        print("  " + "-" * 96)
        for prev, cur in zip(peak_rows, peak_rows[1:]):
            phase = (cur["label"].split(":", 1)[1].strip()
                     if ":" in cur["label"] else cur["label"])
            delta = cur["alloc_mb"] - prev["alloc_mb"]
            print(f"  {phase:<44s}  {cur['alloc_mb']:>10.1f}  {delta:>+10.1f}  "
                  f"{cur['peak_mb']:>10.1f}  {cur['rsrvd_mb']:>10.1f}")

    # ── TRUE-peak tensor census (taken at the actual nvidia-smi peak) ────
    # This is the answer to "what occupies the GB the user saw in nvidia-smi".
    # It captures BOTH the persistent fields AND every transient tensor that
    # was still alive when fluid_step returned (force-output temps, multigrid
    # coarse-level allocations, division-buffer intermediates, etc.).
    if census_at_true_peak:
        _true_total = sum(r["bytes"] for r in census_at_true_peak) / (1024 ** 2)
        _true_alloc = census_at_true_peak_alloc[0]
        _true_rsrv  = census_at_true_peak_rsrv[0]
        _true_nvml  = census_at_true_peak_nvml[0]
        print(f"\nTensor census at TRUE PEAK (right after fluid_step, BEFORE cleanup):")
        print(f"  alloc={_true_alloc:.1f} MB  reserved={_true_rsrv:.1f} MB  "
              f"nvml={_true_nvml if _true_nvml is not None else 'n/a'} MB  "
              f"(census-visible total: {_true_total:.1f} MB)")
        _gap_true = _true_alloc - _true_total
        if abs(_gap_true) > 50:
            print(f"  └─ alloc − census = {_gap_true:+.1f} MB "
                  f"(C++-owned / not gc-visible: cuBLAS workspace, MuJoCo interop, "
                  f"torch.compile buffers if compile=True)")
        print(f"  {'Shape':<32s}  {'Dtype':<14s}  {'Count':>5s}  {'Total MB':>9s}  Attribute names")
        print("  " + "-" * 100)
        for row in census_at_true_peak[:20]:
            shape = "(" + ", ".join(str(s) for s in row["shape"]) + ")"
            names = ", ".join(row.get("names", []))
            print(f"  {shape:<32s}  {row['dtype']:<14s}  "
                  f"{row['count']:>5d}  {_mb(row['bytes']):>9.1f}  {names}")
        if len(census_at_true_peak) > 20:
            _remaining = sum(r["bytes"] for r in census_at_true_peak[20:]) / (1024**2)
            print(f"  ... ({len(census_at_true_peak)-20} more groups, {_remaining:.1f} MB)")

    if census_at_peak:
        _census_total_str = f"{_census_total_peak:.1f} MB" if _census_total_peak else "?"
        print(f"\nTensor census at peak step (AFTER cleanup, {len(census_at_peak)} groups, "
              f"total={_census_total_str}; alloc at census={_census_alloc:.1f} MB):")
        print(f"  NOTE: this is the PERSISTENT BASELINE (after release+empty_cache).")
        print(f"        For the actual peak working set, see TRUE PEAK census above.")
        print(f"  {'Shape':<32s}  {'Dtype':<14s}  {'Count':>5s}  {'Total MB':>9s}  Attribute names")
        print("  " + "-" * 100)
        for row in census_at_peak[:10]:
            shape = "(" + ", ".join(str(s) for s in row["shape"]) + ")"
            names = ", ".join(row.get("names", []))
            print(f"  {shape:<32s}  {row['dtype']:<14s}  "
                  f"{row['count']:>5d}  {_mb(row['bytes']):>9.1f}  {names}")
        if len(census_at_peak) > 10:
            _remaining = sum(r["bytes"] for r in census_at_peak[10:]) / (1024**2)
            print(f"  ... ({len(census_at_peak)-10} more groups, {_remaining:.1f} MB)")

    # ── Memory budget decomposition (the answer to "where do the GB go") ───
    # Three distinct allocation snapshots matter and they are NOT the same:
    #   1. census/alloc AT the snapshot moment (right after fluid_step)
    #      — what gc.get_objects() can find and what memory_allocated() reports
    #   2. peak alloc DURING the step (max_memory_allocated, since reset)
    #      — captures transients that lived briefly inside fluid_step and were
    #        freed before the snapshot ran (Python locals: primes, sdf/body
    #        temps, multigrid V-cycle clones, the inline projection diff, etc.)
    #   3. peak reserved (memory_reserved at the snapshot, which equals the
    #      caching-allocator high-water mark) and nvidia-smi (reserved+context)
    if census_at_true_peak:
        _persistent       = _census_alloc      # alloc post-cleanup ≈ persistent
        _snapshot_alloc   = _true_alloc        # alloc when census ran
        _peak_alloc_in_step = step_peak        # max_memory_allocated during step
        _peak_rsrv_in_step  = (peak_rsrv if peak_rsrv is not None
                               else _true_rsrv)
        _peak_nvml        = _true_nvml         # nvml at snapshot ≈ peak
        _post_step_trans  = _snapshot_alloc - _persistent
        _freed_in_step    = max(0.0, _peak_alloc_in_step - _snapshot_alloc)
        _pool_gap         = _peak_rsrv_in_step - _peak_alloc_in_step
        _ctx_gap          = (_peak_nvml - _peak_rsrv_in_step
                             if _peak_nvml is not None else None)
        print(f"\n── DECOMPOSITION OF PEAK MEMORY (what nvidia-smi shows) ────────────────")
        print(f"  IMPORTANT: the TRUE PEAK census above shows only tensors STILL ALIVE")
        print(f"  at snapshot time. Transients allocated and freed inside fluid_step are")
        print(f"  invisible to gc but still drove the actual peak.")
        print()
        print(f"  [persistent fields, post-cleanup]      {_persistent:8.1f} MB"
              f"  ← gc-visible in TRUE PEAK census")
        print(f"  [+ still-live transients at snapshot]  +{_post_step_trans:8.1f} MB"
              f"  ← gc-visible in TRUE PEAK census")
        print(f"  ──────────────────────────────────────────────────────────")
        print(f"  = alloc AT snapshot                    {_snapshot_alloc:8.1f} MB"
              f"  [= census total above]")
        print(f"  [+ freed-inside-fluid_step transients] +{_freed_in_step:8.1f} MB"
              f"  ← INVISIBLE to gc (primes, sdf/body temps,")
        print(f"  {'':41s}      multigrid clones, projection diff, etc.)")
        print(f"  ──────────────────────────────────────────────────────────")
        print(f"  = peak alloc DURING step               {_peak_alloc_in_step:8.1f} MB"
              f"  [from max_memory_allocated]")
        print(f"  [+ caching-allocator pool overhead]    +{_pool_gap:8.1f} MB"
              f"  (freed slabs kept warm by the allocator)")
        print(f"  ──────────────────────────────────────────────────────────")
        print(f"  = peak reserved (PyTorch holds)        {_peak_rsrv_in_step:8.1f} MB"
              f"  [from memory_reserved]")
        if _ctx_gap is not None:
            print(f"  [+ CUDA context / CUBIN / cuBLAS]      +{_ctx_gap:8.1f} MB"
                  f"  (constant ~286 MB; cuBLAS handle etc.)")
            print(f"  ──────────────────────────────────────────────────────────")
            print(f"  = nvidia-smi (this process)            {_peak_nvml:8.1f} MB")
        print()
        print(f"  To SEE the {_freed_in_step:.0f} MB of inside-step transients while they are alive,")
        print(f"  re-run with LILYTORCH_MEM_DBG=1 — that env var enables alloc/peak prints")
        print(f"  before each sub-step inside _fluid_step_kernel_3d() so you can pinpoint")
        print(f"  which phase (Kernel A, Kernel B, projection, V-cycle) drove the peak.")

    if census_final:
        census_total = sum(r["bytes"] for r in census_final) / (1024 ** 2)
        print(f"\nTensor census at END (after gc+empty_cache) — top 20 by size:")
        print(f"  Total PyTorch tensors found: {census_total:.1f} MB")
        print(f"  {'Shape':<32s}  {'Dtype':<14s}  {'Count':>5s}  {'Total MB':>9s}  Attribute names")
        print("  " + "-" * 100)
        for row in census_final[:20]:
            shape = "(" + ", ".join(str(s) for s in row["shape"]) + ")"
            names = ", ".join(row.get("names", []))
            print(f"  {shape:<32s}  {row['dtype']:<14s}  "
                  f"{row['count']:>5d}  {_mb(row['bytes']):>9.1f}  {names}")
        if final_alloc_mb - census_total > 50:
            print(f"  NOTE: {final_alloc_mb - census_total:.1f} MB allocated but not found "
                  f"by gc — likely C++-owned tensors (torch.compile static bufs, etc.)")
    print(sep, flush=True)


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
        sim_dict = yaml.full_load(f) or {}
    keep = {"lilytorch.integration.extensions.FluidExtension"}
    sim_dict["extensions"] = [
        ext for ext in sim_dict.get("extensions", [])
        if ext.get("loader", "") in keep
    ]
    sim_dict.setdefault("runtime", {})["headless"] = True
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
                _env = os.environ.copy()
                _env.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                                "expandable_segments:True")
                ret = subprocess.call(cmd, env=_env)
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
    _plot_comparison(all_results, args, out_dir)


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
            print(f"  {'Shape':<32s}  {'Dtype':<14s}  {'Count':>5s}  {'Total MB':>9s}  Attribute names")
            print("  " + "-" * 100)
            for row in rec["census_at_peak"][:10]:
                shape  = "(" + ", ".join(str(s) for s in row["shape"]) + ")"
                names  = ", ".join(row.get("names", []))
                print(f"  {shape:<32s}  {row['dtype']:<14s}  "
                      f"{row['count']:>5d}  {_mb(row['bytes']):>9.1f}  {names}")

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


def _plot_comparison(
    all_results: dict[tuple[str, int], dict],
    args: argparse.Namespace,
    out_dir: str,
) -> None:
    """Save a matplotlib figure summarising the memory comparison.

    Three subplots:
      1. Body-scaling — persistent baseline MB vs N bodies (python vs kernel).
      2. Per-phase waterfall — allocated MB at each step phase for n_bodies=1.
      3. Tensor census — top tensors at peak step for each mode (n_bodies=1).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    spec = get_dimension_spec(args.dim)
    body_counts = sorted({k[1] for k in all_results})
    modes_present = [m for m in _MODES if any(k[0] == m for k in all_results)]
    colors = {"python": "#1f77b4", "kernel": "#ff7f0e"}

    single_body_results = {
        m: all_results[(m, 1)]
        for m in modes_present
        if (m, 1) in all_results
    }

    # ── decide layout ──────────────────────────────────────────────────
    n_cols = 3 if single_body_results else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(
        f"GPU memory — {spec.label} pinned 1guilla  "
        f"(grid {grid_label((args.Nx, args.Ny) if spec.dim == 2 else (args.Nx, args.Ny, args.Nz))},"
        f"  {args.poisson_method},  {args.dtype})",
        fontsize=11,
    )

    # ── subplot 1 : body-scaling ───────────────────────────────────────
    ax1 = axes[0]
    x = np.array(body_counts)
    for mode in modes_present:
        ys = [
            _persistent_baseline(all_results[(mode, nb)])
            if (mode, nb) in all_results else float("nan")
            for nb in body_counts
        ]
        ax1.plot(x, ys, "o-", color=colors.get(mode, None), label=mode, linewidth=2)
    ax1.set_xlabel("Number of bodies (B)")
    ax1.set_ylabel("Persistent baseline (MB)")
    ax1.set_title("Body-scaling: persistent memory")
    ax1.set_xticks(x)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    if n_cols < 2:
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"memory_comparison_{spec.short_tag}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\n[memory_comparison] figure saved → {out_path}")
        return

    # ── subplot 2 : per-phase waterfall (n_bodies=1) ───────────────────
    ax2 = axes[1]
    phase_names: list[str] = []
    for mode, rec in single_body_results.items():
        rows = _peak_step_rows(rec)
        if rows:
            phase_names = [
                (r["label"].split(":", 1)[1].strip() if ":" in r["label"] else r["label"])
                for r in rows
            ]
            break

    bar_w = 0.8 / max(len(single_body_results), 1)
    x_idx = np.arange(len(phase_names))
    for i, (mode, rec) in enumerate(single_body_results.items()):
        rows = _peak_step_rows(rec)
        alloc_vals = [r["alloc_mb"] for r in rows] if rows else []
        offset = (i - (len(single_body_results) - 1) / 2) * bar_w
        ax2.bar(x_idx + offset, alloc_vals[:len(phase_names)],
                width=bar_w, color=colors.get(mode, None),
                alpha=0.8, label=mode)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(
        [p.replace("after ", "").replace("(union SDF+body_vel)", "update") for p in phase_names],
        rotation=35, ha="right", fontsize=8,
    )
    ax2.set_ylabel("Allocated MB")
    ax2.set_title("Per-phase alloc (n_bodies=1, peak step)")
    ax2.legend()
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    # ── subplot 3 : tensor census (n_bodies=1) ─────────────────────────
    ax3 = axes[2]
    census_data: dict[str, list[tuple[str, float]]] = {}
    for mode, rec in single_body_results.items():
        census = rec.get("census_at_peak", [])
        census_data[mode] = [
            (
                # names first (if known), then shape+dtype as fallback
                (", ".join(row["names"]) if row.get("names") else "")
                + "\n(" + "×".join(str(s) for s in row["shape"]) + ")  " + row["dtype"],
                _mb(row["bytes"]),
            )
            for row in census[:10]
        ]

    # merge tensor labels (union of top-10 from each mode)
    seen: dict[str, dict[str, float]] = {}
    for mode, entries in census_data.items():
        for label, mb in entries:
            seen.setdefault(label, {})[mode] = mb
    # sort by max across modes
    sorted_labels = sorted(seen, key=lambda l: max(seen[l].values()), reverse=True)[:12]
    y_idx = np.arange(len(sorted_labels))
    bar_h = 0.8 / max(len(single_body_results), 1)
    for i, mode in enumerate(single_body_results):
        vals = [seen.get(lbl, {}).get(mode, 0.0) for lbl in sorted_labels]
        offset = (i - (len(single_body_results) - 1) / 2) * bar_h
        ax3.barh(y_idx + offset, vals, height=bar_h,
                 color=colors.get(mode, None), alpha=0.8, label=mode)
    ax3.set_yticks(y_idx)
    ax3.set_yticklabels(sorted_labels, fontsize=7)
    ax3.invert_yaxis()
    ax3.set_xlabel("Total MB (unique storages)")
    ax3.set_title("Tensor census at peak step (n_bodies=1)")
    ax3.legend()
    ax3.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"memory_comparison_{spec.short_tag}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[memory_comparison] figure saved → {out_path}")


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
