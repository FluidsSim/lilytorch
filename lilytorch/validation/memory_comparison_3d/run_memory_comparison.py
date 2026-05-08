#!/usr/bin/env python3
"""
GPU memory comparison across the solver modes.

This is the *measurement* counterpart to ``docs/memory_analysis.md``.  It
runs the same FARMS-coupled 3-D 1guilla free-swimming scenario under the
reference Python path and the native streamed kernel path, and reports
per-phase / per-tensor memory usage so the biggest bottlenecks can be
identified directly on hardware.

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
                post-fluid-step native force pass.

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

    # Run both modes and emit a comparison table:
    python run_memory_comparison.py --Nx 256 --Ny 64 --Nz 64 --n_steps 80

    # Re-run a single mode:
    python run_memory_comparison.py --mode kernel \
        --Nx 256 --Ny 64 --Nz 64 --n_steps 80

The script intentionally uses a coarsened grid by default so that all
two modes fit on a 12 GB GPU (the ``python`` path needs roughly
2× the memory of the kernel path on the same grid).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time


# ════════════════════════════════════════════════════════════════════════
#  CLI parsing — shared by driver and worker
# ════════════════════════════════════════════════════════════════════════

_MODES = ("python", "kernel")
_MODE_ALIASES = {
    "no_kernels": "python",
    "kernels": "kernel",
}
_LEGACY_RESULT_FILES = {
    "python": "memory_no_kernels.json",
    "kernel": "memory_kernels.json",
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
    return os.path.join(out_dir, f"memory_{mode}.json")


def _existing_result_path(out_dir: str, mode: str) -> str:
    canonical = _result_path(out_dir, mode)
    if os.path.exists(canonical):
        return canonical
    legacy_name = _LEGACY_RESULT_FILES.get(mode)
    if legacy_name is not None:
        legacy = os.path.join(out_dir, legacy_name)
        if os.path.exists(legacy):
            return legacy
    return canonical


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GPU memory comparison across python / "
            "kernel solver modes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=_canonical_mode, choices=_MODES, default=None,
        help=(
            "Run a single configuration (worker mode).  Without this flag "
            "the script becomes the driver and spawns one subprocess per "
            "mode."
        ),
    )
    parser.add_argument("--Nx", type=int, default=256)
    parser.add_argument(
        "--Ny", type=int, default=None,
        help=(
            "Grid cells in y.  If omitted, derived from --Nx and the "
            "SimConfig domain extents to ensure isotropic spacing."
        ),
    )
    parser.add_argument(
        "--Nz", type=int, default=None,
        help=(
            "Grid cells in z.  If omitted, derived from --Nx and the "
            "SimConfig domain extents to ensure isotropic spacing."
        ),
    )
    parser.add_argument(
        "--n_steps", type=int, default=80,
        help=(
            "Total simulation steps to run inside each worker.  Snapshots "
            "are taken at warm-up (~10), peak (~n_steps/2), and end."
        ),
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
        "--out_dir", default=None,
        help=(
            "Directory where worker JSON results land.  Defaults to "
            "``./results/`` next to this script."
        ),
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter used to launch worker subprocesses.",
    )
    parser.add_argument(
        "--keep_existing", action="store_true",
        help=(
            "(Driver only) Skip a mode if its JSON file already exists.  "
            "Useful for incremental sweeps."
        ),
    )
    return parser.parse_args()


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
    """In-process FARMS run for a single configuration.

    The structure mirrors ``run_memory_profile_free_3d.py`` but with three
    additions:

    1. The chosen ``mode`` is *applied to the SimConfig instance* before
       ``FluidExtension`` instantiates ``BDIMhandler`` / ``FluidSolver``.
    2. Snapshots are sparse (warmup + a single peak step + end) so the
       data is comparable across modes without overwhelming the JSON
       output.
    3. A tensor census is taken at the peak step so the *which* of the
       memory difference (not just *how much*) is visible.
    """
    # --- imports deferred so ``--help`` does not need torch / FARMS ---
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "[memory_comparison worker] CUDA is required — this tool "
            "measures GPU memory.  CPU runs are not informative because "
            "PyTorch on CPU does not honour memory_allocated()/peak."
        )

    mode: str = args.mode
    out_dir: str = args.out_dir

    # ── 1. Build the SimConfig and apply mode-specific overrides ─────
    import lilytorch.farms_examples._1guillasim.gen_configs_one_free_3d as cfg_mod

    cfg = cfg_mod.SimConfig()
    cfg.Nx = args.Nx
    # Derive Ny / Nz from Nx and the domain extents so the grid stays
    # isotropic (dx == dy == dz), unless the user passed explicit values.
    dx = (cfg.xmax - cfg.xmin) / cfg.Nx
    cfg.Ny = args.Ny if args.Ny is not None else max(1, round((cfg.ymax - cfg.ymin) / dx))
    cfg.Nz = args.Nz if args.Nz is not None else max(1, round((cfg.zmax - cfg.zmin) / dx))

    cfg.n_iterations = args.n_steps + 2
    cfg.save_every   = args.n_steps + 999      # disable IO
    cfg.save_frames  = False
    cfg.save         = False
    cfg.headless     = True

    # The gen_configs_one_free_3d SimConfig overrides use_bdim=False (it is
    # used standalone without fluid coupling in its normal run).  Force it
    # back on here so FluidExtension is injected into the YAML.
    cfg.use_bdim = True

    # Mode-specific solver method — flows through base_sim_config into
    # bdim_yaml.solver and is read by FluidSolver.
    if mode == "python":
        cfg.solver_method = "python"
    elif mode == "kernel":
        cfg.solver_method = "kernel"
    else:
        raise ValueError(f"unknown mode '{mode}'")

    # Strip non-fluid extensions so the measurement is not polluted by
    # FlowViewer, ExperimentLogger, CameraRecording, etc.
    _orig_extra_ext = cfg.extra_simulation_extensions
    cfg.extra_simulation_extensions = lambda *_a, **_kw: []

    print(f"[memory_comparison worker] mode={mode}  "
          f"Nx={cfg.Nx} Ny={cfg.Ny} Nz={cfg.Nz}  "
          f"n_steps={args.n_steps}  warmup={args.warmup_steps}",
          flush=True)

    # ── 2. Generate the FARMS configs ────────────────────────────────
    from lilytorch.util.paths import gen_new_folder, save_path
    output_folder = gen_new_folder(save_path)
    os.makedirs(output_folder, exist_ok=True)

    cfg.gen_animat_config(output_folder)
    cfg.gen_arena_config(output_folder)
    cfg.gen_simulation_config(output_folder)
    cfg.gen_experiment_config(output_folder)

    # As a belt-and-braces measure, also strip non-fluid extensions
    # directly in the YAML — some BaseSimConfig versions inject
    # extensions that the override above does not reach.
    _strip_yaml_extensions(os.path.join(output_folder, "simulation_config.yaml"))

    os.chdir(output_folder)

    # ── 3. Install profiler hooks ────────────────────────────────────
    records: list[dict] = []
    census_at_peak: list[dict] = []

    def _record(label: str) -> dict:
        s = _snap(label)
        records.append(s)
        print(
            f"  [{label:>50s}]  alloc={s['alloc_mb']:8.1f} MB   "
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
        # Replicate the behaviour of the real ``step`` but instrument
        # selected boundaries with snapshots.  Use the original method
        # whenever we are not in a measurement step to keep overhead
        # low.
        idx = step_idx[0]
        step_idx[0] = idx + 1

        is_warmup_end = (idx == args.warmup_steps)
        is_peak       = (idx == peak_step)
        is_end        = (idx == args.n_steps - 1)
        is_traced     = is_warmup_end or is_peak or is_end

        if not is_traced:
            return _orig_step(self, task, physics)

        # ── traced step: replicate ``BDIMhandler.step`` body manually.
        # Must stay in sync with BDIMhandler.step in BDIMhandler.py.
        iteration = self.iteration
        timestep  = self.pars["solver"]["dt"]
        if iteration >= self.pars["solver"]["nt"]:
            return

        t  = iteration * timestep
        fs = self.fluid_solver

        if fs.terminate:
            self.iteration += 1
            return

        # Reset the peak counter to capture the per-phase deltas of
        # *this step only*.
        _reset_peak()
        _record(f"step {idx:03d} [{mode}]: before")

        # 1. update — body kinematics → SDF/body-velocity grids
        self.update(t, iteration, dt=timestep)
        _record(f"step {idx:03d} [{mode}]: after update (SDF+body_vel)")

        # 2. mu / normals
        if self.ndim == 3:
            fs._recompute_mu_normals()
        else:
            fs._recompute_mu_normals()
        _record(f"step {idx:03d} [{mode}]: after mu/normals")

        # 3. fluid step (Heun: predictor + corrector)
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

        # 4b. free stress/pforce tensors immediately after forces — this
        # matches the real step's ``fs.__dict__.update(_FS_FREE_AFTER_FORCES_3D)``
        # which nulls xstress_tensor, ystress_tensor, zstress_tensor,
        # pforce_x/y/z before plotting and apply_forces.
        from lilytorch.integration.BDIMhandler import _FS_FREE_AFTER_FORCES_3D
        if self.ndim == 3:
            fs.__dict__.update(_FS_FREE_AFTER_FORCES_3D)
        _record(f"step {idx:03d} [{mode}]: after force-field release")

        # 5. plotting (save=False so this is a near-no-op, but must match
        # the real step so the memory footprint sequence is identical)
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

        # 7. release BDIM intermediates.
        # _release_bdim_fields behaviour is mode-specific:
        #   python          → frees mu/normals + force fields + div
        #   kernel          → keeps mu/normal packed buffers alive
        #                     (they are persistent across steps when
        #                     _mu_normals_union=True)
        fs._release_bdim_fields()
        _record(f"step {idx:03d} [{mode}]: after release")

        # Tensor census at the peak step gives the *attribution*
        # (which buffers dominate which mode).
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
    out_path = _result_path(out_dir, mode)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[memory_comparison worker] wrote {out_path}", flush=True)


def _strip_yaml_extensions(yaml_path: str) -> None:
    """Strip every non-fluid extension from ``simulation_config.yaml``.

    Mirrors the strip step in ``run_cost_analysis.py``.  Run unconditionally
    so the worker is robust to BaseSimConfig versions that inject
    FlowViewer / CameraRecording even when ``extra_simulation_extensions``
    returns ``[]``.
    """
    import yaml
    with open(yaml_path, "r") as f:
        sim_dict = yaml.safe_load(f) or {}
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
    """Spawn one subprocess per mode, then aggregate."""
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print(" GPU memory comparison: python vs kernel solver modes")
    print("=" * 78)
    ny_str = str(args.Ny) if args.Ny is not None else "auto"
    nz_str = str(args.Nz) if args.Nz is not None else "auto"
    print(f"  Grid:        {args.Nx} × {ny_str} × {nz_str}  (Ny/Nz auto = isotropic from domain)")
    print(f"  Steps:       {args.n_steps}  "
          f"(warmup={args.warmup_steps}, peak step={args.peak_step or '~0.6·n_steps'})")
    print(f"  Output dir:  {out_dir}")
    print("=" * 78)

    results: dict[str, dict] = {}
    for mode in _MODES:
        existing_path = _existing_result_path(out_dir, mode)
        if args.keep_existing and os.path.exists(existing_path):
            print(f"  [{mode}] reusing existing {existing_path}")
        else:
            print(f"\n  ── Running mode = {mode} ──────────────────────")
            cmd = [
                args.python, os.path.abspath(__file__),
                "--mode", mode,
                "--Nx", str(args.Nx),
                "--n_steps", str(args.n_steps),
                "--warmup_steps", str(args.warmup_steps),
                "--out_dir", out_dir,
            ]
            if args.Ny is not None:
                cmd += ["--Ny", str(args.Ny)]
            if args.Nz is not None:
                cmd += ["--Nz", str(args.Nz)]
            if args.peak_step is not None:
                cmd += ["--peak_step", str(args.peak_step)]
            t0 = time.time()
            ret = subprocess.call(cmd)
            dt = time.time() - t0
            if ret != 0:
                print(f"  [{mode}] subprocess failed (exit {ret}) — "
                      f"skipping in summary")
                continue
            print(f"  [{mode}] completed in {dt:.1f}s")
        out_path = _existing_result_path(out_dir, mode)
        try:
            with open(out_path) as f:
                results[mode] = json.load(f)
            results[mode]["mode"] = mode
        except FileNotFoundError:
            continue

    if not results:
        print("\nNo mode completed successfully — nothing to summarise.")
        return

    _print_comparison(results)


# ── Comparison report ─────────────────────────────────────────────────

def _persistent_baseline(rec: dict) -> float:
    """Allocated MB right *before* the first traced step.

    The first traced step is the warmup step (step == warmup_steps).
    """
    target_label = f"step {rec['warmup_steps']:03d}"
    for r in rec["records"]:
        if r["label"].startswith(target_label) and "before" in r["label"]:
            return r["alloc_mb"]
    # Fallback: use "after simulation complete" minus what we know was
    # released — this is a coarser estimate.
    return rec["records"][-1]["alloc_mb"]


def _peak_step_rows(rec: dict) -> list[dict]:
    target_label = f"step {rec['peak_step']:03d}"
    return [r for r in rec["records"] if r["label"].startswith(target_label)]


def _phase_deltas(rows: list[dict]) -> list[tuple[str, float, float]]:
    """Return (phase, delta_alloc_mb, peak_mb) for each consecutive pair."""
    out = []
    for prev, cur in zip(rows, rows[1:]):
        # Pull the phase name out of the record label, which looks like
        # ``step 040 [kernel]: after fluid_step``.
        cur_phase = cur["label"].split(":", 1)[1].strip() if ":" in cur["label"] else cur["label"]
        out.append((cur_phase, cur["alloc_mb"] - prev["alloc_mb"], cur["peak_mb"]))
    return out


def _print_comparison(results: dict[str, dict]) -> None:
    print("\n\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)

    # ── 1. Top-line resident + peak per mode ──────────────────────────
    print(f"\n{'Mode':<22s} {'Persistent MB':>14s} {'Step peak MB':>14s} "
          f"{'Final peak MB':>14s} {'Reserved MB':>14s}")
    print("-" * 78)
    for mode, rec in results.items():
        persistent_mb = _persistent_baseline(rec)
        peak_rows     = _peak_step_rows(rec)
        step_peak_mb  = max((r["peak_mb"] for r in peak_rows), default=float("nan"))
        print(f"{mode:<22s} {persistent_mb:14.1f} {step_peak_mb:14.1f} "
              f"{rec['final_peak_mb']:14.1f} {rec['final_rsrv_mb']:14.1f}")

    # ── 2. Per-phase peak (cf. mu_normals/forces split) ───────────────
    print("\nPer-phase PEAK alloc at the chosen peak step (MB)")
    print("-" * 78)

    # Gather phase names from the first available result so the table
    # is consistent across modes.
    any_rows = next(iter(results.values()))
    sample_phases = [
        cur["label"].split(":", 1)[1].strip() if ":" in cur["label"] else cur["label"]
        for cur in _peak_step_rows(any_rows)[1:]   # skip "before"
    ]
    header = f"{'Phase':<32s}" + "".join(f"{m:>15s}" for m in results)
    print(header)
    print("-" * len(header))
    for phase_idx, phase_name in enumerate(sample_phases):
        cells = []
        for mode, rec in results.items():
            rows = _peak_step_rows(rec)
            if phase_idx + 1 < len(rows):
                cells.append(f"{rows[phase_idx + 1]['peak_mb']:15.1f}")
            else:
                cells.append(f"{'-':>15s}")
        print(f"{phase_name:<32s}" + "".join(cells))

    # ── 3. Per-phase delta (MB allocated by *that* phase) ─────────────
    print("\nPer-phase DELTA alloc (MB allocated by the phase, signed)")
    print("-" * 78)
    print(header)
    print("-" * len(header))
    for phase_idx, phase_name in enumerate(sample_phases):
        cells = []
        for mode, rec in results.items():
            rows = _peak_step_rows(rec)
            deltas = _phase_deltas(rows)
            if phase_idx < len(deltas):
                cells.append(f"{deltas[phase_idx][1]:+15.1f}")
            else:
                cells.append(f"{'-':>15s}")
        print(f"{phase_name:<32s}" + "".join(cells))

    # ── 4. Top tensor census per mode (the *what*) ────────────────────
    print("\n\nTOP-10 LARGEST TENSORS PER MODE (at peak step)")
    print("=" * 78)
    for mode, rec in results.items():
        print(f"\n[{mode}]")
        print(f"  {'Shape':<35s}  {'Dtype':<18s}  {'Count':>5s}  {'Total MB':>10s}")
        print("  " + "-" * 76)
        for row in rec["census_at_peak"][:10]:
            shape = "(" + ", ".join(str(s) for s in row["shape"]) + ")"
            print(f"  {shape:<35s}  {row['dtype']:<18s}  "
                  f"{row['count']:>5d}  {_mb(row['bytes']):>10.1f}")

    # ── 5. Pairwise savings (kernel vs python baseline) ───────────────
    if "python" in results:
        print("\n\nSAVINGS RELATIVE TO python (negative = kernel path uses LESS memory)")
        print("=" * 78)
        ref_rec     = results["python"]
        ref_persist = _persistent_baseline(ref_rec)
        ref_peak    = max(
            (r["peak_mb"] for r in _peak_step_rows(ref_rec)),
            default=ref_rec.get("final_peak_mb", float("nan")),
        )
        for mode, rec in results.items():
            if mode == "python":
                continue
            persist_mb = _persistent_baseline(rec)
            peak_mb    = max(
                (r["peak_mb"] for r in _peak_step_rows(rec)),
                default=rec.get("final_peak_mb", float("nan")),
            )
            d_persist  = persist_mb - ref_persist
            d_peak     = peak_mb    - ref_peak
            print(f"  [{mode:<18s}]  ΔPersistent = {d_persist:+8.1f} MB   "
                  f"ΔStep peak = {d_peak:+8.1f} MB   "
                  f"({d_peak / max(ref_peak, 1.0) * 100:+5.1f}% step peak)")


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results",
        )
    if args.mode is None:
        _run_driver(args)
    else:
        _run_worker(args)


if __name__ == "__main__":
    main()
