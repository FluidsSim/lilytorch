#!/usr/bin/env python3
"""
Memory-vs-grid-size scaling sweep.

Spawns ``run_memory_comparison.py`` once per grid size in ``--N_list``
(one fresh CUDA context per run — clean baseline), collects each worker's
JSON output, and plots how the four memory layers scale with N:

  1. persistent baseline    (torch.cuda.memory_allocated, post-cleanup)
  2. peak alloc during step (torch.cuda.max_memory_allocated)
  3. peak reserved          (torch.cuda.memory_reserved)
  4. nvidia-smi process RSS (peak step, via pynvml or subprocess)

Two panels are produced:
  * Linear y-axis showing the four curves vs N (cube-root of #cells).
  * Log-log showing peak-alloc vs cells (N³) with an ideal O(N³) reference
    line — verifies that scaling is linear in cell count (which it should
    be: all dominant tensors are N³ × dtype bytes).

A summary table is printed to stdout and the plot is written next to the
script under ``scaling_results/scaling_<mode>_b<bodies>.png``.

Example
-------
    # Default sweep (N = 96, 128, 192, 256, 320, 384, 448, 512), kernel mode, 1 body
    python run_memory_scaling.py

    # Custom list + reuse existing results
    python run_memory_scaling.py --N_list 128,256,384,512 --keep_existing

    # Python (reference) path for comparison
    python run_memory_scaling.py --mode python --N_list 128,192,256
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt

# Path to the per-grid worker script (handles a single Nx,Ny,Nz config).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(_SCRIPT_DIR, "run_memory_comparison.py")


# ════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Grid-size memory-scaling sweep for the FluidSolver kernel-mode "
            "3-D pipeline.  Spawns one worker per N; plots peak memory vs N."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--N_list", type=str,
        default="96,128,192,256,320,384,448,512",
        help=("Comma-separated list of cubic grid sizes to sweep "
              "(N×N×N).  Default: 96,128,192,256,320,384,448,512."),
    )
    p.add_argument(
        "--mode", default="kernel", choices=["kernel", "python"],
        help="Solver path under test (default: kernel).",
    )
    p.add_argument("--n_bodies", type=int, default=1)
    p.add_argument("--n_steps", type=int, default=10,
                   help="Worker steps per N (default: 10 — enough for steady state).")
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--poisson_method", default="multigrid",
                   choices=["multigrid", "mgcg", "fft"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--out_dir", default=None,
                   help="Output directory.  Defaults to "
                        "./scaling_results/ next to this script.")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter used to launch workers.")
    p.add_argument("--keep_existing", action="store_true",
                   help="Skip a worker run if its JSON already exists.")
    p.add_argument("--compile_adv_diff", action="store_true", default=False,
                   help="Forward to worker: torch.compile the adv-diff solver.")
    args = p.parse_args()

    args.N_list = [int(x) for x in args.N_list.split(",") if x.strip()]
    if args.out_dir is None:
        args.out_dir = os.path.join(_SCRIPT_DIR, "scaling_results")
    return args


# ════════════════════════════════════════════════════════════════════════
#  Worker dispatch
# ════════════════════════════════════════════════════════════════════════

def _worker_json_path(out_dir: str, mode: str, n_bodies: int, N: int) -> str:
    """Per-N subdirectory keeps each worker's result isolated."""
    sub = os.path.join(out_dir, f"N{N:04d}")
    return sub, os.path.join(sub, f"memory_{mode}_b{n_bodies:02d}.json")


def _run_one(args: argparse.Namespace, N: int) -> dict | None:
    """Spawn the worker for an N×N×N grid; return parsed JSON or None on failure."""
    sub_dir, json_path = _worker_json_path(args.out_dir, args.mode, args.n_bodies, N)
    os.makedirs(sub_dir, exist_ok=True)

    if args.keep_existing and os.path.exists(json_path):
        print(f"  [N={N:4d}] reusing existing JSON: {json_path}")
    else:
        cmd = [
            args.python, WORKER,
            "--dim", "3",
            "--mode", args.mode,
            "--n_bodies", str(args.n_bodies),
            "--Nx", str(N), "--Ny", str(N), "--Nz", str(N),
            "--n_steps", str(args.n_steps),
            "--warmup_steps", str(args.warmup_steps),
            "--poisson_method", args.poisson_method,
            "--dtype", args.dtype,
            "--out_dir", sub_dir,
        ]
        if args.compile_adv_diff:
            cmd.append("--compile_adv_diff")

        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        t0 = time.time()
        print(f"  [N={N:4d}] spawning worker ({args.mode}, n_bodies={args.n_bodies}) ...",
              flush=True)
        ret = subprocess.call(cmd, env=env)
        dt = time.time() - t0
        if ret != 0:
            print(f"  [N={N:4d}] worker FAILED (exit {ret}) — likely OOM, skipping")
            return None
        print(f"  [N={N:4d}] completed in {dt:.1f}s")

    if not os.path.exists(json_path):
        print(f"  [N={N:4d}] no JSON written at {json_path}, skipping")
        return None
    with open(json_path) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════
#  Metric extraction
# ════════════════════════════════════════════════════════════════════════

def _collect_metrics(rec: dict) -> dict:
    """Pull the four-layer peak values out of a worker JSON record.

    Returns a dict with keys
      persistent_mb, peak_alloc_mb, peak_rsrv_mb, peak_nvml_mb, baseline_rsrv_mb.

    Uses the same definitions the worker's inline summary prints:
      * persistent  = census_at_peak_alloc_mb (alloc after release+empty_cache)
      * peak_alloc  = max peak_mb across the peak-step records
      * peak_rsrv   = max rsrvd_mb across "after fluid_step" records
      * peak_nvml   = peak_step_nvml_mb (captured before cleanup)
      * baseline_rsrv = rsrvd_mb between steps (after release)
    """
    persistent = rec.get("census_at_peak_alloc_mb") or 0.0

    # Peak alloc: max peak_mb from the records that include "after fluid_step".
    peak_alloc = 0.0
    peak_rsrv  = 0.0
    baseline_rsrv = 0.0
    for r in rec.get("records", []):
        label = r.get("label", "")
        if "after fluid_step" in label:
            peak_alloc = max(peak_alloc, r.get("peak_mb", 0.0))
            peak_rsrv  = max(peak_rsrv,  r.get("rsrvd_mb", 0.0))
        if "after release" in label:
            # The "after release" snapshot reflects the inter-step reserved low-water.
            baseline_rsrv = max(baseline_rsrv, r.get("rsrvd_mb", 0.0))

    peak_nvml = rec.get("peak_step_nvml_mb")
    if peak_nvml is None:
        peak_nvml = rec.get("census_at_true_peak_nvml_mb")

    return dict(
        persistent_mb=persistent,
        peak_alloc_mb=peak_alloc,
        peak_rsrv_mb=peak_rsrv,
        peak_nvml_mb=peak_nvml,
        baseline_rsrv_mb=baseline_rsrv,
    )


# ════════════════════════════════════════════════════════════════════════
#  Plotting + table
# ════════════════════════════════════════════════════════════════════════

def _print_summary(data: dict[int, dict], args: argparse.Namespace) -> None:
    print()
    print("=" * 100)
    print(f"  MEMORY SCALING — mode={args.mode}  n_bodies={args.n_bodies}  "
          f"poisson={args.poisson_method}  dtype={args.dtype}")
    print("=" * 100)
    print(f"  {'N':>5} {'cells (M)':>10}  "
          f"{'persistent':>12} {'peak_alloc':>12} {'peak_rsrv':>12} {'nvml':>12}")
    print("  " + "-" * 80)
    for n in sorted(data.keys()):
        d = data[n]
        cells_m = (n ** 3) / 1e6
        nvml_s = f"{d['peak_nvml_mb']:8.0f} MB" if d['peak_nvml_mb'] is not None else "     n/a"
        print(f"  {n:>5d} {cells_m:>9.2f}   "
              f"{d['persistent_mb']:>9.0f} MB "
              f"{d['peak_alloc_mb']:>9.0f} MB "
              f"{d['peak_rsrv_mb']:>9.0f} MB "
              f"{nvml_s}")
    print("=" * 100)

    # Persistent / peak-alloc bytes-per-cell, useful as a sanity check
    # (should be ~constant if scaling is linear in N³).
    print()
    print("  Bytes per cell (persistent and peak_alloc, fp32 ≈ 4 bytes/cell baseline):")
    print(f"  {'N':>5} {'persist/cell':>14} {'peak/cell':>12}")
    for n in sorted(data.keys()):
        d = data[n]
        cells = n ** 3
        ppc = (d["persistent_mb"] * 1024 * 1024) / cells if cells else 0
        peakpc = (d["peak_alloc_mb"] * 1024 * 1024) / cells if cells else 0
        print(f"  {n:>5d} {ppc:>10.1f} B    {peakpc:>9.1f} B")
    print()


def _plot(data: dict[int, dict], args: argparse.Namespace) -> str:
    N = sorted(data.keys())
    if not N:
        return ""
    cells = [n ** 3 for n in N]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel 1: linear axes vs N — the human-readable view ─────────
    ax = axes[0]
    ax.plot(N, [data[n]["persistent_mb"] / 1024 for n in N],
            "o-", label="persistent baseline (alloc post-cleanup)", color="C0")
    ax.plot(N, [data[n]["peak_alloc_mb"]  / 1024 for n in N],
            "s-", label="peak alloc (max_memory_allocated)", color="C1")
    ax.plot(N, [data[n]["peak_rsrv_mb"]   / 1024 for n in N],
            "^-", label="peak reserved (memory_reserved)", color="C2")
    have_nvml = [(n, data[n]["peak_nvml_mb"]) for n in N if data[n]["peak_nvml_mb"]]
    if have_nvml:
        ax.plot([n for n, _ in have_nvml],
                [v / 1024 for _, v in have_nvml],
                "d-", label="nvidia-smi (peak step, this PID)", color="C3")
    ax.set_xlabel("Grid size N (cube root of #cells)")
    ax.set_ylabel("Memory (GiB)")
    ax.set_title(f"Peak memory vs grid size — mode={args.mode}, "
                 f"n_bodies={args.n_bodies}, {args.dtype}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(N)
    ax.set_xticklabels([str(n) for n in N], rotation=0)

    # ── Panel 2: log-log vs N³ with ideal-scaling reference line ─────
    ax = axes[1]
    peak = [data[n]["peak_alloc_mb"] for n in N]
    pers = [data[n]["persistent_mb"] for n in N]
    ax.loglog(cells, peak, "s-", label="peak alloc", color="C1")
    ax.loglog(cells, pers, "o-", label="persistent baseline", color="C0")
    # Ideal-linear (O(N³)) reference, normalised to the largest data point.
    ref_n = max(range(len(N)), key=lambda i: peak[i])
    ref = [c * peak[ref_n] / cells[ref_n] for c in cells]
    ax.loglog(cells, ref, "k--", label="ideal O(N³) linear",
              alpha=0.6, linewidth=1.2)
    ax.set_xlabel("Cells (N³)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title("Log-log scaling — does peak grow linearly with N³?")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.suptitle(
        f"Lilytorch memory scaling — {args.mode}, b={args.n_bodies}, "
        f"poisson={args.poisson_method}, dtype={args.dtype}",
        y=1.02, fontsize=11,
    )
    plt.tight_layout()
    out_path = os.path.join(
        args.out_dir,
        f"scaling_{args.mode}_b{args.n_bodies:02d}_{args.poisson_method}.pdf",
    )
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    return out_path


def _save_csv(data: dict[int, dict], args: argparse.Namespace) -> str:
    """Write a compact CSV of the scaling table for downstream analysis."""
    csv_path = os.path.join(
        args.out_dir,
        f"scaling_{args.mode}_b{args.n_bodies:02d}_{args.poisson_method}.csv",
    )
    with open(csv_path, "w") as f:
        f.write("N,cells,persistent_mb,peak_alloc_mb,peak_rsrv_mb,"
                "peak_nvml_mb,baseline_rsrv_mb\n")
        for n in sorted(data.keys()):
            d = data[n]
            nvml = d["peak_nvml_mb"] if d["peak_nvml_mb"] is not None else ""
            f.write(f"{n},{n**3},{d['persistent_mb']:.2f},"
                    f"{d['peak_alloc_mb']:.2f},{d['peak_rsrv_mb']:.2f},"
                    f"{nvml},{d['baseline_rsrv_mb']:.2f}\n")
    return csv_path


# ════════════════════════════════════════════════════════════════════════
#  Driver
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sep = "=" * 80
    print(sep)
    print(" Memory-scaling sweep")
    print(sep)
    print(f"  Mode:        {args.mode}")
    print(f"  n_bodies:    {args.n_bodies}")
    print(f"  N values:    {args.N_list}")
    print(f"  n_steps:     {args.n_steps}  (warmup={args.warmup_steps})")
    print(f"  Poisson:     {args.poisson_method}  dtype={args.dtype}")
    print(f"  Out dir:     {args.out_dir}")
    print(sep)

    data: dict[int, dict] = {}
    for N in args.N_list:
        print()
        print(f"── N = {N:4d} ─────────────────────────────────────")
        rec = _run_one(args, N)
        if rec is None:
            continue
        m = _collect_metrics(rec)
        data[N] = m
        print(f"      persistent = {m['persistent_mb']:7.1f} MB   "
              f"peak_alloc = {m['peak_alloc_mb']:7.1f} MB   "
              f"peak_rsrv = {m['peak_rsrv_mb']:7.1f} MB   "
              f"nvml = {m['peak_nvml_mb']}")

    if not data:
        print("\nNo successful runs — nothing to plot.")
        return

    _print_summary(data, args)
    plot_path = _plot(data, args)
    csv_path = _save_csv(data, args)
    print(f"  Plot saved to: {plot_path}")
    print(f"  CSV  saved to: {csv_path}")
    print()


if __name__ == "__main__":
    main()
