# GPU memory comparison: kernel vs no-kernel solver paths

This directory contains a runnable counterpart to
`docs/memory_analysis.md`.  The script measures GPU memory usage of the
two solver configurations on identical hardware so you can ground-truth
the analysis numbers and identify the dominant memory bottleneck of each
path.

## What it compares

| Mode               | YAML keys                                                      | Description |
|--------------------|----------------------------------------------------------------|-------------|
| `no_kernels`       | `solver.use_kernels: false`                                    | Pure-PyTorch reference path. Per-body SDFs and body velocities are materialised on the **full** fluid grid. |
| `kernels`          | `solver.solver_method: "kernel"`                              | Current native streamed kernel path. Geometry is updated through the kernel-mode body metadata path and per-body forces are evaluated later from the post-fluid-step fields without exposing the retired historical force-path toggle. |

## Usage

```bash
# Run both modes and emit a comparison table.
# (Default grid is 256x64x64; shrink for smaller GPUs.)
python lilytorch/validation/memory_comparison_3d/run_memory_comparison.py \
    --Nx 256 --Ny 64 --Nz 64 --n_steps 80

# Re-run a single mode (clean CUDA context):
python lilytorch/validation/memory_comparison_3d/run_memory_comparison.py \
  --mode kernels --Nx 256 --n_steps 80

# Reuse JSON files from a previous run (skip already-computed modes):
python lilytorch/validation/memory_comparison_3d/run_memory_comparison.py \
    --keep_existing --Nx 256 --Ny 64 --Nz 64 --n_steps 80
```

The driver runs each mode in its **own subprocess** so torch.compile
caches and CUDA contexts are isolated.  Per-mode JSON results land in
`results/memory_<mode>.json`.

## What it reports

For each mode the worker captures:

1. **Persistent baseline** — `torch.cuda.memory_allocated()` immediately
   before the first traced step (after `--warmup_steps` steps of
   torch.compile / CUDA-graph warm-up).  This is the *resident*
   footprint that does not move between steps.
2. **Per-phase snapshots** — `alloc / peak / reserved` measured at
   every solver phase boundary (`update`, `mu/normals`, `fluid_step`,
   `forces`, `apply_forces`, `release`) for the warm-up step, the
   chosen peak step (default ≈ 0.6·`n_steps`), and the last step.
3. **Tensor census at the peak step** — top-30 tensors grouped by
   `(shape, dtype)`.  This is what attributes the memory difference to
   specific buffers (e.g. shows directly that `no_kernels` allocates a
   `(B, Nx, Ny, Nz)` per-body SDF that the kernel paths do not).
4. **Final peak** — `torch.cuda.max_memory_allocated()` over the
   whole run.

The driver then prints:

* Top-line table: `Persistent / Step peak / Final peak / Reserved` per mode.
* Per-phase peak alloc and per-phase delta alloc, side-by-side across
  modes — this is where the *which phase* of which path uses the most
  memory becomes obvious.
* Top-10 largest tensors per mode at the peak step.
* Savings of each kernel path relative to `no_kernels` in MB and as
  a percentage of step peak.

## Requirements

* CUDA-capable GPU with PyTorch built with CUDA support.  CPU runs are
  rejected because `torch.cuda.memory_allocated()` is the only reliable
  cross-version measurement; CPU footprint of these paths is dominated
  by working-set allocations that PyTorch does not surface.
* The C++/CUDA kernels in `lilytorch/src/kernels/` must be built
  (`pip install -e .` runs the extension build automatically).
* The same FARMS dependencies as `run_memory_profile_free_3d.py` —
  the script reuses the FARMS in-process flow because that is the only
  way to drive `BDIMhandler` end-to-end.

## Output schema

Each `memory_<mode>.json` is:

```json
{
  "mode": "kernels",
  "Nx": 256, "Ny": 64, "Nz": 64,
  "n_steps": 80, "warmup_steps": 15, "peak_step": 48,
  "device": "NVIDIA RTX 4080 SUPER",
  "torch":  "2.4.0+cu121",
  "records": [
    {"label": "step 048 [kernels]: before",
     "alloc_mb": 1234.5, "peak_mb": 1234.5, "rsrvd_mb": 1280.0},
    ...
  ],
  "census_at_peak": [
    {"shape": [256, 64, 64], "dtype": "torch.float32",
     "count": 6, "bytes": 100663296},
    ...
  ],
  "final_peak_mb": 1830.4,
  "final_rsrv_mb": 2048.0
}
```

so any post-processing (CSV export, plotting, paper figures) can read
these files directly without re-running the simulation.

## Caveats

* **Not bit-equal across modes.**  The two paths use different
  reduction orders and (for `no_kernels`) different masking
  conventions, so trajectories diverge after a few hundred steps.
  Memory measurement is unaffected — the script does not assume the
  states match — but do not draw physical conclusions from this tool.
* **Reserved ≠ allocated.**  PyTorch's caching allocator may hold up
  to ~2× the live footprint as reserved memory.  The "Reserved MB"
  column tells you what `nvidia-smi` will show; the "Persistent /
  Step peak" columns are what you need to compare paths.
* **`torch.compile` overhead.**  The first ~15 steps trigger
  CUDA-graph captures whose temporary buffers are *not* counted by
  `memory_allocated()` but show up in `memory_reserved()`.  That is
  why the script has a configurable `--warmup_steps`.  Increase if
  you see the persistent baseline drift between the warm-up step and
  the peak step.
