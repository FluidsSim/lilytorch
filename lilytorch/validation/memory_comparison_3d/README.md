# GPU memory comparison: python vs kernel solver modes (2-D and 3-D)

This directory contains a runnable counterpart to
`docs/memory_analysis.md`.  The script measures GPU memory of the two
solver configurations on identical hardware across **both** 2-D and 3-D
grids, and with a **variable number of bodies** — the key study is whether
resident memory scales with body count or stays constant.

It uses the same **pinned 1guilla** benchmark as
`lilytorch/validation/cost_analysis/` so memory numbers are directly
comparable to the timing numbers there.

## The central question

| Mode     | Per-body SDF storage | Memory scales with B? |
|----------|----------------------|-----------------------|
| `python` | B full-grid tensors (one per body per staggered grid) | **Yes** — linear in B |
| `kernel` | **One** union SDF tensor, regardless of B | **No** — constant in B |

In `python` mode each body in the composite body object keeps its own
`sdf_u / sdf_v / [sdf_w] / body_u / body_v / [body_w]` tensors on the
full fluid grid.  In `kernel` mode, per-body SDF tensors are computed
inside the batched update kernel and immediately discarded; only the
**union minimum** SDF is stored in `composite_body.sdf_val[_u/_v/_w]`.
This O(1)-in-B property is confirmed empirically by the body-scaling sweep
(`--n_bodies_sweep`).

## What it compares

| Mode     | YAML key                            | Description |
|----------|-------------------------------------|-------------|
| `python` | `solver.solver_method: "python"`   | Pure-PyTorch reference path.  Per-body SDFs and body velocities are materialised on the **full** fluid grid, once per body. |
| `kernel` | `solver.solver_method: "kernel"`   | Native streamed kernel path.  Geometry is merged into a union SDF; persistent packed mu/normals buffers are kept across steps (union AABB mode). |

## Usage

```bash
# 2-D body-sweep (default: 1, 2, 4 bodies) — the primary study
python lilytorch/validation/memory_comparison_3d/run_memory_comparison.py \
    --dim 2 --Nx 512 --Ny 128 --n_steps 60 --n_bodies_sweep 1,2,4

# 3-D body-sweep
python lilytorch/validation/memory_comparison_3d/run_memory_comparison.py \
    --dim 3 --Nx 256 --Ny 64 --Nz 64 --n_steps 60 --n_bodies_sweep 1,2,4

# Single mode / single body count (worker mode — used by the driver internally):
python lilytorch/validation/memory_comparison_3d/run_memory_comparison.py \
    --dim 2 --mode kernel --n_bodies 2 --Nx 512 --Ny 128 --n_steps 60

# Reuse existing JSON files (incremental sweep):
python lilytorch/validation/memory_comparison_3d/run_memory_comparison.py \
    --dim 2 --Nx 512 --n_bodies_sweep 1,2,4,8 --keep_existing
```

The driver runs each **(mode × n_bodies)** combination in its **own
subprocess** so torch.compile caches and CUDA contexts are isolated.
Per-worker JSON results land in `results/dim2/` or `results/dim3/`
as `memory_<mode>_b<N>.json`.

## What it reports

### 1. Body-scaling table (the primary result)
Persistent baseline and step-peak memory for each (mode, n_bodies)
combination.  Shows flat kernel vs linear python growth directly.
A per-mode slope estimate (MB per additional body) is printed when
≥ 2 body counts are measured.

### 2. Single-body per-phase breakdown
For the n_bodies=1 case: per-phase peak and delta allocations at the
chosen peak step.  Useful for understanding **where** in the step each
mode allocates.

### 3. Tensor census at the peak step (the *what*)
Top-10 largest GPU tensors at the peak step for each (mode, n_bodies)
combination.  Look for:
- **python**: tensors with shape `(Nx, Ny[, Nz])` whose `count` grows
  with N bodies — those are the per-body SDF / body-velocity buffers.
- **kernel**: the same shapes appear with `count = 1` (or not at all)
  because only the union SDF is stored.

### 4. python vs kernel savings (n_bodies=1)
Top-line ΔPersistent and ΔStep-peak in MB and as a percentage.

## Output schema

Each `memory_<mode>_b<N>.json` contains:

```json
{
  "mode": "kernel",
  "dim": 3,
  "n_bodies": 2,
  "Nx": 256, "Ny": 64, "Nz": 64,
  "n_steps": 60,
  "warmup_steps": 15,
  "peak_step": 36,
  "device": "NVIDIA ...",
  "torch": "2.x.y",
  "records": [ ... ],
  "census_at_peak": [ ... ],
  "final_peak_mb": 1234.5,
  "final_rsrv_mb": 2048.0
}
```

## Requirements

* CUDA-capable GPU.
* The C++/CUDA kernels in `lilytorch/src/kernels/` must be built
  (`pip install -e .`).
* The same FARMS dependencies as `run_cost_analysis.py`.

