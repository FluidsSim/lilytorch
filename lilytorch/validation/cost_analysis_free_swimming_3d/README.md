# Cost analysis — 3-D free-swimming 1guilla

Computational-cost benchmark for the LilyTorch BDIM solver driving the
free-swimming 1guilla simulation.  Measures per-step wall-time
breakdowns across grid resolutions and produces paper-quality figures.

## Files

| File | Purpose |
|------|---------|
| `run_cost_analysis.py`          | Single-grid benchmark. Runs FARMS in-process with CUDA-synced timers around every major sub-kernel; writes CSV + per-grid figures. |
| `run_multigrid_cost_analysis.py`| Driver that launches the single-grid script in isolated subprocesses across several grids, then calls `plot_scaling.py`. |
| `run_scaling_conditions_pipeline.py` | Multi-condition wrapper around the multigrid driver. After substantial testing only two methods are exposed: the production method (`nbforces_opt`) and a no-cropping / no-batching reference (`nboff`). |
| `plot_scaling.py`               | Reads all `cost_breakdown_*.csv` files and produces multi-resolution scaling figures (stacked bars, log–log, % distribution). |

## Running

```bash
# Single grid
python run_cost_analysis.py --Nx 128 --Ny 32 --Nz 32

# Multi-grid scan (default "medium" preset)
python run_multigrid_cost_analysis.py
python run_multigrid_cost_analysis.py --preset production
python run_multigrid_cost_analysis.py --grids 128:32:32,256:64:64,512:128:128

# Regenerate only the combined scaling plots
python plot_scaling.py --data_dir figures/
```

## What is measured

`run_cost_analysis.py` monkey-patches the live `BDIMhandler` /
`FluidSolver` / `AdvDiffSolver` / `PoissonSolver` instances after the
pre-compilation phase, wrapping the **production** methods with
CUDA-synchronised timer blocks.  The production code paths are never
reimplemented, so the captured CUDA graphs / `torch.compile` recordings
are the ones being measured.

### Timer hierarchy

```
TOTAL step                               ← outermost, wraps one handler.step()
├─ 1  SDF update                         ← parent (wrapper overhead only)
│  └─ 1b   SDF eval                      ← LEAF  → "Body update (SDF eval)"
├─ 2  mu + normals                       ← LEAF  → "mu + normals"
├─ 3  fluid_step                         ← parent (wrapper overhead + BCs)
│  ├─ 3a   advection + diffusion         ← LEAF  → "Convection & diffusion"
│  ├─ 3b   BDIM meta-equation            ← LEAF  → "BDIM meta-equation"
│  └─ 3c   projection                    ← LEAF  → "Projection (pressure)"
│     ├─ 3c.i   Jacobi smoothing         ← (diagnostic only, NOT in categories)
│     └─ 3c.ii  V-cycle (top-level)      ← (diagnostic only, NOT in categories)
├─ 4  forces                             ← LEAF  → "Forces"
├─ 5  plotting & saving                  ← LEAF  → "Plotting & saving"
└─ 6  apply_forces (FARMS)               ← LEAF  → "FARMS step"
```

The paper categories map onto the **leaf** timers only, via label
prefixes chosen so that each leaf matches exactly one category and no
parent / sub-timer is double-counted:

| Category                   | Prefix   |
|----------------------------|----------|
| Body update (SDF eval)     | `"1b"`   |
| mu + normals               | `"2 "`   |
| Convection & diffusion     | `"3a  "` |
| BDIM meta-equation         | `"3b"`   |
| Projection (pressure)      | `"3c "`  |
| Forces                     | `"4 "`   |
| Plotting & saving          | `"5 "`   |
| FARMS (apply_forces)       | `"6 "`   |

> **Note on `"3c "`**: the trailing space is load-bearing.  On grids
> ≥ 500 000 cells the Poisson internals (`3c.i`, `3c.ii`) are
> additionally instrumented for diagnostics.  Those sub-timers are
> nested *inside* the `3c   projection` block and would double-count it
> if the prefix were just `"3c"`.  The trailing-space form matches only
> the outer leaf.

### "Other (residual)" — what it contains

```
Other = TOTAL step − Σ (leaf categories above)
```

This is a **true residual** and by construction makes
`Σ categories ≡ TOTAL step` (100% coverage).  In practice, "Other"
captures:

* Parent-timer wrapper overhead (`1  SDF update`, `3  fluid_step`)
  minus their leaves (should be tiny — the parent is effectively the
  leaf plus a few Python frames).
* `set_BCs` calls inside `_fluid_step_3d` that are not wrapped
  individually.
* `_release_bdim_fields()` and other attribute bookkeeping between
  numbered stages.
* Python-level overhead between timer blocks (attribute lookups,
  tuple packing, `types.MethodType` trampoline).
* Any GPU work that happens outside the labeled blocks (e.g. async
  kernels launched by previous stages still draining when the next
  timer starts — though the CUDA `synchronize()` inside the timer
  should prevent this from contaminating the *next* block).

The console report prints a coverage-check line
(`Σ categories / TOTAL step`) that should equal 100.00% to within
floating-point epsilon.

### What is deliberately excluded

* **Smagorinsky LES** (`cfg.smagorinsky_cs = 0.0`).  Disabled both in
  the advection–diffusion viscosity and in the force-computation path.
  A safety-net `RuntimeError` fires if a future override re-enables it
  without updating this script.
* **FlowViewer / CameraRecording / ExperimentLogger / MjcfSaver /
  DataLogger.**  Stripped from the simulation YAML so only
  `FluidExtension` runs — we measure pure solver cost, not
  visualisation or I/O pipelines.
* **Pre-compilation steps.**  The first `--precompile` (default 30)
  steps run the original (untimed) code path to trigger all
  `torch.compile` CUDA-graph captures.  Timing starts only after
  those are done, so compilation cost is not folded into the
  statistics.
* **The first `--discard_first` timed steps** (default 3) are dropped
  from the summary to exclude the post-patch CUDA-graph re-capture
  settling overhead.  They are still recorded in
  `cost_perstep_*.csv` marked `discarded`.

## Output

Each run writes to `figures/` (or `--out_dir`):

| File                              | Content                                   |
|-----------------------------------|-------------------------------------------|
| `cost_breakdown_<NxMxK>.csv`      | Aggregate mean/std/total/% per timer.     |
| `cost_perstep_<NxMxK>.csv`        | Raw per-step timings (for variance diag). |
| `cost_barh_<NxMxK>.pdf`/`.png`    | Horizontal bar — category breakdown.      |
| `cost_pie_<NxMxK>.pdf`/`.png`     | Pie — relative contribution.              |
| `cost_detailed_<NxMxK>.pdf`/`.png`| All sub-timers (not just paper categories)|
| `flow_fields_<NxMxK>.pdf`/`.png`  | Pressure, |u|, ω_z, SDF slice sanity plot.|
| `cost_scaling_stacked.pdf`        | Multi-grid stacked-bar comparison.        |
| `cost_scaling_loglog.pdf`         | Log-log scaling with O(N) and O(N log N). |
| `cost_scaling_pct.pdf`            | Relative (%) distribution vs grid.        |

## Suggested improvements

These are documented here per the original task request; they are
**not implemented** in this folder so as not to enlarge scope.

1. **Arithmetic-intensity / throughput columns.**  Add GFLOP/s and
   GB/s estimates next to ms/step by annotating each leaf timer with
   a closed-form cell-count multiplier (e.g. ≈ 24·N flops for the
   advection step, ≈ 7·N bytes read/write).  This decouples "is the
   solver slow?" from "is the GPU fed?".

2. **Warmup robustness.**  The current heuristic is `discard_first=3`.
   A more principled approach would be to compute the running mean of
   `TOTAL step` and start the window when the coefficient of variation
   drops below, say, 5 % over 5 consecutive steps.  Saves having to
   tune the discard count per grid.

3. **Memory profiling.**  Hook `torch.cuda.memory_allocated()` /
   `torch.cuda.max_memory_allocated()` at each timer boundary to
   report a per-stage peak-memory table.  Would also help diagnose
   why `_release_bdim_fields()` is in the hot path.

4. **Poisson inner-loop coverage.**  When `_instrument_poisson_internals`
   is active, add a coverage check
   (`3c.i + 3c.ii + top-level overhead == 3c`) to the console report,
   similar to the `Σ categories / TOTAL step` check.  Currently the
   internals are diagnostic-only — they are not used to verify that
   Jacobi + V-cycle account for the full projection time.

5. **Variance reporting in scaling plots.**  `cost_scaling_loglog.pdf`
   currently shows means only.  Error bars from the per-step CSV
   (already saved as `cost_perstep_*.csv`) would make the plot
   publication-ready.

6. **Dedicated `set_BCs` timer.**  `set_BCs` currently folds into
   "Other".  Moving it into its own leaf category would shrink the
   residual and clarify whether BC application is non-trivial at
   large N.

7. **CPU↔GPU transfer accounting.**  `6 apply_forces (FARMS)` bundles
   the MuJoCo physics step with the GPU→CPU force transfer and the
   CPU→GPU pose transfer.  Splitting these three (`transfer_out`,
   `mujoco_step`, `transfer_in`) would reveal whether the bottleneck
   at large N is the GPU kernel or the PCIe round-trip.

8. **Automatic grid-shape sanity check.**  `run_multigrid_cost_analysis.py`
   hard-codes Nx ≥ 128 so the fish body fits.  A pre-flight check that
   loads `cfg_mod.SimConfig()` and measures the fish extent would let
   us warn for arbitrary user-supplied `--grids` before spending an
   hour on a broken configuration.

## Design invariants (please preserve)

* Timer wrappers call the **original** (possibly compiled) methods —
  never reimplement a solver path, because that would desync the CUDA
  graph recorded during precompile.
* `Σ leaf categories + Other ≡ TOTAL step` — any new leaf timer must
  be added to `CATEGORIES` with a prefix that matches exactly that
  leaf and nothing else (use a disambiguating trailing space if the
  label would otherwise also match sub-timer prefixes).
* The `FluidExtension`-only simulation YAML must continue to strip
  non-fluid extensions so the benchmark stays reproducible.
