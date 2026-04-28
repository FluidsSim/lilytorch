# Forces-method microbenchmark

Self-contained microbenchmark that compares the cost of the
`(SDF update + forces)` pipeline across the four most relevant
implementations of the BDIM forces stage that have been tested in this
repository.

## Why a separate benchmark?

The full ``cost_analysis_free_swimming_3d/run_cost_analysis.py``
benchmark drives the actual free-swimming 1guilla simulation through
FARMS / MuJoCo and is the right tool to measure end-to-end production
cost.  It needs the full FARMS stack (FARMS, MuJoCo, scikit-fmm,
Open3D, …) and a CUDA-synced timer harness, which is heavy and
GPU-oriented.  A standalone microbenchmark of just the forces stage
fits two needs that the full benchmark cannot serve:

1. **CPU / sandboxed environments.**  The microbenchmark runs without
   any FARMS dependency and in pure-CPU mode, so it can be executed in
   environments where the full benchmark is not available.

2. **Body-fraction sweeps in isolation.**  The user raised a concrete
   concern: with the new cached cc-SDF path, ``bdim_forces_3d_multi``
   reads each body's AABB sub-block from `sparse_cc_flat` instead of
   re-sampling.  AABB cropping is supposed to help when the swimmer
   covers a small fraction of the domain — but does it still pay off
   when the body fills most of the domain?  Sweeping the per-body AABB
   fraction is much cheaper here (no MuJoCo, no warm-up of the full
   solver) than re-deriving it inside ``run_cost_analysis.py``.

## What is the "new method"?

The recent ``copilot/optimize-speed-memory-high-priority-implementation``
PR changed ``bdim_forces_3d_multi`` so that, instead of re-sampling the
body SDF on the fly per cell (the way ``--streaming_forces_3d`` did
before), it reads the cached cc-SDF that ``streaming_sdf_min_3d_multi``
already wrote into ``sparse_cc_flat`` during the SDF update stage.
Functionally this **merges** the SDF compute and the forces compute
through the cache: the per-cell trilinear interpolation cost is paid
once during the SDF stage and re-used during the forces stage.

The benchmark therefore reports timings split into:

* ``SDF stage`` — ``streaming_sdf_min_3d_multi`` (writes the cache)
* ``Forces stage`` — the chosen forces variant
* ``Total`` — sum of the two

so the cache-shift between stages is visible in the numbers.

## Methods compared

| ID                       | What it is                                                                                                                |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------|
| ``kernel_cached``        | **NEW** — current production path. ``bdim_forces_3d_multi`` reads `sparse_cc_flat`.                                       |
| ``kernel_resample``      | **OLD** — pre-PR baseline. Uses ``bdim_forces_3d_multi_legacy_resample``, the legacy kernel recovered from git history.   |
| ``pytorch_narrow_batch`` | Pure-PyTorch reduction over packed `(B, D, D, D)` AABB sub-blocks. Mirrors solver.py's ``--force_narrow_batch`` path.     |
| ``pytorch_full_grid``    | Pure-PyTorch reduction over the full `(Nx, Ny, Nz)` grid. The naive no-cropping baseline the user explicitly flagged.    |

The two kernel variants share the same C++/OpenMP optimisations, so
their delta isolates the cost of "re-sample SDF per cell" vs "read SDF
from cache".

The legacy resample kernel is registered alongside the production op as
``lilytorch_kernels::bdim_forces_3d_multi_legacy_resample`` purely for
this benchmark — it must not be used in production code.

## Running

```bash
# Default sweep (grid 80×64×56, B = 4, fractions 0.10/0.30/0.50/0.70)
python bench_forces_methods.py

# Larger grid
python bench_forces_methods.py --grid 128 96 80 --reps 10

# Just two fractions, faster
python bench_forces_methods.py --fractions 0.20 0.60 --reps 4 --warmup 1
```

CSV + PNG/PDF land in ``figures/forces_methods/``.

## Reading the result (default sweep on CPU, 80×64×56, B = 4)

```text
body_aabb_frac = 0.10  →  AABB  8× 8× 8 (per-body  0.2% of cells, union ≲   0.7%)
  method                         SDF      Forces       Total   Forces/Total
  kernel_cached              5.80 ms     0.10 ms     5.90 ms          1.7 %
  kernel_resample            4.40 ms     0.16 ms     4.57 ms          3.6 %
  pytorch_narrow_batch       4.41 ms     1.09 ms     5.57 ms         19.6 %
  pytorch_full_grid          3.39 ms    24.37 ms    27.87 ms         87.5 %

body_aabb_frac = 0.30  →  AABB 24×19×17 (per-body  2.7% of cells, union ≲  10.8%)
  kernel_cached              8.93 ms     0.54 ms     9.46 ms          5.7 %
  kernel_resample            8.93 ms     1.51 ms    10.46 ms         14.5 %
  pytorch_narrow_batch       9.07 ms     3.13 ms    12.25 ms         25.6 %
  pytorch_full_grid          8.81 ms    32.68 ms    42.02 ms         77.8 %

body_aabb_frac = 0.50  →  AABB 40×32×28 (per-body 12.5% of cells, union ≲  50.0%)
  kernel_cached             22.22 ms     1.81 ms    24.07 ms          7.5 %
  kernel_resample           21.93 ms     6.64 ms    28.59 ms         23.2 %
  pytorch_narrow_batch      21.33 ms    13.48 ms    34.91 ms         38.6 %
  pytorch_full_grid         21.76 ms    29.03 ms    50.89 ms         57.1 %

body_aabb_frac = 0.70  →  AABB 56×45×39 (per-body 34.3% of cells, union ≲ 100.0%)
  kernel_cached             55.25 ms     4.53 ms    59.78 ms          7.6 %
  kernel_resample           55.57 ms    18.02 ms    73.59 ms         24.5 %
  pytorch_narrow_batch      55.01 ms    57.50 ms   112.76 ms         51.0 %
  pytorch_full_grid         55.20 ms    30.11 ms    85.31 ms         35.3 %
```

### Take-aways

1. **The new cached method wins at every body fraction.**
   Forces-stage cost only:

   | AABB frac | cached | resample | pytorch_NB | pytorch_FG | cached vs next-best |
   |-----------|-------:|---------:|-----------:|-----------:|--------------------:|
   | 0.10      | 0.10   | 0.16     | 1.09       | 24.37      | 1.6×                |
   | 0.30      | 0.54   | 1.51     | 3.13       | 32.68      | 2.8×                |
   | 0.50      | 1.81   | 6.64     | 13.48      | 29.03      | 3.7×                |
   | 0.70      | 4.53   | 18.02    | 57.50      | 30.11      | 4.0×                |

2. **Cache vs resample.**  The win grows with body size — at large
   bodies each re-sampled cell pays a 4× penalty over reading from the
   cache.  So the merge-via-cache that the new method performs is most
   valuable in exactly the regime the user worried about.

3. **The user's concern about cropping overhead is real — but only
   for the PyTorch narrow-batch path.**  At 70 % body fraction the
   pytorch_narrow_batch path costs 57 ms in forces, while the
   pytorch_full_grid path costs only 30 ms — the slice-write packing
   into `(B, D, D, D)` becomes net-negative when each AABB nearly
   covers the whole grid.  The C++ ``kernel_cached`` path does **not**
   suffer from this: 4.5 ms forces at 70 % fraction, still 6× faster
   than full-grid PyTorch.  The C++ kernel pays no per-body packing
   cost — each thread strides directly into the (already-packed)
   `sparse_cc_flat` cache.

4. **SDF stage dominates total cost at every fraction** (7–9× the
   forces cost in the cached method).  The recent PR therefore moves
   the bottleneck more squarely into ``streaming_sdf_min_3d_multi``;
   any future "where to optimise next" decisions should take the SDF
   stage as the primary target, not forces.

   **→ Followed up in step #3 (next section).**

5. **Caveats.**
   * Numbers are **CPU, single-thread, OpenMP-disabled-by-default
     PyTorch CPU build**.  GPU numbers will differ but the relative
     ordering is expected to hold (the resample kernel does strictly
     more work per cell than the cache kernel; the PyTorch full-grid
     path always touches every cell).
   * ``pytorch_full_grid`` does not produce per-body forces (it
     integrates over the *union* SDF), so it is included as a cost
     reference only.  In production, "no narrow band" actually means
     the full-grid C++/CUDA shared-stress kernel followed by per-body
     reduction — that hybrid is between ``kernel_cached`` and
     ``pytorch_full_grid`` in cost and is not separately benchmarked
     here.
   * The synthetic scene uses 4 spherical bodies on a small grid.  For
     end-to-end production validation use ``run_cost_analysis.py`` once
     a CUDA-equipped environment is available.

## How to extend

* **More methods** — add a ``forces_<name>(...)`` function and a new
  entry in the ``METHODS`` list.  The harness automatically times it
  with the same warmup / repetitions and adds it to the CSV / plot.
* **Larger grids** — pass ``--grid Nx Ny Nz``.  At 256³ the SDF stage
  alone is ~1 s/step in CPU mode; consider ``--threads`` to use more
  cores.
* **Per-body force comparison** — both kernel variants and
  ``pytorch_narrow_batch`` produce ``(B, 12)`` outputs.  Adding an
  assertion that they match within tight tolerance would extend the
  bench to a correctness check; this is currently delegated to the
  separate ``test_bdim_forces_self.py`` unit test.

## Step #3 — `streaming_sdf_min_3d_multi` micro-optimisation

Take-away (4) above identified the SDF stage as the next target.  The
follow-up changes (in `streaming_sdf_cpu.cpp` and `cuda/streaming_sdf.cu`)
exploit two algebraic simplifications that the previous code did not:

1. **Rotation CSE.**  The 4 sample positions per cell (cc + 3 face
   staggers) differ in world space by `±h/2` along **one** world axis
   each.  In body frame this is just `body_cc + Δ_k` where
   `Δ_k = -half_h · col_k(R_T)` is a per-body constant.  So 3 of the 4
   full-matrix rotations per cell were redundant.  The rotation is now
   computed once per cell (9 mul + 6 add) and the 3 face points are
   derived by 3 vector adds each.  Saves 27 mul + 18 add per cell.

2. **Uniform-grid trilinear.**  The body SDF tables `bx`, `by`, `bz`
   are uniform-grid by construction (BDIM builds them with `linspace`
   and the kernel already takes `inv_dx`, `inv_dy`, `inv_dz`,
   `inv_vol = 1/(dx·dy·dz)`).  Corner weights therefore reduce to
   `(1-frac, frac)` per axis — analytically.  The new
   `trilinear_sample_uniform` helper does **not** need:
   * the 6 axis-table loads per sample (`bx[ix]`, `bx[ix+1]`, …)
     ≡ 24 loads removed per cell across the 4 face samples,
   * the slow `floor` calls (3 per sample, 12 per cell on x86),
   * the trailing `* inv_vol` multiply (which cancels `dx·dy·dz`
     algebraically).
   It also evaluates the corner sum in factored form, cutting the
   multiply count from ~21 to ~14 per sample.

The op signatures are unchanged — the body axis tables and the
`bxL/byL/bzL/inv_vol` entries of `body_meta` are still accepted (callers
pre-allocate them) but go unused inside the new kernel.  Outputs match
the pre-change kernel to within float-associativity rounding (≤ 2 ULP
relative for fp64; verified by the new
`lilytorch/src/kernels/test_streaming_sdf_self.py`).

### Measured speed-up (CPU, 80×64×56, B = 4)

```text
                        SDF stage (ms)            Total per-step (ms)
  AABB frac     before    after  speedup        before   after
  0.10           5.39     5.24    1.03×           5.47    5.30
  0.30           8.37     7.45    1.12×           8.71    7.80
  0.50          16.79     9.72    1.73×          18.11   11.05
  0.70          42.47    20.38    2.08×          45.92   23.84
```

The speed-up grows with body size because per-cell arithmetic dominates
when the AABB is large (relative to the fixed OpenMP fork/join
overhead).  At the largest body fraction the **total** per-step cost
**halves** (45.9 → 23.8 ms).

The optimisation is mirrored in the CUDA kernel; on GPU the rotation
CSE saves register pressure and the uniform-grid path eliminates 24
global-memory loads per thread for the body axis tables.  Magnitude of
the GPU speed-up is hardware-dependent; verify with
`run_cost_analysis.py` once a CUDA-equipped environment is available.
