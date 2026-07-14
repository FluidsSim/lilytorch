# Handoff — two pre-existing ghost-cell defects

Both were found while gating the Warp removal (commit `bf7e3a7`). **Neither was
introduced by it**, both reproduce on unmodified `b055aab`, and neither changes
interior physics today. Fix them anyway: issue A makes the solver
bit-irreproducible, which quietly destroys our ability to write bit-exact
regression gates, and issue B makes CPU and CUDA disagree on the pressure by a
constant, which forces every cross-backend comparison to special-case it.

Ground rules from `cuda_native_port_plan.md` still apply — in particular #4
(every CUDA kernel keeps a CPU twin) and #3 (no task is done without a parity
test and a before/after ms/step).

Vocabulary used below:

* **live cell** — interior, or a *face* ghost (exactly ONE index on a boundary
  plane). These are the only cells a 5/7-point stencil can read.
* **dead cell** — an *edge* or *corner* ghost (TWO or THREE indices on a
  boundary plane). Never read by any stencil in the solver.

---

## Issue A — `apply_bcs_{2,3}d` CUDA races on edge/corner cells (NON-DETERMINISM)

### Diagnosis (confirmed, not a guess)

`cuda/streaming_sdf.cu :: apply_bcs_3d_kernel` (and its 2-D twin in
`streaming_sdf_2d.cu`) dispatches **one BC op per `blockIdx.z`**:

```cpp
const int op = blockIdx.z;      // ALL BC ops run CONCURRENTLY
```

Every op writes one boundary plane of one component. A cell lying on **two**
boundary planes therefore receives **two concurrent writes from two different
source cells** — e.g. for all-Neumann, cell `v[0, j, 0]` is written by both

* the x-face op: `v[0,j,0] = v[1,j,0]`, and
* the z-face op: `v[0,j,0] = v[0,j,1]`.

The winner is whichever block lands last. That is a write-write race, and it is
non-deterministic run to run.

The CPU twin (`streaming_sdf_cpu.cpp :: apply_bcs_3d_cpu`) loops
`for (int op = 0; op < total; ++op)` **sequentially**, so it is deterministic
(last op wins) — which is also why CPU and CUDA disagree on these cells.

### Reproduction

```
3-D solver, N=24, default config (abdquickest + multigrid), CUDA, f64.
Run the identical sim twice, compare v0.
```

Non-deterministic **from step 1**. At 50 steps the interior is bit-exact and
`u0`/`w0`/`p0` are bit-exact; only `v0` differs, in a handful of cells, ALL of
which have exactly two indices on a boundary plane (e.g. `[0,2,0]`, `[0,5,0]`,
`[0,15,0]` — i.e. i=0 AND k=0). Magnitude in those cells: O(1e-1) at step 1,
O(1e-3) by step 50.

(Only `v` shows it for this config because of which BC ops happen to collide;
`u`/`w` are not immune in general.)

### The fix

Give the edge/corner writes a **deterministic tie-break**. Two acceptable
designs — pick whichever benches better, they are both cheap:

1. **Serialise by axis.** Launch one kernel per `axis` (so 2 launches in 2-D, 3
   in 3-D), each handling all ops for that axis. Cells are then written by at
   most one op per launch, and the inter-launch ordering fixes the winner. This
   matches the CPU twin's "last op wins" *only if* the CPU loop is also reordered
   to axis-major — do that, and make it explicit in a comment.
2. **Ownership rule inside the single launch.** Have each thread write a cell
   only if its `axis` is the *lowest* boundary axis of that cell; the op whose
   axis does not own the cell early-returns. One launch, no race, and the winner
   is defined by the rule rather than by the schedule. This is the cheaper
   option and I'd start here.

Whichever you pick, **the CPU twin must use the same rule** so the two backends
agree cell-for-cell, dead cells included.

### Gate

* New test in `tests/test_advection.py` (next to the existing `apply_bcs_*`
  tests): call `apply_bcs_{2,3}d` on CUDA **12 times on identical input** and
  assert the result is bit-identical every time. Today's kernel passes this on
  the synthetic `_bcs_problem_3d` descriptors but fails in the solver's
  all-Neumann config — so build the descriptors from a real `AdvDiffSolver`
  BC config (all-Neumann), not the synthetic set, or the test will pass
  vacuously.
* CPU twin == CUDA kernel over **all** cells (not just live cells). This is the
  test that currently cannot be written; when it passes, the issue is closed.
* Re-run the 50-step 3-D determinism check: two identical runs must now be
  bit-identical in *every* field including ghosts.
* The existing `tests/test_poisson_driver.py::test_smoother_3d_cpu_eq_cuda`
  masks dead cells via `_live_cells()`. Leave that alone — it is masking issue B,
  not this one.

---

## Issue B — the Poisson gauge is a mean over dead ghost cells

### Diagnosis

Every whole-solve Poisson driver ends with a gauge fix computed over the **full
ghost-padded tensor**:

```cpp
auto pmean = p.to(at::kDouble).mean();   // includes edge/corner ghosts
p.sub_(pmean.to(p.scalar_type()));
```

(`cuda/poisson_solve.cu` ~lines 291, 331, 448, 541; `multigrid_cpu.cpp`
`poisson_solve_multigrid_{2,3}d_cpu_impl`.)

Those dead corner cells hold different garbage on the two backends:

* `apply_neumann_bc_*` fills face ghosts; the CPU version leaves corners
  untouched, and the two backends do not agree on them.
* The CUDA Jacobi's ping-pong (`cuda/multigrid_smoothers.cu :: jacobi_sweep_3d_cuda`)
  `cudaMemcpyAsync`s the whole `tmp` buffer back over `p` when `nsmoothing` is
  **odd**. `tmp` is `at::zeros_like(p)` and its corners are never written, so
  p's corners get **zeroed** on odd sweeps and keep their prior value on even
  ones.

Net effect: the gauge constant differs between CPU and CUDA, so the two backends
return pressures that differ by a **constant offset**. Measured: interior
`p` agrees to ~3.5e-6 absolute with the residual difference being a pure
constant (3-D: std of the interior difference = 2e-18, i.e. exactly a constant).

Harmless *today* — the solver only ever consumes ∇p — but it is why
`tests/test_poisson_driver.py::test_poisson_cpu_agrees_with_cuda` has to compare
interiors *modulo a constant*, and it is a landmine for anyone who later reads
`p` absolutely (e.g. a pressure probe, a Bernoulli check, a reported Cp).

### The fix

Make the gauge depend only on cells that are actually part of the solution:

* Compute `pmean` over the **interior** only (`p[1:-1, 1:-1(, 1:-1)]`), in both
  the CUDA and CPU drivers, in all six `poisson_solve_*` entry points.
* Then re-apply the ghost BC so the ring stays consistent with the shifted
  interior.

This changes the returned `p` by a constant — which is exactly the point — so
expect frozen-value tests to move:

* `tests/test_forces.py::test_python_eulerian_force_path_cpu_regression` reads
  pressure forces; it will need re-freezing (it already documents that it tracks
  the Poisson driver). Re-freeze it and say why in the docstring.
* Nothing else should move. If a *velocity* field changes, stop — that means
  something is reading `p` absolutely and you have found a real bug.

While you are in there, consider making `apply_neumann_bc_*` fill the full ghost
ring (corners included) identically on both backends. That is a strictly good
change and it removes the last reason for the two backends' dead cells to differ.
Do NOT "fix" the Jacobi ping-pong by making `tmp` non-zeroed — the zeroing is
deliberate (see the comment there: uninitialised memory once leaked NaN into `p`
and blew up the coupled solve).

### Gate

* `tests/test_poisson_driver.py::test_poisson_cpu_agrees_with_cuda` should be
  tightened to compare the interior **without** the `d - d.mean()` step, and
  ideally the full padded tensor once the ghost ring is consistent.
* `test_smoother_3d_cpu_eq_cuda`'s `_live_cells()` mask should be removable.
* 50-step physics parity vs `bf7e3a7`: **velocity must be bit-identical**;
  pressure may shift by a constant.
* Suite green (`360 pass / 0 fail / 1 skip` today, minus whatever you re-freeze).

---

## Suggested order

Do **A first**. It is self-contained, and once the solver is bit-reproducible you
can gate B with an exact 50-step velocity comparison instead of a tolerance.
