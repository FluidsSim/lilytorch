# Handoff — two pre-existing ghost-cell defects  ✅ BOTH CLOSED

Both were found while gating the Warp removal (commit `bf7e3a7`). **Neither was
introduced by it**, both reproduced on unmodified `b055aab`.

**Status: fixed.** Issue A turned out to be worse than diagnosed below — see
"Correction" — and was a real interior-physics bug on CUDA, not just a
reproducibility annoyance.

Vocabulary used below:

* **live cell** — interior, or a *face* ghost (exactly ONE index on a boundary
  plane).
* **dead cell** — an *edge* or *corner* ghost (TWO or THREE indices on a
  boundary plane).

> ## ⚠ Correction to the original diagnosis
>
> The original write-up asserted that dead cells are "never read by any stencil
> in the solver", so the race was "numerically harmless" and only cost us
> bit-reproducibility. **That is false.** The wide / cross-term advection
> stencil *does* read edge and corner ghosts: perturbing **only** the dead
> ghosts of `u0/v0/w0` and taking a single step moves the **interior** velocity
> by ~8e-3 (3-D) / ~3e-3 (2-D).
>
> So the race was feeding schedule-dependent garbage into the interior. Measured
> on the 3-D all-Neumann config, 50 steps: CUDA's **interior** velocity
> disagreed with the (deterministic) CPU twin by **2.2e-3**. After the fix they
> agree to **3e-15**.
>
> (The claim was easy to believe and hard to check: a first attempt to test it
> came back "advection does not read dead cells" — but `AdvDiffSolver.solve()`
> returns *persistent buffers*, so that probe was comparing a tensor with
> itself. Clone before you compare.)

---

## Issue A — `apply_bcs_{2,3}d` CUDA raced on edge/corner cells ✅ FIXED

### What was wrong

`apply_bcs_{3,2}d_kernel` dispatched **one BC op per `blockIdx.z`**, so all ops
ran concurrently. A cell on two boundary planes got two concurrent writes from
two different source cells (all-Neumann: `v[0,j,0]` written both as `v[1,j,0]`
by the x-face op and as `v[0,j,1]` by the z-face op). Winner = whichever block
landed last ⇒ non-deterministic from step 1, and disagreeing with the
(sequential) CPU twin.

Worse, Neumann and Dirichlet ops shared one launch ("stage 1"), and a Dirichlet
wall-slot write (index 1 along the component's own axis) overlaps a Neumann /
reflective ghost plane of another axis at a **live** cell — so Dirichlet configs
had a live-cell write-write race too, and a read-write race on the Neumann
source.

### The fix — `csrc/bc_ops.h` (single-sourced, both backends)

Ops now run in three **ordered stages** — Neumann → Dirichlet → reflective —
one kernel launch each on CUDA, one loop each on CPU. That is the order the
eager Python reference applies them in, so every cross-kind overlap keeps the
value it had. Within a stage:

* **Ownership.** A cell claimed by several ops of the stage is written only by
  the lowest-indexed of them (= lowest axis, with the current packing). One
  writer per cell ⇒ no write-write race.
* **Composed source.** The owner reads its source stepped inward along its own
  axis *and* along every other axis whose same-stage op also claims the cell.
  That lands on a cell no op of the stage writes (destinations are ghost planes
  0 / n−1, sources are 1 / n−2) ⇒ no read-write race either. For all-Neumann
  this reproduces exactly what the old sequential CPU loop produced, so those
  cells do not move.

Only cells contested *within* a stage are dead cells, so live-cell values are
unchanged on CPU by construction.

### Gates (all green)

* `test_apply_bcs_solver_config_deterministic` — 12 identical CUDA calls,
  bit-identical, on descriptors built from a **real `AdvDiffSolver`** BC config
  (all-Neumann / mixed / all-Dirichlet, 2-D and 3-D). The old kernel fails 5 of
  those 6. It passes on the synthetic `_bcs_problem_*` descriptors, which are
  hand-picked disjoint — the trap the original handoff flagged.
* `test_apply_bcs_solver_config_cpu_eq_cuda` — CPU twin == CUDA over **all**
  cells, dead ghosts included. The comparison that could not be written before.
* 50-step 3-D all-Neumann solver run, CUDA f64: two identical runs are now
  bit-identical in **every** field including ghosts.
* CPU physics unchanged: 3-D velocity bit-identical before/after; 2-D moves only
  at 1e-16 (the issue-B gauge shift changes the rounding of ∇p).

---

## Issue B — the Poisson gauge averaged over dead ghost corners ✅ FIXED

### What was wrong

All six whole-solve drivers ended with `p -= p.mean()` over the **full padded
tensor**. The dead corners hold backend-dependent garbage (the CUDA Jacobi's
ping-pong `cudaMemcpyAsync`s a *zeroed* scratch buffer over `p` when
`nsmoothing` is odd; the CPU twin leaves them alone), so the gauge constant
itself was backend-dependent and CPU and CUDA returned pressures differing by a
constant.

### The fix — `csrc/poisson_gauge.h` (single-sourced, both backends)

* `gauge_fix(p)` — mean over the **interior** only, subtracted from the whole
  tensor (which leaves the Neumann ring consistent: ghost and interior neighbour
  shift by the same constant, so no re-BC is needed).
* `apply_neumann_bc_full(p)` — full ghost-ring refresh, corners included, in dim
  order. The CPU drivers used to call the smoothers' *face-only* pass here and
  left the corners stale.

Applied at all 10 gauge sites in `cuda/poisson_solve.cu`, both CPU drivers in
`multigrid_cpu.cpp`, and the 4 Python CG sites in `poisson_mult.py`
(`_gauge_fix`). The smoothers' per-sweep BC stays face-only — it is the hot path
and the drivers re-derive the ring at the end anyway.

### Gates (all green)

* `test_poisson_cpu_agrees_with_cuda` now compares the **raw** values with no
  `d - d.mean()` step, asserts the constant is actually pinned (removing the
  mean may not improve the agreement), and compares the **full padded tensor**,
  ghost ring included. All six method×ndim combos pass.
* **The frozen force snapshot did NOT move** and was not re-frozen. `p` shifts by
  ~13 in that config, and the pressure force is unchanged at rtol 1e-9 — a
  closed-surface ∮p·n integral is gauge-invariant. This is exactly the check the
  original handoff proposed ("if a velocity moves, something is reading p
  absolutely"): nothing does.
* `test_smoother_3d_cpu_eq_cuda`'s `_live_cells()` mask is **not** removable, and
  the test now says why: RBGS is compared on every cell (mask dropped), while
  odd-`nsmoothing` Jacobi legitimately zeroes the dead ghosts on CUDA (the
  zeroing is deliberate — uninitialised memory once leaked NaN into `p`). The
  test now asserts the mask is needed *only* there, so it can't quietly start
  hiding something else.

---

## Cost

600-step 0.4-gate benchmark, RTX 4080 SUPER, f32, same session:

| | before | after |
|---|---|---|
| 2-D | 9.68 / 9.74 ms/step | 9.86 / 9.98 ms/step |
| 3-D | 25.99 / 26.80 ms/step | 25.02 / 25.10 ms/step |

A wash — the extra launch (3 stages vs 2) and the per-thread ownership loop
(≤ 18 int compares) cost nothing measurable; the interior-mean gauge runs once
per solve. Suite: 372 pass / 1 skip (was 360 + 12 new).

## Follow-up worth considering

The advection stencil reading edge/corner ghosts is now *correct and
deterministic*, but it means those cells are load-bearing — they are not the
"don't care" region the codebase's comments still describe them as. Worth a
pass over the remaining "corners aren't read" comments, and worth deciding
whether the composed-inward-step value is the BC the scheme actually wants
there (it is what the CPU has always used, so this change did not alter it).
