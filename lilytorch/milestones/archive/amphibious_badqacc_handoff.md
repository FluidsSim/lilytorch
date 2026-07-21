# Amphibious Pool BADQACC — RESOLVED (2026-07-14)

## Outcome

Fixed. `gen_config_amphibious.py` now runs **410+ iterations with bounded velocities**
(previously: NaN at iteration 15 → `mjWARN_BADQACC`).

The load-bearing change is **one line**: `rho_air: 1.2 → 12.5` in the `two_phase`
block (833:1 → 80:1 density ratio).

**It was never a `cuda_native_port` kernel regression.** See "Corrections" below —
several premises of the original handoff were wrong, and following them was a dead end.

## Root cause

The projection coefficient is `c = dt·μ₀/ρ`. In air, `c = dt/ρ_air` is
`ρ_water/ρ_air` = **833×** larger than in water, so the air converts *any* residual
pressure error into velocity with a gain of 833.

The geometric V-cycle contracts only **~0.99 per cycle** on this operator (measured
directly on the dumped coefficients — roughly 160 cycles per digit of error
reduction). At 10–30 cycles/step the pressure is nowhere near converged. A sub-1%
pressure error is enough: the air accelerated **linearly from step 1** (~1.7 m/s per
step ≈ 1700 m/s² unopposed), horizontally, at the top of the domain — measured at
`alpha=0`, `sdf>0`, i.e. in the **air**, **outside** the body. Velocity → NaN at
iteration 15–16 → huge coupling forces → MuJoCo BADQACC.

The coefficient corruption (`ch` going negative, then to ~0.9) at iteration 15 is a
*downstream symptom* of the velocity runaway (VOF pushes `alpha` out of [0,1]), not
the cause.

At 80:1 the same pressure error is amplified 66× less: `max|u|` at step 1 drops from
2.8 → 0.25 m/s and the air velocity **saturates** at ~1.2 m/s instead of diverging.

## Corrections to the original handoff

1. **"Runs fine on `cuda_kernels`, crashes on `cuda_native_port`" → the Poisson is
   IDENTICAL on both branches.** Verified head-to-head: cuda_kernels' pure-torch
   `_vcycle_jac_3d` and native `mg_vcycle_3d`, run on the *same* dumped operator,
   produce **identical** error traces (both converge on uniform coefficients; both
   crawl at ρ≈0.99 on the real one). The transfer ops (`restrict_face`,
   `restrict_residual`, `prolongate`) and the V-cycle recursion are the same code on
   both branches. So the kernels were the wrong place to look.

2. **cuda_kernels' Jacobi is NOT "asynchronous in-place".** It computes the full
   stencil sum before overwriting `p`, i.e. it is synchronous — same as the native
   double-buffered kernel. The config comment claiming otherwise (and blaming the
   "native CUDA synchronous-Jacobi driver") was false and has been removed.

3. **"Survives 120 s on cuda_kernels" is probably a wall-clock artifact.** Startup
   (mesh SDF generation + MuJoCo) eats most of 120 s. Given (1), cuda_kernels
   almost certainly blows up too, just later. This was never re-checked.

4. **The config had comments asserting the opposite of the truth.** It carried
   `convexify = True` and `zero_pressure_inside = True` annotated *"matches
   cuda_kernels"* — cuda_kernels has **both False**, plus `multigrid`/`jacobi` and no
   `python_body_update`. All of these have been reverted to the real cuda_kernels
   values. Prefer `git show cuda_kernels:<path>` over any in-file "matches X" comment.

   Note: reverting the config alone was **not** sufficient — it removed the NaN but
   MuJoCo still BADQACC'd at iteration 30. `rho_air` is the actual fix.

## Measured and RULED OUT — do not re-litigate

- **Poisson null-space / body disconnection.** Built the connectivity graph of the
  real operator: exactly **1** connected component, anchored, at every coefficient
  threshold (0 … 1e-7). No sealed pocket, no floating gauge.
- **`poisson_jcap_tol`.** No effect at any value from 1e-12 to 3e-6 (and it *hurts*
  above 1e-6). A cell's diagonal is dominated by its **largest** face coefficient, so
  BDIM band cells never land in the freeze window.
- **μ₀ face construction** — `avg(μ₀(sdf))` (the `_mu0_cc` path) vs `μ₀(avg(sdf))`
  (the python path): identical convergence traces.
- **More Poisson work.** 100 CG iterations + 2 preconditioner V-cycles + 10 smoothing
  sweeps made the blow-up arrive **sooner**, not later. Do not tune cycle counts here.

## Fragmented link geometry (detached tail / head shards) — also fixed

Separate from the blow-up: the eel rendered with its **tail detached** and **ragged
shards at the head**. Not a CUDA-graph/data_ptr problem (reproduced with graphs off).

The raw `link*_collision.stl` are **non-watertight**, so with `convexify=False`
open3d's ray-parity sign test speckles them — **link8 fragments into 17 components**
(a 2624-voxel core plus ~15 satellites of up to 194 voxels) and link0 keeps a stray
speck. `body.py` dropped only components **< 8 voxels** (an *absolute* cutoff), so all
16 of link8's big satellites survived. They are not cosmetic: they act as **phantom
BDIM obstacles** in the flow.

**`convexify = True` is a trap** — it produces perfectly clean geometry (1 component
per link) but **blows the sim up by ~iter 60** (`max|p|` 7.6e3 → 6.8e4 → BADQACC),
because the convex hulls of adjacent eel links overlap. That is the known hull-overlap
instability, and it bites the **eulerian** force path too, not just lagrangian. Keep
`convexify = False`.

**Candidate fix — TRIED AND VERIFIED, BUT NOT APPLIED (reverted at the user's
request; `body.py` is pristine).** Make the island filter in `body.py`'s 3-D SDF
tabulation **relative**: drop components smaller than `max(8, 0.10 × largest)` instead
of `< 8`. Measured result: exactly 1 component per link; a **no-op for any watertight
mesh** (single component ⇒ nothing dropped), so no other example changes; eel rendered
as one connected chain; physics unchanged (`max|u|` 1.19, `max|p|` ~7600 — identical to
baseline), stable past iteration 150.

So the geometry fragmentation is **still present** in the current tree. It is a
`body.py` mesh-cleanup issue, not a config one — do not try to fix it by convexifying.

The 2-D twin (`component_sizes < 4`) should be left alone regardless: a 2-D *slice*
through a single 3-D link can legitimately be multi-component, so a relative filter
there could delete real geometry.

**Latent trap:** the SDF interp cache (`interp_data_3d/`) is keyed on the **mesh
filename only**, not on `convexify`. With `compute_sdf=False` a convexify change would
silently reuse the wrong cached SDF. This config sets `compute_sdf=True`, which hides it.

## Still open

1. **CUDA-graph re-capture crash (~iter 200), a separate pre-existing bug** that the
   blow-up was previously masking. The pre-Poisson graph key includes the `data_ptr`s
   of the per-step scratch staggered SDF/body fields, which are freed and reallocated
   every step (`solver._fluid_step_fused_3d`, the `bdim_fields_scratch` block). An
   allocator reshuffle mints a new key mid-run, triggering a re-capture that dies with
   `RuntimeError: CUDA error: operation failed due to a previous error during capture`.
   **Worked around** with `solver["graph_capture_debug"] = True` (eager; costs speed,
   not correctness — eager and graphed traces are bit-identical). The real fix is to
   keep those scratch buffers pointer-stable, or drop them from the key.

2. **The Poisson is genuinely weak** (ρ≈0.99/cycle) for a two-phase operator with a
   large body. Capping `rho_air` is a *mitigation*, not a cure. Restoring physical air
   (1.2) requires a solver that actually converges — matrix-dependent/Galerkin coarse
   operators, or a proper coarse solve. Same recurring failure as the surface-pool eel
   (`project_surface_pool_refine_blowup`).

3. `max|w|` still creeps slowly (0.02 → ~0.79 over 400 iters) before flattening. Worth
   watching over a longer run; may be physical (surface waves + the swimming fish).

## Harness lesson (cost hours)

A standalone multigrid test that forms `b = A·p_true` **must set the Neumann ghost
cells on `p_true` first**. `mg_residual_3d` uses the ghosts as given, while
`jacobi_sweep_3d` overwrites them with `p_ghost = p_interior` — with unset (zero)
ghosts the two use *different operators*, `p_true` is not a fixed point, and the
V-cycle falsely reads as **divergent**.

Gate any such harness on two things:
- `p_true` must be a fixed point of the smoother (drift ~1e-16), and
- a **uniform-coefficient control** must converge — weighted Jacobi on an M-matrix
  cannot diverge, so if the control "diverges", the harness is broken, not the solver.

## How to run

```bash
cd /data/andreaferrario/lilytorch
/data/andreaferrario/venv_ns_312/bin/python lilytorch/examples/amphibious_pool/gen_config_amphibious.py --no-run
cd /data/andreaferrario/ns_data && bash run.sh
```
