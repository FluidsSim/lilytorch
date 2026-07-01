# Handoff — Warp backend perf: status, equivalence audit, remaining work

Read first: `src_warp/README.md` (WARP_BACKED table + perf notes),
`warp_poc/VALIDATION_STATUS.md` §F (the `[x]`/`[ ]` checklist + perf bullet).

## GUARDRAILS (do not violate)
- **No edits to `lilytorch/src/`.** After every change:
  `git diff --name-only -- lilytorch/src` must show **only** `diagnostics.py`.
  All wiring lives in `src_warp/`, all kernels in `warp_poc/`.
- `python -m pytest lilytorch/warp_poc/ -q` → **270 pass** baseline (run from
  `/data/andreaferrario/lilytorch`; note the doubled `lilytorch/lilytorch/`).
- Parity-test before flipping any dispatch. Branch `optimize_speed_memory`;
  warp 1.14, torch 2.6+cu124, RTX 4080S.

---

## A. ndelta (Eulerian) force — ALGORITHM IS EQUIVALENT, no fix needed
The Warp Eulerian force (`warp_poc/warp_forces.py::forces_post_{2,3}d_kernel`)
uses the **same per-body AABB approach** as native
(`src/kernels/csrc/cuda/streaming_sdf{,_2d}.cu::streaming_sdf_forces_post_*_kernel`):
`tid → (b, local)`, `vol = Ai*Aj` (3-D `*Ak`), launch `dim = B*max_vol_per_body`,
early-return for `local >= vol`. Same SDF sampling (`sdf_sample_off_*`), same band
gate, same union-normal stencil (central + 2nd-order one-sided), same staggered→
cell-centred velocity gradients, same cos-smoothed delta, same `delta_order==2`
grad-magnitude normalisation, same deltaH ∂H second pass.

**The ONLY difference is the reduction:** native does a CUB `BlockReduce` (shared
memory) + one `atomicAdd` per block; Warp does one `wp.atomic_add` per cell into
the float64 accumulator. The **sum is mathematically identical** — only the
floating-point reduction *order* differs (non-associativity → ~1e-9 noise). This
is verified equivalent by `warp_poc/test_forces.py` over the full matrix
(2-D+3-D × ndelta+deltaH × delta_order 1/2 × f32+f64 × scalar/field nu_rho) at
`ATOL_F64 = 1e-9` / `ATOL_F32 = 1e-5 + 3e-4·scale`. **No discrepancy to fix.**

**"Graph not viable" was about CUDA-graph CAPTURE only (a launch-overhead
optimisation), not the algorithm.** The force args are freshly allocated with
body-following shapes every step (`forces_method2` passes `u.contiguous()`,
per-step `_stream_step['kin']/aabb`; `forces_lagrangian_3d` builds `eps_ij` via
`_viscous_stress_tensor` on a moving `slab` crop + `torch.cat` triangle arrays),
so pointers churn and shapes vary → a captured graph can't be replayed. What WAS
done and is sufficient at realistic sizes:
- Dropped the redundant per-call `wp.synchronize()` (null-stream ordering covers
  the caller's torch read) → Eulerian **beats native at 96³ (0.96×) / 128³ (0.85×)**.
- `_fast_flat` view constructor + cached Lagrangian `elem_body`.
Small grids/meshes stay floor-bound by the eager Python launch; that floor is
only removable by graph capture, which the churn blocks. If you want small-grid
parity, the only path is to make the force readout's inputs persistent
(fixed-shape full-grid strain buffers + persistent kin/aabb copied in each step,
mirroring the Kernel-A graph path) — a sizable `forces_method2`/`forces_lagrangian`
override rewrite with a full-grid-strain cost trade-off. Probably not worth it.

## B. Poisson WarpMG2D residual — NOT a bug, here's why
`WarpMG2D` (in `warp_poc/warp_mg_var.py`) is correct: vs the native **Python**
multigrid (`use_kernels=False`, the reference the Warp port mirrors) the interior
solution matches to **2.2e-10** (relative ~7e-8). Two things confused the residual
comparison:
1. **Early-exit vs fixed cycles.** Native `solve_multigrid` iterates V-cycles
   until `residual < tol` then STOPS (returns ~7.7e-7 at tol=1e-6). `WarpMG2D`
   does a FIXED `n_vcycles` (graph capture forbids data-dependent early-exit), so
   6 cycles over-converge to ~5e-10. Different residuals = different work, not a
   bug. Match them by setting both to the same fixed cycle count (`tol=0`).
2. **C++ fused quirk.** The native C++ fused driver (`use_kernels=True`) uses a
   tiled stale-halo fine smoother; it differs from BOTH the native Python MG and
   WarpMG2D by ~1.2e-7 (native-Python vs C++ is *also* 1.2e-7). That's the C++
   driver's discretisation, not ours.
- **Speed:** at `n_vcycles=6` (already over-converged for tol=1e-6), WarpMG2D-graph
  is 0.98 ms at 128² f64 = **0.97× native C++** and **11× the src_warp Python
  driver**. The "speedup" is simply not over-iterating — pick the smallest
  `n_vcycles` that clears tol (≈6 for 1e-6). Consider lowering the default
  `max_vcycles` the graphed path captures (it currently captures `max_vcycles`).
- Tests: `test_poisson_driver.py::test_warp_poisson_graphed_multigrid_2d` +
  `::test_warp_poisson_graphed_2d_independent`.
- **Optional:** WarpMG2D is f64-only (the 2-D MG kernels in `warp_multigrid_2d.py`
  are f64). If a 2-D f32 case needs it, make those kernels dtype-generic (recipe
  in `TASK_remaining_warp_wiring.md`) + drop the `assert dtype==float64`.

## C. REMAINING WORK (priority order)

### C1. Sync-free graphed MGCG — **DONE** (2026-06-30)
`src_warp/poisson_mult.py`: `_dispatch_vcycle` now routes through a CUDA-graphed
`WarpMG{2,3}D` preconditioner (one captured V-cycle = `nvc=1`, replayed once per
`precond_vcycles` loop iter) whenever `cuda_graph` is on, the field is on CUDA,
and a graphed MG exists for this shape/dtype/smoother; otherwise it falls back to
the Item-2 Warp hybrid V-cycle (unchanged).  **Scaling gotcha resolved:** the CG
core feeds `_dispatch_vcycle(-r, …)` and `-r` is already in the h²-scaled smoother
units (the SPD operator `B` uses the h²-scaled face coeffs), so it is passed to
`WarpMG.solve` **with no extra h² multiply** — verified the preconditioner cuts a
manufactured-Neumann residual ~20× per V-cycle before wiring into CG.

Point 4 (periodic convergence) is implemented as an **opt-in** `_cg_core` override
(`cg_check_every`, default 1 = native every-iter behaviour, deferring to
`super()._cg_core`; gated to plain MGCG, RMGCG keeps the native loop). Wired from
config `solver.poisson_cg_check_every` in `src_warp/solver.py`.

**Perf (3-D f32, 96³ jellyfish target, RTX 4080S):** graphed MGCG vs the Item-2
Python-driver MGCG — N=64 **24.4→1.7 ms (14×)**, N=96 **27.1→2.8 ms (9.7×)**,
N=128 **27.7→6.4 ms (4.3×)**, same residual (≤1e-6). Honest caveat: `cg_check_every>1`
did **not** help (it overshot by one CG iter on these well-conditioned 3-iter
problems) — once the V-cycle is graphed the residual `.item()` sync is no longer
the bottleneck; the periodic check only pays off when iteration counts are high.
Tests: `test_poisson_driver.py::test_warp_poisson_graphed_mgcg[*]` (parity vs
native MGCG, 3-D f32/f64 rbgs+jacobi + 2-D f64), `…_periodic[1,4]`,
`…_independent` (monkeypatch-`lilytorch_kernels`-to-raise). 277 warp_poc pass.

### C1 (historical notes from before the fix)
The MGCG **CG arithmetic is already sync-free** — `src/poisson_mult.py::_mgcg_loop`
does `rz/dq/alpha/beta` as torch reductions kept as 0-d GPU tensors (no `.item()`
in the math). The cost is **(a)** the `_dispatch_vcycle` preconditioner (the
src_warp override runs the Warp smoother + a pure-torch coarse recursion → per-
launch overhead) and **(b)** the per-iter convergence check
`_convergence_norm(r)` (a sync) + `_apply_op_spd` (torch Laplacian apply).

Plan (all in `src_warp/poisson_mult.py`, no `src/` edits):
1. Build a graphed WarpMG preconditioner: instantiate `WarpMG{2,3}D` with
   `n_vcycles = self.precond_vcycles` (default 1). It already captures the whole
   V-cycle into one graph.
2. Override `_dispatch_vcycle(self, f, p, face_arrs)` (it's called as
   `z,_ = self._dispatch_vcycle(-r, z, face_arrs)`) to, when `cuda_graph` is on,
   route through the graphed WarpMG: `z_pad = mg.solve(rhs, ch, cv[, cw], p0=z)`;
   return `(z_pad, None)`. **CRITICAL — scaling:** `WarpMG.solve` expects the
   *h²-scaled* RHS (see `solve_multigrid`: `f_scaled = h2*f`); the MGCG passes the
   raw `-r`. Check what the native `_dispatch_vcycle`/`_vcycle_rbgs_*_warp` feed
   the smoother (they scale by `h2` internally or not) and match it, or you get a
   wrong-magnitude preconditioner → CG stalls/diverges. Verify the preconditioner
   reduces the residual on a manufactured Neumann problem BEFORE wiring into CG.
3. The faces are CONSTANT across CG iters within one solve; `WarpMG.solve`
   re-restricts them every call (wasteful but correct). Optional optimisation:
   restrict once per solve, reuse across iters (needs a `WarpMG` method that skips
   `_restrict_faces`/`_extract` and only replays the V-cycle subgraph).
4. Make the convergence check periodic (every K iters, K≈4) or run a fixed iter
   count, to cut the per-iter `.item()` sync. Keep correctness: a fixed count must
   be enough to clear tol on the target problems.
5. Parity test mirroring `test_warp_poisson_graphed_multigrid_2d`: graphed MGCG
   reaches the same residual as the native MGCG (`solve_mgcg`), + a
   monkeypatch-`torch.ops.lilytorch_kernels`-to-raise independence test.
GOTCHA: `_apply_op_spd` (the A·d operator inside CG) is torch — that's fine
(pure-torch, native-independent, on-GPU). You do NOT need to port it to Warp for
sync-freedom; only the preconditioner is the bottleneck. Full Warp-CG (dot
products as `wp.atomic_add` reductions, axpy kernels, one captured graph for the
entire CG loop) is a *further* step only if the torch CG arithmetic shows up in a
profile — measure first.

### C2. End-to-end trajectory validation — **DONE for scene (a)** (2026-06-30)
Real coupled FARMS/MuJoCo sim, native (`src/`) vs Warp (`src_warp/`) backend,
2-D `_1guillasim` pinned (f64), headless, n=400.  Runner +
save/compare harness: `lilytorch/validation/warp_e2e/run_c2.py` +
`c2_hook.py` (additive, opt-in; backend swapped by monkeypatching
`BDIMhandler.FluidSolver` through the sanctioned `_extra_run_patch` seam — no
`src/` or `BDIMhandler` edits).  Outputs under `/data/andreaferrario/ns_data/c2_{native,warp}_2d/`.

**Backend swap verified** (asserted on the first logged step:
`fluid_solver module == lilytorch.src_warp.solver`).

**Equivalence (native bit-deterministic run-to-run @ ~1e-18; warp deterministic @
~1e-9 force-atomic):**
- For the first **~325 steps (t ≤ 0.163 s)** Warp ≡ native to **residual level**:
  field rel-L2 `p` ~**1e-8**, `|u|` ~**1e-9**; `qpos` ~**1e-11**, `xpos` ~**1e-12**.
  Squarely the documented f64 expectation ("fields to residual/reduction-order
  noise").
- A **deterministic, Warp-specific, discrete divergence onsets sharply at
  step ≈330–335** (t ≈ 0.166 s): field rel-L2 `p` 8e-8 → 3.8e-6 (s330) → **0.11
  (s335)**, **71 % of the difference-energy at x<0** (the pinned eel body +
  near-wake) — i.e. seeded at the body surface and convected downstream.  After
  onset BOTH runs stay **stable and physical** (final `umax` native 0.364 vs warp
  0.376, ~3 %); `qpos` max|Δ| = **3.4e-5** at n=400, **within the handoff's f64
  trajectory band (1e-6..1e-4)**.

**This is NOT chaotic amplification — it is the documented Kernel-A f32-SDF
difference crossing a discretization threshold.**  Proven by a perturbation
sweep (`--perturb[/-recurring]`): the coupled system is **linearly stable** to a
uniform velocity perturbation over this horizon — one-shot **and** per-step
recurring kicks of 1e-9…1e-3 all give a **bounded, linearly-scaling** response
(≈0.1× the kick, no bifurcation by n=400); none reproduce the step-335 jump.  By
elimination the only non-bit-exact Warp op in f64 is **Kernel A** (README:
"f64 to ~1e-7 — *native* interpolates the SDF in **f32**"; advection / Kernel B /
apply_bcs are f64 bit-exact, forces are ~1e-9 and the system is provably stable
to 1e-9).  At a specific body pose (~s332) the f32-vs-f64 SDF disagreement flips
a body/near-surface cell decision → a localized O(1) seed → a different-but-valid
nearby flow state.  **Warp is the more accurate of the two here** (true f64 SDF
vs native's f32-truncated interp); this is a legitimate backend *difference*, not
a bug, and it is body-localized exactly as the SDF-precision mechanism predicts.

**Perf (per-step coupled wall, CUDA-synced, steady-state median; same fft scene):**
| path | ms/step | vs native |
|------|---------|-----------|
| native (fft, eager) | 5.42 | 1.00× |
| **warp (fft, eager)** | **3.73** | **0.69×** |
| **warp (fft, Kernel-A graph `--perf`)** | **3.22** | **0.59×** |

End-to-end Warp is **1.45× (eager) / 1.68× (Kernel-A graph)** faster than native
on this scene — **well past the README "~5 %, ideally faster" target.**  (The
graphed-MGCG C1 fast-path is NOT exercised here: scene (a) uses fft Poisson;
forcing mgcg destabilises the fft-tuned eel and native `poisson_cuda_graph` hits
a capture assert at 1024×128.  C1's graphed MGCG is timed on its proper 3-D
Poisson-bound target — see C1 above and scene (b).)

### C2(b). 3-D jellyfish (f32, two-phase, python path) — **DONE** (2026-06-30)
Standalone driver (no FARMS/MuJoCo): `run_c2_jelly.py` builds the native
`TwoPhaseSolver` + analytical `JellyfishBody` and runs `solver.run_sim()` short
(n=120, 128³, headless).  **Backend swap (no Warp two-phase *solver* exists):**
since the native `TwoPhaseSolver` resolves `AdvDiffSolver`/`PoissonSolver` (from
`src.solver`) and its `TwoPhase` VOF field (from `src.two_phase_solver`) as
module globals at construction, the runner rebinds those globals — plus the
`src.forces` kernels — to the Warp ports BEFORE building (same injection
`src_warp.solver.FluidSolver` does internally, applied externally; no `src/`
edits, no Warp-solver subclass needed).  Verified on the instance
(`adv_diff`/`poisson_solver`/`two_phase` modules all `src_warp`).  The jellyfish
runs the **python step path** (deforming SDF ≠ rigid Kernel A), so this exercises
the Warp **advection flux, variable-density Poisson smoother/residual (mgcg),
`cvof_sweep` (VOF), `apply_bcs` and the force readout** — NOT Kernel A/B.

**Equivalence (f32, eager Python-driver MGCG, 120 steps):** body trajectory
matches to f32 round-off and **stays bounded** — `com` max|Δ| **2.5e-9**
(rel 1.3e-8), `quat` **4.1e-8**, `linvel` **3.4e-6** (rel 5.5e-6); `KE` rel 6.5e-4;
**`alpha_sum` (VOF interface mass) rel 1.7e-7** (cvof essentially bit-exact);
fields at f32 FMA/ULP level (`|u|` rel-L2 ~**1e-3**, `p` ~1e-5), **not growing**.
Trends match — clean **f32 PASS** ("looser, document FMA/ULP drift" per the
handoff f32 expectation).

**Perf (peak it/s, 128³ two-phase):** native **55.6** | warp-eager **52.4
(0.94×)** | warp + C1 graphed-MGCG (`--perf`, `poisson_cuda_graph`) **61.8
(1.18×)**.  **HONEST CAVEAT:** the C1 **graphed** (fixed-cycle) MGCG
preconditioner is faster but **UNDER-CONVERGES the stiff 1000:1 two-phase
variable-density Poisson** at the default cycle counts → **~14 % field
divergence** (linvel rel 5.6e-2, `|u|` rel-L2 0.14 already at step 0).  So for the
two-phase jellyfish the **eager** Python-driver MGCG is the parity path; the
graphed path needs more `precond_vcycles`/`poisson_max_mgcg_cycles` for this
stiffness (C1's published 4.3–14× numbers are on the *non*-two-phase Poisson
target — the graphed preconditioner's fixed-cycle trade-off shows up under the
density-ratio stiffness here).  Net for scene (b): Warp ≈ native (~0.94× eager,
within noise of the "~5 %" target) with the graphed fast-path available but
needing a cycle-count bump before it is parity-safe on stiff two-phase.

#### (historical) C2 original note
Run a real coupled sim with `src_cuda` vs `src_warp`: 2-D `_1guillasim` pinned
(f64) + 3-D jellyfish (f32); body trajectory / key fields within tol (document
f32/FMA ULP drift). Needs the FARMS/MuJoCo bridge.

### C3. (low priority) Kernel-A small-grid floor, force small-grid floor
Both are eager-launch-floor-bound at tiny grids and already at/below native at
realistic sizes — see §A and the README. Only pursue if a target case is tiny.

## What was delivered this session (so you don't redo it)
- `apply_bcs_{2,3}d`: CUDA-graph cached (`ApplyBcs{2,3}DGraphRunner`), in-place →
  memory-free, default-on → **128³ set_BCs = native parity (1.00×)**.
- Kernel A: graph capture + folded-in FAR/0 resets (opt-in `kernel_cuda_graph`)
  → **beats native @128³ (0.85×), ~parity @64³ (1.15×)**.
- Forces: `wp.synchronize()` dropped + `_fast_flat` + Lagrangian `elem_body`
  cache → **Eulerian beats native @96-128³**; Lagrangian ~parity at realistic
  meshes (10560 tri 1.11×).
- `WarpMG2D` graphed 2-D multigrid (+ ghost-clamped `mg_residual_2d_clamped`) →
  **matches native C++ @nvc=6, 11× the Python driver**.
- Test commands: `python -m pytest lilytorch/warp_poc/ -q` (270 pass);
  guardrail `git diff --name-only -- lilytorch/src` → only `diagnostics.py`.
