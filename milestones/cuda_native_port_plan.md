# cuda_native_port — plan: native CUDA/C++ kernels + warp_port architecture

**Branch**: `cuda_native_port` (created from `warp_port`).
**Reference branch**: `cuda_kernels` (existing native CUDA/C++ kernels in
`lilytorch/src/kernels/csrc/`, plan docs in `milestones/`).

## Why

The Warp port (`warp_port`) turned out to be a dead end for the CPU story:
Warp does not support multithreaded CPU execution, so the original motivation
(single-source kernels replacing the dual CUDA + OpenMP-C++ maintenance
burden) does not hold. Decision: **return to the native CUDA/C++ kernel
approach**, but keep everything else the `warp_port` branch got right:

- The flattened, deduplicated `src/` tree (no `kernels/` package facade, no
  dead native/reference/demo tiers, tests centralized in `lilytorch/tests/`).
- The whole-step CUDA-graph capture architecture (`graph_capture.py`
  `WholeStepGraphRunner`, `multigrid_graph.py`, forces post-graph) —
  3.63 → 3.12 ms/step on the coupled benchmark, floor ≈ 2.5.
- Bug fixes made only on `warp_port` (must be re-expressed in native kernels):
  - ghost-ring gauge fix: apply `BC(p)` **before** `p -= p.mean()`
  - grow-only `max_vol` watermark (stale-AABB truncation fix, 4ce4946)
  - in-place projection corrections (`addcmul_`/`add_`) — pointer-stable
    `u0/v0/w0` (fresh-tensor graph-cache leak caused the salamander OOM)
  - deltaH force readout rewritten as static full-grid (killed the D2H sync)
  - adaptive MG early-exit + `float(tensor)` sync purge
  - Dirichlet mask in MG; static full-grid `bdim_forcing_2d`

## What exists where

| Piece | `cuda_kernels` (native) | `warp_port` (Warp) |
|---|---|---|
| Kernel sources | `src/kernels/csrc/cuda/*.cu` (~7k lines: streaming_sdf {2d,3d}, multigrid smoothers/transfer, poisson_solve, advection_flux, cvof_sweep, lagrangian_forces) + CPU twins (`*_cpu.cpp`, ~3.7k lines) | warp modules flattened into `src/` (`streaming_sdf.py`, `bdim.py`, `multigrid_graph.py`, …) |
| Op registry | `TORCH_LIBRARY(lilytorch_kernels)` in `ops.cpp`; python wrappers in `kernels/ops.py` (apply_bcs, bdim_coeff[_sigma], interp, jacobi/rbgs sweeps, mg_residual, whole-solve poisson_{multigrid,mgcg,rmgcg}, restrict/prolongate, streaming_sdf_stag_multi, streaming_sdf_forces_post, lagrangian_forces) | `facade.py` aggregator |
| Whole-step graph | **missing** (this plan, Phase 1) | `graph_capture.py` WholeStepGraphRunner |
| per-body key buffers | plan doc only (`milestones/per_body_key_buffers.md`) | n/a |

## Ground rules (apply to every phase)

1. **2D/3D from one source where it doesn't cost performance.** In C++/CUDA
   prefer a dim-templated kernel body (`template <int DIM>`) with thin 2D/3D
   launchers; duplicate only when the fused 3D memory layout genuinely
   differs (e.g. streaming SDF stagger).
2. **Simplify first.** Every phase must *delete* at least as much as it adds
   where possible: no per-simulation special cases, no duplicate 2D/3D python
   orchestration, no retired band-aids carried over.
3. **Parity gates.** No task is "done" without (a) the full pytest suite
   green, (b) a parity test vs the previous path (Warp result or torch
   reference) at ≤1e-10 rel for the same fp width, (c) a before/after
   ms/step number on the standard coupled benchmark.
4. **CPU path stays alive**: every CUDA kernel keeps (or gains) an
   `at::parallel_for` CPU twin. CUDA-graph capture is GPU-only; CPU runs the
   same ops eagerly.
5. Two-phase changes stay in `two_phase*.py` / their tests — never in
   `forces/solver/body.py` core paths.

## Agent budget strategy

Two pools:

- **Claude (Opus/Fable, limited credits)** — anything architectural,
  capture-safety reasoning, numerical-correctness review, and writing the
  *specs + parity tests* that gate the cheap agent. Short, high-leverage
  sessions.
- **DeepSeek v4 (cheap, less reliable)** — mechanical work executed against
  a written spec **with the parity tests already in place before it starts**.
  Rules for DeepSeek tasks: never touches `solver.py` step orchestration or
  graph-capture logic; must run the named test command and paste output;
  output is reviewed by a short Claude session before commit.

The pattern for every mixed task: *Claude writes the spec + failing/oracle
test → DeepSeek implements → Claude reviews diff + benchmark.*

---

## Phase 0 — Foundation: native kernels on the warp_port tree

**Goal**: `cuda_native_port` builds and passes the full suite with the native
extension restored, Warp kept temporarily as a test-only parity oracle.

- **0.1** ✅ Restore the native extension from `cuda_kernels`:
  `git checkout cuda_kernels -- lilytorch/src/kernels/csrc` plus `ops.py`,
  `interpolation.py`, `build.sh`, setup glue. Place it as `src/csrc/` +
  `src/native.py` (follow the flattened-tree convention — no `kernels/`
  package resurrection). Get it compiling.
- **0.2** Re-point the call sites: `facade.py` currently aggregates Warp
  kernels; swap each entry to the native op, one subsystem at a time
  (BCs → bdim coeff → streaming SDF → advection → poisson → forces).
  Keep the Warp modules importable under `tests/` as oracles until Phase 1
  lands, then delete them.
- **0.3** Port the warp_port-only fixes listed in "Why" into the native
  kernels/py-orchestration (most live in python orchestration and transfer
  directly; the gauge fix and Dirichlet mask touch `poisson_mult.py` /
  `poisson_solve.cu`).
- **0.4** Gate: full suite green; 600-step coupled parity vs `warp_port`
  head ≤ ~1e-8; benchmark ms/step recorded (expect roughly the
  `cuda_kernels`-era numbers, i.e. faster than Warp per-kernel eager).

**Agents**: 0.1 DeepSeek (mechanical restore + build fixes) after Claude
writes the target layout; 0.2–0.3 **Claude** (call-site semantics and the
fix ports are exactly where silent breakage happens); 0.4 DeepSeek runs
benchmarks, Claude reviews.

### 0.2 progress log (session 2026-07-10, Claude)

**Key architectural finding.** `facade.py` no longer "aggregates Warp kernels"
(the warp single-source refactor flattened each op into its own module). The
real call-sites are the per-module `*_warp` launch wrappers, and — critically —
**every core-path subsystem is fused into a Warp CUDA-graph runner**:
- pre-Poisson region (SL advection + diffusion + `bdim_forcing` + `apply_bcs`)
  → `graph_capture.WholeStepGraphRunner` (`wp.ScopedCapture`), driven from
  `solver._fluid_step_fused_{2,3}d`;
- force readout → `forces.ForcesPostGraph` (`wp.ScopedCapture`);
- streaming SDF `body_update_{2,3}d` → per-bridge graph in `facade.py`;
- Poisson → `WarpMG{2,3}D` graphed V-cycle in `multigrid_graph.py`.

So "swap to native" is not a rename: native ops cannot be recorded by a Warp
`ScopedCapture`, so **each subsystem must run native EAGERLY in Phase 0** (the
runner's eager path), and the native whole-step CUDA-graph runner is re-added in
**Phase 1** (port `WholeStepGraphRunner`/`ForcesPostGraph` from
`wp.ScopedCapture` → `torch.cuda.CUDAGraph`). Phase 0's 0.4 gate only asks for
`cuda_kernels`-era (eager) numbers, so this is the intended staging.

**CPU-twin gap (ground rule 4).** The restored native extension registers CPU
(`at::parallel_for`) twins for only: `apply_bcs`, `bdim_coeff{,_sigma}`,
`interp`, `lagrangian_forces`, `mg_residual`, `rbgs_sweep`,
`streaming_sdf_forces_post`, `streaming_sdf_stag`. **CUDA-only (no CPU twin):**
`cvof_sweep`, `advect_flux_add`, `jacobi_sweep`, `restrict_*`, `prolongate_*`,
`poisson_solve_{multigrid,mgcg,rmgcg}`. A subsystem whose CPU path is exercised
(two-phase cvof, CPU Poisson, CPU advection) **cannot** go native until its CPU
twin lands — else it violates ground rule 4. This gates cvof/advection/poisson.

**Done this session (native on the fast path, Warp kept as importable oracle):**
- `native.py`: added the missing `advect_flux_add` + `cvof_sweep` wrappers
  (registered in `ops.cpp` but previously unwrapped) with fake-tensor impls.
- **Lagrangian** (`forces.py` imports → `native.lagrangian_forces_{2,3}d`):
  exact positional drop-in, CPU + CUDA. Parity test added
  (`test_lagrangian.py::test_{2,3}d_native_eq_warp`, ≤1e-9 vs Warp).
- **Forces-post** (`forces.py` module aliases + `ForcesPostGraph.run` →
  native, eager): CPU path via native CPU twin, CUDA path native-eager.
  `test_forces.py::test_forces_{2,3}d_graph_replay_eq_eager` repurposed as the
  native-vs-warp parity gate (f64 ~3e-9, f32 ~2e-7; dtype-aware tol).
- **cvof**: native op is CUDA-only → **left on Warp** in `two_phase.py`
  (would regress CPU two-phase). CUDA parity gate added
  (`test_cvof.py::test_cvof_native_eq_warp`) documenting native==Warp on GPU,
  ready to swap once the CPU twin exists.

Full fast suite green (286 passed / 1 skip; the 2 `test_python_eulerian_force_
path_cpu*` failures are **pre-existing** on this WIP branch — a stale frozen
snapshot, confirmed by stashing my diff). Nothing committed (awaiting user).

**Re-scope decision (2026-07-10).** The literal 0.2 ("swap each subsystem to
native EAGER, one at a time, then re-graph in Phase 1") is the wrong shape for
the core path, for two reasons found this session:

- *Piecemeal eager-native benchmarks as a large regression.* The whole reason
  the pre-Poisson region is one Warp graph is to collapse ~130 µs of per-launch
  host overhead into a ~3 µs replay. Converting its members to eager native
  one-at-a-time re-exposes all of it, so 0.4's "cuda_kernels-era numbers" gate
  reads as a regression on every intermediate commit.
- *Graph-capturability is a kernel-shape constraint, not just a runner
  rewrite.* warp_port made `bdim_forcing` a **static full-grid** launch
  (pose-independent dim) precisely so it is CUDA-graph-capturable. The narrow
  native `bdim_coeff_{2,3}d` launches over the per-step **dirty AABB**
  (pose-dependent dim) → `torch.cuda.CUDAGraph` can't capture it cleanly. So
  Phase 1 likely needs a native *static-full-grid* bdim kernel anyway (mirror
  the warp one), and eager `bdim_coeff`-over-AABB caller glue would be
  throwaway. **TODO: verify against `streaming_sdf.cu` bdim launcher.**

So the remaining core-path work is reorganized into **four tracks** (A/B/C/D)
instead of the linear 0.2 list. The isolated subsystems stay in 0.2 (native
eager is a durable end state for them); the fused pre-Poisson region moves into
Phase 1 (convert straight to native-graph, no eager intermediate).

**Track A — isolated subsystems, native eager (durable; finish under 0.2).**
Not inside the whole-step graph, so eager native is the final state.
- ✅ lagrangian, ✅ forces-post (done this session).
- ✅ **interpolation**: `interpolation.py` now re-exports
  `native.RegularGridInterpolator` (CUDA + CPU twin) and deletes its duplicate
  Warp-backed class; the SL `@wp.func` samplers + the `interp_{2,3}d_warp`
  scattered-gather kernels stay as the parity oracle. Parity gates added
  (`test_interpolation.py::test_interp_{2,3}d_native_eq_warp`,
  `test_rgi_class_matches_warp_kernel`; CPU + CUDA, ≤1e-13 f64 / 1e-6 f32).
  Also fixed a pre-existing truncated assert in `test_interp_3d_cpu_eq_gpu`.
- ☐ cvof: **blocked on a CPU twin (Track C)** — native op is CUDA-only and
  two-phase runs on CPU. CUDA parity gate already in place.

**Track B — core pre-Poisson region → native CUDA-graph (executed AS Phase 1).**
BCs (`apply_bcs`), bdim, advection, streaming-SDF are all inside
`WholeStepGraphRunner`. Do NOT convert them to eager native piecemeal. Build
the native `torch.cuda.CUDAGraph` whole-step runner first (Phase 1.2), then
move each op into it, confronting the static-dim constraint up front:
- **BCs**: `adv_diff_solver.set_BCs` → `native.apply_bcs_{2,3}d`; build the
  neu/dir/ref descriptor tensors once (static). Has CPU twin.
- **bdim**: native `bdim_coeff` is narrower than warp `bdim_forcing` (which
  fuses BDIM + Maertens–Weymouth `div_corr` + full-grid pass-through +
  device-`rect_dev`). Either (a) write a native static-full-grid `bdim_forcing`
  mirroring the warp kernel (preferred — directly graphable, keeps the fusion),
  or (b) caller does static `u0.copy_(u_prime)` + `bdim_coeff` over the AABB
  (dynamic dim → not graphable, rejected). Reference the warp `bdim.py` kernel.
- **streaming-SDF**: `body_update_{2,3}d` (facade bridges) →
  `native.streaming_sdf_stag_{2,3}d_multi` (needs 4 persistent int64 key
  scratch buffers `key_cc/u/v[/w]`). Has CPU twin.
- **advection**: native has only `advect_flux_add` (per-direction flux-add,
  CUDA-only) — NOT the fused SL kernel, and there is **no** native diffusion
  op. Needs the `cuda_kernels` multi-launch flux path + a diffusion decision.
  Largest item; blocked on CPU twins (Track C).
- Correctness stays testable eagerly (capture-disabled path + per-op parity
  tests), so no testability is lost by skipping the eager-native commits.

**Track C — write the missing `at::parallel_for` CPU twins (parallel; DeepSeek).**
Satisfies ground rule 4 and unblocks cvof/advection/poisson. CUDA-only today:
`cvof_sweep`, `advect_flux_add`, `jacobi_sweep`, `restrict_*`, `prolongate_*`,
`poisson_solve_{multigrid,mgcg,rmgcg}`. Mechanical C++ mirroring each `.cu`,
gated by the existing CPU-vs-CUDA parity tests. Spec-driven, no orchestration.

**Track D — Poisson, native whole-solve (independent; own graph, not Track B).**
`poisson_mult.py` `WarpMG{2,3}D` V-cycle → native `poisson_solve_*`. Keep
`WarpMG` as the CPU fallback until Track C lands the CPU twin. 0.3's ghost-ring
gauge fix is already in `poisson_solve.cu` (see 0.3 log); the **Dirichlet mask**
lands here, where a live native path makes it testable.

**Suggested order:** A (finish interpolation) → C (unblock, runs in parallel) →
D (Poisson + Dirichlet, own graph) → B (core region, executed as Phase 1). The
fastest de-risk of the whole port is proving native == warp_port at the 600-step
coupled parity gate (0.4), which mostly depends on B + D landing.

### 0.3 progress log (session 2026-07-10, Claude)

**Inherited fixes (branched from `warp_port` → already present, verified):**
grow-only `max_vol` watermark (`facade.py:85/174`), in-place projection
corrections (`solver.py` `addcmul_`/`add_`), deltaH static full-grid
(`two_phase.py`), static full-grid `bdim_forcing_2d`/`_3d` (`bdim.py`), adaptive
MG early-exit + `float(tensor)` sync purge (`poisson_mult.py`). These "transfer
directly" because the branch point is warp_port — no action needed. The native
`poisson_solve.cu` MGCG/RMGCG drivers already carry the adaptive early-exit
(`tol >= 0` residual check) and are host-sync-free.

**Ported this session — ghost-ring gauge fix → `poisson_solve.cu`.** The two
`poisson_solve_multigrid_{2,3}d` drivers subtracted `p.mean()` over the *whole
ghost-padded* array without refreshing the ghost ring first. The per-sweep BC
inside the smoother (`multigrid_smoothers.cu`) writes only the FACE ghosts the
stencil reads, leaving the edge/corner ghosts stale — so the gauge mean was
biased by whatever the corners held. Fix: relocated the `apply_neumann_bc`
helper above the multigrid drivers and call it before the mean (mirrors the
warp `self.BC(p)` before `p -= p.mean()`). The MGCG/RMGCG drivers already refresh
`p` inside the CG loop, so only the two multigrid drivers were affected.
Verified: with a 1e3 seed in the corner ghosts, native-vs-warp interior fields
went from Δ≈1.6 (a full gauge offset) to ≤1e-8 (rbgs) / ~1e-7 (jacobi, solver
noise). Parity test added: `test_poisson_driver.py::
test_native_multigrid_gauge_matches_warp` (2D+3D × f32/f64 × rbgs/jacobi).
Rebuilt the extension (build.sh does not touch `poisson_solve.cu` — touch it +
rm its `.o` manually, or add it to the script). Full `test_poisson_driver.py`
green (58 passed). Nothing committed (awaiting user).

**Deferred — native Dirichlet mask.** The warp path pins `p==0` in masked cells
at every level/sweep, OR-downsamples the mask to coarse levels, and skips the
gauge mean when a mask is set (free-surface GFM only). Porting that into the
native CUDA multigrid means a mask-aware smoother + coarse-mask build +
mask-aware residual in `multigrid_smoothers.cu` — a substantial change that
CANNOT be validated through the production path yet (native poisson is still on
WarpMG; the mask is exercised only by the experimental GFM solver). It is
therefore bundled with the native-poisson swap (remaining-0.2 item 5), where a
live native path makes it testable. Doing it now would add speculative,
unvalidated CUDA (violates ground rule 2). `poisson_mult.py`'s Dirichlet-mask
orchestration is inherited from warp_port and unaffected.

## Phase 1 — Single fluid-step CUDA graph over native kernels

**This is Track B** (see the 0.2 re-scope): the core pre-Poisson region is
converted **directly** from Warp-graph to native-graph — there is no
eager-native intermediate for these ops. Building the native graph runner
(1.2) comes first, then each op (BCs, bdim, streaming-SDF, advection) is moved
into it, so the static-launch-dim constraint is handled at conversion time and
no throwaway eager caller-glue is written.

**Prerequisites**: Track C (CPU twins for `advect_flux_add` etc.) so the CPU
path survives ground rule 4; Track D (native Poisson) is independent and can
land before or after. Correctness is validated eagerly (capture-disabled
runner path + per-op native-vs-warp parity tests) *before* the graph is
switched on.

**Goal**: the whole pre-Poisson region (semi-Lagrangian advection, diffusion
accumulate, bdim forcing, apply_bcs) replays as **one** CUDA graph, for
2D and 3D, constant **and** variable diffusion — replacing per-op python
dispatch, mirroring `warp_port`'s `WholeStepGraphRunner`.

- **1.1 Capture-safety audit (Claude)**: for each native op in the step
  region, verify: no D2H reads, no allocations inside the op, static
  shapes/pointers across steps, dirty-rect data staged into persistent
  device buffers *outside* the graph (the `stage()`/`issue()` split from
  `graph_capture.py` — reuse that file, it is backend-agnostic in design).
  Known offenders to fix: anything sized to the per-step dirty AABB
  (lesson: static full-grid `bdim_forcing`), variable-diffusion coefficient
  updates.
- **1.2 Graph runner (Claude)**: port `WholeStepGraphRunner` from
  `wp.ScopedCapture` to `torch.cuda.CUDAGraph` (or raw
  `cudaStreamBeginCapture` in C++ if torch's pool semantics fight the
  extension's launches). Keyed cache on pointer signature, max-graphs
  eviction, eager fallback, replay/capture counters — all as on warp_port.
  **Must evict**, never pin (salamander OOM lesson).
### 1.2 progress log (session 2026-07-10, Claude)

**Runner landed** — `graph_capture.NativeWholeStepGraphRunner`, alongside the
Warp class (which stays production until the region ops go native; delete it
in row 8). Same `run(key, device, issue, stage)` contract and
capture-on-2nd-sighting life-cycle, so the solver swap is a constructor
change. Backend differences, by design:

- **Recording scope**: `torch.cuda.graph()` records whatever `issue()`
  enqueues on torch's CURRENT stream — verified all native ops launch via
  `at::cuda::getCurrentCUDAStream()` and the `.cu` files contain no
  syncs/mallocs (partial 1.1 audit). Raw `wp.launch` goes to Warp's stream
  and is silently DROPPED from the replay (the warp_port `use_cuda_graphs`
  lesson) — documented as a hard constraint on `issue()`.
- **LRU eviction, never pinning** (salamander OOM lesson): cache is an
  `OrderedDict`; when full the least-recently-replayed graph is `reset()`
  (frees its private memory pool) to admit the new capture — unlike the Warp
  runner's "cache full → stay eager", which pins dead graphs under pointer
  churn. `evictions` counter added; `_seen` book-keeping bounded.
- **Capture recipe**: pre-capture eager `issue()` on a side stream (torch's
  required warm-up; also this step's sole real output — capture records
  without executing), then `torch.cuda.graph(g)`. Temporaries allocated
  inside capture live in the graph's private pool and are reused on replay;
  per-step outputs must land in persistent buffers keyed by pointer.
- **Staging**: `stage()` still runs outside the graph, but replay is
  stream-ordered with `stage()`'s `copy_` on the same torch stream — the
  hard `torch.cuda.synchronize` the Warp backend needed is unnecessary
  (drop it from the solver's `stage()` at swap time).
- **Phase-2 composition note (as required by 2.x)**: host-side branches
  inside `issue()` are frozen at capture — regime A/B selection must be
  folded into `key`.

Tests: `tests/test_whole_step_capture_native.py` (5 tests, CUDA-gated) drive
the runner through a REAL native extension op (`mg_residual_2d`, allocates
inside the region) mixed with plain torch ops: capture/replay **bit-exact**
vs eager (`torch.equal`), staging freshness across replays, LRU eviction +
immediate recapture of a returning evicted key, CPU eager fallback, 100-replay
stress. All pass; existing Warp capture tests unaffected. Micro-bench:
replay submit 5.5 µs vs 13 µs eager for a 2-op region (win scales with
region op count). Remaining for row 7: full 1.1 per-op audit of the actual
region ops (advection/diffusion/bdim/BCs) as they are converted — folded
into row 8 where each op is moved into the runner.

### Bug fixes found via the megakernel investigation (session 2026-07-10, Claude)

Profiling the GPU-time split (verdict: pre-Poisson region is only ~5% of step
GPU time — a unified fluid-step megakernel is NOT worth it; the GPU time is
Poisson ~33% + projection/assembly elementwise chains) surfaced two bugs:

- **Native Poisson `p0=None` crash (row 6 gap) — FIXED.** `FluidSolver.project`
  passes `p0=None` on the `poisson_method: multigrid` cold-start fast path
  (WarpMG zeroed its level-0 buffer in place); `_native_multigrid` forwarded
  the `None` into the torch op → "tensor does not have a device". The native
  Poisson swap had therefore never run end-to-end through the solver (driver
  tests pass a real `p`). Fix: persistent zeroed padded buffer in
  `poisson_mult._native_multigrid` (no per-step alloc, pointer-stable).
- **`bdim_forcing_{2,3}d_warp` per-call dummy alloc inside capture — FIXED.**
  With `bdim_body_div_correction: False` (the solver default), the mw-off
  branch did `u0.new_zeros(1)` per call — a torch alloc on the legacy stream
  that crashes the whole-step `wp.ScopedCapture` ("operation would make the
  legacy stream depend on a capturing blocking stream") whenever the graph
  key repeats (streaming bodies, or allocator address reuse on the analytical
  path), and would otherwise bake a freed pointer into the captured graph.
  Fix: module-level persistent `_mw_dummy` cache per (device, dtype) in
  `bdim.py`. Verified in the exact failure mode (mw-off GPU capture + 9
  replays, finite fields). Carry-over note for row 8: the native runner
  inherits the same constraint — no allocs on the capture path for optional
  args; pass persistent dummies.

Full suite after both fixes: 345 passed / 1 skipped; only the 2 documented
pre-existing `test_python_eulerian_force_path_cpu*` failures remain (stale
frozen snapshot, present before these changes).

- **1.3 Variable vs constant diffusion**: one kernel with a
  `diff` tensor argument; constant case passes a broadcast/scalar path
  decided at capture time, not per step. No separate python branch per case.
- **1.4 2D/3D**: same runner, dim decided by key; kernels templated per
  ground rule 1.
- **1.5 Gate (DeepSeek runs, Claude reviews)**: parity eager-vs-graph
  bit-exact; benchmark 2D + 3D, constant + variable diffusion; expect
  ≥ the warp_port whole-step win (~130 µs → ~3 µs submit overhead for the
  region).

### Item 8 spec (session 2026-07-10, Claude) — move BCs / bdim / streaming-SDF / advection into the native runner

**Read this whole section before touching code.** This is the DeepSeek work
order for row 8. It replaces the four Warp subsystems in the pre-Poisson region
with native ops and switches the solver from `WholeStepGraphRunner` (Warp) to
`NativeWholeStepGraphRunner` (`torch.cuda.CUDAGraph`, already built — see the
1.2 log). The Warp modules stay importable as parity oracles until the whole
region is native and the 0.4 gate is green; only then delete them (row 8 tail).

**End state.** `FluidSolver._fluid_step_fused_{2,3}d`'s `_run_preproj` closure
issues ONLY native `torch.ops.lilytorch_kernels.*` ops + plain torch ops (no
`wp.launch`), so `NativeWholeStepGraphRunner` records the entire region on
torch's current stream and replays it in one host launch. `stage()` loses its
`torch.cuda.synchronize` (the native runner is stream-ordered — see 1.2 log).

**Hard constraints (from the 1.2 log + the megakernel bug fixes) — a violation
is a silent wrong-physics bug, not a crash:**
1. **No `wp.launch` inside `_run_preproj`.** Warp launches go to Warp's stream
   and are dropped from the torch graph replay (the `use_cuda_graphs` lesson).
   Every op in the closure must be a native `TORCH_LIBRARY` op (they launch via
   `at::cuda::getCurrentCUDAStream()`) or a plain torch op.
2. **No allocations on the capture path for optional args.** Pass persistent
   dummies for mw-off `sdf_cc`/`div_corr` (the `_mw_dummy` lesson) — the native
   `bdim_forcing` must accept a 1-element dummy the kernel never reads when
   `mw_on == 0`.
3. **Static launch dims.** Every region kernel launches over the FULL grid
   (pose-independent). The per-step dirty AABB is read from a device-resident
   int32 `rect` tensor staged OUTSIDE the graph — never a python-int launch
   bound. This is why the native `bdim_coeff` (AABB-dim launch) can NOT be
   reused and a new static-full-grid `bdim_forcing` is required.
4. **Pointer-stable outputs.** Every per-step output lands in a persistent
   buffer whose `data_ptr()` is in the runner `key`. No fresh `torch.empty_like`
   inside the closure on the steady-state path (the `_sl_out`/`_diff_out`
   buffers are already persistent — keep them).

**Native op inventory for the region (what exists vs. what DeepSeek writes):**

| Subsystem | native op today | action |
|---|---|---|
| BCs | `apply_bcs_{2,3}d` ✅ (CUDA + CPU twin) | swap the call only |
| streaming-SDF | `streaming_sdf_stag_{2,3}d_multi` ✅ (CUDA + CPU twin) | swap the facade bridge to native |
| bdim | `bdim_coeff_{2,3}d` (AABB-dim, no pass-through, no MW, no rect tensor) | **WRITE** static-full-grid `bdim_forcing_{2,3}d` (CUDA + CPU twin + op reg) |
| advection | `advect_flux_add` ✅ (flux path only) — NO native SL kernel, NO native diffusion | **WRITE** `sl_advect_{2,3}d` + `diffuse_add` (CUDA + CPU twin + op reg) |

**Do the sub-items in this order (each lands + its parity gate goes green
before the next). Order = ascending risk / ascending new-CUDA volume, so the
runner swap (8.E) rides on already-proven native ops.** All gates live in the
new file `tests/test_native_step_region.py` (Claude wrote the harness + oracles;
each native op's test is `SKIP` until its wrapper exists in `native.py`, then
must flip to `PASS` — a skipped item-8 test is a NOT-DONE item).

- **8.A — BCs (swap only, no new kernel).** In `advection.AdvDiffSolver.set_BCs`
  replace `apply_bcs_2d_warp` / `apply_bcs_3d_warp` with
  `native.apply_bcs_2d` / `native.apply_bcs_3d`, passing the SAME `cache` dict
  from `_build_fused_bc_cache{,_2d}` (the descriptor pack is backend-agnostic —
  the native op signature already matches `native.py:apply_bcs_{2,3}d`). Keep
  the eager python fallback untouched. Gate:
  `test_native_step_region.py::test_apply_bcs_{2,3}d_native_eq_warp` (2D+3D ×
  f32/f64) — **PASSES today** (native `apply_bcs` already exists), proving the
  swap is behavior-preserving. Do it first anyway to lock the gate before the
  set_BCs edit. **Parity caveat (verified this session):** native and Warp
  `apply_bcs` agree bit-exactly on the interior and all *face* ghosts, but
  differ at the multi-boundary **corner/edge ghost cells** (≥2 axes on a
  boundary) — the Neumann corner tie-break order differs. Those cells are never
  read by the 5/7-point stencil, so it is a harmless dead-cell difference; the
  gate masks them via `_live_bc_mask` (compare only cells with ≤1 axis on a
  boundary). Do NOT "fix" the corner divergence — it is immaterial and both
  paths are self-consistent.

- **8.B — streaming-SDF (swap bridge to native op).** The facade Warp bridges
  (`body_update_{2,3}d` in `facade.py`) run BEFORE the fluid step (in
  `composite_body.update()` via `BDIMhandler`) with their OWN per-bridge Warp
  graph. Replace them with a native path calling
  `native.streaming_sdf_stag_{2,3}d_multi`. Requirements:
  * Allocate the 4 (3-D) / 3 (2-D) persistent int64 key scratch buffers
    (`key_cc/u/v[/w]`, size `>= prod(grid)`) ONCE at solver/handler init —
    never per call (alloc-on-capture-path hazard). Same for the blend
    `num_*`/`den_*` buffers (pass zero-size or persistent zeros when
    `blend_eps == 0`).
  * The native op does NOT fold the `+FAR`/`0` prefill that the Warp bridge
    memsets inside its captured graph. Do the prefill with persistent torch
    `fill_`/`zero_` on the current stream INSIDE the captured region (so it is
    recorded), OR keep it in `stage()` — decide by whether streaming is folded
    into the whole-step graph (see note below).
  * **Scope decision (recommended):** keep streaming-SDF as its OWN native
    captured region first (mirror the facade bridge's per-bridge graph, but via
    `NativeWholeStepGraphRunner` or a second instance), NOT folded into the
    pre-Poisson whole-step graph. It runs at a different point in the step and
    consumes different inputs (body table + kinematics), so folding it in is a
    later optimization, not required for the 0.4 parity gate. Note this keeps
    two graph launches/step — the ~µs overhead is acceptable (mirrors Phase 3
    option 3). Gate:
    `test_native_step_region.py::test_streaming_sdf_native_eq_warp` (2D+3D,
    single body + two separated bodies + multi-link; native vs the
    `facade.body_update_{2,3}d` Warp bridge, ≤1e-9 f64 / 1e-6 f32).

- **8.C — bdim (WRITE native `bdim_forcing_{2,3}d`).** Port the Warp kernels in
  `bdim.py` (`bdim_forcing_{2,3}d_kernel` + `bdim_one_axis_{2,3}d`) to CUDA +
  an `at::parallel_for` CPU twin, register in `ops.cpp`, wrap in `native.py`.
  This is a SUPERSET of the existing native `bdim_coeff` (which stays for the
  BDIM-σ path); do NOT delete `bdim_coeff`. The kernel must reproduce, line for
  line vs. `bdim.py` (which is itself a faithful port of the retired native
  `bdim_one_axis` in `streaming_sdf.cu`):
  * static FULL-GRID 1-D launch (`dim = prod(grid)`, last axis fastest);
  * inside the device-resident `rect` AABB: BDIM2 velocity → `u0/v0[/w0]` +
    Weymouth-Yue Poisson coeff → face-grid `ch/cv[/cw]` (mind the 3-D
    face-grid offset `(i-1)*c_stride_i + (j-1)*c_stride_j + (k-1)` — 2-D writes
    at full-grid `g`);
  * OUTSIDE the AABB: pass-through `u0[g] = u_prime[g]` (etc.), leave `ch/cv`
    at their persistent `dt/rho` prefill;
  * `mw_on != 0`: full-grid Maertens-Weymouth `div_corr[g] = (1-mu0_cc)*div(u_b)`
    from the cell-centred `sdf_cc` / `eps_mw`.
  Proposed op schema (mirror the Warp wrapper arg order so the solver swap is a
  near-rename; `rect` is a device int32 tensor of length 4 (2-D) / 6 (3-D)):
  ```
  bdim_forcing_2d(Tensor u_prime, Tensor v_prime,
                  Tensor sdf_u, Tensor sdf_v, Tensor body_u, Tensor body_v,
                  Tensor(a!) u0, Tensor(b!) v0, Tensor(c!) ch, Tensor(d!) cv,
                  Tensor sdf_cc, Tensor(e!) div_corr, Tensor rect,
                  float eps, float rho_f, float dt, float h_grid,
                  float eps_mw, float inv_dx, float inv_dy,
                  int mu0_projection, int mw_on) -> ()
  ```
  (3-D adds `w_prime/sdf_w/body_w/w0/cw`, `inv_dz`, and 6-long `rect`.) The
  `native.py` wrapper mirrors `bdim.bdim_forcing_{2,3}d_warp` including the
  `_mw_dummy`-style persistent dummy for the mw-off `sdf_cc`/`div_corr`. Gate:
  `test_native_step_region.py::test_bdim_forcing_native_eq_warp` (2D+3D ×
  f32/f64 × mw-on/off × dirty-AABB and full-grid rect; ≤1e-9 f64 / 2e-7 f32,
  matching the forces-post dtype tols). **f64 must be bit-parity** with the
  Warp oracle (both are faithful ports of the same `.cu`).

- **8.D — advection (WRITE native `sl_advect_{2,3}d` + `diffuse_add`).** Largest
  item. Two new kernels (CUDA + CPU twin + op reg + `native.py` wrapper):
  * `sl_advect_{2,3}d` — fused RK2 midpoint semi-Lagrangian back-trace, one
    launch for all staggered components, writing persistent `out_*`. Mirror
    `advection.sl_advect_{2,3}d_warp` / `sl_advect_{2,3}d_kernel` exactly
    (same per-component grid origins/inv-spacings, same trilinear/biquadratic
    sampler as `interp_{2,3}d`). Signature mirrors the Warp wrapper arg order.
  * `diffuse_add` — explicit-diffusion in-place Laplacian accumulate; mirror
    `diffusion.diffuse_add_(target, copy_buf, dt, *, dh, nu_eff=None, nu=None)`.
    **1.3:** ONE kernel, `nu_eff` a Tensor arg; constant case decided at
    capture time by passing a 1-element `nu_eff` (or a null-tensor sentinel +
    scalar `nu`) — NOT a per-step python branch. Add the CPU twin (the flux
    path's `advect_flux_add`/`cvof` CPU twins from Track C are the model).
  Then in `AdvDiffSolver._solve_semi_lagrangian_warp` swap `sl_advect_*_warp` →
  native and `diffusion.diffuse_add_`'s Warp launch → the native `diffuse_add`
  (keep the persistent `_sl_out`/`_diff_out` buffers). Gates:
  `test_native_step_region.py::test_sl_advect_native_eq_warp` and
  `::test_diffuse_add_native_eq_warp` (2D+3D × f32/f64 × const+variable nu_eff;
  ≤1e-9 f64 / 1e-6 f32). SL back-trace is interpolation-heavy → f64 may differ
  at ~1e-12 from fma ordering; assert ≤1e-9 f64, document if tighter fails.

- **8.E — runner swap + drop the sync.** Only after 8.A–8.D are green: in
  `solver.py` change `WholeStepGraphRunner(` → `NativeWholeStepGraphRunner(`
  at both `_preproj_graph_{2,3}d` init sites (import from `graph_capture`), and
  delete the `torch.cuda.synchronize(u.device)` line from BOTH `stage()`
  closures (the native runner is stream-ordered with `stage()`'s `copy_`; the
  sync was a Warp-stream artifact — 1.2 log). Do NOT touch the `key` tuples,
  the `_run_preproj` structure, or the projection/forces after it. Gate:
  extend `tests/test_whole_step_capture_native.py` with a solver-driven
  `test_fluid_step_region_graph_eq_eager` — a real `FluidSolver` step (2D+3D,
  a streaming body) with the runner forced eager vs. captured, `torch.equal`
  bit-exact on `u0/v0[/w0]`, `ch/cv[/cw]`, and the projected output. Confirm
  `runner.captures>0`, `runner.replays>0`, `runner.evictions` sane over ~20
  steps under body streaming (pointer churn must NOT pin — salamander lesson).

- **8.F — 0.4 gate (row 9, DeepSeek runs / Claude reviews).** 600-step coupled
  parity vs `warp_port` head ≤~1e-8 + before/after ms/step on the standard
  coupled benchmark (2D + 3D, const + variable diffusion). Only after this is
  green: delete the now-dead Warp region kernels (`bdim.py`,
  `sl_advect_*`/`diffuse_add`/`apply_bcs_*_warp` in `advection.py`+`diffusion.py`,
  the `facade.py` bridges) and `WholeStepGraphRunner` from `graph_capture.py`,
  updating imports. Keep the Warp cvof/interp oracles only where a native CPU
  twin still gates a test.

**DeepSeek rules for this task (from the agent-budget section):** never edit
`solver.py` step orchestration beyond the two mechanical swaps in 8.E; never
touch the `key`/`stage` logic or `NativeWholeStepGraphRunner`; port each `.cu`
line-for-line against the Warp oracle (read the oracle first); run the named
test command and paste full output for every sub-item; one Claude review per
sub-item before commit. The parity tests are ALREADY in place
(`tests/test_native_step_region.py`) — a sub-item is done when its test flips
from SKIP to PASS with no other regression (full fast suite still 345 pass /
1 skip, minus the 2 pre-existing `test_python_eulerian_force_path_cpu*`).

## Phase 2 — per_body_key_buffers (independent; after Phase 0)

**Goal**: implement `milestones/per_body_key_buffers.md` (copy it onto this
branch from `cuda_kernels`) — eliminate union-AABB waste in the streaming-SDF
init/decode passes via Regime A (disjoint bodies: direct write, zero
init/decode) and Regime B (overlap-only resolve).

The doc is already a complete spec with an 8-step implementation order.

- **2.1** (Claude, short): reconcile the doc with the post-Phase-0 file
  layout; write the parity tests first: single body, two separated bodies,
  multi-link (salamander) — byte-identical vs the union-AABB path.
- **2.2** (DeepSeek): implement steps 1–5 of the doc's order (overlap
  detection → direct-write kernel → dispatch → per-body keys + resolve).
- **2.3** (DeepSeek bench + Claude review): benches per doc steps 6–7;
  then delete `streaming_sdf_init_keys_3d_kernel` + full-grid decode
  (doc step 8) — keep the deletion, no fallback rot.
- **Interaction with Phase 1**: regime selection (A/B) is per-step host
  logic → it becomes part of the graph **key**, not a branch inside the
  graph. Note this in the runner design (1.2) so the two phases compose.

## Phase 3 — Evaluate fluid_step ⊕ Poisson unification

**Goal**: a bench-backed **decision memo**, then implement only if it wins.

Facts to weigh (Claude, one focused session):
- `poisson_solve_{multigrid,mgcg,rmgcg}` are already whole-solve C++ ops —
  the python overhead there is mostly gone; the remaining cost is the CG
  `.item()` residual sync and the adaptive early-exit host decision.
- Adaptive MG early-exit (a warp_port perf win) requires a host read →
  incompatible with putting the solve inside one big graph. Options:
  1. **Fixed-iteration graphed solve** + residual check every N steps
     outside the graph (recapture/adjust N adaptively at the *step* level).
  2. CUDA ≥12.4 conditional graph nodes (device-side while-loop). Powerful
     but locks the toolkit version and complicates the C++ — likely fails
     ground rule 2.
  3. Keep Poisson as its own captured graph (as `multigrid_graph.py`
     already does on warp_port) chained on the same stream — two graph
     launches per step instead of one; overhead ~µs. **Expected outcome:**
     option 3 or 1; a fully unified single graph is probably ≤ a few µs/step
     better and not worth the rigidity — but measure, don't assume.
- Deliverable: `milestones/step_poisson_unification_memo.md` with numbers
  from Phase 1's benchmark harness; implementation (if any) spec'd for
  DeepSeek.

## Phase 4 — Two-phase solver: speed + complexity reduction

**Goal**: faster two-phase stepping and a substantially smaller
`two_phase_solver.py` (currently ~1150 lines both branches).

Known facts (from prior profiling): variable-coefficient Poisson is ~75% of
two-phase cost; **mgcg beats multigrid 2.2×** there; `cvof_sweep` native
kernel already exists (4.5–5.6×); deltaH is the correct force readout.

- **4.1 Profile on the native path** (DeepSeek runs the harness, Claude
  interprets): per-region timings for a standard two-phase case (sphere
  water-entry 3D + surface-pool 2D), post-Phase-1 so single-phase overhead
  is already gone.
- **4.2 Poisson**: make `mgcg` the two-phase default; check whether the
  variable-coefficient smoothers/residual are the same kernels as
  single-phase with a coeff tensor (they should be — unify if forked).
- **4.3 Fold two-phase extras into the step graph**: cvof sweep, alpha →
  density/viscosity update, density smoothing — all static-shape full-grid
  ops, so they append to the Phase-1 graph region. Variable coeff already
  covered by 1.3.
- **4.4 Complexity pass (Claude)**: reduce `TwoPhaseSolver` to a thin
  override of `FluidSolver` (alpha advection, coeff assembly, force
  readout); delete retired band-aids (gauge anchors, dead
  `advect_scalar`-style paths, experiment-specific toggles that no config
  uses — grep configs first; "duplicates" have been load-bearing before,
  trace reachability before deleting).
- **4.5 Gate**: two-phase suite + uniform-density parity tests (machine
  precision vs single-phase), sphere water-entry validation still PASSES,
  ms/step before/after.

## Suggested execution order & session budget

The 0.2 core-path work is reorganized into Tracks A–D (see the 0.2 re-scope);
the table below reflects that. Track B *is* Phase 1.

| # | Task | Track / Phase | Agent | Est. sessions | Status |
|---|---|---|---|---|---|
| 1 | 0.1 restore + build | 0.1 | DeepSeek (Claude spec) | 1–2 | ✅ |
| 2 | lagrangian + forces-post → native | A / 0.2 | Claude | — | ✅ |
| 3 | ghost-ring gauge fix → `poisson_solve.cu` | 0.3 | Claude | — | ✅ |
| 4 | interpolation → native + parity | A / 0.2 | Claude | — | ✅ |
| 5 | CPU twins (cvof, advect_flux_add, jacobi, transfer, poisson_solve) | C | DeepSeek (Claude spec) | 2 | ✅ |
| 6 | Poisson native whole-solve + Dirichlet mask | D | Claude spec + DeepSeek | 1–2 | ✅ (Poisson swap only) |
| 7 | native `torch.cuda.CUDAGraph` whole-step runner (1.1–1.2) | B / Phase 1 | **Claude** | 2 | ✅ (1.2 + partial 1.1; per-op audit folded into row 8) |
| 8 | move BCs/bdim/streaming-SDF/advection into the runner (1.3–1.5) | B / Phase 1 | Claude spec + DeepSeek | 2–3 | ◐ spec + parity tests DONE (see "Item 8 spec" above + `tests/test_native_step_region.py`); DeepSeek to implement 8.A–8.E |
| 9 | 0.4 gate: 600-step coupled parity + benchmark | 0.4 | DeepSeek runs, Claude reviews | 1 | ☐ |
| 10 | 2.1 per-body-buffer tests | Phase 2 | **Claude** | 1 | ☐ |
| 11 | 2.2–2.3 per-body buffers | Phase 2 | DeepSeek (Claude review) | 2 | ☐ |
| 12 | 3 unification memo | Phase 3 | **Claude** | 1 | ☐ |
| 13 | 4.1–4.3 two-phase perf | Phase 4 | DeepSeek (Claude spec/review) | 2 | ☐ |
| 14 | 4.4–4.5 two-phase simplification | Phase 4 | **Claude** | 1–2 | ☐ |

**Dependencies:** Track C unblocks the CPU side of D (Poisson) and B
(advection). Track D is otherwise independent (its own graph). Track B / Phase 1
is the last correctness-critical piece and the main perf win; the 0.4 gate runs
after B + D land. Phases 2–4 are independent of each other; if credits run low,
defer Phase 3 (pure evaluation) and 4.4 (pure cleanup) — they don't block
correctness or the main perf win.

Claude-critical sessions: the graph-runner build (row 7) and the
bdim/BCs/advection conversion specs (row 8); everything mechanical is
spec-driven DeepSeek work bounded by pre-written parity tests.
