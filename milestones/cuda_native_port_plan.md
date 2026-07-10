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

- **0.1** Restore the native extension from `cuda_kernels`:
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

## Phase 1 — Single fluid-step CUDA graph over native kernels

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
- **1.3 Variable vs constant diffusion**: one kernel with a
  `diff` tensor argument; constant case passes a broadcast/scalar path
  decided at capture time, not per step. No separate python branch per case.
- **1.4 2D/3D**: same runner, dim decided by key; kernels templated per
  ground rule 1.
- **1.5 Gate (DeepSeek runs, Claude reviews)**: parity eager-vs-graph
  bit-exact; benchmark 2D + 3D, constant + variable diffusion; expect
  ≥ the warp_port whole-step win (~130 µs → ~3 µs submit overhead for the
  region).

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

| # | Task | Agent | Est. sessions |
|---|---|---|---|
| 1 | 0.1 restore + build | DeepSeek (Claude spec) | 1–2 |
| 2 | 0.2–0.3 call sites + fix ports | **Claude** | 2–3 |
| 3 | 0.4 gate | DeepSeek | 1 |
| 4 | 1.1–1.2 audit + graph runner | **Claude** | 2 |
| 5 | 1.3–1.5 diffusion/dims + gate | DeepSeek (Claude review) | 1–2 |
| 6 | 2.1 tests | **Claude** | 1 |
| 7 | 2.2–2.3 per-body buffers | DeepSeek (Claude review) | 2 |
| 8 | 3 unification memo | **Claude** | 1 |
| 9 | 4.1–4.3 two-phase perf | DeepSeek (Claude spec/review) | 2 |
| 10 | 4.4–4.5 two-phase simplification | **Claude** | 1–2 |

Claude-critical sessions: ~7–9 total, each scoped to one row. Everything
else is spec-driven DeepSeek work bounded by pre-written parity tests.
Phases 2 and 3 are independent of each other and of 4; if credits run low,
defer 3 (pure evaluation) and 4.4 (pure cleanup) — they don't block
correctness or the main perf win (Phases 0–1).
