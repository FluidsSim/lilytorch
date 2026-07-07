# Host-bound coupled step — speed-up work plan

Date: 2026-07-06 · branch `warp_port` · status: **PLANNED — tasks specced for
parallel agents**

## Context and baseline (measured, do not re-derive)

Reference case: coupled salamander 2-D (`examples/salamander_gamepad/
gen_configs_swim_2d.py`, 1024×512, f32, MG Poisson, `convection_method:
"implicit"` = semi-Lagrangian), RTX 4080 SUPER, headless, working tree of
2026-07-06 (includes the CUDA-graph streaming/forces defaults and the
`FluxAddGraphRunner` / `ApplyBcs*GraphRunner` spikes).

**5.35 ms/step wall. GPU busy ≈ 1.9 ms/step. ~65 % of every step is host-side
Python/dispatch** (submit-only time ≈ GPU-synchronised time for every phase).
GPU-side kernel tuning is pointless until host time < GPU time.

Per-step phase breakdown (sync-attributed):

| phase | ms/step | notes |
|---|---|---|
| semi-Lagrangian advection | **2.39** | 10 eager `RegularGridInterpolator` calls = 1.80; diffuse 0.17; torch midpoint/`.clone()` glue ≈ 0.4 |
| Poisson project + correct | 1.28 | rbgs sweeps 0.77 GPU (336 halfsweeps, 2–3 v-cycles); ~0.4 host around the graph |
| Heun glue / copies / misc in `step_` | ≈ 1.0 | 11 `aten::copy_` + 8 DtoD memcpys/step |
| `bdim_forcing_2d` | 0.30 | 1 eager Warp launch; dirty-AABB dim blocks capture |
| `set_BCs` | 0.25 | graph runner spike in tree, still partially eager |
| MW body-div correction | 0.14 | eager torch ops |
| forces + stream_sdf + apply_forces | 0.47 | already graphed — near floor |
| FARMS/MuJoCo remainder | ≈ 0.4 | out of scope |

Counters: 14.7 eager `wp.launch`/step (10 = SL interp), 4.35 graph
replays/step, 0 CUDA `.item()` syncs/step.

**Target: ~2.5 ms/step (≈ 2.1×)** — GPU floor 1.9–2.0 + MuJoCo 0.4.

## Measurement protocol (mandatory for every task)

```
python lilytorch/benchmarks/prof_coupled_step.py
```

Runs a 377-step headless coupled salamander sim and writes
`benchmarks/prof_coupled_step_report.json`: untouched throughput window
(steps 150–249), sync-attributed phases (250–329), submit-only phases
(330–369), torch.profiler CUDA window (370–376), per-step launch/replay/sync
counts. The harness injects `benchmarks/_prof_coupled_step_patch.py` into the
FARMS subprocess via `_extra_run_patch()`.

Every task PR must include: before/after `throughput_ms_per_step_untouched`,
the phase row(s) it claims to improve, and the full pytest suite green
(baseline: 228 passed / 1 skipped). Physics gates per task below.

## Task 0 — validate + commit the current working tree (BLOCKS all others)

The tree carries uncommitted spikes: graph-default forces/streaming, static
full-grid deltaH, `FluxAddGraphRunner`, `ApplyBcs{2,3}DGraphRunner`, dirty-AABB
`.item()` changes in BDIMhandler (~680 lines across 20 files). The milestone
`cuda_graph_streaming_forces_spike.md` lists live-FARMS validation as pending.

1. Run a real coupled salamander run ≥ 3000 steps, quiet water: check
   `fluid_solver._forces_post_graph_2d.replays` grows ~1/step, no
   ForcesPostGraph churn RuntimeWarning, drags match a pre-change commit ≤1e-9.
2. Commit (no Co-Authored-By trailer — user preference).

All other tasks branch from that commit. Do not stack on a dirty tree.

## Task A — fused semi-Lagrangian advection kernel (expected −2.0 ms/step)

**The single biggest lever.** `AdvDiffSolver._solve_semi_lagrangian`
(`src/advection.py:375-423`) does, per 2-D step: 10 interpolator calls
(each = 5 `wp.from_torch` wraps + eager `wp.launch`, ~120 µs host apiece),
~12 fresh `.clone()` allocations, and eager torch arithmetic for
midpoint/departure points. GPU work is ~150 µs; the rest is host overhead.

Spec:
* One Warp kernel per velocity component set (or one fused kernel writing both
  staggered components): RK2 back-trace entirely in registers —
  sample (u,v) at the grid point, form midpoint `x − 0.5·dt·u(x)`, sample at
  midpoint, form departure `x − dt·u_mid`, sample the advected component at
  departure. **The interpolation is QUADRATIC, not bilinear** — the SL
  interpolators are built with `method="quadratic"`
  (`advection.py:_init_semi_lagrangian`); the in-kernel sampler must be
  `biquadratic_sample_off_2d` (`interpolation.py:251`), which already exists.
  Follow the T2a fused-flux kernel pattern (`advect_flux_add_warp`, same file
  conventions, dtype-generic via `wp.overload` f32/f64). Free micro-win while
  here: the stage-1 `d == i` sample is the component at its own nodes — an
  identity read, no interpolation needed (only 8 of the current 10 samples do
  real work).
* Inputs: the component fields + grid descriptors, all persistent buffers.
  Output into a persistent `vel_new` buffer (no per-step allocation) — this
  makes the launch pointer-stable and graph-capturable; add a graph runner
  like `FluxAddGraphRunner` (advection.py:1158) keyed on pointers+scalars.
* Keep the explicit-diffusion pass separate at first (it is 0.17 ms); fuse
  later only if profiling justifies it.
* Boundary handling: replicate the interpolator's clamping semantics exactly
  (read `RegularGridInterpolator` for edge behaviour) — parity gate below.
* 3-D twin (`interp_3d`/triquadratic path) is a follow-up, same recipe;
  land 2-D first.

Acceptance:
* New pytest in `tests/test_advection.py`: fused kernel vs the existing
  python SL path, f32/f64, CPU-vs-GPU, random fields + random BCs, tolerance
  machine-precision-tight (this is a re-implementation of the same math, not
  a scheme change).
* Coupled salamander 600-step drag trace vs Task-0 baseline ≤1e-6 relative
  (interp order identical ⇒ should be far tighter).
* Profile: `advdiff_solve_sl` row from 2.39 → ≤0.4 ms/step.

Pitfall: the python path must stay selectable for CPU and as test oracle —
do NOT delete it; gate on `is_cuda` like the other runners.

## Task B — graphable bdim_forcing + MW-div fold + copy audit (expected −0.5…−0.8 ms/step)

**DONE 2026-07-06** (measured on the tree carrying Task A's in-flight SL work,
same-tree before/after): throughput 3.626 → 3.117 ms/step (−0.51);
`bdim_forcing` 0.316 → 0.047; `mw_body_div_corr` 0.129 → 0 (row gone — folded
into the kernel, ~11 µs GPU inline); −2 `aten::copy_` and −2 DtoD per step
(the upfront u0/v0 full-grid copies).  Implementation: full-grid static
`bdim_forcing_2d_kernel` (BDIM inside the device-resident rect descriptor,
pass-through `u0 = u'` outside, flag-gated full-grid MW term) +
`BdimForcing2DGraph` runner (pinned-host → device async rect staging,
2nd-sighting capture, pointer+scalar keyed).  Note on sub-item 1's
BDIMhandler `.item()` claim: the streaming path computes the dirty AABB in
host numpy from MuJoCo poses — there were no GPU `.item()` syncs to remove
(counter: 0/step before and after); lines 640-643 are the non-streaming
python body path, off the critical path.  Parity tests in `test_bdim.py`
(pass-through exactness, MW fold vs torch oracle, graph-vs-eager over a
moving rect, capture/replay counts).  3-D twin not done (same recipe applies).

Three sub-items, one agent:

1. **`bdim_forcing_2d` static rewrite** (`src/bdim.py:448-479`, kernel at
   :404): the launch dim is the per-step dirty AABB (`dirty_Ai·dirty_Aj`), so
   the graph runners can't capture it. Rewrite exactly like the deltaH force
   pass (see `milestones/cuda_graph_streaming_forces_spike.md`): full-grid
   static launch; each thread early-returns outside the dirty band, using a
   device-resident dirty-rect descriptor staged with an async `copy_` (the
   kin/aabb staging pattern in `forces.ForcesPostGraph._stage`). Then add a
   graph runner. Also removes the host-side `.item()` AABB math in
   `BDIMhandler` (~lines 640-643) from the fluid critical path if it becomes
   device-resident. Note the early-return must preserve the semantics that
   cells OUTSIDE the dirty rect keep `ch/cv` and `u0/v0` untouched — that is
   already true (kernel only writes inside the rect), so full-grid + return is
   exactly result-preserving. −0.25 ms.
2. **Fold `_mw_body_div_correction`** (solver.py, called from
   `_fluid_step_fused_2d` step 5b) into the same kernel or a second static
   Warp kernel: it is a divergence of the staggered body velocity into the
   Poisson RHS — a few lines in-kernel, kills ~0.14 ms of eager torch ops.
   Gate: `bdim_body_div_correction` flag must still work.
3. **Copy audit in the fused step** (`solver.py:_fluid_step_fused_2d`):
   `self.u0.copy_(primes[0])` (line ~1788) is followed by `bdim_forcing`
   reading `primes` and writing `u0` over the full dirty rect — the upfront
   full-grid copy may be reducible to outside-rect only, or removable if the
   kernel writes all cells it reads. 11 `aten::copy_` + 8 DtoD per step ≈
   0.3 ms GPU + dispatch; document each copy you keep with why. **Landmine:**
   `u0/v0` must remain the persistent `self._vel` rows — ForcesPostGraph
   captured pointers depend on it (see spike doc "Pointer stability").

Acceptance: parity test graph-vs-eager bdim_forcing over a moving-pose
multi-step sim (mirror `tests/test_forces.py::test_forces_2d_graph_replay_eq_eager`
structure); suite green; profile rows `bdim_forcing` ≤0.05, `mw_body_div_corr`
≈0, fluid "other" visibly down.

## Task C — finish + commit the BC/flux graph runners (expected −0.2 ms/step)

**DONE 2026-07-06.** The runners already existed — `ApplyBcs{2,3}DGraphRunner`
(advection.py:1751,:1935) and `FluxAddGraphRunner` (a `_WarpGraphRunner` factory,
advection.py:1286) — but were `[SPIKE]`-marked with nothing proving they replay
under a real run. Finished/hardened them (no new kernels):

1. **Replay counters** — each runner now exposes `replays`/`captures`/`eager`
   counts (`_WarpGraphRunner` added them to `__slots__`). Empirically confirmed
   the flux runner is *not* silently eager in a steady solve: 6 steps → 27
   replays / 18 captures (the caching allocator double-buffers `rhs`, so each of
   the 9 (component,direction) signatures captures twice, then replays). The
   profiler already tallies aggregate replays by patching `wp.capture_launch`,
   so the per-runner counters are for tests/inspection only.
2. **Recapture keys** — audited: dtype (`str(u.dtype)`) and the flux path's
   `dt_dh`/`C_courant`/`scheme_id`/`face_dim` were already keyed. **Hardened**
   the ApplyBcs keys with the op counts `N_neu/N_dir/N_ref` — they are baked into
   the captured graph's launch dims + kernel scalars, so a pool-reused descriptor
   pointer with a changed op count would otherwise replay a stale graph.
3. **Tests** in `tests/test_advection.py`:
   `test_apply_bcs_{2,3}d_graph_replay_eq_eager` (f32/f64, stable-buffer
   multi-step graph-vs-eager, bit-exact, asserts captures==1/eager==1/replays==K-1),
   `test_apply_bcs_2d_graph_recaptures_on_new_dtype`, and
   `test_flux_graph_runner_replays_after_warmup` (asserts captures>0 and
   replays≥captures in a real `AdvDiffSolver.solve` loop).
4. Dropped the `[SPIKE]` markers (advection.py:345 + the two runner docstrings).

Suite green: 275 passed / 1 skipped (full `tests/`). The BC `set_BCs` row was
not re-profiled in isolation — the runners were already wired into `set_BCs`;
this task made them provably-replaying and safe, not faster in a new way.

If Task A lands first, the flux runner matters only for the explicit-convection
examples — still finished; other examples use those schemes.

## Task D — Poisson host share + initial guess (expected −0.3…−0.5 ms/step)

**DONE 2026-07-07** (sub-item 1 landed; sub-item 2 investigated + CLOSED as
inapplicable to the reference case). Same-tree before/after over the phase
window (steps 250–329, the stable signal — the throughput window is too noisy
for a 0.05 ms change): `project(poisson+correct)` 1.2165 → 1.1652 ms sync,
1.1628 → 1.1449 ms submit; −3 `aten::copy_` and −1 DtoD per step. Suite green
(275 passed / 1 skipped). Changes are provably bit-identical (no drag-trace
validation needed — see below), so no physics gate was run.

`project` ≈ 1.17 ms/step: ~0.9 GPU (one heavy v-cycle inside the captured
WarpMG2D graph) + ~0.25 host around it. The −0.3…−0.5 target assumed the solve
burned multiple v-cycles with host churn around each; the profiling refutes
that — see sub-item 2.

1. **Host-wrapper trim (DONE).** Two allocation/dispatch cuts, both in the
   multigrid path (`poisson_mult.solve_multigrid` + `solver.project`):
   * `WarpMG{2,3}D.residual_inf()` — the adaptive early-exit residual now runs
     on the multigrid's OWN persistent level-0 buffers (pw / fw / the
     in-graph-extracted face pairs cp0..cm2 / rw), replacing the old per-check
     closure that re-sliced the y/z-face coefficients (real non-contiguous
     copies) and allocated a fresh residual field + 6 `wp.from_torch` wraps
     every convergence check. Bit-identical to the old `mg_residual_*_warp`
     wrapper (verified: f64 field diff 1.7e-17, f32 2.3e-8 — pure gauge-shift
     roundoff, the residual is computed pre-BC/pre-mean exactly as before).
     `solve_multigrid` returns the persistent residual FIELD (callers do
     `r.abs().max().item()` or GFM `r[~mask]`, so the field shape is kept).
   * `project` passes `p0=None` to `solve_multigrid` when not warm-starting;
     `WarpMG.solve` zeros its level-0 buffer in place, skipping the per-step
     full-grid `torch.zeros_like` alloc + copy-into-graph. MGCG/RMGCG still get
     a real tensor (`x = p0.clone()`), so only the multigrid method opts out.
     Bit-identical (zeros in, zeros in).
   * NOT done, by design: `p -= p.mean()` and `self.BC(p)` were left alone.
     The gauge subtraction is gradient-invariant (cancels in the velocity
     correction) and only matters for warm-start/plotting; removing it changes
     the returned p by a constant (plot/warm-start regression) for ~50 µs GPU.
     BC(p) fusion (4 slice-copies → 1 Warp kernel) saves ~15 µs host but risks
     the ghost-ring gauge landmine (`self.BC(p)` must run BEFORE `p -= p.mean()`
     — fixed once already); not worth it outside a Task-E whole-step capture.
2. **Extrapolated initial guess — CLOSED, inapplicable.** The premise (cut the
   typical v-cycle count 2–3 → 1–2) does not hold for the reference salamander:
   with adaptive early-exit + heavy smoothing (nu1=nu2=nsmoothing=10) the solve
   **already converges in ONE v-cycle/step** and early-exits at tol. Measured
   directly: enabling plain `poisson_warm_start` changed nothing —
   `rbgs_halfsweep` 2352 → 2352 launches, `mg_residual_2d` 56 → 56, project
   1.1656 → 1.1695 ms sync (noise). The ~336 rbgs halfsweeps/step is a single
   heavy multi-level v-cycle, not several cheap ones. So `p_guess = 2·p_n −
   p_{n−1}` would add a persistent history buffer + torch ops for ZERO cycle
   saving on this case, and plain reuse was already REJECTED for the two-phase
   variable-coeff path. Recording and closing per the plan's own escape hatch.
   (If a future case is genuinely budget-limited — many cycles/step, not
   early-exiting — revisit: the WarpMG buffers are already pointer-stable, so
   an opt-in `poisson_warm_start: extrapolate` is a small add there.)

## Task E — whole-fluid-step capture (after A+B+C; expected → ~2.5 ms/step total)

End-game: once advection (A), bdim_forcing (B) and BCs (C) are pointer-stable
static launches, capture `_fluid_step_fused_2d` minus the MG solve as one
`wp.ScopedCapture` graph (or two: pre-Poisson and post-Poisson), leaving the
adaptive MG loop as the only per-step host decision (it already replays
per-v-cycle graphs). Staging pattern for per-step scalars/poses: async `copy_`
into persistent buffers, exactly like `ForcesPostGraph._stage`.

Hard rules learned on this branch (violations were production bugs):
* **Never** use `torch.cuda.make_graphed_callables` / torch-side capture
  around Warp launches — Warp kernels run on Warp's stream and are silently
  dropped from torch replays (frozen physics, no error). The solver refuses
  `use_cuda_graphs` for this reason. Warp-native `wp.ScopedCapture` only, or
  route via `wp.ScopedStream(wp.stream_from_torch())` if torch capture is
  ever required.
* Anything allocated fresh per step inside the captured region breaks replay
  correctness silently — audit with the pointer-churn method from the spike
  (log `data_ptr()` signatures over 20 steps before capturing).
* GL viewer interop maps on a non-blocking stream (err 906/901 fix) — test
  E with the FlowViewer2D enabled as well as headless.

Acceptance: graph-vs-eager parity over ≥600 coupled steps ≤1e-9 drag;
`throughput_ms_per_step_untouched` ≤ 3.0; replay counters grow ~1/step;
suite green; a live viewer run stays visually sane.

## Ordering / parallelism

```
Task 0 (validate+commit tree)          — first, blocking
  ├─ Task A (fused SL kernel)          — parallel ok
  ├─ Task B (bdim_forcing/copies)      — parallel ok
  ├─ Task C (BC/flux runners)          — parallel ok (same file as A: advection.py — coordinate or serialize A→C)
  └─ Task D (Poisson)                  — parallel ok
        └─ Task E (whole-step graph)   — after A+B+C land
```

Expected cumulative: 5.35 → ~3.2 (A) → ~2.7 (B+C) → ~2.4–2.5 (D, E) ms/step.

## Out of scope (measured, not worth it)

Forces readout / streaming SDF / apply_forces (0.47 combined, already
graphed); FARMS/MuJoCo side (0.4); GPU kernel micro-opts (striped atomics,
tile pre-reduction — see spike doc); MuJoCo viewer (proven innocent).

## Global rules for all agents

* Branch from the Task-0 commit on `warp_port`; one PR per task.
* No Co-Authored-By trailer in commits.
* Two-phase code stays out of core files (`forces.py`/`solver.py`/`body.py`)
  — if a change would leak two-phase logic into core, stop and redesign.
* Sim plots/frames for validation go to `/data/andreaferrario/ns_data/`.
* When benchmarking across branches/subprocesses, verify
  `lilytorch.__file__` points at the tree under test (the harness prints it)
  — editable installs have shadowed benchmarks before.
* "Python duplicates" are often load-bearing (CPU path, test oracle) — trace
  reachability before deleting anything.
