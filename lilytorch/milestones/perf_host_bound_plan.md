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

`ApplyBcs2DGraphRunner` / `FluxAddGraphRunner` (advection.py) are marked
[SPIKE] and `set_BCs` still costs 0.25 ms/step. Finish: verify the runners
actually replay in the coupled run (add replay counters, assert >0 after
warm-up in a test), harden the signature keys (dtype/dt changes must
recapture — the landmines are documented in the FluxAddGraphRunner docstring),
and cover with tests in `tests/test_advection.py`. If Task A lands first, the
flux runner matters only for the explicit-convection examples — still finish
it; other examples use those schemes.

## Task D — Poisson host share + initial guess (expected −0.3…−0.5 ms/step)

`project` = 1.28 ms/step: ~0.9 GPU (already inside the captured v-cycle
graphs, `multigrid_graph.py` WarpMG2D), ~0.4 host around it.

1. Profile-guided trim of the host wrapper: RHS assembly (divergence),
   `p -= p.mean()`, BCs on p, and the per-cycle residual-check bool sync
   (2–3/step, unavoidable for adaptive exit but the surrounding ops can move
   onto the graph or into a fused Warp RHS kernel).
   **Landmine:** `self.BC(p)` must run BEFORE `p -= p.mean()` (ghost-ring
   gauge bug, fixed once already — don't reintroduce).
2. Experiment (cheap, high leverage): linear pressure extrapolation initial
   guess `p_guess = 2·p_n − p_{n−1}` (persistent history buffer). If it drops
   the typical v-cycle count 2–3 → 1–2, that's ~0.3 ms GPU saved per step.
   Must be validated on physics: 3000-step quiet-water control + drag trace
   vs baseline; make it opt-in config (`poisson_warm_start: extrapolate`)
   if results shift beyond 1e-9. Note: plain warm-start reuse was REJECTED
   for the two-phase variable-coeff case previously — this is the
   *extrapolated* variant on the single-phase path; if it also fails, record
   why and close the item.

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
