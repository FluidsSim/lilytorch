# Remote-agent task: implement LT7 — Granular flow (sand) via μ(I)-rheology

You are a remote agent implementing **dense granular flow** in LilyTorch as an
incompressible fluid with a pressure-dependent, shear-rate-dependent effective
viscosity (the **μ(I) rheology**). This is a self-contained task; the human will
review your branch later.

**Read first, in this order:**
1. `milestones/granular_design.md` — the full design (this task implements it).
2. `lilytorch/src/two_phase_solver.py` — the `TwoPhaseSolver(FluidSolver)` you subclass.
3. `lilytorch/src/operations.py:295` (`carreau_viscosity`) and
   `lilytorch/src/operations.py:223` (`strain_rate_magnitude`) — the ops you clone/reuse.
4. `lilytorch/src/extras.py:193` (`_compute_nu_t`) — the dispatch hook you override.
5. Reference papers named in `granular_design.md` §2 (Lagrée–Staron–Popinet 2011 is
   the template; Jop et al. 2006 is the law; Barker & Gray 2017 is the stabiliser).

---

## 0. Hard constraints (read carefully — these override the design doc's suggestions)

The design doc proposes editing `operations.py` and `extras.py`. **Do NOT do that.**
The human requires that you **do not modify any existing file in the codebase.**
Achieve everything via *new files* and the *subclass-and-override* pattern instead.

**Off-limits (must show zero `git diff`):** `solver.py`, `extras.py`, `operations.py`,
`two_phase.py`, `two_phase_solver.py`, `body.py`, `forces.py`, `advection.py`,
`diffusion.py`, `base_sim_config.py`, anything under `lilytorch/integration/`,
anything under `lilytorch/FARMS_V2/`, and any other pre-existing source file.

- The granular viscosity closure goes in a **new** module, not appended to `operations.py`.
- The dispatch into `_compute_nu_t` is done by **overriding** `_compute_nu_t` in your
  `GranularSolver` subclass — not by editing `extras.py`. (It's an instance method on
  `FluidSolver`; a subclass override fully replaces it. Call `super()._compute_nu_t`
  for the off path so two-phase/single-phase behaviour is preserved when granular is off.)
- You may **read and call** existing functions/ops freely (e.g.
  `ops.strain_rate_magnitude`, the VOF advection, the projection, BDIM) — just don't edit them.
- **No FARMS / no MuJoCo.** Do not import `farms_*`. Validate everything standalone
  (pure fluid + prescribed/analytic bodies). But keep the inherited BDIM/forces path
  **untouched and intact** so the human can wire `BDIMhandler` → MuJoCo later with zero
  changes to your code. Expose a clean constructor API so a later config plumb is trivial.
- Config plumbing into `BaseSimConfig` is **out of scope** (it would touch core). For
  validation, construct `GranularSolver` directly in your scripts and pass the `granular`
  dict via constructor kwargs.
- Before any commit, run `git diff --stat` against the off-limits list and confirm it is
  empty. This mirrors the existing project rule in
  `feedback-no-core-source-for-two-phase`.

## Branch & workflow

- Branch **from `optimize_speed_memory`**. Name it `granular-mu-i` (or similar).
  ```
  git fetch && git switch optimize_speed_memory && git switch -c granular-mu-i
  ```
- Commit per milestone (G0…G4 below) with clear messages. End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **Do not merge, do not open a PR, do not push to `main`.** Leave the branch for review.
- Keep everything gated on a `use_granular` flag so that with granular **off**, the
  two-phase and single-phase paths are byte-for-byte unchanged.

---

## 1. What to build (new files only)

```
lilytorch/src/granular.py            # NEW: granular_viscosity op(s)
lilytorch/src/granular_solver.py     # NEW: GranularSolver(TwoPhaseSolver)
lilytorch/src/test_granular.py       # NEW: op + solver unit tests (fp64 for op checks)
validation/granular_2d/              # NEW: column collapse, angle of repose, hopper
bench/bench_granular.py              # NEW: speed + peak-memory benchmark
milestones/granular_results.md       # NEW: your final report (numbers + plots)
```

### 1a. `granular.py` — the viscosity op
Implement `granular_viscosity(vel, p, h, ndim, *, mu_s, mu_2, I0, d, rho_s, rho,
gamma_min=1e-6, p_min=0.0, nu_max=None, regularization="barker2017")`, a near-clone of
`ops.carreau_viscosity` but with a **pressure-dependent** yield term `μ_s·p` instead of a
constant `τ_y`. See `granular_design.md` §4 for the reference implementation. Reuse
`ops.strain_rate_magnitude` (do not reimplement it). Implement **both** the raw μ(I) and
the **Barker & Gray (2017) regularised** closure, selectable by `regularization`
(`"none"` | `"barker2017"`). The `nu_max` CFL clamp is load-bearing (jammed cells have
γ̇→0 ⇒ ν→∞); cap with the same diffusion-CFL bound the Carreau path uses.

### 1b. `granular_solver.py` — `GranularSolver(TwoPhaseSolver)`
- Add a `granular` config dict + `use_granular` flag (schema in `granular_design.md` §5).
- Override `_compute_nu_t(self, *vel)`: when `use_granular`, return
  `granular_viscosity(vel, self.p0, ...)` using the **lagged** pressure `self.p0` (the
  one real subtlety — see §3.1 of the design). Otherwise delegate to `super()`.
- Also override `_compute_nu_rho_for_forces` so the force path uses the same ν_eff
  (granular reaction = granular-pressure + frictional stress). Keep it consistent with
  the `_compute_nu_t` override; do not duplicate the strain-rate computation if you can
  cache it.
- Set `self.nu` to a small floor (`nu_floor`) so `ν_total = self.nu + ν_eff` is bounded
  below, exactly as the Carreau path does.
- Reuse TwoPhase VOF for the **granular/air free surface** (the pile surface): dense phase
  density `ρ_grain_bulk = phi·rho_s` feeds the existing `rho_water` slot; air is the light
  phase. No new state.
- Optional `granular_pressure_subiter` (default 1) re-evaluating ν_eff with the new p0 —
  reuse the existing `consistent_n_cycles` fixed-point loop pattern; do not invent new
  plumbing.

---

## 2. Speed & memory (a first-class requirement, not an afterthought)

The branch is `optimize_speed_memory`; the human wants this implementation **fast and
memory-lean** from the start:

- **No avoidable full-grid temporaries.** Mirror the in-place / persistent-buffer style
  already used in the solver and in `ops.carreau_viscosity`. Reuse pre-allocated buffers
  across steps; fuse element-wise chains; prefer `torch.clamp_`/in-place where safe.
- **fp32 production path, fp64 only for op-level correctness tests.** The op must run in
  whatever dtype the field is in (don't hard-cast to fp64).
- Keep the op **`torch.compile`-friendly** (no data-dependent Python control flow on
  tensor values; use tensor ops for the clamps/branches). The two-phase path is
  compile-capable; don't regress that.
- Write `bench/bench_granular.py` modelled on the existing `bench_memory.py` /
  `bench_python_overhead.py`. It must report, at **128², 256², and 128³**:
  - wall-time per step (warm, median of N steps), and
  - **peak CUDA memory** via `torch.cuda.reset_peak_memory_stats()` +
    `torch.cuda.max_memory_allocated()`,
  for **granular-on vs the inherited two-phase baseline (granular-off)**. Assert/report
  that the granular overhead in both time and peak memory is small (target: viscosity
  closure adds no more than a couple of full-grid buffers and a modest % wall-time).
  Put the measured numbers in `granular_results.md`.

---

## 3. Validation ladder (run these; they are the acceptance gate)

Mirror the two-phase validation style (`lilytorch/src/test_two_phase.py`,
`validation/two_phase_2d/`). From `granular_design.md` §8:

1. **Op unit test (fp64).** Uniform shear with known `p`, `γ̇`: assert `ν_eff` equals the
   analytic `μ(I)·p/(ρ·γ̇)`; assert the `nu_max` and `p_min` clamps activate. Also test the
   Barker & Gray regularised branch against its analytic form.
2. **Pure shear / Bagnold profile.** Steady simple shear under gravity-loaded `p`; compare
   the velocity profile to the μ(I) Bagnold prediction.
3. **Angle of repose.** Release a heap; the static surface angle must settle to
   `atan(μ_s)` within a few degrees.
4. **Granular column collapse (2-D) — the headline acceptance test.** Runout length and
   final deposit height vs. initial aspect ratio `a = H/L`. Target: within ~10–15 % of the
   experimental correlation (Lube / Lajeunesse / Lagrée–Staron–Popinet) across
   `a ∈ [0.5, 10]`. **Do not declare success before this passes.**
5. **Hopper / silo discharge (optional, qualitative).** Beverloo scaling of mass flow vs.
   aperture width.

**Regularisation is mandatory before trusting any result.** μ(I) is mathematically
ill-posed (Barker et al. 2015): expect grid-dependent blow-ups that look like bugs but are
intrinsic. Implement the Barker & Gray (2017) closure (G2) before interpreting column-
collapse instability. Reuse the existing `check_explosion` (already called in
`TwoPhaseSolver.finalize_step`) and extend the `LILYTORCH_UMAX_PROBE` diagnostic in your
subclass to also print local `I`, `γ̇`, `p` so ill-posedness can be told apart from a bug.

---

## 4. Milestones (commit at each boundary)

- **G0 — Viscosity op + unit tests.** `granular.py` + tests 1–2. Pure PyTorch, no solver
  wiring. Fully testable in isolation.
- **G1 — `GranularSolver` skeleton.** Subclass, `granular` schema, `use_granular`,
  `_compute_nu_t` / `_compute_nu_rho_for_forces` overrides reading lagged `self.p0`, VOF
  for the granular/air surface. Validate angle of repose (test 3).
- **G2 — Regularisation + diagnostics.** Barker & Gray (2017); extend `UMAX_PROBE`.
- **G3 — Column collapse validation.** The acceptance test (test 4). Tune
  `nu_max`/`γ̇_min`/`p_min`; document the stable CFL envelope.
- **G4 — Benchmark + report.** `bench/bench_granular.py` numbers; write
  `milestones/granular_results.md`. (A prescribed rigid body ploughed through the bed is a
  nice-to-have to exercise the inherited force path **without MuJoCo** — optional.)

G0–G3 are the bulk of the work. FARMS/MuJoCo coupling (design §7, G5 immersed granular)
is explicitly **deferred to the human** — just make sure nothing you do blocks it.

---

## 5. Definition of done

- New branch `granular-mu-i` off `optimize_speed_memory`, committed per milestone, not merged.
- `git diff` against every off-limits file is **empty**.
- `test_granular.py` passes (op tests in fp64).
- Column collapse (test 4) within ~10–15 % of the experimental correlation, regularised.
- `bench/bench_granular.py` runs and reports wall-time + peak memory, granular-on vs
  baseline, at 128²/256²/128³, with small measured overhead.
- `milestones/granular_results.md` documents: what was built, validation numbers + plots,
  benchmark numbers, the stable CFL/parameter envelope, and any open issues (esp. the
  shared triple-point singularity noted in design §6).
- With `use_granular=False`, two-phase/single-phase behaviour is unchanged.
