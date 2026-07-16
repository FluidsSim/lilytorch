# Handoff: make the eulerian and lagrangian force readouts agree — and match published references

**Branch:** `cuda_native_port`  ·  **Written:** 2026-07-16  ·  **Status:** SUPERSEDED by §9 (session 2, same day):
Phases 0-2 done. The §1 "cubic over-read law" is FALSIFIED in real flows — read §9 before acting on §§1-6.

## The goal

Both force readouts must be **usable** and must **agree with each other**, and both must reproduce the
repo's published-reference benchmarks. This is not "pick a winner":

- `lagrangian` is **noisier and less stable** in coupled swimmer runs (the reason eulerian is wanted).
- `eulerian` is smooth but is **currently biased** (see below), which is why the zebrafish swims too
  slowly under it.

The task that triggered this: `examples/zebrafish_ki_project/gen_configs_pd_3d_slow_fast.py` swims
markedly slower with `force_method: "eulerian"` than with `"lagrangian"`. That discrepancy is now
explained. Fixing it is the job.

---

## 1. What is established (with evidence)

**The eulerian VISCOUS readout over-reads by `((R+ε)/R)³`, where R is the local body radius.
Pressure is fine. The lagrangian formula is exact.**

### Root cause

[`streaming_sdf.cu:404`](../src/csrc/cuda/streaming_sdf.cu#L404) — and identically in the CPU twin
[`streaming_sdf_cpu.cpp:697`](../src/csrc/streaming_sdf_cpu.cpp#L697) and the python path
[`forces.py:143`](../src/forces.py#L143):

```c
const scalar_t d_visc = s_cc_body - eps_solver;   // viscous band shifted OUT by one eps
if (d_visc > -eps_body && d_visc < eps_body)  delta_visc = ...;
if (s_cc_body > -eps_body && s_cc_body < eps_body)  delta_pres = ...;   // pressure NOT shifted
```

The viscous band is centred at `φ = +ε`, so it integrates `σ·n` over the **offset iso-surface
`{φ=ε}`**, not the body surface — and **no correction is applied for that**. By the divergence
theorem `∮_{φ=ε} σ·n dS = ∫_{V(R+ε)} ∇·σ dV`, so the force is scaled by the **enclosed-volume**
ratio (cubic, not the area's quadratic).

The shift is not gratuitous: it exists to escape the BDIM band and read uncontaminated fluid strain
(see the rationale at [`lagrangian_forces.cu:202`](../src/csrc/cuda/lagrangian_forces.cu#L202)).
The bug is that the surface it then integrates over is never accounted for.

### Evidence — physical oracle

Two analytic cases on a sphere with exact closed-form answers (divergence theorem):

- **A (pressure):** `p = -G·x`, `u = 0` → `F_p = G·V`
- **B (viscous):** `u_x = c·y²`, `p = 0` (divergence-free) → `F_v = 2·ν·ρ·c·V`

The sphere SDF is an exact distance function (`|∇φ| = 1`), which excludes `force_delta_order` from
the picture entirely. Driving the **native** op (`streaming_sdf_forces_post_3d`), CPU/float64:

| R/h | eps_m | ndelta F_p | deltaH F_p | lagr F_p | **ndelta F_v** | **deltaH F_v** | **lagr F_v** |
|---|---|---|---|---|---|---|---|
| 6.2 | 1.0 | 1.01 | 1.01 | 1.00 | **1.58** | **1.58** | **1.00** |
| 6.2 | 2.0 | 1.04 | 1.04 | 1.00 | **2.37** | **2.37** | **1.00** |
| 9.4 | 1.0 | 1.00 | 1.00 | 1.00 | 1.36 | 1.36 | 1.00 |
| 12.6 | 1.0 | 1.00 | 1.00 | 1.00 | 1.26 | 1.26 | 1.00 |
| 19.0 | 1.0 | 1.00 | 1.00 | 1.00 | 1.17 | 1.17 | 1.00 |
| 19.0 | 2.0 | 1.00 | 1.00 | 1.00 | 1.35 | 1.35 | 1.00 |

(ratios to exact). The measured inflation matches `((R+ε)/R)³` within 1-2% across the whole sweep.
`deltaH` and `ndelta` viscous outputs are **bit-identical** (`1.6886733232106366`) — `deltaH` only
replaces the pressure readout, so it is **not** a candidate fix for this.

The error is only **first order in h** at fixed `eps_multiplier`: you would need `R/h ≈ 100` in the
*thin* direction to get eulerian viscous within a few percent. Refinement is not a workaround.

### Reproduction

Two scripts, added with this handoff:

- `validation/force_readout_oracle/oracle_python_path.py` — python eulerian vs lagrangian via a real
  `FluidSolver`.
- `validation/force_readout_oracle/oracle_native_three_way.py` — ndelta vs deltaH vs lagrangian via
  the native op with a hand-built single-sphere scene (this is the path production runs).

Both print ratio-to-exact tables. Run them first to confirm the environment reproduces the above.

---

## 2. What is NOT established

Be careful here — these are open, not assumed.

1. **The lagrangian in-situ bias.** The oracle imposes analytic fields, so there is no BDIM band to
   contaminate the sample. It validates the lagrangian *formula*, and says **nothing** about the
   live-run readout, which currently samples at `lagrangian_sample_offset = 0.0` — i.e. exactly at
   the surface, inside the band, against unconstrained interior pressure. Sign and magnitude unknown.
2. **Why lagrangian is noisier / less stable** in coupled runs. Candidates in the appendix; none
   measured.
3. **Whether fixing the eulerian viscous alone makes the two agree** in a live coupled run. Probably
   necessary, not obviously sufficient.

---

## 3. Falsifiable predictions — do these first

The law predicts a specific over-read for each benchmark at its own resolution. These are cheap to
confirm and will either validate or kill the whole analysis in a few hours.

| Benchmark | R/h | eps_m | **predicted eulerian viscous over-read** | pressure |
|---|---|---|---|---|
| `validation/cylinder_drag_2d` (Re=550) | 25.6 | 2.0 (default) | **1.25×** | ~1.00 |
| `examples/single_sphere_drop_gazzola` | 10.7 | 2.0 (from yaml) | **1.67×** | ~1.00 |
| zebrafish thin dimension | ~2-3 | 1.0 | **2.4-3.4×** | ~1.00 |

If the decomposed viscous drag in `cylinder_drag_2d` does **not** come out ~25% high under eulerian
relative to lagrangian, the analysis is wrong — stop and re-derive.

---

## 4. The benchmarks

### (a) `validation/cylinder_drag_2d/` — **start here**

Koumoutsakos & Leonard (1995), impulsively started flow past a cylinder at Re=550. Reference data:
`data_to_save/koumoutsatokos_keonard_1995.csv`.

Why it is the right first target:
- `run_cylinder_drag.py` **already has a `FORCE_METHOD = "eulerian" | "lagrangian"` toggle** at the
  top, plus `BDIM_MU0_PROJECTION` and `ZERO_PRESSURE_INSIDE`.
- `plot_cylinder_drag.py` produces a **decomposed viscous/pressure plot**
  (`cylinder_drag_decomposed_Re550`) — which isolates exactly the channel under suspicion.
- Impulsively started flow is viscous-dominated at early time, so the discrepancy should be loud
  in the first convective time.
- 2D, 512², cheap.

Deliverable: overlay eulerian vs lagrangian vs reference, decomposed. Confirm the 1.25× viscous gap.

### (b) `examples/single_sphere_drop_gazzola/` — the integral test

Settling cylinder vs Namkoong et al. (2008); reference terminal velocity `U_t = -0.025 m/s`
(`plot_data.py:61`), `rho_body = 1005.96`, `rho_fluid = 996.0`, `D = 0.005`, `nu = 8e-7`,
`eps_multiplier = 2.0`, `poisson_method: fft`, currently `force_method: "method2"` (= eulerian).

Why it matters: **at terminal velocity, drag balances net weight exactly.** So `U_t` is a direct,
integral test of the readout with a published number — no decomposition needed. Terminal
`Re = U·D/ν ≈ 156`, where viscous is a large share of Cd. An over-read of viscous drag → terminal
velocity too low. *This is the same symptom as the zebrafish*, on a case with a reference answer.

Note `single_sphere_drop_gazzola_low_density` is a second density ratio — a useful second point.

### (c) `examples/single_sphere_drop_coquerelle_3d/` — 3D confirmation

Several `experiment_config_d0p{2,3}_nu0p{02,05,1}.yaml` configs — a d×ν sweep, which is ideal
because the law predicts the over-read varies with `R/h` across them. Known trap: see
`project_coquerelle_regression_f650945` — set `bdim_mu0_projection: False`.

### (d) Also available

`validation/error_analysis_cylinder_2d/` (`run_bdim_consistent.py`, `plot_error_analysis_MW.py`) —
convergence tooling that may already do half of the sweep work.

---

## 5. Suggested plan

- **Phase 0** — Run both oracle scripts. Confirm the table reproduces.
- **Phase 1** — `cylinder_drag_2d`, both methods, decomposed plot vs Koumoutsakos & Leonard. Confirm
  or kill the 1.25× prediction. *Gate: do not proceed until this is settled.*
- **Phase 2** — `single_sphere_drop_gazzola`, both methods, terminal velocity vs Namkoong. This also
  measures the lagrangian in-situ bias for the first time (open question #1), since it is a real
  coupled run with a real BDIM band and a published answer.
- **Phase 3** — Design the fix (§6) using what Phases 1-2 revealed. Re-validate on both.
- **Phase 4** — 3D: `single_sphere_drop_coquerelle_3d` sweep. Then re-run the zebrafish and check the
  two methods agree.
- **Phase 5** — Promote the oracle to a real test in `tests/test_forces.py`. See §7 — its absence is
  why this survived.

---

## 6. Fix design space — nothing here is validated, choose after Phases 1-2

- **Option A — un-shift the viscous band.** Then the strain is band-contaminated.
  **Do NOT try to recover it by dividing by μ0.** This was tested and fails: the blend gives
  `ε(u) = μ0·ε(u') + [d⊗∇μ0 + ∇μ0⊗d] + (μ1 term)` (with `ε(u_b) = 0` exactly for rigid bodies), and
  the dropped `∇μ0` term is `O(slip/ε)` — the same order as the term kept. Dividing by μ0 amplifies
  it by `1/μ0`, diverging as `μ0 → 0`. Measured: **2.9× error at μ0=0.55**, only correct at μ0=1
  where no correction is needed. Physically, the `∇μ0` term is *where the wall shear stress lives*
  in BDIM — it is not contamination to remove.
- **Option B — keep the shift, correct the surface measure** by the local Jacobian (`∇·n`).
  Partial only: the measured `((R+ε)/R)³` decomposes as area `(a/R)²` × real traction growth `(a/R)`.
  A Jacobian correction removes the first factor and leaves the second. Better, not exact.
- **Option C — extrapolate `σ` from the clean region (`φ > ε`) back to `φ = 0`.** Probably the
  principled option; costs a stencil.
- **Option D — fallback:** make lagrangian the single readout and fix its noise instead. The user
  wants both methods usable, so treat this as a last resort.

Whatever is chosen must keep CPU/CUDA/python parity (all three carry the convention today) and
should be validated on Phases 1-2 *before* the zebrafish.

---

## 7. Traps — read before touching anything

- **A parity test is only worth its oracle.** Every force test in `tests/test_forces.py` is
  CPU-vs-GPU parity or a frozen snapshot. **None checks physics.** The bug above and the one in §8
  are invisible to all of them. This is the same lesson as `project_warp_masked_broken_cpu_twins`.
- **`_build_surface_3d` leaves `tri_*_world` in a bbox-centred LOCAL frame.** Only
  `BDIMhandler._refresh_lagrangian_tris_3d` moves it to world. A static-body harness that skips the
  handler silently samples the wrong locations and produces a *plausible-looking* wrong answer.
  (This cost an iteration here — the tell was that two unrelated physics tests returned the
  identical ratio.) Supply an exact world-frame triangulation instead; validate it with
  `∮ x·n dS = 3V` before trusting any force it produces.
- **`forces_method2_3d` full-grid branch crashes**: reads `comp.sdf_vals` with no fallback
  ([`forces.py:800`](../src/forces.py#L800)). Workaround as in
  `validation/two_phase_3d/run_drop_sphere_3d.py:329`.
- **`deltaH` exists only in the native branches** — the python fallback in `forces_method2_3d` has no
  deltaH path. Testing it requires driving the native op.
- **`eps_multiplier` is not a free knob.** Raising it quiets lagrangian band noise but worsens the
  eulerian viscous over-read *cubically*. The two methods currently want opposite values.
- **Regime B is not the force path.** `streaming_sdf_regime_b.cu` registers
  `streaming_sdf_stag_{2,3}d_resolve` (the body-update/streaming stage). Forces are a separate op,
  `streaming_sdf_forces_post_3d`, in `streaming_sdf.cu`. Easy to conflate.

---

## 8. Side findings — fix separately, do not conflate

1. **`force_delta_order = 2` is inverted.** [`solver.py:348`](../src/solver.py#L348) says the delta is
   "divided by `|∇SDF|` so that the volume integral gives the correct surface measure"; the coarea
   formula needs a **multiply**. Measured on a sphere with `φ = g·(r−R)` (true area 1.13097):

   | \|∇φ\| | order 1 | order 2 as coded | multiply by g |
   |---|---|---|---|
   | 0.5 | 2.27479 | 4.54957 | 1.13739 |
   | 1.0 | 1.13264 | 1.13264 | 1.13264 |
   | 2.0 | 0.56536 | 0.28268 | 1.13073 |

   Order 2 doubles the error instead of removing it. Inert at `|∇φ| = 1`, which is why analytical-body
   tests never caught it. Affects mesh bodies (the zebrafish is one). Not currently active in the
   zebrafish config (`force_delta_order` defaults to 1).
2. **The `sdf_vals` crash** in §7.

---

## Appendix — the zebrafish config, for context

`examples/zebrafish_ki_project/gen_configs_pd_3d_slow_fast.py`. Original complaint: noisy force
readings under lagrangian. Unresolved candidates identified but **not measured**:

- `lagrangian_sample_offset` unset → `0.0` → samples on the surface, inside the BDIM band, against
  unconstrained interior pressure (`zero_pressure_inside = False`). The code comment at
  `solver.py:418` says to set it to ~eps. `h = 1.95e-4`, so ~`4.0e-4` is ~2h.
- `convexify = True` with **no 3-D contour mask**: per-link triangulations are convex hulls that
  overlap at joints, so buried triangles integrate interior pressure at full area weight. The 2-D
  fix (`_apply_contour_mask_2d`) has no 3-D counterpart. `deltaH` sidesteps this by construction
  (union surface + softmin partition) — that is deltaH's real value, not the viscous issue.
- `eps_multiplier = 1.0` (solver default is 2.0) — narrow band, sharper but noisier.
- Explicit coupling with a neutrally-buoyant 50 µg body over 16 links → added-mass ratio ~1;
  `force_relaxation` / implicit coupling are both already wired and commented out in the config.
- Fish geometry: 16 links, ~18 mm total, thin dimension ~2-3 cells.

`examples/zebrafish_ki_project/verify_energy_balance.py` (`--tmax 0.3`) checks
`dE_k/dt = P_act − dissipation` and is the natural arbiter for which readout is right in a live run.

---

# §9 — SESSION 2 RESULTS (2026-07-16, Phases 0-2 executed). READ THIS FIRST.

## 9.1 Phase 1 gate FAILED → §1's law is dead in real flows

`cylinder_drag_2d` Re=550, both methods, decomposed, vs Koumoutsakos & Leonard (all with
`ZERO_PRESSURE_INSIDE=0`, see 9.3):

| Nx | τ | visc_E | visc_L | vE/vL | pres_E/pres_L | tot_E | tot_L | Cd_ref |
|---|---|---|---|---|---|---|---|---|
| 512 | 3 | 0.086 | 0.097 | **0.89** | 0.98 | 1.199 | 1.237 | 1.288 |
| 512 | 7 | 0.070 | 0.078 | **0.89** | 0.97 | 1.006 | 1.040 | 1.020 |
| 1024 | 3 | 0.119 | 0.121 | **0.98** | 0.98 | 1.272 | 1.293 | 1.288 |
| 1024 | 7 | 0.096 | 0.098 | **0.98** | 0.99 | 1.029 | 1.045 | 1.020 |

Eulerian viscous is 11% **LOW** (not 25% high), and the gap closes ~linearly in h.
**Corrected model:** the viscous band DOES integrate over `{φ=ε}` (the §1 code fact stands), but the
error is `O(ε·∂σ/∂n)` with **flow-dependent sign** — the oracle's `u=cy²` field has stress *growing*
off the wall (→ cubic over-read); a real boundary layer has shear *decaying* off the wall
(→ under-read). There is no universal multiplicative law; §3's predicted factors are wrong.

Decisive experiment: `shift_sweep.py` (in the scratchpad this session; re-create from §9.5) evaluates
both readouts on ONE frozen τ=7 field while sweeping the band shift / sample offset:
eulerian viscous = 0.049 (s=0, band eats BDIM-contaminated strain) → peak 0.074 (s=1.5h) → 0.053 (s=3h);
lagrangian = 0.078 (off=0) → 0.049 (off=2h). Neither limit is clean; both readouts are "sample σ at
distance ~s from the wall" devices. Python path at s=2h reproduced the live native kernel bit-for-bit.

## 9.2 Phase 2: gazzola settling — both methods reproduce Namkoong

After repairing the example (9.4): lagrangian **U_t = −0.0248 m/s (99.3%** of the −0.025 reference,
from contours.h5); eulerian **U_t = −0.0259 m/s (103.6%**, from fields.h5 SDF centroid — the eulerian
run does not refresh `cnt_update`, contours.h5 is all zeros). Eulerian slightly under-reads drag,
consistent with 9.1. §3's predicted 1.67× viscous over-read (→ much slower settling): killed.
Runs: `/data/andreaferrario/ns_data/namkoong_sphere_drop_{eulerian,lagrangian}/`.

## 9.3 NEW BUG: `zero_pressure_inside=True` HALVES the eulerian pressure force

Measured pE ratio 0.513 on the cylinder. The pressure delta band spans both sides of the surface
(see the mu0-masking note at `forces.py:107`); wiping interior p removes ~half the integral.
The user confirmed: do not zero pressure inside with the eulerian readout.
**Audit needed** — these set `zero_pressure_inside=True` (fine for pure-lagrangian-offset runs, wrong
if combined with eulerian forces): `salamander/gen_configs_underwater_walking_3d.py`,
`salamander/gen_configs_swim_2d.py`, `salamander/gen_configs_paddle_2d.py`,
`zebrafishsim/gen_configs_pd_3d.py`, `_1guillasim/gen_configs_pd_vary_f.py`.
`zebrafish_ki_project` already has `False` (so this is NOT the zebrafish cause).
`run_cylinder_drag.py`'s default was flipped to False this session.

## 9.4 Repairs made this session (uncommitted, working tree)

- `validation/cylinder_drag_2d/run_cylinder_drag.py`: yaml path `src/configs/` →
  `examples/standalone/configs/` (file had moved); `FORCE_METHOD` / `BDIM_MU0_PROJECTION` /
  `ZERO_PRESSURE_INSIDE` / `NX` now env-overridable; zpi default False.
- `examples/single_sphere_drop_gazzola/simulation_config.yaml`: `Nx: 512 → 768` (domain was doubled
  in 72cbbec without keeping dx=dy; solver asserted).
- `examples/single_sphere_drop_gazzola/sphere.yaml`: `density: 1 → 1005.96`. BDIMhandler buoyancy
  uses V=mass/animat-link-density; density=1 gave 192.99 N buoyancy vs 0.194 N weight → sphere left
  the domain in 5 iterations.
- `src/plotting.py`: "body exited domain" warning now prints com_pos + domain bounds.

## 9.5 What remains (plan for the next agent)

1. **Zebrafish is still unexplained.** The O(ε·∂σ/∂n) error is O(1) at thin-dimension R/h≈2-3, but
   its in-situ sign/size is unmeasured. Run `gen_configs_pd_3d_slow_fast.py` twice
   (eulerian vs lagrangian), same seed/gait; log per-link decomposed forces + swim speed; arbiter =
   `verify_energy_balance.py --tmax 0.3`. Cheap variant first: save a mid-run 3D field snapshot +
   poses, then port shift_sweep to 3D (drive `streaming_sdf_forces_post_3d` at several `eps_solver`
   values on the frozen snapshot) — if the viscous readout swings O(1) with s, that IS the zebrafish story.
2. **Fix design** (was §6): the error is a sampling offset, so Option C is the principled one —
   sample/extrapolate σ back to φ=0 from the clean side, e.g. evaluate the band integral at s=ε and
   s=2ε and Richardson-extrapolate to s=0 (two deltas, one extra pass, no new stencil); reject
   Option A (μ0-division diverges, §6 measurement stands). Gate any fix on: cylinder 512²/1024²
   decomposed vs K&L, gazzola U_t, coquerelle 3D sweep, THEN zebrafish agreement.
3. **Phase 5 physics test**: promote `validation/force_readout_oracle/oracle_native_three_way.py`
   into `tests/test_forces.py` (analytic sphere fields; assert lagrangian ≈ exact and eulerian
   pressure ≈ exact; pin the current eulerian viscous offset behaviour with a comment so a future fix
   flips the assertion intentionally). Every existing force test is parity-only; this closes that hole.
4. Commit the session's repairs (9.4) + shift-sweep as a script under `validation/force_readout_oracle/`.
5. Re-check `single_sphere_drop_gazzola_low_density` and `coquerelle_3d` after any fix
   (`bdim_mu0_projection: False` for coquerelle, see memory).

Open questions #2 (lagrangian noise in coupled runs) and the §8 side findings
(`force_delta_order=2` inversion, `sdf_vals` crash) are untouched.
