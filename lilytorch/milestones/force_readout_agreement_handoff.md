# Handoff: make the eulerian and lagrangian force readouts agree — and match published references

**Branch:** `cuda_native_port`  ·  **Written:** 2026-07-16  ·  **Status:** SUPERSEDED by §9 (session 2)
and §10 (session 3, same day). **§10 explains the zebrafish and reconciles §1 with §9 — read it first.**
Phases 0-2 done; the §1 "cubic over-read law" is not a universal law (§9.1) but IS the dominant term
when `eps/R ~ 1`, which is exactly the zebrafish (§10).

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

---

# §10 — SESSION 3 RESULTS (2026-07-16). The zebrafish is EXPLAINED.

§9.5 item 1 is closed, item 3 is done, item 4 is done. The cheap variant worked: no
zebrafish A/B swim race was needed.

## 10.1 The measurement

`gen_zfish_snapshot.py` runs the **production** `gen_configs_pd_3d_slow_fast` config headless
(same grid/gait/physics; viewers stripped) and dumps every argument the live solver hands the force
op at step 300. `shift_sweep_3d.py` then re-drives `streaming_sdf_forces_post_3d` on that frozen
field at any band shift, and — when the snapshot comes from a lagrangian run, the only path that
refreshes the world-frame triangulation (§7 trap) — the lagrangian readout too. **Both readouts, one
field, no trajectory divergence.** The whole loop is ~15 s.

Totals over all 16 links, x = swimming axis, frozen at step 300 (`snap_lagr.pt`):

| | `Fv_x` | `Fp_x` | **net `Fx`** |
|---|---|---|---|
| eulerian @ live `s = 2h` | −2.02e-05 | +2.62e-05 | **+6.00e-06** |
| lagrangian @ live `off = 0` | −5.11e-06 | +1.98e-05 | **+1.47e-05** |
| ratio | **3.96×** | 1.33× | **0.41×** |

**The eulerian readout reports 4× the viscous drag the lagrangian does, leaving 2.4× less net
forward force. That is the "swims markedly slower under eulerian" complaint, quantitatively.**
Pressure (`Fp_x`) is constant across the whole shift sweep, as it must be — only the viscous band is
shifted — which is a clean internal check that the sweep moves what it claims to.

## 10.2 Root cause: the band is WIDER THAN THE FISH

Measured from the snapshot SDF:

- max inscribed radius of the whole animal = **2.86 h** (`min(sdf_cc)`)
- `eps_body` (delta half-width) = **2.0 h**, `eps_solver` (band shift) = **2.0 h**
- **95.1%** of interior cells lie within `eps_body` of a surface

So for this body the smoothed delta is not a surface-localised measure at all: the band launched
from one side reaches through the centreline and out the other. There is no `eps/R << 1` regime to
be asymptotic in — here `eps/R ≈ 0.7`.

The §1 mechanism then dominates. By the divergence theorem the shifted band integrates `∇·σ` over
the volume enclosed by `{φ = eps_solver}`, and that inflation is measurable directly on the real
geometry:

| s/h | `V(φ<s)/V(φ<0)` |
|---|---|
| 1.0 | 1.587 |
| 2.0 | **2.257** |
| 4.0 | 4.031 |

Measured eulerian over-read from `s=0` to `s=2h`: **2.554×** (2.640× on the eulerian-trajectory
snapshot). Predicted by pure volume inflation: **2.257×**. Within ~13% — **the geometric inflation
is the leading term for the zebrafish**, the residual being the flow-dependent `∂σ/∂n` variation of §9.1.

**This reconciles §1 and §9.1.** Both are the same object — the readout is a volume integral over an
inflated region, so its error ≈ (volume inflation) × (how `∇·σ` varies across the shell):
- cylinder, `R/h=25.6`, `eps/R≈0.08`: inflation only 1.25×, and boundary-layer decay of `∂σ/∂n`
  overwhelms it → net **0.89×** (reads LOW). §9.1 stands.
- zebrafish, `R/h≈2.9`, `eps/R≈0.7`: inflation 2.26× dwarfs the decay term → net **~2.5×** (reads
  HIGH). §3's predicted 2.4-3.4× for the zebrafish thin dimension was right, for the right reason —
  it just is not a law that survives to fat bodies.

## 10.3 The two readouts are NOT the same device — matching the offsets does not make them agree

**§9.1's "both readouts are just sample-σ-at-distance-s devices" framing is WRONG, and this kills the
natural idea of reconciling them by setting `lagrangian_sample_offset = eps_solver`.** Tested on the
sphere oracle (R/h=12.6, exact geometry, no joints, `u_x = c·y²`), sweeping BOTH knobs together:

| s/h | eulerian /exact | lagrangian /exact | eul/lag | `((R+s)/R)³` |
|---|---|---|---|---|
| 0.0 | 1.004 | 0.999 | **1.005** | 1.000 |
| 1.0 | 1.260 | 1.079 | 1.168 | **1.257** |
| 2.0 | 1.559 | 1.158 | 1.346 | **1.556** |
| 3.0 | 1.900 | 1.237 | 1.535 | **1.898** |

They agree to **0.5% at s=0** and then diverge monotonically. The eulerian tracks `((R+s)/R)³` to three
decimals — §10.2's volume-inflation law, confirmed exactly on a clean geometry. The reason is
structural:

- **eulerian(s)** = `∮_{φ=s} σ·n dS` — integrates over the **offset iso-surface**, whose measure
  inflates with s.
- **lagrangian(off)** = `∮_{S_body} σ(x + off·n)·n dS` — integrates over the **true body surface**
  (fixed area, from the triangulation) and merely *samples* σ further out.

So raising the two knobs together makes them disagree **more**, not less. Matched at s=h they are
1.17× apart on the sphere and **1.86× apart on the fish** (−1.59e-05 vs −8.56e-06).

**The s=0 residual — now decomposed (§10.3c).** On the sphere the two agree at s=0 (1.005×); on the
fish they do **not** (`eul/lag = 1.551`). That gap is fish-specific and is now accounted for.

## 10.3c The fish's s=0 gap = |∇φ|≠1 × thinness. §8 IS part of the zebrafish story.

`eul/lag = 1.551` at s=0 decomposes into two measured factors whose product is **1.428** (8.6%
residual):

**(a) `|∇φ| ≠ 1` — the coarea factor the readout never applies: ×1.218.**
The zebrafish is a **mesh body**: its union SDF is 16 tabulated per-link SDFs, interpolated and
min-combined. Measured in the band, `mean|∇φ| = 0.82` (`0.75` right at the surface) — the medial
axis of a body only 2.86h thick, plus interpolation smoothing and min-combine kinks. The coarea
identity `∮_{φ=0} g dS = ∫ δ(φ) g |∇φ| dV` needs a **multiply by `|∇φ|`**; `delta_order = 1` applies
**none**, so the readout over-reads by `~⟨1/|∇φ|⟩`.

Measured directly on the fish snapshot via the `delta_order` knob (`order2 = order1 / |∇φ|`, so the
ratio *is* `⟨1/|∇φ|⟩`):

| s/h | order1 | order2 | `order2/order1 = ⟨1/|∇φ|⟩` | `eul/lag(0)` | **corrected** `× ⟨|∇φ|⟩` |
|---|---|---|---|---|---|
| 0.0 | −7.923e-06 | −9.648e-06 | **1.218** | 1.551 | **1.273** |
| 1.0 | −1.592e-05 | −1.726e-05 | 1.084 | 3.117 | 2.875 |
| 2.0 | −2.023e-05 | −2.093e-05 | 1.034 | 3.960 | 3.828 |

**This promotes §8 from "side finding, do not conflate" to part of the main story.** `delta_order=2`
exists to apply exactly this correction, and it was **inverted** — it divided where the coarea
formula multiplies, so enabling it made a mesh body **1.218× worse**, not better.

**FIXED 2026-07-16 (§10.9).** But measuring the fix deflates the estimate in this section's first
draft (which guessed "~22%" from `⟨1/|∇φ|⟩`; the kernel applies the factor per-cell, and Jensen makes
those differ). Measured post-fix on the snapshot:

| s/h | order1 (live) | order2 FIXED | o1/lag(0) | **o2/lag(0)** |
|---|---|---|---|---|
| 0.0 | −7.923e-06 | −6.903e-06 | 1.551 | **1.351** |
| 1.0 | −1.592e-05 | −1.495e-05 | 3.117 | 2.925 |
| 2.0 | −2.023e-05 | −1.965e-05 | 3.960 | **3.846** |

So the coarea factor is worth **13% at s=0 but only 2.9% at the live shift s=2h** — because at
s=2h the band sits out in the fluid, where the union SDF *is* a clean distance function
(`|∇φ|≈1`); `|∇φ|` only degrades near and inside the thin body. **The fix is correct and worth
having, but it does NOT rescue the zebrafish** — the band shift and thinness dominate. Do not oversell it.

**(b) Thinness — the band cannot localise a surface: ×1.173.**
Isolated on the sphere oracle at the fish's thinness with `eps_body = 2h` (exact analytic sphere, so
`|∇φ| = 1` exactly, single closed triangulation, no joints, `∇·σ` uniform — every other factor
removed):

| R/h | eul(0)/exact | lag(0)/exact | eul(0)/lag(0) |
|---|---|---|---|
| **3.0** | 1.173 | 0.999 | **1.173** |
| 4.6 | 1.074 | 0.999 | 1.075 |
| 6.2 | 1.040 | 0.999 | 1.041 |
| 12.6 | 1.010 | 0.999 | 1.010 |

Pure geometry, and it vanishes as `R/h` grows. **The lagrangian formula stays exact (0.999) at every
`R/h`** — it does not care that the body is thin. Note this is NOT the δ-weighted volume-average
effect: that model predicts `⟨V⟩_δ(0)/V(0) = 0.955` for the fish, i.e. ~1.0, so band-smearing through
a thin body is **not** the mechanism — tested and rejected.

**RETRACTED — there are NO buried triangles.** An earlier version of this section claimed the
triangulation carried 1.57× the union surface area from buried inter-link faces. **That was wrong**,
an artifact of estimating the union area by coarea (`∫δ|∇φ|dV`) on a body thinner than the band,
where `|∇φ|→0` on the medial axis deflates the estimate. The triangulation is clean:
- `∮ x·n dS / 3 = 5.013e-08 m³` vs the SDF's union volume `5.029e-08 m³` — **ratio 0.997**, so the 16
  links tile the fish with negligible gap or overlap (the §7 validation, passed).
- **100% of all 174,436 triangle centroids** sit within 1h of the union surface (`φ/h ∈ [−0.81,
  0.77]`, median −0.10). Zero are buried.

So `convexify=False` is doing its job here, and the lagrangian's per-link forces are **not** polluted
by joint caps. Open question #2 (lagrangian noise) needs a different explanation.

## 10.3b Neither live setting is the truth

On the frozen fish field:

| off/h | lagrangian `Fv_x` |  | s/h | eulerian `Fv_x` |
|---|---|---|---|---|
| 0.0 | −5.11e-06 |  | 0.0 | −7.92e-06 |
| 1.0 | −8.56e-06 |  | 2.0 | −2.02e-05 |
| **1.5** | **−8.76e-06** |  | 3.0 | −2.25e-05 |
| 2.0 | −8.04e-06 |  | 4.0 | −3.22e-05 |

The lagrangian has a genuine smooth **maximum at off ≈ 1.5h** — the classic "escape the BDIM band,
before interpolation/decay eats it" trade-off — and its live `off = 0` sits on the contaminated side
of that peak, under-reading by ~40%. The eulerian has **no plateau at all**: it varies ~4× over the
plausible range of s with nothing to extrapolate from. (Linear Richardson from s=2h,4h returns
−8.2e-06, close to lagrangian's peak — suggestive that truth ≈ −8.5e-06, i.e. eulerian@2h over-reads
~2.4× and lagrangian@0 under-reads ~0.6× — but the eulerian curve's shape does not justify trusting
that construction here.)

## 10.4 Recommendation (revises §6 / §9.5 item 2)

1. **The zebrafish should use `lagrangian` with `lagrangian_sample_offset ≈ h` (≈ 2.0e-4 m).**
   **Now settled live — see §10.10**, which supersedes the tentative "1.5h" reasoning this item
   originally carried. Any offset in [0.5h, 3h] gives the same swim speed to within 2% (a genuine
   plateau, not a tuned optimum) and closure in 93-109%; `h` sits mid-plateau. The live default of
   **0.0 is the worst setting tested** (closure 74.4%, speed 20% below the plateau) because it is the
   one point still inside the BDIM band. `solver.py:418`'s own comment already advises ~eps.
   (The sphere oracle shows the lagrangian is *exact* at off=0 and degrades ~8%/h with offset when
   there is NO BDIM band — §10.3. That cost is real but is plainly the lesser one in a live run: the
   band contamination it escapes is worth 26 points of closure.)
2. **Do not try to rescue the eulerian viscous readout on bodies this thin.** Option C (extrapolate
   σ back to φ=0) is still the principled fix for *resolved* bodies, but at `eps/R ≈ 0.7` there is no
   asymptotic regime to extrapolate from — the band is wider than the body. The honest fix for thin
   swimmers is the lagrangian readout, or a finer grid (the 1024×256×128 block already commented out
   in the config would put `R/h ≈ 5.7`, `eps/R ≈ 0.35`).
3. **`force_delta_order = 2` inversion: FIXED (§10.9).** It is now safe to enable order 2 for mesh
   bodies, and it is the only error here with a known-correct answer. But it buys only **2.9%** at
   the live band shift (§10.3c) — worth taking, not a cure.
4. Any eulerian fix must still be gated on the §9.5 campaign (cylinder 512²/1024² vs K&L, gazzola
   `U_t`, coquerelle 3D) — but note it can now ALSO be gated on this snapshot in ~15 s.

## 10.5 Done this session

- `tests/test_forces.py`: **the suite's first physics force tests** (§9.5 item 3) — 6 tests, ~1 s,
  promoting `oracle_native_three_way.py`. Assert lagrangian ≈ exact (both channels) and eulerian
  pressure ≈ exact (both submethods); **pin** the eulerian viscous offset over-read against the
  `((R+eps)/R)³` model with a docstring saying a fix should flip the assertion; and pin
  deltaH-viscous == ndelta-viscous (so deltaH is never mistaken for a candidate fix). Every other
  force test in the repo is parity-or-snapshot only.
- `validation/force_readout_oracle/`: `zfish_snapshot_hook.py`, `gen_zfish_snapshot.py`,
  `shift_sweep_3d.py` (+ `shift_sweep_2d.py` from session 2).

## 10.6 Two PRE-EXISTING failures found, NOT caused by this work or the refactor

`test_python_eulerian_force_path_cpu_regression` and `..._cpu_eq_gpu` fail. Confirmed **pre-existing
at 39fb3b4** (session-2 HEAD) by building that commit in a separate worktree: the numbers are
**bit-identical** there, so the `e5ad1f0..ec64de6` refactor chain (`bdim_forcing→bdim_apply`, file
splits, dead-`bdim_coeff` deletion) is **physics-neutral** — a clean refactor.

- CPU is `[0.48735056267128085, -0.5337453893897146, 28.237438779046265, -11.85863205998823]` vs a
  frozen expectation of `[0.4901618826264682, ...]` — a **~6e-3 relative** drift, where every past
  re-freeze in that docstring was ~1e-9/2e-8 ("arithmetic-order roundoff").
- CPU vs GPU now differ by ~4e-3 relative, though the docstring records that unifying the Poisson
  driver had *made that test pass*.

The test drives the **python** force branch (it asserts `_kernel_step is None`), so the divergence is
upstream in the fields — projection / `bdim_apply` / Poisson — not in the readout. Someone should
bisect it; it is a different bug from this handoff and is deliberately left untouched here.
**Note it means a CPU/CUDA physics divergence is live on this branch right now.**

## 10.7 Reproduce

```bash
python -m pytest lilytorch/tests/test_forces.py -k oracle -q        # ~1 s
python -m lilytorch.validation.force_readout_oracle.oracle_native_three_way
ZFISH_SNAP_FORCE_METHOD=lagrangian \
  ZFISH_SNAP_OUT=/data/andreaferrario/ns_data/zfish_force_snapshot/snap_lagr.pt \
  python -m lilytorch.validation.force_readout_oracle.gen_zfish_snapshot   # ~15 s
python -m lilytorch.validation.force_readout_oracle.shift_sweep_3d \
  /data/andreaferrario/ns_data/zfish_force_snapshot/snap_lagr.pt
```

`gen_zfish_snapshot.py` needs a rebuilt native extension (`python setup.py build_ext --inplace`) —
the op set moved with the refactor chain.

## 10.9 FIX APPLIED: `force_delta_order = 2` now multiplies by |∇φ| (§8 closed)

The coarea identity `∮_{φ=0} f dS = ∫ f δ(φ) |∇φ| dV` needs a **multiply**; every site divided. The
`Towers (2008)` citation in the comments was misremembered — Towers discretises `∫f δ(φ)|∇φ|`. The
oracle settles it independently: for `φ = g·(r−R)`, `∫δ_ε(φ)dV = A/g`, so recovering `A` requires
`×|∇φ|`.

Changed (all five sites, kept single-source):
- `csrc/cuda/eulerian_forces.cu` — 2-D and 3-D kernels
- `csrc/ops_2d.cpp`, `csrc/ops_3d.cpp` — the CPU twins
- `forces.py` — the python 2-D and 3-D paths
- `solver.py` — the docstring that *specified* the wrong formula (it said "divided by |∇SDF| so that
  the volume integral gives the correct surface measure")

The `1e-3` min-clamp on `grad_mag` went with it: it existed only to guard the division. Under a
multiply, `|∇φ|→0` cells *should* contribute ~0 — they are medial-axis, not surface.

**Risk: low.** `delta_order` defaults to **1**, and every edit is inside a `delta_order == 2` branch
(`sdf_grad_mag` is only even computed `if self.force_delta_order == 2`), so **no current production
run changes**. It is also inert at `|∇φ| = 1`, so analytical bodies are untouched at any order.

**Verification.** `test_delta_order2_applies_the_coarea_factor` (new) drives a deliberately scaled
SDF at `|∇φ| = 0.5, 1.0, 2.0` and asserts order 1 reads `exact/|∇φ|` while order 2 recovers `exact` —
it replaces the test that pinned the inverted behaviour, as that test's docstring instructed. The
CPU-vs-GPU parity tests at `delta_order=2` still pass, so both twins moved together.
Full suite: **362 passed, 10 failed** — and the same **10 failures reproduce exactly at the pre-fix
commit 2246f45** (verified in a separate worktree), i.e. all pre-existing: 2 are §10.6, 3 are
`test_two_phase` uniform-parity, 5 are `test_whole_step_capture_native`. None are caused by this fix.

## 10.10 LIVE ARBITRATION (§10.8 item 1 CLOSED) — off=0 is the worst setting in the repo

Eight headless production-config runs (0.6 s each, ~40 s wall), `{stack}/{case}` layout, via
`gen_zfish_readout_arbitration.py`. Arbiter = hydrodynamic-power closure
(`zfish_pbdim_closure.py`), the check the freq-cross study used (90-102% per case):

    P_BDIM = -Σ_links [F_i·v_i + τ_i·ω_i]   from drags.h5 (the READOUT forces)
    closure = ∫P_BDIM dt / (ΔE_k + E_diss)  denominator from the FLUID fields alone

The denominator knows nothing about the readout, so a readout that over-reports drag claims more
work than the water received → closure > 100%.

| case | closure | speed [mm/s] | speed [BL/s] |
|---|---|---|---|
| `eul_order1` | **128.3%** | 44.8 | 2.49 |
| `eul_order2` (the §10.9 fix) | **115.8%** | 42.1 | 2.34 |
| `lagr_off0` **(today's default)** | **74.4%** | 65.4 | 3.64 |
| `lagr_off0p5h` | **97.8%** | 77.7 | 4.32 |
| `lagr_off1h` | 107.8% | 79.5 | 4.41 |
| `lagr_off1p5h` | 109.1% | 78.7 | 4.37 |
| `lagr_off2h` | 104.9% | 78.4 | 4.36 |
| `lagr_off3h` | 93.3% | 78.7 | 4.37 |

**Three results, in order of confidence:**

1. **`lagrangian_sample_offset = 0` — the current default — is the single worst setting tested.**
   Closure **74.4%**: it under-reports the power delivered to the fluid by a quarter, worse than
   either eulerian. And its speed (65.4) is an outlier below a **flat plateau at ~78 mm/s that every
   offset from 0.5h to 3h agrees on within 2%**. Once the sample escapes the BDIM band, the answer
   stops depending on where you put it — off=0 is the one point still inside the band. That flatness
   is the strongest signal in the table: it is a real plateau, not a tuned optimum.
2. **The eulerian genuinely over-reports** (closure 116-128%) and swims ~45% slower than the
   lagrangian plateau. Consistent with §10.2. **Not usable on a body this thin.**
3. **The §10.9 `delta_order=2` fix is worth more live than the frozen field predicted**: eulerian
   closure **128.3% → 115.8%** (12.5 points) — vs the 2.9% the frozen snapshot suggested (§10.3c).
   The frozen estimate held the fields fixed; in a coupled run the correction compounds. It improves
   the eulerian but does not save it.

**Nuance — closure is a weaker discriminator than speed here, and that is physical.** The eulerian is
only 16% off in closure yet 45% off in speed. No contradiction: closure measures *total* power
exchange, which pressure dominates; speed is set by *net* streamwise force, a difference of two
larger numbers, where a 4× viscous over-read bites hard. Do not read "116%" as "the eulerian is
roughly fine".

**Caveats.** One run per setting; 0.6 s (0.3 s analysis window); the eight runs follow different
trajectories, so this compares each readout's self-consistency, not readouts on identical fluid
states (that is what §10.1's frozen snapshot is for). The 97.8-109.1% spread across offsets ≥0.5h is
plausibly run-to-run noise; the 74.4% is not.

**`verify_energy_balance.py` is NOT the arbiter for this question** — it returns *identical* numbers
to 5 significant figures for every readout (`<P_act>` 3.5183e-02 W, residual 102.7%). Expected, and
already recorded in `project_zebrafish_energy_balance`: actuator power is ~5000-12000× the power
reaching the fluid (only ~0.02% of muscle work enters the water), and a position-controlled fish
tracks the same gait regardless of readout, so P_act cannot see the difference. Use
`zfish_pbdim_closure.py`.

## 10.8 What is still open

1. ~~Verify the §10.4 recommendation in a live run.~~ **DONE — §10.10.** What remains: apply
   `lagrangian_sample_offset ≈ 2.0e-4` to the production configs (left to the user; the zebrafish
   config was being actively edited), and re-run the longer 2001-step case to confirm the plateau
   holds beyond 0.6 s.
2. The §10.6 CPU/GPU divergence (independent bug).
3. Open question #2: *why* lagrangian is noisier in coupled runs — still unmeasured, and §10.3c
   **eliminated** the most attractive hypothesis (buried joint caps sampling unconstrained interior
   pressure): there are no buried triangles. Remaining angle: `off=0` sits on a steep part of the
   offset curve, so pose jitter maps straight into force jitter; the `off≈1.5h` peak is stationary
   and should be quieter. Cheap to test on snapshots from consecutive steps.
4. **The 8.6% residual** in §10.3c's decomposition (1.428 modelled vs 1.551 measured). Likely the
   BDIM band, which no oracle here reproduces (the oracle imposes analytic fields — §2 open
   question #1 is still open in that sense).
5. §8's `force_delta_order=2` inversion is now a §10.4 action item, not a side finding. The
   `sdf_vals` crash remains untouched.
