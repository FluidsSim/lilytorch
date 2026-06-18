# Handoff: replace the two-phase force gauge band-aid with a gauge-invariant force

**Status:** open task, ready to start. **Owner:** new agent (fresh session).
**Created:** 2026-06-18.

## TL;DR
The two-phase body-force computation is not discretely gauge-invariant, so a
uniform/DC pressure offset (dominated by the hydrostatic head when fluid gravity
is on) leaks into a spurious net force. The current workaround
(`two_phase.gauge_anchor_forces`, which subtracts the band-mean pressure) is a
band-aid: it over-corrects and leaves a residual artifact that biases the
self-propelled swim speed. **Goal: compute the force so it is gauge-invariant by
construction — adding any constant to `p` changes the force by exactly 0 — and
delete the anchor.** Buoyancy and the dynamic load must still emerge naturally.

## Context / what's been found (verify against current code — citations may drift)
The two-phase body force is an **eulerian smoothed-delta surface integral**
`F = ∫ (σ·n − p·n) δ_ε(φ) dV`:
- `lilytorch/src/forces.py`: `_forces_shared` (pressure force density `−p·n`),
  `_forces_body_integrate_3d`, `forces_method2` / `forces_method2_3d`.
- 3-D CUDA kernel path: `streaming_sdf_forces_post_3d` (`.cu` + `.cpp` twins).
- Lagrangian path: `forces_lagrangian_*` (watertight surface, already Σ(A·n)=0;
  most accurate; natural oracle).

This integral requires `Σ n·δ_ε = 0`, which fails for a coarsely-resolved body
(~7–13 cells), so a uniform pressure leaks into a spurious force
`≈ −p_baseline · Σ n·δ_ε`. With fluid gravity ON (two-phase) the baseline is the
large hydrostatic head, so the leak swamps the real ~10 Pa dynamic load.

Current workaround = `TwoPhaseSolver._anchor_pressure_for_forces`
(`lilytorch/src/two_phase_solver.py`): subtract the BDIM-band-mean pressure from
`p` (out-of-place) before the integral. **This is the band-aid to replace.**

### Evidence the band-aid is methodologically wrong
- It's a single global constant → assumes all bodies at one depth; over/under-corrects.
- Decisive measurement (all `SpawnMode.TRANSVERSE`, frelax=0.5, submerged eel,
  forward = −dx/dt; harness below): toggling **only** the gauge on the *same flow*
  swings the swim speed **−0.092 → +0.152 m/s (sign flip)**. Ladder (all
  TRANSVERSE, frelax=0.5, z=−0.07, identical gait/BDIM, one knob changed):
  ```
  single-phase  (no gravity)                 +0.075   <- clean reference
  single-phase + gravity (no gauge)          -0.042   REVERSED
  two-phase uniform-rho + gravity, gauge OFF  -0.092   REVERSED
  two-phase uniform-rho + gravity, gauge ON   +0.152   over-corrected
  two-phase uniform-rho, NO gravity (gauge)   +0.103
  two-phase REAL air(1.2) + gravity, gauge ON +0.115
  ```
  Physically gravity/hydrostatic must give **zero** horizontal force on a
  heave-locked body, so the +0.05 (gravity) and +0.028 (code path) shifts are
  numerical force-readout artifacts; only the −0.037 (real air interface =
  wave-making drag) is physical. The FLOW is correct (the gauge is out-of-place,
  never touches `p0`); only the FORCE READOUT is contaminated.

## Goal (acceptance definition)
Force computation **discretely gauge-invariant**: `p → p + C` changes the
computed force by exactly 0 (machine precision), with **no anchor / mean
subtraction**. Buoyancy (pressure *variation* across the body) and the dynamic
load still emerge. Then `gauge_anchor_forces` is unnecessary → delete it.

## Suggested approaches (option A is the standard; decide after a code/lit check)
- **A. Control-volume momentum-flux integral (recommended).**
  `F = −d/dt ∫_CV ρu dV − ∮_∂CV (ρ u⊗u + pI − τ)·n dS` over a grid-aligned box CV
  enclosing the body. Over a closed discrete CV, `∮ p n dS = 0` identically for
  uniform p → gauge-invariant by construction (WaterLily / Weymouth–Yue standard).
- **B. Direct BDIM/penalization force (momentum exchange).** Force = the BDIM
  penalization integrated over the body region (momentum added to enforce
  `u → u_body`). Inherently pressure-gauge-free; check buoyancy separation.
- **C. Discretely-invariant surface integral.** Keep the surface form but enforce
  `Σ n·δ = 0` exactly using the *discrete normals* (not a pressure mean). Closer
  to a corrected band-aid; prefer A/B.

## Validation gates (must pass)
1. **Static-body gauge test:** stationary body in hydrostatic two-phase fluid →
   `Fx ≈ 0`, `Fz ≈ Archimedes`, **with NO gauge anchor**. (Band-aid reference:
   static surface sphere Fx −19.2 N → −0.006 N *with* anchor; new method ≈0
   *without*.) Add a unit test asserting force invariance under `p += C`.
2. **Submerged self-propelled A/B** (`gen_config_submerged_diag.py`, all
   `TRANSVERSE`, frelax=0.5, z=−0.07): two-phase submerged swim speed must
   **converge to the single-phase value** (≈ clean reference) once the artifact
   is gone. Knobs: `DIAG_SINGLEPHASE / GRAVITY / RHO_AIR / TP_NOGRAVITY / GAUGE /
   FREESLIP_TOP / SPAWNMODE / FRELAX`. Plots/frames → `/data/andreaferrario/ns_data/`
   (standard location; see memory `feedback-plots-in-ns-data`).
3. **Buoyancy preserved:** floating cylinder ≈ 0.91× Archimedes; 3-D drop-sphere
   Cz ≈ 1 (Lagrangian watertight path); `single_sphere_drop_two_phase_3d`
   float/sink unchanged.
4. Existing two-phase tests (`test_two_phase.py`, ~18) pass; single-phase 1guilla
   drags bit-stable.

## Constraints / pitfalls
- **Core-source rule:** memory `feedback-no-core-source-for-two-phase` says keep
  two-phase changes out of `forces.py`/`solver.py`/`body.py`. BUT a
  gauge-invariant force is a *general* improvement (helps single-phase too).
  **Decide scope with the user first:** implement generally in `forces.py`
  (cleaner, benefits all) vs as a `TwoPhaseSolver` override (respects the rule).
- **Three force paths** must stay consistent: eulerian python
  (`forces_method2[_3d]`), 3-D CUDA kernel (`streaming_sdf_forces_post_3d`,
  rebuild `_C.so`), Lagrangian (`forces_lagrangian_*`, the oracle).
- The fluid **pressure/flow is already correct** (dp/dz = −ρg exact) — do NOT
  touch the Poisson/flow. This is purely a force-readout change.
- `force_relaxation=0.5` in the harness is only an FSI stabiliser; not part of
  this task.

## Pointers
- Memory: `project_two_phase_force_gauge_leak` (mechanism + the 2026-06-18
  over-correction finding), `project_surface_eel_overspeed`,
  `project_two_phase_force_methods`, `feedback-no-core-source-for-two-phase`,
  `feedback-plots-in-ns-data`.
- To-do: `milestones/to_do_list.md` → `HP5b → RESULT-2` (full ladder + actionable).
- Harness: `lilytorch/farms_examples/_1guillasim/experiments/gen_config_submerged_diag.py`,
  `lilytorch/integration/speed_logger.py`; analysis `_submerged_diag/analyze_speed.py`,
  `_submerged_diag/plot_diag_kinematics.py`.

## First step
Reproduce the gauge-toggle sign-flip (`DIAG_GAUGE=1` vs `0` on the two-phase
uniform+gravity submerged case) to confirm the artifact, then prototype approach
A and re-run validation gate #1 (static-body invariance) before anything else.
