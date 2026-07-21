# Handoff: the single-phase vs two-phase swim-speed discrepancy is a GRAVITY / under-resolved-hydrostatic FLOW problem

**For:** a fresh-session agent. **Created:** 2026-06-19. **Status:** root cause
established; surface measurement DONE; **STAGE 1 IMPLEMENTED + VALIDATED**; Stage 2
(two-phase) still open. **Prereq reading:** memory `project_two_phase_force_gauge_leak`,
`project_surface_eel_overspeed`, `project_two_phase_poisson_bottleneck`.

---

## STATUS UPDATE (2026-06-19, later same day) — §6 + Stage 1 DONE

**§6 surface measurement:** the spurious gravity flow at the real swimming depth
(z=−0.0115, 833:1) is the SAME ORDER as submerged — KE grav/nograv **2.28×**
(surface) vs **2.54×** (submerged), and larger in ABSOLUTE terms (excess 2.78e-6
vs 2.16e-6). So the fix is justified where the eel actually swims. (Added a
`KEFLOW_Z` spawn-depth knob to `_run_keflow.py`.)

**Stage 1 (single-phase / uniform density) IMPLEMENTED + VALIDATED** (user-authorized
core edit). solver.py: `_build_hydrostatic_reference` (`p_h=ρ·(g·x)`),
`_apply_gravity_body_force` now adds `dt·g − (dt/ρ)∇p_h` (interior cancellation to
machine precision). two_phase_solver.py: `_build_hydrostatic_reference`→None so
two-phase + all no-gravity runs stay BYTE-IDENTICAL. forces.py/body.py untouched.
- §5.1 KE gate: single-phase grav/nograv **25.2× → 1.000×** at LOOSE tol (= the
  tight-tol brute force, at no perf cost), submerged AND surface.
- §5.2 statics: quiescent + `p=ρg·depth` exact (dp/dy=−9810 Pa/m).
- §5.4: two-phase tests 9/9 pass.
- **§5.5 full SWIM A/B (decisive):** single-phase, real force coupling, N=6000,
  frelax=0.5 — wb-gravity **+0.0712 m/s = no-gravity ref +0.0712** (identical
  trajectory). Adding well-balanced gravity has ZERO effect on the swim. ✓

**LESSON for Stage 2 (cost me one wrong iteration):** §3.3 below is RIGHT but easy
to misread — the force readout must use the DYNAMIC pressure `p_d` ALONE. Do NOT
add `p_h` back into the body-force band quadrature `Σ -p n δ_ε`: that feeds the
large hydrostatic head through the discretely NON-gauge-invariant integral and
LEAKS a spurious horizontal force (it REVERSED the single-phase swim to −0.049).
The hydrostatic/buoyancy must enter ANALYTICALLY (Archimedes) or stay external
(FARMS/MuJoCo handle it for single-phase) — never through the quadrature.

---

## 0. The question

The self-propelled eel swims at ~0.128 m/s (robot). Single-phase underwater
(no gravity) reproduces this and is trusted. The two-phase solver (needed for the
free surface) **over-reads the swim speed**. We want two-phase to agree with
single-phase. Many *force-readout* fixes were tried over prior sessions and ALL
failed: `gauge_anchor_forces` (band-aid, over-corrects), `gauge_invariant_forces`
(approach C, exact gauge removal), `partial_heaviside_forces` (∂H weight). They
converge to ~the same biased speed → the bias is NOT in the force readout.

## 1. RESULT — root cause (this session, firmly established)

**The discrepancy is a spurious FLOW driven by an under-resolved hydrostatic
pressure gradient when gravity is on — not a force-readout problem, and not
two-phase-specific in origin.**

Decisive evidence (all submerged z=−0.07, `force_scaling=0` so the body motion is
identical with/without fluid gravity, measuring FLUID-only kinetic energy via the
harness in §4):

| test | config | Poisson | KE result |
|---|---|---|---|
| 2D standalone | single-phase, moving body | tight mgcg | gravity on/off = **1.00×** (no spurious flow) |
| FARMS, uniform ρ | single- & two-phase | loose 1e-4 | gravity = **25×** nograv; single ≡ two-phase EXACTLY |
| FARMS, single-phase | uniform ρ | tight (mg 1e-8 / cg 1e-9) | gravity → **1.0×** (FIXED) |
| FARMS, **real 833:1** | two-phase | loose 1e-4 | gravity = **2.5×** |
| FARMS, **real 833:1** | two-phase | multigrid 1e-8 | **2.5× — NO change** |
| FARMS, **real 833:1** | two-phase | mgcg 1e-8 | **175× — BLOWS UP** |

Reading of the table:
- Physics: uniform-density incompressible flow is gravity-invariant
  (`g=∇(g·x)` absorbs into pressure, zero velocity effect). So ANY gravity-induced
  velocity is purely the Poisson leaving `∇p` not exactly cancelling `ρg`.
- `single-phase ≡ two-phase` at uniform density ⇒ NOT a two-phase bug; it's
  gravity + the (BDIM) projection. The trusted single-phase reference is immune
  only because it runs WITHOUT gravity.
- **Tolerance fixes single-phase** (row 3) but **cannot fix real two-phase**:
  multigrid STALLS before reaching 1e-8 on the 833:1-stiff grid (→ no change;
  this is exactly why the user's `poisson_tol=1e-8` in `gen_config_surface_pool.py`
  did nothing), and mgcg reaches it but DESTABILIZES (blows up).

**Magnitude correction (important):** the dramatic "25× / +0.136 overspeed"
numbers were measured at UNIFORM density (`rho_air=1000`, a test artifact that
inflates the hydrostatic). The REAL 833:1 case is much milder: **~2.5× fluid KE →
~14% overspeed** (submerged real two-phase +0.091 vs single-phase +0.080, from
the earlier `_submerged_diag/speed_g2_tp_real_anchor.csv` vs `speed_g2_ref_sp.csv`
ladder). Do NOT quote the 25%/+0.136 numbers for the real case.

## 2. What is STILL OPEN (do these before committing to the fix)

1. **The effect at the SURFACE is untested.** EVERY quantitative test above was
   SUBMERGED (z=−0.07). The eel actually swims at the surface (z=−0.0115),
   straddling the 833:1 interface — the hydrostatic kink sits right at the body.
   The effect there could be larger or smaller. **This is the #1 thing to measure
   next** (cheap; §4). If it's small, the fix may not be worth a core edit; if
   large, it clearly is.
2. **The fix (hydrostatic split) is unproven** — a well-founded hypothesis, but so
   was ∂H, and that failed. Validate before trusting (§3, §5).
3. Why the real effect (2.5×) ≪ uniform (25×): plausibly the light-air column
   carries little hydrostatic, but not verified.

## 3. The proposed FIX — well-balanced gravity (hydrostatic pressure split)

Split `p = p_h + p_d`. `p_h` = analytic hydrostatic with `(1/ρ)∇p_h = g` on faces
(using the SAME face density the projection uses); `p_d` = the dynamic part the
Poisson solves. `p_h` carries the entire stiff hydrostatic, so the Poisson is left
with a small, well-conditioned `p_d` → converges at loose tol, no spurious flow,
no mgcg blow-up. It is a STANDARD well-balanced scheme (ocean/atmosphere/free-
surface codes always solve for pressure perturbations about a hydrostatic
reference) — exact decomposition, NO tuning parameter, fixes the FLOW not the
readout. This is the opposite of the failed `gauge_anchor` band-aid.

### Where it goes (concrete)
1. **`p_h` builder** (new method in `lilytorch/src/solver.py`):
   - single-phase/uniform: `p_h = ρ g·x` (exact, ~5 lines).
   - two-phase: cumulative integral of `ρ_face·g` along the gravity axis from
     `TwoPhase.recip_density_face` (`lilytorch/src/two_phase.py:137`), recomputed
     each step from `alpha` (handles the interface kink). The bulk of the work.
2. **Predictor** — `FluidSolver._apply_gravity_body_force`
   (`lilytorch/src/solver.py:765`): replace `vel += dt·g` with the pre-balanced
   `vel += dt·g − c·∇p_h` (≈0 in still fluid). Guard on `use_gravity`.
3. **Force/buoyancy readout**: stored `p0` becomes `p_d`, so the body-force
   integral must use `p_d + p_h` (p_h → analytic Archimedes; p_d → dynamic load).
   This CHANGES the two-phase buoyancy decomposition from "emergent (full-p band
   integral)" back to "analytic Archimedes + dynamic" — the displaced-volume split
   the two-phase design deliberately removed (see `project_two_phase_force_methods`).
   **This is the main risk; re-validate buoyancy.**
4. Plotting/diagnostics that read `p0` as physical pressure need `p_d + p_h`.

### Risk
- **No-gravity cases (the vast majority, incl. the 1guilla underwater reference):
  `p_h=0`, BIT-IDENTICAL.** Guard on `use_gravity` ⇒ zero risk there.
- single-phase + gravity: exact `ρg·z`, low risk.
- two-phase: moving-interface `p_h` + the buoyancy-decomposition change = the risk.
- It is a CORE edit (solver.py + the force-pressure path), benefits single-phase
  too (not a two-phase hack) → needs explicit user authorization, like the
  rho_body Stage-2 kernel edit. Confirm scope with the user first.

### Recommended staging
- **Stage 1** — single-phase/uniform `p_h` + predictor balance. Small, safe,
  proves the mechanism end-to-end via the KE gate.
- **Stage 2** — two-phase variable-density `p_h` + buoyancy reconciliation (the
  valuable but risky part), gated by the buoyancy tests. Only after Stage 1.

## 4. The harness that already exists (don't rebuild it)

All throwaway, on disk:
- `lilytorch/examples/_1guillasim/experiments/_run_keflow.py` — FARMS runner.
  Subclasses the surface-pool config; `force_scaling=0` (fluid exerts no force on
  the body → identical body motion across runs), TRANSVERSE, z=−0.07,
  `diagnostics_every=1`. Env knobs:
  `KEFLOW_N`, `KEFLOW_TAG`, `KEFLOW_NOGRAV` (1=strip fluid gravity),
  `KEFLOW_SINGLEPHASE` (1=plain FluidSolver, no two_phase block),
  `KEFLOW_REALAIR` (1=keep rho_air=1.2 / 833:1; default overrides to 1000=uniform),
  `KEFLOW_PTOL` / `KEFLOW_PMETHOD` (mgcg|multigrid) / `KEFLOW_PCYC`.
- `lilytorch/examples/_1guillasim/experiments/_ke_ext.py` — a `FluidExtension`
  subclass that records FLUID kinetic energy each step (total + fluid-only, the
  latter masks out the body interior `sdf>eps`). Writes
  `_submerged_diag/keflow_<TAG>.csv` (cols `it,ke_tot,ke_fluid`). NOTE:
  `TwoPhaseSolver.finalize_step` does NOT call the base diagnostics, so we compute
  KE in the extension after each `before_step`.
- `_submerged_diag/confirm_wellbalanced.py` — STANDALONE 2D check (no FARMS):
  static & moving analytical body, gravity on/off, tight tol → showed gravity
  on/off = 1.00× (the clean control).
- Result CSVs already in `_submerged_diag/keflow_*.csv`.

Run from the experiments dir, e.g.:
```
cd lilytorch/examples/_1guillasim/experiments
KEFLOW_N=400 KEFLOW_REALAIR=1 KEFLOW_NOGRAV=1 KEFLOW_TAG=tpreal_nograv python _run_keflow.py
KEFLOW_N=400 KEFLOW_REALAIR=1 KEFLOW_NOGRAV=0 KEFLOW_TAG=tpreal_grav   python _run_keflow.py
```
Compare mean `ke_fluid` over `it>=200`. Plots/frames → `/data/andreaferrario/ns_data/`.

## 5. Validation gates for the fix (in order)
1. **KE gate** (cheap): real two-phase grav/nograv → ~1.0 (now 2.5×); single-phase
   +gravity → 1.0; NO blow-up.
2. Hydrostatic statics: a still column stays quiescent (KE~0) AND `p = ρg·depth`.
3. Buoyancy preserved: floating cylinder ≈0.91×Arch; 3-D drop-sphere Cz≈1;
   `single_sphere_drop_two_phase_3d` float/sink unchanged.
4. The 11 two-phase unit tests (`lilytorch/src/test_two_phase.py`) pass.
5. Swim speed: submerged real two-phase drops +0.091 → toward +0.080.

## 6. Recommended FIRST action for the new agent

**Measure the effect at the SURFACE before any core edit** (§2.1). Add a surface
spawn (z=−0.0115) variant to `_run_keflow.py` (or a knob) with `KEFLOW_REALAIR=1`,
gravity on vs off, and read `ke_fluid`. Decide with the user whether the
surface-case magnitude justifies the staged core fix in §3. If yes → Stage 1.

## 7. Hard constraints / pitfalls
- The fluid pressure/flow EXCEPT for this gravity issue is fine; do not chase the
  force readout again (anchor/∂H/approach-C all failed — the flow is the problem).
- Core edits need user authorization (no-core rule); keep no-gravity cases
  byte-identical (guard on `use_gravity`).
- For real two-phase do NOT just tighten `poisson_tol` — multigrid stalls (no-op)
  and mgcg blows up (proven §1).
- Quote the REAL-case magnitude (~14%), not the uniform-density 25%/+0.136.
- Harness files `_run_keflow.py` / `_ke_ext.py` / `confirm_wellbalanced.py` and the
  `keflow_*.csv` are throwaway — clean them up when done.
