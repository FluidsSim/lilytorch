# Stage 2 handoff: well-balanced gravity for the TWO-PHASE solver

> ## ⚠️ STAGE 2 OUTCOME (2026-06-19, attempted) — recommended approach does NOT work
>
> The recommended §2 path (flat-reference variable-density hydrostatic split)
> was implemented faithfully in `two_phase.py` / `two_phase_solver.py` and tested.
> **Result: it does not reduce the two-phase swim-speed / KE excess, and it makes
> the surface case worse.** It is shipped **behind an opt-in flag
> `solver.two_phase.wb_gravity` (DEFAULT OFF)** so production is byte-identical;
> do NOT enable it without re-checking the below.
>
> **Why (the premise is wrong for two-phase):** unlike single-phase Stage 1,
> the two-phase **variable-coefficient** projection ALREADY balances the static
> bulk hydrostatic to machine precision even at the loose production cycle cap.
> Controlled isolation (`_submerged_diag/confirm_wb_twophase.py`, capped
> multigrid+jacobi = production solver):
> - quiescent flat column, gravity on, **legacy KE ≈ 9e-13** (machine zero);
> - **static submerged body** (the mu0 hole) gravity on, **legacy KE ≈ 5.7e-14
>   (uniform) / 1.1e-10 (real 833:1)** — still machine zero.
> So there is no under-resolved BULK hydrostatic to remove. The wb split's
> interior cancellation is exact (verified to 9e-15), but `gradient()` zeros the
> reference gradient at the domain boundary, leaving an uncancelled `dt·g` jump
> → it *introduces* ~1.8e-6 spurious KE.
> - **Real-grid KE gate** (`_run_keflow.py`, REALAIR, loose): submerged
>   legacy/nograv **4.10×**, wb/nograv **4.18×** (no help); surface wb/nograv
>   **5.86×** (worse). Target was ~1.0×. FAILS the §3.1 gate.
>
> **Implication:** the real-grid two-phase swim excess (the modest ~2.5× of §5,
> *not* the 25× uniform-density single-phase artifact) is **moving-body / BDIM-
> band / interface-coupling driven, not a static under-resolved bulk
> hydrostatic** — so a static hydrostatic-reference split (this whole §2/§4
> family) cannot fix it. Next real levers are in the moving-body coupling, not a
> gravity reference. **Decision deferred to the user** (handoff §5 flagged the
> cost/benefit as a judgement call). The implementation + study harness are kept
> for inspection; nothing is committed.
>
> **WHERE THE REAL PROBLEM IS (followed up 2026-06-20):** the two-phase swim
> over-read is the FORCE READOUT (it integrates the full hydrostatic pressure for
> emergent buoyancy → a spurious horizontal force that scales LINEARLY with depth;
> single-phase reads p_d only and is depth-flat). The flow is fine. Static-pressure
> readout fixes (anchor / ∂H / gauge-invariant / minimal control-volume `−∫∇p`,
> which is the ∂H dual) are all dead ends; the residual points at the unsteady /
> added-mass momentum terms. Full handoff + the decisive next experiment:
> **`milestones/two_phase_force_readout_next_agent.md`**. Memory:
> `project_two_phase_force_gauge_leak` (CAPSTONE 2026-06-20).
>
> ---

**For:** a fresh-session agent. **Created:** 2026-06-19. **Status:** Stage 1
(single-phase) SHIPPED + validated; Stage 2 (two-phase) ATTEMPTED — recommended
approach does not work (see the OUTCOME box above), shipped opt-in/default-off.
**Prereq reading (in order):**
1. `milestones/hydrostatic_gravity_handoff.md` — the root-cause brief + the
   STATUS UPDATE at the top (Stage 1 result).
2. memory `project_two_phase_force_gauge_leak` — full diagnostic history.
3. memory `project_two_phase_force_methods`, `project_two_phase_poisson_bottleneck`,
   `project_surface_eel_overspeed`.

---

## 0. Where Stage 1 left things (read this first)

**Root cause (established):** the two-phase swim-speed overspeed is a spurious
FLOW driven by an under-resolved hydrostatic pressure gradient when gravity is on
(the loose production `poisson_tol=1e-4` cannot resolve the stiff ~1000 Pa
hydrostatic on the anisotropic grid). It is NOT a force-readout problem
(anchor / ∂H / approach-C all failed) and NOT two-phase-specific in origin
(single- and two-phase at uniform density are identical).

**Stage 1 fix (single-phase / uniform density), shipped in `lilytorch/src/solver.py`:**
- `_build_hydrostatic_reference()` builds the cell-centred analytic
  `p_h = ρ·(g·x)` (uniform density).
- `_apply_gravity_body_force()` now adds `dt·g − (dt/ρ)·∇p_h` instead of `dt·g`.
  The analytic `∇p_h = ρg` cancels the body force in the interior to machine
  precision → the Poisson never sees the stiff hydrostatic → no spurious flow.
- Gated by `self._wb_gravity` (set in `_init_gravity`). No-gravity runs are
  byte-identical (guard). `TwoPhaseSolver._build_hydrostatic_reference()` returns
  `None` → two-phase currently keeps the LEGACY uniform `dt·g` (unchanged).
- **The force readout uses the DYNAMIC pressure `p_d` ALONE.** This is the key
  Stage-1 lesson and it is SINGLE-PHASE-SPECIFIC (see §1).

**Stage 1 validation (all pass):** KE gate single-phase grav/nograv 25.2×→1.000×
at loose tol (submerged + surface); statics quiescent + `p=ρg·depth` exact;
two-phase units 9/9; full swim A/B (N=6000, real coupling) wb-gravity +0.0712 =
no-gravity ref +0.0712 (identical). Mechanism proven.

---

## 1. The single-phase vs two-phase READOUT distinction (do not get this wrong)

This tripped Stage 1 once (cost a wrong iteration) and is the #1 thing to
internalise:

- **Single-phase (Stage 1): buoyancy is EXTERNAL** (FARMS/MuJoCo apply Archimedes
  to the rigid body). So the fluid force readout must use `p_d` ONLY. Adding
  `p_h` back into the band quadrature `Σ −p n δ_ε` both DOUBLE-COUNTS buoyancy
  AND leaks a spurious HORIZONTAL force through the discretely non-gauge-invariant
  quadrature → it REVERSED the single-phase swim (−0.049 vs +0.071). Verified.

- **Two-phase (Stage 2): buoyancy is EMERGENT** — the BDIMhandler DISABLES the
  external FARMS buoyancy and the band quadrature of the REAL (full) pressure
  recovers buoyancy from the fluid (floating cylinder 0.91× Archimedes,
  drop-sphere Cz≈1). So in two-phase the readout MUST see the full physical
  pressure `p = p_d + p_h`. The Stage-1 "p_d only" rule does NOT carry over.

**Consequence:** the recommended Stage 2 (§2) changes ONLY the FLOW (the
predictor), and RECONSTRUCTS the full pressure `p_d + p_h` for the force readout
so the existing emergent-buoyancy quadrature (+ `gauge_anchor_forces`) is
behaviourally UNCHANGED. No buoyancy-decomposition surgery in the recommended
path. (The handoff §3.3 "analytic Archimedes" decomposition is the fallback in
§4, only if reconstruction proves insufficient.)

---

## 2. RECOMMENDED Stage 2 — fix the FLOW only, keep the readout behaviour

All edits in `lilytorch/src/two_phase_solver.py` (+ a helper in
`lilytorch/src/two_phase.py` if convenient) per
`feedback-no-core-source-for-two-phase`. `git diff` on `forces.py`/`body.py`/the
single-phase parts of `solver.py` must stay empty.

### 2.1 Build a hydrostatic reference `p_h` (variable density)
Override (or add a sibling to) `TwoPhaseSolver._build_hydrostatic_reference`.
`p_h` must satisfy the DISCRETE balance `(p_h[i]−p_h[i−1])/h = ρ_face[i]·g` on
each face along the gravity axis (so the same backward-difference `gradient()`
used by the projection cancels exactly). Build it by a **cumulative sum of the
face density along the gravity axis**:

```
ρ_face = 1 / two_phase.recip_density_face(axis)     # two_phase.py:137
p_h[k] = p_h[k−1] + h · ρ_face[k] · g_axis          # cumsum along gravity axis
```

**CRITICAL design choice — which density profile is the reference?**
- The original root-cause handoff (§3.1) said "recomputed each step from the
  instantaneous `alpha`". **BE CAREFUL: an instantaneous reference cancels the
  hydrostatic of the CURRENT interface shape, including the WAVES — i.e. it
  removes the wave restoring force and can kill the wave-making drag that is the
  whole reason two-phase is needed.** With an instantaneous reference the
  predictor increment is ≈0 everywhere and gravity drives essentially no flow.
- **Recommended: a FIXED FLAT-INTERFACE reference** (water below the equilibrium
  waterline z≈0, air above, with the same smoothed transition). This cancels only
  the stiff EQUILIBRIUM bulk hydrostatic (the part the loose Poisson
  under-resolves → the spurious flow), and leaves wave + body perturbations as
  REAL dynamics. This is the standard well-balanced / "pressure perturbation
  about a hydrostatic reference" scheme.
- You MUST test both against the wave-drag / swim-speed gate (§3.5). If the fixed
  reference leaves residual spurious flow because the interface has drifted far
  from flat, consider a slowly-updated (horizontally-averaged) reference — but
  NOT the raw instantaneous one.

### 2.2 Predictor balance (variable density)
Override `TwoPhaseSolver._apply_gravity_body_force` (it already exists for the
`_consistent_momentum` short-circuit — extend it). For each velocity face
component on a gravity axis:

```
vel += dt · ( g − (1/ρ_face_inst) · ∇p_h )
```

- `ρ_face_inst` = the INSTANTANEOUS face density (`1/recip_density_face`), the
  SAME coefficient the projection uses. In the still reference column
  `ρ_face_inst == ρ_face_ref` → increment = `dt·(g − g) = 0` (quiescent). Under
  the body / displaced interface the bracket ≠ 0 = the real reduced-gravity /
  buoyancy driving term.
- `∇p_h` = the reference gradient (built in 2.1), via the same `self.gradient`.
- Do NOT mu0-weight the balance (Stage 1 didn't; the body enters through mu0 in
  the projection). VERIFY with the KE gate that the body band stays quiescent.
- In `_consistent_momentum` mode gravity is applied inside the conservative
  advection — keep that short-circuit; the well-balanced term must be added in
  the SAME place (consistency). Start with the default (non-consistent) path.

### 2.3 Reconstruct full pressure for the readout
In `TwoPhaseSolver.advance_and_compute_loads` (it already overrides the base —
see two_phase_solver.py:756), after `fluid_step` solves `p_d`, set
`p = p_d + p_h` BEFORE the (deep-interior) `zero_pressure_inside`, the force
calls, and storage. This makes the force readout, `self.p0`, and plotting see the
full physical pressure — i.e. behaviourally identical to today's full-p
quadrature + `gauge_anchor_forces`. (This is the OPPOSITE of the single-phase
Stage-1 rule, for the reason in §1.)

---

## 3. Validation gates (run in this order; STOP if an early one fails)

Harness already on disk (throwaway): `_run_keflow.py` + `_ke_ext.py` (fluid-only
KE, `force_scaling=0`), `gen_config_submerged_diag.py` (real-coupling swim A/B via
`SpeedLogger`). Knobs documented in those files; I added `KEFLOW_Z` (spawn depth).
Results → `_submerged_diag/*.csv`; plots/frames → `/data/andreaferrario/ns_data/`.

1. **KE gate (cheap, decisive for the FLOW).** Real two-phase (`KEFLOW_REALAIR=1`),
   grav vs nograv, loose tol, both z=−0.07 AND z=−0.0115:
   `KEFLOW_N=400 KEFLOW_REALAIR=1 KEFLOW_NOGRAV={0,1} KEFLOW_Z={-0.07,-0.0115}`.
   Current (Stage-1, two-phase still legacy): grav/nograv **2.54× submerged /
   2.28× surface**. TARGET after Stage 2: **~1.0×**, NO blow-up, at LOOSE tol.
   (Do NOT just tighten tol — multigrid stalls, mgcg blows up; proven.)
2. **Hydrostatic statics:** a still two-phase column stays quiescent (KE~0) and
   `p = ρ_w g·depth` below the interface, ~0 above. Use a standalone like
   `_submerged_diag/confirm_wellbalanced.py` adapted to two-phase, or a
   quiescent-tank unit test.
3. **Buoyancy PRESERVED (the regression guard):** floating cylinder ≈0.91×
   Archimedes; 3-D drop-sphere Cz≈1; `single_sphere_drop_two_phase_3d` float
   (ρ0.5 settles at Archimedes draft) / sink (ρ1.5 descends) UNCHANGED. With the
   recommended §2.3 reconstruction these should be ~identical to today.
4. **Two-phase unit tests:** `lilytorch/src/test_two_phase.py` (9 tests) pass.
5. **Swim speed (the actual goal):** submerged real two-phase self-propelled
   should drop from +0.091 toward the single-phase reference +0.080; the SURFACE
   self-propelled (the production case) should stay ≈ robot 0.128 (today
   `gauge_anchor_forces`=True already gives +0.127 at the surface — do not
   regress it). Run via `gen_config_submerged_diag.py` (two-phase path, default)
   and/or a headless `gen_config_surface_pool.py` A/B.

---

## 4. FALLBACK (only if §2.3 reconstruction still biases the swim)

If reconstructing `p_d + p_h` and running the existing quadrature still
over-reads (i.e. the residual was partly readout after all), switch to the
handoff-§3.3 ANALYTIC decomposition: force = quadrature(`p_d`) [dynamic +
perturbation buoyancy] + ANALYTIC Archimedes from the reference `p_h`
(displaced-volume integral, vertical, gauge-clean — NEVER through the
quadrature). This is the displaced-volume split the two-phase design deliberately
removed (`project_two_phase_force_methods`), so re-validate gate 3 hard. This is
the higher-risk path; only take it if the data demands it.

---

## 5. Hard constraints / pitfalls

- **Core-edit authorization:** Stage 1 (single-phase solver.py) was explicitly
  authorized. Stage 2 should live in `two_phase_solver.py` / `two_phase.py`
  (allowed). If you find you MUST touch core `solver.py`/`forces.py`/`body.py`,
  STOP and get explicit user authorization first.
- **Keep no-gravity + single-phase byte-identical.** All Stage 2 code behind the
  two-phase + `use_gravity` path. `git diff` on the single-phase code paths and
  on forces.py/body.py must stay empty.
- **Real-case magnitude is modest (~14% / 2.5× KE), not the 25× uniform-density
  artifact.** Quote the real numbers. The cost/benefit of Stage 2 is a judgement
  call the user is aware of — confirm scope before a large effort.
- **Do NOT chase the force readout** (anchor/∂H/approach-C all failed; the FLOW is
  the problem). The recommended Stage 2 deliberately does not touch the readout.
- **`gauge_anchor_forces` stays ON** at the surface (it gives the correct
  magnitude there). Stage 2 fixes the flow underneath it.
- **Harness files are throwaway** (`_run_keflow.py`, `_ke_ext.py`,
  `confirm_wellbalanced.py`, `keflow_*.csv`, `speed_sp_swim_*.csv`,
  `gen_config_submerged_diag.py` DIAG_SINGLEPHASE path). Clean up when done.

---

## 6. First action for the Stage 2 agent

1. Re-run the KE gate (§3.1) on the CURRENT code to confirm the 2.54×/2.28×
   baseline reproduces (sanity that nothing drifted).
2. Implement §2.1 with a FIXED flat reference, §2.2 predictor, §2.3
   reconstruction.
3. Re-run §3.1 → expect ~1.0×. If it works, run §3.3 (buoyancy regression) and
   §3.5 (swim speed). Decide with the user before committing.
