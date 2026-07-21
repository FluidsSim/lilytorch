# Next-agent handoff: the two-phase SWIM-SPEED over-read (force readout)

> ## RESULT (2026-06-20) — the §1 decisive test is RUN. Verdict: **STATIC, not unsteady.**
>
> The depth-linear horizontal over-read is **fully present on a COMPLETELY STATIC
> (zero-gait, non-moving) body**; undulation adds almost nothing to the mean Fx.
> Per the §1 decision rule this is a **static/pressure** artifact, so the §2
> control-volume *momentum* readout is the WRONG thing to build (it targets the
> unsteady/added-mass path, which the data exonerates — and it is the SBP-dual of
> the already-rejected ∂H anyway).
>
> **Numbers** (`_run_keflow.py`, uniform density, prescribed motion / force_scaling=0,
> mean Fx over last 200 of 400 steps; harness extended with KEFLOW_GAUGE / KEFLOW_GLIDE
> / KEFLOW_STATIC; CSVs `_submerged_diag/force_dec_*.csv`):
>
> | mode | gauge | Fx @ z=−0.07 | Fx @ z=−0.13 | ratio | depth ratio = 1.857 |
> |------|-------|-------------|-------------|-------|------|
> | **static** (no motion) | OFF | **+0.386** | **+0.708** | **1.835** | raw hydrostatic leak |
> | **static** (no motion) | ON  | −0.119 | −0.241 | 2.02 | gauge OVER-corrects, residual still ∝depth |
> | undulate | ON | −0.142 | −0.268 | 1.89 | undulation adds only ~−0.02 to the mean |
> | glide* | ON | −0.074 | −0.206 | 2.79 | *transient (impulsive start), not steady-state |
>
> A static submerged body's true horizontal force is **0** (incompressible flow sees
> only pressure gradients; the hydrostatic gradient is vertical). The +0.386→+0.708
> raw leak is the discrete `ρg·Σ z nₓ δ` quadrature error, exactly linear in depth.
> The shipped uniform **`gauge_anchor_forces` band-mean subtraction does NOT remove
> the depth-linearity** — it subtracts a depth-proportional amount but the wrong one,
> over-correcting to a depth-proportional residual of opposite sign.
>
> **OPEN TENSION the user must weigh:** this seems to contradict memory
> `project_two_phase_force_gauge_leak` (∂H/approach-C *zeroed* the static Fx yet moved
> the self-propelled swim only 0.137→0.134). Reconciliations: (a) ∂H didn't truly
> zero the force in the *running multi-link* eel (per-link seams on the union SDF) —
> a correct **per-body / depth-aware anchor** on the real geometry might; (b) the
> self-propelled swim-speed bias is a DIFFERENT quantity than this prescribed-motion
> static leak. The handoff itself flagged branch (a): "maybe per-body anchoring or the
> interface, not unsteady terms." **Decision required before any build.**
>
> ### FOLLOW-UP (2026-06-20, same session): per-body anchor IMPLEMENTED + TESTED — does NOT fix depth-linearity.
> Built opt-in `two_phase.gauge_anchor_per_body` (TwoPhaseSolver, no core edits):
> exact per-body post-correction `F_i[b] -= p̄_b · U_i[b]` via the p≡1 unit-pressure
> defect `U_i[b]`; per-body band-mean from `_sdf_sparse`/`sdf_vals`; needs
> `solver_method='python'` (loud RuntimeError on the kernel-streaming union-only
> path). Static body, python path: global anchor Fx **−0.119→−0.241 (ratio 2.02)**,
> per-body **−0.098→−0.214 (ratio 2.20)** — depth-linearity UNCHANGED (buoyancy Fz
> preserved +17.79). **Why:** the depth-dependence is a *weighting* mismatch, not a
> *lumping* one — the leak `Σ p nₓδ` is n·δ-weighted, but ANY anchor subtracts a
> *uniform* band-mean `p̄·Σnₓδ`; `p̄_uniform ≠` the n·δ-weighted mean of p and both
> ∝depth, so the mismatch ∝depth survives no matter how you partition the bodies.
> Only changing the WEIGHT (`nₓδ→∂ₓH`, SBP) zeros the static Fx — and that (∂H) was
> already shown to NOT move the swim. So the anchor family (global / per-body /
> approach-C) is exhausted; the crux is the ∂H paradox (zeros static, doesn't fix
> swim). Harness knobs added: KEFLOW_GAUGE / KEFLOW_PERBODY / KEFLOW_PYTHON /
> KEFLOW_GLIDE / KEFLOW_STATIC (+ root-slide-x qvel inject in `_ke_ext.py`).
>
> ### BREAKTHROUGH (2026-06-20, same session): the ∂H paradox is RESOLVED — the leak is the INTER-LINK SEAMS; ∂H over the UNION SDF kills it.
> Re-implemented opt-in `two_phase.partial_heaviside_forces` (∂H weight
> `F_i=-Σ p ∂_iH_ε(φ_b) h^D`, `_heaviside_smooth`=exact antiderivative of the cosine
> delta; python path). Verified CORRECT on a clean analytic sphere: Fx=Fy=−0.00000
> (exact SBP), Fz/Arch=1.006. **Real 9-link eel, per-link ∂H: Fx +0.549→+1.020
> (ratio EXACTLY 1.857 = depth) — worse than the anchors.** Diagnostic
> `partial_heaviside_union` (ONE ∂H over the UNION SDF): **Fx −0.007→−0.014 (≈0,
> depth-flat), Fz=13.9 (accurate buoyancy, cf n·δ 17.8)**. ⇒ the per-link leak is
> ENTIRELY the inter-link seams (each link's standalone SDF has a spurious surface
> where it abuts its neighbour, interior to the real body; ∂H samples hydrostatic p
> there). The union has no internal seam → SBP cancels. **The prior ∂H "failure" was
> a seam artifact, NOT ∂H being wrong.**
>
> **THE FIX (proposed, confirm scope w/ user before the heavy swim):** union-∂H force
> density `f_i=-p ∂_iH(φ_union)` distributed to links by a PARTITION OF UNITY on the
> per-link SDF (softmin, like the existing `body_velocity_blend`), so `Σ_k F_k=union`
> (seam-free) AND each link gets a force+torque for MuJoCo. Validate: static depth-
> flat (§4 probe), `Σ_k F_k==union`, buoyancy/drop-sphere unchanged, THE swim gate.
> Harness knobs: KEFLOW_PHEAVI / KEFLOW_PHEAVI_UNION.
>
> ### SHIPPED + SWIM-VALIDATED (2026-06-20, same session): partition-of-unity union-∂H removes 59% of the overspeed.
> Built `two_phase.partial_heaviside_partition` (+ `partial_heaviside_blend_cells`,
> default 1.5): union-∂H force density `f_i=-p ∂_iH(φ_union)` split to links by a
> softmin partition of unity `w_b=softmax(-φ_b/τ)`, τ=blend·h; cropped to the union
> AABB; python path only (loud guard otherwise). Cheap gates PASS: static Fx
> −0.007→−0.014 (≈60× smaller than any anchor, ~depth-flat), `Σ_k F_k==union` exactly
> (net == union diagnostic), buoyancy Fz≈13.9 preserved, 9/9 two-phase unit tests.
> **DECISIVE SWIM GATE (submerged z=−0.07, uniform ρ, N=6000, frelax=0.5, same gait):**
> single-phase ref surge **−0.0797** (net 0.122) | n·δ anchor **−0.1316** (net 0.172,
> +65% overspeed) | **partition-∂H −0.1011 (net 0.139, +27%)** = **59% of the n·δ
> excess removed**, STABLE to completion. First readout fix to materially move the
> swim (prior: per-link ∂H 0.137→0.134, anchors ~0). RESIDUAL +27% remains (small
> static residual + likely the unsteady/added-mass terms — the handoff §2 path, now
> the next frontier with the static leak mostly handled). Harness: KEFLOW_PHEAVI_PART
> / DIAG_PHEAVI_PART (+ DIAG_PYTHON for the matched n·δ baseline). Code:
> two_phase_solver.py only (no core edits); cost = python path + per-step union-AABB
> B-stack (heavy at production grid → optimization TODO if adopted).
>
> ### RESOLUTION (2026-06-20): the gravity-independent residual was a REFERENCE ARTIFACT, not a two-phase bug.
> No-gravity python A/B (uniform ρ, submerged, net|v|): single-phase `zero_pressure_inside`
> (zeros p at sdf<0) **0.1217** | single-phase FULL band (`DIAG_NO_ZPI=1`) **0.1440** |
> two-phase n·δ FULL band **0.1455** | two-phase ∂H FULL band 0.1363. **Single-phase
> full-band ≈ two-phase n·δ full-band (~1%)** ⇒ with a CONSISTENT band treatment the
> solvers AGREE. The trusted 0.080/0.122 single-phase reference was biased LOW because
> its `zero_pressure_inside` truncates HALF the δ-support of its own n·δ band; two-phase
> keeps the full band (needed for buoyancy), so it only *looked* like overspeed vs the
> hobbled reference. Pressure A/B confirmed the dynamic field around the body is identical
> (user's "flat-in-x" = colour-scale effect). **Net: two-phase swim issue = (1) REAL
> gravity-coupled hydrostatic readout leak — partition-∂H fixes it — + (2) band-treatment
> mismatch vs a truncated reference (not a bug).** OPEN: which band treatment is physically
> correct (full Weymouth–Yue vs sdf<0 truncation) → validate vs drop-sphere Cd / robot
> 0.128; ∂H(0.136) vs full-band n·δ(0.145) ~6% gap. Knobs `DIAG_NO_ZPI`, `KEFLOW_DUMP_P`.
>
> ### BENCHMARK CLOSE-OUT (2026-06-20): FULL band is correct; sdf<0 truncation under-reads 40%.
> Standalone static submerged sphere, exact F_z=ρgV (new
> `validation/two_phase_3d/band_treatment_check.py`, 64³): **FULL-band n·δ 1.024× |
> sdf<0-truncated 0.604× | ∂H 1.025× | exact 1.000.** ⇒ full band correct (n·δ≈∂H, 2.4%
> over at 16 pts/D → 0 with refinement); single-phase `zero_pressure_inside` truncates
> the band and under-reads force 40% — THAT biased the single-phase swim reference LOW,
> so two-phase (full band) was never truly overspeeding. **Final picture: two-phase swim
> over-read = ONLY the gravity-coupled hydrostatic seam-leak → partition-∂H fixes it; the
> "gravity-independent residual" was a truncated-reference artifact, not a bug.** Fix for
> single-phase quantitative force: disable `zero_pressure_inside` (full band).
>
> ### LEAK = n·δ QUADRATURE (not zeroing); LAGRANGIAN is the backed-up cure (2026-06-21).
> `∂H` is non-standard (not WaterLily; no lit ref) → use with caution. Isolation
> (`band_treatment_check.py::leak_isolation`): tilted box + ANALYTIC hydrostatic p, NO
> solver/zeroing → full-band n·δ leaks F_x=+0.099 (true 0); ∂H=−0.00000. So the leak is
> intrinsic to n·δ on an asymmetric coarse body; `zero_pressure_inside` exonerated as its
> cause. Sphere buoyancy incl. Lagrangian: n·δ 1.024× | trunc 0.604× | ∂H 1.025× |
> **LAGRANGIAN 0.991×** | exact 1.0. ⇒ the Lagrangian watertight integral (Uhlmann 2005 /
> Kempe-Fröhlich 2012, `force_method="lagrangian"`, `Σ A·n=0`) is the LITERATURE-BACKED
> gauge-invariant cure — same leak-cancellation as ∂H, standard mechanism, best accuracy.
> NEXT: prefer Lagrangian over ∂H; (a) confirm its F_x≈0 on an asymmetric solver body;
> (b) solve the thin-coarse-eel Lagrangian robustness (the open obstacle).
>
> ---

**For:** a fresh-session agent. **Created:** 2026-06-20.
**One-line problem:** the two-phase (water+air) self-propelled 1guilla eel swims
faster than the single-phase solver and faster than the robot (≈0.128 m/s), and
the bias grows with depth. We have narrowed *where* it is not, and have a single
decisive experiment left to run before building any fix.

**Prereq reading (in order):**
1. memory `project_two_phase_force_gauge_leak` — the full history + the **CAPSTONE
   (2026-06-20)** section at the top of the body, which is the current summary.
2. memory `project_stage2_wb_gravity_two_phase_negative` — why the well-balanced-
   gravity (flow) fix was tried and reverted.
3. memory `project_surface_eel_overspeed`, `project_two_phase_force_methods`,
   `project_two_phase_poisson_bottleneck`.

---

## 0. The result you must internalise first (don't re-derive these the hard way)

**The FLOW is fine; the FORCE READOUT is the suspect. Do NOT chase the flow.**

- Single-phase is gravity-invariant (KE grav/nograv **1.00×**, with the shipped
  Stage-1 well-balanced gravity in `solver.py`).
- **Two-phase at UNIFORM density is gravity-invariant too — 1.02×, with NO fix.**
  The variable-coefficient projection already balances the static hydrostatic to
  machine precision even at the loose production cycle cap. (This killed the
  "under-resolved hydrostatic flow" theory and the Stage-2 well-balanced-gravity
  attempt — implemented, did not help, reverted. See
  `hydrostatic_gravity_stage2_handoff.md` OUTCOME box.)
- Two-phase real-air KE grav/nograv is **4.10×**, but that is the real air/water
  INTERFACE (free-surface/baroclinic physics), depth-flatish, and scales with the
  density contrast (1:1→1.02, 10:1→3.12, 80:1→3.58, 833:1→4.10). It is the
  *flow*, measured at force_scaling=0, and is NOT the swim-speed driver.

**The swim-speed driver is the force readout, and the leak is LINEAR in depth.**
Prescribed-motion probe (force_scaling=0 ⇒ identical body kinematics; log the
computed net hydro force): single-phase horizontal force is depth-FLAT
(Fx −5.48e-3 @ z=−0.07 → −5.79e-3 @ −0.13, ratio 1.06); **two-phase UNIFORM
density Fx +0.373 → +0.698, ratio 1.869 ≈ depth ratio 1.857** = linear in
hydrostatic pressure. Uniform density has no interface, so this is purely the
hydrostatic-pressure readout artifact `ρg·Σ z n_x δ`. It is genuinely numerical: a
fully-submerged body's horizontal force cannot depend on absolute depth.
(Self-propelled two-phase submerged is ~15% faster than single-phase, net 0.137 vs
0.119; real-air & uniform self-propelled BLOW UP from FSI, so the prescribed-motion
force probe — stable — is the measurement of record.)

**Every STATIC-pressure readout fix is a known dead end.** `gauge_anchor_forces`
(band-mean subtract, shipped, ON at the surface) removes most of the depth leak but
over-corrects; the exactly-gauge-invariant approach-C and the `∂H` weight were both
implemented, made the static force clean, and **failed the swim gate**
(0.137→0.134, vs single-phase 0.080). Critically: the minimal control-volume
readout `−∫_B ∇p dV` is the discrete divergence-theorem DUAL of `∂H`
(`Σ p ∂_iH = −Σ(∂_i p)H`), so it would fail the same way — **do not implement it
expecting a different result.**

## 1. The remaining hypothesis (unproven) and the decisive test

**Hypothesis:** the unconverged residual is in the UNSTEADY / added-mass momentum
terms — `∂_t∫_B ρu` and the convective flux `∮ ρu⊗u·n` — which no static-pressure
readout contains. Only a full control-volume momentum balance does.

**Run THIS before building anything (cheap, decisive):**
- Prescribed **steady glide** (no undulation, body translating at constant speed)
  vs **undulating**, at matched depth, two-phase, `gauge_anchor` ON, log the
  computed net horizontal force.
  - If the over-read survives WITHOUT motion → it is still a *pressure/static*
    artifact (and the static fixes' failure needs re-examining — maybe per-body
    anchoring or the interface, not unsteady terms).
  - If it only appears WITH motion → it is the unsteady/added-mass path, and a
    control-volume *momentum* readout (not `−∫∇p`) is the candidate worth building.
- To make a steady glide: in `_run_keflow.py` the body is `SpawnMode.TRANSVERSE`
  with the gait driving it. Either zero the gait amplitude (prescribe a constant
  drift) or add a prescribed constant-velocity translation with no joint actuation.
  Confirm the body actually moves (so the convective/unsteady terms are exercised)
  but does not undulate.

## 2. If the test says "unsteady": the candidate fix

A per-link **control-volume momentum balance** for the force:
`F_k = −d/dt ∫_{B_k} ρu dV − ∮_{∂B_k}(ρu⊗u + pI − τ)·n dS`, evaluated on a control
volume hugging each link (BDIM band). This is exactly gauge-invariant, keeps
buoyancy EMERGENT (it falls out of the pressure-flux term from the real `∇p`, not
an analytic Archimedes — the user explicitly rejected analytic buoyancy as
non-physical), and includes the added-mass term the static readout misses.
**Known wrinkle:** per-link decomposition at the seams of the multi-link eel
(overlapping/abutting BDIM bands) — use a partition-of-unity on the per-link mu0,
and validate that Σ_k F_k equals the whole-body force.

**Do NOT** use the analytic-Archimedes split (analytic vertical buoyancy + p_d
horizontal): the user correctly objected that it makes buoyancy a *prescribed*
force, not emergent — it would not track the real interface/waves and defeats the
purpose of two-phase. Buoyancy must stay emergent from the real fluid pressure.

## 3. Validation gates (in order; STOP if an early one fails)

1. **Prescribed-motion depth-linearity gone:** re-run the thrust-vs-depth probe
   (§4 harness); the new readout's horizontal force must be depth-FLAT (like
   single-phase), gauge_anchor OFF, for uniform density AND real air.
2. **Buoyancy preserved + emergent:** floating cylinder ≈0.91× Archimedes; 3-D
   drop-sphere Cz≈1; `single_sphere_drop_two_phase_3d` float/sink unchanged. The
   vertical force must still emerge from the fluid (not be imposed).
3. **Two-phase unit tests:** `lilytorch/src/test_two_phase.py` (9 tests) pass.
4. **THE swim gate:** submerged self-propelled two-phase must converge toward the
   single-phase reference **+0.080**; the SURFACE production case must stay
   ≈ robot **0.128** (today `gauge_anchor`=True gives +0.127 at the surface — do
   not regress it). Single-phase submerged (+0.080, net 0.119) is the trustworthy
   reference (no hydrostatic in its readout). Note the real-air SELF-PROPELLED runs
   are FSI-unstable; use `DIAG_COUPLING=aitken` (implicit) or a capped density
   ratio to get a stable swim, or measure thrust under prescribed motion.

## 4. Harness (all throwaway, already on disk)

In `lilytorch/examples/_1guillasim/experiments/`:
- **`_run_keflow.py`** — prescribed-motion (force_scaling=0, `SpawnMode.TRANSVERSE`)
  KE + force probe. Knobs: `KEFLOW_N`, `KEFLOW_Z` (depth), `KEFLOW_REALAIR`
  (0=uniform 1000, 1=real air), `KEFLOW_RHOAIR` (override air density, e.g. 12.5),
  `KEFLOW_NOGRAV`, `KEFLOW_SINGLEPHASE`, `KEFLOW_TAG`, `KEFLOW_PTOL/PMETHOD/PCYC`.
- **`_ke_ext.py`** — `KEFluidExtension`: writes `_submerged_diag/keflow_<tag>.csv`
  (fluid KE) AND `force_<tag>.csv` (net hydro Fx,Fy,Fz; raw, gauge_anchor as
  configured). Force read from `fs.friction_force_lin_{x,y,z}` + `pressure_force_{x,y,z}`.
- **`gen_config_submerged_diag.py`** — self-propelled swim (real coupling, SpeedLogger
  → `speed_<tag>.csv`). Knobs: `DIAG_N`, `DIAG_SPAWNZ`, `DIAG_SINGLEPHASE`(+`DIAG_GRAVITY`),
  `DIAG_RHO_AIR` (=1000 → uniform), `DIAG_TP_NOGRAVITY`, `DIAG_GAUGE` (0=anchor off),
  `DIAG_COUPLING=aitken` (implicit, for stability), `DIAG_FRELAX`, `DIAG_TAG`.
- **`_submerged_diag/analyze_speed.py`** — steady-state swim-speed reader (net|v|,
  path|v|, surge, lateral over the tail). forward = −x (link0=head at −x end).
- **`_submerged_diag/confirm_wb_twophase.py`** — standalone two-phase
  quiescent/static-body well-balanced check (shows legacy already balanced;
  evidence the flow is fine). Uses the production capped-multigrid+jacobi solver.

Result CSVs from this session in `_submerged_diag/`: `force_thr_*.csv`
(depth-linearity proof), `keflow_cmp_*.csv` (flow KE comparison),
`speed_dsp_*.csv` (self-propelled depth sweep). Clean up when done.

## 5. Hard constraints

- **No core edits.** Confine to `two_phase_solver.py` / `two_phase.py` /
  `test_two_phase.py` / validation examples — never `forces.py` / `body.py` /
  `solver.py` (memory `feedback-no-core-source-for-two-phase`). `git diff` on core
  must stay empty. If you believe a core edit is unavoidable, STOP and get explicit
  user authorization (Stage-1/rho_body were the only authorized exceptions).
- **Keep no-gravity + single-phase byte-identical.** All changes behind the
  two-phase + use_gravity path.
- **Buoyancy stays emergent** (user requirement) — no analytic/prescribed buoyancy.
- **The kernel path** keeps only the union SDF; a per-link readout needs
  `solver_method='python'` (loud error otherwise, as `∂H`/approach-C did).
- The real-case payoff is modest (~14% submerged speed / robot-match at surface);
  the cost/benefit of a full CV-momentum readout is a judgement call — confirm
  scope with the user before a large build.

## 6. First actions

1. Reproduce the depth-linearity (§4 thrust probe, uniform vs single-phase) to
   confirm nothing drifted.
2. Run the **steady-glide vs undulating** decisive test (§1).
3. Report the result to the user and decide together whether to build the
   control-volume momentum readout (§2) — do NOT build the minimal `−∫∇p` form
   (it is the rejected `∂H` in disguise).
