# Lilytorch — TODO

Memory vars: `sdf_val_{u,v,w}`, `{u,v,w,p}0`, `n{x,y,z}_{u,v,w}`, `body_{u,v,w}`,
`mu{0,1}_{u,v,w}`, `diff_{u,v,w}`.

---

# ═══════════════════════════════════════════════════════════
# NEEDS WORK
# ═══════════════════════════════════════════════════════════

# HIGH PRIORITY

<!-- HP5. **Conservative momentum transport CUDA kernel for two-phase solver** (2026-06-18).

  **Motivation (REVISED 2026-06-18).**  The non-conservative two-phase VOF
  transport was thought to be stable only to ~100:1 density ratio, but testing
  shows it is **stable at the physical 833:1 ratio** (rho_air=1.2) in kernel
  mode without consistent momentum — no blowup observed.  The density cap
  (rho_air=12.5, 80:1) was therefore an unnecessary precaution.  However,
  the harmonic-mean face density at the waterline (ρ_face ≈ 2.4 with rho_air=1.2,
  ~417× smaller than water) still suppresses pressure gradients → reduces wave
  resistance → surface-swimming robot is ~1.5× faster than single-phase
  underwater (0.22 vs 0.15 m/s; experiment 0.128 m/s).  Changing rho_air from
  12.5 to 1.2 does NOT change the speed bias (see HP5b), so the harmonic mean
  is not the dominant mechanism — the body-in-air effect (dorsal surface
  unloaded) is the leading hypothesis.  Conservative momentum transport
  (Nangia et al. 2019) may still reduce the bias by bounding air velocities,
  but the primary speed-bias mechanism is now thought to be structural
  (body-in-air), not a density-ratio instability.

  **Reference implementation.** `TwoPhaseSolver._consistent_advect()` in
  `lilytorch/src/two_phase_solver.py` — pure Python, already correct but
  requires `solver_method='python'` (~10× slower than kernel mode).

  **What to build.**
  * CUDA kernel for conservative mass-momentum advection (replaces Kernel A
    for two-phase): compute upwind mass flux F = u_adv * rho_upwind at faces,
    evolve density and momentum with the SAME flux, recover u = (ρu)/ρ.
  * Integrate into kernel pipeline: `solver.py` → `_fluid_step_kernel_3d`
    dispatch, `two_phase_solver.py` → `project()` coefficient rescale
    (unchanged), `finalize_step` — skip standalone VOF advection when alpha
    is synced from evolved density.
  * BDIM (Kernel B) and projection stay identical.
  * Validation: (a) rho_air=rho_water → match single-phase speed; (b) rho_air=1.2
    → verify still stable, compare speed to current non-conservative at rho_air=1.2
    (baseline: ~0.22 m/s).  **Note:** the speed bias (0.22 vs 0.15 single-phase)
    is now known to be robust to rho_air, so do NOT expect conservative transport
    alone to close the gap — see HP5b for the body-in-air diagnosis.

HP5b. **Diagnose two-phase surface speed bias** (2026-06-18 — investigation in progress).

  **Observation.** Two-phase surface swimming reaches ~0.22 m/s at steady state
  vs ~0.15 m/s for single-phase underwater (same grid, same gait, same robot).
  Experiment: 0.128 m/s.  The speed bias is **robust** — does NOT depend on:
  * `rho_air` (12.5 → 1.2): no change — rules out harmonic-mean density cap
    as the dominant mechanism.
  * `consistent_momentum` on/off: no change — rules out non-conservative
    advection instability as the cause.  (NB: config key typo
    `"converved_momentum"` in `gen_config_surface_pool.py` line 195;
    solver reads `"consistent_momentum"` — the feature was never active in
    these tests.  Fix the typo before re-testing.)
  * `air_transparent_body` on/off: no change — rules out BDIM velocity
    masking in air as the cause.
  * Alpha-weighting in forces (`p = p * alpha` in `_two_phase_forces`):
    breaks buoyancy (body sinks), cannot be used.

  **Leading hypothesis — body-in-air effect (structural, not a bug).**
  At steady state the robot floats at the waterline; dorsal body surface is
  in air (drag ≈ 0), ventral surface + tail are in water (full thrust).
  Same thrust, ~30% less drag → faster.  This is **physically correct**
  for a surface swimmer — the question is whether +47% (0.15→0.22) is the
  right magnitude, or whether additional mechanisms (free-surface pressure
  release reducing confinement) are also contributing. -->

  **Agent TODO — isolate the mechanism:**
  1. ~~Fix config typo~~ ✅ DONE (2026-06-18): `"converved_momentum"` →
     `"consistent_momentum"` in `gen_config_surface_pool.py` line 195
     (set to `False` — requires `solver_method='python'` when enabled).
  2. Run two-phase surface sim (current config, kernel mode, rho_air=1.2)
     and single-phase underwater sim.  At steady state, compare:
     * Per-link drag force (Fx) — is drag_two_phase ≈ 0.7 × drag_single?
       (expect ~30% reduction from dorsal surface unloading).
     * Per-link thrust force — should be similar (tail submerged in both).
     * If drag reduction >30%, additional confinement-release effect exists.
  3. Run the two-phase sim with WATERLINE raised high enough that the robot
     is fully submerged at steady state (e.g. WATERLINE = 0.10).  If speed
     drops to ~0.15 → body-in-air effect confirmed as the sole mechanism.
     (Note: buoyancy will push the robot up; the WATERLINE must be high
     enough that equilibrium depth is still fully submerged.)
  4. If none of the above resolves the bias, instrument `_two_phase_forces`
     to print per-step pressure-force components on water-only vs air-only
     band cells, looking for spurious air suction from gauge anchoring.

  **RESULT (2026-06-18 — body-in-air hypothesis REFUTED; gap is free-DOF
  dynamics).**  Ran the decisive submerged-vs-surface A/B (new harness
  `gen_config_submerged_diag.py` + `speed_logger.py`, output `_submerged_diag/`).
  Because the eel is lighter than water (floats), it was held under with
  `SpawnMode.TRANSVERSE0` (slide-x, slide-y -> surge/sway free, heave/roll/
  pitch/yaw LOCKED); the free-yaw `TRANSVERSE` variant + explicit coupling
  diverged when fully submerged (added-mass blow-up, `mjWARN_BADQACC`), cured
  with `force_relaxation=0.5` (cycle-mean force preserved -> speed unbiased).
  Three matched runs (frelax=0.5, spawn x=4.75, swim speed = net head
  translation / T):
      two-phase SURFACE   (z=-0.0115, dorsal in air)  : 0.184 m/s
      two-phase SUBMERGED (z=-0.07, all in water)     : 0.174 m/s
      single-phase SUBMERGED (z=-0.07, infinite water): 0.164 m/s
  All within ~6-12%.  Conclusions:
  * **Body-in-air unloading is only ~6%** (surface 0.184 vs submerged 0.174) —
    NOT the ~45% HP5b assumed.  **Leading hypothesis refuted.**
  * **two-phase submerged ~= single-phase submerged** (0.174 vs 0.164) — the
    two-phase momentum transport does NOT over-thrust vs single-phase for a
    submerged body (single-phase steady-window is even marginally higher).
    **=> HP5's conservative-momentum kernel will NOT close the speed bias**
    (its value is stability/perf only).
  * The big original 47% gap (free two-phase 0.22 vs free single 0.15) is NOT
    reproduced once DOFs are locked (~12% left) => it lives in the **free
    vertical/rotational surface dynamics** (heave/roll/pitch/yaw — body riding
    the interface), the same DOFs the lock suppresses.
  **=> NEXT (to confirm):** rerun the SAME A/B with `SpawnMode.FREE` (or
  TRANSVERSE, yaw free) under implicit Aitken coupling (cures the added-mass
  blow-up the explicit path hits when submerged), reproduce the 0.22 vs 0.15
  gap, then lock DOFs one at a time (heave only, then +roll/pitch, then +yaw)
  to find which freedom carries the speed-up.  Caveat: locking heave removes
  the body's surface-riding, so the ~6% body-in-air number is for a FIXED-draft
  body, not a freely-bobbing one.

  **RESULT-2 (2026-06-18 — WHY single-fluid-underwater != two-fluid-underwater:
  the gauge_anchor_forces fix OVER-corrects, leaving a residual hydrostatic
  force-readout artifact that biases the self-propelled swim speed).** A
  fully-controlled isolating ladder (ALL `SpawnMode.TRANSVERSE`, frelax=0.5,
  z=-0.07, same gait/BDIM; only one knob changed per run; new diag knobs
  DIAG_SINGLEPHASE/GRAVITY/RHO_AIR/TP_NOGRAVITY/GAUGE/FREESLIP_TOP), forward =
  -dx/dt slope:
      single-phase  (no gravity)                 +0.075   <- clean reference
      single-phase + gravity (no gauge)          -0.042   REVERSED
      two-phase uniform-rho + gravity, gauge OFF  -0.092   REVERSED
      two-phase uniform-rho + gravity, gauge ON   +0.152
      two-phase uniform-rho, NO gravity (gauge)   +0.103
      two-phase REAL air(1.2) + gravity, gauge ON +0.115
  Decisive: toggling ONLY the gauge on the SAME flow swings -0.092 -> +0.152
  (sign-flip).  Decomposition of the single(0.075)->two-phase-real(0.115) gap:
  +0.05 gravity/hydrostatic force-integral leak (the coarse-body `Sum p n delta`
  doesn't cancel the large hydrostatic baseline; gauge mitigates but OVER-
  corrects -> residual), +0.028 two-phase code path (within TRANSVERSE noise),
  -0.037 the ONLY physical effect = wave-making drag from the interface ~7 cells
  above (correctly a slowdown).  Confinement/rigid-lid REFUTED (free-slip top:
  0.075->0.075).  PHYSICS: gravity/hydrostatic must give ZERO horizontal force on
  a heave-locked body, so the +0.05/+0.028 are NUMERICAL force-readout artifacts;
  the gauge fix is out-of-place (flow is fine, only the force readout is approx-
  corrected).  => For a SUBMERGED body single-phase is the trustworthy estimate;
  two-phase submerged is biased HIGH.  Two-phase is needed only NEAR the surface
  (real wave drag) and there carries the residual artifact.  See
  [[project_two_phase_force_gauge_leak]], [[project_surface_eel_overspeed]].
  **=> ACTIONABLE:** improve the eulerian band force to be discretely gauge-
  invariant (per-body / depth-varying anchor, or a control-volume momentum-flux
  integral) to kill the residual hydrostatic leak -> would make two-phase
  submerged forces match single-phase.  **Promoted to its own item HP6.**

HP6. **Discretely gauge-invariant two-phase band force** (the live force-readout
  fix; was buried in HP5b RESULT-2).  **ACTIVE WORK** — latest commit `8196127`
  ("gauge-invariant force handoff + submerged diag harness") + the fresh
  `_submerged_diag/force_dec_*.csv` come from this line.

  **Problem.** The Eulerian band force `ρg·Σ z nₓ δ` over-reads a horizontal
  force that is LINEAR in depth on a body whose true horizontal force is 0
  (a submerged body sees only the *vertical* hydrostatic gradient).  The shipped
  `gauge_anchor_forces` band-mean subtraction OVER-corrects to a depth-linear
  residual of opposite sign; the opt-in `gauge_anchor_per_body` was implemented
  and tested and ALSO does not remove the depth-linearity.

  **Decisive new result (2026-06-20).** The leak is **STATIC, not unsteady** —
  fully present on a zero-gait non-moving body; undulation adds ~−0.02 to the
  mean Fx.  Therefore the control-volume *momentum* readout (§2 of the old brief)
  is the WRONG build (it targets the added-mass/unsteady path the data exonerates,
  and is the SBP-dual of the already-rejected ∂H).  Open tension to reconcile:
  this seemingly contradicts `project_two_phase_force_gauge_leak` (∂H zeroed the
  static Fx yet barely moved the self-propelled swim) — either ∂H didn't truly
  zero the force on the running multi-link union SDF (per-link seams), or the
  swim-speed bias is a different quantity than the static leak.  **A direction
  decision is required before any further build.**

  **HANDOFF BRIEF (current):** `milestones/two_phase_force_readout_next_agent.md`
  (problem, evidence, the 2026-06-20 static verdict, dead-ends, harness pointers).
  NOTE: the old `milestones/gauge_invariant_force_handoff.md` referenced by HP5b
  was DELETED/superseded by this brief.
  Constraint: confine to `two_phase*.py` / validation (no `forces.py`/`solver.py`/
  `body.py` edits — see `feedback-no-core-source-for-two-phase`).

HP4. ~~**Stabilise the one-fluid GFM free surface** (handoff 2026-06-17).~~
  **CANCELLED / REMOVED 2026-06-23.** The one-fluid free-surface method
  (`free_surface_solver.py`, `poisson_gfm.py`, `validation/free_surface/`,
  `run_1guilla_fs.py`, the BDIMhandler `solver.free_surface` branch, and the
  associated docs sections) has been deleted from the codebase as outdated. The
  GFM moving-interface approach never converged under refinement (see the
  session log below for the conclusive analysis); quantitative free-surface
  work uses single-phase underwater, and `TwoPhaseSolver` remains for
  qualitative free-surface cases. The notes below are retained as a historical
  record of why the GFM route was abandoned.

  **One-line status.** The two-phase surface model is sound for *qualitative*
  free-surface work but over-predicts swim speed ~2× from under-resolved wave
  drag; **single-phase underwater already matches the robot (0.128 m/s)** and is
  the tool for quantitative speed. The one-fluid free surface is the right
  structural fix (removes air noise, statics validated exactly) — it just needs
  the explicit surface mode stabilized, which is this HP4 task.

  **Progress (2026-06-17).** Steps (1)–(3) DONE at proof-of-concept level:
  * (1) **Implicit free-surface coupling** — hydrostatic-pressure correction
    (η''(x) term in Poisson RHS) stabilises the standing wave at RES=75:
    wave oscillates with amplitude ~0.02 m, no blowup (|u|max ~0.67 m/s).
    Gaussian smoothing of η(x) added to suppress stair-step noise.
  * (2) **Non-diffusive advection** — self-contained Weymouth–Yue VOF
    implemented for true staggered MAC grid (interior-only arrays). Works
    correctly; column-height transport limited by directional-splitting
    cancellation in divergence-free regions (known WY property).
  * (3) **3‑D extension** — `poisson_gfm.py` extended to 3‑D
    (`gfm_solve_cg_3d`, `gfm_grad_3d`, `level_set_height_3d`,
    `_div_of_faces_3d`).  3‑D hydrostatic validation PASSES (p error ~3.5%
    from half-cell GFM discretization, |u|~1e-17, air p=0).  **NOT yet
    wired into `FreeSurfaceSolver.project`** (still uses staircase mask).
  * Remaining for step (4): period tuning (+22% → goal <5% — root cause is
    GFM discretization of free-surface BC, not source-term calibration),
    wire GFM into solver, 3‑D standing wave, surface-pool eel.

  **Progress (2026-06-17, session 2 — period root cause FOUND + halved).**
  Root cause of the +24% period was located: it is NOT the GFM BC and NOT the
  η'' source-term scaling. The `ETAXX_SCALE` knob does NOT move the period
  (1.0→+22%, 1.45→+24%, *worse*), proving the hydrostatic source is not the
  restoring mechanism. The real cause is the **Weymouth–Yue VOF under-
  transporting the surface**: with the narrow-band velocity zeroing,
  `explicit_wy` now goes fully STABLE but the surface FREEZES (h flat,
  amplitude=0, dα=0) — the WY directional-split flux cancellation in
  divergence-free columns means the surface barely advances → wave appears
  slow → long period.
  * **New `height_function` method** (in `gfm_wave_standalone.py`, selectable
    via `FS_METHOD=height_function`): single-valued surface advanced by the
    exact kinematic eq `Dh/Dt = v_surf − u_surf·∂h/∂x` (v_surf interpolated at
    y=h(x) from the y-faces); alpha/phi rebuilt from h each step. Removes the
    VOF freeze. Plus: (a) exact volume conservation (uniform shift, closed
    box) kills the spurious downward drift; (b) biharmonic (∝k⁴)
    hyperviscosity `h −= (1/16)·∂⁴h/∂x⁴` suppresses the height-function
    odd-even (2-cell) sawtooth — annihilates the sawtooth in one step yet
    damps the 75-cell wave only ~0.3% over the run (a per-step Gaussian
    compounds and kills the wave — rejected).
  * **RESULT (RES=75):** `height_function` → **T_sim=0.646 vs 0.574, +12.5%**
    (was +24%), **amplitude retained 73%** (was 65%), stable. Period error
    HALVED. NOTE: |u|max still slowly creeps 0.26→0.71 over 3 periods — the
    biharmonic only *contains* a residual instability, doesn't cure it.
  * **OPEN (user flagged: filters are non-physical).** The 2-cell sawtooth is
    a genuine odd-even instability of the EXPLICIT height-function coupling
    (surface advanced by pointwise v_surf then fed back through GFM θ). The
    physically-clean cure that removes the NEED for any filter is implicit/
    semi-implicit surface coupling (couple the kinematic + pressure solve so
    the fast mode can't go odd-even) — proper next step, more work. The
    biharmonic is the least-intrusive stopgap (k⁴-targeted, ~0.3% on the
    physical wave). Remaining +12.5% period is then likely the half-cell GFM
    BC discretization (cf. the 3.5% hydrostatic p error) + the residual
    surface dissipation.
  * Diagnostic: period detection now linearly detrends `hs` before zero-
    crossings (a residual drift otherwise reports nan).

  **Progress (2026-06-17, session 3 — IMPLICIT coupling: FILTER REMOVED).**
  User flagged the biharmonic as non-physical and chose to build the implicit
  coupling. DONE: new default method `implicit_robin` in
  `gfm_wave_standalone.py` (`FS_METHOD=implicit_robin`). The dynamic+kinematic
  free-surface conditions are coupled INTO the pressure solve, turning the GFM
  Dirichlet p=0 into a **Robin BC** at each column's surface y-face:
  `p + g·dt²·∂p/∂z = ρ·g·η_pred`, η_pred = (h−H_ref) + dt·w* (DISPLACEMENT,
  not absolute height — the first cut blew up because I used absolute h → ~1900
  Pa source). Implemented as a delta on top of the existing Dirichlet GFM
  operator (`_apply_A_2d`): extra +diagonal at (i,jt) keeps it SPD, plus a
  known source moved to the RHS; surface y-face gradient overridden in the
  velocity correction. Self-contained CG. **No CUDA/solver changes.**
  * **RESULT (RES=75): STABLE WITH NO FILTER, amp 81% retained** (best of all
    methods), mean level correct, |u| saturates ~1.75 (bounded, not growing).
    Period **+62%** (pure form) — restoring under-resolved by the half-cell
    GFM surface BC (the dynamic-pressure BC carries weaker restoring than a
    body force; cf. explicit body-force height_function was +12.5%).
  * **θ-split period tuning** (`FS_GBODY`, default 0): split gravity into an
    implicit-BC part and an explicit body-force fraction. `FS_GBODY=0.07` →
    **T+15%, amp 84%, still filter-free**; but the knob is STIFF/nonlinear
    (0.12 → −55% overshoot, amp 40%), so it is a tuning parameter, not a clean
    cure. Robust tuning-free default is FS_GBODY=0 (+62%).
  * **NET:** the artificial filter is GONE — implicit coupling gives filter-
    free stability with the best amplitude retention (81–84%). Remaining
    period error (+15% tuned / +62% pure) is now squarely the **half-cell GFM
    free-surface-BC discretization** (consistent with the 3.5% hydrostatic p
    error). That — a higher-order/consistent GFM surface gradient — is the
    real next lever for <5% period, NOT more source/knob calibration.
  * Comparison table of all methods is in the file's METHOD-selector header.

  **Progress (2026-06-18, session 4 — CONVERGENCE TEST: implicit_robin is
  resolution-fragile, NOT yet convergent).** Added `FS_RES` env knob and swept
  the pure `implicit_robin` period vs grid:
    RES=75  → T+62%, amp 81%   (the earlier "good" point)
    RES=110 → collapses (amp 7%, period nan)
    RES=150 → T−69%, amp 23%
  The period does NOT converge and amplitude DECAYS under refinement → the
  +62% at RES=75 is **not** a half-cell discretization floor; it sat in a lucky
  stability window. **Root cause (analysed):** the implicit coupling strength
  `β = g·dt²/(θh) ∝ h` (because dt ∝ h here) → the implicit ∂p/∂z term
  VANISHES as the grid refines, so the single-face Robin stops suppressing the
  column odd-even mode and the restoring reverts to effectively explicit-in-η
  → decay/instability. The fully-implicit derivation itself is correct
  (`p + g·dt²·∂p/∂z = ρg·η_pred`, η_pred = η^n + dt·w*, verified by
  back-substituting w^{n+1}=w*−c∂p/∂z); the FLAW is imposing it at only ONE
  y-face per column — on a sloped surface the interface-cut **x-faces** still
  get the Dirichlet p=0 (not p=ρgη), so the implicit BC is incomplete and the
  scheme is inconsistent at finite slope.
  **→ CONCRETE NEXT STEP (convergent fix):** generalise the GFM ghost to impose
  a NONZERO interface pressure p_I = ρg·η_disp(x_interface) on **all**
  interface-cut faces (both x and y), treated implicitly (each cut face carries
  its own β and source). For the single-valued height surface η is known per
  column, so p_I at a y-face = ρg·η_i and at an x-face = ρg·(η_i+η_{i+1})/2.
  This is the proper 2-D implicit GFM free surface; expect it to converge.
  Re-run the FS_RES sweep as the acceptance test BEFORE wiring into the solver.

  **Progress (2026-06-18, session 5 — all-cut-faces RULED OUT; true root cause
  is the coupling ORDER).** Implemented `implicit_robin_full` (method in
  `gfm_wave_standalone.py`): the implicit Robin p=ρgη is imposed on EVERY
  interface-cut face (x and y), each with its own θ/β/source (diagonal per cut
  face → still SPD). FS_RES sweep: RES75 T+82%/amp43%, RES110 −46%/amp5%,
  RES150 −16%/amp78% — STILL non-convergent. So the slope/x-face Dirichlet
  leak was NOT the cause.
  **TRUE ROOT CAUSE (conclusive):** the GFM-Robin couples the surface
  implicitly only through the FIRST normal derivative `∂p/∂z`, so the
  stabilising coefficient `β = g·dt²/(θh) ∝ h` (because dt ∝ h here) and the
  implicit term VANISHES as h→0. A refinement-robust implicit free surface
  (Casulli-style semi-implicit) instead couples the surface elevation through
  a HELMHOLTZ / second-derivative term `g·dt²·∇²η`, coefficient `g·dt²/h² ∝
  const` when dt ∝ h — one order of h STRONGER. No GFM-Robin patch (single-
  face OR all-faces) can be refinement-robust; the coupling is structurally
  one order too weak. **Both `implicit_robin` and `implicit_robin_full` are
  therefore dead-ends for production** (they only "work" in a luck-of-dt window
  at one resolution).
  **→ CORRECTED NEXT STEP:** build a Casulli-type semi-implicit free surface —
  treat the surface-elevation gradient implicitly in momentum AND the velocity
  divergence implicitly in the kinematic equation, yielding a Helmholtz solve
  for η^{n+1} (coefficient g·dt², does NOT vanish). For THIS depth (kH≈2.14,
  not shallow) the depth-AVERAGED Casulli would give the wrong dispersion
  (ω²=gHk² vs gk·tanh kH), so it must be the FULL 2-D (x–z) version: couple
  η^{n+1} into the full pressure Poisson via the Helmholtz surface term while
  keeping the vertical structure (this is how 3-D ocean codes do it). Acceptance
  test = the FS_RES sweep must show a CONVERGING period. This is real work, not
  a patch — best started with fresh budget.

  **Progress (2026-06-18, session 6 — CONVERGENT implicit free surface PASSES
  (linear).** Before reaching for Casulli I did the von-Neumann analysis of the
  Robin BC on a clean FIXED linear domain: η^{n+1}=η_pred/(1+μ),
  μ=g·dt²·k·tanh(kH); discrete map |λ|²=1/(1+μ)<1 (stable, decays
  ~1/√(1+μ)/step), phase→√μ ⇒ ω_num→√(gk·tanh kH) EXACT as μ→0. At RES=75
  μ≈3e-4 ⇒ near-perfect. **CONCLUSION: the Robin concept was never the problem
  — the non-convergence was entirely the moving-interface GFM machinery
  (θ-clamp, narrow-band zeroing, VOF-free kinematic, volume shifts).**
  * **NEW: `lilytorch/validation/free_surface/linear_wave_robin.py`** — clean
    LINEARIZED standing wave, fixed domain (water 0..H, surface linearised at
    z=H), MAC grid, implicit Robin top BC, NO GFM / NO VOF / NO narrow-band /
    NO filter / NO body-force fudge.
  * **One real bug found:** the surface cell's continuity RHS must include the
    PREDICTOR surface-face flux vtop* (`rhs[:,Nz-1] -= vtop*/(c·h)`); it is
    omitted by the interior-only `_div_of_faces` and is zero only while
    vtop≈0, so leaving it out blows up *after* the surface starts moving
    (clean start → accelerating blow-up). With it added: perfect.
  * **RESULT — FS_RES convergence sweep (acceptance test):**
      RES=50  T err +0.2%, amp 86%
      RES=75  T err −0.0%, amp 90%
      RES=110 T err +0.0%, amp 94%
      RES=150 T err +0.1%, amp 95%
    Period EXACT at all resolutions; amplitude retention IMPROVES with
    refinement (→100% as μ→0) — i.e. it CONVERGES. This is the property both
    `implicit_robin` and `implicit_robin_full` failed. **The project's
    load-bearing standalone standing-wave acceptance test now PASSES (linear,
    filter-free).** No Casulli reformulation needed — the implicit Robin is the
    correct, convergent scheme; it just has to be applied on a clean MAC grid.
  ── REMAINING (the implicit core is now proven): port this clean implicit-Robin
     surface BC onto the MOVING interface — rebuild the moving-interface step
     using the lesson above (include the predictor surface-flux term; avoid the
     θ-clamp/narrow-band artifacts that broke the GFM probes), re-pass the
     FS_RES sweep with a moving interface + finite amplitude; THEN wire into
     `FreeSurfaceSolver.project` (replace staircase mask), 3-D standing wave,
     surface-pool eel.

  **Progress (2026-06-18, session 7 — moving-interface port ATTEMPTED,
  BLOCKED by GFM).**  Two attempts to port the clean Robin BC to the moving
  interface by weakening/removing the GFM artifacts:
  * `implicit_robin_clean` (θ floor 0.005, NO narrow-band) → **catastrophic
    blowup** (|u|→2700 m/s).  The narrow-band zeroing IS load-bearing for
    the moving-interface case — without it, air cells accumulate unbounded
    GFM velocities.  Unlike `linear_wave_robin.py` (which has NO air cells),
    the moving-interface domain has air above the surface that must be
    controlled.
  * `implicit_robin_clean` v2 (θ floor 0.005, narrow-band KEPT) → **still
    blows up** (|u|→4400 m/s).  The milder θ floor increases the Robin
    diagonal by 63× (β/(1+β))/(θ·h²) ∝ 1/θ², making the operator stiff
    enough to destabilise the CG + time integration.  The GFM θ-clamp at
    _TH_MIN=0.05 is load-bearing for the operator conditioning.
  * **CONCLUSION:** both GFM artifacts (θ-clamp, narrow-band zeroing) are
    NECESSARY for the moving-interface case.  They cannot be simply
    "removed" or "weakened."  The convergent Robin BC can only be applied on
    a clean MAC grid without air cells (as in `linear_wave_robin.py`).
  * **→ PATH FORWARD:** the moving-interface port needs a FUNDAMENTALLY
    DIFFERENT air treatment — not GFM ghost cells + narrow-band zeroing, but
    a one-fluid approach where the air is NOT a computational domain (e.g.
    cut-cell, embedded boundary, or a level-set method that only solves in
    Ω_water).  This is more work than a patch; best scoped as a separate
    milestone with fresh budget.  The LINEAR implicit-Robin acceptance test
    IS the validated core — it just needs the right moving-interface
    machinery wrapped around it.
  * **FS_RES verification re-run (2026-06-18):** linear_wave_robin.py PASSES
    at all resolutions (RES=50/75/110/150, T err ≤0.2%).

  **Progress (2026-06-18, session 8 — GFM gradient wired into solver).**
  * `FreeSurfaceSolver.gradient()` overridden to return GFM sub-cell gradient
    (padded to full-grid shape) when `free_surface.use_gfm_gradient=True`.
    The level set φ is built from alpha each `project()`.  This gives a more
    accurate velocity correction at the interface while the pressure solve
    still uses the staircase `dirichlet_mask` (multigrid/MGCG unchanged).
    Config key: `free_surface.use_gfm_gradient` (default False).
  * **GFM pressure solve NOT wired** — the standalone GFM CG solver cannot
    replace multigrid/MGCG without significant refactoring, and the GFM-based
    implicit-Robin approaches (`implicit_robin`, `implicit_robin_full`,
    `implicit_robin_clean`) are all resolution-fragile or unstable on the
    moving interface.  The GFM machinery (θ-clamp, narrow-band zeroing) is
    load-bearing for stability but breaks convergence under refinement.
  * **→ REALISTIC PATH FORWARD:** the convergent implicit-Robin BC is proven
    on a clean MAC grid (`linear_wave_robin.py`).  Porting it to the moving
    interface requires a non-GFM air treatment (cut-cell, embedded boundary,
    or one-fluid solve-in-Ω_water-only).  This is scoped as future work.
    The GFM gradient override is a pragmatic partial improvement available now.

  **Progress (2026-06-18, session 10 — exhaustive moving-interface attempts
  CONCLUDE: GFM-based approaches cannot converge).**
  * `robin_gfm_free` + staircase x-faces: |u|→10^135 (blows up).
  * `implicit_robin` + physical-distance narrow band: RES=75 T+62%, RES=110
    collapses, RES=150 T−69% — still non-convergent.
  * **FINAL ASSESSMENT:** the implicit Robin BC is proven correct and
    convergent on a clean MAC grid (`linear_wave_robin.py`, T err ≤0.2% at
    all RES).  ALL GFM-based moving-interface implementations
    (`implicit_robin`, `implicit_robin_full`, `implicit_robin_clean`,
    `robin_gfm_free`) fail to converge under refinement.  The GFM machinery
    (θ-clamp, narrow-band zeroing, GFM gradient at water-air faces) is
    simultaneously load-bearing for stability AND convergence-breaking.
  * **→ ONLY VIABLE PATH:** a one-fluid solve-in-Ω_water-only approach
    (cut-cell, embedded boundary, or σ-coordinate transform) that avoids
    air cells entirely — analogous to `linear_wave_robin.py` but with a
    moving top boundary.  This requires a new solver architecture, not a
    patch on the existing GFM.  Best scoped as a separate milestone with
    fresh budget.
  * The GFM gradient override in `FreeSurfaceSolver` (session 8) remains a
    pragmatic partial improvement for production (better velocity correction
    at the interface with the existing staircase pressure solve).

  ── GOAL ──
  A free-surface solver that captures surface WAVES so the surface-swimming
  eel's wave-making drag is resolved. The two-phase VOF over-predicts swim
  speed ~2× because that drag is under-resolved/smeared (robot 0.128 m/s vs
  sim ~0.20–0.26; ruled out viewer, force-readout, air-density). One-fluid
  (air = constant-p void) ALSO removes the air-force noise structurally.

  ── WHAT EXISTS (the two-phase solver is untouched) ──
  * `lilytorch/src/free_surface_solver.py` — `FreeSurfaceSolver(TwoPhaseSolver)`:
    uniform water density, `p=0` in air (staircase mask via the `dirichlet_mask`
    hook in `poisson_mult.py`), velocity-extend into air, gauge-anchor +
    air-transparent-body OFF.
  * `lilytorch/src/poisson_gfm.py` — self-contained ghost-fluid Poisson (2‑D):
    height level-set `φ = y − h(x)`, θ-scaled ghost gradient placing `p=0` on
    the *exact* interface, Jacobi-CG, operator `A = div(gfm_grad)`.  NOT yet
    wired into `FreeSurfaceSolver.project` (that still uses the staircase mask).
  * `lilytorch/validation/free_surface/run_hydrostatic_fs.py` — **PASSES:**
    water hydrostatic 0.00%, |u| ~ 1e−7, air p = 0.
  * `lilytorch/validation/free_surface/gfm_wave_standalone.py` — self-contained
    2‑D MAC standing-wave probe that **exposes the failure**.

  ── PRECISE FAILURE MODE ──
  **Statics work. Waves don't.** The explicit free-surface mode is UNSTABLE:
  * freeze the surface face → stable but amplitude = 0 (no wave);
  * let the surface move → |u| grows **0.007 → 3 m/s** (three orders of
    magnitude), surface collapses, smaller dt only delays it.
  This is the **classic explicit free-surface instability** — the fast
  gravity-wave mode (ω² = gk tanh kH) is treated explicitly and its CFL
  constraint is not the advective one but a much tighter surface-gravity one.
  This is exactly why the old GFM was retired.

  ── ORDERED NEXT STEPS ──
  **(1) Implicit free-surface coupling** — treat the surface pressure/gravity
      implicitly so the fast gravity-wave mode can't blow up (the load-bearing
      fix).  This means the Poisson RHS must include the gravity contribution
      at the *future* surface position, or equivalently the surface kinematic
      BC (Dh/Dt = v_surface) must be coupled to the pressure solve.
  **(2) Non-diffusive interface advection + clean reinitialised level set** —
      the standalone reuses 1st-order upwind α → φ noisy → noisy θ → noise
      feeds the instability growth.  Reuse the Weymouth-Yue VOF
      (`two_phase.py`) or implement a proper level-set method with HJ-WENO
      advection and Sussman reinitialisation.
  **(3) 3‑D + BDIM integration** — extend `poisson_gfm.py` to 3‑D, wire the
      GFM grad/solve into `FreeSurfaceSolver.project` (replacing the current
      staircase `p=0` mask).  Re-validate hydrostatic first.
  **(4) Validate standing-wave period then eel** — standalone standing wave
      (period + amplitude retention) BEFORE any MAC/BDIM integration.  Target:
      stable oscillation at the analytic period ω² = gk tanh kH with ~100%
      amplitude over several periods.  Then a surface-pool eel config using
      `FreeSurfaceSolver` vs two-phase vs robot 0.128 m/s.

  **Validation order is load-bearing:** standalone standing wave MUST pass
  before any 3‑D or BDIM integration.

---

# BUGS

BG6. **Multigrid residual restriction over-scaled (LOW priority, DEFER — regression risk)**
  — `_restrict_residual_2d/3d` in `poisson_mult.py` sums over fine cells without
  normalization (×4 in 2D, ×8 in 3D).  WaterLily uses `0.5 × sum` (×2/×4); the face-
  coefficient restriction uses `0.5 × sum` (×1/×2).  Lilytorch residual:face ratio = 4,
  WaterLily ratio = 2.  **Investigated 2026-06-11:** analysis confirms 2× discrepancy vs
  WaterLily; solver converges in all tests anyway — the over-scaling is absorbed by the
  post-smooth.  The potential fix is `* 0.5` on `_restrict_residual_*` (not `* 0.25`).
  Deferred: changing normalization on a working solver needs full regression coverage
  before merging.

---

# MEMORY / PERF (optimize_speed_memory branch)

Target: ~8 GiB peak alloc on 3D runs. Do in sequence; remeasure after each stage.

> **MEASURED (2026-06-05, `bench_memory.py`, 448×224×224):** baseline peak **5.749 GiB**
> on the standalone **PYTHON path**, located in **`_recompute_mu_normals`** (mu0/mu1 +
> normals build) — NOT advection (~3.36) and NOT the multigrid solve (~3.58). BUT the
> python path's mu/normals build does **not exist in kernel mode** (Kernel B computes
> them in registers, no buffers), so this peak is python-path-only and is **not
> representative of kernel-mode production**, which is what the T2/T3 items target.
> ⇒ The memory work must be re-baselined on the **kernel path** (needs BDIMhandler/FARMS)
> before T3a/T3b/T2a can be prioritised by measurement. The python-path standalone bench
> proved the wrong path for these items. See `lilytorch/validation/cost_analysis/MEMORY_BASELINE.md`.

MP6. **T3b** Preallocate the V-cycle coarse-level pyramid at `__init__` instead of
  `torch.zeros` inside the recursion. ~0.5-1 GiB transient. (Python-path multigrid solve
  stayed ≤3.58 GiB; re-baseline on kernel path before judging peak benefit.)
MP8. **T2b** Dirty-AABB-sized Kernel-A temps (`sdf_*_tmp`, `b*_tmp`: full-grid → AABB+halo).
  Needs `streaming_sdf.cu` changes; no peak movement until T2a.
MP9. **T2c** Two-pass Kernel B for `primes` elimination (write to AABB scratch, copy back).
  ~1.5 GiB; no peak movement until T2a.
MP10. ~~**T2d — fused CUDA kernel for SCALAR advection**~~ ✅ DONE (2026-06-23,
  as the W&Y `cvof_sweep` kernel — see below). **The original framing was stale:**
  it assumed `advection.advect_scalar` carried the two-phase α-transport, but
  (a) `advect_scalar` has **zero live callers** — its only consumer was the
  free-surface level-set transport, deleted with HP4; and (b) the actual VOF
  α-transport is `TwoPhase._cvof_sweep` (`two_phase.py`), the **Weymouth & Yue**
  conservative scheme, which `advect_flux_add` **cannot** express (W&Y is a
  sequential operator-split sweep with a divergence-correction term
  `a_i·(u_R−u_L)` and a bounded donor face value, not a sum-over-d accumulation of
  QUICK/vanLeer fluxes). So reusing `advect_flux_add` was a dead end; instead a
  **dedicated W&Y CUDA kernel** was written.
  * **NEW op `lilytorch_kernels::cvof_sweep`** in
    `lilytorch/src/kernels/csrc/cuda/cvof_sweep.cu` (schema in `ops.cpp`): one
    launch per (sweep, direction) replacing the ~8 full-grid temporaries (3
    edge-clamped shifts, 2 limited slopes, 2 donor faces, the flux tensor) of the
    Python sweep; computes `out[i] = a[i] + cfl·(F(i)−F(i+1)+a[i]·(u[i+1]−u[i]))`
    with the W&Y van-Leer-limited Courant-corrected donor face, all in registers.
    Edge-clamp (Neumann) neighbour reads + explicit strides (velocity components
    are strided `_vel` row views). 2-D + 3-D.
  * **Wired** into `TwoPhase._cvof_sweep` (`two_phase.py`): dispatches to the
    kernel on CUDA when the extension exports the op (cached
    `_cvof_kernel_available()` probe), else falls back to the renamed
    `_cvof_sweep_python` oracle (CPU path / un-built extension). **No core-source
    edits** (confined to `two_phase.py` per `feedback-no-core-source-for-two-phase`).
  * **Parity** (`test_two_phase.py`, new `test_cvof_sweep_kernel_parity_*`):
    kernel vs `_cvof_sweep_python` on identical CUDA tensors — **rel 0.0 (2-D
    f32+f64, contiguous AND strided-velocity)**, 1.1e-16 (3-D f64), 6e-8 (3-D f32,
    FMA). End-to-end `tp.advect` 100-step boundedness + mass-conservation test on
    CUDA also passes. **Speedup 4.5–5.6×** at typical sizes (2D-256, 3D-64/128;
    1.3× at 2D-512), measured incl. the shared clone+pad overhead.
  * NOTE: `advect_scalar` left in place (harmless dead code; a future passive-
    scalar/dye/level-set consumer could be CUDA-accelerated via `advect_flux_add`
    cheaply, but there is none today). The variable-coeff Poisson still dominates
    the two-phase step at ~75% (see `project_two_phase_poisson_bottleneck`); this
    removes the α-transport slice of the ~6% Python cost.

---

# 2D/3D SOLVER UNIFICATION (remaining)

Steps 1-4 + apply_forces merge ✅. Remaining:

SU1. **Step 5 — stacked-tensor storage — SCOPE NOW BOUNDED BY MEMORY.**
  - **Velocity DONE (2026-06-16):** `u0/v0/w0` → one contiguous `FluidSolver._vel`
    (D,*grid), exposed via compat `@property` views (reads = zero-copy row view;
    sets copy into the row; `w0` raises AttributeError in 2-D so `getattr(fs,'w0')`
    keeps "absent-in-2D" semantics for the viewers). **Memory-neutral** (same
    3·grid floats, no duplicate tensors, kernels get contiguous views) +
    **bit-identical** drags on the kernel(GPU) + python(CPU) 1guilla A/B vs the
    pre-refactor baseline. Solver owns no other per-component trio (all
    `sdf_val/body/mu/normal` live on `composite_body`).
  - **Field trios B–E (`sdf_val_{u,v,w}`, `body_{u,v,w}`, `mu0/mu1`, normals)
    DELIBERATELY NOT STACKED — memory.** In kernel mode these are **per-step
    temporaries** (`fluid_step` allocs `sdf_u_tmp/bU_tmp/…`, Kernel A fills, Kernel
    B consumes for fused BDIM+adv-diff, then freed — the Phase-I optimization).
    Stacking them into persistent `comp` buffers would PIN that memory → regression
    of exactly what this branch optimizes. They are also spread across 6+ body
    classes (analytical/fish/mesh/multi-animat) with aliasing + `torch.where`
    unions. ⇒ SU1's correct scope is the persistent velocity only; the rest stay
    freed temporaries. See `feedback-no-stacking-perstep-temporaries` memory.

(SU2 — Step 6 BDIMhandler update merge — ✅ DONE; see the DONE section below.)

Per-step rules: branch from `optimize_speed_memory`, one PR per step, validate 2D
(`_1guillasim` pinned) + 3D (jellyfish) + cost_analysis (<5% wall-clock regression),
rel-err <1e-6 on integrated quantities. No semantics changes.

---

# GPU UTILISATION (small-tank / small-grid regime) — benchmark 2026-06-11

At small grid sizes the GPU is underutilised because **Python kernel-dispatch overhead
dominates compute** (each multigrid V-cycle dispatches 50-100+ small kernels, each ~10-50 µs
Python cost). Three strategies benchmarked on an RTX 4080 SUPER:

| Strategy | 2D 128×64 | 3D 64×32×32 | 3D 128×64×64 | Δmem |
|----------|-----------|-------------|--------------|------|
| `adv_diff_streams` | 0.98× | 0.99× | 1.01× | 0 |
| **`compile_project`** | **2.74×** | **2.32×** | **2.62×** | **≈0** |
| `use_cuda_graphs` | 1.11× | 1.17× | 1.03× | +14 MiB |
| `compile_project` + streams | 2.78× | 2.27× | 2.70× | 0 |

GU1. **`compile_project=True`** is the clear winner: 2.3–2.8× speedup, zero memory overhead,
  works with all schemes including abdquickest. Enable in `solver.solver.compile_project`.
GU2. **`use_cuda_graphs`** gives modest 1.03–1.17×; benefit shrinks with grid size; +14 MiB
  at 128³. Incompatible with abdquickest (gracefully skips with a log message).
GU3. **`adv_diff_streams`** never helps — dispatch overhead is in the Poisson V-cycle, not
  advection; streams add overhead. Not worth enabling alone.
GU4. Combined `compile_project + adv_diff_streams` gives marginal extra gain over compile alone.

(All three config options have been implemented — see DONE section.)

### Native-path GPU-util — RE-PROFILED 2026-06-14 (after the native CUDA Poisson port)

The 2026-06-11 analysis above is for the **Python** projection path (the
`compile_project` knob compiles that path). Production now runs the **native CUDA
Poisson** path (mgcg/rmgcg), which is already 10–17× faster than Python. Re-profiling
*that* path (`bench_python_overhead.py`) shows it is **NOT Python-dispatch bound** (the
whole solve is a single Python op call) and **NOT sync bound** (MP11 removed the
alpha/beta syncs). The residual low GPU-util (22% @ 2D-N64 → 50% @ 3D-N96) is
**CUDA kernel-launch latency**: the C++ V-cycle issues ~2100 *tiny* kernels per solve and
the GPU idles between launches. Confirmed: idle ≈ (#launches) × ~1.5 µs.

GU6. ~~**CUDA-graph capture of the native solve**~~ ✅ SHIPPED (2026-06-15). Replays the
  ~1300-kernel native V-cycle with ~1 host launch on launch-bound small grids. **Enabler**
  (already done 2026-06-14): the native drivers take `tol < 0` = sync-free fixed-cycle mode
  (skips the residual-norm `.item()` short-circuit) → host-sync-free → graph-capturable;
  `tol >= 0` unchanged. **Wired into `PoissonSolver`** (`poisson_mult.py`): `_solve_mgcg_native`
  / `_solve_multigrid_native` now route through `_graphed_native_solve` when enabled — lazily
  captures one graph per (method, ndim, input-shape) into static p/f/face buffers, then each
  step `copy_`s the live RHS + BDIM face coeffs + warm-start guess in, replays, copies p back
  out (residual cloned). Capture failure → permanent eager fallback. Config:
  `poisson_cuda_graph` (default False) + `poisson_cuda_graph_max_cells` (interior-cell gate,
  default 64³) wired through `base_sim_config.py`. **mgcg + multigrid only** (rmgcg's
  deflation basis changes / fft uses a different solver — both untouched, automatically
  out of scope). **Verified:** graph vs eager bit-exact (`|dp|=0`, 2D/3D, both methods);
  gating correct (N=96 skips, N=48 captures); speedup 9.8×/8.1× (2D-N64/128), 4.0× (3D-N48)
  → 1.04× (3D-N128) — regime-dependent as the POC predicted, hence the size gate;
  mgcg/rmgcg native self-tests still PASS. Harness `/tmp/verify_gu6.py` + `/tmp/verify_gu6_perf.py`.
  ⇒ **SMALL-GRID-ONLY win** (launch-bound); net-NEGATIVE on large compute-bound 3D, so the
  size gate is load-bearing. A full megakernel rewrite would share the large-grid limitation
  AND hit a cooperative-launch occupancy ceiling (~64³) — graphs, size-gated, are the right
  tool. RBGS red/black can't be fused (Gauss-Seidel dep).
  **NOTE on T4 (`--poisson_compile`):** SUBSUMED — no distinct work remains. The literal T4
  (`torch.compile` the *Python* multigrid path) was built (`14bbbfa`) then deliberately
  removed (`4de22b2`, "unnecessary via kernel mode"); its surviving reframe ("CUDA-graph
  capture of the whole native solve") IS this GU6 item, now shipped. The only forward-looking
  follow-on is an optional device-side convergence flag so the captured graph can early-exit
  when converged instead of running the fixed cycle budget (GU6 uses `tol<0` fixed cycles).

---

# PER-STEP HOT-PATH OVERHEAD (measure first — found 2026-06-05 while doing T1/diagnostics)

These run on EVERY step; none is measured for wall-clock cost yet. Time them
(e.g. with the cost_analysis harness) before/after gating.

PH3. **H3 `diagnostics_every=100` is now default-ON** in `base_sim_config.py` (2026-06-11).
  Adds a small recurring vorticity/divergence + host-sync cost. Defensible given the
  blow-up-debugging history, but RATIFY: keep at 100, or set 0 (opt-in)?

---

# LOW PRIORITY

LP2. Crank-Nicolson diffusion — current explicit limit `dt < h²/(2ν·ndim)` is not a
  bottleneck now, relevant only if dt is pushed aggressively.
LP3. **eps configurable** — BDIM transition thickness is hardcoded `2h`; add `eps_cells`
  config key (3h-4h smoother on coarse grids).
LP6. **F1 AABB cull force integration** — δ(sdf−ε) is evaluated over the whole domain per
  body but is nonzero only within ε. Slice to each body's AABB+ε. 10-100× for small
  swimmers in big pools.
LP8. **F2** drag records: CPU pinned memory + async copy instead of GPU `nt` pre-alloc.
LP9. **Adaptive / CFL-limited timestep** (found 2026-06-23) — `dt` is FIXED for the
  whole run (no `self.dt =` reassignment anywhere in `solver.py`); `diagnostics.py`
  only *warns* on CFL>0.5, it never adapts. The blow-up-prone cases (toy boat,
  surface eel, sphere-near-wall) would benefit from a CFL-limited dt as a safety
  lever — recompute `dt = cfl_target·h/|u|max` each step (or every N steps) with a
  config cap. Caveat: checkpoint/restart (LT6) and the Adams-Bashforth flux history
  assume constant dt, so a variable dt touches both. Keep opt-in (default fixed).
LP10. **`COMPOSITEmesh2sdf.transform_3d` is a `NotImplementedError`** (found 2026-06-23,
  `body.py:1052`) — blocked on a missing `mesh2sdf.transform_3d`. Latent: any code
  path that rotates/translates a COMPOSITEmesh2sdf body crashes at runtime. Either
  implement the underlying mesh transform or assert-guard the call site so the
  failure is explicit at construction.
LP11. **Pleurodeles full-3D example is a stub** (found 2026-06-23) —
  `farms_examples/pleurodeles/gen_configs_swim_full3d.py:25` has a placeholder SDF
  filename TODO ("replace once the full-3D SDF is ready"); the example cannot run
  as-is. Either finish the full-3D SDF + wire it, or mark the file experimental.
LP12. ~~**Empty orphan dir `validation/free_surface_2d/`** (found 2026-06-23) — contains
  only `__pycache__`.~~ ✅ Deleted 2026-06-23 with the rest of the free-surface method.

---

# TEST INFRA / REPO HYGIENE (found 2026-06-23)

TI1. **No CI, no test aggregation.** There is no `.github/`, no `conftest.py`,
  `pytest.ini`, or `tox.ini`; the ~26 `test_*.py` files are scattered across
  `src/`, `src/kernels/`, and `integration/` and are run by hand. Meanwhile the
  SU1 per-step rule MANDATES "validate 2D (`_1guillasim`) + 3D (jellyfish) +
  cost_analysis (<5% regression), rel-err <1e-6" with nothing automating it.
  Add at minimum a `pytest` config that collects the self-tests (the CUDA ones
  skip cleanly without a GPU via `pytest.importorskip`), and ideally a CI
  workflow running the CPU-path parity oracles on push. Enabler for trusting the
  refactor-heavy `optimize_speed_memory` branch. NOTE: kernel parity tests need a
  built `_C.so` matching the runtime torch — gate or build-in-CI accordingly.
TI2. ~~**Repo-root clutter / un-ignored diag output.**~~ ✅ DONE (2026-06-23).
  The scratch dirs/CSV dumps (`_submerged_diag/`, `_flip_diag/`, `_overlap_study/`,
  root `run_*.py`) were already relocated/dropped in commit `bbdaa4d` ("Repo tidy").
  Remaining preventative half done: added directory-only `.gitignore` rules
  `_*_diag/` and `_*_study/` (trailing slash => the `_overlap_diag.py`/`_region_diag.py`
  harness scripts stay tracked; the in-dir `force_*.csv` dumps are covered without
  touching the deliberately-commented `# *.csv` rule). Verified with `git check-ignore`.

---

# ARCHITECTURE / PORTABILITY (strategy 2026-06-14)

## Part 1 — Drop FARMS; support pluggable rigid-body engines (Isaac Sim, MuJoCo)
Goal: get rid of FARMS entirely and make the rigid-body engine swappable
(MuJoCo today, Isaac Sim / Isaac Lab next).

Coupling map (investigated): **`BDIMhandler` already does NOT import FARMS** — it
speaks MuJoCo directly (`data.xpos/xquat/xipos`, `model.body_mass`, `geom_*`,
`xfrc_applied`/`mj_applyFT`). FARMS only owns the *outer* layer:
  - `integration/extensions.py` — `FluidExtension(TaskExtension)` hooks
    (`initialize_episode`, `before_step`), experiment options, HDF5 IO.
  - `farms_examples/base_sim_config.py` + `gen_configs_*` — animat model + scene gen.
  - swimmer controllers (CPG networks, PD controllers).
  - the viewers (all `import farms` for the MuJoCo viewer).

AP1. [x] ✅ **`RigidBodyBackend` adapter — DONE (2026-06-23).** New module
      `lilytorch/integration/rigid_body_backend.py`: `RigidBodyBackend` ABC +
      `FarmsMujocoBackend` impl + relocated `MujocoCheckpoint` (re-exported from
      BDIMhandler as `_MujocoCheckpoint` for the existing checkpoint test). The
      adapter surface: `get_body_poses_velocities(source,iteration)`,
      `get_body_mass_radius`, `apply_xfrc`, `gravity_z`, `set_contact_params`,
      `checkpoint()`, `bind_step(task,physics)`. ALL MuJoCo/FARMS access (the
      ~16 `physics.*` + `task.maps`/`task.units` sites: pose/velocity reads both
      sensors+physics paths, mass/rbound, gravity, contact tuning, xfrc force
      write, implicit-coupling checkpoint) moved behind it; BDIMhandler keeps the
      coupling logic (2-D slicing consumers, buoyancy formula, force scaling/
      relaxation, IQN-ILS loop). Behavior-preserving: net −116 lines in
      BDIMhandler; `git grep physics.model|physics.data|task.units|task.maps`
      in BDIMhandler now empty. **Verified:** 21/21 integration regression tests
      identical to baseline (pose-source, checkpoint, strong/fsi coupling, all
      update parities); focused bit-identical unit check of apply_xfrc/
      get_body_mass_radius/gravity/contact vs the original inline formulas;
      both gather paths confirmed logic-identical to git HEAD by normalized diff.
      Core source (solver/forces/body) untouched. (Full live `_1guillasim`
      coupled smoke not run — heavyweight/needs display; recommended as manual
      belt-and-suspenders before merge.) ⇒ unblocks AP2/AP3/AP5.
AP2. [ ] **MuJoCo backend** implementing the adapter from raw `mujoco.MjModel/MjData`
      (or dm_control `Physics`) — no FARMS dependency.
AP3. [ ] **Standalone driver loop** (~100 lines): load model, step physics, call
      `BDIMhandler.step()` each tick (replaces the `before_step` hook).
AP4. [ ] Replace controllers/viewers (FARMS-based) with engine-native equivalents
      (`mujoco.viewer`; note `xfrc_applied` viewer pitfall — use `qfrc_applied`).
AP5. [ ] **Isaac Lab backend** — exposes body state as **torch GPU tensors**, so the
      coupling becomes GPU-resident (no numpy/CPU round-trip the MuJoCo path pays).
      Strong fit; the adapter is the enabler. (MuJoCo-Warp/MJX is a GPU alternative.)
Feasibility: MODERATE — numerics are already FARMS-free; the work is outer-loop
(driver/config/controllers/viewers), not the solver.

### Caveat A — model/file-description format (investigated 2026-06-15)
FARMS owns TWO things behind the `.sdf` (SDFormat XML) authoring format, and they
are SEPARABLE:
  1. **SDF parser** (`farms_core.io.sdf.ModelSDF.read`) — used by BOTH the FARMS
     converter AND lilytorch's own `BodyMesh`/`CompositeBody` (body.py), but the
     latter only pulls *mesh files + link poses* to build its OWN signed-distance
     tables. Shedding it is EASY (mostly pure-Python; vendor it, or author in
     URDF/MJCF instead).
  2. **SDF→MJCF converter** (`farms_mujoco/.../mjcf.py`, ~1725 lines) — the heavy,
     MuJoCo-specific part (units, spawn modes, joint-axis rotation, inertial frames,
     mesh composites, hfields, materials). Faithfully reproducing it is MODERATE–HARD,
     but **largely AVOIDABLE**: MuJoCo loads MJCF/URDF natively → author there and
     delete the conversion step.
  Note BDIMhandler needs NEITHER (geometry from lilytorch `body.sdf`, poses from
  MuJoCo `data.xpos/xquat`).
AP5b. [ ] **Pick a portable model format.** SDFormat is the LEAST supported outside
      Gazebo. Engine-native: MuJoCo=MJCF, Isaac Sim/Lab=USD (via URDF/USD importers),
      Newton=USD/MJCF/URDF importers. Realistic common interchange = **URDF or MJCF**.
      Recommendation: separate "geometry/inertia spec" (what BDIM needs: meshes +
      link tree + poses + masses, format-agnostic) from "engine model" (native
      MJCF/USD/URDF), and standardise authoring on URDF/MJCF rather than SDF→X.

### Caveat B — logging (investigated 2026-06-15)
FARMS provides efficient preallocated Cython typed arrays (`LinkSensorArray`,
`JointSensorArray`, `ContactsArray`, `XfrcArray`, shape `[n_iter,n_links,size]`) →
HDF5 via `dict_to_hdf5` (extensions.py `DataLogger`/`end_episode`).
AP5c. [ ] **Backend-neutral `SensorLogger`** to replace FARMS `AnimatData` logging.
      Feasibility EASY–MODERATE: lilytorch already has HDF5 logging patterns
      (`diagnostics.py`, `save_drags_h5`). Only loss is the Cython buffer
      micro-efficiency — negligible vs. the fluid solve. Pull per-step link/joint
      state straight from the engine via the `RigidBodyBackend` adapter.

## Part 2 — Single-source CPU+GPU kernels (kill the .cpp/.cu double-write)
Pain: kernels are hand-written TWICE (CPU `.cpp` + CUDA `.cu`) → more code, more bugs.
Double-written today: **streaming_sdf (2d/3d), lagrangian_forces (2d/3d), rbgs**.
(Poisson driver/transfer/advection are CUDA-only + pure-PyTorch CPU fallback.)

Strategy (two kernel classes):
AP6. [ ] **Fusible stencil/pointwise** (rbgs sweep, residual, restriction, advection
      flux) → push through **`torch.compile`/Inductor**: write once in PyTorch,
      auto-generates C++ (CPU) + Triton (GPU, incl. ROCm). No hand kernel.
AP7. [ ] **Irregular scatter/gather** (streaming_sdf, lagrangian_forces) → **Warp**
      (chosen): single Python `@wp.kernel` → CPU + CUDA, zero-copy torch interop.
      Driver-style kernels (poisson_solve mgcg/multigrid loops) stay `.cu` — Warp is
      kernel-level, no C++ driver-with-control-flow equivalent (use CUDA-graph capture
      if unified). AMD: Warp's HIP backend is weak — if AMD becomes hard-required,
      Taichi (Vulkan) or SYCL/Kokkos (HIP) for those kernels instead.

**Warp RBGS POC done 2026-06-14** (`/tmp/poc_warp_rbgs.py`, pure test):
  - Interop: zero-copy `wp.from_torch` (warp wrote the torch tensor, same ptr); a
    native `torch.ops` CUDA kernel consumed warp output on one stream; same
    `@wp.kernel` ran on CPU and CUDA. Correctness within 0.9% of native.
  - Perf (ms/2-sweep): 256² native 0.010 / warp 0.094 / warp+graph 0.019;
    2048² native 0.179 / warp 0.664 / warp+graph 0.650; pytorch 5–35× slower than warp.
  - Findings: eager Warp is launch-bound (6 launches vs native's 1 fused) → **CUDA
    graph capture** removes most of it (~2× native at typical sizes). Residual gap
    (2–3.6×) at large grids = native is hand-TILED (shared mem, all sweeps fused, 1
    global pass) vs naive warp's 6 global passes. **To match native, write a tiled
    warp kernel (`wp.tile`).** Net: Warp gives single-source + crushes the PyTorch
    path; matching a hand-tuned kernel needs tiling work.
AP8. [ ] Port streaming_sdf + lagrangian_forces to Warp (tiled where bandwidth-bound);
      keep self-tests as oracles; retire the `.cpp` twins.

---

# LONG TERM

LT1. **LES for high-Reynolds** — extend the existing Smagorinsky SGS model into a full LES
  workflow (WALE/dynamic-Smagorinsky options, wall treatment) for turbulent high-Re
  regimes where DNS is intractable.
LT2. **AMR (Adaptive Mesh Refinement)** — refine the grid only near bodies and in the wake;
  the enabler for high-Re cases at tractable cost (pairs with LES).
LT3. **Near-boundary stress stencils** — velocity gradients use central differences,
  degrading to 1st-order near immersed bodies. One-sided / ghost-cell stencils would
  improve force accuracy and reduce oscillations.
LT4. **5th-order Hermite smoothstep** for the BDIM delta — `0.5*(1+d/ε+sin(πd/ε)/π)` has
  cancellation at `d≈±ε`; Hermite is more robust and drops sin/cos.
LT5. **2nd-order body coupling** — body SDF/velocity are updated once per step, so Heun's
  corrector uses body state at *t* not *t+dt/2* → coupling is effectively 1st-order.
  Update body to *t+dt* after the predictor and feed the corrector.
LT6. **Checkpoint/restart** — periodic full-state save (iteration, drag records,
  Adams-Bashforth flux, body poses) so a crash at iter 999k of a 1M run isn't fatal.
  Current `_load_initial_conditions` restores only `u,v,[w],p` → warm restart diverges
  from a continuous run.
LT7. **Granular flow (sand) via μ(I)-rheology** — new physics: dense dry/immersed granular
  media as an incompressible fluid with a pressure-dependent, shear-rate-dependent
  effective viscosity. Implemented as `GranularSolver(TwoPhaseSolver)` reusing projection,
  BDIM, advection–diffusion, forces, and FARMS/MuJoCo coupling **unchanged**; adds only
  (a) a pressure-dependent μ(I) viscosity closure and (b) granular stabilisation. ~80% of
  the machinery already exists (`_compute_nu_t`, `ops.carreau_viscosity`,
  `ops.strain_rate_magnitude`, VOF free surface for the pile surface). Pitched as the
  cheapest genuinely-new physics LilyTorch can add. Full design + milestone plan:
  `milestones/granular_design.md`.
LT8. **Elastic-body FSI via Cosserat rods (PyElastica coupling)** — simulate flexible
  slender bodies (flapping fins, flagella, soft swimmers, plant stalks) in the BDIM solver.
  **Key insight: the SDF is the easy part — no raycasting, no Bezier.** A Cosserat rod is a
  chain of tapered capsules, and the analytical capsule-chain SDF already exists in the
  codebase (`body.py: segment()` 2D round-cone, `capsule_3d`/`sdUnevenCapsule` 3D); the
  deforming centerline+radius SDF machinery is exactly what `BodyFish*` already does each
  step: `sdf(x) = smin_i segment(x, p_i, p_{i+1}, r_i, r_{i+1})`, AABB-cropped. Bezier is
  strictly worse (no closed-form distance, sub-grid fidelity wasted below ~2h).
  **Architecture:** PyElastica is the structural solver (like MuJoCo for rigid bodies) —
  slots under the planned `RigidBodyBackend`/`StructuralBackend` adapter (AP1) returning
  per-node positions/velocities/radii. Loop: query rod state → rebuild capsule-chain SDF →
  body velocity = nearest-node velocity → BDIM+projection → bin hydro forces to centerline
  nodes as distributed loads+moments → feed back to PyElastica. PyElastica is CPU/Numba but
  a rod is ~100 nodes (negligible vs. fluid). New `BodyCosseratRod` + `PyElasticaBackend`.
  **Real difficulties (NOT the SDF):** (1) **added-mass instability** — light flexible bodies
  are the worst case; MUST use the already-shipped implicit Aitken/IQN-ILS coupling
  (`BDIMhandler._step_implicit`), prefer Aitken to start (IQN reuse-poisoning trap); (2)
  **force→moment attribution** — Cosserat elements carry bending/twist, so accumulate r×f
  torque per node, not just force (extend the Lagrangian-marker force path); (3) **grid
  resolution floor** — BDIM is volume-penalization, needs rod radius ≳ 2-3h (sub-grid fibers
  need a different IB+drag-law model, out of scope); (4) **joint seam smoothness** — per-
  segment min creates concave creases → noisy normals; use polynomial smooth-min for the
  union (+ existing `body_velocity_blend_eps_cells` for the velocity side). Reuses ~3 existing
  pieces: deforming-fish SDF builder, Lagrangian force attribution, implicit coupling.
  NOTE: elastic *sheets / 3D soft solids* are the genuinely-hard SDF case (deforming FEM mesh
  → per-step distance-to-deforming-triangles + robust winding-number sign); start with rods.

LT9. SPH simulation support (?).
LT10. Monolithic strongly-coupled fluid + multi-rigid-body solver (?) — hard, would require dropping MuJoCo.
LT11. Refactor: extract `FluidSolver.__init__` (~500 lines) into `_setup_grid/_models/
  _poisson/_output`; add `BaseSimConfig.generate_config()` (dry-run YAML without launch);
  add type hints.

---

# ═══════════════════════════════════════════════════════════
# ✅ DONE
# ═══════════════════════════════════════════════════════════

## High priority

HP4. ~~**Wire in `FlowDiagnostics`.**~~ ✅ `FlowDiagnostics` moved to its own module
  `lilytorch/src/diagnostics.py` (out of solver.py), instantiated in `FluidSolver.__init__`
  when `diagnostics_every > 0`, called from `finalize_step` on the post-projection field,
  and saved to `diagnostics.h5` at the end of `run_sim`/`run_from_initial`. Default
  `diagnostics_every = 100` in `base_sim_config.py` (0 disables). Subsumes the old F4 item.

## Bugs

BG1. ✅ **`_forces_lagrangian_2d_python_ref` undefined variables (2026-06-11)** —
  `eps_ij`, `nu_rho`, and `nu_rho_const` were used inside the per-body loop but never
  computed in this function.  The production `forces_lagrangian_2d` computes them before
  the loop via `_viscous_stress_tensor` + `_compute_nu_rho_for_forces`; the reference was
  missing the same preamble → silently crashes at runtime when called for tests/debugging.
  **Fix:** added the preamble (mirror of `forces_lagrangian_2d` lines 1094-1122) before
  the body loop in `_forces_lagrangian_2d_python_ref`.

BG2. ✅ **ABDQUICKEST hardcoded `C=0.1` (2026-06-11)** — `abdquickest(u, c, d, C=0.1)` used
  a fixed Courant number regardless of the actual flow CFL.  For CFL > 0.1 the TVD limiter
  is overly optimistic (less diffusive than it should be), and for CFL→0.5 the scheme is
  no longer TVD-guaranteed.  `C` should be `|u|·dt/h` (the actual advective Courant).
  **Fix:** in `AdvDiffSolver._solve_convective`, compute the step's max CFL before the
  flux loop and pass it as `C` to `abdquickest`.  Stored on `self._scheme_name` to avoid
  checking at every (i,d) iteration.

BG3. ✅ **Semi-Lagrangian back-tracing upgraded to RK2 (2026-06-11)** — `_solve_semi_lagrangian`
  used 1st-order Euler back-tracing `x_dep = x − u(x)·dt`.  Replaced with the 2-stage
  midpoint method: `x_mid = x − 0.5·dt·u(x)`, then `x_dep = x − dt·u(x_mid)`.  This is
  2nd-order accurate in the Lagrangian path at the cost of one extra interpolation per
  component per step.  Also removed the spurious `.clone().detach()` calls (unnecessary
  under `torch.no_grad()`) — now plain `.clone()`.

BG4. ✅ **Dead `_use_legacy_sparse_forces_2d` code removed (2026-06-11)** — the flag was
  hardcoded `False` since the sparse-AABB force path was unified; the dead AABB-union
  block (lines 455-476) and the `if/else` cache branch (lines 502-511) were removed from
  `forces_method2`.

BG5. ✅ **`_vcycle_rbgs_2d/3d`: red/black masks reused between pre-smooth and post-smooth
  (2026-06-11)** — masks are shape-dependent only so the post-smooth can reuse the ones
  built for the pre-smooth at the same level; removed the duplicate `_rb_masks_*` call.

BG7. ✅ **`strain_rate_magnitude` cross-derivative stagger fix (2026-06-11)** — `dudy` (at
  x-faces) and `dvdx` (at y-faces) were at different stagger positions before being summed
  into S12 = 0.5·(∂u/∂y + ∂v/∂x); this made the Smagorinsky eddy viscosity physically
  inconsistent in the cross terms.  **Fix:** added `_stag_to_cc` helper in `operations.py`
  that averages each cross-derivative to cell centres before combining.  Verified: pure
  shear gives `|S|=1.0`, solid rotation gives `|S|≈0` (machine precision) — previous code
  gave spurious non-zero for solid rotation.

## Memory / perf

MP1. ~~**H1 per-step `torch.cuda.empty_cache()`**~~ ✅ (2026-06-11). Gated to every
  `empty_cache_every` steps (default 200, config key `empty_cache_every` in
  `base_sim_config.py`).
MP2. ~~**H2 per-step host sync in `check_explosion`**~~ ✅ (2026-06-11). Throttled to every
  `check_explosion_every` steps (default 50, config key `check_explosion_every` in
  `base_sim_config.py`).
MP3. ~~**T1a**~~ ✅ Inlined `_tvd_face` into `van_leer`/`abdquickest`/`cubista` in
  `advection.py`, chaining in-place on the owned `psi` and reusing the live `denom`;
  `_tvd_face` helper removed. Verified bit-exact (fp32+fp64, incl. denom≈0 branch).
MP4. ~~**T1b**~~ ✅ `div` is now a local in `solver.py:project()` (was a persistent
  `self.div`) and is `del`-ed right after each `_poisson_solve` returns, before the
  gradient/correction allocations. ~0.5 GiB transient + removes a persistent field.
MP5. ~~**T3a**~~ ✅ (2026-06-11) Eliminated the `div` field on the multigrid/MGCG path:
  `ops.divergence_interior()` computes the interior-only RHS (no ghost cells) directly
  in `project()`; for the Python path it is scaled in-place by `h²` before the solve
  (`pre_scaled=True` kwarg skips the redundant `f_scaled = h²·f` copy inside
  `solve_multigrid`/`solve_mgcg`).  Kernel path (`use_kernels=True`) is unaffected
  (native CUDA kernel applies h² internally).  FFT path unchanged (still uses full-grid
  `div`).  Dead `'div'` entry removed from `_BDIM_FIELD_NAMES`.
MP7. ~~**T2a**~~ ✅ (2026-06-12) Fused CUDA `advect_flux_add` kernel written in
  `lilytorch/src/kernels/csrc/cuda/advection_flux.cu` and registered as
  `torch.ops.lilytorch_kernels.advect_flux_add`. Replaces the Python
  `_flux → F[:-1]-F[1:] → rhs.add_()` chain (which allocated ~4 full-grid tensors per
  (i,d) pair) with a single kernel launch that accumulates the flux divergence in
  registers and writes directly into rhs.  Handles all 5 schemes (QUICK, ABDQUICKEST,
  vanLeer, CDS, CUBISTA) via compile-time template specialisation.  Handles
  non-contiguous fv/p views via explicit stride parameters; rhs strides are also passed
  so face_dim-dependent layout is handled correctly.  Activated automatically in
  `AdvDiffSolver._solve_convective` on CUDA (skips `_get_step_scheme` sync for
  ABDQUICKEST).  Measured **3.5–3.9× speedup** on 128³; 260/260 flux parity checks +
  10/10 full `_solve_convective` parity checks passed at machine precision (fp64 rel_err
  ≤ 1e-16).
MP11. ~~**T5 Pipelined / communication-avoiding CG (native Poisson driver)**~~ ✅
  (2026-06-14) The native `mgcg`/`rmgcg` CG loop did ~4 host `.item()` syncs per iteration
  (alpha ×2, beta, residual-norm), each a CPU↔GPU pipeline stall (native 2D solves at only
  ~23% GPU-util). **Fix:** keep `alpha`/`beta` as 0-dim *device* scalars (drop the
  `.item()` calls) and fuse the axpy updates with `addcmul_`/`mul_` (a 0-dim tensor
  broadcasts — identical kernel count, no extra temps): `x += α·d` → `x_in.addcmul_(d_in,
  alpha)`, `r -= α·q` → `r.addcmul_(q, alpha, -1.0)`, `d = β·d + z` → `d_in.mul_(beta)
  .add_(z_in)`. This removes the alpha/beta D→H syncs, leaving the residual-norm check as
  the only per-iter host sync (~1/iter, as targeted). Applied to all 4 drivers
  (mgcg/rmgcg × 2D/3D) in `poisson_solve.cu`. **Measured ~1.13–1.18× wall-clock** on
  sync-bound small-2D solves (N=32/64/128). Parity: rmgcg(kdef=0) still bit-identical to
  mgcg (0.00e+00); all 3 Poisson self-tests PASS. NOTE: `addcmul_` is non-FMA (matches the
  Python `_cg_core` reference formula exactly, vs the old scalar `add_` which used FMA) —
  this only perturbs the LAST ULPs, visible solely when over-iterating f32 *past its
  rounding floor* into the chaotic loss-of-orthogonality regime (which the Python path
  exhibits too); production runs 3 MGCG cycles, far below it. The f32 N=16 parity case in
  `test_poisson_solve_mgcg_self.py` was retuned to compare in the converged regime
  (`max_cycles=6`). Follow-on: this was the prerequisite for CUDA-graph capture of the whole
  solve — now SHIPPED as GU6 (which uses the `tol<0` fixed-cycle mode to be sync-free rather
  than needing a device-side convergence flag; an early-exit device flag remains an optional
  enhancement). The plain `multigrid` driver already syncs only once/cycle (residual norm),
  so it was left unchanged.

## 2D/3D solver unification — kernel parity

SU2. ~~**Step 6 — merge BDIMhandler `_update_2d/_3d` + `_update_*_streaming_multi`**~~
  ✅ (2026-06-15, branch `optimize_speed_memory`). Both per-dim pairs collapsed
  into dimension-generic methods driven by `self._sim_axes = range(ndim)`:
  - **Python path:** `_update_2d`+`_update_3d` → `_update_python` (per-axis field
    lists for the staggered SDF/velocity union loop; dim-specific only in
    body-frame compose, rotation helper, rigid-body velocity formula, AABB
    descriptor, and the Lagrangian/contour tail — factored into
    `_refresh_lagrangian_contour_2d`, `_refresh_lagrangian_tris_3d`,
    `_apply_contour_mask_2d`).
  - **Kernel path (production):** `_update_2d_streaming_multi`+
    `_update_3d_streaming_multi` → `_update_streaming_multi` + helpers
    `_stream_kin_static`, `_stream_static_pack` (writes the exact
    `_kernel_static_{2,3}d` names `forces.py`/`solver.fluid_step` read),
    `_stream_lagrangian_refresh`. Per-dim packed layouts (kin row D*D+3D+(3|1),
    body_meta 3D+1, dirty-AABB keys) reproduced exactly; 2-D identity local frame
    makes the unified compose einsum (`R@I`, `urdf+R@0`) a bit-exact no-op.
  `_init_update` dispatches the unified methods. The 4 legacy methods are RETAINED
  as the parity oracles for `test_update_python_parity.py` +
  `test_update_streaming_parity.py` (both bit-identical, max|Δ|=0, across
  Eulerian/blend/Lagrangian/contour × 2D/3D, streaming incl. the prev-union dirty
  branch). Also fixed a pre-existing stale name in `test_update_2d_mirrors_3d.py`
  (`_body_aabb_indices_2d`→`_body_aabb_local_2d`). **FSI regression:** GPU
  kernel-mode + CPU python-mode end-to-end A/B on the real 3D multi-link 1guilla
  (225×75×13, 8 steps, Lagrangian) — per-link drags+torques bit-identical
  unified-vs-legacy on BOTH paths. (2-D kernel covered by unit parity only — no
  non-stale 2-D coupled example exists on HEAD.) **Follow-up (low-risk cleanup):**
  delete the 4 legacy oracle methods (~700 lines, the net line-count win) once a
  full-length run confirms the unified path in situ.

SU3. ~~**K9**~~ ✅ Added `is_cuda` TORCH_CHECK to `apply_bcs_2d_cuda` (mirrors 3D).
SU4. ~~**K10**~~ ✅ `apply_bcs_2d/3d_kernel` now compute `src_lin` unconditionally
  (Dirichlet's value is harmlessly discarded), dropping the dead `src_lin = 0` init and
  the `if (kind != 1)` branch. Rebuilt `_C.so`; CPU↔CUDA parity exact (fp32+fp64).

## GPU utilisation

✅ **All three config options implemented (2026-06-12):** `solver.compile_project`,
`solver.use_cuda_graphs`, `solver.adv_diff_streams` wired into `FluidSolver.__init__`
and exposed in `BaseSimConfig` (all default `False`). `compile_project=True` is the
recommended opt-in for GPU production runs; `torch.compile` has a 30–100 s first-compile
overhead (amortised over long sims). Benchmark script:
`lilytorch/validation/cost_analysis/bench_gpu_util.py`.

GU5. ~~**Fuse the per-axis Neumann BC into one launch**~~ ✅ (2026-06-14) The BC mirror
  (`apply_neumann_bc_2d/3d` in `multigrid_smoothers.cu`) was the single largest launch-count
  contributor — one kernel **per axis pair** (3 in 3D), called by every smoother half-sweep
  and residual eval (~1170 launches/solve @ 3D-N96). Fused into ONE launch using
  `blockIdx.{y,z}` as the axis-pair selector. Safe because the 5-/7-point stencils only read
  **face** ghosts (each axis pair writes a disjoint interior-face slab → race-free; the
  never-read edge/corner ghosts are left stale). **Measured:** launches/solve 2101→1321
  (−37%), wall-clock **−10% (2D-N64) to −25% (3D-N48)**, −17% (3D-N96). multigrid/mgcg/rmgcg
  self-tests still PASS (BC exercised through the full solve vs Python ref).

## Per-step hot-path overhead

PH1. ~~**H1 per-step `torch.cuda.empty_cache()`**~~ ✅ (2026-06-11) — see MP1.
PH2. ~~**H2 per-step host sync in `check_explosion`**~~ ✅ (2026-06-11) — see MP2.
PH5. **TwoPhaseSolver re-introduced the per-step `empty_cache()` (regresses PH1).**
  `TwoPhaseSolver.finalize_step` (`lilytorch/src/two_phase_solver.py`) overrides the base
  and calls `torch.cuda.empty_cache()` **unconditionally every step**, bypassing the
  `empty_cache_every` throttle that fixed PH1 for the base solver — the dominant overhead
  making the two-phase path (e.g. `gen_config_surface_pool.py`) much slower than the one-way
  path (`gen_config_full_pool.py`) at identical grid/physics. The override also ran
  `check_explosion` every step (regressed PH2). **check_explosion FIXED (2026-06-15):** the
  override now gates it on `check_explosion_every` (default 50) like the base. **empty_cache
  stopgap (2026-06-15):** gated the flush on `empty_cache_every` in the override + set
  `empty_cache_every = 10**9` in `gen_config_surface_pool.py` so it never fires. **Remaining
  proper fix:** delete the `empty_cache()` call from the override entirely (it is cosmetic —
  only lowers nvidia-smi reserved memory — and unnecessary in a fixed-shape loop; the moving
  AABB force crops are bounded). Confine to `two_phase_solver.py` (no core-source edits).
PH4. ~~Delete legacy `adv_diff.py`~~ ✅ (2026-06-11) — repointed the lone importer
  (`benchmarks/run_compile_advdiff_bench.py`) to `lilytorch.src.advection` (drop-in: identical
  `AdvDiffSolver` API), removed the file, fixed the "kept on disk as legacy" docstrings.

## Low priority

LP4. ~~**Cache `_compute_union_aabb` across BDIM + coefficient passes**~~ ✅ (2026-06-11).
  The AABB was computed twice per step on the kernel path: once in `_apply_bdim_all_axes`
  and once inside `_compute_bdim_coefficients`.  Now computed once in
  `_fluid_step_kernel_{2,3}d` and reused by both; `_bdim_union_aabb` is reset to `None`
  only after `_compute_bdim_coefficients` returns.
LP5. ~~**Harmonic mean for variable viscosity**~~ ✅ (2026-06-11). `diffusion.py:
  variable_laplacian` now uses the harmonic mean `2·νᵢ·νⱼ/(νᵢ+νⱼ)` for face viscosity
  instead of the arithmetic mean.  More accurate for strongly varying viscosity
  (Carreau/Herschel-Bulkley); backward-compatible (identical for constant ν).
LP7. ~~**F3 cache CC normals**~~ ✅ (2026-06-11). `forces_method1/2/2_3d` now store
  `self.normal_{x,y,z}` on the first call in a step; `_release_bdim_fields` clears them
  after the step.  On the python path `_recompute_mu_normals` already sets them, so no
  change.  On the kernel path and for implicit coupling sub-iterations this avoids a
  redundant `torch.gradient` call per iteration.
