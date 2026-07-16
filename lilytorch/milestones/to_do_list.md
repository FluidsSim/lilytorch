# Lilytorch — TODO

Memory vars: `sdf_val_{u,v,w}`, `{u,v,w,p}0`, `n{x,y,z}_{u,v,w}`, `body_{u,v,w}`,
`mu{0,1}_{u,v,w}`, `diff_{u,v,w}`.

---

# ═══════════════════════════════════════════════════════════
# NEEDS WORK
# ═══════════════════════════════════════════════════════════

# questions — ANSWERED (2026-07-16)

Investigated with file:line evidence; each answer feeds a CLEANUP task below.

1. **advect_flux_add vs advect_flux_accumulate** — `accumulate` is the sole
   production kernel (`advection.py:407`, fused all-direction, graph-captured).
   `add` is legacy: the CPU twin's own header says "superseded in production"
   (`advection_flux_cpu.cpp:218-224`); only callers are
   `tests/test_advection.py:61,121`. → strip `add` (CL1).
2. **Fold sl_advect into advection_schemes.h as a scheme?** — **No.** SL is
   departure-point interpolation (samples u,v,w at off-grid points; different
   op signature), not a face-flux scheme; forcing it through the flux dispatch
   would complicate the graph-captured production kernel for zero gain. The
   real duplication is the *samplers* → dedupe those instead (CL10). Keeping
   sl_advect.cu as its own file is fine.
3. **Interpolation duplication** — bilinear/biquadratic/trilinear/triquadratic
   samplers are re-implemented in ~11 translation units under 3 naming
   families: `*_sample_off*` (sl_advect.cu + cpu twin), `*_sample_uniform*`
   (streaming_sdf{,_2d,_direct,_regime_b}.cu + 3 cpu twins), `lf_*`
   (lagrangian_forces{,_2d}.cu + 2 cpu twins). → one `common/interp.h` (CL10).
4. **lagrangian_forces_2d.cu → lagrangian_forces.cu** — separate op
   registrations, copy-paste dimension twins (~300 lines each). File-level
   merge is cheap (CL9); full dimension-templating not worth it.
5. **common/ folder** — headers shared by .cpp and .cu today:
   `advection_schemes.h`, `bc_ops.h`, `strain_rate.h`, `poisson_gauge.h`
   (`poisson_scratch.h` is CUDA-only). Move to `csrc/common/` (CL8); setup.py
   globs only .cpp/.cu so no build change.
6. **Streaming ops** — partially right. Production uses FOUR streaming ops:
   `streaming_sdf_stag_{2d,3d}_resolve` (facade.py:101/128) **and**
   `streaming_sdf_forces_post_{2d,3d}` (forces.py:361/618). Dead: the
   `_direct` pair — `cuda/streaming_sdf_direct.cu` is a whole-file delete, but
   their CPU impls live *inside* `streaming_sdf_cpu{,_2d}.cpp` which also host
   the live forces_post ops → per-op surgery, not whole-file (CL2).
7. **body.py stale methods** — verified-dead (definition-only grep hits,
   double-checked): `Body.compute_normals_3d_batched` (1193),
   `Body.mu_funcs_batched` (1215), `Body.phi` (1245),
   `BodyFishExperimental.thk_liu` (2123), `BodyMesh.resample_closed_contour`
   (2287), `CompositeBodyMesh.gaussian_kernel` (2816), module fns
   `resample_contour_exact_spacing` (716), `compute_inertias_2d` (787), whole
   class `COMPOSITEmesh2sdf` (1027-1079, never instantiated — deleting it also
   resolves LP10), and the broken `body_type == "composite_segment_body"`
   branch (934-941: `CompositeSegmentBody` is defined nowhere → latent
   NameError). **NOT dead despite looking it** (eval/lambda/torch.compile
   reachable — do NOT delete): `rotate_grid_3d`, `capsule_3d`, `box_3d`,
   `sdUnevenCapsule`. → CL3.
8. **eps consistency** — THREE sources: `solver.eps = eps_multiplier*h`
   (solver.py:187), `comp.eps` (synced at solver.py:543 +
   two_phase_solver.py:218), and `comp.bodies[i].eps` (constructor default
   0.05, **never re-synced**). Consumers reading the *per-child* eps:
   forces.py:317/447/582/755 (δ-width!), two_phase_solver.py:817 (heaviside),
   BDIMhandler.py:549 (AABB margins). They coincide today only because
   body_from_yaml is passed eps=self.eps — a coincidence, not an invariant.
   Fix: solver.eps is the single authority; `Body.eps` becomes a property
   delegating to one owner (or one `_sync_eps()` iterating children); collapse
   the dual kernel args (`comp_eps`+`eps_mw`, `eps_body`+`eps_solver`) to one
   value. → CL11.
9. **facade.py** — only real content is the grow-only
   `_priv_cache`/`_regime_b_priv` buffer pool + two dirty-AABB early-out
   guards; sole production caller is BDIMhandler.py:1718/1733. Move pool +
   guards into native.py (as `native.body_update_{2,3}d`), update tests
   (test_forces.py:20, test_per_body_buffers.py:45,241) and keep the
   monkey-patch seam in `validation/cost_analysis/run_cost_analysis.py:389-407`
   working (it patches the name imported into BDIMhandler). `USE_REGIME_B_ONLY`
   is never read; `_native_body_update_*` aliases have no callers. → CL4.
10. **Force noise** — causes: δ-band force integration on a non-body-fitted
   grid + 1st-order near-boundary gradients + pressure-band oscillation.
   Levers, cheapest first: (a) widen BDIM eps (LP3 `eps_cells` 3-4h — smoother
   δ, slightly diffused boundary); (b) LT4 Hermite smoothstep kernel (kills
   the d≈±ε cancellation); (c) temporal filtering of the *applied* force —
   `force_relaxation` already exists in BDIMhandler (0.5 used in the HP5b
   diag, cycle-mean preserved); (d) `force_method` Lagrangian (marker-based,
   different noise character); (e) LT3 one-sided near-boundary stress stencils
   (the real fix, most work). → CL12 benchmarks the levers.

# CLEANUP (repo consolidation — per-task model assignment)

Model policy: `[fable]` (Claude Fable/Opus) = judgment-heavy / fp-parity- or
correctness-sensitive; `[deepseek-v4]` = mechanical with a precise spec.
**Every phase gates on:** rebuild ext (`python setup.py build_ext --inplace`) →
full pytest suite (baseline ~372 pass / 1 skip) → one coupled smoke sim.
Lesson encoded in CL3/CL10: "dead" python/CUDA duplicates have repeatedly been
load-bearing — trace eval/lambda/compile reachability before deleting.

## Phase A — dead-code deletion (independent, any order)

CL1. `[deepseek-v4]` **Strip `advect_flux_add`** ✅ (2026-07-16): op def
  ops.cpp:346, CUDA kernel cuda/advection_flux.cu:87/485, CPU impl in
  advection_flux_cpu.cpp (+ its "superseded" doc header), wrappers
  native.py:1200/1219. Rewrite tests/test_advection.py oracles against
  `advect_flux_accumulate` (keep per-scheme coverage; fix stale docstring).
CL2. `[deepseek-v4]` **Strip streaming `_direct` ops**: defs ops.cpp:86/99;
  delete cuda/streaming_sdf_direct.cu (whole file); excise the `_direct`
  impls from streaming_sdf_cpu.cpp:1044 and streaming_sdf_cpu_2d.cpp:1033
  (files STAY — they host live forces_post); wrappers native.py:171/198;
  update tests/test_per_body_buffers.py:219/225. Do NOT touch the regime_b
  files or the forces_post ops.
CL3. `[deepseek-v4]` **Delete verified-dead body.py members** ✅ (2026-07-16): (exact list in
  answer 7, honoring its explicit KEEP list). Replace the
  `composite_segment_body` branch with an explicit `ValueError`.
CL4. `[deepseek-v4]` **Dissolve facade.py** per answer 9 (preserve
  `_priv_cache` grow-only semantics + the cost_analysis monkey-patch seam).
CL5. `[deepseek-v4]` **Repo hygiene batch** ✅ (2026-07-16): fix-or-delete
  `lilytorch/src/build.sh` (removed stale `src/kernels/` cleanup lines;
  setup.py is the single build entry point); deleted dangling
  `docs/api/diffusion.rst` automodule (module deleted); dropped deprecated
  `solver_method` from base_sim_config.py; inlined the 30-line
  `src/interpolation.py` re-export shim — all 5 importers now import
  `RegularGridInterpolator` from `native` directly, file deleted; scrubbed
  Warp-era docstring mentions in tests (scene_2d/3d.py, test_forces.py,
  test_poisson_driver.py, test_two_phase.py, test_whole_step_capture_native.py);
  triaged bare skip at test_pose_source.py:128 (replaced `pass` with
  explanatory docstring); moved `milestones/verify_body_update_refactor.py`
  → `tests/`.
CL6. `[deepseek-v4, user sign-off per file]` **Tracked artifacts** ✅ (2026-07-16):
  removed `pleurosim_a0.3.zip` (100 MB), `simulation.hdf5` (35 MB),
  stale Screenshot; added `*.hdf5` to .gitignore.  Working-tree removal +
  gitignore only — NO history rewrite.
CL7. `[deepseek-v4]` **Milestone-doc pruning**: archive/mark-stale the
  Warp-era plans (unified_graph_capture_plan{,2}.md,
  cuda_graph_streaming_forces_spike.md, remove_solver_modes_handoff.md,
  perf_host_bound_plan.md); consolidate the root `/milestones/` dir
  (completed handoffs, incl. cuda_native_port_plan.md) into
  `lilytorch/milestones/archive/`; refresh stale references in this file
  (e.g. TI1 mentions the dissolved `src/kernels/`).

## Phase B — structure moves (after Phase A merges)

CL8. `[deepseek-v4]` **`csrc/common/`**: move advection_schemes.h, bc_ops.h,
  strain_rate.h, poisson_gauge.h (+ poisson_scratch.h for tidiness); update
  includes (.cpp → `"common/x.h"`, .cu → `"../common/x.h"`). No setup.py
  change (it globs .cpp/.cu only).
CL9. `[deepseek-v4]` **Merge lagrangian 2D into 3D files**: concatenate
  cuda/lagrangian_forces_2d.cu → cuda/lagrangian_forces.cu and
  lagrangian_forces_cpu_2d.cpp → lagrangian_forces_cpu.cpp. Keep BOTH op
  registrations; rename colliding statics. Pure file merge, no templating.

## Phase C — dedup + correctness (judgment; after Phase B)

CL10. `[fable]` **Shared `csrc/common/interp.h`**: unify the 3 sampler
  families across ~11 TUs into `__host__ __device__` inline fns; migrate ONE
  family at a time. **fp-parity-sensitive** — the suite gates at 2 ULP and
  `--use_fast_math` already proved capable of breaking 9 gates; after each
  family run the parity tests; any drift >2 ULP → keep that family's local
  copy and document why.
CL11. `[fable]` **eps single source of truth** per answer 8. Cross-cutting
  correctness (forces δ-width, two-phase heaviside, BDIMhandler AABB margins
  all read child eps); verify bit-identical forces on sphere + 1guilla
  before/after.

## Phase D — research / independent

CL12. `[fable]` **Force-noise reduction study** per answer 10: benchmark noise
  spectrum vs {eps_cells 2h/3h/4h} × {sin vs Hermite BDIM kernel} ×
  {force_relaxation 0/0.3/0.5} × {eulerian vs lagrangian force_method} on the
  sphere validation case; promote winners to defaults. Implements LP3 + LT4
  along the way.
CL13. `[deepseek-v4]` **TI1 promotion — pytest bootstrap**: conftest.py +
  pytest.ini collecting lilytorch/tests/, GPU tests auto-skip without CUDA,
  optional GH Actions CPU-parity workflow. Enabler for all phases; can run
  first.
CL14. `[deepseek-v4]` **Reconcile pleurodeles full-3D stub (LP11)**: the "not
  available yet" SDF `pleurosim_v0.3.sdf` now EXISTS in
  examples/sdfs/pleurodeles/ — wire it into gen_configs_swim_full3d.py and
  smoke-run, or mark the example experimental.




# HIGH PRIORITY


- Review the streaming method used currently. Instead of doing atomic min operations in between body sdfs, could we do instead a priority body list defined by the robot sdf file (parent - first priority and childer less). So higher priority bodies get access to the sdf? Would that work?
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

(HP6 — discretely gauge-invariant two-phase band force — ✅ DONE; shipped as
 `force_submethod="deltaH"`. See the DONE section below.)

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

---

# 2D/3D SOLVER UNIFICATION ✅ COMPLETE (2026-06-24)

Steps 1-6 + apply_forces merge ✅. SU1 is bounded-and-done (velocity stacked;
the rest deliberately not stacked — memory). SU2 + its legacy-oracle cleanup ✅
done (see below). **Nothing remaining in this section.**

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

(SU2 — Step 6 BDIMhandler update merge — ✅ DONE; see the DONE section below.
 Legacy-oracle CLEANUP also ✅ DONE 2026-06-24 — see SU2 follow-up note below.)

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
  config key (3h-4h smoother on coarse grids). → implemented as part of **CL12**.
LP6. **F1 AABB cull force integration** — δ(sdf−ε) is evaluated over the whole domain per
  body but is nonzero only within ε. Slice to each body's AABB+ε. 10-100× for small
  swimmers in big pools.
  **PARTIALLY ADDRESSED (status 2026-06-24).** The per-body SDF is now stored
  AABB-sparse (`comp._sdf_sparse[bi] = (aabb, sdf_sub)`, populated by the 2-D/3-D
  Python update), so the per-body distance table is no longer dense. BUT the
  bandwidth-heavy *shared* stress/pressure-force density (`_forces_shared_compiled`)
  is NOT cropped: the crop scaffolding exists in `forces.py` but `u_aabb` is
  hardcoded `None` (~line 466), so it always takes the full-domain branch, and the
  δ-force `sdf_vals` is then reconstructed back to a dense `(B,Ni,Nj)` tensor with
  `_FAR` fill before the δ(sdf−ε) integration → the integration still runs over the
  whole grid. **Remaining work:** drive `u_aabb` from the union of `_sdf_sparse`
  AABBs (+halo), keep `sdf_vals` sparse through the Towers δ-force loop, and index
  the force kernel relative to the cropped slab.
LP8. **F2** drag records: CPU pinned memory + async copy instead of GPU `nt` pre-alloc.
LP9. **Adaptive / CFL-limited timestep** (found 2026-06-23) — `dt` is FIXED for the
  whole run (no `self.dt =` reassignment anywhere in `solver.py`); `diagnostics.py`
  only *warns* on CFL>0.5, it never adapts. The blow-up-prone cases (toy boat,
  surface eel, sphere-near-wall) would benefit from a CFL-limited dt as a safety
  lever — recompute `dt = cfl_target·h/|u|max` each step (or every N steps) with a
  config cap. Caveat: checkpoint/restart (LT6) and the Adams-Bashforth flux history
  assume constant dt, so a variable dt touches both. Keep opt-in (default fixed).
LP10. ~~`COMPOSITEmesh2sdf.transform_3d` NotImplementedError~~ — absorbed by
  **CL3**: the whole `COMPOSITEmesh2sdf` class is dead (never instantiated)
  and gets deleted, resolving this item.
LP11. **Pleurodeles full-3D example is a stub** (found 2026-06-23) —
  `examples/pleurodeles/gen_configs_swim_full3d.py:25` has a placeholder SDF
  filename TODO ("replace once the full-3D SDF is ready"); the example cannot run
  as-is. UPDATE 2026-07-16: the SDF now exists → tracked as **CL14**.

---

# TEST INFRA / REPO HYGIENE (found 2026-06-23)

TI1. **No CI, no test aggregation.** There is no `.github/`, no `conftest.py`,
  `pytest.ini`, or `tox.ini`; the `test_*.py` files (now centralized in
  `lilytorch/tests/`) are run by hand. → tracked as **CL13**. Meanwhile the
  SU1 per-step rule MANDATES "validate 2D (`_1guillasim`) + 3D (jellyfish) +
  cost_analysis (<5% regression), rel-err <1e-6" with nothing automating it.
  Add at minimum a `pytest` config that collects the self-tests (the CUDA ones
  skip cleanly without a GPU via `pytest.importorskip`), and ideally a CI
  workflow running the CPU-path parity oracles on push. Enabler for trusting the
  refactor-heavy `optimize_speed_memory` branch. NOTE: kernel parity tests need a
  built `_C.so` matching the runtime torch — gate or build-in-CI accordingly.

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
  - `examples/base_sim_config.py` + `gen_configs_*` — animat model + scene gen.
  - swimmer controllers (CPG networks, PD controllers).
  - the viewers (all `import farms` for the MuJoCo viewer).

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
AP7. [x] **Irregular scatter/gather** (streaming_sdf, lagrangian_forces) → **Warp**
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
  - **UPDATE 2026-06-24** (`warp_poc/warp_poisson.py`): for the variable-coeff **3-D RBGS**
    the ~1.5× gap was closed AND reversed WITHOUT `wp.tile` — Warp+graph now **0.5–1.0× of
    native (faster) at 32³–128³**, bit-parity 1.3e-7, CPU==GPU. Two opts: (1) flat 1-D
    array addressing (precomputed base + ±stride) instead of 3-D `wp.array` stride-recompute;
    (2) fold homogeneous Neumann into the half-sweep via index clamp (ghost=self at bdry) →
    ZERO BC launches (native pays 2/sweep). NB the 2026-06-14 RBGS gap was vs the hand-TILED
    *2-D* native; the *3-D* native rbgs is flat (untiled), which is why flat Warp matches it.
    `wp.tile` remains the lever only where native itself is shared-mem tiled (2-D rbgs).

**Warp streaming_sdf POC done 2026-06-24** (`lilytorch/warp_poc/`, isolated from main codebase).
  **PARTIAL VERDICT (Kernel A only): for the streaming-SDF kernel Warp is viable — the
  FANNED design matches/beats native at all B, is single-source CPU+GPU, and CUDA-graph
  captured.** This validates ONE of the per-step kernels; the full kernel-mode replacement
  is NOT yet earned — advection, Kernel B (BDIM coeff + normals), Poisson stencils,
  lagrangian_forces, and (critically) PEAK MEMORY are untested. Full testing checklist:
  `lilytorch/warp_poc/VALIDATION_STATUS.md`. Two streaming-SDF designs benchmarked:
  - **(a) Sequential-per-body** (`run_eager`/`run_graph`): B kernel launches (one per body),
    CUDA-graph captures all B into one replay.  Race-free (each thread owns a unique AABB
    cell → conditional compare-swap, no atomics).  Cost scales with B.
  - **(b) Fanned all-body** (`run_fanned_eager`/`run_graph_fanned`): **2 launches, CONSTANT
    in B** (dim = B·max_vol).  Pass B = `wp.atomic_min` on cc/u/v/w across all bodies fanned;
    Pass C = recompute per-body face SDF, write that body's velocity where SDF == stored min
    (bit-identical recompute ⇒ exact winner select).  This is the structural analogue of the
    native 3-pass fanned kernel, but using **float `atomic_min` instead of the uint64
    packed-key atomicMin** (`wp.bit_cast` is absent in Warp 1.14, so the packed key can't be
    built — the equality-decode replaces it).
  - **Parity** (B=1/3/9, eager + graph, 64×32×32 + 128×64×64; pytest 8/8 PASS): both designs
    SDF rel err < 5e-4 (abs < 2 ULP at overlap cells — tie noise only), body-vel rel < 1e-4.
    Eager vs graph: **bit-identical**.
  - **Perf** (RTX 4080 Super, warmup=10, reps=100; ratio vs native, <1.0 = Warp faster):

    | Grid      | B | mode        | native  | seq-graph     | **fan-graph**     |
    |-----------|---|-------------|---------|---------------|-------------------|
    | 64×32×32  | 3 | kernel-only | 0.012ms | 0.010 (0.85×) | **0.007 (0.61×)** |
    | 64×32×32  | 3 | +reset      | 0.026ms | 0.021 (0.80×) | **0.018 (0.69×)** |
    | 64×32×32  | 9 | kernel-only | 0.013ms | 0.029 (2.28×) | **0.008 (0.64×)** |
    | 64×32×32  | 9 | +reset      | 0.025ms | 0.043 (1.72×) | **0.018 (0.74×)** |
    | 128×64×64 | 3 | kernel-only | 0.019ms | 0.017 (0.90×) | **0.022 (1.16×)** |
    | 128×64×64 | 3 | +reset      | 0.037ms | 0.035 (0.94×) | **0.039 (1.05×)** |
    | 128×64×64 | 9 | kernel-only | 0.020ms | 0.031 (1.50×) | **0.019 (0.96×)** |
    | 128×64×64 | 9 | +reset      | 0.035ms | 0.055 (1.56×) | **0.037 (1.05×)** |

  - **Larger grids** (warmup=10, reps=50; with-reset = production-relevant, both pay reset):

    | Grid        | B | mode        | native  | seq-graph     | **fan-graph**     |
    |-------------|---|-------------|---------|---------------|-------------------|
    | 192×96×96   | 9 | +reset      | 0.112ms | 0.089 (0.79×) | **0.088 (0.79×)** |
    | 256×128×128 | 9 | +reset      | 0.426ms | 0.320 (0.75×) | **0.332 (0.78×)** |
    | 384×192×192 | 9 | +reset      | 1.886ms | 1.049 (0.56×) | **1.116 (0.59×)** |

    **Warp's advantage GROWS with grid size** — 0.79× @192³ → 0.59× @384³.  (The
    native *kernel-only* time scales super-linearly at large grids — 0.044→0.241→1.280ms
    for 192/256/384 @B=9, far worse than the ~3× cell-count growth — so the kernel-only
    gap looks even larger, ~0.2–0.5×; not over-claimed since the reset-inclusive row is
    the apples-to-apples production number.  Worth a look why native streaming_sdf scales
    poorly there, but not load-bearing for the Warp verdict.)

  - **Conclusions:**
    * **CUDA graph capture is essential** — eager mode is 3–14× slower (B Python submissions).
    * **Fanned mode (b) is the winner: 0.59–1.16× of native at ALL B and ALL grids, const in B.**
      It resolves the sequential design's small-grid B=9 slowdown (1.4–2.3×) by collapsing
      to 2 launches; at ≥192³ both Warp designs beat native and the lead widens with grid.
    * Sequential mode (a) is fine for B≤3 (fish) but degrades ~linearly with B.
    * The float-`atomic_min` + equality-decode **fully sidesteps** the `wp.bit_cast` gap — no
      need to wait for Warp 1.15.  Cost vs native: Pass C recomputes the trilinear SDF (native
      reads it back from the packed key), but the 2-launch GPU saving dominates.
    * **Single-source achieved:** the same `@wp.kernel` runs on CPU and CUDA (Warp codegen).

AP8. [ ] Port streaming_sdf + lagrangian_forces to Warp and retire the `.cpp`/`.cu` twins.
      **streaming_sdf design is DECIDED (AP7 POC):** use the FANNED float-atomicMin +
      equality-decode kernel (`warp_poc/warp_kernels.py: streaming_sdf_fanned_{min,decode}_3d`),
      2 launches constant in B, CUDA-graph captured — matches/beats native at all body counts.
      Remaining for production: (1) wire `WarpStreamingSDF` into the kernel dispatch in
      `solver.py`/`BDIMhandler` behind a `use_warp_kernels` toggle; (2) port the 2-D variant
      + the blend-eps path (`num_*`/`den_*` softmin) currently stubbed in the POC; (3) port
      `lagrangian_forces` (simpler — atomicAdd scatter, no argmin); (4) keep the `.cpp`/`.cu`
      self-tests as parity oracles before deleting them. AABB-sized output buffers (MP8/T2b)
      compose cleanly since the fanned kernel already indexes by global g.

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
  cancellation at `d≈±ε`; Hermite is more robust and drops sin/cos. → benchmarked/
  implemented as part of **CL12**.
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
