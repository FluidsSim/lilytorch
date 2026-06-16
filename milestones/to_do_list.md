# Lilytorch — TODO

Memory vars: `sdf_val_{u,v,w}`, `{u,v,w,p}0`, `n{x,y,z}_{u,v,w}`, `body_{u,v,w}`,
`mu{0,1}_{u,v,w}`, `diff_{u,v,w}`.

---

# ═══════════════════════════════════════════════════════════
# NEEDS WORK
# ═══════════════════════════════════════════════════════════

# HIGH PRIORITY

HP3. **Polish repo & docs** — review/correct outdated documentation, including `docs/`.

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

# 2D/3D SOLVER UNIFICATION (remaining)

Steps 1-4 + apply_forces merge ✅. Remaining:

SU1. **Step 5 — stacked-tensor storage.** Replace `(u0,v0,w0)`, `(nx,ny,nz)`,
  `(mu0_{u,v,w})` etc. with `(D, *grid)` tensors. Deepest refactor (every callsite,
  FARMS bridge, kernels, plotting, HDF5). Needs explicit user sign-off.

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

AP1. [ ] **Define a `RigidBodyBackend` adapter** = the only surface BDIMhandler needs:
      `get_body_poses() -> pos,quat,com`, `get_body_velocities()`,
      `get_body_mass_inertia()`, `apply_force_torque(body,F,T)`, `step(dt)`, `gravity`.
      Refactor BDIMhandler's ~10 MuJoCo-specific access sites behind it.
      **This single refactor both decouples FARMS and enables Isaac.** Do it FIRST.
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
  (`run_compile_advdiff_bench.py`) to `lilytorch.src.advection` (drop-in: identical
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
