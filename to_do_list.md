# Lilytorch — TODO

Memory vars: `sdf_val_{u,v,w}`, `{u,v,w,p}0`, `n{x,y,z}_{u,v,w}`, `body_{u,v,w}`,
`mu{0,1}_{u,v,w}`, `diff_{u,v,w}`.

---

# HIGH PRIORITY

- **Two-phase body∩interface band instability (elongated static float)** — a body held in
  a STATIC FLOAT straddling the air/water interface blows up (DSYHS boat `it~51`,
  volume-matched 11:1 spheroid `it~47-61`), while a *compact* float (sphere) and the
  *descending/entry* regime (three-sphere drop, water-entry) are STABLE. Localised to the
  **air–water–solid triple point / waterline band**, with a ~static body — it is **not**
  FSI (the free body moves <0.5 mm before blow-up), **not** the propeller/masses/CFL/mesh/
  resolution/coupling, and **not** the pure two-phase transport (body-FREE hydrostatic is
  stable for the same scheme). See `memory/project_toy_boat_two_phase_debug.md` for the full
  bisection.
  - ✅ **FIXED — air-transparent body** (`two_phase.air_transparent_body`, default ON).
    Masks the BDIM fluid fraction μ₀ by the VOF water fraction α: μ₀_eff = 1 − α·(1−μ₀).
    In water (α=1) → normal BDIM; in air (α=0) → body transparent (μ₀_eff=1). Eliminates
    the triple-point singularity by ensuring c = dt·μ₀_eff/ρ never collapses to 0 in the
    air phase. Physically justified: air forces ~1000× smaller than water. Implemented in
    both kernel path (`_rescale_kernel_coeffs_two_phase`) and Python path
    (`_compute_bdim_coefficients` + `_apply_bdim_all_axes`). Verified stable: sphere
    (4000/4000), 11:1 spheroid (4000/4000), single-body hull at waterline (>3 min).
  - ⚠️ **Multi-body convex-hull seams** remain unstable at coarse resolution (h≥0.05).
    The hull, keel, and rudder convex hulls overlap; the running-min SDF union hard-switches
    body velocity at the seams → grid-scale divergence → blow-up. Single-body hull-only is
    stable; multi-body only stable at fine resolution (h=0.00333, `gen_configs_small.py`).
    `body_velocity_blend_eps_cells` made it worse at 2.0 cells; untested at other values.
  - *Tried, did NOT cure:* GFM/sharp-interface, `p_rgh` reduced-pressure, interface damping,
    density smoothing, body-aware VOF (reinit / flux-mask / μ0-gated gravity), consistent
    momentum alone, three-phase density EVERYWHERE (hurts — `ρ_s` must stay out of the
    momentum/transport), `ρ_flow·g` gravity (hurts on BDIM), naive rigid re-imposition.
  - *Path forward (research-grade, likely needs core / new FSI architecture):* explicit
    **contact-line / triple-point** treatment at the waterline–body corner; or the
    **DLM / Brinkman-penalization** multiphase-WSI scheme (Nangia 2019 `arXiv:1901.07892`;
    Bhalla IBAMR `arXiv:1904.04078`; `github.com/IBAMR/IBAMR`). Note BDIM ≈ Brinkman, so the
    fix should be replicable in the BDIM framework rather than swapping methods.

- 🔴 **Lagrangian force method incompatible with two-phase flow** — `force_method="lagrangian"`
  gives ~3× spurious buoyancy + large pitch torque for surface-straddling bodies, even at
  fine resolution (explodes at iter 12, h=0.00333). Eulerian (`force_method="eulerian"`)
  works correctly. Root cause NOT identified despite testing: pressure gauge (all-Neumann
  mean subtraction → negative air pressure), `zero_pressure_inside`, `convexify`, and
  `air_transparent_body`. The surface integral ∮ p n dS over the marching-cubes
  triangulation diverges violently from the volumetric band integral ∫ p n δ_ε(φ) dV.
  **Needs investigation:** compare forces from both methods on a single step, check marching-
  cubes mesh quality (watertightness, normal consistency), verify interpolation stencil
  doesn't sample interior cells with unphysical p. Currently workaround: use eulerian.

- **Polish repo & docs** — review/correct outdated documentation, including `docs/`.

- ~~**Wire in `FlowDiagnostics`.**~~ DONE. `FlowDiagnostics` moved to its own module
  `lilytorch/src/diagnostics.py` (out of solver.py), instantiated in `FluidSolver.__init__`
  when `diagnostics_every > 0`, called from `finalize_step` on the post-projection field,
  and saved to `diagnostics.h5` at the end of `run_sim`/`run_from_initial`. Default
  `diagnostics_every = 100` in `base_sim_config.py` (0 disables). Subsumes the old F4 item.

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

- ~~**T1a**~~ DONE. Inlined `_tvd_face` into `van_leer`/`abdquickest`/`cubista` in
  `advection.py`, chaining in-place on the owned `psi` and reusing the live `denom`;
  `_tvd_face` helper removed. Verified bit-exact (fp32+fp64, incl. denom≈0 branch).
- ~~**T1b**~~ DONE. `div` is now a local in `solver.py:project()` (was a persistent
  `self.div`) and is `del`-ed right after each `_poisson_solve` returns, before the
  gradient/correction allocations. ~0.5 GiB transient + removes a persistent field.
- **T3a** Eliminate the `div` field — inline `divergence()` into the multigrid RHS.
  ~543 MB persistent + ~0.5 GiB transient. (Persistent part already captured by T1b.
  Needs kernel-path re-baseline to confirm transient peak benefit — python-path multigrid
  solve was only ~1 GiB over resident.)
- **T3b** Preallocate the V-cycle coarse-level pyramid at `__init__` instead of
  `torch.zeros` inside the recursion. ~0.5-1 GiB transient. (Python-path multigrid solve
  stayed ≤3.58 GiB; re-baseline on kernel path before judging peak benefit.)
- **T2a** Fused CUDA `_flux` kernel (QUICK/ABDQUICKEST stencil in registers; the
  `_flux` in `advection.py`, NOT `adv_diff.py`). ~3 GiB + 5-10× adv-diff speedup.
  NOTE: there is currently **NO native advection kernel** — `AdvDiffSolver.solve` is
  pure-PyTorch (many ATen ops + full-grid temps), optionally `torch.compile`d (which did
  NOT lower peak in testing). T2a = write a new `.cu` flux kernel. On the python path
  advection was only ~0.7 GiB transient; the "~3 GiB" estimate likely refers to the
  kernel path — re-baseline on kernel mode first.
- **T2b** Dirty-AABB-sized Kernel-A temps (`sdf_*_tmp`, `b*_tmp`: full-grid → AABB+halo).
  Needs `streaming_sdf.cu` changes; no peak movement until T2a.
- **T2c** Two-pass Kernel B for `primes` elimination (write to AABB scratch, copy back).
  ~1.5 GiB; no peak movement until T2a.
- **T4** (architectural, 1+ wk each, measure first): fp16 SDF+body-vel temps;
  mixed-precision velocity fields (fp16 storage, fp32 compute); `--poisson_compile`
  CUDA-graph capture.

---

# 2D/3D SOLVER UNIFICATION (remaining)

Steps 1-4 + apply_forces merge DONE. Remaining:

- **Step 5 — stacked-tensor storage.** Replace `(u0,v0,w0)`, `(nx,ny,nz)`,
  `(mu0_{u,v,w})` etc. with `(D, *grid)` tensors. Deepest refactor (every callsite,
  FARMS bridge, kernels, plotting, HDF5). Needs explicit user sign-off.
- **Step 6 remainder — merge BDIMhandler `_update_2d/_3d` +
  `_update_*_streaming_multi`** (~1000 lines). Replace per-plane branches with a
  `self._sim_axes` index array; needs full FSI regression coverage.

Per-step rules: branch from `optimize_speed_memory`, one PR per step, validate 2D
(`_1guillasim` pinned) + 3D (jellyfish) + cost_analysis (<5% wall-clock regression),
rel-err <1e-6 on integrated quantities. No semantics changes.

### Kernel parity (remaining minor)
- ~~**K9**~~ DONE. Added `is_cuda` TORCH_CHECK to `apply_bcs_2d_cuda` (mirrors 3D).
- ~~**K10**~~ DONE. `apply_bcs_2d/3d_kernel` now compute `src_lin` unconditionally
  (Dirichlet's value is harmlessly discarded), dropping the dead `src_lin = 0` init and
  the `if (kind != 1)` branch. Rebuilt `_C.so`; CPU↔CUDA parity exact (fp32+fp64).

---

# PER-STEP HOT-PATH OVERHEAD (measure first — found 2026-06-05 while doing T1/diagnostics)

These run on EVERY step; none is measured for wall-clock cost yet. Time them
(e.g. with the cost_analysis harness) before/after gating.

- **H1 per-step `torch.cuda.empty_cache()`** — `solver.py` `finalize_step` calls it every
  step ("to reduce nvidia-smi usage"). Confirmed via `bench_memory.py`: `cur` drops to the
  resident floor after every step → the allocator cache is dumped and re-grown each step
  (churn + fragmentation risk). Try gating to every N steps (or off) and time it. Likely a
  real throughput cost; trades speed for a prettier `nvidia-smi`.
- **H2 per-step host sync in `check_explosion`** — `torch.stack([isfinite…]) … .cpu().numpy()`
  every step is a device→host sync on the critical path (pipeline stall). Throttle to every
  N steps (reuse the `diagnostics_every` cadence idea). Blow-ups don't need per-step detection.
- **H3 `diagnostics_every=100` is now default-ON** in `base_sim_config.py` (this session).
  Adds a small recurring vorticity/divergence + host-sync cost. Defensible given the
  blow-up-debugging history, but RATIFY: keep at 100, or set 0 (opt-in)?
- ~~Delete legacy `adv_diff.py`~~ DONE (this session) — repointed the lone importer
  (`run_compile_advdiff_bench.py`) to `lilytorch.src.advection` (drop-in: identical
  `AdvDiffSolver` API), removed the file, fixed the "kept on disk as legacy" docstrings.

---

# LOW PRIORITY

- Analytical 2D salamander swimmer sim (via `control.py` + `gamepad.py`).
- Crank-Nicolson diffusion — current explicit limit `dt < h²/(2ν·ndim)` is not a
  bottleneck now, relevant only if dt is pushed aggressively.
- **eps configurable** — BDIM transition thickness is hardcoded `2h`; add `eps_cells`
  config key (3h-4h smoother on coarse grids).
- **F1 AABB cull force integration** — δ(sdf−ε) is evaluated over the whole domain per
  body but is nonzero only within ε. Slice to each body's AABB+ε. 10-100× for small
  swimmers in big pools.
- **F3 cache CC normals** — recomputed via `torch.gradient` every force call; cache
  alongside staggered normals at body update.
- **F2** drag records: CPU pinned memory + async copy instead of GPU `nt` pre-alloc.

---

# LONG TERM

- **LES for high-Reynolds** — extend the existing Smagorinsky SGS model into a full LES
  workflow (WALE/dynamic-Smagorinsky options, wall treatment) for turbulent high-Re
  regimes where DNS is intractable.
- **AMR (Adaptive Mesh Refinement)** — refine the grid only near bodies and in the wake;
  the enabler for high-Re cases at tractable cost (pairs with LES).
- **Near-boundary stress stencils** — velocity gradients use central differences,
  degrading to 1st-order near immersed bodies. One-sided / ghost-cell stencils would
  improve force accuracy and reduce oscillations.
- **5th-order Hermite smoothstep** for the BDIM delta — `0.5*(1+d/ε+sin(πd/ε)/π)` has
  cancellation at `d≈±ε`; Hermite is more robust and drops sin/cos.
- **2nd-order body coupling** — body SDF/velocity are updated once per step, so Heun's
  corrector uses body state at *t* not *t+dt/2* → coupling is effectively 1st-order.
  Update body to *t+dt* after the predictor and feed the corrector.
- **Checkpoint/restart** — periodic full-state save (iteration, drag records,
  Adams-Bashforth flux, body poses) so a crash at iter 999k of a 1M run isn't fatal.
  Current `_load_initial_conditions` restores only `u,v,[w],p` → warm restart diverges
  from a continuous run.
- SPH simulation support (?).
- Monolithic strongly-coupled fluid + multi-rigid-body solver (?) — hard, would require dropping MuJoCo.
- Refactor: extract `FluidSolver.__init__` (~500 lines) into `_setup_grid/_models/
  _poisson/_output`; add `BaseSimConfig.generate_config()` (dry-run YAML without launch);
  add type hints.
