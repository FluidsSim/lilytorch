
# instructions
Read the HIGH PRIORITY todo list to implement (in to_do_list.md) and start to work on from top to bottom (higher to lower priority list). Then give me a step to step guide for testing the various implementations.
<!--
be independent and self testing. Install the necessary packages explained in the README under the installation instructions (you can install pytorch in C++ mode, but you also must install the FARMS packages). Read the HIGH PRIORITY next steps to implement in the repository and start to work on from top to bottom (higher to lower priority list). -->


---- MEMORY VARS -----
sdf_val_u, sdf_val_v, sdf_val_w
u0, v0, w0, p0
u, v, w, p
nx_u, ny_u, nz_u
nx_v, ny_v, nz_v
nx_w, ny_w, nz_w
body_u, body_v, body_w
mu0_u, mu1_u
mu0_v, mu1_v
mu0_w, mu1_w
diff_u, diff_v, diff_w


# HIGH PRIORITY:
- Dropping _compute_variable_density_coefficients: Opus 4.7 suggested to completely remove the rho = rho_body + (rho−rho_body)·μ₀ terms (unless dealing with problems where rho_body<<rho_fluid, i.e. where added mass causes instabilities. It suggested to use the original BDIM2 implementation for c_h = dt·μ₀/(rho_body + (rho−rho_body)·μ₀)
- Change the printing to use pylog, with info, warning and error colors.

- **(suggestion) Softmin SDF blending of multibody body velocities.**
  Multibody swimmers (eel, salamander, pleurodeles) currently force
  `bdim_mu0_projection = False` to avoid the blow-up that the BDIM2
  μ₀-weighted coefficient `c = dt·μ₀/ρ_eff` causes when bodies (rigid
  links connected by joints) interpenetrate.  Root cause: per-cell
  *winning-body* assignment makes `body_u/v/w` discontinuous on the
  SDF-intersection curve between two links moving at different velocities;
  in low-μ₀ cells that discontinuity becomes a large `div(u*)` the
  degenerate μ₀-weighted projection cannot remove.  **Fix**: replace
  winning-body with a smooth SDF-weighted blend
  `u(x) = Σ_k exp(-(sdf_k - sdf_min)/σ)·u_k / Σ_k exp(...)`
  so `body_u` is continuous across intersection surfaces.  This would
  let multibody swimmers keep `bdim_mu0_projection = True` (better
  BDIM2 surface accuracy) without blowing up.
  Status (2026-05-28): a pure-Python prototype was implemented + removed
  — correctness was verified on a synthetic 2-body case (σ→0 recovers
  winning-body; σ=ε matches the softmin formula to 1e-10), but it
  slowed kernel-mode steps too much (per-step Python re-iteration over
  bodies with `body.sdf(...)` evaluation) and the visible improvement
  on the salamander_gamepad swim_2d sim did not justify the cost.
  Required for viability: a **fused CUDA accumulation kernel** that
  mirrors `streaming_sdf_min_rho_3d_multi_kernel`'s per-(body, cell)
  iteration but uses `atomicAdd` of `(exp(-Δ/σ)·u_k, exp(-Δ/σ))` into
  per-stagger accumulators, plus a finalize divide.  Cost would then
  be ~1 SDF eval per body-cell + a few FMAs + one `exp` (≪ 5% of step
  time).  Wiring (config knob, BDIMhandler callable stash,
  fluid_step hook) is documented in the conversation that removed the
  prototype — re-implement starting from a CUDA kernel rather than the
  Python loop.
  **Engineering gotcha for any future re-implementation**: kernel mode
  deliberately skips `_bind_staggered_grids` (memory optimisation —
  see `body.py:_StaggeredGrids` docstring), so `comp.Xu_stag/Yu_stag
  /Zu_stag` are NOT allocated.  Build per-AABB staggered coords on
  the fly via ``torch.meshgrid(fluid_solver.x[i0:i1] - h/2, ...)`` —
  never touch `comp.X*_stag` (the previous Python prototype crashed
  on pleurodeles 3D for this reason).

structure should be as follows: BDIMhandler should have a compute a body update

Review options and propose what to do. This should have a careful modifications in all the examples scipts in farms_examples/.
- Polish the repository, review and correct outdated documentation, also in the docs/ folder.

## Tier 1 — Python-only (each <4 hours, low risk)
- [ ] **T1a Inline `_tvd_face` into `abdquickest`/`cubista` + chain remaining temps in-place.**
  Saves ~0.5 GiB peak inside the scheme function. Bit-exact testable. Files: `lilytorch/src/adv_diff.py`.
- [ ] **T1b Free `self.div` immediately after `_poisson_solve` returns** (it's only read once
  for the RHS, never used after). Saves ~0.5 GiB transient peak. File: `lilytorch/src/solver.py:project()`.
- [ ] **T1c Consolidate `_compute_variable_density_coefficients` 2-D/3-D return paths** so the 2-D
  `ch_cc` temp can be dropped earlier on the FFT Poisson path. Marginal effect in 3-D.

Expected combined: ~1 GiB off peak → **~8 GiB peak alloc, ~9.2 GB nvidia-smi.**

## Tier 2 — Targeted CUDA work (1-3 days each)
- [ ] **T2a Fused CUDA kernel for `_flux`.** One kernel reads `fv` + 4 cells of `p`, computes the
  QUICK/ABDQUICKEST stencil in registers, writes one flux value. Drops
  `denom, rf, psi, B1, B2, cond, flux_in, cat output` materialisation entirely. **~3 GiB peak**
  (adv-diff drops 8.82 → ~5.5 GiB) and 5-10× speedup as a side effect. Only pays off when
  combined with multigrid optimisation (T3 below), otherwise multigrid still sets the peak.
- [ ] **T2b Dirty-AABB-sized Kernel-A temps** (the originally-deferred #1 from the May session).
  `sdf_*_tmp` and `b*_tmp` go from full-grid (6 × ~543 MB = 3.26 GiB) to AABB+halo (~10 MB
  for one fish at 512³). Needs CUDA kernel changes in `streaming_sdf.cu` (init, min_rho,
  decode, BDIM kernels — AABB-local indexing with 1-cell halo handling). Saves ~3 GiB at
  marker 4 `cur` but **doesn't move the peak** until T2a is done.
- [ ] **T2c Two-pass Kernel B for `primes` elimination.** Write BDIM result to dirty-AABB-sized
  scratch (~0.5 MB), then copy back to `u0[AABB]`. Lets us `del primes` immediately after
  `self.u0.copy_(primes[0])`. Saves 1.5 GiB at marker 4 `cur`, **no peak movement until T2a.**

## Tier 3 — Multigrid memory (each ~1 day)
- [ ] **T3a Eliminate the `div` field.** Currently `divergence(u, v, w)` returns a full-grid
  tensor stored on `self.div`. Inline it into the multigrid RHS computation so no full-grid
  `div` tensor exists. Saves 543 MB persistent + ~0.5 GiB transient.
- [ ] **T3b Preallocate V-cycle coarse-level pyramid.** `_vcycle_rbgs_3d` currently calls
  `torch.zeros(coarse_shape, …)` inside the recursion (one alloc per level per V-cycle).
  Allocate a static pyramid at `__init__` and reuse. Saves ~0.5-1 GiB transient at fine level.
- [ ] **T3c Try `solve_mgcg` with `precond_vcycles=1`.** MGCG converges with fewer V-cycle calls
  and the working set inside CG is smaller (no need to keep restricted face arrays alive across
  recursion). Already supported via `poisson_method="mgcg"` — just benchmark and decide.

## Tier 4 — Bigger architectural (1+ week)
- [ ] **T4a fp16 SDF + body-vel** for kernel-mode temps. Adv-diff and project stay fp32.
  Saves ~1.5 GiB persistent (sdf_val) + ~1.5 GiB transient (if combined with T2b).
- [ ] **T4b Mixed-precision velocity fields.** fp16 storage for u0/v0/w0, fp32 compute.
  ~1 GiB persistent + ~0.5 GiB transient. Needs careful BDIM2 + multigrid stability checks.
- [ ] **T4c Try `--poisson_compile` flag.** CUDA-graph capture of the V-cycle. Mostly helps
  allocator pool fragmentation; effect on peak varies — just measure.

## Recommended sequence
1. T1a → ~8.7 GiB peak (free, an hour).
2. T3a + T3b → ~7.5-8 GiB peak (half-day, eliminates multigrid as the limiting factor).
3. Stop and remeasure. If headroom is enough, stop here.
4. Else T2a (fused `_flux` kernel) → ~5.5 GiB peak + adv-diff speedup. Then T2b + T2c become
   the new limiting factors.

# LOW PRIORITY:
- Can advection/poisson solvers be improved? I.e. by implementing a cuda/c++ kernel instead of torch.compile?
- Test an analytical 2d swimmer simulation of the salamander swimming in 2d (use the control.py and gamepad.py extension and figure out how to set it up)
- Consider Crank-Nicolson for diffusion. Current explicit diffusion has stability limit dt < h²/(2ν·ndim). Not a bottleneck now (dt_diff ≈ 4.2s ≫ dt_cfl), but becomes relevant if dt is increased aggressively per A5.


# LONG TERM GOALS:
- Velocity gradients for the stress tensor use central differences, which degrade to
1st-order near immersed boundaries. One-sided or ghost-cell stencils for cells near
the body would improve force accuracy and reduce oscillations.
- The expressions `0.5*(1 + d/ε + sin(π·d/ε)/π)` have cancellation when d ≈ ±ε.
A 5th-order Hermite smoothstep is more robust numerically and avoids sin/cos.
- How to handle bodies outside the water (at the interface). Volume of fluids methods (?)
- Add sph simulation support (?)
- Strongly coupled solver - Monolithic fluid multi rigid body solver (?) --> hard, it would require dropping Mujoco

- **Semi-implicit FSI coupling (Aitken-Picard ± Robin-flavoured update).**
  Address added-mass instability when explicit + `force_relaxation` is not enough
  (light-mass swimmers, floppy joints, large dt, anything with m_added/m_body ≳ 2).
  Current state: pleurodeles 3D + salamander 2D both run stably via the
  `force_relaxation` low-pass on the force feedback (β=0.3 sufficient). Implicit
  coupling is the principled answer if relaxation's transient lag becomes a problem
  or stability margin needs to be guaranteed across all configs.

  **Why Colonius/Taira IBPM doesn't transfer directly**: their monolithic saddle-
  point system treats the surface force as a Lagrange multiplier enforcing a sharp
  Dirichlet no-slip. BDIM's velocity blend `u* = (1-μ₀)u_fluid + μ₀ u_body` has no
  Lagrange multiplier — the body's effect on the fluid is implicit in the blending,
  there's no sharp `Γ` to impose a BC on, and MuJoCo's body solver is a black-box
  Neumann interface. The Colonius algebra doesn't apply.

  **Practical extension (BDIM-MuJoCo)**:
  1. **Aitken-Picard sub-iteration** (~2-3 days). Within each timestep, sub-iterate
     fluid + body until (force, body_state) are self-consistent. Pseudo-code:
     ```
     for step n:
         checkpoint MuJoCo state, fluid state
         F^{(0)} = F^{n-1}  # extrapolate
         for k = 0, 1, 2, ...:
             apply F^{(k)} via xfrc_applied, sub-step MuJoCo → body_state^{(k+1)}
             stamp u_body from body_state^{(k+1)} into fluid grid
             run BDIM fluid step → compute F^{(k+1)}
             apply Aitken Δ² acceleration on F^{(k+1)}
             if |F^{(k+1)} - F^{(k)}| / |F| < tol: break
             restore MuJoCo + fluid state to start of step
         commit final F, body_state, fluid state
     ```
     Typically converges in 2-4 inner iterations → 2-4× per-step cost.
     Main implementation challenge: MuJoCo state checkpoint/restore (mj_saveState/
     mj_loadState) inside the FARMS lifecycle. Fluid checkpoint is straightforward
     (u, v, [w], p tensor copies).

  2. **Robin-flavoured update** (~1-2 more days, optional acceleration on top of
     Aitken). Replace the plain stamp `u* = (1-μ₀)u_fluid + μ₀ u_body` with a
     Robin-style update:
     ```
     u* = (1-μ₀)u_fluid + μ₀ (u_body + κ · (F_applied^{k} - F_predicted))
     ```
     where κ = compliance estimate (∝ 1/m_body) and F_predicted = current fluid
     force on the current u_body. Encodes "added-mass preview" inside the iteration.
     Typically drops Aitken iteration count from 3-4 → 1-2 (Badia, Nobile, Vergara
     2008 + Causin, Gerbeau, Nobile 2005). Marginal complexity for marginal win;
     only worth it if Aitken alone converges too slowly.

  **When to do this**: only when at least one of the following is true:
   (a) a real swimmer config can't be stabilised by `force_relaxation` at any β
       (i.e., even β=0.05 still explodes — would mean m_added/m_body ≫ 2).
   (b) the transient lag of low-β relaxation (~1/β iters) is unacceptable for the
       application (e.g., gamepad control responsiveness).
   (c) you want a stability guarantee across all parameter regimes for publication
       / public release.
  As of 2026-05-29 none of (a)-(c) apply; relaxation handles every tested swimmer
  config with β ≤ 0.3. Document as a "if you ever need it, here's the recipe"
  rather than a blocker.

- AMR (Adaptive Mesh Refinement) - refine grid only near bodies and in the wake.


# CHECKS
- After projection, the residual divergence is stored (self.div) but never checked.
Adding `div_max = self.divergence(u,v,w).abs().max()` every N steps catches Poisson
under-convergence before it cascades into NaN.


# IMPROVEMENT SUGGESTIONS (from deep code review, March 2026)




## B. STABILITY — Solver Robustness


### B6. Post-projection divergence monitoring



## C. CODE QUALITY

### C7. No type hints
No Python type annotations anywhere. Adding them improves IDE support, catches bugs,
and serves as documentation.


## D. FEATURES

### D1. Checkpoint/restart system
No automatic periodic checkpointing that saves full solver state (iteration, drag records,
body positions, Poisson state). A crash at iter 999k of a 1M-step run means starting over.


---

# IMPROVEMENT SUGGESTIONS (from deep code review, March 2026 — second pass)

## E. NUMERICAL ACCURACY



### E2. All-Neumann Poisson compatibility condition not enforced
When `poisson_bc_type = "neumann"`, the system ∇²p = f is only solvable if ∫f dV = 0.
There is no check or enforcement of this. Floating-point drift in the divergence of u* can
violate it. Subtracting `mean(f)` from the RHS before the FFT/multigrid solve would make the
solver robust and is standard practice.

### E3. Heun body update is called only once per step — coupling is effectively 1st-order
Heun's method calls the advection-diffusion solver twice (predictor + corrector). The body SDF
and velocity are updated only once at the start of the step, so the corrector uses the body
state at time *t*, not *t + dt/2*. The fluid–body coupling is therefore 1st-order accurate
even though the fluid integration is 2nd-order. At minimum, document this as a known limitation;
ideally, update the body to *t + dt* after the predictor and feed it into the corrector.

### E4. `eps = 2*h` hardcoded — should be configurable
The BDIM transition thickness is fixed at `2h` in `solver.py:577` with no config override.
For coarser grids or larger bodies, `3h`–`4h` gives smoother IBM forcing. Add `eps_cells: 2`
(integer) as a solver config key.


## F. PERFORMANCE

### F1. AABB culling for force integration
`_forces_body_batch_3d` (and 2D equivalent) evaluates the smoothed delta δ(sdf − ε) across the
**entire** domain for every body, but δ is non-zero only within ε of the body surface. Each body
already has a bounding box. Slice the grid to each body's AABB + ε before the force integral.
For a small swimmer in a large pool this is a 10–100× reduction in flops.

### F2. Drag records allocate on GPU with full `nt` pre-allocation
`viscous_drag_record` and `pressure_drag_record` pre-allocate `(n_bodies, n_force_comp, nt)` on
device (solver.py:853). Move them to CPU pinned memory and use `tensor.pin_memory()` + async
`copy_()`. This frees device memory and overlaps force accumulation with GPU compute.

### F3. CC normals recomputed in every force call
Both `forces_method2` and `forces_method2_3d` call `compute_normals(sdf_val)` via
`torch.gradient` on the full SDF each step for the cell-centred normals. The staggered normals
(`normal_x_u`, etc.) are already cached from the body update. Cache the CC normals too, computed
once per body update alongside the staggered ones.

### F4. `FlowDiagnostics` default `check_every = 1` is expensive
With `check_every=1`, every step computes `vorticity_fn()` (another `torch.gradient` over the
full velocity field). The config key `diagnostics_every` is not set in `base_sim_config.py` so
it silently defaults to 1. The default should be `save_every` or at least 10.


## G. CORRECTNESS / ROBUSTNESS

### G1. Restart is incomplete — drag records and body state are not saved/restored
`_load_initial_conditions` restores u, v, [w], p but not: `viscous_drag_record`,
`pressure_drag_record`, Adams-Bashforth previous-step flux, or body initial state from FARMS.
A warm restart will produce a different trajectory from a continuous run at the same point.
A complete checkpoint should serialise all solver state plus the FARMS body poses.

## H. ARCHITECTURE


### H2. `FluidSolver.__init__` is ~500 lines — extract into setup helpers
The constructor initialises the grid, time integration, non-Newtonian models (Carreau,
Smagorinsky, yield damping), sponge layer, both Poisson solvers, force buffers, diagnostics,
output paths, and initial conditions. Extract into private methods: `_setup_grid()`,
`_setup_models()`, `_setup_poisson()`, `_setup_output()`. Easier to navigate, test, and
override in subclasses.

### H3. `BaseSimConfig.run()` couples config generation and simulation launch
Running a config to inspect the generated YAML requires launching a full FARMS simulation.
Add a `BaseSimConfig.generate_config()` method that returns the parameter dicts without
launching, and have `run()` call it internally. This makes unit testing and dry-runs trivial.

