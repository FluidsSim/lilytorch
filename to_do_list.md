
# instructions

You are working starting from the optimize_speed_memory branch. Checkout from there and create a new branch for your implementations. be independent and self testing. Install the necessary packages explained in the README under the installation instructions (you can install pytorch in C++ mode, but you also must install the FARMS packages). Read the HIGH PRIORITY next steps to implement in the repository and start to work on from top to bottom (higher to lower priority list).


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

# DONE:
- The bdim_forces_3d_multi_kernel recompute the body cc sdf on the fly. I think this is a waste of computation. Indeed i think that forces can be computed together with the union sdf in streaming_sdf_min_3d_multi_kernel. Inside this kernel the body sdf is computed, there is where the delta force functions, and forces can be computed. This should save significant computational time.
- the gen_config_full_pool.py simulation in cpu mode does not seem to utilize multiple cores.
- The nbforces cost analysis plotted in cost_scaling_loglog.png reveals some scaling of the costs of "Other (residual)" and "Body update (SDF)". I do not understand why, since the cropping approach with aabb boxes should (in my view), maintain the same cost at different scales. Unless the domain size remains the same and just the number of grid points increases, in which case the portion of the domain that includes the body increases, so the operations should indeed increase. Please clarify.


# HIGH PRIORITY:
- Simplify methods for cost optimization. After substantial testing of the different methods for running the cost analysis in run_scaling_conditions_pipeline.py the best method is nbforces_opt. Remove all the other methods, except for keeping the old one for reference (no cropping, no batching method for reference).
- Implement cuda kernels for 2d simulations in the style used for 2d simulations and test them (by running a 2d simulation example)
- Run the cost analysis similar to run_scaling_conditions_pipeline.py for 2d simulations.
- Implement 2nd order accurate force method also for the cuda/C++ kernels (currently only in non cuda/c++ kernel mode)
- Implement triquadratic interpolation option similar to that implemented in pytorch_interpolation for evaluating the sdf functions in the cuda/C++ kernels. This should be optionally set by the user via a meta parameter.
- replace pytorch_interpolation with existing precompiled cuda/c++ kernels or write new ones if necessary in the kernel/ folder.
- Combine solver.py and BDIMhandler in a single simulation file (just solver.py). BDIMhandler should only keep whatever is necessary for handling the coupling with FARMS, if possible. Review options and propose what to do.
- Move all non standard computations (sponge layer, carreau, etc) in a dedicated file in src/extras.py
- Polish the repository
- Can advection be improved? I.e. by implementing a cuda/c++ kernel?
- Can poisson solve be improved? I.e. by implementing a cuda/c++ kernel?
- Move force computations in an ad-hoc file (src/forces.py)
- Do a systematic memory cost analysis. I think that the latest method with cuda kernels dynamically rewrites the body velocities and writed the forces computation inside the kernel whenever body properties are needed. This reduces the memory footprint by not storing the sdfs/body velocities of each rigid body (just a unique composite union body properties are stored).

# LOW PRIORITY:
- Test an analytical 2d swimmer simulation
- Consider Crank-Nicolson for diffusion. Current explicit diffusion has stability limit dt < h²/(2ν·ndim). Not a bottleneck now (dt_diff ≈ 4.2s ≫ dt_cfl), but becomes relevant if dt is increased aggressively per A5.


# LONG TERM GOALS:
- How to handle bodies outside the water (at the interface). Volume of fluids methods (?)
- Add sph simulation support (?)
- Monolithic fluid multi rigid body solver (?)
- AMR (Adaptive Mesh Refinement) - refine grid only near bodies and in the wake.


# IMPROVEMENT SUGGESTIONS (from deep code review, March 2026)




## B. STABILITY — Solver Robustness


### B6. Post-projection divergence monitoring
After projection, the residual divergence is stored (self.div) but never checked.
Adding `div_max = self.divergence(u,v,w).abs().max()` every N steps catches Poisson
under-convergence before it cascades into NaN.


### B8. Sponge/buffer layer for outflow
Current BCs are Dirichlet or Neumann only. For external flows with vortex shedding,
reflected pressure waves from the outlet contaminate the near-body solution.
A convective outflow BC or exponential sponge layer (damping toward freestream in the
last ~10% of the domain) is standard practice.

### B9. Polynomial Heaviside instead of trig in `mu_funcs` (body.py)
The expressions `0.5*(1 + d/ε + sin(π·d/ε)/π)` have cancellation when d ≈ ±ε.
A 5th-order Hermite smoothstep is more robust numerically and avoids sin/cos.

### B10. Wall-distance-aware stencils for stress
Velocity gradients for the stress tensor use central differences, which degrade to
1st-order near immersed boundaries. One-sided or ghost-cell stencils for cells near
the body would improve force accuracy and reduce oscillations.


## C. CODE QUALITY

### C1. No formal test suite
All "tests" are standalone scripts under src/old/. Missing: pytest structure,
convergence-rate verification (h-refinement), manufactured solution tests,
BDIM conservation checks, analytical force validation (Stokes drag).



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

### F5. No `torch.compile` warm-up before the simulation loop
Compiled kernels (adv-diff, BDIM meta, forces) trigger JIT tracing on their first real call,
causing a latency spike at iteration 0. Add one dummy forward pass (with detach/no_grad) during
`initialize_episode` — before the loop — to absorb tracing overhead.


## G. CORRECTNESS / ROBUSTNESS

### G1. Restart is incomplete — drag records and body state are not saved/restored
`_load_initial_conditions` restores u, v, [w], p but not: `viscous_drag_record`,
`pressure_drag_record`, Adams-Bashforth previous-step flux, or body initial state from FARMS.
A warm restart will produce a different trajectory from a continuous run at the same point.
A complete checkpoint should serialise all solver state plus the FARMS body poses.

### G2. Async I/O futures are never pruned — potential memory leak
`_io_futures` is an ever-growing list (solver.py:915). Over a 100k-step run with
`save_every=100`, 1000 `Future` objects accumulate. Completed futures should be pruned
periodically (e.g., every `save_every` steps), and uncaught I/O exceptions in futures should
be surfaced rather than silently dropped.


## H. ARCHITECTURE

### H1. `forces_method1` is dead code
`force_method` defaults to `"method2"` and 3D always uses `forces_method2_3d`. `forces_method1`
(solver.py:1036) requires `body.cnt_update`, `body.mask`, `body.ds` — none of which are
populated in any current example. The ~80-line method should be removed or formally deprecated
with a docstring note to avoid misleading future contributors.

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

