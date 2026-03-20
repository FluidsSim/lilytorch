
MEMORY COUNT 3D

---- FS variables -----
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




python lilytorch/src/video_postprocess.py /data/andreaferrario/ns_data/cylinder_3d/Re_27_laminar --fields omega_z_3d vel_mag_3d --slow-factor 4

ALREADY RUN:

- Run a full self-propelled 3d 1guilla simulation to check if the force computation in 3d works
- Run error analysis for the flow past cylinder in 2d following run_error_analysis_cylinder.py (adapting this script)
- The pool appears above the swimmer position. I want the swimmer to be inside the pool. Also, the water arena should be programmatically be generated to have water inside the pool only. Also the sizes of the pool borders should be scaled with the pool size - make it automatically generate them from the generation files in the config run. It would be nice to have some cool textures for the pool as well.
- Add a proper analusis of the computational cost of the main parts of the code for 1guilla 3d swimming, for example what is the cost of the different parts of the solver, the computation of the sdf properties, the posson solver, the advection, etc. Suggest a small list of cost to tests
- Move the yaml files in a dedicated folder. Fix the yaml files to adhere to the most recent
- I want to understand if it is possible to make a unique shared BDIMhandler for all mujoco simulations (if possible - check that they can share it) 2d and 3d simulations. First check all BDIMhandler files and see how they differ. Clarify if it is possible to define hyperparameters that are generated via the config generation files instead of the BDIMhandler. This would simplified greatly the repo.
- Bilinear/trilinear prolongation instead of piecewise-constant injection — would improve V-cycle quality and reduce CG iterations
- Red-Black Gauss-Seidel instead of Jacobi — 2x smoothing rate per sweep
- Run a computational analysis of the solver for the 1guilla free swimming simulation, in particular i would like a good (paper quality) plot of the cost of the main operations: FARMS step, body update, computation of body interpolation (mu) and normals (grouped together), convection+diffusion, projection (pressure solve). Suggest some more if you think it is worthed including. Add these tests in the validation folder.
- Find memory bottlenecks and potential memory improvements
- torch.compile() on the Jacobi stencil — would fuse the elementwise ops into a single GPU kernel
- Implement faster solver for the Poisson equation, i.e. test different smoother, or preconjugate gradient multigrid method
- Clean the code and suggest improvements for body.py
- i would like to add a ground plane that looks good at the bottom of the pool to avoid seeing the stars. also i want to generate the recording camera in mujoco to be at a good position to see the swimmer
- Speed up force computations
- Cache SDF rotation transforms in 3D updates
In `_update_3d()` (BDIMhandler), the rotation matrix from Euler angles is recomputed for
every body every timestep, then applied to the full SDF grid. For slowly rotating bodies,
cache the previous rotation and only recompute when angle change exceeds a threshold.
- Run 2d sphere coquerelle and gazzola tests again
- Pre-allocate operator output buffers in operations.py
`gradient()` creates `torch.zeros_like(var)` for dvar_dx/dy/dz every call. Same for
`divergence()` and `vorticity()`. Called multiple times per Heun sub-step. Pass
pre-allocated buffers or cache them on the solver.
- Run the drag cylinder test in 2d
- I prefer the old 2d plots with white backgroud. Please return to those
- It seems that many of the functions/parameters in the configs for running the simulations in farms_examples are shared. I think it would be better to have a single common config class and that each run config can be a class that inherits this master class and modifies its attributes for its specific simulation settings. Do that
- set a sphinx documentation system, with API, mathematical formulas, scheme descriptions, boundary conditions explanations, and parameters that can be used
- The solver only checks for NaN. Monitoring E_k = 0.5·Σ(u²+v²+w²)·h^d and enstrophy
would catch slow blow-ups, non-physical energy growth, or excessive dissipation
- how difficult would be to add a turbulence Smagorinsky model
(ν_t = (Cs·Δ)²|S̄|) to model additive eddy viscosity? What does the model do exactly and how expensive is it?

# HIGH PRIORITY:
- when running 1guilla slime exp i noticed that the gpu is not fully used - check why
- Test a simulation in 2d and one in 3d with an analytically moving body, both analytically defined
- Test a simulation in 2d and one in 3d with an analytically moving body defined via a mesh file.
- Create new 2d/3d salamander simulation
- Make new zebrafish swimming models
- Polish the repository
- Schooling experiment with many zebrafish in 2d
- Test solution of the Poisson equation using PINNs/CNNs - ask agent
- Compare 1guilla pinned simulation against PIV data (need a fined grid)
- Compare 1guilla with dyes experiments

# LOW PRIORITY:
- Test an analytical 2d swimmer simulation
- Consider Crank-Nicolson for diffusion. Current explicit diffusion has stability limit dt < h²/(2ν·ndim). Not a bottleneck now (dt_diff ≈ 4.2s ≫ dt_cfl), but becomes relevant if dt is increased aggressively per A5.


# LONG TERM GOALS:
- How to handle bodies outside the water (at the interface)
- Add volume of fluids methods for handling water surface breaking (?)
- Add sph simulation support (?)
- Monolithic fluid multi rigid body solver (?)
- Simulate a submarine
- AMR (Adaptive Mesh Refinement) - refine grid only near bodies and in the wake. Would dramatically reduce cost for external flow problems where most of the domain is smooth freestream.


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

### D4. LES subgrid model (Smagorinsky)
No turbulence model limits the solver to low-Re. A Smagorinsky model
(ν_t = (Cs·Δ)²|S̄|) would be straightforward: additive eddy viscosity to self.nu.


### D6. Performance profiling hooks
No timing infrastructure to identify which phase (advection, Poisson, BDIM, SDF, force,
I/O) dominates cost. Add `torch.cuda.Event`-based timers around each phase, toggled by a
`profile=True` flag.

### D7.


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

