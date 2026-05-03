
# instructions
You are working starting from the optimize_speed_memory branch. Checkout from there and create a new branch for your implementations. Read the HIGH PRIORITY todo list to implement (in to_do_list.md) and start to work on from top to bottom (higher to lower priority list). Then give me a step to step guide for testing the various implementations.
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

# DONE:
- Implement cuda kernels for 2d simulations in the style used for 3d simulations: `streaming_sdf_min_2d` / `streaming_sdf_min_2d_multi` (CC + face-staggered SDF + body velocity running-min, with bilinear and biquadratic-Lagrange interpolation), `bdim_forces_2d_multi` (Phase-D fused force/torque integration over each body's AABB, reading the cached cc-SDF from `sparse_cc_flat`) and `apply_bcs_2d` (Phase-H fused Neumann + Dirichlet ghost-line writes for `(u, v)`).  CPU + CUDA implementations validated against pure-PyTorch references at machine precision in `test_streaming_sdf_2d_self.py`, `test_bdim_forces_2d_self.py`, `test_apply_bcs_2d_self.py`.  Wired through the solver: `BDIMhandler._update_2d_streaming_multi` (mirrors `_update_3d_streaming_multi`), `solver.streaming_sdf_2d` / `streaming_forces_2d` config flags (coupled together because 2-D has no narrow-band forces fallback), and `AdvDiffSolver.set_BCs` dispatches to the fused CUDA op in 2-D when the velocity tensors are CUDA + contiguous + same-floating-dtype.  TODO follow-up: 2-D analogue of the cost-analysis pipeline (`validation/cost_analysis_free_swimming_2d/`).
- Build the 2-D cost-analysis pipeline `validation/cost_analysis_free_swimming_2d/` (copy + adapt of `run_cost_analysis.py`, `run_multigrid_cost_analysis.py`, `run_scaling_conditions_pipeline.py`, `plot_scaling.py`) targeting a 2-D scene exercising the new `streaming_sdf_2d` / `streaming_forces_2d` path.
- The bdim_forces_3d_multi_kernel recompute the body cc sdf on the fly. I think this is a waste of computation. Indeed i think that forces can be computed together with the union sdf in streaming_sdf_min_3d_multi_kernel. Inside this kernel the body sdf is computed, there is where the delta force functions, and forces can be computed. This should save significant computational time.
- the gen_config_full_pool.py simulation in cpu mode does not seem to utilize multiple cores.
- The nbforces cost analysis plotted in cost_scaling_loglog.png reveals some scaling of the costs of "Other (residual)" and "Body update (SDF)". I do not understand why, since the cropping approach with aabb boxes should (in my view), maintain the same cost at different scales. Unless the domain size remains the same and just the number of grid points increases, in which case the portion of the domain that includes the body increases, so the operations should indeed increase. Please clarify.
- Simplify methods for cost optimization. After substantial testing of the different methods for running the cost analysis in run_scaling_conditions_pipeline.py the best method is nbforces_opt. Remove all the other methods, except for keeping the old one for reference (no cropping, no batching method for reference).
- Implement triquadratic interpolation option for evaluating the sdf functions in the cuda/C++ kernels (`streaming_sdf_min_3d`, `streaming_sdf_min_3d_multi`). Mirrors the pytorch_interpolation biquadratic-Lagrange algorithm extended to 3D (3x3x3 stencil, falls back to trilinear in the boundary layer). Optionally enabled by the user via the `sdf_interp_method` solver config key (`"trilinear"` (default) | `"triquadratic"`), exposed on `BaseSimConfig`. CPU implementation validated against a pure-PyTorch reference in `lilytorch/src/kernels/test_streaming_sdf_self.py`.
- Move all non standard computations (sponge layer, carreau, etc) in a dedicated file in src/extras.py
- Move force computations in an ad-hoc file (src/forces.py)
- Implement 2nd order accurate force method also for the cuda/C++ kernel solver version (currently only in non cuda/c++ kernel mode)
- replace pytorch_interpolation with existing precompiled cuda/c++ kernels or write new ones if necessary in the kernel/ folder.
- [STAGE 1 IN PROGRESS — branch `copilot/stage1-variant-cleanup`] There are a number of strange settings in solver.py (and maybe BDIMhandler.py) that depend on some old code, where different variants of the solvers were implemented, such as cropping, batching, streaming, etc methods. Currently (and this is how i want it to be), there should only be two solver variants. One is in pure python code, it should be suboptimal (no batching, no cropping), and another should use the cuda/c++ kernels in kernels/. The first approach should also allow compilation of advection+diff, forces, sdf, poisson. The second version should compile forces and sdf by default, but there should also be an option to compile the other two. You should read through the code and clean this feature.
  - The 9 individual variant flags (`force_narrow_band`, `force_narrow_batch`, `force_shared_union`, `mu_normals_union`, `bdim_union`, `streaming_sdf_3d`, `streaming_forces_3d`, `streaming_sdf_2d`, `streaming_forces_2d`) are collapsed into a single user-facing switch `solver.use_kernels` (default `True`).  `use_kernels` is independent of `use_gpu` — the latter still selects CPU vs CUDA device.
- Confirm that the sdf and force computations done with the cuda/c++ kernels is computing on each body aabb boxes, and not on the union aabb (narrow band on individual bodies). This is the most efficient approach, as the sdf/force computations are restricted to where it is needed. Is it possible to implement this method also for the bdim update, setting of the variable coefficients, mu/normals, and perhaps other parts? I think that, whilst currently the cuda/c++ kernels separately handle force and sdf computation, this could be merged, in the sense that the forces could be computed for each body at the same time (at the same body iteration) when the sdf is computed. This would avoid the need to store several CC sdfs for later force computation, saving memory, and reusing the body normals (although these are not cell centered). Additionally, bdim update, setting of the variable coefficients, mu/normals could in my view also be computed in the same kernel function locally. At the end, to save memory, the mu0 of the union sdf is needed for the correction step so it must be computed, but the memory of each body velocity, sdfs, normals, etc
can be released after the loop of that body is computed. This should, in my logic, save a lot of memory, whist maintaining the same accuracy.
- Check that the code works for dtype float32 and float64 (double). Especially the cuda/c++ kernels (but also the rest of the code) should be using one or the other depending on the user request.
- Implement a gamepad connected with the fluid solver. The implementation should be restricted to the salamander_gamepad/ folder. The current gen_configs_swim_2d.py is copied from the salamander/ folder, but should be modifies to use the salamander_gamepad/control.py. Also you need to modify this latter file to make it work as a pd controller as the salamander.pd_controller_swim.PositionController, and allow it to make the animal turn speedup/slowdown and turn left/right as in a videogame, following the rules indicated in the current control.py script.


# HIGH PRIORITY:
- Fix flow 2d gpu viewer (remove calls in base_sim_config), simulations sometimes stops midway, advection compilation generates flickering, multigrid method does not work with poisson_compile  = True.
- Sometimes the simulation of salamander gamepad breaks midway and does not terminate.
- Do a systematic memory cost analysis. I think that the latest method with cuda kernels dynamically rewrites the body velocities and writed the forces computation inside the kernel whenever body properties are needed. This reduces the memory footprint by not storing the sdfs/body velocities of each rigid body (just a unique composite union body properties are stored).
- Combine solver.py and BDIMhandler in a single simulation file (just solver.py). BDIMhandler should only keep whatever is necessary for handling the coupling with FARMS, if possible. Review options and propose what to do. This should have a careful modifications in all the examples scipts in farms_examples/.
- Polish the repository, review and correct outdated documentation, also in the docs/ folder.

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

