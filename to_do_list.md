

python lilytorch/src/video_postprocess.py /data/andreaferrario/ns_data/1guilla_self_propelled/swimming2 --fields omega_z_3d vel_mag_3d --format 



RUN:
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

HIGH PRIORITY:
- Test a simulation in 2d and one in 3d with an analytically moving body, both analytically defined
- Test an analytical 2d swimmer simulation
- Test a simulation in 2d and one in 3d with an analytically moving body defined via a mesh file.
- Make a new 2d/3d salamander simulation for the salamander and zebrafish models
- Polish the repository
- Run the drag cylinder test in 2d



LOW PRIORITY:


LONG TERM GOALS:
- How to handle bodies outside the water (at the interface)
- Add volume of fluids methods for handling water surface breaking
- Add sph simulation support
- Monolithic fluid multi rigid body solver (?)


---

# IMPROVEMENT SUGGESTIONS (from deep code review, March 2026)


### A7. Pre-allocate velocity buffers in `_solve_convective` (adv_diff.py)
`vel_new = [v.clone() for v in vel]` allocates 2 (2D) or 3 (3D) full-grid clones every
timestep. With Heun, this happens twice per step. Pre-allocate `self._vel_buf` at init
and use `torch.Tensor.copy_()` instead.







### A10. Batch GPU→CPU force transfers in `_apply_forces_3d`
Currently calls `.cpu().numpy()` 12 separate times (6 friction + 6 pressure), each
triggering a CUDA sync. Stack all force tensors into one, do a single `.cpu().numpy()`,
then slice on CPU.



### A14. Narrow-band force computation in `forces_method2_3d`
9 calls to dpdx/dpdy/dpdz (3 vel × 3 dirs) each allocate a full-grid tensor. The
stress tensor only matters where `delta != 0` (near the body surface). Computing gradients
only in a narrow band (~2ε from surface) would shrink computation dramatically.

### A15. Unify derivative implementations
`compute_dpdx/dpdy/dpdz` use `torch.gradient(edge_order=2)` while `gradient()`,
`divergence()`, `vorticity()` use manual slice-based FD. The former has extra overhead
for edge handling. Unifying on slice-based FD would be faster and consistent.




## B. STABILITY — Solver Robustness

### B1. Add runtime CFL monitoring
The `clf()` method in adv_diff.py computes CFL but is never called during simulation.
Add a CFL check in `step_()` or `solve_heun()`:
```python
cfl = self.adv_diff_solver.clf(u, v, w)
if cfl > 0.5:
    warnings.warn(f"CFL = {cfl:.3f} exceeds 0.5 at iter {iteration}")
```

### B2. Implement adaptive time-stepping
Use CFL helper to adjust dt dynamically:
```python
cfl = self.adv_diff_solver.clf(u, v, w)
self.dt = min(self.dt_max, cfl_target * self.h / (u_max + 1e-12))
```
Prevents blowups during impulsive events (body startup, vortex shedding onset).

### B3. Use `zero_pressure_inside = True`
Already implemented as a config key. Zeros pressure inside body SDF, preventing
pressure spikes at the fluid-solid interface. Especially important in 3D.

### B4. Consider Crank-Nicolson for diffusion
Current explicit diffusion has stability limit dt < h²/(2ν·ndim). Not a bottleneck now
(dt_diff ≈ 4.2s ≫ dt_cfl), but becomes relevant if dt is increased aggressively per A5.

### B5. Tighten Poisson tolerance for long simulations
Default `poisson_tol=1e-4` accumulates pressure errors over thousands of steps.
For >10k steps, use 1e-5 or 1e-6. FFT solver gives machine precision at no extra cost.

### B6. Post-projection divergence monitoring
After projection, the residual divergence is stored (self.div) but never checked.
Adding `div_max = self.divergence(u,v,w).abs().max()` every N steps catches Poisson
under-convergence before it cascades into NaN.

### B7. Energy and enstrophy monitoring
The solver only checks for NaN. Monitoring E_k = 0.5·Σ(u²+v²+w²)·h^d and enstrophy
would catch slow blow-ups, non-physical energy growth, or excessive dissipation.

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

### B11. Initial velocity masThe highest-impact, lowest-effort wins are A1 (switch to FFT Poisson — already implemented), A5 (increase dt — just a config change), A11 (warm-start pressure — a few lines), and B1 (CFL monitoring — a few lines).

king inside body
Already implemented (u *= (1-mu) at startup). Ensure all configs enable it — prevents
impulsive pressure spike at t=0 that can cause early divergence.

### B12. Symmetry-breaking perturbation
`perturbation_amplitude` config key adds small random noise. Needed for free-swimming
to prevent getting stuck in unstable symmetric equilibria that cause sudden blowups.


## C. CODE QUALITY

### C1. No formal test suite
All "tests" are standalone scripts under src/old/. Missing: pytest structure,
convergence-rate verification (h-refinement), manufactured solution tests,
BDIM conservation checks, analytical force validation (Stokes drag).

### C2. No documentation system
No Sphinx, mkdocs, or rendered docs. At minimum, a docs/ folder with mathematical
formulation, scheme descriptions, BC explanations, and API reference.

### C3. Heavy top-level imports in body.py
Module-level import of open3d, cv2, matplotlib, skfmm, skimage, scipy.interpolate
all load at import time. Many are only needed for one-time SDF construction.
Lazy imports with importlib would cut startup time.

### C4. ~300+ lines of commented-out dead code in solver.py
`forces_method1` has ~250 lines of dead alternatives, debug embed() calls, inline
matplotlib imports. Move to version control history.

### C5. BDIM meta-equation duplicated 6 times
Same `mu0*u' + (1-mu0)*u_body + mu1*normal_derivative(...)` pattern at 6 locations.
Extract into a single `_apply_bdim(vel_prime, body_vel, mu0, m_m0, mu1, normals)`.

### C6. `_recompute_mu_normals` duplicated between solver.py and BDIMhandler.py
Near-identical logic in both files. One should call the other.

### C7. No type hints
No Python type annotations anywhere. Adding them improves IDE support, catches bugs,
and serves as documentation.

### C8. Typo: `costum_update` throughout solver.py and BDIMhandler.py
Should be `custom_update`. ~20+ occurrences.

### C9. Synchronous I/O in `save_results`
`.cpu().numpy()` + `np.save()` per field, each a sync point. Use `torch.save()` or
move saving to a background thread with `concurrent.futures.ThreadPoolExecutor`.


## D. FEATURES

### D1. Checkpoint/restart system
No automatic periodic checkpointing that saves full solver state (iteration, drag records,
body positions, Poisson state). A crash at iter 999k of a 1M-step run means starting over.

### D2. Move drag record to CPU / memory-mapped file
`viscous_drag_record = torch.zeros((n_bodies, n_force_comp, nt))` lives on GPU for the
entire run. For nt=1M and 10 bodies, that's ~120 MB of GPU memory holding mostly zeros.
Write incrementally to CPU buffer or memory-mapped file.

### D3. Asynchronous/background I/O
Both field saving and VTK export block the simulation loop. Use
`concurrent.futures.ThreadPoolExecutor` to pin data to CPU and queue writes.

### D4. LES subgrid model (Smagorinsky)
No turbulence model limits the solver to low-Re. A Smagorinsky model
(ν_t = (Cs·Δ)²|S̄|) would be straightforward: additive eddy viscosity to self.nu.

### D5. Compressed output format
`np.save` produces uncompressed files. For 512×128×128, each field is ~32 MB.
Use `np.savez_compressed` or HDF5 (h5py) for 5–10× storage reduction.

### D6. Performance profiling hooks
No timing infrastructure to identify which phase (advection, Poisson, BDIM, SDF, force,
I/O) dominates cost. Add `torch.cuda.Event`-based timers around each phase, toggled by a
`profile=True` flag.

### D7. AMR (Adaptive Mesh Refinement)
Long-term: refine grid only near bodies and in the wake. Would dramatically reduce
cost for external flow problems where most of the domain is smooth freestream.



