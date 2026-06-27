# `src_warp/` — Warp single-source kernel backend

Parallel backend tree to [`src_cuda/`](../src_cuda/README.md).  Kernel-dispatching
solver modules here call the single-source `@wp.kernel` ports (in
[`warp_poc/`](../warp_poc/)) through the unified backend API in
[`kernel/`](kernel/__init__.py).  **One kernel source runs on CPU *and* GPU**,
retiring the hand-written `.cu`/`.cpp` twins for the ops that are wired in.

This is the §F "end-to-end swap-in" of the Warp port, realized as a parallel
backend tree (rather than an in-`src/` `use_warp_kernels` flag), per the repo
owner's request.  `lilytorch/src/` is **untouched**; both backend trees import
the kernel-agnostic modules (`body`, `plotting`, `poisson_fft`, `operations`,
`diagnostics`, …) from there.

## Layout
```
src_warp/
  kernel/__init__.py   # unified backend API: Warp where wired, native fallback
  solver.py            # FluidSolver: injects Warp sub-solvers at __init__
  advection.py         # AdvDiffSolver: flux on Warp advect_flux_add
  two_phase.py         # TwoPhase: VOF sweep on Warp cvof_sweep
  poisson_mult.py      # PoissonSolver (native; driver-assembly = remaining)
  forces.py            # native readout (lagrangian routing = remaining)
```

## What runs on Warp today (`kernel.WARP_BACKED`)
| op | path | status |
|----|------|--------|
| `advect_flux_add` | `AdvDiffSolver._solve_convective` fused-flux | **Warp**, dtype-generic (f64 bit-exact vs native all 5 schemes 2-D+3-D; f32 to single precision), CPU==GPU |
| `cvof_sweep` | `TwoPhase._cvof_sweep` | **Warp**, dtype-generic (f64 bit-exact 2-D+3-D; f32 to single precision), CPU==GPU |
| `streaming_sdf_stag_2d_multi` (Kernel A, 2-D) | `_fluid_step_kernel_2d` via marshalling bridge | **Warp**, dtype-generic (f32 bit-exact, f64 to ~1e-7 — native interpolates the SDF in f32), CPU==GPU |
| `bdim_coeff_2d` (Kernel B, 2-D) | `_fluid_step_kernel_2d` | **Warp**, dtype-generic (f64 bit-exact, f32 to single precision) — no native fallback |

`advect_flux_add` / `cvof_sweep` are *signature-identical drop-ins* (pure
dispatch-target swap).  The 2-D Kernel A/B are wired through a **marshalling
bridge** (`kernel._KernelA2DBridge`) re-expressing the live flat-table tensors
into `WarpStreamingSDF2D`'s setup/update/run API, plus an overridden
`FluidSolver._fluid_step_kernel_2d`.  Kernel A is now **dtype-generic** (one
`@wp.kernel` source, `wp.overload` f32+f64); Kernel B's Warp port is f64-only
(an f32 solver keeps native Kernel B).  The σ path stays native (the streaming
bridge does not emit the packed `key_*`).  Verified in
`warp_poc/test_src_trees.py` (incl. the **CPU single-source payoff** and the f64
Kernel A/B chain-vs-native parity).

## What still falls back to native (the §F remaining work)
The Warp kernels for all of these are **already ported and parity-clean** in
`warp_poc/` (165 tests); what remains is the *in-solver wiring*, not the kernels:

1. **Kernel A/B (3-D)** (`streaming_sdf_stag_3d_multi`, `bdim_coeff_3d`) — the
   2-D path is wired (above); the 3-D path still needs its marshalling bridge +
   an **f32 Kernel B Warp variant** (3-D solvers run f32, while the Kernel B port
   is f64; mirror the dtype-generic treatment done for 2-D Kernel A).  Then
   override `_fluid_step_kernel_3d`.
2. **Poisson driver** — `poisson_solve_*` is a monolithic C++ op (no injectable
   smoother seam).  Warp routing = assembling the mgcg/multigrid **outer driver**
   in Python from the Warp smoother + transfer ops; `WarpVCycle`/`WarpVCycle2D`
   already prove the composition converges.  Remaining: wrap it as a
   `poisson_solve_*`-compatible `PoissonSolver` backend.
3. **`apply_bcs_*`** — Warp wrapper uses one `max_face_dim` vs native
   `(max_dim0, max_dim1)`; safe only on cubic faces.  Remaining: pass both face
   dims through.
4. **Forces** — Eulerian `streaming_sdf_forces_post_*` is the one unported native
   op; the Warp `lagrangian_forces_*` IS ported but with a decomposed arg list.
   SU1 validation routes forces through `force_method=lagrangian` (native).

## Validate end-to-end (next)
- 2-D `_1guillasim` pinned + 3-D jellyfish trajectory match vs `src_cuda`, <5%
  wall-clock — needs item 1 (and 2) above wired.
- Full CPU run on Warp kernels — the per-op CPU single-source is proven
  (`test_src_trees.py::test_cvof_warp_cpu_matches_python_reference`); the
  end-to-end CPU run follows once the step path is fully Warp.

## Tests
```
python -m pytest warp_poc/test_src_trees.py -q   # 13 structure/parity/CPU tests
python -m pytest warp_poc/ -q                     # 165 total
```
