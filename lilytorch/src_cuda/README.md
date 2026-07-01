# `src_cuda/` — hand-written CUDA/C++ kernel backend

Parallel backend tree to [`src_warp/`](../src_warp/README.md).  Kernel-dispatching
solver modules here call the hand-written `.cu`/`.cpp` kernels (registered as
`torch.ops.lilytorch_kernels.*` and surfaced through `lilytorch.src.kernels.ops`)
through the unified backend API in [`kernel/`](kernel/__init__.py).

Because `lilytorch.src` already *is* the native/CUDA path, the modules here are
thin re-exports of their `lilytorch.src` counterparts — `src_cuda.solver.FluidSolver`
**is** `lilytorch.src.solver.FluidSolver`.  The value of this tree is the explicit
`kernel/` backend boundary, mirrored one-for-one by `src_warp`, so a sim can be
pointed at either backend by importing from `lilytorch.src_cuda` vs
`lilytorch.src_warp` while the kernel-agnostic modules (`body`, `plotting`,
`poisson_fft`, `operations`, `diagnostics`, …) stay shared in `lilytorch.src`.

## Layout
```
src_cuda/
  kernel/__init__.py   # unified backend API → native ops (WARP_BACKED = {})
  solver.py            # re-export FluidSolver (native Kernel A/B + Poisson)
  advection.py         # re-export AdvDiffSolver (advect_flux_add, apply_bcs)
  poisson_mult.py      # re-export PoissonSolver (mgcg/multigrid/rmgcg)
  forces.py            # re-export native force readout
  two_phase.py         # re-export TwoPhase (cvof_sweep)
```

The unified kernel API contract is documented in
[`kernel/__init__.py`](kernel/__init__.py); `src_warp` exposes the same names.
