"""Unified kernel-backend API — **CUDA/native** implementation.

This module defines the *contract* that both backend trees expose under the
same names: :mod:`lilytorch.src_cuda.kernel` (here, native ``.cu``/``.cpp``) and
:mod:`lilytorch.src_warp.kernel` (single-source ``@wp.kernel``).  The
kernel-dispatching solver modules in each tree import their ops from the sibling
``kernel`` package, so the *only* thing that distinguishes ``src_cuda`` from
``src_warp`` is which backend fills these names.

Op groups (all present in both backends; ``src_warp`` falls back to these native
ops where the Warp port is not yet wired — see that module's ``WARP_BACKED``):

* streaming SDF / BDIM (Kernel A & B): ``streaming_sdf_stag_{2,3}d_multi``,
  ``bdim_coeff_{2,3}d``, ``bdim_coeff_sigma_{2,3}d``
* advection / BC: ``advect_flux_add``, ``apply_bcs_{2,3}d``, ``interp_{2,3}d``
* two-phase: ``cvof_sweep``
* forces: ``lagrangian_forces_{2,3}d``, ``streaming_sdf_forces_post_{2,3}d``
* Poisson driver handle: ``K`` (module exposing ``poisson_solve_*`` + the
  multigrid smoother/transfer ops); injected into ``PoissonSolver._K``.
"""

import torch

# Importing ``_C`` registers ``torch.ops.lilytorch_kernels.*`` (advect_flux_add,
# cvof_sweep, apply_bcs, …) and the abstract impls used by ``ops``.
from lilytorch.src.kernels import _C as _C  # noqa: F401
from lilytorch.src.kernels import ops as _ops

BACKEND = "cuda"

# ── Kernel A / Kernel B (streaming SDF + BDIM coefficients) ────────────────
streaming_sdf_stag_2d_multi = _ops.streaming_sdf_stag_2d_multi
streaming_sdf_stag_3d_multi = _ops.streaming_sdf_stag_3d_multi
bdim_coeff_2d = _ops.bdim_coeff_2d
bdim_coeff_3d = _ops.bdim_coeff_3d
bdim_coeff_sigma_2d = _ops.bdim_coeff_sigma_2d
bdim_coeff_sigma_3d = _ops.bdim_coeff_sigma_3d

# ── advection / boundary conditions / interpolation ───────────────────────
# advect_flux_add and cvof_sweep are registered only as torch.ops (no wrapper
# in ``ops``); apply_bcs / interp exist in both places — use torch.ops for the
# pointer-arithmetic ops to match the native call sites verbatim.
advect_flux_add = torch.ops.lilytorch_kernels.advect_flux_add
apply_bcs_2d = _ops.apply_bcs_2d
apply_bcs_3d = _ops.apply_bcs_3d
interp_2d = _ops.interp_2d
interp_3d = _ops.interp_3d

# ── two-phase VOF ─────────────────────────────────────────────────────────
cvof_sweep = torch.ops.lilytorch_kernels.cvof_sweep

# ── forces ────────────────────────────────────────────────────────────────
lagrangian_forces_2d = _ops.lagrangian_forces_2d
lagrangian_forces_3d = _ops.lagrangian_forces_3d
streaming_sdf_forces_post_2d = _ops.streaming_sdf_forces_post_2d
streaming_sdf_forces_post_3d = _ops.streaming_sdf_forces_post_3d

# ── Poisson driver handle (PoissonSolver._K) ──────────────────────────────
# The native ``ops`` module exposes poisson_solve_{mgcg,multigrid,rmgcg}_{2,3}d
# plus the rbgs/jacobi/mg_residual/restrict/prolongate kernel-level ops.
K = _ops

# Every op in this backend is native.
WARP_BACKED = frozenset()
