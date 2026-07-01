"""Single-source Warp kernels for BDIM-IB CFD.

Every op below is backed by an ``@wp.kernel`` port in ``src/kernels/warp_*.py``
(one source runs on CPU and CUDA), exposed through the :mod:`~.facade` module.
The hand-written CUDA/C++ ``_C.so`` extension has been retired.

3-D ops:
* ``streaming_sdf_stag_3d_multi``  -- Phase-I streaming SDF + body velocities
* ``bdim_coeff_3d``                -- Phase-I fused BDIM2 + Poisson coeffs
* ``streaming_sdf_forces_post_3d`` -- post-fluid-step force integration
* ``apply_bcs_3d``                 -- 3-D BC writes (Neumann + Dirichlet)

2-D ops mirror the 3-D ones with the z-axis stripped.
"""
import torch  # noqa: F401

# Warp single-source kernels (facade) — these op names dispatch to the
# ``@wp.kernel`` ports in ``src/kernels/warp_*.py``.
from .facade import (
    streaming_sdf_stag_3d_multi,
    bdim_coeff_3d,
    bdim_coeff_sigma_3d,
    streaming_sdf_forces_post_3d,
    apply_bcs_3d,
    streaming_sdf_stag_2d_multi,
    bdim_coeff_2d,
    bdim_coeff_sigma_2d,
    streaming_sdf_forces_post_2d,
    apply_bcs_2d,
    interp_2d,
    interp_3d,
    lagrangian_forces_2d,
    lagrangian_forces_3d,
    advect_flux_add,
    cvof_sweep,
)

from .interpolation import (
    RegularGridInterpolator,
    RegularGridInterpolator3D,
    RegularGridInterpolatorAutomatic,
)

__all__ = [
    "streaming_sdf_stag_3d_multi",
    "bdim_coeff_3d",
    "bdim_coeff_sigma_3d",
    "streaming_sdf_forces_post_3d",
    "apply_bcs_3d",
    "streaming_sdf_stag_2d_multi",
    "bdim_coeff_2d",
    "bdim_coeff_sigma_2d",
    "streaming_sdf_forces_post_2d",
    "apply_bcs_2d",
    "interp_2d",
    "interp_3d",
    "lagrangian_forces_2d",
    "lagrangian_forces_3d",
    "advect_flux_add",
    "cvof_sweep",
    "RegularGridInterpolator",
    "RegularGridInterpolator3D",
    "RegularGridInterpolatorAutomatic",
]
