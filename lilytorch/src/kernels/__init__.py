"""Single-source Warp kernels for BDIM-IB CFD.

Every op below is backed by an ``@wp.kernel`` port in this package
(``streaming_sdf{,_2d}.py``, ``bdim{,_2d}.py``, ``forces.py``, ``misc_{2,3}d.py``,
``advection.py``, ``cvof.py``, ``poisson*.py``, ``multigrid*.py``) — one kernel
source runs on CPU and CUDA — and is exposed through the :mod:`~.facade` module.
The hand-written CUDA/C++ ``_C.so`` extension has been retired.

3-D ops:
* ``body_update_3d``  -- Phase-I streaming SDF + body velocities
* ``bdim_forcing_3d``                -- Phase-I fused BDIM2 + Poisson coeffs
* ``streaming_sdf_forces_post_3d`` -- post-fluid-step force integration
* ``apply_bcs_3d``                 -- 3-D BC writes (Neumann + Dirichlet)

2-D ops mirror the 3-D ones with the z-axis stripped.
"""
import torch  # noqa: F401

from .facade import (
    body_update_3d,
    bdim_forcing_3d,
    bdim_forcing_sigma_3d,
    streaming_sdf_forces_post_3d,
    apply_bcs_3d,
    body_update_2d,
    bdim_forcing_2d,
    bdim_forcing_sigma_2d,
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
    "body_update_3d",
    "bdim_forcing_3d",
    "bdim_forcing_sigma_3d",
    "streaming_sdf_forces_post_3d",
    "apply_bcs_3d",
    "body_update_2d",
    "bdim_forcing_2d",
    "bdim_forcing_sigma_2d",
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
