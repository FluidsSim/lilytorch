"""Native CUDA ops for BDIM-IB CFD (ported from pytorch_interpolation).

Importing this package loads the compiled ``_C.so`` extension which
registers four operators under the ``lilytorch_kernels`` torch library:

* ``streaming_sdf_min_3d``       -- one-body fused SDF / face-velocity update
* ``streaming_sdf_min_3d_multi`` -- multi-body fused SDF / face-velocity update
* ``bdim_forces_3d_multi``       -- per-body force / torque integration
* ``apply_bcs_3d``               -- fused 3-D BC writes (Neumann + Dirichlet)

CUDA kernels live in ``csrc/cuda/streaming_sdf.cu``; the C++ glue,
schemas and CPU stubs live in ``csrc/ops.cpp``.
"""
import torch  # noqa: F401 — ensures libtorch is loaded before _C.so dlopen
from . import _C  # noqa: F401  -- triggers TORCH_LIBRARY registration
from .ops import (
    streaming_sdf_min_3d,
    streaming_sdf_min_3d_multi,
    bdim_forces_3d_multi,
    apply_bcs_3d,
)

__all__ = [
    "streaming_sdf_min_3d",
    "streaming_sdf_min_3d_multi",
    "bdim_forces_3d_multi",
    "apply_bcs_3d",
]
