"""Native CUDA ops for BDIM-IB CFD (ported from pytorch_interpolation).

Importing this package loads the compiled ``_C.so`` extension which
registers operators under the ``lilytorch_kernels`` torch library:

3-D ops:
* ``streaming_sdf_min_3d``       -- one-body fused SDF / face-velocity update
* ``streaming_sdf_min_3d_multi`` -- multi-body fused SDF / face-velocity update
* ``bdim_forces_3d_multi``       -- per-body force / torque integration
* ``apply_bcs_3d``               -- fused 3-D BC writes (Neumann + Dirichlet)

2-D ops (mirror the 3-D ones with the z-axis stripped):
* ``streaming_sdf_min_2d``       -- one-body fused 2-D SDF / face-velocity update
* ``streaming_sdf_min_2d_multi`` -- multi-body fused 2-D SDF / face-velocity update

CUDA kernels live in ``csrc/cuda/streaming_sdf*.cu``; the C++ glue and
schemas live in ``csrc/ops.cpp``; CPU implementations live in
``csrc/streaming_sdf_cpu*.cpp``.
"""
import sys
from pathlib import Path

import torch  # noqa: F401 -- ensures libtorch is loaded before _C.so dlopen


def _load_native_extension():
    try:
        from . import _C  # noqa: F401  -- triggers TORCH_LIBRARY registration
        return _C
    except ImportError as exc:
        repo_root = Path(__file__).resolve().parents[3]
        abi_fn = getattr(torch, "compiled_with_cxx11_abi", None)
        abi = abi_fn() if callable(abi_fn) else "unknown"
        detail = ""
        if "undefined symbol" in str(exc):
            detail = (
                " The compiled extension appears to be incompatible with the "
                f"active PyTorch runtime ({torch.__version__}, "
                f"cxx11abi={abi}). This usually means `_C.so` was built in a "
                "different environment or via pip build isolation."
            )
        raise ImportError(
            "Failed to import `lilytorch.src.kernels._C`."
            f"{detail} Rebuild the extension against the current environment "
            "with:\n"
            f"  cd {repo_root}\n"
            f"  {sys.executable} setup.py build_ext --inplace\n"
            "For editable installs, use `pip install -e . --no-build-isolation` "
            "after installing PyTorch in that same environment."
        ) from exc


_C = _load_native_extension()

from .ops import (
    streaming_sdf_min_3d,
    streaming_sdf_min_3d_multi,
    bdim_forces_3d_multi,
    apply_bcs_3d,
    streaming_sdf_min_2d,
    streaming_sdf_min_2d_multi,
)

__all__ = [
    "streaming_sdf_min_3d",
    "streaming_sdf_min_3d_multi",
    "bdim_forces_3d_multi",
    "apply_bcs_3d",
    "streaming_sdf_min_2d",
    "streaming_sdf_min_2d_multi",
]
