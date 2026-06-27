"""``src_cuda.advection`` — native-backed advection-diffusion.

Re-export of :mod:`lilytorch.src.advection`.  Kernel dispatch (``advect_flux_add``,
``apply_bcs_*``) goes through the native ``torch.ops.lilytorch_kernels.*`` ops,
i.e. :mod:`lilytorch.src_cuda.kernel`.  The Warp analogue is
:mod:`lilytorch.src_warp.advection`.
"""
from lilytorch.src.advection import *  # noqa: F401,F403
from lilytorch.src.advection import AdvDiffSolver, SCHEMES  # noqa: F401
from lilytorch.src_cuda import kernel  # noqa: F401
