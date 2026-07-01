"""``src_cuda.solver`` — native-backed fluid solver.

Re-export of :mod:`lilytorch.src.solver`.  ``FluidSolver`` here dispatches Kernel
A/B (``streaming_sdf_stag_*`` / ``bdim_coeff_*``) and the Poisson driver to the
hand-written CUDA/C++ kernels.  This is byte-identical to constructing
``lilytorch.src.solver.FluidSolver`` — the value is the explicit backend
namespace, symmetric with :mod:`lilytorch.src_warp.solver`.
"""
from lilytorch.src.solver import *  # noqa: F401,F403
from lilytorch.src.solver import FluidSolver  # noqa: F401
from lilytorch.src_cuda import kernel  # noqa: F401

BACKEND = "cuda"
