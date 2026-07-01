"""``src_cuda.poisson_mult`` — native-backed multigrid/MGCG/RMGCG Poisson.

Re-export of :mod:`lilytorch.src.poisson_mult`; the solver's ``_K`` handle is the
native ``lilytorch.src.kernels.ops`` module (== :mod:`lilytorch.src_cuda.kernel`'s
``K``).  Warp analogue: :mod:`lilytorch.src_warp.poisson_mult`.
"""
from lilytorch.src.poisson_mult import *  # noqa: F401,F403
from lilytorch.src.poisson_mult import PoissonSolver  # noqa: F401
from lilytorch.src_cuda import kernel  # noqa: F401
