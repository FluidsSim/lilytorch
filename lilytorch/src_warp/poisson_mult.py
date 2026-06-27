"""``src_warp.poisson_mult`` — multigrid/MGCG/RMGCG Poisson (native for now).

The Warp port has the smoother (``rbgs``/``jacobi``), the multigrid transfer
ops (``mg_residual``/``restrict_*``/``prolongate_add``), and an assembled,
geometrically-converging ``WarpVCycle`` (2-D + 3-D) — all parity-clean vs native
(see ``warp_poc/VALIDATION_STATUS.md`` §D/§E).  Routing the *live* solver to
Warp means assembling the mgcg/multigrid **outer driver** in Python from those
Warp kernels (the native ``poisson_solve_*`` is a monolithic C++ op with no
Python-injectable smoother seam).  That driver-assembly is the §F remaining
work; until then this subclass is the native solver, exposed under the Warp
namespace for symmetry.
"""
from lilytorch.src.poisson_mult import PoissonSolver as _BasePoissonSolver  # noqa: F401


class PoissonSolver(_BasePoissonSolver):
    """Native Poisson driver under the Warp namespace (driver assembly = §F)."""

    pass
