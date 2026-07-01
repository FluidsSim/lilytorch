"""Backend selection helper — maps a config ``solver.backend`` value to the
solver class to instantiate.

There is no in-``src/`` backend flag (the Warp backend is a parallel module
tree, per the repo owner's request).  This helper centralizes the
``"native"`` / ``"warp"`` choice so the integration bridge
(:mod:`lilytorch.integration.BDIMhandler`) and the standalone example drivers
can opt in to the Warp backend from YAML (``solver.backend: warp``) instead of
hand-swapping the solver symbol.  Default is ``"native"`` — existing configs are
unaffected.
"""


def resolve_solver_class(backend, two_phase):
    """Return the ``FluidSolver`` / ``TwoPhaseSolver`` class for ``backend``.

    ``backend`` is ``None`` / ``"native"`` (native CUDA backend) or ``"warp"``
    (single-source Warp backend, :mod:`lilytorch.src_warp`).  ``two_phase``
    selects the two-phase (water + air) solver over the single-phase one.
    """
    if backend in (None, "native"):
        from lilytorch.src.solver import FluidSolver
        from lilytorch.src.two_phase_solver import TwoPhaseSolver
        return TwoPhaseSolver if two_phase else FluidSolver
    if backend == "warp":
        if two_phase:
            from lilytorch.src_warp.two_phase_solver import TwoPhaseSolver
            return TwoPhaseSolver
        from lilytorch.src_warp.solver import FluidSolver
        return FluidSolver
    raise ValueError(
        f"solver.backend must be 'native' or 'warp', got {backend!r}.")
