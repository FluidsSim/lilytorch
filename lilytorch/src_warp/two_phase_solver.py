"""``src_warp.two_phase_solver`` — two-phase solver wired to the Warp backend.

Mirrors :class:`lilytorch.src_warp.solver.FluidSolver` for the two-phase
(water + real air) path: subclasses the native
:class:`lilytorch.src.two_phase_solver.TwoPhaseSolver` and injects the Warp
sub-solvers (advection flux, Poisson smoother/residual) **and** the Warp
``cvof_sweep`` VOF field by temporarily swapping the module-global
``AdvDiffSolver`` / ``PoissonSolver`` (in ``lilytorch.src.solver``) and
``TwoPhase`` (in ``lilytorch.src.two_phase_solver``) for the duration of
``__init__`` only — the same localized dependency injection used by the
single-phase Warp ``FluidSolver``, needing no ``lilytorch.src`` edits and
leaving no global state behind.

The two-phase body-force readout (``_two_phase_forces`` → ``_forces.forces_method2``)
dispatches the module-global ``streaming_sdf_forces_post_{2,3}d`` /
``_lagrangian_forces_{2,3}d_kernel``; those are routed to the Warp ports by a
localized per-call swap in the overridden :meth:`_two_phase_forces` (same pattern
as ``FluidSolver.forces_method2``).

MRO: ``(_BaseTwoPhaseSolver, _WarpFluidSolver, src.solver.FluidSolver)`` — the
two-phase methods take priority, the Warp ``FluidSolver``'s kernel-step / force
overrides (e.g. ``_fluid_step_kernel_3d`` for a two-phase *kernel*-mode run) are
inherited where the two-phase solver does not override them.
"""
import lilytorch.src.solver as _solver_mod
import lilytorch.src.two_phase_solver as _tps_mod
import lilytorch.src.forces as _forces_mod
from lilytorch.src.two_phase_solver import TwoPhaseSolver as _BaseTwoPhaseSolver

from lilytorch.src_warp.solver import FluidSolver as _WarpFluidSolver
from lilytorch.src_warp.advection import AdvDiffSolver as _WarpAdvDiffSolver
from lilytorch.src_warp.poisson_mult import PoissonSolver as _WarpPoissonSolver
from lilytorch.src_warp.two_phase import TwoPhase as _WarpTwoPhase
from lilytorch.src_warp import kernel

BACKEND = "warp"


class TwoPhaseSolver(_BaseTwoPhaseSolver, _WarpFluidSolver):
    """``TwoPhaseSolver`` with Warp sub-solvers + Warp VOF injected at construction."""

    def __init__(self, *args, **kwargs):
        _save_adv = _solver_mod.AdvDiffSolver
        _save_poi = _solver_mod.PoissonSolver
        _save_tp = _tps_mod.TwoPhase
        _solver_mod.AdvDiffSolver = _WarpAdvDiffSolver
        _solver_mod.PoissonSolver = _WarpPoissonSolver
        _tps_mod.TwoPhase = _WarpTwoPhase
        try:
            _BaseTwoPhaseSolver.__init__(self, *args, **kwargs)
        finally:
            _solver_mod.AdvDiffSolver = _save_adv
            _solver_mod.PoissonSolver = _save_poi
            _tps_mod.TwoPhase = _save_tp

        # Mirror the Warp ``FluidSolver`` post-init hooks (its __init__ is not on
        # this MRO path, so replicate them): opt-in Kernel-A graph flag, periodic
        # MGCG convergence check, persistent kernel-step buffer caches.
        pars = args[0] if args else kwargs.get("pars", {})
        try:
            self._kernel_cuda_graph = bool(
                pars["solver"].get("kernel_cuda_graph", False))
        except Exception:
            self._kernel_cuda_graph = False
        try:
            cge = int(pars["solver"].get("poisson_cg_check_every", 1))
            if getattr(self, "poisson_solver", None) is not None:
                self.poisson_solver.cg_check_every = cge
        except Exception:
            pass
        self._kbuf2d = None
        self._kbuf3d = None

    # ── two-phase body-force readout on Warp ─────────────────────────────────
    def _two_phase_forces(self, fn3d, vels, p, iteration):
        _s2 = _forces_mod.streaming_sdf_forces_post_2d
        _s3 = _forces_mod.streaming_sdf_forces_post_3d
        _l2 = _forces_mod._lagrangian_forces_2d_kernel
        _l3 = _forces_mod._lagrangian_forces_3d_kernel
        _forces_mod.streaming_sdf_forces_post_2d = kernel.streaming_sdf_forces_post_2d
        _forces_mod.streaming_sdf_forces_post_3d = kernel.streaming_sdf_forces_post_3d
        _forces_mod._lagrangian_forces_2d_kernel = kernel.lagrangian_forces_2d
        _forces_mod._lagrangian_forces_3d_kernel = kernel.lagrangian_forces_3d
        try:
            return super()._two_phase_forces(fn3d, vels, p, iteration)
        finally:
            _forces_mod.streaming_sdf_forces_post_2d = _s2
            _forces_mod.streaming_sdf_forces_post_3d = _s3
            _forces_mod._lagrangian_forces_2d_kernel = _l2
            _forces_mod._lagrangian_forces_3d_kernel = _l3
