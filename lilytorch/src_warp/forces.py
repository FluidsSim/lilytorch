"""``src_warp.forces`` — hydrodynamic force readout.

Two force readouts exist in ``lilytorch.src.forces``:

* **Lagrangian** ``lagrangian_forces_{2,3}d`` — **on Warp** (Item 3a).
  ``warp_poc/warp_lagrangian.py`` is now dtype-generic (f32+f64; per-element
  math in the field dtype, float64 atomic accumulator — exactly the native
  ``AT_DISPATCH``/``double* out`` contract), and its Python wrappers take the
  *same* arg list as the native ``ops.lagrangian_forces_{2,3}d`` (decomposed
  eps_xx…/tri_*… plus ``method``/``sample_offset``/``out=``).  The
  :mod:`lilytorch.src_warp.kernel` facade exposes drop-in shims, and
  :class:`lilytorch.src_warp.solver.FluidSolver` overrides
  ``forces_lagrangian_{2,3}d`` to route the inherited readout's single kernel
  call through them (a localized module-global swap — no ``lilytorch.src``
  edits).  This is the most accurate readout for fully-resolved bodies, so it is
  the recommended SU1 force routing.
* **Eulerian** ``streaming_sdf_forces_post_{2,3}d`` — **on Warp** (Item 3b).
  ``warp_poc/warp_forces.py`` ports the n·δ viscous+pressure band integral (+ the
  deltaH ∂H pressure pass), dtype-generic (f32/f64); the native CUB block-reduction
  is replaced by a per-cell ``wp.atomic_add`` into the float64 accumulator (same
  sum, reduction-order noise only).  Wired behind the inherited ``forces_method2``
  / ``forces_method2_3d`` by the solver-subclass module-global swap.

This module re-exports the native readout under the Warp namespace; both force
routings (Lagrangian + Eulerian) are installed by the solver subclass, not here.
"""
from lilytorch.src.forces import *  # noqa: F401,F403
