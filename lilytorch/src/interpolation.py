"""Scattered-point interpolation on a uniform grid.

``RegularGridInterpolator`` (a drop-in for pytorch_interpolation's) is
re-exported from :mod:`lilytorch.src.native`: the native CUDA/C++
``interp_{2,3}d`` ops, each with an ``at::parallel_for`` CPU twin.  No
grid_sample overhead, no coordinate normalisation, border-clamp semantics.

method="linear"    → bilinear (2-D) / trilinear (3-D)  [default]
method="quadratic" → biquadratic (2-D) / triquadratic (3-D), falls back
                     to linear at grid boundaries (< 3 cells on any axis)

2-D / 3-D is determined from the number of axes supplied to the
constructor (must be 2 or 3; 1 raises ValueError).

The uniform-grid samplers themselves (bilinear/biquadratic/sdf_sample in 2-D,
trilinear/triquadratic in 3-D) are single-sourced in C++/CUDA and shared by the
streaming-SDF and forces kernels; they are not exposed to Python.
"""
from __future__ import annotations

# ``native`` is a true leaf (imports only the ``_C`` extension + torch), so this
# does not break the "interpolation must not import solver/two_phase/facade" rule.
from lilytorch.src.native import RegularGridInterpolator

__all__ = [
    "RegularGridInterpolator",
]
