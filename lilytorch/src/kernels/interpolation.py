"""Scattered-point interpolators backed by the Warp ``interp_{2,3}d`` kernels.

Drop-in replacement for pytorch_interpolation's RegularGridInterpolator,
RegularGridInterpolatorAutomatic, and RegularGridInterpolator3D.

Uses the same bilinear/biquadratic (2-D) and trilinear/triquadratic (3-D)
device functions as the streaming-SDF kernels — no grid_sample overhead,
no coordinate normalisation, and the same border-clamp semantics.

method="linear"    → bilinear (2-D) / trilinear (3-D)  [default]
method="quadratic" → biquadratic (2-D) / triquadratic (3-D), falls back
                     to linear at grid boundaries (< 3 cells on any axis)

2-D / 3-D is determined from the number of axes supplied to the
constructor (must be 2 or 3; 1 raises ValueError).
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .facade import interp_2d, interp_3d

__all__ = [
    "RegularGridInterpolator",
    "RegularGridInterpolator3D",
    "RegularGridInterpolatorAutomatic",
]

_VALID_METHODS = ("linear", "quadratic")


class RegularGridInterpolator:
    """Scattered-point interpolator on a uniform regular grid.

    Automatically selects the 2-D or 3-D kernel from the number of axes.

    Parameters
    ----------
    points : tuple of 1-D Tensors
        Axis coordinate arrays — exactly 2 (2-D) or 3 (3-D).
    F : Tensor
        Grid values, shape ``(Nx, Ny)`` or ``(Nx, Ny, Nz)``.
    method : "linear" | "quadratic"
        Interpolation order.  Default "linear" (bilinear / trilinear).
    fill_value : any
        Kept for API compatibility; ignored.  Out-of-bounds queries are
        always clamped to the nearest border value by the kernel.
    """

    def __init__(
        self,
        points: Sequence[Tensor],
        F: Tensor,
        method: str = "linear",
        fill_value=None,          # kept for drop-in compatibility; ignored
    ) -> None:
        ndim = len(points)
        if ndim < 2:
            raise ValueError(
                f"At least 2 grid axes (x, y) are required; got {ndim}. "
                "Pass (x, y) for 2-D or (x, y, z) for 3-D."
            )
        if ndim > 3:
            raise ValueError(
                f"At most 3 grid axes are supported; got {ndim}."
            )
        if method not in _VALID_METHODS:
            raise ValueError(
                f"method must be 'linear' or 'quadratic', got {method!r}."
            )

        self._ndim   = ndim
        self._method = method

        # Store axis tensors for attribute access compatibility
        axes = [ax.detach() for ax in points]
        self.x  = axes[0]
        self.y  = axes[1]
        self.z  = axes[2] if ndim == 3 else None

        # Pre-compute grid metadata (uniform spacing assumed)
        self.dx = float((axes[0][-1] - axes[0][0]) / max(axes[0].numel() - 1, 1))
        self.dy = float((axes[1][-1] - axes[1][0]) / max(axes[1].numel() - 1, 1))
        self.dz = (
            float((axes[2][-1] - axes[2][0]) / max(axes[2].numel() - 1, 1))
            if ndim == 3 else None
        )

        self._bx0    = float(axes[0][0])
        self._by0    = float(axes[1][0])
        self._bz0    = float(axes[2][0]) if ndim == 3 else 0.0
        self._inv_dx = 1.0 / self.dx if self.dx != 0.0 else 0.0
        self._inv_dy = 1.0 / self.dy if self.dy != 0.0 else 0.0
        self._inv_dz = (1.0 / self.dz if (self.dz is not None and self.dz != 0.0) else 0.0)
        self._Mx     = int(axes[0].numel())
        self._My     = int(axes[1].numel())
        self._Mz     = int(axes[2].numel()) if ndim == 3 else 1

        self._F = F.contiguous()

    # ------------------------------------------------------------------
    # .F property — reassigned every step in the CFD loop
    # ------------------------------------------------------------------
    @property
    def F(self) -> Tensor:
        return self._F

    @F.setter
    def F(self, val: Tensor) -> None:
        self._F = val.contiguous() if not val.is_contiguous() else val

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------
    def __call__(self, *coords: Tensor) -> Tensor:
        """Interpolate at scattered query points.

        Parameters
        ----------
        *coords : Tensor
            One 1-D tensor per dimension — ``(xq, yq)`` for 2-D or
            ``(xq, yq, zq)`` for 3-D.  All must have the same numel.

        Returns
        -------
        Tensor
            Interpolated values, same shape as each coordinate tensor.
        """
        if len(coords) != self._ndim:
            raise ValueError(
                f"Expected {self._ndim} coordinate tensors, got {len(coords)}."
            )
        orig_shape = coords[0].shape
        flat = [c.reshape(-1) for c in coords]

        if self._ndim == 2:
            result = interp_2d(
                self._F, flat[0], flat[1],
                self._bx0, self._by0,
                self._inv_dx, self._inv_dy,
                self._Mx, self._My,
                self._method,
            )
        else:
            result = interp_3d(
                self._F, flat[0], flat[1], flat[2],
                self._bx0, self._by0, self._bz0,
                self._inv_dx, self._inv_dy, self._inv_dz,
                self._Mx, self._My, self._Mz,
                self._method,
            )

        return result.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Aliases — same class handles 2-D and 3-D; "Automatic" / "3D" suffixes are
# kept for backward-compatible imports from sites that used to import them
# from pytorch_interpolation.
# ---------------------------------------------------------------------------

RegularGridInterpolatorAutomatic = RegularGridInterpolator
RegularGridInterpolator3D        = RegularGridInterpolator
