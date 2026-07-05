"""Scattered-point interpolators backed by the Warp ``interp_{2,3}d`` kernels.

Drop-in replacement for pytorch_interpolation's RegularGridInterpolator.

Also the home of the shared uniform-grid sampler ``@wp.func``s
(``bilinear``/``biquadratic``/``sdf_sample`` in 2-D, ``trilinear``/
``triquadratic`` in 3-D) — no grid_sample overhead, no coordinate
normalisation, and the same border-clamp semantics.  The streaming-SDF and
forces kernels import these from here.

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

# The scattered-gather Warp kernels these interpolators call (interp_2d /
# interp_3d) are defined at the bottom of this module — merged from the former
# misc_2d.py / misc_3d.py.  They pull the shared sampler @wp.func's from
# streaming_sdf (a leaf); this module is a leaf itself and must never import
# ``solver``, ``two_phase`` or ``facade``.

__all__ = [
    "RegularGridInterpolator",
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


# ═════════════════════════════════════════════════════════════════════════════
#  Shared uniform-grid sampler @wp.func's + scattered-gather Warp kernels.
#  The samplers (bilinear/biquadratic/sdf_sample in 2-D, trilinear/triquadratic
#  in 3-D) live here as the single source; streaming_sdf.py and forces.py import
#  them.  interp_2d / interp_3d back the RegularGridInterpolator class above.
#  All dtype-generic f32+f64 via wp.overload.
# ═════════════════════════════════════════════════════════════════════════════
from typing import Any  # noqa: E402
import warp as wp  # noqa: E402

wp.init()


# ─────────────────────────────────────────────────────────────────────────────
#  Trilinear interpolation on a uniform grid (with flat-array offset)
# ─────────────────────────────────────────────────────────────────────────────
@wp.func
def trilinear_sample_off(
    F:      wp.array(dtype=Any),
    F_off:  int,
    Mx: int, My: int, Mz: int,
    bx0: Any, by0: Any, bz0: Any,
    inv_dx: Any, inv_dy: Any, inv_dz: Any,
    xq: Any, yq: Any, zq: Any,
):
    """Trilinear interp into F_flat at offset F_off with border clamp."""
    zero = type(bx0)(0.0)
    one = type(bx0)(1.0)
    tx = wp.clamp((xq - bx0) * inv_dx, zero, type(bx0)(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, zero, type(bx0)(My - 1))
    tz = wp.clamp((zq - bz0) * inv_dz, zero, type(bx0)(Mz - 1))

    ix = wp.min(int(tx), Mx - 2)
    iy = wp.min(int(ty), My - 2)
    iz = wp.min(int(tz), Mz - 2)

    fx = tx - type(bx0)(ix)
    fy = ty - type(bx0)(iy)
    fz = tz - type(bx0)(iz)
    wx0 = one - fx
    wy0 = one - fy
    wz0 = one - fz

    s2   = Mz
    s1   = My * Mz
    base = F_off + ix * s1 + iy * s2 + iz

    return (
        wx0 * (wy0 * (wz0 * F[base]              + fz * F[base + 1]) +
               fy  * (wz0 * F[base + s2]          + fz * F[base + s2 + 1])) +
        fx  * (wy0 * (wz0 * F[base + s1]          + fz * F[base + s1 + 1]) +
               fy  * (wz0 * F[base + s1 + s2]     + fz * F[base + s1 + s2 + 1]))
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Bilinear / biquadratic interpolation on a uniform grid (flat offset)
# ─────────────────────────────────────────────────────────────────────────────
@wp.func
def bilinear_sample_off_2d(
    F:      wp.array(dtype=Any),
    F_off:  int,
    Mx: int, My: int,
    bx0: Any, by0: Any,
    inv_dx: Any, inv_dy: Any,
    xq: Any, yq: Any,
):
    """Bilinear interp into F_flat at offset F_off with border clamp."""
    tx = wp.clamp((xq - bx0) * inv_dx, type(bx0)(0.0), type(bx0)(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, type(bx0)(0.0), type(bx0)(My - 1))

    ix = wp.min(int(tx), Mx - 2)
    iy = wp.min(int(ty), My - 2)

    fx = tx - type(bx0)(ix)
    fy = ty - type(bx0)(iy)
    wx0 = type(bx0)(1.0) - fx
    wy0 = type(bx0)(1.0) - fy

    s1   = My
    base = F_off + ix * s1 + iy

    return (
        wx0 * (wy0 * F[base]      + fy * F[base + 1]) +
        fx  * (wy0 * F[base + s1] + fy * F[base + s1 + 1])
    )


@wp.func
def biquadratic_sample_off_2d(
    F:      wp.array(dtype=Any),
    F_off:  int,
    Mx: int, My: int,
    bx0: Any, by0: Any,
    inv_dx: Any, inv_dy: Any,
    xq: Any, yq: Any,
):
    """Biquadratic interp; falls back to bilinear at the table border (Mx/My<3
    or ix/iy<1), exactly matching ``biquadratic_sample_uniform_2d``."""
    tx = wp.clamp((xq - bx0) * inv_dx, type(bx0)(0.0), type(bx0)(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, type(bx0)(0.0), type(bx0)(My - 1))

    ix = wp.min(int(tx), Mx - 2)
    iy = wp.min(int(ty), My - 2)

    if ix < 1 or iy < 1 or Mx < 3 or My < 3:
        return bilinear_sample_off_2d(F, F_off, Mx, My, bx0, by0,
                                      inv_dx, inv_dy, xq, yq)

    fx = tx - type(bx0)(ix)
    fy = ty - type(bx0)(iy)

    half = type(bx0)(0.5)
    one = type(bx0)(1.0)
    wxm = half * fx * (fx - one)
    wx0 = one - fx * fx
    wxp = half * fx * (fx + one)
    wym = half * fy * (fy - one)
    wy0 = one - fy * fy
    wyp = half * fy * (fy + one)

    s1   = My
    base = F_off + (ix - 1) * s1 + (iy - 1)

    out = type(bx0)(0.0)
    # dx = 0
    col0 = wym * F[base]      + wy0 * F[base + 1]      + wyp * F[base + 2]
    out += wxm * col0
    # dx = 1
    b1 = base + s1
    col1 = wym * F[b1] + wy0 * F[b1 + 1] + wyp * F[b1 + 2]
    out += wx0 * col1
    # dx = 2
    b2 = base + 2 * s1
    col2 = wym * F[b2] + wy0 * F[b2 + 1] + wyp * F[b2 + 2]
    out += wxp * col2
    return out


@wp.func
def sdf_sample_off_2d(
    interp_method: int,
    F:      wp.array(dtype=Any),
    F_off:  int,
    Mx: int, My: int,
    bx0: Any, by0: Any,
    inv_dx: Any, inv_dy: Any,
    xq: Any, yq: Any,
):
    if interp_method == 1:
        return biquadratic_sample_off_2d(F, F_off, Mx, My, bx0, by0,
                                         inv_dx, inv_dy, xq, yq)
    return bilinear_sample_off_2d(F, F_off, Mx, My, bx0, by0,
                                  inv_dx, inv_dy, xq, yq)


@wp.kernel
def interp_2d_kernel(
    F:  wp.array(dtype=Any),
    xq: wp.array(dtype=Any),
    yq: wp.array(dtype=Any),
    N: int, Mx: int, My: int,
    bx0: Any, by0: Any,
    inv_dx: Any, inv_dy: Any,
    interp_method: int,
    G: wp.array(dtype=Any),
):
    tid = wp.tid()
    if tid >= N:
        return
    G[tid] = sdf_sample_off_2d(interp_method, F, 0, Mx, My,
                               bx0, by0, inv_dx, inv_dy, xq[tid], yq[tid])


for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(interp_2d_kernel,
                {"F": _A, "xq": _A, "yq": _A, "G": _A,
                 "bx0": _dt, "by0": _dt, "inv_dx": _dt, "inv_dy": _dt})


def interp_2d_warp(F_t, xq_t, yq_t, bx0, by0, inv_dx, inv_dy, Mx, My,
                   method="linear"):
    """Warp port of native ``interp_2d``.  Returns G shape (N,) in F's dtype."""
    wdev = "cuda:0" if F_t.device.type == "cuda" else "cpu"
    N = int(xq_t.numel())
    G = torch.empty(N, dtype=F_t.dtype, device=F_t.device)
    if N == 0:
        return G
    im = 1 if method == "quadratic" else 0
    wpf = wp.float64 if F_t.dtype == torch.float64 else wp.float32
    tdt = F_t.dtype
    wp.launch(interp_2d_kernel, dim=N,
              inputs=[wp.from_torch(F_t.reshape(-1).contiguous()),
                      wp.from_torch(xq_t.reshape(-1).contiguous().to(tdt)),
                      wp.from_torch(yq_t.reshape(-1).contiguous().to(tdt)),
                      N, int(Mx), int(My),
                      wpf(bx0), wpf(by0),
                      wpf(inv_dx), wpf(inv_dy), im,
                      wp.from_torch(G)],
              device=wdev)
    return G


@wp.func
def triquadratic_sample_off(
    F: wp.array(dtype=Any), F_off: int,
    Mx: int, My: int, Mz: int,
    bx0: Any, by0: Any, bz0: Any,
    inv_dx: Any, inv_dy: Any, inv_dz: Any,
    xq: Any, yq: Any, zq: Any,
):
    zero = type(xq)(0.0)
    one = type(xq)(1.0)
    half = type(xq)(0.5)
    tx = wp.clamp((xq - bx0) * inv_dx, zero, type(xq)(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, zero, type(xq)(My - 1))
    tz = wp.clamp((zq - bz0) * inv_dz, zero, type(xq)(Mz - 1))
    ix = wp.min(int(tx), Mx - 2)
    iy = wp.min(int(ty), My - 2)
    iz = wp.min(int(tz), Mz - 2)
    if ix < 1 or iy < 1 or iz < 1 or Mx < 3 or My < 3 or Mz < 3:
        return trilinear_sample_off(F, F_off, Mx, My, Mz, bx0, by0, bz0,
                                    inv_dx, inv_dy, inv_dz, xq, yq, zq)
    fx = tx - type(xq)(ix); fy = ty - type(xq)(iy); fz = tz - type(xq)(iz)
    wxm = half*fx*(fx-one); wx0 = one-fx*fx; wxp = half*fx*(fx+one)
    wym = half*fy*(fy-one); wy0 = one-fy*fy; wyp = half*fy*(fy+one)
    wzm = half*fz*(fz-one); wz0 = one-fz*fz; wzp = half*fz*(fz+one)
    s2 = Mz
    s1 = My * Mz
    base = F_off + (ix-1)*s1 + (iy-1)*s2 + (iz-1)
    out = zero
    for dx in range(3):
        wx = wxm
        if dx == 1: wx = wx0
        if dx == 2: wx = wxp
        b0 = base + dx*s1
        plane = zero
        for dy in range(3):
            wy = wym
            if dy == 1: wy = wy0
            if dy == 2: wy = wyp
            b1 = b0 + dy*s2
            row = wzm*F[b1] + wz0*F[b1+1] + wzp*F[b1+2]
            plane = plane + wy*row
        out = out + wx*plane
    return out


@wp.kernel
def interp_3d_kernel(
    F: wp.array(dtype=Any),
    xq: wp.array(dtype=Any), yq: wp.array(dtype=Any),
    zq: wp.array(dtype=Any),
    N: int, Mx: int, My: int, Mz: int,
    bx0: Any, by0: Any, bz0: Any,
    idx: Any, idy: Any, idz: Any,
    interp_method: int,
    G: wp.array(dtype=Any),
):
    tid = wp.tid()
    if tid >= N:
        return
    if interp_method == 1:
        G[tid] = triquadratic_sample_off(F, 0, Mx, My, Mz, bx0, by0, bz0,
                                         idx, idy, idz, xq[tid], yq[tid], zq[tid])
    else:
        G[tid] = trilinear_sample_off(F, 0, Mx, My, Mz, bx0, by0, bz0,
                                      idx, idy, idz, xq[tid], yq[tid], zq[tid])


for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(interp_3d_kernel,
                {"F": _A, "xq": _A, "yq": _A, "zq": _A, "G": _A,
                 "bx0": _dt, "by0": _dt, "bz0": _dt,
                 "idx": _dt, "idy": _dt, "idz": _dt})


def interp_3d_warp(F_t, xq_t, yq_t, zq_t, bx0, by0, bz0, idx, idy, idz,
                   Mx, My, Mz, method="linear"):
    """Warp port of native ``interp_3d``.  Returns G shape (N,) in F's dtype."""
    wdev = "cuda:0" if F_t.device.type == "cuda" else "cpu"
    N = int(xq_t.numel())
    G = torch.empty(N, dtype=F_t.dtype, device=F_t.device)
    if N == 0:
        return G
    im = 1 if method == "quadratic" else 0
    tdt = F_t.dtype
    wpf = wp.float64 if tdt == torch.float64 else wp.float32
    cast = lambda t: wp.from_torch(t.reshape(-1).contiguous().to(tdt))
    wp.launch(interp_3d_kernel, dim=N,
              inputs=[wp.from_torch(F_t.reshape(-1).contiguous()),
                      cast(xq_t), cast(yq_t), cast(zq_t),
                      N, int(Mx), int(My), int(Mz),
                      wpf(bx0), wpf(by0), wpf(bz0),
                      wpf(idx), wpf(idy), wpf(idz),
                      im, wp.from_torch(G)],
              device=wdev)
    return G


# Names used by the RegularGridInterpolator classes above (resolved at call time).
interp_2d = interp_2d_warp
interp_3d = interp_3d_warp
