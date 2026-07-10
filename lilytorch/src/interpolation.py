"""Scattered-point interpolation + shared uniform-grid samplers.

``RegularGridInterpolator`` (a drop-in for pytorch_interpolation's) is
re-exported from :mod:`lilytorch.src.native` — the native CUDA/C++
``interp_{2,3}d`` ops (cuda_native_port Track A).  The Warp ``interp_{2,3}d_warp``
scattered-gather kernels below are retained as the parity oracle.

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

import torch  # noqa: F401  (kept for the Warp scattered-gather kernels below)
from torch import Tensor  # noqa: F401

# cuda_native_port Phase 0.2 (Track A): the scattered-point interpolator IS the
# native CUDA/C++ ``RegularGridInterpolator`` (backed by the ``interp_{2,3}d``
# ops — both a CUDA kernel and an ``at::parallel_for`` CPU twin).  It is a
# verbatim drop-in for the former Warp-backed class defined here (same
# constructor, ``.F`` setter, ``.x/.y/.z`` metadata, ``__call__``).  The Warp
# scattered-gather kernels ``interp_{2,3}d_warp`` remain at the bottom of this
# module as the parity oracle (tests/test_interpolation.py), and the shared
# uniform-grid ``@wp.func`` samplers stay for the streaming-SDF / forces kernels.
#
# ``native`` is a true leaf (imports only the ``_C`` extension + torch), so this
# does not break the "interpolation must not import solver/two_phase/facade" rule.
from lilytorch.src.native import RegularGridInterpolator

__all__ = [
    "RegularGridInterpolator",
]


# ═════════════════════════════════════════════════════════════════════════════
#  Shared uniform-grid sampler @wp.func's + scattered-gather Warp kernels.
#  The samplers (bilinear/biquadratic/sdf_sample in 2-D, trilinear/triquadratic
#  in 3-D) live here as the single source; streaming_sdf.py and forces.py import
#  them.  ``interp_{2,3}d_warp`` are the parity oracle for the native
#  ``RegularGridInterpolator`` (re-exported at the top of this module).
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
