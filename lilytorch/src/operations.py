"""
Differential operators and derived quantities (gradient, divergence,
vorticity, …) used by the fluid solver.

Extracted from solver.py so they can be maintained independently.
All functions are standalone — they take explicit grid parameters
(h, ndim, dx, dy, dz) instead of relying on ``self``.
"""

from __future__ import annotations

import torch
import warp as wp

wp.init()

# ------------------------------------------------------------------
#  Warp helpers  (same pattern as diffusion.py)
# ------------------------------------------------------------------


def _wp_device(t: torch.Tensor) -> str:
    return f"cuda:{t.device.index}" if t.is_cuda else "cpu"


def _wp_dtype(t: torch.Tensor):
    return wp.float64 if t.dtype == torch.float64 else wp.float32


def _flat(t: torch.Tensor):
    """Zero-copy flat Warp view (f32/f64) over t's storage."""
    assert t.dtype in (torch.float64, torch.float32), "warp ops: f32/f64 only"
    elem = t.element_size()
    remaining = (t.untyped_storage().nbytes() - t.storage_offset() * elem) // elem
    return wp.array(ptr=t.data_ptr(), dtype=_wp_dtype(t),
                    shape=(int(remaining),), device=_wp_device(t))


_FLAT_VIEW_CACHE: dict = {}


def _flat_cached(t: torch.Tensor):
    """Cached :func:`_flat` — reuse the Warp view for a given buffer/layout."""
    elem = t.element_size()
    remaining = (t.untyped_storage().nbytes() - t.storage_offset() * elem) // elem
    key = (t.data_ptr(), int(remaining), t.dtype, _wp_device(t))
    view = _FLAT_VIEW_CACHE.get(key)
    if view is None:
        view = _flat(t)
        _FLAT_VIEW_CACHE[key] = view
    return view


# =====================================================================
#  Warp strain-rate magnitude kernel  (unified 2-D / 3-D, f32 / f64)
#
#  2-D is handled as the degenerate 3-D case with Nz=1, Niz=1 so the
#  z-derivative terms vanish.  Matches the stencil of the legacy
#  torch.gradient(edge_order=2) + _stag_to_cc path bit-for-bit at
#  interior points.
# =====================================================================

from typing import Any  # noqa: E402


@wp.kernel
def strain_rate_magnitude_warp_kernel(
    u: wp.array(dtype=Any),
    v: wp.array(dtype=Any),
    w: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
    is_3d: int,                         # compile-time: 0 for 2-D, 1 for 3-D
    Nx: int, Ny: int, Nz: int,         # full grid dims (Nz >= 1)
    su_x: int, su_y: int, su_z: int,   # u strides (su_z=0 for 2-D)
    sv_x: int, sv_y: int, sv_z: int,   # v strides
    sw_x: int, sw_y: int, sw_z: int,   # w strides (all 0 for 2-D)
    so_x: int, so_y: int, so_z: int,   # out strides
    inv_2h: Any,                        # 1/(2h)
):
    """Compute strain-rate magnitude  |S̄| = sqrt(2·S_ij·S_ij)  at every
    grid point of the u-array (x-face MAC grid), matching the reference
    ``torch.gradient(…, edge_order=2) + _stag_to_cc`` path.

    2-D / 3-D unified: when ``is_3d=0`` Warp dead-code-eliminates the z
    branches; Nz=1 and all z-strides are 0."""
    tid = wp.tid()
    total = Nx * Ny * Nz
    if tid >= total:
        return

    # Decode flat tid → (i, j, k); k is fastest-varying.
    k = tid % Nz
    ij = tid // Nz
    j = ij % Ny
    i = ij // Ny

    idx_u = i * su_x + j * su_y + k * su_z
    idx_v = i * sv_x + j * sv_y + k * sv_z
    idx_o = i * so_x + j * so_y + k * so_z

    two = type(u[0])(2.0)
    half = type(u[0])(0.5)

    # ── 1-D gradient helper (matches torch.gradient edge_order=2) ──
    # grad1d(f, i, stride, N) → (f[i+1] - f[i-1])/(2h)  interior
    #                           one-sided 3-pt at boundaries

    # ---- dudx (gradient of u along dim 0, at u-grid point (i,j,k)) ----
    if i == 0:
        dudx = (-type(u[0])(3.0) * u[idx_u]
                + type(u[0])(4.0) * u[idx_u + su_x]
                - u[idx_u + 2 * su_x]) * inv_2h
    elif i == Nx - 1:
        dudx = (type(u[0])(3.0) * u[idx_u]
                - type(u[0])(4.0) * u[idx_u - su_x]
                + u[idx_u - 2 * su_x]) * inv_2h
    else:
        dudx = (u[idx_u + su_x] - u[idx_u - su_x]) * inv_2h

    # ---- dudy_raw (gradient of u along dim 1, at u-grid point) ----
    if j == 0:
        dudy_raw = (-type(u[0])(3.0) * u[idx_u]
                    + type(u[0])(4.0) * u[idx_u + su_y]
                    - u[idx_u + 2 * su_y]) * inv_2h
    elif j == Ny - 1:
        dudy_raw = (type(u[0])(3.0) * u[idx_u]
                    - type(u[0])(4.0) * u[idx_u - su_y]
                    + u[idx_u - 2 * su_y]) * inv_2h
    else:
        dudy_raw = (u[idx_u + su_y] - u[idx_u - su_y]) * inv_2h

    # ---- dvdx_raw (gradient of v along dim 0, at v-grid point) ----
    if i == 0:
        dvdx_raw = (-type(v[0])(3.0) * v[idx_v]
                    + type(v[0])(4.0) * v[idx_v + sv_x]
                    - v[idx_v + 2 * sv_x]) * inv_2h
    elif i == Nx - 1:
        dvdx_raw = (type(v[0])(3.0) * v[idx_v]
                    - type(v[0])(4.0) * v[idx_v - sv_x]
                    + v[idx_v - 2 * sv_x]) * inv_2h
    else:
        dvdx_raw = (v[idx_v + sv_x] - v[idx_v - sv_x]) * inv_2h

    # ---- dvdy (gradient of v along dim 1, at v-grid point) ----
    if j == 0:
        dvdy = (-type(v[0])(3.0) * v[idx_v]
                + type(v[0])(4.0) * v[idx_v + sv_y]
                - v[idx_v + 2 * sv_y]) * inv_2h
    elif j == Ny - 1:
        dvdy = (type(v[0])(3.0) * v[idx_v]
                - type(v[0])(4.0) * v[idx_v - sv_y]
                + v[idx_v - 2 * sv_y]) * inv_2h
    else:
        dvdy = (v[idx_v + sv_y] - v[idx_v - sv_y]) * inv_2h

    # ---- _stag_to_cc for dudy (average along dim 0, replicate at last) ----
    if i < Nx - 1:
        # dudy_raw[i+1] = grad of u along y at (i+1, j, k)
        idx_u_xp1 = idx_u + su_x
        if j == 0:
            dudy_raw_xp1 = (-type(u[0])(3.0) * u[idx_u_xp1]
                            + type(u[0])(4.0) * u[idx_u_xp1 + su_y]
                            - u[idx_u_xp1 + 2 * su_y]) * inv_2h
        elif j == Ny - 1:
            dudy_raw_xp1 = (type(u[0])(3.0) * u[idx_u_xp1]
                            - type(u[0])(4.0) * u[idx_u_xp1 - su_y]
                            + u[idx_u_xp1 - 2 * su_y]) * inv_2h
        else:
            dudy_raw_xp1 = (u[idx_u_xp1 + su_y] - u[idx_u_xp1 - su_y]) * inv_2h
        dudy = half * (dudy_raw + dudy_raw_xp1)
    else:
        # i == Nx-1: replicate from i=Nx-2
        # Recompute dudy at Nx-2 using the same formula
        idx_u_xm1 = idx_u - su_x
        if j == 0:
            dudy_raw_xm1 = (-type(u[0])(3.0) * u[idx_u_xm1]
                            + type(u[0])(4.0) * u[idx_u_xm1 + su_y]
                            - u[idx_u_xm1 + 2 * su_y]) * inv_2h
        elif j == Ny - 1:
            dudy_raw_xm1 = (type(u[0])(3.0) * u[idx_u_xm1]
                            - type(u[0])(4.0) * u[idx_u_xm1 - su_y]
                            + u[idx_u_xm1 - 2 * su_y]) * inv_2h
        else:
            dudy_raw_xm1 = (u[idx_u_xm1 + su_y] - u[idx_u_xm1 - su_y]) * inv_2h
        # dudy_raw at i (Nx-1) is dudy_raw computed above
        # dudy_cc at Nx-2 = 0.5*(dudy_raw_xm1 + dudy_raw)
        dudy = half * (dudy_raw_xm1 + dudy_raw)

    # ---- _stag_to_cc for dvdx (average along dim 1, replicate at last) ----
    if j < Ny - 1:
        idx_v_yp1 = idx_v + sv_y
        if i == 0:
            dvdx_raw_yp1 = (-type(v[0])(3.0) * v[idx_v_yp1]
                            + type(v[0])(4.0) * v[idx_v_yp1 + sv_x]
                            - v[idx_v_yp1 + 2 * sv_x]) * inv_2h
        elif i == Nx - 1:
            dvdx_raw_yp1 = (type(v[0])(3.0) * v[idx_v_yp1]
                            - type(v[0])(4.0) * v[idx_v_yp1 - sv_x]
                            + v[idx_v_yp1 - 2 * sv_x]) * inv_2h
        else:
            dvdx_raw_yp1 = (v[idx_v_yp1 + sv_x] - v[idx_v_yp1 - sv_x]) * inv_2h
        dvdx = half * (dvdx_raw + dvdx_raw_yp1)
    else:
        # j == Ny-1: replicate from j=Ny-2
        idx_v_ym1 = idx_v - sv_y
        if i == 0:
            dvdx_raw_ym1 = (-type(v[0])(3.0) * v[idx_v_ym1]
                            + type(v[0])(4.0) * v[idx_v_ym1 + sv_x]
                            - v[idx_v_ym1 + 2 * sv_x]) * inv_2h
        elif i == Nx - 1:
            dvdx_raw_ym1 = (type(v[0])(3.0) * v[idx_v_ym1]
                            - type(v[0])(4.0) * v[idx_v_ym1 - sv_x]
                            + v[idx_v_ym1 - 2 * sv_x]) * inv_2h
        else:
            dvdx_raw_ym1 = (v[idx_v_ym1 + sv_x] - v[idx_v_ym1 - sv_x]) * inv_2h
        dvdx = half * (dvdx_raw_ym1 + dvdx_raw)

    # ---- Assemble 2-D S_ij S_ij ----
    S2 = dudx * dudx + dvdy * dvdy + half * (dudy + dvdx) * (dudy + dvdx)

    if is_3d:
        idx_w = i * sw_x + j * sw_y + k * sw_z

        # dudz_raw = grad of u along dim 2 (z)
        if k == 0:
            dudz_raw = (-type(u[0])(3.0) * u[idx_u]
                        + type(u[0])(4.0) * u[idx_u + su_z]
                        - u[idx_u + 2 * su_z]) * inv_2h
        elif k == Nz - 1:
            dudz_raw = (type(u[0])(3.0) * u[idx_u]
                        - type(u[0])(4.0) * u[idx_u - su_z]
                        + u[idx_u - 2 * su_z]) * inv_2h
        else:
            dudz_raw = (u[idx_u + su_z] - u[idx_u - su_z]) * inv_2h

        # dwdx_raw = grad of w along dim 0 (x)
        if i == 0:
            dwdx_raw = (-type(w[0])(3.0) * w[idx_w]
                        + type(w[0])(4.0) * w[idx_w + sw_x]
                        - w[idx_w + 2 * sw_x]) * inv_2h
        elif i == Nx - 1:
            dwdx_raw = (type(w[0])(3.0) * w[idx_w]
                        - type(w[0])(4.0) * w[idx_w - sw_x]
                        + w[idx_w - 2 * sw_x]) * inv_2h
        else:
            dwdx_raw = (w[idx_w + sw_x] - w[idx_w - sw_x]) * inv_2h

        # dvdz_raw = grad of v along dim 2 (z)
        if k == 0:
            dvdz_raw = (-type(v[0])(3.0) * v[idx_v]
                        + type(v[0])(4.0) * v[idx_v + sv_z]
                        - v[idx_v + 2 * sv_z]) * inv_2h
        elif k == Nz - 1:
            dvdz_raw = (type(v[0])(3.0) * v[idx_v]
                        - type(v[0])(4.0) * v[idx_v - sv_z]
                        + v[idx_v - 2 * sv_z]) * inv_2h
        else:
            dvdz_raw = (v[idx_v + sv_z] - v[idx_v - sv_z]) * inv_2h

        # dwdy_raw = grad of w along dim 1 (y)
        if j == 0:
            dwdy_raw = (-type(w[0])(3.0) * w[idx_w]
                        + type(w[0])(4.0) * w[idx_w + sw_y]
                        - w[idx_w + 2 * sw_y]) * inv_2h
        elif j == Ny - 1:
            dwdy_raw = (type(w[0])(3.0) * w[idx_w]
                        - type(w[0])(4.0) * w[idx_w - sw_y]
                        + w[idx_w - 2 * sw_y]) * inv_2h
        else:
            dwdy_raw = (w[idx_w + sw_y] - w[idx_w - sw_y]) * inv_2h

        # dwdz = grad of w along dim 2 (z)
        if k == 0:
            dwdz = (-type(w[0])(3.0) * w[idx_w]
                    + type(w[0])(4.0) * w[idx_w + sw_z]
                    - w[idx_w + 2 * sw_z]) * inv_2h
        elif k == Nz - 1:
            dwdz = (type(w[0])(3.0) * w[idx_w]
                    - type(w[0])(4.0) * w[idx_w - sw_z]
                    + w[idx_w - 2 * sw_z]) * inv_2h
        else:
            dwdz = (w[idx_w + sw_z] - w[idx_w - sw_z]) * inv_2h

        # _stag_to_cc for dudz (average along dim 0)
        if i < Nx - 1:
            idx_u_xp1 = idx_u + su_x
            if k == 0:
                dudz_raw_xp1 = (-type(u[0])(3.0) * u[idx_u_xp1]
                                + type(u[0])(4.0) * u[idx_u_xp1 + su_z]
                                - u[idx_u_xp1 + 2 * su_z]) * inv_2h
            elif k == Nz - 1:
                dudz_raw_xp1 = (type(u[0])(3.0) * u[idx_u_xp1]
                                - type(u[0])(4.0) * u[idx_u_xp1 - su_z]
                                + u[idx_u_xp1 - 2 * su_z]) * inv_2h
            else:
                dudz_raw_xp1 = (u[idx_u_xp1 + su_z] - u[idx_u_xp1 - su_z]) * inv_2h
            dudz = half * (dudz_raw + dudz_raw_xp1)
        else:
            idx_u_xm1 = idx_u - su_x
            if k == 0:
                dudz_raw_xm1 = (-type(u[0])(3.0) * u[idx_u_xm1]
                                + type(u[0])(4.0) * u[idx_u_xm1 + su_z]
                                - u[idx_u_xm1 + 2 * su_z]) * inv_2h
            elif k == Nz - 1:
                dudz_raw_xm1 = (type(u[0])(3.0) * u[idx_u_xm1]
                                - type(u[0])(4.0) * u[idx_u_xm1 - su_z]
                                + u[idx_u_xm1 - 2 * su_z]) * inv_2h
            else:
                dudz_raw_xm1 = (u[idx_u_xm1 + su_z] - u[idx_u_xm1 - su_z]) * inv_2h
            dudz = half * (dudz_raw_xm1 + dudz_raw)

        # _stag_to_cc for dwdx (average along dim 2)
        if k < Nz - 1:
            idx_w_zp1 = idx_w + sw_z
            if i == 0:
                dwdx_raw_zp1 = (-type(w[0])(3.0) * w[idx_w_zp1]
                                + type(w[0])(4.0) * w[idx_w_zp1 + sw_x]
                                - w[idx_w_zp1 + 2 * sw_x]) * inv_2h
            elif i == Nx - 1:
                dwdx_raw_zp1 = (type(w[0])(3.0) * w[idx_w_zp1]
                                - type(w[0])(4.0) * w[idx_w_zp1 - sw_x]
                                + w[idx_w_zp1 - 2 * sw_x]) * inv_2h
            else:
                dwdx_raw_zp1 = (w[idx_w_zp1 + sw_x] - w[idx_w_zp1 - sw_x]) * inv_2h
            dwdx = half * (dwdx_raw + dwdx_raw_zp1)
        else:
            idx_w_zm1 = idx_w - sw_z
            if i == 0:
                dwdx_raw_zm1 = (-type(w[0])(3.0) * w[idx_w_zm1]
                                + type(w[0])(4.0) * w[idx_w_zm1 + sw_x]
                                - w[idx_w_zm1 + 2 * sw_x]) * inv_2h
            elif i == Nx - 1:
                dwdx_raw_zm1 = (type(w[0])(3.0) * w[idx_w_zm1]
                                - type(w[0])(4.0) * w[idx_w_zm1 - sw_x]
                                + w[idx_w_zm1 - 2 * sw_x]) * inv_2h
            else:
                dwdx_raw_zm1 = (w[idx_w_zm1 + sw_x] - w[idx_w_zm1 - sw_x]) * inv_2h
            dwdx = half * (dwdx_raw_zm1 + dwdx_raw)

        # _stag_to_cc for dvdz (average along dim 1)
        if j < Ny - 1:
            idx_v_yp1 = idx_v + sv_y
            if k == 0:
                dvdz_raw_yp1 = (-type(v[0])(3.0) * v[idx_v_yp1]
                                + type(v[0])(4.0) * v[idx_v_yp1 + sv_z]
                                - v[idx_v_yp1 + 2 * sv_z]) * inv_2h
            elif k == Nz - 1:
                dvdz_raw_yp1 = (type(v[0])(3.0) * v[idx_v_yp1]
                                - type(v[0])(4.0) * v[idx_v_yp1 - sv_z]
                                + v[idx_v_yp1 - 2 * sv_z]) * inv_2h
            else:
                dvdz_raw_yp1 = (v[idx_v_yp1 + sv_z] - v[idx_v_yp1 - sv_z]) * inv_2h
            dvdz = half * (dvdz_raw + dvdz_raw_yp1)
        else:
            idx_v_ym1 = idx_v - sv_y
            if k == 0:
                dvdz_raw_ym1 = (-type(v[0])(3.0) * v[idx_v_ym1]
                                + type(v[0])(4.0) * v[idx_v_ym1 + sv_z]
                                - v[idx_v_ym1 + 2 * sv_z]) * inv_2h
            elif k == Nz - 1:
                dvdz_raw_ym1 = (type(v[0])(3.0) * v[idx_v_ym1]
                                - type(v[0])(4.0) * v[idx_v_ym1 - sv_z]
                                + v[idx_v_ym1 - 2 * sv_z]) * inv_2h
            else:
                dvdz_raw_ym1 = (v[idx_v_ym1 + sv_z] - v[idx_v_ym1 - sv_z]) * inv_2h
            dvdz = half * (dvdz_raw_ym1 + dvdz_raw)

        # _stag_to_cc for dwdy (average along dim 2)
        if k < Nz - 1:
            idx_w_zp1 = idx_w + sw_z
            if j == 0:
                dwdy_raw_zp1 = (-type(w[0])(3.0) * w[idx_w_zp1]
                                + type(w[0])(4.0) * w[idx_w_zp1 + sw_y]
                                - w[idx_w_zp1 + 2 * sw_y]) * inv_2h
            elif j == Ny - 1:
                dwdy_raw_zp1 = (type(w[0])(3.0) * w[idx_w_zp1]
                                - type(w[0])(4.0) * w[idx_w_zp1 - sw_y]
                                + w[idx_w_zp1 - 2 * sw_y]) * inv_2h
            else:
                dwdy_raw_zp1 = (w[idx_w_zp1 + sw_y] - w[idx_w_zp1 - sw_y]) * inv_2h
            dwdy = half * (dwdy_raw + dwdy_raw_zp1)
        else:
            idx_w_zm1 = idx_w - sw_z
            if j == 0:
                dwdy_raw_zm1 = (-type(w[0])(3.0) * w[idx_w_zm1]
                                + type(w[0])(4.0) * w[idx_w_zm1 + sw_y]
                                - w[idx_w_zm1 + 2 * sw_y]) * inv_2h
            elif j == Ny - 1:
                dwdy_raw_zm1 = (type(w[0])(3.0) * w[idx_w_zm1]
                                - type(w[0])(4.0) * w[idx_w_zm1 - sw_y]
                                + w[idx_w_zm1 - 2 * sw_y]) * inv_2h
            else:
                dwdy_raw_zm1 = (w[idx_w_zm1 + sw_y] - w[idx_w_zm1 - sw_y]) * inv_2h
            dwdy = half * (dwdy_raw_zm1 + dwdy_raw)

        # Add 3-D terms to S2
        S2 += dwdz * dwdz
        S2 += half * (dudz + dwdx) * (dudz + dwdx)
        S2 += half * (dvdz + dwdy) * (dvdz + dwdy)

    # |S̄| = sqrt(2 · S_ij·S_ij)
    out[idx_o] = wp.sqrt(two * S2)


# Register float32 + float64 specialisations
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(strain_rate_magnitude_warp_kernel, {
        "u": _A, "v": _A, "w": _A, "out": _A,
        "inv_2h": _dt,
    })


# ── Persistent output buffer (module-level, like diffusion.py) ──
_smag_persist: torch.Tensor | None = None


def _strain_rate_magnitude_warp_eager(u_t, v_t, w_t, h):
    """Eager (non-graph) launch of :func:`strain_rate_magnitude_warp_kernel`.

    Returns a persistent full-grid ``|S̄|`` tensor with the same shape as
    ``u_t``.  The buffer is reused across calls with the same shape/dtype/
    device; only the interior values are recomputed each invocation.
    Ghost cells are written to match the legacy ``torch.gradient(…,
    edge_order=2) + _stag_to_cc`` path exactly.
    """
    global _smag_persist
    if (_smag_persist is None
            or _smag_persist.shape != u_t.shape
            or _smag_persist.dtype != u_t.dtype
            or _smag_persist.device != u_t.device):
        _smag_persist = torch.empty_like(u_t)

    out_t = _smag_persist

    ndim = u_t.ndim
    shape = u_t.shape
    Nx, Ny = shape[0], shape[1]
    Nz = shape[2] if ndim == 3 else 1
    is_3d = 1 if (ndim == 3 and w_t is not None) else 0

    u_stride = u_t.stride()
    v_stride = v_t.stride()
    o_stride = out_t.stride()

    su_x, su_y = int(u_stride[0]), int(u_stride[1])
    su_z = int(u_stride[2]) if ndim == 3 else 0
    sv_x, sv_y = int(v_stride[0]), int(v_stride[1])
    sv_z = int(v_stride[2]) if ndim == 3 else 0
    so_x, so_y = int(o_stride[0]), int(o_stride[1])
    so_z = int(o_stride[2]) if ndim == 3 else 0

    if is_3d:
        w_stride = w_t.stride()
        sw_x, sw_y = int(w_stride[0]), int(w_stride[1])
        sw_z = int(w_stride[2])
        _w_flat = _flat_cached(w_t)
    else:
        sw_x = sw_y = sw_z = 0
        # Dummy w array (never read — Warp eliminates the dead branch)
        _w_flat = _flat_cached(torch.empty(1, dtype=u_t.dtype, device=u_t.device))

    wpf = _wp_dtype(u_t)
    dev = _wp_device(u_t)

    wp.launch(
        strain_rate_magnitude_warp_kernel,
        dim=Nx * Ny * Nz,
        inputs=[
            _flat_cached(u_t), _flat_cached(v_t), _w_flat,
            _flat_cached(out_t),
            is_3d,
            Nx, Ny, Nz,
            su_x, su_y, su_z,
            sv_x, sv_y, sv_z,
            sw_x, sw_y, sw_z,
            so_x, so_y, so_z,
            wpf(0.5 / float(h)),
        ],
        device=dev,
    )
    return out_t


# ------------------------------------------------------------------
# Reciprocal helper — accepts float or Tensor, returns Tensor reciprocal
# ------------------------------------------------------------------
def _recip(val):
    """Return the reciprocal of *val* as a Tensor.

    Multiply-by-reciprocal is deterministic across CPU and CUDA
    (avoids potential division rounding differences in torch ops).
    """
    if isinstance(val, torch.Tensor):
        return val.reciprocal()
    return 1.0 / val  # plain float — exact


# ------------------------------------------------------------------
# First-order partial derivatives
# ------------------------------------------------------------------
def compute_dpdx(p, h):
    """Compute dp/dx via central difference (multiply-by-reciprocal)."""
    dp = torch.zeros_like(p)
    inv_2h = _recip(2.0 * h) if isinstance(h, torch.Tensor) else 1.0 / (2.0 * h)
    if p.ndim == 2:
        dp[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) * inv_2h
    else:
        dp[1:-1, 1:-1, 1:-1] = (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) * inv_2h
    return dp


def compute_dpdy(p, h):
    """Compute dp/dy via central difference (multiply-by-reciprocal)."""
    dp = torch.zeros_like(p)
    inv_2h = _recip(2.0 * h) if isinstance(h, torch.Tensor) else 1.0 / (2.0 * h)
    if p.ndim == 2:
        dp[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) * inv_2h
    else:
        dp[1:-1, 1:-1, 1:-1] = (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) * inv_2h
    return dp


def compute_dpdz(p, h):
    """Compute dp/dz (3-D only, multiply-by-reciprocal)."""
    dp = torch.zeros_like(p)
    inv_2h = _recip(2.0 * h) if isinstance(h, torch.Tensor) else 1.0 / (2.0 * h)
    dp[1:-1, 1:-1, 1:-1] = (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) * inv_2h
    return dp


# ------------------------------------------------------------------
# Gradient
# ------------------------------------------------------------------
def gradient(var, h, ndim):
    """
    Compute gradient(var) using multiply-by-reciprocal (CPU/GPU parity).
    2-D → (dvar_dx, dvar_dy),  3-D → (dvar_dx, dvar_dy, dvar_dz).
    """
    inv_h = _recip(h)
    dvar_dx = torch.zeros_like(var)
    dvar_dy = torch.zeros_like(var)
    if ndim == 2:
        dvar_dx[1:-1, 1:-1] = (var[1:-1, 1:-1] - var[:-2, 1:-1]) * inv_h
        dvar_dy[1:-1, 1:-1] = (var[1:-1, 1:-1] - var[1:-1, :-2]) * inv_h
        return (dvar_dx, dvar_dy)
    else:
        dvar_dz = torch.zeros_like(var)
        dvar_dx[1:-1, 1:-1, 1:-1] = (var[1:-1, 1:-1, 1:-1] - var[:-2, 1:-1, 1:-1]) * inv_h
        dvar_dy[1:-1, 1:-1, 1:-1] = (var[1:-1, 1:-1, 1:-1] - var[1:-1, :-2, 1:-1]) * inv_h
        dvar_dz[1:-1, 1:-1, 1:-1] = (var[1:-1, 1:-1, 1:-1] - var[1:-1, 1:-1, :-2]) * inv_h
        return (dvar_dx, dvar_dy, dvar_dz)


# ------------------------------------------------------------------
# Divergence
# ------------------------------------------------------------------
def divergence(u, v, dx, dy, w=None, dz=None):
    """Compute the divergence using multiply-by-reciprocal (CPU/GPU parity)."""
    inv_dx = _recip(dx)
    inv_dy = _recip(dy)
    div = torch.zeros_like(u)
    if w is None:
        div[1:-1, 1:-1] = ((u[2:, 1:-1] - u[1:-1, 1:-1]) * inv_dx
                         + (v[1:-1, 2:] - v[1:-1, 1:-1]) * inv_dy)
    else:
        inv_dz = _recip(dz)
        div[1:-1, 1:-1, 1:-1] = (
            (u[2:, 1:-1, 1:-1] - u[1:-1, 1:-1, 1:-1]) * inv_dx
          + (v[1:-1, 2:, 1:-1] - v[1:-1, 1:-1, 1:-1]) * inv_dy
          + (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, 1:-1]) * inv_dz
        )
    return div


def divergence_interior(u, v, dx, dy, w=None, dz=None):
    """Interior-only divergence: returns shape (Nx, Ny) or (Nx, Ny, Nz).

    Equivalent to ``divergence(...)[1:-1, 1:-1, ...]`` but allocates only the
    interior cells, skipping the ghost-cell wrapper.  Used by the multigrid
    path (T3a) to eliminate the full-grid ``div`` buffer during the Poisson
    solve.
    """
    inv_dx = _recip(dx)
    inv_dy = _recip(dy)
    if w is None:
        return ((u[2:, 1:-1] - u[1:-1, 1:-1]) * inv_dx
              + (v[1:-1, 2:] - v[1:-1, 1:-1]) * inv_dy)
    inv_dz = _recip(dz)
    return ((u[2:, 1:-1, 1:-1] - u[1:-1, 1:-1, 1:-1]) * inv_dx
          + (v[1:-1, 2:, 1:-1] - v[1:-1, 1:-1, 1:-1]) * inv_dy
          + (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, 1:-1]) * inv_dz)


# ------------------------------------------------------------------
# Normal derivative
# ------------------------------------------------------------------
def normal_derivative(var, h, ndim, normal_x, normal_y, normal_z=None):
    """Compute the normal derivative dvar/dn = n · ∇var."""
    nd = normal_x * compute_dpdx(var, h) + normal_y * compute_dpdy(var, h)
    if ndim == 3 and normal_z is not None:
        nd = nd + normal_z * compute_dpdz(var, h)
    return nd


# ------------------------------------------------------------------
# Vorticity
# ------------------------------------------------------------------
def vorticity(u, v, h, ndim, w=None):
    """
    Compute vorticity.
    2-D: scalar  omega = dv/dx - du/dy
    3-D: magnitude ``|ω|`` = sqrt(ωx² + ωy² + ωz²)
    """
    inv_h = _recip(h)
    if ndim == 2 or w is None:
        dvdx = torch.zeros_like(u)
        dudy = torch.zeros_like(u)
        dvdx[1:-1, 1:-1] = (v[1:-1, 1:-1] - v[:-2, 1:-1]) * inv_h
        dudy[1:-1, 1:-1] = (u[1:-1, 1:-1] - u[1:-1, :-2]) * inv_h
        return dvdx - dudy
    else:
        # 3-D vorticity magnitude.
        # Start at index 2 so backward differences never reach into
        # ghost cells (index 0), which can have BC-inconsistent values
        # and produce spurious boundary vorticity.
        ox = torch.zeros_like(u)
        ox[2:-2, 2:-2, 2:-2] = (
            (w[2:-2, 2:-2, 2:-2] - w[2:-2, 1:-3, 2:-2]) * inv_h -
            (v[2:-2, 2:-2, 2:-2] - v[2:-2, 2:-2, 1:-3]) * inv_h
        )
        oy = torch.zeros_like(u)
        oy[2:-2, 2:-2, 2:-2] = (
            (u[2:-2, 2:-2, 2:-2] - u[2:-2, 2:-2, 1:-3]) * inv_h -
            (w[2:-2, 2:-2, 2:-2] - w[1:-3, 2:-2, 2:-2]) * inv_h
        )
        oz = torch.zeros_like(u)
        oz[2:-2, 2:-2, 2:-2] = (
            (v[2:-2, 2:-2, 2:-2] - v[1:-3, 2:-2, 2:-2]) * inv_h -
            (u[2:-2, 2:-2, 2:-2] - u[2:-2, 1:-3, 2:-2]) * inv_h
        )
        return torch.sqrt(ox**2 + oy**2 + oz**2)


def vorticity_components(u, v, w, h):
    """
    Return the three signed vorticity components (omega_x, omega_y, omega_z)
    as a dict, plus the magnitude.  Only meaningful in 3-D.
    """
    inv_h = _recip(h)
    ox = torch.zeros_like(u)
    ox[2:-2, 2:-2, 2:-2] = (
        (w[2:-2, 2:-2, 2:-2] - w[2:-2, 1:-3, 2:-2]) * inv_h -
        (v[2:-2, 2:-2, 2:-2] - v[2:-2, 2:-2, 1:-3]) * inv_h
    )
    oy = torch.zeros_like(u)
    oy[2:-2, 2:-2, 2:-2] = (
        (u[2:-2, 2:-2, 2:-2] - u[2:-2, 2:-2, 1:-3]) * inv_h -
        (w[2:-2, 2:-2, 2:-2] - w[1:-3, 2:-2, 2:-2]) * inv_h
    )
    oz = torch.zeros_like(u)
    oz[2:-2, 2:-2, 2:-2] = (
        (v[2:-2, 2:-2, 2:-2] - v[1:-3, 2:-2, 2:-2]) * inv_h -
        (u[2:-2, 2:-2, 2:-2] - u[2:-2, 1:-3, 2:-2]) * inv_h
    )
    return {"omega_x": ox, "omega_y": oy, "omega_z": oz,
            "omega_mag": torch.sqrt(ox**2 + oy**2 + oz**2)}


# ------------------------------------------------------------------
# Cross products
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Strain-rate magnitude and Smagorinsky eddy viscosity
# ------------------------------------------------------------------

def strain_rate_magnitude(vel, h, ndim):
    """Compute ``|S̄|`` = sqrt(2 * S_ij * S_ij) — pure-Warp implementation.

    Replaces the legacy ``torch.gradient(edge_order=2) + _stag_to_cc`` path
    with a single Warp kernel that is graph-capturable inside a
    ``wp.ScopedCapture``.

    Parameters
    ----------
    vel  : tuple of tensors (u, v) in 2-D or (u, v, w) in 3-D.
    h    : float — uniform grid spacing.
    ndim : int — 2 or 3.

    Returns
    -------
    ``|S̄|`` tensor with same shape as vel[0], backed by a persistent
    buffer (no per-step allocation).
    """
    if ndim == 2:
        u, v = vel
        return _strain_rate_magnitude_warp_eager(u, v, None, h)
    else:
        u, v, w = vel
        return _strain_rate_magnitude_warp_eager(u, v, w, h)


def smagorinsky_viscosity(vel, h, ndim, cs=0.1):
    """Compute the Smagorinsky eddy viscosity  ν_t = (Cs·Δ)² ``|S̄|``.

    Parameters
    ----------
    vel  : tuple of tensors — velocity components.
    h    : float — uniform grid spacing (used as filter width Δ).
    ndim : int — 2 or 3.
    cs   : float — Smagorinsky constant (typically 0.1–0.2).

    Returns
    -------
    ν_t tensor with same shape as vel[0].
    """
    S_mag = strain_rate_magnitude(vel, h, ndim)
    return (cs * h) ** 2 * S_mag


def carreau_viscosity(vel, h, ndim, nu_0, nu_inf, lam, n,
                     tau_y=0.0, rho=1000.0, nu_max=None):
    """Compute Carreau (or Herschel-Bulkley–Carreau) viscosity field.

    Without yield stress (tau_y = 0):
        ν(γ̇) = ν_∞ + (ν_0 − ν_∞) · [1 + (λ·γ̇)²]^((n−1)/2)

    With yield stress (tau_y > 0):
        ν(γ̇) = τ_y / (ρ · max(γ̇, γ̇_min)) + ν_∞ + (ν_0 − ν_∞) · [1 + (λ·γ̇)²]^((n−1)/2)

    The total viscosity is clamped to ``nu_max`` to guarantee diffusion CFL
    stability.  When ``nu_max`` is None no clamping is applied (pure Carreau
    without yield stress is bounded by ν_0 anyway).

    Parameters
    ----------
    vel    : tuple of tensors — velocity components (u, v) or (u, v, w).
    h      : float — uniform grid spacing.
    ndim   : int — 2 or 3.
    nu_0   : float — zero-shear-rate kinematic viscosity [m²/s].
    nu_inf : float — infinite-shear-rate kinematic viscosity [m²/s].
    lam    : float — relaxation time [s].
    n      : float — power-law index (< 1 for shear-thinning).
    tau_y  : float — yield stress [Pa]. Default 0 (pure Carreau).
    rho    : float — fluid density [kg/m³]. Only used when tau_y > 0.
    nu_max : float or None — hard upper bound on ν for CFL stability.

    Returns
    -------
    ν tensor (spatially varying kinematic viscosity) with same shape as vel[0].
    """
    S_mag = strain_rate_magnitude(vel, h, ndim)
    nu = nu_inf + (nu_0 - nu_inf) * (1.0 + (lam * S_mag) ** 2) ** ((n - 1.0) / 2.0)
    if tau_y > 0.0:
        gamma_dot_reg = torch.clamp(S_mag, min=1e-6)
        nu = nu + tau_y / (rho * gamma_dot_reg)
    if nu_max is not None:
        nu = torch.clamp(nu, max=nu_max)
    return nu


def cross_product_2d(ax, ay, bx, by):
    """Element-wise 2-D cross product (scalar result)."""
    return ax * by - ay * bx


def cross_product_3d(ax, ay, az, bx, by, bz):
    """Element-wise 3-D cross product  a × b.

    Returns (cx, cy, cz) tensors.
    """
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return cx, cy, cz
