"""Warp single-source Lagrangian surface-force kernels (AP7 scatter, part 2).

Port of the native ``lagrangian_forces_3d`` / ``lagrangian_forces_2d`` CUDA/CPP
kernel pair (``src/kernels/csrc/cuda/lagrangian_forces*.cu``,
``..._cpu*.cpp``): fused per-body surface integration of the hydrodynamic
traction ``t = nu*rho*(eps . n) - p n`` over a precomputed surface
(triangulation in 3-D, closed contour in 2-D), scattered into a per-body
12-channel (3-D) / 6-channel (2-D) force+torque row via ``atomicAdd``.

This is the *other* irregular-scatter (Warp-class) kernel besides Kernel A.
Unlike Kernel A it has no argmin — just a gather (trilinear/triquadratic field
sample a fixed offset along the outward normal) followed by an ``atomic_add``
accumulation, so it is the simplest scatter demonstrator.

Design notes / faithful-port details (read the .cu before changing anything):
  * Native parallelises one thread per surface element with a 2-D grid
    ``(blocksPerBody, B)``; here we launch a flat ``dim=N_total`` and recover the
    owning body via a precomputed ``elem_body`` map (host-built from offsets).
    Numerically identical; the atomic accumulation order differs from the
    reference either way, giving ~1e-12 (float64) reduction noise — same as the
    native self-test's tolerance.
  * **Precision (single source, both dtypes).**  Mirrors the native
    ``AT_DISPATCH_FLOATING_TYPES`` exactly: every per-element quantity (sampling,
    strain·normal, pressure, trapezoidal/area weight, moment arm) is computed in
    the *field* element type ``scalar_t`` (``Any`` Warp generic — float32 *or*
    float64), and only the final products are cast to ``double`` for the
    ``atomicAdd`` into the always-float64 ``out`` accumulator (native writes
    ``(double)(tvx*wq)`` into a ``double* out``).  ``wp.overload`` registers the
    f32 + f64 specialisations (Warp 1.14 needs the dtypes pre-registered).  f64
    is bit-identical to the original concrete kernel; f32 matches native f32 to
    single precision.  ``out`` / ``elem_body`` / ``offs`` stay concrete.
  * Field sampling mirrors ``lf_trilinear_3d_d`` / ``lf_triquadratic_3d_d``
    exactly: clamp ``t`` to ``[0, M-1]``, ``ix=(int)t`` then clamp to ``M-2``,
    flat row-major base index ``ix*My*Mz + iy*Mz + iz``, quadratic falls back to
    linear when ``ix<1`` (or ``M<3``).
  * Flat 1-D ``wp.array`` addressing (HANDOFF lesson 2): fields are passed
    flattened; base index + constant stride offsets, mirroring the native CUDA.
"""
from __future__ import annotations

from typing import Any

import warp as wp
import torch

wp.init()


# ─────────────────────────────────────────────────────────────────────────────
#  3-D field sampling (flat row-major F, strides s1=My*Mz, s2=Mz)
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def lf_trilinear_3d(
    F: wp.array(dtype=Any),
    Mx: wp.int32, My: wp.int32, Mz: wp.int32,
    bx0: Any, by0: Any, bz0: Any,
    inv_dx: Any, inv_dy: Any, inv_dz: Any,
    xq: Any, yq: Any, zq: Any,
):
    zero = type(xq)(0.0)
    one = type(xq)(1.0)
    tx = wp.clamp((xq - bx0) * inv_dx, zero, type(xq)(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, zero, type(xq)(My - 1))
    tz = wp.clamp((zq - bz0) * inv_dz, zero, type(xq)(Mz - 1))

    ix = wp.int32(tx)
    if ix > Mx - 2:
        ix = Mx - 2
    iy = wp.int32(ty)
    if iy > My - 2:
        iy = My - 2
    iz = wp.int32(tz)
    if iz > Mz - 2:
        iz = Mz - 2

    fx = tx - type(xq)(ix)
    fy = ty - type(xq)(iy)
    fz = tz - type(xq)(iz)
    wx0 = one - fx
    wy0 = one - fy
    wz0 = one - fz

    s2 = Mz
    s1 = My * Mz
    base = ix * s1 + iy * s2 + iz

    return (
        wx0 * (
            wy0 * (wz0 * F[base] + fz * F[base + 1])
            + fy * (wz0 * F[base + s2] + fz * F[base + s2 + 1])
        )
        + fx * (
            wy0 * (wz0 * F[base + s1] + fz * F[base + s1 + 1])
            + fy * (wz0 * F[base + s1 + s2] + fz * F[base + s1 + s2 + 1])
        )
    )


@wp.func
def lf_triquadratic_3d(
    F: wp.array(dtype=Any),
    Mx: wp.int32, My: wp.int32, Mz: wp.int32,
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

    ix = wp.int32(tx)
    if ix > Mx - 2:
        ix = Mx - 2
    iy = wp.int32(ty)
    if iy > My - 2:
        iy = My - 2
    iz = wp.int32(tz)
    if iz > Mz - 2:
        iz = Mz - 2

    res = zero
    if ix < 1 or iy < 1 or iz < 1 or Mx < 3 or My < 3 or Mz < 3:
        res = lf_trilinear_3d(
            F, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, xq, yq, zq)
    else:
        fx = tx - type(xq)(ix)
        fy = ty - type(xq)(iy)
        fz = tz - type(xq)(iz)

        wxm = half * fx * (fx - one)
        wx0 = one - fx * fx
        wxp = half * fx * (fx + one)
        wym = half * fy * (fy - one)
        wy0 = one - fy * fy
        wyp = half * fy * (fy + one)
        wzm = half * fz * (fz - one)
        wz0 = one - fz * fz
        wzp = half * fz * (fz + one)

        s2 = Mz
        s1 = My * Mz
        base = (ix - 1) * s1 + (iy - 1) * s2 + (iz - 1)

        out = zero
        for dx in range(3):
            wx = wxm
            if dx == 1:
                wx = wx0
            if dx == 2:
                wx = wxp
            b0 = base + dx * s1
            plane = zero
            for dy in range(3):
                wy = wym
                if dy == 1:
                    wy = wy0
                if dy == 2:
                    wy = wyp
                b1 = b0 + dy * s2
                row = wzm * F[b1] + wz0 * F[b1 + 1] + wzp * F[b1 + 2]
                plane += wy * row
            out += wx * plane
        res = out
    return res


@wp.func
def lf_sample_3d(
    method: wp.int32,
    F: wp.array(dtype=Any),
    Mx: wp.int32, My: wp.int32, Mz: wp.int32,
    bx0: Any, by0: Any, bz0: Any,
    inv_dx: Any, inv_dy: Any, inv_dz: Any,
    xq: Any, yq: Any, zq: Any,
):
    res = type(xq)(0.0)
    if method == 1:
        res = lf_triquadratic_3d(
            F, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, xq, yq, zq)
    else:
        res = lf_trilinear_3d(
            F, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, xq, yq, zq)
    return res


@wp.kernel
def lagrangian_forces_3d_kernel(
    exx: wp.array(dtype=Any), eyy: wp.array(dtype=Any),
    ezz: wp.array(dtype=Any), exy: wp.array(dtype=Any),
    exz: wp.array(dtype=Any), eyz: wp.array(dtype=Any),
    pp: wp.array(dtype=Any), nrho: wp.array(dtype=Any),
    nrho_scalar: wp.int32,
    cx: wp.array(dtype=Any), cy: wp.array(dtype=Any),
    cz: wp.array(dtype=Any),
    nxp: wp.array(dtype=Any), nyp: wp.array(dtype=Any),
    nzp: wp.array(dtype=Any),
    area: wp.array(dtype=Any),
    elem_body: wp.array(dtype=wp.int32),
    com: wp.array(dtype=Any),
    Mx: wp.int32, My: wp.int32, Mz: wp.int32,
    bx0: Any, by0: Any, bz0: Any,
    inv_dx: Any, inv_dy: Any, inv_dz: Any,
    interp_method: wp.int32,
    sample_offset: Any,
    out: wp.array(dtype=wp.float64),   # (B*12,)
):
    t = wp.tid()
    b = elem_body[t]

    qx = cx[t]
    qy = cy[t]
    qz = cz[t]
    nxv = nxp[t]
    nyv = nyp[t]
    nzv = nzp[t]
    A = area[t]

    # Sample fields a distance sample_offset OUTSIDE the body along n.
    qxs = qx + sample_offset * nxv
    qys = qy + sample_offset * nyv
    qzs = qz + sample_offset * nzv

    e_xx = lf_sample_3d(interp_method, exx, Mx, My, Mz, bx0, by0, bz0,
                        inv_dx, inv_dy, inv_dz, qxs, qys, qzs)
    e_yy = lf_sample_3d(interp_method, eyy, Mx, My, Mz, bx0, by0, bz0,
                        inv_dx, inv_dy, inv_dz, qxs, qys, qzs)
    e_zz = lf_sample_3d(interp_method, ezz, Mx, My, Mz, bx0, by0, bz0,
                        inv_dx, inv_dy, inv_dz, qxs, qys, qzs)
    e_xy = lf_sample_3d(interp_method, exy, Mx, My, Mz, bx0, by0, bz0,
                        inv_dx, inv_dy, inv_dz, qxs, qys, qzs)
    e_xz = lf_sample_3d(interp_method, exz, Mx, My, Mz, bx0, by0, bz0,
                        inv_dx, inv_dy, inv_dz, qxs, qys, qzs)
    e_yz = lf_sample_3d(interp_method, eyz, Mx, My, Mz, bx0, by0, bz0,
                        inv_dx, inv_dy, inv_dz, qxs, qys, qzs)

    nu_rho_m = nrho[0]
    if nrho_scalar == 0:
        nu_rho_m = lf_sample_3d(interp_method, nrho, Mx, My, Mz, bx0, by0, bz0,
                                inv_dx, inv_dy, inv_dz, qxs, qys, qzs)

    tvx = nu_rho_m * (e_xx * nxv + e_xy * nyv + e_xz * nzv)
    tvy = nu_rho_m * (e_xy * nxv + e_yy * nyv + e_yz * nzv)
    tvz = nu_rho_m * (e_xz * nxv + e_yz * nyv + e_zz * nzv)

    p_m = lf_sample_3d(interp_method, pp, Mx, My, Mz, bx0, by0, bz0,
                       inv_dx, inv_dy, inv_dz, qxs, qys, qzs)
    tpx = -p_m * nxv
    tpy = -p_m * nyv
    tpz = -p_m * nzv

    rx = qx - com[b * 3 + 0]
    ry = qy - com[b * 3 + 1]
    rz = qz - com[b * 3 + 2]

    o = b * 12
    wp.atomic_add(out, o + 0, wp.float64(tvx * A))
    wp.atomic_add(out, o + 1, wp.float64(tvy * A))
    wp.atomic_add(out, o + 2, wp.float64(tvz * A))
    wp.atomic_add(out, o + 3, wp.float64((ry * tvz - rz * tvy) * A))
    wp.atomic_add(out, o + 4, wp.float64((rz * tvx - rx * tvz) * A))
    wp.atomic_add(out, o + 5, wp.float64((rx * tvy - ry * tvx) * A))
    wp.atomic_add(out, o + 6, wp.float64(tpx * A))
    wp.atomic_add(out, o + 7, wp.float64(tpy * A))
    wp.atomic_add(out, o + 8, wp.float64(tpz * A))
    wp.atomic_add(out, o + 9, wp.float64((ry * tpz - rz * tpy) * A))
    wp.atomic_add(out, o + 10, wp.float64((rz * tpx - rx * tpz) * A))
    wp.atomic_add(out, o + 11, wp.float64((rx * tpy - ry * tpx) * A))


# ─────────────────────────────────────────────────────────────────────────────
#  2-D field sampling (flat row-major F, stride s1=My)
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def lf_bilinear_2d(
    F: wp.array(dtype=Any),
    Mx: wp.int32, My: wp.int32,
    bx0: Any, by0: Any,
    inv_dx: Any, inv_dy: Any,
    xq: Any, yq: Any,
):
    zero = type(xq)(0.0)
    one = type(xq)(1.0)
    tx = wp.clamp((xq - bx0) * inv_dx, zero, type(xq)(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, zero, type(xq)(My - 1))

    ix = wp.int32(tx)
    if ix > Mx - 2:
        ix = Mx - 2
    iy = wp.int32(ty)
    if iy > My - 2:
        iy = My - 2

    fx = tx - type(xq)(ix)
    fy = ty - type(xq)(iy)
    wx0 = one - fx
    wy0 = one - fy

    s1 = My
    base = ix * s1 + iy
    return (
        wx0 * (wy0 * F[base] + fy * F[base + 1])
        + fx * (wy0 * F[base + s1] + fy * F[base + s1 + 1])
    )


@wp.func
def lf_biquadratic_2d(
    F: wp.array(dtype=Any),
    Mx: wp.int32, My: wp.int32,
    bx0: Any, by0: Any,
    inv_dx: Any, inv_dy: Any,
    xq: Any, yq: Any,
):
    zero = type(xq)(0.0)
    one = type(xq)(1.0)
    half = type(xq)(0.5)
    tx = wp.clamp((xq - bx0) * inv_dx, zero, type(xq)(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, zero, type(xq)(My - 1))

    ix = wp.int32(tx)
    if ix > Mx - 2:
        ix = Mx - 2
    iy = wp.int32(ty)
    if iy > My - 2:
        iy = My - 2

    res = zero
    if ix < 1 or iy < 1 or Mx < 3 or My < 3:
        res = lf_bilinear_2d(F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq)
    else:
        fx = tx - type(xq)(ix)
        fy = ty - type(xq)(iy)

        wxm = half * fx * (fx - one)
        wx0 = one - fx * fx
        wxp = half * fx * (fx + one)
        wym = half * fy * (fy - one)
        wy0 = one - fy * fy
        wyp = half * fy * (fy + one)

        s1 = My
        base = (ix - 1) * s1 + (iy - 1)
        out = zero
        for dx in range(3):
            wx = wxm
            if dx == 1:
                wx = wx0
            if dx == 2:
                wx = wxp
            b0 = base + dx * s1
            row = wym * F[b0] + wy0 * F[b0 + 1] + wyp * F[b0 + 2]
            out += wx * row
        res = out
    return res


@wp.func
def lf_sample_2d(
    method: wp.int32,
    F: wp.array(dtype=Any),
    Mx: wp.int32, My: wp.int32,
    bx0: Any, by0: Any,
    inv_dx: Any, inv_dy: Any,
    xq: Any, yq: Any,
):
    res = type(xq)(0.0)
    if method == 1:
        res = lf_biquadratic_2d(F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq)
    else:
        res = lf_bilinear_2d(F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq)
    return res


@wp.kernel
def lagrangian_forces_2d_kernel(
    exx: wp.array(dtype=Any), exy: wp.array(dtype=Any),
    eyy: wp.array(dtype=Any),
    pp: wp.array(dtype=Any), nrho: wp.array(dtype=Any),
    nrho_scalar: wp.int32,
    cnt_x: wp.array(dtype=Any), cnt_y: wp.array(dtype=Any),
    offs: wp.array(dtype=wp.int32),
    elem_body: wp.array(dtype=wp.int32),
    com: wp.array(dtype=Any),
    Mx: wp.int32, My: wp.int32,
    bx0: Any, by0: Any,
    inv_dx: Any, inv_dy: Any,
    interp_method: wp.int32,
    sample_offset: Any,
    out: wp.array(dtype=wp.float64),   # (B*6,)
):
    g = wp.tid()
    b = elem_body[g]
    i0 = offs[b]
    i1 = offs[b + 1]
    M = i1 - i0
    if M <= 1:
        return

    k = g - i0
    km = k - 1
    if k == 0:
        km = M - 1
    kp = k + 1
    if k == M - 1:
        kp = 0
    gm = i0 + km
    gp = i0 + kp

    qx = cnt_x[g]
    qy = cnt_y[g]
    half = type(qx)(0.5)

    # Tangent via central diff on the closed contour → outward normal.
    tx = (cnt_x[gp] - cnt_x[gm]) * half
    ty = (cnt_y[gp] - cnt_y[gm]) * half
    L = wp.sqrt(tx * tx + ty * ty)
    if L < type(qx)(1e-30):
        L = type(qx)(1e-30)
    tx = tx / L
    ty = ty / L
    nx = ty
    ny = -tx

    qxs = qx + sample_offset * nx
    qys = qy + sample_offset * ny

    e_xx = lf_sample_2d(interp_method, exx, Mx, My, bx0, by0,
                        inv_dx, inv_dy, qxs, qys)
    e_xy = lf_sample_2d(interp_method, exy, Mx, My, bx0, by0,
                        inv_dx, inv_dy, qxs, qys)
    e_yy = lf_sample_2d(interp_method, eyy, Mx, My, bx0, by0,
                        inv_dx, inv_dy, qxs, qys)

    nu_rho_m = nrho[0]
    if nrho_scalar == 0:
        nu_rho_m = lf_sample_2d(interp_method, nrho, Mx, My, bx0, by0,
                                inv_dx, inv_dy, qxs, qys)

    tvx = nu_rho_m * (e_xx * nx + e_xy * ny)
    tvy = nu_rho_m * (e_xy * nx + e_yy * ny)

    p_m = lf_sample_2d(interp_method, pp, Mx, My, bx0, by0,
                       inv_dx, inv_dy, qxs, qys)
    tpx = -p_m * nx
    tpy = -p_m * ny

    # Lumped trapezoidal weight on closed contour: w_k = 0.5*(ds_{k-1}+ds_k)
    dxm = qx - cnt_x[gm]
    dym = qy - cnt_y[gm]
    dxp = cnt_x[gp] - qx
    dyp = cnt_y[gp] - qy
    dsm = wp.sqrt(dxm * dxm + dym * dym)
    dsp = wp.sqrt(dxp * dxp + dyp * dyp)
    wq = half * (dsm + dsp)

    rx = qx - com[b * 2 + 0]
    ry = qy - com[b * 2 + 1]

    o = b * 6
    wp.atomic_add(out, o + 0, wp.float64(tvx * wq))
    wp.atomic_add(out, o + 1, wp.float64(tvy * wq))
    wp.atomic_add(out, o + 2, wp.float64((rx * tvy - ry * tvx) * wq))
    wp.atomic_add(out, o + 3, wp.float64(tpx * wq))
    wp.atomic_add(out, o + 4, wp.float64(tpy * wq))
    wp.atomic_add(out, o + 5, wp.float64((rx * tpy - ry * tpx) * wq))


# Register float32 + float64 specialisations (generic args only; out/elem_body/
# offs stay concrete — out is always the float64 accumulator, native style).
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(lagrangian_forces_3d_kernel, {
        "exx": _A, "eyy": _A, "ezz": _A, "exy": _A, "exz": _A, "eyz": _A,
        "pp": _A, "nrho": _A, "cx": _A, "cy": _A, "cz": _A,
        "nxp": _A, "nyp": _A, "nzp": _A, "area": _A, "com": _A,
        "bx0": _dt, "by0": _dt, "bz0": _dt,
        "inv_dx": _dt, "inv_dy": _dt, "inv_dz": _dt,
        "sample_offset": _dt,
    })
    wp.overload(lagrangian_forces_2d_kernel, {
        "exx": _A, "exy": _A, "eyy": _A, "pp": _A, "nrho": _A,
        "cnt_x": _A, "cnt_y": _A, "com": _A,
        "bx0": _dt, "by0": _dt, "inv_dx": _dt, "inv_dy": _dt,
        "sample_offset": _dt,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Python launch wrappers (mirror the native ops.py signatures)
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_MAP = {"linear": 0, "quadratic": 1}


def _wp_dtype(t: torch.Tensor):
    return wp.float64 if t.dtype == torch.float64 else wp.float32


def _torch_dtype(t: torch.Tensor):
    return torch.float64 if t.dtype == torch.float64 else torch.float32


def _fast_flat(x: torch.Tensor, wpf, tdt, wdev):
    """Low-overhead zero-copy flat Warp view.  When ``x`` is already contiguous
    and the working dtype, build the ``wp.array`` directly from ``data_ptr``
    (~2× cheaper than ``wp.from_torch`` — it skips the no-op
    ``.contiguous()/.to()`` and the torch→warp inference); otherwise fall back to
    the safe ``wp.from_torch`` path.  Cuts the per-call wrapping floor of the
    force readouts (~17 wraps) which graph capture cannot remove here (the
    strain/triangle buffers are freshly allocated with body-following shapes each
    step)."""
    if x.dtype == tdt and x.is_contiguous():
        return wp.array(ptr=x.data_ptr(), dtype=wpf, shape=(x.numel(),),
                        device=wdev)
    return wp.from_torch(x.reshape(-1).contiguous().to(tdt))


_ELEM_BODY_CACHE = {}


def _elem_body_from_offsets(offsets: torch.Tensor, n_total: int,
                            device) -> torch.Tensor:
    """Map each surface element index → owning body, from a (B+1,) prefix sum.

    Cached on the offsets buffer identity: the per-body element counts are a
    static topology (they do not change as the body moves), and the
    ``repeat_interleave`` below otherwise forces a per-call D2H sync to size its
    output — a latency floor on every Lagrangian force readout."""
    key = (offsets.data_ptr(), int(n_total), str(device), int(offsets.numel()))
    eb = _ELEM_BODY_CACHE.get(key)
    if eb is not None:
        return eb
    counts = (offsets[1:] - offsets[:-1]).to(torch.long)
    B = counts.numel()
    eb = torch.repeat_interleave(torch.arange(B, device=device), counts.to(device))
    assert eb.numel() == n_total, (eb.numel(), n_total)
    eb = eb.to(torch.int32).contiguous()
    _ELEM_BODY_CACHE[key] = eb
    return eb


def lagrangian_forces_3d_warp(
        eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz,
        p, nu_rho_field,
        tri_centroid, tri_normal, tri_area,
        tri_offsets, com_pos,
        bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
        Mx, My, Mz, method="linear", sample_offset=0.0, out=None):
    """Warp port of ``lagrangian_forces_3d``; returns (B, 12) float64 torch.

    Dtype-generic: per-element math runs in ``p.dtype`` (f32/f64), the ``out``
    accumulator is always float64 (native ``double* out``).  ``out`` may be a
    pre-allocated ``(B, 12)`` float64 buffer (zeroed in place, mirroring the
    native ``out=`` convention)."""
    interp_method = _METHOD_MAP[method]
    dev = p.device
    wdev = "cuda:0" if dev.type == "cuda" else "cpu"
    wpf = _wp_dtype(p)
    tdt = _torch_dtype(p)
    B = int(com_pos.shape[0])
    T_total = int(tri_centroid.shape[1])

    def f(x):
        return _fast_flat(x, wpf, tdt, wdev)

    nrho_scalar = 1 if nu_rho_field.numel() == 1 else 0
    elem_body = _elem_body_from_offsets(tri_offsets, T_total, dev)

    if out is None:
        out = torch.zeros((B, 12), dtype=torch.float64, device=dev)
    else:
        out.zero_()
    out_w = wp.from_torch(out.reshape(-1))

    wp.launch(
        lagrangian_forces_3d_kernel, dim=T_total,
        inputs=[
            f(eps_xx), f(eps_yy), f(eps_zz), f(eps_xy), f(eps_xz), f(eps_yz),
            f(p), f(nu_rho_field), nrho_scalar,
            f(tri_centroid[0]), f(tri_centroid[1]), f(tri_centroid[2]),
            f(tri_normal[0]), f(tri_normal[1]), f(tri_normal[2]),
            f(tri_area),
            wp.from_torch(elem_body.contiguous()),
            f(com_pos),
            int(Mx), int(My), int(Mz),
            wpf(bx0), wpf(by0), wpf(bz0),
            wpf(inv_dx), wpf(inv_dy), wpf(inv_dz),
            int(interp_method), wpf(sample_offset),
            out_w,
        ],
        device=wdev)
    # No wp.synchronize(): the caller reads ``out`` through torch (null-stream
    # ordering after the Warp launch); the full-device sync was a latency floor.
    return out.reshape(B, 12)


def lagrangian_forces_2d_warp(
        eps_xx, eps_xy, eps_yy,
        p, nu_rho_field,
        cnt_flat, cnt_offsets, com_pos,
        bx0, by0, inv_dx, inv_dy,
        Mx, My, method="linear", sample_offset=0.0, out=None):
    """Warp port of ``lagrangian_forces_2d``; returns (B, 6) float64 torch.

    Dtype-generic (see :func:`lagrangian_forces_3d_warp`)."""
    interp_method = _METHOD_MAP[method]
    dev = p.device
    wdev = "cuda:0" if dev.type == "cuda" else "cpu"
    wpf = _wp_dtype(p)
    tdt = _torch_dtype(p)
    B = int(com_pos.shape[0])
    M_total = int(cnt_flat.shape[1])

    def f(x):
        return _fast_flat(x, wpf, tdt, wdev)

    nrho_scalar = 1 if nu_rho_field.numel() == 1 else 0
    elem_body = _elem_body_from_offsets(cnt_offsets, M_total, dev)
    offs_i32 = wp.from_torch(cnt_offsets.to(torch.int32).contiguous())

    if out is None:
        out = torch.zeros((B, 6), dtype=torch.float64, device=dev)
    else:
        out.zero_()
    out_w = wp.from_torch(out.reshape(-1))

    wp.launch(
        lagrangian_forces_2d_kernel, dim=M_total,
        inputs=[
            f(eps_xx), f(eps_xy), f(eps_yy),
            f(p), f(nu_rho_field), nrho_scalar,
            f(cnt_flat[0]), f(cnt_flat[1]),
            offs_i32,
            wp.from_torch(elem_body.contiguous()),
            f(com_pos),
            int(Mx), int(My),
            wpf(bx0), wpf(by0), wpf(inv_dx), wpf(inv_dy),
            int(interp_method), wpf(sample_offset),
            out_w,
        ],
        device=wdev)
    # No wp.synchronize(): the caller reads ``out`` through torch (null-stream
    # ordering after the Warp launch); the full-device sync was a latency floor.
    return out.reshape(B, 6)
