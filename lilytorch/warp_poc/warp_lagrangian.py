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
  * Everything runs in **float64** to match the native self-test, which feeds
    float64 fields (``AT_DISPATCH`` → ``scalar_t = double``) and accumulates in
    ``double`` via ``atomicAdd``.  ``wp.atomic_add`` on a ``wp.float64`` array
    works on CPU and CUDA → single source.
  * Field sampling mirrors ``lf_trilinear_3d_d`` / ``lf_triquadratic_3d_d``
    exactly: clamp ``t`` to ``[0, M-1]``, ``ix=(int)t`` then clamp to ``M-2``,
    flat row-major base index ``ix*My*Mz + iy*Mz + iz``, quadratic falls back to
    linear when ``ix<1`` (or ``M<3``).
  * Flat 1-D ``wp.array`` addressing (HANDOFF lesson 2): fields are passed
    flattened; base index + constant stride offsets, mirroring the native CUDA.
"""
from __future__ import annotations

import warp as wp
import torch

wp.init()

_ZERO = wp.constant(wp.float64(0.0))
_ONE = wp.constant(wp.float64(1.0))
_HALF = wp.constant(wp.float64(0.5))


# ─────────────────────────────────────────────────────────────────────────────
#  3-D field sampling (flat row-major F, strides s1=My*Mz, s2=Mz)
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def lf_trilinear_3d(
    F: wp.array(dtype=wp.float64),
    Mx: wp.int32, My: wp.int32, Mz: wp.int32,
    bx0: wp.float64, by0: wp.float64, bz0: wp.float64,
    inv_dx: wp.float64, inv_dy: wp.float64, inv_dz: wp.float64,
    xq: wp.float64, yq: wp.float64, zq: wp.float64,
) -> wp.float64:
    tx = wp.clamp((xq - bx0) * inv_dx, _ZERO, wp.float64(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, _ZERO, wp.float64(My - 1))
    tz = wp.clamp((zq - bz0) * inv_dz, _ZERO, wp.float64(Mz - 1))

    ix = wp.int32(tx)
    if ix > Mx - 2:
        ix = Mx - 2
    iy = wp.int32(ty)
    if iy > My - 2:
        iy = My - 2
    iz = wp.int32(tz)
    if iz > Mz - 2:
        iz = Mz - 2

    fx = tx - wp.float64(ix)
    fy = ty - wp.float64(iy)
    fz = tz - wp.float64(iz)
    wx0 = _ONE - fx
    wy0 = _ONE - fy
    wz0 = _ONE - fz

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
    F: wp.array(dtype=wp.float64),
    Mx: wp.int32, My: wp.int32, Mz: wp.int32,
    bx0: wp.float64, by0: wp.float64, bz0: wp.float64,
    inv_dx: wp.float64, inv_dy: wp.float64, inv_dz: wp.float64,
    xq: wp.float64, yq: wp.float64, zq: wp.float64,
) -> wp.float64:
    tx = wp.clamp((xq - bx0) * inv_dx, _ZERO, wp.float64(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, _ZERO, wp.float64(My - 1))
    tz = wp.clamp((zq - bz0) * inv_dz, _ZERO, wp.float64(Mz - 1))

    ix = wp.int32(tx)
    if ix > Mx - 2:
        ix = Mx - 2
    iy = wp.int32(ty)
    if iy > My - 2:
        iy = My - 2
    iz = wp.int32(tz)
    if iz > Mz - 2:
        iz = Mz - 2

    res = _ZERO
    if ix < 1 or iy < 1 or iz < 1 or Mx < 3 or My < 3 or Mz < 3:
        res = lf_trilinear_3d(
            F, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, xq, yq, zq)
    else:
        fx = tx - wp.float64(ix)
        fy = ty - wp.float64(iy)
        fz = tz - wp.float64(iz)

        wxm = _HALF * fx * (fx - _ONE)
        wx0 = _ONE - fx * fx
        wxp = _HALF * fx * (fx + _ONE)
        wym = _HALF * fy * (fy - _ONE)
        wy0 = _ONE - fy * fy
        wyp = _HALF * fy * (fy + _ONE)
        wzm = _HALF * fz * (fz - _ONE)
        wz0 = _ONE - fz * fz
        wzp = _HALF * fz * (fz + _ONE)

        s2 = Mz
        s1 = My * Mz
        base = (ix - 1) * s1 + (iy - 1) * s2 + (iz - 1)

        out = _ZERO
        for dx in range(3):
            wx = wxm
            if dx == 1:
                wx = wx0
            if dx == 2:
                wx = wxp
            b0 = base + dx * s1
            plane = _ZERO
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
    F: wp.array(dtype=wp.float64),
    Mx: wp.int32, My: wp.int32, Mz: wp.int32,
    bx0: wp.float64, by0: wp.float64, bz0: wp.float64,
    inv_dx: wp.float64, inv_dy: wp.float64, inv_dz: wp.float64,
    xq: wp.float64, yq: wp.float64, zq: wp.float64,
) -> wp.float64:
    res = _ZERO
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
    exx: wp.array(dtype=wp.float64), eyy: wp.array(dtype=wp.float64),
    ezz: wp.array(dtype=wp.float64), exy: wp.array(dtype=wp.float64),
    exz: wp.array(dtype=wp.float64), eyz: wp.array(dtype=wp.float64),
    pp: wp.array(dtype=wp.float64), nrho: wp.array(dtype=wp.float64),
    nrho_scalar: wp.int32,
    cx: wp.array(dtype=wp.float64), cy: wp.array(dtype=wp.float64),
    cz: wp.array(dtype=wp.float64),
    nxp: wp.array(dtype=wp.float64), nyp: wp.array(dtype=wp.float64),
    nzp: wp.array(dtype=wp.float64),
    area: wp.array(dtype=wp.float64),
    elem_body: wp.array(dtype=wp.int32),
    com: wp.array(dtype=wp.float64),
    Mx: wp.int32, My: wp.int32, Mz: wp.int32,
    bx0: wp.float64, by0: wp.float64, bz0: wp.float64,
    inv_dx: wp.float64, inv_dy: wp.float64, inv_dz: wp.float64,
    interp_method: wp.int32,
    sample_offset: wp.float64,
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
    wp.atomic_add(out, o + 0, tvx * A)
    wp.atomic_add(out, o + 1, tvy * A)
    wp.atomic_add(out, o + 2, tvz * A)
    wp.atomic_add(out, o + 3, (ry * tvz - rz * tvy) * A)
    wp.atomic_add(out, o + 4, (rz * tvx - rx * tvz) * A)
    wp.atomic_add(out, o + 5, (rx * tvy - ry * tvx) * A)
    wp.atomic_add(out, o + 6, tpx * A)
    wp.atomic_add(out, o + 7, tpy * A)
    wp.atomic_add(out, o + 8, tpz * A)
    wp.atomic_add(out, o + 9, (ry * tpz - rz * tpy) * A)
    wp.atomic_add(out, o + 10, (rz * tpx - rx * tpz) * A)
    wp.atomic_add(out, o + 11, (rx * tpy - ry * tpx) * A)


# ─────────────────────────────────────────────────────────────────────────────
#  2-D field sampling (flat row-major F, stride s1=My)
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def lf_bilinear_2d(
    F: wp.array(dtype=wp.float64),
    Mx: wp.int32, My: wp.int32,
    bx0: wp.float64, by0: wp.float64,
    inv_dx: wp.float64, inv_dy: wp.float64,
    xq: wp.float64, yq: wp.float64,
) -> wp.float64:
    tx = wp.clamp((xq - bx0) * inv_dx, _ZERO, wp.float64(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, _ZERO, wp.float64(My - 1))

    ix = wp.int32(tx)
    if ix > Mx - 2:
        ix = Mx - 2
    iy = wp.int32(ty)
    if iy > My - 2:
        iy = My - 2

    fx = tx - wp.float64(ix)
    fy = ty - wp.float64(iy)
    wx0 = _ONE - fx
    wy0 = _ONE - fy

    s1 = My
    base = ix * s1 + iy
    return (
        wx0 * (wy0 * F[base] + fy * F[base + 1])
        + fx * (wy0 * F[base + s1] + fy * F[base + s1 + 1])
    )


@wp.func
def lf_biquadratic_2d(
    F: wp.array(dtype=wp.float64),
    Mx: wp.int32, My: wp.int32,
    bx0: wp.float64, by0: wp.float64,
    inv_dx: wp.float64, inv_dy: wp.float64,
    xq: wp.float64, yq: wp.float64,
) -> wp.float64:
    tx = wp.clamp((xq - bx0) * inv_dx, _ZERO, wp.float64(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, _ZERO, wp.float64(My - 1))

    ix = wp.int32(tx)
    if ix > Mx - 2:
        ix = Mx - 2
    iy = wp.int32(ty)
    if iy > My - 2:
        iy = My - 2

    res = _ZERO
    if ix < 1 or iy < 1 or Mx < 3 or My < 3:
        res = lf_bilinear_2d(F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq)
    else:
        fx = tx - wp.float64(ix)
        fy = ty - wp.float64(iy)

        wxm = _HALF * fx * (fx - _ONE)
        wx0 = _ONE - fx * fx
        wxp = _HALF * fx * (fx + _ONE)
        wym = _HALF * fy * (fy - _ONE)
        wy0 = _ONE - fy * fy
        wyp = _HALF * fy * (fy + _ONE)

        s1 = My
        base = (ix - 1) * s1 + (iy - 1)
        out = _ZERO
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
    F: wp.array(dtype=wp.float64),
    Mx: wp.int32, My: wp.int32,
    bx0: wp.float64, by0: wp.float64,
    inv_dx: wp.float64, inv_dy: wp.float64,
    xq: wp.float64, yq: wp.float64,
) -> wp.float64:
    res = _ZERO
    if method == 1:
        res = lf_biquadratic_2d(F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq)
    else:
        res = lf_bilinear_2d(F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq)
    return res


@wp.kernel
def lagrangian_forces_2d_kernel(
    exx: wp.array(dtype=wp.float64), exy: wp.array(dtype=wp.float64),
    eyy: wp.array(dtype=wp.float64),
    pp: wp.array(dtype=wp.float64), nrho: wp.array(dtype=wp.float64),
    nrho_scalar: wp.int32,
    cnt_x: wp.array(dtype=wp.float64), cnt_y: wp.array(dtype=wp.float64),
    offs: wp.array(dtype=wp.int32),
    elem_body: wp.array(dtype=wp.int32),
    com: wp.array(dtype=wp.float64),
    Mx: wp.int32, My: wp.int32,
    bx0: wp.float64, by0: wp.float64,
    inv_dx: wp.float64, inv_dy: wp.float64,
    interp_method: wp.int32,
    sample_offset: wp.float64,
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

    # Tangent via central diff on the closed contour → outward normal.
    tx = (cnt_x[gp] - cnt_x[gm]) * _HALF
    ty = (cnt_y[gp] - cnt_y[gm]) * _HALF
    L = wp.sqrt(tx * tx + ty * ty)
    if L < wp.float64(1e-30):
        L = wp.float64(1e-30)
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
    wq = _HALF * (dsm + dsp)

    rx = qx - com[b * 2 + 0]
    ry = qy - com[b * 2 + 1]

    o = b * 6
    wp.atomic_add(out, o + 0, tvx * wq)
    wp.atomic_add(out, o + 1, tvy * wq)
    wp.atomic_add(out, o + 2, (rx * tvy - ry * tvx) * wq)
    wp.atomic_add(out, o + 3, tpx * wq)
    wp.atomic_add(out, o + 4, tpy * wq)
    wp.atomic_add(out, o + 5, (rx * tpy - ry * tpx) * wq)


# ─────────────────────────────────────────────────────────────────────────────
#  Python launch wrappers (mirror the native ops.py signatures)
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_MAP = {"linear": 0, "quadratic": 1}


def _elem_body_from_offsets(offsets: torch.Tensor, n_total: int,
                            device) -> torch.Tensor:
    """Map each surface element index → owning body, from a (B+1,) prefix sum."""
    counts = (offsets[1:] - offsets[:-1]).to(torch.long)
    B = counts.numel()
    eb = torch.repeat_interleave(torch.arange(B, device=device), counts.to(device))
    assert eb.numel() == n_total, (eb.numel(), n_total)
    return eb.to(torch.int32)


def lagrangian_forces_3d_warp(
        eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz,
        p, nu_rho_field,
        tri_centroid, tri_normal, tri_area,
        tri_offsets, com_pos,
        bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
        Mx, My, Mz, method="linear", sample_offset=0.0):
    """Warp port of ``lagrangian_forces_3d``; returns (B, 12) float64 torch."""
    interp_method = _METHOD_MAP[method]
    dev = p.device
    wdev = "cuda:0" if dev.type == "cuda" else "cpu"
    B = int(com_pos.shape[0])
    T_total = int(tri_centroid.shape[1])

    def f(x):
        return wp.from_torch(x.reshape(-1).contiguous().to(torch.float64))

    nrho_scalar = 1 if nu_rho_field.numel() == 1 else 0
    elem_body = _elem_body_from_offsets(tri_offsets, T_total, dev)

    out = wp.zeros(B * 12, dtype=wp.float64, device=wdev)
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
            float(bx0), float(by0), float(bz0),
            float(inv_dx), float(inv_dy), float(inv_dz),
            int(interp_method), float(sample_offset),
            out,
        ],
        device=wdev)
    wp.synchronize()
    return wp.to_torch(out).reshape(B, 12)


def lagrangian_forces_2d_warp(
        eps_xx, eps_xy, eps_yy,
        p, nu_rho_field,
        cnt_flat, cnt_offsets, com_pos,
        bx0, by0, inv_dx, inv_dy,
        Mx, My, method="linear", sample_offset=0.0):
    """Warp port of ``lagrangian_forces_2d``; returns (B, 6) float64 torch."""
    interp_method = _METHOD_MAP[method]
    dev = p.device
    wdev = "cuda:0" if dev.type == "cuda" else "cpu"
    B = int(com_pos.shape[0])
    M_total = int(cnt_flat.shape[1])

    def f(x):
        return wp.from_torch(x.reshape(-1).contiguous().to(torch.float64))

    nrho_scalar = 1 if nu_rho_field.numel() == 1 else 0
    elem_body = _elem_body_from_offsets(cnt_offsets, M_total, dev)
    offs_i32 = wp.from_torch(cnt_offsets.to(torch.int32).contiguous())

    out = wp.zeros(B * 6, dtype=wp.float64, device=wdev)
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
            float(bx0), float(by0), float(inv_dx), float(inv_dy),
            int(interp_method), float(sample_offset),
            out,
        ],
        device=wdev)
    wp.synchronize()
    return wp.to_torch(out).reshape(B, 6)
