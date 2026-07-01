"""Warp single-source **Eulerian** surface-force readout (Phase D).

Port of the native ``streaming_sdf_forces_post_{2,3}d`` CUDA kernels
(``src/kernels/csrc/cuda/streaming_sdf{,_2d}.cu``,
``streaming_sdf_forces_post_*_kernel`` + the deltaH ``forces_post_deltaH_pressure_*``
second pass).  This is the n·δ viscous+pressure band integral used by the
``forces_method2`` Eulerian readout — the *last* native custom kernel without a
Warp port (the other Warp-class scatter, the Lagrangian readout, is in
``warp_lagrangian.py``).

For every body ``b`` and every cell in the body's dirty AABB the kernel:
  1. re-samples the body-local cc-SDF ``s_body`` from the streamed flat table
     (``F_flat``/``body_meta``/``kin``; reuses ``sdf_sample_off_2d`` /
     ``sdf_sample_off_3d`` — the exact Kernel-A sampler);
  2. tests the BDIM band ``band_lo < s_body < band_hi``;
  3. builds the **union** normal ``n = ∇sdf_cc/|∇sdf_cc|`` (central / 2nd-order
     one-sided at the grid edge);
  4. forms the viscous stress ``σ·n`` from the post-step velocity (cell-centred
     gradients) and the pressure traction ``−p n``;
  5. weights by the smoothed δ (visc band centred on ``eps_solver``, pres band on
     0; ``delta_order==2`` divides by |∇s_body| sampled at ±h);
  6. accumulates ``[fv, t_v, fp, t_p] · h^d`` into the per-body float64 row.

**Block-reduction note.**  The native kernel block-reduces (CUB ``BlockReduce``)
before one ``atomicAdd`` per channel per block — purely an atomic-contention
optimisation.  This port does one ``wp.atomic_add`` per cell into the float64
``out`` accumulator: the *sum is identical*, only the float64 reduction order
(hence ~1e-9 noise) differs — exactly the tolerance the native self-test uses.

**Precision (single source, both dtypes).**  Per-element math runs in the field
element type (``Any`` generic, f32/f64; matches the native ``AT_DISPATCH``
``scalar_t``); only the final products cast to ``double`` for the atomic into the
always-float64 ``out`` (native ``double* out``).  ``wp.overload`` registers
f32+f64.  ``F_offsets``/``body_shapes``/``aabb_*`` stay int64, the keys/ids int.

**deltaH submethod.**  ``force_submethod==1`` zeros the main kernel's pressure δ
(``with_pressure=0`` → only viscous channels) and runs a second pass
(``forces_post_deltaH_pressure_2d``) over the union AABB: the union-∂H pressure
force density split to bodies by a softmin partition of unity (temperature
``ph_tau``).  Faithful port; same atomic-scatter class.
"""
from __future__ import annotations

from typing import Any

import warp as wp
import torch

wp.init()

from lilytorch.src.kernels.streaming_sdf_2d import sdf_sample_off_2d
from lilytorch.src.kernels.streaming_sdf import trilinear_sample_off


_PI = 3.141592653589793


# ─────────────────────────────────────────────────────────────────────────────
#  3-D SDF sampling with F-offset (faithful port of sdf_sample_dispatch).
#  trilinear comes from warp_kernels (F_off-aware); triquadratic ported here.
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def _triquadratic_sample_off_3d(
    F:      wp.array(dtype=Any),
    F_off:  int,
    Mx: int, My: int, Mz: int,
    bx0: Any, by0: Any, bz0: Any,
    inv_dx: Any, inv_dy: Any, inv_dz: Any,
    xq: Any, yq: Any, zq: Any,
):
    zero = type(bx0)(0.0)
    one = type(bx0)(1.0)
    half = type(bx0)(0.5)
    tx = wp.clamp((xq - bx0) * inv_dx, zero, type(bx0)(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, zero, type(bx0)(My - 1))
    tz = wp.clamp((zq - bz0) * inv_dz, zero, type(bx0)(Mz - 1))

    ix = wp.min(int(tx), Mx - 2)
    iy = wp.min(int(ty), My - 2)
    iz = wp.min(int(tz), Mz - 2)

    if ix < 1 or iy < 1 or iz < 1 or Mx < 3 or My < 3 or Mz < 3:
        return trilinear_sample_off(F, F_off, Mx, My, Mz, bx0, by0, bz0,
                                    inv_dx, inv_dy, inv_dz, xq, yq, zq)

    fx = tx - type(bx0)(ix)
    fy = ty - type(bx0)(iy)
    fz = tz - type(bx0)(iz)

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
    base = F_off + (ix - 1) * s1 + (iy - 1) * s2 + (iz - 1)

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
            rrow = wzm * F[b1] + wz0 * F[b1 + 1] + wzp * F[b1 + 2]
            plane += wy * rrow
        out += wx * plane
    return out


@wp.func
def _sdf_sample_off_3d(
    interp_method: int,
    F:      wp.array(dtype=Any),
    F_off:  int,
    Mx: int, My: int, Mz: int,
    bx0: Any, by0: Any, bz0: Any,
    inv_dx: Any, inv_dy: Any, inv_dz: Any,
    xq: Any, yq: Any, zq: Any,
):
    if interp_method == 1:
        return _triquadratic_sample_off_3d(F, F_off, Mx, My, Mz, bx0, by0, bz0,
                                           inv_dx, inv_dy, inv_dz, xq, yq, zq)
    return trilinear_sample_off(F, F_off, Mx, My, Mz, bx0, by0, bz0,
                                inv_dx, inv_dy, inv_dz, xq, yq, zq)


# ═════════════════════════════════════════════════════════════════════════════
#  2-D main kernel — n·δ viscous (+ optional pressure) band integral
# ═════════════════════════════════════════════════════════════════════════════

@wp.kernel
def forces_post_2d_kernel(
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx:          wp.array(dtype=Any),
    gy:          wp.array(dtype=Any),
    Ngx: wp.int32, Ngy: wp.int32,
    sdf_cc:      wp.array(dtype=Any),
    interp_method: wp.int32,
    u_prev:      wp.array(dtype=Any),
    v_prev:      wp.array(dtype=Any),
    p_prev:      wp.array(dtype=Any),
    nu_rho_field: wp.array(dtype=Any),
    nu_rho_field_size: wp.int32,
    inv_h: Any, eps_body: Any, eps_solver: Any, h2: Any,
    delta_order: wp.int32,
    with_pressure: wp.int32,
    max_vol: wp.int32,
    out: wp.array(dtype=wp.float64),   # (B*6,)
):
    tid = wp.tid()
    b = tid // max_vol
    local = tid - b * max_vol
    Ai = wp.int32(aabb_dim[b * 2 + 0])
    Aj = wp.int32(aabb_dim[b * 2 + 1])
    vol = Ai * Aj
    if local >= vol:
        return

    zero = type(inv_h)(0.0)
    one = type(inv_h)(1.0)
    half = type(inv_h)(0.5)
    two = type(inv_h)(2.0)

    di = local // Aj
    dj = local - di * Aj
    i = wp.int32(aabb_lo[b * 2 + 0]) + di
    j = wp.int32(aabb_lo[b * 2 + 1]) + dj
    g_idx = i * Ngy + j

    F_off = wp.int32(F_offsets[b])
    Mx = wp.int32(body_shapes[b * 2 + 0])
    My = wp.int32(body_shapes[b * 2 + 1])

    bx0 = body_meta[b * 7 + 0]
    by0 = body_meta[b * 7 + 1]
    idx_ = body_meta[b * 7 + 4]
    idy_ = body_meta[b * 7 + 5]

    r00 = kin[b * 11 + 0]
    r01 = kin[b * 11 + 1]
    r10 = kin[b * 11 + 2]
    r11 = kin[b * 11 + 3]
    bp_x = kin[b * 11 + 4]
    bp_y = kin[b * 11 + 5]
    cm_x = kin[b * 11 + 6]
    cm_y = kin[b * 11 + 7]

    xc = gx[i]
    yc = gy[j]
    dx_w = xc - bp_x
    dy_w = yc - bp_y
    bxq = r00 * dx_w + r01 * dy_w
    byq = r10 * dx_w + r11 * dy_w

    s_cc_body = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                  bx0, by0, idx_, idy_, bxq, byq)

    band_lo = wp.min(eps_solver - eps_body, -eps_body)
    band_hi = wp.max(eps_solver + eps_body, eps_body)
    if s_cc_body <= band_lo or s_cc_body >= band_hi:
        return

    nu_rho_val = nu_rho_field[0]
    if nu_rho_field_size != 1:
        nu_rho_val = nu_rho_field[g_idx]

    # ── union normal n = ∇sdf_cc / |∇sdf_cc| (central / 2nd-order one-sided) ──
    dsdx_union = zero
    if Ngx >= 3:
        if i == 0:
            dsdx_union = (type(inv_h)(-3.0) * sdf_cc[j]
                          + type(inv_h)(4.0) * sdf_cc[Ngy + j]
                          - sdf_cc[2 * Ngy + j]) * half * inv_h
        elif i == Ngx - 1:
            dsdx_union = (type(inv_h)(3.0) * sdf_cc[(Ngx - 1) * Ngy + j]
                          - type(inv_h)(4.0) * sdf_cc[(Ngx - 2) * Ngy + j]
                          + sdf_cc[(Ngx - 3) * Ngy + j]) * half * inv_h
        else:
            dsdx_union = (sdf_cc[(i + 1) * Ngy + j]
                          - sdf_cc[(i - 1) * Ngy + j]) * half * inv_h
    elif Ngx == 2:
        dsdx_union = (sdf_cc[Ngy + j] - sdf_cc[j]) * inv_h

    dsdy_union = zero
    row = i * Ngy
    if Ngy >= 3:
        if j == 0:
            dsdy_union = (type(inv_h)(-3.0) * sdf_cc[row]
                          + type(inv_h)(4.0) * sdf_cc[row + 1]
                          - sdf_cc[row + 2]) * half * inv_h
        elif j == Ngy - 1:
            dsdy_union = (type(inv_h)(3.0) * sdf_cc[row + (Ngy - 1)]
                          - type(inv_h)(4.0) * sdf_cc[row + (Ngy - 2)]
                          + sdf_cc[row + (Ngy - 3)]) * half * inv_h
        else:
            dsdy_union = (sdf_cc[row + (j + 1)]
                          - sdf_cc[row + (j - 1)]) * half * inv_h
    elif Ngy == 2:
        dsdy_union = (sdf_cc[i * Ngy + 1] - sdf_cc[i * Ngy]) * inv_h

    union_norm = wp.sqrt(dsdx_union * dsdx_union + dsdy_union * dsdy_union)
    union_inv_norm = zero
    if union_norm > zero:
        union_inv_norm = one / union_norm
    nx = dsdx_union * union_inv_norm
    ny = dsdy_union * union_inv_norm

    # ── clamped neighbour indices ────────────────────────────────────────────
    im1 = i - 1
    if i <= 0:
        im1 = 0
    ip1 = i + 1
    if i + 1 >= Ngx:
        ip1 = i
    im2 = i - 2
    if i <= 1:
        im2 = 0
    ip2 = i + 2
    if i + 2 >= Ngx:
        ip2 = Ngx - 1
    jm1 = j - 1
    if j <= 0:
        jm1 = 0
    jp1 = j + 1
    if j + 1 >= Ngy:
        jp1 = j
    jm2 = j - 2
    if j <= 1:
        jm2 = 0
    jp2 = j + 2
    if j + 2 >= Ngy:
        jp2 = Ngy - 1

    # ── velocity gradients (staggered → cell-centred) ────────────────────────
    if i + 1 < Ngx:
        dudx = (u_prev[ip1 * Ngy + j] - u_prev[i * Ngy + j]) * inv_h
    else:
        dudx = (u_prev[i * Ngy + j] - u_prev[im1 * Ngy + j]) * inv_h

    if j + 1 < Ngy:
        dvdy = (v_prev[i * Ngy + jp1] - v_prev[i * Ngy + j]) * inv_h
    else:
        dvdy = (v_prev[i * Ngy + j] - v_prev[i * Ngy + jm1]) * inv_h

    u_cc_jm2 = half * (u_prev[i * Ngy + jm2] + u_prev[ip1 * Ngy + jm2])
    u_cc_jm1 = half * (u_prev[i * Ngy + jm1] + u_prev[ip1 * Ngy + jm1])
    u_cc_j0 = half * (u_prev[i * Ngy + j] + u_prev[ip1 * Ngy + j])
    u_cc_jp1 = half * (u_prev[i * Ngy + jp1] + u_prev[ip1 * Ngy + jp1])
    u_cc_jp2 = half * (u_prev[i * Ngy + jp2] + u_prev[ip1 * Ngy + jp2])

    if Ngy >= 3:
        if j == 0:
            dudy = (type(inv_h)(-3.0) * u_cc_j0 + type(inv_h)(4.0) * u_cc_jp1
                    - u_cc_jp2) * half * inv_h
        elif j == Ngy - 1:
            dudy = (type(inv_h)(3.0) * u_cc_j0 - type(inv_h)(4.0) * u_cc_jm1
                    + u_cc_jm2) * half * inv_h
        else:
            dudy = (u_cc_jp1 - u_cc_jm1) * half * inv_h
    else:
        dudy = (u_cc_jp1 - u_cc_jm1) * half * inv_h

    v_cc_im2 = half * (v_prev[im2 * Ngy + j] + v_prev[im2 * Ngy + jp1])
    v_cc_im1 = half * (v_prev[im1 * Ngy + j] + v_prev[im1 * Ngy + jp1])
    v_cc_i0 = half * (v_prev[i * Ngy + j] + v_prev[i * Ngy + jp1])
    v_cc_ip1 = half * (v_prev[ip1 * Ngy + j] + v_prev[ip1 * Ngy + jp1])
    v_cc_ip2 = half * (v_prev[ip2 * Ngy + j] + v_prev[ip2 * Ngy + jp1])

    if Ngx >= 3:
        if i == 0:
            dvdx = (type(inv_h)(-3.0) * v_cc_i0 + type(inv_h)(4.0) * v_cc_ip1
                    - v_cc_ip2) * half * inv_h
        elif i == Ngx - 1:
            dvdx = (type(inv_h)(3.0) * v_cc_i0 - type(inv_h)(4.0) * v_cc_im1
                    + v_cc_im2) * half * inv_h
        else:
            dvdx = (v_cc_ip1 - v_cc_im1) * half * inv_h
    else:
        dvdx = (v_cc_ip1 - v_cc_im1) * half * inv_h

    xs = nu_rho_val * (two * dudx * nx + (dudy + dvdx) * ny)
    ys = nu_rho_val * ((dvdx + dudy) * nx + two * dvdy * ny)

    p_c = p_prev[g_idx]
    pxv = -p_c * nx
    pyv = -p_c * ny

    # ── smoothed delta weights ───────────────────────────────────────────────
    inv_2eps = half / eps_body
    pi_ov_eb = type(inv_h)(_PI) / eps_body

    delta_visc = zero
    delta_pres = zero
    d_visc = s_cc_body - eps_solver
    if d_visc > -eps_body and d_visc < eps_body:
        delta_visc = (one + wp.cos(pi_ov_eb * d_visc)) * inv_2eps
    if s_cc_body > -eps_body and s_cc_body < eps_body:
        delta_pres = (one + wp.cos(pi_ov_eb * s_cc_body)) * inv_2eps
    if with_pressure == 0:
        delta_pres = zero

    if delta_order == 2 and (delta_visc > zero or delta_pres > zero):
        h_grid = one / inv_h
        s_xp = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                 bx0, by0, idx_, idy_,
                                 bxq + r00 * h_grid, byq + r10 * h_grid)
        s_xm = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                 bx0, by0, idx_, idy_,
                                 bxq - r00 * h_grid, byq - r10 * h_grid)
        s_yp = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                 bx0, by0, idx_, idy_,
                                 bxq + r01 * h_grid, byq + r11 * h_grid)
        s_ym = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                 bx0, by0, idx_, idy_,
                                 bxq - r01 * h_grid, byq - r11 * h_grid)
        dsdx = (s_xp - s_xm) * half * inv_h
        dsdy = (s_yp - s_ym) * half * inv_h
        grad_mag = wp.sqrt(dsdx * dsdx + dsdy * dsdy)
        min_grad = type(inv_h)(1e-3)
        if grad_mag < min_grad:
            grad_mag = min_grad
        inv_grad = one / grad_mag
        delta_visc = delta_visc * inv_grad
        delta_pres = delta_pres * inv_grad

    arm_x = xc - cm_x
    arm_y = yc - cm_y

    fv_x = wp.float64(xs * delta_visc)
    fv_y = wp.float64(ys * delta_visc)
    fp_x = wp.float64(pxv * delta_pres)
    fp_y = wp.float64(pyv * delta_pres)
    ax = wp.float64(arm_x)
    ay = wp.float64(arm_y)
    h2_d = wp.float64(h2)

    o = b * 6
    wp.atomic_add(out, o + 0, fv_x * h2_d)
    wp.atomic_add(out, o + 1, fv_y * h2_d)
    wp.atomic_add(out, o + 2, (ax * fv_y - ay * fv_x) * h2_d)
    wp.atomic_add(out, o + 3, fp_x * h2_d)
    wp.atomic_add(out, o + 4, fp_y * h2_d)
    wp.atomic_add(out, o + 5, (ax * fp_y - ay * fp_x) * h2_d)


# ── 2-D deltaH (∂H) pressure second pass (force_submethod == 1) ──────────────

@wp.func
def _heaviside_smooth_2d(phi: Any, inv_eps: Any):
    pi = type(phi)(_PI)
    one = type(phi)(1.0)
    half = type(phi)(0.5)
    x = phi * inv_eps
    x = wp.clamp(x, -one, one)
    return half * (one + x + wp.sin(pi * x) / pi)


@wp.kernel
def forces_post_deltaH_pressure_2d_kernel(
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx:          wp.array(dtype=Any),
    gy:          wp.array(dtype=Any),
    Ngx: wp.int32, Ngy: wp.int32,
    sdf_cc:      wp.array(dtype=Any),
    interp_method: wp.int32,
    p_prev:      wp.array(dtype=Any),
    inv_h: Any, inv_eps: Any, inv_tau: Any, h2: Any,
    B: wp.int32,
    uli0: wp.int32, ulj0: wp.int32, ULi: wp.int32, ULj: wp.int32,
    out: wp.array(dtype=wp.float64),
):
    local = wp.tid()
    uvol = ULi * ULj
    if local >= uvol:
        return
    di = local // ULj
    dj = local - di * ULj
    i = uli0 + di
    j = ulj0 + dj
    if i < 0 or i >= Ngx or j < 0 or j >= Ngy:
        return

    zero = type(inv_h)(0.0)
    half = type(inv_h)(0.5)
    one = type(inv_h)(1.0)

    # ∇H of the smoothed union Heaviside (central / 2nd-order one-sided).
    gHx = zero
    if Ngx >= 3:
        if i == 0:
            gHx = (type(inv_h)(-3.0) * _heaviside_smooth_2d(sdf_cc[0 * Ngy + j], inv_eps)
                   + type(inv_h)(4.0) * _heaviside_smooth_2d(sdf_cc[1 * Ngy + j], inv_eps)
                   - _heaviside_smooth_2d(sdf_cc[2 * Ngy + j], inv_eps)) * half * inv_h
        elif i == Ngx - 1:
            gHx = (type(inv_h)(3.0) * _heaviside_smooth_2d(sdf_cc[(Ngx - 1) * Ngy + j], inv_eps)
                   - type(inv_h)(4.0) * _heaviside_smooth_2d(sdf_cc[(Ngx - 2) * Ngy + j], inv_eps)
                   + _heaviside_smooth_2d(sdf_cc[(Ngx - 3) * Ngy + j], inv_eps)) * half * inv_h
        else:
            gHx = (_heaviside_smooth_2d(sdf_cc[(i + 1) * Ngy + j], inv_eps)
                   - _heaviside_smooth_2d(sdf_cc[(i - 1) * Ngy + j], inv_eps)) * half * inv_h
    elif Ngx == 2:
        gHx = (_heaviside_smooth_2d(sdf_cc[1 * Ngy + j], inv_eps)
               - _heaviside_smooth_2d(sdf_cc[0 * Ngy + j], inv_eps)) * inv_h

    gHy = zero
    rr = i * Ngy
    if Ngy >= 3:
        if j == 0:
            gHy = (type(inv_h)(-3.0) * _heaviside_smooth_2d(sdf_cc[rr + 0], inv_eps)
                   + type(inv_h)(4.0) * _heaviside_smooth_2d(sdf_cc[rr + 1], inv_eps)
                   - _heaviside_smooth_2d(sdf_cc[rr + 2], inv_eps)) * half * inv_h
        elif j == Ngy - 1:
            gHy = (type(inv_h)(3.0) * _heaviside_smooth_2d(sdf_cc[rr + (Ngy - 1)], inv_eps)
                   - type(inv_h)(4.0) * _heaviside_smooth_2d(sdf_cc[rr + (Ngy - 2)], inv_eps)
                   + _heaviside_smooth_2d(sdf_cc[rr + (Ngy - 3)], inv_eps)) * half * inv_h
        else:
            gHy = (_heaviside_smooth_2d(sdf_cc[rr + (j + 1)], inv_eps)
                   - _heaviside_smooth_2d(sdf_cc[rr + (j - 1)], inv_eps)) * half * inv_h
    elif Ngy == 2:
        gHy = (_heaviside_smooth_2d(sdf_cc[rr + 1], inv_eps)
               - _heaviside_smooth_2d(sdf_cc[rr + 0], inv_eps)) * inv_h

    if gHx == zero and gHy == zero:
        return

    g = i * Ngy + j
    p_c = p_prev[g]
    fdx = -p_c * gHx
    fdy = -p_c * gHy
    xc = gx[i]
    yc = gy[j]
    sdfu = sdf_cc[g]

    # softmin partition of unity Z = Σ_b exp(-(s_b - s_union)/tau)
    Z = zero
    for b in range(B):
        i0 = wp.int32(aabb_lo[b * 2 + 0])
        j0 = wp.int32(aabb_lo[b * 2 + 1])
        Ai = wp.int32(aabb_dim[b * 2 + 0])
        Aj = wp.int32(aabb_dim[b * 2 + 1])
        if i >= i0 and i < i0 + Ai and j >= j0 and j < j0 + Aj:
            F_off = wp.int32(F_offsets[b])
            Mx = wp.int32(body_shapes[b * 2 + 0])
            My = wp.int32(body_shapes[b * 2 + 1])
            dx_w = xc - kin[b * 11 + 4]
            dy_w = yc - kin[b * 11 + 5]
            bxq = kin[b * 11 + 0] * dx_w + kin[b * 11 + 1] * dy_w
            byq = kin[b * 11 + 2] * dx_w + kin[b * 11 + 3] * dy_w
            s_b = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                    body_meta[b * 7 + 0], body_meta[b * 7 + 1],
                                    body_meta[b * 7 + 4], body_meta[b * 7 + 5],
                                    bxq, byq)
            Z += wp.exp(-(s_b - sdfu) * inv_tau)
    if Z <= zero:
        return
    inv_Z = one / Z
    h2_d = wp.float64(h2)

    for b in range(B):
        i0 = wp.int32(aabb_lo[b * 2 + 0])
        j0 = wp.int32(aabb_lo[b * 2 + 1])
        Ai = wp.int32(aabb_dim[b * 2 + 0])
        Aj = wp.int32(aabb_dim[b * 2 + 1])
        if i >= i0 and i < i0 + Ai and j >= j0 and j < j0 + Aj:
            F_off = wp.int32(F_offsets[b])
            Mx = wp.int32(body_shapes[b * 2 + 0])
            My = wp.int32(body_shapes[b * 2 + 1])
            dx_w = xc - kin[b * 11 + 4]
            dy_w = yc - kin[b * 11 + 5]
            bxq = kin[b * 11 + 0] * dx_w + kin[b * 11 + 1] * dy_w
            byq = kin[b * 11 + 2] * dx_w + kin[b * 11 + 3] * dy_w
            s_b = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                    body_meta[b * 7 + 0], body_meta[b * 7 + 1],
                                    body_meta[b * 7 + 4], body_meta[b * 7 + 5],
                                    bxq, byq)
            wb = wp.exp(-(s_b - sdfu) * inv_tau) * inv_Z
            fbx = wb * fdx
            fby = wb * fdy
            ax = xc - kin[b * 11 + 6]
            ay = yc - kin[b * 11 + 7]
            o = b * 6
            wp.atomic_add(out, o + 3, wp.float64(fbx) * h2_d)
            wp.atomic_add(out, o + 4, wp.float64(fby) * h2_d)
            wp.atomic_add(out, o + 5,
                          (wp.float64(ax) * wp.float64(fby)
                           - wp.float64(ay) * wp.float64(fbx)) * h2_d)


# Register float32 + float64 specialisations (generic args only).
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(forces_post_2d_kernel, {
        "F_flat": _A, "body_meta": _A, "kin": _A, "gx": _A, "gy": _A,
        "sdf_cc": _A, "u_prev": _A, "v_prev": _A, "p_prev": _A,
        "nu_rho_field": _A,
        "inv_h": _dt, "eps_body": _dt, "eps_solver": _dt, "h2": _dt,
    })
    wp.overload(forces_post_deltaH_pressure_2d_kernel, {
        "F_flat": _A, "body_meta": _A, "kin": _A, "gx": _A, "gy": _A,
        "sdf_cc": _A, "p_prev": _A,
        "inv_h": _dt, "inv_eps": _dt, "inv_tau": _dt, "h2": _dt,
    })


# ═════════════════════════════════════════════════════════════════════════════
#  3-D main kernel — n·δ viscous (+ optional pressure) band integral
# ═════════════════════════════════════════════════════════════════════════════

@wp.kernel
def forces_post_3d_kernel(
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx:          wp.array(dtype=Any),
    gy:          wp.array(dtype=Any),
    gz:          wp.array(dtype=Any),
    Ngx: wp.int32, Ngy: wp.int32, Ngz: wp.int32,
    sdf_cc:      wp.array(dtype=Any),
    interp_method: wp.int32,
    u_prev:      wp.array(dtype=Any),
    v_prev:      wp.array(dtype=Any),
    w_prev:      wp.array(dtype=Any),
    p_prev:      wp.array(dtype=Any),
    nu_rho_field: wp.array(dtype=Any),
    nu_rho_field_size: wp.int32,
    inv_h: Any, eps_body: Any, eps_solver: Any, h3: Any,
    delta_order: wp.int32,
    with_pressure: wp.int32,
    max_vol: wp.int32,
    out: wp.array(dtype=wp.float64),   # (B*12,)
):
    tid = wp.tid()
    b = tid // max_vol
    local = tid - b * max_vol
    Ai = wp.int32(aabb_dim[b * 3 + 0])
    Aj = wp.int32(aabb_dim[b * 3 + 1])
    Ak = wp.int32(aabb_dim[b * 3 + 2])
    vol = Ai * Aj * Ak
    if local >= vol:
        return

    zero = type(inv_h)(0.0)
    one = type(inv_h)(1.0)
    half = type(inv_h)(0.5)
    two = type(inv_h)(2.0)

    di = local // (Aj * Ak)
    rem = local - di * (Aj * Ak)
    dj = rem // Ak
    dk = rem - dj * Ak
    i = wp.int32(aabb_lo[b * 3 + 0]) + di
    j = wp.int32(aabb_lo[b * 3 + 1]) + dj
    k = wp.int32(aabb_lo[b * 3 + 2]) + dk
    g_idx = (i * Ngy + j) * Ngz + k

    F_off = wp.int32(F_offsets[b])
    Mx = wp.int32(body_shapes[b * 3 + 0])
    My = wp.int32(body_shapes[b * 3 + 1])
    Mz = wp.int32(body_shapes[b * 3 + 2])
    bx0 = body_meta[b * 10 + 0]
    by0 = body_meta[b * 10 + 1]
    bz0 = body_meta[b * 10 + 2]
    idx_ = body_meta[b * 10 + 6]
    idy_ = body_meta[b * 10 + 7]
    idz_ = body_meta[b * 10 + 8]

    r00 = kin[b * 21 + 0]
    r01 = kin[b * 21 + 1]
    r02 = kin[b * 21 + 2]
    r10 = kin[b * 21 + 3]
    r11 = kin[b * 21 + 4]
    r12 = kin[b * 21 + 5]
    r20 = kin[b * 21 + 6]
    r21 = kin[b * 21 + 7]
    r22 = kin[b * 21 + 8]
    bp_x = kin[b * 21 + 9]
    bp_y = kin[b * 21 + 10]
    bp_z = kin[b * 21 + 11]
    cm_x = kin[b * 21 + 12]
    cm_y = kin[b * 21 + 13]
    cm_z = kin[b * 21 + 14]

    xc = gx[i]
    yc = gy[j]
    zc = gz[k]
    dx_w = xc - bp_x
    dy_w = yc - bp_y
    dz_w = zc - bp_z
    bxq = r00 * dx_w + r01 * dy_w + r02 * dz_w
    byq = r10 * dx_w + r11 * dy_w + r12 * dz_w
    bzq = r20 * dx_w + r21 * dy_w + r22 * dz_w

    s_cc_body = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz,
                                   bx0, by0, bz0, idx_, idy_, idz_,
                                   bxq, byq, bzq)

    band_lo = wp.min(eps_solver - eps_body, -eps_body)
    band_hi = wp.max(eps_solver + eps_body, eps_body)
    if s_cc_body <= band_lo or s_cc_body >= band_hi:
        return

    nu_rho_val = nu_rho_field[0]
    if nu_rho_field_size != 1:
        nu_rho_val = nu_rho_field[g_idx]

    # ── union normal n = ∇sdf_cc / |∇sdf_cc| ─────────────────────────────────
    sij = i * Ngy * Ngz + j * Ngz + k
    dsdx_union = zero
    if Ngx >= 3:
        if i == 0:
            dsdx_union = (type(inv_h)(-3.0) * sdf_cc[(0 * Ngy + j) * Ngz + k]
                          + type(inv_h)(4.0) * sdf_cc[(1 * Ngy + j) * Ngz + k]
                          - sdf_cc[(2 * Ngy + j) * Ngz + k]) * half * inv_h
        elif i == Ngx - 1:
            dsdx_union = (type(inv_h)(3.0) * sdf_cc[((Ngx - 1) * Ngy + j) * Ngz + k]
                          - type(inv_h)(4.0) * sdf_cc[((Ngx - 2) * Ngy + j) * Ngz + k]
                          + sdf_cc[((Ngx - 3) * Ngy + j) * Ngz + k]) * half * inv_h
        else:
            dsdx_union = (sdf_cc[((i + 1) * Ngy + j) * Ngz + k]
                          - sdf_cc[((i - 1) * Ngy + j) * Ngz + k]) * half * inv_h
    elif Ngx == 2:
        dsdx_union = (sdf_cc[(1 * Ngy + j) * Ngz + k]
                      - sdf_cc[(0 * Ngy + j) * Ngz + k]) * inv_h

    dsdy_union = zero
    if Ngy >= 3:
        if j == 0:
            dsdy_union = (type(inv_h)(-3.0) * sdf_cc[(i * Ngy + 0) * Ngz + k]
                          + type(inv_h)(4.0) * sdf_cc[(i * Ngy + 1) * Ngz + k]
                          - sdf_cc[(i * Ngy + 2) * Ngz + k]) * half * inv_h
        elif j == Ngy - 1:
            dsdy_union = (type(inv_h)(3.0) * sdf_cc[(i * Ngy + (Ngy - 1)) * Ngz + k]
                          - type(inv_h)(4.0) * sdf_cc[(i * Ngy + (Ngy - 2)) * Ngz + k]
                          + sdf_cc[(i * Ngy + (Ngy - 3)) * Ngz + k]) * half * inv_h
        else:
            dsdy_union = (sdf_cc[(i * Ngy + (j + 1)) * Ngz + k]
                          - sdf_cc[(i * Ngy + (j - 1)) * Ngz + k]) * half * inv_h
    elif Ngy == 2:
        dsdy_union = (sdf_cc[(i * Ngy + 1) * Ngz + k]
                      - sdf_cc[(i * Ngy + 0) * Ngz + k]) * inv_h

    dsdz_union = zero
    if Ngz >= 3:
        if k == 0:
            dsdz_union = (type(inv_h)(-3.0) * sdf_cc[sij - k + 0]
                          + type(inv_h)(4.0) * sdf_cc[sij - k + 1]
                          - sdf_cc[sij - k + 2]) * half * inv_h
        elif k == Ngz - 1:
            dsdz_union = (type(inv_h)(3.0) * sdf_cc[sij]
                          - type(inv_h)(4.0) * sdf_cc[sij - 1]
                          + sdf_cc[sij - 2]) * half * inv_h
        else:
            dsdz_union = (sdf_cc[sij + 1] - sdf_cc[sij - 1]) * half * inv_h
    elif Ngz == 2:
        dsdz_union = (sdf_cc[(i * Ngy + j) * Ngz + 1]
                      - sdf_cc[(i * Ngy + j) * Ngz + 0]) * inv_h

    union_norm = wp.sqrt(dsdx_union * dsdx_union + dsdy_union * dsdy_union
                         + dsdz_union * dsdz_union)
    inv_norm = zero
    if union_norm > zero:
        inv_norm = one / union_norm
    nx = dsdx_union * inv_norm
    ny = dsdy_union * inv_norm
    nz = dsdz_union * inv_norm

    # ── clamped neighbour indices ────────────────────────────────────────────
    im1 = i - 1
    if i <= 0:
        im1 = 0
    ip1 = i + 1
    if i + 1 >= Ngx:
        ip1 = i
    im2 = i - 2
    if i <= 1:
        im2 = 0
    ip2 = i + 2
    if i + 2 >= Ngx:
        ip2 = Ngx - 1
    jm1 = j - 1
    if j <= 0:
        jm1 = 0
    jp1 = j + 1
    if j + 1 >= Ngy:
        jp1 = j
    jm2 = j - 2
    if j <= 1:
        jm2 = 0
    jp2 = j + 2
    if j + 2 >= Ngy:
        jp2 = Ngy - 1
    km1 = k - 1
    if k <= 0:
        km1 = 0
    kp1 = k + 1
    if k + 1 >= Ngz:
        kp1 = k
    km2 = k - 2
    if k <= 1:
        km2 = 0
    kp2 = k + 2
    if k + 2 >= Ngz:
        kp2 = Ngz - 1

    # flat index helper g(i,j,k) = (i*Ngy+j)*Ngz+k
    # ── normal derivatives (forward; backward at upper boundary) ─────────────
    if i + 1 < Ngx:
        dudx = (u_prev[(ip1 * Ngy + j) * Ngz + k] - u_prev[(i * Ngy + j) * Ngz + k]) * inv_h
    else:
        dudx = (u_prev[(i * Ngy + j) * Ngz + k] - u_prev[(im1 * Ngy + j) * Ngz + k]) * inv_h
    if j + 1 < Ngy:
        dvdy = (v_prev[(i * Ngy + jp1) * Ngz + k] - v_prev[(i * Ngy + j) * Ngz + k]) * inv_h
    else:
        dvdy = (v_prev[(i * Ngy + j) * Ngz + k] - v_prev[(i * Ngy + jm1) * Ngz + k]) * inv_h
    if k + 1 < Ngz:
        dwdz = (w_prev[(i * Ngy + j) * Ngz + kp1] - w_prev[(i * Ngy + j) * Ngz + k]) * inv_h
    else:
        dwdz = (w_prev[(i * Ngy + j) * Ngz + k] - w_prev[(i * Ngy + j) * Ngz + km1]) * inv_h

    # dudy (u CC = 0.5*(u[i,j,k]+u[i+1,j,k]))
    u_cc_jm2 = half * (u_prev[(i * Ngy + jm2) * Ngz + k] + u_prev[(ip1 * Ngy + jm2) * Ngz + k])
    u_cc_jm1 = half * (u_prev[(i * Ngy + jm1) * Ngz + k] + u_prev[(ip1 * Ngy + jm1) * Ngz + k])
    u_cc_j0 = half * (u_prev[(i * Ngy + j) * Ngz + k] + u_prev[(ip1 * Ngy + j) * Ngz + k])
    u_cc_jp1 = half * (u_prev[(i * Ngy + jp1) * Ngz + k] + u_prev[(ip1 * Ngy + jp1) * Ngz + k])
    u_cc_jp2 = half * (u_prev[(i * Ngy + jp2) * Ngz + k] + u_prev[(ip1 * Ngy + jp2) * Ngz + k])
    if Ngy >= 3:
        if j == 0:
            dudy = (type(inv_h)(-3.0) * u_cc_j0 + type(inv_h)(4.0) * u_cc_jp1 - u_cc_jp2) * half * inv_h
        elif j == Ngy - 1:
            dudy = (type(inv_h)(3.0) * u_cc_j0 - type(inv_h)(4.0) * u_cc_jm1 + u_cc_jm2) * half * inv_h
        else:
            dudy = (u_cc_jp1 - u_cc_jm1) * half * inv_h
    else:
        dudy = (u_cc_jp1 - u_cc_jm1) * half * inv_h

    # dudz
    u_cc_km2 = half * (u_prev[(i * Ngy + j) * Ngz + km2] + u_prev[(ip1 * Ngy + j) * Ngz + km2])
    u_cc_km1 = half * (u_prev[(i * Ngy + j) * Ngz + km1] + u_prev[(ip1 * Ngy + j) * Ngz + km1])
    u_cc_k0 = half * (u_prev[(i * Ngy + j) * Ngz + k] + u_prev[(ip1 * Ngy + j) * Ngz + k])
    u_cc_kp1 = half * (u_prev[(i * Ngy + j) * Ngz + kp1] + u_prev[(ip1 * Ngy + j) * Ngz + kp1])
    u_cc_kp2 = half * (u_prev[(i * Ngy + j) * Ngz + kp2] + u_prev[(ip1 * Ngy + j) * Ngz + kp2])
    if Ngz >= 3:
        if k == 0:
            dudz = (type(inv_h)(-3.0) * u_cc_k0 + type(inv_h)(4.0) * u_cc_kp1 - u_cc_kp2) * half * inv_h
        elif k == Ngz - 1:
            dudz = (type(inv_h)(3.0) * u_cc_k0 - type(inv_h)(4.0) * u_cc_km1 + u_cc_km2) * half * inv_h
        else:
            dudz = (u_cc_kp1 - u_cc_km1) * half * inv_h
    else:
        dudz = (u_cc_kp1 - u_cc_km1) * half * inv_h

    # dvdx (v CC = 0.5*(v[i,j,k]+v[i,j+1,k]))
    v_cc_im2 = half * (v_prev[(im2 * Ngy + j) * Ngz + k] + v_prev[(im2 * Ngy + jp1) * Ngz + k])
    v_cc_im1 = half * (v_prev[(im1 * Ngy + j) * Ngz + k] + v_prev[(im1 * Ngy + jp1) * Ngz + k])
    v_cc_i0 = half * (v_prev[(i * Ngy + j) * Ngz + k] + v_prev[(i * Ngy + jp1) * Ngz + k])
    v_cc_ip1 = half * (v_prev[(ip1 * Ngy + j) * Ngz + k] + v_prev[(ip1 * Ngy + jp1) * Ngz + k])
    v_cc_ip2 = half * (v_prev[(ip2 * Ngy + j) * Ngz + k] + v_prev[(ip2 * Ngy + jp1) * Ngz + k])
    if Ngx >= 3:
        if i == 0:
            dvdx = (type(inv_h)(-3.0) * v_cc_i0 + type(inv_h)(4.0) * v_cc_ip1 - v_cc_ip2) * half * inv_h
        elif i == Ngx - 1:
            dvdx = (type(inv_h)(3.0) * v_cc_i0 - type(inv_h)(4.0) * v_cc_im1 + v_cc_im2) * half * inv_h
        else:
            dvdx = (v_cc_ip1 - v_cc_im1) * half * inv_h
    else:
        dvdx = (v_cc_ip1 - v_cc_im1) * half * inv_h

    # dvdz
    v_cc_km2 = half * (v_prev[(i * Ngy + j) * Ngz + km2] + v_prev[(i * Ngy + jp1) * Ngz + km2])
    v_cc_km1 = half * (v_prev[(i * Ngy + j) * Ngz + km1] + v_prev[(i * Ngy + jp1) * Ngz + km1])
    v_cc_k0 = half * (v_prev[(i * Ngy + j) * Ngz + k] + v_prev[(i * Ngy + jp1) * Ngz + k])
    v_cc_kp1 = half * (v_prev[(i * Ngy + j) * Ngz + kp1] + v_prev[(i * Ngy + jp1) * Ngz + kp1])
    v_cc_kp2 = half * (v_prev[(i * Ngy + j) * Ngz + kp2] + v_prev[(i * Ngy + jp1) * Ngz + kp2])
    if Ngz >= 3:
        if k == 0:
            dvdz = (type(inv_h)(-3.0) * v_cc_k0 + type(inv_h)(4.0) * v_cc_kp1 - v_cc_kp2) * half * inv_h
        elif k == Ngz - 1:
            dvdz = (type(inv_h)(3.0) * v_cc_k0 - type(inv_h)(4.0) * v_cc_km1 + v_cc_km2) * half * inv_h
        else:
            dvdz = (v_cc_kp1 - v_cc_km1) * half * inv_h
    else:
        dvdz = (v_cc_kp1 - v_cc_km1) * half * inv_h

    # dwdx (w CC = 0.5*(w[i,j,k]+w[i,j,k+1]))
    w_cc_im2 = half * (w_prev[(im2 * Ngy + j) * Ngz + k] + w_prev[(im2 * Ngy + j) * Ngz + kp1])
    w_cc_im1 = half * (w_prev[(im1 * Ngy + j) * Ngz + k] + w_prev[(im1 * Ngy + j) * Ngz + kp1])
    w_cc_i0 = half * (w_prev[(i * Ngy + j) * Ngz + k] + w_prev[(i * Ngy + j) * Ngz + kp1])
    w_cc_ip1 = half * (w_prev[(ip1 * Ngy + j) * Ngz + k] + w_prev[(ip1 * Ngy + j) * Ngz + kp1])
    w_cc_ip2 = half * (w_prev[(ip2 * Ngy + j) * Ngz + k] + w_prev[(ip2 * Ngy + j) * Ngz + kp1])
    if Ngx >= 3:
        if i == 0:
            dwdx = (type(inv_h)(-3.0) * w_cc_i0 + type(inv_h)(4.0) * w_cc_ip1 - w_cc_ip2) * half * inv_h
        elif i == Ngx - 1:
            dwdx = (type(inv_h)(3.0) * w_cc_i0 - type(inv_h)(4.0) * w_cc_im1 + w_cc_im2) * half * inv_h
        else:
            dwdx = (w_cc_ip1 - w_cc_im1) * half * inv_h
    else:
        dwdx = (w_cc_ip1 - w_cc_im1) * half * inv_h

    # dwdy
    w_cc_jm2 = half * (w_prev[(i * Ngy + jm2) * Ngz + k] + w_prev[(i * Ngy + jm2) * Ngz + kp1])
    w_cc_jm1 = half * (w_prev[(i * Ngy + jm1) * Ngz + k] + w_prev[(i * Ngy + jm1) * Ngz + kp1])
    w_cc_j0 = half * (w_prev[(i * Ngy + j) * Ngz + k] + w_prev[(i * Ngy + j) * Ngz + kp1])
    w_cc_jp1 = half * (w_prev[(i * Ngy + jp1) * Ngz + k] + w_prev[(i * Ngy + jp1) * Ngz + kp1])
    w_cc_jp2 = half * (w_prev[(i * Ngy + jp2) * Ngz + k] + w_prev[(i * Ngy + jp2) * Ngz + kp1])
    if Ngy >= 3:
        if j == 0:
            dwdy = (type(inv_h)(-3.0) * w_cc_j0 + type(inv_h)(4.0) * w_cc_jp1 - w_cc_jp2) * half * inv_h
        elif j == Ngy - 1:
            dwdy = (type(inv_h)(3.0) * w_cc_j0 - type(inv_h)(4.0) * w_cc_jm1 + w_cc_jm2) * half * inv_h
        else:
            dwdy = (w_cc_jp1 - w_cc_jm1) * half * inv_h
    else:
        dwdy = (w_cc_jp1 - w_cc_jm1) * half * inv_h

    xs = nu_rho_val * (two * dudx * nx + (dudy + dvdx) * ny + (dudz + dwdx) * nz)
    ys = nu_rho_val * ((dvdx + dudy) * nx + two * dvdy * ny + (dvdz + dwdy) * nz)
    zs = nu_rho_val * ((dwdx + dudz) * nx + (dwdy + dvdz) * ny + two * dwdz * nz)

    p_c = p_prev[g_idx]
    pxv = -p_c * nx
    pyv = -p_c * ny
    pzv = -p_c * nz

    inv_2eps = half / eps_body
    pi_ov_eb = type(inv_h)(_PI) / eps_body
    delta_visc = zero
    delta_pres = zero
    d_visc = s_cc_body - eps_solver
    if d_visc > -eps_body and d_visc < eps_body:
        delta_visc = (one + wp.cos(pi_ov_eb * d_visc)) * inv_2eps
    if s_cc_body > -eps_body and s_cc_body < eps_body:
        delta_pres = (one + wp.cos(pi_ov_eb * s_cc_body)) * inv_2eps
    if with_pressure == 0:
        delta_pres = zero

    if delta_order == 2 and (delta_visc > zero or delta_pres > zero):
        h_grid = one / inv_h
        s_xp = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                  idx_, idy_, idz_, bxq + r00 * h_grid, byq + r10 * h_grid, bzq + r20 * h_grid)
        s_xm = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                  idx_, idy_, idz_, bxq - r00 * h_grid, byq - r10 * h_grid, bzq - r20 * h_grid)
        s_yp = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                  idx_, idy_, idz_, bxq + r01 * h_grid, byq + r11 * h_grid, bzq + r21 * h_grid)
        s_ym = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                  idx_, idy_, idz_, bxq - r01 * h_grid, byq - r11 * h_grid, bzq - r21 * h_grid)
        s_zp = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                  idx_, idy_, idz_, bxq + r02 * h_grid, byq + r12 * h_grid, bzq + r22 * h_grid)
        s_zm = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                  idx_, idy_, idz_, bxq - r02 * h_grid, byq - r12 * h_grid, bzq - r22 * h_grid)
        dxx = s_xp - s_xm
        dyy = s_yp - s_ym
        dzz = s_zp - s_zm
        grad_mag = wp.sqrt((dxx * dxx + dyy * dyy + dzz * dzz) * type(inv_h)(0.25) * inv_h * inv_h)
        min_grad = type(inv_h)(1e-3)
        if grad_mag < min_grad:
            grad_mag = min_grad
        inv_grad = one / grad_mag
        delta_visc = delta_visc * inv_grad
        delta_pres = delta_pres * inv_grad

    ax = wp.float64(xc - cm_x)
    ay = wp.float64(yc - cm_y)
    az = wp.float64(zc - cm_z)
    fv_x = wp.float64(xs * delta_visc)
    fv_y = wp.float64(ys * delta_visc)
    fv_z = wp.float64(zs * delta_visc)
    fp_x = wp.float64(pxv * delta_pres)
    fp_y = wp.float64(pyv * delta_pres)
    fp_z = wp.float64(pzv * delta_pres)
    h3_d = wp.float64(h3)

    o = b * 12
    wp.atomic_add(out, o + 0, fv_x * h3_d)
    wp.atomic_add(out, o + 1, fv_y * h3_d)
    wp.atomic_add(out, o + 2, fv_z * h3_d)
    wp.atomic_add(out, o + 3, (ay * fv_z - az * fv_y) * h3_d)
    wp.atomic_add(out, o + 4, (az * fv_x - ax * fv_z) * h3_d)
    wp.atomic_add(out, o + 5, (ax * fv_y - ay * fv_x) * h3_d)
    wp.atomic_add(out, o + 6, fp_x * h3_d)
    wp.atomic_add(out, o + 7, fp_y * h3_d)
    wp.atomic_add(out, o + 8, fp_z * h3_d)
    wp.atomic_add(out, o + 9, (ay * fp_z - az * fp_y) * h3_d)
    wp.atomic_add(out, o + 10, (az * fp_x - ax * fp_z) * h3_d)
    wp.atomic_add(out, o + 11, (ax * fp_y - ay * fp_x) * h3_d)


# ── 3-D deltaH (∂H) pressure second pass ─────────────────────────────────────

@wp.func
def _heaviside_smooth_3d(phi: Any, inv_eps: Any):
    pi = type(phi)(_PI)
    one = type(phi)(1.0)
    half = type(phi)(0.5)
    x = phi * inv_eps
    x = wp.clamp(x, -one, one)
    return half * (one + x + wp.sin(pi * x) / pi)


@wp.kernel
def forces_post_deltaH_pressure_3d_kernel(
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx:          wp.array(dtype=Any),
    gy:          wp.array(dtype=Any),
    gz:          wp.array(dtype=Any),
    Ngx: wp.int32, Ngy: wp.int32, Ngz: wp.int32,
    sdf_cc:      wp.array(dtype=Any),
    interp_method: wp.int32,
    p_prev:      wp.array(dtype=Any),
    inv_h: Any, inv_eps: Any, inv_tau: Any, h3: Any,
    B: wp.int32,
    uli0: wp.int32, ulj0: wp.int32, ulk0: wp.int32,
    ULi: wp.int32, ULj: wp.int32, ULk: wp.int32,
    out: wp.array(dtype=wp.float64),
):
    localt = wp.tid()
    uvol = ULi * ULj * ULk
    if localt >= uvol:
        return
    di = localt // (ULj * ULk)
    rem = localt - di * (ULj * ULk)
    dj = rem // ULk
    dk = rem - dj * ULk
    i = uli0 + di
    j = ulj0 + dj
    k = ulk0 + dk
    if i < 0 or i >= Ngx or j < 0 or j >= Ngy or k < 0 or k >= Ngz:
        return

    zero = type(inv_h)(0.0)
    one = type(inv_h)(1.0)
    half = type(inv_h)(0.5)

    gHx = zero
    if Ngx >= 3:
        if i == 0:
            gHx = (type(inv_h)(-3.0) * _heaviside_smooth_3d(sdf_cc[(0 * Ngy + j) * Ngz + k], inv_eps)
                   + type(inv_h)(4.0) * _heaviside_smooth_3d(sdf_cc[(1 * Ngy + j) * Ngz + k], inv_eps)
                   - _heaviside_smooth_3d(sdf_cc[(2 * Ngy + j) * Ngz + k], inv_eps)) * half * inv_h
        elif i == Ngx - 1:
            gHx = (type(inv_h)(3.0) * _heaviside_smooth_3d(sdf_cc[((Ngx - 1) * Ngy + j) * Ngz + k], inv_eps)
                   - type(inv_h)(4.0) * _heaviside_smooth_3d(sdf_cc[((Ngx - 2) * Ngy + j) * Ngz + k], inv_eps)
                   + _heaviside_smooth_3d(sdf_cc[((Ngx - 3) * Ngy + j) * Ngz + k], inv_eps)) * half * inv_h
        else:
            gHx = (_heaviside_smooth_3d(sdf_cc[((i + 1) * Ngy + j) * Ngz + k], inv_eps)
                   - _heaviside_smooth_3d(sdf_cc[((i - 1) * Ngy + j) * Ngz + k], inv_eps)) * half * inv_h
    elif Ngx == 2:
        gHx = (_heaviside_smooth_3d(sdf_cc[(1 * Ngy + j) * Ngz + k], inv_eps)
               - _heaviside_smooth_3d(sdf_cc[(0 * Ngy + j) * Ngz + k], inv_eps)) * inv_h

    gHy = zero
    if Ngy >= 3:
        if j == 0:
            gHy = (type(inv_h)(-3.0) * _heaviside_smooth_3d(sdf_cc[(i * Ngy + 0) * Ngz + k], inv_eps)
                   + type(inv_h)(4.0) * _heaviside_smooth_3d(sdf_cc[(i * Ngy + 1) * Ngz + k], inv_eps)
                   - _heaviside_smooth_3d(sdf_cc[(i * Ngy + 2) * Ngz + k], inv_eps)) * half * inv_h
        elif j == Ngy - 1:
            gHy = (type(inv_h)(3.0) * _heaviside_smooth_3d(sdf_cc[(i * Ngy + (Ngy - 1)) * Ngz + k], inv_eps)
                   - type(inv_h)(4.0) * _heaviside_smooth_3d(sdf_cc[(i * Ngy + (Ngy - 2)) * Ngz + k], inv_eps)
                   + _heaviside_smooth_3d(sdf_cc[(i * Ngy + (Ngy - 3)) * Ngz + k], inv_eps)) * half * inv_h
        else:
            gHy = (_heaviside_smooth_3d(sdf_cc[(i * Ngy + (j + 1)) * Ngz + k], inv_eps)
                   - _heaviside_smooth_3d(sdf_cc[(i * Ngy + (j - 1)) * Ngz + k], inv_eps)) * half * inv_h
    elif Ngy == 2:
        gHy = (_heaviside_smooth_3d(sdf_cc[(i * Ngy + 1) * Ngz + k], inv_eps)
               - _heaviside_smooth_3d(sdf_cc[(i * Ngy + 0) * Ngz + k], inv_eps)) * inv_h

    gHz = zero
    base_ij = (i * Ngy + j) * Ngz
    if Ngz >= 3:
        if k == 0:
            gHz = (type(inv_h)(-3.0) * _heaviside_smooth_3d(sdf_cc[base_ij + 0], inv_eps)
                   + type(inv_h)(4.0) * _heaviside_smooth_3d(sdf_cc[base_ij + 1], inv_eps)
                   - _heaviside_smooth_3d(sdf_cc[base_ij + 2], inv_eps)) * half * inv_h
        elif k == Ngz - 1:
            gHz = (type(inv_h)(3.0) * _heaviside_smooth_3d(sdf_cc[base_ij + (Ngz - 1)], inv_eps)
                   - type(inv_h)(4.0) * _heaviside_smooth_3d(sdf_cc[base_ij + (Ngz - 2)], inv_eps)
                   + _heaviside_smooth_3d(sdf_cc[base_ij + (Ngz - 3)], inv_eps)) * half * inv_h
        else:
            gHz = (_heaviside_smooth_3d(sdf_cc[base_ij + (k + 1)], inv_eps)
                   - _heaviside_smooth_3d(sdf_cc[base_ij + (k - 1)], inv_eps)) * half * inv_h
    elif Ngz == 2:
        gHz = (_heaviside_smooth_3d(sdf_cc[base_ij + 1], inv_eps)
               - _heaviside_smooth_3d(sdf_cc[base_ij + 0], inv_eps)) * inv_h

    if gHx == zero and gHy == zero and gHz == zero:
        return

    g = (i * Ngy + j) * Ngz + k
    p_c = p_prev[g]
    fdx = -p_c * gHx
    fdy = -p_c * gHy
    fdz = -p_c * gHz
    xc = gx[i]
    yc = gy[j]
    zc = gz[k]
    sdfu = sdf_cc[g]

    Z = zero
    for b in range(B):
        i0 = wp.int32(aabb_lo[b * 3 + 0])
        j0 = wp.int32(aabb_lo[b * 3 + 1])
        k0 = wp.int32(aabb_lo[b * 3 + 2])
        Ai = wp.int32(aabb_dim[b * 3 + 0])
        Aj = wp.int32(aabb_dim[b * 3 + 1])
        Ak = wp.int32(aabb_dim[b * 3 + 2])
        if (i >= i0 and i < i0 + Ai and j >= j0 and j < j0 + Aj
                and k >= k0 and k < k0 + Ak):
            F_off = wp.int32(F_offsets[b])
            Mx = wp.int32(body_shapes[b * 3 + 0])
            My = wp.int32(body_shapes[b * 3 + 1])
            Mz = wp.int32(body_shapes[b * 3 + 2])
            dx_w = xc - kin[b * 21 + 9]
            dy_w = yc - kin[b * 21 + 10]
            dz_w = zc - kin[b * 21 + 11]
            bxq = kin[b * 21 + 0] * dx_w + kin[b * 21 + 1] * dy_w + kin[b * 21 + 2] * dz_w
            byq = kin[b * 21 + 3] * dx_w + kin[b * 21 + 4] * dy_w + kin[b * 21 + 5] * dz_w
            bzq = kin[b * 21 + 6] * dx_w + kin[b * 21 + 7] * dy_w + kin[b * 21 + 8] * dz_w
            s_b = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz,
                                     body_meta[b * 10 + 0], body_meta[b * 10 + 1],
                                     body_meta[b * 10 + 2], body_meta[b * 10 + 6],
                                     body_meta[b * 10 + 7], body_meta[b * 10 + 8],
                                     bxq, byq, bzq)
            Z += wp.exp(-(s_b - sdfu) * inv_tau)
    if Z <= zero:
        return
    inv_Z = one / Z
    h3_d = wp.float64(h3)

    for b in range(B):
        i0 = wp.int32(aabb_lo[b * 3 + 0])
        j0 = wp.int32(aabb_lo[b * 3 + 1])
        k0 = wp.int32(aabb_lo[b * 3 + 2])
        Ai = wp.int32(aabb_dim[b * 3 + 0])
        Aj = wp.int32(aabb_dim[b * 3 + 1])
        Ak = wp.int32(aabb_dim[b * 3 + 2])
        if (i >= i0 and i < i0 + Ai and j >= j0 and j < j0 + Aj
                and k >= k0 and k < k0 + Ak):
            F_off = wp.int32(F_offsets[b])
            Mx = wp.int32(body_shapes[b * 3 + 0])
            My = wp.int32(body_shapes[b * 3 + 1])
            Mz = wp.int32(body_shapes[b * 3 + 2])
            dx_w = xc - kin[b * 21 + 9]
            dy_w = yc - kin[b * 21 + 10]
            dz_w = zc - kin[b * 21 + 11]
            bxq = kin[b * 21 + 0] * dx_w + kin[b * 21 + 1] * dy_w + kin[b * 21 + 2] * dz_w
            byq = kin[b * 21 + 3] * dx_w + kin[b * 21 + 4] * dy_w + kin[b * 21 + 5] * dz_w
            bzq = kin[b * 21 + 6] * dx_w + kin[b * 21 + 7] * dy_w + kin[b * 21 + 8] * dz_w
            s_b = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz,
                                     body_meta[b * 10 + 0], body_meta[b * 10 + 1],
                                     body_meta[b * 10 + 2], body_meta[b * 10 + 6],
                                     body_meta[b * 10 + 7], body_meta[b * 10 + 8],
                                     bxq, byq, bzq)
            wb = wp.exp(-(s_b - sdfu) * inv_tau) * inv_Z
            fbx = wb * fdx
            fby = wb * fdy
            fbz = wb * fdz
            ax = wp.float64(xc - kin[b * 21 + 12])
            ay = wp.float64(yc - kin[b * 21 + 13])
            az = wp.float64(zc - kin[b * 21 + 14])
            fbxd = wp.float64(fbx)
            fbyd = wp.float64(fby)
            fbzd = wp.float64(fbz)
            o = b * 12
            wp.atomic_add(out, o + 6, fbxd * h3_d)
            wp.atomic_add(out, o + 7, fbyd * h3_d)
            wp.atomic_add(out, o + 8, fbzd * h3_d)
            wp.atomic_add(out, o + 9, (ay * fbzd - az * fbyd) * h3_d)
            wp.atomic_add(out, o + 10, (az * fbxd - ax * fbzd) * h3_d)
            wp.atomic_add(out, o + 11, (ax * fbyd - ay * fbxd) * h3_d)


# Register float32 + float64 specialisations for the 3-D kernels.
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(forces_post_3d_kernel, {
        "F_flat": _A, "body_meta": _A, "kin": _A, "gx": _A, "gy": _A, "gz": _A,
        "sdf_cc": _A, "u_prev": _A, "v_prev": _A, "w_prev": _A, "p_prev": _A,
        "nu_rho_field": _A,
        "inv_h": _dt, "eps_body": _dt, "eps_solver": _dt, "h3": _dt,
    })
    wp.overload(forces_post_deltaH_pressure_3d_kernel, {
        "F_flat": _A, "body_meta": _A, "kin": _A, "gx": _A, "gy": _A, "gz": _A,
        "sdf_cc": _A, "p_prev": _A,
        "inv_h": _dt, "inv_eps": _dt, "inv_tau": _dt, "h3": _dt,
    })


# ═════════════════════════════════════════════════════════════════════════════
#  Host wrappers (mirror the native ops.py signatures)
# ═════════════════════════════════════════════════════════════════════════════

def _wp_dtype(t):
    return wp.float64 if t.dtype == torch.float64 else wp.float32


def _torch_dtype(t):
    return torch.float64 if t.dtype == torch.float64 else torch.float32


def _fast_flat(x, wpf, tdt, wdev):
    """Low-overhead zero-copy flat Warp view (~2× cheaper than ``wp.from_torch``
    on the common contiguous/right-dtype path; safe fallback otherwise)."""
    if x.dtype == tdt and x.is_contiguous():
        return wp.array(ptr=x.data_ptr(), dtype=wpf, shape=(x.numel(),),
                        device=wdev)
    return wp.from_torch(x.reshape(-1).contiguous().to(tdt))


def _fast_flat_i64(x, wdev):
    """Low-overhead zero-copy flat int64 Warp view."""
    if x.dtype == torch.int64 and x.is_contiguous():
        return wp.array(ptr=x.data_ptr(), dtype=wp.int64, shape=(x.numel(),),
                        device=wdev)
    return wp.from_torch(x.reshape(-1).contiguous().to(torch.int64))


def streaming_sdf_forces_post_2d_warp(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim, gx, gy,
        h_grid, max_vol_per_body, sdf_cc, interp_method,
        u_prev, v_prev, p_prev, nu_rho_field,
        eps_body, eps_solver, h2, delta_order,
        out, force_submethod=0, ph_tau=0.0):
    """Warp port of ``streaming_sdf_forces_post_2d``.  Writes ``out`` (B*6 or
    (B,6) float64) in place — the native wrapper does NOT zero ``out`` (the
    caller's persistent buffer is pre-zeroed each step), so neither do we."""
    B = int(aabb_dim.shape[0])
    if B <= 0 or int(max_vol_per_body) <= 0:
        return
    wdev = "cuda:0" if F_flat.device.type == "cuda" else "cpu"
    wpf = _wp_dtype(F_flat)
    tdt = _torch_dtype(F_flat)
    Ngx = int(gx.numel())
    Ngy = int(gy.numel())
    with_pressure = 1 if int(force_submethod) == 0 else 0

    def f(x):
        return _fast_flat(x, wpf, tdt, wdev)

    def fi(x):
        return _fast_flat_i64(x, wdev)

    out_w = wp.from_torch(out.reshape(-1))
    wp.launch(
        forces_post_2d_kernel, dim=B * int(max_vol_per_body),
        inputs=[
            f(F_flat), fi(F_offsets), fi(body_shapes), f(body_meta), f(kin),
            fi(aabb_lo), fi(aabb_dim), f(gx), f(gy),
            Ngx, Ngy, f(sdf_cc), int(interp_method),
            f(u_prev), f(v_prev), f(p_prev), f(nu_rho_field),
            int(nu_rho_field.numel()),
            wpf(1.0 / h_grid), wpf(eps_body), wpf(eps_solver), wpf(h2),
            int(delta_order), int(with_pressure), int(max_vol_per_body),
            out_w,
        ],
        device=wdev)

    if int(force_submethod) != 0:
        lo = aabb_lo.to("cpu").to(torch.int64)
        dim = aabb_dim.to("cpu").to(torch.int64)
        ulo = [Ngx, Ngy]
        uhi = [0, 0]
        for b in range(B):
            for d in range(2):
                a0 = int(lo[b, d])
                a1 = a0 + int(dim[b, d])
                ulo[d] = min(ulo[d], a0)
                uhi[d] = max(uhi[d], a1)
        Ng = [Ngx, Ngy]
        halo = 2
        for d in range(2):
            ulo[d] = max(ulo[d] - halo, 0)
            uhi[d] = min(uhi[d] + halo, Ng[d])
        ULi = uhi[0] - ulo[0]
        ULj = uhi[1] - ulo[1]
        if ULi > 0 and ULj > 0:
            tau = ph_tau if ph_tau > 0.0 else 1e-9
            wp.launch(
                forces_post_deltaH_pressure_2d_kernel, dim=ULi * ULj,
                inputs=[
                    f(F_flat), fi(F_offsets), fi(body_shapes), f(body_meta),
                    f(kin), fi(aabb_lo), fi(aabb_dim), f(gx), f(gy),
                    Ngx, Ngy, f(sdf_cc), int(interp_method), f(p_prev),
                    wpf(1.0 / h_grid), wpf(1.0 / eps_body), wpf(1.0 / tau),
                    wpf(h2), int(B),
                    int(ulo[0]), int(ulo[1]), int(ULi), int(ULj),
                    out_w,
                ],
                device=wdev)
    # No wp.synchronize(): the caller reads ``out`` through torch, which orders
    # after the Warp launch on the (legacy null) default stream — the explicit
    # full-device sync was a per-call latency floor, not a correctness need.


def streaming_sdf_forces_post_3d_warp(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim, gx, gy, gz,
        h_grid, max_vol_per_body, sdf_cc, interp_method,
        u_prev, v_prev, w_prev, p_prev, nu_rho_field,
        eps_body, eps_solver, h3, delta_order,
        out, force_submethod=0, ph_tau=0.0):
    """Warp port of ``streaming_sdf_forces_post_3d``.  Accumulates into ``out``
    (B*12 or (B,12) float64) in place (native does not zero it)."""
    B = int(aabb_dim.shape[0])
    if B <= 0 or int(max_vol_per_body) <= 0:
        return
    wdev = "cuda:0" if F_flat.device.type == "cuda" else "cpu"
    wpf = _wp_dtype(F_flat)
    tdt = _torch_dtype(F_flat)
    Ngx = int(gx.numel())
    Ngy = int(gy.numel())
    Ngz = int(gz.numel())
    with_pressure = 1 if int(force_submethod) == 0 else 0

    def f(x):
        return _fast_flat(x, wpf, tdt, wdev)

    def fi(x):
        return _fast_flat_i64(x, wdev)

    out_w = wp.from_torch(out.reshape(-1))
    wp.launch(
        forces_post_3d_kernel, dim=B * int(max_vol_per_body),
        inputs=[
            f(F_flat), fi(F_offsets), fi(body_shapes), f(body_meta), f(kin),
            fi(aabb_lo), fi(aabb_dim), f(gx), f(gy), f(gz),
            Ngx, Ngy, Ngz, f(sdf_cc), int(interp_method),
            f(u_prev), f(v_prev), f(w_prev), f(p_prev), f(nu_rho_field),
            int(nu_rho_field.numel()),
            wpf(1.0 / h_grid), wpf(eps_body), wpf(eps_solver), wpf(h3),
            int(delta_order), int(with_pressure), int(max_vol_per_body),
            out_w,
        ],
        device=wdev)

    if int(force_submethod) != 0:
        lo = aabb_lo.to("cpu").to(torch.int64)
        dim = aabb_dim.to("cpu").to(torch.int64)
        ulo = [Ngx, Ngy, Ngz]
        uhi = [0, 0, 0]
        for b in range(B):
            for d in range(3):
                a0 = int(lo[b, d])
                a1 = a0 + int(dim[b, d])
                ulo[d] = min(ulo[d], a0)
                uhi[d] = max(uhi[d], a1)
        Ng = [Ngx, Ngy, Ngz]
        halo = 2
        for d in range(3):
            ulo[d] = max(ulo[d] - halo, 0)
            uhi[d] = min(uhi[d] + halo, Ng[d])
        ULi = uhi[0] - ulo[0]
        ULj = uhi[1] - ulo[1]
        ULk = uhi[2] - ulo[2]
        if ULi > 0 and ULj > 0 and ULk > 0:
            tau = ph_tau if ph_tau > 0.0 else 1e-9
            wp.launch(
                forces_post_deltaH_pressure_3d_kernel, dim=ULi * ULj * ULk,
                inputs=[
                    f(F_flat), fi(F_offsets), fi(body_shapes), f(body_meta),
                    f(kin), fi(aabb_lo), fi(aabb_dim), f(gx), f(gy), f(gz),
                    Ngx, Ngy, Ngz, f(sdf_cc), int(interp_method), f(p_prev),
                    wpf(1.0 / h_grid), wpf(1.0 / eps_body), wpf(1.0 / tau),
                    wpf(h3), int(B),
                    int(ulo[0]), int(ulo[1]), int(ulo[2]),
                    int(ULi), int(ULj), int(ULk),
                    out_w,
                ],
                device=wdev)
    # No wp.synchronize(): the caller reads ``out`` through torch, which orders
    # after the Warp launch on the (legacy null) default stream — the explicit
    # full-device sync was a per-call latency floor, not a correctness need.
