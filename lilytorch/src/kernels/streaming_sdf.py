"""Warp single-source CPU+GPU kernels for BDIM streaming SDF (AP7) — 3-D.

Sequential per-body design
──────────────────────────
Each body gets its own kernel launch (B launches per step).  Within a single
body's launch, every thread processes a unique AABB cell (global index g is
unique per-thread), so the compare-swap on sdf + body_vel is race-free without
atomics.  CUDA-graph capture collapses the B launches to a single host call.

This sidesteps the need for the 64-bit packed-key atomicMin trick used by the
native kernel (which required the non-portable `__float_as_uint`/`atomicCAS`
not available in Warp's built-in set).

The fanned all-body path additionally honours the smooth velocity-blend
(`blend_eps > 0`): accumulate Σ w_i v_i and Σ w_i with
w_i = sigmoid(-s_i/blend_eps) via `wp.atomic_add`, then the decode divides by
Σ w_i — bit-mirroring the native num_*/den_* fields (matches the 2-D file).

**Precision (single source, both dtypes).**  Every value-carrying array and
float scalar is a Warp *generic* (``Any``): float literals are materialised in
the bound element type via ``type(x)(literal)``.  ``wp.overload`` registers the
``float32`` *and* ``float64`` specialisations up front (Warp 1.14 does not
reliably re-specialise a generic kernel implicitly across dtypes in one
process).  ``float32`` codegen is unchanged from the original concrete kernels,
so the existing parity tests stay bit-identical; ``float64`` is what an f64
solver uses.

All @wp.func / @wp.kernel are at module level (Warp codegen requirement).
"""
from __future__ import annotations

from typing import Any, Optional

import warp as wp
import torch

wp.init()

# ─────────────────────────────────────────────────────────────────────────────
#  Trilinear interpolation on a uniform body SDF grid (with flat-array offset)
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
#  Per-body streaming SDF kernel (sequential design — no blend, matches 2-D)
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def streaming_sdf_one_body_3d(
    # All-body packed arrays (static, loaded once)
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),    # [B]
    body_shapes: wp.array(dtype=wp.int64),    # [B*3] flat
    body_meta:   wp.array(dtype=Any),         # [B*10] flat
    # Per-step packed arrays (updated before each step / graph replay)
    kin:         wp.array(dtype=Any),         # [B*21] flat
    aabb_lo:     wp.array(dtype=wp.int64),    # [B*3] flat
    aabb_dim:    wp.array(dtype=wp.int64),    # [B*3] flat
    # This body's index (compile-time constant per launch in graph)
    b:  int,
    # Fluid grid
    gx: wp.array(dtype=Any),
    gy: wp.array(dtype=Any),
    gz: wp.array(dtype=Any),
    half_h:  Any,
    max_vol: int,
    Ngy: int, Ngz: int,
    # Outputs (full-grid flat)
    sdf_cc: wp.array(dtype=Any),
    sdf_u:  wp.array(dtype=Any),
    sdf_v:  wp.array(dtype=Any),
    sdf_w:  wp.array(dtype=Any),
    body_u: wp.array(dtype=Any),
    body_v: wp.array(dtype=Any),
    body_w: wp.array(dtype=Any),
):
    """Process one body's AABB: sample SDF at cc+3 face positions; compare-swap.

    Launch with dim=max_vol.  Threads past the body's actual volume return early.
    Thread local-id maps 1-to-1 to AABB cells (unique g per thread → no races).
    """
    local = wp.tid()
    Ai = int(aabb_dim[b * 3 + 0])
    Aj = int(aabb_dim[b * 3 + 1])
    Ak = int(aabb_dim[b * 3 + 2])
    vol = Ai * Aj * Ak
    if local >= vol:
        return

    # AABB-local → global (i, j, k)
    di  = local // (Aj * Ak)
    rem = local - di * (Aj * Ak)
    dj  = rem // Ak
    dk  = rem - dj * Ak
    i   = int(aabb_lo[b * 3 + 0]) + di
    j   = int(aabb_lo[b * 3 + 1]) + dj
    k   = int(aabb_lo[b * 3 + 2]) + dk
    g   = i * Ngy * Ngz + j * Ngz + k

    # Body SDF table params
    F_off  = int(F_offsets[b])
    Mx     = int(body_shapes[b * 3 + 0])
    My     = int(body_shapes[b * 3 + 1])
    Mz     = int(body_shapes[b * 3 + 2])
    bx0    = body_meta[b * 10 + 0]
    by0    = body_meta[b * 10 + 1]
    bz0    = body_meta[b * 10 + 2]
    inv_dx = body_meta[b * 10 + 6]
    inv_dy = body_meta[b * 10 + 7]
    inv_dz = body_meta[b * 10 + 8]

    # Kinematics: R_T (row-major 3×3), body position, CoM, lin/ang vel
    K0   = b * 21
    r00  = kin[K0 + 0]; r01 = kin[K0 + 1]; r02 = kin[K0 + 2]
    r10  = kin[K0 + 3]; r11 = kin[K0 + 4]; r12 = kin[K0 + 5]
    r20  = kin[K0 + 6]; r21 = kin[K0 + 7]; r22 = kin[K0 + 8]
    bp_x = kin[K0 + 9]; bp_y = kin[K0 + 10]; bp_z = kin[K0 + 11]
    cm_x = kin[K0 + 12]; cm_y = kin[K0 + 13]; cm_z = kin[K0 + 14]
    lv_x = kin[K0 + 15]; lv_y = kin[K0 + 16]; lv_z = kin[K0 + 17]
    av_x = kin[K0 + 18]; av_y = kin[K0 + 19]; av_z = kin[K0 + 20]

    # World cc position → body-local frame
    xc = gx[i]; yc = gy[j]; zc = gz[k]
    dx = xc - bp_x; dy = yc - bp_y; dz = zc - bp_z
    bxq = r00*dx + r01*dy + r02*dz
    byq = r10*dx + r11*dy + r12*dz
    bzq = r20*dx + r21*dy + r22*dz

    # Pre-rotate face offsets into body frame
    neg_hh = -half_h
    du_x = neg_hh*r00; du_y = neg_hh*r10; du_z = neg_hh*r20
    dv_x = neg_hh*r01; dv_y = neg_hh*r11; dv_z = neg_hh*r21
    dw_x = neg_hh*r02; dw_y = neg_hh*r12; dw_z = neg_hh*r22

    # cc: atomic_min (cc has no associated body velocity in this kernel)
    s_cc = trilinear_sample_off(F_flat, F_off, Mx, My, Mz,
                                 bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
                                 bxq, byq, bzq)
    wp.atomic_min(sdf_cc, g, s_cc)

    # u-face: conditional compare-swap (no race: g unique within this body's kernel)
    s_u = trilinear_sample_off(F_flat, F_off, Mx, My, Mz,
                                bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
                                bxq + du_x, byq + du_y, bzq + du_z)
    if s_u < sdf_u[g]:
        sdf_u[g] = s_u
        body_u[g] = lv_x + av_y * (zc - cm_z) - av_z * (yc - cm_y)

    # v-face
    s_v = trilinear_sample_off(F_flat, F_off, Mx, My, Mz,
                                bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
                                bxq + dv_x, byq + dv_y, bzq + dv_z)
    if s_v < sdf_v[g]:
        sdf_v[g] = s_v
        body_v[g] = lv_y + av_z * (xc - cm_x) - av_x * (zc - cm_z)

    # w-face
    s_w = trilinear_sample_off(F_flat, F_off, Mx, My, Mz,
                                bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
                                bxq + dw_x, byq + dw_y, bzq + dw_z)
    if s_w < sdf_w[g]:
        sdf_w[g] = s_w
        body_w[g] = lv_z + av_x * (yc - cm_y) - av_y * (xc - cm_x)


# ─────────────────────────────────────────────────────────────────────────────
#  Fanned all-bodies kernels (constant in B — 1 launch each, matches native)
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def _fan_decode_tid(
    tid: int, max_vol: int,
    aabb_lo:  wp.array(dtype=wp.int64),
    aabb_dim: wp.array(dtype=wp.int64),
    Ngy: int, Ngz: int,
):
    """Map a fanned thread id → (valid, b, i, j, k, g). g<0 ⇒ skip."""
    b     = tid // max_vol
    local = tid - b * max_vol
    Ai = int(aabb_dim[b * 3 + 0])
    Aj = int(aabb_dim[b * 3 + 1])
    Ak = int(aabb_dim[b * 3 + 2])
    vol = Ai * Aj * Ak
    if local >= vol:
        return b, -1, -1, -1, -1
    di  = local // (Aj * Ak)
    rem = local - di * (Aj * Ak)
    dj  = rem // Ak
    dk  = rem - dj * Ak
    i   = int(aabb_lo[b * 3 + 0]) + di
    j   = int(aabb_lo[b * 3 + 1]) + dj
    k   = int(aabb_lo[b * 3 + 2]) + dk
    g   = i * Ngy * Ngz + j * Ngz + k
    return b, i, j, k, g


@wp.func
def _fan_faces(
    b: int, i: int, j: int, k: int,
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    gx: wp.array(dtype=Any),
    gy: wp.array(dtype=Any),
    gz: wp.array(dtype=Any),
    half_h: Any,
):
    """Return (s_cc, s_u, s_v, s_w) sampled SDFs for body b at cell (i,j,k)."""
    F_off  = int(F_offsets[b])
    Mx     = int(body_shapes[b * 3 + 0])
    My     = int(body_shapes[b * 3 + 1])
    Mz     = int(body_shapes[b * 3 + 2])
    bx0    = body_meta[b * 10 + 0]
    by0    = body_meta[b * 10 + 1]
    bz0    = body_meta[b * 10 + 2]
    inv_dx = body_meta[b * 10 + 6]
    inv_dy = body_meta[b * 10 + 7]
    inv_dz = body_meta[b * 10 + 8]

    K0   = b * 21
    r00  = kin[K0 + 0]; r01 = kin[K0 + 1]; r02 = kin[K0 + 2]
    r10  = kin[K0 + 3]; r11 = kin[K0 + 4]; r12 = kin[K0 + 5]
    r20  = kin[K0 + 6]; r21 = kin[K0 + 7]; r22 = kin[K0 + 8]
    bp_x = kin[K0 + 9]; bp_y = kin[K0 + 10]; bp_z = kin[K0 + 11]

    xc = gx[i]; yc = gy[j]; zc = gz[k]
    dx = xc - bp_x; dy = yc - bp_y; dz = zc - bp_z
    bxq = r00*dx + r01*dy + r02*dz
    byq = r10*dx + r11*dy + r12*dz
    bzq = r20*dx + r21*dy + r22*dz

    neg_hh = -half_h
    du_x = neg_hh*r00; du_y = neg_hh*r10; du_z = neg_hh*r20
    dv_x = neg_hh*r01; dv_y = neg_hh*r11; dv_z = neg_hh*r21
    dw_x = neg_hh*r02; dw_y = neg_hh*r12; dw_z = neg_hh*r22

    s_cc = trilinear_sample_off(F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                inv_dx, inv_dy, inv_dz, bxq, byq, bzq)
    s_u  = trilinear_sample_off(F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                inv_dx, inv_dy, inv_dz,
                                bxq + du_x, byq + du_y, bzq + du_z)
    s_v  = trilinear_sample_off(F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                inv_dx, inv_dy, inv_dz,
                                bxq + dv_x, byq + dv_y, bzq + dv_z)
    s_w  = trilinear_sample_off(F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                inv_dx, inv_dy, inv_dz,
                                bxq + dw_x, byq + dw_y, bzq + dw_z)
    return s_cc, s_u, s_v, s_w


@wp.kernel
def streaming_sdf_fanned_min_3d(
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx: wp.array(dtype=Any),
    gy: wp.array(dtype=Any),
    gz: wp.array(dtype=Any),
    half_h:  Any,
    max_vol: int,
    Ngy: int, Ngz: int,
    blend_eps: Any,
    sdf_cc: wp.array(dtype=Any),
    sdf_u:  wp.array(dtype=Any),
    sdf_v:  wp.array(dtype=Any),
    sdf_w:  wp.array(dtype=Any),
    num_u:  wp.array(dtype=Any),
    num_v:  wp.array(dtype=Any),
    num_w:  wp.array(dtype=Any),
    den_u:  wp.array(dtype=Any),
    den_v:  wp.array(dtype=Any),
    den_w:  wp.array(dtype=Any),
):
    """Pass B: fanned all-body atomic-min of cc/u/v/w SDFs (1 launch).
    Also accumulates Σ w_i v_i / Σ w_i into num/den when blend_eps>0."""
    tid = wp.tid()
    b, i, j, k, g = _fan_decode_tid(tid, max_vol, aabb_lo, aabb_dim, Ngy, Ngz)
    if g < 0:
        return
    s_cc, s_u, s_v, s_w = _fan_faces(
        b, i, j, k, F_flat, F_offsets, body_shapes, body_meta, kin,
        gx, gy, gz, half_h)
    wp.atomic_min(sdf_cc, g, s_cc)
    wp.atomic_min(sdf_u,  g, s_u)
    wp.atomic_min(sdf_v,  g, s_v)
    wp.atomic_min(sdf_w,  g, s_w)

    if blend_eps > type(blend_eps)(0.0):
        K0   = b * 21
        cm_x = kin[K0 + 12]; cm_y = kin[K0 + 13]; cm_z = kin[K0 + 14]
        lv_x = kin[K0 + 15]; lv_y = kin[K0 + 16]; lv_z = kin[K0 + 17]
        av_x = kin[K0 + 18]; av_y = kin[K0 + 19]; av_z = kin[K0 + 20]
        xc = gx[i]; yc = gy[j]; zc = gz[k]
        vU = lv_x + av_y * (zc - cm_z) - av_z * (yc - cm_y)
        vV = lv_y + av_z * (xc - cm_x) - av_x * (zc - cm_z)
        vW = lv_z + av_x * (yc - cm_y) - av_y * (xc - cm_x)
        one = type(blend_eps)(1.0)
        wU = one / (one + wp.exp(s_u / blend_eps))
        wV = one / (one + wp.exp(s_v / blend_eps))
        wW = one / (one + wp.exp(s_w / blend_eps))
        wp.atomic_add(num_u, g, wU * vU); wp.atomic_add(den_u, g, wU)
        wp.atomic_add(num_v, g, wV * vV); wp.atomic_add(den_v, g, wV)
        wp.atomic_add(num_w, g, wW * vW); wp.atomic_add(den_w, g, wW)


@wp.kernel
def streaming_sdf_fanned_decode_3d(
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx: wp.array(dtype=Any),
    gy: wp.array(dtype=Any),
    gz: wp.array(dtype=Any),
    half_h:  Any,
    max_vol: int,
    Ngy: int, Ngz: int,
    blend_eps: Any,
    sdf_u:  wp.array(dtype=Any),
    sdf_v:  wp.array(dtype=Any),
    sdf_w:  wp.array(dtype=Any),
    body_u: wp.array(dtype=Any),
    body_v: wp.array(dtype=Any),
    body_w: wp.array(dtype=Any),
    num_u:  wp.array(dtype=Any),
    num_v:  wp.array(dtype=Any),
    num_w:  wp.array(dtype=Any),
    den_u:  wp.array(dtype=Any),
    den_v:  wp.array(dtype=Any),
    den_w:  wp.array(dtype=Any),
    # BDIM-σ key emission (emit_keys == 0 → key_* are dummies, untouched).
    # See the 2-D decode kernel: write the winning body-id (lowest-id-wins via
    # int64 ``atomic_min``) into key_u/key_v/key_w; the σ bdim_forcing masks
    # ``key & 0xffffffff``.  Unlike 2-D (full-grid keys indexed by g), the
    # native 3-D keys are dirty_vol-sized and indexed by the AABB-local
    # ``g_local`` (matching ``bdim_forcing_sigma_3d``'s read) — so the dirty
    # origin / strides are passed in to recompute it.
    emit_keys: int,
    key_u: wp.array(dtype=wp.int64),
    key_v: wp.array(dtype=wp.int64),
    key_w: wp.array(dtype=wp.int64),
    di0: int, dj0: int, dk0: int, dAj: int, dAk: int,
):
    """Pass C: write the winning body's face velocity where SDF == stored min.
    With blend_eps>0, instead writes Σ w_i v_i / Σ w_i (the softmin blend)."""
    tid = wp.tid()
    b, i, j, k, g = _fan_decode_tid(tid, max_vol, aabb_lo, aabb_dim, Ngy, Ngz)
    if g < 0:
        return
    s_cc, s_u, s_v, s_w = _fan_faces(
        b, i, j, k, F_flat, F_offsets, body_shapes, body_meta, kin,
        gx, gy, gz, half_h)

    K0   = b * 21
    cm_x = kin[K0 + 12]; cm_y = kin[K0 + 13]; cm_z = kin[K0 + 14]
    lv_x = kin[K0 + 15]; lv_y = kin[K0 + 16]; lv_z = kin[K0 + 17]
    av_x = kin[K0 + 18]; av_y = kin[K0 + 19]; av_z = kin[K0 + 20]
    xc = gx[i]; yc = gy[j]; zc = gz[k]

    blend = blend_eps > type(blend_eps)(0.0)
    den_tol = type(blend_eps)(1e-6)

    if blend and den_u[g] > den_tol:
        body_u[g] = num_u[g] / den_u[g]
    elif s_u == sdf_u[g]:
        body_u[g] = lv_x + av_y * (zc - cm_z) - av_z * (yc - cm_y)

    if blend and den_v[g] > den_tol:
        body_v[g] = num_v[g] / den_v[g]
    elif s_v == sdf_v[g]:
        body_v[g] = lv_y + av_z * (xc - cm_x) - av_x * (zc - cm_z)

    if blend and den_w[g] > den_tol:
        body_w[g] = num_w[g] / den_w[g]
    elif s_w == sdf_w[g]:
        body_w[g] = lv_z + av_x * (yc - cm_y) - av_y * (xc - cm_x)

    if emit_keys != 0:
        g_local = (i - di0) * (dAj * dAk) + (j - dj0) * dAk + (k - dk0)
        if s_u == sdf_u[g]:
            wp.atomic_min(key_u, g_local, wp.int64(b))
        if s_v == sdf_v[g]:
            wp.atomic_min(key_v, g_local, wp.int64(b))
        if s_w == sdf_w[g]:
            wp.atomic_min(key_w, g_local, wp.int64(b))


# ── Register float32 + float64 specialisations up front ─────────────────────
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(streaming_sdf_one_body_3d, {
        "F_flat": _A, "body_meta": _A, "kin": _A,
        "gx": _A, "gy": _A, "gz": _A, "half_h": _dt,
        "sdf_cc": _A, "sdf_u": _A, "sdf_v": _A, "sdf_w": _A,
        "body_u": _A, "body_v": _A, "body_w": _A,
    })
    wp.overload(streaming_sdf_fanned_min_3d, {
        "F_flat": _A, "body_meta": _A, "kin": _A,
        "gx": _A, "gy": _A, "gz": _A, "half_h": _dt, "blend_eps": _dt,
        "sdf_cc": _A, "sdf_u": _A, "sdf_v": _A, "sdf_w": _A,
        "num_u": _A, "num_v": _A, "num_w": _A,
        "den_u": _A, "den_v": _A, "den_w": _A,
    })
    wp.overload(streaming_sdf_fanned_decode_3d, {
        "F_flat": _A, "body_meta": _A, "kin": _A,
        "gx": _A, "gy": _A, "gz": _A, "half_h": _dt, "blend_eps": _dt,
        "sdf_u": _A, "sdf_v": _A, "sdf_w": _A,
        "body_u": _A, "body_v": _A, "body_w": _A,
        "num_u": _A, "num_v": _A, "num_w": _A,
        "den_u": _A, "den_v": _A, "den_w": _A,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Python wrapper: persistent arrays + CUDA-graph capture
# ─────────────────────────────────────────────────────────────────────────────

class WarpStreamingSDF:
    """Warp streaming SDF with CUDA-graph support — two interchangeable designs.

    (a) SEQUENTIAL per-body  (`run_eager` / `capture_graph` / `run_graph`)
        B kernel launches (one per body).  No velocity blend (matches 2-D).

    (b) FANNED all-body  (`run_fanned_eager` / `capture_graph_fanned` /
        `run_graph_fanned`)  ← RECOMMENDED.
        2 kernel launches, CONSTANT in B (each dim = B·max_vol).  Honours the
        smooth velocity-blend path when ``blend_eps > 0`` (num/den softmin),
        mirroring the native ``body_update_3d``.

    ``dtype`` selects the float precision of every value-carrying array/scalar
    (``wp.float32`` default — bit-identical to the original; ``wp.float64`` for
    an f64 solver).  Static input tensors are cast to the matching torch dtype
    in :meth:`setup`; the caller's output arrays must already be that dtype.
    """

    def __init__(self, Ngx: int, Ngy: int, Ngz: int, device: str = "cuda:0",
                 dtype=wp.float32):
        self.Ngx = Ngx; self.Ngy = Ngy; self.Ngz = Ngz
        self.device = device
        self._wpf = dtype
        self._tdtype = torch.float64 if dtype == wp.float64 else torch.float32
        self._graph: Optional[wp.Graph] = None
        self._B   = 0
        self._max_vol = 0
        self._half_h  = 0.5
        self._blend_eps = 0.0

    def setup(
        self,
        F_flat_t:      torch.Tensor,
        F_offsets_t:   torch.Tensor,   # [B] int64
        body_shapes_t: torch.Tensor,   # [B, 3] int64
        body_meta_t:   torch.Tensor,   # [B, 10] float
        gx_t: torch.Tensor, gy_t: torch.Tensor, gz_t: torch.Tensor,
        h: float,
        max_vol: int,
        blend_eps: float = 0.0,
    ):
        """Convert static per-body tensors to persistent Warp arrays."""
        B = int(F_offsets_t.shape[0])
        self._B = B
        self._max_vol = int(max_vol)
        self._half_h  = float(h) * 0.5
        self._blend_eps = float(blend_eps)

        td = self._tdtype
        # Static (never change)
        self._F_flat      = wp.from_torch(F_flat_t.to(td).contiguous())
        self._F_offsets   = wp.from_torch(F_offsets_t.contiguous())
        self._body_shapes = wp.from_torch(body_shapes_t.reshape(-1).contiguous())
        self._body_meta   = wp.from_torch(body_meta_t.to(td).reshape(-1).contiguous())
        self._gx = wp.from_torch(gx_t.to(td).contiguous())
        self._gy = wp.from_torch(gy_t.to(td).contiguous())
        self._gz = wp.from_torch(gz_t.to(td).contiguous())

        # Dynamic (updated per step)
        self._kin      = wp.zeros(B * 21, dtype=self._wpf, device=self.device)
        self._aabb_lo  = wp.zeros(B * 3,  dtype=wp.int64,   device=self.device)
        self._aabb_dim = wp.zeros(B * 3,  dtype=wp.int64,   device=self.device)

        # Blend accumulators (full-grid), only used when blend_eps>0.
        N = self.Ngx * self.Ngy * self.Ngz
        nb = N if self._blend_eps > 0.0 else 1
        self._num_u = wp.zeros(nb, dtype=self._wpf, device=self.device)
        self._num_v = wp.zeros(nb, dtype=self._wpf, device=self.device)
        self._num_w = wp.zeros(nb, dtype=self._wpf, device=self.device)
        self._den_u = wp.zeros(nb, dtype=self._wpf, device=self.device)
        self._den_v = wp.zeros(nb, dtype=self._wpf, device=self.device)
        self._den_w = wp.zeros(nb, dtype=self._wpf, device=self.device)

    def update_kinematics(
        self,
        kin_t:      torch.Tensor,   # [B, 21] float
        aabb_lo_t:  torch.Tensor,   # [B, 3]  int64
        aabb_dim_t: torch.Tensor,   # [B, 3]  int64
        max_vol:    int = None,
    ):
        """Copy per-step body poses into persistent Warp arrays.

        Must be called before run_eager() / run_graph() each step.

        ``max_vol`` is the CURRENT step's max per-body AABB volume.  The
        launch dimension must cover it: a deforming body's AABB can grow
        past the setup-time value, and a stale ``_max_vol`` silently
        truncates the highest-index cells of the AABB (they keep FAR sdf /
        zero body velocity).  Grow-only watermark; growth invalidates any
        captured graph (its launch dims are frozen).
        """
        if max_vol is not None and int(max_vol) > self._max_vol:
            self._max_vol = int(max_vol)
            self._graph = None
        wp.copy(self._kin,
                wp.from_torch(kin_t.to(self._tdtype).reshape(-1).contiguous()))
        wp.copy(self._aabb_lo,  wp.from_torch(aabb_lo_t.reshape(-1).contiguous()))
        wp.copy(self._aabb_dim, wp.from_torch(aabb_dim_t.reshape(-1).contiguous()))

    def _zero_blend(self):
        if self._blend_eps > 0.0:
            self._num_u.zero_(); self._num_v.zero_(); self._num_w.zero_()
            self._den_u.zero_(); self._den_v.zero_(); self._den_w.zero_()

    def _launch_all_bodies(self, sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW):
        """Launch one kernel per body sequentially (B kernel launches)."""
        hh = self._wpf(self._half_h)
        for b in range(self._B):
            wp.launch(
                streaming_sdf_one_body_3d,
                dim=self._max_vol,
                inputs=[
                    self._F_flat, self._F_offsets,
                    self._body_shapes, self._body_meta,
                    self._kin, self._aabb_lo, self._aabb_dim,
                    b,
                    self._gx, self._gy, self._gz,
                    hh, self._max_vol,
                    self.Ngy, self.Ngz,
                    sdf_cc, sdf_u, sdf_v, sdf_w,
                    bU, bV, bW,
                ],
                device=self.device,
            )

    def run_eager(self, sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW):
        """Run all B body kernels eagerly (B Python kernel submissions)."""
        self._launch_all_bodies(sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW)

    def capture_graph(self, sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW):
        """Capture the B per-body launches as one CUDA graph."""
        with wp.ScopedCapture(device=self.device) as capture:
            self._launch_all_bodies(sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW)
        self._graph = capture.graph

    def run_graph(self):
        """Replay the captured graph (1 host call, B kernel executions)."""
        if self._graph is None:
            raise RuntimeError("call capture_graph() first")
        wp.capture_launch(self._graph)

    # ── Fanned mode (constant in B: 2 launches regardless of body count) ──────

    def _key_dummy(self):
        if getattr(self, "_kdummy", None) is None:
            self._kdummy = wp.zeros(1, dtype=wp.int64, device=self.device)
        return self._kdummy

    def _launch_fanned(self, sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW,
                       key_u=None, key_v=None, key_w=None, emit_keys=0,
                       dirty=(0, 0, 0, 0, 0)):
        """Launch the 2 fanned kernels (min + decode), each dim = B*max_vol."""
        self._zero_blend()
        dim = self._B * self._max_vol
        hh = self._wpf(self._half_h)
        be = self._wpf(self._blend_eps)
        ku = key_u if emit_keys else self._key_dummy()
        kv = key_v if emit_keys else self._key_dummy()
        kw = key_w if emit_keys else self._key_dummy()
        di0, dj0, dk0, dAj, dAk = (int(x) for x in dirty)
        wp.launch(
            streaming_sdf_fanned_min_3d, dim=dim,
            inputs=[
                self._F_flat, self._F_offsets,
                self._body_shapes, self._body_meta,
                self._kin, self._aabb_lo, self._aabb_dim,
                self._gx, self._gy, self._gz,
                hh, self._max_vol,
                self.Ngy, self.Ngz, be,
                sdf_cc, sdf_u, sdf_v, sdf_w,
                self._num_u, self._num_v, self._num_w,
                self._den_u, self._den_v, self._den_w,
            ],
            device=self.device,
        )
        wp.launch(
            streaming_sdf_fanned_decode_3d, dim=dim,
            inputs=[
                self._F_flat, self._F_offsets,
                self._body_shapes, self._body_meta,
                self._kin, self._aabb_lo, self._aabb_dim,
                self._gx, self._gy, self._gz,
                hh, self._max_vol,
                self.Ngy, self.Ngz, be,
                sdf_u, sdf_v, sdf_w,
                bU, bV, bW,
                self._num_u, self._num_v, self._num_w,
                self._den_u, self._den_v, self._den_w,
                int(emit_keys), ku, kv, kw,
                di0, dj0, dk0, dAj, dAk,
            ],
            device=self.device,
        )

    def run_fanned_eager(self, sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW,
                         key_u=None, key_v=None, key_w=None, emit_keys=0,
                         dirty=(0, 0, 0, 0, 0)):
        """Run the 2 fanned kernels eagerly (2 Python submissions, any B)."""
        self._launch_fanned(sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW,
                            key_u=key_u, key_v=key_v, key_w=key_w,
                            emit_keys=emit_keys, dirty=dirty)

    def capture_graph_fanned(self, sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW):
        """Capture the 2 fanned launches as one CUDA graph."""
        with wp.ScopedCapture(device=self.device) as capture:
            self._launch_fanned(sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW)
        self._graph_fanned = capture.graph

    def run_graph_fanned(self):
        """Replay the captured fanned graph (1 host call, 2 kernel executions)."""
        if getattr(self, "_graph_fanned", None) is None:
            raise RuntimeError("call capture_graph_fanned() first")
        wp.capture_launch(self._graph_fanned)
