"""Warp single-source CPU+GPU kernels for BDIM streaming SDF — 2-D variant (AP7).

2-D analogue of `warp_kernels.py` (`body_update_3d`) with the
z-axis stripped, mirroring the native `streaming_sdf_2d.cu` line-for-line.

Per-body packed-array layouts (2-D, matching the native `.cu`):
  body_shapes : [B*2]   = (Mx, My)
  body_meta   : [B*7]   = (bx0, by0, bx_last, by_last, inv_dx, inv_dy, inv_vol)
  kin         : [B*11]  = R_T(4, row-major 2x2) + bp(2) + cm(2) + lv(2) + om(1)

Two interchangeable designs (same as the 3-D file):
  (a) sequential per-body : B launches, race-free compare-swap (g unique/thread).
  (b) fanned all-body     : 2 launches, const in B, float `wp.atomic_min` +
                            equality-decode (no `wp.bit_cast` in Warp 1.14).

Both honour the smooth velocity-blend path (`blend_eps > 0`): accumulate
Σ w_i v_i and Σ w_i with w_i = sigmoid(-s_i/blend_eps) via `wp.atomic_add`,
then the decode divides by Σ w_i — bit-mirroring the native num_*/den_* fields.

**Precision (single source, both dtypes).**  Every value-carrying array and
float scalar is a Warp *generic* (``Any``): float literals are materialised in
the bound element type via ``type(x)(literal)``.  ``wp.overload`` registers the
``float32`` *and* ``float64`` specialisations up front (Warp 1.14 does not
reliably re-specialise a generic kernel implicitly across dtypes in one
process; pre-registering is the supported path).  ``float32`` codegen is
unchanged from the original concrete kernels, so the existing parity tests stay
bit-identical; ``float64`` is what the in-solver bridge uses for an f64 solver.

All @wp.func / @wp.kernel are module level (Warp codegen requirement).
"""
from __future__ import annotations

from typing import Any

import warp as wp
import torch

wp.init()


# ─────────────────────────────────────────────────────────────────────────────
#  Bilinear / biquadratic interpolation on a uniform body SDF grid (flat offset)
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


# ─────────────────────────────────────────────────────────────────────────────
#  Shared decode of a fanned thread id  →  (b, i, j, g)
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def _fan_decode_tid_2d(
    tid: int, max_vol: int,
    aabb_lo:  wp.array(dtype=wp.int64),
    aabb_dim: wp.array(dtype=wp.int64),
    Ngy: int,
):
    """Map a fanned thread id → (b, i, j, g). g<0 ⇒ skip (thread past body vol)."""
    b     = tid // max_vol
    local = tid - b * max_vol
    Ai = int(aabb_dim[b * 2 + 0])
    Aj = int(aabb_dim[b * 2 + 1])
    vol = Ai * Aj
    if local >= vol:
        return b, -1, -1, -1
    di = local // Aj
    dj = local - di * Aj
    i  = int(aabb_lo[b * 2 + 0]) + di
    j  = int(aabb_lo[b * 2 + 1]) + dj
    g  = i * Ngy + j
    return b, i, j, g


@wp.func
def _fan_faces_2d(
    b: int, i: int, j: int,
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    gx: wp.array(dtype=Any),
    gy: wp.array(dtype=Any),
    half_h: Any,
    interp_method: int,
):
    """Return (s_cc, s_u, s_v) sampled SDFs for body b at cell (i,j)."""
    F_off  = int(F_offsets[b])
    Mx     = int(body_shapes[b * 2 + 0])
    My     = int(body_shapes[b * 2 + 1])
    bx0    = body_meta[b * 7 + 0]
    by0    = body_meta[b * 7 + 1]
    inv_dx = body_meta[b * 7 + 4]
    inv_dy = body_meta[b * 7 + 5]

    K0   = b * 11
    r00  = kin[K0 + 0]; r01 = kin[K0 + 1]
    r10  = kin[K0 + 2]; r11 = kin[K0 + 3]
    bp_x = kin[K0 + 4]; bp_y = kin[K0 + 5]

    xc = gx[i]; yc = gy[j]
    dx = xc - bp_x; dy = yc - bp_y
    bxq = r00 * dx + r01 * dy
    byq = r10 * dx + r11 * dy

    neg_hh = -half_h
    du_x = neg_hh * r00; du_y = neg_hh * r10
    dv_x = neg_hh * r01; dv_y = neg_hh * r11

    s_cc = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My, bx0, by0,
                             inv_dx, inv_dy, bxq, byq)
    s_u  = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My, bx0, by0,
                             inv_dx, inv_dy, bxq + du_x, byq + du_y)
    s_v  = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My, bx0, by0,
                             inv_dx, inv_dy, bxq + dv_x, byq + dv_y)
    return s_cc, s_u, s_v


# ─────────────────────────────────────────────────────────────────────────────
#  Fanned all-bodies kernels (constant in B — 2 launches, matches native)
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def streaming_sdf_fanned_min_2d(
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx: wp.array(dtype=Any),
    gy: wp.array(dtype=Any),
    half_h:  Any,
    max_vol: int,
    Ngy: int,
    interp_method: int,
    blend_eps: Any,
    sdf_cc: wp.array(dtype=Any),
    sdf_u:  wp.array(dtype=Any),
    sdf_v:  wp.array(dtype=Any),
    num_u:  wp.array(dtype=Any),
    num_v:  wp.array(dtype=Any),
    den_u:  wp.array(dtype=Any),
    den_v:  wp.array(dtype=Any),
):
    """Pass B: fanned all-body atomic-min of cc/u/v SDFs (1 launch).
    Also accumulates Σ w_i v_i / Σ w_i into num/den when blend_eps>0."""
    tid = wp.tid()
    b, i, j, g = _fan_decode_tid_2d(tid, max_vol, aabb_lo, aabb_dim, Ngy)
    if g < 0:
        return
    s_cc, s_u, s_v = _fan_faces_2d(
        b, i, j, F_flat, F_offsets, body_shapes, body_meta, kin,
        gx, gy, half_h, interp_method)
    wp.atomic_min(sdf_cc, g, s_cc)
    wp.atomic_min(sdf_u,  g, s_u)
    wp.atomic_min(sdf_v,  g, s_v)

    if blend_eps > type(blend_eps)(0.0):
        K0   = b * 11
        cm_x = kin[K0 + 6]; cm_y = kin[K0 + 7]
        lv_x = kin[K0 + 8]; lv_y = kin[K0 + 9]
        om   = kin[K0 + 10]
        xc = gx[i]; yc = gy[j]
        vU = lv_x - om * (yc - cm_y)
        vV = lv_y + om * (xc - cm_x)
        one = type(blend_eps)(1.0)
        wU = one / (one + wp.exp(s_u / blend_eps))
        wV = one / (one + wp.exp(s_v / blend_eps))
        wp.atomic_add(num_u, g, wU * vU); wp.atomic_add(den_u, g, wU)
        wp.atomic_add(num_v, g, wV * vV); wp.atomic_add(den_v, g, wV)


@wp.kernel
def streaming_sdf_fanned_decode_2d(
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx: wp.array(dtype=Any),
    gy: wp.array(dtype=Any),
    half_h:  Any,
    max_vol: int,
    Ngy: int,
    interp_method: int,
    blend_eps: Any,
    sdf_u:  wp.array(dtype=Any),
    sdf_v:  wp.array(dtype=Any),
    body_u: wp.array(dtype=Any),
    body_v: wp.array(dtype=Any),
    num_u:  wp.array(dtype=Any),
    num_v:  wp.array(dtype=Any),
    den_u:  wp.array(dtype=Any),
    den_v:  wp.array(dtype=Any),
    # BDIM-σ key emission (emit_keys == 0 → key_u/key_v are dummies, untouched).
    # When on, write the winning body-id (the body whose face SDF equals the
    # stored running-min) into the low 32 bits of key_u/key_v via int64
    # ``atomic_min`` → lowest-id-wins tie-break, mirroring the native packed
    # ``atomicMin`` (SDF high bits, body-id low bits).  The σ bdim_forcing only
    # reads ``key & 0xffffffff`` (body-id), so the high SDF bits are not needed.
    emit_keys: int,
    key_u: wp.array(dtype=wp.int64),
    key_v: wp.array(dtype=wp.int64),
):
    """Pass C: write the winning body's face velocity where SDF == stored min.
    With blend_eps>0, instead writes Σ w_i v_i / Σ w_i (the softmin blend)."""
    tid = wp.tid()
    b, i, j, g = _fan_decode_tid_2d(tid, max_vol, aabb_lo, aabb_dim, Ngy)
    if g < 0:
        return
    s_cc, s_u, s_v = _fan_faces_2d(
        b, i, j, F_flat, F_offsets, body_shapes, body_meta, kin,
        gx, gy, half_h, interp_method)

    K0   = b * 11
    cm_x = kin[K0 + 6]; cm_y = kin[K0 + 7]
    lv_x = kin[K0 + 8]; lv_y = kin[K0 + 9]
    om   = kin[K0 + 10]
    xc = gx[i]; yc = gy[j]

    blend = blend_eps > type(blend_eps)(0.0)
    den_tol = type(blend_eps)(1e-6)

    if blend and den_u[g] > den_tol:
        body_u[g] = num_u[g] / den_u[g]
    elif s_u == sdf_u[g]:
        body_u[g] = lv_x - om * (yc - cm_y)

    if blend and den_v[g] > den_tol:
        body_v[g] = num_v[g] / den_v[g]
    elif s_v == sdf_v[g]:
        body_v[g] = lv_y + om * (xc - cm_x)

    if emit_keys != 0:
        if s_u == sdf_u[g]:
            wp.atomic_min(key_u, g, wp.int64(b))
        if s_v == sdf_v[g]:
            wp.atomic_min(key_v, g, wp.int64(b))


# ─────────────────────────────────────────────────────────────────────────────
#  Sequential per-body kernel (B launches; race-free compare-swap)
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def streaming_sdf_one_body_2d(
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    b:  int,
    gx: wp.array(dtype=Any),
    gy: wp.array(dtype=Any),
    half_h:  Any,
    max_vol: int,
    Ngy: int,
    interp_method: int,
    sdf_cc: wp.array(dtype=Any),
    sdf_u:  wp.array(dtype=Any),
    sdf_v:  wp.array(dtype=Any),
    body_u: wp.array(dtype=Any),
    body_v: wp.array(dtype=Any),
):
    local = wp.tid()
    Ai = int(aabb_dim[b * 2 + 0])
    Aj = int(aabb_dim[b * 2 + 1])
    vol = Ai * Aj
    if local >= vol:
        return
    di = local // Aj
    dj = local - di * Aj
    i  = int(aabb_lo[b * 2 + 0]) + di
    j  = int(aabb_lo[b * 2 + 1]) + dj
    g  = i * Ngy + j

    s_cc, s_u, s_v = _fan_faces_2d(
        b, i, j, F_flat, F_offsets, body_shapes, body_meta, kin,
        gx, gy, half_h, interp_method)

    K0   = b * 11
    cm_x = kin[K0 + 6]; cm_y = kin[K0 + 7]
    lv_x = kin[K0 + 8]; lv_y = kin[K0 + 9]
    om   = kin[K0 + 10]
    xc = gx[i]; yc = gy[j]

    wp.atomic_min(sdf_cc, g, s_cc)
    if s_u < sdf_u[g]:
        sdf_u[g] = s_u
        body_u[g] = lv_x - om * (yc - cm_y)
    if s_v < sdf_v[g]:
        sdf_v[g] = s_v
        body_v[g] = lv_y + om * (xc - cm_x)


# ── Register float32 + float64 specialisations up front ─────────────────────
# (only the generic ``Any`` args need types; the int / int64 args stay as
#  declared.  Pre-registering avoids unreliable implicit re-specialisation.)
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(streaming_sdf_fanned_min_2d, {
        "F_flat": _A, "body_meta": _A, "kin": _A, "gx": _A, "gy": _A,
        "half_h": _dt, "blend_eps": _dt,
        "sdf_cc": _A, "sdf_u": _A, "sdf_v": _A,
        "num_u": _A, "num_v": _A, "den_u": _A, "den_v": _A,
    })
    wp.overload(streaming_sdf_fanned_decode_2d, {
        "F_flat": _A, "body_meta": _A, "kin": _A, "gx": _A, "gy": _A,
        "half_h": _dt, "blend_eps": _dt,
        "sdf_u": _A, "sdf_v": _A, "body_u": _A, "body_v": _A,
        "num_u": _A, "num_v": _A, "den_u": _A, "den_v": _A,
    })
    wp.overload(streaming_sdf_one_body_2d, {
        "F_flat": _A, "body_meta": _A, "kin": _A, "gx": _A, "gy": _A,
        "half_h": _dt,
        "sdf_cc": _A, "sdf_u": _A, "sdf_v": _A, "body_u": _A, "body_v": _A,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Python wrapper
# ─────────────────────────────────────────────────────────────────────────────

class WarpStreamingSDF2D:
    """2-D Warp streaming SDF — sequential + fanned designs, CUDA-graph capture.

    See `warp_kernels.WarpStreamingSDF` (3-D) for the design rationale.  This
    2-D class additionally wires the smooth velocity-blend path (num/den/eps).

    ``dtype`` selects the float precision of every value-carrying array/scalar
    (``wp.float32`` default — bit-identical to the original; ``wp.float64`` for
    an f64 solver).  Static input tensors are cast to the matching torch dtype
    in :meth:`setup`; the caller's output arrays must already be that dtype.
    """

    def __init__(self, Ngx: int, Ngy: int, device: str = "cuda:0",
                 dtype=wp.float32):
        self.Ngx = Ngx; self.Ngy = Ngy
        self.device = device
        self._wpf = dtype
        self._tdtype = torch.float64 if dtype == wp.float64 else torch.float32
        self._B = 0
        self._max_vol = 0
        self._half_h = 0.5
        self._interp = 0
        self._blend_eps = 0.0

    def setup(self, F_flat_t, F_offsets_t, body_shapes_t, body_meta_t,
              gx_t, gy_t, h, max_vol, interp_method: int = 0,
              blend_eps: float = 0.0):
        B = int(F_offsets_t.shape[0])
        self._B = B
        self._max_vol = int(max_vol)
        self._half_h = float(h) * 0.5
        self._interp = int(interp_method)
        self._blend_eps = float(blend_eps)

        td = self._tdtype
        self._F_flat      = wp.from_torch(F_flat_t.to(td).contiguous())
        self._F_offsets   = wp.from_torch(F_offsets_t.contiguous())
        self._body_shapes = wp.from_torch(body_shapes_t.reshape(-1).contiguous())
        self._body_meta   = wp.from_torch(body_meta_t.to(td).reshape(-1).contiguous())
        self._gx = wp.from_torch(gx_t.to(td).contiguous())
        self._gy = wp.from_torch(gy_t.to(td).contiguous())

        self._kin      = wp.zeros(B * 11, dtype=self._wpf, device=self.device)
        self._aabb_lo  = wp.zeros(B * 2,  dtype=wp.int64,  device=self.device)
        self._aabb_dim = wp.zeros(B * 2,  dtype=wp.int64,  device=self.device)

        # Blend accumulators (full-grid), only used when blend_eps>0.
        N = self.Ngx * self.Ngy
        nb = N if self._blend_eps > 0.0 else 1
        self._num_u = wp.zeros(nb, dtype=self._wpf, device=self.device)
        self._num_v = wp.zeros(nb, dtype=self._wpf, device=self.device)
        self._den_u = wp.zeros(nb, dtype=self._wpf, device=self.device)
        self._den_v = wp.zeros(nb, dtype=self._wpf, device=self.device)

    def update_kinematics(self, kin_t, aabb_lo_t, aabb_dim_t):
        wp.copy(self._kin,
                wp.from_torch(kin_t.to(self._tdtype).reshape(-1).contiguous()))
        wp.copy(self._aabb_lo,  wp.from_torch(aabb_lo_t.reshape(-1).contiguous()))
        wp.copy(self._aabb_dim, wp.from_torch(aabb_dim_t.reshape(-1).contiguous()))

    def _zero_blend(self):
        if self._blend_eps > 0.0:
            self._num_u.zero_(); self._num_v.zero_()
            self._den_u.zero_(); self._den_v.zero_()

    def _key_dummy(self):
        if getattr(self, "_kdummy", None) is None:
            self._kdummy = wp.zeros(1, dtype=wp.int64, device=self.device)
        return self._kdummy

    # ── fanned mode ──────────────────────────────────────────────────────────
    def _launch_fanned(self, sdf_cc, sdf_u, sdf_v, bU, bV,
                       key_u=None, key_v=None, emit_keys=0):
        self._zero_blend()
        dim = self._B * self._max_vol
        be = self._wpf(self._blend_eps)
        hh = self._wpf(self._half_h)
        ku = key_u if emit_keys else self._key_dummy()
        kv = key_v if emit_keys else self._key_dummy()
        wp.launch(streaming_sdf_fanned_min_2d, dim=dim,
                  inputs=[self._F_flat, self._F_offsets, self._body_shapes,
                          self._body_meta, self._kin, self._aabb_lo, self._aabb_dim,
                          self._gx, self._gy, hh,
                          self._max_vol, self.Ngy, self._interp, be,
                          sdf_cc, sdf_u, sdf_v,
                          self._num_u, self._num_v, self._den_u, self._den_v],
                  device=self.device)
        wp.launch(streaming_sdf_fanned_decode_2d, dim=dim,
                  inputs=[self._F_flat, self._F_offsets, self._body_shapes,
                          self._body_meta, self._kin, self._aabb_lo, self._aabb_dim,
                          self._gx, self._gy, hh,
                          self._max_vol, self.Ngy, self._interp, be,
                          sdf_u, sdf_v, bU, bV,
                          self._num_u, self._num_v, self._den_u, self._den_v,
                          int(emit_keys), ku, kv],
                  device=self.device)

    def run_fanned_eager(self, sdf_cc, sdf_u, sdf_v, bU, bV,
                         key_u=None, key_v=None, emit_keys=0):
        self._launch_fanned(sdf_cc, sdf_u, sdf_v, bU, bV,
                            key_u=key_u, key_v=key_v, emit_keys=emit_keys)

    def capture_graph_fanned(self, sdf_cc, sdf_u, sdf_v, bU, bV):
        with wp.ScopedCapture(device=self.device) as cap:
            self._launch_fanned(sdf_cc, sdf_u, sdf_v, bU, bV)
        self._graph_fanned = cap.graph

    def run_graph_fanned(self):
        if getattr(self, "_graph_fanned", None) is None:
            raise RuntimeError("call capture_graph_fanned() first")
        wp.capture_launch(self._graph_fanned)

    # ── sequential mode ──────────────────────────────────────────────────────
    def _launch_seq(self, sdf_cc, sdf_u, sdf_v, bU, bV):
        hh = self._wpf(self._half_h)
        for b in range(self._B):
            wp.launch(streaming_sdf_one_body_2d, dim=self._max_vol,
                      inputs=[self._F_flat, self._F_offsets, self._body_shapes,
                              self._body_meta, self._kin, self._aabb_lo,
                              self._aabb_dim, b, self._gx, self._gy,
                              hh, self._max_vol, self.Ngy,
                              self._interp, sdf_cc, sdf_u, sdf_v, bU, bV],
                      device=self.device)

    def run_eager(self, sdf_cc, sdf_u, sdf_v, bU, bV):
        self._launch_seq(sdf_cc, sdf_u, sdf_v, bU, bV)
