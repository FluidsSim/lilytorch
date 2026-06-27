"""Warp single-source CPU+GPU kernels for BDIM streaming SDF (AP7).

Sequential per-body design
──────────────────────────
Each body gets its own kernel launch (B launches per step).  Within a single
body's launch, every thread processes a unique AABB cell (global index g is
unique per-thread), so the compare-swap on sdf + body_vel is race-free without
atomics.  CUDA-graph capture collapses the B launches to a single host call.

This sidesteps the need for the 64-bit packed-key atomicMin trick used by the
native kernel (which required the non-portable `__float_as_uint`/`atomicCAS`
not available in Warp's built-in set).

Architecture comparison vs native kernel
─────────────────────────────────────────
  Native : 3 launches  (init_keys + fanned-all-bodies + decode) — fused, 1 global pass
  Warp   : B launches  (one per body) — sequential, simple compare-swap
  Both   : CUDA-graph capture → 1 graph replay, near-zero Python overhead

All @wp.func / @wp.kernel are at module level (Warp codegen requirement).
"""
from __future__ import annotations

import warp as wp
import torch
import numpy as np
from typing import Optional, Tuple

wp.init()

# ─────────────────────────────────────────────────────────────────────────────
#  Trilinear interpolation on a uniform body SDF grid (with flat-array offset)
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def trilinear_sample_off(
    F:      wp.array(dtype=wp.float32),
    F_off:  int,
    Mx: int, My: int, Mz: int,
    bx0: wp.float32, by0: wp.float32, bz0: wp.float32,
    inv_dx: wp.float32, inv_dy: wp.float32, inv_dz: wp.float32,
    xq: wp.float32, yq: wp.float32, zq: wp.float32,
) -> wp.float32:
    """Trilinear interp into F_flat at offset F_off with border clamp."""
    tx = wp.clamp((xq - bx0) * inv_dx, wp.float32(0.0), wp.float32(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, wp.float32(0.0), wp.float32(My - 1))
    tz = wp.clamp((zq - bz0) * inv_dz, wp.float32(0.0), wp.float32(Mz - 1))

    ix = wp.min(int(tx), Mx - 2)
    iy = wp.min(int(ty), My - 2)
    iz = wp.min(int(tz), Mz - 2)

    fx = tx - wp.float32(ix)
    fy = ty - wp.float32(iy)
    fz = tz - wp.float32(iz)
    wx0 = wp.float32(1.0) - fx
    wy0 = wp.float32(1.0) - fy
    wz0 = wp.float32(1.0) - fz

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
#  Per-body streaming SDF kernel
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def streaming_sdf_one_body_3d(
    # All-body packed arrays (static, loaded once)
    F_flat:      wp.array(dtype=wp.float32),
    F_offsets:   wp.array(dtype=wp.int64),    # [B]
    body_shapes: wp.array(dtype=wp.int64),    # [B*3] flat
    body_meta:   wp.array(dtype=wp.float32),  # [B*10] flat
    # Per-step packed arrays (updated before each step / graph replay)
    kin:         wp.array(dtype=wp.float32),  # [B*21] flat
    aabb_lo:     wp.array(dtype=wp.int64),    # [B*3] flat
    aabb_dim:    wp.array(dtype=wp.int64),    # [B*3] flat
    # This body's index (compile-time constant per launch in graph)
    b:  int,
    # Fluid grid
    gx: wp.array(dtype=wp.float32),
    gy: wp.array(dtype=wp.float32),
    gz: wp.array(dtype=wp.float32),
    half_h:  wp.float32,
    max_vol: int,
    Ngy: int, Ngz: int,
    # Outputs (full-grid flat)
    sdf_cc: wp.array(dtype=wp.float32),
    sdf_u:  wp.array(dtype=wp.float32),
    sdf_v:  wp.array(dtype=wp.float32),
    sdf_w:  wp.array(dtype=wp.float32),
    body_u: wp.array(dtype=wp.float32),
    body_v: wp.array(dtype=wp.float32),
    body_w: wp.array(dtype=wp.float32),
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
    # u-face: world -h/2 along x → R_T @ [-h/2, 0, 0]
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
#
#  Pass B (min)   : dim = B*max_vol; every (body, cell) thread does
#                   wp.atomic_min on sdf_cc/u/v/w → resolves all-body argmin
#                   without the uint64 packed key (wp.bit_cast not in Warp 1.14).
#  Pass C (decode): dim = B*max_vol; recompute the face SDF; where it equals the
#                   stored winning value, write that body's face velocity.  The
#                   trilinear recompute is bit-identical to Pass B (same inputs),
#                   so the winner's equality test is exact.  Ties (overlap cells)
#                   resolve last-writer-wins — same tie behaviour as native.

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
    F_flat:      wp.array(dtype=wp.float32),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=wp.float32),
    kin:         wp.array(dtype=wp.float32),
    gx: wp.array(dtype=wp.float32),
    gy: wp.array(dtype=wp.float32),
    gz: wp.array(dtype=wp.float32),
    half_h: wp.float32,
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
    F_flat:      wp.array(dtype=wp.float32),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=wp.float32),
    kin:         wp.array(dtype=wp.float32),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx: wp.array(dtype=wp.float32),
    gy: wp.array(dtype=wp.float32),
    gz: wp.array(dtype=wp.float32),
    half_h:  wp.float32,
    max_vol: int,
    Ngy: int, Ngz: int,
    sdf_cc: wp.array(dtype=wp.float32),
    sdf_u:  wp.array(dtype=wp.float32),
    sdf_v:  wp.array(dtype=wp.float32),
    sdf_w:  wp.array(dtype=wp.float32),
):
    """Pass B: fanned all-body atomic-min of cc/u/v/w SDFs (1 launch)."""
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


@wp.kernel
def streaming_sdf_fanned_decode_3d(
    F_flat:      wp.array(dtype=wp.float32),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=wp.float32),
    kin:         wp.array(dtype=wp.float32),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx: wp.array(dtype=wp.float32),
    gy: wp.array(dtype=wp.float32),
    gz: wp.array(dtype=wp.float32),
    half_h:  wp.float32,
    max_vol: int,
    Ngy: int, Ngz: int,
    sdf_u:  wp.array(dtype=wp.float32),
    sdf_v:  wp.array(dtype=wp.float32),
    sdf_w:  wp.array(dtype=wp.float32),
    body_u: wp.array(dtype=wp.float32),
    body_v: wp.array(dtype=wp.float32),
    body_w: wp.array(dtype=wp.float32),
):
    """Pass C: write the winning body's face velocity where SDF == stored min."""
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

    if s_u == sdf_u[g]:
        body_u[g] = lv_x + av_y * (zc - cm_z) - av_z * (yc - cm_y)
    if s_v == sdf_v[g]:
        body_v[g] = lv_y + av_z * (xc - cm_x) - av_x * (zc - cm_z)
    if s_w == sdf_w[g]:
        body_w[g] = lv_z + av_x * (yc - cm_y) - av_y * (xc - cm_x)


# ─────────────────────────────────────────────────────────────────────────────
#  Python wrapper: persistent arrays + CUDA-graph capture
# ─────────────────────────────────────────────────────────────────────────────

class WarpStreamingSDF:
    """Warp streaming SDF with CUDA-graph support — two interchangeable designs.

    (a) SEQUENTIAL per-body  (`run_eager` / `capture_graph` / `run_graph`)
        B kernel launches (one per body).  Within each launch g is unique per
        thread → conditional compare-swap is race-free without atomics.  Cost
        scales ~linearly with B.  Good for small B (fish, 3 links).

    (b) FANNED all-body  (`run_fanned_eager` / `capture_graph_fanned` /
        `run_graph_fanned`)  ← RECOMMENDED.
        2 kernel launches, CONSTANT in B (each dim = B·max_vol):
          Pass B = wp.atomic_min on cc/u/v/w across all bodies fanned;
          Pass C = recompute per-body face SDF, write that body's velocity where
                   SDF == the stored min (bit-identical recompute → exact winner).
        Structural analogue of the native 3-pass fanned kernel, using float
        atomic_min instead of the uint64 packed-key atomicMin (wp.bit_cast is
        absent in Warp 1.14, so the packed key can't be built — equality-decode
        replaces it).  Benchmarks at 0.6–1.2× of native at all B (see AP7).

    Both designs benefit from CUDA-graph capture (essential: eager is 3–14×
    slower from per-launch Python overhead).  Both pass parity vs the native
    kernel (SDF rel < 5e-4, body-vel rel < 1e-4).

    Usage (fanned, recommended)
    ───────────────────────────
        wsdf = WarpStreamingSDF(Ngx, Ngy, Ngz, device="cuda:0")
        wsdf.setup(F_flat, F_offsets, body_shapes, body_meta, gx, gy, gz, h, max_vol)
        wsdf.update_kinematics(kin, aabb_lo, aabb_dim)            # per step
        wsdf.capture_graph_fanned(sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW)  # once
        wsdf.run_graph_fanned()                                   # 1 host call, any B
    """

    def __init__(self, Ngx: int, Ngy: int, Ngz: int, device: str = "cuda:0"):
        self.Ngx = Ngx; self.Ngy = Ngy; self.Ngz = Ngz
        self.device = device
        self._graph: Optional[wp.Graph] = None
        self._B   = 0
        self._max_vol = 0
        self._half_h  = 0.5

        # Persistent output arrays (filled in setup)
        self._out: dict = {}

    def setup(
        self,
        F_flat_t:      torch.Tensor,
        F_offsets_t:   torch.Tensor,   # [B] int64
        body_shapes_t: torch.Tensor,   # [B, 3] int64
        body_meta_t:   torch.Tensor,   # [B, 10] float32
        gx_t: torch.Tensor, gy_t: torch.Tensor, gz_t: torch.Tensor,
        h: float,
        max_vol: int,
    ):
        """Convert static per-body tensors to persistent Warp arrays."""
        B = int(F_offsets_t.shape[0])
        self._B = B
        self._max_vol = max_vol
        self._half_h  = float(h) * 0.5

        # Static (never change)
        self._F_flat      = wp.from_torch(F_flat_t.contiguous())
        self._F_offsets   = wp.from_torch(F_offsets_t.contiguous())
        self._body_shapes = wp.from_torch(body_shapes_t.reshape(-1).contiguous())
        self._body_meta   = wp.from_torch(body_meta_t.reshape(-1).contiguous())
        self._gx = wp.from_torch(gx_t.contiguous())
        self._gy = wp.from_torch(gy_t.contiguous())
        self._gz = wp.from_torch(gz_t.contiguous())

        # Dynamic (updated per step)
        self._kin      = wp.zeros(B * 21, dtype=wp.float32, device=self.device)
        self._aabb_lo  = wp.zeros(B * 3,  dtype=wp.int64,   device=self.device)
        self._aabb_dim = wp.zeros(B * 3,  dtype=wp.int64,   device=self.device)

    def update_kinematics(
        self,
        kin_t:      torch.Tensor,   # [B, 21] float32
        aabb_lo_t:  torch.Tensor,   # [B, 3]  int64
        aabb_dim_t: torch.Tensor,   # [B, 3]  int64
    ):
        """Copy per-step body poses into persistent Warp arrays.

        Must be called before run_eager() / run_graph() each step.
        """
        wp.copy(self._kin,      wp.from_torch(kin_t.reshape(-1).contiguous()))
        wp.copy(self._aabb_lo,  wp.from_torch(aabb_lo_t.reshape(-1).contiguous()))
        wp.copy(self._aabb_dim, wp.from_torch(aabb_dim_t.reshape(-1).contiguous()))

    def _launch_all_bodies(
        self,
        sdf_cc: wp.array, sdf_u: wp.array,
        sdf_v:  wp.array, sdf_w: wp.array,
        bU: wp.array,     bV: wp.array, bW: wp.array,
    ):
        """Launch one kernel per body sequentially (B kernel launches)."""
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
                    wp.float32(self._half_h), self._max_vol,
                    self.Ngy, self.Ngz,
                    sdf_cc, sdf_u, sdf_v, sdf_w,
                    bU, bV, bW,
                ],
                device=self.device,
            )

    def run_eager(
        self,
        sdf_cc: wp.array, sdf_u: wp.array,
        sdf_v:  wp.array, sdf_w: wp.array,
        bU: wp.array,     bV: wp.array, bW: wp.array,
    ):
        """Run all B body kernels eagerly (B Python kernel submissions)."""
        self._launch_all_bodies(sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW)

    def capture_graph(
        self,
        sdf_cc: wp.array, sdf_u: wp.array,
        sdf_v:  wp.array, sdf_w: wp.array,
        bU: wp.array,     bV: wp.array, bW: wp.array,
    ):
        """Capture the B per-body launches as one CUDA graph.

        The graph references the persistent kin/aabb/output arrays; calling
        update_kinematics() before run_graph() ensures new poses are used.
        """
        with wp.ScopedCapture(device=self.device) as capture:
            self._launch_all_bodies(sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW)
        self._graph = capture.graph

    def run_graph(self):
        """Replay the captured graph (1 host call, B kernel executions)."""
        if self._graph is None:
            raise RuntimeError("call capture_graph() first")
        wp.capture_launch(self._graph)

    # ── Fanned mode (constant in B: 2 launches regardless of body count) ──────

    def _launch_fanned(
        self,
        sdf_cc: wp.array, sdf_u: wp.array,
        sdf_v:  wp.array, sdf_w: wp.array,
        bU: wp.array,     bV: wp.array, bW: wp.array,
    ):
        """Launch the 2 fanned kernels (min + decode), each dim = B*max_vol."""
        dim = self._B * self._max_vol
        wp.launch(
            streaming_sdf_fanned_min_3d, dim=dim,
            inputs=[
                self._F_flat, self._F_offsets,
                self._body_shapes, self._body_meta,
                self._kin, self._aabb_lo, self._aabb_dim,
                self._gx, self._gy, self._gz,
                wp.float32(self._half_h), self._max_vol,
                self.Ngy, self.Ngz,
                sdf_cc, sdf_u, sdf_v, sdf_w,
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
                wp.float32(self._half_h), self._max_vol,
                self.Ngy, self.Ngz,
                sdf_u, sdf_v, sdf_w,
                bU, bV, bW,
            ],
            device=self.device,
        )

    def run_fanned_eager(
        self,
        sdf_cc: wp.array, sdf_u: wp.array,
        sdf_v:  wp.array, sdf_w: wp.array,
        bU: wp.array,     bV: wp.array, bW: wp.array,
    ):
        """Run the 2 fanned kernels eagerly (2 Python submissions, any B)."""
        self._launch_fanned(sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW)

    def capture_graph_fanned(
        self,
        sdf_cc: wp.array, sdf_u: wp.array,
        sdf_v:  wp.array, sdf_w: wp.array,
        bU: wp.array,     bV: wp.array, bW: wp.array,
    ):
        """Capture the 2 fanned launches as one CUDA graph."""
        with wp.ScopedCapture(device=self.device) as capture:
            self._launch_fanned(sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW)
        self._graph_fanned = capture.graph

    def run_graph_fanned(self):
        """Replay the captured fanned graph (1 host call, 2 kernel executions)."""
        if getattr(self, "_graph_fanned", None) is None:
            raise RuntimeError("call capture_graph_fanned() first")
        wp.capture_launch(self._graph_fanned)
