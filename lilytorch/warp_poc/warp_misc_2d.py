"""Warp single-source 2-D pointwise/gather kernels: interp_2d + apply_bcs_2d.

Ports of native `interp_2d` (scattered-point bilinear/biquadratic gather) and
`apply_bcs_2d` (fused Neumann / Dirichlet / reflective ghost-line writes), the
last two stencil/pointwise stages of the per-step pipeline (VALIDATION_STATUS
§D).  interp reuses the same `sdf_sample_off_2d` device func as Kernel A, so the
gather is bit-identical to `streaming_sdf_stag_2d_multi`'s SDF sampling.
"""
from __future__ import annotations

import warp as wp
import torch

wp.init()

from lilytorch.warp_poc.warp_kernels_2d import sdf_sample_off_2d


# ─────────────────────────────────────────────────────────────────────────────
#  interp_2d : one thread per query point
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def interp_2d_kernel(
    F:  wp.array(dtype=wp.float32),
    xq: wp.array(dtype=wp.float32),
    yq: wp.array(dtype=wp.float32),
    N: int, Mx: int, My: int,
    bx0: wp.float32, by0: wp.float32,
    inv_dx: wp.float32, inv_dy: wp.float32,
    interp_method: int,
    G: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid >= N:
        return
    G[tid] = sdf_sample_off_2d(interp_method, F, 0, Mx, My,
                               bx0, by0, inv_dx, inv_dy, xq[tid], yq[tid])


def interp_2d_warp(F_t, xq_t, yq_t, bx0, by0, inv_dx, inv_dy, Mx, My,
                   method="linear"):
    """Warp port of native ``interp_2d``.  Returns G shape (N,) float32."""
    wdev = "cuda:0" if F_t.device.type == "cuda" else "cpu"
    N = int(xq_t.numel())
    G = torch.empty(N, dtype=torch.float32, device=F_t.device)
    if N == 0:
        return G
    im = 1 if method == "quadratic" else 0
    wp.launch(interp_2d_kernel, dim=N,
              inputs=[wp.from_torch(F_t.reshape(-1).contiguous()),
                      wp.from_torch(xq_t.reshape(-1).contiguous().to(torch.float32)),
                      wp.from_torch(yq_t.reshape(-1).contiguous().to(torch.float32)),
                      N, int(Mx), int(My),
                      wp.float32(bx0), wp.float32(by0),
                      wp.float32(inv_dx), wp.float32(inv_dy), im,
                      wp.from_torch(G)],
              device=wdev)
    return G


# ─────────────────────────────────────────────────────────────────────────────
#  apply_bcs_2d : fused Neumann / Dirichlet / reflective ghost-line writes
#
#  Faithful port of ``apply_bcs_2d_kernel`` — launch dim = (total_ops, max_line);
#  op kind decoded from concatenated descriptor ranges:
#    op < N_neu                 : Neumann copy   (dst = src interior)
#    N_neu <= op < N_neu+N_dir  : Dirichlet write (dst = value)
#    else                       : reflective     (dst = 2*value - src)
#  comp 0/1 selects the u/v field; axis 0/1 the boundary normal.  Reflective
#  runs in a SECOND launch (read-after-write on the adjacent interior cell).
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def apply_bcs_2d_kernel(
    u: wp.array(dtype=wp.float64),
    v: wp.array(dtype=wp.float64),
    shapes: wp.array(dtype=wp.int64),     # [4] = (uNx,uNy, vNx,vNy)
    neu_desc: wp.array(dtype=wp.int32),   # [N_neu*3]
    N_neu: int,
    dir_desc: wp.array(dtype=wp.int32),   # [N_dir*3]
    dir_val:  wp.array(dtype=wp.float64),
    N_dir: int,
    ref_desc: wp.array(dtype=wp.int32),   # [N_ref*4]
    ref_val:  wp.array(dtype=wp.float64),
    N_ref: int,
):
    op, line = wp.tid()
    total = N_neu + N_dir + N_ref
    if op >= total:
        return

    kind = int(0)
    comp = int(0)
    axis = int(0)
    dst_along = int(0)
    src_along = int(0)
    value = wp.float64(0.0)

    if op < N_neu:
        kind = 0
        comp = neu_desc[op * 3 + 0]
        axis = neu_desc[op * 3 + 1]
        side = neu_desc[op * 3 + 2]
        sz = int(shapes[comp * 2 + axis])
        if side == 0:
            dst_along = 0; src_along = 1
        else:
            dst_along = sz - 1; src_along = sz - 2
    elif op < N_neu + N_dir:
        d = op - N_neu
        kind = 1
        comp = dir_desc[d * 3 + 0]
        axis = dir_desc[d * 3 + 1]
        offset = dir_desc[d * 3 + 2]
        sz = int(shapes[comp * 2 + axis])
        if offset >= 0:
            dst_along = offset
        else:
            dst_along = sz + offset
        value = dir_val[d]
    else:
        r = op - N_neu - N_dir
        kind = 2
        comp = ref_desc[r * 4 + 0]
        axis = ref_desc[r * 4 + 1]
        dst_off = ref_desc[r * 4 + 2]
        src_off = ref_desc[r * 4 + 3]
        sz = int(shapes[comp * 2 + axis])
        if dst_off >= 0:
            dst_along = dst_off
        else:
            dst_along = sz + dst_off
        if src_off >= 0:
            src_along = src_off
        else:
            src_along = sz + src_off
        value = ref_val[r]

    Nx = int(shapes[comp * 2 + 0])
    Ny = int(shapes[comp * 2 + 1])
    dim0_max = Ny
    if axis != 0:
        dim0_max = Nx
    if line >= dim0_max:
        return

    if axis == 0:
        dst_lin = dst_along * Ny + line
        src_lin = src_along * Ny + line
    else:
        dst_lin = line * Ny + dst_along
        src_lin = line * Ny + src_along

    if comp == 0:
        if kind == 0:
            u[dst_lin] = u[src_lin]
        elif kind == 1:
            u[dst_lin] = value
        else:
            u[dst_lin] = wp.float64(2.0) * value - u[src_lin]
    else:
        if kind == 0:
            v[dst_lin] = v[src_lin]
        elif kind == 1:
            v[dst_lin] = value
        else:
            v[dst_lin] = wp.float64(2.0) * value - v[src_lin]


def _i32(t, wdev):
    if t is None or t.numel() == 0:
        return wp.zeros(1, dtype=wp.int32, device=wdev)
    return wp.from_torch(t.reshape(-1).contiguous().to(torch.int32))


def _f64(t, wdev):
    if t is None or t.numel() == 0:
        return wp.zeros(1, dtype=wp.float64, device=wdev)
    return wp.from_torch(t.reshape(-1).contiguous().to(torch.float64))


def apply_bcs_2d_warp(u, v, shapes, neu_desc, dir_desc, dir_val,
                      ref_desc, ref_val, max_line_dim):
    """Warp port of native ``apply_bcs_2d``; mutates u/v in place."""
    wdev = "cuda:0" if u.device.type == "cuda" else "cpu"
    N_neu = int(neu_desc.size(0)) if neu_desc is not None and neu_desc.numel() else 0
    N_dir = int(dir_desc.size(0)) if dir_desc is not None and dir_desc.numel() else 0
    N_ref = int(ref_desc.size(0)) if ref_desc is not None and ref_desc.numel() else 0
    if N_neu + N_dir + N_ref == 0 or max_line_dim <= 0:
        return
    uw = wp.from_torch(u.reshape(-1))
    vw = wp.from_torch(v.reshape(-1))
    shw = wp.from_torch(shapes.reshape(-1).contiguous().to(torch.int64))
    neu = _i32(neu_desc, wdev); dirw = _i32(dir_desc, wdev); refw = _i32(ref_desc, wdev)
    dvw = _f64(dir_val, wdev); rvw = _f64(ref_val, wdev)

    # Stage 1: Neumann + Dirichlet (N_ref=0).
    if N_neu + N_dir > 0:
        wp.launch(apply_bcs_2d_kernel, dim=(N_neu + N_dir, int(max_line_dim)),
                  inputs=[uw, vw, shw, neu, N_neu, dirw, dvw, N_dir,
                          refw, rvw, 0],
                  device=wdev)
    # Stage 2: reflective (N_neu=N_dir=0 so op index maps into ref range).
    if N_ref > 0:
        wp.launch(apply_bcs_2d_kernel, dim=(N_ref, int(max_line_dim)),
                  inputs=[uw, vw, shw, neu, 0, dirw, dvw, 0,
                          refw, rvw, N_ref],
                  device=wdev)
