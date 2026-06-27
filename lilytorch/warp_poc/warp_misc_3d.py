"""Warp 3-D pointwise/gather kernels: interp_3d + apply_bcs_3d (coverage fill).

3-D analogues of `warp_misc_2d.py`.  interp_3d reuses the trilinear sampler from
`warp_kernels.py` (Kernel A) and adds the triquadratic branch (faithful port of
`triquadratic_sample_uniform` in streaming_sdf.cu).  apply_bcs_3d mirrors
`apply_bcs_3d_kernel`: per BC op, write a ghost FACE (2-D launch over the face),
Neumann/Dirichlet in stage 1, reflective in stage 2.
"""
from __future__ import annotations

import warp as wp
import torch

wp.init()

from lilytorch.warp_poc.warp_kernels import trilinear_sample_off


@wp.func
def triquadratic_sample_off(
    F: wp.array(dtype=wp.float32), F_off: int,
    Mx: int, My: int, Mz: int,
    bx0: wp.float32, by0: wp.float32, bz0: wp.float32,
    inv_dx: wp.float32, inv_dy: wp.float32, inv_dz: wp.float32,
    xq: wp.float32, yq: wp.float32, zq: wp.float32,
) -> wp.float32:
    tx = wp.clamp((xq - bx0) * inv_dx, wp.float32(0.0), wp.float32(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, wp.float32(0.0), wp.float32(My - 1))
    tz = wp.clamp((zq - bz0) * inv_dz, wp.float32(0.0), wp.float32(Mz - 1))
    ix = wp.min(int(tx), Mx - 2)
    iy = wp.min(int(ty), My - 2)
    iz = wp.min(int(tz), Mz - 2)
    if ix < 1 or iy < 1 or iz < 1 or Mx < 3 or My < 3 or Mz < 3:
        return trilinear_sample_off(F, F_off, Mx, My, Mz, bx0, by0, bz0,
                                    inv_dx, inv_dy, inv_dz, xq, yq, zq)
    fx = tx - wp.float32(ix); fy = ty - wp.float32(iy); fz = tz - wp.float32(iz)
    half = wp.float32(0.5)
    wxm = half*fx*(fx-1.0); wx0 = 1.0-fx*fx; wxp = half*fx*(fx+1.0)
    wym = half*fy*(fy-1.0); wy0 = 1.0-fy*fy; wyp = half*fy*(fy+1.0)
    wzm = half*fz*(fz-1.0); wz0 = 1.0-fz*fz; wzp = half*fz*(fz+1.0)
    s2 = Mz
    s1 = My * Mz
    base = F_off + (ix-1)*s1 + (iy-1)*s2 + (iz-1)
    out = wp.float32(0.0)
    for dx in range(3):
        wx = wxm
        if dx == 1: wx = wx0
        if dx == 2: wx = wxp
        b0 = base + dx*s1
        plane = wp.float32(0.0)
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
    F: wp.array(dtype=wp.float32),
    xq: wp.array(dtype=wp.float32), yq: wp.array(dtype=wp.float32),
    zq: wp.array(dtype=wp.float32),
    N: int, Mx: int, My: int, Mz: int,
    bx0: wp.float32, by0: wp.float32, bz0: wp.float32,
    idx: wp.float32, idy: wp.float32, idz: wp.float32,
    interp_method: int,
    G: wp.array(dtype=wp.float32),
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


def interp_3d_warp(F_t, xq_t, yq_t, zq_t, bx0, by0, bz0, idx, idy, idz,
                   Mx, My, Mz, method="linear"):
    wdev = "cuda:0" if F_t.device.type == "cuda" else "cpu"
    N = int(xq_t.numel())
    G = torch.empty(N, dtype=torch.float32, device=F_t.device)
    if N == 0:
        return G
    im = 1 if method == "quadratic" else 0
    f32 = lambda t: wp.from_torch(t.reshape(-1).contiguous().to(torch.float32))
    wp.launch(interp_3d_kernel, dim=N,
              inputs=[wp.from_torch(F_t.reshape(-1).contiguous()),
                      f32(xq_t), f32(yq_t), f32(zq_t),
                      N, int(Mx), int(My), int(Mz),
                      wp.float32(bx0), wp.float32(by0), wp.float32(bz0),
                      wp.float32(idx), wp.float32(idy), wp.float32(idz),
                      im, wp.from_torch(G)],
              device=wdev)
    return G


# ─── apply_bcs_3d ────────────────────────────────────────────────────────────

@wp.kernel
def apply_bcs_3d_kernel(
    u: wp.array(dtype=wp.float64), v: wp.array(dtype=wp.float64),
    w: wp.array(dtype=wp.float64),
    shapes: wp.array(dtype=wp.int64),     # [3*3] = (Nx,Ny,Nz) per comp
    neu_desc: wp.array(dtype=wp.int32), N_neu: int,
    dir_desc: wp.array(dtype=wp.int32), dir_val: wp.array(dtype=wp.float64), N_dir: int,
    ref_desc: wp.array(dtype=wp.int32), ref_val: wp.array(dtype=wp.float64), N_ref: int,
):
    op, i, j = wp.tid()
    total = N_neu + N_dir + N_ref
    if op >= total:
        return
    kind = int(0); comp = int(0); axis = int(0)
    dst_along = int(0); src_along = int(0); value = wp.float64(0.0)
    if op < N_neu:
        kind = 0
        comp = neu_desc[op*3+0]; axis = neu_desc[op*3+1]; side = neu_desc[op*3+2]
        sz = int(shapes[comp*3+axis])
        if side == 0:
            dst_along = 0; src_along = 1
        else:
            dst_along = sz-1; src_along = sz-2
    elif op < N_neu + N_dir:
        d = op - N_neu; kind = 1
        comp = dir_desc[d*3+0]; axis = dir_desc[d*3+1]; offset = dir_desc[d*3+2]
        sz = int(shapes[comp*3+axis])
        if offset >= 0:
            dst_along = offset
        else:
            dst_along = sz + offset
        value = dir_val[d]
    else:
        r = op - N_neu - N_dir; kind = 2
        comp = ref_desc[r*4+0]; axis = ref_desc[r*4+1]
        dst_off = ref_desc[r*4+2]; src_off = ref_desc[r*4+3]
        sz = int(shapes[comp*3+axis])
        if dst_off >= 0:
            dst_along = dst_off
        else:
            dst_along = sz + dst_off
        if src_off >= 0:
            src_along = src_off
        else:
            src_along = sz + src_off
        value = ref_val[r]

    Nx = int(shapes[comp*3+0]); Ny = int(shapes[comp*3+1]); Nz = int(shapes[comp*3+2])
    if axis == 0:
        d0 = Ny; d1 = Nz
    elif axis == 1:
        d0 = Nx; d1 = Nz
    else:
        d0 = Nx; d1 = Ny
    if i >= d0 or j >= d1:
        return
    s1 = Ny * Nz
    s2 = Nz
    if axis == 0:
        dst = dst_along*s1 + i*s2 + j
        src = src_along*s1 + i*s2 + j
    elif axis == 1:
        dst = i*s1 + dst_along*s2 + j
        src = i*s1 + src_along*s2 + j
    else:
        dst = i*s1 + j*s2 + dst_along
        src = i*s1 + j*s2 + src_along

    if comp == 0:
        if kind == 0: u[dst] = u[src]
        elif kind == 1: u[dst] = value
        else: u[dst] = wp.float64(2.0)*value - u[src]
    elif comp == 1:
        if kind == 0: v[dst] = v[src]
        elif kind == 1: v[dst] = value
        else: v[dst] = wp.float64(2.0)*value - v[src]
    else:
        if kind == 0: w[dst] = w[src]
        elif kind == 1: w[dst] = value
        else: w[dst] = wp.float64(2.0)*value - w[src]


def _i32(t, wdev):
    if t is None or t.numel() == 0:
        return wp.zeros(1, dtype=wp.int32, device=wdev)
    return wp.from_torch(t.reshape(-1).contiguous().to(torch.int32))


def _f64(t, wdev):
    if t is None or t.numel() == 0:
        return wp.zeros(1, dtype=wp.float64, device=wdev)
    return wp.from_torch(t.reshape(-1).contiguous().to(torch.float64))


def apply_bcs_3d_warp(u, v, w, shapes, neu_desc, dir_desc, dir_val,
                      ref_desc, ref_val, max_face_dim):
    wdev = "cuda:0" if u.device.type == "cuda" else "cpu"
    N_neu = int(neu_desc.size(0)) if neu_desc is not None and neu_desc.numel() else 0
    N_dir = int(dir_desc.size(0)) if dir_desc is not None and dir_desc.numel() else 0
    N_ref = int(ref_desc.size(0)) if ref_desc is not None and ref_desc.numel() else 0
    if N_neu + N_dir + N_ref == 0 or max_face_dim <= 0:
        return
    uw = wp.from_torch(u.reshape(-1)); vw = wp.from_torch(v.reshape(-1)); ww = wp.from_torch(w.reshape(-1))
    shw = wp.from_torch(shapes.reshape(-1).contiguous().to(torch.int64))
    neu = _i32(neu_desc, wdev); dirw = _i32(dir_desc, wdev); refw = _i32(ref_desc, wdev)
    dvw = _f64(dir_val, wdev); rvw = _f64(ref_val, wdev)
    M = int(max_face_dim)
    if N_neu + N_dir > 0:
        wp.launch(apply_bcs_3d_kernel, dim=(N_neu + N_dir, M, M),
                  inputs=[uw, vw, ww, shw, neu, N_neu, dirw, dvw, N_dir, refw, rvw, 0],
                  device=wdev)
    if N_ref > 0:
        wp.launch(apply_bcs_3d_kernel, dim=(N_ref, M, M),
                  inputs=[uw, vw, ww, shw, neu, 0, dirw, dvw, 0, refw, rvw, N_ref],
                  device=wdev)
