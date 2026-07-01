"""Warp 3-D pointwise/gather kernels: interp_3d + apply_bcs_3d (coverage fill).

3-D analogues of `warp_misc_2d.py`.  interp_3d reuses the trilinear sampler from
`warp_kernels.py` (Kernel A) and adds the triquadratic branch (faithful port of
`triquadratic_sample_uniform` in streaming_sdf.cu).  apply_bcs_3d mirrors
`apply_bcs_3d_kernel`: per BC op, write a ghost FACE (2-D launch over the face),
Neumann/Dirichlet in stage 1, reflective in stage 2.
"""
from __future__ import annotations

from typing import Any

import warp as wp
import torch

wp.init()

from lilytorch.src.kernels.streaming_sdf import trilinear_sample_off


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


# ─── apply_bcs_3d ────────────────────────────────────────────────────────────

@wp.kernel
def apply_bcs_3d_kernel(
    u: wp.array(dtype=Any), v: wp.array(dtype=Any),
    w: wp.array(dtype=Any),
    shapes: wp.array(dtype=wp.int64),     # [3*3] = (Nx,Ny,Nz) per comp
    neu_desc: wp.array(dtype=wp.int32), N_neu: int,
    dir_desc: wp.array(dtype=wp.int32), dir_val: wp.array(dtype=Any), N_dir: int,
    ref_desc: wp.array(dtype=wp.int32), ref_val: wp.array(dtype=Any), N_ref: int,
):
    op, i, j = wp.tid()
    total = N_neu + N_dir + N_ref
    if op >= total:
        return
    kind = int(0); comp = int(0); axis = int(0)
    dst_along = int(0); src_along = int(0); value = type(u[0])(0.0)
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

    two = type(u[0])(2.0)
    if comp == 0:
        if kind == 0: u[dst] = u[src]
        elif kind == 1: u[dst] = value
        else: u[dst] = two*value - u[src]
    elif comp == 1:
        if kind == 0: v[dst] = v[src]
        elif kind == 1: v[dst] = value
        else: v[dst] = two*value - v[src]
    else:
        if kind == 0: w[dst] = w[src]
        elif kind == 1: w[dst] = value
        else: w[dst] = two*value - w[src]


# Register float32 + float64 specialisations (only the value arrays are generic).
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(apply_bcs_3d_kernel,
                {"u": _A, "v": _A, "w": _A, "dir_val": _A, "ref_val": _A})


def _i32(t, wdev):
    if t is None or t.numel() == 0:
        return wp.zeros(1, dtype=wp.int32, device=wdev)
    return wp.from_torch(t.reshape(-1).contiguous().to(torch.int32))


def _valf(t, wdev, tdtype):
    """Flat Warp view of a value array cast to the field dtype (f32/f64)."""
    if t is None or t.numel() == 0:
        wpf = wp.float64 if tdtype == torch.float64 else wp.float32
        return wp.zeros(1, dtype=wpf, device=wdev)
    return wp.from_torch(t.reshape(-1).contiguous().to(tdtype))


def apply_bcs_3d_warp(u, v, w, shapes, neu_desc, dir_desc, dir_val,
                      ref_desc, ref_val, max_dim0, max_dim1=None):
    """Warp port of native ``apply_bcs_3d``; mutates u/v/w in place.

    Dtype-generic (f32/f64).  ``max_dim0``/``max_dim1`` are the two face-grid
    extents (native passes both, see ``_build_fused_bc_cache``); a single
    positional arg keeps backward-compat for cubic faces (``max_dim1=max_dim0``).
    Threads outside a face's own ``(d0, d1)`` early-return inside the kernel, so
    launching the per-face max is correct for non-cubic grids.
    """
    wdev = "cuda:0" if u.device.type == "cuda" else "cpu"
    N_neu = int(neu_desc.size(0)) if neu_desc is not None and neu_desc.numel() else 0
    N_dir = int(dir_desc.size(0)) if dir_desc is not None and dir_desc.numel() else 0
    N_ref = int(ref_desc.size(0)) if ref_desc is not None and ref_desc.numel() else 0
    M0 = int(max_dim0)
    M1 = int(max_dim1) if max_dim1 is not None else M0
    if N_neu + N_dir + N_ref == 0 or M0 <= 0 or M1 <= 0:
        return
    uw = wp.from_torch(u.reshape(-1)); vw = wp.from_torch(v.reshape(-1)); ww = wp.from_torch(w.reshape(-1))
    shw = wp.from_torch(shapes.reshape(-1).contiguous().to(torch.int64))
    neu = _i32(neu_desc, wdev); dirw = _i32(dir_desc, wdev); refw = _i32(ref_desc, wdev)
    dvw = _valf(dir_val, wdev, u.dtype); rvw = _valf(ref_val, wdev, u.dtype)
    if N_neu + N_dir > 0:
        wp.launch(apply_bcs_3d_kernel, dim=(N_neu + N_dir, M0, M1),
                  inputs=[uw, vw, ww, shw, neu, N_neu, dirw, dvw, N_dir, refw, rvw, 0],
                  device=wdev)
    if N_ref > 0:
        wp.launch(apply_bcs_3d_kernel, dim=(N_ref, M0, M1),
                  inputs=[uw, vw, ww, shw, neu, 0, dirw, dvw, 0, refw, rvw, N_ref],
                  device=wdev)


class ApplyBcs3DGraphRunner:
    """CUDA-graph-cached ``apply_bcs_3d`` — 3-D analogue of
    :class:`lilytorch.src.kernels.misc_2d.ApplyBcs2DGraphRunner`.  In-place ghost
    writes into the persistent u/v/w fields (no extra memory); captured on the
    second sighting of a stable (u, v, w, descriptor, face-dims) signature, eager
    otherwise.  CPU delegates to :func:`apply_bcs_3d_warp`."""

    def __init__(self):
        self._graphs = {}
        self._seen = {}

    def __call__(self, u, v, w, shapes, neu_desc, dir_desc, dir_val,
                 ref_desc, ref_val, max_dim0, max_dim1=None):
        N_neu = int(neu_desc.size(0)) if neu_desc is not None and neu_desc.numel() else 0
        N_dir = int(dir_desc.size(0)) if dir_desc is not None and dir_desc.numel() else 0
        N_ref = int(ref_desc.size(0)) if ref_desc is not None and ref_desc.numel() else 0
        M0 = int(max_dim0)
        M1 = int(max_dim1) if max_dim1 is not None else M0
        if N_neu + N_dir + N_ref == 0 or M0 <= 0 or M1 <= 0:
            return
        if u.device.type != "cuda":
            return apply_bcs_3d_warp(u, v, w, shapes, neu_desc, dir_desc, dir_val,
                                     ref_desc, ref_val, M0, M1)
        key = (u.data_ptr(), v.data_ptr(), w.data_ptr(), shapes.data_ptr(),
               neu_desc.data_ptr(), dir_desc.data_ptr(), ref_desc.data_ptr(),
               M0, M1, str(u.dtype))
        ent = self._graphs.get(key)
        if ent is None:
            n = self._seen.get(key, 0) + 1
            self._seen[key] = n
            if n < 2:
                return apply_bcs_3d_warp(u, v, w, shapes, neu_desc, dir_desc,
                                         dir_val, ref_desc, ref_val, M0, M1)
            wdev = "cuda:0"
            uw = wp.from_torch(u.reshape(-1)); vw = wp.from_torch(v.reshape(-1))
            ww = wp.from_torch(w.reshape(-1))
            shw = wp.from_torch(shapes.reshape(-1).contiguous().to(torch.int64))
            neu = _i32(neu_desc, wdev); dirw = _i32(dir_desc, wdev)
            refw = _i32(ref_desc, wdev)
            dvw = _valf(dir_val, wdev, u.dtype); rvw = _valf(ref_val, wdev, u.dtype)

            def _launch():
                if N_neu + N_dir > 0:
                    wp.launch(apply_bcs_3d_kernel, dim=(N_neu + N_dir, M0, M1),
                              inputs=[uw, vw, ww, shw, neu, N_neu, dirw, dvw,
                                      N_dir, refw, rvw, 0], device=wdev)
                if N_ref > 0:
                    wp.launch(apply_bcs_3d_kernel, dim=(N_ref, M0, M1),
                              inputs=[uw, vw, ww, shw, neu, 0, dirw, dvw, 0,
                                      refw, rvw, N_ref], device=wdev)

            _launch()
            with wp.ScopedCapture(device=wdev) as cap:
                _launch()
            ent = (cap.graph, (uw, vw, ww, shw, neu, dirw, refw, dvw, rvw))
            self._graphs[key] = ent
        wp.capture_launch(ent[0])
