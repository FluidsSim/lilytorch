"""Warp single-source 2-D pointwise/gather kernels: interp_2d + apply_bcs_2d.

Ports of native `interp_2d` (scattered-point bilinear/biquadratic gather) and
`apply_bcs_2d` (fused Neumann / Dirichlet / reflective ghost-line writes), the
last two stencil/pointwise stages of the per-step pipeline (VALIDATION_STATUS
§D).  interp reuses the same `sdf_sample_off_2d` device func as Kernel A, so the
gather is bit-identical to `streaming_sdf_stag_2d_multi`'s SDF sampling.
"""
from __future__ import annotations

from typing import Any

import warp as wp
import torch

wp.init()

from lilytorch.src.kernels.streaming_sdf_2d import sdf_sample_off_2d


# ─────────────────────────────────────────────────────────────────────────────
#  interp_2d : one thread per query point
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def interp_2d_kernel(
    F:  wp.array(dtype=Any),
    xq: wp.array(dtype=Any),
    yq: wp.array(dtype=Any),
    N: int, Mx: int, My: int,
    bx0: Any, by0: Any,
    inv_dx: Any, inv_dy: Any,
    interp_method: int,
    G: wp.array(dtype=Any),
):
    tid = wp.tid()
    if tid >= N:
        return
    G[tid] = sdf_sample_off_2d(interp_method, F, 0, Mx, My,
                               bx0, by0, inv_dx, inv_dy, xq[tid], yq[tid])


for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(interp_2d_kernel,
                {"F": _A, "xq": _A, "yq": _A, "G": _A,
                 "bx0": _dt, "by0": _dt, "inv_dx": _dt, "inv_dy": _dt})


def interp_2d_warp(F_t, xq_t, yq_t, bx0, by0, inv_dx, inv_dy, Mx, My,
                   method="linear"):
    """Warp port of native ``interp_2d``.  Returns G shape (N,) in F's dtype."""
    wdev = "cuda:0" if F_t.device.type == "cuda" else "cpu"
    N = int(xq_t.numel())
    G = torch.empty(N, dtype=F_t.dtype, device=F_t.device)
    if N == 0:
        return G
    im = 1 if method == "quadratic" else 0
    wpf = wp.float64 if F_t.dtype == torch.float64 else wp.float32
    tdt = F_t.dtype
    wp.launch(interp_2d_kernel, dim=N,
              inputs=[wp.from_torch(F_t.reshape(-1).contiguous()),
                      wp.from_torch(xq_t.reshape(-1).contiguous().to(tdt)),
                      wp.from_torch(yq_t.reshape(-1).contiguous().to(tdt)),
                      N, int(Mx), int(My),
                      wpf(bx0), wpf(by0),
                      wpf(inv_dx), wpf(inv_dy), im,
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
    u: wp.array(dtype=Any),
    v: wp.array(dtype=Any),
    shapes: wp.array(dtype=wp.int64),     # [4] = (uNx,uNy, vNx,vNy)
    neu_desc: wp.array(dtype=wp.int32),   # [N_neu*3]
    N_neu: int,
    dir_desc: wp.array(dtype=wp.int32),   # [N_dir*3]
    dir_val:  wp.array(dtype=Any),
    N_dir: int,
    ref_desc: wp.array(dtype=wp.int32),   # [N_ref*4]
    ref_val:  wp.array(dtype=Any),
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
    value = type(u[0])(0.0)

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

    two = type(u[0])(2.0)
    if comp == 0:
        if kind == 0:
            u[dst_lin] = u[src_lin]
        elif kind == 1:
            u[dst_lin] = value
        else:
            u[dst_lin] = two * value - u[src_lin]
    else:
        if kind == 0:
            v[dst_lin] = v[src_lin]
        elif kind == 1:
            v[dst_lin] = value
        else:
            v[dst_lin] = two * value - v[src_lin]


# Register float32 + float64 specialisations (only the value arrays are generic).
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(apply_bcs_2d_kernel,
                {"u": _A, "v": _A, "dir_val": _A, "ref_val": _A})


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


def apply_bcs_2d_warp(u, v, shapes, neu_desc, dir_desc, dir_val,
                      ref_desc, ref_val, max_line_dim):
    """Warp port of native ``apply_bcs_2d``; mutates u/v in place.

    Dtype-generic: u/v and the BC value arrays run in the field dtype
    (float32 or float64).
    """
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
    dvw = _valf(dir_val, wdev, u.dtype); rvw = _valf(ref_val, wdev, u.dtype)

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


class ApplyBcs2DGraphRunner:
    """CUDA-graph-cached ``apply_bcs_2d``: the eager wrapper's per-call host floor
    (~90 µs: ~17 µs Warp-array wrapping + 2× ~36 µs ``wp.launch`` submission) is
    replaced by a single ``wp.capture_launch`` (~3 µs) once the (u, v, descriptor,
    max_line) pointer signature is stable.  Ghost writes are in-place into the
    persistent velocity fields, so there is **no extra memory** vs native.

    Churn guard: a pointer signature is only captured on its **second** sighting,
    so one-shot tensors (e.g. fresh projection outputs) stay eager and never pay
    the (expensive) capture cost.  CPU and the unstable first-sighting path
    delegate to :func:`apply_bcs_2d_warp` (bit-identical)."""

    def __init__(self):
        self._graphs = {}   # key -> (graph, cached wp arrays)
        self._seen = {}     # key -> sighting count

    def __call__(self, u, v, shapes, neu_desc, dir_desc, dir_val,
                 ref_desc, ref_val, max_line_dim):
        N_neu = int(neu_desc.size(0)) if neu_desc is not None and neu_desc.numel() else 0
        N_dir = int(dir_desc.size(0)) if dir_desc is not None and dir_desc.numel() else 0
        N_ref = int(ref_desc.size(0)) if ref_desc is not None and ref_desc.numel() else 0
        if N_neu + N_dir + N_ref == 0 or max_line_dim <= 0:
            return
        if u.device.type != "cuda":     # CPU: eager (no graph capture)
            return apply_bcs_2d_warp(u, v, shapes, neu_desc, dir_desc, dir_val,
                                     ref_desc, ref_val, max_line_dim)
        key = (u.data_ptr(), v.data_ptr(), shapes.data_ptr(),
               neu_desc.data_ptr(), dir_desc.data_ptr(), ref_desc.data_ptr(),
               int(max_line_dim), str(u.dtype))
        ent = self._graphs.get(key)
        if ent is None:
            n = self._seen.get(key, 0) + 1
            self._seen[key] = n
            if n < 2:        # first sighting → eager (might be a one-shot ptr)
                return apply_bcs_2d_warp(u, v, shapes, neu_desc, dir_desc,
                                         dir_val, ref_desc, ref_val, max_line_dim)
            wdev = "cuda:0"
            uw = wp.from_torch(u.reshape(-1)); vw = wp.from_torch(v.reshape(-1))
            shw = wp.from_torch(shapes.reshape(-1).contiguous().to(torch.int64))
            neu = _i32(neu_desc, wdev); dirw = _i32(dir_desc, wdev)
            refw = _i32(ref_desc, wdev)
            dvw = _valf(dir_val, wdev, u.dtype); rvw = _valf(ref_val, wdev, u.dtype)
            ml = int(max_line_dim)

            def _launch():
                if N_neu + N_dir > 0:
                    wp.launch(apply_bcs_2d_kernel, dim=(N_neu + N_dir, ml),
                              inputs=[uw, vw, shw, neu, N_neu, dirw, dvw, N_dir,
                                      refw, rvw, 0], device=wdev)
                if N_ref > 0:
                    wp.launch(apply_bcs_2d_kernel, dim=(N_ref, ml),
                              inputs=[uw, vw, shw, neu, 0, dirw, dvw, 0,
                                      refw, rvw, N_ref], device=wdev)

            _launch()  # warm-up / JIT (idempotent ghost writes)
            with wp.ScopedCapture(device=wdev) as cap:
                _launch()
            ent = (cap.graph, (uw, vw, shw, neu, dirw, refw, dvw, rvw))
            self._graphs[key] = ent
        wp.capture_launch(ent[0])
