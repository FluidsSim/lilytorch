"""Warp single-source port of the two-phase **cvof_sweep** kernel (MP10/T2d).

Faithful float64 port of native `cvof_sweep` (`cvof_sweep.cu`): the Weymouth &
Yue (2010) conservative-VOF directional sweep, per interior cell i along
`face_dim` (transverse dims FULL):

    out[i] = a[i] + cfl * ( F(i) - F(i+1) + a[i]*(u[i+1] - u[i]) )

with the upwind van-Leer-limited W&Y face flux F(k) and EDGE-CLAMP neighbour
reads — exactly mirroring `TwoPhase._cvof_sweep`.  ``out`` must be preallocated
as ``a.clone()`` (only interior cells along face_dim are overwritten).

Handles 2-D and 3-D with arbitrary strides (the velocity component is a strided
row view of the stacked _vel tensor): we build a zero-copy flat 1-D Warp view
via ``wp.array(ptr=t.data_ptr(), shape=(remaining,))`` and pass each tensor's
per-dim element strides as int kernel args, doing the native pointer arithmetic
(HANDOFF lesson 14).  Native cvof_sweep is CUDA-only → CPU validated by Warp
CPU == Warp GPU.  Confined to the two-phase path (no core-source edits).
"""
from __future__ import annotations

from typing import Any

import warp as wp
import torch

wp.init()


@wp.func
def cv_vleer(db: Any, df: Any):
    sum = db + df
    denom = sum
    if sum == type(db)(0.0):
        denom = type(db)(1.0)
    s = type(db)(2.0) * db * df / denom
    res = type(db)(0.0)
    if db * df > type(db)(0.0):
        res = s
    return res


@wp.func
def cv_face(
    a: wp.array(dtype=Any), u: wp.array(dtype=Any),
    a_base: wp.int64, u_base: wp.int64,
    k: wp.int32, N: wp.int32, a_s_fd: wp.int64, u_s_fd: wp.int64, cfl: Any,
):
    km1 = k - 1
    if km1 < 0:
        km1 = 0
    km2 = k - 2
    if km2 < 0:
        km2 = 0
    kp1 = k + 1
    if kp1 > N - 1:
        kp1 = N - 1

    ak = a[a_base + wp.int64(k) * a_s_fd]
    am1 = a[a_base + wp.int64(km1) * a_s_fd]
    am2 = a[a_base + wp.int64(km2) * a_s_fd]
    ap1 = a[a_base + wp.int64(kp1) * a_s_fd]
    ud = u[u_base + wp.int64(k) * u_s_fd]
    C = ud * cfl

    half = type(ak)(0.5)
    one = type(ak)(1.0)
    s_pos = cv_vleer(am1 - am2, ak - am1)
    f_pos = am1 + half * (one - C) * s_pos

    s_neg = cv_vleer(ak - am1, ap1 - ak)
    f_neg = ak - half * (one + C) * s_neg

    face = f_neg
    if C >= type(ak)(0.0):
        face = f_pos
    return ud * face


@wp.kernel
def cvof_sweep_kernel(
    a:   wp.array(dtype=Any),
    u:   wp.array(dtype=Any),
    out: wp.array(dtype=Any),
    Nfd: wp.int32, Nt1: wp.int32, Nt2: wp.int32,
    a_s_fd: wp.int64, a_s_t1: wp.int64, a_s_t2: wp.int64,
    u_s_fd: wp.int64, u_s_t1: wp.int64, u_s_t2: wp.int64,
    o_s_fd: wp.int64, o_s_t1: wp.int64, o_s_t2: wp.int64,
    cfl: Any,
):
    tid = wp.tid()
    Ni = Nfd - 2
    i_t2 = tid % Nt2
    rem = tid / Nt2
    i_t1 = rem % Nt1
    i_in = rem / Nt1
    if i_in >= Ni:
        return
    i = i_in + 1

    a_t = wp.int64(i_t1) * a_s_t1 + wp.int64(i_t2) * a_s_t2
    u_t = wp.int64(i_t1) * u_s_t1 + wp.int64(i_t2) * u_s_t2
    o_t = wp.int64(i_t1) * o_s_t1 + wp.int64(i_t2) * o_s_t2

    ai = a[a_t + wp.int64(i) * a_s_fd]
    FL = cv_face(a, u, a_t, u_t, i, Nfd, a_s_fd, u_s_fd, cfl)
    FR = cv_face(a, u, a_t, u_t, i + 1, Nfd, a_s_fd, u_s_fd, cfl)
    uL = u[u_t + wp.int64(i) * u_s_fd]
    uR = u[u_t + wp.int64(i + 1) * u_s_fd]

    res = ai + cfl * ((FL - FR) + ai * (uR - uL))
    out[o_t + wp.int64(i) * o_s_fd] = res


# Register float32 + float64 specialisations (only the generic args need types).
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(cvof_sweep_kernel, {"a": _A, "u": _A, "out": _A, "cfl": _dt})


def _wp_dtype(t: torch.Tensor):
    return wp.float64 if t.dtype == torch.float64 else wp.float32


def _flat(t: torch.Tensor, wdev: str):
    """Zero-copy flat 1-D Warp view (f32/f64) honouring storage_offset+strides."""
    elem = t.element_size()
    remaining = (t.untyped_storage().nbytes() - t.storage_offset() * elem) // elem
    return wp.array(ptr=t.data_ptr(), dtype=_wp_dtype(t), shape=(int(remaining),),
                    device=wdev)


def cvof_sweep_warp(a: torch.Tensor, u_d: torch.Tensor, cfl: float,
                    face_dim: int, out: torch.Tensor):
    """Warp port of native ``cvof_sweep``; writes interior of ``out`` in place.
    Generic in dtype (f32/f64), selected from ``a``."""
    ndim = a.dim()
    assert ndim in (2, 3)
    wdev = "cuda:0" if a.device.type == "cuda" else "cpu"
    _cfl = _wp_dtype(a)(cfl)
    Nfd = int(a.size(face_dim))
    t_dims = [d for d in range(ndim) if d != face_dim]
    Nt1 = int(a.size(t_dims[0])); a_s_t1 = a.stride(t_dims[0])
    u_s_t1 = u_d.stride(t_dims[0]); o_s_t1 = out.stride(t_dims[0])
    if len(t_dims) > 1:
        Nt2 = int(a.size(t_dims[1])); a_s_t2 = a.stride(t_dims[1])
        u_s_t2 = u_d.stride(t_dims[1]); o_s_t2 = out.stride(t_dims[1])
    else:
        Nt2 = 1; a_s_t2 = 0; u_s_t2 = 0; o_s_t2 = 0
    Ni = Nfd - 2
    if Ni <= 0 or Nt1 <= 0 or Nt2 <= 0:
        return
    wp.launch(cvof_sweep_kernel, dim=Ni * Nt1 * Nt2,
              inputs=[_flat(a, wdev), _flat(u_d, wdev), _flat(out, wdev),
                      Nfd, Nt1, Nt2,
                      wp.int64(a.stride(face_dim)), wp.int64(a_s_t1), wp.int64(a_s_t2),
                      wp.int64(u_d.stride(face_dim)), wp.int64(u_s_t1), wp.int64(u_s_t2),
                      wp.int64(out.stride(face_dim)), wp.int64(o_s_t1), wp.int64(o_s_t2),
                      _cfl],
              device=wdev)
