"""Warp single-source advection kernel (AP6 counter-demo, part 2).

First-order upwind flux-divergence advection of a cell-centred scalar by a
cell-centred velocity field, with the flux differencing fused in-kernel (same
idea as the native `advect_flux_add`, minus the high-order limiter):

    q_new[c] = q[c] - dt * ( d(uq)/dx + d(vq)/dy + d(wq)/dz )_upwind

Face velocity = average of the two adjacent cell-centred velocities; the face
scalar is the upstream cell (sign of the face velocity).  All fields are
ghost-padded (N+2)³ with clamp (zero-gradient) ghosts.

Demonstrates the *fusible stencil* class is writable as one `@wp.kernel`
running on CPU + CUDA, validated against a PyTorch reference of the same scheme.
"""
from __future__ import annotations

from typing import Any

import warp as wp
import torch

wp.init()


@wp.func
def _flux(uf: wp.float32, q_lo: wp.float32, q_hi: wp.float32) -> wp.float32:
    """Face flux uf*q with first-order upwind scalar."""
    if uf > 0.0:
        return uf * q_lo
    return uf * q_hi


@wp.kernel
def advect_upwind_3d(
    q:  wp.array(dtype=wp.float32, ndim=3),    # padded (N+2)³
    u:  wp.array(dtype=wp.float32, ndim=3),    # padded (N+2)³
    v:  wp.array(dtype=wp.float32, ndim=3),
    w:  wp.array(dtype=wp.float32, ndim=3),
    dt: wp.float32, inv_h: wp.float32,
    out: wp.array(dtype=wp.float32, ndim=3),   # interior N³
):
    i, j, k = wp.tid()
    ci = i + 1; cj = j + 1; ck = k + 1
    qc = q[ci, cj, ck]

    # x faces
    uW = 0.5 * (u[ci - 1, cj, ck] + u[ci, cj, ck])
    uE = 0.5 * (u[ci, cj, ck] + u[ci + 1, cj, ck])
    fW = _flux(uW, q[ci - 1, cj, ck], qc)
    fE = _flux(uE, qc, q[ci + 1, cj, ck])

    # y faces
    vS = 0.5 * (v[ci, cj - 1, ck] + v[ci, cj, ck])
    vN = 0.5 * (v[ci, cj, ck] + v[ci, cj + 1, ck])
    fS = _flux(vS, q[ci, cj - 1, ck], qc)
    fN = _flux(vN, qc, q[ci, cj + 1, ck])

    # z faces
    wB = 0.5 * (w[ci, cj, ck - 1] + w[ci, cj, ck])
    wT = 0.5 * (w[ci, cj, ck] + w[ci, cj, ck + 1])
    fB = _flux(wB, q[ci, cj, ck - 1], qc)
    fT = _flux(wT, qc, q[ci, cj, ck + 1])

    div = (fE - fW + fN - fS + fT - fB) * inv_h
    out[i, j, k] = qc - dt * div


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch reference of the identical scheme (validation oracle)
# ─────────────────────────────────────────────────────────────────────────────

def advect_upwind_torch(q_pad, u_pad, v_pad, w_pad, dt, inv_h):
    """Vectorised reference; q_pad/... padded (N+2)³. Returns interior N³."""
    c  = (slice(1, -1),) * 3
    qc = q_pad[c]

    def faces(a_pad, ax):
        # adjacent cell-centred values along axis ax (lo neighbour, hi neighbour)
        sl_lo = [slice(1, -1)] * 3; sl_lo[ax] = slice(0, -2)
        sl_hi = [slice(1, -1)] * 3; sl_hi[ax] = slice(2, None)
        return a_pad[tuple(sl_lo)], a_pad[tuple(sl_hi)]

    out = qc.clone()
    for ax, vel in enumerate((u_pad, v_pad, w_pad)):
        q_lo, q_hi = faces(q_pad, ax)
        u_lo, u_hi = faces(vel, ax)
        uc = vel[c]
        uW = 0.5 * (u_lo + uc); uE = 0.5 * (uc + u_hi)
        fW = torch.where(uW > 0, uW * q_lo, uW * qc)
        fE = torch.where(uE > 0, uE * qc,  uE * q_hi)
        out = out - dt * (fE - fW) * inv_h
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  HIGH-ORDER LIMITER PORT — faithful single-source replica of the native fused
#  `advect_flux_add` CUDA kernel (src/kernels/csrc/cuda/advection_flux.cu).
#
#  The native op is called once per (velocity component i, spatial direction d):
#      advect_flux_add(fv, p, rhs, dt_dh, C_courant, scheme_id, face_dim)
#  and accumulates IN PLACE:   rhs[i_fd] += dt_dh * (F_left - F_right)
#  where, for the interior cell i_fd in [0, Nfd-3]:
#      F_left  = flux at global face f_L = i_fd
#      F_right = flux at global face f_R = i_fd + 1
#
#  Layout (from _field_for_flux / _face_vel):
#      p  : (Nfd,   Nt1[, Nt2]) — FULL on face_dim, interior on the rest
#      fv : (Nfd-1, Nt1[, Nt2])
#      rhs: (Ni_0, Ni_1[, Ni_2]) in ORIGINAL grid-dim order, C-contiguous
#           → rhs.stride(face_dim) is NOT Nt1*Nt2 unless face_dim is outermost,
#             so the rhs strides are passed SEPARATELY (HANDOFF caveat).
#
#  Dtype-generic (single source): the kernel/scheme funcs are written over a
#  Warp generic float (``Any``); float literals are materialised in the bound
#  type via ``type(x)(literal)`` and ``wp.overload`` registers the float32 AND
#  float64 specialisations.  float64 lands on bit-parity with the native op
#  (AT_DISPATCH_FLOATING_TYPES → scalar_t = double); the codegen for f64 is
#  unchanged from the original concrete kernel, so the existing parity tests stay
#  bit-identical.  float32 is for an f32 solver (the native op then runs
#  scalar_t = float, so f32 parity is at single precision).
#  Flat 1-D array addressing + explicit element strides mirror the native
#  pointer arithmetic (lesson 2/13); the same kernel serves 2-D (Nt2=1,
#  s_t2=0) and 3-D, exactly like the `.cu`.
#
#  Scheme IDs (match _CUDA_SCHEME_IDS in advection.py and the .cu enum):
#      0 = QUICK   1 = ABDQUICKEST   2 = vanLeer   3 = CDS   4 = CUBISTA
# ═════════════════════════════════════════════════════════════════════════════

_AD_TINY_F64 = 1e-30


@wp.func
def _median3(a: Any, b: Any, c: Any):
    # max(min(a,b), min(max(a,b), c)) — native median3
    return wp.max(wp.min(a, b), wp.min(wp.max(a, b), c))


# All five scheme funcs share the uniform signature (u, c, d, C) — C is the
# Courant number, used only by ABDQUICKEST and ignored by the rest.  The uniform
# signature lets a single kernel template (`_make_flux_kernel`) close over any
# one of them, giving COMPILE-TIME scheme specialization (one kernel per scheme,
# the scheme call fully inlined) — the Warp analogue of the native
# `template <int scheme_id>` dispatch.  A runtime `if scheme_id==…` branch keeps
# all five code paths live in one kernel → ~1.4× slower than native in 3-D;
# specialization closes that to ~1.0× (HANDOFF lesson 17).


@wp.func
def _scheme_quick(u: Any, c: Any, d: Any, C: Any):
    inner = _median3(type(c)(10.0) * c - type(c)(9.0) * u, c, d)
    outer = (type(c)(5.0) * c + type(c)(2.0) * d - u) / type(c)(6.0)
    return _median3(outer, c, inner)


@wp.func
def _scheme_abdquickest(u: Any, c: Any, d: Any, C: Any):
    zero = type(c)(0.0); half = type(c)(0.5)
    one = type(c)(1.0); two = type(c)(2.0); three = type(c)(3.0)
    denom = d - c
    res = c
    if wp.abs(denom) >= type(c)(_AD_TINY_F64):
        rf = (c - u) / denom
        C2 = C * C
        C_upper = two * (one - C)
        scale = (one - C2) / (three - three * C)
        offset = (two + C2 - three * C) / (three - three * C)
        psi = wp.min(rf * scale + offset, C_upper)
        psi = wp.min(psi, rf * C_upper)
        psi = wp.max(psi, zero)
        res = c + half * denom * psi
    return res


@wp.func
def _scheme_van_leer(u: Any, c: Any, d: Any, C: Any):
    denom = d - c
    res = c
    if wp.abs(denom) >= type(c)(_AD_TINY_F64):
        rf = (c - u) / denom
        abs_rf = wp.abs(rf)
        psi = (rf + abs_rf) / (type(c)(1.0) + abs_rf)
        res = c + type(c)(0.5) * denom * psi
    return res


@wp.func
def _scheme_cds(u: Any, c: Any, d: Any, C: Any):
    return type(c)(0.5) * (c + d)


@wp.func
def _scheme_cubista(u: Any, c: Any, d: Any, C: Any):
    zero = type(c)(0.0); half = type(c)(0.5)
    denom = d - c
    res = c
    if wp.abs(denom) >= type(c)(_AD_TINY_F64):
        rf = (c - u) / denom
        psi = wp.min(type(c)(0.75) * rf + type(c)(0.25), type(c)(1.5))
        psi = wp.min(psi, rf * type(c)(1.5))
        psi = wp.max(psi, zero)
        res = c + half * denom * psi
    return res


def _make_flux_kernel(scheme):
    """Build a scheme-SPECIALIZED ``advect_flux_add`` kernel.

    The kernel closes over the ``scheme`` ``@wp.func`` (Warp resolves the closure
    at kernel-creation time and inlines it), so there is no runtime scheme branch
    and the four other scheme code paths are not even present — one compiled
    kernel per scheme, exactly like native's ``template <int scheme_id>``.
    """
    @wp.kernel
    def advect_flux_add_kernel(
        p:   wp.array(dtype=Any),   # flat storage view (full on face_dim)
        fv:  wp.array(dtype=Any),   # flat storage view
        rhs: wp.array(dtype=Any),   # flat storage view, accumulated in place
        Nfd: wp.int32, Nt1: wp.int32, Nt2: wp.int32,
        p_s_fd: wp.int32,   p_s_t1: wp.int32,   p_s_t2: wp.int32,
        fv_s_fd: wp.int32,  fv_s_t1: wp.int32,  fv_s_t2: wp.int32,
        rhs_s_fd: wp.int32, rhs_s_t1: wp.int32, rhs_s_t2: wp.int32,
        dt_dh: Any, C: Any,
    ):
        gid = wp.tid()
        Ni_fd = Nfd - 2
        NT = Nt1 * Nt2
        if gid >= Ni_fd * NT:
            return

        # flat decode: t2 fastest (matches native threadIdx.x → t2 coalescing)
        i_fd = gid / NT
        rem = gid - i_fd * NT
        i_t1 = rem / Nt2
        i_t2 = rem - i_t1 * Nt2

        tp = i_t1 * p_s_t1 + i_t2 * p_s_t2
        tfv = i_t1 * fv_s_t1 + i_t2 * fv_s_t2

        f_L = i_fd
        f_R = i_fd + 1
        fv_L = fv[tfv + f_L * fv_s_fd]
        fv_R = fv[tfv + f_R * fv_s_fd]
        zero = type(fv_L)(0.0)

        # ---- left face flux ----
        pc = p[tp + f_L * p_s_fd]
        pd = p[tp + (f_L + 1) * p_s_fd]
        F_L = type(pc)(0.0)
        if fv_L > zero:
            if f_L == 0:
                # lo boundary: no upstream → CDS fallback
                F_L = fv_L * type(pc)(0.5) * (pc + pd)
            else:
                pu = p[tp + (f_L - 1) * p_s_fd]
                F_L = fv_L * scheme(pu, pc, pd, C)
        else:
            # negative flow: f_L is never the hi boundary (max = Nfd-3 < Nfd-2)
            pdd = p[tp + (f_L + 2) * p_s_fd]
            F_L = fv_L * scheme(pdd, pd, pc, C)

        # ---- right face flux ----
        pc2 = p[tp + f_R * p_s_fd]
        pd2 = p[tp + (f_R + 1) * p_s_fd]
        F_R = type(pc2)(0.0)
        if fv_R > zero:
            # positive flow: f_R is never the lo boundary (min = 1 > 0)
            pu2 = p[tp + (f_R - 1) * p_s_fd]
            F_R = fv_R * scheme(pu2, pc2, pd2, C)
        else:
            if f_R == Nfd - 2:
                # hi boundary: no downstream → CDS fallback
                F_R = fv_R * type(pc2)(0.5) * (pc2 + pd2)
            else:
                pdd2 = p[tp + (f_R + 2) * p_s_fd]
                F_R = fv_R * scheme(pdd2, pd2, pc2, C)

        ridx = i_fd * rhs_s_fd + i_t1 * rhs_s_t1 + i_t2 * rhs_s_t2
        rhs[ridx] = rhs[ridx] + dt_dh * (F_L - F_R)

    return advect_flux_add_kernel


# One compiled kernel per scheme_id (keys match _CUDA_SCHEME_IDS / the .cu enum).
_FLUX_KERNELS = {
    0: _make_flux_kernel(_scheme_quick),        # QUICK
    1: _make_flux_kernel(_scheme_abdquickest),  # ABDQUICKEST
    2: _make_flux_kernel(_scheme_van_leer),     # vanLeer
    3: _make_flux_kernel(_scheme_cds),          # CDS
    4: _make_flux_kernel(_scheme_cubista),      # CUBISTA
}

# Register float32 + float64 specialisations for every scheme kernel.
for _k in _FLUX_KERNELS.values():
    for _dt in (wp.float32, wp.float64):
        _A = wp.array(dtype=_dt)
        wp.overload(_k, {"p": _A, "fv": _A, "rhs": _A, "dt_dh": _dt, "C": _dt})


# ─────────────────────────────────────────────────────────────────────────────
#  Host wrapper: mirror the native op signature, mutate rhs in place.
# ─────────────────────────────────────────────────────────────────────────────

def _wp_device(t: torch.Tensor) -> str:
    return f"cuda:{t.device.index}" if t.is_cuda else "cpu"


def _wp_dtype(t: torch.Tensor):
    return wp.float64 if t.dtype == torch.float64 else wp.float32


def _flat(t: torch.Tensor):
    """Zero-copy flat Warp view (f32/f64) over t's storage, honouring the
    storage offset so element 0 == t's logical [0,0,..] (native pointer base)."""
    assert t.dtype in (torch.float64, torch.float32), "warp advection: f32/f64 only"
    elem = t.element_size()
    remaining = (t.untyped_storage().nbytes() - t.storage_offset() * elem) // elem
    return wp.array(ptr=t.data_ptr(), dtype=_wp_dtype(t),
                    shape=(int(remaining),), device=_wp_device(t))


def advect_flux_add_warp(fv_t, p_t, rhs_t, dt_dh, C_courant, scheme_id, face_dim):
    """Warp port of the retired native ``advect_flux_add`` op.

    Accumulates ``rhs_t[i_fd] += dt_dh * (F_left - F_right)`` in place for one
    (velocity component, spatial direction) pair.  Faithful to the original C++
    wrapper's stride extraction (transverse dims gathered in order, skipping
    face_dim; rhs strides taken in ORIGINAL grid-dim order).
    """
    ndim = p_t.dim()
    assert ndim in (2, 3)
    Nfd = p_t.size(face_dim)
    Ni_fd = Nfd - 2
    if Ni_fd <= 0:
        return

    t_dims = [d for d in range(ndim) if d != face_dim]
    Nt1 = p_t.size(t_dims[0]) if len(t_dims) > 0 else 1
    Nt2 = p_t.size(t_dims[1]) if len(t_dims) > 1 else 1

    p_s_fd = p_t.stride(face_dim)
    p_s_t1 = p_t.stride(t_dims[0]) if len(t_dims) > 0 else 0
    p_s_t2 = p_t.stride(t_dims[1]) if len(t_dims) > 1 else 0
    fv_s_fd = fv_t.stride(face_dim)
    fv_s_t1 = fv_t.stride(t_dims[0]) if len(t_dims) > 0 else 0
    fv_s_t2 = fv_t.stride(t_dims[1]) if len(t_dims) > 1 else 0
    rhs_s_fd = rhs_t.stride(face_dim)
    rhs_s_t1 = rhs_t.stride(t_dims[0]) if len(t_dims) > 0 else 0
    rhs_s_t2 = rhs_t.stride(t_dims[1]) if len(t_dims) > 1 else 0

    kernel = _FLUX_KERNELS[int(scheme_id)]   # compile-time scheme specialization
    dev = _wp_device(p_t)
    wpf = _wp_dtype(p_t)
    n_threads = Ni_fd * Nt1 * Nt2
    wp.launch(
        kernel,
        dim=n_threads,
        inputs=[
            _flat(p_t), _flat(fv_t), _flat(rhs_t),
            int(Nfd), int(Nt1), int(Nt2),
            int(p_s_fd), int(p_s_t1), int(p_s_t2),
            int(fv_s_fd), int(fv_s_t1), int(fv_s_t2),
            int(rhs_s_fd), int(rhs_s_t1), int(rhs_s_t2),
            wpf(float(dt_dh)), wpf(float(C_courant)),
        ],
        device=dev,
    )
