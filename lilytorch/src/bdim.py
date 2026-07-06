"""Warp single-source **bdim_forcing (3-D)** — fused BDIM2 coefficient + FD normals.

Port of the native ``bdim_forcing_3d`` CUDA kernel
(``src/kernels/csrc/cuda/streaming_sdf.cu``, ``bdim_one_axis_3d`` device helper).
For every cell in the dirty AABB,
and for each of the three staggered face grids, the kernel:

  1. reads the face SDF ``phi = sdf[g]`` and forms the smoothed Heaviside /
     delta weights ``mu0, mu1`` (matches ``body._mu_normals_batched`` exactly);
  2. forms the **unit normal** ``n = grad(phi)/|grad(phi)|`` by central
     differences (clamped neighbours at the grid edge);
  3. takes the BDIM2 normal derivative ``nd = n . grad(phi' - body_vel)``;
  4. writes the persistent velocity ``u0 = mu0*(u' - body) + body + mu1*nd``
     and the Weymouth-&-Yue Poisson coefficient ``c = (mu0_proj? dt*mu0 : dt)/rho_f``.

The *whole point* of this kernel (kernel-mode's headline memory claim) is that
``mu0``, ``mu1`` and the three normal components live **only in registers** —
they are never materialised as global tensors (which is what the PyTorch
"python path" does, inflating peak memory; see ``bench_bdim_memory.py``).  This
file demonstrates Warp keeps the same register-resident profile as the native
CUDA kernel: no global scratch is allocated, only the (pre-existing) output
tensors are written.

**Precision (single source, both dtypes).**  The funcs/kernel are Warp generics
(``Any``): float literals are materialised in the bound element type via
``type(x)(literal)`` and ``wp.overload`` registers the float32 *and* float64
specialisations (Warp 1.14 needs the dtypes pre-registered).  float64 lands on
bit-parity with the native op (``scalar_t = double``) — the codegen is
unchanged from the original concrete kernel, so existing f64 parity stays
bit-identical; float32 matches native f32 to single precision.

**Coefficient layout (HANDOFF lesson 9, 3-D-specific).**  Unlike 2-D, the 3-D
native writes the Poisson coefficient at the *face-grid* flat offset
``(i-1)*c_stride_i + (j-1)*c_stride_j + (k-1)`` (``ch`` is sized
``(Ngx-1, Ngy-2, Ngz-2)`` etc.).  This kernel reproduces that exactly; the 3-D
parity test asserts the face-grid layout.

Read the ``.cu`` before changing anything; the discretization is matched line
for line (HANDOFF rule).
"""
from __future__ import annotations

from typing import Any

import warp as wp
import torch

wp.init()

# native uses scalar_t(M_PI) with scalar_t = double → the IEEE-754 double M_PI.
_PI_F64 = 3.141592653589793


# ─────────────────────────────────────────────────────────────────────────────
#  Per-axis device helper (faithful port of bdim_one_axis_3d)
#
#  All grid arrays are FLAT row-major (Ngx, Ngy, Ngz) → base index + ±stride
#  offsets, mirroring the native CUDA addressing (HANDOFF lesson 2).
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def _mu0_mu1(phi: Any, eps: Any):
    """Smoothed Heaviside (mu0) and first moment (mu1) of phi/eps."""
    zero = type(phi)(0.0)
    mu0 = zero
    mu1 = zero
    if phi <= -eps:
        mu0 = zero
        mu1 = zero
    elif phi >= eps:
        mu0 = type(phi)(1.0)
        mu1 = zero
    else:
        half = type(phi)(0.5)
        one = type(phi)(1.0)
        two = type(phi)(2.0)
        quarter = type(phi)(0.25)
        pi = type(phi)(_PI_F64)
        deps = phi / eps
        s = wp.sin(pi * deps)
        c = wp.cos(pi * deps)
        mu0 = half * (one + deps + s / pi)
        mu1 = eps * (
            quarter
            - quarter * deps * deps
            - (s * deps + (one + c) / pi) / (two * pi)
        )
    return mu0, mu1


@wp.func
def bdim_one_axis_3d(
    phi_prime: wp.array(dtype=Any),
    sdf: wp.array(dtype=Any),
    body: wp.array(dtype=Any),
    phi_out: wp.array(dtype=Any),   # written
    c_out: wp.array(dtype=Any),     # written (face grid)
    eps: Any, rho_f: Any, dt: Any, inv_2h: Any,
    Ngx: wp.int32, Ngy: wp.int32, Ngz: wp.int32,
    i: wp.int32, j: wp.int32, k: wp.int32,
    c_stride_i: wp.int32, c_stride_j: wp.int32,
    c_hi_i: wp.int32, c_hi_j: wp.int32, c_hi_k: wp.int32,
    mu0_proj: wp.int32,
):
    zero = type(eps)(0.0)
    one = type(eps)(1.0)
    stride_i = Ngy * Ngz
    stride_j = Ngz
    g = i * stride_i + j * stride_j + k

    # Clamped neighbour indices (body at the grid edge can't read OOB).
    im = i - 1
    if i <= 0:
        im = 0
    ip = i + 1
    if i >= Ngx - 1:
        ip = Ngx - 1
    jm = j - 1
    if j <= 0:
        jm = 0
    jp = j + 1
    if j >= Ngy - 1:
        jp = Ngy - 1
    km = k - 1
    if k <= 0:
        km = 0
    kp = k + 1
    if k >= Ngz - 1:
        kp = Ngz - 1

    g_im = im * stride_i + j * stride_j + k
    g_ip = ip * stride_i + j * stride_j + k
    g_jm = i * stride_i + jm * stride_j + k
    g_jp = i * stride_i + jp * stride_j + k
    g_km = i * stride_i + j * stride_j + km
    g_kp = i * stride_i + j * stride_j + kp

    phi = sdf[g]
    mu0, mu1 = _mu0_mu1(phi, eps)

    # Unit normal n = grad(phi)/|grad(phi)|, central differences.
    nx = (sdf[g_ip] - sdf[g_im]) * inv_2h
    ny = (sdf[g_jp] - sdf[g_jm]) * inv_2h
    nz = (sdf[g_kp] - sdf[g_km]) * inv_2h
    nn = wp.sqrt(nx * nx + ny * ny + nz * nz)
    if nn > zero:
        inv_nn = one / nn
        nx = nx * inv_nn
        ny = ny * inv_nn
        nz = nz * inv_nn

    # BDIM2 normal derivative of (phi_prime - body); zero at grid boundary.
    b_c = body[g]
    diff_c = phi_prime[g] - b_c

    ddx = zero
    if i > 0 and i < Ngx - 1:
        ddx = ((phi_prime[g_ip] - body[g_ip]) -
               (phi_prime[g_im] - body[g_im])) * inv_2h
    ddy = zero
    if j > 0 and j < Ngy - 1:
        ddy = ((phi_prime[g_jp] - body[g_jp]) -
               (phi_prime[g_jm] - body[g_jm])) * inv_2h
    ddz = zero
    if k > 0 and k < Ngz - 1:
        ddz = ((phi_prime[g_kp] - body[g_kp]) -
               (phi_prime[g_km] - body[g_km])) * inv_2h
    nd = nx * ddx + ny * ddy + nz * ddz

    phi_out[g] = mu0 * diff_c + b_c + mu1 * nd

    if i >= 1 and j >= 1 and k >= 1 and i <= c_hi_i and j <= c_hi_j and k <= c_hi_k:
        # Poisson coefficient.
        mu0_c = mu0
        cval = dt / rho_f
        if mu0_proj != 0:
            cval = dt * mu0_c / rho_f
        c_out[(i - 1) * c_stride_i + (j - 1) * c_stride_j + (k - 1)] = cval


@wp.kernel
def bdim_forcing_3d_kernel(
    u_prime: wp.array(dtype=Any),
    v_prime: wp.array(dtype=Any),
    w_prime: wp.array(dtype=Any),
    sdf_u: wp.array(dtype=Any),
    sdf_v: wp.array(dtype=Any),
    sdf_w: wp.array(dtype=Any),
    body_u: wp.array(dtype=Any),
    body_v: wp.array(dtype=Any),
    body_w: wp.array(dtype=Any),
    u0: wp.array(dtype=Any),
    v0: wp.array(dtype=Any),
    w0: wp.array(dtype=Any),
    ch: wp.array(dtype=Any),
    cv: wp.array(dtype=Any),
    cw: wp.array(dtype=Any),
    eps: Any, rho_f: Any, dt: Any, inv_2h: Any,
    Ngx: wp.int32, Ngy: wp.int32, Ngz: wp.int32,
    di0: wp.int32, dj0: wp.int32, dk0: wp.int32,
    mu0_proj: wp.int32,
    dAj: wp.int32, dAk: wp.int32,
):
    # FLAT 1-D launch + native's exact decode (k fastest → coalesced global
    # stores).  Mirroring the native thread→cell map (`dk = local % dAk`) — not
    # just flat *array* addressing but a flat *launch* — closes a ~1.15×→~1.05×
    # gap vs the 3-D `wp.tid()` launch, whose thread-ID mapping coalesces worse.
    local = wp.tid()
    dk = local % dAk
    rem = local / dAk
    dj = rem % dAj
    di = rem / dAj
    i = di0 + di
    j = dj0 + dj
    k = dk0 + dk

    # ch: x-face grid (Ngx-1, Ngy-2, Ngz-2)
    bdim_one_axis_3d(
        u_prime, sdf_u, body_u, u0, ch,
        eps, rho_f, dt, inv_2h, Ngx, Ngy, Ngz, i, j, k,
        (Ngy - 2) * (Ngz - 2), (Ngz - 2), Ngx - 1, Ngy - 2, Ngz - 2, mu0_proj)
    # cv: y-face grid (Ngx-2, Ngy-1, Ngz-2)
    bdim_one_axis_3d(
        v_prime, sdf_v, body_v, v0, cv,
        eps, rho_f, dt, inv_2h, Ngx, Ngy, Ngz, i, j, k,
        (Ngy - 1) * (Ngz - 2), (Ngz - 2), Ngx - 2, Ngy - 1, Ngz - 2, mu0_proj)
    # cw: z-face grid (Ngx-2, Ngy-2, Ngz-1)
    bdim_one_axis_3d(
        w_prime, sdf_w, body_w, w0, cw,
        eps, rho_f, dt, inv_2h, Ngx, Ngy, Ngz, i, j, k,
        (Ngy - 2) * (Ngz - 1), (Ngz - 1), Ngx - 2, Ngy - 2, Ngz - 1, mu0_proj)


# Register float32 + float64 specialisations (generic args only).
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(bdim_forcing_3d_kernel, {
        "u_prime": _A, "v_prime": _A, "w_prime": _A,
        "sdf_u": _A, "sdf_v": _A, "sdf_w": _A,
        "body_u": _A, "body_v": _A, "body_w": _A,
        "u0": _A, "v0": _A, "w0": _A, "ch": _A, "cv": _A, "cw": _A,
        "eps": _dt, "rho_f": _dt, "dt": _dt, "inv_2h": _dt,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Python launch wrappers (mirror the native ops.py signatures)
# ─────────────────────────────────────────────────────────────────────────────

def _wdev(t: torch.Tensor) -> str:
    return "cuda:0" if t.device.type == "cuda" else "cpu"


def _wp_dtype(t: torch.Tensor):
    return wp.float64 if t.dtype == torch.float64 else wp.float32


def bdim_forcing_3d_warp(
        u_prime, v_prime, w_prime,
        sdf_u, sdf_v, sdf_w,
        body_u, body_v, body_w,
        u0, v0, w0, ch, cv, cw,
        eps, rho_f, dt, h_grid,
        dirty_i0, dirty_j0, dirty_k0,
        dirty_Ai, dirty_Aj, dirty_Ak,
        mu0_projection=1):
    """Warp port of ``bdim_forcing_3d``.  Writes ``u0/v0/w0`` and ``ch/cv/cw``
    in place inside the dirty AABB; returns nothing (mirrors the native op).
    Generic in dtype (f32/f64), selected from ``u0``.

    The output tensors are wrapped zero-copy via ``wp.from_torch``; the kernel
    allocates **no** global scratch — mu/normals stay in registers.
    """
    if dirty_Ai * dirty_Aj * dirty_Ak <= 0:
        return
    wdev = _wdev(u0)
    wpf = _wp_dtype(u0)
    Ngx, Ngy, Ngz = u0.shape

    def f(x):
        return wp.from_torch(x.reshape(-1))  # zero-copy view

    wp.launch(
        bdim_forcing_3d_kernel,
        dim=int(dirty_Ai) * int(dirty_Aj) * int(dirty_Ak),
        inputs=[
            f(u_prime), f(v_prime), f(w_prime),
            f(sdf_u), f(sdf_v), f(sdf_w),
            f(body_u), f(body_v), f(body_w),
            f(u0), f(v0), f(w0), f(ch), f(cv), f(cw),
            wpf(eps), wpf(rho_f), wpf(dt),
            wpf(0.5 / h_grid),
            int(Ngx), int(Ngy), int(Ngz),
            int(dirty_i0), int(dirty_j0), int(dirty_k0),
            int(mu0_projection),
            int(dirty_Aj), int(dirty_Ak),
        ],
        device=wdev)


# ═════════════════════════════════════════════════════════════════════════════
#  2-D VARIANT — BDIM2 forcing + Poisson coefficients (bdim_forcing_2d)
#  Merged from the former bdim_2d.py.  z-axis-stripped analogue of the 3-D
#  kernel above; shares _PI_F64/_wdev/_wp_dtype (above).
# ═════════════════════════════════════════════════════════════════════════════
@wp.func
def _mu0_mu1_2d(phi: Any, eps: Any):
    zero = type(phi)(0.0)
    mu0 = zero
    mu1 = zero
    if phi <= -eps:
        mu0 = zero
        mu1 = zero
    elif phi >= eps:
        mu0 = type(phi)(1.0)
        mu1 = zero
    else:
        half = type(phi)(0.5)
        one = type(phi)(1.0)
        two = type(phi)(2.0)
        quarter = type(phi)(0.25)
        pi = type(phi)(_PI_F64)
        deps = phi / eps
        s = wp.sin(pi * deps)
        c = wp.cos(pi * deps)
        mu0 = half * (one + deps + s / pi)
        mu1 = eps * (
            quarter
            - quarter * deps * deps
            - (s * deps + (one + c) / pi) / (two * pi)
        )
    return mu0, mu1


@wp.func
def bdim_one_axis_2d(
    phi_prime: wp.array(dtype=Any),
    sdf: wp.array(dtype=Any),
    body: wp.array(dtype=Any),
    phi_out: wp.array(dtype=Any),   # written (full grid g)
    c_out: wp.array(dtype=Any),     # written (full grid g)
    eps: Any, rho_f: Any, dt: Any, inv_2h: Any,
    Ngx: wp.int32, Ngy: wp.int32,
    i: wp.int32, j: wp.int32,
    mu0_proj: wp.int32,
):
    zero = type(eps)(0.0)
    one = type(eps)(1.0)
    stride_i = Ngy
    g = i * stride_i + j

    im = i - 1
    if i <= 0:
        im = 0
    ip = i + 1
    if i >= Ngx - 1:
        ip = Ngx - 1
    jm = j - 1
    if j <= 0:
        jm = 0
    jp = j + 1
    if j >= Ngy - 1:
        jp = Ngy - 1

    g_im = im * stride_i + j
    g_ip = ip * stride_i + j
    g_jm = i * stride_i + jm
    g_jp = i * stride_i + jp

    phi = sdf[g]
    mu0, mu1 = _mu0_mu1_2d(phi, eps)

    nx = (sdf[g_ip] - sdf[g_im]) * inv_2h
    ny = (sdf[g_jp] - sdf[g_jm]) * inv_2h
    nn = wp.sqrt(nx * nx + ny * ny)
    if nn > zero:
        inv_nn = one / nn
        nx = nx * inv_nn
        ny = ny * inv_nn

    b_c = body[g]
    diff_c = phi_prime[g] - b_c

    ddx = zero
    if i > 0 and i < Ngx - 1:
        ddx = ((phi_prime[g_ip] - body[g_ip]) -
               (phi_prime[g_im] - body[g_im])) * inv_2h
    ddy = zero
    if j > 0 and j < Ngy - 1:
        ddy = ((phi_prime[g_jp] - body[g_jp]) -
               (phi_prime[g_jm] - body[g_jm])) * inv_2h
    nd = nx * ddx + ny * ddy

    phi_out[g] = mu0 * diff_c + b_c + mu1 * nd

    # Poisson coefficient at the SAME full-grid index g (2-D: no face-grid offset).
    mu0_c = mu0
    cval = dt / rho_f
    if mu0_proj != 0:
        cval = dt * mu0_c / rho_f
    c_out[g] = cval


@wp.kernel
def bdim_forcing_2d_kernel(
    u_prime: wp.array(dtype=Any),
    v_prime: wp.array(dtype=Any),
    sdf_u: wp.array(dtype=Any),
    sdf_v: wp.array(dtype=Any),
    body_u: wp.array(dtype=Any),
    body_v: wp.array(dtype=Any),
    u0: wp.array(dtype=Any),
    v0: wp.array(dtype=Any),
    ch: wp.array(dtype=Any),
    cv: wp.array(dtype=Any),
    eps: Any, rho_f: Any, dt: Any, inv_2h: Any,
    Ngx: wp.int32, Ngy: wp.int32,
    di0: wp.int32, dj0: wp.int32,
    mu0_proj: wp.int32,
    dAj: wp.int32,
):
    # Flat 1-D launch, native's exact decode (j fastest → coalesced).
    local = wp.tid()
    dj = local % dAj
    di = local / dAj
    i = di0 + di
    j = dj0 + dj

    bdim_one_axis_2d(
        u_prime, sdf_u, body_u, u0, ch,
        eps, rho_f, dt, inv_2h, Ngx, Ngy, i, j, mu0_proj)
    bdim_one_axis_2d(
        v_prime, sdf_v, body_v, v0, cv,
        eps, rho_f, dt, inv_2h, Ngx, Ngy, i, j, mu0_proj)


# Register float32 + float64 specialisations (generic args only).
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(bdim_forcing_2d_kernel, {
        "u_prime": _A, "v_prime": _A, "sdf_u": _A, "sdf_v": _A,
        "body_u": _A, "body_v": _A, "u0": _A, "v0": _A, "ch": _A, "cv": _A,
        "eps": _dt, "rho_f": _dt, "dt": _dt, "inv_2h": _dt,
    })



def bdim_forcing_2d_warp(
        u_prime, v_prime, sdf_u, sdf_v, body_u, body_v,
        u0, v0, ch, cv,
        eps, rho_f, dt, h_grid,
        dirty_i0, dirty_j0, dirty_Ai, dirty_Aj,
        mu0_projection=1):
    """Warp port of ``bdim_forcing_2d``.
    Writes u0/v0 and ch/cv (full-grid) in place inside the dirty AABB.
    Generic in dtype (f32/f64), selected from ``u0``."""
    if int(dirty_Ai) * int(dirty_Aj) <= 0:
        return
    wdev = _wdev(u0)
    wpf = _wp_dtype(u0)
    Ngx, Ngy = u0.shape

    def f(x):
        return wp.from_torch(x.reshape(-1))

    wp.launch(
        bdim_forcing_2d_kernel,
        dim=int(dirty_Ai) * int(dirty_Aj),
        inputs=[
            f(u_prime), f(v_prime), f(sdf_u), f(sdf_v),
            f(body_u), f(body_v), f(u0), f(v0), f(ch), f(cv),
            wpf(eps), wpf(rho_f), wpf(dt),
            wpf(0.5 / h_grid),
            int(Ngx), int(Ngy),
            int(dirty_i0), int(dirty_j0),
            int(mu0_projection),
            int(dirty_Aj),
        ],
        device=wdev)
