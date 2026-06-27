"""Warp single-source **Kernel B** — fused BDIM2 coefficient + FD normals.

Port of the native ``bdim_coeff_3d`` / ``bdim_coeff_sigma_3d`` CUDA kernels
(``src/kernels/csrc/cuda/streaming_sdf.cu``, ``bdim_one_axis_3d`` /
``bdim_one_axis_sigma_3d`` device helpers).  For every cell in the dirty AABB,
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

Everything runs in **float64** to land on bit-parity with the native op
(``AT_DISPATCH_FLOATING_TYPES`` → ``scalar_t = double`` when fed float64
tensors).  The SAME ``@wp.kernel`` runs on Warp ``device="cpu"`` (→ C++/OpenMP)
and ``"cuda:0"``.

Read the ``.cu`` before changing anything; the discretization is matched line
for line (HANDOFF rule).
"""
from __future__ import annotations

import warp as wp
import torch

wp.init()

_ZERO = wp.constant(wp.float64(0.0))
_ONE = wp.constant(wp.float64(1.0))
_HALF = wp.constant(wp.float64(0.5))
_TWO = wp.constant(wp.float64(2.0))
_QUARTER = wp.constant(wp.float64(0.25))
# native uses scalar_t(M_PI) with scalar_t = double → the IEEE-754 double M_PI.
_PI = wp.constant(wp.float64(3.141592653589793))
_MASK32 = wp.constant(wp.int64(0xFFFFFFFF))


# ─────────────────────────────────────────────────────────────────────────────
#  Per-axis device helper (faithful port of bdim_one_axis_3d)
#
#  All grid arrays are FLAT row-major (Ngx, Ngy, Ngz) → base index + ±stride
#  offsets, mirroring the native CUDA addressing (HANDOFF lesson 2).
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def _mu0_mu1(phi: wp.float64, eps: wp.float64):
    """Smoothed Heaviside (mu0) and first moment (mu1) of phi/eps."""
    mu0 = _ZERO
    mu1 = _ZERO
    if phi <= -eps:
        mu0 = _ZERO
        mu1 = _ZERO
    elif phi >= eps:
        mu0 = _ONE
        mu1 = _ZERO
    else:
        deps = phi / eps
        s = wp.sin(_PI * deps)
        c = wp.cos(_PI * deps)
        mu0 = _HALF * (_ONE + deps + s / _PI)
        mu1 = eps * (
            _QUARTER
            - _QUARTER * deps * deps
            - (s * deps + (_ONE + c) / _PI) / (_TWO * _PI)
        )
    return mu0, mu1


@wp.func
def _mu0_only(phi: wp.float64, eps: wp.float64) -> wp.float64:
    """Smoothed Heaviside mu0 alone (used for the BDIM-σ shifted coefficient)."""
    res = _ZERO
    if phi <= -eps:
        res = _ZERO
    elif phi >= eps:
        res = _ONE
    else:
        deps = phi / eps
        res = _HALF * (_ONE + deps + wp.sin(_PI * deps) / _PI)
    return res


@wp.func
def bdim_one_axis_3d(
    phi_prime: wp.array(dtype=wp.float64),
    sdf: wp.array(dtype=wp.float64),
    body: wp.array(dtype=wp.float64),
    phi_out: wp.array(dtype=wp.float64),   # written
    c_out: wp.array(dtype=wp.float64),     # written (face grid)
    eps: wp.float64, rho_f: wp.float64, dt: wp.float64, inv_2h: wp.float64,
    Ngx: wp.int32, Ngy: wp.int32, Ngz: wp.int32,
    i: wp.int32, j: wp.int32, k: wp.int32,
    c_stride_i: wp.int32, c_stride_j: wp.int32,
    c_hi_i: wp.int32, c_hi_j: wp.int32, c_hi_k: wp.int32,
    mu0_proj: wp.int32,
    # BDIM-σ extras (sigma_on == 0 → ignored, plain bdim_coeff_3d):
    sigma_on: wp.int32,
    key: wp.array(dtype=wp.int64),
    sigma_shifts: wp.array(dtype=wp.float32),
    n_sigma: wp.int32,
    di0: wp.int32, dj0: wp.int32, dk0: wp.int32,
    dAj: wp.int32, dAk: wp.int32,
):
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
    if nn > _ZERO:
        inv_nn = _ONE / nn
        nx = nx * inv_nn
        ny = ny * inv_nn
        nz = nz * inv_nn

    # BDIM2 normal derivative of (phi_prime - body); zero at grid boundary.
    b_c = body[g]
    diff_c = phi_prime[g] - b_c

    ddx = _ZERO
    if i > 0 and i < Ngx - 1:
        ddx = ((phi_prime[g_ip] - body[g_ip]) -
               (phi_prime[g_im] - body[g_im])) * inv_2h
    ddy = _ZERO
    if j > 0 and j < Ngy - 1:
        ddy = ((phi_prime[g_jp] - body[g_jp]) -
               (phi_prime[g_jm] - body[g_jm])) * inv_2h
    ddz = _ZERO
    if k > 0 and k < Ngz - 1:
        ddz = ((phi_prime[g_kp] - body[g_kp]) -
               (phi_prime[g_km] - body[g_km])) * inv_2h
    nd = nx * ddx + ny * ddy + nz * ddz

    phi_out[g] = mu0 * diff_c + b_c + mu1 * nd

    if i >= 1 and j >= 1 and k >= 1 and i <= c_hi_i and j <= c_hi_j and k <= c_hi_k:
        # Poisson coefficient.  BDIM-σ shifts the SDF for the mu0 used here only.
        mu0_c = mu0
        if sigma_on != 0:
            local = (i - di0) * (dAj * dAk) + (j - dj0) * dAk + (k - dk0)
            body_idx = wp.int32(key[local] & _MASK32)
            sigma_shift = _ZERO
            if body_idx < n_sigma:
                sigma_shift = wp.float64(sigma_shifts[body_idx])
            mu0_c = _mu0_only(phi - sigma_shift, eps)
        cval = dt / rho_f
        if mu0_proj != 0:
            cval = dt * mu0_c / rho_f
        c_out[(i - 1) * c_stride_i + (j - 1) * c_stride_j + (k - 1)] = cval


@wp.kernel
def bdim_coeff_3d_kernel(
    u_prime: wp.array(dtype=wp.float64),
    v_prime: wp.array(dtype=wp.float64),
    w_prime: wp.array(dtype=wp.float64),
    sdf_u: wp.array(dtype=wp.float64),
    sdf_v: wp.array(dtype=wp.float64),
    sdf_w: wp.array(dtype=wp.float64),
    body_u: wp.array(dtype=wp.float64),
    body_v: wp.array(dtype=wp.float64),
    body_w: wp.array(dtype=wp.float64),
    u0: wp.array(dtype=wp.float64),
    v0: wp.array(dtype=wp.float64),
    w0: wp.array(dtype=wp.float64),
    ch: wp.array(dtype=wp.float64),
    cv: wp.array(dtype=wp.float64),
    cw: wp.array(dtype=wp.float64),
    eps: wp.float64, rho_f: wp.float64, dt: wp.float64, inv_2h: wp.float64,
    Ngx: wp.int32, Ngy: wp.int32, Ngz: wp.int32,
    di0: wp.int32, dj0: wp.int32, dk0: wp.int32,
    mu0_proj: wp.int32,
    sigma_on: wp.int32,
    key_u: wp.array(dtype=wp.int64),
    key_v: wp.array(dtype=wp.int64),
    key_w: wp.array(dtype=wp.int64),
    sigma_shifts: wp.array(dtype=wp.float32),
    n_sigma: wp.int32,
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
        (Ngy - 2) * (Ngz - 2), (Ngz - 2), Ngx - 1, Ngy - 2, Ngz - 2, mu0_proj,
        sigma_on, key_u, sigma_shifts, n_sigma, di0, dj0, dk0, dAj, dAk)
    # cv: y-face grid (Ngx-2, Ngy-1, Ngz-2)
    bdim_one_axis_3d(
        v_prime, sdf_v, body_v, v0, cv,
        eps, rho_f, dt, inv_2h, Ngx, Ngy, Ngz, i, j, k,
        (Ngy - 1) * (Ngz - 2), (Ngz - 2), Ngx - 2, Ngy - 1, Ngz - 2, mu0_proj,
        sigma_on, key_v, sigma_shifts, n_sigma, di0, dj0, dk0, dAj, dAk)
    # cw: z-face grid (Ngx-2, Ngy-2, Ngz-1)
    bdim_one_axis_3d(
        w_prime, sdf_w, body_w, w0, cw,
        eps, rho_f, dt, inv_2h, Ngx, Ngy, Ngz, i, j, k,
        (Ngy - 2) * (Ngz - 1), (Ngz - 1), Ngx - 2, Ngy - 2, Ngz - 1, mu0_proj,
        sigma_on, key_w, sigma_shifts, n_sigma, di0, dj0, dk0, dAj, dAk)


# ─────────────────────────────────────────────────────────────────────────────
#  Python launch wrappers (mirror the native ops.py signatures)
# ─────────────────────────────────────────────────────────────────────────────

def _wdev(t: torch.Tensor) -> str:
    return "cuda:0" if t.device.type == "cuda" else "cpu"


def _empty_key(wdev):
    """1-element placeholder int64/float32 arrays for the non-sigma path."""
    return (wp.zeros(1, dtype=wp.int64, device=wdev),
            wp.zeros(1, dtype=wp.float32, device=wdev))


def bdim_coeff_3d_warp(
        u_prime, v_prime, w_prime,
        sdf_u, sdf_v, sdf_w,
        body_u, body_v, body_w,
        u0, v0, w0, ch, cv, cw,
        eps, rho_f, dt, h_grid,
        dirty_i0, dirty_j0, dirty_k0,
        dirty_Ai, dirty_Aj, dirty_Ak,
        mu0_projection=1,
        *, key_u=None, key_v=None, key_w=None, sigma_shifts=None,
        graph=None):
    """Warp port of ``bdim_coeff_3d`` (and, with the keyword args, the σ
    variant ``bdim_coeff_sigma_3d``).  Writes ``u0/v0/w0`` and ``ch/cv/cw``
    in place inside the dirty AABB; returns nothing (mirrors the native op).

    The output tensors are wrapped zero-copy via ``wp.from_torch``; the kernel
    allocates **no** global scratch — mu/normals stay in registers.
    """
    if dirty_Ai * dirty_Aj * dirty_Ak <= 0:
        return
    wdev = _wdev(u0)
    Ngx, Ngy, Ngz = u0.shape

    def f(x):
        return wp.from_torch(x.reshape(-1))  # zero-copy view, already float64

    sigma_on = 1 if sigma_shifts is not None else 0
    if sigma_on:
        ku = wp.from_torch(key_u.reshape(-1))
        kv = wp.from_torch(key_v.reshape(-1))
        kw = wp.from_torch(key_w.reshape(-1))
        ss = wp.from_torch(sigma_shifts.reshape(-1).to(torch.float32))
        n_sigma = int(sigma_shifts.numel())
    else:
        ku, ss = _empty_key(wdev)
        kv = kw = ku
        n_sigma = 0

    wp.launch(
        bdim_coeff_3d_kernel,
        dim=int(dirty_Ai) * int(dirty_Aj) * int(dirty_Ak),
        inputs=[
            f(u_prime), f(v_prime), f(w_prime),
            f(sdf_u), f(sdf_v), f(sdf_w),
            f(body_u), f(body_v), f(body_w),
            f(u0), f(v0), f(w0), f(ch), f(cv), f(cw),
            wp.float64(eps), wp.float64(rho_f), wp.float64(dt),
            wp.float64(0.5 / h_grid),
            int(Ngx), int(Ngy), int(Ngz),
            int(dirty_i0), int(dirty_j0), int(dirty_k0),
            int(mu0_projection),
            int(sigma_on), ku, kv, kw, ss, int(n_sigma),
            int(dirty_Aj), int(dirty_Ak),
        ],
        device=wdev)
