"""Warp single-source **bdim_forcing (2-D)** — fused BDIM2 coefficient + FD normals.

Port of native ``bdim_forcing_2d`` / ``bdim_forcing_sigma_2d``
(``streaming_sdf_2d.cu``: ``bdim_one_axis_2d`` / ``bdim_one_axis_sigma_2d``).

Unlike the 3-D kernel, the 2-D native writes the Poisson coefficient at the
**full-grid** flat index ``c_out[g]`` (g = i*Ngy + j) — *both* the CUDA and the
CPU ops do (``streaming_sdf_cpu_2d.cpp: bdim_one_axis_2d_cpu``).  So there is NO
CPU/CUDA coefficient-layout discrepancy in 2-D (cf. HANDOFF lesson 9, which is
3-D-only); ``ch``/``cv`` are full-grid ``(Ngx, Ngy)`` arrays and coeff parity
holds vs both CPU and CUDA.

**Precision (single source, both dtypes).**  The kernel/funcs are Warp generics
(``Any``): float literals are materialised in the bound element type via
``type(x)(literal)`` and ``wp.overload`` registers the float32 *and* float64
specialisations (Warp 1.14 needs the dtypes pre-registered).  float64 lands on
bit-parity with the native op (``scalar_t = double``) — codegen unchanged from
the original concrete kernel, so existing f64 parity stays bit-identical; float32
matches native f32 to single precision.  ``key_*`` stays int64 and
``sigma_shifts`` stays float32 (cast to the working type inside the kernel),
independent of the solver dtype.  No global scratch — mu0/mu1/normals stay
register-resident, identical to native (the kernel-mode memory claim).
"""
from __future__ import annotations

from typing import Any

import warp as wp
import torch

wp.init()

_MASK32 = wp.constant(wp.int64(0xFFFFFFFF))
_PI_F64 = 3.141592653589793


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
def _mu0_only_2d(phi: Any, eps: Any):
    res = type(phi)(0.0)
    if phi <= -eps:
        res = type(phi)(0.0)
    elif phi >= eps:
        res = type(phi)(1.0)
    else:
        half = type(phi)(0.5)
        one = type(phi)(1.0)
        pi = type(phi)(_PI_F64)
        deps = phi / eps
        res = half * (one + deps + wp.sin(pi * deps) / pi)
    return res


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
    # BDIM-σ extras (sigma_on == 0 → ignored):
    sigma_on: wp.int32,
    key: wp.array(dtype=wp.int64),
    sigma_shifts: wp.array(dtype=wp.float32),
    n_sigma: wp.int32,
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
    if sigma_on != 0:
        body_idx = wp.int32(key[g] & _MASK32)
        sigma_shift = zero
        if body_idx < n_sigma:
            sigma_shift = type(eps)(sigma_shifts[body_idx])
        mu0_c = _mu0_only_2d(phi - sigma_shift, eps)
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
    sigma_on: wp.int32,
    key_u: wp.array(dtype=wp.int64),
    key_v: wp.array(dtype=wp.int64),
    sigma_shifts: wp.array(dtype=wp.float32),
    n_sigma: wp.int32,
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
        eps, rho_f, dt, inv_2h, Ngx, Ngy, i, j, mu0_proj,
        sigma_on, key_u, sigma_shifts, n_sigma)
    bdim_one_axis_2d(
        v_prime, sdf_v, body_v, v0, cv,
        eps, rho_f, dt, inv_2h, Ngx, Ngy, i, j, mu0_proj,
        sigma_on, key_v, sigma_shifts, n_sigma)


# Register float32 + float64 specialisations (generic args only).
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(bdim_forcing_2d_kernel, {
        "u_prime": _A, "v_prime": _A, "sdf_u": _A, "sdf_v": _A,
        "body_u": _A, "body_v": _A, "u0": _A, "v0": _A, "ch": _A, "cv": _A,
        "eps": _dt, "rho_f": _dt, "dt": _dt, "inv_2h": _dt,
    })


def _wdev(t: torch.Tensor) -> str:
    return "cuda:0" if t.device.type == "cuda" else "cpu"


def _wp_dtype(t: torch.Tensor):
    return wp.float64 if t.dtype == torch.float64 else wp.float32


def _empty_key(wdev):
    return (wp.zeros(1, dtype=wp.int64, device=wdev),
            wp.zeros(1, dtype=wp.float32, device=wdev))


def bdim_forcing_2d_warp(
        u_prime, v_prime, sdf_u, sdf_v, body_u, body_v,
        u0, v0, ch, cv,
        eps, rho_f, dt, h_grid,
        dirty_i0, dirty_j0, dirty_Ai, dirty_Aj,
        mu0_projection=1,
        *, key_u=None, key_v=None, sigma_shifts=None):
    """Warp port of ``bdim_forcing_2d`` (and σ variant with the keyword args).
    Writes u0/v0 and ch/cv (full-grid) in place inside the dirty AABB.
    Generic in dtype (f32/f64), selected from ``u0``."""
    if int(dirty_Ai) * int(dirty_Aj) <= 0:
        return
    wdev = _wdev(u0)
    wpf = _wp_dtype(u0)
    Ngx, Ngy = u0.shape

    def f(x):
        return wp.from_torch(x.reshape(-1))

    sigma_on = 1 if sigma_shifts is not None else 0
    if sigma_on:
        ku = wp.from_torch(key_u.reshape(-1))
        kv = wp.from_torch(key_v.reshape(-1))
        ss = wp.from_torch(sigma_shifts.reshape(-1).to(torch.float32))
        n_sigma = int(sigma_shifts.numel())
    else:
        ku, ss = _empty_key(wdev)
        kv = ku
        n_sigma = 0

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
            int(sigma_on), ku, kv, ss, int(n_sigma),
            int(dirty_Aj),
        ],
        device=wdev)
