"""Parity tests: Warp Eulerian force readout vs native CUDA.

Validates ``streaming_sdf_forces_post_{2,3}d_warp`` (``warp_forces.py``) against
the native ops on the synthetic flat-table scenes (``scene_2d.make_synthetic_scene_2d``
/ ``bench_viability.make_synthetic_scene``).  The union ``sdf_cc`` is populated by
the native streaming kernel first (so the band / union-normal paths are exercised
on a real union SDF), then both force readouts consume the same fields.

Covers: ``force_submethod`` 0 (n·δ) and 1 (deltaH ∂H pressure), ``delta_order``
1 and 2, scalar + full-field nu_rho, linear sampling, f32 + f64.

Run:  pytest lilytorch/warp_poc/test_forces.py -v
"""
from __future__ import annotations

import pytest
import torch

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import (
        body_update_2d as native_stream_2d,
        streaming_sdf_forces_post_2d as native_forces_2d,
        body_update_3d as native_stream_3d,
        streaming_sdf_forces_post_3d as native_forces_3d,
    )
    _NATIVE = True
except Exception:
    _NATIVE = False

from lilytorch.src.kernels.forces import (
    streaming_sdf_forces_post_2d_warp,
    streaming_sdf_forces_post_3d_warp,
)
from lilytorch.src.kernels.scene_2d import make_synthetic_scene_2d
from lilytorch.src.kernels.bench_viability import make_synthetic_scene

SKIP_NO_NATIVE = pytest.mark.skipif(not _NATIVE, reason="native _C.so unavailable")
SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(),
                                  reason="native forces kernel is CUDA-only")

# float64 atomic reduction-order noise (native block-reduce vs Warp per-cell);
# f32 also carries single-precision / FMA drift.
ATOL_F64 = 1e-9
RTOL_F32 = 3e-4
ATOL_F32 = 1e-5


def _fill_union_sdf_2d(sc, dtype, dev):
    """Run the native streaming kernel to populate a real union ``sdf_cc``."""
    Ngx, Ngy = sc["Ngx"], sc["Ngy"]
    opt = dict(dtype=dtype, device=dev)
    sdf_cc = torch.full((Ngx, Ngy), 1e4, **opt)
    sdf_u = torch.full((Ngx, Ngy), 1e4, **opt)
    sdf_v = torch.full((Ngx, Ngy), 1e4, **opt)
    bu = torch.zeros((Ngx, Ngy), **opt)
    bv = torch.zeros((Ngx, Ngy), **opt)
    N = Ngx * Ngy
    kcc = torch.empty(N, dtype=torch.int64, device=dev)
    ku = torch.empty(N, dtype=torch.int64, device=dev)
    kv = torch.empty(N, dtype=torch.int64, device=dev)
    nu_ = torch.empty(1, **opt); nv_ = torch.empty(1, **opt)
    du_ = torch.empty(1, **opt); dv_ = torch.empty(1, **opt)
    di0, dj0, dAi, dAj = sc["dirty_bounds"]
    native_stream_2d(sc["F_flat"], sc["F_offsets"], sc["body_shapes"],
                     sc["body_meta"], sc["kin"], sc["aabb_lo"], sc["aabb_dim"],
                     sc["gx"], sc["gy"], float(sc["h"]), int(sc["max_vol"]),
                     sdf_cc, sdf_u, sdf_v, bu, bv, kcc, ku, kv,
                     0, di0, dj0, dAi, dAj, nu_, nv_, du_, dv_, 0.0)
    return sdf_cc


def _run_2d(dev, dtype, submethod, delta_order, scalar_nrho):
    sc = make_synthetic_scene_2d(96, 64, 3, device=dev, dtype=dtype)
    for k in ("F_flat", "body_meta", "kin", "gx", "gy"):
        sc[k] = sc[k].to(dtype)
    Ngx, Ngy, h = sc["Ngx"], sc["Ngy"], float(sc["h"])
    sdf_cc = _fill_union_sdf_2d(sc, dtype, dev)

    torch.manual_seed(7)
    u = torch.randn(Ngx, Ngy, dtype=dtype, device=dev)
    v = torch.randn(Ngx, Ngy, dtype=dtype, device=dev)
    p = torch.randn(Ngx, Ngy, dtype=dtype, device=dev)
    if scalar_nrho:
        nrho = torch.tensor([0.13], dtype=dtype, device=dev)
    else:
        nrho = (torch.randn(Ngx, Ngy, dtype=dtype, device=dev).abs() + 0.05)

    eps_body = 2.0 * h
    eps_solver = 0.0
    h2 = h * h
    B = sc["aabb_dim"].shape[0]
    ph_tau = 0.5 * h if submethod else 0.0
    common = dict(force_submethod=submethod, ph_tau=ph_tau)

    out_n = torch.zeros(B, 6, dtype=torch.float64, device=dev)
    native_forces_2d(
        sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
        sc["kin"], sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"],
        h, int(sc["max_vol"]), sdf_cc, 0, u, v, p, nrho,
        eps_body, eps_solver, h2, delta_order, out_n, **common)

    out_w = torch.zeros(B, 6, dtype=torch.float64, device=dev)
    streaming_sdf_forces_post_2d_warp(
        sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
        sc["kin"], sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"],
        h, int(sc["max_vol"]), sdf_cc, 0, u, v, p, nrho,
        eps_body, eps_solver, h2, delta_order, out_w, **common)
    return out_w.cpu(), out_n.cpu()


def _check(w, n, dtype):
    err = (w - n).abs().max().item()
    if dtype == torch.float32:
        scale = n.abs().max().item()
        assert err <= ATOL_F32 + RTOL_F32 * scale, f"f32 err {err:.3e} scale {scale:.3e}"
    else:
        assert err < ATOL_F64, f"f64 err {err:.3e}"
    # not all-zero (the scene must exercise the band)
    assert n.abs().max().item() > 0, "scene produced no in-band force"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("delta_order", [1, 2])
@pytest.mark.parametrize("scalar_nrho", [True, False])
def test_forces_2d_ndelta(dtype, delta_order, scalar_nrho):
    w, n = _run_2d("cuda", dtype, 0, delta_order, scalar_nrho)
    _check(w, n, dtype)


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("delta_order", [1, 2])
def test_forces_2d_deltaH(dtype, delta_order):
    w, n = _run_2d("cuda", dtype, 1, delta_order, False)
    _check(w, n, dtype)


# ─── 3-D ─────────────────────────────────────────────────────────────────────

def _fill_union_sdf_3d(sc, dtype, dev):
    Ngx, Ngy, Ngz = sc["Ngx"], sc["Ngy"], sc["Ngz"]
    opt = dict(dtype=dtype, device=dev)
    sdf_cc = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    sdf_u = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    sdf_v = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    sdf_w = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    bu = torch.zeros((Ngx, Ngy, Ngz), **opt)
    bv = torch.zeros((Ngx, Ngy, Ngz), **opt)
    bw = torch.zeros((Ngx, Ngy, Ngz), **opt)
    di0, dj0, dk0, dAi, dAj, dAk = sc["dirty_bounds"]
    dvol = int(dAi) * int(dAj) * int(dAk)
    kcc = torch.empty(dvol, dtype=torch.int64, device=dev)
    ku = torch.empty(dvol, dtype=torch.int64, device=dev)
    kv = torch.empty(dvol, dtype=torch.int64, device=dev)
    kw = torch.empty(dvol, dtype=torch.int64, device=dev)
    nu_ = torch.empty(1, **opt); nv_ = torch.empty(1, **opt); nw_ = torch.empty(1, **opt)
    du_ = torch.empty(1, **opt); dv_ = torch.empty(1, **opt); dw_ = torch.empty(1, **opt)
    native_stream_3d(sc["F_flat"], sc["F_offsets"], sc["body_shapes"],
                     sc["body_meta"], sc["kin"], sc["aabb_lo"], sc["aabb_dim"],
                     sc["gx"], sc["gy"], sc["gz"], float(sc["h"]), int(sc["max_vol"]),
                     sdf_cc, sdf_u, sdf_v, sdf_w, bu, bv, bw,
                     kcc, ku, kv, kw, 0,
                     di0, dj0, dk0, dAi, dAj, dAk,
                     nu_, nv_, nw_, du_, dv_, dw_, 0.0)
    return sdf_cc


def _run_3d(dev, dtype, submethod, delta_order, scalar_nrho):
    sc = make_synthetic_scene(48, 32, 32, 3, device=dev, dtype=dtype)
    for kk in ("F_flat", "body_meta", "kin", "gx", "gy", "gz"):
        sc[kk] = sc[kk].to(dtype)
    Ngx, Ngy, Ngz, h = sc["Ngx"], sc["Ngy"], sc["Ngz"], float(sc["h"])
    sdf_cc = _fill_union_sdf_3d(sc, dtype, dev)

    torch.manual_seed(11)
    u = torch.randn(Ngx, Ngy, Ngz, dtype=dtype, device=dev)
    v = torch.randn(Ngx, Ngy, Ngz, dtype=dtype, device=dev)
    w = torch.randn(Ngx, Ngy, Ngz, dtype=dtype, device=dev)
    p = torch.randn(Ngx, Ngy, Ngz, dtype=dtype, device=dev)
    if scalar_nrho:
        nrho = torch.tensor([0.11], dtype=dtype, device=dev)
    else:
        nrho = (torch.randn(Ngx, Ngy, Ngz, dtype=dtype, device=dev).abs() + 0.05)

    eps_body = 2.0 * h
    eps_solver = 0.0
    h3 = h * h * h
    B = sc["aabb_dim"].shape[0]
    ph_tau = 0.5 * h if submethod else 0.0
    common = dict(force_submethod=submethod, ph_tau=ph_tau)

    out_n = torch.zeros(B, 12, dtype=torch.float64, device=dev)
    native_forces_3d(
        sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
        sc["kin"], sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["gz"],
        h, int(sc["max_vol"]), sdf_cc, 0, u, v, w, p, nrho,
        eps_body, eps_solver, h3, delta_order, out_n, **common)

    out_w = torch.zeros(B, 12, dtype=torch.float64, device=dev)
    streaming_sdf_forces_post_3d_warp(
        sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
        sc["kin"], sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["gz"],
        h, int(sc["max_vol"]), sdf_cc, 0, u, v, w, p, nrho,
        eps_body, eps_solver, h3, delta_order, out_w, **common)
    return out_w.cpu(), out_n.cpu()


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("delta_order", [1, 2])
@pytest.mark.parametrize("scalar_nrho", [True, False])
def test_forces_3d_ndelta(dtype, delta_order, scalar_nrho):
    w, n = _run_3d("cuda", dtype, 0, delta_order, scalar_nrho)
    _check(w, n, dtype)


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("delta_order", [1, 2])
def test_forces_3d_deltaH(dtype, delta_order):
    w, n = _run_3d("cuda", dtype, 1, delta_order, False)
    _check(w, n, dtype)
