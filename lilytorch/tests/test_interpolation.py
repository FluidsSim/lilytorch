"""Warp interp_2d / interp_3d scattered-gather checks: Warp CPU == Warp GPU.

Merged from the former test_misc_2d.py / test_misc_3d.py (the interp half); the
apply_bcs half moved to test_advection.py alongside the AdvDiffSolver.
"""
from __future__ import annotations

import pytest
import torch
import warp as wp

from lilytorch.src.interpolation import (
    interp_2d_warp, interp_3d_warp, RegularGridInterpolator,
)
from lilytorch.src.native import interp_2d as interp_2d_native
from lilytorch.src.native import interp_3d as interp_3d_native

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
_DEVS = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


# ─── interp_2d ───────────────────────────────────────────────────────────────

def _interp_problem_2d(dev, Mx=30, My=24, N=4000, seed=5, dtype=torch.float32):
    torch.manual_seed(seed)
    xs = torch.linspace(-0.5, 0.5, Mx)
    ys = torch.linspace(-0.4, 0.4, My)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    F = (torch.sin(3 * X) * torch.cos(2 * Y)).to(dtype)
    bx0, by0 = float(xs[0]), float(ys[0])
    inv_dx = 1.0 / float(xs[1] - xs[0])
    inv_dy = 1.0 / float(ys[1] - ys[0])
    # queries spanning (and slightly past) the domain → exercise clamp
    xq = (torch.rand(N) * 1.2 - 0.6).to(dtype)
    yq = (torch.rand(N) * 1.0 - 0.5).to(dtype)
    return (F.to(dev), xq.to(dev), yq.to(dev), bx0, by0, inv_dx, inv_dy, Mx, My)


@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_interp_2d_cpu_eq_gpu(method, dtype):
    # CPU vs GPU to ~1 ULP of the working dtype: the CUDA codegen may contract
    # the bilinear blend FMAs while the CPU path does not.
    pc = _interp_problem_2d("cpu", dtype=dtype)
    pg = _interp_problem_2d("cuda:0", dtype=dtype)
    gc = interp_2d_warp(*pc, method)
    gg = interp_2d_warp(*pg, method)
    wp.synchronize()
    assert gc.dtype == dtype
    d = (gc - gg.cpu()).abs().max().item()
    tol = 1e-6 if dtype == torch.float32 else 1e-14
    assert d < tol, f"interp {method} {dtype} cpu vs gpu maxdiff {d:.3e}"


# ─── interp_3d ───────────────────────────────────────────────────────────────

def _interp_problem_3d(dev, Mx=20, My=18, Mz=16, N=3000, seed=5, dtype=torch.float32):
    torch.manual_seed(seed)
    xs = torch.linspace(-0.5, 0.5, Mx); ys = torch.linspace(-0.4, 0.4, My); zs = torch.linspace(-0.3, 0.3, Mz)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    F = (torch.sin(3*X)*torch.cos(2*Y)*torch.sin(Z)).to(dtype)
    b = (float(xs[0]), float(ys[0]), float(zs[0]))
    inv = (1.0/float(xs[1]-xs[0]), 1.0/float(ys[1]-ys[0]), 1.0/float(zs[1]-zs[0]))
    xq = (torch.rand(N)*1.2-0.6).to(dtype)
    yq = (torch.rand(N)*1.0-0.5).to(dtype)
    zq = (torch.rand(N)*0.8-0.4).to(dtype)
    return F.to(dev), xq.to(dev), yq.to(dev), zq.to(dev), b, inv, (Mx, My, Mz)


@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_interp_3d_cpu_eq_gpu(method, dtype):
    # CPU vs GPU to ~1 ULP of the working dtype: the CUDA codegen may contract
    # the trilinear blend FMAs while the CPU path does not.
    Fc, xc, yc, zc, b, inv, M = _interp_problem_3d("cpu", dtype=dtype)
    Fg, xg, yg, zg, _, _, _ = _interp_problem_3d("cuda:0", dtype=dtype)
    gc = interp_3d_warp(Fc, xc, yc, zc, *b, *inv, *M, method)
    gg = interp_3d_warp(Fg, xg, yg, zg, *b, *inv, *M, method)
    wp.synchronize()
    assert gc.dtype == dtype
    d = (gc - gg.cpu()).abs().max().item()
    tol = 1e-6 if dtype == torch.float32 else 1e-14
    assert d < tol, f"interp3d {method} {dtype} cpu vs gpu maxdiff {d:.3e}"


# ─── cuda_native_port Track A parity gate: native interp == Warp oracle ──────
# Scattered gathers (no atomics) → tight parity; only FMA-contraction drift.

@pytest.mark.parametrize("dev", _DEVS)
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_interp_2d_native_eq_warp(dev, method, dtype):
    p = _interp_problem_2d(dev, dtype=dtype)
    w = interp_2d_warp(*p, method)
    n = interp_2d_native(*p, method)
    wp.synchronize()
    d = (w - n).abs().max().item()
    tol = 1e-6 if dtype == torch.float32 else 1e-13
    assert d < tol, f"interp2d native vs warp {dev} {method} {dtype} maxdiff {d:.3e}"


@pytest.mark.parametrize("dev", _DEVS)
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_interp_3d_native_eq_warp(dev, method, dtype):
    F, xq, yq, zq, b, inv, M = _interp_problem_3d(dev, dtype=dtype)
    w = interp_3d_warp(F, xq, yq, zq, *b, *inv, *M, method)
    n = interp_3d_native(F, xq, yq, zq, *b, *inv, *M, method)
    wp.synchronize()
    d = (w - n).abs().max().item()
    tol = 1e-6 if dtype == torch.float32 else 1e-13
    assert d < tol, f"interp3d native vs warp {dev} {method} {dtype} maxdiff {d:.3e}"


@pytest.mark.parametrize("dev", _DEVS)
@pytest.mark.parametrize("method", ["linear", "quadratic"])
def test_rgi_class_matches_warp_kernel(dev, method):
    """The re-exported native RegularGridInterpolator == the Warp scattered
    gather it replaces, end-to-end (constructor metadata + __call__)."""
    F, xq, yq, bx0, by0, inv_dx, inv_dy, Mx, My = _interp_problem_2d(
        dev, dtype=torch.float64)
    ax = (torch.linspace(bx0, bx0 + (Mx - 1) / inv_dx, Mx, dtype=torch.float64).to(dev),
          torch.linspace(by0, by0 + (My - 1) / inv_dy, My, dtype=torch.float64).to(dev))
    rgi = RegularGridInterpolator(ax, F, method=method)
    got = rgi(xq, yq)
    ref = interp_2d_warp(F, xq, yq, bx0, by0, inv_dx, inv_dy, Mx, My, method)
    wp.synchronize()
    d = (got - ref).abs().max().item()
    assert d < 1e-12, f"RGI vs warp {dev} {method} maxdiff {d:.3e}"
