"""Warp interp_3d / apply_bcs_3d single-source checks: Warp CPU == Warp GPU."""
from __future__ import annotations

import pytest
import torch
import warp as wp

from lilytorch.src.kernels.misc_3d import interp_3d_warp, apply_bcs_3d_warp

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def _interp_problem(dev, Mx=20, My=18, Mz=16, N=3000, seed=5, dtype=torch.float32):
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
    Fc, xc, yc, zc, b, inv, M = _interp_problem("cpu", dtype=dtype)
    Fg, xg, yg, zg, _, _, _ = _interp_problem("cuda:0", dtype=dtype)
    gc = interp_3d_warp(Fc, xc, yc, zc, *b, *inv, *M, method)
    gg = interp_3d_warp(Fg, xg, yg, zg, *b, *inv, *M, method)
    wp.synchronize()
    assert gc.dtype == dtype
    d = (gc - gg.cpu()).abs().max().item()
    tol = 1e-6 if dtype == torch.float32 else 1e-14
    assert d < tol, f"interp3d {method} {dtype} cpu vs gpu maxdiff {d:.3e}"


def _bcs_problem(dev, Nx=20, Ny=16, Nz=14, seed=9):
    torch.manual_seed(seed)
    u = torch.randn(Nx, Ny, Nz, dtype=torch.float64)
    v = torch.randn(Nx, Ny, Nz, dtype=torch.float64)
    w = torch.randn(Nx, Ny, Nz, dtype=torch.float64)
    shapes = torch.tensor([[Nx, Ny, Nz]]*3, dtype=torch.int64)
    # disjoint stage-1 ops (no shared face cell): u x-faces, v y-faces, w z-Dirichlet
    neu = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 1, 0]], dtype=torch.int32)
    dird = torch.tensor([[2, 2, 0], [2, 2, -1]], dtype=torch.int32)
    dirv = torch.tensor([2.5, -1.3], dtype=torch.float64)
    refd = torch.tensor([[1, 1, -1, -2]], dtype=torch.int32)   # v y-face reflective (stage2)
    refv = torch.tensor([0.4], dtype=torch.float64)
    M = max(Nx, Ny, Nz)
    to = lambda t: t.to(dev)
    return (to(u), to(v), to(w), to(shapes), to(neu), to(dird), to(dirv), to(refd), to(refv), M)


def _run_bcs(dev, f32=False):
    u, v, w, shapes, neu, dird, dirv, refd, refv, M = _bcs_problem(dev)
    if f32:
        u = u.float(); v = v.float(); w = w.float()
        dirv = dirv.float(); refv = refv.float()
    uw, vw, ww = u.clone().contiguous(), v.clone().contiguous(), w.clone().contiguous()
    apply_bcs_3d_warp(uw, vw, ww, shapes, neu, dird, dirv, refd, refv, M, M)
    wp.synchronize()
    return uw, vw, ww


@SKIP_NO_CUDA
@pytest.mark.parametrize("f32", [False, True], ids=["f64", "f32"])
def test_apply_bcs_3d_cpu_eq_gpu(f32):
    """BC writes are copies / value sets → CPU == GPU bit-exact."""
    rc = _run_bcs("cpu", f32)
    rg = _run_bcs("cuda:0", f32)
    for a, b, nm in zip(rc, rg, ("u", "v", "w")):
        d = (a - b.cpu()).abs().max().item()
        assert d == 0.0, f"bcs3d cpu vs gpu {nm} mismatch {d:.3e}"


@SKIP_NO_CUDA
def test_apply_bcs_3d_noncubic_dual_facedims():
    """Non-cubic grid with separate (max_dim0, max_dim1): CPU == GPU exactly."""
    Nx, Ny, Nz = 24, 14, 10
    torch.manual_seed(3)
    u = torch.randn(Nx, Ny, Nz, dtype=torch.float64)
    v = torch.randn(Nx, Ny, Nz, dtype=torch.float64)
    w = torch.randn(Nx, Ny, Nz, dtype=torch.float64)
    shapes = torch.tensor([[Nx, Ny, Nz]] * 3, dtype=torch.int64)
    # Disjoint per-component ops (no shared stage-1 cell — overlaps are
    # order-undefined on GPU) that still exercise all three face axes so
    # dim0/dim1 differ per face:
    #   u z-face (axis2): d0=Nx=24, d1=Ny=14  → drives max_dim0, max_dim1
    #   w x-face (axis0): d0=Ny=14, d1=Nz=10
    #   v y-face (axis1): d0=Nx=24, d1=Nz=10
    neu = torch.tensor([[0, 2, 1], [2, 0, 0]], dtype=torch.int32)
    dird = torch.tensor([[1, 1, -1]], dtype=torch.int32)
    dirv = torch.tensor([1.1], dtype=torch.float64)
    refd = torch.zeros((0, 4), dtype=torch.int32)
    refv = torch.zeros((0,), dtype=torch.float64)
    max_dim0 = int(max(Ny, Nx))
    max_dim1 = int(max(Nz, Ny))

    def run(dev):
        to = lambda t: t.to(dev)
        args = [to(x) for x in (u, v, w, shapes, neu, dird, dirv, refd, refv)]
        uw, vw, ww = (args[0].clone().contiguous(), args[1].clone().contiguous(),
                      args[2].clone().contiguous())
        apply_bcs_3d_warp(uw, vw, ww, *args[3:9], max_dim0, max_dim1)
        wp.synchronize()
        return uw, vw, ww

    rc, rg = run("cpu"), run("cuda:0")
    for a, b, nm in zip(rc, rg, ("u", "v", "w")):
        d = (a - b.cpu()).abs().max().item()
        assert d == 0.0, f"noncubic cpu vs gpu {nm} mismatch {d:.3e}"
