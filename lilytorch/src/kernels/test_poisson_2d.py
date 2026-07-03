"""2-D Warp Poisson smoothers: convergence + CPU == GPU single-source checks."""
from __future__ import annotations

import pytest
import torch
import warp as wp

from lilytorch.src.poisson_2d import WarpRBGS2D

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def _problem(Nx, Ny, dev, seed=11):
    """Build a random positive-coefficient problem ONCE on CPU then .to(dev)."""
    torch.manual_seed(seed)
    p = torch.zeros((Nx + 2, Ny + 2), dtype=torch.float32)
    p[1:-1, 1:-1] = torch.randn(Nx, Ny)
    f = torch.randn(Nx, Ny, dtype=torch.float32)
    coeffs = [0.5 + torch.rand(Nx, Ny, dtype=torch.float32) for _ in range(4)]
    return (p.to(dev), f.to(dev), [c.to(dev) for c in coeffs])


def _warp_run(prob, dev, kind, n=1, w=1.0):
    p, f, coeffs = prob
    p = p.clone()
    s = WarpRBGS2D(p.shape[0] - 2, p.shape[1] - 2, device=dev)
    s.setup(p, f, coeffs)
    if kind == "rbgs":
        s.sweep(n)
    else:
        s.jacobi(n, w)
    wp.synchronize()
    return wp.to_torch(s.p).reshape(p.shape).clone()


@SKIP_NO_CUDA
def test_rbgs_converges():
    """Global Warp RBGS reduces the residual geometrically (manufactured Poisson)."""
    Nx, Ny = 64, 64
    prob = _problem(Nx, Ny, "cuda:0")
    p, f, coeffs = prob
    s = WarpRBGS2D(Nx, Ny, device="cuda:0")
    s.setup(p.clone(), f, coeffs)
    r0 = s.residual_norm()
    s.sweep(50)
    r1 = s.residual_norm()
    assert r1 < 0.5 * r0, f"residual {r0:.3e} -> {r1:.3e}"


@SKIP_NO_CUDA
def test_jacobi_converges():
    """Damped Jacobi also contracts (the smoother the graphed 2-D MG uses)."""
    Nx, Ny = 64, 64
    prob = _problem(Nx, Ny, "cuda:0")
    p, f, coeffs = prob
    s = WarpRBGS2D(Nx, Ny, device="cuda:0")
    s.setup(p.clone(), f, coeffs)
    r0 = s.residual_norm()
    s.jacobi(80, 0.8)
    r1 = s.residual_norm()
    assert r1 < 0.5 * r0, f"residual {r0:.3e} -> {r1:.3e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("kind", ["rbgs", "jacobi"])
def test_cpu_eq_gpu(kind):
    Nx, Ny = 48, 40
    pc = _warp_run(_problem(Nx, Ny, "cpu"), "cpu", kind, 3)
    pg = _warp_run(_problem(Nx, Ny, "cuda:0"), "cuda:0", kind, 3)
    d = (pc[1:-1, 1:-1] - pg[1:-1, 1:-1].cpu()).abs().max().item()
    assert d < 1e-4, f"{kind} cpu vs gpu {d:.3e}"
