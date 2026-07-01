"""3-D fused tiled smoothers (Jacobi + RBGS) — parity vs reference + convergence."""
from __future__ import annotations

import pytest
import torch
import warp as wp

from lilytorch.src.kernels.poisson_tiled_3d import WarpTiledSmoother3D, TILE

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def _problem(N, seed=11):
    torch.manual_seed(seed)
    p = torch.zeros(N + 2, N + 2, N + 2, dtype=torch.float32, device="cuda:0")
    p[1:-1, 1:-1, 1:-1] = torch.randn(N, N, N, device="cuda:0")
    f = torch.randn(N, N, N, dtype=torch.float32, device="cuda:0")
    c = [0.4 + torch.rand(N, N, N, dtype=torch.float32, device="cuda:0") for _ in range(6)]
    return p, f, c


def _neumann(p):
    p[0] = p[1]; p[-1] = p[-2]
    p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
    p[:, :, 0] = p[:, :, 1]; p[:, :, -1] = p[:, :, -2]
    return p


def _stencil(pp, c):
    c0, c1, c2, c3, c4, c5 = c
    return (c0*pp[2:, 1:-1, 1:-1] + c1*pp[:-2, 1:-1, 1:-1]
            + c2*pp[1:-1, 2:, 1:-1] + c3*pp[1:-1, :-2, 1:-1]
            + c4*pp[1:-1, 1:-1, 2:] + c5*pp[1:-1, 1:-1, :-2])


def _torch_jacobi_singletile(p, f, c, nsweep, w):
    p = p.clone(); J = sum(c)
    for _ in range(nsweep):
        p[1:-1, 1:-1, 1:-1] = w * ((-f + _stencil(p, c)) / J) + (1-w) * p[1:-1, 1:-1, 1:-1]
    return p


def _torch_rbgs_singletile(p, f, c, nsweep):
    p = p.clone(); J = sum(c); N = f.shape[0]
    idx = torch.arange(N, device=f.device)
    red = (idx.view(-1,1,1) + idx.view(1,-1,1) + idx.view(1,1,-1)) % 2 == 0
    for _ in range(nsweep):
        p[1:-1, 1:-1, 1:-1] = torch.where(red, (-f + _stencil(p, c))/J, p[1:-1, 1:-1, 1:-1])
        p[1:-1, 1:-1, 1:-1] = torch.where(~red, (-f + _stencil(p, c))/J, p[1:-1, 1:-1, 1:-1])
    return p


@SKIP_NO_CUDA
@pytest.mark.parametrize("ns,w", [(1, 1.0), (5, 0.8)])
def test_tiled_jacobi_3d_singletile(ns, w):
    N = TILE
    p, f, c = _problem(N)
    _neumann(p)
    pw = WarpTiledSmoother3D(N).jacobi(p, f, c, ns, w)
    wp.synchronize()
    ref = _torch_jacobi_singletile(p, f, c, ns, w)
    d = (pw[1:-1, 1:-1, 1:-1] - ref[1:-1, 1:-1, 1:-1]).abs().max().item()
    assert d < 1e-5, f"3D tiled jacobi ns={ns} maxdiff {d:.2e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("ns", [1, 4])
def test_tiled_rbgs_3d_singletile(ns):
    N = TILE
    p, f, c = _problem(N)
    _neumann(p)
    pw = WarpTiledSmoother3D(N).rbgs(p, f, c, ns)
    wp.synchronize()
    ref = _torch_rbgs_singletile(p, f, c, ns)
    d = (pw[1:-1, 1:-1, 1:-1] - ref[1:-1, 1:-1, 1:-1]).abs().max().item()
    assert d < 1e-5, f"3D tiled rbgs ns={ns} maxdiff {d:.2e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("kind", ["jacobi", "rbgs"])
def test_tiled_3d_converges(kind):
    N = 32
    torch.manual_seed(3)
    c = [0.4 + 0.1*torch.rand(N, N, N, dtype=torch.float32, device="cuda:0") for _ in range(6)]
    f = torch.randn(N, N, N, dtype=torch.float32, device="cuda:0")
    p = torch.zeros(N + 2, N + 2, N + 2, dtype=torch.float32, device="cuda:0")
    sm = WarpTiledSmoother3D(N)
    r = lambda pp: (f - _stencil(pp, c) + sum(c)*pp[1:-1, 1:-1, 1:-1]).norm().item()
    r0 = r(p)
    for _ in range(20):
        p = sm.jacobi(p, f, c, 4, 0.8) if kind == "jacobi" else sm.rbgs(p, f, c, 2)
        _neumann(p)
    wp.synchronize()
    assert r(p) < 0.5 * r0, f"{kind} residual {r0:.3e} -> {r(p):.3e}"
