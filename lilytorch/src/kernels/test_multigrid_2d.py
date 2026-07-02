"""2-D Warp multigrid ops: CPU == GPU single-source checks + a converging V-cycle.

Validates that the Warp kernel-level ops compose with a Python/CUDA-graph
multigrid DRIVER in 2-D (the Poisson outer-driver composition claim).
"""
from __future__ import annotations

import pytest
import torch
import warp as wp

from lilytorch.src.kernels.multigrid_2d import (
    mg_residual_2d_warp, restrict_residual_2d_warp, restrict_face_2d_warp,
    prolongate_add_2d_warp, WarpVCycle2D,
)

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
JCAP = 1e-30


def _coeffs(Nx, Ny, dev, seed=2):
    torch.manual_seed(seed)
    return [(0.5 + torch.rand(Nx, Ny, dtype=torch.float64)).to(dev) for _ in range(4)]


@SKIP_NO_CUDA
def test_restrict_residual_2d_cpu_eq_gpu():
    Nxf, Nyf = 64, 48
    torch.manual_seed(1)
    r = torch.randn(Nxf, Nyf, dtype=torch.float64)
    rc = torch.empty(Nxf // 2, Nyf // 2, dtype=torch.float64)
    restrict_residual_2d_warp(r, rc)
    rg = torch.empty(Nxf // 2, Nyf // 2, dtype=torch.float64, device="cuda:0")
    restrict_residual_2d_warp(r.cuda(), rg)
    wp.synchronize()
    assert (rc - rg.cpu()).abs().max().item() == 0.0


@SKIP_NO_CUDA
@pytest.mark.parametrize("face_dim", [0, 1])
def test_restrict_face_2d_cpu_eq_gpu(face_dim):
    n = 64
    torch.manual_seed(1)
    if face_dim == 0:
        src = torch.randn(n + 1, n, dtype=torch.float64)
        dc = torch.empty(n // 2 + 1, n // 2, dtype=torch.float64)
    else:
        src = torch.randn(n, n + 1, dtype=torch.float64)
        dc = torch.empty(n // 2, n // 2 + 1, dtype=torch.float64)
    restrict_face_2d_warp(src, dc, face_dim)
    dg = torch.empty(dc.shape, dtype=torch.float64, device="cuda:0")
    restrict_face_2d_warp(src.cuda(), dg, face_dim)
    wp.synchronize()
    assert (dc - dg.cpu()).abs().max().item() == 0.0


@SKIP_NO_CUDA
def test_prolongate_add_2d_cpu_eq_gpu():
    nc, nf = 32, 64
    torch.manual_seed(1)
    ec = torch.randn(nc + 2, nc + 2, dtype=torch.float64)
    p0 = torch.randn(nf + 2, nf + 2, dtype=torch.float64)
    pc = p0.clone()
    prolongate_add_2d_warp(ec, pc)
    pg = p0.clone().cuda()
    prolongate_add_2d_warp(ec.cuda(), pg)
    wp.synchronize()
    # 1-ULP float64 FMA-contraction difference between the CPU and CUDA codegen
    # on the bilinear weighted sum.
    assert (pc - pg.cpu()).abs().max().item() < 1e-14


@SKIP_NO_CUDA
def test_vcycle_2d_converges():
    """All-Warp 2-D V-cycle reduces the residual geometrically on a
    manufactured zero-mean Neumann Poisson — the driver-composition payoff."""
    N = 64
    torch.manual_seed(0)
    f = torch.randn(N, N, dtype=torch.float64, device="cuda:0")
    f -= f.mean()
    vc = WarpVCycle2D(N, device="cuda:0")
    vc.set_rhs(f)
    r0 = vc.residual_norm()
    rates = []
    for _ in range(10):
        vc.cycle()
        rates.append(vc.residual_norm())
    assert rates[-1] < 1e-6 * r0, f"residual {r0:.2e} -> {rates[-1]:.2e}"
    # geometric: each cycle contracts
    assert rates[3] < 0.3 * r0


@SKIP_NO_CUDA
def test_transfer_ops_cpu_eq_gpu():
    Nx, Ny = 48, 32
    torch.manual_seed(5)
    pc = torch.randn(Nx + 2, Ny + 2, dtype=torch.float64)
    fc = torch.randn(Nx, Ny, dtype=torch.float64)
    cc = _coeffs(Nx, Ny, "cpu")
    rc = mg_residual_2d_warp(pc, fc, *cc, JCAP)
    rg = mg_residual_2d_warp(pc.cuda(), fc.cuda(), *[c.cuda() for c in cc], JCAP)
    wp.synchronize()
    assert (rc - rg.cpu()).abs().max().item() < 1e-12
