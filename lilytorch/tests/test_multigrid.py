"""Warp multigrid ops: CPU == GPU single-source checks + a Warp V-cycle.

Exercises the ported building blocks (smoother, residual, transfers) on
manufactured fields, then drives the assembled ``WarpVCycle`` on a
constant-coefficient Neumann-Laplacian Poisson and asserts the residual
contracts geometrically.

Run:  pytest lilytorch/src/kernels/test_multigrid.py -v
      python -m lilytorch.src.test_multigrid
"""
from __future__ import annotations

import pytest
import torch
import warp as wp

from lilytorch.src.multigrid import (
    jacobi_sweep_3d_warp, mg_residual_3d_warp,
    restrict_residual_3d_warp, restrict_face_3d_warp, prolongate_add_3d_warp,
    WarpVCycle,
    mg_residual_2d_warp, restrict_residual_2d_warp, restrict_face_2d_warp,
    prolongate_add_2d_warp, WarpVCycle2D,
)

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

JCAP = 1e-30


def _prob(N, dev):
    """Manufactured padded p + interior f + 6 positive face coefficients."""
    torch.manual_seed(11)
    p = torch.randn(N + 2, N + 2, N + 2, dtype=torch.float64)
    f = torch.randn(N, N, N, dtype=torch.float64)
    coefs = [torch.rand(N, N, N, dtype=torch.float64) + 0.5 for _ in range(6)]
    return [t.to(dev) for t in [p, f, *coefs]]


def _maxerr(a, b):
    return (a.cpu() - b.cpu()).abs().max().item()


# ─── CPU == GPU (single source) ──────────────────────────────────────────────

@SKIP_NO_CUDA
def test_cpu_eq_gpu_residual():
    p, f, *c = _prob(24, "cpu")
    rc = mg_residual_3d_warp(p, f, *c, JCAP)
    pg = [t.cuda() for t in [p, f, *c]]
    rg = mg_residual_3d_warp(pg[0], pg[1], *pg[2:], JCAP)
    assert _maxerr(rc, rg) < 1e-12


@SKIP_NO_CUDA
@pytest.mark.parametrize("nsmooth", [1, 2, 3])
def test_cpu_eq_gpu_jacobi(nsmooth):
    p, f, *c = _prob(24, "cpu")
    pc = p.clone()
    jacobi_sweep_3d_warp(pc, f, *c, JCAP, 0.8, nsmooth)
    pg = [t.cuda() for t in [p, f, *c]]
    pgc = pg[0].clone()
    jacobi_sweep_3d_warp(pgc, pg[1], *pg[2:], JCAP, 0.8, nsmooth)
    err = _maxerr(pc[1:-1, 1:-1, 1:-1], pgc[1:-1, 1:-1, 1:-1])
    assert err < 1e-12, f"nsmooth={nsmooth}: {err:.2e}"


@SKIP_NO_CUDA
def test_cpu_eq_gpu_restrict_residual():
    N = 32; Nc = N // 2
    torch.manual_seed(3)
    r = torch.randn(N, N, N, dtype=torch.float64)
    rc = torch.empty(Nc, Nc, Nc, dtype=torch.float64)
    restrict_residual_3d_warp(r, rc)
    rg = torch.empty(Nc, Nc, Nc, dtype=torch.float64, device="cuda:0")
    restrict_residual_3d_warp(r.cuda(), rg)
    assert _maxerr(rc, rg) == 0.0


@SKIP_NO_CUDA
@pytest.mark.parametrize("face_dim", [0, 1, 2])
def test_cpu_eq_gpu_restrict_face(face_dim):
    # fine face shape: +1 in face_dim
    Nc = 16
    shp = [2 * Nc, 2 * Nc, 2 * Nc]
    shp[face_dim] = 2 * Nc + 1
    torch.manual_seed(4)
    src = torch.randn(*shp, dtype=torch.float64)
    cshp = [Nc, Nc, Nc]
    cshp[face_dim] = Nc + 1
    dc = torch.empty(*cshp, dtype=torch.float64)
    restrict_face_3d_warp(src, dc, face_dim)
    dg = torch.empty(*cshp, dtype=torch.float64, device="cuda:0")
    restrict_face_3d_warp(src.cuda(), dg, face_dim)
    assert _maxerr(dc, dg) == 0.0


@SKIP_NO_CUDA
def test_cpu_eq_gpu_prolongate():
    Nc = 16; Nf = 32
    torch.manual_seed(6)
    ec = torch.randn(Nc + 2, Nc + 2, Nc + 2, dtype=torch.float64)
    p0 = torch.randn(Nf + 2, Nf + 2, Nf + 2, dtype=torch.float64)
    pc = p0.clone()
    prolongate_add_3d_warp(ec, pc)
    pg = p0.clone().cuda()
    prolongate_add_3d_warp(ec.cuda(), pg)
    # 1-ULP f64 FMA-contraction difference between the CPU and CUDA codegen
    # on the trilinear weighted sum.
    assert _maxerr(pc, pg) < 1e-14


# ─── V-cycle convergence (integration of the ported ops) ─────────────────────

@SKIP_NO_CUDA
def test_vcycle_converges():
    N = 32
    torch.manual_seed(5)
    f = torch.randn(N, N, N, dtype=torch.float64, device="cuda:0")
    f -= f.mean()                      # compatible (zero-mean) Neumann RHS
    vc = WarpVCycle(N, device="cuda:0", nu1=2, nu2=2)
    vc.set_rhs(f)
    r0 = vc.residual_norm()
    norms = [r0]
    for _ in range(12):
        vc.cycle()
        norms.append(vc.residual_norm())
    assert norms[-1] < 1e-4 * norms[0], f"no contraction: {norms[0]:.2e}->{norms[-1]:.2e}"
    # geometric: average contraction factor per cycle < 0.5
    factor = (norms[-1] / norms[0]) ** (1.0 / (len(norms) - 1))
    assert factor < 0.6, f"weak contraction factor {factor:.3f}"


if __name__ == "__main__":
    import warp as wp  # noqa
    if torch.cuda.is_available():
        N = 32
        f = torch.randn(N, N, N, dtype=torch.float64, device="cuda:0")
        f -= f.mean()
        vc = WarpVCycle(N, device="cuda:0")
        vc.set_rhs(f)
        r0 = vc.residual_norm()
        print(f"  V-cycle residual: {r0:.3e}", end="")
        for _ in range(10):
            vc.cycle()
        print(f" -> {vc.residual_norm():.3e}")


# ═════════════════════════════════════════════════════════════════════════════
#  2-D transfer ops + WarpVCycle2D — merged from the former test_multigrid_2d.py
# ═════════════════════════════════════════════════════════════════════════════
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
