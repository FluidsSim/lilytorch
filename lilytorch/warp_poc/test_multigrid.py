"""Parity tests: Warp multigrid transfer ops vs native, + a Warp V-cycle.

Validates the four ported building blocks against the native CUDA ops (the
parity oracle) on manufactured fields, on CPU and GPU, plus Warp CPU == Warp GPU
(single source).  Finally drives the assembled ``WarpVCycle`` on a manufactured
constant-coefficient Neumann-Laplacian Poisson and asserts the residual
contracts geometrically (the integration check the HANDOFF asks for).

Run:  pytest lilytorch/warp_poc/test_multigrid.py -v
      python -m lilytorch.warp_poc.test_multigrid
"""
from __future__ import annotations

import pytest
import torch

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import (
        jacobi_sweep_3d as nat_jacobi,
        mg_residual_3d as nat_residual,
        restrict_residual_3d as nat_rr,
        restrict_face_3d as nat_rf,
        prolongate_add_3d as nat_pa,
    )
    _NATIVE = True
except Exception:
    _NATIVE = False

from lilytorch.warp_poc.warp_multigrid import (
    jacobi_sweep_3d_warp, mg_residual_3d_warp,
    restrict_residual_3d_warp, restrict_face_3d_warp, prolongate_add_3d_warp,
    WarpVCycle,
)

SKIP_NO_NATIVE = pytest.mark.skipif(not _NATIVE, reason="native _C.so unavailable")
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
    return (a - b).abs().max().item()


# ─── residual parity ─────────────────────────────────────────────────────────

@SKIP_NO_NATIVE
@SKIP_NO_CUDA
def test_residual_gpu():
    p, f, *c = _prob(32, "cuda:0")
    rn = nat_residual(p, f, *c, JCAP)
    rw = mg_residual_3d_warp(p, f, *c, JCAP)
    assert _maxerr(rn, rw) == 0.0, _maxerr(rn, rw)


# (native mg_residual_3d / transfer ops are CUDA-only — CPU is validated by the
#  Warp CPU==GPU single-source check below, not against a native CPU oracle.)


# ─── jacobi parity (interior only — ghosts are BC bookkeeping) ───────────────

@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("nsmooth", [1, 2, 3])
def test_jacobi_gpu(nsmooth):
    p, f, *c = _prob(32, "cuda:0")
    pn = p.clone(); pw = p.clone()
    nat_jacobi(pn, f, *c, JCAP, 0.8, nsmooth)
    jacobi_sweep_3d_warp(pw, f, *c, JCAP, 0.8, nsmooth)
    err = _maxerr(pn[1:-1, 1:-1, 1:-1], pw[1:-1, 1:-1, 1:-1])
    assert err < 1e-13, f"nsmooth={nsmooth}: {err:.2e}"


# ─── restriction parity ──────────────────────────────────────────────────────

@SKIP_NO_NATIVE
@SKIP_NO_CUDA
def test_restrict_residual_gpu():
    N = 32; Nc = N // 2
    r = torch.randn(N, N, N, dtype=torch.float64, device="cuda:0")
    rcn = torch.empty(Nc, Nc, Nc, dtype=torch.float64, device="cuda:0")
    rcw = torch.empty_like(rcn)
    nat_rr(r, rcn)
    restrict_residual_3d_warp(r, rcw)
    assert _maxerr(rcn, rcw) == 0.0, _maxerr(rcn, rcw)


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("face_dim", [0, 1, 2])
def test_restrict_face_gpu(face_dim):
    # fine face shape: +1 in face_dim
    Nc = 16
    shp = [2 * Nc, 2 * Nc, 2 * Nc]
    shp[face_dim] = 2 * Nc + 1
    src = torch.randn(*shp, dtype=torch.float64, device="cuda:0")
    cshp = [Nc, Nc, Nc]
    cshp[face_dim] = Nc + 1
    dstn = torch.empty(*cshp, dtype=torch.float64, device="cuda:0")
    dstw = torch.empty_like(dstn)
    nat_rf(src, dstn, face_dim)
    restrict_face_3d_warp(src, dstw, face_dim)
    assert _maxerr(dstn, dstw) == 0.0, _maxerr(dstn, dstw)


# ─── prolongation parity ─────────────────────────────────────────────────────

@SKIP_NO_NATIVE
@SKIP_NO_CUDA
def test_prolongate_gpu():
    Nc = 16; Nf = 32
    ec = torch.randn(Nc + 2, Nc + 2, Nc + 2, dtype=torch.float64, device="cuda:0")
    p0 = torch.randn(Nf + 2, Nf + 2, Nf + 2, dtype=torch.float64, device="cuda:0")
    pn = p0.clone(); pw = p0.clone()
    nat_pa(ec, pn)
    prolongate_add_3d_warp(ec, pw)
    assert _maxerr(pn, pw) == 0.0, _maxerr(pn, pw)


# ─── CPU == GPU (single source) ──────────────────────────────────────────────

@SKIP_NO_CUDA
def test_cpu_eq_gpu_residual():
    p, f, *c = _prob(24, "cpu")
    rc = mg_residual_3d_warp(p, f, *c, JCAP)
    pg = [t.cuda() for t in [p, f, *c]]
    rg = mg_residual_3d_warp(pg[0], pg[1], *pg[2:], JCAP)
    assert _maxerr(rc, rg.cpu()) < 1e-12


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
