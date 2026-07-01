"""Parity: 2-D Warp Poisson smoothers vs native rbgs_sweep_2d / jacobi_sweep_2d.

See `warp_poisson_2d` for the tiled-vs-thread-per-cell parity nuance:
  * Jacobi nsmoothing=1 → bit-exact on any grid.
  * RBGS  nsmoothing=1 → bit-exact only single-tile (Nx<=8, Ny<=32); otherwise
    converges + matches native within the block-approx gap.
"""
from __future__ import annotations

import pytest
import torch
import warp as wp

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import rbgs_sweep_2d, jacobi_sweep_2d
    _NATIVE = True
except Exception:
    _NATIVE = False

from lilytorch.src.kernels.poisson_2d import WarpRBGS2D

SKIP_NO_NATIVE = pytest.mark.skipif(not _NATIVE, reason="native _C.so unavailable")
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


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("w", [1.0, 0.8])
def test_jacobi_bitexact_multiblock(w):
    """Jacobi nsmoothing=1 is bit-exact on a multi-tile grid."""
    Nx, Ny = 96, 64
    prob = _problem(Nx, Ny, "cuda:0")
    p, f, coeffs = prob
    pn = p.clone()
    jacobi_sweep_2d(pn, f, *coeffs, 1e-30, w, 1)
    torch.cuda.synchronize()
    pw = _warp_run(prob, "cuda:0", "jacobi", 1, w)
    d = (pn[1:-1, 1:-1] - pw[1:-1, 1:-1]).abs().max().item()
    assert d == 0.0, f"jacobi w={w} maxdiff {d:.3e}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
def test_rbgs_bitexact_singletile():
    """RBGS nsmoothing=1 bit-exact when the grid fits one 8x32 tile."""
    Nx, Ny = 8, 32
    prob = _problem(Nx, Ny, "cuda:0")
    p, f, coeffs = prob
    pn = p.clone()
    rbgs_sweep_2d(pn, f, *coeffs, 1e-30, 1)
    torch.cuda.synchronize()
    pw = _warp_run(prob, "cuda:0", "rbgs", 1)
    d = (pn[1:-1, 1:-1] - pw[1:-1, 1:-1]).abs().max().item()
    assert d == 0.0, f"rbgs single-tile maxdiff {d:.3e}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
def test_rbgs_multiblock_diff_only_at_tile_seams():
    """On a multi-tile grid the global Warp RBGS is bit-exact EXCEPT at the
    8x32 tile seams: native's red half-sweep reads only pre-sweep values (so
    every red cell matches), and the black half-sweep differs ONLY where a red
    neighbour is read across a tile boundary.  We assert bit-exactness on every
    interior cell that is not adjacent to a seam in either axis."""
    TI, TJ = 8, 32  # native RBGS_2D_I, RBGS_2D_J
    Nx, Ny = 96, 64
    prob = _problem(Nx, Ny, "cuda:0")
    p, f, coeffs = prob
    pn = p.clone()
    rbgs_sweep_2d(pn, f, *coeffs, 1e-30, 1)
    torch.cuda.synchronize()
    pw = _warp_run(prob, "cuda:0", "rbgs", 1)
    diff = (pn[1:-1, 1:-1] - pw[1:-1, 1:-1]).abs()           # (Nx,Ny)
    gi = torch.arange(Nx, device="cuda:0").view(-1, 1)
    gj = torch.arange(Ny, device="cuda:0").view(1, -1)
    seam_i = (gi % TI == 0) | (gi % TI == TI - 1)
    seam_j = (gj % TJ == 0) | (gj % TJ == TJ - 1)
    off_seam = (~seam_i) & (~seam_j)
    assert diff[off_seam].max().item() == 0.0, (
        f"off-seam max diff {diff[off_seam].max().item():.3e} (should be bit-exact)")
    # And confirm the seam cells are where the (expected) differences live.
    assert diff.max().item() > 0.0, "expected nonzero diff at tile seams"


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
@pytest.mark.parametrize("kind", ["rbgs", "jacobi"])
def test_cpu_eq_gpu(kind):
    Nx, Ny = 48, 40
    pc = _warp_run(_problem(Nx, Ny, "cpu"), "cpu", kind, 3)
    pg = _warp_run(_problem(Nx, Ny, "cuda:0"), "cuda:0", kind, 3)
    d = (pc[1:-1, 1:-1] - pg[1:-1, 1:-1].cpu()).abs().max().item()
    assert d < 1e-4, f"{kind} cpu vs gpu {d:.3e}"
