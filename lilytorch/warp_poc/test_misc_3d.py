"""Parity: Warp interp_3d / apply_bcs_3d vs native (3-D coverage fill)."""
from __future__ import annotations

import pytest
import torch
import warp as wp

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import interp_3d as nat_interp_3d
    from lilytorch.src.kernels.ops import apply_bcs_3d as nat_apply_bcs_3d
    _NATIVE = True
except Exception:
    _NATIVE = False

from lilytorch.warp_poc.warp_misc_3d import interp_3d_warp, apply_bcs_3d_warp

SKIP_NO_NATIVE = pytest.mark.skipif(not _NATIVE, reason="native _C.so unavailable")
SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def _interp_problem(dev, Mx=20, My=18, Mz=16, N=3000, seed=5):
    torch.manual_seed(seed)
    xs = torch.linspace(-0.5, 0.5, Mx); ys = torch.linspace(-0.4, 0.4, My); zs = torch.linspace(-0.3, 0.3, Mz)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    F = (torch.sin(3*X)*torch.cos(2*Y)*torch.sin(Z)).float()
    b = (float(xs[0]), float(ys[0]), float(zs[0]))
    inv = (1.0/float(xs[1]-xs[0]), 1.0/float(ys[1]-ys[0]), 1.0/float(zs[1]-zs[0]))
    xq = torch.rand(N)*1.2-0.6; yq = torch.rand(N)*1.0-0.5; zq = torch.rand(N)*0.8-0.4
    return F.to(dev), xq.to(dev), yq.to(dev), zq.to(dev), b, inv, (Mx, My, Mz)


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["linear", "quadratic"])
def test_interp_3d_gpu(method):
    F, xq, yq, zq, b, inv, M = _interp_problem("cuda:0")
    gn = nat_interp_3d(F, xq, yq, zq, *b, *inv, *M, method)
    gw = interp_3d_warp(F, xq, yq, zq, *b, *inv, *M, method)
    wp.synchronize()
    d = (gn.float() - gw).abs().max().item()
    assert d < 1e-6, f"interp3d {method} maxdiff {d:.3e}"


@SKIP_NO_NATIVE
@pytest.mark.parametrize("method", ["linear", "quadratic"])
def test_interp_3d_cpu(method):
    F, xq, yq, zq, b, inv, M = _interp_problem("cpu")
    gn = nat_interp_3d(F, xq, yq, zq, *b, *inv, *M, method)
    gw = interp_3d_warp(F, xq, yq, zq, *b, *inv, *M, method)
    wp.synchronize()
    d = (gn.float() - gw).abs().max().item()
    assert d == 0.0, f"interp3d cpu {method} maxdiff {d:.3e}"


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


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
def test_apply_bcs_3d_gpu():
    u, v, w, shapes, neu, dird, dirv, refd, refv, M = _bcs_problem("cuda:0")
    un, vn, wn = u.clone().contiguous(), v.clone().contiguous(), w.clone().contiguous()
    uw, vw, ww = u.clone().contiguous(), v.clone().contiguous(), w.clone().contiguous()
    nat_apply_bcs_3d(un, vn, wn, shapes, neu, dird, dirv, refd, refv, M, M)
    apply_bcs_3d_warp(uw, vw, ww, shapes, neu, dird, dirv, refd, refv, M)
    wp.synchronize()
    for a, b, nm in ((un, uw, "u"), (vn, vw, "v"), (wn, ww, "w")):
        assert (a - b).abs().max().item() == 0.0, f"bcs3d {nm} mismatch"


@SKIP_NO_NATIVE
def test_apply_bcs_3d_cpu():
    u, v, w, shapes, neu, dird, dirv, refd, refv, M = _bcs_problem("cpu")
    un, vn, wn = u.clone().contiguous(), v.clone().contiguous(), w.clone().contiguous()
    uw, vw, ww = u.clone().contiguous(), v.clone().contiguous(), w.clone().contiguous()
    nat_apply_bcs_3d(un, vn, wn, shapes, neu, dird, dirv, refd, refv, M, M)
    apply_bcs_3d_warp(uw, vw, ww, shapes, neu, dird, dirv, refd, refv, M)
    wp.synchronize()
    for a, b, nm in ((un, uw, "u"), (vn, vw, "v"), (wn, ww, "w")):
        assert (a - b).abs().max().item() == 0.0, f"bcs3d cpu {nm} mismatch"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
def test_apply_bcs_3d_gpu_f32():
    """f32 dtype-generic parity (bit-exact: BC writes are copies / value sets)."""
    u, v, w, shapes, neu, dird, dirv, refd, refv, M = _bcs_problem("cuda:0")
    u = u.float(); v = v.float(); w = w.float(); dirv = dirv.float(); refv = refv.float()
    un, vn, wn = u.clone().contiguous(), v.clone().contiguous(), w.clone().contiguous()
    uw, vw, ww = u.clone().contiguous(), v.clone().contiguous(), w.clone().contiguous()
    nat_apply_bcs_3d(un, vn, wn, shapes, neu, dird, dirv, refd, refv, M, M)
    apply_bcs_3d_warp(uw, vw, ww, shapes, neu, dird, dirv, refd, refv, M, M)
    wp.synchronize()
    for a, b, nm in ((un, uw, "u"), (vn, vw, "v"), (wn, ww, "w")):
        assert (a - b).abs().max().item() == 0.0, f"bcs3d f32 {nm} mismatch"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
def test_apply_bcs_3d_noncubic_dual_facedims():
    """Non-cubic grid with separate (max_dim0, max_dim1) matches native exactly."""
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
    to = lambda t: t.to("cuda:0")
    args = [to(x) for x in (u, v, w, shapes, neu, dird, dirv, refd, refv)]
    un, vn, wn = (args[0].clone().contiguous(), args[1].clone().contiguous(),
                  args[2].clone().contiguous())
    uw, vw, ww = (args[0].clone().contiguous(), args[1].clone().contiguous(),
                  args[2].clone().contiguous())
    nat_apply_bcs_3d(un, vn, wn, *args[3:9], max_dim0, max_dim1)
    apply_bcs_3d_warp(uw, vw, ww, *args[3:9], max_dim0, max_dim1)
    wp.synchronize()
    for a, b, nm in ((un, uw, "u"), (vn, vw, "v"), (wn, ww, "w")):
        assert (a - b).abs().max().item() == 0.0, f"noncubic {nm} mismatch"


@SKIP_NO_NATIVE
@pytest.mark.parametrize("method", ["linear", "quadratic"])
def test_interp_3d_cpu_f64(method):
    """f64 dtype-generic interp parity (CPU bit-exact: no FMA contraction)."""
    F, xq, yq, zq, b, inv, M = _interp_problem("cpu")
    F = F.double(); xq = xq.double(); yq = yq.double(); zq = zq.double()
    gn = nat_interp_3d(F, xq, yq, zq, *b, *inv, *M, method)
    gw = interp_3d_warp(F, xq, yq, zq, *b, *inv, *M, method)
    wp.synchronize()
    assert gw.dtype == torch.float64
    d = (gn - gw).abs().max().item()
    assert d == 0.0, f"interp3d f64 cpu {method} maxdiff {d:.3e}"
