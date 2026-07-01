"""Parity: Warp interp_2d / apply_bcs_2d vs native (VALIDATION_STATUS §D)."""
from __future__ import annotations

import pytest
import torch
import warp as wp

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import interp_2d as native_interp_2d
    from lilytorch.src.kernels.ops import apply_bcs_2d as native_apply_bcs_2d
    _NATIVE = True
except Exception:
    _NATIVE = False

from lilytorch.warp_poc.warp_misc_2d import interp_2d_warp, apply_bcs_2d_warp

SKIP_NO_NATIVE = pytest.mark.skipif(not _NATIVE, reason="native _C.so unavailable")
SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


# ─── interp_2d ───────────────────────────────────────────────────────────────

def _interp_problem(dev, Mx=30, My=24, N=4000, seed=5):
    torch.manual_seed(seed)
    xs = torch.linspace(-0.5, 0.5, Mx)
    ys = torch.linspace(-0.4, 0.4, My)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    F = (torch.sin(3 * X) * torch.cos(2 * Y)).float()
    bx0, by0 = float(xs[0]), float(ys[0])
    inv_dx = 1.0 / float(xs[1] - xs[0])
    inv_dy = 1.0 / float(ys[1] - ys[0])
    # queries spanning (and slightly past) the domain → exercise clamp
    xq = (torch.rand(N) * 1.2 - 0.6)
    yq = (torch.rand(N) * 1.0 - 0.5)
    return (F.to(dev), xq.to(dev), yq.to(dev), bx0, by0, inv_dx, inv_dy, Mx, My)


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["linear", "quadratic"])
def test_interp_2d_gpu(method):
    # GPU parity to ~1 float32 ULP: native CUDA and Warp CUDA contract the
    # bilinear blend FMAs in a slightly different order (the CPU path below is
    # bit-exact since neither contracts).
    F, xq, yq, bx0, by0, idx, idy, Mx, My = _interp_problem("cuda:0")
    gn = native_interp_2d(F, xq, yq, bx0, by0, idx, idy, Mx, My, method)
    gw = interp_2d_warp(F, xq, yq, bx0, by0, idx, idy, Mx, My, method)
    wp.synchronize()
    d = (gn.float() - gw).abs().max().item()
    assert d < 1e-6, f"interp {method} maxdiff {d:.3e}"


@SKIP_NO_NATIVE
@pytest.mark.parametrize("method", ["linear", "quadratic"])
def test_interp_2d_cpu(method):
    F, xq, yq, bx0, by0, idx, idy, Mx, My = _interp_problem("cpu")
    gn = native_interp_2d(F, xq, yq, bx0, by0, idx, idy, Mx, My, method)
    gw = interp_2d_warp(F, xq, yq, bx0, by0, idx, idy, Mx, My, method)
    wp.synchronize()
    d = (gn.float() - gw).abs().max().item()
    assert d == 0.0, f"interp cpu {method} maxdiff {d:.3e}"


# ─── apply_bcs_2d ────────────────────────────────────────────────────────────

def _bcs_problem(dev, Nx=40, Ny=32, seed=9):
    torch.manual_seed(seed)
    u = torch.randn(Nx, Ny, dtype=torch.float64)
    v = torch.randn(Nx, Ny, dtype=torch.float64)
    shapes = torch.tensor([[Nx, Ny], [Nx, Ny]], dtype=torch.int64)
    # Descriptors chosen so no two STAGE-1 (Neumann+Dirichlet) ops share a
    # cell — overlapping stage-1 writes are order-undefined on GPU (native's
    # own note), so a deterministic bit-exact comparison needs disjoint ops.
    # Neumann: u rows 0 and Nx-1 (axis0, both sides).
    neu = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.int32)
    # Dirichlet: v cols 0 and Ny-1 (axis1).  Different field from Neumann.
    dird = torch.tensor([[1, 1, 0], [1, 1, -1]], dtype=torch.int32)
    dirv = torch.tensor([2.5, -1.3], dtype=torch.float64)
    # Reflective (stage 2 → runs last, deterministic even at corners): u col Ny-1.
    refd = torch.tensor([[0, 1, -1, -2]], dtype=torch.int32)
    refv = torch.tensor([0.4], dtype=torch.float64)
    max_line = max(Nx, Ny)
    to = lambda t: t.to(dev)
    return (to(u), to(v), to(shapes), to(neu), to(dird), to(dirv),
            to(refd), to(refv), max_line)


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
def test_apply_bcs_2d_gpu():
    p = _bcs_problem("cuda:0")
    u, v, shapes, neu, dird, dirv, refd, refv, ml = p
    un, vn = u.clone().contiguous(), v.clone().contiguous()
    uw, vw = u.clone().contiguous(), v.clone().contiguous()
    native_apply_bcs_2d(un, vn, shapes, neu, dird, dirv, refd, refv, ml)
    apply_bcs_2d_warp(uw, vw, shapes, neu, dird, dirv, refd, refv, ml)
    wp.synchronize()
    du = (un - uw).abs().max().item(); dv = (vn - vw).abs().max().item()
    assert du == 0.0 and dv == 0.0, f"bcs maxdiff u={du:.3e} v={dv:.3e}"


@SKIP_NO_NATIVE
def test_apply_bcs_2d_cpu():
    p = _bcs_problem("cpu")
    u, v, shapes, neu, dird, dirv, refd, refv, ml = p
    un, vn = u.clone().contiguous(), v.clone().contiguous()
    uw, vw = u.clone().contiguous(), v.clone().contiguous()
    native_apply_bcs_2d(un, vn, shapes, neu, dird, dirv, refd, refv, ml)
    apply_bcs_2d_warp(uw, vw, shapes, neu, dird, dirv, refd, refv, ml)
    wp.synchronize()
    du = (un - uw).abs().max().item(); dv = (vn - vw).abs().max().item()
    assert du == 0.0 and dv == 0.0, f"bcs cpu maxdiff u={du:.3e} v={dv:.3e}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
def test_apply_bcs_2d_gpu_f32():
    """f32 dtype-generic parity (bit-exact: BC writes are copies / value sets)."""
    p = _bcs_problem("cuda:0")
    u, v, shapes, neu, dird, dirv, refd, refv, ml = p
    u = u.float(); v = v.float(); dirv = dirv.float(); refv = refv.float()
    un, vn = u.clone().contiguous(), v.clone().contiguous()
    uw, vw = u.clone().contiguous(), v.clone().contiguous()
    native_apply_bcs_2d(un, vn, shapes, neu, dird, dirv, refd, refv, ml)
    apply_bcs_2d_warp(uw, vw, shapes, neu, dird, dirv, refd, refv, ml)
    wp.synchronize()
    du = (un - uw).abs().max().item(); dv = (vn - vw).abs().max().item()
    assert du == 0.0 and dv == 0.0, f"bcs f32 maxdiff u={du:.3e} v={dv:.3e}"


@SKIP_NO_NATIVE
@pytest.mark.parametrize("method", ["linear", "quadratic"])
def test_interp_2d_cpu_f64(method):
    """f64 dtype-generic interp parity (CPU bit-exact: no FMA contraction)."""
    F, xq, yq, bx0, by0, idx, idy, Mx, My = _interp_problem("cpu")
    F = F.double(); xq = xq.double(); yq = yq.double()
    gn = native_interp_2d(F, xq, yq, bx0, by0, idx, idy, Mx, My, method)
    gw = interp_2d_warp(F, xq, yq, bx0, by0, idx, idy, Mx, My, method)
    wp.synchronize()
    assert gw.dtype == torch.float64
    d = (gn - gw).abs().max().item()
    assert d == 0.0, f"interp f64 cpu {method} maxdiff {d:.3e}"
