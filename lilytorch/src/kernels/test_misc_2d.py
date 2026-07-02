"""Warp interp_2d / apply_bcs_2d single-source checks: Warp CPU == Warp GPU."""
from __future__ import annotations

import pytest
import torch
import warp as wp

from lilytorch.src.kernels.misc_2d import interp_2d_warp, apply_bcs_2d_warp

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


# ─── interp_2d ───────────────────────────────────────────────────────────────

def _interp_problem(dev, Mx=30, My=24, N=4000, seed=5, dtype=torch.float32):
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
    pc = _interp_problem("cpu", dtype=dtype)
    pg = _interp_problem("cuda:0", dtype=dtype)
    gc = interp_2d_warp(*pc, method)
    gg = interp_2d_warp(*pg, method)
    wp.synchronize()
    assert gc.dtype == dtype
    d = (gc - gg.cpu()).abs().max().item()
    tol = 1e-6 if dtype == torch.float32 else 1e-14
    assert d < tol, f"interp {method} {dtype} cpu vs gpu maxdiff {d:.3e}"


# ─── apply_bcs_2d ────────────────────────────────────────────────────────────

def _bcs_problem(dev, Nx=40, Ny=32, seed=9):
    torch.manual_seed(seed)
    u = torch.randn(Nx, Ny, dtype=torch.float64)
    v = torch.randn(Nx, Ny, dtype=torch.float64)
    shapes = torch.tensor([[Nx, Ny], [Nx, Ny]], dtype=torch.int64)
    # Descriptors chosen so no two STAGE-1 (Neumann+Dirichlet) ops share a
    # cell — overlapping stage-1 writes are order-undefined on GPU, so a
    # deterministic bit-exact comparison needs disjoint ops.
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


def _run_bcs(dev, f32=False):
    u, v, shapes, neu, dird, dirv, refd, refv, ml = _bcs_problem(dev)
    if f32:
        u = u.float(); v = v.float(); dirv = dirv.float(); refv = refv.float()
    uw, vw = u.clone().contiguous(), v.clone().contiguous()
    apply_bcs_2d_warp(uw, vw, shapes, neu, dird, dirv, refd, refv, ml)
    wp.synchronize()
    return uw, vw


@SKIP_NO_CUDA
@pytest.mark.parametrize("f32", [False, True], ids=["f64", "f32"])
def test_apply_bcs_2d_cpu_eq_gpu(f32):
    """BC writes are copies / value sets → CPU == GPU bit-exact."""
    uc, vc = _run_bcs("cpu", f32)
    ug, vg = _run_bcs("cuda:0", f32)
    du = (uc - ug.cpu()).abs().max().item()
    dv = (vc - vg.cpu()).abs().max().item()
    assert du == 0.0 and dv == 0.0, f"bcs cpu vs gpu maxdiff u={du:.3e} v={dv:.3e}"
