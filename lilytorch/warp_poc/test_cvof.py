"""Parity: Warp cvof_sweep vs native (CUDA-only) + Warp CPU == Warp GPU.

Exercises 2-D and 3-D, every face_dim, and a STRIDED velocity row view (the
production case: a component of the stacked _vel tensor) to validate the
flat-pointer + explicit-stride path (HANDOFF lesson 14).
"""
from __future__ import annotations

import pytest
import torch
import warp as wp

try:
    import lilytorch.src.kernels  # noqa: F401
    _NATIVE = torch.ops.lilytorch_kernels.cvof_sweep is not None
except Exception:
    _NATIVE = False

from lilytorch.warp_poc.warp_cvof import cvof_sweep_warp

SKIP_NO_NATIVE = pytest.mark.skipif(not _NATIVE, reason="native cvof_sweep unavailable")
SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
CFL = 0.3


def _alpha_vel(shape, dev, seed, strided):
    torch.manual_seed(seed)
    a = torch.rand(shape, dtype=torch.float64)
    if strided:
        # velocity as a trailing-axis component → genuinely strided row view.
        vel = torch.randn(*shape, 2, dtype=torch.float64)
        u = vel[..., 0]
    else:
        u = torch.randn(shape, dtype=torch.float64)
    a = a.to(dev)
    u = u.to(dev) if not strided else vel.to(dev)[..., 0]
    return a, u


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("shape", [(40, 32), (24, 20, 18)])
@pytest.mark.parametrize("strided", [False, True])
def test_cvof_gpu_parity(shape, strided):
    a, u = _alpha_vel(shape, "cuda:0", 3, strided)
    for fd in range(len(shape)):
        out_n = a.clone()
        torch.ops.lilytorch_kernels.cvof_sweep(a, u, CFL, fd, out_n)
        out_w = a.clone()
        cvof_sweep_warp(a, u, CFL, fd, out_w)
        wp.synchronize()
        d = (out_n - out_w).abs().max().item()
        assert d == 0.0, f"shape={shape} strided={strided} fd={fd} maxdiff {d:.3e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("shape", [(40, 32), (24, 20, 18)])
def test_cvof_cpu_eq_gpu(shape):
    for fd in range(len(shape)):
        ac, uc = _alpha_vel(shape, "cpu", 4, False)
        ag, ug = _alpha_vel(shape, "cuda:0", 4, False)
        oc = ac.clone(); cvof_sweep_warp(ac, uc, CFL, fd, oc)
        og = ag.clone(); cvof_sweep_warp(ag, ug, CFL, fd, og)
        wp.synchronize()
        d = (oc - og.cpu()).abs().max().item()
        assert d < 1e-12, f"shape={shape} fd={fd} cpu vs gpu {d:.3e}"
