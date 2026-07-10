"""Warp cvof_sweep single-source check: Warp CPU == Warp GPU.

Exercises 2-D and 3-D, every face_dim, and a STRIDED velocity row view (the
production case: a component of the stacked _vel tensor) to validate the
flat-pointer + explicit-stride path (HANDOFF lesson 14).
"""
from __future__ import annotations

import pytest
import torch
import warp as wp

from lilytorch.src.cvof import cvof_sweep_warp
from lilytorch.src.native import cvof_sweep as cvof_sweep_native

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


@SKIP_NO_CUDA
@pytest.mark.parametrize("shape", [(40, 32), (24, 20, 18)])
@pytest.mark.parametrize("strided", [False, True])
def test_cvof_cpu_eq_gpu(shape, strided):
    for fd in range(len(shape)):
        ac, uc = _alpha_vel(shape, "cpu", 4, strided)
        ag, ug = _alpha_vel(shape, "cuda:0", 4, strided)
        oc = ac.clone(); cvof_sweep_warp(ac, uc, CFL, fd, oc)
        og = ag.clone(); cvof_sweep_warp(ag, ug, CFL, fd, og)
        wp.synchronize()
        d = (oc - og.cpu()).abs().max().item()
        assert d < 1e-12, f"shape={shape} strided={strided} fd={fd} cpu vs gpu {d:.3e}"


# cuda_native_port Phase 0.2 parity gate: native cvof_sweep == Warp oracle.
# CUDA-only — the native ``cvof_sweep`` op has no CPU twin yet (ground rule 4),
# so two_phase.py keeps the Warp cvof on CPU.  This gate documents that the
# native CUDA kernel is bit-parity with the oracle, ready for the swap once the
# ``at::parallel_for`` CPU twin lands.
@SKIP_NO_CUDA
@pytest.mark.parametrize("shape", [(40, 32), (24, 20, 18)])
@pytest.mark.parametrize("strided", [False, True])
def test_cvof_native_eq_warp(shape, strided):
    for fd in range(len(shape)):
        a, u = _alpha_vel(shape, "cuda:0", 4, strided)
        o_warp = a.clone(); cvof_sweep_warp(a, u, CFL, fd, o_warp)
        o_nat = a.clone(); cvof_sweep_native(a, u, CFL, fd, o_nat)
        wp.synchronize()
        d = (o_warp - o_nat).abs().max().item()
        assert d < 1e-12, f"shape={shape} strided={strided} fd={fd} native vs warp {d:.3e}"
