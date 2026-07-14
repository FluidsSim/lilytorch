"""Native ``cvof_sweep`` single-source check: CPU twin == CUDA kernel.

Exercises 2-D and 3-D, every face_dim, and a STRIDED velocity row view (the
production case: a component of the stacked _vel tensor) to validate the
flat-pointer + explicit-stride path (HANDOFF lesson 14).
"""
from __future__ import annotations

import pytest
import torch

from lilytorch.src.native import cvof_sweep

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
    """The ``at::parallel_for`` CPU twin agrees with the CUDA kernel."""
    for fd in range(len(shape)):
        ac, uc = _alpha_vel(shape, "cpu", 4, strided)
        ag, ug = _alpha_vel(shape, "cuda:0", 4, strided)
        oc = ac.clone(); cvof_sweep(ac, uc, CFL, fd, oc)
        og = ag.clone(); cvof_sweep(ag, ug, CFL, fd, og)
        d = (oc - og.cpu()).abs().max().item()
        assert d < 1e-12, f"shape={shape} strided={strided} fd={fd} cpu vs gpu {d:.3e}"
