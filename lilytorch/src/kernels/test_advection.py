"""Advection Warp-kernel tests.

Two layers:
  (1) First-order upwind Warp kernel vs a PyTorch reference (CPU+GPU) — the
      original viability demo.
  (2) HIGH-ORDER LIMITER port (`advect_flux_add_warp`) single-source check:
      the SAME @wp.kernel on CPU == GPU, plus in-place accumulate semantics.
      Covers all five schemes (QUICK / ABDQUICKEST / vanLeer / CDS / CUBISTA),
      2-D and 3-D, every (velocity component i, direction d) pair (so the
      rhs-stride caveat — face_dim ≠ outermost in rhs — is exercised for d>0),
      built from the genuine strided slice views of advection.py.

Run:  pytest lilytorch/src/kernels/test_advection.py -v
      python -m lilytorch.src.kernels.test_advection
"""
from __future__ import annotations

import itertools

import pytest
import torch
import warp as wp

from lilytorch.src.kernels.advection import (
    advect_upwind_3d,
    advect_upwind_torch,
    advect_flux_add_warp,
)
from lilytorch.src.advection import (
    _face_vel,
    _field_for_flux,
    _inner,
)

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

# scheme_id → name (matches _CUDA_SCHEME_IDS in advection.py)
_SCHEMES = {0: "quick", 1: "abdquickest", 2: "vanLeer", 3: "cds", 4: "cubista"}


# ─────────────────────────────────────────────────────────────────────────────
#  (1) first-order upwind viability demo (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _pad(x):
    return torch.nn.functional.pad(
        x.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]


def _run_upwind(dev, N=32, seed=7):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.rand((N, N, N), generator=g, device=dev)
    u = torch.rand((N, N, N), generator=g, device=dev) - 0.5
    v = torch.rand((N, N, N), generator=g, device=dev) - 0.5
    w = torch.rand((N, N, N), generator=g, device=dev) - 0.5
    qp, up, vp, wp_ = (_pad(x) for x in (q, u, v, w))
    dt, inv_h = 0.1, float(N)

    ref = advect_upwind_torch(qp, up, vp, wp_, dt, inv_h)

    out = wp.zeros((N, N, N), dtype=wp.float32, device=dev)
    wp.launch(advect_upwind_3d, dim=(N, N, N),
              inputs=[wp.from_torch(qp.contiguous()), wp.from_torch(up.contiguous()),
                      wp.from_torch(vp.contiguous()), wp.from_torch(wp_.contiguous()),
                      dt, inv_h, out], device=dev)
    wp.synchronize()
    return wp.to_torch(out), ref


@SKIP_NO_CUDA
def test_upwind_matches_reference_gpu():
    out, ref = _run_upwind("cuda:0")
    rel = (out - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-5, f"GPU warp vs torch-ref rel {rel:.3e}"


def test_upwind_matches_reference_cpu():
    out, ref = _run_upwind("cpu")
    rel = (out - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-5, f"CPU warp vs torch-ref rel {rel:.3e}"


# ─────────────────────────────────────────────────────────────────────────────
#  (2) high-order limiter single-source + accumulate semantics
# ─────────────────────────────────────────────────────────────────────────────

def _make_vel(ndim, N, dev, seed=11):
    """Padded MAC velocity field built ONCE on CPU then moved (lesson 5: a
    device-seeded generator yields different sequences per device)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (N + 2,) * ndim
    vel = [torch.rand(shape, generator=g, dtype=torch.float64) - 0.5
           for _ in range(ndim)]
    return [v.to(dev) for v in vel]


def _courant(scheme_id):
    # ABDQUICKEST uses the Courant number; the others ignore it.
    return 0.37 if scheme_id == 1 else 0.0


def _warp(vel, i, d, scheme_id, dt_dh, ndim):
    inner = _inner(ndim)
    fv = _face_vel(vel, i, d, ndim)
    p = _field_for_flux(vel[i], d, ndim)
    rhs = torch.zeros_like(vel[i][inner])
    advect_flux_add_warp(fv, p, rhs, dt_dh, _courant(scheme_id), scheme_id, d)
    wp.synchronize()
    return rhs


@SKIP_NO_CUDA
def test_flux_add_accumulates_in_place():
    """The op must ADD into a pre-seeded rhs (not overwrite), like production."""
    ndim, N, dev = 3, 20, "cuda:0"
    vel = _make_vel(ndim, N, dev, seed=5)
    inner = _inner(ndim)
    g = torch.Generator(device="cpu").manual_seed(99)
    seed_rhs = (torch.rand(vel[0][inner].shape, generator=g, dtype=torch.float64)).to(dev)
    i, d, sid, dt_dh = 0, 2, 0, 0.2  # d != 0 → rhs.stride(face_dim) ≠ outermost
    fv = _face_vel(vel, i, d, ndim)
    p = _field_for_flux(vel[i], d, ndim)

    delta = _warp(vel, i, d, sid, dt_dh, ndim)  # zero-seeded → pure flux term
    got = seed_rhs.clone()
    advect_flux_add_warp(fv, p, got, dt_dh, 0.0, sid, d)
    wp.synchronize()
    # 1-ULP f64 slack: the kernel fuses seed + dt_dh·flux in one expression,
    # while the reference adds them in two rounding steps.
    assert (got - (seed_rhs + delta)).abs().max().item() < 1e-15


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("scheme_id", sorted(_SCHEMES))
def test_flux_add_warp_cpu_equals_gpu(ndim, scheme_id):
    """Single-source check: the SAME @wp.kernel on CPU == GPU."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    N = 20 if ndim == 3 else 40
    dt_dh = 0.15
    vel_cpu = _make_vel(ndim, N, "cpu")
    vel_gpu = [v.to("cuda:0") for v in vel_cpu]
    worst = 0.0
    for i, d in itertools.product(range(ndim), range(ndim)):
        rc = _warp(vel_cpu, i, d, scheme_id, dt_dh, ndim)
        rg = _warp(vel_gpu, i, d, scheme_id, dt_dh, ndim).cpu()
        worst = max(worst, (rc - rg).abs().max().item())
    assert worst < 1e-12, f"{_SCHEMES[scheme_id]} {ndim}-D CPU vs GPU {worst:.3e}"


if __name__ == "__main__":
    for dev in (["cuda:0"] if torch.cuda.is_available() else []) + ["cpu"]:
        out, ref = _run_upwind(dev)
        rel = (out - ref).abs().max().item() / ref.abs().max().item()
        print(f"  upwind {dev:7s}: rel {rel:.2e}  {'PASS' if rel < 1e-5 else 'FAIL'}")
