"""Advection Warp-kernel tests (single production path).

The fused high-order limiter kernel ``advect_flux_add_warp`` is the *only*
convective path (CPU C++/OpenMP and CUDA).  Three layers:

  (1) CPU regression anchor — a frozen (sum, |max|) snapshot of the rhs the
      kernel produces for a fixed seed, for all five schemes (QUICK /
      ABDQUICKEST / vanLeer / CDS / CUBISTA), 2-D and 3-D.  The snapshot was
      captured when the kernel was still validated bit-for-bit against the
      (now-removed) PyTorch reference, so it pins correctness without CUDA.
  (2) Single-source integrity — the SAME @wp.kernel on CPU == GPU (needs CUDA).
  (3) In-place accumulate semantics (rhs += flux, not overwrite).

Every (velocity component i, direction d) pair is exercised so the rhs-stride
caveat (face_dim ≠ outermost in rhs) is covered for d>0, using the genuine
strided slice views from advection.py.

Run:  pytest lilytorch/tests/test_advection.py -v
"""
from __future__ import annotations

import itertools

import pytest
import torch
import warp as wp

from lilytorch.src.advection import (
    advect_flux_add_warp,
    _face_vel,
    _field_for_flux,
    _inner,
)

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

# scheme_id → name (matches _CUDA_SCHEME_IDS in advection.py)
_SCHEMES = {0: "quick", 1: "abdquickest", 2: "vanLeer", 3: "cds", 4: "cubista"}


def _make_vel(ndim, N, dev, seed=11):
    """Padded MAC velocity field built ONCE on CPU then moved (a device-seeded
    generator yields different sequences per device)."""
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


# ─────────────────────────────────────────────────────────────────────────────
#  (1) CPU regression anchor  (no CUDA required)
# ─────────────────────────────────────────────────────────────────────────────

# (ndim, scheme_id) → (sum(rhs), max|rhs|), summed over all (i, d) pairs at
# seed 11, dt_dh 0.15, N=40 (2-D) / N=20 (3-D).  Captured against the kernel
# while it was bit-parity with the removed PyTorch scheme reference.
_REGRESSION = {
    (2, 0): (0.05990405142928154, 0.049610416505477574),
    (2, 1): (0.07741790648773922, 0.049610416505477574),
    (2, 2): (0.07369885379628983, 0.049610416505477574),
    (2, 3): (0.047893404100937315, 0.04886254959572121),
    (2, 4): (0.0762824918396289, 0.049610416505477574),
    (3, 0): (1.1405044788006227, 0.060893124887524866),
    (3, 1): (1.0385373002973184, 0.060893124887524866),
    (3, 2): (1.064901094990712, 0.060893124887524866),
    (3, 3): (1.022053726682977, 0.052619659932840825),
    (3, 4): (1.078541541139582, 0.060893124887524866),
}


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("scheme_id", sorted(_SCHEMES))
def test_flux_add_warp_cpu_regression(ndim, scheme_id):
    """Kernel output on CPU matches the frozen validated snapshot."""
    N = 20 if ndim == 3 else 40
    vel = _make_vel(ndim, N, "cpu")
    dt_dh = 0.15
    s, a = 0.0, 0.0
    for i, d in itertools.product(range(ndim), range(ndim)):
        rhs = _warp(vel, i, d, scheme_id, dt_dh, ndim)
        s += float(rhs.sum())
        a = max(a, float(rhs.abs().max()))
    exp_s, exp_a = _REGRESSION[(ndim, scheme_id)]
    assert abs(s - exp_s) < 1e-12, f"{_SCHEMES[scheme_id]} {ndim}-D sum {s!r}"
    assert abs(a - exp_a) < 1e-12, f"{_SCHEMES[scheme_id]} {ndim}-D max {a!r}"


# ─────────────────────────────────────────────────────────────────────────────
#  (2) single-source integrity  +  (3) accumulate semantics
# ─────────────────────────────────────────────────────────────────────────────

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
