"""Parity tests: Warp **bdim_forcing** (`bdim_forcing_3d` + FD normals) vs native.

Validates ``bdim_forcing_3d_warp`` (and its BDIM-σ keyword path) against the native
CUDA op — the parity oracle — on manufactured sphere-SDF + random-field scenes:
full-grid and interior dirty-AABB sub-blocks, ``mu0_projection`` 0/1, and the
σ-shifted-coefficient variant.  Also checks Warp CPU == Warp GPU (single source).

Note on the native CPU op: ``bdim_forcing_3d_cpu`` writes the Poisson coefficient
at the *padded-grid* flat index (``c_out[g]``), whereas the CUDA op writes the
*compact face grid* (``c_out[(i-1)·…]``) — the two native impls use different
``ch/cv/cw`` layouts.  The production path (``solver.py``) is CUDA + the compact
layout, which is what this Warp port reproduces.  So coefficient parity is taken
vs native **CUDA**; on CPU only the velocity fields ``u0/v0/w0`` (padded-`g`
write, identical in both native impls) are compared to native CPU.

Run:  pytest lilytorch/warp_poc/test_bdim.py -v
      python -m lilytorch.src.kernels.test_bdim
"""
from __future__ import annotations

import pytest
import torch

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import (
        bdim_forcing_3d as native_3d,
        bdim_forcing_sigma_3d as native_sigma_3d,
    )
    _NATIVE = True
except Exception:
    _NATIVE = False

from lilytorch.src.kernels.bdim import bdim_forcing_3d_warp

SKIP_NO_NATIVE = pytest.mark.skipif(not _NATIVE, reason="native _C.so unavailable")
SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

NGX, NGY, NGZ, H = 40, 34, 30, 0.05
RHO, DT = 1000.0, 1e-3
EPS = 2 * H


# ─── scene builder (sphere face SDFs + smooth/random body & advdiff fields) ───

def _fields(dev):
    """Build the manufactured problem ONCE on CPU then ``.to(dev)`` — torch's
    per-device generators differ for the same seed (HANDOFF lesson 5)."""
    torch.manual_seed(7)
    xs = (torch.arange(NGX) - NGX / 2.0) * H
    ys = (torch.arange(NGY) - NGY / 2.0) * H
    zs = (torch.arange(NGZ) - NGZ / 2.0) * H
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    R = 0.35

    def sph(ox, oy, oz):
        return (torch.sqrt((X - ox) ** 2 + (Y - oy) ** 2 + (Z - oz) ** 2) - R).double()

    su, sv, sw = sph(0.5 * H, 0, 0), sph(0, 0.5 * H, 0), sph(0, 0, 0.5 * H)

    def rnd():
        return torch.randn(NGX, NGY, NGZ, dtype=torch.float64)

    flds = [su, sv, sw, rnd(), rnd(), rnd(), rnd(), rnd(), rnd()]
    return [t.to(dev) for t in flds]


def _cbuf(dev, dtype=torch.float64):
    base = DT / RHO
    return (torch.full((NGX - 1, NGY - 2, NGZ - 2), base, dtype=dtype, device=dev),
            torch.full((NGX - 2, NGY - 1, NGZ - 2), base, dtype=dtype, device=dev),
            torch.full((NGX - 2, NGY - 2, NGZ - 1), base, dtype=dtype, device=dev))


def _run_warp(F, dirty, mu0_proj, sigma=None):
    su, sv, sw, bu, bv, bw, up, vp, wp_ = F
    u0, v0, w0 = up.clone(), vp.clone(), wp_.clone()
    ch, cv, cw = _cbuf(su.device, su.dtype)
    kw = {}
    if sigma is not None:
        ku, kv, kwk, ss = sigma
        kw = dict(key_u=ku, key_v=kv, key_w=kwk, sigma_shifts=ss)
    bdim_forcing_3d_warp(up, vp, wp_, su, sv, sw, bu, bv, bw,
                       u0, v0, w0, ch, cv, cw, EPS, RHO, DT, H, *dirty,
                       mu0_proj, **kw)
    return u0, v0, w0, ch, cv, cw


def _run_native(F, dirty, mu0_proj, sigma=None):
    su, sv, sw, bu, bv, bw, up, vp, wp_ = F
    u0, v0, w0 = up.clone(), vp.clone(), wp_.clone()
    ch, cv, cw = _cbuf(su.device, su.dtype)
    if sigma is None:
        native_3d(up, vp, wp_, su, sv, sw, bu, bv, bw,
                  u0, v0, w0, ch, cv, cw, EPS, RHO, DT, H, *dirty, mu0_proj)
    else:
        ku, kv, kwk, ss = sigma
        native_sigma_3d(up, vp, wp_, su, sv, sw, bu, bv, bw,
                        u0, v0, w0, ch, cv, cw, ku, kv, kwk, ss,
                        EPS, RHO, DT, H, *dirty, mu0_proj)
    return u0, v0, w0, ch, cv, cw


def _fields_f32(dev):
    return [t.float() for t in _fields(dev)]


def _maxerr(a, b):
    return [(x - y).abs().max().item() for x, y in zip(a, b)]


# ─── GPU parity vs native CUDA (the production path: compact face-grid c) ─────

@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_gpu_parity_full(mu0_proj):
    F = _fields("cuda:0")
    dirty = (0, 0, 0, NGX, NGY, NGZ)
    err = _maxerr(_run_native(F, dirty, mu0_proj), _run_warp(F, dirty, mu0_proj))
    assert max(err) == 0.0, f"full mu0_proj={mu0_proj}: {err}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_gpu_parity_subblock(mu0_proj):
    F = _fields("cuda:0")
    dirty = (5, 4, 3, 20, 18, 16)   # interior AABB sub-block
    err = _maxerr(_run_native(F, dirty, mu0_proj), _run_warp(F, dirty, mu0_proj))
    assert max(err) == 0.0, f"subblock mu0_proj={mu0_proj}: {err}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_gpu_parity_full_f32(mu0_proj):
    """float32: Warp dtype-generic port vs native f32 (single-precision tol)."""
    F = _fields_f32("cuda:0")
    dirty = (0, 0, 0, NGX, NGY, NGZ)
    err = _maxerr(_run_native(F, dirty, mu0_proj), _run_warp(F, dirty, mu0_proj))
    # f32 FMA/ULP drift between the native CUDA codegen and Warp's.
    assert max(err) < 1e-5, f"full f32 mu0_proj={mu0_proj}: {err}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_gpu_parity_subblock_f32(mu0_proj):
    F = _fields_f32("cuda:0")
    dirty = (5, 4, 3, 20, 18, 16)
    err = _maxerr(_run_native(F, dirty, mu0_proj), _run_warp(F, dirty, mu0_proj))
    assert max(err) < 1e-5, f"subblock f32 mu0_proj={mu0_proj}: {err}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_gpu_parity_sigma(mu0_proj):
    F = _fields("cuda:0")
    dirty = (0, 0, 0, NGX, NGY, NGZ)
    dvol = NGX * NGY * NGZ
    torch.manual_seed(3)
    g = torch.Generator(device="cuda:0").manual_seed(3)
    ku = torch.randint(0, 2, (dvol,), dtype=torch.int64, device="cuda:0", generator=g)
    kv = torch.randint(0, 2, (dvol,), dtype=torch.int64, device="cuda:0", generator=g)
    kw = torch.randint(0, 2, (dvol,), dtype=torch.int64, device="cuda:0", generator=g)
    ss = torch.tensor([0.05, 0.12], dtype=torch.float32, device="cuda:0")
    sig = (ku, kv, kw, ss)
    err = _maxerr(_run_native(F, dirty, mu0_proj, sig),
                  _run_warp(F, dirty, mu0_proj, sig))
    assert max(err) == 0.0, f"sigma mu0_proj={mu0_proj}: {err}"


# ─── CPU: velocity-field parity vs native CPU (padded-g write, same layout) ───

@SKIP_NO_NATIVE
def test_cpu_velocity_parity():
    F = _fields("cpu")
    su, sv, sw, bu, bv, bw, up, vp, wp_ = F
    dirty = (0, 0, 0, NGX, NGY, NGZ)
    # native CPU writes c at padded-grid index → feed it full-grid c scratch so
    # it doesn't corrupt the heap; we only compare velocities here.
    u0n, v0n, w0n = up.clone(), vp.clone(), wp_.clone()
    cf = lambda: torch.full((NGX, NGY, NGZ), DT / RHO, dtype=torch.float64)
    native_3d(up, vp, wp_, su, sv, sw, bu, bv, bw,
              u0n, v0n, w0n, cf(), cf(), cf(), EPS, RHO, DT, H, *dirty, 1)
    uw = _run_warp(F, dirty, 1)
    err = _maxerr((u0n, v0n, w0n), uw[:3])
    assert max(err) == 0.0, f"cpu velocity: {err}"


# ─── single source: Warp CPU == Warp GPU ──────────────────────────────────────

@SKIP_NO_CUDA
def test_cpu_eq_gpu():
    dirty = (0, 0, 0, NGX, NGY, NGZ)
    wc = _run_warp(_fields("cpu"), dirty, 1)
    wg = _run_warp(_fields("cuda:0"), dirty, 1)
    err = [(a.cpu() - b.cpu()).abs().max().item() for a, b in zip(wc, wg)]
    # float64 reduction noise on the normalised normal / mu1 polynomial.
    assert max(err) < 1e-12, f"warp cpu vs gpu: {err}"


if __name__ == "__main__":
    if torch.cuda.is_available():
        F = _fields("cuda:0")
        for mp in (1, 0):
            e = _maxerr(_run_native(F, (0, 0, 0, NGX, NGY, NGZ), mp),
                        _run_warp(F, (0, 0, 0, NGX, NGY, NGZ), mp))
            print(f"  GPU full   mu0_proj={mp}  maxerr={max(e):.2e}")
    print("  (run via pytest for the full matrix)")
