"""Parity: Warp **Kernel B (2-D)** (`bdim_coeff_2d` + FD normals) vs native.

2-D analogue of `test_bdim.py`.  In 2-D the native writes the Poisson coeff at
the full-grid index ``c_out[g]`` in BOTH the CUDA and CPU ops, so coeff parity
is taken vs CUDA *and* CPU (no layout discrepancy).  Also: Warp CPU == Warp GPU.
"""
from __future__ import annotations

import pytest
import torch

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import (
        bdim_coeff_2d as native_2d,
        bdim_coeff_sigma_2d as native_sigma_2d,
    )
    _NATIVE = True
except Exception:
    _NATIVE = False

from lilytorch.src.kernels.bdim_2d import bdim_coeff_2d_warp

SKIP_NO_NATIVE = pytest.mark.skipif(not _NATIVE, reason="native _C.so unavailable")
SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

NGX, NGY, H = 48, 40, 0.05
RHO, DT = 1000.0, 1e-3
EPS = 2 * H


def _fields(dev):
    torch.manual_seed(7)
    xs = (torch.arange(NGX) - NGX / 2.0) * H
    ys = (torch.arange(NGY) - NGY / 2.0) * H
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    R = 0.35

    def disc(ox, oy):
        return (torch.sqrt((X - ox) ** 2 + (Y - oy) ** 2) - R).double()

    su, sv = disc(0.5 * H, 0), disc(0, 0.5 * H)

    def rnd():
        return torch.randn(NGX, NGY, dtype=torch.float64)

    flds = [su, sv, rnd(), rnd(), rnd(), rnd()]
    return [t.to(dev) for t in flds]


def _cbuf(dev):
    base = DT / RHO
    return (torch.full((NGX, NGY), base, dtype=torch.float64, device=dev),
            torch.full((NGX, NGY), base, dtype=torch.float64, device=dev))


def _run_warp(F, dirty, mu0_proj, sigma=None):
    su, sv, bu, bv, up, vp = F
    u0, v0 = up.clone(), vp.clone()
    ch, cv = _cbuf(su.device)
    kw = {}
    if sigma is not None:
        ku, kv, ss = sigma
        kw = dict(key_u=ku, key_v=kv, sigma_shifts=ss)
    bdim_coeff_2d_warp(up, vp, su, sv, bu, bv, u0, v0, ch, cv,
                       EPS, RHO, DT, H, *dirty, mu0_proj, **kw)
    return u0, v0, ch, cv


def _run_native(F, dirty, mu0_proj, sigma=None):
    su, sv, bu, bv, up, vp = F
    u0, v0 = up.clone(), vp.clone()
    ch, cv = _cbuf(su.device)
    if sigma is None:
        native_2d(up, vp, su, sv, bu, bv, u0, v0, ch, cv,
                  EPS, RHO, DT, H, *dirty, mu0_proj)
    else:
        ku, kv, ss = sigma
        native_sigma_2d(up, vp, su, sv, bu, bv, u0, v0, ch, cv, ku, kv, ss,
                        EPS, RHO, DT, H, *dirty, mu0_proj)
    return u0, v0, ch, cv


def _maxerr(a, b):
    return [(x - y).abs().max().item() for x, y in zip(a, b)]


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_gpu_parity_full(mu0_proj):
    F = _fields("cuda:0")
    dirty = (0, 0, NGX, NGY)
    err = _maxerr(_run_native(F, dirty, mu0_proj), _run_warp(F, dirty, mu0_proj))
    assert max(err) == 0.0, f"full mu0_proj={mu0_proj}: {err}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_gpu_parity_subblock(mu0_proj):
    F = _fields("cuda:0")
    dirty = (5, 4, 24, 20)
    err = _maxerr(_run_native(F, dirty, mu0_proj), _run_warp(F, dirty, mu0_proj))
    assert max(err) == 0.0, f"subblock mu0_proj={mu0_proj}: {err}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_gpu_parity_full_f32(mu0_proj):
    """float32: the dtype-generic Kernel B runs on Warp (no native fallback) and
    matches the native f32 op to single precision (not bit-exact: f32 FMA/order)."""
    dev = "cuda:0"
    su, sv, bu, bv, up, vp = [t.float() for t in _fields(dev)]
    dirty = (0, 0, NGX, NGY)
    base = DT / RHO

    def cb():
        return (torch.full((NGX, NGY), base, dtype=torch.float32, device=dev),
                torch.full((NGX, NGY), base, dtype=torch.float32, device=dev))

    u0n, v0n = up.clone(), vp.clone(); chn, cvn = cb()
    native_2d(up, vp, su, sv, bu, bv, u0n, v0n, chn, cvn,
              EPS, RHO, DT, H, *dirty, mu0_proj)
    u0w, v0w = up.clone(), vp.clone(); chw, cvw = cb()
    bdim_coeff_2d_warp(up, vp, su, sv, bu, bv, u0w, v0w, chw, cvw,
                       EPS, RHO, DT, H, *dirty, mu0_proj)
    for nm, a, b in (("u0", u0n, u0w), ("v0", v0n, v0w),
                     ("ch", chn, chw), ("cv", cvn, cvw)):
        assert b.dtype == torch.float32
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), \
            f"{nm} f32 mu0_proj={mu0_proj}: {(a - b).abs().max().item():.2e}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_gpu_parity_sigma(mu0_proj):
    F = _fields("cuda:0")
    dirty = (0, 0, NGX, NGY)
    dvol = NGX * NGY
    g = torch.Generator(device="cuda:0").manual_seed(3)
    ku = torch.randint(0, 2, (dvol,), dtype=torch.int64, device="cuda:0", generator=g)
    kv = torch.randint(0, 2, (dvol,), dtype=torch.int64, device="cuda:0", generator=g)
    ss = torch.tensor([0.05, 0.12], dtype=torch.float32, device="cuda:0")
    sig = (ku, kv, ss)
    err = _maxerr(_run_native(F, dirty, mu0_proj, sig),
                  _run_warp(F, dirty, mu0_proj, sig))
    assert max(err) == 0.0, f"sigma mu0_proj={mu0_proj}: {err}"


@SKIP_NO_NATIVE
def test_cpu_parity():
    """2-D: coeff + velocity parity vs native CPU (full-grid c, no discrepancy)."""
    F = _fields("cpu")
    dirty = (0, 0, NGX, NGY)
    err = _maxerr(_run_native(F, dirty, 1), _run_warp(F, dirty, 1))
    assert max(err) == 0.0, f"cpu parity: {err}"


@SKIP_NO_CUDA
def test_cpu_eq_gpu():
    dirty = (0, 0, NGX, NGY)
    wc = _run_warp(_fields("cpu"), dirty, 1)
    wg = _run_warp(_fields("cuda:0"), dirty, 1)
    err = [(a.cpu() - b.cpu()).abs().max().item() for a, b in zip(wc, wg)]
    assert max(err) < 1e-12, f"warp cpu vs gpu: {err}"
