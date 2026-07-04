"""Warp **bdim_forcing (3-D)** single-source checks: Warp CPU == Warp GPU.

Exercises ``bdim_forcing_3d_warp`` (and its BDIM-σ keyword path) on manufactured
sphere-SDF + random-field scenes: full-grid and interior dirty-AABB sub-blocks,
``mu0_projection`` 0/1, and the σ-shifted-coefficient variant.

Run:  pytest lilytorch/src/kernels/test_bdim.py -v
"""
from __future__ import annotations

import pytest
import torch

from lilytorch.src.bdim import bdim_forcing_3d_warp, bdim_forcing_2d_warp

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


def _sigma(dev):
    dvol = NGX * NGY * NGZ
    g = torch.Generator(device="cpu").manual_seed(3)
    ku = torch.randint(0, 2, (dvol,), dtype=torch.int64, generator=g)
    kv = torch.randint(0, 2, (dvol,), dtype=torch.int64, generator=g)
    kw = torch.randint(0, 2, (dvol,), dtype=torch.int64, generator=g)
    ss = torch.tensor([0.05, 0.12], dtype=torch.float32)
    return ku.to(dev), kv.to(dev), kw.to(dev), ss.to(dev)


# ─── single source: Warp CPU == Warp GPU ──────────────────────────────────────

@SKIP_NO_CUDA
@pytest.mark.parametrize("dirty", [(0, 0, 0, NGX, NGY, NGZ),
                                   (5, 4, 3, 20, 18, 16)],
                         ids=["full", "subblock"])
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_cpu_eq_gpu(dirty, mu0_proj):
    wc = _run_warp(_fields("cpu"), dirty, mu0_proj)
    wg = _run_warp(_fields("cuda:0"), dirty, mu0_proj)
    err = [(a.cpu() - b.cpu()).abs().max().item() for a, b in zip(wc, wg)]
    # float64 reduction noise on the normalised normal / mu1 polynomial.
    assert max(err) < 1e-12, f"warp cpu vs gpu: {err}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_cpu_eq_gpu_sigma(mu0_proj):
    dirty = (0, 0, 0, NGX, NGY, NGZ)
    wc = _run_warp(_fields("cpu"), dirty, mu0_proj, _sigma("cpu"))
    wg = _run_warp(_fields("cuda:0"), dirty, mu0_proj, _sigma("cuda:0"))
    err = [(a.cpu() - b.cpu()).abs().max().item() for a, b in zip(wc, wg)]
    assert max(err) < 1e-12, f"warp cpu vs gpu (sigma): {err}"


if __name__ == "__main__":
    if torch.cuda.is_available():
        wc = _run_warp(_fields("cpu"), (0, 0, 0, NGX, NGY, NGZ), 1)
        wg = _run_warp(_fields("cuda:0"), (0, 0, 0, NGX, NGY, NGZ), 1)
        e = [(a.cpu() - b.cpu()).abs().max().item() for a, b in zip(wc, wg)]
        print(f"  CPU vs GPU full  maxerr={max(e):.2e}")
    print("  (run via pytest for the full matrix)")


# ═════════════════════════════════════════════════════════════════════════════
#  2-D bdim_forcing (bdim_forcing_2d_warp) — merged from the former
#  test_bdim_2d.py.  Symbols carry a `_2d` suffix so both dims coexist.
# ═════════════════════════════════════════════════════════════════════════════
NGX_2D, NGY_2D, H_2D = 48, 40, 0.05
RHO_2D, DT_2D = 1000.0, 1e-3
EPS_2D = 2 * H_2D


def _fields_2d(dev):
    torch.manual_seed(7)
    xs = (torch.arange(NGX_2D) - NGX_2D / 2.0) * H_2D
    ys = (torch.arange(NGY_2D) - NGY_2D / 2.0) * H_2D
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    R = 0.35

    def disc(ox, oy):
        return (torch.sqrt((X - ox) ** 2 + (Y - oy) ** 2) - R).double()

    su, sv = disc(0.5 * H_2D, 0), disc(0, 0.5 * H_2D)

    def rnd():
        return torch.randn(NGX_2D, NGY_2D, dtype=torch.float64)

    flds = [su, sv, rnd(), rnd(), rnd(), rnd()]
    return [t.to(dev) for t in flds]


def _cbuf_2d(dev):
    base = DT_2D / RHO_2D
    return (torch.full((NGX_2D, NGY_2D), base, dtype=torch.float64, device=dev),
            torch.full((NGX_2D, NGY_2D), base, dtype=torch.float64, device=dev))


def _run_warp_2d(F, dirty, mu0_proj, sigma=None):
    su, sv, bu, bv, up, vp = F
    u0, v0 = up.clone(), vp.clone()
    ch, cv = _cbuf_2d(su.device)
    kw = {}
    if sigma is not None:
        ku, kv, ss = sigma
        kw = dict(key_u=ku, key_v=kv, sigma_shifts=ss)
    bdim_forcing_2d_warp(up, vp, su, sv, bu, bv, u0, v0, ch, cv,
                       EPS_2D, RHO_2D, DT_2D, H_2D, *dirty, mu0_proj, **kw)
    return u0, v0, ch, cv


def _sigma_2d(dev):
    dvol = NGX_2D * NGY_2D
    g = torch.Generator(device="cpu").manual_seed(3)
    ku = torch.randint(0, 2, (dvol,), dtype=torch.int64, generator=g)
    kv = torch.randint(0, 2, (dvol,), dtype=torch.int64, generator=g)
    ss = torch.tensor([0.05, 0.12], dtype=torch.float32)
    return ku.to(dev), kv.to(dev), ss.to(dev)


@SKIP_NO_CUDA
@pytest.mark.parametrize("dirty", [(0, 0, NGX_2D, NGY_2D), (5, 4, 24, 20)],
                         ids=["full", "subblock"])
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_cpu_eq_gpu_2d(dirty, mu0_proj):
    wc = _run_warp_2d(_fields_2d("cpu"), dirty, mu0_proj)
    wg = _run_warp_2d(_fields_2d("cuda:0"), dirty, mu0_proj)
    err = [(a.cpu() - b.cpu()).abs().max().item() for a, b in zip(wc, wg)]
    assert max(err) < 1e-12, f"warp cpu vs gpu: {err}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_cpu_eq_gpu_sigma_2d(mu0_proj):
    dirty = (0, 0, NGX_2D, NGY_2D)
    wc = _run_warp_2d(_fields_2d("cpu"), dirty, mu0_proj, _sigma_2d("cpu"))
    wg = _run_warp_2d(_fields_2d("cuda:0"), dirty, mu0_proj, _sigma_2d("cuda:0"))
    err = [(a.cpu() - b.cpu()).abs().max().item() for a, b in zip(wc, wg)]
    assert max(err) < 1e-12, f"warp cpu vs gpu (sigma): {err}"
