"""Warp **bdim_forcing (2-D)** single-source checks: Warp CPU == Warp GPU.

2-D analogue of `test_bdim.py`.  Covers full-grid + dirty-subblock launches,
both `mu0_proj` settings, and the σ (body-id key) variant.
"""
from __future__ import annotations

import pytest
import torch

from lilytorch.src.kernels.bdim_2d import bdim_forcing_2d_warp

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
    bdim_forcing_2d_warp(up, vp, su, sv, bu, bv, u0, v0, ch, cv,
                       EPS, RHO, DT, H, *dirty, mu0_proj, **kw)
    return u0, v0, ch, cv


def _sigma(dev):
    dvol = NGX * NGY
    g = torch.Generator(device="cpu").manual_seed(3)
    ku = torch.randint(0, 2, (dvol,), dtype=torch.int64, generator=g)
    kv = torch.randint(0, 2, (dvol,), dtype=torch.int64, generator=g)
    ss = torch.tensor([0.05, 0.12], dtype=torch.float32)
    return ku.to(dev), kv.to(dev), ss.to(dev)


@SKIP_NO_CUDA
@pytest.mark.parametrize("dirty", [(0, 0, NGX, NGY), (5, 4, 24, 20)],
                         ids=["full", "subblock"])
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_cpu_eq_gpu(dirty, mu0_proj):
    wc = _run_warp(_fields("cpu"), dirty, mu0_proj)
    wg = _run_warp(_fields("cuda:0"), dirty, mu0_proj)
    err = [(a.cpu() - b.cpu()).abs().max().item() for a, b in zip(wc, wg)]
    assert max(err) < 1e-12, f"warp cpu vs gpu: {err}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_cpu_eq_gpu_sigma(mu0_proj):
    dirty = (0, 0, NGX, NGY)
    wc = _run_warp(_fields("cpu"), dirty, mu0_proj, _sigma("cpu"))
    wg = _run_warp(_fields("cuda:0"), dirty, mu0_proj, _sigma("cuda:0"))
    err = [(a.cpu() - b.cpu()).abs().max().item() for a, b in zip(wc, wg)]
    assert max(err) < 1e-12, f"warp cpu vs gpu (sigma): {err}"
