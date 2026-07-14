"""Native **bdim_forcing (3-D)** single-source checks: CPU twin == CUDA kernel.

Exercises ``bdim_forcing_3d`` on manufactured
sphere-SDF + random-field scenes: full-grid and interior dirty-AABB sub-blocks,
``mu0_projection`` 0/1.

Run:  pytest lilytorch/tests/test_bdim.py -v
"""
from __future__ import annotations

import pytest
import torch

from lilytorch.src.native import bdim_forcing_3d, bdim_forcing_2d

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


def _run_native(F, dirty, mu0_proj):
    su, sv, sw, bu, bv, bw, up, vp, wp_ = F
    u0, v0, w0 = up.clone(), vp.clone(), wp_.clone()
    ch, cv, cw = _cbuf(su.device, su.dtype)
    bdim_forcing_3d(up, vp, wp_, su, sv, sw, bu, bv, bw,
                       u0, v0, w0, ch, cv, cw, EPS, RHO, DT, H, *dirty,
                       mu0_proj)
    return u0, v0, w0, ch, cv, cw


# ─── single source: native CPU twin == CUDA kernel ────────────────────────────

@SKIP_NO_CUDA
@pytest.mark.parametrize("dirty", [(0, 0, 0, NGX, NGY, NGZ),
                                   (5, 4, 3, 20, 18, 16)],
                         ids=["full", "subblock"])
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_cpu_eq_gpu(dirty, mu0_proj):
    wc = _run_native(_fields("cpu"), dirty, mu0_proj)
    wg = _run_native(_fields("cuda:0"), dirty, mu0_proj)
    err = [(a.cpu() - b.cpu()).abs().max().item() for a, b in zip(wc, wg)]
    # float64 reduction noise on the normalised normal / mu1 polynomial.
    assert max(err) < 1e-12, f"cpu vs gpu: {err}"


# ─── 3-D full-grid rewrite semantics: pass-through outside the dirty AABB ────

@pytest.mark.parametrize("dev", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_outside_rect_passthrough_3d(dev):
    """Outside the dirty AABB the kernel must write u0/v0/w0 = the advected
    primes (the upfront full-grid copies it replaces) and leave the face-grid
    ch/cv/cw untouched."""
    su, sv, sw, bu, bv, bw, up, vp, wp_ = _fields(dev)
    u0 = torch.full_like(up, 777.0)     # garbage: NOT clones of the primes
    v0 = torch.full_like(vp, 777.0)
    w0 = torch.full_like(wp_, 777.0)
    ch, cv, cw = _cbuf(dev)
    refs = tuple(c.clone() for c in (ch, cv, cw))
    rect = (5, 4, 3, 20, 18, 16)
    bdim_forcing_3d(up, vp, wp_, su, sv, sw, bu, bv, bw,
                         u0, v0, w0, ch, cv, cw,
                         EPS, RHO, DT, H, *rect, 1)
    i0, j0, k0, Ai, Aj, Ak = rect
    out = torch.ones_like(up, dtype=torch.bool)
    out[i0:i0 + Ai, j0:j0 + Aj, k0:k0 + Ak] = False
    for got, prime, name in ((u0, up, "u0"), (v0, vp, "v0"), (w0, wp_, "w0")):
        assert torch.equal(got[out], prime[out]), f"{name} != prime outside AABB"
        assert not torch.equal(got[~out], prime[~out]), f"no BDIM write in {name}"
    # face-grid coeffs outside the AABB stay at the dt/rho prefill: compare
    # against a full-AABB reference restricted to the outside complement.
    for c, ref, gs in zip((ch, cv, cw), refs, (c.shape for c in refs)):
        outside = torch.ones(gs, dtype=torch.bool)
        outside[max(i0 - 1, 0):i0 + Ai, max(j0 - 1, 0):j0 + Aj,
                max(k0 - 1, 0):k0 + Ak] = False
        assert torch.equal(c[outside], ref[outside]), "coeff touched outside AABB"


# ─── 3-D Maertens–Weymouth fold: in-kernel div_corr == torch oracle ──────────

def _mw_oracle_3d(bU, bV, bW, sdf_cc, eps, h):
    div = torch.zeros_like(bU)
    div[1:-1, 1:-1, 1:-1] = (
        (bU[2:, 1:-1, 1:-1] - bU[1:-1, 1:-1, 1:-1]) * (1.0 / h)
        + (bV[1:-1, 2:, 1:-1] - bV[1:-1, 1:-1, 1:-1]) * (1.0 / h)
        + (bW[1:-1, 1:-1, 2:] - bW[1:-1, 1:-1, 1:-1]) * (1.0 / h))
    deps = (sdf_cc / eps).clamp(-1.0, 1.0)
    mu0 = 0.5 * (1.0 + deps + torch.sin(torch.pi * deps) / torch.pi)
    return (1.0 - mu0) * div


@pytest.mark.parametrize("dev", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_mw_div_corr_fold_3d(dev, dtype):
    su, sv, sw, bu, bv, bw, up, vp, wp_ = [t.to(dtype) for t in _fields(dev)]
    sdf_cc = ((su + sv + sw) / 3.0).contiguous()
    u0, v0, w0 = up.clone(), vp.clone(), wp_.clone()
    ch, cv, cw = _cbuf(dev, dtype)
    div_corr = torch.full_like(sdf_cc, -333.0)
    eps_mw = 1.7 * EPS
    bdim_forcing_3d(up, vp, wp_, su, sv, sw, bu, bv, bw,
                         u0, v0, w0, ch, cv, cw,
                         EPS, RHO, DT, H, 5, 4, 3, 20, 18, 16, 1,
                         sdf_cc=sdf_cc, div_corr=div_corr, eps_mw=eps_mw,
                         inv_dx=1.0 / H, inv_dy=1.0 / H, inv_dz=1.0 / H)
    ref = _mw_oracle_3d(bu, bv, bw, sdf_cc, eps_mw, H)
    # ulp-level torch-vs-kernel sin/FMA differences, relative to the field scale
    rtol = 1e-13 if dtype == torch.float64 else 1e-6
    err = (div_corr - ref).abs().max().item() / ref.abs().max().item()
    assert err < rtol, f"MW fold vs torch oracle (rel): {err:.3e}"
    assert (div_corr == -333.0).sum().item() == 0     # full-grid write


# ─── 3-D rect_dev: pre-allocated == per-call eager over a moving-AABB run ────

def _rectdev_steps_bdim_3d(dtype, mw_on):
    dev = "cuda:0"
    su, sv, sw, bu, bv, bw, up, vp, wp_ = [t.to(dtype) for t in _fields(dev)]
    sdf_cc = ((su + sv + sw) / 3.0).contiguous()
    u0g, v0g, w0g = (torch.empty_like(t) for t in (up, vp, wp_))
    chg, cvg, cwg = _cbuf(dev, dtype)
    dcg = torch.zeros_like(sdf_cc) if mw_on else None
    kw = dict(eps_mw=1.7 * EPS, inv_dx=1.0 / H, inv_dy=1.0 / H,
              inv_dz=1.0 / H) if mw_on else {}
    rect_dev = torch.empty(6, dtype=torch.int32, device=dev)

    for step in range(8):
        rect = (4 + step, 3 + step, 2 + step, 18, 16, 14)   # moving AABB
        up.add_(0.01); bu.mul_(1.001)                       # live-data check
        rect_dev.copy_(torch.tensor(rect, dtype=torch.int32, device=dev))
        bdim_forcing_3d(up, vp, wp_, su, sv, sw, bu, bv, bw,
                             u0g, v0g, w0g, chg, cvg, cwg,
                             EPS, RHO, DT, H, *rect, 1,
                             sdf_cc=(sdf_cc if mw_on else None),
                             div_corr=dcg, rect_dev=rect_dev, **kw)
        u0e, v0e, w0e = (torch.empty_like(t) for t in (up, vp, wp_))
        che, cve, cwe = chg.clone(), cvg.clone(), cwg.clone()
        dce = torch.zeros_like(sdf_cc) if mw_on else None
        bdim_forcing_3d(up, vp, wp_, su, sv, sw, bu, bv, bw,
                             u0e, v0e, w0e, che, cve, cwe,
                             EPS, RHO, DT, H, *rect, 1,
                             sdf_cc=(sdf_cc if mw_on else None),
                             div_corr=dce, **kw)
        outs = [(u0g, u0e), (v0g, v0e), (w0g, w0e),
                (chg, che), (cvg, cve), (cwg, cwe)]
        if mw_on:
            outs.append((dcg, dce))
        for a, b in outs:
            err = (a - b).abs().max().item()
            assert err == 0.0, f"step {step}: rect_dev vs eager err {err:.3e}"
    return rect_dev


@SKIP_NO_CUDA
@pytest.mark.parametrize("mw_on", [False, True], ids=["plain", "mw"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_bdim_3d_rectdev_eq_eager(dtype, mw_on):
    """Pre-allocated rect_dev produces bit-identical results to eager path."""
    _rectdev_steps_bdim_3d(dtype, mw_on)


# ─── solver-level MW wiring: fused step's div_corr == the method oracle ──────

@pytest.mark.parametrize("ndim", [2, 3])
def test_solver_mw_fold_wiring(ndim):
    """Run the fused single-phase step with ``bdim_body_div_correction`` on
    and check the kernel-written ``_mw_div_corr_persist`` matches the retained
    torch oracle ``_mw_body_div_correction`` on the live solver state (i.e.
    the solver passes the right sdf_cc/eps/inv_d{x,y,z} into the kernel)."""
    from lilytorch.tests.test_two_phase import (
        _parity_pars, _taylor_green_ic, _set_ic, _step_n)
    from lilytorch.src.solver import FluidSolver

    N, dt, nu, rho = (20 if ndim == 3 else 32), 2.0e-3, 1.0e-2, 1000.0
    body = (["lambda x, y, z: sphere(x,y,z,xt=0.5,yt=0.5,zt=0.5,r=0.15)"]
            if ndim == 3 else
            ["lambda x, y: circle(x,y,xt=0.5,yt=0.5,r=0.15)"])
    pars = _parity_pars(ndim, N, dt, nu, rho, body)
    pars["solver"]["bdim_body_div_correction"] = True

    s = FluidSolver(pars, dtype=torch.float64, compute_forces=False)
    _set_ic(s, _taylor_green_ic(s))
    _step_n(s, 4)

    comp = s.composite_body
    if ndim == 3:
        ref = s._mw_body_div_correction(comp.body_u, comp.body_v, comp.body_w)
    else:
        ref = s._mw_body_div_correction(comp.body_u, comp.body_v)
    err = (s._mw_div_corr_persist - ref).abs().max().item()
    assert err < 1e-13, f"solver MW wiring vs oracle: {err:.3e}"
    for f in (s.u0, s.v0, s.p0) + ((s.w0,) if ndim == 3 else ()):
        assert torch.isfinite(f).all()


if __name__ == "__main__":
    if torch.cuda.is_available():
        wc = _run_native(_fields("cpu"), (0, 0, 0, NGX, NGY, NGZ), 1)
        wg = _run_native(_fields("cuda:0"), (0, 0, 0, NGX, NGY, NGZ), 1)
        e = [(a.cpu() - b.cpu()).abs().max().item() for a, b in zip(wc, wg)]
        print(f"  CPU vs GPU full  maxerr={max(e):.2e}")
    print("  (run via pytest for the full matrix)")


# ═════════════════════════════════════════════════════════════════════════════
#  2-D bdim_forcing (bdim_forcing_2d) — merged from the former
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


def _run_native_2d(F, dirty, mu0_proj):
    su, sv, bu, bv, up, vp = F
    u0, v0 = up.clone(), vp.clone()
    ch, cv = _cbuf_2d(su.device)
    bdim_forcing_2d(up, vp, su, sv, bu, bv, u0, v0, ch, cv,
                       EPS_2D, RHO_2D, DT_2D, H_2D, *dirty, mu0_proj)
    return u0, v0, ch, cv


@SKIP_NO_CUDA
@pytest.mark.parametrize("dirty", [(0, 0, NGX_2D, NGY_2D), (5, 4, 24, 20)],
                         ids=["full", "subblock"])
@pytest.mark.parametrize("mu0_proj", [1, 0])
def test_cpu_eq_gpu_2d(dirty, mu0_proj):
    wc = _run_native_2d(_fields_2d("cpu"), dirty, mu0_proj)
    wg = _run_native_2d(_fields_2d("cuda:0"), dirty, mu0_proj)
    err = [(a.cpu() - b.cpu()).abs().max().item() for a, b in zip(wc, wg)]
    assert max(err) < 1e-12, f"cpu vs gpu: {err}"


# ─── full-grid rewrite semantics: pass-through outside the dirty rect ────────

@pytest.mark.parametrize("dev", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_outside_rect_passthrough_2d(dev):
    """Outside the dirty AABB the kernel must write u0 = u_prime / v0 = v_prime
    (the upfront full-grid copy it replaces) and leave ch/cv untouched."""
    su, sv, bu, bv, up, vp = _fields_2d(dev)
    u0 = torch.full_like(up, 777.0)     # garbage: NOT a clone of up
    v0 = torch.full_like(vp, 777.0)
    ch, cv = _cbuf_2d(dev)
    ch_ref, cv_ref = ch.clone(), cv.clone()
    rect = (5, 4, 24, 20)
    bdim_forcing_2d(up, vp, su, sv, bu, bv, u0, v0, ch, cv,
                         EPS_2D, RHO_2D, DT_2D, H_2D, *rect, 1)
    i0, j0, Ai, Aj = rect
    out = torch.ones_like(up, dtype=torch.bool)
    out[i0:i0 + Ai, j0:j0 + Aj] = False
    assert torch.equal(u0[out], up[out]), "u0 != u_prime outside rect"
    assert torch.equal(v0[out], vp[out]), "v0 != v_prime outside rect"
    assert torch.equal(ch[out], ch_ref[out]), "ch touched outside rect"
    assert torch.equal(cv[out], cv_ref[out]), "cv touched outside rect"
    inside = ~out
    assert not torch.equal(u0[inside], up[inside]), "no BDIM write inside rect"


# ─── Maertens–Weymouth fold: in-kernel div_corr == torch oracle ──────────────

def _mw_oracle(bU, bV, sdf_cc, eps, dx, dy):
    """Verbatim FluidSolver._mw_body_div_correction + ops.divergence."""
    div = torch.zeros_like(bU)
    div[1:-1, 1:-1] = ((bU[2:, 1:-1] - bU[1:-1, 1:-1]) * (1.0 / dx)
                       + (bV[1:-1, 2:] - bV[1:-1, 1:-1]) * (1.0 / dy))
    deps = (sdf_cc / eps).clamp(-1.0, 1.0)
    mu0 = 0.5 * (1.0 + deps + torch.sin(torch.pi * deps) / torch.pi)
    return (1.0 - mu0) * div


@pytest.mark.parametrize("dev", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_mw_div_corr_fold_2d(dev, dtype):
    su, sv, bu, bv, up, vp = [t.to(dtype) for t in _fields_2d(dev)]
    sdf_cc = 0.5 * (su + sv)            # any smooth CC SDF-like field
    u0, v0 = up.clone(), vp.clone()
    ch, cv = (c.to(dtype) for c in _cbuf_2d(dev))
    div_corr = torch.full_like(sdf_cc, -333.0)
    eps_mw = 1.7 * EPS_2D
    bdim_forcing_2d(up, vp, su, sv, bu, bv, u0, v0, ch, cv,
                         EPS_2D, RHO_2D, DT_2D, H_2D,
                         5, 4, 24, 20, 1,
                         sdf_cc=sdf_cc, div_corr=div_corr,
                         eps_mw=eps_mw, inv_dx=1.0 / H_2D, inv_dy=1.0 / H_2D)
    ref = _mw_oracle(bu, bv, sdf_cc, eps_mw, H_2D, H_2D)
    # ulp-level torch-vs-kernel sin/FMA differences, relative to the field scale
    rtol = 1e-13 if dtype == torch.float64 else 1e-6
    err = (div_corr - ref).abs().max().item() / ref.abs().max().item()
    assert err < rtol, f"MW fold vs torch oracle (rel): {err:.3e}"
    # full-grid write: no stale sentinel survives anywhere
    assert (div_corr == -333.0).sum().item() == 0


# ─── rect_dev: pre-allocated == per-call eager over a moving-AABB run ────────

def _rectdev_steps_bdim_2d(dtype, mw_on):
    """8 steps with drifting fields and a MOVING dirty rect through one
    pre-allocated rect_dev, checked per-step against a fresh eager launch
    on cloned outputs."""
    dev = "cuda:0"
    su, sv, bu, bv, up, vp = [t.to(dtype) for t in _fields_2d(dev)]
    sdf_cc = (0.5 * (su + sv)).contiguous()
    ch_base = DT_2D / RHO_2D
    # persistent buffers (pointer-stable, as in the solver)
    u0g = torch.empty_like(up); v0g = torch.empty_like(vp)
    chg = torch.full_like(up, ch_base); cvg = torch.full_like(vp, ch_base)
    dcg = torch.zeros_like(sdf_cc) if mw_on else None
    kw = dict(eps_mw=1.7 * EPS_2D, inv_dx=1.0 / H_2D, inv_dy=1.0 / H_2D) \
        if mw_on else {}
    rect_dev = torch.empty(4, dtype=torch.int32, device=dev)

    for step in range(8):
        rect = (4 + step, 3 + step, 22, 18)      # moving dirty rect
        up.add_(0.01); bu.mul_(1.001)            # live-data check
        rect_dev.copy_(torch.tensor(rect, dtype=torch.int32, device=dev))
        bdim_forcing_2d(up, vp, su, sv, bu, bv, u0g, v0g, chg, cvg,
                             EPS_2D, RHO_2D, DT_2D, H_2D, *rect, 1,
                             sdf_cc=(sdf_cc if mw_on else None),
                             div_corr=dcg, rect_dev=rect_dev, **kw)
        # eager reference on fresh clones (chg/cvg state must match: clone)
        u0e = torch.empty_like(up); v0e = torch.empty_like(vp)
        che, cve = chg.clone(), cvg.clone()
        dce = torch.zeros_like(sdf_cc) if mw_on else None
        bdim_forcing_2d(up, vp, su, sv, bu, bv, u0e, v0e, che, cve,
                             EPS_2D, RHO_2D, DT_2D, H_2D, *rect, 1,
                             sdf_cc=(sdf_cc if mw_on else None),
                             div_corr=dce, **kw)
        outs = [(u0g, u0e), (v0g, v0e), (chg, che), (cvg, cve)]
        if mw_on:
            outs.append((dcg, dce))
        for a, b in outs:
            err = (a - b).abs().max().item()
            assert err == 0.0, f"step {step}: rect_dev vs eager err {err:.3e}"
    return rect_dev


@SKIP_NO_CUDA
@pytest.mark.parametrize("mw_on", [False, True], ids=["plain", "mw"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_bdim_2d_rectdev_eq_eager(dtype, mw_on):
    """Pre-allocated rect_dev produces bit-identical results to eager path."""
    _rectdev_steps_bdim_2d(dtype, mw_on)
