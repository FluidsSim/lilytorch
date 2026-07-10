"""Parity gates for cuda_native_port **item 8** — moving the pre-Poisson step
region (BCs / bdim / streaming-SDF / advection) from Warp to native kernels.

These tests are the DeepSeek work order's acceptance criteria (see the "Item 8
spec" in ``milestones/cuda_native_port_plan.md``).  Each native op is validated
against its Warp oracle (the Warp modules stay importable until the whole region
is native and the 0.4 gate is green).

Life-cycle: the ops that DeepSeek still has to write (`native.bdim_forcing_*`,
`native.sl_advect_*`, `native.diffuse_add`) do not exist yet, so their tests
**SKIP** today.  A sub-item of item 8 is DONE only when its test flips from
SKIP → PASS with no other regression.  ``native.apply_bcs_*`` already exists
(Phase 0.2), so 8.A's gate runs now.

All gates are CUDA-only: the whole-step region graph is GPU-only (CPU runs the
same ops eagerly and is covered by the existing suite).
"""
from __future__ import annotations

import pytest
import torch
import warp as wp

from lilytorch.src import native, diffusion
from lilytorch.src.bdim import bdim_forcing_2d_warp, bdim_forcing_3d_warp
from lilytorch.src.advection import (
    AdvDiffSolver,
    sl_advect_2d_warp, sl_advect_3d_warp,
    apply_bcs_2d_warp, apply_bcs_3d_warp,
)

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
DEV = "cuda:0"

# dtype-aware parity tolerances (mirror the forces-post / interpolation gates).
_TOL = {torch.float64: 1e-9, torch.float32: 2e-7}
_SL_TOL = {torch.float64: 1e-9, torch.float32: 5e-6}  # SL back-trace is interpolation-heavy


def _need(op_name: str):
    """Skip until the native wrapper `native.<op_name>` exists (DeepSeek adds it)."""
    if not hasattr(native, op_name):
        pytest.skip(f"item 8: native.{op_name} not implemented yet")


def _maxdiff(a, b):
    return (a - b).abs().max().item()


def _live_bc_mask(shape):
    """True for cells a 5/7-point stencil can read: interior (0 axes on a
    boundary) + face ghosts (exactly 1 axis on a boundary).  Corner/edge cells
    (>=2 axes on a boundary) are never read, and Neumann corner tie-breaks
    differ harmlessly between the native and Warp apply_bcs — exclude them so
    the swap gate does not chase a dead-cell non-bug (see item-8 spec 8.A)."""
    on_bnd = None
    for ax, n in enumerate(shape):
        idx = torch.arange(n)
        b = (idx == 0) | (idx == n - 1)
        b = b.reshape([n if d == ax else 1 for d in range(len(shape))])
        on_bnd = b.to(torch.int64) if on_bnd is None else on_bnd + b.to(torch.int64)
    return (on_bnd <= 1)


# =====================================================================
# 8.C — bdim_forcing_{2,3}d  (NEW native static-full-grid kernel)
# native.bdim_forcing_{2,3}d MUST mirror bdim.bdim_forcing_{2,3}d_warp's
# python signature exactly, so the solver swap is a rename and this test
# calls both identically.
# =====================================================================

def _bdim_inputs_2d(dtype, seed):
    torch.manual_seed(seed)
    Ngx, Ngy = 22, 18
    h = 0.05
    g = lambda: torch.randn(Ngx, Ngy, dtype=dtype, device=DEV).contiguous()
    sdf = lambda: (torch.randn(Ngx, Ngy, dtype=dtype, device=DEV) * (3 * h)).contiguous()
    d = dict(
        u_prime=g(), v_prime=g(), sdf_u=sdf(), sdf_v=sdf(),
        body_u=g(), body_v=g(),
        ch_shape=(Ngx, Ngy), cv_shape=(Ngx, Ngy),
        Ng=(Ngx, Ngy), h=h,
    )
    return d


def _bdim_inputs_3d(dtype, seed):
    torch.manual_seed(seed)
    Ngx, Ngy, Ngz = 14, 12, 10
    h = 0.05
    g = lambda: torch.randn(Ngx, Ngy, Ngz, dtype=dtype, device=DEV).contiguous()
    sdf = lambda: (torch.randn(Ngx, Ngy, Ngz, dtype=dtype, device=DEV) * (3 * h)).contiguous()
    d = dict(
        u_prime=g(), v_prime=g(), w_prime=g(),
        sdf_u=sdf(), sdf_v=sdf(), sdf_w=sdf(),
        body_u=g(), body_v=g(), body_w=g(),
        ch_shape=(Ngx - 1, Ngy - 2, Ngz - 2),
        cv_shape=(Ngx - 2, Ngy - 1, Ngz - 2),
        cw_shape=(Ngx - 2, Ngy - 2, Ngz - 1),
        Ng=(Ngx, Ngy, Ngz), h=h,
    )
    return d


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("mw", [False, True])
@pytest.mark.parametrize("full_grid", [False, True])
def test_bdim_forcing_2d_native_eq_warp(dtype, mw, full_grid):
    _need("bdim_forcing_2d")
    inp = _bdim_inputs_2d(dtype, seed=7)
    Ngx, Ngy = inp["Ng"]
    h, rho, dt, eps = inp["h"], 1.0, 0.01, 2.0 * inp["h"]
    rect = (0, 0, Ngx, Ngy) if full_grid else (3, 2, Ngx - 8, Ngy - 6)

    # Pre-generate MW tensors ONCE so Warp and native see identical inputs.
    sdf_cc_mw = div_mw = None
    if mw:
        sdf_cc_mw = (torch.randn(Ngx, Ngy, dtype=dtype, device=DEV) * (3 * h)).contiguous()
        div_mw = torch.zeros(Ngx, Ngy, dtype=dtype, device=DEV)

    def run(fn):
        u0 = inp["u_prime"].clone(); v0 = inp["v_prime"].clone()
        ch = torch.full(inp["ch_shape"], dt / rho, dtype=dtype, device=DEV)
        cv = torch.full(inp["cv_shape"], dt / rho, dtype=dtype, device=DEV)
        fn(inp["u_prime"], inp["v_prime"], inp["sdf_u"], inp["sdf_v"],
           inp["body_u"], inp["body_v"], u0, v0, ch, cv,
           eps, rho, dt, h, *rect, 1,
           sdf_cc_mw, div_mw, 1.0 / h if mw else 1.0, 1.0 / h, 1.0 / h)
        return u0, v0, ch, cv, div_mw

    # NB: sdf_cc/div/eps_mw/inv_d* are keyword-tail args on the Warp wrapper;
    # pass positionally in the mirrored native wrapper (DeepSeek: keep the order).
    # Here we call through a thin shim to keep both calls identical.
    def shim_warp(u_prime, v_prime, sdf_u, sdf_v, body_u, body_v, u0, v0, ch, cv,
                  eps, rho, dt, h, i0, j0, Ai, Aj, mu0p, sdf_cc, div, eps_mw, idx, idy):
        bdim_forcing_2d_warp(u_prime, v_prime, sdf_u, sdf_v, body_u, body_v,
                             u0, v0, ch, cv, eps, rho, dt, h, i0, j0, Ai, Aj,
                             mu0_projection=mu0p, sdf_cc=sdf_cc, div_corr=div,
                             eps_mw=eps_mw, inv_dx=idx, inv_dy=idy)

    def shim_native(u_prime, v_prime, sdf_u, sdf_v, body_u, body_v, u0, v0, ch, cv,
                    eps, rho, dt, h, i0, j0, Ai, Aj, mu0p, sdf_cc, div, eps_mw, idx, idy):
        native.bdim_forcing_2d(u_prime, v_prime, sdf_u, sdf_v, body_u, body_v,
                               u0, v0, ch, cv, eps, rho, dt, h, i0, j0, Ai, Aj,
                               mu0_projection=mu0p, sdf_cc=sdf_cc, div_corr=div,
                               eps_mw=eps_mw, inv_dx=idx, inv_dy=idy)

    uw, vw, chw, cvw, dw = run(shim_warp)
    un, vn, chn, cvn, dn = run(shim_native)
    wp.synchronize()
    tol = _TOL[dtype]
    assert _maxdiff(uw, un) <= tol, f"u0 {_maxdiff(uw, un):.2e}"
    assert _maxdiff(vw, vn) <= tol, f"v0 {_maxdiff(vw, vn):.2e}"
    assert _maxdiff(chw, chn) <= tol, f"ch {_maxdiff(chw, chn):.2e}"
    assert _maxdiff(cvw, cvn) <= tol, f"cv {_maxdiff(cvw, cvn):.2e}"
    if mw:
        assert _maxdiff(dw, dn) <= tol, f"div_corr {_maxdiff(dw, dn):.2e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("mw", [False, True])
@pytest.mark.parametrize("full_grid", [False, True])
def test_bdim_forcing_3d_native_eq_warp(dtype, mw, full_grid):
    _need("bdim_forcing_3d")
    inp = _bdim_inputs_3d(dtype, seed=11)
    Ngx, Ngy, Ngz = inp["Ng"]
    h, rho, dt, eps = inp["h"], 1.0, 0.01, 2.0 * inp["h"]
    rect = ((0, 0, 0, Ngx, Ngy, Ngz) if full_grid
            else (2, 2, 1, Ngx - 6, Ngy - 5, Ngz - 4))

    # Pre-generate MW tensors ONCE so Warp and native see identical inputs.
    sdf_cc_mw = div_mw = None
    if mw:
        sdf_cc_mw = (torch.randn(Ngx, Ngy, Ngz, dtype=dtype, device=DEV) * (3 * h)).contiguous()
        div_mw = torch.zeros(Ngx, Ngy, Ngz, dtype=dtype, device=DEV)

    def run(fn):
        u0 = inp["u_prime"].clone(); v0 = inp["v_prime"].clone(); w0 = inp["w_prime"].clone()
        ch = torch.full(inp["ch_shape"], dt / rho, dtype=dtype, device=DEV)
        cv = torch.full(inp["cv_shape"], dt / rho, dtype=dtype, device=DEV)
        cw = torch.full(inp["cw_shape"], dt / rho, dtype=dtype, device=DEV)
        fn(inp["u_prime"], inp["v_prime"], inp["w_prime"],
           inp["sdf_u"], inp["sdf_v"], inp["sdf_w"],
           inp["body_u"], inp["body_v"], inp["body_w"],
           u0, v0, w0, ch, cv, cw, eps, rho, dt, h, *rect, 1,
           sdf_cc_mw, div_mw, 1.0 / h if mw else 1.0, 1.0 / h, 1.0 / h, 1.0 / h)
        return u0, v0, w0, ch, cv, cw, div_mw

    def shim(op):
        def _call(u_p, v_p, w_p, su, sv, sw, bu, bv, bw, u0, v0, w0, ch, cv, cw,
                  eps, rho, dt, h, i0, j0, k0, Ai, Aj, Ak, mu0p,
                  sdf_cc, div, eps_mw, idx, idy, idz):
            op(u_p, v_p, w_p, su, sv, sw, bu, bv, bw, u0, v0, w0, ch, cv, cw,
               eps, rho, dt, h, i0, j0, k0, Ai, Aj, Ak,
               mu0_projection=mu0p, sdf_cc=sdf_cc, div_corr=div,
               eps_mw=eps_mw, inv_dx=idx, inv_dy=idy, inv_dz=idz)
        return _call

    rw = run(shim(bdim_forcing_3d_warp))
    rn = run(shim(native.bdim_forcing_3d))
    wp.synchronize()
    tol = _TOL[dtype]
    names = ["u0", "v0", "w0", "ch", "cv", "cw"]
    for nm, a, b in zip(names, rw[:6], rn[:6]):
        assert _maxdiff(a, b) <= tol, f"{nm} {_maxdiff(a, b):.2e}"
    if mw:
        assert _maxdiff(rw[6], rn[6]) <= tol, f"div_corr {_maxdiff(rw[6], rn[6]):.2e}"


# =====================================================================
# 8.D — sl_advect_{2,3}d + diffuse_add  (NEW native kernels)
# Axes/interpolator metadata are sourced from a real AdvDiffSolver so the
# native op is called with exactly the same args as the Warp oracle.
# =====================================================================

def _sl_solver(ndim, dtype, N=24):
    x = torch.linspace(0.0, 1.0, N, dtype=dtype, device=DEV)
    kw = dict(device=torch.device(DEV), dt=1e-3, x=x, y=x.clone(), nu=1e-3,
              method="semi-lagrangian")
    if ndim == 3:
        kw["z"] = x.clone()
    return AdvDiffSolver(**kw)


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sl_advect_2d_native_eq_warp(dtype):
    _need("sl_advect_2d")
    torch.manual_seed(3)
    s = _sl_solver(2, dtype)
    N = s.n[0]
    u = torch.randn(N, N, dtype=dtype, device=DEV).contiguous()
    v = torch.randn(N, N, dtype=dtype, device=DEV).contiguous()
    s.solve(u.clone(), v.clone())          # populates _sl_axes_dev / _interps
    gxu, gyu, gxv, gyv = s._sl_axes_dev
    iu, iv = s._interps
    meta = (iu._bx0, iu._by0, iu._inv_dx, iu._inv_dy,
            iv._bx0, iv._by0, iv._inv_dx, iv._inv_dy, s.dt)

    ow = (torch.empty_like(u), torch.empty_like(v))
    on = (torch.empty_like(u), torch.empty_like(v))
    sl_advect_2d_warp(u, v, ow[0], ow[1], gxu, gyu, gxv, gyv, *meta)
    native.sl_advect_2d(u, v, on[0], on[1], gxu, gyu, gxv, gyv, *meta)
    wp.synchronize()
    tol = _SL_TOL[dtype]
    assert _maxdiff(ow[0], on[0]) <= tol
    assert _maxdiff(ow[1], on[1]) <= tol


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sl_advect_3d_native_eq_warp(dtype):
    _need("sl_advect_3d")
    torch.manual_seed(4)
    s = _sl_solver(3, dtype, N=18)
    N = s.n[0]
    u = torch.randn(N, N, N, dtype=dtype, device=DEV).contiguous()
    v = torch.randn(N, N, N, dtype=dtype, device=DEV).contiguous()
    w = torch.randn(N, N, N, dtype=dtype, device=DEV).contiguous()
    s.solve(u.clone(), v.clone(), w.clone())
    axes = s._sl_axes_dev
    iu, iv, iw = s._interps
    meta = (iu._bx0, iu._by0, iu._bz0, iu._inv_dx, iu._inv_dy, iu._inv_dz,
            iv._bx0, iv._by0, iv._bz0, iv._inv_dx, iv._inv_dy, iv._inv_dz,
            iw._bx0, iw._by0, iw._bz0, iw._inv_dx, iw._inv_dy, iw._inv_dz, s.dt)
    ow = tuple(torch.empty_like(u) for _ in range(3))
    on = tuple(torch.empty_like(u) for _ in range(3))
    sl_advect_3d_warp(u, v, w, *ow, *axes, *meta)
    native.sl_advect_3d(u, v, w, *on, *axes, *meta)
    wp.synchronize()
    tol = _SL_TOL[dtype]
    for a, b in zip(ow, on):
        assert _maxdiff(a, b) <= tol


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("variable", [False, True])
def test_diffuse_add_native_eq_warp(dtype, ndim, variable):
    _need("diffuse_add")
    torch.manual_seed(5)
    N = 20
    shape = (N,) * ndim
    dh = (0.05,) * ndim
    dt, nu = 1e-3, 2e-3
    base = torch.randn(shape, dtype=dtype, device=DEV).contiguous()
    nu_eff = None
    if variable:
        nu_eff = (nu + 0.5 * nu * torch.rand(shape, dtype=dtype, device=DEV)).contiguous()

    tw = base.clone(); cw = torch.empty_like(tw)
    tn = base.clone(); cn = torch.empty_like(tn)
    diffusion.diffuse_add_(tw, cw, dt, dh=dh, nu_eff=nu_eff, nu=nu)
    native.diffuse_add(tn, cn, dt, dh=dh, nu_eff=nu_eff, nu=nu)
    wp.synchronize()
    assert _maxdiff(tw, tn) <= _TOL[dtype]


# =====================================================================
# 8.A — apply_bcs_{2,3}d  (swap only; native op already exists)
# Validates the native BC op through the REAL descriptor cache built by
# AdvDiffSolver, exactly as set_BCs will call it after the swap.
# =====================================================================

@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_apply_bcs_2d_native_eq_warp(dtype):
    torch.manual_seed(6)
    N = 20
    x = torch.linspace(0.0, 1.0, N, dtype=dtype, device=DEV)
    s = AdvDiffSolver(device=torch.device(DEV), dt=1e-3, x=x, y=x.clone(), nu=1e-3,
                      BC_type_u=("N", "D", "D", "N"), BC_values_u=(0, 0.5, -0.3, 0),
                      BC_type_v=("D", "N", "N", "D"), BC_values_v=(0.2, 0, 0, -0.1),
                      method="semi-lagrangian")
    base_u = torch.randn(N, N, dtype=dtype, device=DEV).contiguous()
    base_v = torch.randn(N, N, dtype=dtype, device=DEV).contiguous()
    cache = s._build_fused_bc_cache_2d((base_u, base_v))

    uw, vw = base_u.clone(), base_v.clone()
    apply_bcs_2d_warp(uw, vw, cache["shapes"], cache["neu_desc"], cache["dir_desc"],
                      cache["dir_val"], cache["ref_desc"], cache["ref_val"],
                      cache["max_line_dim"])
    un, vn = base_u.clone(), base_v.clone()
    native.apply_bcs_2d(un, vn, cache["shapes"], cache["neu_desc"], cache["dir_desc"],
                        cache["dir_val"], cache["ref_desc"], cache["ref_val"],
                        cache["max_line_dim"])
    wp.synchronize()
    m = _live_bc_mask(uw.shape).to(DEV)
    assert _maxdiff(uw[m], un[m]) == 0.0
    assert _maxdiff(vw[m], vn[m]) == 0.0


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_apply_bcs_3d_native_eq_warp(dtype):
    torch.manual_seed(8)
    N = 16
    x = torch.linspace(0.0, 1.0, N, dtype=dtype, device=DEV)
    s = AdvDiffSolver(device=torch.device(DEV), dt=1e-3, x=x, y=x.clone(), z=x.clone(),
                      nu=1e-3,
                      BC_type_u=("N", "D", "D", "N"), BC_values_u=(0, 0.5, -0.3, 0),
                      BC_type_v=("D", "N", "N", "D"), BC_values_v=(0.2, 0, 0, -0.1),
                      BC_type_w=("N", "N", "D", "D", "N", "D"),
                      BC_values_w=(0, 0, 0.1, -0.2, 0, 0.3),
                      method="semi-lagrangian")
    base = [torch.randn(N, N, N, dtype=dtype, device=DEV).contiguous() for _ in range(3)]
    cache = s._build_fused_bc_cache(tuple(base))

    warp_t = [b.clone() for b in base]
    apply_bcs_3d_warp(*warp_t, cache["shapes"], cache["neu_desc"], cache["dir_desc"],
                      cache["dir_val"], cache["ref_desc"], cache["ref_val"],
                      cache["max_dim0"], cache["max_dim1"])
    nat_t = [b.clone() for b in base]
    native.apply_bcs_3d(*nat_t, cache["shapes"], cache["neu_desc"], cache["dir_desc"],
                        cache["dir_val"], cache["ref_desc"], cache["ref_val"],
                        cache["max_dim0"], cache["max_dim1"])
    wp.synchronize()
    m = _live_bc_mask(warp_t[0].shape).to(DEV)
    for a, b in zip(warp_t, nat_t):
        assert _maxdiff(a[m], b[m]) == 0.0


# =====================================================================
# 8.B — streaming-SDF native path.
# Unit parity for streaming_sdf_stag_{2,3}d_multi requires a full body-table
# oracle (body_shapes/body_meta/kin/aabb marshalling).  Rather than duplicate
# that marshalling here, 8.B's gate is (a) the existing coupled/streaming
# suite (which exercises facade.body_update end-to-end and will break if the
# native swap diverges), plus (b) the solver-driven bit-exact step gate added
# to tests/test_whole_step_capture_native.py in 8.E.  DeepSeek: if a standalone
# unit oracle is wanted, build it from a single BodyAnalytical via BDIMhandler's
# body-table packing and compare native.streaming_sdf_stag_2d_multi against the
# facade.body_update_2d Warp bridge (single body, two separated bodies, salamander
# multi-link; ≤1e-9 f64 / 1e-6 f32).
