"""§F end-to-end structure tests: ``src_cuda`` vs ``src_warp`` backend trees.

Validates the parallel backend layout requested for the Warp swap-in:

* the two ``kernel/`` facades expose the *same* op contract;
* ``src_cuda`` is the native path (``FluidSolver`` identity);
* ``src_warp`` wires the signature-clean drop-in ops (``advect_flux_add``,
  ``cvof_sweep``) to single-source Warp kernels and is otherwise native;
* the Warp-wired ops produce **bit-identical** results to native when reached
  through the live solver sub-classes (``AdvDiffSolver._solve_convective`` and
  ``TwoPhase._cvof_sweep``);
* the **CPU end-to-end payoff**: the Warp ``cvof_sweep`` runs on CPU from the
  same kernel source and matches the native python reference.

Native kernel ops are CUDA-only, so the GPU-parity tests skip without CUDA; the
CPU single-source test always runs.
"""
import numpy as np
import pytest
import torch

from lilytorch.src_cuda import kernel as kcuda
from lilytorch.src_warp import kernel as kwarp
from lilytorch.src_cuda import solver as scuda
from lilytorch.src_warp import solver as swarp
from lilytorch.src import solver as sbase

import lilytorch.src_cuda.advection as adv_cuda
import lilytorch.src_warp.advection as adv_warp
import lilytorch.src_cuda.two_phase as tp_cuda
import lilytorch.src_warp.two_phase as tp_warp

CUDA = torch.cuda.is_available()
_OP_NAMES = [
    "streaming_sdf_stag_2d_multi", "streaming_sdf_stag_3d_multi",
    "bdim_coeff_2d", "bdim_coeff_3d", "bdim_coeff_sigma_2d", "bdim_coeff_sigma_3d",
    "advect_flux_add", "apply_bcs_2d", "apply_bcs_3d", "interp_2d", "interp_3d",
    "cvof_sweep", "lagrangian_forces_2d", "lagrangian_forces_3d",
    "streaming_sdf_forces_post_2d", "streaming_sdf_forces_post_3d", "K",
]


# ─────────────────────────── structure / contract ──────────────────────────
def test_kernel_facades_expose_same_contract():
    for name in _OP_NAMES:
        assert hasattr(kcuda, name), f"src_cuda.kernel missing {name}"
        assert hasattr(kwarp, name), f"src_warp.kernel missing {name}"
    assert kcuda.BACKEND == "cuda"
    assert kwarp.BACKEND == "warp"


def test_warp_backed_set():
    # CUDA backend runs nothing on Warp; Warp backend runs the clean drop-ins
    # plus the 2-D Kernel A/B marshalling bridge.
    assert kcuda.WARP_BACKED == frozenset()
    assert kwarp.WARP_BACKED == frozenset({
        "advect_flux_add", "cvof_sweep",
        "streaming_sdf_stag_2d_multi", "bdim_coeff_2d",
    })


def test_solver_identities():
    # src_cuda is the native solver verbatim; src_warp subclasses it.
    assert scuda.FluidSolver is sbase.FluidSolver
    assert issubclass(swarp.FluidSolver, sbase.FluidSolver)
    assert swarp.FluidSolver is not sbase.FluidSolver
    # warp tree injects its own advection / two-phase sub-solvers
    assert adv_warp.AdvDiffSolver is not adv_cuda.AdvDiffSolver
    assert issubclass(adv_warp.AdvDiffSolver, adv_cuda.AdvDiffSolver)
    assert tp_warp.TwoPhase is not tp_cuda.TwoPhase
    assert issubclass(tp_warp.TwoPhase, tp_cuda.TwoPhase)


# ─────────────────────────── advection flux parity ─────────────────────────
def _build_adv(AdvCls, device, method="quick", dtype=torch.float64):
    n = 32
    x = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    y = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    return AdvCls(device, dt=1e-3, x=x, y=y, nu=1e-6, method=method)


@pytest.mark.skipif(not CUDA, reason="native advect_flux_add is CUDA-only")
@pytest.mark.parametrize("method", ["quick", "vanLeer", "cds", "cubista"])
def test_advection_flux_warp_matches_native(method):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    n = 32
    u0 = torch.rand(n, n, generator=g, device=dev, dtype=torch.float64)
    v0 = torch.rand(n, n, generator=g, device=dev, dtype=torch.float64)

    a_native = _build_adv(adv_cuda.AdvDiffSolver, dev, method)
    a_warp = _build_adv(adv_warp.AdvDiffSolver, dev, method)
    assert a_warp._is_cuda and a_native._is_cuda

    out_n = a_native._solve_convective(u0.clone(), v0.clone())
    out_w = a_warp._solve_convective(u0.clone(), v0.clone())
    for cn, cw in zip(out_n, out_w):
        # advect_flux_add_warp is bit-exact vs native (validated in warp_poc).
        assert torch.equal(cn, cw), f"{method}: max|Δ|={ (cn-cw).abs().max().item() }"


@pytest.mark.skipif(not CUDA, reason="native advect_flux_add is CUDA-only")
@pytest.mark.parametrize("method", ["quick", "vanLeer", "cds", "cubista"])
def test_advection_flux_warp_matches_native_f32(method):
    """float32 solver: the dtype-generic Warp flux runs (no f64 assert) and
    matches the native f32 op to single precision (not bit-exact: f32 FMA/order)."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    n = 32
    u0 = torch.rand(n, n, generator=g, device=dev, dtype=torch.float32)
    v0 = torch.rand(n, n, generator=g, device=dev, dtype=torch.float32)

    a_native = _build_adv(adv_cuda.AdvDiffSolver, dev, method, dtype=torch.float32)
    a_warp = _build_adv(adv_warp.AdvDiffSolver, dev, method, dtype=torch.float32)

    out_n = a_native._solve_convective(u0.clone(), v0.clone())
    out_w = a_warp._solve_convective(u0.clone(), v0.clone())
    for cn, cw in zip(out_n, out_w):
        assert cn.dtype == torch.float32 and cw.dtype == torch.float32
        assert torch.allclose(cn, cw, rtol=1e-4, atol=1e-6), \
            f"{method} f32: max|Δ|={(cn - cw).abs().max().item():.2e}"


# ─────────────────────────── cvof parity (GPU) ─────────────────────────────
def _build_tp(TPCls, device, dtype=torch.float64):
    n = 24
    x = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    y = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    init = lambda X, Y: (X < 0.5).to(dtype)
    return TPCls(x, y, h=1.0 / (n - 1), alpha_init=init, device=device)


@pytest.mark.skipif(not CUDA, reason="native cvof_sweep is CUDA-only")
def test_cvof_warp_matches_native_gpu():
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(1)
    tp_n = _build_tp(tp_cuda.TwoPhase, dev)
    tp_w = _build_tp(tp_warp.TwoPhase, dev)
    a = tp_n.alpha.clone()
    u = (torch.rand_like(a, dtype=torch.float64) - 0.5)
    for d in range(2):
        out_n = tp_n._cvof_sweep(a.clone(), u, d, dt=1e-3)
        out_w = tp_w._cvof_sweep(a.clone(), u, d, dt=1e-3)
        assert torch.equal(out_n, out_w), (
            f"cvof d={d}: max|Δ|={(out_n - out_w).abs().max().item()}"
        )


@pytest.mark.skipif(not CUDA, reason="native cvof_sweep is CUDA-only")
def test_cvof_warp_matches_native_gpu_f32():
    """float32 two-phase: the dtype-generic Warp cvof runs (no f64-only path)
    and matches the native f32 op to single precision."""
    dev = torch.device("cuda")
    tp_n = _build_tp(tp_cuda.TwoPhase, dev, dtype=torch.float32)
    tp_w = _build_tp(tp_warp.TwoPhase, dev, dtype=torch.float32)
    a = tp_n.alpha.clone()
    u = (torch.rand_like(a, dtype=torch.float32) - 0.5)
    for d in range(2):
        out_n = tp_n._cvof_sweep(a.clone(), u, d, dt=1e-3)
        out_w = tp_w._cvof_sweep(a.clone(), u, d, dt=1e-3)
        assert out_w.dtype == torch.float32
        assert torch.allclose(out_n, out_w, rtol=1e-4, atol=1e-6), (
            f"cvof f32 d={d}: max|Δ|={(out_n - out_w).abs().max().item():.2e}"
        )


# ────────────── Kernel A/B (2-D) marshalling-bridge parity ──────────────────
from lilytorch.warp_poc.scene_2d import make_synthetic_scene_2d

_RHO, _DT = 1000.0, 1e-3


def _run_chain(stream_fn, bdim_fn, sc, up, vp, mu0_proj):
    """Native-positional Kernel A (streaming) → Kernel B (bdim) chain, exactly
    as ``_fluid_step_kernel_2d`` calls them.  Returns (u0, v0, ch, cv)."""
    dev = up.device
    Ngx, Ngy = sc["Ngx"], sc["Ngy"]
    opt = dict(dtype=up.dtype, device=dev)
    sdf_cc = torch.full((Ngx, Ngy), 1e4, **opt)
    sdf_u = torch.full((Ngx, Ngy), 1e4, **opt)
    sdf_v = torch.full((Ngx, Ngy), 1e4, **opt)
    bu = torch.zeros((Ngx, Ngy), **opt)
    bv = torch.zeros((Ngx, Ngy), **opt)
    N = Ngx * Ngy
    kcc = torch.empty(N, dtype=torch.int64, device=dev)
    ku = torch.empty(N, dtype=torch.int64, device=dev)
    kv = torch.empty(N, dtype=torch.int64, device=dev)
    nu_ = torch.empty(1, **opt); nv_ = torch.empty(1, **opt)
    du_ = torch.empty(1, **opt); dv_ = torch.empty(1, **opt)
    di0, dj0, dAi, dAj = sc["dirty_bounds"]

    stream_fn(sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
              sc["kin"], sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"],
              float(sc["h"]), int(sc["max_vol"]),
              sdf_cc, sdf_u, sdf_v, bu, bv, kcc, ku, kv,
              0, di0, dj0, dAi, dAj, nu_, nv_, du_, dv_, 0.0)

    u0, v0 = up.clone(), vp.clone()
    ch = torch.full((Ngx, Ngy), _DT / _RHO, **opt)
    cv = torch.full((Ngx, Ngy), _DT / _RHO, **opt)
    eps = 2.0 * float(sc["h"])
    bdim_fn(up, vp, sdf_u, sdf_v, bu, bv, u0, v0, ch, cv,
            eps, _RHO, _DT, float(sc["h"]), di0, dj0, dAi, dAj, mu0_proj)
    return u0, v0, ch, cv


@pytest.mark.skipif(not CUDA, reason="native Kernel A/B are CUDA-only")
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("B", [1, 3])
def test_kernelAB_2d_bridge_matches_native(B, dtype):
    """The ``src_warp.kernel`` 2-D dispatch (Kernel A bridge + Kernel B, both now
    dtype-generic on Warp — no native fallback) chained as the solver step calls
    it, vs the native chain, at f32 AND f64.  Agreement is at the documented
    Kernel-A SDF gate (~1e-7 f64 / f32-eps f32, native interpolates the SDF in
    f32) propagated through Kernel B."""
    dev = torch.device("cuda")
    sc = make_synthetic_scene_2d(96, 64, B, device=dev, dtype=dtype)
    for k in ("F_flat", "body_meta", "kin", "gx", "gy"):
        sc[k] = sc[k].to(dtype)
    torch.manual_seed(3)
    up = torch.randn(sc["Ngx"], sc["Ngy"], dtype=dtype, device=dev)
    vp = torch.randn(sc["Ngx"], sc["Ngy"], dtype=dtype, device=dev)

    u_n, v_n, ch_n, cv_n = _run_chain(
        kcuda.streaming_sdf_stag_2d_multi, kcuda.bdim_coeff_2d,
        sc, up.clone(), vp.clone(), 1)
    u_w, v_w, ch_w, cv_w = _run_chain(
        kwarp.streaming_sdf_stag_2d_multi, kwarp.bdim_coeff_2d,
        sc, up.clone(), vp.clone(), 1)

    for nm, n, w in (("u0", u_n, u_w), ("v0", v_n, v_w),
                     ("ch", ch_n, ch_w), ("cv", cv_n, cv_w)):
        assert w.dtype == dtype
        rel = (n - w).abs() / n.abs().clamp_min(1e-8)
        assert rel.max().item() < 5e-4, f"B={B} {dtype} {nm} rel {rel.max():.2e}"


# ─────────────────── CPU end-to-end payoff (single source) ──────────────────
def test_cvof_warp_cpu_matches_python_reference():
    """One Warp ``@wp.kernel`` source serves the CPU path: the Warp cvof on CPU
    matches the native python reference sweep (native CUDA op unavailable on CPU).
    """
    dev = torch.device("cpu")
    tp_w = _build_tp(tp_warp.TwoPhase, dev)
    a = tp_w.alpha.clone()
    torch.manual_seed(2)
    u = (torch.rand_like(a, dtype=torch.float64) - 0.5)
    for d in range(2):
        out_warp = tp_w._cvof_sweep(a.clone(), u, d, dt=1e-3)        # Warp on CPU
        out_ref = tp_w._cvof_sweep_python(a.clone(), u, d, dt=1e-3)  # python ref
        diff = (out_warp - out_ref).abs().max().item()
        assert diff < 1e-12, f"cvof CPU d={d}: max|Δ|={diff}"
