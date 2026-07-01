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
import warp as wp

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
    # plus the 2-D and 3-D Kernel A/B marshalling bridges.
    assert kcuda.WARP_BACKED == frozenset()
    assert {
        "advect_flux_add", "cvof_sweep",
        "streaming_sdf_stag_2d_multi", "bdim_coeff_2d",
        "streaming_sdf_stag_3d_multi", "bdim_coeff_3d",
        "apply_bcs_2d", "apply_bcs_3d", "interp_2d", "interp_3d",
        "bdim_coeff_sigma_2d", "bdim_coeff_sigma_3d",
    }.issubset(kwarp.WARP_BACKED)


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


@pytest.mark.skipif(not CUDA, reason="native Kernel A is CUDA-only")
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_kernelA_2d_graph_replay_matches_native(dtype):
    """Perf path: the bridge's CUDA-graph fast path (use_graph=True) captures on
    the 2nd sighting of stable output buffers and replays.  Verify the replayed
    streaming output matches native — incl. a *kinematics change between calls*
    (update_kinematics runs outside the graph, so the replay must pick up the
    new body pose)."""
    dev = torch.device("cuda")
    sc = make_synthetic_scene_2d(96, 64, 2, device=dev, dtype=dtype)
    for k in ("F_flat", "body_meta", "kin", "gx", "gy"):
        sc[k] = sc[k].to(dtype)
    Ngx, Ngy = sc["Ngx"], sc["Ngy"]
    opt = dict(dtype=dtype, device=dev)
    di0, dj0, dAi, dAj = sc["dirty_bounds"]
    N = Ngx * Ngy
    # persistent output buffers (stable pointers → graph capture triggers)
    sdf = [torch.empty((Ngx, Ngy), **opt) for _ in range(3)]
    bdy = [torch.empty((Ngx, Ngy), **opt) for _ in range(2)]
    ks = [torch.empty(N, dtype=torch.int64, device=dev) for _ in range(3)]
    nn = [torch.empty(1, **opt) for _ in range(4)]

    def run(stream_fn, kin, use_graph):
        # Native needs the SDF pre-filled to FAR; for the Warp graph path we
        # instead POISON the buffers (the bridge folds the FAR/0 reset into the
        # captured graph, so a correct fold must overwrite the poison).
        poison = use_graph
        for s in sdf: s.fill_(-7.0 if poison else 1e4)
        for b in bdy: b.fill_(-7.0 if poison else 0.0)
        stream_fn(sc["F_flat"], sc["F_offsets"], sc["body_shapes"],
                  sc["body_meta"], kin, sc["aabb_lo"], sc["aabb_dim"],
                  sc["gx"], sc["gy"], float(sc["h"]), int(sc["max_vol"]),
                  sdf[0], sdf[1], sdf[2], bdy[0], bdy[1], ks[0], ks[1], ks[2],
                  0, di0, dj0, dAi, dAj, nn[0], nn[1], nn[2], nn[3], 0.0,
                  **({"use_graph": True} if use_graph else {}))
        return sdf[1].clone(), sdf[2].clone(), bdy[0].clone(), bdy[1].clone()

    kin2 = sc["kin"].clone()  # perturb body kinematics for the replay step
    kin2[8:10] += 0.5  # linear-velocity components (2-D kin layout: …lv(2)…)

    # warp: call 1 (eager), call 2 (captures graph), call 3 (replay w/ new kin)
    run(kwarp.streaming_sdf_stag_2d_multi, sc["kin"], True)
    run(kwarp.streaming_sdf_stag_2d_multi, sc["kin"], True)
    w = run(kwarp.streaming_sdf_stag_2d_multi, kin2, True)  # graph replay
    n = run(kcuda.streaming_sdf_stag_2d_multi, kin2, False)  # native ref
    wp.synchronize()
    for nm, cn, cw in zip(("sdf_u", "sdf_v", "bU", "bV"), n, w):
        rel = (cn - cw).abs() / cn.abs().clamp_min(1e-6)
        assert rel.max().item() < 5e-4, f"graph replay {nm} rel {rel.max():.2e}"


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


def _run_chain_sigma_2d(stream_fn, bdim_fn, sc, up, vp, sigma_shifts,
                        emit_keys_kw):
    """σ chain: Kernel A (streaming, emitting body-id keys) → σ Kernel B (reads
    the keys).  Returns (u0, v0, ch, cv, key_u&mask, key_v&mask)."""
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
              0, di0, dj0, dAi, dAj, nu_, nv_, du_, dv_, 0.0, **emit_keys_kw)

    u0, v0 = up.clone(), vp.clone()
    ch = torch.full((Ngx, Ngy), _DT / _RHO, **opt)
    cv = torch.full((Ngx, Ngy), _DT / _RHO, **opt)
    eps = 2.0 * float(sc["h"])
    bdim_fn(up, vp, sdf_u, sdf_v, bu, bv, u0, v0, ch, cv,
            ku, kv, sigma_shifts,
            eps, _RHO, _DT, float(sc["h"]), di0, dj0, dAi, dAj, 1)
    mask = (1 << 32) - 1
    return u0, v0, ch, cv, ku & mask, kv & mask


@pytest.mark.skipif(not CUDA, reason="native Kernel A/B are CUDA-only")
@pytest.mark.parametrize("B", [2, 3])
def test_kernelAB_2d_sigma_chain_matches_native(B):
    """Item 5: the Warp streaming bridge emits the winning body-id into
    ``key_u/key_v`` (``emit_keys``) and the Warp σ Kernel B reads it — no native
    fallback on the σ path.  Run at f32 (native quantises the union SDF to f32
    for its packed key, so f32 makes the winner selection identical), compare
    the σ-corrected fields AND the decoded body-ids vs native."""
    dev = torch.device("cuda")
    dtype = torch.float32
    sc = make_synthetic_scene_2d(96, 64, B, device=dev, dtype=dtype)
    for k in ("F_flat", "body_meta", "kin", "gx", "gy"):
        sc[k] = sc[k].to(dtype)
    torch.manual_seed(4)
    up = torch.randn(sc["Ngx"], sc["Ngy"], dtype=dtype, device=dev)
    vp = torch.randn(sc["Ngx"], sc["Ngy"], dtype=dtype, device=dev)
    # one non-zero σ shift per body
    ss = (0.02 + 0.03 * torch.arange(B, device=dev)).to(torch.float32)

    # native σ wrapper takes the packed-signature args via the facade.
    def _native_sigma(up_, vp_, su, sv, bu, bv, u0, v0, ch, cv,
                      ku, kv, ss_, eps, rho, dt, h, di0, dj0, dAi, dAj, mp):
        kcuda.bdim_coeff_sigma_2d(up_, vp_, su, sv, bu, bv, u0, v0, ch, cv,
                                  ku, kv, ss_, eps, rho, dt, h,
                                  di0, dj0, dAi, dAj, mp)

    def _warp_sigma(up_, vp_, su, sv, bu, bv, u0, v0, ch, cv,
                    ku, kv, ss_, eps, rho, dt, h, di0, dj0, dAi, dAj, mp):
        kwarp.bdim_coeff_2d(up_, vp_, su, sv, bu, bv, u0, v0, ch, cv,
                            eps, rho, dt, h, di0, dj0, dAi, dAj, mp,
                            key_u=ku, key_v=kv, sigma_shifts=ss_)

    u_n, v_n, ch_n, cv_n, ku_n, kv_n = _run_chain_sigma_2d(
        kcuda.streaming_sdf_stag_2d_multi, _native_sigma, sc,
        up.clone(), vp.clone(), ss, {})
    u_w, v_w, ch_w, cv_w, ku_w, kv_w = _run_chain_sigma_2d(
        kwarp.streaming_sdf_stag_2d_multi, _warp_sigma, sc,
        up.clone(), vp.clone(), ss, {"emit_keys": True})

    # decoded body-ids must agree within the dirty sub-block (native only
    # initialises / writes keys there; the σ Kernel B only reads them there).
    di0, dj0, dAi, dAj = sc["dirty_bounds"]
    Ngx, Ngy = sc["Ngx"], sc["Ngy"]
    sl = (slice(di0, di0 + dAi), slice(dj0, dj0 + dAj))
    for nm, n, w in (("u", ku_n, ku_w), ("v", kv_n, kv_w)):
        nn = n.reshape(Ngx, Ngy)[sl]
        ww = w.reshape(Ngx, Ngy)[sl]
        assert torch.equal(nn, ww), f"{nm} body-id keys differ in dirty block"
    for nm, n, w in (("u0", u_n, u_w), ("v0", v_n, v_w),
                     ("ch", ch_n, ch_w), ("cv", cv_n, cv_w)):
        rel = (n - w).abs() / n.abs().clamp_min(1e-6)
        assert rel.max().item() < 5e-4, f"σ B={B} {nm} rel {rel.max():.2e}"


# ────────────── Kernel A/B (3-D) marshalling-bridge parity ──────────────────
from lilytorch.warp_poc.bench_viability import make_synthetic_scene


def _run_chain_3d(stream_fn, bdim_fn, sc, up, vp, wp_, mu0_proj):
    """Native-positional 3-D Kernel A (streaming) → Kernel B (bdim) chain, as
    ``_fluid_step_kernel_3d`` calls them.  Returns (u0, v0, w0, ch, cv, cw)."""
    dev = up.device
    Ngx, Ngy, Ngz = sc["Ngx"], sc["Ngy"], sc["Ngz"]
    opt = dict(dtype=up.dtype, device=dev)
    sdf_cc = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    sdf_u = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    sdf_v = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    sdf_w = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    bu = torch.zeros((Ngx, Ngy, Ngz), **opt)
    bv = torch.zeros((Ngx, Ngy, Ngz), **opt)
    bw = torch.zeros((Ngx, Ngy, Ngz), **opt)
    di0, dj0, dk0, dAi, dAj, dAk = sc["dirty_bounds"]
    dvol = int(dAi) * int(dAj) * int(dAk)
    kcc = torch.empty(dvol, dtype=torch.int64, device=dev)
    ku = torch.empty(dvol, dtype=torch.int64, device=dev)
    kv = torch.empty(dvol, dtype=torch.int64, device=dev)
    kw = torch.empty(dvol, dtype=torch.int64, device=dev)
    nu_ = torch.empty(1, **opt); nv_ = torch.empty(1, **opt); nw_ = torch.empty(1, **opt)
    du_ = torch.empty(1, **opt); dv_ = torch.empty(1, **opt); dw_ = torch.empty(1, **opt)

    stream_fn(sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
              sc["kin"], sc["aabb_lo"], sc["aabb_dim"],
              sc["gx"], sc["gy"], sc["gz"],
              float(sc["h"]), int(sc["max_vol"]),
              sdf_cc, sdf_u, sdf_v, sdf_w, bu, bv, bw,
              kcc, ku, kv, kw, 0,
              di0, dj0, dk0, dAi, dAj, dAk,
              nu_, nv_, nw_, du_, dv_, dw_, 0.0)

    u0, v0, w0 = up.clone(), vp.clone(), wp_.clone()
    base = _DT / _RHO
    ch = torch.full((Ngx - 1, Ngy - 2, Ngz - 2), base, **opt)
    cv = torch.full((Ngx - 2, Ngy - 1, Ngz - 2), base, **opt)
    cw = torch.full((Ngx - 2, Ngy - 2, Ngz - 1), base, **opt)
    eps = 2.0 * float(sc["h"])
    bdim_fn(up, vp, wp_, sdf_u, sdf_v, sdf_w, bu, bv, bw,
            u0, v0, w0, ch, cv, cw,
            eps, _RHO, _DT, float(sc["h"]),
            di0, dj0, dk0, dAi, dAj, dAk, mu0_proj)
    return u0, v0, w0, ch, cv, cw


@pytest.mark.skipif(not CUDA, reason="native 3-D Kernel A is CUDA-only")
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_kernelA_3d_graph_replay_matches_native(dtype):
    """3-D analogue of :func:`test_kernelA_2d_graph_replay_matches_native`: the
    bridge captures the fanned streaming on the 2nd sighting of stable buffers and
    replays it; the replay must match native after a kinematics change (the body
    pose update runs outside the captured graph)."""
    dev = torch.device("cuda")
    sc = make_synthetic_scene(48, 32, 32, 2, device=dev, dtype=dtype)
    for k in ("F_flat", "body_meta", "kin", "gx", "gy", "gz"):
        sc[k] = sc[k].to(dtype)
    Ngx, Ngy, Ngz = sc["Ngx"], sc["Ngy"], sc["Ngz"]
    opt = dict(dtype=dtype, device=dev)
    di0, dj0, dk0, dAi, dAj, dAk = sc["dirty_bounds"]
    dvol = int(dAi) * int(dAj) * int(dAk)
    sdf = [torch.empty((Ngx, Ngy, Ngz), **opt) for _ in range(4)]
    bdy = [torch.empty((Ngx, Ngy, Ngz), **opt) for _ in range(3)]
    nn = [torch.empty(1, **opt) for _ in range(6)]
    # dummy keys (non-σ graph path ignores them)
    ksd = [torch.empty(1, dtype=torch.int64, device=dev) for _ in range(4)]
    ksn = [torch.empty(dvol, dtype=torch.int64, device=dev) for _ in range(4)]

    def run(stream_fn, kin, keys, use_graph):
        # Poison the Warp graph-path buffers (its FAR/0 reset is folded into the
        # captured graph); native gets the conventional FAR pre-fill.
        poison = use_graph
        for s in sdf: s.fill_(-7.0 if poison else 1e4)
        for b in bdy: b.fill_(-7.0 if poison else 0.0)
        stream_fn(sc["F_flat"], sc["F_offsets"], sc["body_shapes"],
                  sc["body_meta"], kin, sc["aabb_lo"], sc["aabb_dim"],
                  sc["gx"], sc["gy"], sc["gz"], float(sc["h"]), int(sc["max_vol"]),
                  sdf[0], sdf[1], sdf[2], sdf[3], bdy[0], bdy[1], bdy[2],
                  keys[0], keys[1], keys[2], keys[3], 0,
                  di0, dj0, dk0, dAi, dAj, dAk,
                  nn[0], nn[1], nn[2], nn[3], nn[4], nn[5], 0.0,
                  **({"use_graph": True} if use_graph else {}))
        return [sdf[i].clone() for i in (1, 2, 3)] + [b.clone() for b in bdy]

    kin2 = sc["kin"].clone()
    kin2[15:18] += 0.5  # 3-D kin layout: …lv(3) at offset 15…
    run(kwarp.streaming_sdf_stag_3d_multi, sc["kin"], ksd, True)
    run(kwarp.streaming_sdf_stag_3d_multi, sc["kin"], ksd, True)
    w = run(kwarp.streaming_sdf_stag_3d_multi, kin2, ksd, True)
    n = run(kcuda.streaming_sdf_stag_3d_multi, kin2, ksn, False)
    wp.synchronize()
    for nm, cn, cw in zip(("sdf_u", "sdf_v", "sdf_w", "bU", "bV", "bW"), n, w):
        rel = (cn - cw).abs() / cn.abs().clamp_min(1e-6)
        assert rel.max().item() < 5e-4, f"3-D graph replay {nm} rel {rel.max():.2e}"


@pytest.mark.skipif(not CUDA, reason="native 3-D Kernel A/B are CUDA-only")
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("B", [1, 3])
def test_kernelAB_3d_bridge_matches_native(B, dtype):
    """The ``src_warp.kernel`` 3-D dispatch (Kernel A bridge + Kernel B, both
    dtype-generic on Warp) chained as ``_fluid_step_kernel_3d`` calls it, vs the
    native chain, at f32 AND f64."""
    dev = torch.device("cuda")
    sc = make_synthetic_scene(48, 32, 32, B, device=dev, dtype=dtype)
    for k in ("F_flat", "body_meta", "kin", "gx", "gy", "gz"):
        sc[k] = sc[k].to(dtype)
    torch.manual_seed(3)
    up = torch.randn(sc["Ngx"], sc["Ngy"], sc["Ngz"], dtype=dtype, device=dev)
    vp = torch.randn(sc["Ngx"], sc["Ngy"], sc["Ngz"], dtype=dtype, device=dev)
    wp_ = torch.randn(sc["Ngx"], sc["Ngy"], sc["Ngz"], dtype=dtype, device=dev)

    n = _run_chain_3d(kcuda.streaming_sdf_stag_3d_multi, kcuda.bdim_coeff_3d,
                      sc, up.clone(), vp.clone(), wp_.clone(), 1)
    w = _run_chain_3d(kwarp.streaming_sdf_stag_3d_multi, kwarp.bdim_coeff_3d,
                      sc, up.clone(), vp.clone(), wp_.clone(), 1)

    for nm, cn, cw in zip(("u0", "v0", "w0", "ch", "cv", "cw"), n, w):
        assert cw.dtype == dtype
        rel = (cn - cw).abs() / cn.abs().clamp_min(1e-8)
        assert rel.max().item() < 5e-4, f"B={B} {dtype} {nm} rel {rel.max():.2e}"


def _run_chain_sigma_3d(stream_fn, bdim_fn, sc, up, vp, wp_, sigma_shifts,
                        emit_keys_kw):
    """3-D σ chain: Kernel A (streaming, dirty-local body-id keys) → σ Kernel B.
    Returns (u0, v0, w0, ch, cv, cw, key_u&mask, key_v&mask, key_w&mask)."""
    dev = up.device
    Ngx, Ngy, Ngz = sc["Ngx"], sc["Ngy"], sc["Ngz"]
    opt = dict(dtype=up.dtype, device=dev)
    sdf = [torch.full((Ngx, Ngy, Ngz), 1e4, **opt) for _ in range(4)]
    sdf_cc, sdf_u, sdf_v, sdf_w = sdf
    bu = torch.zeros((Ngx, Ngy, Ngz), **opt)
    bv = torch.zeros((Ngx, Ngy, Ngz), **opt)
    bw = torch.zeros((Ngx, Ngy, Ngz), **opt)
    di0, dj0, dk0, dAi, dAj, dAk = sc["dirty_bounds"]
    dvol = int(dAi) * int(dAj) * int(dAk)
    kcc = torch.empty(dvol, dtype=torch.int64, device=dev)
    ku = torch.empty(dvol, dtype=torch.int64, device=dev)
    kv = torch.empty(dvol, dtype=torch.int64, device=dev)
    kw = torch.empty(dvol, dtype=torch.int64, device=dev)
    nu_ = torch.empty(1, **opt); nv_ = torch.empty(1, **opt); nw_ = torch.empty(1, **opt)
    du_ = torch.empty(1, **opt); dv_ = torch.empty(1, **opt); dw_ = torch.empty(1, **opt)

    stream_fn(sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
              sc["kin"], sc["aabb_lo"], sc["aabb_dim"],
              sc["gx"], sc["gy"], sc["gz"],
              float(sc["h"]), int(sc["max_vol"]),
              sdf_cc, sdf_u, sdf_v, sdf_w, bu, bv, bw,
              kcc, ku, kv, kw, 0,
              di0, dj0, dk0, dAi, dAj, dAk,
              nu_, nv_, nw_, du_, dv_, dw_, 0.0, **emit_keys_kw)

    u0, v0, w0 = up.clone(), vp.clone(), wp_.clone()
    base = _DT / _RHO
    ch = torch.full((Ngx - 1, Ngy - 2, Ngz - 2), base, **opt)
    cv = torch.full((Ngx - 2, Ngy - 1, Ngz - 2), base, **opt)
    cw = torch.full((Ngx - 2, Ngy - 2, Ngz - 1), base, **opt)
    eps = 2.0 * float(sc["h"])
    bdim_fn(up, vp, wp_, sdf_u, sdf_v, sdf_w, bu, bv, bw,
            u0, v0, w0, ch, cv, cw, ku, kv, kw, sigma_shifts,
            eps, _RHO, _DT, float(sc["h"]),
            di0, dj0, dk0, dAi, dAj, dAk, 1)
    mask = (1 << 32) - 1
    return u0, v0, w0, ch, cv, cw, ku & mask, kv & mask, kw & mask


@pytest.mark.skipif(not CUDA, reason="native 3-D Kernel A/B are CUDA-only")
@pytest.mark.parametrize("B", [2, 3])
def test_kernelAB_3d_sigma_chain_matches_native(B):
    """Item 5 (3-D): the Warp streaming bridge emits the winning body-id into the
    dirty-local ``key_{u,v,w}`` (``emit_keys``) and the Warp σ Kernel B reads it
    — no native fallback on the σ path.  f32 to match native's f32 packed key."""
    dev = torch.device("cuda")
    dtype = torch.float32
    sc = make_synthetic_scene(48, 32, 32, B, device=dev, dtype=dtype)
    for k in ("F_flat", "body_meta", "kin", "gx", "gy", "gz"):
        sc[k] = sc[k].to(dtype)
    torch.manual_seed(4)
    up = torch.randn(sc["Ngx"], sc["Ngy"], sc["Ngz"], dtype=dtype, device=dev)
    vp = torch.randn(sc["Ngx"], sc["Ngy"], sc["Ngz"], dtype=dtype, device=dev)
    wp_ = torch.randn(sc["Ngx"], sc["Ngy"], sc["Ngz"], dtype=dtype, device=dev)
    ss = (0.02 + 0.03 * torch.arange(B, device=dev)).to(torch.float32)

    def _native_sigma(up_, vp_, wp__, su, sv, sw, bu, bv, bw, u0, v0, w0,
                      ch, cv, cw, ku, kv, kw, ss_, eps, rho, dt, h,
                      di0, dj0, dk0, dAi, dAj, dAk, mp):
        kcuda.bdim_coeff_sigma_3d(up_, vp_, wp__, su, sv, sw, bu, bv, bw,
                                  u0, v0, w0, ch, cv, cw, ku, kv, kw, ss_,
                                  eps, rho, dt, h, di0, dj0, dk0, dAi, dAj, dAk, mp)

    def _warp_sigma(up_, vp_, wp__, su, sv, sw, bu, bv, bw, u0, v0, w0,
                    ch, cv, cw, ku, kv, kw, ss_, eps, rho, dt, h,
                    di0, dj0, dk0, dAi, dAj, dAk, mp):
        kwarp.bdim_coeff_3d(up_, vp_, wp__, su, sv, sw, bu, bv, bw,
                            u0, v0, w0, ch, cv, cw, eps, rho, dt, h,
                            di0, dj0, dk0, dAi, dAj, dAk, mp,
                            key_u=ku, key_v=kv, key_w=kw, sigma_shifts=ss_)

    n = _run_chain_sigma_3d(kcuda.streaming_sdf_stag_3d_multi, _native_sigma,
                            sc, up.clone(), vp.clone(), wp_.clone(), ss, {})
    w = _run_chain_sigma_3d(kwarp.streaming_sdf_stag_3d_multi, _warp_sigma,
                            sc, up.clone(), vp.clone(), wp_.clone(), ss,
                            {"emit_keys": True})

    # body-id keys are dirty_vol-sized & dirty-local → compare in full.
    for nm, kn, kw_ in (("u", n[6], w[6]), ("v", n[7], w[7]), ("w", n[8], w[8])):
        assert torch.equal(kn, kw_), f"{nm} body-id keys differ"
    for nm, cn, cw in zip(("u0", "v0", "w0", "ch", "cv", "cw"), n[:6], w[:6]):
        rel = (cn - cw).abs() / cn.abs().clamp_min(1e-6)
        assert rel.max().item() < 5e-4, f"σ B={B} {nm} rel {rel.max():.2e}"


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


# ─────────────── Lagrangian-force routing (src_warp method override) ─────────
import lilytorch.src.forces as _forces_mod


@pytest.mark.parametrize("ndim", [2, 3])
def test_lagrangian_force_override_routes_to_warp(ndim):
    """``src_warp.solver.FluidSolver`` overrides ``forces_lagrangian_{2,3}d`` to
    swap the ``lilytorch.src.forces`` module-global kernel for the Warp facade
    shim for the duration of the (inherited) force readout, then restore it — the
    same localized injection used for the sub-solvers.  Verify (a) the override
    is present, (b) the inherited body sees the Warp shim as its kernel, and (c)
    the global is restored afterward (no leakage into the native ``src`` path)."""
    name = f"forces_lagrangian_{ndim}d"
    gname = f"_lagrangian_forces_{ndim}d_kernel"
    facade = getattr(kwarp, f"lagrangian_forces_{ndim}d")

    # (a) override present (not the inherited native function).
    assert getattr(swarp.FluidSolver, name) is not getattr(sbase.FluidSolver, name)

    inst = swarp.FluidSolver.__new__(swarp.FluidSolver)
    captured = {}

    def _probe(self, *a, **k):
        captured["kernel"] = getattr(_forces_mod, gname)

    before = getattr(_forces_mod, gname)
    orig = getattr(sbase.FluidSolver, name)
    setattr(sbase.FluidSolver, name, _probe)
    try:
        getattr(inst, name)(*([None] * (ndim + 1)), 0)  # (u,[v,]w/p,..., iteration)
    finally:
        setattr(sbase.FluidSolver, name, orig)

    # (b) the inherited body ran with the Warp shim bound as the kernel.
    assert captured["kernel"] is facade
    # (c) the module global is restored (native path unaffected).
    assert getattr(_forces_mod, gname) is before


@pytest.mark.parametrize("ndim", [2, 3])
def test_eulerian_force_override_routes_to_warp(ndim):
    """``forces_method2{,_3d}`` (inherited) call the module-global
    ``streaming_sdf_forces_post_{2,3}d``; the ``src_warp.solver`` override swaps
    it for the Warp Eulerian shim for the call, then restores it."""
    name = "forces_method2" if ndim == 2 else "forces_method2_3d"
    gname = f"streaming_sdf_forces_post_{ndim}d"
    facade = getattr(kwarp, gname)

    assert getattr(swarp.FluidSolver, name) is not getattr(sbase.FluidSolver, name)

    inst = swarp.FluidSolver.__new__(swarp.FluidSolver)
    captured = {}

    def _probe(self, *a, **k):
        captured["kernel"] = getattr(_forces_mod, gname)

    before = getattr(_forces_mod, gname)
    orig = getattr(sbase.FluidSolver, name)
    setattr(sbase.FluidSolver, name, _probe)
    try:
        getattr(inst, name)(*([None] * (ndim + 1)), 0)
    finally:
        setattr(sbase.FluidSolver, name, orig)

    assert captured["kernel"] is facade
    assert getattr(_forces_mod, gname) is before


# ─────────────── set_BCs routing (src_warp.advection override) ──────────────
def _build_adv_nd(AdvCls, device, ndim, dtype=torch.float64, n=24):
    x = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    y = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    # All-Dirichlet-0 on EVERY face so each boundary ghost is set to 0:
    # corner/edge write-races resolve to the same value, making the
    # native↔warp comparison deterministic (Neumann faces would race).
    nf = 2 * ndim
    D = ("D",) * nf
    Z = (0,) * nf
    if ndim == 2:
        return AdvCls(device, dt=1e-3, x=x, y=y, nu=1e-6, method="quick",
                      BC_type_u=D, BC_values_u=Z, BC_type_v=D, BC_values_v=Z)
    z = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    return AdvCls(device, dt=1e-3, x=x, y=y, z=z, nu=1e-6, method="quick",
                  BC_type_u=D, BC_values_u=Z, BC_type_v=D, BC_values_v=Z,
                  BC_type_w=D, BC_values_w=Z)


@pytest.mark.skipif(not CUDA, reason="native apply_bcs fused path is CUDA-only")
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_set_bcs_override_routes_to_warp(ndim, dtype):
    """The ``src_warp.advection.AdvDiffSolver.set_BCs`` override dispatches the
    fused ghost-line writes through the Warp ``apply_bcs_{2,3}d`` kernel (the
    native ``set_BCs`` calls ``torch.ops…`` directly, so this is a real method
    override, not the module-global swap).  Verify (a) the override is present,
    (b) the Warp facade op is dispatched exactly once, (c) the Dirichlet-0
    ghost faces are actually written to 0.  Numerical bit-exactness of the
    kernel itself (incl. f32 + non-cubic dual face-dims) is covered per-op in
    ``test_misc_{2,3}d`` with disjoint ops — a full closed-box comparison is
    structurally non-deterministic (edge/corner cells are written by multiple
    ops in one stage launch, an order-undefined GPU race in native too)."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(7)
    shape = (24,) * ndim
    orig = [torch.rand(shape, generator=g, device=dev, dtype=dtype).contiguous()
            for _ in range(ndim)]
    fw = [f.clone() for f in orig]

    a_n = _build_adv_nd(adv_cuda.AdvDiffSolver, dev, ndim, dtype)
    a_w = _build_adv_nd(adv_warp.AdvDiffSolver, dev, ndim, dtype)

    # (a) override present (not inherited native set_BCs).
    assert type(a_w).set_BCs is not type(a_n).set_BCs

    # (b) routing: the Warp graph-cached apply_bcs runner fires exactly once.
    runner = getattr(a_w, f"_bcs_runner_{ndim}d")  # property → lazily-built runner
    called = {"n": 0}

    class _Spy:
        def __call__(self, *a, **k):
            called["n"] += 1
            return runner(*a, **k)

    setattr(a_w, f"_bcs_graph_{ndim}d", _Spy())  # property returns this spy
    a_w.set_BCs(*fw)
    wp.synchronize()
    assert called["n"] == 1, "Warp apply_bcs runner was not dispatched"
    assert fw[0].dtype == dtype

    # (c) the kernel actually mutated the boundary ghost ring of every field
    #     (x-low ghost face), and left the deep interior untouched.
    for f, o in zip(fw, orig):
        assert not torch.equal(f[0], o[0]), "ghost face not written"
        interior = (slice(2, -2),) * ndim
        assert torch.equal(f[interior], o[interior]), "interior was modified"


# ─────────────────── Item 6: step-level independence ────────────────────────
# Every custom-kernel op the live src_warp fluid step touches must run on Warp.
# (a) static: WARP_BACKED ⊇ the step's custom ops.  (b) dynamic: with
# torch.ops.lilytorch_kernels monkeypatched to raise, the step's ops still run
# (they dispatch to Warp single-source kernels, never the native CUDA/C++ ops).

# Custom ops a coupled kernel step calls (see src/solver._fluid_step_kernel_*,
# advection.set_BCs, two_phase._cvof_sweep, forces.forces_method2*/lagrangian).
_STEP_CUSTOM_OPS = {
    "advect_flux_add", "cvof_sweep",
    "streaming_sdf_stag_2d_multi", "bdim_coeff_2d", "bdim_coeff_sigma_2d",
    "streaming_sdf_stag_3d_multi", "bdim_coeff_3d", "bdim_coeff_sigma_3d",
    "apply_bcs_2d", "apply_bcs_3d",
    "lagrangian_forces_2d", "lagrangian_forces_3d",
    "streaming_sdf_forces_post_2d", "streaming_sdf_forces_post_3d",
    "rbgs_sweep_2d", "rbgs_sweep_3d", "mg_residual_2d", "mg_residual_3d",
}


def test_warp_backed_covers_step_custom_ops():
    """Item 6 (2): every custom op the fluid step dispatches is Warp-backed."""
    missing = _STEP_CUSTOM_OPS - set(kwarp.WARP_BACKED)
    assert not missing, f"step ops not on Warp: {sorted(missing)}"


class _BoomOps:
    def __getattr__(self, name):
        raise RuntimeError(f"native op {name} must not be called (Warp path)")


@pytest.mark.skipif(not CUDA, reason="native Kernel A/B are CUDA-only")
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_no_native_kernel_calls_2d(monkeypatch, dtype):
    """Item 6 (1): the 2-D step's custom ops run with no native dispatch — build
    everything first, then patch torch.ops.lilytorch_kernels to raise."""
    dev = torch.device("cuda")
    # Build scenes / solvers BEFORE patching (native scene builders may run ops).
    sc = make_synthetic_scene_2d(96, 64, 3, device=dev, dtype=dtype)
    for k in ("F_flat", "body_meta", "kin", "gx", "gy"):
        sc[k] = sc[k].to(dtype)
    torch.manual_seed(1)
    up = torch.randn(sc["Ngx"], sc["Ngy"], dtype=dtype, device=dev)
    vp = torch.randn(sc["Ngx"], sc["Ngy"], dtype=dtype, device=dev)
    ss = (0.02 + 0.03 * torch.arange(3, device=dev)).to(torch.float32)
    adv = _build_adv_nd(adv_warp.AdvDiffSolver, dev, 2, dtype)
    bcf = [torch.randn(24, 24, dtype=dtype, device=dev).contiguous() for _ in range(2)]
    tp = _build_tp(tp_warp.TwoPhase, dev, dtype=dtype)
    al = tp.alpha.clone(); uvel = torch.rand_like(al) - 0.5

    def _warp_sigma(up_, vp_, su, sv, bu, bv, u0, v0, ch, cv, ku, kv, ss_,
                    eps, rho, dt, h, di0, dj0, dAi, dAj, mp):
        kwarp.bdim_coeff_2d(up_, vp_, su, sv, bu, bv, u0, v0, ch, cv,
                            eps, rho, dt, h, di0, dj0, dAi, dAj, mp,
                            key_u=ku, key_v=kv, sigma_shifts=ss_)

    monkeypatch.setattr(torch.ops, "lilytorch_kernels", _BoomOps())
    # Kernel A + B (+σ) chains.
    _run_chain(kwarp.streaming_sdf_stag_2d_multi, kwarp.bdim_coeff_2d,
               sc, up.clone(), vp.clone(), 1)
    _run_chain_sigma_2d(kwarp.streaming_sdf_stag_2d_multi, _warp_sigma, sc,
                        up.clone(), vp.clone(), ss, {"emit_keys": True})
    # advection flux + fused BCs + two-phase VOF sweep.
    adv._solve_convective(up.clone(), vp.clone())
    adv.set_BCs(*[b.clone() for b in bcf])
    tp._cvof_sweep(al.clone(), uvel, 0, dt=1e-3)
    wp.synchronize()


@pytest.mark.skipif(not CUDA, reason="native Kernel A/B are CUDA-only")
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_no_native_kernel_calls_3d(monkeypatch, dtype):
    """Item 6 (1): the 3-D step's custom ops run with no native dispatch."""
    dev = torch.device("cuda")
    sc = make_synthetic_scene(48, 32, 32, 3, device=dev, dtype=dtype)
    for k in ("F_flat", "body_meta", "kin", "gx", "gy", "gz"):
        sc[k] = sc[k].to(dtype)
    torch.manual_seed(1)
    up = torch.randn(sc["Ngx"], sc["Ngy"], sc["Ngz"], dtype=dtype, device=dev)
    vp = torch.randn(sc["Ngx"], sc["Ngy"], sc["Ngz"], dtype=dtype, device=dev)
    wpv = torch.randn(sc["Ngx"], sc["Ngy"], sc["Ngz"], dtype=dtype, device=dev)
    ss = (0.02 + 0.03 * torch.arange(3, device=dev)).to(torch.float32)
    adv = _build_adv_nd(adv_warp.AdvDiffSolver, dev, 3, dtype)
    bcf = [torch.randn(20, 20, 20, dtype=dtype, device=dev).contiguous()
           for _ in range(3)]

    def _warp_sigma(up_, vp_, wp__, su, sv, sw, bu, bv, bw, u0, v0, w0,
                    ch, cv, cw, ku, kv, kw, ss_, eps, rho, dt, h,
                    di0, dj0, dk0, dAi, dAj, dAk, mp):
        kwarp.bdim_coeff_3d(up_, vp_, wp__, su, sv, sw, bu, bv, bw,
                            u0, v0, w0, ch, cv, cw, eps, rho, dt, h,
                            di0, dj0, dk0, dAi, dAj, dAk, mp,
                            key_u=ku, key_v=kv, key_w=kw, sigma_shifts=ss_)

    monkeypatch.setattr(torch.ops, "lilytorch_kernels", _BoomOps())
    _run_chain_3d(kwarp.streaming_sdf_stag_3d_multi, kwarp.bdim_coeff_3d,
                  sc, up.clone(), vp.clone(), wpv.clone(), 1)
    _run_chain_sigma_3d(kwarp.streaming_sdf_stag_3d_multi, _warp_sigma, sc,
                        up.clone(), vp.clone(), wpv.clone(), ss,
                        {"emit_keys": True})
    adv._solve_convective(up.clone(), vp.clone(), wpv.clone())
    adv.set_BCs(*[b.clone() for b in bcf])
    wp.synchronize()


@pytest.mark.skipif(not CUDA, reason="needs the GPU reference for CPU==GPU")
@pytest.mark.parametrize("sigma", [False, True])
def test_kernelAB_2d_chain_cpu_eq_gpu(sigma):
    """Item 6 (5): the 2-D Kernel A + Kernel B (+σ) chain runs end-to-end on the
    CPU Warp single-source kernels and matches the GPU Warp result — the one
    kernel source serves CPU and GPU (the §F CPU end-to-end payoff)."""
    dtype = torch.float64

    def _scene(dev):
        sc = make_synthetic_scene_2d(48, 32, 2, device=dev, dtype=dtype)
        for k in ("F_flat", "body_meta", "kin", "gx", "gy"):
            sc[k] = sc[k].to(dtype)
        return sc

    torch.manual_seed(5)
    up = torch.randn(50, 34, dtype=dtype)  # (Ngx, Ngy) for Nx=48,Ny=32 → +2 ghost
    vp = torch.randn(50, 34, dtype=dtype)
    ss = torch.tensor([0.03, 0.06], dtype=torch.float32)

    def run(dev):
        sc = _scene(torch.device(dev))
        u = up.to(dev); v = vp.to(dev); s = ss.to(dev)
        if not sigma:
            return _run_chain(kwarp.streaming_sdf_stag_2d_multi,
                              kwarp.bdim_coeff_2d, sc, u.clone(), v.clone(), 1)

        def _ws(up_, vp_, su, sv, bu, bv, u0, v0, ch, cv, ku, kv, ss_,
                eps, rho, dt, h, di0, dj0, dAi, dAj, mp):
            kwarp.bdim_coeff_2d(up_, vp_, su, sv, bu, bv, u0, v0, ch, cv,
                                eps, rho, dt, h, di0, dj0, dAi, dAj, mp,
                                key_u=ku, key_v=kv, sigma_shifts=ss_)
        return _run_chain_sigma_2d(kwarp.streaming_sdf_stag_2d_multi, _ws, sc,
                                   u.clone(), v.clone(), s,
                                   {"emit_keys": True})[:4]

    cpu = run("cpu")
    gpu = run("cuda:0")
    wp.synchronize()
    for nm, c, g in zip(("u0", "v0", "ch", "cv"), cpu, gpu):
        d = (c.cpu() - g.cpu()).abs().max().item()
        assert d < 1e-12, f"cpu vs gpu {nm} maxdiff {d:.2e}"
