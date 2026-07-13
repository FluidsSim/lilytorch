"""Warp Eulerian force readout single-source checks: Warp CPU == Warp GPU.

Exercises ``streaming_sdf_forces_post_{2,3}d_warp`` (``forces.py``) on the
synthetic flat-table scenes (``scene_2d.make_synthetic_scene_2d`` /
``bench_viability.make_synthetic_scene``).  The union ``sdf_cc`` is populated
by the Warp streaming bridge first (so the band / union-normal paths are
exercised on a real union SDF), then the force readout consumes the same
fields on both devices.

Covers: ``force_submethod`` 0 (n·δ) and 1 (deltaH ∂H pressure), ``delta_order``
1 and 2, scalar + full-field nu_rho, f32 + f64.

Run:  pytest lilytorch/src/kernels/test_forces.py -v
"""
from __future__ import annotations

import pytest
import torch

from lilytorch.src.facade import body_update_2d, body_update_3d
from lilytorch.src.forces import (
    streaming_sdf_forces_post_2d_warp,
    streaming_sdf_forces_post_3d_warp,
)
from lilytorch.tests.scene_2d import make_synthetic_scene_2d
from lilytorch.benchmarks.bench_viability import make_synthetic_scene

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(),
                                  reason="needs CUDA for the GPU half")

# Native CPU/CUDA streaming SDF now agrees to ~1e-12 (f64) after the
# sampler accumulation-order fix.  delta_order=1 force tests pass at
# ATOL=1e-12; delta_order=2 amplifies residual ~1e-12 SDF differences
# through the delta-function gradient, producing ~1e-8 force errors on
# the 3-D path (a Warp force-kernel runtime artifact, not a native bug).
ATOL_F64 = 1e-12
RTOL_F64 = 1e-8
RTOL_F32 = 3e-4
ATOL_F32 = 1e-5


def _rand_fields(shape, dtype, dev, seed):
    """Build ONCE on CPU then move — torch's per-device generators differ for
    the same seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(shape, generator=g, dtype=dtype).to(dev)
            for _ in range(len(shape) + 2)]


def _fill_union_sdf_2d(sc, dtype, dev, kin=None, aabb_lo=None, aabb_dim=None,
                       max_vol=None, out=None):
    """Run the Warp streaming bridge to populate a real union ``sdf_cc``.

    ``kin``/``aabb_*``/``max_vol`` default to the scene's; pass the current
    step's values (and a persistent ``out``) to refresh the union in place,
    exactly as BDIMhandler does every step — the deltaH readout requires
    ``sdf_cc`` and the AABBs to be CONSISTENT (a grown AABB over a stale
    union puts FAR cells inside the softmin partition → exp overflow)."""
    Ngx, Ngy = sc["Ngx"], sc["Ngy"]
    opt = dict(dtype=dtype, device=dev)
    kin = sc["kin"] if kin is None else kin
    aabb_lo = sc["aabb_lo"] if aabb_lo is None else aabb_lo
    aabb_dim = sc["aabb_dim"] if aabb_dim is None else aabb_dim
    max_vol = int(sc["max_vol"]) if max_vol is None else int(max_vol)
    sdf_cc = torch.empty((Ngx, Ngy), **opt) if out is None else out
    sdf_cc.fill_(1e4)
    sdf_u = torch.full((Ngx, Ngy), 1e4, **opt)
    sdf_v = torch.full((Ngx, Ngy), 1e4, **opt)
    bu = torch.zeros((Ngx, Ngy), **opt)
    bv = torch.zeros((Ngx, Ngy), **opt)
    nu_ = torch.empty(1, **opt); nv_ = torch.empty(1, **opt)
    du_ = torch.empty(1, **opt); dv_ = torch.empty(1, **opt)
    body_update_2d(sc["F_flat"], sc["F_offsets"], sc["body_shapes"],
                   sc["body_meta"], kin, aabb_lo, aabb_dim,
                   sc["gx"], sc["gy"], float(sc["h"]), max_vol,
                   sdf_cc, sdf_u, sdf_v, bu, bv,
                   0, 0, 0, Ngx, Ngy, nu_, nv_, du_, dv_, 0.0)
    return sdf_cc


def _run_2d(dev, dtype, submethod, delta_order, scalar_nrho):
    sc = make_synthetic_scene_2d(96, 64, 3, device=dev, dtype=dtype)
    for k in ("F_flat", "body_meta", "kin", "gx", "gy"):
        sc[k] = sc[k].to(dtype)
    Ngx, Ngy, h = sc["Ngx"], sc["Ngy"], float(sc["h"])
    sdf_cc = _fill_union_sdf_2d(sc, dtype, dev)

    u, v, p, nrho_f = _rand_fields((Ngx, Ngy), dtype, dev, seed=7)
    if scalar_nrho:
        nrho = torch.tensor([0.13], dtype=dtype, device=dev)
    else:
        nrho = nrho_f.abs() + 0.05

    eps_body = 2.0 * h
    eps_solver = 0.0
    h2 = h * h
    B = sc["aabb_dim"].shape[0]
    ph_tau = 0.5 * h if submethod else 0.0

    out_w = torch.zeros(B, 6, dtype=torch.float64, device=dev)
    streaming_sdf_forces_post_2d_warp(
        sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
        sc["kin"], sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"],
        h, int(sc["max_vol"]), sdf_cc, 0, u, v, p, nrho,
        eps_body, eps_solver, h2, delta_order, out_w,
        force_submethod=submethod, ph_tau=ph_tau)
    return out_w.cpu()


def _check(g, c, dtype):
    err = (g - c).abs().max().item()
    scale = g.abs().max().item()
    if dtype == torch.float32:
        assert err <= ATOL_F32 + RTOL_F32 * scale, f"f32 err {err:.3e} scale {scale:.3e}"
    else:
        assert err <= ATOL_F64 + RTOL_F64 * scale, f"f64 err {err:.3e} scale {scale:.3e}"
    # not all-zero (the scene must exercise the band)
    assert scale > 0, "scene produced no in-band force"


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("delta_order", [1, 2])
@pytest.mark.parametrize("scalar_nrho", [True, False])
def test_forces_2d_ndelta_cpu_eq_gpu(dtype, delta_order, scalar_nrho):
    g = _run_2d("cuda", dtype, 0, delta_order, scalar_nrho)
    c = _run_2d("cpu", dtype, 0, delta_order, scalar_nrho)
    _check(g, c, dtype)


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("delta_order", [1, 2])
def test_forces_2d_deltaH_cpu_eq_gpu(dtype, delta_order):
    g = _run_2d("cuda", dtype, 1, delta_order, False)
    c = _run_2d("cpu", dtype, 1, delta_order, False)
    _check(g, c, dtype)


# ─── 3-D ─────────────────────────────────────────────────────────────────────

def _fill_union_sdf_3d(sc, dtype, dev, kin=None, aabb_lo=None, aabb_dim=None,
                       max_vol=None, out=None):
    """3-D twin of :func:`_fill_union_sdf_2d` (same refresh semantics)."""
    Ngx, Ngy, Ngz = sc["Ngx"], sc["Ngy"], sc["Ngz"]
    opt = dict(dtype=dtype, device=dev)
    kin = sc["kin"] if kin is None else kin
    aabb_lo = sc["aabb_lo"] if aabb_lo is None else aabb_lo
    aabb_dim = sc["aabb_dim"] if aabb_dim is None else aabb_dim
    max_vol = int(sc["max_vol"]) if max_vol is None else int(max_vol)
    sdf_cc = torch.empty((Ngx, Ngy, Ngz), **opt) if out is None else out
    sdf_cc.fill_(1e4)
    sdf_u = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    sdf_v = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    sdf_w = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    bu = torch.zeros((Ngx, Ngy, Ngz), **opt)
    bv = torch.zeros((Ngx, Ngy, Ngz), **opt)
    bw = torch.zeros((Ngx, Ngy, Ngz), **opt)
    nu_ = torch.empty(1, **opt); nv_ = torch.empty(1, **opt); nw_ = torch.empty(1, **opt)
    du_ = torch.empty(1, **opt); dv_ = torch.empty(1, **opt); dw_ = torch.empty(1, **opt)
    body_update_3d(sc["F_flat"], sc["F_offsets"], sc["body_shapes"],
                   sc["body_meta"], kin, aabb_lo, aabb_dim,
                   sc["gx"], sc["gy"], sc["gz"], float(sc["h"]), max_vol,
                   sdf_cc, sdf_u, sdf_v, sdf_w, bu, bv, bw,
                   0, 0, 0, 0, Ngx, Ngy, Ngz,
                   nu_, nv_, nw_, du_, dv_, dw_, 0.0)
    return sdf_cc


def _run_3d(dev, dtype, submethod, delta_order, scalar_nrho):
    sc = make_synthetic_scene(48, 32, 32, 3, device=dev, dtype=dtype)
    for kk in ("F_flat", "body_meta", "kin", "gx", "gy", "gz"):
        sc[kk] = sc[kk].to(dtype)
    Ngx, Ngy, Ngz, h = sc["Ngx"], sc["Ngy"], sc["Ngz"], float(sc["h"])
    sdf_cc = _fill_union_sdf_3d(sc, dtype, dev)

    u, v, w, p, nrho_f = _rand_fields((Ngx, Ngy, Ngz), dtype, dev, seed=11)
    if scalar_nrho:
        nrho = torch.tensor([0.11], dtype=dtype, device=dev)
    else:
        nrho = nrho_f.abs() + 0.05

    eps_body = 2.0 * h
    eps_solver = 0.0
    h3 = h * h * h
    B = sc["aabb_dim"].shape[0]
    ph_tau = 0.5 * h if submethod else 0.0

    out_w = torch.zeros(B, 12, dtype=torch.float64, device=dev)
    streaming_sdf_forces_post_3d_warp(
        sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
        sc["kin"], sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["gz"],
        h, int(sc["max_vol"]), sdf_cc, 0, u, v, w, p, nrho,
        eps_body, eps_solver, h3, delta_order, out_w,
        force_submethod=submethod, ph_tau=ph_tau)
    return out_w.cpu()


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("delta_order", [1, 2])
@pytest.mark.parametrize("scalar_nrho", [True, False])
def test_forces_3d_ndelta_cpu_eq_gpu(dtype, delta_order, scalar_nrho):
    g = _run_3d("cuda", dtype, 0, delta_order, scalar_nrho)
    c = _run_3d("cpu", dtype, 0, delta_order, scalar_nrho)
    _check(g, c, dtype)


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("delta_order", [1, 2])
def test_forces_3d_deltaH_cpu_eq_gpu(dtype, delta_order):
    g = _run_3d("cuda", dtype, 1, delta_order, False)
    c = _run_3d("cpu", dtype, 1, delta_order, False)
    _check(g, c, dtype)


# ── Live NON-streaming python force path (forces_method2 python branch) ───────
# The torch-tensor force path (``_forces_shared`` / ``_forces_body_batch`` in
# forces.py) is NOT dead: it is the general fallback taken whenever the Warp
# streaming buffers (``comp._kernel_step`` / ``_kernel_static_2d``) are absent —
# i.e. any direct ``FluidSolver`` (no BDIMhandler), analytical composite bodies,
# and the drag/lift validation benchmarks.  The 3c dedup only removed the dead
# ``_forces_lagrangian_*_python_ref`` oracles; this locks the surviving path
# end-to-end (finite, deterministic on CPU, and Warp-CPU == Warp-GPU parity).

def _run_python_eulerian(device):
    from lilytorch.tests.test_two_phase import (
        _parity_pars, _taylor_green_ic, _set_ic, _step_n)
    from lilytorch.src.solver import FluidSolver
    body = ["lambda x, y: circle(x,y,xt=0.5,yt=0.5,r=0.12)"]
    pars = _parity_pars(2, 48, 2.0e-3, 1.0e-2, 1000.0, body)
    pars["solver"]["use_gpu"] = (device == "cuda")
    pars["solver"]["force_method"] = "eulerian"
    sp = FluidSolver(pars, dtype=torch.float64, compute_forces=True)
    _set_ic(sp, _taylor_green_ic(sp))
    _step_n(sp, 5)
    # confirms we exercised the python branch, not the Warp streaming readout
    assert getattr(sp.composite_body, "_kernel_step", None) is None
    return torch.tensor([
        float(sp.friction_force_lin_x.reshape(-1)[0]),
        float(sp.friction_force_lin_y.reshape(-1)[0]),
        float(sp.pressure_force_x.reshape(-1)[0]),
        float(sp.pressure_force_y.reshape(-1)[0]),
    ], dtype=torch.float64)


def test_python_eulerian_force_path_cpu_regression():
    """Frozen CPU snapshot of the non-streaming python eulerian force readout
    (float64 is deterministic).  Guards the load-bearing torch-tensor path that
    has no other unit coverage.

    Re-frozen when the Poisson driver unified onto the single WarpMG V-cycle:
    the force *readout* path is unchanged, but the pressure it reads now comes
    from WarpMG on CPU instead of the retired hybrid torch V-cycle, shifting the
    values by ~3e-9 relative (arithmetic-order roundoff, not convergence).

    Re-frozen again when ``solve_multigrid`` became adaptive (early-exit at
    ``tol`` between one-v-cycle graph replays, restoring the retired native
    driver's convergence semantics): the solve stops after fewer V-cycles once
    converged, shifting the pressure — and these readouts — by ~7e-9 relative."""
    got = _run_python_eulerian("cpu")
    expected = torch.tensor(
        [0.4901618791749159, -0.5369408627991937,
         28.151615663974535, -11.757473401771366], dtype=torch.float64)
    assert torch.allclose(got, expected, rtol=1e-9, atol=1e-11), \
        f"python eulerian force drift: {got.tolist()} vs {expected.tolist()}"


@SKIP_NO_CUDA
def test_python_eulerian_force_path_cpu_eq_gpu():
    """The non-streaming python force path is single-source across devices."""
    c = _run_python_eulerian("cpu")
    g = _run_python_eulerian("cuda")
    assert torch.allclose(c, g, rtol=1e-8, atol=1e-8), \
        f"CPU vs GPU python eulerian force: {c.tolist()} vs {g.tolist()}"


# ── Native streaming force readout (ForcesPostGraph) == Warp oracle ──────────
# cuda_native_port Phase 0.2 parity gate.  Drives the ForcesPostGraph readout —
# now the NATIVE CUDA op, run eagerly (the Warp CUDA-graph capture is retired
# here; the native whole-step graph runner is Phase 1) — over a multi-step
# "simulation": per-step FRESH kin/aabb tensors (as BDIMhandler produces),
# moving poses, in-place fluid-field updates, and a mid-run max_vol growth that
# exercises the grow-only watermark.  The reference runs the Warp oracle on
# identical data; agreement is atomic-accumulation-order roundoff only (native
# CUDA vs Warp CUDA), so the tolerance is fp-width-aware rather than bit-exact.

# native-vs-warp atomic-order divergence: f64 ~3e-9, f32 ~2e-7 (measured);
# leave headroom for other scenes / larger force magnitudes.
def _forces_parity_atol(dtype):
    return 1e-6 if dtype == torch.float64 else 1e-3


def _graph_steps_2d(dtype, submethod=0):
    dev = "cuda"
    from lilytorch.src.forces import ForcesPostGraph
    sc = make_synthetic_scene_2d(96, 64, 3, device=dev, dtype=dtype)
    for k in ("F_flat", "body_meta", "kin", "gx", "gy"):
        sc[k] = sc[k].to(dtype)
    Ngx, Ngy, h = sc["Ngx"], sc["Ngy"], float(sc["h"])
    sdf_cc = _fill_union_sdf_2d(sc, dtype, dev)

    u, v, p, _ = _rand_fields((Ngx, Ngy), dtype, dev, seed=7)
    nrho = torch.tensor([0.13], dtype=dtype, device=dev)
    eps_body, eps_solver, h2 = 2.0 * h, 0.0, h * h
    ph_tau = 0.5 * h if submethod else 0.0
    B = sc["aabb_dim"].shape[0]

    fg = ForcesPostGraph(2)
    out_g = torch.zeros(B, 6, dtype=torch.float64, device=dev)
    out_e = torch.zeros(B, 6, dtype=torch.float64, device=dev)
    kin0 = sc["kin"].clone()

    for step in range(8):
        # fresh per-step tensors, drifting pose (mirrors BDIMhandler's pack)
        kin = kin0.clone()
        kin[:, 4:6] += 0.002 * step        # body_pos drift
        aabb_lo = sc["aabb_lo"].clone()
        aabb_dim = sc["aabb_dim"].clone()
        if step >= 5:                       # AABB growth → watermark recapture
            for b in range(B):              # grow every body: the max-vol one too
                room = Ngx - int(aabb_lo[b, 0]) - int(aabb_dim[b, 0])
                aabb_dim[b, 0] += min(8, room)
        max_vol = int(aabb_dim.prod(dim=1).max())
        u.add_(0.01); p.mul_(1.001)         # live-data check
        # Re-stream the union SDF with the CURRENT pose/AABBs into the SAME
        # buffer (as BDIMhandler does every step): the deltaH softmin needs
        # sdf_cc consistent with the AABBs, and the graph needs the pointer
        # stable.
        _fill_union_sdf_2d(sc, dtype, dev, kin=kin, aabb_lo=aabb_lo,
                           aabb_dim=aabb_dim, max_vol=max_vol, out=sdf_cc)

        fg.run(sc["F_flat"], sc["F_offsets"], sc["body_shapes"],
               sc["body_meta"], kin, aabb_lo, aabb_dim,
               (sc["gx"], sc["gy"]), h, max_vol, sdf_cc, 0,
               (u, v), p, nrho, eps_body, eps_solver, h2, 1, out_g,
               force_submethod=submethod, ph_tau=ph_tau)

        out_e.zero_()
        streaming_sdf_forces_post_2d_warp(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
            kin, aabb_lo, aabb_dim, sc["gx"], sc["gy"],
            h, max_vol, sdf_cc, 0, u, v, p, nrho,
            eps_body, eps_solver, h2, 1, out_e,
            force_submethod=submethod, ph_tau=ph_tau)

        err = (out_g - out_e).abs().max().item()
        atol = _forces_parity_atol(dtype)
        assert err < atol, f"step {step}: native vs warp err {err:.3e} >= {atol:.1e}"
        assert out_e.abs().max().item() > 0, f"step {step}: no in-band force"
    return fg


@SKIP_NO_CUDA
@pytest.mark.parametrize("submethod", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_forces_2d_graph_replay_eq_eager(dtype, submethod):
    fg = _graph_steps_2d(dtype, submethod)
    # Phase 0: the readout is native and runs eagerly every step (no Warp
    # capture/replay).  All 8 steps count as eager; the per-step native-vs-warp
    # parity is checked inside _graph_steps_2d.
    assert fg.eager_calls == 8, f"eager_calls={fg.eager_calls}"
    assert fg.captures == 0 and fg.replays == 0


def _graph_steps_3d(dtype, submethod=0):
    dev = "cuda"
    from lilytorch.src.forces import ForcesPostGraph
    sc = make_synthetic_scene(48, 32, 32, 3, device=dev, dtype=dtype)
    for k in ("F_flat", "body_meta", "kin", "gx", "gy", "gz"):
        sc[k] = sc[k].to(dtype)
    Ngx, Ngy, Ngz, h = sc["Ngx"], sc["Ngy"], sc["Ngz"], float(sc["h"])
    sdf_cc = _fill_union_sdf_3d(sc, dtype, dev)

    u, v, w, p, _ = _rand_fields((Ngx, Ngy, Ngz), dtype, dev, seed=11)
    nrho = torch.tensor([0.11], dtype=dtype, device=dev)
    eps_body, eps_solver, h3 = 2.0 * h, 0.0, h * h * h
    ph_tau = 0.5 * h if submethod else 0.0
    B = sc["aabb_dim"].shape[0]

    fg = ForcesPostGraph(3)
    out_g = torch.zeros(B, 12, dtype=torch.float64, device=dev)
    out_e = torch.zeros(B, 12, dtype=torch.float64, device=dev)
    kin0 = sc["kin"].clone()

    for step in range(8):
        kin = kin0.clone()
        kin[:, 9:12] += 0.002 * step
        aabb_lo = sc["aabb_lo"].clone()
        aabb_dim = sc["aabb_dim"].clone()
        if step >= 5:
            for b in range(B):
                room = Ngx - int(aabb_lo[b, 0]) - int(aabb_dim[b, 0])
                aabb_dim[b, 0] += min(4, room)
        max_vol = int(aabb_dim.prod(dim=1).max())
        u.add_(0.01); p.mul_(1.001)
        _fill_union_sdf_3d(sc, dtype, dev, kin=kin, aabb_lo=aabb_lo,
                           aabb_dim=aabb_dim, max_vol=max_vol, out=sdf_cc)

        fg.run(sc["F_flat"], sc["F_offsets"], sc["body_shapes"],
               sc["body_meta"], kin, aabb_lo, aabb_dim,
               (sc["gx"], sc["gy"], sc["gz"]), h, max_vol, sdf_cc, 0,
               (u, v, w), p, nrho, eps_body, eps_solver, h3, 1, out_g,
               force_submethod=submethod, ph_tau=ph_tau)

        out_e.zero_()
        streaming_sdf_forces_post_3d_warp(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
            kin, aabb_lo, aabb_dim, sc["gx"], sc["gy"], sc["gz"],
            h, max_vol, sdf_cc, 0, u, v, w, p, nrho,
            eps_body, eps_solver, h3, 1, out_e,
            force_submethod=submethod, ph_tau=ph_tau)

        err = (out_g - out_e).abs().max().item()
        atol = _forces_parity_atol(dtype)
        assert err < atol, f"step {step}: native vs warp err {err:.3e} >= {atol:.1e}"
        assert out_e.abs().max().item() > 0, f"step {step}: no in-band force"
    return fg


@SKIP_NO_CUDA
@pytest.mark.parametrize("submethod", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_forces_3d_graph_replay_eq_eager(dtype, submethod):
    fg = _graph_steps_3d(dtype, submethod)
    # Phase 0: native readout runs eagerly every step (see the 2-D twin).
    assert fg.eager_calls == 8, f"eager_calls={fg.eager_calls}"
    assert fg.captures == 0 and fg.replays == 0
