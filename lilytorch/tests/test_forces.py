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

# float64 atomic reduction-order noise (per-cell accumulation order differs
# between devices); f32 also carries single-precision / FMA drift.
ATOL_F64 = 1e-9
RTOL_F32 = 3e-4
ATOL_F32 = 1e-5


def _rand_fields(shape, dtype, dev, seed):
    """Build ONCE on CPU then move — torch's per-device generators differ for
    the same seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(shape, generator=g, dtype=dtype).to(dev)
            for _ in range(len(shape) + 2)]


def _fill_union_sdf_2d(sc, dtype, dev):
    """Run the Warp streaming bridge to populate a real union ``sdf_cc``."""
    Ngx, Ngy = sc["Ngx"], sc["Ngy"]
    opt = dict(dtype=dtype, device=dev)
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
    body_update_2d(sc["F_flat"], sc["F_offsets"], sc["body_shapes"],
                   sc["body_meta"], sc["kin"], sc["aabb_lo"], sc["aabb_dim"],
                   sc["gx"], sc["gy"], float(sc["h"]), int(sc["max_vol"]),
                   sdf_cc, sdf_u, sdf_v, bu, bv, kcc, ku, kv,
                   0, di0, dj0, dAi, dAj, nu_, nv_, du_, dv_, 0.0)
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
    if dtype == torch.float32:
        scale = g.abs().max().item()
        assert err <= ATOL_F32 + RTOL_F32 * scale, f"f32 err {err:.3e} scale {scale:.3e}"
    else:
        assert err < ATOL_F64, f"f64 err {err:.3e}"
    # not all-zero (the scene must exercise the band)
    assert g.abs().max().item() > 0, "scene produced no in-band force"


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

def _fill_union_sdf_3d(sc, dtype, dev):
    Ngx, Ngy, Ngz = sc["Ngx"], sc["Ngy"], sc["Ngz"]
    opt = dict(dtype=dtype, device=dev)
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
    body_update_3d(sc["F_flat"], sc["F_offsets"], sc["body_shapes"],
                   sc["body_meta"], sc["kin"], sc["aabb_lo"], sc["aabb_dim"],
                   sc["gx"], sc["gy"], sc["gz"], float(sc["h"]), int(sc["max_vol"]),
                   sdf_cc, sdf_u, sdf_v, sdf_w, bu, bv, bw,
                   kcc, ku, kv, kw, 0,
                   di0, dj0, dk0, dAi, dAj, dAk,
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
    values by ~3e-9 relative (arithmetic-order roundoff, not convergence)."""
    got = _run_python_eulerian("cpu")
    expected = torch.tensor(
        [0.49016188278759837, -0.5369408657638242,
         28.151615096217956, -11.75747282288945], dtype=torch.float64)
    assert torch.allclose(got, expected, rtol=1e-9, atol=1e-11), \
        f"python eulerian force drift: {got.tolist()} vs {expected.tolist()}"


@SKIP_NO_CUDA
def test_python_eulerian_force_path_cpu_eq_gpu():
    """The non-streaming python force path is single-source across devices."""
    c = _run_python_eulerian("cpu")
    g = _run_python_eulerian("cuda")
    assert torch.allclose(c, g, rtol=1e-8, atol=1e-8), \
        f"CPU vs GPU python eulerian force: {c.tolist()} vs {g.tolist()}"
