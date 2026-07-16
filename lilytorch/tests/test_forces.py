"""Native Eulerian force readout single-source checks: CPU twin == CUDA kernel.

Exercises ``streaming_sdf_forces_post_{2,3}d`` (``forces.py``) on the
synthetic flat-table scenes (``scene_2d.make_synthetic_scene_2d`` /
``scene_3d.make_synthetic_scene``).  The union ``sdf_cc`` is populated
by the native streaming bridge first (so the band / union-normal paths are
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
    streaming_sdf_forces_post_2d,
    streaming_sdf_forces_post_3d,
)
from lilytorch.tests.scene_2d import make_synthetic_scene_2d
from lilytorch.tests.scene_3d import make_synthetic_scene

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(),
                                  reason="needs CUDA for the GPU half")

# The CPU/CUDA streaming SDF agrees to ~1e-12 (f64).  delta_order=1 force
# tests pass at ATOL=1e-12; delta_order=2 amplifies residual ~1e-12 SDF
# differences through the delta-function gradient, producing ~1e-8 force
# errors on the 3-D path.
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
    """Run the native streaming bridge to populate a real union ``sdf_cc``.

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
    body_update_2d(sc["F_flat"], sc["F_offsets"], sc["body_shapes"],
                   sc["body_meta"], kin, aabb_lo, aabb_dim,
                   sc["gx"], sc["gy"], float(sc["h"]), max_vol,
                   sdf_cc, sdf_u, sdf_v, bu, bv,
                   0, 0, 0, Ngx, Ngy)
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
    streaming_sdf_forces_post_2d(
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
    body_update_3d(sc["F_flat"], sc["F_offsets"], sc["body_shapes"],
                   sc["body_meta"], kin, aabb_lo, aabb_dim,
                   sc["gx"], sc["gy"], sc["gz"], float(sc["h"]), max_vol,
                   sdf_cc, sdf_u, sdf_v, sdf_w, bu, bv, bw,
                   0, 0, 0, 0, Ngx, Ngy, Ngz)
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
    streaming_sdf_forces_post_3d(
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
# forces.py) is NOT dead: it is the general fallback taken whenever the native
# streaming buffers (``comp._kernel_step`` / ``_kernel_static_2d``) are absent —
# i.e. any direct ``FluidSolver`` (no BDIMhandler), analytical composite bodies,
# and the drag/lift validation benchmarks.  The 3c dedup only removed the dead
# ``_forces_lagrangian_*_python_ref`` oracles; this locks the surviving path
# end-to-end (finite, deterministic on CPU, and CPU == GPU parity).

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
    # confirms we exercised the python branch, not the native streaming readout
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

    The force *readout* path is stable; the frozen values track the pressure it
    reads, so they have been re-frozen whenever the Poisson driver changed
    (arithmetic-order roundoff, not convergence — each shift was ~1e-9
    relative).

    Re-frozen when the CPU Poisson moved onto the native
    ``poisson_solve_multigrid_*`` C++ driver (the same one CUDA already used):
    the pressure it reads shifts by ~2e-8 relative.  That unification also made
    ``test_python_eulerian_force_path_cpu_eq_gpu`` pass, which had been failing
    for as long as CPU and CUDA ran different V-cycles.

    NOT re-frozen when the Poisson gauge moved to an interior-only mean (which
    shifts this config's p by ~13): a closed-surface ∮p·n integral is
    gauge-invariant, and these values did not move at rtol 1e-9.  That is the
    intended check — if they HAD moved, something would be reading p absolutely.
    """
    got = _run_python_eulerian("cpu")
    expected = torch.tensor(
        [0.4901618826264682, -0.5369408657033692,
         28.151615122937677, -11.757472849432448], dtype=torch.float64)
    assert torch.allclose(got, expected, rtol=1e-9, atol=1e-11), \
        f"python eulerian force drift: {got.tolist()} vs {expected.tolist()}"


@SKIP_NO_CUDA
def test_python_eulerian_force_path_cpu_eq_gpu():
    """The non-streaming python force path is single-source across devices."""
    c = _run_python_eulerian("cpu")
    g = _run_python_eulerian("cuda")
    assert torch.allclose(c, g, rtol=1e-8, atol=1e-8), \
        f"CPU vs GPU python eulerian force: {c.tolist()} vs {g.tolist()}"


# ── Streaming force readout: the ForcesPostGraph wrapper == the raw op ───────
# Drives the ForcesPostGraph readout over a multi-step "simulation": per-step
# FRESH kin/aabb tensors (as BDIMhandler produces), moving poses, in-place
# fluid-field updates, and a mid-run max_vol growth that exercises the grow-only
# watermark.  The reference calls the raw native op on identical data, so this
# gates the staging / watermark bookkeeping around the kernel, not the kernel.
# Agreement is atomic-accumulation-order roundoff only, so the tolerance is
# fp-width-aware rather than bit-exact.

# atomic-order divergence: f64 ~3e-9, f32 ~2e-7 (measured);
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
        streaming_sdf_forces_post_2d(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
            kin, aabb_lo, aabb_dim, sc["gx"], sc["gy"],
            h, max_vol, sdf_cc, 0, u, v, p, nrho,
            eps_body, eps_solver, h2, 1, out_e,
            force_submethod=submethod, ph_tau=ph_tau)

        err = (out_g - out_e).abs().max().item()
        atol = _forces_parity_atol(dtype)
        assert err < atol, f"step {step}: graph wrapper vs raw op err {err:.3e} >= {atol:.1e}"
        assert out_e.abs().max().item() > 0, f"step {step}: no in-band force"
    return fg


@SKIP_NO_CUDA
@pytest.mark.parametrize("submethod", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_forces_2d_graph_replay_eq_eager(dtype, submethod):
    fg = _graph_steps_2d(dtype, submethod)
    # The readout runs eagerly every step; all 8 steps count as eager, and the
    # per-step parity against the raw op is checked inside _graph_steps_2d.
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
        streaming_sdf_forces_post_3d(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
            kin, aabb_lo, aabb_dim, sc["gx"], sc["gy"], sc["gz"],
            h, max_vol, sdf_cc, 0, u, v, w, p, nrho,
            eps_body, eps_solver, h3, 1, out_e,
            force_submethod=submethod, ph_tau=ph_tau)

        err = (out_g - out_e).abs().max().item()
        atol = _forces_parity_atol(dtype)
        assert err < atol, f"step {step}: graph wrapper vs raw op err {err:.3e} >= {atol:.1e}"
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


# ─── physical oracle: absolute accuracy, not parity ─────────────────────────
#
# Everything above this line compares the readout against ITSELF on another
# device or against a frozen snapshot.  Such a test passes just as happily when
# both sides are wrong.  The two cases below impose analytic fields on a sphere
# whose force integral is known in closed form (divergence theorem), so they
# measure each readout's real error.  Companion script with the full R/h × eps
# sweep: validation/force_readout_oracle/oracle_native_three_way.py.
#
#   A (pressure):  p = -G·x,     u = 0   ->  F_p = G·V
#   B (viscous):   u_x = c·y²,   p = 0   ->  F_v = 2·ν·ρ·c·V   (divergence-free)
#
# The sphere SDF is an exact distance function (|∇φ| = 1), which keeps
# force_delta_order out of the picture entirely.

_ORC_R = 0.2                       # sphere radius
_ORC_C = 0.5                       # sphere centre (all axes)
_ORC_NU, _ORC_RHO = 1.0e-2, 1000.0
_ORC_G = 3.0                       # pressure gradient, case A
_ORC_CSH = 2.0                     # shear coefficient, case B
_ORC_V = (4.0 / 3.0) * 3.141592653589793 * _ORC_R ** 3


def _oracle_scene(N, eps_mult):
    """Single exact sphere on an N³ unit-cube grid, hand-built for the native op."""
    h = 1.0 / (N - 1)
    g = torch.linspace(0.0, (N - 1) * h, N, dtype=torch.float64)

    M, half = 161, 0.4                      # body SDF table (fine local grid)
    bl = torch.linspace(-half, half, M, dtype=torch.float64)
    BX, BY, BZ = torch.meshgrid(bl, bl, bl, indexing="ij")
    F_flat = (torch.sqrt(BX**2 + BY**2 + BZ**2) - _ORC_R).ravel().contiguous()
    inv_d = 1.0 / float(bl[1] - bl[0])
    body_meta = torch.tensor(
        [[float(bl[0])] * 3 + [float(bl[-1])] * 3 + [inv_d] * 3 + [inv_d**3]],
        dtype=torch.float64)

    X, Y, Z = torch.meshgrid(g, g, g, indexing="ij")
    sdf_cc = (torch.sqrt((X - _ORC_C)**2 + (Y - _ORC_C)**2 + (Z - _ORC_C)**2)
              - _ORC_R).ravel().contiguous()

    return dict(
        h=h, g=g, X=X, Y=Y, N=N, eps=eps_mult * h,
        F_flat=F_flat, F_offsets=torch.tensor([0], dtype=torch.int64),
        body_shapes=torch.tensor([[M, M, M]], dtype=torch.int64),
        body_meta=body_meta, sdf_cc=sdf_cc,
        # kin: R_T(9) identity + body_pos(3) + com(3) + lin_vel(3) + ang_vel(3)
        kin=torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0, 1,
                           _ORC_C, _ORC_C, _ORC_C, _ORC_C, _ORC_C, _ORC_C,
                           0, 0, 0, 0, 0, 0]], dtype=torch.float64),
        aabb_lo=torch.zeros((1, 3), dtype=torch.int64),
        aabb_dim=torch.tensor([[N, N, N]], dtype=torch.int64),
    )


def _oracle_eulerian(sc, submethod, u, v, w, p):
    """-> (F_visc_x, F_pres_x) from the native eulerian readout."""
    out = torch.zeros((1, 12), dtype=torch.float64)
    streaming_sdf_forces_post_3d(
        sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
        sc["kin"], sc["aabb_lo"], sc["aabb_dim"],
        sc["g"], sc["g"], sc["g"], sc["h"], sc["N"] ** 3,
        sc["sdf_cc"], 0,
        u.ravel().contiguous(), v.ravel().contiguous(),
        w.ravel().contiguous(), p.ravel().contiguous(),
        torch.tensor([_ORC_NU * _ORC_RHO], dtype=torch.float64),
        sc["eps"], sc["eps"], sc["h"] ** 3, 1, out, submethod, 1.5 * sc["h"],
    )
    return float(out[0, 0]), float(out[0, 6])


def _oracle_lagrangian(sc, u, v, w, p):
    """-> (F_visc_x, F_pres_x) from the lagrangian readout, same fields."""
    from lilytorch.src.forces import _viscous_stress_tensor
    from lilytorch.tests.test_lagrangian import _build_sphere_tris

    tc, tn, ta = _build_sphere_tris(_ORC_C, _ORC_C, _ORC_C, _ORC_R, 160, 90)
    e = _viscous_stress_tensor((u, v, w), sc["h"])
    out = torch.zeros((1, 12), dtype=torch.float64)
    torch.ops.lilytorch_kernels.lagrangian_forces_3d.default(
        e[0][0].contiguous(), e[1][1].contiguous(), e[2][2].contiguous(),
        e[0][1].contiguous(), e[0][2].contiguous(), e[1][2].contiguous(),
        p.contiguous(), torch.tensor([_ORC_NU * _ORC_RHO], dtype=torch.float64),
        tc, tn, ta, torch.tensor([0, tc.shape[1]], dtype=torch.int64),
        torch.tensor([[_ORC_C, _ORC_C, _ORC_C]], dtype=torch.float64),
        float(sc["g"][0]), float(sc["g"][0]), float(sc["g"][0]),
        1.0 / sc["h"], 1.0 / sc["h"], 1.0 / sc["h"],
        sc["N"], sc["N"], sc["N"], 0, 0.0, out,
    )
    return float(out[0, 0]), float(out[0, 6])


@pytest.mark.parametrize("submethod", [0, 1])
def test_oracle_eulerian_pressure_matches_exact(submethod):
    """Case A: the eulerian pressure readout is accurate — both submethods."""
    sc = _oracle_scene(48, 1.0)
    z = torch.zeros_like(sc["X"])
    _, fp = _oracle_eulerian(sc, submethod, z, z.clone(), z.clone(),
                             -_ORC_G * sc["X"])
    exact = _ORC_G * _ORC_V
    assert fp == pytest.approx(exact, rel=0.02), \
        f"submethod {submethod}: F_p {fp:.6e} vs exact {exact:.6e}"


def test_oracle_lagrangian_matches_exact():
    """The lagrangian formula is exact in both channels (this is the yardstick
    the eulerian offsets below are measured against)."""
    sc = _oracle_scene(48, 1.0)
    z = torch.zeros_like(sc["X"])
    _, fp = _oracle_lagrangian(sc, z, z.clone(), z.clone(), -_ORC_G * sc["X"])
    fv, _ = _oracle_lagrangian(sc, _ORC_CSH * sc["Y"] ** 2, z.clone(),
                               z.clone(), z.clone())
    assert fp == pytest.approx(_ORC_G * _ORC_V, rel=0.01)
    assert fv == pytest.approx(2 * _ORC_NU * _ORC_RHO * _ORC_CSH * _ORC_V,
                               rel=0.01)


@pytest.mark.parametrize("submethod", [0, 1])
def test_oracle_eulerian_viscous_reads_offset_surface(submethod):
    """PINS A KNOWN DEFECT — this assertion is meant to be flipped by a fix.

    The eulerian viscous band is shifted out by one eps (streaming_sdf.cu:404,
    streaming_sdf_cpu.cpp:697, forces.py:143) to escape the BDIM band, so it
    integrates σ·n over {φ = eps} rather than the body surface, with no
    correction for the offset.  The resulting error is O(eps·∂σ/∂n): its sign is
    FLOW-DEPENDENT, not a universal multiplicative law.  Here the imposed field
    u_x = c·y² has stress growing away from the wall, so the readout over-reads,
    and (this field being quadratic) it lands on the enclosed-volume ratio
    ((R+eps)/R)³.  In a real boundary layer, shear DECAYS off the wall and the
    same mechanism under-reads instead — measured ~11% low on cylinder_drag_2d
    at Re=550 (see milestones/force_readout_agreement_handoff.md §9.1).

    When the readout is corrected to φ=0, this test should assert
    ``fv ≈ exact`` like the lagrangian one above.
    """
    sc = _oracle_scene(48, 1.0)
    z = torch.zeros_like(sc["X"])
    fv, _ = _oracle_eulerian(sc, submethod, _ORC_CSH * sc["Y"] ** 2, z.clone(),
                             z.clone(), z.clone())
    exact = 2 * _ORC_NU * _ORC_RHO * _ORC_CSH * _ORC_V
    predicted = exact * ((_ORC_R + sc["eps"]) / _ORC_R) ** 3
    assert fv == pytest.approx(predicted, rel=0.02), \
        f"offset-surface model broke: F_v {fv:.6e}, model {predicted:.6e}"
    assert fv > 1.3 * exact, "over-read vanished — did the band shift change?"


def _oracle_lagrangian_offset(sc, u, v, w, p, sample_offset):
    """Lagrangian readout with a non-zero sample offset (σ read off the wall)."""
    from lilytorch.src.forces import _viscous_stress_tensor
    from lilytorch.tests.test_lagrangian import _build_sphere_tris

    tc, tn, ta = _build_sphere_tris(_ORC_C, _ORC_C, _ORC_C, _ORC_R, 160, 90)
    e = _viscous_stress_tensor((u, v, w), sc["h"])
    out = torch.zeros((1, 12), dtype=torch.float64)
    torch.ops.lilytorch_kernels.lagrangian_forces_3d.default(
        e[0][0].contiguous(), e[1][1].contiguous(), e[2][2].contiguous(),
        e[0][1].contiguous(), e[0][2].contiguous(), e[1][2].contiguous(),
        p.contiguous(), torch.tensor([_ORC_NU * _ORC_RHO], dtype=torch.float64),
        tc, tn, ta, torch.tensor([0, tc.shape[1]], dtype=torch.int64),
        torch.tensor([[_ORC_C, _ORC_C, _ORC_C]], dtype=torch.float64),
        float(sc["g"][0]), float(sc["g"][0]), float(sc["g"][0]),
        1.0 / sc["h"], 1.0 / sc["h"], 1.0 / sc["h"],
        sc["N"], sc["N"], sc["N"], 0, float(sample_offset), out,
    )
    return float(out[0, 0])


def _oracle_eulerian_shift(sc, u, v, w, p, eps_solver):
    """Eulerian readout at an explicit band shift (eps_body held fixed)."""
    out = torch.zeros((1, 12), dtype=torch.float64)
    streaming_sdf_forces_post_3d(
        sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
        sc["kin"], sc["aabb_lo"], sc["aabb_dim"],
        sc["g"], sc["g"], sc["g"], sc["h"], sc["N"] ** 3,
        sc["sdf_cc"], 0,
        u.ravel().contiguous(), v.ravel().contiguous(),
        w.ravel().contiguous(), p.ravel().contiguous(),
        torch.tensor([_ORC_NU * _ORC_RHO], dtype=torch.float64),
        sc["eps"], float(eps_solver), sc["h"] ** 3, 1, out, 0, 1.5 * sc["h"],
    )
    return float(out[0, 0])


def test_oracle_readouts_are_not_the_same_device_at_matched_offset():
    """The two readouts do NOT agree by matching eps_solver to the lagrangian
    sample offset — they are structurally different integrals:

        eulerian(s)   = ∮_{φ=s} σ·n dS          -- the OFFSET ISO-SURFACE,
                                                   whose measure inflates with s
        lagrangian(o) = ∮_{S_body} σ(x+o·n)·n dS -- the TRUE surface (fixed
                                                   area), σ merely SAMPLED at o

    So they coincide only at zero, and raising both knobs together makes them
    diverge, not converge.  Pins the two facts that follow (handoff §10.3):
    they agree at s=0, and the eulerian's growth IS the volume inflation.
    """
    sc = _oracle_scene(64, 1.0)
    h = sc["h"]
    z = torch.zeros_like(sc["X"])
    u = _ORC_CSH * sc["Y"] ** 2
    args = (u, z.clone(), z.clone(), z.clone())

    # At zero offset both reduce to the true surface integral -> they agree.
    e0 = _oracle_eulerian_shift(sc, *args, 0.0)
    l0 = _oracle_lagrangian_offset(sc, *args, 0.0)
    assert e0 == pytest.approx(l0, rel=0.02), \
        f"readouts should coincide at zero offset: {e0:.6e} vs {l0:.6e}"

    # Matched at s = 2h they do not: the eulerian has inflated by the enclosed
    # volume ratio while the lagrangian's surface measure has not moved.
    e2 = _oracle_eulerian_shift(sc, *args, 2.0 * h)
    l2 = _oracle_lagrangian_offset(sc, *args, 2.0 * h)
    assert e2 / l2 > 1.25, \
        f"expected the readouts to DIVERGE at matched offset, got {e2/l2:.3f}"
    assert e2 / e0 == pytest.approx(((_ORC_R + 2 * h) / _ORC_R) ** 3, rel=0.03), \
        "eulerian growth should track the enclosed-volume ratio"


def test_oracle_eulerian_thinness_error_at_zero_shift():
    """Even with NO band shift, the eulerian over-reads on a body comparable to
    the band half-width — and the error vanishes as R/h grows.

    This isolates pure geometric thinness: exact analytic sphere (|∇φ| = 1, so
    no coarea factor in play), a single closed triangulation (no joints),
    uniform ∇·σ, and s = 0 (so no offset-iso-surface inflation).  At the
    zebrafish's thinness (R/h ≈ 3, eps_body = 2h) this alone is worth ~1.17x,
    one of the two factors behind the fish's 1.551x s=0 gap (handoff §10.3c).

    The lagrangian is exact at every R/h — it does not care about thinness.
    """
    z_ = None
    ratios = {}
    for N, key in ((16, "thin"), (64, "fat")):
        sc = _oracle_scene(N, 2.0)          # eps_body = 2h, as the fish runs
        z = torch.zeros_like(sc["X"])
        args = (_ORC_CSH * sc["Y"] ** 2, z.clone(), z.clone(), z.clone())
        e0 = _oracle_eulerian_shift(sc, *args, 0.0)
        l0 = _oracle_lagrangian_offset(sc, *args, 0.0)
        exact = 2 * _ORC_NU * _ORC_RHO * _ORC_CSH * _ORC_V
        assert l0 == pytest.approx(exact, rel=0.02), \
            f"lagrangian should be exact regardless of thinness (N={N})"
        ratios[key] = e0 / l0
    # R/h ~ 3: a real over-read; R/h ~ 12.6: essentially gone.
    assert ratios["thin"] > 1.10, f"expected thinness over-read, got {ratios['thin']:.3f}"
    assert ratios["fat"] < 1.03, f"should converge with R/h, got {ratios['fat']:.3f}"
    assert ratios["thin"] > ratios["fat"]


def test_delta_order2_is_inverted_vs_coarea():
    """PINS A KNOWN BUG (handoff §8, promoted to §10.3c) — flip this on fix.

    The coarea identity ∮_{φ=0} g dS = ∫ δ(φ) g |∇φ| dV needs a MULTIPLY by
    |∇φ|; ``solver.py`` says delta_order=2 "divides by |∇SDF| so that the volume
    integral gives the correct surface measure", and the code does divide.  So
    order 2 doubles the |∇φ| error instead of removing it.

    Inert at |∇φ| = 1 (which is why analytical-body tests never caught it), so
    this drives a DELIBERATELY SCALED SDF, φ = g·(r−R), where the true force is
    unchanged but |∇φ| = g.  Correct behaviour would be order2 ≈ exact for any
    g; as coded it lands at exact/g² instead.  On the zebrafish's real mesh SDF
    (|∇φ| ≈ 0.82) this is worth ~1.22x of live over-read.
    """
    exact = 2 * _ORC_NU * _ORC_RHO * _ORC_CSH * _ORC_V
    for g in (0.5, 2.0):
        sc = _oracle_scene(64, 1.0)
        # rescale BOTH the union SDF and the body table -> |∇φ| = g, same body
        sc["sdf_cc"] = sc["sdf_cc"] * g
        sc["F_flat"] = sc["F_flat"] * g
        sc["eps"] = sc["eps"] * g          # band must scale with the SDF units
        z = torch.zeros_like(sc["X"])
        args = (_ORC_CSH * sc["Y"] ** 2, z.clone(), z.clone(), z.clone())
        f1 = _oracle_eulerian_shift(sc, *args, 0.0)
        out = torch.zeros((1, 12), dtype=torch.float64)
        streaming_sdf_forces_post_3d(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
            sc["kin"], sc["aabb_lo"], sc["aabb_dim"],
            sc["g"], sc["g"], sc["g"], sc["h"], sc["N"] ** 3,
            sc["sdf_cc"], 0,
            args[0].ravel().contiguous(), args[1].ravel().contiguous(),
            args[2].ravel().contiguous(), args[3].ravel().contiguous(),
            torch.tensor([_ORC_NU * _ORC_RHO], dtype=torch.float64),
            sc["eps"], 0.0, sc["h"] ** 3, 2, out, 0, 1.5 * sc["h"],
        )
        f2 = float(out[0, 0])
        # order 2 as coded = order 1 / |∇φ|; the coarea fix would MULTIPLY.
        assert f2 == pytest.approx(f1 / g, rel=0.05), \
            f"|∇φ|={g}: order2 {f2:.5e} is not order1/|∇φ| ({f1/g:.5e})"
        # ...so it moves AWAY from the exact answer that order1*|∇φ| recovers.
        assert f1 * g == pytest.approx(exact, rel=0.06), \
            f"|∇φ|={g}: order1*|∇φ| should recover exact, got {f1*g:.5e}"


def test_oracle_deltaH_viscous_is_ndelta_viscous():
    """deltaH only replaces the PRESSURE readout, so it cannot be a candidate
    fix for the viscous offset above: the two viscous outputs are bit-identical.
    """
    sc = _oracle_scene(48, 1.0)
    z = torch.zeros_like(sc["X"])
    u = _ORC_CSH * sc["Y"] ** 2
    fv0, _ = _oracle_eulerian(sc, 0, u, z.clone(), z.clone(), z.clone())
    fv1, _ = _oracle_eulerian(sc, 1, u, z.clone(), z.clone(), z.clone())
    assert fv0 == fv1
