"""Advection Warp-kernel tests (single production path).

The fused high-order limiter kernel ``advect_flux_add_warp`` is the *only*
convective path (CPU C++/OpenMP and CUDA).  Three layers:

  (1) CPU regression anchor — a frozen (sum, |max|) snapshot of the rhs the
      kernel produces for a fixed seed, for all five schemes (QUICK /
      ABDQUICKEST / vanLeer / CDS / CUBISTA), 2-D and 3-D.  The snapshot was
      captured when the kernel was still validated bit-for-bit against the
      (now-removed) PyTorch reference, so it pins correctness without CUDA.
  (2) Single-source integrity — the SAME @wp.kernel on CPU == GPU (needs CUDA).
  (3) In-place accumulate semantics (rhs += flux, not overwrite).

Every (velocity component i, direction d) pair is exercised so the rhs-stride
caveat (face_dim ≠ outermost in rhs) is covered for d>0, using the genuine
strided slice views from advection.py.

Run:  pytest lilytorch/tests/test_advection.py -v
"""
from __future__ import annotations

import itertools

import pytest
import torch
import warp as wp

from lilytorch.src.advection import (
    advect_flux_add_warp,
    _face_vel,
    _field_for_flux,
    _inner,
    apply_bcs_2d_warp,
    apply_bcs_3d_warp,
)

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

# scheme_id → name (matches _CUDA_SCHEME_IDS in advection.py)
_SCHEMES = {0: "quick", 1: "abdquickest", 2: "vanLeer", 3: "cds", 4: "cubista"}


def _make_vel(ndim, N, dev, seed=11):
    """Padded MAC velocity field built ONCE on CPU then moved (a device-seeded
    generator yields different sequences per device)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (N + 2,) * ndim
    vel = [torch.rand(shape, generator=g, dtype=torch.float64) - 0.5
           for _ in range(ndim)]
    return [v.to(dev) for v in vel]


def _courant(scheme_id):
    # ABDQUICKEST uses the Courant number; the others ignore it.
    return 0.37 if scheme_id == 1 else 0.0


def _warp(vel, i, d, scheme_id, dt_dh, ndim):
    inner = _inner(ndim)
    fv = _face_vel(vel, i, d, ndim)
    p = _field_for_flux(vel[i], d, ndim)
    rhs = torch.zeros_like(vel[i][inner])
    advect_flux_add_warp(fv, p, rhs, dt_dh, _courant(scheme_id), scheme_id, d)
    wp.synchronize()
    return rhs


# ─────────────────────────────────────────────────────────────────────────────
#  (1) CPU regression anchor  (no CUDA required)
# ─────────────────────────────────────────────────────────────────────────────

# (ndim, scheme_id) → (sum(rhs), max|rhs|), summed over all (i, d) pairs at
# seed 11, dt_dh 0.15, N=40 (2-D) / N=20 (3-D).  Captured against the kernel
# while it was bit-parity with the removed PyTorch scheme reference.
_REGRESSION = {
    (2, 0): (0.05990405142928154, 0.049610416505477574),
    (2, 1): (0.07741790648773922, 0.049610416505477574),
    (2, 2): (0.07369885379628983, 0.049610416505477574),
    (2, 3): (0.047893404100937315, 0.04886254959572121),
    (2, 4): (0.0762824918396289, 0.049610416505477574),
    (3, 0): (1.1405044788006227, 0.060893124887524866),
    (3, 1): (1.0385373002973184, 0.060893124887524866),
    (3, 2): (1.064901094990712, 0.060893124887524866),
    (3, 3): (1.022053726682977, 0.052619659932840825),
    (3, 4): (1.078541541139582, 0.060893124887524866),
}


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("scheme_id", sorted(_SCHEMES))
def test_flux_add_warp_cpu_regression(ndim, scheme_id):
    """Kernel output on CPU matches the frozen validated snapshot."""
    N = 20 if ndim == 3 else 40
    vel = _make_vel(ndim, N, "cpu")
    dt_dh = 0.15
    s, a = 0.0, 0.0
    for i, d in itertools.product(range(ndim), range(ndim)):
        rhs = _warp(vel, i, d, scheme_id, dt_dh, ndim)
        s += float(rhs.sum())
        a = max(a, float(rhs.abs().max()))
    exp_s, exp_a = _REGRESSION[(ndim, scheme_id)]
    assert abs(s - exp_s) < 1e-12, f"{_SCHEMES[scheme_id]} {ndim}-D sum {s!r}"
    assert abs(a - exp_a) < 1e-12, f"{_SCHEMES[scheme_id]} {ndim}-D max {a!r}"


# ─────────────────────────────────────────────────────────────────────────────
#  (2) single-source integrity  +  (3) accumulate semantics
# ─────────────────────────────────────────────────────────────────────────────

@SKIP_NO_CUDA
def test_flux_add_accumulates_in_place():
    """The op must ADD into a pre-seeded rhs (not overwrite), like production."""
    ndim, N, dev = 3, 20, "cuda:0"
    vel = _make_vel(ndim, N, dev, seed=5)
    inner = _inner(ndim)
    g = torch.Generator(device="cpu").manual_seed(99)
    seed_rhs = (torch.rand(vel[0][inner].shape, generator=g, dtype=torch.float64)).to(dev)
    i, d, sid, dt_dh = 0, 2, 0, 0.2  # d != 0 → rhs.stride(face_dim) ≠ outermost
    fv = _face_vel(vel, i, d, ndim)
    p = _field_for_flux(vel[i], d, ndim)

    delta = _warp(vel, i, d, sid, dt_dh, ndim)  # zero-seeded → pure flux term
    got = seed_rhs.clone()
    advect_flux_add_warp(fv, p, got, dt_dh, 0.0, sid, d)
    wp.synchronize()
    # 1-ULP f64 slack: the kernel fuses seed + dt_dh·flux in one expression,
    # while the reference adds them in two rounding steps.
    assert (got - (seed_rhs + delta)).abs().max().item() < 1e-15


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("scheme_id", sorted(_SCHEMES))
def test_flux_add_warp_cpu_equals_gpu(ndim, scheme_id):
    """Single-source check: the SAME @wp.kernel on CPU == GPU."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    N = 20 if ndim == 3 else 40
    dt_dh = 0.15
    vel_cpu = _make_vel(ndim, N, "cpu")
    vel_gpu = [v.to("cuda:0") for v in vel_cpu]
    worst = 0.0
    for i, d in itertools.product(range(ndim), range(ndim)):
        rc = _warp(vel_cpu, i, d, scheme_id, dt_dh, ndim)
        rg = _warp(vel_gpu, i, d, scheme_id, dt_dh, ndim).cpu()
        worst = max(worst, (rc - rg).abs().max().item())
    assert worst < 1e-12, f"{_SCHEMES[scheme_id]} {ndim}-D CPU vs GPU {worst:.3e}"


# ═════════════════════════════════════════════════════════════════════════════
#  FluxAddGraphRunner — CUDA-graph-cached flux launches must equal eager
# ═════════════════════════════════════════════════════════════════════════════

@SKIP_NO_CUDA
@pytest.mark.parametrize("scheme", ["quick", "abdquickest"])
def test_flux_graph_runner_matches_eager(scheme):
    """A multi-step 3-D advection run with the CUDA-graph flux runner must
    reproduce the eager run bit-for-bit (the graph bakes f32 scalars, so replay
    is exact).  This is the guard against the ``b423298`` failure mode where a
    graph "runs but the Warp launches are dropped": such a graph would leave rhs
    unchanged, so the fields would NOT match eager — and the ``evolved`` check
    below asserts the fields actually moved, so a no-op graph fails loudly.
    ``abdquickest`` is forced onto the eager path (live Courant number); it is
    included to confirm that exclusion still evolves correctly."""
    import os
    from lilytorch.src.advection import AdvDiffSolver, FluxAddGraphRunner

    N, dev = 34, torch.device("cuda:0")
    x = torch.linspace(0.0, 1.0, N, device=dev, dtype=torch.float32)
    torch.manual_seed(0)
    seed = [torch.randn(N, N, N, device=dev, dtype=torch.float32) * 0.1
            for _ in range(3)]

    def run(graph_on):
        s = AdvDiffSolver(dev, dt=1e-3, x=x, y=x.clone(), nu=1e-3,
                          z=x.clone(), method=scheme)
        # Force the desired dispatch regardless of the LILY_FLUX_GRAPH env var.
        from lilytorch.src import advection as _adv
        s._flux_graph_runner = (FluxAddGraphRunner() if graph_on
                                else _adv.advect_flux_add_warp)
        u, v, w = (f.clone() for f in seed)
        for _ in range(25):        # ≥20 steps; runner captures on 2nd sighting
            out = s.solve(u, v, w)
            u.copy_(out[0]); v.copy_(out[1]); w.copy_(out[2])
        torch.cuda.synchronize()
        return u, v, w

    ue, ve, we = run(False)
    ug, vg, wg = run(True)

    # Fields must have actually evolved (a no-op graph would leave seed intact).
    evolved = (ue - seed[0]).abs().max().item()
    assert evolved > 1e-3, f"fields did not evolve ({evolved:.3e}) — bad harness"

    for a, b, name in ((ue, ug, "u"), (ve, vg, "v"), (we, wg, "w")):
        d = (a - b).abs().max().item()
        assert d == 0.0, f"graph vs eager {name} mismatch: {d:.3e} (dropped launches?)"


@SKIP_NO_CUDA
def test_flux_graph_runner_replays_after_warmup():
    """The flux runner must actually take the graph fast path in a steady solve
    loop — not silently stay eager because the rhs pointer churns.  After warm-up
    ``replays`` must be positive and at least the number of captured graphs (each
    stable (component, direction) signature is captured once then replayed)."""
    from lilytorch.src.advection import AdvDiffSolver, FluxAddGraphRunner
    N, dev = 24, torch.device("cuda:0")
    x = torch.linspace(0.0, 1.0, N, device=dev, dtype=torch.float32)
    torch.manual_seed(0)
    s = AdvDiffSolver(dev, dt=1e-3, x=x, y=x.clone(), nu=1e-3,
                      z=x.clone(), method="quick")
    runner = FluxAddGraphRunner()
    s._flux_graph_runner = runner
    u, v, w = (torch.randn(N, N, N, device=dev, dtype=torch.float32) * 0.1
               for _ in range(3))
    for _ in range(6):
        out = s.solve(u, v, w)
        u.copy_(out[0]); v.copy_(out[1]); w.copy_(out[2])
    torch.cuda.synchronize()
    assert runner.captures > 0, "flux runner never captured a graph (pointer churn?)"
    assert runner.replays >= runner.captures, (
        f"replays={runner.replays} < captures={runner.captures} — graphs not reused")


# ═════════════════════════════════════════════════════════════════════════════
#  apply_bcs_2d / apply_bcs_3d — fused BC ghost writes (Warp CPU == GPU)
#  Merged from the former test_misc_2d.py / test_misc_3d.py.
# ═════════════════════════════════════════════════════════════════════════════

# ─── apply_bcs_2d ─────────────────────────────────────────────────────────────

def _bcs_problem_2d(dev, Nx=40, Ny=32, seed=9):
    torch.manual_seed(seed)
    u = torch.randn(Nx, Ny, dtype=torch.float64)
    v = torch.randn(Nx, Ny, dtype=torch.float64)
    shapes = torch.tensor([[Nx, Ny], [Nx, Ny]], dtype=torch.int64)
    # Descriptors chosen so no two STAGE-1 (Neumann+Dirichlet) ops share a
    # cell — overlapping stage-1 writes are order-undefined on GPU, so a
    # deterministic bit-exact comparison needs disjoint ops.
    # Neumann: u rows 0 and Nx-1 (axis0, both sides).
    neu = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.int32)
    # Dirichlet: v cols 0 and Ny-1 (axis1).  Different field from Neumann.
    dird = torch.tensor([[1, 1, 0], [1, 1, -1]], dtype=torch.int32)
    dirv = torch.tensor([2.5, -1.3], dtype=torch.float64)
    # Reflective (stage 2 → runs last, deterministic even at corners): u col Ny-1.
    refd = torch.tensor([[0, 1, -1, -2]], dtype=torch.int32)
    refv = torch.tensor([0.4], dtype=torch.float64)
    max_line = max(Nx, Ny)
    to = lambda t: t.to(dev)
    return (to(u), to(v), to(shapes), to(neu), to(dird), to(dirv),
            to(refd), to(refv), max_line)


def _run_bcs_2d(dev, f32=False):
    u, v, shapes, neu, dird, dirv, refd, refv, ml = _bcs_problem_2d(dev)
    if f32:
        u = u.float(); v = v.float(); dirv = dirv.float(); refv = refv.float()
    uw, vw = u.clone().contiguous(), v.clone().contiguous()
    apply_bcs_2d_warp(uw, vw, shapes, neu, dird, dirv, refd, refv, ml)
    wp.synchronize()
    return uw, vw


@SKIP_NO_CUDA
@pytest.mark.parametrize("f32", [False, True], ids=["f64", "f32"])
def test_apply_bcs_2d_cpu_eq_gpu(f32):
    """BC writes are copies / value sets → CPU == GPU bit-exact."""
    uc, vc = _run_bcs_2d("cpu", f32)
    ug, vg = _run_bcs_2d("cuda:0", f32)
    du = (uc - ug.cpu()).abs().max().item()
    dv = (vc - vg.cpu()).abs().max().item()
    assert du == 0.0 and dv == 0.0, f"bcs cpu vs gpu maxdiff u={du:.3e} v={dv:.3e}"


# ─── apply_bcs_3d ─────────────────────────────────────────────────────────────


def _bcs_problem_3d(dev, Nx=20, Ny=16, Nz=14, seed=9):
    torch.manual_seed(seed)
    u = torch.randn(Nx, Ny, Nz, dtype=torch.float64)
    v = torch.randn(Nx, Ny, Nz, dtype=torch.float64)
    w = torch.randn(Nx, Ny, Nz, dtype=torch.float64)
    shapes = torch.tensor([[Nx, Ny, Nz]]*3, dtype=torch.int64)
    # disjoint stage-1 ops (no shared face cell): u x-faces, v y-faces, w z-Dirichlet
    neu = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 1, 0]], dtype=torch.int32)
    dird = torch.tensor([[2, 2, 0], [2, 2, -1]], dtype=torch.int32)
    dirv = torch.tensor([2.5, -1.3], dtype=torch.float64)
    refd = torch.tensor([[1, 1, -1, -2]], dtype=torch.int32)   # v y-face reflective (stage2)
    refv = torch.tensor([0.4], dtype=torch.float64)
    M = max(Nx, Ny, Nz)
    to = lambda t: t.to(dev)
    return (to(u), to(v), to(w), to(shapes), to(neu), to(dird), to(dirv), to(refd), to(refv), M)


def _run_bcs_3d(dev, f32=False):
    u, v, w, shapes, neu, dird, dirv, refd, refv, M = _bcs_problem_3d(dev)
    if f32:
        u = u.float(); v = v.float(); w = w.float()
        dirv = dirv.float(); refv = refv.float()
    uw, vw, ww = u.clone().contiguous(), v.clone().contiguous(), w.clone().contiguous()
    apply_bcs_3d_warp(uw, vw, ww, shapes, neu, dird, dirv, refd, refv, M, M)
    wp.synchronize()
    return uw, vw, ww


@SKIP_NO_CUDA
@pytest.mark.parametrize("f32", [False, True], ids=["f64", "f32"])
def test_apply_bcs_3d_cpu_eq_gpu(f32):
    """BC writes are copies / value sets → CPU == GPU bit-exact."""
    rc = _run_bcs_3d("cpu", f32)
    rg = _run_bcs_3d("cuda:0", f32)
    for a, b, nm in zip(rc, rg, ("u", "v", "w")):
        d = (a - b.cpu()).abs().max().item()
        assert d == 0.0, f"bcs3d cpu vs gpu {nm} mismatch {d:.3e}"


@SKIP_NO_CUDA
def test_apply_bcs_3d_noncubic_dual_facedims():
    """Non-cubic grid with separate (max_dim0, max_dim1): CPU == GPU exactly."""
    Nx, Ny, Nz = 24, 14, 10
    torch.manual_seed(3)
    u = torch.randn(Nx, Ny, Nz, dtype=torch.float64)
    v = torch.randn(Nx, Ny, Nz, dtype=torch.float64)
    w = torch.randn(Nx, Ny, Nz, dtype=torch.float64)
    shapes = torch.tensor([[Nx, Ny, Nz]] * 3, dtype=torch.int64)
    # Disjoint per-component ops (no shared stage-1 cell — overlaps are
    # order-undefined on GPU) that still exercise all three face axes so
    # dim0/dim1 differ per face:
    #   u z-face (axis2): d0=Nx=24, d1=Ny=14  → drives max_dim0, max_dim1
    #   w x-face (axis0): d0=Ny=14, d1=Nz=10
    #   v y-face (axis1): d0=Nx=24, d1=Nz=10
    neu = torch.tensor([[0, 2, 1], [2, 0, 0]], dtype=torch.int32)
    dird = torch.tensor([[1, 1, -1]], dtype=torch.int32)
    dirv = torch.tensor([1.1], dtype=torch.float64)
    refd = torch.zeros((0, 4), dtype=torch.int32)
    refv = torch.zeros((0,), dtype=torch.float64)
    max_dim0 = int(max(Ny, Nx))
    max_dim1 = int(max(Nz, Ny))

    def run(dev):
        to = lambda t: t.to(dev)
        args = [to(x) for x in (u, v, w, shapes, neu, dird, dirv, refd, refv)]
        uw, vw, ww = (args[0].clone().contiguous(), args[1].clone().contiguous(),
                      args[2].clone().contiguous())
        apply_bcs_3d_warp(uw, vw, ww, *args[3:9], max_dim0, max_dim1)
        wp.synchronize()
        return uw, vw, ww

    rc, rg = run("cpu"), run("cuda:0")
    for a, b, nm in zip(rc, rg, ("u", "v", "w")):
        d = (a - b.cpu()).abs().max().item()
        assert d == 0.0, f"bcs3d noncubic cpu vs gpu {nm} mismatch {d:.3e}"


# ─── ApplyBcs{2,3}DGraphRunner — CUDA-graph replay must equal eager ───────────
#  Guards two things at once: (1) the captured graph replays bit-for-bit vs the
#  eager apply_bcs_*_warp path over a multi-step run with a *stable* buffer
#  (a dropped-launch graph would leave the ghost cells stale → mismatch), and
#  (2) the replay/capture counters behave — after the 2nd-sighting capture,
#  every later call replays (the plan's "prove the runners actually replay").

@SKIP_NO_CUDA
@pytest.mark.parametrize("f32", [False, True], ids=["f64", "f32"])
def test_apply_bcs_2d_graph_replay_eq_eager(f32):
    from lilytorch.src.advection import ApplyBcs2DGraphRunner
    u0, v0, shapes, neu, dird, dirv, refd, refv, ml = _bcs_problem_2d("cuda:0")
    if f32:
        u0 = u0.float(); v0 = v0.float(); dirv = dirv.float(); refv = refv.float()

    K = 8
    torch.manual_seed(1)
    # Deterministic in-place interior perturbations so the Neumann/reflective
    # ghost writes (which read the interior) actually change step to step —
    # identically for both runs — while the buffer pointer stays fixed.
    noise = [torch.randn_like(u0) for _ in range(K)]

    def run(use_graph):
        u = u0.clone().contiguous(); v = v0.clone().contiguous()
        runner = ApplyBcs2DGraphRunner() if use_graph else None
        for k in range(K):
            u.add_(noise[k]); v.add_(noise[k])
            if use_graph:
                runner(u, v, shapes, neu, dird, dirv, refd, refv, ml)
            else:
                apply_bcs_2d_warp(u, v, shapes, neu, dird, dirv, refd, refv, ml)
        wp.synchronize()
        return u, v, runner

    ue, ve, _ = run(False)
    ug, vg, runner = run(True)
    du = (ue - ug).abs().max().item(); dv = (ve - vg).abs().max().item()
    assert du == 0.0 and dv == 0.0, f"graph vs eager maxdiff u={du:.3e} v={dv:.3e}"
    # Stable signature: call 1 eager (first sighting), calls 2..K replay.
    assert runner.captures == 1, f"captures={runner.captures} (expected 1)"
    assert runner.eager == 1, f"eager={runner.eager} (expected 1)"
    assert runner.replays == K - 1, f"replays={runner.replays} (expected {K-1})"


@SKIP_NO_CUDA
@pytest.mark.parametrize("f32", [False, True], ids=["f64", "f32"])
def test_apply_bcs_3d_graph_replay_eq_eager(f32):
    from lilytorch.src.advection import ApplyBcs3DGraphRunner
    u0, v0, w0, shapes, neu, dird, dirv, refd, refv, M = _bcs_problem_3d("cuda:0")
    if f32:
        u0 = u0.float(); v0 = v0.float(); w0 = w0.float()
        dirv = dirv.float(); refv = refv.float()

    K = 8
    torch.manual_seed(2)
    noise = [torch.randn_like(u0) for _ in range(K)]

    def run(use_graph):
        u = u0.clone().contiguous(); v = v0.clone().contiguous()
        w = w0.clone().contiguous()
        runner = ApplyBcs3DGraphRunner() if use_graph else None
        for k in range(K):
            u.add_(noise[k]); v.add_(noise[k]); w.add_(noise[k])
            if use_graph:
                runner(u, v, w, shapes, neu, dird, dirv, refd, refv, M, M)
            else:
                apply_bcs_3d_warp(u, v, w, shapes, neu, dird, dirv, refd, refv, M, M)
        wp.synchronize()
        return u, v, w, runner

    ue, ve, we, _ = run(False)
    ug, vg, wg, runner = run(True)
    for a, b, nm in ((ue, ug, "u"), (ve, vg, "v"), (we, wg, "w")):
        d = (a - b).abs().max().item()
        assert d == 0.0, f"graph vs eager {nm} mismatch: {d:.3e} (dropped launches?)"
    assert runner.captures == 1, f"captures={runner.captures} (expected 1)"
    assert runner.eager == 1, f"eager={runner.eager} (expected 1)"
    assert runner.replays == K - 1, f"replays={runner.replays} (expected {K-1})"


@SKIP_NO_CUDA
def test_apply_bcs_2d_graph_recaptures_on_new_dtype():
    """A single runner instance handles both dtypes: ``str(u.dtype)`` is in the
    signature key, so an f64 signature never replays an f32 graph.  Both dtypes
    stay correct and each captures its own graph."""
    from lilytorch.src.advection import ApplyBcs2DGraphRunner
    runner = ApplyBcs2DGraphRunner()
    for dt in (torch.float64, torch.float32):
        u0, v0, shapes, neu, dird, dirv, refd, refv, ml = _bcs_problem_2d("cuda:0")
        if dt == torch.float32:
            u0 = u0.float(); v0 = v0.float(); dirv = dirv.float(); refv = refv.float()
        u = u0.clone().contiguous(); v = v0.clone().contiguous()
        ue = u.clone(); ve = v.clone()
        for _ in range(4):
            runner(u, v, shapes, neu, dird, dirv, refd, refv, ml)
            apply_bcs_2d_warp(ue, ve, shapes, neu, dird, dirv, refd, refv, ml)
        wp.synchronize()
        assert (u - ue).abs().max().item() == 0.0
        assert (v - ve).abs().max().item() == 0.0
    assert runner.captures == 2, f"captures={runner.captures} (expected one per dtype)"


# ═════════════════════════════════════════════════════════════════════════════
#  Fused semi-Lagrangian advection (2-D) — sl_advect_2d_kernel
#  One Warp launch replaces the 10 python RegularGridInterpolator calls of the
#  RK2 back-trace.  The pure-Python reference lives here as a standalone oracle
#  (moved from AdvDiffSolver._solve_semi_lagrangian_python).
# ═════════════════════════════════════════════════════════════════════════════

from lilytorch.src.advection import (
    AdvDiffSolver,
    _inner,
    SLAdvect2DGraphRunner,
    SLAdvect3DGraphRunner,
)
from lilytorch.src import diffusion


def _sl_python_oracle(solver, *vel, nu_t=None):
    """Pure-Python semi-Lagrangian RK2 back-trace — independent reference
    oracle for the Warp fused kernels.  Moved here from the now-removed
    ``AdvDiffSolver._solve_semi_lagrangian_python``."""
    ndim = solver.ndim
    shape = tuple(solver.n)
    for i in range(ndim):
        solver._interps[i].F = vel[i]
    vel_new = list(vel)
    half_dt = 0.5 * solver.dt
    for i in range(ndim):
        vel_at_i = [
            solver._interps[d](*solver._flat_coords[i]).clone()
            for d in range(ndim)
        ]
        midpoint = [
            solver._flat_coords[i][d] - half_dt * vel_at_i[d]
            for d in range(ndim)
        ]
        vel_at_mid = [
            solver._interps[d](*midpoint).clone()
            for d in range(ndim)
        ]
        departure = [
            solver._flat_coords[i][d] - solver.dt * vel_at_mid[d]
            for d in range(ndim)
        ]
        vel_new[i] = solver._interps[i](*departure).reshape(shape).clone()
    inner = _inner(ndim)
    for i in range(ndim):
        vel_new[i][inner] += diffusion.diffuse(
            vel_new[i], solver.dt, nu=solver.nu, nu_t=nu_t,
            inv_dh2=solver._inv_dh2, dh=solver.dh,
        )
    return tuple(vel_new)


def _sl_solver_2d(dev, dtype, N=48, M=40, dt=0.03, nu=1e-3, seed=7):
    """2-D SL solver on a non-square grid + random MAC fields.

    dt is large enough that departure points cross cell boundaries and leave
    the domain near the border, exercising the clamp + bilinear-fallback
    paths of the biquadratic sampler."""
    x = torch.linspace(0.0, 1.0, N, device=dev, dtype=dtype)
    y = torch.linspace(0.0, 0.8, M, device=dev, dtype=dtype)
    s = AdvDiffSolver(torch.device(dev), dt=dt, x=x, y=y, nu=nu,
                      method="implicit")
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = (torch.rand((N, M), generator=g, dtype=torch.float64) - 0.5).to(
        device=dev, dtype=dtype)
    v = (torch.rand((N, M), generator=g, dtype=torch.float64) - 0.5).to(
        device=dev, dtype=dtype)
    return s, u, v


@pytest.mark.parametrize("dev", ["cpu",
                                 pytest.param("cuda:0", marks=SKIP_NO_CUDA)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32],
                         ids=["f64", "f32"])
def test_sl_fused_matches_python_2d(dev, dtype):
    """Fused RK2 kernel == python interpolator path (same math, same
    biquadratic sampler @wp.func).  CPU is bit-exact; on CUDA the only
    drift source is FMA contraction in the midpoint/departure arithmetic
    (observed 3.5e-15 f64 / 2.8e-6 f32 → tolerances give ~3x headroom)."""
    s, u, v = _sl_solver_2d(dev, dtype)
    ref = _sl_python_oracle(s, u.clone(), v.clone())
    got = s._solve_semi_lagrangian_warp(u.clone(), v.clone())
    wp.synchronize()
    tol = 1e-14 if dtype is torch.float64 else 1e-5
    for a, b, nm in zip(ref, got, ("u", "v")):
        d = (a - b).abs().max().item()
        assert d < tol, f"sl fused vs python {nm} ({dev}, {dtype}): {d:.3e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32],
                         ids=["f64", "f32"])
def test_sl_fused_cpu_equals_gpu(dtype):
    """Single-source check: the SAME sl_advect_2d_kernel on CPU == GPU."""
    sc, uc, vc = _sl_solver_2d("cpu", dtype)
    sg, ug, vg = _sl_solver_2d("cuda:0", dtype)
    rc = sc._solve_semi_lagrangian_warp(uc, vc)
    rg = sg._solve_semi_lagrangian_warp(ug, vg)
    wp.synchronize()
    tol = 1e-14 if dtype is torch.float64 else 1e-5
    for a, b, nm in zip(rc, rg, ("u", "v")):
        d = (a - b.cpu()).abs().max().item()
        assert d < tol, f"sl kernel cpu vs gpu {nm} ({dtype}): {d:.3e}"


@pytest.mark.parametrize("dev", ["cpu",
                                 pytest.param("cuda:0", marks=SKIP_NO_CUDA)])
def test_sl_dispatch_takes_warp_path_2d(dev):
    """The production entry point (`solve`) must actually take the fused
    Warp path on contiguous 2-D fields (CPU + CUDA) — guards against a
    silent fall-back to the python interpolator path."""
    s, u, v = _sl_solver_2d(dev, torch.float32)
    assert s._sl_out is None
    out = s.solve(u, v)
    assert s._sl_out is not None, f"solve() on {dev} did not take the fused Warp path"
    assert out[0].data_ptr() == s._sl_out[0].data_ptr()


@SKIP_NO_CUDA
def test_sl_graph_runner_matches_eager():
    """Multi-step SL run with the CUDA-graph runner must reproduce the eager
    kernel run bit-for-bit (same kernel, baked pointers).  Mirrors
    test_flux_graph_runner_matches_eager, including the guard against the
    ``b423298`` dropped-launch failure mode: a no-op graph would leave the
    persistent out-buffers stale, caught by both the eager comparison and
    the ``evolved`` check."""
    from lilytorch.src import advection as _adv

    def run(graph_on):
        s, u, v = _sl_solver_2d("cuda:0", torch.float32)
        s._sl_graph_runner = (SLAdvect2DGraphRunner() if graph_on
                              else _adv.sl_advect_2d_warp)
        u0, v0 = u.clone(), v.clone()
        for _ in range(25):        # runner captures on 2nd sighting
            out = s.solve(u, v)
            u.copy_(out[0]); v.copy_(out[1])
        torch.cuda.synchronize()
        return u0, u.clone(), v.clone()

    seed_u, ue, ve = run(False)
    _, ug, vg = run(True)

    evolved = (ue - seed_u).abs().max().item()
    assert evolved > 1e-3, f"fields did not evolve ({evolved:.3e}) — bad harness"

    for a, b, name in ((ue, ug, "u"), (ve, vg, "v")):
        d = (a - b).abs().max().item()
        assert d == 0.0, f"graph vs eager {name} mismatch: {d:.3e} (dropped launches?)"


def test_sl_warp_path_always_taken(monkeypatch):
    """The Warp path is now the ONLY production path — even with
    LILY_SL_KERNEL=0 the dispatch goes through the fused kernel."""
    monkeypatch.setenv("LILY_SL_KERNEL", "0")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    s, u, v = _sl_solver_2d(dev, torch.float32)
    out = s.solve(u, v)
    assert s._sl_out is not None, "Warp path must always be taken"
    assert out[0].data_ptr() == s._sl_out[0].data_ptr()


# ═════════════════════════════════════════════════════════════════════════════
#  Fused semi-Lagrangian advection (3-D) — sl_advect_3d_kernel
#  One Warp launch replaces the 21 python RegularGridInterpolator calls.
# ═════════════════════════════════════════════════════════════════════════════


def _sl_solver_3d(dev, dtype, Nx=24, Ny=20, Nz=16, dt=0.02, nu=1e-3, seed=3):
    """3-D SL solver on a non-cubic grid + random MAC fields.

    dt is large enough that departure points cross cell boundaries and leave
    the domain near the border, exercising the clamp + trilinear-fallback
    paths of the triquadratic sampler."""
    x = torch.linspace(0.0, 1.0, Nx, device=dev, dtype=dtype)
    y = torch.linspace(0.0, 0.8, Ny, device=dev, dtype=dtype)
    z = torch.linspace(0.0, 0.6, Nz, device=dev, dtype=dtype)
    s = AdvDiffSolver(torch.device(dev), dt=dt, x=x, y=y, nu=nu, z=z,
                      method="implicit")
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (Nx, Ny, Nz)
    u = (torch.rand(shape, generator=g, dtype=torch.float64) - 0.5).to(
        device=dev, dtype=dtype)
    v = (torch.rand(shape, generator=g, dtype=torch.float64) - 0.5).to(
        device=dev, dtype=dtype)
    w = (torch.rand(shape, generator=g, dtype=torch.float64) - 0.5).to(
        device=dev, dtype=dtype)
    return s, u, v, w


@pytest.mark.parametrize("dev", ["cpu",
                                 pytest.param("cuda:0", marks=SKIP_NO_CUDA)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32],
                         ids=["f64", "f32"])
def test_sl_fused_matches_python_3d(dev, dtype):
    """Fused 3-D RK2 kernel == python interpolator path (same math, same
    triquadratic sampler @wp.func).  CPU is bit-exact; CUDA tolerance
    accounts for FMA contraction drift."""
    s, u, v, w = _sl_solver_3d(dev, dtype)
    ref = _sl_python_oracle(s, u.clone(), v.clone(), w.clone())
    got = s._solve_semi_lagrangian_warp(u.clone(), v.clone(), w.clone())
    wp.synchronize()
    tol = 1e-14 if dtype is torch.float64 else 1e-5
    for a, b, nm in zip(ref, got, ("u", "v", "w")):
        d = (a - b).abs().max().item()
        assert d < tol, f"sl3d fused vs python {nm} ({dev}, {dtype}): {d:.3e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32],
                         ids=["f64", "f32"])
def test_sl_fused_cpu_equals_gpu_3d(dtype):
    """Single-source check: the SAME sl_advect_3d_kernel on CPU == GPU."""
    sc, uc, vc, wc = _sl_solver_3d("cpu", dtype)
    sg, ug, vg, wg = _sl_solver_3d("cuda:0", dtype)
    rc = sc._solve_semi_lagrangian_warp(uc, vc, wc)
    rg = sg._solve_semi_lagrangian_warp(ug, vg, wg)
    wp.synchronize()
    tol = 1e-14 if dtype is torch.float64 else 1e-5
    for a, b, nm in zip(rc, rg, ("u", "v", "w")):
        d = (a - b.cpu()).abs().max().item()
        assert d < tol, f"sl3d kernel cpu vs gpu {nm} ({dtype}): {d:.3e}"


@pytest.mark.parametrize("dev", ["cpu",
                                 pytest.param("cuda:0", marks=SKIP_NO_CUDA)])
def test_sl_dispatch_takes_warp_path_3d(dev):
    """The production entry point (`solve`) must take the fused Warp
    path on contiguous 3-D fields (CPU + CUDA)."""
    s, u, v, w = _sl_solver_3d(dev, torch.float32)
    assert s._sl_out is None
    out = s.solve(u, v, w)
    assert s._sl_out is not None, f"solve() 3-D on {dev} did not take the fused Warp path"
    assert len(s._sl_out) == 3
    assert out[0].data_ptr() == s._sl_out[0].data_ptr()


@SKIP_NO_CUDA
def test_sl_graph_runner_matches_eager_3d():
    """Multi-step 3-D SL run with the CUDA-graph runner must reproduce the
    eager kernel run bit-for-bit."""
    from lilytorch.src import advection as _adv

    def run(graph_on):
        s, u, v, w = _sl_solver_3d("cuda:0", torch.float32)
        s._sl_graph_runner_3d = (SLAdvect3DGraphRunner() if graph_on
                                 else _adv.sl_advect_3d_warp)
        u0, v0, w0 = u.clone(), v.clone(), w.clone()
        for _ in range(25):        # runner captures on 2nd sighting
            out = s.solve(u, v, w)
            u.copy_(out[0]); v.copy_(out[1]); w.copy_(out[2])
        torch.cuda.synchronize()
        return u0, u.clone(), v.clone(), w.clone()

    seed_u, ue, ve, we = run(False)
    _, ug, vg, wg = run(True)

    evolved = (ue - seed_u).abs().max().item()
    assert evolved > 1e-3, f"3-D fields did not evolve ({evolved:.3e}) — bad harness"

    for a, b, name in ((ue, ug, "u"), (ve, vg, "v"), (we, wg, "w")):
        d = (a - b).abs().max().item()
        assert d == 0.0, f"graph vs eager {name} mismatch: {d:.3e} (dropped launches?)"
