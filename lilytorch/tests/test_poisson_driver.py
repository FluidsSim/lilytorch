"""Convergence tests for the native Poisson driver.

``poisson_mult.PoissonSolver`` dispatches to the native CUDA / C++ kernels:

* CUDA — the whole-solve drivers ``poisson_solve_{multigrid,mgcg,rmgcg}_{2,3}d``
  run the entire solve (V-cycle tree, CG loop, deflation, gauge) in one C++ call.
* CPU — ``multigrid`` has a whole-solve twin; ``mgcg`` / ``rmgcg`` run the CG
  loop in Python over the native ``mg_vcycle_{2,3}d`` preconditioner.

These check it converges on a manufactured Neumann Poisson for every
(method, ndim, dtype, smoother) combination on BOTH devices, and that the two
backends agree.
"""
from __future__ import annotations

import pytest
import torch

from lilytorch.src.poisson_mult import PoissonSolver

CUDA = torch.cuda.is_available()
SKIP_NO_CUDA = pytest.mark.skipif(not CUDA, reason="CUDA not available")
_DEVS = ["cpu"] + (["cuda"] if CUDA else [])
_METHODS = ["solve_multigrid", "solve_mgcg", "solve_rmgcg"]


def _problem(ndim, dtype, N=48, device="cpu"):
    torch.manual_seed(0)
    h = 1.0 / N
    shp = (N,) * ndim
    f = torch.randn(*shp, dtype=dtype)
    f -= f.mean()                                    # compatible (Neumann) RHS
    pshp = tuple(n + 2 for n in shp)
    p = torch.zeros(pshp, dtype=dtype)
    o = dict(dtype=dtype)
    if ndim == 2:
        faces = dict(ch=torch.full((N + 1, N), 1.0, **o),
                     cv=torch.full((N, N + 1), 1.0, **o))
    else:
        faces = dict(ch=torch.full((N + 1, N, N), 1.0, **o),
                     cv=torch.full((N, N + 1, N), 1.0, **o),
                     cw=torch.full((N, N, N + 1), 1.0, **o))
    # Built once on CPU then moved: a device-seeded generator would otherwise
    # hand the two devices different problems.
    return (h, f.to(device), p.to(device),
            {k: v.to(device) for k, v in faces.items()})


def _solve(method, ndim, dtype, smoother, device, N=48):
    h, f, p, faces = _problem(ndim, dtype, N=N, device=device)
    s = PoissonSolver(dtype=dtype, device=device, h=h, tol=1e-6, max_vcycles=40,
                      max_cycles=50, nsmoothing=2, smoother=smoother,
                      verbose=False,
                      recycle_k=2 if method == "solve_rmgcg" else 0)
    p_out, r = getattr(s, method)(
        f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    return p_out, r.abs().max().item(), f.abs().max().item()


@pytest.mark.parametrize("device", _DEVS)
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("smoother", ["rbgs", "jacobi"])
def test_poisson_converges(method, ndim, dtype, smoother, device):
    """The native driver drives the residual well below the RHS scale.

    The bar is deliberately loose (1e-3 · |f|): plain multigrid with an
    undamped (w=1) Jacobi smoother is a genuinely slow solver, and this test is
    gating "the driver converges", not "the smoother is optimal".  The MGCG
    paths land ~6 orders below this.
    """
    _, r, fscale = _solve(method, ndim, dtype, smoother, device)
    assert r < 1e-3 * fscale, \
        f"{method} {ndim}D {dtype} {smoother} on {device}: residual {r:.2e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("ndim", [2, 3])
def test_poisson_cpu_agrees_with_cuda(method, ndim):
    """The CPU and CUDA backends return the same pressure — no constant allowed.

    Compared on the raw values, with NO ``d - d.mean()`` step: both drivers end
    with the same full ghost-ring Neumann pass and the same INTERIOR-only gauge
    (csrc/poisson_gauge.h), so the arbitrary Neumann constant is pinned to the
    same value on both backends.  It used to be a mean over the FULL padded
    tensor, dead corners included — and those corners hold backend-dependent
    garbage (the CUDA Jacobi's odd-sweep ping-pong copies a zeroed scratch
    buffer back over p), so the gauge constant itself was backend-dependent and
    the two backends returned pressures differing by a constant.

    The whole padded tensor is compared, ghost ring included: the ring is now
    re-derived from the interior on both sides.

    Not bit-exact: the CUDA 2-D smoothers are tiled (stale cross-block halo
    reads by design), so the backends take slightly different iteration paths to
    the same answer.
    """
    dtype = torch.float64
    pc, rc, fscale = _solve(method, ndim, dtype, "rbgs", "cpu")
    pg, rg, _ = _solve(method, ndim, dtype, "rbgs", "cuda")

    assert rc < 1e-3 * fscale and rg < 1e-3 * fscale, \
        f"{method} {ndim}D did not converge (cpu {rc:.2e}, cuda {rg:.2e})"

    inner = (slice(1, -1),) * ndim
    scale = pc[inner].abs().max().item()

    # Raw interior difference — a constant offset would show up here.
    d = pc[inner] - pg.cpu()[inner]
    err = d.abs().max().item()
    assert err < 1e-4 * scale, \
        f"{method} {ndim}D cpu vs cuda interior pressure: {err:.3e} (scale {scale:.3e})"

    # The gauge constant must be pinned: removing the mean may not improve the
    # agreement, or the two backends are still differing by a constant.
    demeaned = (d - d.mean()).abs().max().item()
    assert err <= demeaned * 1.5 + 1e-12 * scale, \
        (f"{method} {ndim}D cpu vs cuda pressure still differs by a CONSTANT "
         f"(raw {err:.3e} vs de-meaned {demeaned:.3e}) — the gauge is not "
         f"interior-only on one of the backends")

    # Full padded tensor, ghost ring included.
    errf = (pc - pg.cpu()).abs().max().item()
    assert errf < 1e-4 * scale, \
        f"{method} {ndim}D cpu vs cuda padded pressure (ghosts): {errf:.3e}"


@pytest.mark.parametrize("device", _DEVS)
@pytest.mark.parametrize("ndim", [2, 3])
def test_mgcg_beats_plain_multigrid(ndim, device):
    """MGCG must converge at least as far as plain multigrid on the same budget.

    Guards the preconditioner wiring: on the CPU ``solve_mgcg`` runs the Python
    CG loop over ``mg_vcycle_*``, and a mis-scaled or gauge-fixed V-cycle there
    would still "converge" — just far worse than the plain multigrid it wraps.
    """
    _, r_mg, fscale = _solve("solve_multigrid", ndim, torch.float64, "rbgs", device)
    _, r_cg, _ = _solve("solve_mgcg", ndim, torch.float64, "rbgs", device)
    assert r_cg <= max(r_mg, 1e-12 * fscale), \
        f"{ndim}D on {device}: mgcg {r_cg:.2e} worse than multigrid {r_mg:.2e}"


# ── MGCG on a STIFF variable-coefficient operator ────────────────────────────
# The uniform-coefficient problem above is too easy to expose a broken CG
# preconditioner: the multigrid V-cycle is only *mildly* non-symmetric there.
# The real regime (two-phase 80:1–833:1 density jumps) is where a non-symmetric
# preconditioner made mgcg/rmgcg converge WORSE than plain multigrid.  These
# gates build that stiff operator (coeff jump across a z-plane) and check the
# two properties the 2026-07-15 fix restored:
#   1. the CG-preconditioner V-cycle (full-weighting restriction = P^T) is
#      SYMMETRIC to ~machine precision, and
#   2. mgcg converges at least as far as multigrid at EQUAL V-cycle budget.
# Before the fix (sum-of-children restriction != trilinear^T) both fail: the
# one-cycle operator was ~4–55% asymmetric and mgcg-30 was ~8x worse than
# multigrid-30 at 833:1.

def _stiff_faces(ndim, dtype, N, ratio, device):
    """Face coefficients with a `ratio`:1 jump across the mid-plane of the last
    axis (mimics the projection coeff c=dt/rho across an air/water interface)."""
    o = dict(dtype=dtype, device=device)
    lo, hi = 1.0, ratio                       # c is `ratio`x larger in the "air"
    shp = (N,) * ndim
    c = torch.full(shp, lo, **o)
    half = N // 2
    c[(...,)] = torch.where(
        torch.arange(N, device=device).expand(shp) < half,
        torch.tensor(lo, **o), torch.tensor(hi, **o))
    cpad = torch.nn.functional.pad(
        c[(None, None)], (1, 1) * ndim, mode="replicate")[0, 0]
    faces = {}
    for d, lab in enumerate(["ch", "cv", "cw"][:ndim]):
        fwd = [slice(1, -1)] * ndim; fwd[d] = slice(1, None)
        bwd = [slice(1, -1)] * ndim; bwd[d] = slice(None, -1)
        faces[lab] = (0.5 * (cpad[tuple(fwd)] + cpad[tuple(bwd)])).contiguous()
    return faces


@SKIP_NO_CUDA
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("ratio", [80.0, 833.0])
def test_mgcg_preconditioner_is_symmetric(ndim, ratio):
    """One CG-preconditioner V-cycle must be a SYMMETRIC operator: <Mu,v>=<u,Mv>."""
    dtype, device, N = torch.float64, "cuda", 32
    h = 1.0 / N
    faces = _stiff_faces(ndim, dtype, N, ratio, device)
    s = PoissonSolver(dtype=dtype, device=device, h=h, tol=-1.0, max_vcycles=1,
                      max_cycles=1, nsmoothing=5, w=1, smoother="jacobi",
                      verbose=False)
    face_arrs = [faces[k] for k in ["ch", "cv", "cw"][:ndim]]
    inner = (slice(1, -1),) * ndim
    pshp = (N + 2,) * ndim

    def M(vec):
        z = torch.zeros(pshp, dtype=dtype, device=device)
        z, _ = s._dispatch_vcycle(vec, z, face_arrs)
        return z[inner].clone()

    torch.manual_seed(0)
    u = torch.randn((N,) * ndim, dtype=dtype, device=device); u -= u.mean()
    v = torch.randn((N,) * ndim, dtype=dtype, device=device); v -= v.mean()
    a = (M(u) * v).sum().item()
    b = (u * M(v)).sum().item()
    rel = abs(a - b) / max(abs(a), abs(b), 1e-30)
    assert rel < 1e-9, \
        f"{ndim}D {ratio}:1 preconditioner not symmetric: <Mu,v>={a:.4e} <u,Mv>={b:.4e} rel={rel:.2e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("ratio", [80.0, 833.0])
def test_mgcg_beats_multigrid_stiff(ndim, ratio):
    """At EQUAL V-cycle budget, mgcg must not converge worse than multigrid on a
    stiff jump operator (the property a non-symmetric preconditioner destroyed)."""
    dtype, device, N, K = torch.float64, "cuda", 48, 20
    h = 1.0 / N
    faces = _stiff_faces(ndim, dtype, N, ratio, device)
    f = torch.randn((N,) * ndim, dtype=dtype, device=device); f -= f.mean()
    p0 = torch.zeros((N + 2,) * ndim, dtype=dtype, device=device)

    def run(method, max_vc, max_cg):
        s = PoissonSolver(dtype=dtype, device=device, h=h, tol=-1.0,
                          max_vcycles=max_vc, max_cycles=max_cg, nsmoothing=5,
                          w=1, smoother="jacobi", verbose=False)
        _, r = getattr(s, method)(f.clone(), p0.clone(),
                                  **{k: v.clone() for k, v in faces.items()})
        return r.abs().max().item()

    r_mg = run("solve_multigrid", K, 0)     # K V-cycles
    r_cg = run("solve_mgcg", 1, K)          # K CG iters = K V-cycles of work
    assert r_cg <= r_mg * 1.05, \
        f"{ndim}D {ratio}:1: mgcg {r_cg:.2e} worse than multigrid {r_mg:.2e} at equal budget"


# ── 3-D smoother CPU twins == CUDA kernels ───────────────────────────────────
# Regression gate for a CPU rbgs_sweep_3d parity bug.  The
# CPU Poisson previously ran on a different MG backend, so this twin was never compared to CUDA,
# and it was wrong two ways:
#
#   * a bogus row-level ``(i+j) & 1`` guard sat on TOP of the correct per-cell
#     ``(i+j+k) & 1`` colour test, so it skipped half the red cells and half the
#     black cells outright; and
#   * it never refreshed the Neumann ghost ring between the red and black
#     half-sweeps (the CUDA twin does), so the black cells read stale mirrors.
#
# RBGS is compared on EVERY cell, edge/corner ghosts included.
#
# JACOBI is compared on LIVE cells only (interior + face ghosts — the ones a
# 7-point stencil can reach).  That mask is NOT a stand-in for the apply_bcs
# race (fixed) nor for the Poisson gauge (fixed): the CUDA Jacobi's ping-pong
# copies a *zeroed* scratch buffer back over p when nsmoothing is ODD, so it
# zeroes the edge/corner ghosts while the CPU twin leaves them at their prior
# value.  The zeroing is deliberate — uninitialised memory once leaked NaN into
# p and blew up the coupled solve (see the comment in jacobi_sweep_3d_cuda) —
# and it is harmless because every whole-solve driver ends by re-deriving the
# full ghost ring from the interior (csrc/poisson_gauge.h).  So the mask stays,
# and it is asserted below that it is only ever needed for odd-sweep Jacobi.
#
# The 2-D CUDA smoothers are TILED (stale cross-block halo reads by design), so
# only the 3-D smoothers are expected to match a plain sequential CPU sweep.

def _live_cells(n, ndim):
    """True where a 7-point stencil can read: at most ONE axis on a boundary."""
    idx = torch.arange(n)
    on_b = sum(((idx == 0) | (idx == n - 1))
               .reshape([-1 if d == a else 1 for d in range(ndim)]).long()
               for a in range(ndim))
    return on_b <= 1


@SKIP_NO_CUDA
@pytest.mark.parametrize("smoother", ["rbgs", "jacobi"])
@pytest.mark.parametrize("nsmoothing", [1, 2, 3])
def test_smoother_3d_cpu_eq_cuda(smoother, nsmoothing):
    from lilytorch.src import native

    torch.manual_seed(1)
    N, dt = 8, torch.float64
    p = torch.randn(N + 2, N + 2, N + 2, dtype=dt)
    f = torch.randn(N, N, N, dtype=dt)
    cs = [torch.rand(N, N, N, dtype=dt) + 0.5 for _ in range(6)]

    def run(dev):
        pp = p.clone().to(dev).contiguous()
        args = [pp, f.to(dev)] + [c.to(dev) for c in cs]
        if smoother == "rbgs":
            native.rbgs_sweep_3d(*args, 1e-12, nsmoothing)
        else:
            native.jacobi_sweep_3d(*args, 1e-12, 1.0, nsmoothing)
        return pp.cpu()

    diff = (run("cpu") - run("cuda")).abs()
    live = _live_cells(N + 2, 3)

    d = diff[live].max().item()
    assert d < 1e-12, \
        f"{smoother}_sweep_3d nsmoothing={nsmoothing}: cpu vs cuda {d:.3e}"

    # The dead ghosts must agree too, EXCEPT for odd-nsmoothing Jacobi, whose
    # CUDA ping-pong deliberately zeroes them (see the header comment).  Pinning
    # that here keeps the mask honest: if any other case starts needing it, this
    # fails instead of hiding behind the mask.
    dead = diff[~live].max().item()
    if smoother == "jacobi" and nsmoothing % 2 == 1:
        return
    assert dead < 1e-12, \
        (f"{smoother}_sweep_3d nsmoothing={nsmoothing}: cpu vs cuda disagree on "
         f"the edge/corner ghosts by {dead:.3e} — only odd-sweep Jacobi is "
         f"allowed to (its ping-pong zeroes them)")
