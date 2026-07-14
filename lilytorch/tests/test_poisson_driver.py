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
    """The CPU and CUDA backends solve the same problem to the same pressure.

    Compared on the INTERIOR and modulo an additive constant, which is what a
    pure-Neumann Poisson actually determines.  Two things make the raw padded
    tensors differ without any physics differing:

    * The ghost ring's edge/corner cells are never read by the 5/7-point
      stencil, and the two backends leave different garbage there (the CUDA
      Jacobi's odd-sweep ping-pong copies a zeroed scratch buffer back over p).
    * Each driver's gauge is a mean over the FULL padded tensor, dead corners
      included — so that garbage shifts the whole field by a constant.  The
      solver only ever consumes ∇p, so the constant is immaterial.

    Not bit-exact either way: the CUDA 2-D smoothers are tiled (stale
    cross-block halo reads by design), so the backends take slightly different
    iteration paths to the same answer.
    """
    dtype = torch.float64
    pc, rc, fscale = _solve(method, ndim, dtype, "rbgs", "cpu")
    pg, rg, _ = _solve(method, ndim, dtype, "rbgs", "cuda")

    assert rc < 1e-3 * fscale and rg < 1e-3 * fscale, \
        f"{method} {ndim}D did not converge (cpu {rc:.2e}, cuda {rg:.2e})"

    inner = (slice(1, -1),) * ndim
    d = pc[inner] - pg.cpu()[inner]
    d = d - d.mean()                       # drop the arbitrary Neumann constant
    err = d.abs().max().item()
    scale = pc[inner].abs().max().item()
    assert err < 1e-4 * scale, \
        f"{method} {ndim}D cpu vs cuda interior pressure: {err:.3e} (scale {scale:.3e})"


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


# ── 3-D smoother CPU twins == CUDA kernels ───────────────────────────────────
# Regression gate for the CPU rbgs_sweep_3d bug the Warp removal exposed.  The
# CPU Poisson used to run on WarpMG, so this twin was never compared to CUDA,
# and it was wrong two ways:
#
#   * a bogus row-level ``(i+j) & 1`` guard sat on TOP of the correct per-cell
#     ``(i+j+k) & 1`` colour test, so it skipped half the red cells and half the
#     black cells outright; and
#   * it never refreshed the Neumann ghost ring between the red and black
#     half-sweeps (the CUDA twin does), so the black cells read stale mirrors.
#
# Compared on LIVE cells only — the ones a 7-point stencil can reach (interior +
# face ghosts).  The edge/corner ghosts are never read, and the CUDA Jacobi's
# odd-sweep ping-pong copies a zeroed scratch buffer back over p, so the two
# backends legitimately leave different garbage there.
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

    live = _live_cells(N + 2, 3)
    d = (run("cpu") - run("cuda")).abs()[live].max().item()
    assert d < 1e-12, \
        f"{smoother}_sweep_3d nsmoothing={nsmoothing}: cpu vs cuda {d:.3e}"
