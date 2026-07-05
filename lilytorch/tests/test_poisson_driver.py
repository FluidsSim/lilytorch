"""Convergence tests for the Warp-backed Poisson driver.

``poisson_mult.PoissonSolver`` runs the multigrid / MGCG outer driver in
Python with the fine-level smoother + residual on Warp kernels.  These tests
check it converges to tolerance on a manufactured Neumann Poisson, for both
smoothers, 2-D + 3-D, f32 + f64, multigrid + MGCG, and the CUDA-graphed
all-Warp variants (``cuda_graph=True``).

Also asserts independence: with ``torch.ops.lilytorch_kernels`` monkeypatched
to raise, the driver still solves (it touches no custom torch op).
"""
from __future__ import annotations

import pytest
import torch

from lilytorch.src.poisson_mult import PoissonSolver

CUDA = torch.cuda.is_available()
SKIP_NO_CUDA = pytest.mark.skipif(not CUDA, reason="CUDA not available")
DEV = "cuda" if CUDA else "cpu"


def _problem(ndim, dtype, N=48, device=DEV):
    torch.manual_seed(0)
    h = 1.0 / N
    shp = (N,) * ndim
    f = torch.randn(*shp, dtype=dtype, device=device)
    f -= f.mean()                                    # compatible (Neumann) RHS
    pshp = tuple(n + 2 for n in shp)
    p = torch.zeros(pshp, dtype=dtype, device=device)
    o = dict(dtype=dtype, device=device)
    if ndim == 2:
        faces = dict(ch=torch.full((N + 1, N), 1.0, **o),
                     cv=torch.full((N, N + 1), 1.0, **o))
    else:
        faces = dict(ch=torch.full((N + 1, N, N), 1.0, **o),
                     cv=torch.full((N, N + 1, N), 1.0, **o),
                     cw=torch.full((N, N, N + 1), 1.0, **o))
    return h, f, p, faces


def _solve(SolverCls, method, ndim, dtype, smoother):
    h, f, p, faces = _problem(ndim, dtype)
    s = SolverCls(dtype=dtype, device=DEV, h=h, tol=1e-6, max_vcycles=40,
                  max_cycles=50, nsmoothing=2, smoother=smoother, verbose=False)
    _, r = getattr(s, method)(f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    return r.abs().max().item()


@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["solve_multigrid", "solve_mgcg"])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("smoother", ["rbgs", "jacobi"])
def test_warp_poisson_converges(method, ndim, dtype, smoother):
    """The Warp driver converges to a small residual."""
    rw = _solve(PoissonSolver, method, ndim, dtype, smoother)
    assert rw < 1e-3, f"{method} {ndim}D {dtype} {smoother}: warp residual {rw:.2e}"


@SKIP_NO_CUDA
def test_warp_poisson_independent_of_native_ops(monkeypatch):
    """With lilytorch_kernels monkeypatched to raise, the Warp Poisson driver
    still solves (it dispatches no native custom-kernel op)."""
    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError(f"native op {name} must not be called (Warp path)")
    monkeypatch.setattr(torch.ops, "lilytorch_kernels", _Boom())
    rw = _solve(PoissonSolver, "solve_mgcg", 3, torch.float32, "rbgs")
    assert rw < 1e-3, f"independent solve residual {rw:.2e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("smoother", ["rbgs", "jacobi"])
def test_warp_poisson_graphed_multigrid(dtype, smoother):
    """The CUDA-graphed all-Warp 3-D multigrid (cuda_graph=True) converges to
    the same residual as the ungraphed Python-driver path."""
    h, f, p, faces = _problem(3, dtype, N=48)
    rn = _solve(PoissonSolver, "solve_multigrid", 3, dtype, smoother)
    s = PoissonSolver(dtype=dtype, device=DEV, h=h, tol=1e-6, max_vcycles=15,
                    nsmoothing=2, smoother=smoother, verbose=False,
                    cuda_graph=True)
    _, r = s.solve_multigrid(f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    rw = r.abs().max().item()
    assert rw < 1e-3, f"graphed {dtype} {smoother}: residual {rw:.2e}"
    assert rw < 50.0 * max(rn, 1e-8), f"graphed {dtype} {smoother}: {rw:.2e} vs {rn:.2e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("smoother", ["rbgs", "jacobi"])
def test_warp_poisson_graphed_multigrid_2d(dtype, smoother):
    """The CUDA-graphed all-Warp 2-D multigrid (``WarpMG2D``) converges to the
    same residual as the ungraphed Python-driver path — dtype-generic (f32+f64)
    and rbgs/jacobi, matching the 3-D driver."""
    h, f, p, faces = _problem(2, dtype, N=64)
    rn = _solve(PoissonSolver, "solve_multigrid", 2, dtype, smoother)
    s = PoissonSolver(dtype=dtype, device=DEV, h=h, tol=1e-6, max_vcycles=15,
                    nsmoothing=2, smoother=smoother, verbose=False,
                    cuda_graph=True)
    _, r = s.solve_multigrid(f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    rw = r.abs().max().item()
    assert rw < 1e-3, f"graphed 2-D {dtype} {smoother}: residual {rw:.2e}"
    assert rw < 50.0 * max(rn, 1e-8), (
        f"graphed 2-D {dtype} {smoother}: {rw:.2e} vs {rn:.2e}")


@SKIP_NO_CUDA
def test_warp_poisson_graphed_2d_independent(monkeypatch):
    """The graphed 2-D MG replays with no native custom-kernel op."""
    h, f, p, faces = _problem(2, torch.float64, N=64)
    s = PoissonSolver(dtype=torch.float64, device=DEV, h=h, tol=1e-6, max_vcycles=15,
                    nsmoothing=2, smoother="rbgs", verbose=False,
                    cuda_graph=True)
    # warm-up + capture before patching (capture itself is pure-Warp).
    s.solve_multigrid(f.clone(), p, **{k: v.clone() for k, v in faces.items()})

    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError(f"native op {name} must not be called")
    monkeypatch.setattr(torch.ops, "lilytorch_kernels", _Boom())
    _, r = s.solve_multigrid(f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    assert r.abs().max().item() < 1e-3


@SKIP_NO_CUDA
def test_warp_poisson_graphed_independent(monkeypatch):
    """The graphed multigrid touches no native custom-kernel op."""
    h, f, p, faces = _problem(3, torch.float32, N=32)
    s = PoissonSolver(dtype=torch.float32, device=DEV, h=h, tol=1e-6, max_vcycles=10,
                    nsmoothing=2, smoother="rbgs", verbose=False,
                    cuda_graph=True)
    # warm-up / capture before monkeypatch (capture itself is pure-Warp).
    s.solve_multigrid(f.clone(), p.clone(), **{k: v.clone() for k, v in faces.items()})

    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError(f"native op {name} must not be called")
    monkeypatch.setattr(torch.ops, "lilytorch_kernels", _Boom())
    _, r = s.solve_multigrid(f.clone(), p.clone(),
                             **{k: v.clone() for k, v in faces.items()})
    assert r.abs().max().item() < 1e-2


@SKIP_NO_CUDA
@pytest.mark.parametrize("ndim,dtype,smoother", [
    (3, torch.float32, "rbgs"), (3, torch.float64, "rbgs"),
    (3, torch.float32, "jacobi"),
    (2, torch.float64, "rbgs"), (2, torch.float32, "rbgs"),
    (2, torch.float64, "jacobi"), (2, torch.float32, "jacobi")])
def test_warp_poisson_graphed_mgcg(ndim, dtype, smoother):
    """The CUDA-graphed WarpMG MGCG preconditioner (``cuda_graph=True``,
    ``_dispatch_vcycle`` → one-host-launch V-cycle) reaches the same residual
    as the ungraphed MGCG driver (C1)."""
    h, f, p, faces = _problem(ndim, dtype, N=48)
    rn = _solve(PoissonSolver, "solve_mgcg", ndim, dtype, smoother)
    s = PoissonSolver(dtype=dtype, device=DEV, h=h, tol=1e-6, max_vcycles=2,
                    max_cycles=50, nsmoothing=2, smoother=smoother, verbose=False,
                    cuda_graph=True)
    _, r = s.solve_mgcg(f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    rw = r.abs().max().item()
    assert rw < 1e-3, f"graphed MGCG {ndim}D {dtype} {smoother}: residual {rw:.2e}"
    assert rw < 50.0 * max(rn, 1e-8), (
        f"graphed MGCG {ndim}D {dtype} {smoother}: {rw:.2e} vs ungraphed {rn:.2e}")


@SKIP_NO_CUDA
@pytest.mark.parametrize("check_every", [1, 4])
def test_warp_poisson_graphed_mgcg_periodic(check_every):
    """Periodic convergence check (``cg_check_every``) still converges; K=1 is
    the every-iter behaviour, K=4 cuts the per-iter residual sync (C1 point 4)."""
    h, f, p, faces = _problem(3, torch.float32, N=48)
    s = PoissonSolver(dtype=torch.float32, device=DEV, h=h, tol=1e-6, max_vcycles=2,
                    max_cycles=80, nsmoothing=2, smoother="rbgs", verbose=False,
                    cuda_graph=True)
    s.cg_check_every = check_every
    _, r = s.solve_mgcg(f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    assert r.abs().max().item() < 1e-3, f"periodic K={check_every}: {r.abs().max().item():.2e}"


@SKIP_NO_CUDA
def test_warp_poisson_graphed_mgcg_independent(monkeypatch):
    """The graphed MGCG preconditioner replays with no native custom-kernel op."""
    h, f, p, faces = _problem(3, torch.float32, N=32)
    s = PoissonSolver(dtype=torch.float32, device=DEV, h=h, tol=1e-6, max_vcycles=2,
                    max_cycles=50, nsmoothing=2, smoother="rbgs", verbose=False,
                    cuda_graph=True)
    # warm-up / capture before monkeypatch (capture itself is pure-Warp).
    s.solve_mgcg(f.clone(), p.clone(), **{k: v.clone() for k, v in faces.items()})

    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError(f"native op {name} must not be called")
    monkeypatch.setattr(torch.ops, "lilytorch_kernels", _Boom())
    _, r = s.solve_mgcg(f.clone(), p.clone(),
                        **{k: v.clone() for k, v in faces.items()})
    assert r.abs().max().item() < 1e-3


def test_warp_poisson_cpu():
    """CPU end-to-end: the Warp smoother/residual run on CPU from one source."""
    h, f, p, faces = _problem(2, torch.float64, N=24, device="cpu")
    s = PoissonSolver(dtype=torch.float64, device="cpu", h=h, tol=1e-6,
                    max_vcycles=60, nsmoothing=2, smoother="rbgs", verbose=False)
    _, r = s.solve_multigrid(f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    assert r.abs().max().item() < 1e-3


# ── Manufactured-solution (analytic) Poisson ──────────────────────────────
# Solve ∇²p = f with unit face coefficients for the closed-form Neumann
# eigenfunction p*(x) = Π_d cos(2π x_d) on the unit cube (cell-centred), whose
# exact Laplacian is f = -ndim·(2π)²·p*.  The recovered field must match p* to
# the O(h²) discretisation error — a true accuracy check (not just a residual
# or python-vs-warp parity assertion), anchoring the Warp-only solver.
def _mms_problem(ndim, dtype, N, device):
    import math
    h = 1.0 / N
    k = 2.0 * math.pi
    xc = (torch.arange(N, dtype=dtype, device=device) + 0.5) * h
    grids = torch.meshgrid(*([xc] * ndim), indexing="ij")
    p_true = torch.ones_like(grids[0])
    for g in grids:
        p_true = p_true * torch.cos(k * g)
    p_true = p_true - p_true.mean()
    f = (-ndim * k * k) * p_true
    f = f - f.mean()                                  # compatible (Neumann) RHS
    o = dict(dtype=dtype, device=device)
    if ndim == 2:
        faces = dict(ch=torch.ones(N + 1, N, **o), cv=torch.ones(N, N + 1, **o))
    else:
        faces = dict(ch=torch.ones(N + 1, N, N, **o),
                     cv=torch.ones(N, N + 1, N, **o),
                     cw=torch.ones(N, N, N + 1, **o))
    return h, f, p_true, faces


@pytest.mark.parametrize("method", ["solve_multigrid", "solve_mgcg"])
@pytest.mark.parametrize("smoother", ["rbgs", "jacobi"])
@pytest.mark.parametrize("ndim,N,tol_err", [(2, 64, 3e-3), (3, 32, 1.5e-2)])
def test_warp_poisson_manufactured_solution(method, smoother, ndim, N, tol_err):
    """The Warp driver recovers the analytic p* to the O(h²) truncation error,
    on the active device (CPU always; CUDA when present)."""
    h, f, p_true, faces = _mms_problem(ndim, torch.float64, N, DEV)
    s = PoissonSolver(dtype=torch.float64, device=DEV, h=h, tol=1e-10,
                      max_vcycles=300, max_cycles=300, nsmoothing=2,
                      smoother=smoother, verbose=False)
    p0 = torch.zeros(*([N + 2] * ndim), dtype=torch.float64, device=DEV)
    p, _ = getattr(s, method)(f.clone(), p0,
                              **{k: v.clone() for k, v in faces.items()})
    inner = tuple(slice(1, -1) for _ in range(ndim))
    p_rec = p[inner].clone()
    p_rec = p_rec - p_rec.mean()                      # fix the Neumann null space
    err = (p_rec - p_true).abs().max().item()
    assert err < tol_err, f"MMS {method} {ndim}D {smoother}: max |p-p*| = {err:.2e}"


@pytest.mark.parametrize("ndim,N", [(2, 64), (3, 32)])
@pytest.mark.parametrize("smoother", ["rbgs", "jacobi"])
def test_warp_poisson_dirichlet_mask(ndim, N, smoother):
    """Free-surface GFM: with ``dirichlet_mask`` set, the WarpMG driver pins p=0
    in the flagged (air) cells at every level/sweep and still converges to a
    small residual in the fluid cells — the mask-aware V-cycle that replaced the
    torch _vcycle Dirichlet path."""
    dtype = torch.float64
    h = 1.0 / N
    o = dict(dtype=dtype, device=DEV)
    # variable-coefficient faces so the mask sees a non-trivial operator
    if ndim == 2:
        faces = dict(ch=torch.full((N + 1, N), 0.5, **o),
                     cv=torch.full((N, N + 1), 0.5, **o))
    else:
        faces = dict(ch=torch.full((N + 1, N, N), 0.5, **o),
                     cv=torch.full((N, N + 1, N), 0.5, **o),
                     cw=torch.full((N, N, N + 1), 0.5, **o))
    mask = torch.zeros(*([N] * ndim), dtype=torch.bool, device=DEV)
    mask[tuple([slice(N // 2, None)] * ndim)] = True          # an "air" block
    torch.manual_seed(0)
    f = torch.randn(*([N] * ndim), dtype=dtype, device=DEV).masked_fill(mask, 0.0)
    p0 = torch.zeros(*([N + 2] * ndim), dtype=dtype, device=DEV)

    s = PoissonSolver(dtype=dtype, device=DEV, h=h, tol=1e-10, max_vcycles=60,
                      max_cycles=80, nsmoothing=2, smoother=smoother, verbose=False)
    s.dirichlet_mask = mask
    p, r = s.solve_multigrid(f.clone(), p0, **{k: v.clone() for k, v in faces.items()})
    inner = tuple(slice(1, -1) for _ in range(ndim))
    # (1) Dirichlet cells are pinned exactly to zero
    assert p[inner][mask].abs().max().item() == 0.0
    # (2) the solve converged in the fluid region
    rf = r[~mask].abs().max().item()
    assert rf < 1e-6, f"dirichlet {ndim}D {smoother}: fluid residual {rf:.2e}"
