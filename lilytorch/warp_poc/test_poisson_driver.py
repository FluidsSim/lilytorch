"""Parity: ``src_warp`` Poisson driver (Warp smoother+residual) vs native.

The ``src_warp.poisson_mult.PoissonSolver`` runs the multigrid / MGCG outer
driver in Python with the fine-level smoother + residual on Warp kernels (no
``lilytorch_kernels`` op).  This test checks it converges to the same residual
as the native Python-path solver (``use_kernels=False``) on a manufactured
Neumann Poisson, for both smoothers, 2-D + 3-D, f32 + f64, and multigrid + MGCG.

Parity is residual-level (both reach the tolerance), NOT bit-exact — the native
2-D fine smoother is a tiled stale-halo variant while the Warp port is a global
red-black sweep; the V-cycle converges to the same solution either way.

Also asserts independence: with ``torch.ops.lilytorch_kernels`` monkeypatched to
raise, the Warp driver still solves (it touches no native custom-kernel op).
"""
from __future__ import annotations

import pytest
import torch

from lilytorch.src.poisson_mult import PoissonSolver as NativePoisson
from lilytorch.src_warp.poisson_mult import PoissonSolver as WarpPoisson

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
                  max_cycles=50, nsmoothing=2, smoother=smoother, verbose=False,
                  use_kernels=False)
    _, r = getattr(s, method)(f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    return r.abs().max().item()


@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["solve_multigrid", "solve_mgcg"])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("smoother", ["rbgs", "jacobi"])
def test_warp_poisson_converges(method, ndim, dtype, smoother):
    """Warp driver reaches the same order-of-magnitude residual as native."""
    rn = _solve(NativePoisson, method, ndim, dtype, smoother)
    rw = _solve(WarpPoisson, method, ndim, dtype, smoother)
    # Both must have converged to a small residual; the Warp residual must be in
    # the same ballpark as native (different smoother variant → not identical).
    assert rw < 1e-3, f"{method} {ndim}D {dtype} {smoother}: warp residual {rw:.2e}"
    assert rw < 50.0 * max(rn, 1e-8), (
        f"{method} {ndim}D {dtype} {smoother}: warp {rw:.2e} vs native {rn:.2e}")


@SKIP_NO_CUDA
def test_warp_poisson_independent_of_native_ops(monkeypatch):
    """With lilytorch_kernels monkeypatched to raise, the Warp Poisson driver
    still solves (it dispatches no native custom-kernel op)."""
    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError(f"native op {name} must not be called (Warp path)")
    monkeypatch.setattr(torch.ops, "lilytorch_kernels", _Boom())
    rw = _solve(WarpPoisson, "solve_mgcg", 3, torch.float32, "rbgs")
    assert rw < 1e-3, f"independent solve residual {rw:.2e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("smoother", ["rbgs", "jacobi"])
def test_warp_poisson_graphed_multigrid(dtype, smoother):
    """The CUDA-graphed all-Warp 3-D multigrid (cuda_graph=True) converges to the
    same residual as the native Python-path solver."""
    h, f, p, faces = _problem(3, dtype, N=48)
    rn = _solve(NativePoisson, "solve_multigrid", 3, dtype, smoother)
    s = WarpPoisson(dtype=dtype, device=DEV, h=h, tol=1e-6, max_vcycles=15,
                    nsmoothing=2, smoother=smoother, verbose=False,
                    use_kernels=True, cuda_graph=True)
    _, r = s.solve_multigrid(f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    rw = r.abs().max().item()
    assert rw < 1e-3, f"graphed {dtype} {smoother}: residual {rw:.2e}"
    assert rw < 50.0 * max(rn, 1e-8), f"graphed {dtype} {smoother}: {rw:.2e} vs {rn:.2e}"


@SKIP_NO_CUDA
def test_warp_poisson_graphed_multigrid_2d():
    """The CUDA-graphed all-Warp 2-D multigrid (``WarpMG2D``, f64+rbgs — the 2-D
    eel target) converges to the same residual as the native Python-path solver."""
    h, f, p, faces = _problem(2, torch.float64, N=64)
    rn = _solve(NativePoisson, "solve_multigrid", 2, torch.float64, "rbgs")
    s = WarpPoisson(dtype=torch.float64, device=DEV, h=h, tol=1e-6, max_vcycles=15,
                    nsmoothing=2, smoother="rbgs", verbose=False,
                    use_kernels=True, cuda_graph=True)
    _, r = s.solve_multigrid(f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    rw = r.abs().max().item()
    assert rw < 1e-3, f"graphed 2-D: residual {rw:.2e}"
    assert rw < 50.0 * max(rn, 1e-8), f"graphed 2-D: {rw:.2e} vs {rn:.2e}"


@SKIP_NO_CUDA
def test_warp_poisson_graphed_2d_independent(monkeypatch):
    """The graphed 2-D MG replays with no native custom-kernel op."""
    h, f, p, faces = _problem(2, torch.float64, N=64)
    s = WarpPoisson(dtype=torch.float64, device=DEV, h=h, tol=1e-6, max_vcycles=15,
                    nsmoothing=2, smoother="rbgs", verbose=False,
                    use_kernels=True, cuda_graph=True)
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
    s = WarpPoisson(dtype=torch.float32, device=DEV, h=h, tol=1e-6, max_vcycles=10,
                    nsmoothing=2, smoother="rbgs", verbose=False,
                    use_kernels=True, cuda_graph=True)
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
    (3, torch.float32, "jacobi"), (2, torch.float64, "rbgs")])
def test_warp_poisson_graphed_mgcg(ndim, dtype, smoother):
    """The CUDA-graphed WarpMG MGCG preconditioner (``cuda_graph=True``,
    ``_dispatch_vcycle`` → one-host-launch V-cycle) reaches the same residual as
    native MGCG (C1)."""
    h, f, p, faces = _problem(ndim, dtype, N=48)
    rn = _solve(NativePoisson, "solve_mgcg", ndim, dtype, smoother)
    s = WarpPoisson(dtype=dtype, device=DEV, h=h, tol=1e-6, max_vcycles=2,
                    max_cycles=50, nsmoothing=2, smoother=smoother, verbose=False,
                    use_kernels=True, cuda_graph=True)
    _, r = s.solve_mgcg(f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    rw = r.abs().max().item()
    assert rw < 1e-3, f"graphed MGCG {ndim}D {dtype} {smoother}: residual {rw:.2e}"
    assert rw < 50.0 * max(rn, 1e-8), (
        f"graphed MGCG {ndim}D {dtype} {smoother}: {rw:.2e} vs native {rn:.2e}")


@SKIP_NO_CUDA
@pytest.mark.parametrize("check_every", [1, 4])
def test_warp_poisson_graphed_mgcg_periodic(check_every):
    """Periodic convergence check (``cg_check_every``) still converges; K=1 is the
    native every-iter behaviour, K=4 cuts the per-iter residual sync (C1 point 4)."""
    h, f, p, faces = _problem(3, torch.float32, N=48)
    s = WarpPoisson(dtype=torch.float32, device=DEV, h=h, tol=1e-6, max_vcycles=2,
                    max_cycles=80, nsmoothing=2, smoother="rbgs", verbose=False,
                    use_kernels=True, cuda_graph=True)
    s.cg_check_every = check_every
    _, r = s.solve_mgcg(f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    assert r.abs().max().item() < 1e-3, f"periodic K={check_every}: {r.abs().max().item():.2e}"


@SKIP_NO_CUDA
def test_warp_poisson_graphed_mgcg_independent(monkeypatch):
    """The graphed MGCG preconditioner replays with no native custom-kernel op."""
    h, f, p, faces = _problem(3, torch.float32, N=32)
    s = WarpPoisson(dtype=torch.float32, device=DEV, h=h, tol=1e-6, max_vcycles=2,
                    max_cycles=50, nsmoothing=2, smoother="rbgs", verbose=False,
                    use_kernels=True, cuda_graph=True)
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
    s = WarpPoisson(dtype=torch.float64, device="cpu", h=h, tol=1e-6,
                    max_vcycles=60, nsmoothing=2, smoother="rbgs", verbose=False,
                    use_kernels=False)
    _, r = s.solve_multigrid(f.clone(), p, **{k: v.clone() for k, v in faces.items()})
    assert r.abs().max().item() < 1e-3
