"""Tiled fused Jacobi (shared-mem multi-sweep) — parity vs native + convergence.

Confirms the smoother gap CLOSES with the Warp 1.14 tile API:
  - nsmoothing=1: bit-close to native jacobi_sweep_2d (both = global Jacobi, no
    stale halos used in the first sweep → tiling-independent), float32 ULP.
  - multi-sweep: residual converges geometrically (valid MG smoother).
GPU-only (the tile/shared-mem path is CUDA).
"""
from __future__ import annotations

import pytest
import torch
import warp as wp

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import jacobi_sweep_2d as nat_jac
    _NATIVE = True
except Exception:
    _NATIVE = False

from lilytorch.warp_poc.warp_poisson_tiled_2d import (
    WarpTiledJacobi2D, WarpTiledRBGS2D, TILE)

SKIP_NO_NATIVE = pytest.mark.skipif(not _NATIVE, reason="native _C.so unavailable")
SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def _problem(N, seed=11):
    torch.manual_seed(seed)
    p = torch.zeros(N + 2, N + 2, dtype=torch.float32, device="cuda:0")
    p[1:-1, 1:-1] = torch.randn(N, N, device="cuda:0")
    f = torch.randn(N, N, dtype=torch.float32, device="cuda:0")
    c = [0.5 + torch.rand(N, N, dtype=torch.float32, device="cuda:0") for _ in range(4)]
    return p, f, c


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("N", [64, 256])
@pytest.mark.parametrize("w", [1.0, 0.8])
def test_tiled_jacobi_ns1_matches_native(N, w):
    """1 sweep = global Jacobi (no stale halos) → matches native to float32 ULP,
    independent of the tile decomposition."""
    p, f, c = _problem(N)
    # Apply Neumann ghosts so both paths see the same domain boundary (native
    # applies it internally; the tiled load reads p's ghosts as-is).  Inter-tile
    # halos are already the true global values for a single sweep.
    p[0, :] = p[1, :]; p[-1, :] = p[-2, :]; p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
    pn = p.clone()
    nat_jac(pn, f, *c, 1e-30, w, 1)
    torch.cuda.synchronize()
    pw = WarpTiledJacobi2D(N).smooth(p, f, c, 1, w)
    wp.synchronize()
    d = (pn[1:-1, 1:-1] - pw[1:-1, 1:-1]).abs().max().item()
    assert d < 1e-5, f"N={N} w={w} tiled-jacobi ns=1 vs native maxdiff {d:.2e}"


def _torch_rbgs_singletile(p, f, c, nsweep):
    """Global red-black GS with stale (loaded-once) Neumann ghosts — the
    reference my single-TILE (32x32) tiled RBGS must reproduce bit-for-bit."""
    p = p.clone()   # caller has already applied Neumann ghosts (kept stale here)
    cp0, cm0, cp1, cm1 = c
    J = cp0 + cm0 + cp1 + cm1
    N = f.shape[0]
    gi = torch.arange(N, device=f.device).view(-1, 1)
    gj = torch.arange(N, device=f.device).view(1, -1)
    red = (gi + gj) % 2 == 0
    for _ in range(nsweep):
        U = (-f + cp0*p[2:, 1:-1] + cm0*p[:-2, 1:-1]
             + cp1*p[1:-1, 2:] + cm1*p[1:-1, :-2]) / J
        p[1:-1, 1:-1] = torch.where(red, U, p[1:-1, 1:-1])
        U2 = (-f + cp0*p[2:, 1:-1] + cm0*p[:-2, 1:-1]
              + cp1*p[1:-1, 2:] + cm1*p[1:-1, :-2]) / J
        p[1:-1, 1:-1] = torch.where(~red, U2, p[1:-1, 1:-1])
    return p


@SKIP_NO_CUDA
@pytest.mark.parametrize("nsweep", [1, 5])
def test_tiled_rbgs_singletile_matches_reference(nsweep):
    """At single-tile size (N=TILE) the fused tiled RBGS = global red-black GS
    with stale ghosts → matches the torch reference to float32 ULP."""
    N = TILE
    p, f, c = _problem(N)
    p[0, :] = p[1, :]; p[-1, :] = p[-2, :]; p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
    pw = WarpTiledRBGS2D(N).smooth(p, f, c, nsweep)
    wp.synchronize()
    ref = _torch_rbgs_singletile(p, f, c, nsweep)
    d = (pw[1:-1, 1:-1] - ref[1:-1, 1:-1]).abs().max().item()
    assert d < 1e-5, f"tiled RBGS ns={nsweep} vs ref maxdiff {d:.2e}"


@SKIP_NO_CUDA
def test_tiled_rbgs_converges():
    """Multi-tile fused tiled RBGS reduces the residual (valid MG smoother)."""
    N = 128
    torch.manual_seed(3)
    c = [0.5 + torch.rand(N, N, dtype=torch.float32, device="cuda:0") for _ in range(4)]
    f = torch.randn(N, N, dtype=torch.float32, device="cuda:0")
    p = torch.zeros(N + 2, N + 2, dtype=torch.float32, device="cuda:0")

    def resid(pp):
        cp0, cm0, cp1, cm1 = c
        J = cp0 + cm0 + cp1 + cm1
        s = (cp0*pp[2:, 1:-1] + cm0*pp[:-2, 1:-1] + cp1*pp[1:-1, 2:] + cm1*pp[1:-1, :-2])
        return (f - s + J*pp[1:-1, 1:-1]).norm().item()

    sm = WarpTiledRBGS2D(N)
    r0 = resid(p)
    for _ in range(15):
        p = sm.smooth(p, f, c, 2)
        p[0, :] = p[1, :]; p[-1, :] = p[-2, :]; p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
    wp.synchronize()
    assert resid(p) < 0.5 * r0, f"residual {r0:.3e} -> {resid(p):.3e}"


@SKIP_NO_CUDA
def test_tiled_jacobi_converges():
    """Multi-sweep fused tiled Jacobi reduces the residual on a manufactured
    diagonally-dominant problem (valid MG smoother)."""
    N = 128
    torch.manual_seed(3)
    # diagonally dominant so weighted Jacobi converges
    c = [0.2 + 0.1 * torch.rand(N, N, dtype=torch.float32, device="cuda:0") for _ in range(4)]
    f = torch.randn(N, N, dtype=torch.float32, device="cuda:0")
    p = torch.zeros(N + 2, N + 2, dtype=torch.float32, device="cuda:0")

    def resid(pp):
        cp0, cm0, cp1, cm1 = c
        J = cp0 + cm0 + cp1 + cm1
        s = (cp0 * pp[2:, 1:-1] + cm0 * pp[:-2, 1:-1]
             + cp1 * pp[1:-1, 2:] + cm1 * pp[1:-1, :-2])
        return (f - s + J * pp[1:-1, 1:-1]).norm().item()

    sm = WarpTiledJacobi2D(N)
    r0 = resid(p)
    for _ in range(20):
        p = sm.smooth(p, f, c, 4, 0.8)
        # Neumann ghost refresh (clamp) so the next block-load halo is consistent
        p[0, :] = p[1, :]; p[-1, :] = p[-2, :]; p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
    wp.synchronize()
    r1 = resid(p)
    assert r1 < 0.5 * r0, f"residual {r0:.3e} -> {r1:.3e}"
