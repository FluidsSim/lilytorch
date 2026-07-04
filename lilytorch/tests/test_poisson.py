"""Validate the Warp RBGS Poisson smoother.

  (1) Manufactured-solution convergence — residual drops monotonically.
  (2) Single-source: CPU Warp vs GPU Warp agree.

Run:  python -m lilytorch.src.test_poisson
      pytest lilytorch/src/kernels/test_poisson.py -v
"""
from __future__ import annotations

import pytest
import torch
import warp as wp

from lilytorch.src.poisson import WarpRBGS, WarpRBGS2D

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
SKIP_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def _make_problem(Nx, Ny, Nz, device, seed=0):
    """Random SPD-ish variable-coefficient 7-point problem (padded p)."""
    g = torch.Generator(device=device).manual_seed(seed)
    p = torch.zeros((Nx + 2, Ny + 2, Nz + 2), dtype=torch.float32, device=device)
    p[1:-1, 1:-1, 1:-1] = torch.rand((Nx, Ny, Nz), generator=g, device=device)
    f = torch.rand((Nx, Ny, Nz), generator=g, device=device) - 0.5
    # positive face coefficients → diagonally dominant
    coeffs = [0.5 + torch.rand((Nx, Ny, Nz), generator=g, device=device)
              for _ in range(6)]
    return p, f, coeffs


@SKIP_CUDA
def test_residual_decreases():
    """RBGS smoothing reduces the residual monotonically (solver correctness)."""
    N = 24
    p, f, coeffs = _make_problem(N, N, N, DEV, seed=2)
    sol = WarpRBGS(N, N, N, device=DEV)
    sol.setup(p, f, coeffs)
    r0 = sol.residual_norm()
    sol.sweep(50)
    wp.synchronize()
    r1 = sol.residual_norm()
    assert r1 < 0.2 * r0, f"residual not reduced enough: {r0:.3e} → {r1:.3e}"


@SKIP_CUDA
def test_cpu_gpu_single_source():
    """Same @wp.kernel on CPU and GPU gives the same answer (single-source)."""
    N = 16
    # Build ONE problem on CPU, copy to GPU — both must solve identical data
    # (torch RNG differs across devices for the same seed, so generate once).
    p0, f0, coeffs0 = _make_problem(N, N, N, "cpu", seed=3)

    def run(dev):
        p = p0.to(dev).clone()
        f = f0.to(dev).clone()
        coeffs = [c.to(dev).clone() for c in coeffs0]
        sol = WarpRBGS(N, N, N, device=dev)
        sol.setup(p, f, coeffs)
        sol.sweep(5)
        wp.synchronize()
        return p[1:-1, 1:-1, 1:-1].cpu().clone()

    cpu = run("cpu")
    gpu = run("cuda:0")
    d = (cpu - gpu).abs().max().item()
    assert d < 1e-4, f"CPU vs GPU diverge by {d:.3e}"


def _smoke():
    print("\nWarp Poisson smoother validation")
    print("=" * 50)
    N = 24
    p, f, coeffs = _make_problem(N, N, N, DEV, seed=2)
    s = WarpRBGS(N, N, N, DEV); s.setup(p, f, coeffs)
    r0 = s.residual_norm()
    for it in (10, 25, 50, 100):
        s.sweep(it - (0 if it == 10 else _smoke._prev)); _smoke._prev = it
        print(f"  residual after {it:3d} sweeps: {s.residual_norm():.4e}")
    print(f"  (started at {r0:.4e})")
_smoke._prev = 0


if __name__ == "__main__":
    _smoke()


# ═════════════════════════════════════════════════════════════════════════════
#  2-D smoother (WarpRBGS2D) — merged from the former test_poisson_2d.py
# ═════════════════════════════════════════════════════════════════════════════
SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def _problem(Nx, Ny, dev, seed=11):
    """Build a random positive-coefficient problem ONCE on CPU then .to(dev)."""
    torch.manual_seed(seed)
    p = torch.zeros((Nx + 2, Ny + 2), dtype=torch.float32)
    p[1:-1, 1:-1] = torch.randn(Nx, Ny)
    f = torch.randn(Nx, Ny, dtype=torch.float32)
    coeffs = [0.5 + torch.rand(Nx, Ny, dtype=torch.float32) for _ in range(4)]
    return (p.to(dev), f.to(dev), [c.to(dev) for c in coeffs])


def _warp_run(prob, dev, kind, n=1, w=1.0):
    p, f, coeffs = prob
    p = p.clone()
    s = WarpRBGS2D(p.shape[0] - 2, p.shape[1] - 2, device=dev)
    s.setup(p, f, coeffs)
    if kind == "rbgs":
        s.sweep(n)
    else:
        s.jacobi(n, w)
    wp.synchronize()
    return wp.to_torch(s.p).reshape(p.shape).clone()


@SKIP_NO_CUDA
def test_rbgs_converges():
    """Global Warp RBGS reduces the residual geometrically (manufactured Poisson)."""
    Nx, Ny = 64, 64
    prob = _problem(Nx, Ny, "cuda:0")
    p, f, coeffs = prob
    s = WarpRBGS2D(Nx, Ny, device="cuda:0")
    s.setup(p.clone(), f, coeffs)
    r0 = s.residual_norm()
    s.sweep(50)
    r1 = s.residual_norm()
    assert r1 < 0.5 * r0, f"residual {r0:.3e} -> {r1:.3e}"


@SKIP_NO_CUDA
def test_jacobi_converges():
    """Damped Jacobi also contracts (the smoother the graphed 2-D MG uses)."""
    Nx, Ny = 64, 64
    prob = _problem(Nx, Ny, "cuda:0")
    p, f, coeffs = prob
    s = WarpRBGS2D(Nx, Ny, device="cuda:0")
    s.setup(p.clone(), f, coeffs)
    r0 = s.residual_norm()
    s.jacobi(80, 0.8)
    r1 = s.residual_norm()
    assert r1 < 0.5 * r0, f"residual {r0:.3e} -> {r1:.3e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("kind", ["rbgs", "jacobi"])
def test_cpu_eq_gpu(kind):
    Nx, Ny = 48, 40
    pc = _warp_run(_problem(Nx, Ny, "cpu"), "cpu", kind, 3)
    pg = _warp_run(_problem(Nx, Ny, "cuda:0"), "cuda:0", kind, 3)
    d = (pc[1:-1, 1:-1] - pg[1:-1, 1:-1].cpu()).abs().max().item()
    assert d < 1e-4, f"{kind} cpu vs gpu {d:.3e}"
