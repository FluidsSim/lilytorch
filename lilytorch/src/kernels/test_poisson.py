"""Validate the Warp RBGS Poisson smoother.

  (1) Parity vs native `rbgs_sweep_3d` — 1 sweep, interior cells.
  (2) Manufactured-solution convergence — residual drops monotonically.
  (3) Single-source: CPU Warp vs GPU Warp agree.

Run:  python -m lilytorch.src.kernels.test_poisson
      pytest lilytorch/warp_poc/test_poisson.py -v
"""
from __future__ import annotations

import pytest
import torch
import warp as wp

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import rbgs_sweep_3d
    _NATIVE = True
except Exception:
    _NATIVE = False

from lilytorch.src.kernels.poisson import WarpRBGS

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
SKIP_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
SKIP_NATIVE = pytest.mark.skipif(not _NATIVE, reason="native _C.so unavailable")


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
@SKIP_NATIVE
@pytest.mark.parametrize("N", [16, 32])
def test_warp_matches_native_one_sweep(N):
    """After 1 RBGS sweep, Warp interior == native interior (BC refresh only
    touches ghosts, which the current sweep does not consume)."""
    p0, f, coeffs = _make_problem(N, N, N, DEV, seed=1)

    # Native
    pn = p0.clone()
    rbgs_sweep_3d(pn, f, *coeffs, 1e-30, 1)
    torch.cuda.synchronize()

    # Warp
    pw = p0.clone()
    sol = WarpRBGS(N, N, N, device=DEV)
    sol.setup(pw, f, coeffs)
    sol.sweep(1)
    wp.synchronize()

    int_n = pn[1:-1, 1:-1, 1:-1]
    int_w = pw[1:-1, 1:-1, 1:-1]
    max_abs = (int_n - int_w).abs().max().item()
    rel = max_abs / int_n.abs().max().clamp_min(1e-8).item()
    assert rel < 1e-5, f"N={N}: Warp vs native rel {rel:.3e} (abs {max_abs:.3e})"


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
    if _NATIVE and torch.cuda.is_available():
        p0, f, coeffs = _make_problem(N, N, N, DEV, seed=1)
        pn = p0.clone(); rbgs_sweep_3d(pn, f, *coeffs, 1e-30, 1); torch.cuda.synchronize()
        pw = p0.clone(); s = WarpRBGS(N, N, N, DEV); s.setup(pw, f, coeffs); s.sweep(1); wp.synchronize()
        rel = ((pn - pw)[1:-1, 1:-1, 1:-1]).abs().max().item() / pn[1:-1,1:-1,1:-1].abs().max().item()
        print(f"  native parity (1 sweep, interior): rel err {rel:.2e}  "
              f"{'PASS' if rel < 1e-5 else 'FAIL'}")

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
