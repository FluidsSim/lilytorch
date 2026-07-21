"""End-to-end accuracy tests for the FFT-based Poisson solver.

Covers all combinations of boundary condition (free-space, Neumann)
and dimensionality (2-D, 3-D) using manufactured solutions with
analytically known Laplacians.

Run:  pytest lilytorch/tests/test_poisson_fft.py -v
"""
from __future__ import annotations

import pytest
import torch

from lilytorch.src.poisson_fft import PoissonSolverFFT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device():
    """Return the fastest available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def discrete_laplacian_neumann(phi: torch.Tensor, dh: list[float]) -> torch.Tensor:
    """Discrete Laplacian with Neumann (replicate) ghost cells.

    The FFT Neumann solver exactly inverts the *discrete* Laplacian
    (with replicate / Neumann ghost cells), NOT the continuous one.
    We therefore test with the discrete Laplacian of the exact solution.
    """
    ndim = phi.ndim
    Lh = torch.zeros_like(phi)
    for d in range(ndim):
        # Build ghost-padded view along dimension d
        # ghost[-1] = phi[0], ghost[N] = phi[N-1]  (Neumann)
        sl_lo = [slice(None)] * ndim
        sl_hi = [slice(None)] * ndim
        sl_lo[d] = slice(0, 1)  # first element
        sl_hi[d] = slice(-1, None)  # last element
        padded = torch.cat([phi[tuple(sl_lo)], phi, phi[tuple(sl_hi)]], dim=d)
        # Second difference
        sl_m = [slice(None)] * ndim
        sl_0 = [slice(None)] * ndim
        sl_p = [slice(None)] * ndim
        sl_m[d] = slice(0, -2)
        sl_0[d] = slice(1, -1)
        sl_p[d] = slice(2, None)
        Lh = Lh + (padded[tuple(sl_m)] - 2 * padded[tuple(sl_0)]
                   + padded[tuple(sl_p)]) / dh[d] ** 2
    return Lh


def _rel_errors(approx: torch.Tensor, exact: torch.Tensor) -> tuple[float, float]:
    """Return (Linf, L2) relative errors, using max|exact| as scale."""
    scale = exact.abs().max().item()
    diff = torch.abs(approx - exact)
    linf = diff.max().item() / scale
    l2 = torch.sqrt((diff ** 2).mean()).item() / scale
    return linf, l2


# ---------------------------------------------------------------------------
# Free-space tests
# ---------------------------------------------------------------------------

def test_free_2d():
    """2-D free-space: compact bump  φ(r) = exp(-c/(1-r²))  on [−1,1] disk."""
    dtype = torch.float64
    device = _device()

    nx, ny = 512, 128
    x = torch.linspace(0, 5, nx, dtype=dtype, device=device)
    y = torch.linspace(-5, 5, ny, dtype=dtype, device=device)

    ps = PoissonSolverFFT(x, y, overwrite=True)

    X, Y = torch.meshgrid(x, y, indexing="ij")
    X0, Y0 = 2.0, 2.0
    XX, YY = X - X0, Y - Y0
    RR2 = XX ** 2 + YY ** 2
    c = 6.0

    # source = Laplacian of exp(-c/(1-r²))
    f = (
        4 * c * torch.exp(-c / (1 - RR2))
        * (c * RR2 + XX ** 4 + YY ** 4 + 2 * XX ** 2 * YY ** 2 - 1)
        * (-1 + RR2) ** (-4)
    )
    f[RR2 >= 1] = 0

    phi_exact = torch.exp(-c / (1 - RR2))
    phi_exact[RR2 >= 1] = 0

    phi_approx = ps.solve(f)

    linf, l2 = _rel_errors(phi_approx, phi_exact)
    # 8th-order Hejlesen regularisation → ~O(1e-4) for this resolution.
    assert linf < 1e-3, f"2-D free-space Linf error too large: {linf:.2e}"
    assert l2 < 1e-4, f"2-D free-space L2 error too large: {l2:.2e}"


def test_free_3d():
    r"""3-D free-space: Gaussian  φ(r) = exp(-α r²).

    ∇²φ = (4α²r² − 6α) exp(−α r²)
    """
    dtype = torch.float64
    device = _device()

    N = 64
    x = torch.linspace(-5, 5, N, dtype=dtype, device=device)
    y = torch.linspace(-5, 5, N, dtype=dtype, device=device)
    z = torch.linspace(-5, 5, N, dtype=dtype, device=device)

    ps = PoissonSolverFFT(x, y, z=z, overwrite=True)

    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    alpha = 1.0
    R2 = X ** 2 + Y ** 2 + Z ** 2

    phi_exact = torch.exp(-alpha * R2)
    f = (4.0 * alpha ** 2 * R2 - 6.0 * alpha) * torch.exp(-alpha * R2)

    phi_approx = ps.solve(f)

    linf, l2 = _rel_errors(phi_approx, phi_exact)
    # Gaussian (erf) regularisation is O(σ²) → moderate accuracy at N=64.
    assert linf < 0.5, f"3-D free-space Linf error too large: {linf:.2e}"
    assert l2 < 0.1, f"3-D free-space L2 error too large: {l2:.2e}"


# ---------------------------------------------------------------------------
# Neumann (DCT) tests
# ---------------------------------------------------------------------------

def test_neumann_2d():
    r"""2-D Neumann:  φ = cos(πx/Lx)·cos(πy/Ly).

    Cosine modes satisfy ∂φ/∂n = 0 on all boundaries.
    """
    dtype = torch.float64
    device = _device()

    Lx, Ly = 2.0, 3.0
    nx, ny = 128, 128
    hx, hy = Lx / nx, Ly / ny
    x = torch.linspace(hx / 2, Lx - hx / 2, nx, dtype=dtype, device=device)
    y = torch.linspace(hy / 2, Ly - hy / 2, ny, dtype=dtype, device=device)

    ps = PoissonSolverFFT(x, y, bc_type="neumann")

    X, Y = torch.meshgrid(x, y, indexing="ij")
    phi_exact = torch.cos(torch.pi * X / Lx) * torch.cos(torch.pi * Y / Ly)

    # RHS = discrete Laplacian (Neumann ghost cells)
    f = discrete_laplacian_neumann(phi_exact, [hx, hy])

    phi_approx = ps.solve(f)

    # Remove mean — Neumann solution is unique only up to a constant
    phi_approx = phi_approx - phi_approx.mean()
    phi_exact = phi_exact - phi_exact.mean()

    linf, l2 = _rel_errors(phi_approx, phi_exact)
    assert linf < 1e-10, f"2-D Neumann Linf error too large: {linf:.2e}"
    assert l2 < 1e-10, f"2-D Neumann L2 error too large: {l2:.2e}"


def test_neumann_3d():
    r"""3-D Neumann:  φ = cos(πx/Lx)·cos(2πy/Ly)·cos(πz/Lz).

    Mixed wavenumbers exercise the DCT diagonalisation on all axes.
    """
    dtype = torch.float64
    device = _device()

    Lx, Ly, Lz = 2.0, 3.0, 1.5
    N = 64
    hx, hy, hz = Lx / N, Ly / N, Lz / N
    x = torch.linspace(hx / 2, Lx - hx / 2, N, dtype=dtype, device=device)
    y = torch.linspace(hy / 2, Ly - hy / 2, N, dtype=dtype, device=device)
    z = torch.linspace(hz / 2, Lz - hz / 2, N, dtype=dtype, device=device)

    ps = PoissonSolverFFT(x, y, z=z, bc_type="neumann")

    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    kx = torch.pi / Lx
    ky = 2.0 * torch.pi / Ly
    kz = torch.pi / Lz
    phi_exact = torch.cos(kx * X) * torch.cos(ky * Y) * torch.cos(kz * Z)

    # RHS = discrete Laplacian (Neumann ghost cells)
    f = discrete_laplacian_neumann(phi_exact, [hx, hy, hz])

    phi_approx = ps.solve(f)

    phi_approx = phi_approx - phi_approx.mean()
    phi_exact = phi_exact - phi_exact.mean()

    linf, l2 = _rel_errors(phi_approx, phi_exact)
    assert linf < 1e-10, f"3-D Neumann Linf error too large: {linf:.2e}"
    assert l2 < 1e-10, f"3-D Neumann L2 error too large: {l2:.2e}"
