"""Unit tests for the two-phase VOF helper (:mod:`lilytorch.src.two_phase`).

Run directly (``python -m lilytorch.src.test_two_phase``) or via pytest.
"""

import math
import torch

from lilytorch.src.two_phase import TwoPhase


def _grid(N=40, L=1.0, ndim=2, dtype=torch.float64):
    h = L / N
    coords = [torch.linspace(0.0, L, N, dtype=dtype) for _ in range(ndim)]
    return coords, h


# ---------------------------------------------------------------------------
# Material-field blends (analytic)
# ---------------------------------------------------------------------------
def test_density_and_viscosity_blends():
    (x, y), h = _grid()
    tp = TwoPhase(x, y, h, lambda X, Y: (Y < 0.5).double(),
                  rho_water=1000.0, rho_air=1.0,
                  nu_water=2.0, nu_air=0.5)
    q = tp.recip_density_cc()
    # reciprocal density: 1/ρ_water in water, 1/ρ_air in air
    assert torch.isclose(q[tp.alpha == 1].min(), torch.tensor(1.0 / 1000.0, dtype=q.dtype))
    assert torch.isclose(q[tp.alpha == 0].max(), torch.tensor(1.0 / 1.0, dtype=q.dtype))
    # explicit half-fraction cell: ρ_cc = 500.5 → q = 1/500.5
    tp.alpha.fill_(0.5)
    assert torch.allclose(tp.recip_density_cc(),
                          torch.full_like(tp.alpha, 1.0 / 500.5))
    assert torch.allclose(tp.viscosity_cc(), torch.full_like(tp.alpha, 1.25))


def test_recip_density_face_harmonic():
    """recip_density_face is the arithmetic mean of 1/ρ = reciprocal of the
    harmonic face density."""
    (x, y), h = _grid(N=8)
    # vertical step: left half water, right half air (jump along x = dim 0)
    tp = TwoPhase(x, y, h, lambda X, Y: (X < 0.5).double(),
                  rho_water=1000.0, rho_air=1.0)
    qf = tp.recip_density_face(0)
    # interior all-water / all-air faces reduce to the bulk reciprocal
    assert torch.isclose(qf.min(), torch.tensor(1.0 / 1000.0, dtype=qf.dtype))
    # at the water/air cut face: 0.5*(1/1000 + 1/1) = 0.50050
    cut = qf[(qf > 1.0 / 1000.0 + 1e-9) & (qf < 1.0 / 1.0 - 1e-9)]
    assert cut.numel() > 0
    assert torch.allclose(cut, torch.full_like(cut, 0.5 * (1.0 / 1000.0 + 1.0 / 1.0)))
    # equals the reciprocal of the harmonic density mean 2ρ_iρ_j/(ρ_i+ρ_j)
    harm_density = 2.0 * 1000.0 * 1.0 / (1000.0 + 1.0)
    assert torch.allclose(cut, torch.full_like(cut, 1.0 / harm_density))


def test_arithmetic_face_density_rejected():
    """The legacy arithmetic face-density option is no longer accepted."""
    import pytest
    (x, y), h = _grid(N=8)
    with pytest.raises(TypeError):
        TwoPhase(x, y, h, lambda X, Y: (X < 0.5).double(),
                 face_density="arithmetic")


# ---------------------------------------------------------------------------
# Transport: boundedness + mass conservation
# ---------------------------------------------------------------------------
def _taylor_green(coords, h):
    """Divergence-free velocity from psi = sin(pi x) sin(pi y), zero normal
    velocity on the [0,1]^2 walls (no boundary flux)."""
    X, Y = torch.meshgrid(coords[0], coords[1], indexing="ij")
    u =  math.pi * torch.sin(math.pi * X) * torch.cos(math.pi * Y)   #  d psi/dy
    v = -math.pi * torch.cos(math.pi * X) * torch.sin(math.pi * Y)   # -d psi/dx
    return u, v


def test_boundedness_and_mass_conservation_2d():
    (x, y), h = _grid(N=48)
    cx = cy = 0.5; r = 0.2
    tp = TwoPhase(x, y, h,
                  lambda X, Y: ((X - cx)**2 + (Y - cy)**2 < r**2).double(),
                  rho_water=1000.0, rho_air=1.0)
    u, v = _taylor_green((x, y), h)
    umax = max(u.abs().max().item(), v.abs().max().item())
    dt = 0.2 * h / umax
    V0 = tp.water_volume()
    for _ in range(100):
        tp.advect(u, v, dt=dt)
        assert tp.alpha.min() >= -1e-12 and tp.alpha.max() <= 1.0 + 1e-12
    drift = abs(tp.water_volume() - V0) / V0
    # The Weymouth-Yue scheme conserves volume to round-off for a DISCRETELY
    # divergence-free velocity (as produced by the projection in the real
    # solver — see the dam-break validation's ~round-off vol drift). Here the
    # *analytic* Taylor-Green field sampled at cell centres is not discretely
    # div-free, so the divergence-correction terms leave a small O(h dt Nstep)
    # residual; bound it loosely.
    assert drift < 5e-3, f"water-volume drift too large: {drift:.2e}"


def test_boundedness_3d():
    h = 1.0 / 24
    x = y = z = torch.linspace(0.0, 1.0, 24, dtype=torch.float64)
    tp = TwoPhase(x, y, h, lambda X, Y, Z: (Z < 0.5).double(), z=z)
    u = torch.full_like(tp.alpha, 0.0)
    w = torch.full_like(tp.alpha, 0.1)
    dt = 0.2 * h / 0.1
    for _ in range(20):
        tp.advect(u, u, w, dt=dt)
        assert tp.alpha.min() >= -1e-12 and tp.alpha.max() <= 1.0 + 1e-12


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS  {name}")
    print("All two-phase unit tests passed.")
