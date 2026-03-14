"""
Tests for the sub-grid scale (SGS) / stabilisation models added to AdvDiffSolver.

Background
----------
High-order TVD schemes such as QUICK (FLUXLMT) and ADBQUICKEST have low
numerical dissipation, which can cause spurious oscillations in under-resolved
flows.  Two complementary stabilisation strategies—mirroring what OpenFOAM and
FEniCS do—have been added:

* **Smagorinsky SGS model**  (``sgs_model="smagorinsky"``)
  Classic Smagorinsky (1963) eddy-viscosity closure used in OpenFOAM LES
  (``SmagorinskyModel``).  Adds  ν_t = (Cs·Δ)²·|S̄|  to the molecular viscosity
  where |S̄| is the strain-rate magnitude and Δ = √(dx·dy).

* **Artificial-diffusion model**  (``sgs_model="artificial_diffusion"``)
  SUPG-inspired local artificial diffusion used in FEniCS/DOLFIN advection
  problems.  Adds  ν_art = Cv·h·|u|  proportional to the local velocity
  magnitude and the mesh size h = √(dx·dy).
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from lilytorch.src.adv_diff import AdvDiffSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_solver(N=32, dt=0.001, nu=0.0, method="abdquickest", **kwargs):
    device = torch.device("cpu")
    x = torch.linspace(0.0, 1.0, N)
    y = torch.linspace(0.0, 1.0, N)
    return AdvDiffSolver(device, dt, x, y, nu, method=method, **kwargs)


def gradient_magnitude(u):
    """Mean squared gradient magnitude—lower means more diffused."""
    gx = u[2:, 1:-1] - u[:-2, 1:-1]
    gy = u[1:-1, 2:] - u[1:-1, :-2]
    return (gx ** 2 + gy ** 2).mean().item()


def make_step_profile(N):
    """Square step in the centre of the domain."""
    u = torch.zeros(N, N)
    u[N // 4 : 3 * N // 4, N // 4 : 3 * N // 4] = 1.0
    v = torch.ones(N, N) * 0.3
    return u, v


def run_simulation(solver, N, nt=50):
    u, v = make_step_profile(N)
    for _ in range(nt):
        u, v = solver.solve(u, v)
        solver.set_BCs(u, v)
    return u, v


# ---------------------------------------------------------------------------
# Unit tests for Smagorinsky viscosity
# ---------------------------------------------------------------------------

class TestSmagorinskyViscosity:

    def test_uniform_flow_gives_zero_viscosity(self):
        """Smagorinsky ν_t must vanish for uniform translation (no shear)."""
        N = 32
        s = make_solver(N=N, sgs_model="smagorinsky", Cs=0.17)
        u = torch.ones(N, N)
        v = torch.zeros(N, N)
        nu_t = s.smagorinsky_viscosity(u, v)
        assert nu_t.max().item() < 1e-12, f"Expected ~0, got {nu_t.max().item()}"

    def test_shear_flow_gives_positive_viscosity(self):
        """Smagorinsky ν_t must be strictly positive in a shear flow."""
        N = 32
        s = make_solver(N=N, sgs_model="smagorinsky", Cs=0.17)
        # u = y  → ∂u/∂y ≠ 0
        u = torch.zeros(N, N)
        for j in range(N):
            u[:, j] = float(j) / N
        v = torch.zeros(N, N)
        nu_t = s.smagorinsky_viscosity(u, v)
        assert nu_t.max().item() > 0.0, "Smagorinsky ν_t should be positive in shear flow"
        assert (nu_t >= 0.0).all(), "Smagorinsky ν_t must be non-negative everywhere"

    def test_viscosity_scales_with_cs_squared(self):
        """Doubling Cs should quadruple ν_t (because ν_t ∝ Cs²)."""
        N = 32
        s1 = make_solver(N=N, sgs_model="smagorinsky", Cs=0.1)
        s2 = make_solver(N=N, sgs_model="smagorinsky", Cs=0.2)
        u = torch.zeros(N, N)
        for j in range(N):
            u[:, j] = float(j) / N
        v = torch.zeros(N, N)
        nu_t1 = s1.smagorinsky_viscosity(u, v).mean().item()
        nu_t2 = s2.smagorinsky_viscosity(u, v).mean().item()
        ratio = nu_t2 / nu_t1
        assert abs(ratio - 4.0) < 0.01, f"Expected ratio ~4, got {ratio}"

    def test_output_shape(self):
        """smagorinsky_viscosity must return an (N-2, N-2) tensor."""
        N = 32
        s = make_solver(N=N, sgs_model="smagorinsky")
        u = torch.ones(N, N)
        v = torch.zeros(N, N)
        nu_t = s.smagorinsky_viscosity(u, v)
        assert nu_t.shape == (N - 2, N - 2), f"Unexpected shape {nu_t.shape}"


# ---------------------------------------------------------------------------
# Unit tests for artificial diffusion viscosity
# ---------------------------------------------------------------------------

class TestArtificialDiffusionViscosity:

    def test_zero_velocity_gives_zero_viscosity(self):
        """Artificial diffusion ν_art must vanish when the flow is at rest."""
        N = 32
        s = make_solver(N=N, sgs_model="artificial_diffusion", Cv=0.1)
        u = torch.zeros(N, N)
        v = torch.zeros(N, N)
        nu_art = s.artificial_diffusion_viscosity(u, v)
        assert nu_art.max().item() == 0.0, "ν_art should be 0 for zero velocity"

    def test_positive_velocity_gives_positive_viscosity(self):
        """Artificial diffusion ν_art must be positive when there is flow."""
        N = 32
        s = make_solver(N=N, sgs_model="artificial_diffusion", Cv=0.1)
        u = torch.ones(N, N)
        v = torch.zeros(N, N)
        nu_art = s.artificial_diffusion_viscosity(u, v)
        assert nu_art.min().item() > 0.0, "ν_art should be > 0 for nonzero velocity"

    def test_proportional_to_velocity_magnitude(self):
        """ν_art must double when the velocity doubles (linear scaling)."""
        N = 32
        s = make_solver(N=N, sgs_model="artificial_diffusion", Cv=0.1)
        u1 = torch.ones(N, N)
        v1 = torch.zeros(N, N)
        u2 = 2.0 * torch.ones(N, N)
        v2 = torch.zeros(N, N)
        nu1 = s.artificial_diffusion_viscosity(u1, v1).mean().item()
        nu2 = s.artificial_diffusion_viscosity(u2, v2).mean().item()
        assert abs(nu2 / nu1 - 2.0) < 1e-5, f"Expected ratio ~2, got {nu2/nu1}"

    def test_proportional_to_cv(self):
        """ν_art must scale linearly with Cv."""
        N = 32
        s1 = make_solver(N=N, sgs_model="artificial_diffusion", Cv=0.1)
        s2 = make_solver(N=N, sgs_model="artificial_diffusion", Cv=0.2)
        u = torch.ones(N, N)
        v = torch.zeros(N, N)
        nu1 = s1.artificial_diffusion_viscosity(u, v).mean().item()
        nu2 = s2.artificial_diffusion_viscosity(u, v).mean().item()
        assert abs(nu2 / nu1 - 2.0) < 1e-5, f"Expected ratio ~2, got {nu2/nu1}"

    def test_output_shape(self):
        """artificial_diffusion_viscosity must return an (N-2, N-2) tensor."""
        N = 32
        s = make_solver(N=N, sgs_model="artificial_diffusion")
        u = torch.ones(N, N)
        v = torch.zeros(N, N)
        nu_art = s.artificial_diffusion_viscosity(u, v)
        assert nu_art.shape == (N - 2, N - 2), f"Unexpected shape {nu_art.shape}"


# ---------------------------------------------------------------------------
# Unit tests for compute_nu_sgs dispatcher
# ---------------------------------------------------------------------------

class TestComputeNuSgs:

    def test_none_returns_scalar_zero(self):
        """No SGS model → compute_nu_sgs returns 0."""
        s = make_solver(N=16, sgs_model=None)
        u = torch.ones(16, 16)
        v = torch.zeros(16, 16)
        result = s.compute_nu_sgs(u, v)
        assert result == 0.0

    def test_smagorinsky_dispatches_correctly(self):
        N = 16
        s = make_solver(N=N, sgs_model="smagorinsky")
        u = torch.zeros(N, N)
        for j in range(N):
            u[:, j] = float(j) / N
        v = torch.zeros(N, N)
        nu_sgs = s.compute_nu_sgs(u, v)
        nu_direct = s.smagorinsky_viscosity(u, v)
        assert torch.allclose(nu_sgs, nu_direct)

    def test_artificial_diffusion_dispatches_correctly(self):
        N = 16
        s = make_solver(N=N, sgs_model="artificial_diffusion")
        u = torch.ones(N, N)
        v = torch.zeros(N, N)
        nu_sgs = s.compute_nu_sgs(u, v)
        nu_direct = s.artificial_diffusion_viscosity(u, v)
        assert torch.allclose(nu_sgs, nu_direct)


# ---------------------------------------------------------------------------
# Integration tests: SGS models increase effective diffusion
# ---------------------------------------------------------------------------

class TestSGSIncreaseDiffusion:
    """
    Verify that enabling an SGS model damps under-resolved gradients more
    than the baseline (sgs_model=None) under the same molecular viscosity.
    A square step is advected; we measure mean squared gradient magnitude
    as a proxy for diffusion—lower means smoother / more diffused.
    """

    N  = 48
    NT = 60

    def _run(self, method, sgs_model, **kwargs):
        BC_u = ["D", "D", "D", "D"]
        BC_v = ["D", "D", "D", "D"]
        solver = make_solver(
            N=self.N, dt=0.0005, nu=0.0, method=method,
            sgs_model=sgs_model,
            BC_type_u=BC_u, BC_values_u=[0, 0, 0, 0],
            BC_type_v=BC_v, BC_values_v=[0.3, 0.3, 0.3, 0.3],
            **kwargs,
        )
        u, _ = run_simulation(solver, self.N, nt=self.NT)
        return gradient_magnitude(u)

    @pytest.mark.parametrize("method", ["abdquickest", "quick"])
    def test_smagorinsky_more_diffusive(self, method):
        gm_none = self._run(method, sgs_model=None)
        gm_smag = self._run(method, sgs_model="smagorinsky", Cs=0.17)
        assert gm_smag < gm_none, (
            f"[{method}] Smagorinsky should reduce gradient magnitude "
            f"(got {gm_smag:.6f} vs baseline {gm_none:.6f})"
        )

    @pytest.mark.parametrize("method", ["abdquickest", "quick"])
    def test_artificial_diffusion_more_diffusive(self, method):
        gm_none = self._run(method, sgs_model=None)
        gm_art  = self._run(method, sgs_model="artificial_diffusion", Cv=0.1)
        assert gm_art < gm_none, (
            f"[{method}] Artificial diffusion should reduce gradient magnitude "
            f"(got {gm_art:.6f} vs baseline {gm_none:.6f})"
        )

    @pytest.mark.parametrize("method", ["abdquickest", "quick"])
    def test_larger_coefficient_more_diffusive(self, method):
        """Increasing Cv should produce more diffusion."""
        gm_small = self._run(method, sgs_model="artificial_diffusion", Cv=0.05)
        gm_large = self._run(method, sgs_model="artificial_diffusion", Cv=0.2)
        assert gm_large < gm_small, (
            f"[{method}] Larger Cv should produce more diffusion "
            f"(got {gm_large:.6f} vs {gm_small:.6f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
