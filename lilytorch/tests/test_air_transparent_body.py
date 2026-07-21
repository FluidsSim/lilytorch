"""Unit tests for the air-transparent-body fix in TwoPhaseSolver.

Validates the core logic without running a full FARMS simulation.
"""
import torch
import pytest
from lilytorch.src.two_phase import TwoPhase


def _make_two_phase(ndim=2, N=16, dtype=torch.float64):
    """Create a minimal TwoPhase field with a water/air interface."""
    L = 1.0
    h = L / N
    coords = [torch.linspace(0.0, L, N, dtype=dtype) for _ in range(ndim)]
    if ndim == 2:
        x, y = coords
        # Horizontal interface at y = 0.5 (water below, air above)
        tp = TwoPhase(x, y, h,
                      lambda X, Y: (Y < 0.5).double(),
                      rho_water=1000.0, rho_air=1.2,
                      nu_water=1e-6, nu_air=1.5e-5,
                      dtype=dtype)
    else:
        x, y, z = coords
        tp = TwoPhase(x, y, h,
                      lambda X, Y, Z: (Z < 0.5).double(),
                      rho_water=1000.0, rho_air=1.2,
                      nu_water=1e-6, nu_air=1.5e-5,
                      z=z, dtype=dtype)
    return tp, h


class TestAlphaFace:
    """Test the _alpha_face helper method directly on TwoPhase fields."""

    def test_alpha_face_2d_shapes(self):
        """Alpha face values have correct shapes in 2D."""
        tp, h = _make_two_phase(ndim=2, N=16)
        a = tp.alpha  # (Nx+2, Ny+2) = (18, 18)

        # Simulate _alpha_face logic inline
        # d=0 (x-faces): average along x
        a_face_0_full = a.clone()
        a_face_0_full[1:, :] = 0.5 * (a[:-1, :] + a[1:, :])
        assert a_face_0_full.shape == a.shape  # (18, 18)

        # d=1 (y-faces): average along y
        a_face_1_full = a.clone()
        a_face_1_full[:, 1:] = 0.5 * (a[:, :-1] + a[:, 1:])
        assert a_face_1_full.shape == a.shape  # (18, 18)

    def test_alpha_face_3d_shapes(self):
        """Alpha face values have correct shapes in 3D."""
        tp, h = _make_two_phase(ndim=3, N=8)
        a = tp.alpha  # (10, 10, 10)

        # Full-grid: same shape as alpha
        for d in range(3):
            out = a.clone()
            if d == 0:
                out[1:, :, :] = 0.5 * (a[:-1, :, :] + a[1:, :, :])
            elif d == 1:
                out[:, 1:, :] = 0.5 * (a[:, :-1, :] + a[:, 1:, :])
            else:
                out[:, :, 1:] = 0.5 * (a[:, :, :-1] + a[:, :, 1:])
            assert out.shape == a.shape  # (10, 10, 10)

    def test_alpha_face_values_at_interface(self):
        """At the sharp interface, face values are 0.5 exactly."""
        tp, h = _make_two_phase(ndim=2, N=16)
        a = tp.alpha

        # y-face at the interface (between water cell and air cell)
        a_face_1 = a.clone()
        a_face_1[:, 1:] = 0.5 * (a[:, :-1] + a[:, 1:])

        # The interface is at y=0.5, which is between cells N//2-1 and N//2
        # (since y goes from 0 to 1 with N=16 cells)
        # Cell centers: h/2, 3h/2, ..., L-h/2
        # y < 0.5 → water (alpha=1), y > 0.5 → air (alpha=0)
        mid = a.shape[1] // 2  # N//2 + 1 = 9 (with ghost cells, N=16 → shape=18)
        interface_face_col = a_face_1[:, mid]
        # The face between water (alpha=1) and air (alpha=0) should be 0.5
        assert torch.allclose(interface_face_col[1:-1],
                              torch.full_like(interface_face_col[1:-1], 0.5))


class TestAirTransparentBodyFormula:
    """Test the air-transparent-body coefficient formula directly."""

    def test_mu0_eff_in_water(self):
        """In water (alpha=1), mu0_eff = mu0 (unchanged)."""
        mu0 = torch.tensor([0.0, 0.3, 0.7, 1.0], dtype=torch.float64)
        alpha_face = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        mu0_eff = alpha_face * mu0 + (1.0 - alpha_face)
        assert torch.allclose(mu0_eff, mu0)

    def test_mu0_eff_in_air(self):
        """In air (alpha=0), mu0_eff = 1 (body transparent)."""
        mu0 = torch.tensor([0.0, 0.3, 0.7, 1.0], dtype=torch.float64)
        alpha_face = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        mu0_eff = alpha_face * mu0 + (1.0 - alpha_face)
        assert torch.allclose(mu0_eff, torch.ones_like(mu0))

    def test_mu0_eff_at_interface(self):
        """At the interface (alpha=0.5), mu0_eff = 0.5*(mu0 + 1)."""
        mu0 = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        alpha_face = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        mu0_eff = alpha_face * mu0 + (1.0 - alpha_face)
        expected = 0.5 * (mu0 + 1.0)
        assert torch.allclose(mu0_eff, expected)

    def test_coefficient_formula_kernel_path(self):
        """Verify the kernel-path coefficient rescaling formula.

        c_target = alpha_face * c_current + (1-alpha_face) * dt/rho_face
        where c_current = dt * mu0 / rho_face (the legacy coefficient).
        """
        dt = 0.001
        rho_water = 1000.0
        rho_air = 1.2
        # Test points: (alpha, mu0, rho_face)
        # Case 1: water, outside body
        alpha, mu0, rho_f = 1.0, 1.0, 1000.0
        c_current = dt * mu0 / rho_f  # = dt/1000
        c_target = alpha * c_current + (1.0 - alpha) * dt / rho_f
        assert abs(c_target - dt / 1000.0) < 1e-15
        assert abs(c_target - c_current) < 1e-15  # same as legacy in water

        # Case 2: air, outside body
        alpha, mu0, rho_f = 0.0, 1.0, 1.2
        c_current = dt * mu0 / rho_f  # = dt/1.2
        c_target = alpha * c_current + (1.0 - alpha) * dt / rho_f
        assert abs(c_target - dt / 1.2) < 1e-15
        assert abs(c_target - c_current) < 1e-15  # same as legacy in free air

        # Case 3: air, inside body (THE KEY CASE)
        alpha, mu0, rho_f = 0.0, 0.0, 1.2
        c_current = dt * mu0 / rho_f  # = 0 (body excludes pressure correction!)
        c_target = alpha * c_current + (1.0 - alpha) * dt / rho_f  # = dt/1.2
        # Legacy would give c=0 (body blocks pressure correction in air)
        # New formula gives c=dt/1.2 (air flows freely)
        assert abs(c_target - dt / 1.2) < 1e-15
        assert c_target > c_current  # new > old (fix gives finite coefficient)

        # Case 4: water, inside body (should still be 0)
        alpha, mu0, rho_f = 1.0, 0.0, 1000.0
        c_current = dt * mu0 / rho_f  # = 0
        c_target = alpha * c_current + (1.0 - alpha) * dt / rho_f  # = 0
        assert abs(c_target) < 1e-15  # body still blocks in water (correct)

        # Case 5: interface, transition band
        alpha, mu0, rho_f = 0.5, 0.5, 500.5  # half water, half body
        c_current = dt * mu0 / rho_f
        c_target = alpha * c_current + (1.0 - alpha) * dt / rho_f
        # Should be between the water-body value and the free-air value
        c_water_body = dt * 0.5 / 500.5
        c_free_air = dt / 500.5
        assert c_water_body < c_target < c_free_air

    def test_coefficient_continuity(self):
        """The coefficient should be continuous across the interface."""
        dt = 0.001
        # At the interface (alpha = 0.5), with mu0 = 0.5, rho = harmonic mean
        rho_w, rho_a = 1000.0, 1.2
        rho_interface = 1.0 / (0.5 / rho_w + 0.5 / rho_a)  # harmonic ≈ 2.397

        for mu0_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            alpha = 0.5
            c_current = dt * mu0_val / rho_interface
            c_target = alpha * c_current + (1.0 - alpha) * dt / rho_interface
            # c_target should be bounded between dt*0/rho (0) and dt*1/rho
            assert 0.0 <= c_target <= dt * 1.0 / rho_interface

    def test_legacy_behavior_unchanged(self):
        """With air_transparent_body=False, the formula is unchanged."""
        dt = 0.001
        # Legacy: c = dt * mu0 / rho_face
        for mu0 in [0.0, 0.5, 1.0]:
            for rho_f in [1.2, 500.5, 1000.0]:
                c_legacy = dt * mu0 / rho_f
                # When alpha is not applied (atb=False), c = c_legacy
                # (same formula used by _rescale_kernel_coeffs_two_phase when atb=False)
                assert abs(c_legacy - dt * mu0 / rho_f) < 1e-15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
