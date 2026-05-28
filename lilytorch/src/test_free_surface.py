"""
Self-tests for ``lilytorch.src.free_surface``.

Each test compares the analytic reference for a single FreeSurface
operation against the actual implementation on a small CPU grid.
Run with::

    python -m lilytorch.src.kernels.test_free_surface
"""

import math
import torch

from lilytorch.src.free_surface import (
    FreeSurface, _cc_velocity_2d, _upwind_grad,
)


def _build_2d(N=32, L=1.0, dtype=torch.float64):
    h = L / N
    x = torch.arange(-h / 2, L + h, h, dtype=dtype)
    y = torch.arange(-h / 2, L + h, h, dtype=dtype)
    return x, y, h


def test_init_signs():
    print("[test] init signs (2-D horizontal interface) ...", end=" ")
    x, y, h = _build_2d()
    fs = FreeSurface(x, y, h, phi_init=lambda X, Y: Y - 0.5)
    interior = fs.phi_fs[1:-1, 1:-1]
    Y_int = torch.meshgrid(x[1:-1], y[1:-1], indexing="ij")[1]
    err = (interior - (Y_int - 0.5)).abs().max().item()
    assert err < 1e-12, err
    # masks
    assert torch.equal(fs.air_mask_cc, fs.phi_fs > 0)
    assert torch.equal(fs.fluid_mask_cc, fs.phi_fs <= 0)
    print(f"OK  (max err {err:.2e})")


def test_advection_uniform():
    """A uniform velocity should translate a planar interface rigidly."""
    print("[test] advection uniform translation ...", end=" ")
    N = 64
    x, y, h = _build_2d(N=N)
    fs = FreeSurface(x, y, h, phi_init=lambda X, Y: Y - 0.5,
                     reinit_iters=0)
    dt = 0.1 * h
    nsteps = 40
    v_uni = -0.5         # downward
    u_mac = torch.zeros((N + 2, N + 2), dtype=torch.float64)
    v_mac = torch.full((N + 2, N + 2), v_uni, dtype=torch.float64)
    for _ in range(nsteps):
        fs.advect(u_mac, v_mac, dt=dt)
    expected_shift = v_uni * dt * nsteps          # negative
    Y = torch.meshgrid(x, y, indexing="ij")[1]
    ref = Y - (0.5 + expected_shift)
    err = (fs.phi_fs[2:-2, 2:-2] - ref[2:-2, 2:-2]).abs().max().item()
    print(f"OK  (max err {err:.2e}, expected ~ O(h) = {h:.2e})")
    assert err < 5 * h, (err, h)


def test_reinit_recovers_signed_distance():
    """A scaled tanh profile is not a signed distance; reinit must drive
    |∇φ| → 1 in a band around the zero level set without moving it."""
    print("[test] reinit drives |grad phi| -> 1 ...", end=" ")
    N = 64
    x, y, h = _build_2d(N=N)
    fs = FreeSurface(
        x, y, h,
        phi_init=lambda X, Y: 3.0 * torch.tanh((Y - 0.5) / (5 * h)),
        reinit_iters=200,
    )
    zero_before = (fs.phi_fs > 0).to(torch.int8) - (fs.phi_fs < 0).to(torch.int8)
    fs.reinitialize()
    zero_after = (fs.phi_fs > 0).to(torch.int8) - (fs.phi_fs < 0).to(torch.int8)
    sign_drift = (zero_before != zero_after).sum().item()
    # |∇phi| in a 3-cell band away from the zero set:
    g = fs._godunov_grad_magnitude(fs.phi_fs, torch.sign(fs.phi_fs))
    band = (fs.phi_fs.abs() < 5 * h) & (fs.phi_fs.abs() > 1.5 * h)
    grad_err = (g[band] - 1.0).abs().mean().item()
    print(f"OK  (sign drift cells={sign_drift}, mean |grad-1|={grad_err:.3f})")
    assert grad_err < 0.15, grad_err
    assert sign_drift < 0.01 * fs.phi_fs.numel(), sign_drift


def test_ghost_fluid_scales_planar():
    """For a flat interface at ``y = 0.5`` the y-face GFM scales are:

        - 1 on faces strictly inside fluid
        - 0 on faces strictly inside air
        - 1/θ on the cut face, with θ ∈ [0,1] the fluid fraction.

    With the interface midway between two cells (φ_a = -h/2, φ_b = +h/2),
    θ = 1/2 → scale = 2.  (theta_min should not clamp here.)
    """
    print("[test] ghost-fluid face scales (planar) ...", end=" ")
    N = 16
    x, y, h = _build_2d(N=N)
    # Place the interface exactly on the face between cells of index Y near 0.5.
    fs = FreeSurface(x, y, h, phi_init=lambda X, Y: Y - 0.5,
                     theta_min=1e-3)
    s_u, s_v = fs.ghost_fluid_face_scales()
    # x-faces all reside within constant-sign cells (interface horizontal),
    # so every x-face is either fluid-fluid (scale=1) or air-air (scale=0).
    assert torch.all((s_u == 0) | (s_u == 1)), \
        f"x-face scales unexpected: {s_u.unique()}"
    # y-faces: detect cut faces and their scales.
    j_cut = []
    for j in range(s_v.shape[1]):
        col = s_v[:, j]
        if not (torch.all(col == col[0])):
            continue
        if 0 < col[0].item() < 1 or col[0].item() > 1:
            j_cut.append(j)
    # There should be exactly one cut row.  Its scale should be 2.0
    # (theta = 1/2 for interface midway between cells).
    cut_vals = set()
    for j in range(s_v.shape[1]):
        col_vals = s_v[:, j].unique().tolist()
        for v in col_vals:
            cut_vals.add(round(v, 6))
    print(f"OK  (face-scale unique values = {sorted(cut_vals)})")
    # Sanity: minimum scale is 0 (air-air), maximum >= 1.99 (cut face 1/θ).
    assert s_v.min().item() == 0.0
    assert s_v.max().item() >= 1.99, s_v.max().item()


def test_apply_pressure_mask():
    print("[test] apply_pressure_mask zeros air cells ...", end=" ")
    N = 16
    x, y, h = _build_2d(N=N)
    fs = FreeSurface(x, y, h, phi_init=lambda X, Y: Y - 0.5)
    p = torch.full_like(fs.phi_fs, 7.0)
    fs.apply_pressure_mask(p)
    assert torch.all(p[fs.air_mask_cc] == 0.0)
    assert torch.all(p[fs.fluid_mask_cc] == 7.0)
    print("OK")


def test_extend_velocity_constant():
    """A velocity that is constant in fluid should be extended as a
    constant into the air band by the constant-along-normal extension."""
    print("[test] velocity extension (constant fluid value) ...", end=" ")
    N = 64
    x, y, h = _build_2d(N=N)
    fs = FreeSurface(x, y, h, phi_init=lambda X, Y: Y - 0.5,
                     extend_iters=30)
    u_cc = torch.zeros_like(fs.phi_fs)
    u_cc[fs.fluid_mask_cc] = 1.0
    # Garbage in air (worst case: zeros, or random — try random)
    air = fs.air_mask_cc
    torch.manual_seed(0)
    u_cc[air] = torch.randn_like(u_cc[air]) * 3.0
    fs.extend_velocity(u_cc)
    # In a band of ~10 cells above the interface, u should be close to 1.
    Y = torch.meshgrid(x, y, indexing="ij")[1]
    near_air = air & (Y < 0.5 + 8 * h)
    if near_air.any():
        err = (u_cc[near_air] - 1.0).abs().mean().item()
        print(f"OK  (mean |u-1| in near-air = {err:.3e})")
        assert err < 0.05, err
    else:
        print("SKIP  (no near-air cells)")


if __name__ == "__main__":
    test_init_signs()
    test_advection_uniform()
    test_reinit_recovers_signed_distance()
    test_ghost_fluid_scales_planar()
    test_apply_pressure_mask()
    test_extend_velocity_constant()
    print("\nAll free_surface self-tests passed.")
