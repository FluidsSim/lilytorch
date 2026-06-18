"""3-D hydrostatic validation of the GFM Poisson solver.

Closed box, water below z=0.3, air (void) above, under gravity, at rest.
Expected steady state:
  * velocity ~ 0,
  * p = 0 in the air (alpha < 0.5),
  * p = rho_w g (0.3 - z) in the water (p=0 at the free surface, increasing
    downward) — buoyancy-producing hydrostatic pressure emerges purely
    from the p=0 free-surface BC.
"""
import math, torch
torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)
from lilytorch.src.poisson_gfm import (
    level_set_height_3d, gfm_grad_3d, _div_of_faces_3d, gfm_solve_cg_3d,
)

g, rho = 9.81, 1000.0
Lx, Ly, Lz = 0.5, 0.3, 0.5
H_water = 0.3  # water fills bottom 0.3 m
h = 0.02  # 2 cm cells
Nx, Ny, Nz = round(Lx / h), round(Ly / h), round(Lz / h)
dt = 0.1 * h / math.sqrt(g * Lz)
c_coeff = dt / rho
print(f"3-D GFM hydrostatic {Nx}x{Ny}x{Nz} dx={h*1e3:.0f}mm dt={dt:.2e}")

xc = (torch.arange(Nx) + 0.5) * h
yc = (torch.arange(Ny) + 0.5) * h
zc = (torch.arange(Nz) + 0.5) * h
alpha = (zc[None, None, :] < H_water).double().expand(Nx, Ny, Nz)

u = torch.zeros(Nx - 1, Ny, Nz)
v = torch.zeros(Nx, Ny - 1, Nz)
w = torch.zeros(Nx, Ny, Nz - 1)

def water_zface(a):
    return 0.5 * (a[:, :, :-1] + a[:, :, 1:]) > 0.5

# Add gravity to w on water z-faces
wz = water_zface(alpha)
w_star = w + torch.where(wz, torch.full_like(w, -g * dt), torch.zeros_like(w))

# Build level set
phi = level_set_height_3d(alpha, h, 0.0)

# Solve GFM Poisson
div = _div_of_faces_3d(u, v, w_star, h)
p = gfm_solve_cg_3d(div / c_coeff, phi, h, n_iter=500, tol=1e-8)

# Expected hydrostatic pressure
Zc_full = zc[None, None, :].expand(Nx, Ny, Nz)  # (Nx, Ny, Nz)
p_expected = rho * g * torch.clamp(H_water - Zc_full, min=0.0)

# Check: p should match hydrostatic in water
water_cc = alpha >= 0.5
p_error = (p - p_expected)[water_cc]
rel_error = p_error.abs().max() / p_expected[water_cc].max()
print(f"Max |p - p_hydro| in water: {float(p_error.abs().max()):.3f} Pa")
print(f"Relative error: {float(rel_error)*100:.2f}%")
print(f"p max (expected {float(p_expected.max()):.1f}): {float(p[water_cc].max()):.1f}")

# Check: p ≈ 0 in air
air_cc = alpha < 0.5
p_air_max = float(p[air_cc].abs().max())
print(f"Max |p| in air: {p_air_max:.2e}")

# Check: velocity correction should give ~0 velocity
gx, gy, gz = gfm_grad_3d(p, phi, h)
u_new = u - c_coeff * gx
v_new = v - c_coeff * gy
w_new = w_star - c_coeff * gz
umax = float(max(u_new.abs().max(), v_new.abs().max(), w_new.abs().max()))
print(f"|u|max after correction: {umax:.2e}")

# Allow ~3% pressure error (half-cell GFM discretization at interface)
success = rel_error < 0.05 and p_air_max < 1e-6 and umax < 1e-6
print(f"\n{'PASS' if success else 'FAIL'}: 3-D GFM hydrostatic validation")
