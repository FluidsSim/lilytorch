"""
Error analysis for flow past cylinder in 2D — Maertens & Weymouth (2015) style.

Setup:
  - Diameter D = 120 (grid-unit scale)
  - Domain  30D × 30D  (3600 × 3600)
  - Grid spacings dx = dy ∈ {3, 4, 6, 8, 12}  →  Nx = Ny ∈ {1200, 900, 600, 450, 300}
  - Re = 550  (same as the previous test)
  - CFL = 0.1  →  dt = 0.1 * dx / U
  - Run for 3 convective times  →  t_stop = 3 * D / U = 360
  - Cylinder centred at (0, 0) in domain [-1800, 1800]²

Reference solution: finest grid (dx = 3).
"""

from lilytorch.src.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm
import os
import gc

# ================================================================
# Physical / geometric parameters
# ================================================================
D  = 120          # cylinder diameter (grid-unit scale)
R  = D / 2        # radius = 60
Re = 550          # Reynolds number (matches previous test)
U  = 1.0          # free-stream velocity
nu = U * D / Re   # kinematic viscosity  ≈ 0.2182

rho = 1.0         # density

# Domain: 30D × 30D, centred at origin
half_L = 15 * D   # 1800
xmin, xmax = -half_L, half_L
ymin, ymax = -half_L, half_L

# Cylinder centre
cx, cy = 0.0, 0.0

# Time: 3 convective times
t_conv = D / U
n_conv = 3.0
t_stop = n_conv * t_conv   # 360

# ================================================================
# Grid spacings (coarse → fine)
# ================================================================
dxs = [12, 8, 6, 4, 3]

convection_method = "abdquickest"
output_base = "/data/andreaferrario/ns_data/flow_past_cylinder_error_tests_MW/" + convection_method

print(f"{'='*70}")
print(f"  Maertens & Weymouth (2015) style convergence test")
print(f"{'='*70}")
print(f"  D = {D},  R = {R},  Re = {Re},  U = {U},  nu = {nu:.6f}")
print(f"  Domain = [{xmin}, {xmax}] × [{ymin}, {ymax}]  ({int(xmax-xmin)} × {int(ymax-ymin)})")
print(f"  Cylinder centre = ({cx}, {cy})")
print(f"  t_stop = {t_stop} ({n_conv} convective times)")
print(f"  Grid spacings: {dxs}")
print()

for dx in dxs:
    Nx = int((xmax - xmin) / dx)
    Ny = Nx

    print(f"\n{'='*60}")
    print(f"  dx = dy = {dx}  →  Nx = Ny = {Nx}")
    print(f"{'='*60}\n")

    pars = yaml2pyobject("lilytorch/src/scripts/flow_past_cylinder.yaml")

    # --- Domain ---
    pars["solver"]["xmin"] = xmin
    pars["solver"]["xmax"] = xmax
    pars["solver"]["ymin"] = ymin
    pars["solver"]["ymax"] = ymax
    pars["solver"]["Nx"]   = Nx
    pars["solver"]["Ny"]   = Ny

    # --- Physics ---
    pars["solver"]["nu"]  = nu
    pars["solver"]["rho"] = rho

    # --- Body: cylinder centred at (cx, cy) ---
    pars["body"]["sdf"] = [
        f"lambda x, y: circle(x,y,xt={cx},yt={cy},r={R})"
    ]

    # --- Boundary conditions (same U on both x-faces, zero v on y-faces) ---
    pars["boundary_conditions"]["BC_values_u"] = [U, U, 0.0, 0.0]
    pars["boundary_conditions"]["BC_values_v"] = [0.0, 0.0, 0.0, 0.0]

    # --- Time stepping ---
    dt = 0.1 * dx / U
    pars["solver"]["dt"]                = dt
    pars["solver"]["convection_method"] = convection_method
    pars["solver"]["nt"]                = int(t_stop / dt) + 1

    # --- Output ---
    save_path = f"{output_base}/{dx}/"
    pars["output"]["save_frames"] = True
    pars["output"]["save_every"]  = max(50, pars["solver"]["nt"] // 10)

    print(f"  Re = {U * D / nu:.1f}")
    print(f"  dx = {dx},  dt = {dt:.4f},  nt = {pars['solver']['nt']}")
    print(f"  D/dx = {D/dx:.1f} cells per diameter")
    print(f"  eps = 2*dx = {2*dx}  (eps/D = {2*dx/D:.3f})")

    # =========== Run simulation ===========
    solver = FluidSolver(pars, dtype=torch.float32, compute_forces=True)
    solver.save_path = save_path
    os.makedirs(solver.save_path, exist_ok=True)
    solver.set_initial_conditions()
    u = solver.u0
    v = solver.v0
    p = solver.p0

    for it in tqdm(range(0, solver.nt), desc=f"dx={dx}"):
        t = it * solver.dt
        (u, v, p, stop_sim) = solver.step_(u, v, p, it, t)

    # Save final fields
    uv_path = f"{save_path}/uv_field"
    os.makedirs(uv_path, exist_ok=True)
    np.save(f"{uv_path}/u", u.cpu().numpy())
    np.save(f"{uv_path}/v", v.cpu().numpy())
    np.save(f"{uv_path}/p", p.cpu().numpy())

    print(f"  Saved fields to {uv_path}")

    # Free memory
    del solver, u, v, p
    torch.cuda.empty_cache()
    gc.collect()

print("\n\nAll simulations complete!")
print(f"Results saved under: {output_base}")
