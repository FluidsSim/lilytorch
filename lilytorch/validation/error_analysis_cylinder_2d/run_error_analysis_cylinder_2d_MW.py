"""
Error analysis for flow past cylinder in 2D — Maertens & Weymouth (2015) style.

Setup (non-dimensional, following MW2015 Section 3.1):
  - Diameter D = 1
  - Domain  30D × 30D  (30 × 30)
  - Grid: power-of-2 Nx for multigrid compatibility
      Nx ∈ {128, 256, 512, 1024, 2048}   →   dx = 30/Nx
      D/dx ∈ {4.27, 8.53, 17.07, 34.13, 68.27}
  - NOTE: D/dx < 16 is pre-asymptotic for BDIM2; 2nd-order rates emerge at
    D/dx ≥ 16 (Nx ≥ 512 on this 30D domain).  For a cleaner convergence study
    reduce the domain to 10D × 10D (half_L = 5) to shift the range to
    D/dx ∈ {12.8 … 102.4} for the same Nx values.
      Refinement ratio = 2 between successive levels
  - Re = 100  (MW2015 Section 3.1: "circular cylinder at Re = 100")
  - CFL = 0.1  →  dt = 0.1 * dx / U
  - Run for 3 convective times  →  t_stop = 3 * D / U = 3
  - Cylinder centred at (0, 0) in domain [-15, 15]²
  - ε = 2·dx  (kernel half-width)

Reference solution: finest grid (Nx = 1024, D/dx ≈ 34).
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
# Physical / geometric parameters  (non-dimensional, D = 1)
# ================================================================
D  = 1            # cylinder diameter (non-dimensional)
R  = D / 2        # radius = 0.5
Re = 100          # Reynolds number (MW2015 Section 3.1)
U  = 1.0          # free-stream velocity
nu = U * D / Re   # kinematic viscosity = 0.01

rho = 1.0         # density

half_L = 5 * D
xmin, xmax = -half_L, half_L
ymin, ymax = -half_L, half_L

# Cylinder centre
cx, cy = 0.0, 0.0

# Time: 3 convective times
t_conv = D / U
n_conv = 3.0
t_stop = n_conv * t_conv   # 3

# ================================================================
# Grid sizes (power-of-2 for multigrid; refinement ratio = 2)
#   dx = domain_size / Nx ,  D/dx = Nx / domain_size
# ================================================================
nxs = [128, 256, 512, 1024, 2048]          # coarse → fine
domain_size = xmax - xmin       # 30

convection_method = "abdquickest"
output_base = "/data/andreaferrario/ns_data/flow_past_cylinder_error_tests_MW/" + convection_method

print(f"{'='*70}")
print(f"  Maertens & Weymouth (2015) style convergence test")
print(f"{'='*70}")
print(f"  D = {D},  R = {R},  Re = {Re},  U = {U},  nu = {nu:.6f}")
print(f"  Domain = [{xmin}, {xmax}] × [{ymin}, {ymax}]  ({domain_size} × {domain_size})")
print(f"  Cylinder centre = ({cx}, {cy})")
print(f"  t_stop = {t_stop} ({n_conv} convective times)")
print(f"  Grid sizes Nx: {nxs}")
print()

for Nx in nxs:
    dx = domain_size / Nx
    Ny = Nx

    print(f"\n{'='*60}")
    print(f"  Nx = {Nx}  →  dx = {dx:.6f}  →  D/dx = {D/dx:.2f}")
    print(f"{'='*60}\n")

    pars = yaml2pyobject("lilytorch/src/configs/flow_past_cylinder.yaml")

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

    # --- Solver method: CompositeBodyAnalytical does not use BDIMhandler,
    #     so the kernel path (_kernel_step bookkeeping) is unavailable.
    pars["solver"]["solver_method"] = "python"

    # --- Poisson solver: increase iterations for larger grids ---
    pars["solver"]["poisson_method"]     = "multigrid"
    pars["solver"]["poisson_max_cycles"] = 30           # V-cycles (was 10)
    pars["solver"]["poisson_nsmoothing"] = 10           # Jacobi sweeps per level (was 5)
    pars["solver"]["poisson_tol"]        = 1e-8         # relax tolerance slightly

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
    save_path = f"{output_base}/Nx{Nx}/"
    pars["output"]["save_frames"] = True
    pars["output"]["save_every"]  = max(50, pars["solver"]["nt"] // 10)

    print(f"  Re = {U * D / nu:.1f}")
    print(f"  Nx = {Nx},  dx = {dx:.6f},  dt = {dt:.6f},  nt = {pars['solver']['nt']}")
    print(f"  D/dx = {D/dx:.1f} cells per diameter")
    print(f"  eps = 2·dx = {2*dx:.6f}")

    # =========== Run simulation ===========
    solver = FluidSolver(pars, dtype=torch.float32, compute_forces=False)
    solver.save_path = save_path
    os.makedirs(solver.save_path, exist_ok=True)
    solver.set_initial_conditions()
    u = solver.u0
    v = solver.v0
    p = solver.p0

    for it in tqdm(range(0, solver.nt), desc=f"Nx={Nx}, D/dx={D/dx:.0f}"):
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
