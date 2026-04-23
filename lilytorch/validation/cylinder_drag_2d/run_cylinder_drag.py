"""
Drag coefficient validation for impulsively started flow past a cylinder.

Reproduces Koumoutsakos & Leonard (1995) at Re = 550.
Non-dimensional setup:
  - Diameter D = 2R,  R = 0.1
  - Domain [-0.5, 1.5] × [-1, 1]   (Nx × Ny = 512 × 512)
  - u_inlet = Re · ν / D
  - CFL = 0.1 · dx / u_inlet
  - Run to 7 convective times  (t_stop = 7 · R / u_inlet)
"""

from lilytorch.src.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject
import torch
import numpy as np
import os
import gc
from tqdm import tqdm

# ================================================================
# Physical / geometric parameters
# ================================================================
Re = 550
R  = 0.1                   # cylinder radius
D  = 2 * R                 # diameter
nu = 1e-6                  # kinematic viscosity (from YAML default)
U  = Re * nu / D           # inlet velocity
rho = 1e3                  # density

# Domain
xmin, xmax = -0.5, 1.5
ymin, ymax = -1.0, 1.0
Nx = 512
Ny = 512

# Cylinder centre
cx, cy = 0.0, 0.0

# Time
final_conv_time = 7.0
t_stop = final_conv_time * R / U

# CFL
dx = (xmax - xmin) / Nx
dt = 0.1 * dx / U
nt = int(t_stop / dt) + 1

convection_method = "abdquickest"

output_base = "/data/andreaferrario/ns_data/cylinder_drag_validation/"

# ================================================================
# Setup
# ================================================================
print(f"{'=' * 60}")
print(f"  Cylinder drag validation — Koumoutsakos & Leonard (1995)")
print(f"{'=' * 60}")
print(f"  Re = {Re},  D = {D},  R = {R},  U = {U:.6f}")
print(f"  nu = {nu},  rho = {rho}")
print(f"  Domain = [{xmin}, {xmax}] × [{ymin}, {ymax}]")
print(f"  Nx = {Nx},  Ny = {Ny},  dx = {dx:.6f}")
print(f"  dt = {dt:.6e},  nt = {nt}")
print(f"  t_stop = {t_stop:.4f}  ({final_conv_time} convective times)")
print(f"  Convection method: {convection_method}")
print()

# ================================================================
# Build parameter dict from YAML template
# ================================================================
pars = yaml2pyobject("lilytorch/src/scripts/configs/flow_past_cylinder.yaml")

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

# --- Time stepping ---
pars["solver"]["dt"]                = dt
pars["solver"]["convection_method"] = convection_method
pars["solver"]["nt"]                = nt

# --- Poisson solver ---
pars["solver"]["poisson_verbose"] = True

# --- Body ---
pars["body"]["sdf"] = [
    f"lambda x, y: circle(x,y,xt={cx},yt={cy},r={R})"
]
pars["body"]["update_maps"] = [{
    "rotation": "lambda t: 0*torch.sin(t/5)",
    "translation": [
        "lambda t: 0*torch.sin(t/5)",
        "lambda t: 0*torch.sin(t/5)"
    ]
}]

# --- Boundary conditions ---
pars["boundary_conditions"]["BC_values_u"] = [U, U, 0.0, 0.0]

# --- Output ---
save_path = output_base
pars["output"]["save_frames"] = True
pars["output"]["save_every"]  = max(50, nt // 20)

print(f"  save_every = {pars['output']['save_every']}")
print(f"  Number of iterations: {nt}")

# ================================================================
# Run simulation
# ================================================================
solver = FluidSolver(pars, dtype=torch.float64, compute_forces=True)
solver.save_path = save_path
os.makedirs(solver.save_path, exist_ok=True)
solver.set_initial_conditions()

u = solver.u0
v = solver.v0
p = solver.p0

for it in tqdm(range(solver.nt), desc=f"Re={Re}, Nx={Nx}"):
    t = it * solver.dt
    (u, v, p, stop_sim) = solver.step_(u, v, p, it, t)
    if stop_sim:
        print(f"  Simulation terminated at iteration {it}")
        break

# Save force records
forces_path = save_path
os.makedirs(forces_path, exist_ok=True)
np.save(f"{forces_path}/viscous_drags", solver.viscous_drag_record.cpu().numpy())
np.save(f"{forces_path}/pressure_drags", solver.pressure_drag_record.cpu().numpy())

# Save final fields
uv_path = f"{save_path}/uv_field"
os.makedirs(uv_path, exist_ok=True)
np.save(f"{uv_path}/u", u.cpu().numpy())
np.save(f"{uv_path}/v", v.cpu().numpy())
np.save(f"{uv_path}/p", p.cpu().numpy())

# Save run metadata
metadata = {
    "Re": Re, "R": R, "D": D, "U": U,
    "nu": nu, "rho": rho,
    "Nx": Nx, "Ny": Ny, "dx": dx, "dt": float(dt),
    "nt": nt, "convection_method": convection_method,
    "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
}
np.savez(f"{forces_path}/metadata.npz", **metadata)

print(f"\n  Saved force records + fields to {save_path}")

# Cleanup
del solver, u, v, p
torch.cuda.empty_cache()
gc.collect()

print("Done!")
