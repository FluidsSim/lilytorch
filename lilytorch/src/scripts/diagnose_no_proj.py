"""
Diagnostic: run 3D steps WITHOUT pressure-Poisson projection.
This isolates whether the blowup comes from advection/BDIM or from the Poisson solver.
"""
import sys, os, torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(SCRIPT_DIR, "..", "..", ".."))

from lilytorch.src.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject

yaml_path = os.path.join(SCRIPT_DIR, "flow_past_sphere_3d.yaml")
pars = yaml2pyobject(yaml_path)
pars["solver"]["nt"] = 10
pars["solver"]["poisson_verbose"] = False
pars["output"]["save_frames"] = False

solver = FluidSolver(pars, dtype=torch.float64, compute_forces=False)
solver.set_initial_conditions()

def report(label, **fields):
    parts = [f"{label:30s}"]
    for name, f in fields.items():
        lo, hi = f.min().item(), f.max().item()
        parts.append(f"  {name}: [{lo:+.6e}, {hi:+.6e}]")
    print(" | ".join(parts))

u = solver.u0.clone()
v = solver.v0.clone()
w = solver.w0.clone()
p = torch.zeros_like(u)

print("=" * 100)
print("  3D SIMULATION — NO PRESSURE PROJECTION")
print("=" * 100)
report("Initial", u=u, v=v, w=w)

N_STEPS = 10
for step in range(N_STEPS):
    t = torch.tensor(step * float(solver.dt), device=solver.device, dtype=solver.dtype)

    # --- body update ---
    solver.composite_body.update(t, step, dt=solver.dt)
    (solver.mu0_all, solver.mu1_all) = solver.composite_body.mu_funcs(solver.composite_body.sdf_val)
    solver.m_m0_all = (1 - solver.mu0_all)

    (solver.mu0_all_u, solver.mu1_all_u) = solver.composite_body.mu_funcs(solver.composite_body.sdf_val_u)
    solver.m_m0_all_u = (1 - solver.mu0_all_u)
    (solver.mu0_all_v, solver.mu1_all_v) = solver.composite_body.mu_funcs(solver.composite_body.sdf_val_v)
    solver.m_m0_all_v = (1 - solver.mu0_all_v)
    (solver.mu0_all_w, solver.mu1_all_w) = solver.composite_body.mu_funcs(solver.composite_body.sdf_val_w)
    solver.m_m0_all_w = (1 - solver.mu0_all_w)

    (_, solver.normal_x, solver.normal_y, solver.normal_z, _) = solver.composite_body.compute_sdf_properties(solver.composite_body.sdf_val)
    (_, solver.normal_x_u, solver.normal_y_u, solver.normal_z_u, _) = solver.composite_body.compute_sdf_properties(solver.composite_body.sdf_val_u)
    (_, solver.normal_x_v, solver.normal_y_v, solver.normal_z_v, _) = solver.composite_body.compute_sdf_properties(solver.composite_body.sdf_val_v)
    (_, solver.normal_x_w, solver.normal_y_w, solver.normal_z_w, _) = solver.composite_body.compute_sdf_properties(solver.composite_body.sdf_val_w)

    # --- advection-diffusion ---
    (uprime, vprime, wprime) = solver.adv_diff_solver.solve(u, v, w)

    # --- BDIM2 ---
    body_u = solver.composite_body.body_u
    body_v = solver.composite_body.body_v
    body_w = solver.composite_body.body_w

    uprime = (solver.mu0_all_u * uprime
              + solver.m_m0_all_u * body_u
              + solver.mu1_all_u * solver.normal_derivative(uprime - body_u,
                    solver.normal_x_u, solver.normal_y_u, solver.normal_z_u))
    vprime = (solver.mu0_all_v * vprime
              + solver.m_m0_all_v * body_v
              + solver.mu1_all_v * solver.normal_derivative(vprime - body_v,
                    solver.normal_x_v, solver.normal_y_v, solver.normal_z_v))
    wprime = (solver.mu0_all_w * wprime
              + solver.m_m0_all_w * body_w
              + solver.mu1_all_w * solver.normal_derivative(wprime - body_w,
                    solver.normal_x_w, solver.normal_y_w, solver.normal_z_w))

    # --- BCs ---
    solver.adv_diff_solver.set_BCs(uprime, vprime, wprime)

    # --- NO PROJECTION — skip Poisson solve entirely ---
    u, v, w = uprime, vprime, wprime

    div = solver.divergence(u, v, w)
    report(f"Step {step:3d}", u=u, v=v, w=w, div=div)

    if torch.isnan(u).any():
        print("*** NaN detected — stopping ***")
        break

print("\nDone.")
