"""
Diagnostic script: walk through the FIRST 3D time step sub-operation by sub-operation
and report field ranges at every stage to pinpoint where the blowup originates.
"""

import sys, os, torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(SCRIPT_DIR, "..", "..", ".."))

from lilytorch.src.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject

yaml_path = os.path.join(SCRIPT_DIR, "flow_past_sphere_3d.yaml")
pars = yaml2pyobject(yaml_path)
# Override for quick diagnostics
pars["solver"]["nt"] = 1
pars["solver"]["poisson_verbose"] = True
pars["output"]["save_frames"] = False

print("=" * 70)
print("  3D FIRST-STEP DIAGNOSTIC")
print("=" * 70)

solver = FluidSolver(pars, dtype=torch.float64, compute_forces=False)
solver.set_initial_conditions()

def report(label, **fields):
    print(f"\n--- {label} ---")
    for name, f in fields.items():
        lo, hi = f.min().item(), f.max().item()
        mn = f.mean().item()
        print(f"  {name:10s}  min={lo:+14.6e}  max={hi:+14.6e}  mean={mn:+14.6e}")

# ---- initial conditions ----
u = solver.u0.clone()
v = solver.v0.clone()
w = solver.w0.clone()
p = solver.p0.clone()
report("Initial conditions", u=u, v=v, w=w, p=p)

# ---- Step 0a: update body ----
iteration = 0
t = torch.tensor(0.0, device=solver.device, dtype=solver.dtype)
solver.composite_body.update(t, iteration, dt=solver.dt)

(solver.mu0_all,  solver.mu1_all)  = solver.composite_body.mu_funcs(solver.composite_body.sdf_val)
solver.m_m0_all                    = (1-solver.mu0_all)

(solver.mu0_all_u, solver.mu1_all_u) = solver.composite_body.mu_funcs(solver.composite_body.sdf_val_u)
solver.m_m0_all_u = (1-solver.mu0_all_u)

(solver.mu0_all_v, solver.mu1_all_v) = solver.composite_body.mu_funcs(solver.composite_body.sdf_val_v)
solver.m_m0_all_v = (1-solver.mu0_all_v)

(_, solver.normal_x, solver.normal_y, solver.normal_z, _) = solver.composite_body.compute_sdf_properties(solver.composite_body.sdf_val)
(_, solver.normal_x_u, solver.normal_y_u, solver.normal_z_u, _) = solver.composite_body.compute_sdf_properties(solver.composite_body.sdf_val_u)
(_, solver.normal_x_v, solver.normal_y_v, solver.normal_z_v, _) = solver.composite_body.compute_sdf_properties(solver.composite_body.sdf_val_v)

(solver.mu0_all_w, solver.mu1_all_w) = solver.composite_body.mu_funcs(solver.composite_body.sdf_val_w)
solver.m_m0_all_w = (1-solver.mu0_all_w)
(_, solver.normal_x_w, solver.normal_y_w, solver.normal_z_w, _) = solver.composite_body.compute_sdf_properties(solver.composite_body.sdf_val_w)

report("Body fields",
       mu0=solver.mu0_all, mu1=solver.mu1_all,
       mu0_u=solver.mu0_all_u, mu0_v=solver.mu0_all_v, mu0_w=solver.mu0_all_w,
       sdf=solver.composite_body.sdf_val)

# ---- Step 0b: adv_diff_solver.solve ----
(uprime, vprime, wprime) = solver.adv_diff_solver.solve(u, v, w)
report("After adv_diff solve", uprime=uprime, vprime=vprime, wprime=wprime)

# ---- Step 0c: BDIM2 ----
body_u = solver.composite_body.body_u
body_v = solver.composite_body.body_v
body_w = solver.composite_body.body_w

report("Body velocities", body_u=body_u, body_v=body_v, body_w=body_w)

# normal derivative
nd_u = solver.normal_derivative(uprime - body_u, solver.normal_x_u, solver.normal_y_u, solver.normal_z_u)
nd_v = solver.normal_derivative(vprime - body_v, solver.normal_x_v, solver.normal_y_v, solver.normal_z_v)
nd_w = solver.normal_derivative(wprime - body_w, solver.normal_x_w, solver.normal_y_w, solver.normal_z_w)
report("Normal derivative terms", nd_u=nd_u, nd_v=nd_v, nd_w=nd_w)

uprime_bdim = solver.mu0_all_u * uprime + solver.m_m0_all_u * body_u + solver.mu1_all_u * nd_u
vprime_bdim = solver.mu0_all_v * vprime + solver.m_m0_all_v * body_v + solver.mu1_all_v * nd_v
wprime_bdim = solver.mu0_all_w * wprime + solver.m_m0_all_w * body_w + solver.mu1_all_w * nd_w
report("After BDIM2", u=uprime_bdim, v=vprime_bdim, w=wprime_bdim)

# ---- Step 0d: set_BCs ----
solver.adv_diff_solver.set_BCs(uprime_bdim, vprime_bdim, wprime_bdim)
report("After set_BCs", u=uprime_bdim, v=vprime_bdim, w=wprime_bdim)

# ---- Step 0e: divergence before projection ----
div_pre = solver.divergence(uprime_bdim, vprime_bdim, wprime_bdim)
report("Divergence before projection", div=div_pre)

# ---- Step 0f: project ----
print("\n--- Projection ---")

coeff = solver.dt / solver.rho
ch = coeff * solver.mu0_all_u
cv = coeff * solver.mu0_all_v
cw = coeff * solver.mu0_all_w

report("Projection coefficients", ch=ch, cv=cv, cw=cw)

# Solve poisson
print("\nSolving Poisson equation...")
p_new, r = solver.poisson_solver.solve_multigrid(
    div_pre[1:-1, 1:-1, 1:-1],
    torch.zeros_like(u),
    coeff * torch.ones_like(u),
    ch=ch[1:, 1:-1, 1:-1],
    cv=cv[1:-1, 1:, 1:-1],
    cw=cw[1:-1, 1:-1, 1:],
)
report("Pressure from Poisson", p=p_new, residual=r)

# pressure gradient
(p_x, p_y, p_z) = solver.gradient(p_new)
report("Pressure gradient", px=p_x, py=p_y, pz=p_z)

# velocity correction
u_corr = uprime_bdim - ch * p_x
v_corr = vprime_bdim - cv * p_y
w_corr = wprime_bdim - cw * p_z
report("After projection (final)", u=u_corr, v=v_corr, w=w_corr, p=p_new)

# divergence after projection
div_post = solver.divergence(u_corr, v_corr, w_corr)
report("Divergence AFTER projection", div=div_post)

print("\n" + "=" * 70)
print("  DIAGNOSTIC COMPLETE")
print("=" * 70)
