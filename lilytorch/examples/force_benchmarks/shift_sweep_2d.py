"""Decisive experiment: on ONE saved cylinder field (tau=7, Re=550),
evaluate the eulerian viscous/pressure readout as a function of the
viscous-band shift eps_solver, and the lagrangian readout as a function
of its sample offset. If the s->0 limit of the eulerian viscous matches
the lagrangian value, the band shift explains the whole eulerian-vs-
lagrangian viscous gap seen in the live run (vE/vL ~ 0.89 at s=2h).
"""
import numpy as np
import torch

from lilytorch.src.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject

FIELDS = "/data/andreaferrario/ns_data/cylinder_drag_validation/eulerian_mu0-1_zpi-0/uv_field"

Re, R = 550, 0.1
D = 2 * R
nu = 1e-6
U = Re * nu / D
rho = 1e3
xmin, xmax, ymin, ymax = -0.5, 1.5, -1.0, 1.0
Nx = Ny = 512
dx = (xmax - xmin) / Nx
dt = 0.1 * dx / U

pars = yaml2pyobject("lilytorch/examples/standalone/configs/flow_past_cylinder.yaml")
pars["solver"].update(dict(
    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, Nx=Nx, Ny=Ny,
    nu=nu, rho=rho, dt=dt, convection_method="abdquickest", nt=10,
    force_method="eulerian", bdim_mu0_projection=True,
    zero_pressure_inside=False, use_gpu=False,
))
pars["body"]["sdf"] = [f"lambda x, y: circle(x,y,xt=0.0,yt=0.0,r={R})"]
pars["body"]["update_maps"] = [{
    "rotation": "lambda t: 0*torch.sin(t/5)",
    "translation": ["lambda t: 0*torch.sin(t/5)", "lambda t: 0*torch.sin(t/5)"],
}]
pars["boundary_conditions"]["BC_values_u"] = [U, U, 0.0, 0.0]

solver = FluidSolver(pars, dtype=torch.float64, compute_forces=True)
solver.set_initial_conditions()

dev, dtp = solver.device, solver.dtype
u = torch.from_numpy(np.load(f"{FIELDS}/u.npy")).to(dev, dtp)
v = torch.from_numpy(np.load(f"{FIELDS}/v.npy")).to(dev, dtp)
p = torch.from_numpy(np.load(f"{FIELDS}/p.npy")).to(dev, dtp)
print("field shapes:", u.shape, v.shape, p.shape, " solver.eps =", solver.eps, " h =", solver.h)

comp = solver.composite_body
# force the python fallback branch of forces_method2 (sweepable eps)
comp._kernel_static_2d = None

q = 0.5 * rho * U**2 * D
h = solver.h
eps_production = solver.eps

print("\n=== eulerian: sweep viscous-band shift eps_solver (delta width eps_body fixed) ===")
print(f"{'s/h':>5} {'Cd_visc':>9} {'Cd_pres':>9}")
eul = {}
for s_over_h in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    solver.eps = s_over_h * h
    solver.forces_method2(u, v, p, 0)
    fv = float(solver.friction_force_lin_x[0]) / q
    fp = float(solver.pressure_force_x[0]) / q
    eul[s_over_h] = (fv, fp)
    print(f"{s_over_h:5.2f} {fv:9.4f} {fp:9.4f}")
solver.eps = eps_production

print("\n=== lagrangian: sweep sample offset ===")
print(f"{'off/h':>5} {'Cd_visc':>9} {'Cd_pres':>9}")
lag = {}
for o_over_h in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    solver.lagrangian_sample_offset = o_over_h * h
    solver.forces_lagrangian_2d(u, v, p, 0)
    fv = float(solver.friction_force_lin_x[0]) / q
    fp = float(solver.pressure_force_x[0]) / q
    lag[o_over_h] = (fv, fp)
    print(f"{o_over_h:5.2f} {fv:9.4f} {fp:9.4f}")

print("\nrecorded live-run values at tau=7 (native kernel, s=2h):")
print("  eulerian   Cd_visc 0.0697  Cd_pres 0.9363")
print("  lagrangian Cd_visc 0.0782  Cd_pres 0.9619   (offset 0)")
