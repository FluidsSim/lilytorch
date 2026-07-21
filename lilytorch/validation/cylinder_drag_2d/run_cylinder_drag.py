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
# Quick toggles — set these and re-run.
# Output filenames embed the toggles so different runs don't overwrite.
# ================================================================
FORCE_METHOD          = os.environ.get("FORCE_METHOD", "eulerian")   # "lagrangian" or "eulerian"
BDIM_MU0_PROJECTION   = os.environ.get("BDIM_MU0_PROJECTION", "1") == "1"  # True = paper-correct decoupled body cells; False = uniform dt/rho
ZERO_PRESSURE_INSIDE  = os.environ.get("ZERO_PRESSURE_INSIDE", "0") == "1"  # True halves the eulerian pressure readout: the delta band needs BOTH sides of the surface (see forces.py note on mu0 masking). Measured pE ratio 0.51 with True.

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
Nx = int(os.environ.get("NX", "512"))
Ny = Nx

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

_tag = f"{FORCE_METHOD}_mu0-{int(BDIM_MU0_PROJECTION)}_zpi-{int(ZERO_PRESSURE_INSIDE)}"
output_base = f"/data/andreaferrario/ns_data/cylinder_drag_validation/{_tag}/"

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
print(f"  Toggles: force_method={FORCE_METHOD}  bdim_mu0_projection={BDIM_MU0_PROJECTION}  "
      f"zero_pressure_inside={ZERO_PRESSURE_INSIDE}")
print(f"  Output: {output_base}")
print()

# ================================================================
# Build parameter dict from YAML template
# ================================================================
pars = yaml2pyobject("lilytorch/examples/standalone/configs/flow_past_cylinder.yaml")

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

# --- Force / projection / kernel toggles ---
pars["solver"]["force_method"]         = FORCE_METHOD
pars["solver"]["bdim_mu0_projection"]  = BDIM_MU0_PROJECTION
pars["solver"]["zero_pressure_inside"] = ZERO_PRESSURE_INSIDE

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

# ================================================================
# Cd summary at key convective times (K&L 1995 reference comparison)
# ================================================================
F_visc_x  = solver.viscous_drag_record[0, 0, :it+1].cpu().numpy()
F_pres_x  = solver.pressure_drag_record[0, 0, :it+1].cpu().numpy()
F_total_x = F_visc_x + F_pres_x
Cd = F_total_x / (0.5 * rho * U**2 * D)
np.save(f"{forces_path}/Cd.npy", Cd)
print(f"\n  ==== Cd ({FORCE_METHOD}, mu0={BDIM_MU0_PROJECTION}, zpi={ZERO_PRESSURE_INSIDE}) ====")
for t_target in (0.5, 1.0, 2.0, 3.0, 5.0, 7.0):
    it_t = int(t_target * R / U / dt)
    if it_t < len(Cd):
        print(f"    Cd at t*={t_target:>4.1f} (iter {it_t}): {Cd[it_t]:+.4f}")
print(f"    Cd mean over last 200 iters:           {Cd[-200:].mean():+.4f}")

# ================================================================
# Inline plotting — same as plot_cylinder_drag.py but with toggle
# info in the title / filenames.
# ================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     11,
    "figure.dpi":         200,
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "lines.linewidth":    1.5,
    "mathtext.fontset":   "cm",
})

# Time axes
time = np.arange(len(Cd)) * dt
tau  = time * U / R
viscous_cd  = F_visc_x  / (0.5 * rho * U**2 * D)
pressure_cd = F_pres_x  / (0.5 * rho * U**2 * D)
total_cd    = Cd

# K&L 1995 reference
ref_path = os.path.join(REPO_ROOT, "data_to_save",
                        "koumoutsatokos_keonard_1995.csv")
if os.path.exists(ref_path):
    ref_data = np.genfromtxt(ref_path, delimiter=",")
    tau_ref, cd_ref = ref_data[:, 0], ref_data[:, 1]
else:
    print(f"  WARNING: reference CSV not found at {ref_path}")
    tau_ref, cd_ref = None, None

C_VISCOUS, C_PRESSURE, C_TOTAL, C_REF = "#42a5f5", "#ef5350", "#212121", "#757575"
toggle_str = (f"{FORCE_METHOD}, mu0_proj={int(BDIM_MU0_PROJECTION)}, "
              f"zpi={int(ZERO_PRESSURE_INSIDE)}")

# Figure 1: components + total + reference
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(tau, viscous_cd,  color=C_VISCOUS,  lw=1.2, label=r"Viscous $C_D$")
ax.plot(tau, pressure_cd, color=C_PRESSURE, lw=1.2, label=r"Pressure $C_D$")
ax.plot(tau, total_cd,    color=C_TOTAL,    lw=1.8, label=r"Total $C_D$ (present)")
if tau_ref is not None:
    ax.plot(tau_ref, cd_ref, color=C_REF, ls="--", lw=1.8,
            label="Koumoutsakos & Leonard (1995)")
ax.set_xlabel(r"Convective time $\tau = tU/R$")
ax.set_ylabel(r"Drag coefficient $C_D$")
ax.set_xlim(0, final_conv_time)
ax.set_ylim(0, 2)
ax.set_title(f"Re={int(Re)}, Nx={Nx}  ({toggle_str})", fontsize=10)
ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.85")
fig.tight_layout()
fig_path = os.path.join(forces_path, f"cylinder_drag_Re{int(Re)}_{_tag}.png")
fig.savefig(fig_path)
plt.close(fig)
print(f"  Saved plot: {fig_path}")

print("Done!")

# Cleanup
del solver, u, v, p
torch.cuda.empty_cache()
gc.collect()
