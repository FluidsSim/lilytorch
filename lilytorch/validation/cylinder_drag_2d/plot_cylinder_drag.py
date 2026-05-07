"""
Paper-quality drag coefficient plot for impulsively started flow past a cylinder.

Compares BDIM2 results against Koumoutsakos & Leonard (1995) at Re = 550.
Generates publication-ready PDF + PNG figures.

Usage
-----
    python plot_cylinder_drag.py
    python plot_cylinder_drag.py --data_dir /path/to/simulation/output
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Parse arguments ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

parser = argparse.ArgumentParser(description="Cylinder drag validation plot")
parser.add_argument("--data_dir", type=str,
                    default="/data/andreaferrario/ns_data/cylinder_drag_validation/",
                    help="Directory containing viscous_drags.npy / pressure_drags.npy")
parser.add_argument("--out_dir", type=str, default=None,
                    help="Directory for output figures (default: same as data_dir)")
args = parser.parse_args()

data_dir = args.data_dir
out_dir  = args.out_dir if args.out_dir else data_dir
os.makedirs(out_dir, exist_ok=True)

# ── Publication style ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.5,
    "text.usetex":        False,
    "mathtext.fontset":   "cm",
})

# ── Load simulation metadata ────────────────────────────────────────
meta_path = os.path.join(data_dir, "metadata.npz")
if os.path.exists(meta_path):
    meta = dict(np.load(meta_path, allow_pickle=True))
    Re  = float(meta["Re"])
    R   = float(meta["R"])
    D   = float(meta["D"])
    U   = float(meta["U"])
    rho = float(meta["rho"])
    dt  = float(meta["dt"])
    nt  = int(meta["nt"])
    Nx  = int(meta["Nx"])
    convection_method = str(meta["convection_method"])
else:
    # Fallback defaults matching run_cylinder_drag.py
    Re  = 550
    R   = 0.1
    D   = 2 * R
    nu  = 1e-6
    U   = Re * nu / D
    rho = 1e3
    dt  = None
    nt  = None
    Nx  = 512
    convection_method = "abdquickest"

# ── Load force records ───────────────────────────────────────────────
viscous_forces  = np.load(os.path.join(data_dir, "viscous_drags.npy"))
pressure_forces = np.load(os.path.join(data_dir, "pressure_drags.npy"))

# Shape: (n_bodies, n_force_comp, nt)
nt_actual = viscous_forces.shape[-1]
if dt is None:
    raise ValueError("Cannot reconstruct time axis without metadata. "
                     "Re-run the simulation or provide metadata.npz.")

time = np.arange(nt_actual) * dt

# Drag coefficients  Cd = 2F / (ρ D U²)
viscous_cd  = 2 * viscous_forces[0, 0, :]  / (rho * D * U**2)
pressure_cd = 2 * pressure_forces[0, 0, :] / (rho * D * U**2)
total_cd    = viscous_cd + pressure_cd

# Convective time  τ = t U / R
tau = time * U / R

# ── Load reference data ──────────────────────────────────────────────
ref_path = os.path.join(REPO_ROOT, "data_to_save",
                        "koumoutsatokos_keonard_1995.csv")
if not os.path.exists(ref_path):
    # Try relative to script directory
    ref_path = os.path.join(SCRIPT_DIR, "..", "..", "..",
                            "data_to_save",
                            "koumoutsatokos_keonard_1995.csv")
ref_data = np.genfromtxt(ref_path, delimiter=",")
tau_ref  = ref_data[:, 0]
cd_ref   = ref_data[:, 1]

# ── Colours ──────────────────────────────────────────────────────────
C_VISCOUS  = "#42a5f5"   # blue
C_PRESSURE = "#ef5350"   # red
C_TOTAL    = "#212121"   # near-black
C_REF      = "#757575"   # grey

# =====================================================================
#  Figure 1 — Main drag coefficient comparison
# =====================================================================
fig, ax = plt.subplots(figsize=(5.5, 3.5))

ax.plot(tau, viscous_cd,  color=C_VISCOUS,  ls="-",  lw=1.2,
        label=r"Viscous $C_D$")
ax.plot(tau, pressure_cd, color=C_PRESSURE, ls="-",  lw=1.2,
        label=r"Pressure $C_D$")
ax.plot(tau, total_cd,    color=C_TOTAL,    ls="-",  lw=1.8,
        label=r"Total $C_D$ (present)")
ax.plot(tau_ref, cd_ref,  color=C_REF,      ls="--", lw=1.8,
        label="Koumoutsakos \& Leonard (1995)")

ax.set_xlabel(r"Convective time $\tau = tU/R$")
ax.set_ylabel(r"Drag coefficient $C_D$")
ax.set_xlim(0, 7)
ax.set_ylim(0, 2)

ax.legend(loc="upper right", frameon=True, framealpha=0.9,
          edgecolor="0.85", fancybox=False)

# Annotate Re
ax.text(0.03, 0.03, rf"$Re = {int(Re)}$,  $N_x = {Nx}$",
        transform=ax.transAxes, fontsize=9, color="0.35",
        verticalalignment="bottom")

fig.tight_layout()

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, f"cylinder_drag_Re{int(Re)}.{ext}"))
print(f"Saved: cylinder_drag_Re{int(Re)}.pdf / .png  →  {out_dir}")
plt.close(fig)

# =====================================================================
#  Figure 2 — Decomposed drag with inset for early transient
# =====================================================================
fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))

ax2.fill_between(tau, 0, viscous_cd,  alpha=0.15, color=C_VISCOUS)
ax2.fill_between(tau, viscous_cd, total_cd, alpha=0.15, color=C_PRESSURE)
ax2.plot(tau, viscous_cd,  color=C_VISCOUS,  ls="-", lw=1.2,
         label=r"Viscous $C_D$")
ax2.plot(tau, pressure_cd, color=C_PRESSURE, ls="-", lw=1.2,
         label=r"Pressure $C_D$")
ax2.plot(tau, total_cd,    color=C_TOTAL,    ls="-", lw=1.8,
         label=r"Total $C_D$")
ax2.plot(tau_ref, cd_ref,  color=C_REF,      ls="--", lw=1.8,
         label="K\&L 1995")

ax2.set_xlabel(r"Convective time $\tau = tU/R$")
ax2.set_ylabel(r"Drag coefficient $C_D$")
ax2.set_xlim(0, 7)
ax2.set_ylim(0, 2)

ax2.legend(loc="upper right", frameon=True, framealpha=0.9,
           edgecolor="0.85", fancybox=False, fontsize=8)

# Inset: zoom on early transient (τ < 1)
ax_inset = ax2.inset_axes([0.35, 0.45, 0.35, 0.40])  # [x0, y0, w, h]
mask_sim = tau <= 1.5
mask_ref = tau_ref <= 1.5
ax_inset.plot(tau[mask_sim], total_cd[mask_sim],
              color=C_TOTAL, ls="-", lw=1.5)
ax_inset.plot(tau_ref[mask_ref], cd_ref[mask_ref],
              color=C_REF, ls="--", lw=1.5)
ax_inset.set_xlim(0, 1.5)
ax_inset.set_ylim(0, 2)
ax_inset.set_xlabel(r"$\tau$", fontsize=7, labelpad=1)
ax_inset.set_ylabel(r"$C_D$", fontsize=7, labelpad=1)
ax_inset.tick_params(labelsize=7)
ax_inset.grid(True, alpha=0.2, linewidth=0.4)
for spine in ax_inset.spines.values():
    spine.set_linewidth(0.5)
    spine.set_color("0.5")

ax2.text(0.03, 0.03, rf"$Re = {int(Re)}$,  $N_x = {Nx}$",
         transform=ax2.transAxes, fontsize=9, color="0.35",
         verticalalignment="bottom")

fig2.tight_layout()
for ext in ("pdf", "png"):
    fig2.savefig(os.path.join(out_dir, f"cylinder_drag_decomposed_Re{int(Re)}.{ext}"))
print(f"Saved: cylinder_drag_decomposed_Re{int(Re)}.pdf / .png  →  {out_dir}")
plt.close(fig2)

print("Done!")
