"""
Convergence analysis for BDIM2 flow past cylinder – MW2015 style.

Reads results from run_error_analysis_cylinder_2d_MW.py and computes
L2 / L-inf errors for velocity and pressure, with global / near-body /
far-field / fluid-only breakdowns.  Prints convergence rates and makes
publication-quality plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import os

matplotlib.rc("font", size=14)

# ================================================================
# Configuration — must match run_error_analysis_cylinder_2d_MW.py
# ================================================================
D  = 120
R  = D / 2
cx, cy = 0.0, 0.0

half_L = 15 * D
xmin, xmax = -half_L, half_L
ymin, ymax = -half_L, half_L
Lx = xmax - xmin   # 3600
Ly = ymax - ymin

maindir = "/data/andreaferrario/ns_data/flow_past_cylinder_error_tests_MW/abdquickest/"
figdir  = maindir

# Spacings tested (must exist as subdirectories)
dxs_all = sorted([int(d) for d in os.listdir(maindir)
                   if os.path.isdir(os.path.join(maindir, d)) and d.isdigit()])
print(f"Spacings found: {dxs_all}")

dx_finest = min(dxs_all)      # reference = finest grid (dx=3)
dxs = [d for d in dxs_all if d != dx_finest]  # compare these against finest

# ================================================================
# Helpers
# ================================================================
def make_grid(nx, ny, dx, dy):
    x = np.linspace(xmin + 0.5*dx, xmax - 0.5*dx, nx)
    y = np.linspace(ymin + 0.5*dy, ymax - 0.5*dy, ny)
    return np.meshgrid(x, y, indexing='ij')

def sdf_cylinder(X, Y):
    return np.sqrt((X - cx)**2 + (Y - cy)**2) - R

def rms(err):
    return np.sqrt(np.mean(err**2))

def l_inf(err):
    return np.max(np.abs(err))

def rate(e1, e2, h1, h2):
    if e1 <= 0 or e2 <= 0:
        return float('nan')
    return np.log(e1 / e2) / np.log(h1 / h2)

# ================================================================
# Load finest-grid reference
# ================================================================
ref_path = os.path.join(maindir, str(dx_finest), "uv_field")
u_ref = np.load(os.path.join(ref_path, "u.npy"))
v_ref = np.load(os.path.join(ref_path, "v.npy"))
p_ref = np.load(os.path.join(ref_path, "p.npy"))
N_ref = u_ref.shape[0] - 2  # interior size (excluding ghost cells)
print(f"Reference: dx={dx_finest}, grid shape {u_ref.shape}, N_interior={N_ref}")

# Coarsest spacing for common comparison grid
dx_coarsest = max(dxs)
N_comp = int(Lx / dx_coarsest)  # comparison grid size (interior)

# ================================================================
# Compute errors
# ================================================================
metrics = {
    "L2_u_global": [], "Linf_u_global": [],
    "L2_p_global": [], "Linf_p_global": [],
    "L2_u_fluid": [], "Linf_u_fluid": [],
    "L2_p_fluid": [], "Linf_p_fluid": [],
    "L2_u_far":   [], "Linf_u_far":   [],
    "L2_p_far":   [], "Linf_p_far":   [],
}

body_band_R = 5   # near-body = within 5R from surface

for dx in dxs:
    Nx = int(Lx / dx)
    u = np.load(os.path.join(maindir, str(dx), "uv_field", "u.npy"))
    v = np.load(os.path.join(maindir, str(dx), "uv_field", "v.npy"))
    p = np.load(os.path.join(maindir, str(dx), "uv_field", "p.npy"))

    # Down-sample to common comparison grid
    di     = dx_coarsest // dx        # stride in this field
    di_ref = dx_coarsest // dx_finest  # stride in reference field

    u_c  = u[1:-1:di, 1:-1:di]
    v_c  = v[1:-1:di, 1:-1:di]
    p_c  = p[1:-1:di, 1:-1:di]
    ur_c = u_ref[1:-1:di_ref, 1:-1:di_ref]
    vr_c = v_ref[1:-1:di_ref, 1:-1:di_ref]
    pr_c = p_ref[1:-1:di_ref, 1:-1:di_ref]

    # SDF on comparison grid
    X, Y = make_grid(u_c.shape[0], u_c.shape[1], dx_coarsest, dx_coarsest)
    sdf = sdf_cylinder(X, Y)

    # Masks
    fluid = sdf > 0                                  # strictly outside body
    far   = sdf >= body_band_R * R                    # far-field (>5R from surface)

    eu = u_c - ur_c
    ev = v_c - vr_c
    ep = p_c - pr_c
    emag = np.sqrt(eu**2 + ev**2)

    # Global
    metrics["L2_u_global"].append(rms(emag))
    metrics["Linf_u_global"].append(l_inf(emag))
    metrics["L2_p_global"].append(rms(ep))
    metrics["Linf_p_global"].append(l_inf(np.abs(ep)))

    # Fluid only (sdf > 0)
    metrics["L2_u_fluid"].append(rms(emag[fluid]))
    metrics["Linf_u_fluid"].append(l_inf(emag[fluid]))
    metrics["L2_p_fluid"].append(rms(ep[fluid]))
    metrics["Linf_p_fluid"].append(l_inf(np.abs(ep[fluid])))

    # Far-field
    metrics["L2_u_far"].append(rms(emag[far]))
    metrics["Linf_u_far"].append(l_inf(emag[far]))
    metrics["L2_p_far"].append(rms(ep[far]))
    metrics["Linf_p_far"].append(l_inf(np.abs(ep[far])))

for k in metrics:
    metrics[k] = np.array(metrics[k])

dx_arr = np.array(dxs, dtype=float)

# ================================================================
# Print convergence table
# ================================================================
def print_table(title, keys):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    for key in keys:
        vals = metrics[key]
        print(f"\n  {key}:")
        print(f"    {'dx':>5s}  {'D/dx':>6s}  {'error':>12s}  {'rate':>8s}")
        print(f"    {'-'*38}")
        for i, dx in enumerate(dxs):
            if i == 0:
                r_str = "   ---"
            else:
                r = rate(vals[i-1], vals[i], dx_arr[i-1], dx_arr[i])
                r_str = f"{r:6.2f}"
            print(f"    {dx:5d}  {D/dx:6.1f}  {vals[i]:12.4e}  {r_str}")

print_table("GLOBAL errors",
            ["L2_u_global", "Linf_u_global", "L2_p_global", "Linf_p_global"])
print_table("FLUID-ONLY errors (sdf > 0)",
            ["L2_u_fluid", "Linf_u_fluid", "L2_p_fluid", "Linf_p_fluid"])
print_table("FAR-FIELD errors (sdf > 5R)",
            ["L2_u_far", "Linf_u_far", "L2_p_far", "Linf_p_far"])

# ================================================================
# Figure 1 – Global convergence (velocity + pressure)
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
xlim = [dx_arr[-1]*0.7, dx_arr[0]*1.4]
x_ref = np.array(xlim)

specs = [
    (0, 0, "L2_u_global",   "L2_u_fluid",  r"$L_2$ velocity error"),
    (0, 1, "Linf_u_global", "Linf_u_fluid", r"$L_\infty$ velocity error"),
    (1, 0, "L2_p_global",   "L2_p_fluid",   r"$L_2$ pressure error"),
    (1, 1, "Linf_p_global", "Linf_p_fluid", r"$L_\infty$ pressure error"),
]

for (r, c, key_g, key_f, ylabel) in specs:
    ax = axes[r, c]
    vg = metrics[key_g]
    vf = metrics[key_f]
    ax.loglog(dx_arr, vg, "ko-", ms=8, label="Global")
    ax.loglog(dx_arr, vf, "bs-", ms=7, label="Fluid only")

    # Reference slopes anchored to fluid-only finest point
    s2 = vf[-1] / dx_arr[-1]**2
    s1 = vf[-1] / dx_arr[-1]**1
    ax.loglog(x_ref, s2*x_ref**2, "r--", alpha=0.5, label=r"$O(h^2)$")
    ax.loglog(x_ref, s1*x_ref**1, "b--", alpha=0.5, label=r"$O(h)$")

    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # Annotate fluid-only rates
    for i in range(1, len(dxs)):
        r_val = rate(vf[i-1], vf[i], dx_arr[i-1], dx_arr[i])
        xm = np.sqrt(dx_arr[i-1]*dx_arr[i])
        ym = np.sqrt(vf[i-1]*vf[i])
        ax.annotate(f"{r_val:.2f}", (xm, ym), fontsize=9, color="blue",
                    ha="center", va="bottom")

fig.suptitle(f"BDIM2 Cylinder Convergence (D={D}, Re=550, MW2015 setup)", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "convergence_MW_global.pdf"), bbox_inches="tight", dpi=150)
fig.savefig(os.path.join(figdir, "convergence_MW_global.png"), bbox_inches="tight", dpi=150)
print(f"\nSaved: convergence_MW_global.pdf/png")

# ================================================================
# Figure 2 – Error fields (velocity magnitude) at each resolution
# ================================================================
fig2, axes2 = plt.subplots(1, len(dxs), figsize=(5*len(dxs), 5))
if len(dxs) == 1:
    axes2 = [axes2]

for i, dx in enumerate(dxs):
    u = np.load(os.path.join(maindir, str(dx), "uv_field", "u.npy"))
    v = np.load(os.path.join(maindir, str(dx), "uv_field", "v.npy"))

    di     = dx_coarsest // dx
    di_ref = dx_coarsest // dx_finest
    eu = u[1:-1:di, 1:-1:di] - u_ref[1:-1:di_ref, 1:-1:di_ref]
    ev = v[1:-1:di, 1:-1:di] - v_ref[1:-1:di_ref, 1:-1:di_ref]
    emag = np.sqrt(eu**2 + ev**2)

    ax = axes2[i]
    im = ax.imshow(emag.T, origin="lower",
                   extent=[xmin, xmax, ymin, ymax],
                   cmap="hot",
                   norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=1e-1))
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), "w-", lw=1.5)
    ax.set_title(f"dx={dx}  (D/dx={D/dx:.0f})")
    ax.set_xlabel("x")
    if i == 0:
        ax.set_ylabel("y")
    # Zoom to 3D around body
    zoom = 3*D
    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)
    plt.colorbar(im, ax=ax, shrink=0.8)

fig2.suptitle("Velocity error magnitude (vs finest grid)", fontsize=14, y=1.02)
fig2.tight_layout()
fig2.savefig(os.path.join(figdir, "error_fields_MW.pdf"), bbox_inches="tight", dpi=150)
fig2.savefig(os.path.join(figdir, "error_fields_MW.png"), bbox_inches="tight", dpi=150)
print(f"Saved: error_fields_MW.pdf/png")

# ================================================================
# Figure 3 – Vorticity at each resolution
# ================================================================
fig3, axes3 = plt.subplots(1, len(dxs_all), figsize=(5*len(dxs_all), 5))
if len(dxs_all) == 1:
    axes3 = [axes3]

for i, dx in enumerate(dxs_all):
    u = np.load(os.path.join(maindir, str(dx), "uv_field", "u.npy"))
    v = np.load(os.path.join(maindir, str(dx), "uv_field", "v.npy"))
    omega = np.gradient(v, dx, axis=0, edge_order=2) - np.gradient(u, dx, axis=1, edge_order=2)

    ax = axes3[i]
    vmax = 0.02
    ax.imshow(omega.T, origin="lower",
              extent=[xmin, xmax, ymin, ymax],
              cmap=cm.RdBu, vmin=-vmax, vmax=vmax)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), "k-", lw=1.5)
    ax.set_title(f"dx={dx}  (D/dx={D/dx:.0f})")
    zoom = 3*D
    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)

fig3.suptitle("Vorticity (all resolutions)", fontsize=14, y=1.02)
fig3.tight_layout()
fig3.savefig(os.path.join(figdir, "vorticity_MW.pdf"), bbox_inches="tight", dpi=150)
fig3.savefig(os.path.join(figdir, "vorticity_MW.png"), bbox_inches="tight", dpi=150)
print(f"Saved: vorticity_MW.pdf/png")

# ================================================================
# Summary to file
# ================================================================
summary_path = os.path.join(figdir, "convergence_summary_MW.txt")
with open(summary_path, "w") as f:
    f.write("BDIM2 Convergence Analysis — MW2015 Setup\n")
    f.write(f"D={D}, Re=550, domain=30D×30D, t=3 T_conv\n")
    f.write("=" * 60 + "\n\n")
    for key in sorted(metrics.keys()):
        vals = metrics[key]
        f.write(f"{key}:\n")
        f.write(f"  {'dx':>5s}  {'D/dx':>6s}  {'error':>12s}  {'rate':>8s}\n")
        for i, dx in enumerate(dxs):
            if i == 0:
                r_str = "   ---"
            else:
                r_val = rate(vals[i-1], vals[i], dx_arr[i-1], dx_arr[i])
                r_str = f"{r_val:6.2f}"
            f.write(f"  {dx:5d}  {D/dx:6.1f}  {vals[i]:12.4e}  {r_str}\n")
        f.write("\n")

print(f"\nSaved summary: {summary_path}")
print("Done!")
