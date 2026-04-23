"""
Convergence analysis for BDIM2 flow past cylinder – MW2015 style.

Reads results from run_error_analysis_cylinder_2d_MW.py (power-of-2 grids)
and computes L2 / L-inf errors for velocity and pressure, with global /
near-body / far-field / fluid-only breakdowns.  Prints convergence rates
and makes publication-quality plots.

NOTE on pressure norms:
  In BDIM2 the variable-coefficient Poisson equation  div(μ₀ ∇p) = div(u*)
  has μ₀→0 inside the body, making p unconstrained there.  Including the
  body interior in the global norm inflates the error with non-physical
  values and destroys the convergence rate.  The primary convergence plot
  therefore reports **fluid-only** norms (sdf > 0).
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
D  = 1
R  = D / 2
Re = 100
cx, cy = 0.0, 0.0

half_L = 15 * D
xmin, xmax = -half_L, half_L
ymin, ymax = -half_L, half_L
Lx = xmax - xmin   # 30

maindir = "/data/andreaferrario/ns_data/flow_past_cylinder_error_tests_MW/abdquickest/"
figdir  = maindir

# Detect Nx values from directory names  (Nx256/, Nx512/, Nx1024/)
nxs_all = sorted([int(d[2:]) for d in os.listdir(maindir)
                   if os.path.isdir(os.path.join(maindir, d))
                   and d.startswith('Nx') and d[2:].isdigit()])
print(f"Grid sizes found: Nx = {nxs_all}")

Nx_finest = max(nxs_all)              # reference = finest grid
nxs = [n for n in nxs_all if n != Nx_finest]  # compare these against finest

# Derived quantities
dxs_all = [Lx / n for n in nxs_all]
dx_finest = Lx / Nx_finest
dx_coarsest = Lx / min(nxs_all)

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
ref_path = os.path.join(maindir, f"Nx{Nx_finest}", "uv_field")
u_ref = np.load(os.path.join(ref_path, "u.npy"))
v_ref = np.load(os.path.join(ref_path, "v.npy"))
p_ref = np.load(os.path.join(ref_path, "p.npy"))
N_ref = u_ref.shape[0] - 2  # interior size (excluding ghost cells)
print(f"Reference: Nx={Nx_finest}, dx={dx_finest:.6f}, D/dx={D/dx_finest:.2f}, "
      f"grid shape {u_ref.shape}, N_interior={N_ref}")

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

Nx_coarsest = min(nxs_all)

for Nx in nxs:
    dx = Lx / Nx

    u = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "u.npy"))
    v = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "v.npy"))
    p = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "p.npy"))

    # Down-sample to common comparison grid (coarsest)
    # All Nx are powers of 2, so stride = Nx / Nx_coarsest is exact integer
    di     = Nx // Nx_coarsest
    di_ref = Nx_finest // Nx_coarsest

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
    fluid = sdf > 0                        # strictly outside body
    far   = sdf >= body_band_R * R          # far-field (>5R from surface)

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

dx_arr = np.array([Lx / n for n in nxs])  # grid spacings for coarse grids

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
        print(f"    {'Nx':>6s}  {'dx':>10s}  {'D/dx':>6s}  {'error':>12s}  {'rate':>8s}")
        print(f"    {'-'*52}")
        for i, Nx in enumerate(nxs):
            dx = Lx / Nx
            if i == 0:
                r_str = "   ---"
            else:
                r = rate(vals[i-1], vals[i], dx_arr[i-1], dx_arr[i])
                r_str = f"{r:6.2f}"
            print(f"    {Nx:6d}  {dx:10.6f}  {D/dx:6.1f}  {vals[i]:12.4e}  {r_str}")

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
    (0, 0, "L2_u_fluid",   r"$L_2$ velocity error"),
    (0, 1, "Linf_u_fluid", r"$L_\infty$ velocity error"),
    (1, 0, "L2_p_fluid",   r"$L_2$ pressure error"),
    (1, 1, "Linf_p_fluid", r"$L_\infty$ pressure error"),
]

for (r, c, key_primary, ylabel) in specs:
    ax = axes[r, c]
    vp = metrics[key_primary]
    ax.loglog(dx_arr, vp, "ko-", ms=8, lw=2, label="Fluid only (sdf > 0)")

    # Reference slopes – anchor at the geometric midpoint of the data range
    # and offset vertically so the lines sit clearly above/below the data
    mid_idx = len(dx_arr) // 2
    dx_mid = dx_arr[mid_idx]
    vp_mid = vp[mid_idx]

    s2 = vp_mid / dx_mid**2
    s1 = vp_mid / dx_mid**1
    # Shift reference lines above the data for visual clarity
    shift = 2.5
    ax.loglog(x_ref, shift * s2 * x_ref**2, "r--", lw=1.5, alpha=0.7, label=r"$O(h^2)$")
    ax.loglog(x_ref, shift * s1 * x_ref**1, "b--", lw=1.5, alpha=0.7, label=r"$O(h)$")

    ax.set_xlabel(r"$\Delta x / D$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)

    # Better xticks: show actual dx values with D/dx labels
    ax.set_xticks(dx_arr)
    ax.set_xticklabels([f"{Lx/n:.2g}\n(D/$\Delta x$={n/Lx*D:.0f})" for n in nxs],
                       fontsize=8)
    ax.minorticks_off()

    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="major", ls=":", alpha=0.4)

    # Annotate convergence rates between consecutive points
    for i in range(1, len(nxs)):
        r_val = rate(vp[i-1], vp[i], dx_arr[i-1], dx_arr[i])
        xm = np.sqrt(dx_arr[i-1]*dx_arr[i])
        ym = np.sqrt(vp[i-1]*vp[i])
        ax.annotate(f"{r_val:.2f}", (xm, ym), fontsize=9, color="black",
                    ha="center", va="bottom")

fig.suptitle(f"BDIM2 Cylinder Convergence (D={D}, Re={Re}, MW2015 setup)", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "convergence_MW_global.pdf"), bbox_inches="tight", dpi=150)
fig.savefig(os.path.join(figdir, "convergence_MW_global.png"), bbox_inches="tight", dpi=150)
print(f"\nSaved: convergence_MW_global.pdf/png")

# ================================================================
# Figure 2 – Error fields (velocity magnitude) at each resolution
# ================================================================
fig2, axes2 = plt.subplots(1, len(nxs), figsize=(5*len(nxs), 5))
if len(nxs) == 1:
    axes2 = [axes2]

for i, Nx in enumerate(nxs):
    dx = Lx / Nx
    u = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "u.npy"))
    v = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "v.npy"))

    di     = Nx // Nx_coarsest
    di_ref = Nx_finest // Nx_coarsest
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
    ax.set_title(f"Nx={Nx}  (D/$\\Delta x$ = {D/dx:.0f})")
    ax.set_xlabel("x / D")
    if i == 0:
        ax.set_ylabel("y / D")
    # Zoom around body
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
fig3, axes3 = plt.subplots(1, len(nxs_all), figsize=(5*len(nxs_all), 5))
if len(nxs_all) == 1:
    axes3 = [axes3]

for i, Nx in enumerate(nxs_all):
    dx = Lx / Nx
    u = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "u.npy"))
    v = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "v.npy"))
    omega = np.gradient(v, dx, axis=0, edge_order=2) - np.gradient(u, dx, axis=1, edge_order=2)

    ax = axes3[i]
    vmax = 2.0
    ax.imshow(omega.T, origin="lower",
              extent=[xmin, xmax, ymin, ymax],
              cmap=cm.RdBu, vmin=-vmax, vmax=vmax)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), "k-", lw=1.5)
    ax.set_title(f"Nx={Nx}  (D/$\\Delta x$ = {D/dx:.0f})")
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
    f.write(f"D={D}, Re={Re}, domain=30D×30D, t=3 T_conv\n")
    f.write(f"Power-of-2 grids: Nx = {nxs_all}\n")
    f.write("=" * 60 + "\n\n")
    for key in sorted(metrics.keys()):
        vals = metrics[key]
        f.write(f"{key}:\n")
        f.write(f"  {'Nx':>6s}  {'dx':>10s}  {'D/dx':>6s}  {'error':>12s}  {'rate':>8s}\n")
        for i, Nx in enumerate(nxs):
            dx = Lx / Nx
            if i == 0:
                r_str = "   ---"
            else:
                r_val = rate(vals[i-1], vals[i], dx_arr[i-1], dx_arr[i])
                r_str = f"{r_val:6.2f}"
            f.write(f"  {Nx:6d}  {dx:10.6f}  {D/dx:6.1f}  {vals[i]:12.4e}  {r_str}\n")
        f.write("\n")

print(f"\nSaved summary: {summary_path}")
print("Done!")
