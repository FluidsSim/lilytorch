"""
Convergence analysis for BDIM2 flow past cylinder – Heun corrector.

Computes L2 / L-inf errors for velocity and pressure against the
finest-grid solution (1024), prints convergence rates, and produces
publication-quality log-log plots.

Includes *global*, *near-body*, and *far-field* error breakdowns to
separate immersed-boundary accuracy from interior spatial accuracy.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import os

matplotlib.rc("font", size=14)

# ================================================================
# Configuration
# ================================================================
maindir = "/data/andreaferrario/ns_data/flow_past_cylinder_error_tests/abdquickest/"
figdir  = maindir  # save figures next to data

# Domain
Lx = 2.0  # xmax - xmin
Ly = 2.0
xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0

# Cylinder
R  = 0.1
cx = xmin + 3*R   # -0.7
cy = 0.0

# Gather resolution directories
dirs = sorted([int(d) for d in os.listdir(maindir)
               if os.path.isdir(os.path.join(maindir, d)) and d.isdigit()])
print(f"Resolutions found: {dirs}")
n_finest = dirs[-1]

# ================================================================
# Helper functions
# ================================================================
def make_grid(nx, ny):
    """Cell-centred grid matching the solver layout (ghost cells excluded by [1:-1])."""
    dx = Lx / nx
    dy = Ly / ny
    x = np.linspace(xmin + 0.5*dx, xmax - 0.5*dx, nx)
    y = np.linspace(ymin + 0.5*dy, ymax - 0.5*dy, ny)
    return np.meshgrid(x, y, indexing='ij')  # (nx, ny)


def sdf_cylinder(X, Y):
    return np.sqrt((X - cx)**2 + (Y - cy)**2) - R


def normalized_l2(err):
    """RMS (root-mean-square) error – properly normalized."""
    return np.sqrt(np.mean(err**2))


def l_inf(err):
    return np.max(np.abs(err))


def convergence_rate(e_coarse, e_fine, h_coarse, h_fine):
    if e_coarse <= 0 or e_fine <= 0:
        return float('nan')
    return np.log(e_coarse / e_fine) / np.log(h_coarse / h_fine)


# ================================================================
# Load finest-grid reference and build comparison grids
# ================================================================
u_ref = np.load(os.path.join(maindir, str(n_finest), "uv_field", "u.npy"))
v_ref = np.load(os.path.join(maindir, str(n_finest), "uv_field", "v.npy"))
p_ref = np.load(os.path.join(maindir, str(n_finest), "uv_field", "p.npy"))
print(f"Reference grid shape: {u_ref.shape}  (N={n_finest})")

n_coarse = dirs[0]  # smallest grid – common comparison size

# ================================================================
# Compute errors at each resolution (except finest)
# ================================================================
resolutions = dirs[:-1]  # compare against finest
dx_arr = np.array([Lx / n for n in resolutions])

# Error containers  – dict of { metric_name : [err_per_resolution] }
metrics = {
    "L2_u_global":  [], "Linf_u_global":  [],
    "L2_p_global":  [], "Linf_p_global":  [],
    "L2_u_body":    [], "Linf_u_body":    [],
    "L2_u_far":     [], "Linf_u_far":     [],
    "L2_p_body":    [], "L2_p_far":       [],
    "Linf_p_body":  [], "Linf_p_far":     [],
}

# Band width for near-body region (in diameter units from the surface)
body_band = 5  # 5 R from the cylinder surface

for nx in resolutions:
    # Load coarse solution
    u = np.load(os.path.join(maindir, str(nx), "uv_field", "u.npy"))
    v = np.load(os.path.join(maindir, str(nx), "uv_field", "v.npy"))
    p = np.load(os.path.join(maindir, str(nx), "uv_field", "p.npy"))

    # Down-sample both to the comparison grid (common coarsest)
    # Stride = resolution / n_coarse
    di = nx // n_coarse
    di_ref = n_finest // n_coarse

    # Interior points only – skip ghost cells
    u_c   = u[1:-1:di, 1:-1:di]
    v_c   = v[1:-1:di, 1:-1:di]
    p_c   = p[1:-1:di, 1:-1:di]
    ur_c  = u_ref[1:-1:di_ref, 1:-1:di_ref]
    vr_c  = v_ref[1:-1:di_ref, 1:-1:di_ref]
    pr_c  = p_ref[1:-1:di_ref, 1:-1:di_ref]

    # Build SDF on comparison grid
    X, Y = make_grid(u_c.shape[0], u_c.shape[1])
    sdf  = sdf_cylinder(X, Y)

    # Masks
    near = (sdf >= 0) & (sdf < body_band * R)
    far  = sdf >= body_band * R

    eu = u_c - ur_c
    ev = v_c - vr_c
    ep = p_c - pr_c

    emag = np.sqrt(eu**2 + ev**2)

    # Global errors
    metrics["L2_u_global"].append(normalized_l2(emag))
    metrics["Linf_u_global"].append(l_inf(emag))
    metrics["L2_p_global"].append(normalized_l2(ep))
    metrics["Linf_p_global"].append(l_inf(np.abs(ep)))

    # Near-body
    if near.sum() > 0:
        metrics["L2_u_body"].append(normalized_l2(emag[near]))
        metrics["Linf_u_body"].append(l_inf(emag[near]))
        metrics["L2_p_body"].append(normalized_l2(ep[near]))
        metrics["Linf_p_body"].append(l_inf(np.abs(ep[near])))
    else:
        for k in ["L2_u_body", "Linf_u_body", "L2_p_body", "Linf_p_body"]:
            metrics[k].append(np.nan)

    # Far-field
    if far.sum() > 0:
        metrics["L2_u_far"].append(normalized_l2(emag[far]))
        metrics["Linf_u_far"].append(l_inf(emag[far]))
        metrics["L2_p_far"].append(normalized_l2(ep[far]))
        metrics["Linf_p_far"].append(l_inf(np.abs(ep[far])))
    else:
        for k in ["L2_u_far", "Linf_u_far", "L2_p_far", "Linf_p_far"]:
            metrics[k].append(np.nan)

# Convert to arrays
for k in metrics:
    metrics[k] = np.array(metrics[k])

# ================================================================
# Print convergence table
# ================================================================
def print_table(title, metric_keys):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    for key in metric_keys:
        vals = metrics[key]
        print(f"\n  {key}:")
        print(f"    {'N':>6s}  {'dx':>10s}  {'error':>12s}  {'rate':>8s}")
        print(f"    {'-'*42}")
        for i, nx in enumerate(resolutions):
            dx = Lx / nx
            if i == 0:
                rate_str = "   ---"
            else:
                rate = convergence_rate(vals[i-1], vals[i], dx_arr[i-1], dx_arr[i])
                rate_str = f"{rate:6.2f}"
            print(f"    {nx:6d}  {dx:10.6f}  {vals[i]:12.4e}  {rate_str}")

print_table("GLOBAL errors (velocity magnitude & pressure)",
            ["L2_u_global", "Linf_u_global", "L2_p_global", "Linf_p_global"])

print_table("NEAR-BODY errors (within 5R of surface)",
            ["L2_u_body", "Linf_u_body", "L2_p_body", "Linf_p_body"])

print_table("FAR-FIELD errors (beyond 5R from surface)",
            ["L2_u_far", "Linf_u_far", "L2_p_far", "Linf_p_far"])


# ================================================================
# Figure 1 – Global convergence (velocity + pressure, L2 + Linf)
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

xlim = [dx_arr[-1]*0.7, dx_arr[0]*1.4]
x_ref = np.array(xlim)

plot_specs = [
    (0, 0, "L2_u_global",   r"$L_2$ velocity error", "ko-"),
    (0, 1, "Linf_u_global", r"$L_\infty$ velocity error", "ko-"),
    (1, 0, "L2_p_global",   r"$L_2$ pressure error", "gs-"),
    (1, 1, "Linf_p_global", r"$L_\infty$ pressure error", "gs-"),
]

for (r, c, key, ylabel, fmt) in plot_specs:
    ax = axes[r, c]
    vals = metrics[key]
    ax.loglog(dx_arr, vals, fmt, markersize=8, label="Heun (BDIM2)")
    # Reference slopes
    scale2 = vals[-1] / dx_arr[-1]**2
    scale1 = vals[-1] / dx_arr[-1]**1
    ax.loglog(x_ref, scale2*x_ref**2, "r--", alpha=0.6, label=r"$\mathcal{O}(h^2)$")
    ax.loglog(x_ref, scale1*x_ref**1, "b--", alpha=0.6, label=r"$\mathcal{O}(h^1)$")
    ax.set_xlabel(r"$h = L/N$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # Annotate rates
    for i in range(1, len(resolutions)):
        rate = convergence_rate(vals[i-1], vals[i], dx_arr[i-1], dx_arr[i])
        xm = np.sqrt(dx_arr[i-1]*dx_arr[i])
        ym = np.sqrt(vals[i-1]*vals[i])
        ax.annotate(f"{rate:.2f}", (xm, ym), fontsize=9, color="red",
                    ha="center", va="bottom")

fig.suptitle("BDIM2 Convergence – Flow Past Cylinder (Heun RK2)", fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "convergence_global_heun.pdf"), bbox_inches="tight", dpi=150)
fig.savefig(os.path.join(figdir, "convergence_global_heun.png"), bbox_inches="tight", dpi=150)
print(f"\nSaved: convergence_global_heun.pdf/png")

# ================================================================
# Figure 2 – Near-body vs Far-field comparison (velocity only)
# ================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, norm_name, ylabel in zip(
        axes2,
        ["L2_u", "Linf_u"],
        [r"$L_2$ velocity error", r"$L_\infty$ velocity error"]):

    ax.loglog(dx_arr, metrics[f"{norm_name}_global"], "ko-", ms=8, label="Global")
    ax.loglog(dx_arr, metrics[f"{norm_name}_body"],   "r^-", ms=8, label="Near-body (<5R)")
    ax.loglog(dx_arr, metrics[f"{norm_name}_far"],    "bs-", ms=8, label="Far-field (>5R)")

    scale2 = metrics[f"{norm_name}_far"][-1] / dx_arr[-1]**2
    scale1 = metrics[f"{norm_name}_far"][-1] / dx_arr[-1]**1
    ax.loglog(x_ref, scale2*x_ref**2, "r--", alpha=0.4, label=r"$O(h^2)$")
    ax.loglog(x_ref, scale1*x_ref**1, "b--", alpha=0.4, label=r"$O(h)$")

    ax.set_xlabel(r"$h = L/N$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.4)

fig2.suptitle("Near-body vs Far-field Convergence (Heun)", fontsize=15, y=1.01)
fig2.tight_layout()
fig2.savefig(os.path.join(figdir, "convergence_body_vs_far_heun.pdf"), bbox_inches="tight", dpi=150)
fig2.savefig(os.path.join(figdir, "convergence_body_vs_far_heun.png"), bbox_inches="tight", dpi=150)
print(f"Saved: convergence_body_vs_far_heun.pdf/png")

# ================================================================
# Figure 3 – Error fields at each resolution
# ================================================================
fig3, axes3 = plt.subplots(1, len(resolutions), figsize=(5*len(resolutions), 5))
if len(resolutions) == 1:
    axes3 = [axes3]

for i, nx in enumerate(resolutions):
    u = np.load(os.path.join(maindir, str(nx), "uv_field", "u.npy"))
    v = np.load(os.path.join(maindir, str(nx), "uv_field", "v.npy"))

    di     = nx // n_coarse
    di_ref = n_finest // n_coarse
    eu = u[1:-1:di, 1:-1:di] - u_ref[1:-1:di_ref, 1:-1:di_ref]
    ev = v[1:-1:di, 1:-1:di] - v_ref[1:-1:di_ref, 1:-1:di_ref]
    emag = np.sqrt(eu**2 + ev**2)

    ax = axes3[i]
    im = ax.imshow(emag.T, origin="lower", extent=[xmin, xmax, ymin, ymax],
                   cmap="hot", norm=matplotlib.colors.LogNorm(vmin=1e-8, vmax=1e-2))
    # Draw cylinder outline
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), "w-", lw=1.5)
    ax.set_title(f"N={nx}  (dx={Lx/nx:.4f})")
    ax.set_xlabel("x")
    if i == 0:
        ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, shrink=0.8, label=r"$|\mathbf{e}_u|$")

fig3.suptitle("Velocity error magnitude (vs 1024 reference)", fontsize=15, y=1.02)
fig3.tight_layout()
fig3.savefig(os.path.join(figdir, "error_fields_heun.pdf"), bbox_inches="tight", dpi=150)
fig3.savefig(os.path.join(figdir, "error_fields_heun.png"), bbox_inches="tight", dpi=150)
print(f"Saved: error_fields_heun.pdf/png")

# ================================================================
# Figure 4 – Vorticity snapshots at each resolution
# ================================================================
fig4, axes4 = plt.subplots(1, len(dirs), figsize=(5*len(dirs), 5))
if len(dirs) == 1:
    axes4 = [axes4]

for i, nx in enumerate(dirs):
    u = np.load(os.path.join(maindir, str(nx), "uv_field", "u.npy"))
    v = np.load(os.path.join(maindir, str(nx), "uv_field", "v.npy"))
    dx = Lx / nx
    # Vorticity: dv/dx - du/dy  (interior only)
    omega = np.gradient(v, dx, axis=0, edge_order=2) - np.gradient(u, dx, axis=1, edge_order=2)

    ax = axes4[i]
    vmax = 0.2
    ax.imshow(omega.T, origin="lower", extent=[xmin, xmax, ymin, ymax],
              cmap=cm.RdBu, vmin=-vmax, vmax=vmax)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), "k-", lw=1.5)
    ax.set_title(f"N={nx}")
    ax.set_xlabel("x")
    if i == 0:
        ax.set_ylabel("y")

fig4.suptitle("Vorticity field (steady state)", fontsize=15, y=1.02)
fig4.tight_layout()
fig4.savefig(os.path.join(figdir, "vorticity_fields_heun.pdf"), bbox_inches="tight", dpi=150)
fig4.savefig(os.path.join(figdir, "vorticity_fields_heun.png"), bbox_inches="tight", dpi=150)
print(f"Saved: vorticity_fields_heun.pdf/png")

# ================================================================
# Summary table written to file
# ================================================================
summary_path = os.path.join(figdir, "convergence_summary_heun.txt")
with open(summary_path, "w") as f:
    f.write("BDIM2 Convergence Analysis – Heun (RK2) Corrector\n")
    f.write("=" * 60 + "\n\n")
    for key in sorted(metrics.keys()):
        vals = metrics[key]
        f.write(f"{key}:\n")
        f.write(f"  {'N':>6s}  {'dx':>10s}  {'error':>12s}  {'rate':>8s}\n")
        for i, nx in enumerate(resolutions):
            dx = Lx / nx
            if i == 0:
                rate_str = "   ---"
            else:
                rate = convergence_rate(vals[i-1], vals[i], dx_arr[i-1], dx_arr[i])
                rate_str = f"{rate:6.2f}"
            f.write(f"  {nx:6d}  {dx:10.6f}  {vals[i]:12.4e}  {rate_str}\n")
        f.write("\n")

print(f"\nSaved convergence table: {summary_path}")
print("\nDone!")
