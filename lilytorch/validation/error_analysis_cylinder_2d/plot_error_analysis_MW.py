"""
Convergence analysis for BDIM2 flow past cylinder – MW2015 style.

Reads results from run_error_analysis_cylinder_2d_MW.py (power-of-2 grids)
and computes L2 / L-inf errors for velocity and pressure, with global /
near-body / far-field / fluid-only breakdowns.  Prints convergence rates
and makes publication-quality plots.

NOTE on pressure norms:
  In BDIM2 the variable-coefficient Poisson equation  div(μ₀ ∇p) = div(u*)
  has μ₀→0 inside the body, making p unconstrained there.  Including the
  body interior in the pressure norm inflates the error with non-physical
  values and destroys the convergence rate.

  Following MW2015, the primary convergence figure therefore reports errors
  over the **fluid domain** (sdf > 0): the whole domain outside the body,
  which includes the BDIM transition band.  The body interior (where
  μ₀≈0 and pressure is arbitrary) is always excluded.
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

half_L = 5 * D
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
    "L2_u_global":   [], "Linf_u_global":   [],
    "L2_p_global":   [], "Linf_p_global":   [],
    "L2_u_fluid":    [], "Linf_u_fluid":    [],
    "L2_p_fluid":    [], "Linf_p_fluid":    [],
    "L2_u_interior": [], "Linf_u_interior": [],
    "L2_p_interior": [], "Linf_p_interior": [],
    "L2_u_far":      [], "Linf_u_far":      [],
    "L2_p_far":      [], "Linf_p_far":      [],
}

body_band_R = 5   # near-body = within 5R from surface

Nx_coarsest = min(nxs_all)

for Nx in nxs:
    dx = Lx / Nx

    u = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "u.npy"))
    v = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "v.npy"))
    p = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "p.npy"))

    # Down-sample to common comparison grid (coarsest).
    # All Nx are powers-of-2, so di = Nx / Nx_coarsest is exact integer.
    #
    # IMPORTANT – staggered vs. cell-centred alignment:
    #   u is stored at x-FACE positions (staggered in x, CC in y).
    #   v is stored at y-FACE positions (CC in x, staggered in y).
    #   p is CC in both x and y.
    #
    #   For staggered directions, stride-sampling aligns exactly with
    #   the coarser face positions (x_stag = x - h/2, so face k·di
    #   lands on coarse face k).
    #   For CC directions, stride-sampling gives positions xmin + 0.5*h
    #   + k·di·h, while coarse CC positions are xmin + 0.5*h_c + k·h_c
    #   = xmin + 0.5·di·h + k·di·h.  The offset (di-1)·h/2 introduces
    #   an O(h) interpolation error that caps apparent convergence at
    #   first order even when the true solution is second order.
    #
    #   Fix: average di·di (or 1·di / di·1) cells in the CC direction(s).
    di     = Nx // Nx_coarsest
    di_ref = Nx_finest // Nx_coarsest
    Nc     = Nx_coarsest

    def _restrict(field, di_x, di_y):
        """Restrict interior of 'field' to (Nc, Nc) by striding in staggered
        directions and averaging in cell-centred directions.
        di_x / di_y: stride factor for x/y respectively.
          stride → just take every di-th index (faces align).
          average → average di adjacent cells (CC alignment fix).
        """
        interior = field[1:-1, 1:-1]          # strip ghost cells → (Nx, Ny)
        if di_x == 1:                          # stride in x (staggered)
            tmp = interior[::di_y, :]
        else:                                  # average in x (CC), stride in y
            tmp = interior.reshape(Nc, di_x, -1).mean(axis=1)
        if di_y == 1:
            out = tmp[:, ::di_x]               # stride in y (staggered)
        else:
            out = tmp.reshape(-1, Nc, di_y).mean(axis=2)
        return out

    # u: staggered in x (stride), CC in y (average)
    u_c  = _restrict(u,     1,     di)
    ur_c = _restrict(u_ref, 1,     di_ref)
    # v: CC in x (average), staggered in y (stride)
    v_c  = _restrict(v,     di,    1)
    vr_c = _restrict(v_ref, di_ref, 1)
    # p: CC in both x and y (average in both)
    p_c  = _restrict(p,     di,    di)
    pr_c = _restrict(p_ref, di_ref, di_ref)

    # SDF on comparison grid
    X, Y = make_grid(u_c.shape[0], u_c.shape[1], dx_coarsest, dx_coarsest)
    sdf = sdf_cylinder(X, Y)

    # Masks
    fluid    = sdf > 0                     # strictly outside body
    # Exclude the BDIM transition band (half-width ε = 2·dx_coarsest).
    # Pressure and velocity inside the band are O(1)-corrected by BDIM
    # and do not converge in Linf as dx → 0 (the band always contains
    # O(1)-error cells regardless of resolution).  The interior mask
    # removes these cells to expose the true convergence rate.
    bdim_eps  = 2.0 * dx_coarsest
    interior  = sdf > bdim_eps                # outside BDIM band
    far       = sdf >= body_band_R * R        # far-field (>5R from surface)

    eu = u_c - ur_c
    ev = v_c - vr_c
    # Remove gauge offset: pressure is only defined up to a constant in
    # incompressible flow.  Subtract the mean over fluid cells so that the
    # error reflects only spatial structure, not an arbitrary datum shift.
    fluid_mask_c = sdf > 0
    ep = (p_c - p_c[fluid_mask_c].mean()) - (pr_c - pr_c[fluid_mask_c].mean())
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

    # Fluid interior (sdf > 2·dx_coarsest — BDIM band excluded)
    metrics["L2_u_interior"].append(rms(emag[interior]))
    metrics["Linf_u_interior"].append(l_inf(emag[interior]))
    metrics["L2_p_interior"].append(rms(ep[interior]))
    metrics["Linf_p_interior"].append(l_inf(np.abs(ep[interior])))

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
print_table(f"FLUID-INTERIOR errors (sdf > 2*dx_coarsest = {2*dx_coarsest:.4f}, BDIM band excluded)",
            ["L2_u_interior", "Linf_u_interior", "L2_p_interior", "Linf_p_interior"])
print_table("FAR-FIELD errors (sdf > 5R)",
            ["L2_u_far", "Linf_u_far", "L2_p_far", "Linf_p_far"])

# ================================================================
# Figure 1 – Global convergence (velocity + pressure)
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
xlim = [dx_arr[-1]*0.7, dx_arr[0]*1.4]
x_ref = np.array(xlim)

# Each subplot shows the same norm/component across all four masks.
specs = [
    (0, 0, "L2_u",   r"$L_2$ velocity error"),
    (0, 1, "Linf_u", r"$L_\infty$ velocity error"),
    (1, 0, "L2_p",   r"$L_2$ pressure error"),
    (1, 1, "Linf_p", r"$L_\infty$ pressure error"),
]

# Masks plotted on every subplot, in drawing order.
mask_variants = [
    ("global",   "whole domain",                  "C0", "o", "-"),
    ("fluid",    r"fluid (sdf $> 0$)",             "C1", "s", "-"),
    ("interior", r"interior (sdf $> 2\Delta x$)",  "C2", "^", "-"),
    ("far",      r"far field (sdf $> 5R$)",        "C3", "D", "-"),
]

for (r, c, key_base, ylabel) in specs:
    ax = axes[r, c]

    # Collect all series to anchor reference slopes at the interior curve.
    anchor_key = f"{key_base}_interior"
    vp_anchor  = metrics[anchor_key]
    mid_idx    = len(dx_arr) // 2
    dx_mid     = dx_arr[mid_idx]
    vp_mid     = vp_anchor[mid_idx]

    for suffix, label, color, marker, ls in mask_variants:
        key = f"{key_base}_{suffix}"
        vp  = metrics[key]
        ax.loglog(dx_arr, vp, color=color, marker=marker, ls=ls,
                  ms=7, lw=1.8, label=label)
        # Annotate convergence rates next to the interior curve only
        # to avoid clutter; other curves can be read from the tables.
        if suffix == "interior":
            for i in range(1, len(nxs)):
                r_val = rate(vp[i-1], vp[i], dx_arr[i-1], dx_arr[i])
                xm = np.sqrt(dx_arr[i-1] * dx_arr[i])
                ym = np.sqrt(vp[i-1]     * vp[i])
                ax.annotate(f"{r_val:.2f}", (xm, ym), fontsize=8,
                            color=color, ha="center", va="bottom")

    # Reference slopes anchored to the interior curve
    s2 = vp_mid / dx_mid**2
    s1 = vp_mid / dx_mid**1
    shift = 2.5
    ax.loglog(x_ref, shift * s2 * x_ref**2, "k--", lw=1.2, alpha=0.6, label=r"$O(h^2)$")
    ax.loglog(x_ref, shift * s1 * x_ref**1, "k:",  lw=1.2, alpha=0.6, label=r"$O(h)$")

    ax.set_xlabel(r"$\Delta x / D$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_xticks(dx_arr)
    ax.set_xticklabels([f"{Lx/n:.2g}\n(D/$\\Delta x$={n/Lx*D:.0f})" for n in nxs],
                       fontsize=8)
    ax.minorticks_off()
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="major", ls=":", alpha=0.4)

fig.suptitle(f"BDIM2 Cylinder Convergence (D={D}, Re={Re}, MW2015 setup)", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "convergence_MW_global.pdf"), bbox_inches="tight", dpi=150)
fig.savefig(os.path.join(figdir, "convergence_MW_global.png"), bbox_inches="tight", dpi=150)
print(f"\nSaved: convergence_MW_global.pdf/png")

# ================================================================
# Figure 2 – Error fields: velocity magnitude (row 0) + pressure (row 1)
# ================================================================
zoom   = 3 * D
theta  = np.linspace(0, 2 * np.pi, 100)
n_cols = len(nxs)

fig2, axes2 = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
if n_cols == 1:
    axes2 = axes2.reshape(2, 1)

for i, Nx in enumerate(nxs):
    dx     = Lx / Nx
    di     = Nx // Nx_coarsest
    di_ref = Nx_finest // Nx_coarsest
    Nc     = Nx_coarsest

    u = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "u.npy"))
    v = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "v.npy"))
    p = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "p.npy"))

    # ---- velocity magnitude error (strided – acceptable for visualisation) ----
    eu   = u[1:-1:di, 1:-1:di] - u_ref[1:-1:di_ref, 1:-1:di_ref]
    ev   = v[1:-1:di, 1:-1:di] - v_ref[1:-1:di_ref, 1:-1:di_ref]
    emag = np.sqrt(eu**2 + ev**2)

    # ---- pressure error (block-averaged, gauge-corrected, same as metrics) ----
    p_c  = _restrict(p,     di,     di)
    pr_c = _restrict(p_ref, di_ref, di_ref)
    X, Y = make_grid(p_c.shape[0], p_c.shape[1], dx_coarsest, dx_coarsest)
    sdf_c = sdf_cylinder(X, Y)
    fmask = sdf_c > 0
    ep = (p_c - p_c[fmask].mean()) - (pr_c - pr_c[fmask].mean())

    # ---- row 0: |u_err| ----
    ax = axes2[0, i]
    im0 = ax.imshow(emag.T, origin="lower",
                    extent=[xmin, xmax, ymin, ymax],
                    cmap="hot",
                    norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=1e-1))
    ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), "w-", lw=1.5)
    ax.set_title(f"Nx={Nx}  (D/$\\Delta x$={D/dx:.0f})", fontsize=11)
    ax.set_xlabel("x / D")
    if i == 0:
        ax.set_ylabel(r"$|\mathbf{u}_{err}|$" + "\ny / D")
    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)
    plt.colorbar(im0, ax=ax, shrink=0.8)

    # ---- row 1: p_err (signed, diverging colormap) ----
    # Mask body interior (pressure unconstrained there) and set colour
    # scale from the interior fluid only (sdf > bdim_eps) so that the
    # O(1) errors in the BDIM transition band don't wash out the plot.
    ep_plot = ep.copy()
    ep_plot[sdf_c <= 0] = np.nan          # hide body interior
    int_mask = sdf_c > bdim_eps
    ep_max = max(np.abs(ep[int_mask]).max(), 1e-8) if int_mask.any() else max(np.abs(ep[fmask]).max(), 1e-8)
    ax = axes2[1, i]
    im1 = ax.imshow(ep_plot.T, origin="lower",
                    extent=[xmin, xmax, ymin, ymax],
                    cmap="RdBu_r",
                    vmin=-ep_max, vmax=ep_max)
    ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), "k-", lw=1.5)
    ax.set_xlabel("x / D")
    if i == 0:
        ax.set_ylabel(r"$p_{err}$" + "\ny / D")
    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)
    plt.colorbar(im1, ax=ax, shrink=0.8, label=r"$p_{err}$")

fig2.suptitle("Error fields vs finest grid", fontsize=14, y=1.01)
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
