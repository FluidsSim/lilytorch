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

METHODOLOGY (revised):
  The previous version restricted every grid (including the finest-grid
  reference) down to the coarsest grid before differencing.  For a BDIM
  flow, that approach contaminates restricted-reference values near the
  body: each coarse cell averages 16×16 fine cells when going from
  Nx=2048 → Nx=128, and the averaging unavoidably mixes body-interior
  pressure (arbitrary) and BDIM-damped velocity (≈0) into cells whose
  centre lies just outside the body.  The result is an O(1) "contamination
  ring" of width ~Δx_coarsest that does not decrease with Nx and caps the
  apparent convergence rate at first order or worse.

  Fix: each Nx is compared on its **own grid**, with the reference being
  restricted from Nx_finest → Nx (refinement ratio di_ref = Nx_finest/Nx,
  shrinking as Nx grows).  For pressure (and any field with undefined body
  values) the restriction is mask-aware – it averages only fluid sub-cells
  inside each target cell – so that body-interior values never leak across
  the immersed boundary.  The SDF, the BDIM band mask
  (sdf > 2·Δx_Nx) and the fluid/far masks are evaluated on the Nx grid,
  matching the actual resolution being analysed.
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

# ================================================================
# Helpers
# ================================================================
def cc_coords(Nx, dx):
    """Cell-centred coordinates for an Nx-cell axis."""
    return np.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, Nx)

def face_coords(Nx, dx):
    """Staggered (left-face) coordinates for the Nx interior face slots
    stored in the simulation arrays (excluding ghost cells)."""
    return np.linspace(xmin, xmax - dx, Nx)

def make_cc_grid(Nx, dx):
    x = cc_coords(Nx, dx)
    return np.meshgrid(x, x, indexing='ij')

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

# ----------------------------------------------------------------
# Restriction operators (fine grid → target grid, di = ratio)
#
# Each field is stored as (N+2, N+2) with one ghost cell on every side.
# The interior slice [1:-1, 1:-1] has shape (N, N) where:
#   * u (staggered in x, CC in y): interior face index k corresponds to
#     x_face = xmin + k·Δx_N   (k = 0 .. N-1)
#   * v (CC in x, staggered in y): index k → y_face = xmin + k·Δx_N
#   * p (CC, CC):                  index (i,j) → centre (xmin+(i+0.5)Δx_N, …)
#
# Restriction from Nx_fine to Nx_tgt (di = Nx_fine // Nx_tgt) keeps the
# staggered alignment exact by stride-sampling in staggered directions and
# uses block averaging in CC directions (second-order quadrature).
# ----------------------------------------------------------------
def restrict_field(field, di, stag_x, stag_y):
    """Restrict an (Nf+2, Nf+2) field to (Nt, Nt) with Nt = Nf/di.

    Stride in staggered directions (face k·di lands on coarse face k);
    block-average in cell-centred directions.
    """
    interior = field[1:-1, 1:-1]
    Nf = interior.shape[0]
    if di == 1:
        return interior.copy()
    Nt = Nf // di
    # x axis
    if stag_x:
        tmp = interior[::di, :]
    else:
        tmp = interior.reshape(Nt, di, Nf).mean(axis=1)
    # y axis
    if stag_y:
        out = tmp[:, ::di]
    else:
        out = tmp.reshape(tmp.shape[0], Nt, di).mean(axis=2)
    return out


def restrict_field_masked(field, fluid_mask_fine, di, stag_x, stag_y):
    """Mask-aware restriction.  In CC directions, the block average uses
    only fluid sub-cells of `fluid_mask_fine`; in staggered directions
    stride-sampling is unchanged (no neighbouring body cells to mix in).
    Returns (out, valid) where `valid[i,j]` is True when at least one
    fluid sub-cell contributed to cell (i,j).
    """
    interior = field[1:-1, 1:-1]
    mask     = fluid_mask_fine.astype(field.dtype)
    Nf = interior.shape[0]
    if di == 1:
        return interior.copy(), fluid_mask_fine.copy()
    Nt = Nf // di

    # x axis
    if stag_x:
        tmp_v = interior[::di, :]
        tmp_m = mask[::di, :]
    else:
        weighted = interior * mask
        tmp_v = weighted.reshape(Nt, di, Nf).sum(axis=1)
        tmp_m = mask.reshape(Nt, di, Nf).sum(axis=1)

    # y axis
    if stag_y:
        s_v = tmp_v[:, ::di]
        s_m = tmp_m[:, ::di]
    else:
        if not stag_x:
            # tmp_v already weighted-sum across x; just sum across y
            s_v = tmp_v.reshape(tmp_v.shape[0], Nt, di).sum(axis=2)
            s_m = tmp_m.reshape(tmp_m.shape[0], Nt, di).sum(axis=2)
        else:
            # stag_x stride did not weight; weight now
            weighted_y = tmp_v * tmp_m
            s_v = weighted_y.reshape(tmp_v.shape[0], Nt, di).sum(axis=2)
            s_m = tmp_m.reshape(tmp_m.shape[0], Nt, di).sum(axis=2)

    valid = s_m > 0
    out   = np.where(valid, s_v / np.where(valid, s_m, 1), 0.0)
    # `valid` should be (Nt, Nt) – cell has at least one contributing fluid sub-cell.
    return out, valid


# ================================================================
# Load finest-grid reference + precompute reference-grid SDF
# ================================================================
ref_path = os.path.join(maindir, f"Nx{Nx_finest}", "uv_field")
u_ref = np.load(os.path.join(ref_path, "u.npy"))
v_ref = np.load(os.path.join(ref_path, "v.npy"))
p_ref = np.load(os.path.join(ref_path, "p.npy"))
N_ref = u_ref.shape[0] - 2
print(f"Reference: Nx={Nx_finest}, dx={dx_finest:.6f}, D/dx={D/dx_finest:.2f}, "
      f"grid shape {u_ref.shape}, N_interior={N_ref}")

# SDF on the reference grid – used as fluid mask for mask-aware restriction.
X_ref, Y_ref = make_cc_grid(Nx_finest, dx_finest)
sdf_ref      = sdf_cylinder(X_ref, Y_ref)
fluid_ref    = sdf_ref > 0

# Staggered analogues of the reference SDF (sample SDF at u/v positions).
xf_ref = face_coords(Nx_finest, dx_finest)
yc_ref = cc_coords(Nx_finest, dx_finest)
Xu_ref, Yu_ref = np.meshgrid(xf_ref, yc_ref, indexing='ij')
sdf_u_ref = sdf_cylinder(Xu_ref, Yu_ref)
fluid_u_ref = sdf_u_ref > 0

xc_ref = yc_ref
yf_ref = xf_ref
Xv_ref, Yv_ref = np.meshgrid(xc_ref, yf_ref, indexing='ij')
sdf_v_ref = sdf_cylinder(Xv_ref, Yv_ref)
fluid_v_ref = sdf_v_ref > 0

# ================================================================
# Compute errors – each Nx compared on its own grid
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

# Cache restricted reference fields for later figure use.
ref_cache = {}

for Nx in nxs:
    dx       = Lx / Nx
    di_ref   = Nx_finest // Nx

    u = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "u.npy"))
    v = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "v.npy"))
    p = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "p.npy"))

    # Strip ghost cells.
    u_i = u[1:-1, 1:-1]
    v_i = v[1:-1, 1:-1]
    p_i = p[1:-1, 1:-1]

    # Restrict reference to this Nx.
    #   u/v: BDIM damps them to ~0 inside the body on both grids, so a
    #        plain block-average matches what the Nx-grid simulation also
    #        contains in the body cells used for those staggered fields.
    #        We still use mask-aware restriction to be safe.
    #   p:   unconstrained inside the body (μ₀→0). Must exclude body
    #        sub-cells from the block average to avoid contamination.
    if di_ref == 1:
        ur = u_ref[1:-1, 1:-1]
        vr = v_ref[1:-1, 1:-1]
        pr = p_ref[1:-1, 1:-1]
        pr_valid = np.ones_like(pr, dtype=bool)
    else:
        ur, _ = restrict_field_masked(u_ref, fluid_u_ref, di_ref, True,  False)
        vr, _ = restrict_field_masked(v_ref, fluid_v_ref, di_ref, False, True)
        pr, pr_valid = restrict_field_masked(p_ref, fluid_ref, di_ref, False, False)

    # SDFs on this Nx grid.
    X, Y       = make_cc_grid(Nx, dx)
    sdf        = sdf_cylinder(X, Y)            # cell-centred SDF
    # Staggered SDF positions for u, v.
    Xu, Yu     = np.meshgrid(face_coords(Nx, dx), cc_coords(Nx, dx), indexing='ij')
    sdf_u      = sdf_cylinder(Xu, Yu)
    Xv, Yv     = np.meshgrid(cc_coords(Nx, dx), face_coords(Nx, dx), indexing='ij')
    sdf_v      = sdf_cylinder(Xv, Yv)

    # Masks (cell-centred quantities use `sdf`; u/v errors are interpolated
    # onto the CC grid below so a single mask family covers all metrics).
    fluid    = sdf > 0                          # strictly outside body
    bdim_eps = 2.0 * dx                         # BDIM half-band on THIS grid
    interior = sdf > bdim_eps                   # outside BDIM band
    far      = sdf >= body_band_R * R           # far-field (>5R from surface)

    # Velocity magnitude on cell centres (average of two adjacent faces in
    # the staggered direction).  We average **after** restriction so the
    # error is evaluated at consistent CC locations.
    def faces_to_cc_x(field_xface):
        # field shape (Nx, Ny), x-faces → CC by averaging adjacent x-faces.
        # face k is the LEFT face of cell k; right face is k+1 (or periodic wrap).
        # Use one-sided extrapolation at the last cell (matches typical BDIM
        # output where the last face is set by the BC – kept as-is).
        right = np.empty_like(field_xface)
        right[:-1, :] = field_xface[1:, :]
        right[-1,  :] = field_xface[-1, :]
        return 0.5 * (field_xface + right)

    def faces_to_cc_y(field_yface):
        right = np.empty_like(field_yface)
        right[:, :-1] = field_yface[:, 1:]
        right[:,  -1] = field_yface[:, -1]
        return 0.5 * (field_yface + right)

    u_cc  = faces_to_cc_x(u_i)
    v_cc  = faces_to_cc_y(v_i)
    ur_cc = faces_to_cc_x(ur)
    vr_cc = faces_to_cc_y(vr)

    eu = u_cc - ur_cc
    ev = v_cc - vr_cc
    emag = np.sqrt(eu**2 + ev**2)

    # Pressure gauge correction (incompressible: p only defined up to a
    # constant).  Use the same valid-fluid mask for both fields and for
    # both the gauge and the metric so the comparison is consistent.
    gauge_mask = fluid & pr_valid
    p_mean  = p_i[gauge_mask].mean()
    pr_mean = pr[gauge_mask].mean()
    ep = (p_i - p_mean) - (pr - pr_mean)
    # Cells where the restriction had no fluid sub-cell are not comparable.
    ep = np.where(pr_valid, ep, 0.0)

    # Cache for figures.
    ref_cache[Nx] = dict(emag=emag, ep=ep, pr_valid=pr_valid, sdf=sdf,
                         dx=dx, bdim_eps=bdim_eps, fluid=fluid,
                         interior=interior, far=far)

    # ---- L2 / Linf metrics (RMS = sqrt(mean(e^2)) on the Nx grid). ----
    metrics["L2_u_global"  ].append(rms(emag))
    metrics["Linf_u_global"].append(l_inf(emag))
    metrics["L2_p_global"  ].append(rms(ep[pr_valid]))
    metrics["Linf_p_global"].append(l_inf(ep[pr_valid]))

    metrics["L2_u_fluid"   ].append(rms(emag[fluid]))
    metrics["Linf_u_fluid" ].append(l_inf(emag[fluid]))
    pf = fluid & pr_valid
    metrics["L2_p_fluid"   ].append(rms(ep[pf]))
    metrics["Linf_p_fluid" ].append(l_inf(ep[pf]))

    metrics["L2_u_interior"  ].append(rms(emag[interior]))
    metrics["Linf_u_interior"].append(l_inf(emag[interior]))
    pi_mask = interior & pr_valid
    metrics["L2_p_interior"  ].append(rms(ep[pi_mask]))
    metrics["Linf_p_interior"].append(l_inf(ep[pi_mask]))

    metrics["L2_u_far"  ].append(rms(emag[far]))
    metrics["Linf_u_far"].append(l_inf(emag[far]))
    pfar = far & pr_valid
    metrics["L2_p_far"  ].append(rms(ep[pfar]))
    metrics["Linf_p_far"].append(l_inf(ep[pfar]))

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
print_table("FLUID-INTERIOR errors (sdf > 2*dx_Nx, BDIM band excluded)",
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
    ("global",   "whole domain",                   "C0", "o", "-"),
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
#
#   Both rows now use the same error fields that drove the metrics tables
#   above so figures and tables are mutually consistent.
# ================================================================
zoom   = 3 * D
theta  = np.linspace(0, 2 * np.pi, 100)
n_cols = len(nxs)

fig2, axes2 = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
if n_cols == 1:
    axes2 = axes2.reshape(2, 1)

for i, Nx in enumerate(nxs):
    cache = ref_cache[Nx]
    emag  = cache["emag"]
    ep    = cache["ep"]
    sdf_c = cache["sdf"]
    dx    = cache["dx"]
    bdim_eps_i = cache["bdim_eps"]
    fmask = cache["fluid"]

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
    int_mask = sdf_c > bdim_eps_i
    ep_max = max(np.abs(ep[int_mask]).max(), 1e-8) if int_mask.any() \
             else max(np.abs(ep[fmask]).max(), 1e-8)
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
