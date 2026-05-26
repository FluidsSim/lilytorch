
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

maindir = "/data/andreaferrario/ns_data/flow_past_cylinder_error_tests_MW/data/abdquickest/"
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

def l_2(err):
    return np.sqrt(np.mean(err**2))

def l_inf(err):
    return np.max(np.abs(err))

def rate(e1, e2, h1, h2):
    if e1 <= 0 or e2 <= 0:
        return float('nan')
    return np.log(e1 / e2) / np.log(h1 / h2)

def coarsen_p_fluid(p_int, Nc):
    """Block-average cell-centred pressure (Nf x Nf) down to (Nc x Nc),
    EXCLUDING body-interior (sdf < 0) fine cells.

    Pressure is undefined inside the body (mu0 -> 0, so it floats to a
    gauge-arbitrary value that differs per grid).  A plain block-average of a
    near-surface coarse cell mixes those floating values in, and since they do
    NOT converge it fakes a ~1st-order pressure rate.  Averaging only the fluid
    (sdf > 0) fine cells removes the contamination.

    Velocity needs NO such mask: inside the body it is the well-defined imposed
    body velocity (u_b ~ 0), so a plain block-average is consistent across grids.
    """
    Nf = p_int.shape[0]
    di = Nf // Nc
    if di == 1:
        return p_int
    dxf = Lx / Nf
    Xf, Yf = make_grid(Nf, Nf, dxf, dxf)
    w = (sdf_cylinder(Xf, Yf) > 0).astype(p_int.dtype)
    def _blk(a):
        return a.reshape(Nc, di, -1).mean(axis=1).reshape(Nc, Nc, di).mean(axis=2)
    num, den, plain = _blk(p_int * w), _blk(w), _blk(p_int)
    return np.where(den > 0, num / np.maximum(den, 1e-30), plain)

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
zoom   = 3 * D
n_cols = len(nxs)
fig2, axes2 = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
if n_cols == 1:
    axes2 = axes2.reshape(2, 1)

Linf_u_fluids = []
Linf_p_fluids = []
L2_u_fluids = []
L2_p_fluids = []

for i, Nx in enumerate(nxs):
    dx = Lx / Nx

    u = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "u.npy"))
    v = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "v.npy"))
    p = np.load(os.path.join(maindir, f"Nx{Nx}", "uv_field", "p.npy"))

    di = Nx_finest // Nx  # refinement factor (integer)
    Ny = u.shape[1] - 2   # interior cells in y (= Nx for square domain)

    # ---- Restrict fine reference to coarse grid -------------------------
    # u : staggered in x (stride di), CC in y (block-average di)
    # v : CC in x (block-average di), staggered in y (stride di)
    # p : CC in both (block-average di × di)
    # Striding in a CC direction introduces an O(Δx) positional error that
    # caps apparent convergence at 1st order regardless of scheme accuracy.
    u_ref_int = u_ref[1:-1, 1:-1]   # (Nx_finest, Ny_finest)
    v_ref_int = v_ref[1:-1, 1:-1]
    p_ref_int = p_ref[1:-1, 1:-1]

    u_ref_c = u_ref_int[::di, :].reshape(Nx, Ny, di).mean(axis=2)
    v_ref_c = v_ref_int.reshape(Nx, di, -1).mean(axis=1)[:, ::di]
    p_ref_c = (p_ref_int.reshape(Nx, di, -1).mean(axis=1)
                         .reshape(Nx, Ny, di).mean(axis=2))

    eu = u[1:-1, 1:-1] - u_ref_c
    ev = v[1:-1, 1:-1] - v_ref_c
    # Remove pressure gauge (defined only up to a constant)
    X, Y = make_grid(Nx, Ny, dx, dx)
    fluid_mask = sdf_cylinder(X, Y) > 0
    ep_raw = p[1:-1, 1:-1] - p_ref_c
    ep = ep_raw - ep_raw[fluid_mask].mean()

    # ---- row 0: |u_err| ----
    ax = axes2[0, i]
    ax.set_title(f"Nx={Nx}  |eu|")
    im0 = ax.imshow(np.abs(eu).T, origin="lower",
                    extent=[xmin, xmax, ymin, ymax],
                    cmap="hot",
                    # norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=1e-1)
                    )
    plt.colorbar(im0, ax=ax, shrink=0.8)

    # ---- row 1: |p_err| ----
    ax = axes2[1, i]
    ax.set_title(f"Nx={Nx}  |ep|")
    im0 = ax.imshow(np.abs(ep).T, origin="lower",
                    extent=[xmin, xmax, ymin, ymax],
                    cmap="hot",
                    # norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=1e-1)
                    )
    plt.colorbar(im0, ax=ax, shrink=0.8)

    # Error norms over fluid cells only (body interior excluded).
    # Velocity error = magnitude sqrt(eu^2 + ev^2) (both components).
    emag = np.sqrt(eu**2 + ev**2)
    Linf_u_fluids.append(l_inf(emag[fluid_mask]))
    Linf_p_fluids.append(l_inf(ep[fluid_mask]))
    L2_u_fluids.append(l_2(emag[fluid_mask]))
    L2_p_fluids.append(l_2(ep[fluid_mask]))

fig2.savefig(os.path.join(figdir, "error_analysis_cylinder_2d.png"), dpi=150, bbox_inches="tight")

# ================================================================
# Convergence table — RAW per-grid norms.  Each grid Nx is compared to the
# finest-grid reference restricted to its OWN resolution (no extra coarsening
# / smoothing).  The only block-averaging is the necessary staggered/CC
# restriction of the reference onto grid Nx (see the loop above):
#   u: stride x, average y ;  v: average x, stride y ;  p: average both.
# Velocity error = |sqrt(eu^2 + ev^2)| ; pressure error = gauge-removed ep.
# Norms over the fluid region (sdf > 0).
# ================================================================
dx_arr = np.array([Lx / n for n in nxs])

def conv_rate(e1, e2, h1, h2):
    if e1 <= 0 or e2 <= 0:
        return float('nan')
    return np.log(e1 / e2) / np.log(h1 / h2)

print(f"\n{'='*64}")
print(f"  Convergence (raw per-grid norms).  Reference = Nx={Nx_finest}.")
print(f"  Fluid region (sdf > 0).")
print(f"{'='*64}")
for label, vals in [("Linf u", Linf_u_fluids), ("L2 u", L2_u_fluids),
                    ("Linf p", Linf_p_fluids), ("L2 p", L2_p_fluids)]:
    print(f"\n  {label}")
    print(f"  {'Nx':>6s}  {'dx':>10s}  {'D/dx':>6s}  {'error':>12s}  {'rate':>6s}")
    print(f"  {'-'*50}")
    for i, Nx in enumerate(nxs):
        dx = Lx / Nx
        r = "---" if i == 0 else f"{conv_rate(vals[i-1], vals[i], dx_arr[i-1], dx_arr[i]):.2f}"
        print(f"  {Nx:6d}  {dx:10.6f}  {D/dx:6.1f}  {vals[i]:12.4e}  {r:>6s}")

# ================================================================
# Convergence plot with reference slopes
# ================================================================
fig, ax = plt.subplots(figsize=(8, 6))
dx_arr_all = np.array(dxs_all[:-1])

# Raw per-grid norms (no coarsening/smoothing): velocity magnitude and
# gauge-removed pressure, each grid vs the finest-grid reference restricted to
# its own resolution.
ax.loglog(dx_arr_all, Linf_u_fluids, "o-",  color="C0", label=r"$L_\infty$ u")
ax.loglog(dx_arr_all, Linf_p_fluids, "o--", color="C1", label=r"$L_\infty$ p")
ax.loglog(dx_arr_all, L2_u_fluids,   "s-",  color="C2", label=r"$L_2$ u")
ax.loglog(dx_arr_all, L2_p_fluids,   "s--", color="C3", label=r"$L_2$ p")

# Reference slopes anchored at coarsest dx
x_ref = np.array([dx_arr_all[0] * 0.8, dx_arr_all[-1] * 1.2])
for order, ls, lbl in [(1, ":", "1st order"), (2, "--", "2nd order")]:
    scale = Linf_u_fluids[0] * (dx_arr_all[0] / x_ref[0]) ** order
    ax.loglog(x_ref, scale * (x_ref / x_ref[0]) ** order,
              color="gray", ls=ls, lw=1.2, label=lbl)

ax.set_xlabel(r"$\Delta x$")
ax.set_ylabel("Error (fluid region, sdf > 0)")
ax.set_title(f"BDIM2 convergence — Re={Re}, t=3 (raw per-grid norms)")
ax.legend(fontsize=10)
ax.grid(True, which="both", ls="--", alpha=0.5)
fig.savefig(os.path.join(figdir, "error_vs_dx_fluid.pdf"), dpi=150, bbox_inches="tight")
plt.close(fig)

