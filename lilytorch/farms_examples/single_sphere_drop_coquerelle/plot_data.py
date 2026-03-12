"""
Plot results from the Coquerelle et al. 2D sphere sedimentation test.

Produces:
  1. Normalised settling velocity vs normalised time,
     compared with Koumoutsakos & Leonard (1995) reference.
  2. Horizontal velocity and angular velocity vs time.
  3. Velocity-field snapshots.
  4. Vorticity field snapshot.
  5. Trajectory plot.
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.labelsize":    13,
    "axes.titlesize":    13,
    "legend.fontsize":   9.5,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "lines.linewidth":   1.6,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
})

# ── paths ────────────────────────────────────────────────────────────────
HERE       = pathlib.Path(__file__).resolve().parent
FIG_DIR    = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
REF_DIR    = HERE.parent.parent.parent / "data_to_save"
FMT        = ".png"

# Fluid solver output (CPU run)
FLUID_DIR  = pathlib.Path("/data/andreaferrario/ns_data/2026-03-09T17:04:08.820397")
# FARMS HDF5 (from earlier GPU run – has the same model, first available)
HDF5_PATH  = pathlib.Path("/data/andreaferrario/ns_data/2026-03-09T16:58:45.928485/output/simulation.hdf5")

# ── physical parameters ─────────────────────────────────────────────────
D          = 0.25           # sphere diameter
R          = D / 2          # sphere radius
rho_body   = 1.5            # density ratio
rho_fluid  = 1.0
nu         = 0.01           # kinematic viscosity
g_acc      = 980.0          # gravitational acceleration (Coquerelle convention)
dt_fluid   = 1e-5
save_every = 1000

# Terminal velocity estimate:  U_t = sqrt( (4/3) * g * D * (rho_body/rho_fluid - 1) / C_D )
# Using Re ~ 200  →  C_D ~ 0.8  →  U_t ~ 10 (approximate)
# Better: from the Koumoutsakos data the velocity approaches ~1.0 normalised
# The reference paper uses U_t from the stationary solution; we'll compute
# U_t empirically from the fluid data.

# ── colours ──────────────────────────────────────────────────────────────
C_UX  = "#2166AC"
C_UZ  = "#B2182B"
C_ANG = "#4DAF4A"
C_REF = "#636363"


# =====================================================================
# 1.  HDF5-based velocity plot (if available)
# =====================================================================
def plot_velocities_hdf5():
    if not HDF5_PATH.exists():
        print("  No HDF5 found – skipping HDF5 velocity plot.")
        return
    from farms_core.io.hdf5 import hdf5_to_dict
    from farms_core.sensors.sensor_convention import sc

    data  = hdf5_to_dict(str(HDF5_PATH))
    times = data["times"][:-1]
    sa    = data["animats"][0]["sensors"]["links"]["array"][:-1, 0, :]

    com_vel = sa[:, sc.link_com_velocity_lin_x : sc.link_com_velocity_lin_z + 1]
    ang_vel = sa[:, sc.link_com_velocity_ang_x : sc.link_com_velocity_ang_z + 1]
    com_pos = sa[:, sc.link_com_position_x : sc.link_com_position_z + 1]

    # MuJoCo gravity is [0,0,-9.81] but Coquerelle uses g=980
    # MuJoCo x → fluid x, MuJoCo z → fluid y (settling direction)
    vx = com_vel[:, 0]          # horizontal
    vz = -com_vel[:, 2]         # settling (positive = downward, MuJoCo -z)
    vy_mj = com_vel[:, 1]       # MuJoCo y (should be ~0 for 2D)
    ang_y = ang_vel[:, 1]       # rotation around MuJoCo y

    # Estimate U_t from max settling velocity
    U_t_est = max(np.max(np.abs(vz)), 1e-6)
    t_star  = times * U_t_est / D

    # ── reference: Koumoutsakos & Leonard (1995) ──
    ref = np.genfromtxt(str(REF_DIR / "koumoutsatokos_keonard_1995.csv"), delimiter=",")

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # — top: settling velocity + reference —
    ax = axes[0]
    ax.scatter(ref[:, 0], ref[:, 1], s=18, marker="o",
               facecolors="none", edgecolors=C_REF, linewidths=0.8,
               label="Koumoutsakos & Leonard (1995)", zorder=3)
    ax.plot(t_star, vz / U_t_est, color=C_UZ, label=r"$v_z / U_t$ (BDIM)")
    ax.set_ylabel(r"Normalised settling velocity  $v_z / U_t$")
    ax.set_title("Sphere sedimentation – Coquerelle et al.")
    ax.legend(framealpha=0.92, edgecolor="0.7")

    # — bottom: horizontal + angular —
    ax2 = axes[1]
    ax2.plot(t_star, vx / U_t_est, color=C_UX, label=r"$v_x / U_t$")
    ax2.plot(t_star, D * ang_y / U_t_est, color=C_ANG, label=r"$D\omega / U_t$")
    ax2.set_xlabel(r"$t^{*} = t \, U_t / D$")
    ax2.set_ylabel(r"Normalised velocity")
    ax2.legend(framealpha=0.92, edgecolor="0.7")

    fig.tight_layout()
    fig.savefig(str(FIG_DIR / f"coquerelle_com_velocity{FMT}"))
    print(f"  Saved {FIG_DIR / f'coquerelle_com_velocity{FMT}'}")
    plt.close(fig)

    # ── position trajectory ──
    fig3, ax3 = plt.subplots(figsize=(4.5, 5))
    ax3.plot(com_pos[:, 0], com_pos[:, 2], color="#333333", linewidth=1.5)
    ax3.plot(com_pos[0, 0], com_pos[0, 2], "go", markersize=8, label="Start")
    ax3.plot(com_pos[-2, 0], com_pos[-2, 2], "rs", markersize=8, label="End")
    ax3.set_xlabel("$x$ (MuJoCo)")
    ax3.set_ylabel("$z$ (MuJoCo, settling)")
    ax3.set_title("Sphere trajectory")
    ax3.legend(framealpha=0.92)
    ax3.set_aspect("equal")
    fig3.tight_layout()
    fig3.savefig(str(FIG_DIR / f"coquerelle_trajectory{FMT}"))
    print(f"  Saved {FIG_DIR / f'coquerelle_trajectory{FMT}'}")
    plt.close(fig3)


# =====================================================================
# 2.  Velocity-field snapshots
# =====================================================================
def plot_velocity_fields():
    uv_dir = FLUID_DIR / "uv_field"
    if not uv_dir.exists():
        print("  No UV field data – skipping.")
        return

    xg = np.load(str(uv_dir / "x_grid.npy"))
    yg = np.load(str(uv_dir / "y_grid.npy"))
    X, Y = np.meshgrid(xg, yg, indexing="ij")

    iters = sorted(set(
        int(p.stem.split("_")[1])
        for p in uv_dir.glob("u_*.npy")
    ))
    if not iters:
        print("  No snapshots saved – skipping field plots.")
        return

    n_panels = min(4, len(iters))
    idx = np.linspace(0, len(iters) - 1, n_panels, dtype=int)
    chosen = [iters[i] for i in idx]

    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 6),
                             sharey=True, constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    im = None
    for ax, it in zip(axes, chosen):
        u = np.load(str(uv_dir / f"u_{it}.npy"))
        v = np.load(str(uv_dir / f"v_{it}.npy"))
        speed = np.sqrt(u**2 + v**2)
        t_phys = it * dt_fluid

        vmax = max(np.percentile(speed, 99.5), 1e-9)
        im = ax.pcolormesh(X, Y, speed,
                           cmap="inferno", vmin=0, vmax=vmax,
                           shading="auto", rasterized=True)

        # Sphere outline (starts at x=1, y=4, falls in y)
        th = np.linspace(0, 2 * np.pi, 120)
        cx, cy = 1.0, 4.0  # approximate (ignoring settling offset for display)
        ax.plot(cx + R * np.cos(th), cy + R * np.sin(th),
                color="white", linewidth=1.0)

        ax.set_title(f"$t = {t_phys:.4f}$ s\n(iter {it})", fontsize=10)
        ax.set_xlabel("$x$")
        ax.set_aspect("equal")
        # Zoom around sphere
        ax.set_xlim([cx - 3 * D, cx + 3 * D])
        ax.set_ylim([cy - 4 * D, cy + 2 * D])

    axes[0].set_ylabel("$y$")
    fig.colorbar(im, ax=axes, shrink=0.7, label=r"$|\mathbf{u}|$", pad=0.02)
    fig.suptitle("Velocity magnitude – Coquerelle sphere sedimentation",
                 fontsize=13, y=1.01)
    fig.savefig(str(FIG_DIR / f"coquerelle_velocity_field{FMT}"))
    print(f"  Saved {FIG_DIR / f'coquerelle_velocity_field{FMT}'}")
    plt.close(fig)


# =====================================================================
# 3.  Vorticity field
# =====================================================================
def plot_vorticity():
    uv_dir = FLUID_DIR / "uv_field"
    if not uv_dir.exists():
        return
    xg = np.load(str(uv_dir / "x_grid.npy"))
    yg = np.load(str(uv_dir / "y_grid.npy"))
    dx, dy = xg[1] - xg[0], yg[1] - yg[0]

    iters = sorted(set(int(p.stem.split("_")[1]) for p in uv_dir.glob("u_*.npy")))
    if len(iters) < 2:
        return

    it = iters[-1]
    u = np.load(str(uv_dir / f"u_{it}.npy"))
    v = np.load(str(uv_dir / f"v_{it}.npy"))
    t_phys = it * dt_fluid
    omega = np.gradient(v, dx, axis=0) - np.gradient(u, dy, axis=1)
    X, Y = np.meshgrid(xg, yg, indexing="ij")

    vlim = np.percentile(np.abs(omega), 99)

    fig, ax = plt.subplots(figsize=(5.5, 6))
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    im = ax.pcolormesh(X, Y, omega, cmap="RdBu_r", norm=norm,
                       shading="auto", rasterized=True)

    th = np.linspace(0, 2 * np.pi, 120)
    cx, cy = 1.0, 4.0
    ax.plot(cx + R * np.cos(th), cy + R * np.sin(th), "k-", linewidth=1.0)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim([cx - 3 * D, cx + 3 * D])
    ax.set_ylim([cy - 4 * D, cy + 2 * D])
    ax.set_aspect("equal")
    ax.set_title(f"Vorticity $\\omega_z$ at $t = {t_phys:.4f}$ s")
    fig.colorbar(im, ax=ax, shrink=0.6, label=r"$\omega_z$ [1/s]")
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / f"coquerelle_vorticity{FMT}"))
    print(f"  Saved {FIG_DIR / f'coquerelle_vorticity{FMT}'}")
    plt.close(fig)


# =====================================================================
# 4.  Vertical velocity profile at sphere centre
# =====================================================================
def plot_v_profile():
    uv_dir = FLUID_DIR / "uv_field"
    if not uv_dir.exists():
        return
    xg = np.load(str(uv_dir / "x_grid.npy"))
    yg = np.load(str(uv_dir / "y_grid.npy"))

    iters = sorted(set(int(p.stem.split("_")[1]) for p in uv_dir.glob("v_*.npy")))
    if not iters:
        return

    it = iters[-1]
    v = np.load(str(uv_dir / f"v_{it}.npy"))
    t_phys = it * dt_fluid

    cy = 4.0   # approximate sphere y-centre
    j_cen = np.argmin(np.abs(yg - cy))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(xg, v[:, j_cen], color=C_UZ, linewidth=1.5)
    ax.axhline(0, color="0.5", linewidth=0.5, linestyle="--")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$v_y$")
    ax.set_title(f"Vertical velocity profile at $y \\approx {cy:.1f}$, $t = {t_phys:.4f}$ s")
    ax.set_xlim([0.4, 1.6])
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / f"coquerelle_v_profile{FMT}"))
    print(f"  Saved {FIG_DIR / f'coquerelle_v_profile{FMT}'}")
    plt.close(fig)


# =====================================================================
if __name__ == "__main__":
    print("Plotting Coquerelle sphere sedimentation results …")
    plot_velocities_hdf5()
    plot_velocity_fields()
    plot_vorticity()
    plot_v_profile()
    print("Done.")

