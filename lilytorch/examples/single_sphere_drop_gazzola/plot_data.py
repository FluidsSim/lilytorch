"""
Plot results from the namkoong et al. 2D circle sedimentation test.

Reads fluid fields from ``fields.h5`` and body contours from
``contours.h5`` (the new unified HDF5 storage).

Produces:
  1. Normalised COM velocity (ux, uz, omega) vs normalised time,
     compared with reference data from namkoong et al. 2008
  2. Velocity-field snapshots at selected time steps.
  3. Vorticity field snapshot.
  4. Kinetic & potential energy evolution.
"""

import pathlib
import numpy as np
import h5py
import matplotlib
# matplotlib.use("Agg")
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
FMT        = ".svg"

# Data directory containing fields.h5, contours.h5, and output/simulation.hdf5
DATA_DIR   = pathlib.Path("/data/andreaferrario/ns_data/namkoong_sphere_drop")
FIELDS_H5  = DATA_DIR / "fields.h5"
CONTOURS_H5 = DATA_DIR / "contours.h5"
HDF5_PATH  = DATA_DIR / "output" / "simulation.hdf5"

# ── physical parameters ─────────────────────────────────────────────────
D         = 0.005                    # circle diameter  [m]
R         = D / 2
rho_body  = 1005.96                  # kg/m^3  (matches simulation_config.yaml)
rho_fluid = 996.0                    # kg/m^3
nu        = 8e-7                     # kinematic viscosity [m^2/s]
g_acc     = 9.81                     # gravitational acceleration [m/s^2]
U_t       = -0.025                 # terminal settling velocity [m/s]
mass_2d   = rho_body * np.pi * R**2
dt_fluid  = 1e-3
xlim      = (0, 135)

# ── colours ──────────────────────────────────────────────────────────────
C_UX  = "#2166AC"   # blue   – horizontal velocity
C_UZ  = "#B2182B"   # red    – vertical (settling) velocity
C_ANG = "#4DAF4A"   # green  – angular velocity


# ── helpers ──────────────────────────────────────────────────────────────
def load_grids():
    """Return (x, y) 1-D grid arrays from fields.h5."""
    with h5py.File(FIELDS_H5, "r") as f:
        x = f["grids/x"][:]
        y = f["grids/y"][:]
    return x, y


def load_field(iteration, name):
    """Load a single field array for a given iteration."""
    grp = f"fields/{iteration:06d}"
    with h5py.File(FIELDS_H5, "r") as f:
        return f[grp][name][:]


def list_iterations():
    """Return sorted list of saved iteration numbers."""
    with h5py.File(FIELDS_H5, "r") as f:
        return sorted(int(k) for k in f["fields"].keys())


def load_contour(iteration, body_idx=0):
    """Load body contour (2, N) for a given iteration from contours.h5."""
    if not CONTOURS_H5.exists():
        return None
    with h5py.File(CONTOURS_H5, "r") as f:
        key = f"{iteration:06d}/body_{body_idx}"
        if key in f:
            return f[key][:]
    return None


# =====================================================================
# 1.  Normalised velocity plot  (HDF5 body kinematics)
# =====================================================================
def plot_velocities():
    if not HDF5_PATH.exists():
        print("  simulation.hdf5 not found – skipping velocity plot.")
        return

    from farms_core.io.hdf5 import hdf5_to_dict
    from farms_core.sensors.sensor_convention import sc

    data  = hdf5_to_dict(str(HDF5_PATH))
    times = data["times"][:-1]
    sa    = data["animats"][0]["sensors"]["links"]["array"][:, 0, :]

    com_vel = sa[:, sc.link_com_velocity_lin_x : sc.link_com_velocity_lin_z + 1]
    ang_vel = sa[:, sc.link_com_velocity_ang_x : sc.link_com_velocity_ang_z + 1]
    z_pos   = sa[:, sc.link_com_position_y]  # MuJoCo y = vertical (gravity)

    # Normalised time: t* = t |U_t| / D
    t_star = times * abs(U_t) / D

    # Normalised velocities
    # MuJoCo x -> fluid x (horizontal), MuJoCo y -> vertical (settling)
    ux_norm  =  com_vel[:, 0] / U_t
    uz_norm  = com_vel[:, 2] / U_t   # negate: positive = downward settling
    ang_norm =  -D * ang_vel[:, 1] / U_t

    # from IPython import embed; embed()

    # ── reference data (digitised from namkoong et al.) ──
    ref_up = np.genfromtxt(str(REF_DIR / "up.csv"), delimiter=",")   # settling vel
    ref_vp = np.genfromtxt(str(REF_DIR / "vp.csv"), delimiter=",")   # horizontal
    ref_wp = np.genfromtxt(str(REF_DIR / "wp.csv"), delimiter=",")   # angular

    # Restrict reference data to t* ≤ 120
    ref_up = ref_up[ref_up[:, 0] <= 120]
    ref_vp = ref_vp[ref_vp[:, 0] <= 120]
    ref_wp = ref_wp[ref_wp[:, 0] <= 120]

    # ── velocity figure ──
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Reference (open markers)
    ax.scatter(ref_vp[:, 0], ref_vp[:, 1], s=16, marker=".",
               facecolors="none", edgecolors=C_UX, linewidths=0.8,
               label=r"$u_x/U_t$ (namkoong et al.)")
    ax.scatter(ref_up[:, 0], ref_up[:, 1], s=16, marker=".",
               facecolors="none", edgecolors=C_UZ, linewidths=0.8,
               label=r"$u_z/U_t$ (namkoong et al.)", zorder=3)
    ax.scatter(ref_wp[:, 0], ref_wp[:, 1], s=16, marker=".",
               facecolors="none", edgecolors=C_ANG, linewidths=0.8,
               label=r"$D\omega/U_t$ (namkoong et al.)", zorder=3)

    # Simulation (solid lines)
    ax.plot(t_star, ux_norm,  color=C_UX,  label=r"OUR", linewidth=1)
    ax.plot(t_star, uz_norm,  color=C_UZ,  label=r"OUR", linewidth=1)
    ax.plot(t_star, ang_norm, color=C_ANG, label=r"OUR", linewidth=1)

    ax.set_xlabel(r"$t^{*} = t\,|U_t|/D$")
    ax.set_ylabel(r"Normalised velocity")
    ax.set_title("circle sedimentation")
    ax.set_ylim([-0.5, 1.3])
    ax.legend(ncol=2, loc="center", framealpha=0.92, edgecolor="0.7")
    ax.set_xlim(xlim)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / f"namkoong_com_velocity{FMT}"))
    print(f"  Saved {FIG_DIR / f'namkoong_com_velocity{FMT}'}")
    plt.close(fig)

    # ── energy figure ──
    KE = 0.5 * mass_2d * (com_vel[:, 0]**2 + com_vel[:, 1]**2)
    PE = mass_2d * g_acc * z_pos

    fig2, ax2 = plt.subplots(figsize=(6, 3.8))
    ax2.plot(t_star, 1e6 * KE, color="#D95F02", label="Kinetic energy")
    ax2.plot(t_star, 1e6 * PE, color="#1B9E77", label="Potential energy")
    ax2.set_xlabel(r"$t^{*}$")
    ax2.set_ylabel(r"Energy  [$\mu$J]")
    ax2.set_title("Energy evolution – namkoong")
    ax2.legend(framealpha=0.92, edgecolor="0.7")
    fig2.tight_layout()
    fig2.savefig(str(FIG_DIR / f"namkoong_energy{FMT}"))
    print(f"  Saved {FIG_DIR / f'namkoong_energy{FMT}'}")
    plt.close(fig2)


# =====================================================================
# 2.  Velocity-field snapshots (from fields.h5)
# =====================================================================
def plot_velocity_fields():
    if not FIELDS_H5.exists():
        print("  fields.h5 not found – skipping velocity field plots.")
        return

    xg, yg = load_grids()
    iters = list_iterations()

    if not iters:
        print("  No field data – skipping.")
        return

    # Pick 4 evenly spaced snapshots
    n_panels = min(4, len(iters))
    idx = np.linspace(0, len(iters) - 1, n_panels, dtype=int)
    chosen = [iters[i] for i in idx]

    fig, axes = plt.subplots(1, n_panels, figsize=(3.8 * n_panels, 7),
                             sharey=True, constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    im = None
    for ax, it in zip(axes, chosen):
        u = load_field(it, "u")
        v = load_field(it, "v")
        speed = np.sqrt(u**2 + v**2)
        t_phys = it * dt_fluid

        # Build coordinate arrays matching the actual (Ny, Nx) field shape
        ny_f, nx_f = speed.shape
        x_f = np.linspace(xg[0], xg[-1], nx_f)
        y_f = np.linspace(yg[0], yg[-1], ny_f)
        Xf, Yf = np.meshgrid(x_f, y_f)

        vmax = max(np.percentile(speed, 99.5), 1e-9)
        im = ax.pcolormesh(Xf * 1e3, Yf * 1e3, speed,
                           cmap="inferno", vmin=0, vmax=vmax,
                           shading="nearest", rasterized=True)

        # circle outline from contours.h5 (fall back to approximate circle)
        cnt = load_contour(it)
        if cnt is not None:
            ax.plot(cnt[0] * 1e3, cnt[1] * 1e3,
                    color="white", linewidth=0.8)
        else:
            th = np.linspace(0, 2 * np.pi, 120)
            cx, cy = 0.0, 0.3 - abs(U_t) * t_phys
            ax.plot((cx + R * np.cos(th)) * 1e3,
                    (cy + R * np.sin(th)) * 1e3,
                    color="white", linewidth=0.8)

        ax.set_title(f"$t = {t_phys * 1e3:.1f}$ ms", fontsize=10)
        ax.set_xlabel("$x$ [mm]")
        ax.set_aspect("equal")

        # Zoom around circle using SDF or contour centroid
        if cnt is not None:
            cx = np.mean(cnt[0])
            cy = np.mean(cnt[1])
        else:
            cx, cy = 0.0, 0.3 - abs(U_t) * t_phys
        ax.set_xlim([(cx - 4 * R) * 1e3, (cx + 4 * R) * 1e3])
        ax.set_ylim([(cy - 6 * R) * 1e3, (cy + 6 * R) * 1e3])

    axes[0].set_ylabel("$y$ [mm]")
    fig.colorbar(im, ax=axes, shrink=0.7, label=r"$|\mathbf{u}|$ [m/s]",
                 pad=0.02)
    fig.suptitle("Velocity magnitude – namkoong circle sedimentation",
                 fontsize=13, y=1.01)
    fig.savefig(str(FIG_DIR / f"namkoong_velocity_field{FMT}"))
    print(f"  Saved {FIG_DIR / f'namkoong_velocity_field{FMT}'}")
    plt.close(fig)


# =====================================================================
# 3.  Vorticity field snapshot
# =====================================================================
def plot_vorticity():
    if not FIELDS_H5.exists():
        print("  fields.h5 not found – skipping vorticity plot.")
        return

    xg, yg = load_grids()
    dx, dy = xg[1] - xg[0], yg[1] - yg[0]
    iters = list_iterations()

    if len(iters) < 2:
        return

    it = iters[-1]
    u = load_field(it, "u")
    v = load_field(it, "v")
    t_phys = it * dt_fluid

    omega = np.gradient(v, dx, axis=0) - np.gradient(u, dy, axis=1)
    X, Y = np.meshgrid(xg, yg, indexing="ij")
    vlim = np.percentile(np.abs(omega), 99)

    fig, ax = plt.subplots(figsize=(5, 7))
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    im = ax.pcolormesh(X * 1e3, Y * 1e3, omega,
                       cmap="RdBu_r", norm=norm,
                       shading="nearest", rasterized=True)

    # circle outline from contours.h5
    cnt = load_contour(it)
    if cnt is not None:
        ax.plot(cnt[0] * 1e3, cnt[1] * 1e3, "k-", linewidth=0.8)
    else:
        th = np.linspace(0, 2 * np.pi, 120)
        cx, cy = 0.0, 0.3 - abs(U_t) * t_phys
        ax.plot((cx + R * np.cos(th)) * 1e3,
                (cy + R * np.sin(th)) * 1e3, "k-", linewidth=0.8)

    # Zoom around circle
    if cnt is not None:
        cx = np.mean(cnt[0])
        cy = np.mean(cnt[1])
    else:
        cx, cy = 0.0, 0.3 - abs(U_t) * t_phys

    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$y$ [mm]")
    ax.set_xlim([(cx - 5 * R) * 1e3, (cx + 5 * R) * 1e3])
    ax.set_ylim([(cy - 8 * R) * 1e3, (cy + 8 * R) * 1e3])
    ax.set_aspect("equal")
    ax.set_title(f"Vorticity  $\\omega_z$  at  $t = {t_phys * 1e3:.1f}$ ms")
    fig.colorbar(im, ax=ax, shrink=0.55, label=r"$\omega_z$ [1/s]")
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / f"namkoong_vorticity{FMT}"))
    print(f"  Saved {FIG_DIR / f'namkoong_vorticity{FMT}'}")
    plt.close(fig)


# =====================================================================
# 4.  Vertical velocity profile at the last saved snapshot
# =====================================================================
def plot_v_profile():
    if not FIELDS_H5.exists():
        print("  fields.h5 not found – skipping v-profile plot.")
        return

    xg, yg = load_grids()
    iters = list_iterations()

    if not iters:
        return

    it = iters[-1]
    v = load_field(it, "v")
    t_phys = it * dt_fluid

    # Find circle centre from contour or estimate
    cnt = load_contour(it)
    if cnt is not None:
        cy = np.mean(cnt[1])
    else:
        cy = 0.3 - abs(U_t) * t_phys
    j_cen = np.argmin(np.abs(yg - cy))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(xg * 1e3, v[:, j_cen] / U_t, color=C_UZ, linewidth=1.5)
    ax.axhline(0, color="0.5", linewidth=0.5, linestyle="--")
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel(r"$v_y / U_t$")
    ax.set_title(f"Vertical velocity profile at $t = {t_phys * 1e3:.1f}$ ms")
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / f"namkoong_v_profile{FMT}"))
    print(f"  Saved {FIG_DIR / f'namkoong_v_profile{FMT}'}")
    plt.close(fig)


# =====================================================================
# 5.  Check: fluid velocity inside the body vs body u_z
# =====================================================================
def check_body_velocity():
    """Compare the fluid v-field inside the body (SDF < 0) with u_z from FARMS.

    For a rigid body the BDIM meta-equation enforces u → u_body inside,
    so v (settling direction) averaged over cells deep inside the body
    should closely match the COM settling velocity u_z from MuJoCo.
    """
    if not FIELDS_H5.exists() or not HDF5_PATH.exists():
        print("  fields.h5 or simulation.hdf5 not found – skipping body-vel check.")
        return

    from farms_core.io.hdf5 import hdf5_to_dict
    from farms_core.sensors.sensor_convention import sc

    # Body kinematics (every iteration)
    data  = hdf5_to_dict(str(HDF5_PATH))
    times = data["times"][:-1]
    sa    = data["animats"][0]["sensors"]["links"]["array"][:, 0, :]
    com_vel = sa[:, sc.link_com_velocity_lin_x : sc.link_com_velocity_lin_z + 1]

    xg, yg = load_grids()
    iters  = list_iterations()

    print(f"\n  {'iter':>6s}  {'u_z(body)':>12s}  {'<v>_inside':>12s}  "
          f"{'|diff|':>12s}  {'rel_err':>10s}  {'#cells':>6s}")
    print("  " + "-" * 70)

    uz_body_arr = []
    v_inside_arr = []
    t_arr = []

    for it in iters:
        if it >= len(times):
            continue

        v_field   = load_field(it, "v")
        sdf_field = load_field(it, "sdf")

        # Cells deep inside the body (SDF < -eps, i.e. well inside)
        eps = xg[1] - xg[0]         # one grid spacing
        mask = sdf_field < -eps

        if mask.sum() == 0:
            continue

        # Mean fluid v inside the body
        v_mean_inside = np.mean(v_field[mask])

        # Body settling velocity: MuJoCo z → fluid y
        # In the namkoong controller: lin_axes = [0, 2], so
        # fluid y-velocity = MuJoCo z-velocity = com_vel[:, 2]
        uz_body = float(com_vel[it, 2])

        diff = abs(v_mean_inside - uz_body)
        rel  = diff / max(abs(uz_body), 1e-30)

        uz_body_arr.append(uz_body)
        v_inside_arr.append(v_mean_inside)
        t_arr.append(times[it])

        print(f"  {it:6d}  {uz_body:12.6e}  {v_mean_inside:12.6e}  "
              f"{diff:12.6e}  {rel:10.4e}  {int(mask.sum()):6d}")

    # ── Plot comparison ──
    if len(t_arr) > 1:
        t_arr = np.array(t_arr)
        t_star = t_arr * abs(U_t) / D

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        sharex=True)
        ax1.plot(t_star, uz_body_arr, "o-", color=C_UZ, ms=4,
                 label=r"$u_z$ (body COM)")
        ax1.plot(t_star, v_inside_arr, "s--", color=C_UX, ms=4,
                 label=r"$\langle v \rangle$ inside body")
        ax1.set_ylabel("Velocity [m/s]")
        ax1.set_title("Body velocity vs fluid velocity inside body")
        ax1.legend()

        rel_err = np.abs(np.array(v_inside_arr) - np.array(uz_body_arr)) / np.maximum(np.abs(uz_body_arr), 1e-30)
        ax2.semilogy(t_star, rel_err, "k.-")
        ax2.set_xlabel(r"$t^{*} = t\,|U_t|/D$")
        ax2.set_ylabel("Relative error")
        ax2.set_ylim(bottom=1e-6)

        fig.tight_layout()
        fig.savefig(str(FIG_DIR / f"namkoong_body_vel_check{FMT}"))
        print(f"\n  Saved {FIG_DIR / f'namkoong_body_vel_check{FMT}'}")
        plt.close(fig)


# =====================================================================
if __name__ == "__main__":
    print("Plotting namkoong circle sedimentation results …")
    plot_velocities()
    plot_velocity_fields()
    plot_vorticity()
    plot_v_profile()
    check_body_velocity()
    print("Done.")
