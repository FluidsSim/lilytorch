"""
Plot results from the Coquerelle et al. 2D sphere sedimentation test.

Produces:
  1. Sphere URDF position (x and z) vs time.
  2. Sphere settling velocity uz vs time.
  3. Vertical velocity profile v(x) at t = 0.1 s.
"""

import pathlib
import numpy as np
import h5py
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
HERE      = pathlib.Path(__file__).resolve().parent
FIG_DIR   = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
FMT       = ".png"

DATA_DIR  = pathlib.Path("/data/andreaferrario/ns_data/coquerelle_sphere_drop")
FIELDS_H5 = DATA_DIR / "fields.h5"
HDF5_PATH = DATA_DIR / "output" / "simulation.hdf5"

# ── physical parameters ──────────────────────────────────────────────────
D        = 0.25      # sphere diameter [cm]
R        = D / 2
# Fallback only; prefer inferring dt from simulation.hdf5 times.
dt_fluid = 0.0001
T_PROF   = 0.1       # time for velocity profile [s]

# ── colours ──────────────────────────────────────────────────────────────
C_X  = "#2166AC"
C_Z  = "#B2182B"


# ── helpers ──────────────────────────────────────────────────────────────
def load_grids():
    with h5py.File(FIELDS_H5, "r") as f:
        x = f["grids/x"][:]
        y = f["grids/y"][:]
    return x, y


def load_field(iteration, name):
    with h5py.File(FIELDS_H5, "r") as f:
        return f[f"fields/{iteration:06d}"][name][:]


def list_iterations():
    with h5py.File(FIELDS_H5, "r") as f:
        return sorted(int(k) for k in f["fields"].keys())


def load_hdf5_kinematics():
    """Return (times, com_pos, com_vel) arrays from FARMS HDF5."""
    from farms_core.io.hdf5 import hdf5_to_dict
    from farms_core.sensors.sensor_convention import sc

    data = hdf5_to_dict(str(HDF5_PATH))
    times = data["times"][:-1]
    sa    = data["animats"][0]["sensors"]["links"]["array"][:, 0, :]

    com_pos = sa[:, sc.link_com_position_x : sc.link_com_position_z + 1]
    com_vel = sa[:, sc.link_com_velocity_lin_x : sc.link_com_velocity_lin_z + 1]
    return times, com_pos, com_vel


def infer_dt_from_hdf5(default_dt=dt_fluid):
    """Infer simulation dt from HDF5 time stamps (fallback to default_dt)."""
    if not HDF5_PATH.exists():
        return default_dt
    times, _, _ = load_hdf5_kinematics()
    if len(times) < 2:
        return default_dt
    return float(np.median(np.diff(times)))


def report_settling_metrics():
    """Print peak, tail, and final settling-velocity metrics."""
    if not HDF5_PATH.exists():
        print("  simulation.hdf5 not found – skipping settling metrics.")
        return

    times, com_pos, com_vel = load_hdf5_kinematics()
    z = com_pos[:, 2]
    uz = com_vel[:, 2]

    i_peak = int(np.argmin(uz))
    tail_start = int(0.9 * len(uz))

    print("  Settling metrics:")
    print(f"    dt (from HDF5): {infer_dt_from_hdf5():.8f} s")
    print(f"    peak downward uz: {uz[i_peak]:.6f} cm/s at t={times[i_peak]:.6f}s, z={z[i_peak]:.6f}cm")
    print(f"    mean uz over last 10%: {uz[tail_start:].mean():.6f} cm/s")
    print(f"    final uz: {uz[-1]:.6f} cm/s at t={times[-1]:.6f}s, z={z[-1]:.6f}cm")


# =====================================================================
# 1.  Sphere URDF position vs time
# =====================================================================
def plot_sphere_position():
    if not HDF5_PATH.exists():
        print("  simulation.hdf5 not found – skipping position plot.")
        return

    times, com_pos, _ = load_hdf5_kinematics()

    # MuJoCo: x → fluid x (horizontal), z → fluid y (settling)
    pos_x = com_pos[:, 0]
    pos_z = com_pos[:, 2]

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    axes[0].plot(times, pos_x, color=C_X)
    axes[0].set_ylabel("$x$ [cm]")
    axes[0].set_title("Sphere URDF position – Coquerelle sedimentation")

    axes[1].plot(times, pos_z, color=C_Z)
    axes[1].set_ylabel("$z$ [cm]")
    axes[1].set_xlabel("$t$ [s]")

    fig.tight_layout()
    fig.savefig(str(FIG_DIR / f"coquerelle_sphere_position{FMT}"))
    print(f"  Saved {FIG_DIR / f'coquerelle_sphere_position{FMT}'}")
    plt.close(fig)


# =====================================================================
# 2.  Sphere uz (settling) velocity vs time
# =====================================================================
def plot_sphere_uz():
    if not HDF5_PATH.exists():
        print("  simulation.hdf5 not found – skipping uz plot.")
        return

    times, _, com_vel = load_hdf5_kinematics()

    # MuJoCo z-velocity → settling direction in fluid
    uz = com_vel[:, 2]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, uz, color=C_Z)
    ax.axhline(0, color="0.5", linewidth=0.5, linestyle="--")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel(r"$u_z$ [cm/s]")
    ax.set_title("Sphere settling velocity – Coquerelle sedimentation")
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / f"coquerelle_sphere_uz{FMT}"))
    print(f"  Saved {FIG_DIR / f'coquerelle_sphere_uz{FMT}'}")
    plt.close(fig)


# =====================================================================
# 3.  Vertical velocity profile v(x) at t = T_PROF
# =====================================================================
def plot_v_profile():
    if not FIELDS_H5.exists():
        print("  fields.h5 not found – skipping v-profile plot.")
        return

    # Find the saved iteration closest to T_PROF, skipping snapshots whose
    # SDF-derived body position is anomalous (jumps back to initial value).
    iters     = list_iterations()
    dt_data   = infer_dt_from_hdf5()
    it_target = int(round(T_PROF / dt_data))

    with h5py.File(FIELDS_H5, "r") as f:
        yg_raw = f["grids/y"][:]
        def _sdf_y(it):
            s = f[f"fields/{it:06d}"]["sdf"][1:-1, 1:-1]
            ij = np.unravel_index(np.argmin(s), s.shape)
            return float(yg_raw[1:-1][ij[1]])
        sdf_y0 = _sdf_y(iters[0])
        candidates = [i for i in iters
                      if abs(_sdf_y(i) - sdf_y0) > 0.05]  # must have moved from initial
    if not candidates:
        candidates = iters  # fallback: no valid candidates, use all

    it = min(candidates, key=lambda i: abs(i - it_target))
    t_actual = it * dt_data
    print(f"  v_profile: requested t={T_PROF:.3f}s → using iter {it} (t={t_actual:.4f}s)")

    if not HDF5_PATH.exists():
        print("  simulation.hdf5 not found – FARMS reference unavailable, using SDF only.")

    xg, yg = load_grids()

    v   = load_field(it, "v")
    sdf = load_field(it, "sdf")

    # Sphere centre from SDF minimum (ground truth in fluid coordinates)
    sdf_phys = sdf[1:-1, 1:-1]
    ij_min   = np.unravel_index(np.argmin(sdf_phys), sdf_phys.shape)
    x_sphere_fluid = float(xg[1:-1][ij_min[0]])
    y_sphere_fluid = float(yg[1:-1][ij_min[1]])
    print(f"  SDF centre: x={x_sphere_fluid:.4f}, y={y_sphere_fluid:.4f} cm")

    h = float(yg[1] - yg[0])
    y_stag = yg - h / 2          # v is y-staggered: v[:, j] lives at y_stag[j]
    j_cen = np.argmin(np.abs(y_stag - y_sphere_fluid))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xg[1:-1], v[1:-1, j_cen], color=C_Z, linewidth=1.5)
    ax.axhline(0, color="0.5", linewidth=0.5, linestyle="--")
    ax.axvline(x_sphere_fluid - R, color="0.7", linewidth=0.8, linestyle=":")
    ax.axvline(x_sphere_fluid + R, color="0.7", linewidth=0.8, linestyle=":",
               label=f"sphere extent (x={x_sphere_fluid:.3f}±{R})")
    ax.set_xlabel("$x$ [cm]")
    ax.set_ylabel(r"$v_y$ [cm/s]")
    y_slice = float(y_stag[j_cen])
    ax.set_title(f"Vertical velocity profile at $y = {y_slice:.3f}$ cm "
                 f"(SDF centre $y = {y_sphere_fluid:.3f}$), $t = {t_actual:.4f}$ s")
    ax.legend(framealpha=0.92, edgecolor="0.7")
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / f"coquerelle_v_profile{FMT}"))
    print(f"  Saved {FIG_DIR / f'coquerelle_v_profile{FMT}'}")
    plt.close(fig)

    # ── 2-D imshow of v at the same iteration ──────────────────────────
    # Strip ghost cells; v is y-staggered so y-axis uses y_stag
    xp      = xg[1:-1]
    yp      = y_stag[1:-1]
    v_phys  = v[1:-1, 1:-1]

    vmax_im = np.percentile(np.abs(v_phys), 99)

    fig2, ax2 = plt.subplots(figsize=(5, 9))
    pm = ax2.pcolormesh(xp, yp, v_phys.T,
                        cmap="RdBu_r", vmin=-vmax_im, vmax=vmax_im,
                        shading="auto", rasterized=True)
    fig2.colorbar(pm, ax=ax2, label=r"$v_y$ [cm/s]", shrink=0.6)
    ax2.axhline(y_sphere_fluid, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
    circle = plt.Circle((x_sphere_fluid, y_sphere_fluid), R,
                         fill=False, edgecolor="k", linewidth=1.2)
    ax2.add_patch(circle)
    ax2.set_xlabel("$x$ [cm]")
    ax2.set_ylabel("$y$ [cm]")
    ax2.set_aspect("equal")
    ax2.set_title(f"$v_y$ field at $t = {t_actual:.4f}$ s  (iter {it})")
    fig2.tight_layout()
    fig2.savefig(str(FIG_DIR / f"coquerelle_v_imshow{FMT}"))
    print(f"  Saved {FIG_DIR / f'coquerelle_v_imshow{FMT}'}")
    plt.close(fig2)


# =====================================================================
if __name__ == "__main__":
    print("Plotting Coquerelle sphere sedimentation results …")
    report_settling_metrics()
    plot_sphere_position()
    plot_sphere_uz()
    plot_v_profile()
    print("Done.")
