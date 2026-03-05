"""
Plot OpenFOAM + preCICE coupling results.

Generates:
  1. Force/moment time history from postProcessing/forces
  2. Velocity magnitude contour (2D mid-plane slice) at each saved timestep
  3. Ux and Uy velocity component contours
  4. Pressure contour (2D mid-plane slice) at each saved timestep
  5. Vorticity (omega_z) contour from OpenFOAM postProcess
  6. preCICE convergence log

Usage:
    python plot_results.py [/path/to/output/dir]

If no argument given, uses the latest directory in /data/andreaferrario/ns_data/.
"""

import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def find_latest_output():
    """Find the latest output directory."""
    base = "/data/andreaferrario/ns_data"
    dirs = sorted(glob.glob(os.path.join(base, "2026-*")))
    if not dirs:
        raise FileNotFoundError(f"No output directories in {base}")
    return dirs[-1]


def parse_force_dat(filepath):
    """Parse OpenFOAM force.dat or moment.dat file.

    Returns dict with keys: time, total, pressure, viscous  (each Nx3).
    Handles duplicate entries per timestep (one per PIMPLE iteration) by
    keeping only the last entry for each time value.
    """
    times, total, pressure, viscous = [], [], [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            t = float(parts[0])
            vals = [float(v) for v in parts[1:]]
            # total_x,y,z  pressure_x,y,z  viscous_x,y,z
            times.append(t)
            total.append(vals[0:3])
            pressure.append(vals[3:6])
            viscous.append(vals[6:9])

    times = np.array(times)
    total = np.array(total)
    pressure = np.array(pressure)
    viscous = np.array(viscous)

    # Keep only last entry per timestep (last PIMPLE iteration)
    _, idx = np.unique(times, return_index=True)
    # Actually we want the LAST occurrence, not first
    _, inv = np.unique(times[::-1], return_index=True)
    last_idx = len(times) - 1 - inv
    last_idx = np.sort(last_idx)

    return {
        "time": times[last_idx],
        "total": total[last_idx],
        "pressure": pressure[last_idx],
        "viscous": viscous[last_idx],
    }


def plot_forces(outdir, figdir):
    """Plot force and moment time histories."""
    force_file = os.path.join(outdir, "openfoam_case/postProcessing/forces/0/force.dat")
    moment_file = os.path.join(outdir, "openfoam_case/postProcessing/forces/0/moment.dat")

    if not os.path.exists(force_file):
        print(f"  No force.dat found at {force_file}")
        return

    force = parse_force_dat(force_file)
    moment = parse_force_dat(moment_file) if os.path.exists(moment_file) else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Force components
    ax = axes[0, 0]
    ax.plot(force["time"] * 1000, force["total"][:, 0], "b-", label="Fx total")
    ax.plot(force["time"] * 1000, force["pressure"][:, 0], "b--", label="Fx pressure", alpha=0.6)
    ax.plot(force["time"] * 1000, force["viscous"][:, 0], "b:", label="Fx viscous", alpha=0.6)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Force [N]")
    ax.set_title("X-Force (Drag direction)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(force["time"] * 1000, force["total"][:, 1], "r-", label="Fy total")
    ax.plot(force["time"] * 1000, force["pressure"][:, 1], "r--", label="Fy pressure", alpha=0.6)
    ax.plot(force["time"] * 1000, force["viscous"][:, 1], "r:", label="Fy viscous", alpha=0.6)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Force [N]")
    ax.set_title("Y-Force (Lateral)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(force["time"] * 1000, force["total"][:, 2], "g-", label="Fz total")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Force [N]")
    ax.set_title("Z-Force")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if moment is not None:
        ax = axes[1, 1]
        ax.plot(moment["time"] * 1000, moment["total"][:, 0], "b-", label="Mx")
        ax.plot(moment["time"] * 1000, moment["total"][:, 1], "r-", label="My")
        ax.plot(moment["time"] * 1000, moment["total"][:, 2], "g-", label="Mz")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Moment [N·m]")
        ax.set_title("Moments")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Forces on swimmer body", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "forces.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved forces.png")


def plot_precice_convergence(outdir, figdir):
    """Plot preCICE convergence history."""
    conv_file = os.path.join(outdir, "precice-OpenFOAM-convergence.log")
    if not os.path.exists(conv_file):
        print(f"  No convergence log found")
        return

    data = np.loadtxt(conv_file, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Columns: TimeWindow, Iteration, ResRel(Displacement), ResRel(Force)
    tw = data[:, 0]
    it = data[:, 1]
    res_disp = data[:, 2]
    res_force = data[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    # Color by time window
    unique_tw = np.unique(tw)
    for t in unique_tw:
        mask = tw == t
        ax.semilogy(it[mask], np.maximum(res_disp[mask], 1e-16), "o-", markersize=3,
                     label=f"TW {int(t)}" if t <= 10 else None)
    ax.set_xlabel("Iteration within time window")
    ax.set_ylabel("Relative residual")
    ax.set_title("Displacement convergence")
    ax.grid(True, alpha=0.3)
    if len(unique_tw) <= 10:
        ax.legend(fontsize=7)

    ax = axes[1]
    for t in unique_tw:
        mask = tw == t
        ax.semilogy(it[mask], np.maximum(res_force[mask], 1e-16), "o-", markersize=3,
                     label=f"TW {int(t)}" if t <= 10 else None)
    ax.set_xlabel("Iteration within time window")
    ax.set_ylabel("Relative residual")
    ax.set_title("Force convergence")
    ax.grid(True, alpha=0.3)
    if len(unique_tw) <= 10:
        ax.legend(fontsize=7)

    fig.suptitle("preCICE coupling convergence", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "precice_convergence.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved precice_convergence.png")


def _slice_to_matplotlib(internal, field_name, z_mid=0.2):
    """Slice an OpenFOAM internal mesh at z=z_mid, return (x, y, values) on a
    point-based triangulation suitable for matplotlib tricontourf.

    Returns (Triangulation, values) or (None, None) on failure.
    """
    from matplotlib.tri import Triangulation

    sliced = internal.slice(normal="z", origin=(0, 0, z_mid))
    if sliced is None or (sliced.n_cells == 0 and sliced.n_points == 0):
        return None, None

    # Move cell data to point data if needed
    if field_name not in sliced.point_data and field_name in sliced.cell_data:
        sliced = sliced.cell_data_to_point_data()
    elif field_name not in sliced.point_data and field_name not in sliced.cell_data:
        return None, None

    # Triangulate
    tri = sliced.triangulate()
    x, y = tri.points[:, 0], tri.points[:, 1]

    vals = tri.point_data[field_name]

    # Build connectivity
    faces_flat = tri.faces
    n_faces = tri.n_cells
    # pyvista faces: [n_pts, id0, id1, id2, n_pts, id0, ...]
    faces_arr = faces_flat.reshape(n_faces, 4)[:, 1:4]
    triang = Triangulation(x, y, faces_arr)

    return triang, vals


def _slice_and_plot(internal, field_name, t_val, t_name, figdir, z_mid=0.2,
                     cmap="RdBu_r", label="", title_prefix="", clamp_pct=99):
    """Slice a field at z_mid and save a contour plot. Fast: works on pre-sliced data."""
    from matplotlib.tri import Triangulation

    sliced = internal.slice(normal="z", origin=(0, 0, z_mid))
    if sliced is None or sliced.n_cells == 0:
        return

    # Move cell data to point data if needed
    if field_name not in sliced.point_data and field_name in sliced.cell_data:
        sliced = sliced.cell_data_to_point_data()
    elif field_name not in sliced.point_data:
        return

    tri = sliced.triangulate()
    x, y = tri.points[:, 0], tri.points[:, 1]
    vals = tri.point_data[field_name]

    faces_arr = tri.faces.reshape(tri.n_cells, 4)[:, 1:4]
    triang = Triangulation(x, y, faces_arr)

    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.isfinite(vals) & (np.abs(vals) < 1e10)
    vmax = np.percentile(np.abs(vals[finite]), clamp_pct) if finite.any() else 1.0
    vmax = max(vmax, 0.01)
    levels = np.linspace(-vmax, vmax, 50)

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    cf = ax.tricontourf(triang, vals, levels=levels, cmap=cmap, extend="both")
    plt.colorbar(cf, ax=ax, label=label, shrink=0.8)
    ax.set_xlim(-0.9, 1.5)
    ax.set_ylim(-0.3, 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{title_prefix} at t = {t_val*1000:.1f} ms (z = {z_mid} slice)")
    fig.tight_layout()
    fname = f"{title_prefix.lower().replace(' ', '_')}_t{t_name}.png"
    fig.savefig(os.path.join(figdir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_field_slices(outdir, figdir, only_times=None):
    """Plot vorticity and pressure on z-mid-plane slice for each timestep.

    Key optimisation: slice the 3D mesh (~1M cells) to 2D FIRST (~few thousand
    cells), THEN compute_derivative on the tiny slice for vorticity.

    Args:
        only_times: optional set of timestep directory names (str) to plot.
                    If None, plots all available timesteps.
    """
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
    except ImportError:
        print("  pyvista not installed — skipping field contour plots")
        return

    case_dir = os.path.join(outdir, "openfoam_case")

    # Find reconstructed time directories (not processor*)
    time_dirs = []
    for d in os.listdir(case_dir):
        full = os.path.join(case_dir, d)
        if os.path.isdir(full) and d != "0" and d != "0.orig" and not d.startswith("processor"):
            try:
                t = float(d)
                if os.path.exists(os.path.join(full, "U")) and os.path.exists(os.path.join(full, "p")):
                    time_dirs.append((t, d))
            except ValueError:
                pass

    time_dirs.sort()
    if only_times is not None:
        time_dirs = [(t, d) for t, d in time_dirs if d in only_times]
    if not time_dirs:
        print("  No reconstructed time directories found — run reconstructPar first")
        return

    print(f"  Found {len(time_dirs)} timesteps to plot: {[t for t, _ in time_dirs]}")

    # Create dummy .foam file for pyvista
    foam_path = os.path.join(case_dir, "foam.foam")
    if not os.path.exists(foam_path):
        open(foam_path, "w").close()

    try:
        reader = pv.OpenFOAMReader(foam_path)
    except Exception as e:
        print(f"  Could not open OpenFOAM case with pyvista: {e}")
        return

    available_times = reader.time_values
    print(f"  Available reader times: {list(available_times)}")

    z_mid = 0.2

    for t_val, t_name in time_dirs:
        print(f"  Processing t = {t_name} ...")
        idx = np.argmin(np.abs(np.array(available_times) - t_val))
        reader.set_active_time_value(available_times[idx])

        try:
            mesh = reader.read()
        except Exception as e:
            print(f"  Error reading time {t_name}: {e}")
            continue

        # Get internal mesh
        if isinstance(mesh, pv.MultiBlock):
            internal = mesh["internalMesh"] if "internalMesh" in mesh.keys() else mesh[0]
        else:
            internal = mesh

        if internal is None or internal.n_cells == 0:
            print(f"  Empty mesh at time {t_name}")
            continue

        # ---- Slice to 2D (fast: 1M cells → few thousand) ----
        sliced_2d = internal.slice(normal="z", origin=(0, 0, z_mid))
        if sliced_2d is None or sliced_2d.n_cells == 0:
            print(f"  Empty slice at time {t_name}")
            continue

        sliced_2d = sliced_2d.cell_data_to_point_data()

        # Compute derived quantities from U
        if "U" in sliced_2d.point_data:
            U = sliced_2d.point_data["U"]
            sliced_2d.point_data["Umag"] = np.linalg.norm(U, axis=1)
            sliced_2d.point_data["Ux"] = U[:, 0]
            sliced_2d.point_data["Uy"] = U[:, 1]

        # Extract omega_z from OpenFOAM-computed vorticity field
        if "vorticity" in sliced_2d.point_data:
            sliced_2d.point_data["omega_z"] = sliced_2d.point_data["vorticity"][:, 2]

        # Triangulate once for all plots
        from matplotlib.tri import Triangulation
        tri = sliced_2d.triangulate()
        x, y = tri.points[:, 0], tri.points[:, 1]
        faces_arr = tri.faces.reshape(tri.n_cells, 4)[:, 1:4]
        triang = Triangulation(x, y, faces_arr)

        # --- Helper for symmetric (diverging) contour ---
        def _plot_symmetric(vals, fname, cmap, label, title):
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            finite = np.isfinite(vals) & (np.abs(vals) < 1e10)
            vmax = np.percentile(np.abs(vals[finite]), 99) if finite.any() else 1.0
            vmax = max(vmax, 0.01)
            levels = np.linspace(-vmax, vmax, 50)
            fig, ax = plt.subplots(1, 1, figsize=(16, 4))
            cf = ax.tricontourf(triang, vals, levels=levels, cmap=cmap, extend="both")
            plt.colorbar(cf, ax=ax, label=label, shrink=0.8)
            ax.set_xlim(-0.9, 1.5); ax.set_ylim(-0.3, 0.3); ax.set_aspect("equal")
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            ax.set_title(f"{title} at t = {t_val*1000:.1f} ms (z = {z_mid} slice)")
            fig.tight_layout()
            fig.savefig(os.path.join(figdir, fname), dpi=150)
            plt.close(fig)
            print(f"  Saved {fname}")

        # --- Velocity magnitude ---
        if "Umag" in tri.point_data:
            try:
                umag = np.nan_to_num(tri.point_data["Umag"], nan=0.0, posinf=0.0, neginf=0.0)
                vmax = max(np.percentile(umag[umag > 0], 99), 0.01) if (umag > 0).any() else 0.3
                levels = np.linspace(0, vmax, 50)
                fig, ax = plt.subplots(1, 1, figsize=(16, 4))
                cf = ax.tricontourf(triang, umag, levels=levels, cmap="viridis", extend="max")
                plt.colorbar(cf, ax=ax, label="|U| [m/s]", shrink=0.8)
                ax.set_xlim(-0.9, 1.5); ax.set_ylim(-0.3, 0.3); ax.set_aspect("equal")
                ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
                ax.set_title(f"Velocity magnitude at t = {t_val*1000:.1f} ms (z = {z_mid} slice)")
                fig.tight_layout()
                fig.savefig(os.path.join(figdir, f"velocity_t{t_name}.png"), dpi=150)
                plt.close(fig)
                print(f"  Saved velocity_t{t_name}.png")
            except Exception as e:
                print(f"  Error plotting velocity at {t_name}: {e}")

        # --- Ux component ---
        if "Ux" in tri.point_data:
            try:
                _plot_symmetric(tri.point_data["Ux"], f"Ux_t{t_name}.png",
                                "RdBu_r", "Ux [m/s]", "Velocity Ux")
            except Exception as e:
                print(f"  Error plotting Ux at {t_name}: {e}")

        # --- Uy component ---
        if "Uy" in tri.point_data:
            try:
                _plot_symmetric(tri.point_data["Uy"], f"Uy_t{t_name}.png",
                                "RdBu_r", "Uy [m/s]", "Velocity Uy")
            except Exception as e:
                print(f"  Error plotting Uy at {t_name}: {e}")

        # --- Pressure ---
        if "p" in tri.point_data:
            try:
                _plot_symmetric(tri.point_data["p"], f"pressure_t{t_name}.png",
                                "RdBu_r", "p [Pa/ρ]", "Pressure")
            except Exception as e:
                print(f"  Error plotting pressure at {t_name}: {e}")

        # --- Vorticity omega_z (from OpenFOAM postProcess) ---
        if "omega_z" in tri.point_data:
            try:
                _plot_symmetric(tri.point_data["omega_z"], f"vorticity_t{t_name}.png",
                                "RdBu_r", "ω_z [1/s]", "Vorticity ω_z")
            except Exception as e:
                print(f"  Error plotting vorticity at {t_name}: {e}")
        else:
            print(f"  No vorticity field at t={t_name} — run: postProcess -func vorticity")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Plot OpenFOAM + preCICE results")
    parser.add_argument("outdir", nargs="?", default=None,
                        help="Output directory (default: latest in ns_data)")
    parser.add_argument("--times", nargs="+", default=None,
                        help="Only plot these timestep directory names (e.g. 0.05 0.1)")
    parser.add_argument("--fields-only", action="store_true",
                        help="Skip forces/convergence plots, only do field slices")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir if args.outdir else find_latest_output()
    only_times = set(args.times) if args.times else None

    print(f"Output directory: {outdir}")

    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    if not args.fields_only:
        print("\n[1/3] Plotting forces...")
        plot_forces(outdir, figdir)

        print("\n[2/3] Plotting preCICE convergence...")
        plot_precice_convergence(outdir, figdir)

    # Run OpenFOAM postProcess vorticity if not already done
    case_dir = os.path.join(outdir, "openfoam_case")
    sample_time = None
    for d in sorted(os.listdir(case_dir)):
        try:
            t = float(d)
            if t > 0 and os.path.isdir(os.path.join(case_dir, d)):
                sample_time = d
                break
        except ValueError:
            pass
    if sample_time and not os.path.exists(os.path.join(case_dir, sample_time, "vorticity")):
        print("\n  Running postProcess -func vorticity ...")
        import subprocess
        subprocess.run(
            "source /usr/lib/openfoam/openfoam2312/etc/bashrc && postProcess -func vorticity",
            shell=True, cwd=case_dir, capture_output=True,
        )

    step = "[3/3]" if not args.fields_only else "[1/1]"
    print(f"\n{step} Plotting field slices (Umag, Ux, Uy, pressure, vorticity)...")
    plot_field_slices(outdir, figdir, only_times=only_times)

    print(f"\nAll figures saved to: {figdir}")


if __name__ == "__main__":
    main()
