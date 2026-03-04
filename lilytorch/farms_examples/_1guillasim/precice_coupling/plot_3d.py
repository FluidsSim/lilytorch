"""
3D vorticity isosurface renders — publication-style Gazzola-like plots.

Renders dual omega_z isosurfaces (positive = red, negative = blue) with
the swimmer body, using PyVista off-screen.

Usage:
    python plot_3d.py [/path/to/output/dir]
"""

import sys, os, glob
import numpy as np


def find_latest_output():
    base = "/data/andreaferrario/ns_data"
    dirs = sorted(glob.glob(os.path.join(base, "2026-*")))
    if not dirs:
        raise FileNotFoundError(f"No output directories in {base}")
    return dirs[-1]


def plot_3d_vorticity(outdir, figdir, only_times=None):
    """Render 3D vorticity isosurfaces.

    Args:
        only_times: optional set of timestep directory names (str) to render.
                    If None, renders all available timesteps.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    case_dir = os.path.join(outdir, "openfoam_case")

    # Find ALL reconstructed timesteps with vorticity (for global iteration numbering)
    all_time_dirs = []
    for d in os.listdir(case_dir):
        full = os.path.join(case_dir, d)
        if os.path.isdir(full) and not d.startswith("processor") and d not in ("0", "0.orig"):
            try:
                t = float(d)
                if os.path.exists(os.path.join(full, "vorticity")):
                    all_time_dirs.append((t, d))
            except ValueError:
                pass
    all_time_dirs.sort()

    # Build global index: dir_name -> 1-based iteration number
    iter_map = {d: idx + 1 for idx, (_, d) in enumerate(all_time_dirs)}

    # Filter to requested times
    if only_times is not None:
        time_dirs = [(t, d) for t, d in all_time_dirs if d in only_times]
    else:
        time_dirs = all_time_dirs

    if not time_dirs:
        print("  No timesteps with vorticity field found.")
        print("  Run: postProcess -func vorticity")
        return

    print(f"  Found {len(time_dirs)} timesteps with vorticity: {[t for t, _ in time_dirs]}")

    # Create .foam file
    foam_path = os.path.join(case_dir, "foam.foam")
    if not os.path.exists(foam_path):
        open(foam_path, "w").close()

    reader = pv.OpenFOAMReader(foam_path)
    available_times = reader.time_values

    for t_val, t_name in time_dirs:
        iter_num = iter_map[t_name]
        print(f"  Rendering t = {t_name} (iter {iter_num}) ...")
        idx = np.argmin(np.abs(np.array(available_times) - t_val))
        reader.set_active_time_value(available_times[idx])

        mesh = reader.read()
        if isinstance(mesh, pv.MultiBlock):
            internal = mesh["internalMesh"] if "internalMesh" in mesh.keys() else mesh[0]
        else:
            internal = mesh

        if internal is None or internal.n_cells == 0:
            print(f"    Empty mesh at t={t_name}")
            continue

        # Move to point data for smooth isosurfaces
        internal = internal.cell_data_to_point_data()

        if "vorticity" not in internal.point_data:
            print(f"    No vorticity field at t={t_name}")
            continue

        # Extract omega_z (z-component of vorticity)
        omega_z = internal.point_data["vorticity"][:, 2]
        internal.point_data["omega_z"] = omega_z

        # Determine isosurface threshold — use a percentile of |omega_z|
        abs_oz = np.abs(omega_z)
        nonzero = abs_oz[abs_oz > 0.1]
        if len(nonzero) == 0:
            print(f"    Vorticity too small at t={t_name}")
            continue

        # Use ~80th percentile of non-trivial vorticity as threshold
        threshold = np.percentile(nonzero, 80)
        print(f"    omega_z range: [{omega_z.min():.1f}, {omega_z.max():.1f}], threshold = ±{threshold:.1f}")

        # Generate isosurfaces
        try:
            iso_pos = internal.contour([threshold], scalars="omega_z")
            iso_neg = internal.contour([-threshold], scalars="omega_z")
        except Exception as e:
            print(f"    Isosurface error: {e}")
            continue

        # Get swimmer surface from boundary patch (actual deformed mesh, not static STL)
        swimmer_mesh = None
        if isinstance(mesh, pv.MultiBlock) and "boundary" in mesh.keys():
            boundary = mesh["boundary"]
            if hasattr(boundary, "keys") and "swimmer" in boundary.keys():
                swimmer_mesh = boundary["swimmer"]

        # Use domain center as focal point so the full tank is visible
        bds = internal.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        domain_center = ((bds[0]+bds[1])/2, (bds[2]+bds[3])/2, (bds[4]+bds[5])/2)
        Lx = bds[1] - bds[0]  # ~2.4
        Ly = bds[3] - bds[2]  # ~0.6
        focal = domain_center

        print(f"    Domain center: ({focal[0]:.3f}, {focal[1]:.3f}, {focal[2]:.3f}), "
              f"extent: {Lx:.2f} x {Ly:.2f}")

        # --- Render top-down view only ---
        camera_pos = (focal[0], focal[1], focal[2] + 3.0)

        pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        pl.set_background("white")

        # Positive vorticity — red
        if iso_pos.n_points > 0:
            pl.add_mesh(iso_pos, color="#CC3333", opacity=0.6,
                        smooth_shading=True, label=f"ω_z = +{threshold:.0f}")

        # Negative vorticity — blue
        if iso_neg.n_points > 0:
            pl.add_mesh(iso_neg, color="#3333CC", opacity=0.6,
                        smooth_shading=True, label=f"ω_z = -{threshold:.0f}")

        # Add swimmer body from boundary patch (deformed mesh at current time)
        if swimmer_mesh is not None and swimmer_mesh.n_points > 0:
            pl.add_mesh(swimmer_mesh, color="#888888", opacity=0.9, smooth_shading=True)

        # Domain bounding box (thin outline)
        pl.add_mesh(internal.outline(), color="gray", line_width=0.5, opacity=0.3)

        pl.camera_position = [camera_pos, focal, (1, 0, 0)]
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = max(Lx, Ly) / 2 * 1.05

        pl.add_legend(bcolor="white", face=None, size=(0.2, 0.1))

        fname = f"curl_{iter_num:03d}.png"
        pl.screenshot(os.path.join(figdir, fname))
        pl.close()
        print(f"    Saved {fname}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="3D vorticity isosurface renders")
    parser.add_argument("outdir", nargs="?", default=None,
                        help="Output directory (default: latest in ns_data)")
    parser.add_argument("--times", nargs="+", default=None,
                        help="Only render these timestep directory names (e.g. 0.05 0.1)")
    args = parser.parse_args()

    outdir = args.outdir if args.outdir else find_latest_output()
    only_times = set(args.times) if args.times else None

    print(f"Output directory: {outdir}")
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    print("\nRendering 3D vorticity isosurfaces...")
    plot_3d_vorticity(outdir, figdir, only_times=only_times)
    print(f"\nFigures saved to: {figdir}")

if __name__ == "__main__":
    main()
