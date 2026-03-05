"""
3D vorticity isosurface renders — publication-style Gazzola-like plots.

Renders dual omega_z isosurfaces (positive = red, negative = blue) with
the swimmer body, using PyVista off-screen.

Optimized with multiprocessing.

Usage:
    python plot_3d.py [/path/to/output/dir] [--times T1 T2 ...]
"""

import sys, os, glob
import numpy as np
import multiprocessing
import pyvista as pv
import argparse
import time
import subprocess

# Force off-screen rendering globally
pv.OFF_SCREEN = True

def find_latest_output():
    base = "/data/andreaferrario/ns_data"
    dirs = sorted(glob.glob(os.path.join(base, "2026-*")))
    if not dirs:
        raise FileNotFoundError(f"No output directories in {base}")
    return dirs[-1]

def render_frame(args):
    """Worker function to render a single time step."""
    t_val, t_name, iter_num, outdir, figdir, foam_path, fixed_threshold, percentile, force = args

    fname = f"curl_{iter_num:03d}.png"
    save_path = os.path.join(figdir, fname)

    # Idempotency check: skip if already rendered (unless --force)
    if os.path.exists(save_path) and not force:
        return

    try:
        # Re-import inside worker to ensure clean state
        import pyvista as pv
        pv.OFF_SCREEN = True

        # We reload the reader in each process. This incurs overhead but is safe.
        reader = pv.OpenFOAMReader(foam_path)

        # It's expensive to get time_values repeatedly but robust
        available_times = np.array(reader.time_values)
        if len(available_times) == 0:
             print(f"[{os.getpid()}] No times available in foam.foam")
             return

        idx = np.argmin(np.abs(available_times - t_val))
        reader.set_active_time_value(available_times[idx])

        mesh = reader.read()

        # Handle MultiBlock structure
        if isinstance(mesh, pv.MultiBlock):
            # Try to grab internalMesh block
            internal = mesh["internalMesh"] if "internalMesh" in mesh.keys() else mesh[0]
        else:
            internal = mesh

        if internal is None or internal.n_cells == 0:
            print(f"[{os.getpid()}] Empty mesh at t={t_name}")
            return

        # Move to point data for smooth contours
        internal = internal.cell_data_to_point_data()

        if "vorticity" in internal.point_data:
            omega_z = internal.point_data["vorticity"][:, 2]
        elif "U" in internal.point_data:
            # PyVista computes vorticity via `compute_derivative(vorticity=True)` if available
            try:
                # Triangulate first to avoid Jacobian errors on distorted cells
                internal = internal.triangulate()
                derivs = internal.compute_derivative(scalars="U", gradient=False, vorticity=True)
                omega_z = derivs["vorticity"][:, 2]
            except Exception as e:
                print(f"[{os.getpid()}] Failed to compute vorticity from U at t={t_name}: {e}")
                return
        else:
            print(f"[{os.getpid()}] No vorticity or U field at t={t_name}")
            return

        internal.point_data["omega_z"] = omega_z

        # Threshold logic
        abs_oz = np.abs(omega_z)
        max_oz = np.max(abs_oz)

        # Use a lower noise floor to detecting structure
        noise_floor = 0.01
        nonzero = abs_oz[abs_oz > noise_floor]

        if fixed_threshold is not None:
            threshold = fixed_threshold
        elif len(nonzero) == 0:
            # Fallback for very weak flow — still render the frame (swimmer + domain)
            threshold = None
            print(f"[{os.getpid()}] Vorticity negligible (max={max_oz:.2e}) at t={t_name} — rendering swimmer only")
        else:
            # Dynamic threshold
            threshold = np.percentile(nonzero, percentile)

        if threshold is not None:
            print(f"[{os.getpid()}] t={t_name}: Max |wz|={max_oz:.3f}, Threshold={threshold:.3f}")

        # Generate isosurfaces (skip if no meaningful vorticity)
        iso_pos = None
        iso_neg = None
        if threshold is not None:
            iso_pos = internal.contour([threshold], scalars="omega_z")
            iso_neg = internal.contour([-threshold], scalars="omega_z")


        # Swimmer body from 'boundary'
        swimmer_mesh = None
        if isinstance(mesh, pv.MultiBlock) and "boundary" in mesh.keys():
            boundary = mesh["boundary"]
            if hasattr(boundary, "keys") and "swimmer" in boundary.keys():
                swimmer_mesh = boundary["swimmer"]

        # View setup based on domain bounds
        bds = internal.bounds
        domain_center = ((bds[0]+bds[1])/2, (bds[2]+bds[3])/2, (bds[4]+bds[5])/2)
        Lx = bds[1] - bds[0]
        Ly = bds[3] - bds[2]
        focal = domain_center

        pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        pl.set_background("white")

        if iso_pos is not None and iso_pos.n_points > 0:
            pl.add_mesh(iso_pos, color="#CC3333", opacity=0.6,
                        smooth_shading=True, label=f"ω_z = +{threshold:.0f}")
        if iso_neg is not None and iso_neg.n_points > 0:
            pl.add_mesh(iso_neg, color="#3333CC", opacity=0.6,
                        smooth_shading=True, label=f"ω_z = -{threshold:.0f}")

        if swimmer_mesh is not None and swimmer_mesh.n_points > 0:
            pl.add_mesh(swimmer_mesh, color="#888888", opacity=0.9, smooth_shading=True)

        # Domain box
        pl.add_mesh(internal.outline(), color="gray", line_width=0.5, opacity=0.3)

        # Camera setup (top-down view)
        camera_pos = (focal[0], focal[1], focal[2] + 3.0)
        pl.camera_position = [camera_pos, focal, (1, 0, 0)]
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = max(Lx, Ly) / 2 * 1.05

        # Only add legend if we have labeled meshes (isosurfaces)
        has_labels = (iso_pos is not None and iso_pos.n_points > 0) or \
                     (iso_neg is not None and iso_neg.n_points > 0)
        if has_labels:
            pl.add_legend(bcolor="white", face=None, size=(0.2, 0.1))

        pl.screenshot(save_path)
        pl.close()
        print(f"Saved {fname} (t={t_name})")

    except Exception as e:
        print(f"Error rendering t={t_name}: {e}")

def plot_3d_vorticity(outdir, figdir, only_times=None, dt=0.2,
                      fixed_threshold=None, percentile=65, force=False):
    case_dir = os.path.join(outdir, "openfoam_case")

    # ... (rest of logic same until tasks loop)


    # 1. Discover all valid time directories (skipping 0, processor*, etc.)
    all_time_dirs = []
    if not os.path.exists(case_dir):
        print(f"Case directory {case_dir} not found.")
        return

    for entry in os.scandir(case_dir):
        if entry.is_dir() and not entry.name.startswith("processor") and entry.name != "0.orig":
            try:
                t = float(entry.name)
                # Check for vorticity file existence to confirm reconstruction
                # If vorticity is missing, we can try to compute it from U later,
                # so check for U as well.
                has_vorticity = os.path.exists(os.path.join(entry.path, "vorticity"))
                has_U = os.path.exists(os.path.join(entry.path, "U"))

                if has_vorticity or has_U:
                    all_time_dirs.append((t, entry.name))
            except ValueError:
                pass
    # Sort by time value
    all_time_dirs.sort(key=lambda x: x[0])

    # 2. Build map and filter
    # Use dt to calculate frame index for filenames (starting at 0 for T=0)
    # T=0.0 -> index 0
    # T=0.2 -> index 1 (if dt=0.2).
    # If dt is not provided or <=0, fallback to enumeration.
    if dt > 0:
        iter_map = {d: int(round(val / dt)) for val, d in all_time_dirs}
    else:
        iter_map = {d: idx for idx, (_, d) in enumerate(all_time_dirs)}

    if only_times is not None:
        # Filter to requested times only
        time_dirs = [(t, d) for t, d in all_time_dirs if d in only_times]
    else:
        time_dirs = all_time_dirs

    if not time_dirs:
        print("  No timesteps with vorticity field found for plotting.")
        return

    print(f"  Found {len(time_dirs)} timesteps to process in parallel.")

    # 3. Prepare tasks
    foam_path = os.path.join(case_dir, "foam.foam")
    if not os.path.exists(foam_path):
        open(foam_path, "w").close()

    tasks = []
    for t_val, t_name in time_dirs:
        iter_num = iter_map[t_name]
        tasks.append((t_val, t_name, iter_num, outdir, figdir, foam_path, fixed_threshold, percentile, force))

    # 4. Execute in parallel
    # Use fewer workers than cores to stay responsive on clusters (monitor runs alongside heavy SIM)
    # Reducing to 2 to minimize contention with MPI ranks.
    n_workers = 2
    if len(tasks) > 0:
        with multiprocessing.Pool(processes=n_workers) as pool:
            pool.map(render_frame, tasks)

def main():
    # Set start method to 'fork' (Linux default, usually fastest)
    # or 'spawn' (safer for complex C-extensions like VTK if crashes occur).
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass # Context likely already set

    parser = argparse.ArgumentParser(description="3D vorticity isosurface renders")
    parser.add_argument("outdir", nargs="?", default=None,
                        help="Output directory (default: latest in ns_data)")
    parser.add_argument("--times", nargs="+", default=None,
                        help="Only render these timestep directory names")
    parser.add_argument("--dt", type=float, default=0.2,
                        help="Time step interval for frame indexing (default: 0.2)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed vorticity threshold (overrides dynamic percentile)")
    parser.add_argument("--percentile", type=float, default=65,
                        help="Percentile of active vorticity to use as threshold (default: 65)")
    parser.add_argument("--reconstruct", action="store_true",
                        help="Run reconstructPar -newTimes before plotting (requires OpenFOAM installed)")
    parser.add_argument("--force", action="store_true",
                        help="Re-render frames even if they already exist")
    args = parser.parse_args()

    outdir = args.outdir if args.outdir else find_latest_output()
    only_times = set(args.times) if args.times else None
    dt = args.dt
    fixed_threshold = args.threshold
    percentile = args.percentile
    force = args.force

    # Resolve absolute path
    outdir = os.path.abspath(outdir)
    case_dir = os.path.join(outdir, "openfoam_case")

    print(f"Output directory: {outdir}")

    # Auto-detect if reconstruction is needed (if not explicitly requested)
    if not args.reconstruct:
        proc0 = os.path.join(case_dir, "processor0")
        if os.path.isdir(proc0):
            # Check valid time directories (float-like)
            def is_time(name):
                try:
                    float(name)
                    return True
                except ValueError:
                    return False

            proc_times = {e.name for e in os.scandir(proc0) if e.is_dir() and is_time(e.name)}
            case_times = {e.name for e in os.scandir(case_dir) if e.is_dir() and is_time(e.name)}

            # If processor0 has times that are not in the case dir (excluding 0)
            new_times = {t for t in proc_times if t not in case_times and t != "0"}

            if new_times:
                print(f"  Detected {len(new_times)} un-reconstructed timestep(s).")
                print("  -> Automatically enabling reconstruction.")
                args.reconstruct = True

    # Optional reconstruction step
    if args.reconstruct:
        print(f"Running reconstructPar -newTimes on {case_dir}...")
        try:
            cmd = f"source /usr/lib/openfoam/openfoam2312/etc/bashrc && reconstructPar -case {case_dir} -newTimes > /dev/null"
            subprocess.run(["bash", "-c", cmd], check=True)
            print("Reconstruction complete.")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Reconstruction failed with exit code {e.returncode}. Proceeding with existing data.")

    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    plot_3d_vorticity(outdir, figdir, only_times=only_times, dt=dt,
                      fixed_threshold=fixed_threshold, percentile=percentile, force=force)

if __name__ == "__main__":
    main()
