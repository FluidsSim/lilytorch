"""
Convert IGS (IGES) hull surface files to STL using the system `gmsh` binary.

Reads all .igs files from geometries/IGS/, meshes each surface with gmsh,
and writes binary STL files to DSYHS_STL/.

No Python dependencies beyond the standard library are required — only the
system `gmsh` binary must be on PATH.

Usage:
    python convert_igs2stl.py                # default clmax = 5.0
    python convert_igs2stl.py --clmax 2.0    # finer mesh
    python convert_igs2stl.py --clmax 10.0   # coarser mesh
    python convert_igs2stl.py --jobs 4       # 4 parallel workers
"""

import subprocess
import os
import glob
import sys
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def igs_to_stl_gmsh(input_path: str, output_path: str,
                    mesh_size: float = 5.0) -> None:
    """Convert a single IGS file to STL via gmsh CLI.

    Args:
        input_path: Path to the .igs file.
        output_path: Desired path for the .stl file.
        mesh_size: Target maximum mesh element size (gmsh -clmax).
    """
    cmd = [
        "gmsh",
        input_path,
        "-2",                       # 2D surface mesh
        "-clmax", str(mesh_size),   # max element size
        "-format", "stl",
        "-bin",                     # binary STL (smaller files)
        "-o", output_path,
        "-nopopup",                 # no GUI popup
        "-v", "0",                  # suppress verbose output
    ]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def _convert_one(args: tuple) -> tuple[str, bool, str]:
    """Worker function for parallel execution (module-level for pickling)."""
    filepath, output_dir, mesh_size = args
    basename = os.path.basename(filepath)
    out_path = os.path.join(output_dir, basename.replace(".igs", ".stl"))
    try:
        igs_to_stl_gmsh(filepath, out_path, mesh_size=mesh_size)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        return (basename, True, f"{size_mb:.1f} MB")
    except subprocess.CalledProcessError as exc:
        return (basename, False, str(exc))
    except Exception as exc:
        return (basename, False, str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-convert IGS (IGES) hull surfaces to STL via gmsh.")
    parser.add_argument("--clmax", type=float, default=5.0,
                        help="Maximum mesh element size (default: 5.0)")
    parser.add_argument("--jobs", "-j", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Override input directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir or os.path.join(script_dir, "IGS")
    output_dir = args.output_dir or os.path.join(script_dir, "STL")

    os.makedirs(output_dir, exist_ok=True)

    igs_files = sorted(glob.glob(os.path.join(input_dir, "*.igs")))

    if not igs_files:
        print("No .igs files found in", input_dir)
        sys.exit(1)

    print(f"Found {len(igs_files)} IGS files in {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Mesh size (clmax): {args.clmax}")
    print(f"Parallel workers: {args.jobs}")
    print()

    t0 = time.perf_counter()
    n_success = 0
    n_fail = 0

    if args.jobs <= 1:
        # Sequential
        for i, filepath in enumerate(igs_files, start=1):
            basename = os.path.basename(filepath)
            t_file = time.perf_counter()
            basename, ok, info = _convert_one(
                (filepath, output_dir, args.clmax))
            elapsed = time.perf_counter() - t_file
            if ok:
                n_success += 1
                print(f"  [{i:2d}/{len(igs_files)}] {basename}  "
                      f"({info}, {elapsed:.1f}s)")
            else:
                n_fail += 1
                print(f"  [{i:2d}/{len(igs_files)}] {basename}  "
                      f"FAILED: {info}", file=sys.stderr)
    else:
        # Parallel
        tasks = [(fp, output_dir, args.clmax) for fp in igs_files]
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(_convert_one, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures), start=1):
                basename, ok, info = future.result()
                if ok:
                    n_success += 1
                    print(f"  [{i:2d}/{len(igs_files)}] {basename}  ({info})")
                else:
                    n_fail += 1
                    print(f"  [{i:2d}/{len(igs_files)}] {basename}  "
                          f"FAILED: {info}", file=sys.stderr)

    total_t = time.perf_counter() - t0
    print(f"\nDone: {n_success} converted, {n_fail} failed "
          f"({total_t:.1f}s total)")


if __name__ == "__main__":
    main()
