"""
Fix STL files from the DSYHS conversion:
  1. Mirror half-hulls (Y < 0 only) into full hulls (Y symmetric about 0).
  2. Flip keel/rudder Z so they extend downward from the hull baseline.
  3. Optionally center/translate keel & rudder to match a given hull.

Usage:
    python fix_dsyhs_stl.py                          # fix all, in-place
    python fix_dsyhs_stl.py --no-overwrite           # write to STL_full/
    python fix_dsyhs_stl.py --assemble SYSSER43      # assemble keel+rudder+hull
"""

import struct
import os
import glob
import argparse
import sys
import copy


# ---------------------------------------------------------------------------
#  Low-level binary STL I/O
# ---------------------------------------------------------------------------

def read_stl(path: str) -> tuple[list, list]:
    """Read a binary STL file. Returns (vertices, triangles).

    vertices: list of (x, y, z) tuples
    triangles: list of (v0, v1, v2) index triples (0-based into vertices)
    """
    with open(path, "rb") as f:
        header = f.read(80)
        n_tri = struct.unpack("<I", f.read(4))[0]

        verts = []
        vert_map = {}  # (x,y,z) -> index
        tris = []

        for _ in range(n_tri):
            f.read(12)  # normal (ignored)
            tri = []
            for _ in range(3):
                x, y, z = struct.unpack("<fff", f.read(12))
                key = (round(x, 6), round(y, 6), round(z, 6))
                if key not in vert_map:
                    vert_map[key] = len(verts)
                    verts.append((x, y, z))
                tri.append(vert_map[key])
            tris.append(tuple(tri))
            f.read(2)  # attribute byte count

    return verts, tris


def write_stl(path: str, verts: list, tris: list) -> None:
    """Write a binary STL file."""
    with open(path, "wb") as f:
        f.write(b" " * 80)  # header
        f.write(struct.pack("<I", len(tris)))
        for tri in tris:
            # Compute face normal
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            u = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
            v = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
            nx = u[1] * v[2] - u[2] * v[1]
            ny = u[2] * v[0] - u[0] * v[2]
            nz = u[0] * v[1] - u[1] * v[0]
            f.write(struct.pack("<fff", nx, ny, nz))
            for idx in tri:
                f.write(struct.pack("<fff", *verts[idx]))
            f.write(struct.pack("<H", 0))


# ---------------------------------------------------------------------------
#  Transformations
# ---------------------------------------------------------------------------

def mirror_y(verts: list) -> list:
    """Mirror vertices across the Y=0 (centerplane)."""
    return [(x, -y, z) for (x, y, z) in verts]


def merge_meshes(meshes: list[tuple[list, list]]) -> tuple[list, list]:
    """Merge multiple (verts, tris) meshes into a single mesh."""
    all_verts = []
    all_tris = []
    for verts, tris in meshes:
        offset = len(all_verts)
        all_verts.extend(verts)
        all_tris.extend((a + offset, b + offset, c + offset) for a, b, c in tris)
    return all_verts, all_tris


def translate(verts: list, dx: float = 0.0, dy: float = 0.0,
              dz: float = 0.0) -> list:
    """Translate vertices."""
    return [(x + dx, y + dy, z + dz) for (x, y, z) in verts]


def flip_z(verts: list) -> list:
    """Negate Z (flip keel/rudder to extend downward)."""
    return [(x, y, -z) for (x, y, z) in verts]


# ---------------------------------------------------------------------------
#  Bounding box helper
# ---------------------------------------------------------------------------

def bbox(verts: list) -> tuple:
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    for v in verts:
        for k in range(3):
            if v[k] < mn[k]:
                mn[k] = v[k]
            if v[k] > mx[k]:
                mx[k] = v[k]
    return tuple(mn), tuple(mx)


# ---------------------------------------------------------------------------
#  Main logic
# ---------------------------------------------------------------------------

def is_half_hull(verts: list) -> bool:
    """Return True if the mesh is a half-hull (all Y <= ~0, no positive Y)."""
    mn, mx = bbox(verts)
    return mx[1] < 1.0 and mn[1] < -10.0  # Y max near 0, significant Y span


def is_full_hull(verts: list) -> bool:
    """Return True if Y spans both negative and positive significantly."""
    mn, mx = bbox(verts)
    return mn[1] < -10.0 and mx[1] > 10.0


def make_full_hull(verts: list, tris: list) -> tuple[list, list]:
    """Given a port-side half-hull, mirror it and merge into a full hull."""
    mirrored_verts = mirror_y(verts)
    # Reverse triangle winding for the mirrored side to keep normals outward
    mirrored_tris = [(a, c, b) for (a, b, c) in tris]
    return merge_meshes([(verts, tris), (mirrored_verts, mirrored_tris)])


def process_file(
    input_path: str,
    output_dir: str,
    mirror_hulls: bool = True,
    flip_keel_rudder: bool = True,
) -> str:
    """Process a single STL file. Returns output path."""
    verts, tris = read_stl(input_path)
    basename = os.path.basename(input_path)
    name, _ = os.path.splitext(basename)

    if mirror_hulls and is_half_hull(verts):
        verts, tris = make_full_hull(verts, tris)
        name += "_full"

    if flip_keel_rudder and ("Keel" in basename or "Rudder" in basename):
        verts = flip_z(verts)
        name += "_flipped"

    name += ".stl"
    out_path = os.path.join(output_dir, name)
    write_stl(out_path, verts, tris)
    return out_path


def assemble_hull(
    input_dir: str,
    output_dir: str,
    hull_name: str,
) -> str:
    """Assemble a full hull + keel + rudder into a single STL.

    hull_name: e.g. 'SYSSER43' (matches '*SYSSER43*.stl').
    """
    hull_files = sorted(glob.glob(
        os.path.join(input_dir, f"*{hull_name}*_surface.stl")))
    keel_path = os.path.join(input_dir, "StandardKeel.stl")
    rudder_path = os.path.join(input_dir, "StandardRudder.stl")

    meshes = []

    # Load and mirror hull(s)
    for hf in hull_files:
        verts, tris = read_stl(hf)
        if is_half_hull(verts):
            verts, tris = make_full_hull(verts, tris)
        meshes.append((verts, tris))

    # Add keel at its native coordinates (no Z flip).
    # The keel and hull share X=0=AP, Y=0=centerplane, but appear to use
    # different Z=0 references.  We keep the original vertex positions so
    # the user can see the native layout in Blender and adjust as needed.
    if os.path.exists(keel_path):
        kv, kt = read_stl(keel_path)
        meshes.append((kv, kt))

    # Add rudder at native coordinates (no Z flip, no translation).
    if os.path.exists(rudder_path):
        rv, rt = read_stl(rudder_path)
        meshes.append((rv, rt))

    all_verts, all_tris = merge_meshes(meshes)
    out_path = os.path.join(output_dir, f"{hull_name}_assembled.stl")
    write_stl(out_path, all_verts, all_tris)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Fix DSYHS STL files: mirror half-hulls, flip keel/rudder.")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory with raw STL files (default: ./STL)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ./STL_full)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Don't mirror half-hulls")
    parser.add_argument("--no-flip-keel", action="store_true",
                        help="Don't flip keel/rudder Z")
    parser.add_argument("--assemble", type=str, default=None,
                        help="Assemble hull+keel+rudder for a given hull name "
                             "(e.g. SYSSER43)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir or os.path.join(script_dir, "STL")
    output_dir = args.output_dir or os.path.join(script_dir, "STL_full")
    os.makedirs(output_dir, exist_ok=True)

    if args.assemble:
        out = assemble_hull(input_dir, output_dir, args.assemble)
        print(f"Assembled hull written to: {out}")
        return

    stl_files = sorted(glob.glob(os.path.join(input_dir, "*.stl")))
    if not stl_files:
        print(f"No STL files found in {input_dir}")
        sys.exit(1)

    mirror = not args.no_mirror
    flip = not args.no_flip_keel

    for f in stl_files:
        out = process_file(f, output_dir, mirror_hulls=mirror,
                           flip_keel_rudder=flip)
        print(f"  {os.path.basename(f)}  ->  {os.path.basename(out)}")

    print(f"\nDone. Output in: {output_dir}")


if __name__ == "__main__":
    main()
