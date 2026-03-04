#!/bin/bash
# ============================================================================
# prepare_mesh.sh
# Prepare the OpenFOAM mesh for the swimmer FSI simulation.
#
# Steps:
#   1. Merge individual link STLs into a single swimmer.stl
#   2. Run blockMesh for background mesh
#   3. Run snappyHexMesh to refine around the swimmer body
#   4. Copy snappy result to time 0 and clean up
#
# Usage:
#   cd openfoam_case
#   bash prepare_mesh.sh /path/to/stl_folder
# ============================================================================

set -e

export STL_FOLDER="${1:?Usage: prepare_mesh.sh <stl_folder>}"
CASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Preparing OpenFOAM mesh ==="
echo "  STL folder: $STL_FOLDER"
echo "  Case dir  : $CASE_DIR"

cd "$CASE_DIR"

# --- 1. Merge link STLs into a single triSurface file ---
mkdir -p constant/triSurface

echo "Merging link STLs..."
# Use python to merge STL files, transforming each link to its MuJoCo
# world-frame initial position so the assembled body is correct.
python3 << 'PYEOF'
import sys, os, glob
try:
    from stl import mesh as stl_mesh
    import numpy as np
except ImportError:
    print("ERROR: numpy-stl not installed. Run: pip install numpy-stl")
    sys.exit(1)

stl_folder = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("STL_FOLDER", "")
if not stl_folder:
    stl_folder = os.environ.get("STL_FOLDER", "")

# -- Link world-frame positions from MuJoCo kinematic chain ----------------
# Spawn pose: [-0.65, 0, 0.2]; base link = link1
# These are computed by walking the MuJoCo body tree (MJCF pos attributes).
# The chain is: 1guilla_watertight(-0.65,0,0.2) -> link1(0,0,0) ->
#   link0(-0.177,0,0)  link2(0.060,0,0) -> link3(0.060,0,0) -> ...
spawn_pos = np.array([-0.65, 0.0, 0.2])
link_local_offsets = {  # relative to parent in the chain
    "link1": np.array([0.0,      0.0, 0.0]),   # base (child of spawn)
    "link0": np.array([-0.17701, 0.0, 0.0]),   # child of link1
    "link2": np.array([0.0601,   0.0, 0.0]),   # child of link1
    "link3": np.array([0.06,     0.0, 0.0]),   # child of link2
    "link4": np.array([0.0599,   0.0, 0.0]),   # child of link3
    "link5": np.array([0.059901, 0.0, 0.0]),   # child of link4
    "link6": np.array([0.060199, 0.0, 0.0]),   # child of link5
    "link7": np.array([0.06,     0.0, 0.0]),   # child of link6
    "link8": np.array([0.0598,   0.0, 0.0]),   # child of link7
}
# Parent map (from SDF joints)
parent_map = {
    "link1": None,
    "link0": "link1",
    "link2": "link1",
    "link3": "link2",
    "link4": "link3",
    "link5": "link4",
    "link6": "link5",
    "link7": "link6",
    "link8": "link7",
}

def world_pos(link_name):
    """Recursively compute world position by walking up the chain."""
    if parent_map[link_name] is None:
        return spawn_pos + link_local_offsets[link_name]
    return world_pos(parent_map[link_name]) + link_local_offsets[link_name]

link_world = {name: world_pos(name) for name in link_local_offsets}
print("Link world-frame positions:")
for name, pos in sorted(link_world.items()):
    print(f"  {name}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

# -- Load and transform STLs -----------------------------------------------
stl_files = sorted(glob.glob(os.path.join(stl_folder, "link*_collision.stl")))
if not stl_files:
    print(f"ERROR: No link*_collision.stl files found in {stl_folder}")
    sys.exit(1)

meshes = []
for f in stl_files:
    link_name = os.path.basename(f).replace("_collision.stl", "")
    if link_name not in link_world:
        print(f"  WARNING: {link_name} not in kinematic chain, skipping")
        continue
    m = stl_mesh.Mesh.from_file(f)
    # Translate from local frame to world frame (rotation is identity at t=0)
    offset = link_world[link_name]
    m.vectors += offset  # shape (N, 3, 3) — broadcasts over vertices
    meshes.append(m)
    print(f"  Loaded {f}: {m.vectors.shape[0]} tris, translated by {offset}")

combined = stl_mesh.Mesh(np.concatenate([m.data for m in meshes]))
out_path = os.path.join("constant", "triSurface", "swimmer.stl")
combined.save(out_path)
verts = np.unique(combined.vectors.reshape(-1, 3), axis=0)
print(f"  Wrote combined STL: {out_path} ({combined.vectors.shape[0]} tris)")
print(f"  Bounding box: x=[{verts[:,0].min():.4f}, {verts[:,0].max():.4f}]"
      f"  y=[{verts[:,1].min():.4f}, {verts[:,1].max():.4f}]"
      f"  z=[{verts[:,2].min():.4f}, {verts[:,2].max():.4f}]")
PYEOF

# --- 2. blockMesh ---
echo ""
echo "=== Running blockMesh ==="
blockMesh

# --- 3. snappyHexMesh (parallel for speed) ---
echo ""
echo "=== Decomposing for parallel snappyHexMesh ==="
decomposePar -force

echo "=== Running snappyHexMesh on 24 cores ==="
mpirun -np 24 --oversubscribe snappyHexMesh -overwrite -parallel

echo "=== Reconstructing mesh ==="
reconstructParMesh -constant

# Clean up processor dirs
rm -rf processor*

# --- 4. Copy initial conditions ---
echo ""
echo "=== Setting up initial conditions ==="
# snappyHexMesh -overwrite puts cellLevel/pointLevel into 0/
# We need to merge 0.orig/* into 0/, not copy 0.orig as a subfolder
cp -f 0.orig/* 0/ 2>/dev/null || true
cp -rf 0.orig/* 0/

echo ""
echo "=== Mesh preparation complete ==="
echo "  You can now run: pimpleFoam"
echo "  Or launch the coupled FARMS-OpenFOAM simulation."
