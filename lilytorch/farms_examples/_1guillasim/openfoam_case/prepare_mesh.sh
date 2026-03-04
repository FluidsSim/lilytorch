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

STL_FOLDER="${1:?Usage: prepare_mesh.sh <stl_folder>}"
CASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Preparing OpenFOAM mesh ==="
echo "  STL folder: $STL_FOLDER"
echo "  Case dir  : $CASE_DIR"

cd "$CASE_DIR"

# --- 1. Merge link STLs into a single triSurface file ---
mkdir -p constant/triSurface

echo "Merging link STLs..."
# Use python to merge STL files (handles binary/ASCII transparently)
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

stl_files = sorted(glob.glob(os.path.join(stl_folder, "link*_collision.stl")))
if not stl_files:
    print(f"ERROR: No link*_collision.stl files found in {stl_folder}")
    sys.exit(1)

meshes = []
for f in stl_files:
    m = stl_mesh.Mesh.from_file(f)
    meshes.append(m)
    print(f"  Loaded {f}: {m.vectors.shape[0]} triangles")

combined = stl_mesh.Mesh(np.concatenate([m.data for m in meshes]))
out_path = os.path.join("constant", "triSurface", "swimmer.stl")
combined.save(out_path)
print(f"  Wrote combined STL: {out_path} ({combined.vectors.shape[0]} triangles)")
PYEOF

# --- 2. blockMesh ---
echo ""
echo "=== Running blockMesh ==="
blockMesh

# --- 3. snappyHexMesh ---
echo ""
echo "=== Running snappyHexMesh ==="
snappyHexMesh -overwrite

# --- 4. Copy initial conditions ---
echo ""
echo "=== Setting up initial conditions ==="
cp -r 0.orig 0

echo ""
echo "=== Mesh preparation complete ==="
echo "  You can now run: pimpleFoam"
echo "  Or launch the coupled FARMS-OpenFOAM simulation."
