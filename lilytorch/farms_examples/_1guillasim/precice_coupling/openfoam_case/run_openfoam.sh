#!/bin/bash
# ============================================================================
# run_openfoam.sh
# Launch the OpenFOAM (pimpleFoam) participant for the preCICE coupling.
#
# This script should be run AFTER prepare_mesh.sh and BEFORE (or in parallel
# with) the FARMS participant.
#
# Usage:
#   cd openfoam_case
#   bash run_openfoam.sh [nprocs]
# ============================================================================

set -e

CASE_DIR="$(cd "$(dirname "$0")" && pwd)"
NPROCS="${1:-1}"

cd "$CASE_DIR"

echo "=== Starting OpenFOAM preCICE participant ==="
echo "  Case dir: $CASE_DIR"
echo "  Procs   : $NPROCS"

if [ "$NPROCS" -gt 1 ]; then
    echo "=== Decomposing mesh for $NPROCS processors ==="
    # Generate decomposeParDict if not present
    if [ ! -f system/decomposeParDict ]; then
        cat > system/decomposeParDict << EOF
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}
numberOfSubdomains  $NPROCS;
method              scotch;
EOF
    fi
    decomposePar -force
    echo ""
    echo "=== Running pimpleFoam with $NPROCS MPI ranks ==="
    mpirun -np "$NPROCS" pimpleFoam -parallel
else
    echo "=== Running pimpleFoam (serial) ==="
    pimpleFoam
fi
