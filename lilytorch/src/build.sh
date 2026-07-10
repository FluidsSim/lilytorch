#!/usr/bin/env bash
# In-place rebuild of the lilytorch native kernels extension.
#
# Usage:   bash lilytorch/src/build.sh
# Forces a clean recompile of the CUDA .cu and the cpp glue.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON:-python}"

# Force recompile of all source files.
touch lilytorch/src/csrc/cuda/streaming_sdf.cu
touch lilytorch/src/csrc/cuda/streaming_sdf_2d.cu
touch lilytorch/src/csrc/cuda/advection_flux.cu
touch lilytorch/src/csrc/cuda/cvof_sweep.cu
touch lilytorch/src/csrc/cuda/multigrid_smoothers.cu
touch lilytorch/src/csrc/cuda/multigrid_transfer.cu
touch lilytorch/src/csrc/cuda/poisson_solve.cu
touch lilytorch/src/csrc/ops.cpp
touch lilytorch/src/csrc/streaming_sdf_cpu.cpp
touch lilytorch/src/csrc/streaming_sdf_cpu_2d.cpp
touch lilytorch/src/csrc/rbgs_cpu.cpp
touch lilytorch/src/csrc/multigrid_cpu.cpp
# Clean all old object files for a truly fresh recompile.
rm -f build/temp.linux-*/lilytorch/src/csrc/cuda/*.o 2>/dev/null || true
rm -f build/temp.linux-*/lilytorch/src/csrc/*.o 2>/dev/null || true
# Also clean old kernels/csrc paths for a fresh start
rm -f build/temp.linux-*/lilytorch/src/kernels/csrc/**/*.o 2>/dev/null || true
rm -f build/temp.linux-*/lilytorch/src/kernels/csrc/*.o 2>/dev/null || true

"$PYTHON_BIN" setup.py build_ext --inplace
