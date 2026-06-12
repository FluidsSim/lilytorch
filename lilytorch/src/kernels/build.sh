#!/usr/bin/env bash
# In-place rebuild of the lilytorch native kernels extension.
#
# Usage:   bash lilytorch/src/kernels/build.sh
# Forces a clean recompile of the CUDA .cu and the cpp glue.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON:-python}"

# Force recompile of all source files.
touch lilytorch/src/kernels/csrc/cuda/streaming_sdf.cu
touch lilytorch/src/kernels/csrc/cuda/streaming_sdf_2d.cu
touch lilytorch/src/kernels/csrc/cuda/advection_flux.cu
touch lilytorch/src/kernels/csrc/ops.cpp
touch lilytorch/src/kernels/csrc/streaming_sdf_cpu.cpp
touch lilytorch/src/kernels/csrc/streaming_sdf_cpu_2d.cpp
rm -f build/temp.linux-*/lilytorch/src/kernels/csrc/cuda/streaming_sdf.o \
      build/temp.linux-*/lilytorch/src/kernels/csrc/cuda/streaming_sdf_2d.o \
      build/temp.linux-*/lilytorch/src/kernels/csrc/ops.o \
      build/temp.linux-*/lilytorch/src/kernels/csrc/streaming_sdf_cpu.o \
      build/temp.linux-*/lilytorch/src/kernels/csrc/streaming_sdf_cpu_2d.o 2>/dev/null || true

"$PYTHON_BIN" setup.py build_ext --inplace
