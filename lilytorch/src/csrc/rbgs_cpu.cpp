// =====================================================================
//  rbgs_cpu.cpp — superseded by multigrid_cpu.cpp.
//
//  The real at::parallel_for CPU twins for rbgs_sweep_2d/3d and
//  mg_residual_2d/3d are now in multigrid_cpu.cpp.  This file is kept
//  as a no-op to avoid build-system churn; it registers no ops.
// =====================================================================

#include <torch/library.h>
#include <torch/all.h>

namespace lilytorch_kernels {
// All CPU registrations for these ops now live in multigrid_cpu.cpp.
}  // namespace lilytorch_kernels
