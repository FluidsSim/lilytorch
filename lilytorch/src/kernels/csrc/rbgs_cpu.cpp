// =====================================================================
//  rbgs_cpu.cpp — CPU stubs for rbgs_sweep_2d / rbgs_sweep_3d.
//
//  These ops are CUDA-only; the CPU path simply raises an error.
// =====================================================================

#include <torch/library.h>
#include <torch/all.h>

namespace lilytorch_kernels {

static void rbgs_sweep_2d_cpu(
        at::Tensor /*p*/, at::Tensor /*f*/,
        at::Tensor /*cp0*/, at::Tensor /*cm0*/,
        at::Tensor /*cp1*/, at::Tensor /*cm1*/,
        double /*jcap_tol*/, int64_t /*nsmoothing*/)
{
    TORCH_CHECK(false,
        "rbgs_sweep_2d is a CUDA-only operation. "
        "Move tensors to a CUDA device before calling.");
}

static void rbgs_sweep_3d_cpu(
        at::Tensor /*p*/, at::Tensor /*f*/,
        at::Tensor /*cp0*/, at::Tensor /*cm0*/,
        at::Tensor /*cp1*/, at::Tensor /*cm1*/,
        at::Tensor /*cp2*/, at::Tensor /*cm2*/,
        double /*jcap_tol*/, int64_t /*nsmoothing*/)
{
    TORCH_CHECK(false,
        "rbgs_sweep_3d is a CUDA-only operation. "
        "Move tensors to a CUDA device before calling.");
}

static void mg_residual_2d_cpu(
        at::Tensor /*p*/, at::Tensor /*f*/,
        at::Tensor /*cp0*/, at::Tensor /*cm0*/,
        at::Tensor /*cp1*/, at::Tensor /*cm1*/,
        double /*jcap_tol*/, at::Tensor /*r*/)
{
    TORCH_CHECK(false,
        "mg_residual_2d is a CUDA-only operation. "
        "Move tensors to a CUDA device before calling.");
}

static void mg_residual_3d_cpu(
        at::Tensor /*p*/, at::Tensor /*f*/,
        at::Tensor /*cp0*/, at::Tensor /*cm0*/,
        at::Tensor /*cp1*/, at::Tensor /*cm1*/,
        at::Tensor /*cp2*/, at::Tensor /*cm2*/,
        double /*jcap_tol*/, at::Tensor /*r*/)
{
    TORCH_CHECK(false,
        "mg_residual_3d is a CUDA-only operation. "
        "Move tensors to a CUDA device before calling.");
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("rbgs_sweep_2d", &rbgs_sweep_2d_cpu);
    m.impl("rbgs_sweep_3d", &rbgs_sweep_3d_cpu);
    m.impl("mg_residual_2d", &mg_residual_2d_cpu);
    m.impl("mg_residual_3d", &mg_residual_3d_cpu);
}

}  // namespace lilytorch_kernels
