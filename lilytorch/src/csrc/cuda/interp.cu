// =====================================================================
//  interp.cu — Scattered-point interpolation kernels (2-D and 3-D).
//
//  Contains interp_{2,3}d: bilinear/trilinear (method=0) and
//  biquadratic/triquadratic (method=1) interpolation at arbitrary
//  query points on a uniform grid.
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <c10/util/ArrayRef.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/block/block_reduce.cuh>

#include "../common/bc_ops.h"
#include "../common/interp.h"

namespace lilytorch_kernels {

// ---- 2-D interp kernel + launcher -------------------------------------------
// =====================================================================
//  interp_2d: scattered-point bilinear / biquadratic sampling
//
//  One thread per query point.  Calls the same sdf_sample_dispatch_2d
//  device function used by the streaming/forces kernels (interp_method 0 =
//  bilinear / "linear", 1 = biquadratic / "quadratic").
// =====================================================================
template <typename scalar_t>
__global__ void interp_2d_kernel(
    const scalar_t* __restrict__ F,
    const scalar_t* __restrict__ xq,
    const scalar_t* __restrict__ yq,
    const int N,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    const int interp_method,
    scalar_t* __restrict__ G)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    G[tid] = sdf_sample_dispatch_2d(
        interp_method,
        F, Mx, My,
        bx0, by0,
        inv_dx, inv_dy,
        xq[tid], yq[tid]);
}

void interp_2d_cuda(
    const at::Tensor& F,
    const at::Tensor& xq, const at::Tensor& yq,
    const double bx0, const double by0,
    const double inv_dx, const double inv_dy,
    const int64_t Mx, const int64_t My,
    const int64_t interp_method,
    at::Tensor& G)
{
    const int N = (int)xq.numel();
    if (N == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blockSize = (N <= 128) ? 32 : (N <= 4096) ? 128 : 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // Bind temporaries to named locals: ``.contiguous().to(...)`` may return
    // a fresh tensor whose storage is freed at the end of the full-expression
    // unless held.  The CUDA launch is asynchronous, so a dangling pointer
    // could be read by the kernel after the storage has been recycled.
    auto F_c  = F.contiguous();
    auto xq_c = xq.contiguous().to(F.scalar_type());
    auto yq_c = yq.contiguous().to(F.scalar_type());

    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "interp_2d_cuda", [&] {
        interp_2d_kernel<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
            F_c.data_ptr<scalar_t>(),
            xq_c.data_ptr<scalar_t>(),
            yq_c.data_ptr<scalar_t>(),
            N,
            (int)Mx, (int)My,
            (scalar_t)bx0, (scalar_t)by0,
            (scalar_t)inv_dx, (scalar_t)inv_dy,
            (int)interp_method,
            G.data_ptr<scalar_t>());
    });
}


// ---- 3-D interp kernel + launcher -------------------------------------------
// =====================================================================
//  interp_3d: scattered-point trilinear / triquadratic sampling
//
//  One thread per query point.  Calls the same sdf_sample_dispatch
//  device function used by the streaming/forces kernels (interp_method 0 =
//  trilinear / "linear", 1 = triquadratic / "quadratic").
// =====================================================================
template <typename scalar_t>
__global__ void interp_3d_kernel(
    const scalar_t* __restrict__ F,
    const scalar_t* __restrict__ xq,
    const scalar_t* __restrict__ yq,
    const scalar_t* __restrict__ zq,
    const int N,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    const int interp_method,
    scalar_t* __restrict__ G)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    G[tid] = sdf_sample_dispatch(
        interp_method,
        F, Mx, My, Mz,
        bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz,
        xq[tid], yq[tid], zq[tid]);
}

void interp_3d_cuda(
    const at::Tensor& F,
    const at::Tensor& xq, const at::Tensor& yq, const at::Tensor& zq,
    const double bx0, const double by0, const double bz0,
    const double inv_dx, const double inv_dy, const double inv_dz,
    const int64_t Mx, const int64_t My, const int64_t Mz,
    const int64_t interp_method,
    at::Tensor& G)
{
    const int N = (int)xq.numel();
    if (N == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blockSize = (N <= 128) ? 32 : (N <= 4096) ? 128 : 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // Bind temporaries to named locals: ``.contiguous().to(...)`` may return
    // a fresh tensor whose storage is freed at the end of the full-expression
    // unless held.  The CUDA launch is asynchronous, so a dangling pointer
    // could be read by the kernel after the storage has been recycled.
    auto F_c  = F.contiguous();
    auto xq_c = xq.contiguous().to(F.scalar_type());
    auto yq_c = yq.contiguous().to(F.scalar_type());
    auto zq_c = zq.contiguous().to(F.scalar_type());

    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "interp_3d_cuda", [&] {
        interp_3d_kernel<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
            F_c.data_ptr<scalar_t>(),
            xq_c.data_ptr<scalar_t>(),
            yq_c.data_ptr<scalar_t>(),
            zq_c.data_ptr<scalar_t>(),
            N,
            (int)Mx, (int)My, (int)Mz,
            (scalar_t)bx0, (scalar_t)by0, (scalar_t)bz0,
            (scalar_t)inv_dx, (scalar_t)inv_dy, (scalar_t)inv_dz,
            (int)interp_method,
            G.data_ptr<scalar_t>());
    });
}


TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("interp_3d", &interp_3d_cuda);
    m.impl("interp_2d", &interp_2d_cuda);
}

}  // namespace lilytorch_kernels
