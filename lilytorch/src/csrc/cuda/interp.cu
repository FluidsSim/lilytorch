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

namespace lilytorch_kernels {

// ---- 2-D samplers -----------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ scalar_t bilinear_sample_uniform_2d(
    const scalar_t* __restrict__ F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    const scalar_t Mx_lim = (scalar_t)(Mx - 1);
    const scalar_t My_lim = (scalar_t)(My - 1);
    tx = max((scalar_t)0, min(tx, Mx_lim));
    ty = max((scalar_t)0, min(ty, My_lim));

    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;

    const scalar_t fx = tx - (scalar_t)ix;
    const scalar_t fy = ty - (scalar_t)iy;
    const scalar_t wx0 = (scalar_t)1 - fx, wx1 = fx;
    const scalar_t wy0 = (scalar_t)1 - fy, wy1 = fy;

    const int s1   = My;
    const int base = ix * s1 + iy;

    return (
        wx0 * (wy0 * F[base]      + wy1 * F[base + 1]) +
        wx1 * (wy0 * F[base + s1] + wy1 * F[base + s1 + 1])
    );
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t biquadratic_sample_uniform_2d(
    const scalar_t* __restrict__ F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    const scalar_t Mx_lim = (scalar_t)(Mx - 1);
    const scalar_t My_lim = (scalar_t)(My - 1);
    tx = max((scalar_t)0, min(tx, Mx_lim));
    ty = max((scalar_t)0, min(ty, My_lim));

    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;

    if (ix < 1 || iy < 1 || Mx < 3 || My < 3) {
        return bilinear_sample_uniform_2d<scalar_t>(
            F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
    }

    const scalar_t fx = tx - (scalar_t)ix;
    const scalar_t fy = ty - (scalar_t)iy;

    const scalar_t half = (scalar_t)0.5;
    const scalar_t wxm = half * fx * (fx - (scalar_t)1);
    const scalar_t wx0 = (scalar_t)1 - fx * fx;
    const scalar_t wxp = half * fx * (fx + (scalar_t)1);
    const scalar_t wym = half * fy * (fy - (scalar_t)1);
    const scalar_t wy0 = (scalar_t)1 - fy * fy;
    const scalar_t wyp = half * fy * (fy + (scalar_t)1);

    const int s1   = My;
    const int base = (ix - 1) * s1 + (iy - 1);

    scalar_t out = (scalar_t)0;
    #pragma unroll
    for (int dx = 0; dx < 3; ++dx) {
        const scalar_t wx = (dx == 0) ? wxm : (dx == 1 ? wx0 : wxp);
        const int b0 = base + dx * s1;
        const scalar_t col =
            wym * F[b0] + wy0 * F[b0 + 1] + wyp * F[b0 + 2];
        out += wx * col;
    }
    return out;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sdf_sample_dispatch_2d(
    const int interp_method,
    const scalar_t* __restrict__ F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    if (interp_method == 1) {
        return biquadratic_sample_uniform_2d<scalar_t>(
            F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
    }
    return bilinear_sample_uniform_2d<scalar_t>(
        F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
}


// ---- 3-D samplers -----------------------------------------------------------
// =====================================================================
//  Trilinear sample on a UNIFORM body grid.
//
//  See the matching helper in streaming_sdf_cpu.cpp for the rationale.
//  Body SDF tables are uniform-grid by construction in BDIM, so corner
//  weights reduce to (1-frac, frac) per axis -- this saves the slow
//  ``floor`` calls, the 24 axis-table loads per cell across the 4 face
//  samples, and the trailing ``* inv_vol`` multiply.
// =====================================================================
template <typename scalar_t>
__device__ __forceinline__ scalar_t trilinear_sample_uniform(
    const scalar_t* __restrict__ F,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    scalar_t tz = (zq - bz0) * inv_dz;
    const scalar_t Mx_lim = (scalar_t)(Mx - 1);
    const scalar_t My_lim = (scalar_t)(My - 1);
    const scalar_t Mz_lim = (scalar_t)(Mz - 1);
    tx = max((scalar_t)0, min(tx, Mx_lim));
    ty = max((scalar_t)0, min(ty, My_lim));
    tz = max((scalar_t)0, min(tz, Mz_lim));

    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;
    int iz = (int)tz; if (iz > Mz - 2) iz = Mz - 2;

    const scalar_t fx = tx - (scalar_t)ix;
    const scalar_t fy = ty - (scalar_t)iy;
    const scalar_t fz = tz - (scalar_t)iz;
    const scalar_t wx0 = (scalar_t)1 - fx, wx1 = fx;
    const scalar_t wy0 = (scalar_t)1 - fy, wy1 = fy;
    const scalar_t wz0 = (scalar_t)1 - fz, wz1 = fz;

    const int s2   = Mz;
    const int s1   = My * Mz;
    const int base = ix*s1 + iy*s2 + iz;

    return (
        wx0 * (
          wy0 * (wz0 * F[base]                + wz1 * F[base + 1]) +
          wy1 * (wz0 * F[base + s2]           + wz1 * F[base + s2 + 1])
        ) +
        wx1 * (
          wy0 * (wz0 * F[base + s1]           + wz1 * F[base + s1 + 1]) +
          wy1 * (wz0 * F[base + s1 + s2]      + wz1 * F[base + s1 + s2 + 1])
        )
    );
}

// =====================================================================
//  Triquadratic sample on a UNIFORM body grid.
//
//  Lagrange interpolation on a 3x3x3 stencil [ix-1, ix, ix+1]^3 with the
//  same lower-bracketing convention as ``trilinear_sample_uniform``.
//  Mirrors the CPU implementation line-for-line; see the matching helper
//  in streaming_sdf_cpu.cpp for the algorithmic rationale.
// =====================================================================
template <typename scalar_t>
__device__ __forceinline__ scalar_t triquadratic_sample_uniform(
    const scalar_t* __restrict__ F,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    scalar_t tz = (zq - bz0) * inv_dz;
    const scalar_t Mx_lim = (scalar_t)(Mx - 1);
    const scalar_t My_lim = (scalar_t)(My - 1);
    const scalar_t Mz_lim = (scalar_t)(Mz - 1);
    tx = max((scalar_t)0, min(tx, Mx_lim));
    ty = max((scalar_t)0, min(ty, My_lim));
    tz = max((scalar_t)0, min(tz, Mz_lim));

    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;
    int iz = (int)tz; if (iz > Mz - 2) iz = Mz - 2;

    if (ix < 1 || iy < 1 || iz < 1 ||
        Mx < 3 || My < 3 || Mz < 3) {
        return trilinear_sample_uniform<scalar_t>(
            F, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, xq, yq, zq);
    }

    const scalar_t fx = tx - (scalar_t)ix;
    const scalar_t fy = ty - (scalar_t)iy;
    const scalar_t fz = tz - (scalar_t)iz;

    const scalar_t half = (scalar_t)0.5;
    const scalar_t wxm = half * fx * (fx - (scalar_t)1);
    const scalar_t wx0 = (scalar_t)1 - fx * fx;
    const scalar_t wxp = half * fx * (fx + (scalar_t)1);
    const scalar_t wym = half * fy * (fy - (scalar_t)1);
    const scalar_t wy0 = (scalar_t)1 - fy * fy;
    const scalar_t wyp = half * fy * (fy + (scalar_t)1);
    const scalar_t wzm = half * fz * (fz - (scalar_t)1);
    const scalar_t wz0 = (scalar_t)1 - fz * fz;
    const scalar_t wzp = half * fz * (fz + (scalar_t)1);

    const int s2 = Mz;
    const int s1 = My * Mz;
    const int base = (ix - 1) * s1 + (iy - 1) * s2 + (iz - 1);

    scalar_t out = (scalar_t)0;
    #pragma unroll
    for (int dx = 0; dx < 3; ++dx) {
        const scalar_t wx = (dx == 0) ? wxm : (dx == 1 ? wx0 : wxp);
        const int b0 = base + dx * s1;
        scalar_t plane = (scalar_t)0;
        #pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
            const scalar_t wy = (dy == 0) ? wym : (dy == 1 ? wy0 : wyp);
            const int b1 = b0 + dy * s2;
            const scalar_t row =
                wzm * F[b1]     + wz0 * F[b1 + 1] + wzp * F[b1 + 2];
            plane += wy * row;
        }
        out += wx * plane;
    }
    return out;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sdf_sample_dispatch(
    const int interp_method,
    const scalar_t* __restrict__ F,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    if (interp_method == 1) {
        return triquadratic_sample_uniform<scalar_t>(
            F, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, xq, yq, zq);
    }
    return trilinear_sample_uniform<scalar_t>(
        F, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, xq, yq, zq);
}


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
