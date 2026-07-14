// =====================================================================
//  strain_rate.cu — CUDA kernel for the strain-rate magnitude |S̄|.
//
//  Op: ``lilytorch_kernels::strain_rate_magnitude`` (see ops.cpp)
//      Tensor u, Tensor v, Tensor? w, float h, Tensor(a!) out -> ()
//
//  The math lives in ``csrc/strain_rate.h`` and is shared verbatim with the
//  CPU twin (``csrc/strain_rate_cpu.cpp``); it backs
//  ``operations.strain_rate_magnitude``.
// =====================================================================

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../strain_rate.h"

namespace lilytorch_kernels {

static inline int cdiv_sr(int a, int b) { return (a + b - 1) / b; }

template <typename scalar_t, int DIM>
__global__ void strain_rate_magnitude_kernel(
        const scalar_t* __restrict__ u,
        const scalar_t* __restrict__ v,
        const scalar_t* __restrict__ w,
        scalar_t* __restrict__ out,
        int Nx, int Ny, int Nz,
        int64_t su_x, int64_t su_y, int64_t su_z,
        int64_t sv_x, int64_t sv_y, int64_t sv_z,
        int64_t sw_x, int64_t sw_y, int64_t sw_z,
        int64_t so_x, int64_t so_y, int64_t so_z,
        scalar_t inv_2h) {

    const int64_t tid   = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)Nx * Ny * Nz;
    if (tid >= total) return;

    // k fastest-varying (row-major, matches the tensor layout).
    const int k  = (int)(tid % Nz);
    const int64_t ij = tid / Nz;
    const int j  = (int)(ij % Ny);
    const int i  = (int)(ij / Ny);

    const scalar_t s = lt_strain_rate_at<scalar_t, DIM>(
        u, v, w, i, j, k, Nx, Ny, Nz,
        su_x, su_y, su_z, sv_x, sv_y, sv_z, sw_x, sw_y, sw_z, inv_2h);

    out[i * so_x + j * so_y + k * so_z] = s;
}

void strain_rate_magnitude_cuda(
        at::Tensor u, at::Tensor v, c10::optional<at::Tensor> w,
        double h, at::Tensor out) {

    const int ndim = (int)u.dim();
    TORCH_CHECK(ndim == 2 || ndim == 3,
                "strain_rate_magnitude: u must be 2-D or 3-D, got ", ndim);
    TORCH_CHECK(v.dim() == ndim && out.dim() == ndim,
                "strain_rate_magnitude: u, v, out must have the same ndim");
    TORCH_CHECK(u.sizes() == v.sizes() && u.sizes() == out.sizes(),
                "strain_rate_magnitude: u, v, out must have the same shape");
    TORCH_CHECK(u.is_cuda() && v.is_cuda() && out.is_cuda(),
                "strain_rate_magnitude: all tensors must be CUDA");

    const bool is_3d = (ndim == 3);
    if (is_3d) {
        TORCH_CHECK(w.has_value(),
                    "strain_rate_magnitude: w is required in 3-D");
        TORCH_CHECK(w->dim() == 3 && w->sizes() == u.sizes() && w->is_cuda(),
                    "strain_rate_magnitude: w must be a CUDA tensor shaped like u");
    }

    const int Nx = (int)u.size(0);
    const int Ny = (int)u.size(1);
    const int Nz = is_3d ? (int)u.size(2) : 1;

    // The edge_order=2 one-sided stencil reaches 2 cells inward.
    TORCH_CHECK(Nx >= 3 && Ny >= 3 && (!is_3d || Nz >= 3),
                "strain_rate_magnitude: every dim must have extent >= 3");

    const int64_t su_x = u.stride(0), su_y = u.stride(1);
    const int64_t su_z = is_3d ? u.stride(2) : 0;
    const int64_t sv_x = v.stride(0), sv_y = v.stride(1);
    const int64_t sv_z = is_3d ? v.stride(2) : 0;
    const int64_t so_x = out.stride(0), so_y = out.stride(1);
    const int64_t so_z = is_3d ? out.stride(2) : 0;
    const int64_t sw_x = is_3d ? w->stride(0) : 0;
    const int64_t sw_y = is_3d ? w->stride(1) : 0;
    const int64_t sw_z = is_3d ? w->stride(2) : 0;

    const c10::cuda::CUDAGuard guard(u.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t total = (int64_t)Nx * Ny * Nz;
    const int blk = 256;
    const int grd = cdiv_sr((int)total, blk);

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "strain_rate_magnitude", [&] {
        const scalar_t* wp_ = is_3d ? w->data_ptr<scalar_t>() : nullptr;
        auto launch = [&](auto dim_tag) {
            constexpr int DIM = decltype(dim_tag)::value;
            strain_rate_magnitude_kernel<scalar_t, DIM><<<grd, blk, 0, stream>>>(
                u.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), wp_,
                out.data_ptr<scalar_t>(),
                Nx, Ny, Nz,
                su_x, su_y, su_z, sv_x, sv_y, sv_z,
                sw_x, sw_y, sw_z, so_x, so_y, so_z,
                static_cast<scalar_t>(0.5 / h));
        };
        if (is_3d) launch(std::integral_constant<int, 3>{});
        else       launch(std::integral_constant<int, 2>{});
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("strain_rate_magnitude", &strain_rate_magnitude_cuda);
}

}  // namespace lilytorch_kernels
