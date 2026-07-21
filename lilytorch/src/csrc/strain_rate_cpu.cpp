// =====================================================================
//  strain_rate_cpu.cpp — CPU twin of the strain-rate magnitude kernel
//  (ground rule 4: every CUDA kernel keeps an at::parallel_for CPU twin).
//
//  Shares the whole stencil with the CUDA kernel via ``csrc/strain_rate.h``,
//  so CPU and CUDA are the same arithmetic in the same order.
// =====================================================================

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include "common/strain_rate.h"

namespace lilytorch_kernels {

void strain_rate_magnitude_cpu(
        at::Tensor u, at::Tensor v, c10::optional<at::Tensor> w,
        double h, at::Tensor out) {

    const int ndim = (int)u.dim();
    TORCH_CHECK(ndim == 2 || ndim == 3,
                "strain_rate_magnitude: u must be 2-D or 3-D, got ", ndim);
    TORCH_CHECK(v.dim() == ndim && out.dim() == ndim,
                "strain_rate_magnitude: u, v, out must have the same ndim");
    TORCH_CHECK(u.sizes() == v.sizes() && u.sizes() == out.sizes(),
                "strain_rate_magnitude: u, v, out must have the same shape");

    const bool is_3d = (ndim == 3);
    if (is_3d) {
        TORCH_CHECK(w.has_value(),
                    "strain_rate_magnitude: w is required in 3-D");
        TORCH_CHECK(w->dim() == 3 && w->sizes() == u.sizes(),
                    "strain_rate_magnitude: w must be shaped like u");
    }

    const int Nx = (int)u.size(0);
    const int Ny = (int)u.size(1);
    const int Nz = is_3d ? (int)u.size(2) : 1;

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

    const int64_t total = (int64_t)Nx * Ny * Nz;

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "strain_rate_magnitude_cpu", [&] {
        const scalar_t* up = u.data_ptr<scalar_t>();
        const scalar_t* vp = v.data_ptr<scalar_t>();
        const scalar_t* wp_ = is_3d ? w->data_ptr<scalar_t>() : nullptr;
        scalar_t* op = out.data_ptr<scalar_t>();
        const scalar_t inv_2h = static_cast<scalar_t>(0.5 / h);

        at::parallel_for(0, total, 2048, [&](int64_t begin, int64_t end) {
            for (int64_t tid = begin; tid < end; ++tid) {
                const int k = (int)(tid % Nz);
                const int64_t ij = tid / Nz;
                const int j = (int)(ij % Ny);
                const int i = (int)(ij / Ny);

                const scalar_t s = is_3d
                    ? lt_strain_rate_at<scalar_t, 3>(
                        up, vp, wp_, i, j, k, Nx, Ny, Nz,
                        su_x, su_y, su_z, sv_x, sv_y, sv_z,
                        sw_x, sw_y, sw_z, inv_2h)
                    : lt_strain_rate_at<scalar_t, 2>(
                        up, vp, wp_, i, j, k, Nx, Ny, Nz,
                        su_x, su_y, su_z, sv_x, sv_y, sv_z,
                        sw_x, sw_y, sw_z, inv_2h);

                op[i * so_x + j * so_y + k * so_z] = s;
            }
        });
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("strain_rate_magnitude", &strain_rate_magnitude_cpu);
}

}  // namespace lilytorch_kernels
