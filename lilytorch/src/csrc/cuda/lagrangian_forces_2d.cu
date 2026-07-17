// =====================================================================
//  cuda/lagrangian_forces_2d.cu
//
//  CUDA implementation of ``lagrangian_forces_2d``: fused per-body
//  surface integration of (ν·ρ·ε·n - p n) over each body's CCW-closed
//  contour.  Mirrors the CPU version in lagrangian_forces_cpu_2d.cpp.
//
//  Parallelisation:
//    * gridDim.y = B (one block-row per body).
//    * One thread per contour marker; per-marker traction is computed
//      and accumulated into the body's 6-channel ``out`` row via
//      atomicAdd (double-precision).
//    * Per-marker quadrature weight is the trapezoidal lumped weight
//      ``w_k = 0.5 * (ds_{k-1} + ds_k)`` on the closed contour, which
//      reproduces the segment-wise ``0.5 * (f_i + f_{i+1}) * ds_i``
//      sum used by the CPU reference.
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <c10/util/ArrayRef.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace lilytorch_kernels {

template <typename scalar_t>
__device__ __forceinline__ scalar_t lf_bilinear_2d_d(
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
__device__ __forceinline__ scalar_t lf_biquadratic_2d_d(
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
        return lf_bilinear_2d_d<scalar_t>(
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

    const int s1 = My;
    const int base = (ix - 1) * s1 + (iy - 1);

    scalar_t out = (scalar_t)0;
    #pragma unroll
    for (int dx = 0; dx < 3; ++dx) {
        const scalar_t wx = (dx == 0) ? wxm : (dx == 1 ? wx0 : wxp);
        const int b0 = base + dx * s1;
        const scalar_t row = wym * F[b0] + wy0 * F[b0 + 1] + wyp * F[b0 + 2];
        out += wx * row;
    }
    return out;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t lf_sample_2d_d(
    const int method,
    const scalar_t* __restrict__ F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    if (method == 1) {
        return lf_biquadratic_2d_d<scalar_t>(
            F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
    }
    return lf_bilinear_2d_d<scalar_t>(
        F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
}

template <typename scalar_t>
__global__ void lagrangian_forces_2d_kernel(
    const scalar_t* __restrict__ exx,
    const scalar_t* __restrict__ exy,
    const scalar_t* __restrict__ eyy,
    const scalar_t* __restrict__ pp,
    const scalar_t* __restrict__ nrho,
    const int nrho_scalar,
    const scalar_t* __restrict__ cnt_x,   // (M_total,)
    const scalar_t* __restrict__ cnt_y,   // (M_total,)
    const int64_t* __restrict__ offs,     // (B+1,)
    const scalar_t* __restrict__ com,     // (B, 2)
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    const int interp_method,
    const scalar_t off_pres,
    const scalar_t off_visc,
    double* __restrict__ out)             // (B, 6)
{
    const int b = blockIdx.y;
    const int64_t i0 = offs[b];
    const int64_t i1 = offs[b + 1];
    const int64_t M  = i1 - i0;
    if (M <= 1) return;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    const int64_t k  = (int64_t)tid;
    const int64_t km = (k == 0)     ? (M - 1) : (k - 1);
    const int64_t kp = (k == M - 1) ? 0       : (k + 1);
    const int64_t g  = i0 + k;
    const int64_t gm = i0 + km;
    const int64_t gp = i0 + kp;

    const scalar_t qx = cnt_x[g];
    const scalar_t qy = cnt_y[g];

    // Tangent via central diff on the closed contour
    scalar_t tx = (cnt_x[gp] - cnt_x[gm]) * (scalar_t)0.5;
    scalar_t ty = (cnt_y[gp] - cnt_y[gm]) * (scalar_t)0.5;
    scalar_t L = sqrt(tx * tx + ty * ty);
    if (L < (scalar_t)1e-30) L = (scalar_t)1e-30;
    tx /= L; ty /= L;
    const scalar_t nx =  ty;
    const scalar_t ny = -tx;

    // Sample fields away from the contour along the outward normal, with
    // independent offsets per channel (see lagrangian_forces.cu for the
    // rationale).  off_pres == off_visc reproduces the legacy single knob.
    const scalar_t qxv = qx + off_visc * nx;
    const scalar_t qyv = qy + off_visc * ny;

    const scalar_t e_xx = lf_sample_2d_d<scalar_t>(
        interp_method, exx, Mx, My, bx0, by0, inv_dx, inv_dy, qxv, qyv);
    const scalar_t e_xy = lf_sample_2d_d<scalar_t>(
        interp_method, exy, Mx, My, bx0, by0, inv_dx, inv_dy, qxv, qyv);
    const scalar_t e_yy = lf_sample_2d_d<scalar_t>(
        interp_method, eyy, Mx, My, bx0, by0, inv_dx, inv_dy, qxv, qyv);
    // nu_rho rides with the viscous channel: it multiplies the strain tensor.
    const scalar_t nu_rho_m = nrho_scalar
        ? nrho[0]
        : lf_sample_2d_d<scalar_t>(
            interp_method, nrho, Mx, My, bx0, by0, inv_dx, inv_dy, qxv, qyv);

    const scalar_t tvx = nu_rho_m * (e_xx * nx + e_xy * ny);
    const scalar_t tvy = nu_rho_m * (e_xy * nx + e_yy * ny);

    const scalar_t qxp = qx + off_pres * nx;
    const scalar_t qyp = qy + off_pres * ny;

    const scalar_t p_m = lf_sample_2d_d<scalar_t>(
        interp_method, pp, Mx, My, bx0, by0, inv_dx, inv_dy, qxp, qyp);
    const scalar_t tpx = -p_m * nx;
    const scalar_t tpy = -p_m * ny;

    // Lumped trapezoidal weight on closed contour:
    //   w_k = 0.5 * (ds_{k-1} + ds_k)
    const scalar_t dxm = qx - cnt_x[gm];
    const scalar_t dym = qy - cnt_y[gm];
    const scalar_t dxp = cnt_x[gp] - qx;
    const scalar_t dyp = cnt_y[gp] - qy;
    const scalar_t dsm = sqrt(dxm * dxm + dym * dym);
    const scalar_t dsp = sqrt(dxp * dxp + dyp * dyp);
    const scalar_t wq  = (scalar_t)0.5 * (dsm + dsp);

    const scalar_t com_x = com[b * 2 + 0];
    const scalar_t com_y = com[b * 2 + 1];
    const scalar_t rx = qx - com_x;
    const scalar_t ry = qy - com_y;

    atomicAdd(&out[b * 6 + 0], (double)(tvx * wq));
    atomicAdd(&out[b * 6 + 1], (double)(tvy * wq));
    atomicAdd(&out[b * 6 + 2], (double)((rx * tvy - ry * tvx) * wq));
    atomicAdd(&out[b * 6 + 3], (double)(tpx * wq));
    atomicAdd(&out[b * 6 + 4], (double)(tpy * wq));
    atomicAdd(&out[b * 6 + 5], (double)((rx * tpy - ry * tpx) * wq));
}

void lagrangian_forces_2d_cuda(
    const at::Tensor& eps_xx, const at::Tensor& eps_xy, const at::Tensor& eps_yy,
    const at::Tensor& p, const at::Tensor& nu_rho_field,
    const at::Tensor& cnt_flat, const at::Tensor& cnt_offsets,
    const at::Tensor& com_pos,
    const double bx0, const double by0,
    const double inv_dx, const double inv_dy,
    const int64_t Mx, const int64_t My,
    const int64_t interp_method,
    const double sample_offset_pressure,
    const double sample_offset_friction,
    at::Tensor out)
{
    const int B = (int)com_pos.size(0);
    TORCH_CHECK(out.dim() == 2 && out.size(0) == B && out.size(1) == 6,
                "lagrangian_forces_2d_cuda: out must be (B, 6)");
    TORCH_CHECK(out.scalar_type() == at::kDouble,
                "lagrangian_forces_2d_cuda: out must be float64");
    TORCH_CHECK(cnt_offsets.numel() == B + 1,
                "lagrangian_forces_2d_cuda: cnt_offsets must have B+1 entries");

    out.zero_();
    if (B <= 0) return;

    auto exx_c  = eps_xx.contiguous();
    auto exy_c  = eps_xy.contiguous();
    auto eyy_c  = eps_yy.contiguous();
    auto p_c    = p.contiguous();
    auto nrho_c = nu_rho_field.contiguous();
    auto cnt_c  = cnt_flat.contiguous();
    auto offs_c = cnt_offsets.contiguous().to(at::kLong);
    auto com_c  = com_pos.contiguous();

    const bool nrho_scalar = (nrho_c.numel() == 1);

    // M_max determines blocks per body row.  Compute on the host from
    // offsets (cnt_offsets is small — B+1 entries).
    auto offs_cpu = offs_c.cpu();
    const int64_t* offs_h = offs_cpu.data_ptr<int64_t>();
    int64_t M_max = 0;
    for (int b = 0; b < B; ++b) {
        M_max = std::max<int64_t>(M_max, offs_h[b + 1] - offs_h[b]);
    }
    if (M_max <= 0) return;

    const int blockSize = (M_max <= 32) ? 32 : (M_max <= 128) ? 64 : 128;
    const int blocksPerBody = (int)((M_max + blockSize - 1) / blockSize);
    const dim3 grid((unsigned)blocksPerBody, (unsigned)B, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "lagrangian_forces_2d_cuda", [&] {
        const int64_t M_total = cnt_c.size(1);
        const scalar_t* cnt_x = cnt_c.data_ptr<scalar_t>();
        const scalar_t* cnt_y = cnt_x + M_total;

        lagrangian_forces_2d_kernel<scalar_t>
            <<<grid, blockSize, 0, stream>>>(
                exx_c.data_ptr<scalar_t>(),
                exy_c.data_ptr<scalar_t>(),
                eyy_c.data_ptr<scalar_t>(),
                p_c.data_ptr<scalar_t>(),
                nrho_c.data_ptr<scalar_t>(),
                (int)nrho_scalar,
                cnt_x, cnt_y,
                offs_c.data_ptr<int64_t>(),
                com_c.data_ptr<scalar_t>(),
                (int)Mx, (int)My,
                (scalar_t)bx0, (scalar_t)by0,
                (scalar_t)inv_dx, (scalar_t)inv_dy,
                (int)interp_method,
                (scalar_t)sample_offset_pressure,
                (scalar_t)sample_offset_friction,
                out.data_ptr<double>());
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("lagrangian_forces_2d", &lagrangian_forces_2d_cuda);
}

}  // namespace lilytorch_kernels
