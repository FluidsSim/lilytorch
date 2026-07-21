// =====================================================================
//  cuda/lagrangian_forces.cu
//
//  CUDA implementation of ``lagrangian_forces_3d``: fused per-body
//  surface integration over a precomputed triangulation.
//  Mirrors the CPU version in lagrangian_forces_cpu.cpp.
//
//  Parallelisation:
//    * gridDim.y = B, gridDim.x covers the largest body's triangle list.
//    * One thread per triangle; per-triangle contribution is atomicAdd'd
//      into the body's 12-channel output row.
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <c10/util/ArrayRef.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "../common/interp.h"

namespace lilytorch_kernels {

template <typename scalar_t>
__device__ __forceinline__ scalar_t lf_trilinear_3d_d(
    const scalar_t* __restrict__ F,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    return trilinear_sample_uniform<scalar_t>(
        F, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, xq, yq, zq);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t lf_triquadratic_3d_d(
    const scalar_t* __restrict__ F,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    return triquadratic_sample_uniform<scalar_t>(
        F, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, xq, yq, zq);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t lf_sample_3d_d(
    const int method,
    const scalar_t* __restrict__ F,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    return sdf_sample_dispatch<scalar_t>(
        method, F, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, xq, yq, zq);
}

template <typename scalar_t>
__global__ void lagrangian_forces_3d_kernel(
    const scalar_t* __restrict__ exx,
    const scalar_t* __restrict__ eyy,
    const scalar_t* __restrict__ ezz,
    const scalar_t* __restrict__ exy,
    const scalar_t* __restrict__ exz,
    const scalar_t* __restrict__ eyz,
    const scalar_t* __restrict__ pp,
    const scalar_t* __restrict__ nrho,
    const int nrho_scalar,
    const scalar_t* __restrict__ cx,    // (T_total,)
    const scalar_t* __restrict__ cy,
    const scalar_t* __restrict__ cz,
    const scalar_t* __restrict__ nx_p,
    const scalar_t* __restrict__ ny_p,
    const scalar_t* __restrict__ nz_p,
    const scalar_t* __restrict__ area,  // (T_total,)
    const int64_t* __restrict__ offs,   // (B+1,)
    const scalar_t* __restrict__ com,   // (B, 3)
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    const int interp_method,
    const scalar_t off_pres,
    const scalar_t off_visc,
    double* __restrict__ out)           // (B, 12)
{
    const int b = blockIdx.y;
    const int64_t t0 = offs[b];
    const int64_t t1 = offs[b + 1];
    const int64_t T  = t1 - t0;
    if (T <= 0) return;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T) return;

    const int64_t t = t0 + (int64_t)tid;

    const scalar_t qx = cx[t];
    const scalar_t qy = cy[t];
    const scalar_t qz = cz[t];
    const scalar_t nxv = nx_p[t];
    const scalar_t nyv = ny_p[t];
    const scalar_t nzv = nz_p[t];
    const scalar_t A   = area[t];

    // Sample fields a distance OUTSIDE the body along the outward normal —
    // moves the query out of the BDIM band (where the blended velocity gives
    // ε(u_blend) ≈ μ0·ε(u_fluid) and ``zero_pressure_inside`` zeros interior
    // p), into the pure fluid where μ0=1 and the wall-side limits are
    // recovered.  See ``forces.py:forces_lagrangian_3d`` for the choice of σ.
    //
    // The two channels carry INDEPENDENT offsets: the strain tensor and p are
    // contaminated differently, and the eulerian readout has always sampled
    // them at different iso-surfaces (p at φ=0, σ at φ=eps).  Keeping one
    // knob here made every cross-method comparison confound sampling location
    // with readout.  off_pres == off_visc reproduces the legacy single knob.
    const scalar_t qxv = qx + off_visc * nxv;
    const scalar_t qyv = qy + off_visc * nyv;
    const scalar_t qzv = qz + off_visc * nzv;

    const scalar_t e_xx = lf_sample_3d_d<scalar_t>(
        interp_method, exx, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qxv, qyv, qzv);
    const scalar_t e_yy = lf_sample_3d_d<scalar_t>(
        interp_method, eyy, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qxv, qyv, qzv);
    const scalar_t e_zz = lf_sample_3d_d<scalar_t>(
        interp_method, ezz, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qxv, qyv, qzv);
    const scalar_t e_xy = lf_sample_3d_d<scalar_t>(
        interp_method, exy, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qxv, qyv, qzv);
    const scalar_t e_xz = lf_sample_3d_d<scalar_t>(
        interp_method, exz, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qxv, qyv, qzv);
    const scalar_t e_yz = lf_sample_3d_d<scalar_t>(
        interp_method, eyz, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qxv, qyv, qzv);

    // nu_rho rides with the viscous channel: it multiplies the strain tensor.
    const scalar_t nu_rho_m = nrho_scalar
        ? nrho[0]
        : lf_sample_3d_d<scalar_t>(
            interp_method, nrho, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, qxv, qyv, qzv);

    const scalar_t tvx = nu_rho_m * (e_xx * nxv + e_xy * nyv + e_xz * nzv);
    const scalar_t tvy = nu_rho_m * (e_xy * nxv + e_yy * nyv + e_yz * nzv);
    const scalar_t tvz = nu_rho_m * (e_xz * nxv + e_yz * nyv + e_zz * nzv);

    const scalar_t qxp = qx + off_pres * nxv;
    const scalar_t qyp = qy + off_pres * nyv;
    const scalar_t qzp = qz + off_pres * nzv;

    const scalar_t p_m = lf_sample_3d_d<scalar_t>(
        interp_method, pp, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qxp, qyp, qzp);
    const scalar_t tpx = -p_m * nxv;
    const scalar_t tpy = -p_m * nyv;
    const scalar_t tpz = -p_m * nzv;

    const scalar_t com_x = com[b * 3 + 0];
    const scalar_t com_y = com[b * 3 + 1];
    const scalar_t com_z = com[b * 3 + 2];
    const scalar_t rx = qx - com_x;
    const scalar_t ry = qy - com_y;
    const scalar_t rz = qz - com_z;

    atomicAdd(&out[b * 12 + 0],  (double)(tvx * A));
    atomicAdd(&out[b * 12 + 1],  (double)(tvy * A));
    atomicAdd(&out[b * 12 + 2],  (double)(tvz * A));
    atomicAdd(&out[b * 12 + 3],  (double)((ry * tvz - rz * tvy) * A));
    atomicAdd(&out[b * 12 + 4],  (double)((rz * tvx - rx * tvz) * A));
    atomicAdd(&out[b * 12 + 5],  (double)((rx * tvy - ry * tvx) * A));
    atomicAdd(&out[b * 12 + 6],  (double)(tpx * A));
    atomicAdd(&out[b * 12 + 7],  (double)(tpy * A));
    atomicAdd(&out[b * 12 + 8],  (double)(tpz * A));
    atomicAdd(&out[b * 12 + 9],  (double)((ry * tpz - rz * tpy) * A));
    atomicAdd(&out[b * 12 + 10], (double)((rz * tpx - rx * tpz) * A));
    atomicAdd(&out[b * 12 + 11], (double)((rx * tpy - ry * tpx) * A));
}

void lagrangian_forces_3d_cuda(
    const at::Tensor& eps_xx, const at::Tensor& eps_yy, const at::Tensor& eps_zz,
    const at::Tensor& eps_xy, const at::Tensor& eps_xz, const at::Tensor& eps_yz,
    const at::Tensor& p, const at::Tensor& nu_rho_field,
    const at::Tensor& tri_centroid, const at::Tensor& tri_normal,
    const at::Tensor& tri_area,
    const at::Tensor& tri_offsets, const at::Tensor& com_pos,
    const double bx0, const double by0, const double bz0,
    const double inv_dx, const double inv_dy, const double inv_dz,
    const int64_t Mx, const int64_t My, const int64_t Mz,
    const int64_t interp_method,
    const double sample_offset_pressure,
    const double sample_offset_friction,
    at::Tensor out)
{
    const int B = (int)com_pos.size(0);
    TORCH_CHECK(out.dim() == 2 && out.size(0) == B && out.size(1) == 12,
                "lagrangian_forces_3d_cuda: out must be (B, 12)");
    TORCH_CHECK(out.scalar_type() == at::kDouble,
                "lagrangian_forces_3d_cuda: out must be float64");
    TORCH_CHECK(tri_offsets.numel() == B + 1,
                "lagrangian_forces_3d_cuda: tri_offsets must have B+1 entries");

    out.zero_();
    if (B <= 0) return;

    auto exx_c  = eps_xx.contiguous();
    auto eyy_c  = eps_yy.contiguous();
    auto ezz_c  = eps_zz.contiguous();
    auto exy_c  = eps_xy.contiguous();
    auto exz_c  = eps_xz.contiguous();
    auto eyz_c  = eps_yz.contiguous();
    auto p_c    = p.contiguous();
    auto nrho_c = nu_rho_field.contiguous();
    auto ct_c   = tri_centroid.contiguous();
    auto nm_c   = tri_normal.contiguous();
    auto ar_c   = tri_area.contiguous();
    auto offs_c = tri_offsets.contiguous().to(at::kLong);
    auto com_c  = com_pos.contiguous();

    const bool nrho_scalar = (nrho_c.numel() == 1);

    auto offs_cpu = offs_c.cpu();
    const int64_t* offs_h = offs_cpu.data_ptr<int64_t>();
    int64_t T_max = 0;
    for (int b = 0; b < B; ++b) {
        T_max = std::max<int64_t>(T_max, offs_h[b + 1] - offs_h[b]);
    }
    if (T_max <= 0) return;

    const int blockSize = (T_max <= 64) ? 64 : (T_max <= 512) ? 128 : 256;
    const int blocksPerBody = (int)((T_max + blockSize - 1) / blockSize);
    const dim3 grid((unsigned)blocksPerBody, (unsigned)B, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "lagrangian_forces_3d_cuda", [&] {
        const int64_t T_total = ct_c.size(1);
        const scalar_t* cx = ct_c.data_ptr<scalar_t>();
        const scalar_t* cy = cx + T_total;
        const scalar_t* cz = cy + T_total;
        const scalar_t* nx_p = nm_c.data_ptr<scalar_t>();
        const scalar_t* ny_p = nx_p + T_total;
        const scalar_t* nz_p = ny_p + T_total;

        lagrangian_forces_3d_kernel<scalar_t>
            <<<grid, blockSize, 0, stream>>>(
                exx_c.data_ptr<scalar_t>(),
                eyy_c.data_ptr<scalar_t>(),
                ezz_c.data_ptr<scalar_t>(),
                exy_c.data_ptr<scalar_t>(),
                exz_c.data_ptr<scalar_t>(),
                eyz_c.data_ptr<scalar_t>(),
                p_c.data_ptr<scalar_t>(),
                nrho_c.data_ptr<scalar_t>(),
                (int)nrho_scalar,
                cx, cy, cz,
                nx_p, ny_p, nz_p,
                ar_c.data_ptr<scalar_t>(),
                offs_c.data_ptr<int64_t>(),
                com_c.data_ptr<scalar_t>(),
                (int)Mx, (int)My, (int)Mz,
                (scalar_t)bx0, (scalar_t)by0, (scalar_t)bz0,
                (scalar_t)inv_dx, (scalar_t)inv_dy, (scalar_t)inv_dz,
                (int)interp_method,
                (scalar_t)sample_offset_pressure,
                (scalar_t)sample_offset_friction,
                out.data_ptr<double>());
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("lagrangian_forces_3d", &lagrangian_forces_3d_cuda);
}

// =====================================================================
//  2-D Lagrangian forces (merged from cuda/lagrangian_forces_2d.cu)
// =====================================================================

template <typename scalar_t>
__device__ __forceinline__ scalar_t lf_bilinear_2d_d(
    const scalar_t* __restrict__ F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    return bilinear_sample_uniform_2d<scalar_t>(
        F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t lf_biquadratic_2d_d(
    const scalar_t* __restrict__ F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    return biquadratic_sample_uniform_2d<scalar_t>(
        F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
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
    return sdf_sample_dispatch_2d<scalar_t>(
        method, F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
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
