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

namespace lilytorch_kernels {

template <typename scalar_t>
__device__ __forceinline__ scalar_t lf_trilinear_3d_d(
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

    const int s2 = Mz;
    const int s1 = My * Mz;
    const int base = ix * s1 + iy * s2 + iz;

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

template <typename scalar_t>
__device__ __forceinline__ scalar_t lf_triquadratic_3d_d(
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
        return lf_trilinear_3d_d<scalar_t>(
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
                wzm * F[b1] + wz0 * F[b1 + 1] + wzp * F[b1 + 2];
            plane += wy * row;
        }
        out += wx * plane;
    }
    return out;
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
    if (method == 1) {
        return lf_triquadratic_3d_d<scalar_t>(
            F, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, xq, yq, zq);
    }
    return lf_trilinear_3d_d<scalar_t>(
        F, Mx, My, Mz, bx0, by0, bz0,
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

    const scalar_t e_xx = lf_sample_3d_d<scalar_t>(
        interp_method, exx, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qx, qy, qz);
    const scalar_t e_yy = lf_sample_3d_d<scalar_t>(
        interp_method, eyy, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qx, qy, qz);
    const scalar_t e_zz = lf_sample_3d_d<scalar_t>(
        interp_method, ezz, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qx, qy, qz);
    const scalar_t e_xy = lf_sample_3d_d<scalar_t>(
        interp_method, exy, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qx, qy, qz);
    const scalar_t e_xz = lf_sample_3d_d<scalar_t>(
        interp_method, exz, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qx, qy, qz);
    const scalar_t e_yz = lf_sample_3d_d<scalar_t>(
        interp_method, eyz, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qx, qy, qz);

    const scalar_t nu_rho_m = nrho_scalar
        ? nrho[0]
        : lf_sample_3d_d<scalar_t>(
            interp_method, nrho, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, qx, qy, qz);

    const scalar_t tvx = nu_rho_m * (e_xx * nxv + e_xy * nyv + e_xz * nzv);
    const scalar_t tvy = nu_rho_m * (e_xy * nxv + e_yy * nyv + e_yz * nzv);
    const scalar_t tvz = nu_rho_m * (e_xz * nxv + e_yz * nyv + e_zz * nzv);

    const scalar_t p_m = lf_sample_3d_d<scalar_t>(
        interp_method, pp, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, qx, qy, qz);
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
                out.data_ptr<double>());
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("lagrangian_forces_3d", &lagrangian_forces_3d_cuda);
}

}  // namespace lilytorch_kernels
