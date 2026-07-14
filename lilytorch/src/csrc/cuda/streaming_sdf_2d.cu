// =====================================================================
//  streaming_sdf_2d.cu
//
//  2-D streaming-SDF support kernels (post-2.4: union path removed —
//  production streaming is streaming_sdf_direct.cu + streaming_sdf_regime_b.cu).
//  Keeps: forces-post readout, bdim_coeff, fused BCs, scattered interp.
//  Mirrors ``streaming_sdf.cu`` with the z-axis stripped; see the matching
//  helpers in ``streaming_sdf_cpu_2d.cpp`` for the algorithmic rationale.
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

#include "../bc_ops.h"

namespace lilytorch_kernels {

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

// =====================================================================
//  CUDA dispatch (single body)
// =====================================================================
template <typename scalar_t, int BLOCK_SIZE>
__global__ void streaming_sdf_forces_post_2d_kernel(
    const scalar_t* __restrict__ F_flat,
    const int64_t*  __restrict__ F_offsets,
    const int64_t*  __restrict__ body_shapes,
    const scalar_t* __restrict__ body_meta,
    const scalar_t* __restrict__ kin,
    const int64_t*  __restrict__ aabb_lo,
    const int64_t*  __restrict__ aabb_dim,
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const int Ngx,
    const int Ngy,
    const scalar_t* __restrict__ sdf_cc,
    const int interp_method,
    const scalar_t* __restrict__ u_prev,
    const scalar_t* __restrict__ v_prev,
    const scalar_t* __restrict__ p_prev,
    const scalar_t* __restrict__ nu_rho_field,
    const int64_t   nu_rho_field_size,
    const scalar_t inv_h,
    const scalar_t eps_body,
    const scalar_t eps_solver,
    const scalar_t h2,
    const int delta_order,
    const int with_pressure,
    double* __restrict__ out)
{
    const int b     = blockIdx.y;
    const int local = blockIdx.x * blockDim.x + threadIdx.x;

    const int Ai = (int)aabb_dim[b*2 + 0];
    const int Aj = (int)aabb_dim[b*2 + 1];
    const int vol = Ai * Aj;

    double acc[6];
#pragma unroll
    for (int c = 0; c < 6; ++c) acc[c] = 0.0;

    if (local < vol) {
        const int di = local / Aj;
        const int dj = local - di * Aj;

        const int i0 = (int)aabb_lo[b*2 + 0];
        const int j0 = (int)aabb_lo[b*2 + 1];
        const int i  = i0 + di;
        const int j  = j0 + dj;
        const int g_idx = i * Ngy + j;

        const scalar_t* F  = F_flat + F_offsets[b];
        const int Mx = (int)body_shapes[b*2 + 0];
        const int My = (int)body_shapes[b*2 + 1];

        const scalar_t* M  = body_meta + b*7;
        const scalar_t bx0 = M[0], by0 = M[1];
        const scalar_t idx_ = M[4], idy_ = M[5];

        const scalar_t* K  = kin + b*11;
        const scalar_t r00 = K[0], r01 = K[1];
        const scalar_t r10 = K[2], r11 = K[3];
        const scalar_t bp_x = K[4], bp_y = K[5];
        const scalar_t cm_x = K[6], cm_y = K[7];

        const scalar_t xc = gx[i];
        const scalar_t yc = gy[j];
        const scalar_t dx_w = xc - bp_x;
        const scalar_t dy_w = yc - bp_y;
        const scalar_t bxq = r00 * dx_w + r01 * dy_w;
        const scalar_t byq = r10 * dx_w + r11 * dy_w;

        const scalar_t s_cc_body = sdf_sample_dispatch_2d(
            interp_method, F, Mx, My, bx0, by0, idx_, idy_, bxq, byq);

        const scalar_t band_lo = (eps_solver - eps_body) < (-eps_body)
            ? (eps_solver - eps_body) : (-eps_body);
        const scalar_t band_hi = (eps_solver + eps_body) > (eps_body)
            ? (eps_solver + eps_body) : (eps_body);

        if (s_cc_body > band_lo && s_cc_body < band_hi) {
            const scalar_t nu_rho_val = (nu_rho_field_size == 1)
                ? nu_rho_field[0] : nu_rho_field[g_idx];

            scalar_t dsdx_union = 0;
            if (Ngx >= 3) {
                if (i == 0) {
                    dsdx_union = (
                        (scalar_t)(-3) * sdf_cc[j]
                        + (scalar_t)4 * sdf_cc[Ngy + j]
                        - sdf_cc[2 * Ngy + j]
                    ) * (scalar_t)0.5 * inv_h;
                } else if (i == Ngx - 1) {
                    dsdx_union = (
                        (scalar_t)3 * sdf_cc[(Ngx - 1) * Ngy + j]
                        - (scalar_t)4 * sdf_cc[(Ngx - 2) * Ngy + j]
                        + sdf_cc[(Ngx - 3) * Ngy + j]
                    ) * (scalar_t)0.5 * inv_h;
                } else {
                    dsdx_union = (
                        sdf_cc[(i + 1) * Ngy + j]
                        - sdf_cc[(i - 1) * Ngy + j]
                    ) * (scalar_t)0.5 * inv_h;
                }
            } else if (Ngx == 2) {
                dsdx_union = (sdf_cc[Ngy + j] - sdf_cc[j]) * inv_h;
            }

            scalar_t dsdy_union = 0;
            if (Ngy >= 3) {
                const int row = i * Ngy;
                if (j == 0) {
                    dsdy_union = (
                        (scalar_t)(-3) * sdf_cc[row]
                        + (scalar_t)4 * sdf_cc[row + 1]
                        - sdf_cc[row + 2]
                    ) * (scalar_t)0.5 * inv_h;
                } else if (j == Ngy - 1) {
                    dsdy_union = (
                        (scalar_t)3 * sdf_cc[row + (Ngy - 1)]
                        - (scalar_t)4 * sdf_cc[row + (Ngy - 2)]
                        + sdf_cc[row + (Ngy - 3)]
                    ) * (scalar_t)0.5 * inv_h;
                } else {
                    dsdy_union = (
                        sdf_cc[row + (j + 1)]
                        - sdf_cc[row + (j - 1)]
                    ) * (scalar_t)0.5 * inv_h;
                }
            } else if (Ngy == 2) {
                dsdy_union = (sdf_cc[i * Ngy + 1] - sdf_cc[i * Ngy]) * inv_h;
            }

            const scalar_t union_norm = sqrt(dsdx_union * dsdx_union + dsdy_union * dsdy_union);
            const scalar_t union_inv_norm = union_norm > (scalar_t)0
                ? ((scalar_t)1.0 / union_norm)
                : (scalar_t)0;
            const scalar_t nx = dsdx_union * union_inv_norm;
            const scalar_t ny = dsdy_union * union_inv_norm;

            const int im1 = (i > 0)         ? i-1 : 0;
            const int ip1 = (i+1 < Ngx)     ? i+1 : i;
            const int im2 = (i > 1)         ? i-2 : 0;
            const int ip2 = (i+2 < Ngx)     ? i+2 : (Ngx - 1);
            const int jm1 = (j > 0)         ? j-1 : 0;
            const int jp1 = (j+1 < Ngy)     ? j+1 : j;
            const int jm2 = (j > 1)         ? j-2 : 0;
            const int jp2 = (j+2 < Ngy)     ? j+2 : (Ngy - 1);

            scalar_t dudx;
            if (i + 1 < Ngx) {
                dudx = (u_prev[ip1 * Ngy + j] - u_prev[i * Ngy + j]) * inv_h;
            } else {
                dudx = (u_prev[i * Ngy + j] - u_prev[im1 * Ngy + j]) * inv_h;
            }

            scalar_t dvdy;
            if (j + 1 < Ngy) {
                dvdy = (v_prev[i * Ngy + jp1] - v_prev[i * Ngy + j]) * inv_h;
            } else {
                dvdy = (v_prev[i * Ngy + j] - v_prev[i * Ngy + jm1]) * inv_h;
            }

            const scalar_t u_cc_jm2 = (scalar_t)0.5 * (u_prev[i * Ngy + jm2] + u_prev[ip1 * Ngy + jm2]);
            const scalar_t u_cc_jm1 = (scalar_t)0.5 * (u_prev[i * Ngy + jm1] + u_prev[ip1 * Ngy + jm1]);
            const scalar_t u_cc_j0  = (scalar_t)0.5 * (u_prev[i * Ngy + j  ] + u_prev[ip1 * Ngy + j  ]);
            const scalar_t u_cc_jp1 = (scalar_t)0.5 * (u_prev[i * Ngy + jp1] + u_prev[ip1 * Ngy + jp1]);
            const scalar_t u_cc_jp2 = (scalar_t)0.5 * (u_prev[i * Ngy + jp2] + u_prev[ip1 * Ngy + jp2]);

            scalar_t dudy;
            if (Ngy >= 3) {
                if (j == 0) {
                    dudy = ((scalar_t)(-3) * u_cc_j0 + (scalar_t)4 * u_cc_jp1 - u_cc_jp2)
                         * (scalar_t)0.5 * inv_h;
                } else if (j == Ngy - 1) {
                    dudy = ((scalar_t)3 * u_cc_j0 - (scalar_t)4 * u_cc_jm1 + u_cc_jm2)
                         * (scalar_t)0.5 * inv_h;
                } else {
                    dudy = (u_cc_jp1 - u_cc_jm1) * (scalar_t)0.5 * inv_h;
                }
            } else {
                dudy = (u_cc_jp1 - u_cc_jm1) * (scalar_t)0.5 * inv_h;
            }

            const scalar_t v_cc_im2 = (scalar_t)0.5 * (v_prev[im2 * Ngy + j] + v_prev[im2 * Ngy + jp1]);
            const scalar_t v_cc_im1 = (scalar_t)0.5 * (v_prev[im1 * Ngy + j] + v_prev[im1 * Ngy + jp1]);
            const scalar_t v_cc_i0  = (scalar_t)0.5 * (v_prev[i   * Ngy + j] + v_prev[i   * Ngy + jp1]);
            const scalar_t v_cc_ip1 = (scalar_t)0.5 * (v_prev[ip1 * Ngy + j] + v_prev[ip1 * Ngy + jp1]);
            const scalar_t v_cc_ip2 = (scalar_t)0.5 * (v_prev[ip2 * Ngy + j] + v_prev[ip2 * Ngy + jp1]);

            scalar_t dvdx;
            if (Ngx >= 3) {
                if (i == 0) {
                    dvdx = ((scalar_t)(-3) * v_cc_i0 + (scalar_t)4 * v_cc_ip1 - v_cc_ip2)
                         * (scalar_t)0.5 * inv_h;
                } else if (i == Ngx - 1) {
                    dvdx = ((scalar_t)3 * v_cc_i0 - (scalar_t)4 * v_cc_im1 + v_cc_im2)
                         * (scalar_t)0.5 * inv_h;
                } else {
                    dvdx = (v_cc_ip1 - v_cc_im1) * (scalar_t)0.5 * inv_h;
                }
            } else {
                dvdx = (v_cc_ip1 - v_cc_im1) * (scalar_t)0.5 * inv_h;
            }

            const scalar_t xs = nu_rho_val * (2*dudx*nx + (dudy+dvdx)*ny);
            const scalar_t ys = nu_rho_val * ((dvdx+dudy)*nx + 2*dvdy*ny);

            const scalar_t p_c = p_prev[g_idx];
            const scalar_t pxv = -p_c * nx;
            const scalar_t pyv = -p_c * ny;

            const scalar_t pi_v     = (scalar_t)3.141592653589793;
            const scalar_t inv_2eps = (scalar_t)0.5 / eps_body;
            const scalar_t pi_ov_eb = pi_v / eps_body;

            scalar_t delta_visc = 0;
            scalar_t delta_pres = 0;
            const scalar_t d_visc = s_cc_body - eps_solver;
            if (d_visc > -eps_body && d_visc < eps_body)
                delta_visc = ((scalar_t)1 + cos(pi_ov_eb * d_visc)) * inv_2eps;
            if (s_cc_body > -eps_body && s_cc_body < eps_body)
                delta_pres = ((scalar_t)1 + cos(pi_ov_eb * s_cc_body)) * inv_2eps;
            // deltaH readout supplies the pressure force/torque from a separate
            // union-∂H pass; here we emit only the viscous channels.
            if (!with_pressure) delta_pres = 0;

            if (delta_order == 2 && (delta_visc > 0 || delta_pres > 0)) {
                const scalar_t h_grid = (scalar_t)1.0 / inv_h;
                const scalar_t s_xp = sdf_sample_dispatch_2d(
                    interp_method, F, Mx, My, bx0, by0, idx_, idy_,
                    bxq + r00*h_grid, byq + r10*h_grid);
                const scalar_t s_xm = sdf_sample_dispatch_2d(
                    interp_method, F, Mx, My, bx0, by0, idx_, idy_,
                    bxq - r00*h_grid, byq - r10*h_grid);
                const scalar_t s_yp = sdf_sample_dispatch_2d(
                    interp_method, F, Mx, My, bx0, by0, idx_, idy_,
                    bxq + r01*h_grid, byq + r11*h_grid);
                const scalar_t s_ym = sdf_sample_dispatch_2d(
                    interp_method, F, Mx, My, bx0, by0, idx_, idy_,
                    bxq - r01*h_grid, byq - r11*h_grid);
                const scalar_t dsdx = (s_xp - s_xm) * (scalar_t)0.5 * inv_h;
                const scalar_t dsdy = (s_yp - s_ym) * (scalar_t)0.5 * inv_h;
                scalar_t grad_mag = sqrt(dsdx*dsdx + dsdy*dsdy);
                const scalar_t min_grad = (scalar_t)1e-3;
                if (grad_mag < min_grad) grad_mag = min_grad;
                const scalar_t inv_grad = (scalar_t)1.0 / grad_mag;
                delta_visc *= inv_grad;
                delta_pres *= inv_grad;
            }

            const scalar_t arm_x = xc - cm_x;
            const scalar_t arm_y = yc - cm_y;

            const double fv_x = (double)(xs  * delta_visc);
            const double fv_y = (double)(ys  * delta_visc);
            const double fp_x = (double)(pxv * delta_pres);
            const double fp_y = (double)(pyv * delta_pres);

            acc[0] = fv_x;
            acc[1] = fv_y;
            acc[2] = (double)arm_x * fv_y - (double)arm_y * fv_x;
            acc[3] = fp_x;
            acc[4] = fp_y;
            acc[5] = (double)arm_x * fp_y - (double)arm_y * fp_x;
        }
    }

    // Block-wide sum via CUB BlockReduce (warp shuffles + 1 shmem slot per
    // warp).  Replaces the previous 12 KB / block manual reduction; new
    // shmem footprint is the BlockReduce TempStorage (~128 B), lifting
    // the occupancy cap on consumer SMs.
    using BlockReduceD = cub::BlockReduce<double, BLOCK_SIZE>;
    __shared__ typename BlockReduceD::TempStorage tmp;
    const double h2_d = (double)h2;
#pragma unroll
    for (int c = 0; c < 6; ++c) {
        const double s = BlockReduceD(tmp).Sum(acc[c]);
        if (threadIdx.x == 0)
            atomicAdd(&out[b*6 + c], s * h2_d);
        __syncthreads();
    }
}

// ----------------------------------------------------------------------
//  2-D partial-Heaviside (∂H) pressure-force readout  --  deltaH submethod.
//  See the 3-D analogue in streaming_sdf.cu for the full rationale.
// ----------------------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ scalar_t heaviside_smooth_dev_2d(scalar_t phi, scalar_t inv_eps) {
    const scalar_t pi = (scalar_t)3.141592653589793;
    scalar_t x = phi * inv_eps;
    x = x < (scalar_t)-1 ? (scalar_t)-1 : (x > (scalar_t)1 ? (scalar_t)1 : x);
    return (scalar_t)0.5 * ((scalar_t)1 + x + sin(pi * x) / pi);
}

template <typename scalar_t>
__global__ void forces_post_deltaH_pressure_2d_kernel(
    const scalar_t* __restrict__ F_flat,
    const int64_t*  __restrict__ F_offsets,
    const int64_t*  __restrict__ body_shapes,
    const scalar_t* __restrict__ body_meta,
    const scalar_t* __restrict__ kin,
    const int64_t*  __restrict__ aabb_lo,
    const int64_t*  __restrict__ aabb_dim,
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const int Ngx, const int Ngy,
    const scalar_t* __restrict__ sdf_cc,
    const int interp_method,
    const scalar_t* __restrict__ p_prev,
    const scalar_t inv_h,
    const scalar_t inv_eps,
    const scalar_t inv_tau,
    const scalar_t h2,
    const int B,
    const int uli0, const int ulj0,
    const int ULi, const int ULj,
    double* __restrict__ out)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const int uvol = ULi * ULj;
    if (local >= uvol) return;
    const int di = local / ULj;
    const int dj = local - di * ULj;
    const int i = uli0 + di;
    const int j = ulj0 + dj;
    if (i < 0 || i >= Ngx || j < 0 || j >= Ngy) return;

#define SDF_AT(ii, jj) (sdf_cc[(int64_t)(ii) * Ngy + (jj)])
#define HV_AT(ii, jj) heaviside_smooth_dev_2d<scalar_t>(SDF_AT(ii, jj), inv_eps)
    scalar_t gHx = 0, gHy = 0;
    if (Ngx >= 3) {
        if (i == 0) gHx = ((scalar_t)(-3)*HV_AT(0,j) + (scalar_t)4*HV_AT(1,j) - HV_AT(2,j)) * (scalar_t)0.5 * inv_h;
        else if (i == Ngx-1) gHx = ((scalar_t)3*HV_AT(Ngx-1,j) - (scalar_t)4*HV_AT(Ngx-2,j) + HV_AT(Ngx-3,j)) * (scalar_t)0.5 * inv_h;
        else gHx = (HV_AT(i+1,j) - HV_AT(i-1,j)) * (scalar_t)0.5 * inv_h;
    } else if (Ngx == 2) gHx = (HV_AT(1,j) - HV_AT(0,j)) * inv_h;
    if (Ngy >= 3) {
        if (j == 0) gHy = ((scalar_t)(-3)*HV_AT(i,0) + (scalar_t)4*HV_AT(i,1) - HV_AT(i,2)) * (scalar_t)0.5 * inv_h;
        else if (j == Ngy-1) gHy = ((scalar_t)3*HV_AT(i,Ngy-1) - (scalar_t)4*HV_AT(i,Ngy-2) + HV_AT(i,Ngy-3)) * (scalar_t)0.5 * inv_h;
        else gHy = (HV_AT(i,j+1) - HV_AT(i,j-1)) * (scalar_t)0.5 * inv_h;
    } else if (Ngy == 2) gHy = (HV_AT(i,1) - HV_AT(i,0)) * inv_h;
#undef HV_AT
#undef SDF_AT
    if (gHx == (scalar_t)0 && gHy == (scalar_t)0) return;

    const int64_t g = (int64_t)i * Ngy + j;
    const scalar_t p_c = p_prev[g];
    const scalar_t fdx = -p_c * gHx, fdy = -p_c * gHy;
    const scalar_t xc = gx[i], yc = gy[j];
    const scalar_t sdfu = sdf_cc[g];

    scalar_t Z = 0;
    for (int b = 0; b < B; ++b) {
        const int i0 = (int)aabb_lo[b*2+0], j0 = (int)aabb_lo[b*2+1];
        const int Ai = (int)aabb_dim[b*2+0], Aj = (int)aabb_dim[b*2+1];
        if (i < i0 || i >= i0+Ai || j < j0 || j >= j0+Aj) continue;
        const scalar_t* F = F_flat + F_offsets[b];
        const int Mx = (int)body_shapes[b*2+0], My = (int)body_shapes[b*2+1];
        const scalar_t* M = body_meta + b*7;
        const scalar_t* K = kin + b*11;
        const scalar_t dx_w = xc - K[4], dy_w = yc - K[5];
        const scalar_t bxq = K[0]*dx_w + K[1]*dy_w;
        const scalar_t byq = K[2]*dx_w + K[3]*dy_w;
        const scalar_t s_b = sdf_sample_dispatch_2d(interp_method, F, Mx, My,
            M[0], M[1], M[4], M[5], bxq, byq);
        Z += exp(-(s_b - sdfu) * inv_tau);
    }
    if (Z <= (scalar_t)0) return;
    const scalar_t inv_Z = (scalar_t)1 / Z;
    const double h2_d = (double)h2;

    for (int b = 0; b < B; ++b) {
        const int i0 = (int)aabb_lo[b*2+0], j0 = (int)aabb_lo[b*2+1];
        const int Ai = (int)aabb_dim[b*2+0], Aj = (int)aabb_dim[b*2+1];
        if (i < i0 || i >= i0+Ai || j < j0 || j >= j0+Aj) continue;
        const scalar_t* F = F_flat + F_offsets[b];
        const int Mx = (int)body_shapes[b*2+0], My = (int)body_shapes[b*2+1];
        const scalar_t* M = body_meta + b*7;
        const scalar_t* K = kin + b*11;
        const scalar_t dx_w = xc - K[4], dy_w = yc - K[5];
        const scalar_t bxq = K[0]*dx_w + K[1]*dy_w;
        const scalar_t byq = K[2]*dx_w + K[3]*dy_w;
        const scalar_t s_b = sdf_sample_dispatch_2d(interp_method, F, Mx, My,
            M[0], M[1], M[4], M[5], bxq, byq);
        const scalar_t wb = exp(-(s_b - sdfu) * inv_tau) * inv_Z;
        const scalar_t fbx = wb * fdx, fby = wb * fdy;
        const scalar_t ax = xc - K[6], ay = yc - K[7];
        atomicAdd(&out[b*6 + 3], (double)fbx * h2_d);
        atomicAdd(&out[b*6 + 4], (double)fby * h2_d);
        atomicAdd(&out[b*6 + 5], ((double)ax*(double)fby - (double)ay*(double)fbx) * h2_d);
    }
}

void streaming_sdf_forces_post_2d_cuda(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes, const at::Tensor& body_meta,
    const at::Tensor& kin,
    const at::Tensor& aabb_lo, const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy,
    const double h_grid, const int64_t max_vol_per_body,
    const at::Tensor& sdf_cc,
    const int64_t interp_method,
    const at::Tensor& u_prev, const at::Tensor& v_prev, const at::Tensor& p_prev,
    const at::Tensor& nu_rho_field,
    const double eps_body, const double eps_solver, const double h2,
    const int64_t delta_order,
    const int64_t force_submethod,
    const double ph_tau,
    at::Tensor out)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngx = (int)gx.numel();
    const int Ngy = (int)gy.numel();
    const int with_pressure = (force_submethod == 0) ? 1 : 0;

    // Adaptive blockSize matching the streaming_sdf_stag launcher; CUB's
    // BlockReduce takes block size as a compile-time template parameter,
    // so we fan out to one of the three configured sizes.
    const int blockSize = (max_vol_per_body <= 128)  ? 32
                        : (max_vol_per_body <= 4096) ? 128
                                                     : 256;
    const int nblocks = (int)((max_vol_per_body + blockSize - 1) / blockSize);

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_forces_post_2d_cuda", [&] {
        auto launch = [&](auto block_size_ic) {
            constexpr int BS = decltype(block_size_ic)::value;
            streaming_sdf_forces_post_2d_kernel<scalar_t, BS>
                <<<dim3(nblocks, B, 1), dim3(BS, 1, 1), 0, stream>>>(
                    F_flat.data_ptr<scalar_t>(),
                    F_offsets.data_ptr<int64_t>(),
                    body_shapes.data_ptr<int64_t>(),
                    body_meta.data_ptr<scalar_t>(),
                    kin.data_ptr<scalar_t>(),
                    aabb_lo.data_ptr<int64_t>(),
                    aabb_dim.data_ptr<int64_t>(),
                    gx.data_ptr<scalar_t>(),
                    gy.data_ptr<scalar_t>(),
                    Ngx, Ngy,
                    sdf_cc.data_ptr<scalar_t>(),
                    (int)interp_method,
                    u_prev.data_ptr<scalar_t>(),
                    v_prev.data_ptr<scalar_t>(),
                    p_prev.data_ptr<scalar_t>(),
                    nu_rho_field.data_ptr<scalar_t>(),
                    (int64_t)nu_rho_field.numel(),
                    (scalar_t)(1.0 / h_grid),
                    (scalar_t)eps_body,
                    (scalar_t)eps_solver,
                    (scalar_t)h2,
                    (int)delta_order,
                    with_pressure,
                    out.data_ptr<double>());
        };
        switch (blockSize) {
            case 32:  launch(std::integral_constant<int, 32>{}); break;
            case 128: launch(std::integral_constant<int, 128>{}); break;
            default:  launch(std::integral_constant<int, 256>{}); break;
        }
    });

    if (force_submethod != 0) {
        // deltaH: second pass fills the pressure force/torque from union-∂H.
        auto lo_c  = aabb_lo.to(at::kCPU);
        auto dim_c = aabb_dim.to(at::kCPU);
        const int64_t* loh  = lo_c.data_ptr<int64_t>();
        const int64_t* dimh = dim_c.data_ptr<int64_t>();
        int ulo[2] = {Ngx, Ngy};
        int uhi[2] = {0, 0};
        for (int b = 0; b < B; ++b)
            for (int d = 0; d < 2; ++d) {
                const int a0 = (int)loh[b*2+d];
                const int a1 = a0 + (int)dimh[b*2+d];
                if (a0 < ulo[d]) ulo[d] = a0;
                if (a1 > uhi[d]) uhi[d] = a1;
            }
        const int Ng[2] = {Ngx, Ngy};
        const int halo = 2;
        for (int d = 0; d < 2; ++d) {
            ulo[d] -= halo; if (ulo[d] < 0) ulo[d] = 0;
            uhi[d] += halo; if (uhi[d] > Ng[d]) uhi[d] = Ng[d];
        }
        const int ULi = uhi[0] - ulo[0];
        const int ULj = uhi[1] - ulo[1];
        const int64_t uvol = (int64_t)ULi * ULj;
        if (ULi > 0 && ULj > 0) {
            const double tau = (ph_tau > 0.0) ? ph_tau : 1e-9;
            const int bs = 256;
            const int nb = (int)((uvol + bs - 1) / bs);
            AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "forces_post_deltaH_pressure_2d_cuda", [&] {
                forces_post_deltaH_pressure_2d_kernel<scalar_t>
                    <<<dim3(nb, 1, 1), dim3(bs, 1, 1), 0, stream>>>(
                        F_flat.data_ptr<scalar_t>(), F_offsets.data_ptr<int64_t>(),
                        body_shapes.data_ptr<int64_t>(), body_meta.data_ptr<scalar_t>(),
                        kin.data_ptr<scalar_t>(), aabb_lo.data_ptr<int64_t>(),
                        aabb_dim.data_ptr<int64_t>(), gx.data_ptr<scalar_t>(),
                        gy.data_ptr<scalar_t>(), Ngx, Ngy,
                        sdf_cc.data_ptr<scalar_t>(), (int)interp_method,
                        p_prev.data_ptr<scalar_t>(),
                        (scalar_t)(1.0 / h_grid), (scalar_t)(1.0 / eps_body),
                        (scalar_t)(1.0 / tau), (scalar_t)h2, B,
                        ulo[0], ulo[1], ULi, ULj,
                        out.data_ptr<double>());
            });
        }
    }
}

// =====================================================================
//  apply_bcs_2d (CUDA)
//
//  2-D analogue of ``apply_bcs_3d``: writes one ghost line per op
//  (Neumann copy, Dirichlet constant, or reflective).  One launch per op
//  kind; the write-ownership rule in bc_ops.h keeps the corner ghosts
//  race-free and deterministic.  See CPU impl for the argument layout.
// =====================================================================

template <typename scalar_t>
__global__ void apply_bcs_2d_kernel(
    scalar_t* __restrict__ u,
    scalar_t* __restrict__ v,
    const int64_t* __restrict__ shapes,
    const int kind,
    const int* __restrict__ desc,
    const scalar_t* __restrict__ vals,   // null for Neumann
    const int nops)
{
    using namespace lilytorch_kernels::bcs;

    const int op = blockIdx.y;
    if (op >= nops) return;

    int comp, axis, dst_along, src_along;
    bc_decode(kind, desc, op, shapes, /*ndim=*/2, comp, axis, dst_along, src_along);

    const int Nx = (int)shapes[comp * 2 + 0];
    const int Ny = (int)shapes[comp * 2 + 1];
    const int dim0_max = (axis == 0) ? Ny : Nx;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim0_max) return;

    int c[2];
    if (axis == 0) { c[0] = dst_along; c[1] = i;         }
    else           { c[0] = i;         c[1] = dst_along; }

    int s[2];
    if (!bc_own_and_source(kind, desc, nops, op, shapes, /*ndim=*/2,
                           comp, axis, src_along, c, s))
        return;                                   // a lower-indexed op owns it

    scalar_t* base = (comp == 0) ? u : v;
    const int64_t dst_lin = (int64_t)c[0] * Ny + c[1];
    const int64_t src_lin = (int64_t)s[0] * Ny + s[1];

    if      (kind == BC_KIND_NEUMANN)   base[dst_lin] = base[src_lin];
    else if (kind == BC_KIND_DIRICHLET) base[dst_lin] = vals[op];
    else base[dst_lin] = scalar_t(2) * vals[op] - base[src_lin];
}

void apply_bcs_2d_cuda(
    at::Tensor u, at::Tensor v,
    const at::Tensor& shapes,
    const at::Tensor& neu_desc,
    const at::Tensor& dir_desc,
    const at::Tensor& dir_val,
    const at::Tensor& ref_desc,
    const at::Tensor& ref_val,
    const int64_t max_line_dim)
{
    TORCH_CHECK(u.is_cuda() && v.is_cuda(),
                "apply_bcs_2d_cuda: u/v must be CUDA tensors");
    TORCH_CHECK(u.is_contiguous() && v.is_contiguous(),
                "apply_bcs_2d_cuda: u/v must be contiguous");
    TORCH_CHECK(u.scalar_type() == v.scalar_type(),
                "apply_bcs_2d_cuda: u/v must share dtype");
    TORCH_CHECK(shapes.scalar_type() == at::kLong &&
                shapes.dim() == 2 && shapes.size(0) == 2 && shapes.size(1) == 2,
                "apply_bcs_2d_cuda: shapes must be int64[2,2]");
    TORCH_CHECK(ref_desc.scalar_type() == at::kInt && ref_desc.dim() == 2 &&
                ref_desc.size(1) == 4,
                "apply_bcs_2d_cuda: ref_desc must be int32[N,4]");

    const int N_neu = (int)neu_desc.size(0);
    const int N_dir = (int)dir_desc.size(0);
    const int N_ref = (int)ref_desc.size(0);
    if (N_neu + N_dir + N_ref == 0 || max_line_dim <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blockX = 256;
    const int gridX  = (int)((max_line_dim + blockX - 1) / blockX);

    // One launch per op kind, in the order the eager reference applies them:
    //   Neumann → Dirichlet → reflective.
    // The stage boundaries define the cross-kind overlaps (a reflective op
    // reads the adjacent cell AFTER any Dirichlet wall write to it, a Neumann
    // op BEFORE — as on CPU); within a stage, bc_own_and_source() picks a
    // single writer per cell.
    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_2d_cuda", [&] {
        using namespace lilytorch_kernels::bcs;
        auto launch = [&](int kind, const at::Tensor& desc,
                          const scalar_t* vals, int nops) {
            if (nops == 0) return;
            apply_bcs_2d_kernel<scalar_t>
                <<<dim3(gridX, nops, 1), dim3(blockX, 1, 1), 0, stream>>>(
                    u.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                    shapes.data_ptr<int64_t>(),
                    kind, desc.data_ptr<int>(), vals, nops);
        };
        launch(BC_KIND_NEUMANN, neu_desc, nullptr, N_neu);
        launch(BC_KIND_DIRICHLET, dir_desc,
               (N_dir > 0) ? dir_val.data_ptr<scalar_t>() : nullptr, N_dir);
        launch(BC_KIND_REFLECTIVE, ref_desc,
               (N_ref > 0) ? ref_val.data_ptr<scalar_t>() : nullptr, N_ref);
    });
}

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

// =====================================================================
//  Phase-I memory-reduction 2-D pipeline.  Kernel A streams the union
//  SDF / face velocities without the legacy per-cell winning-density
//  scratch tensor; Kernel B fuses the BDIM2 velocity update with the
//  variable-density Poisson coefficient calculation, computing mu0,
//  mu1 and the unit normal in CUDA thread registers.
// =====================================================================
// =====================================================================
//  Kernel B (2-D): fused BDIM2 + variable-density Poisson coefficients.
//  Mirrors bdim_coeff_3d_cuda from streaming_sdf.cu, with the z axis
//  removed.  See the 3-D version for documentation of the formulas.
// =====================================================================
template <typename scalar_t>
__device__ __forceinline__ void bdim_one_axis_2d(
    const scalar_t* __restrict__ phi_prime,
    const scalar_t* __restrict__ sdf,
    const scalar_t* __restrict__ body,
    const scalar_t eps,
    const scalar_t rho_f,
    const scalar_t dt,
    const scalar_t inv_2h,
    const int Ngx, const int Ngy,
    const int i, const int j,
    scalar_t* __restrict__ phi_out,
    scalar_t* __restrict__ c_out,
    const int mu0_proj)
{
    const int stride_i = Ngy;
    const int g  = i * stride_i + j;

    const int im = (i > 0)         ? (i - 1) : 0;
    const int ip = (i < Ngx - 1)   ? (i + 1) : Ngx - 1;
    const int jm = (j > 0)         ? (j - 1) : 0;
    const int jp = (j < Ngy - 1)   ? (j + 1) : Ngy - 1;

    const int g_im = im * stride_i + j;
    const int g_ip = ip * stride_i + j;
    const int g_jm = i  * stride_i + jm;
    const int g_jp = i  * stride_i + jp;

    const scalar_t phi = sdf[g];
    scalar_t mu0, mu1;
    if (phi <= -eps) {
        mu0 = scalar_t(0);
        mu1 = scalar_t(0);
    } else if (phi >= eps) {
        mu0 = scalar_t(1);
        mu1 = scalar_t(0);
    } else {
        const scalar_t deps = phi / eps;
        const scalar_t pi   = scalar_t(M_PI);
        const scalar_t s    = sin(pi * deps);
        const scalar_t c    = cos(pi * deps);
        mu0 = scalar_t(0.5) * (scalar_t(1) + deps + s / pi);
        mu1 = eps * (
            scalar_t(0.25)
            - scalar_t(0.25) * deps * deps
            - (s * deps + (scalar_t(1) + c) / pi) / (scalar_t(2) * pi)
        );
    }

    scalar_t nx = (sdf[g_ip] - sdf[g_im]) * inv_2h;
    scalar_t ny = (sdf[g_jp] - sdf[g_jm]) * inv_2h;
    const scalar_t nn = sqrt(nx*nx + ny*ny);
    if (nn > scalar_t(0)) {
        const scalar_t inv_nn = scalar_t(1) / nn;
        nx *= inv_nn;
        ny *= inv_nn;
    }

    const scalar_t b_c    = body[g];
    const scalar_t pp_c   = phi_prime[g];
    const scalar_t diff_c = pp_c - b_c;

    scalar_t ddx, ddy;
    if (i > 0 && i < Ngx - 1) {
        ddx = ((phi_prime[g_ip] - body[g_ip]) -
               (phi_prime[g_im] - body[g_im])) * inv_2h;
    } else {
        ddx = scalar_t(0);
    }
    if (j > 0 && j < Ngy - 1) {
        ddy = ((phi_prime[g_jp] - body[g_jp]) -
               (phi_prime[g_jm] - body[g_jm])) * inv_2h;
    } else {
        ddy = scalar_t(0);
    }
    const scalar_t nd = nx * ddx + ny * ddy;

    phi_out[g] = mu0 * diff_c + b_c + mu1 * nd;
    // BDIM2 coefficient dt*mu0/rho_fluid (Weymouth & Yue): the body enters via
    // mu0 only, NOT its density.  mu0_proj == 0 → plain dt/rho (no mu0 numerator).
    c_out[g]   = (mu0_proj ? dt * mu0 : dt) / rho_f;
}

template <typename scalar_t>
__global__ void bdim_coeff_2d_kernel(
    const scalar_t* __restrict__ u_prime,
    const scalar_t* __restrict__ v_prime,
    const scalar_t* __restrict__ sdf_u,
    const scalar_t* __restrict__ sdf_v,
    const scalar_t* __restrict__ body_u,
    const scalar_t* __restrict__ body_v,
    scalar_t* __restrict__ u0,
    scalar_t* __restrict__ v0,
    scalar_t* __restrict__ ch,
    scalar_t* __restrict__ cv,
    const scalar_t eps,
    const scalar_t rho_f,
    const scalar_t dt,
    const scalar_t inv_2h,
    const int Ngx, const int Ngy,
    const int di0, const int dj0,
    const int dAi, const int dAj,
    const int dirty_vol,
    const int mu0_proj)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= dirty_vol) return;
    const int dj = local % dAj;
    const int di = local / dAj;
    const int i = di0 + di;
    const int j = dj0 + dj;

    bdim_one_axis_2d<scalar_t>(
        u_prime, sdf_u, body_u,
        eps, rho_f, dt, inv_2h,
        Ngx, Ngy, i, j, u0, ch, mu0_proj);
    bdim_one_axis_2d<scalar_t>(
        v_prime, sdf_v, body_v,
        eps, rho_f, dt, inv_2h,
        Ngx, Ngy, i, j, v0, cv, mu0_proj);
}

void bdim_coeff_2d_cuda(
    const at::Tensor& u_prime,
    const at::Tensor& v_prime,
    const at::Tensor& sdf_u,
    const at::Tensor& sdf_v,
    const at::Tensor& body_u,
    const at::Tensor& body_v,
    at::Tensor u0, at::Tensor v0,
    at::Tensor ch, at::Tensor cv,
    const double eps,
    const double rho_f,
    const double dt,
    const double h_grid,
    const int64_t dirty_i0, const int64_t dirty_j0,
    const int64_t dirty_Ai, const int64_t dirty_Aj,
    const int64_t mu0_projection)
{
    const int64_t dirty_vol = dirty_Ai * dirty_Aj;
    if (dirty_vol <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngx = (int)u0.size(0);
    const int Ngy = (int)u0.size(1);

    const int blockSize = 256;
    const int nblocks   = (int)((dirty_vol + blockSize - 1) / blockSize);

    AT_DISPATCH_FLOATING_TYPES(u0.scalar_type(), "bdim_coeff_2d_cuda", [&] {
        bdim_coeff_2d_kernel<scalar_t>
            <<<nblocks, blockSize, 0, stream>>>(
                u_prime.data_ptr<scalar_t>(),
                v_prime.data_ptr<scalar_t>(),
                sdf_u.data_ptr<scalar_t>(),
                sdf_v.data_ptr<scalar_t>(),
                body_u.data_ptr<scalar_t>(),
                body_v.data_ptr<scalar_t>(),
                u0.data_ptr<scalar_t>(),
                v0.data_ptr<scalar_t>(),
                ch.data_ptr<scalar_t>(),
                cv.data_ptr<scalar_t>(),
                (scalar_t)eps,
                (scalar_t)rho_f,
                (scalar_t)dt,
                (scalar_t)(0.5 / h_grid),
                Ngx, Ngy,
                (int)dirty_i0, (int)dirty_j0,
                (int)dirty_Ai, (int)dirty_Aj,
                (int)dirty_vol,
                (int)mu0_projection);
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("bdim_coeff_2d",                       &bdim_coeff_2d_cuda);
    m.impl("streaming_sdf_forces_post_2d",          &streaming_sdf_forces_post_2d_cuda);
    m.impl("apply_bcs_2d",                          &apply_bcs_2d_cuda);
    m.impl("interp_2d",                        &interp_2d_cuda);
}

}  // namespace lilytorch_kernels
