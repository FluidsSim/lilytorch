// =====================================================================
//  streaming_sdf_2d.cu
//
//  CUDA implementations of ``streaming_sdf_min_rho_2d_multi`` and
//  ``streaming_sdf_min_rho_2d_multi``.  Mirrors ``streaming_sdf.cu``
//  line-for-line with the z-axis stripped.  See the matching helpers
//  in ``streaming_sdf_cpu_2d.cpp`` for the algorithmic rationale.
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
#include "packed_key.cuh"

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
// =====================================================================
//  streaming_sdf_min_rho_2d_multi  (Phase C: parallel-B atomicMin64)
//
//  2-D analogue of ``streaming_sdf_min_rho_3d_multi``.  Single launch
//  fanned across ``gridDim.y == B`` with packed ``(s, body_id)``
//  64-bit ``atomicMin`` per union SDF field (cc, u, v).  See
//  ``packed_key.cuh`` and the 3-D implementation for the encoding
//  rationale and the init / decode pipeline.
// =====================================================================

// Initialises key arrays only within the dirty sub-block
// [di0, di0+dAi) x [dj0, dj0+dAj), the union of prev and curr union AABBs.
template <typename scalar_t>
__global__ void streaming_sdf_init_keys_2d_kernel(
    const scalar_t* __restrict__ sdf_cc,
    const scalar_t* __restrict__ sdf_u,
    const scalar_t* __restrict__ sdf_v,
    uint64_t* __restrict__ key_cc,
    uint64_t* __restrict__ key_u,
    uint64_t* __restrict__ key_v,
    const int dirty_vol,
    const int B_sentinel,
    const int di0, const int dj0,
    const int dAi, const int dAj,
    const int Ngy)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= dirty_vol) return;
    const int dj = local % dAj;
    const int di = local / dAj;
    const int g  = (di0 + di) * Ngy + (dj0 + dj);
    key_cc[g] = pack_sdf_body_key(sdf_cc[g], B_sentinel);
    key_u [g] = pack_sdf_body_key(sdf_u [g], B_sentinel);
    key_v [g] = pack_sdf_body_key(sdf_v [g], B_sentinel);
}

// Decode kernel: scatter winning ``(s, body_id)`` back to ``sdf_*`` and
// recompute ``bU/bV[g]`` from the winning body's planar kinematics.
// In 2-D the angular velocity is the scalar ``om = K[10]``.
template <typename scalar_t>
__global__ void streaming_sdf_decode_keys_2d_kernel(
    const uint64_t* __restrict__ key_cc,
    const uint64_t* __restrict__ key_u,
    const uint64_t* __restrict__ key_v,
    const scalar_t* __restrict__ kin,           // [B,11]
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const int Ngx, const int Ngy,
    const int B_sentinel,
    scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ sdf_u,
    scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ bU,
    scalar_t* __restrict__ bV)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int Ngrid = Ngx * Ngy;
    if (g >= Ngrid) return;

    const int j = g % Ngy;
    const int i = g / Ngy;

    const uint64_t kc = key_cc[g];
    const uint64_t ku = key_u [g];
    const uint64_t kv = key_v [g];

    const uint32_t bc = unpack_body_id(kc);
    const uint32_t bu = unpack_body_id(ku);
    const uint32_t bv = unpack_body_id(kv);

    if ((int)bc < B_sentinel) sdf_cc[g] = unpack_sdf<scalar_t>(kc);
    if ((int)bu < B_sentinel) sdf_u [g] = unpack_sdf<scalar_t>(ku);
    if ((int)bv < B_sentinel) sdf_v [g] = unpack_sdf<scalar_t>(kv);

    if ((int)bu < B_sentinel) {
        const scalar_t* K = kin + (int)bu * 11;
        const scalar_t cm_y = K[7];
        const scalar_t lv_x = K[8];
        const scalar_t om   = K[10];
        const scalar_t yc = gy[j];
        bU[g] = lv_x - om * (yc - cm_y);
    }
    if ((int)bv < B_sentinel) {
        const scalar_t* K = kin + (int)bv * 11;
        const scalar_t cm_x = K[6];
        const scalar_t lv_y = K[9];
        const scalar_t om   = K[10];
        const scalar_t xc = gx[i];
        bV[g] = lv_y + om * (xc - cm_x);
    }
}

// =====================================================================
//  streaming_sdf_forces_post_2d (CUDA)
//
//  2-D analogue of ``streaming_sdf_forces_post_3d``.  Per-body grid: walks the
//  AABB cells, re-samples body-local cc-SDF support, evaluates
//  smoothed visc/pres deltas, accumulates 6 float64 channels:
//      [fv_x, fv_y, t_v, fp_x, fp_y, t_p]
//  via shared-memory reduction + atomicAdd into out[b, 0..5].
// =====================================================================

// =====================================================================
//  streaming_sdf_min_rho_2d_multi  (Phase C only)
// =====================================================================

template <typename scalar_t>
__global__ void streaming_sdf_min_rho_2d_multi_kernel(
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
    const scalar_t half_h,
    uint64_t* __restrict__ key_cc,
    uint64_t* __restrict__ key_u,
    uint64_t* __restrict__ key_v,
    const int interp_method)
{
    const int b     = blockIdx.y;
    const int local = blockIdx.x * blockDim.x + threadIdx.x;

    const int Ai = (int)aabb_dim[b*2 + 0];
    const int Aj = (int)aabb_dim[b*2 + 1];
    const int vol = Ai * Aj;
    if (local >= vol) return;

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

    const scalar_t xc = gx[i];
    const scalar_t yc = gy[j];
    const scalar_t dx_w = xc - bp_x;
    const scalar_t dy_w = yc - bp_y;
    const scalar_t bxq = r00 * dx_w + r01 * dy_w;
    const scalar_t byq = r10 * dx_w + r11 * dy_w;

    const scalar_t neg_hh = -half_h;
    const scalar_t du_x = neg_hh * r00;
    const scalar_t du_y = neg_hh * r10;
    const scalar_t dv_x = neg_hh * r01;
    const scalar_t dv_y = neg_hh * r11;

    const scalar_t s_cc = sdf_sample_dispatch_2d(
        interp_method, F, Mx, My, bx0, by0, idx_, idy_, bxq, byq);
    atomicMin((unsigned long long*)&key_cc[g_idx],
              (unsigned long long)pack_sdf_body_key(s_cc, b));

    const scalar_t s_u = sdf_sample_dispatch_2d(
        interp_method, F, Mx, My, bx0, by0, idx_, idy_,
        bxq + du_x, byq + du_y);
    atomicMin((unsigned long long*)&key_u[g_idx],
              (unsigned long long)pack_sdf_body_key(s_u, b));

    const scalar_t s_v = sdf_sample_dispatch_2d(
        interp_method, F, Mx, My, bx0, by0, idx_, idy_,
        bxq + dv_x, byq + dv_y);
    atomicMin((unsigned long long*)&key_v[g_idx],
              (unsigned long long)pack_sdf_body_key(s_v, b));
}

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

// 2-D memory-saving decode kernel: extends the basic 2-D decode by also
// stamping ``winning_rho_cc[g]`` for the body that won the cc-SDF.
template <typename scalar_t>
__global__ void streaming_sdf_decode_keys_rho_2d_kernel(
    const uint64_t* __restrict__ key_cc,
    const uint64_t* __restrict__ key_u,
    const uint64_t* __restrict__ key_v,
    const scalar_t* __restrict__ kin,
    const scalar_t* __restrict__ rho_bodies,
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const int Ngy,
    const int B_sentinel,
    scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ sdf_u,
    scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ bU,
    scalar_t* __restrict__ bV,
    scalar_t* __restrict__ winning_rho_cc,
    const int dirty_vol,
    const int di0, const int dj0,
    const int dAi, const int dAj)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= dirty_vol) return;
    const int dj = local % dAj;
    const int di = local / dAj;
    const int i  = di0 + di;
    const int j  = dj0 + dj;
    const int g  = i * Ngy + j;

    const uint64_t kc = key_cc[g];
    const uint64_t ku = key_u [g];
    const uint64_t kv = key_v [g];

    const uint32_t bc = unpack_body_id(kc);
    const uint32_t bu = unpack_body_id(ku);
    const uint32_t bv = unpack_body_id(kv);

    if ((int)bc < B_sentinel) {
        sdf_cc[g] = unpack_sdf<scalar_t>(kc);
        winning_rho_cc[g] = rho_bodies[(int)bc];
    }
    if ((int)bu < B_sentinel) sdf_u[g] = unpack_sdf<scalar_t>(ku);
    if ((int)bv < B_sentinel) sdf_v[g] = unpack_sdf<scalar_t>(kv);

    if ((int)bu < B_sentinel) {
        const scalar_t* K = kin + (int)bu * 11;
        const scalar_t cm_y = K[7];
        const scalar_t lv_x = K[8];
        const scalar_t om   = K[10];
        const scalar_t yc = gy[j];
        bU[g] = lv_x - om * (yc - cm_y);
    }
    if ((int)bv < B_sentinel) {
        const scalar_t* K = kin + (int)bv * 11;
        const scalar_t cm_x = K[6];
        const scalar_t lv_y = K[9];
        const scalar_t om   = K[10];
        const scalar_t xc = gx[i];
        bV[g] = lv_y + om * (xc - cm_x);
    }
}

void streaming_sdf_min_rho_2d_multi_cuda(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes,
    const at::Tensor& body_meta,
    const at::Tensor& kin,
    const at::Tensor& aabb_lo,
    const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy,
    const double h_grid,
    const int64_t max_vol_per_body,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v,
    at::Tensor body_u, at::Tensor body_v,
    const int64_t interp_method,
    const at::Tensor& rho_bodies,
    at::Tensor winning_rho_cc,
    const int64_t dirty_i0, const int64_t dirty_j0,
    const int64_t dirty_Ai, const int64_t dirty_Aj)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngx = (int)gx.numel();
    const int Ngy = (int)gy.numel();
    const int64_t Ngrid = (int64_t)Ngx * Ngy;

    const int blockSize = (max_vol_per_body <= 128) ? 32
                        : (max_vol_per_body <= 4096) ? 128 : 256;

    auto key_opts = at::TensorOptions()
        .dtype(at::kLong).device(sdf_cc.device());
    auto key_cc_t = at::empty({Ngrid}, key_opts);
    auto key_u_t  = at::empty({Ngrid}, key_opts);
    auto key_v_t  = at::empty({Ngrid}, key_opts);

    const int64_t dirty_vol = dirty_Ai * dirty_Aj;

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_min_rho_2d_multi_cuda", [&] {
        const int initBlock = 256;
        const int initBlocks = (int)((dirty_vol + initBlock - 1) / initBlock);
        streaming_sdf_init_keys_2d_kernel<scalar_t>
            <<<initBlocks, initBlock, 0, stream>>>(
                sdf_cc.data_ptr<scalar_t>(),
                sdf_u .data_ptr<scalar_t>(),
                sdf_v .data_ptr<scalar_t>(),
                (uint64_t*)key_cc_t.data_ptr<int64_t>(),
                (uint64_t*)key_u_t .data_ptr<int64_t>(),
                (uint64_t*)key_v_t .data_ptr<int64_t>(),
                (int)dirty_vol, B,
                (int)dirty_i0, (int)dirty_j0,
                (int)dirty_Ai, (int)dirty_Aj,
                Ngy);

        const int blocksPerBody = (int)((max_vol_per_body + blockSize - 1) / blockSize);
        streaming_sdf_min_rho_2d_multi_kernel<scalar_t>
            <<<dim3(blocksPerBody, B, 1), dim3(blockSize, 1, 1), 0, stream>>>(
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
                (scalar_t)(0.5 * h_grid),
                (uint64_t*)key_cc_t.data_ptr<int64_t>(),
                (uint64_t*)key_u_t .data_ptr<int64_t>(),
                (uint64_t*)key_v_t .data_ptr<int64_t>(),
                (int)interp_method);

        streaming_sdf_decode_keys_rho_2d_kernel<scalar_t>
            <<<initBlocks, initBlock, 0, stream>>>(
                (const uint64_t*)key_cc_t.data_ptr<int64_t>(),
                (const uint64_t*)key_u_t .data_ptr<int64_t>(),
                (const uint64_t*)key_v_t .data_ptr<int64_t>(),
                kin.data_ptr<scalar_t>(),
                rho_bodies.data_ptr<scalar_t>(),
                gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(),
                Ngy, B,
                sdf_cc.data_ptr<scalar_t>(),
                sdf_u .data_ptr<scalar_t>(),
                sdf_v .data_ptr<scalar_t>(),
                body_u.data_ptr<scalar_t>(),
                body_v.data_ptr<scalar_t>(),
                winning_rho_cc.data_ptr<scalar_t>(),
                (int)dirty_vol,
                (int)dirty_i0, (int)dirty_j0,
                (int)dirty_Ai, (int)dirty_Aj);
    });
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
    at::Tensor out)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngx = (int)gx.numel();
    const int Ngy = (int)gy.numel();

    // Adaptive blockSize matching the streaming_sdf_min_rho launcher; CUB's
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
                    out.data_ptr<double>());
        };
        switch (blockSize) {
            case 32:  launch(std::integral_constant<int, 32>{}); break;
            case 128: launch(std::integral_constant<int, 128>{}); break;
            default:  launch(std::integral_constant<int, 256>{}); break;
        }
    });
}

// =====================================================================
//  apply_bcs_2d (CUDA)
//
//  2-D analogue of ``apply_bcs_3d``: writes one ghost line per op
//  (Neumann copy or Dirichlet constant).  See CPU impl for the
//  argument layout.
// =====================================================================

template <typename scalar_t>
__global__ void apply_bcs_2d_kernel(
    scalar_t* __restrict__ u,
    scalar_t* __restrict__ v,
    const int64_t* __restrict__ shapes,
    const int* __restrict__ neu_desc,
    const int N_neu,
    const int* __restrict__ dir_desc,
    const scalar_t* __restrict__ dir_val,
    const int N_dir)
{
    const int op = blockIdx.y;
    const int total = N_neu + N_dir;
    if (op >= total) return;

    const bool is_neu = (op < N_neu);
    int comp, axis, dst_along, src_along;
    scalar_t value = scalar_t(0);

    if (is_neu) {
        comp = neu_desc[op * 3 + 0];
        axis = neu_desc[op * 3 + 1];
        const int side = neu_desc[op * 3 + 2];
        const int sz = (int)shapes[comp * 2 + axis];
        if (side == 0) { dst_along = 0;      src_along = 1; }
        else           { dst_along = sz - 1; src_along = sz - 2; }
    } else {
        const int d = op - N_neu;
        comp = dir_desc[d * 3 + 0];
        axis = dir_desc[d * 3 + 1];
        const int offset = dir_desc[d * 3 + 2];
        const int sz = (int)shapes[comp * 2 + axis];
        dst_along = (offset >= 0) ? offset : (sz + offset);
        src_along = 0;
        value = dir_val[d];
    }

    const int Nx = (int)shapes[comp * 2 + 0];
    const int Ny = (int)shapes[comp * 2 + 1];
    const int dim0_max = (axis == 0) ? Ny : Nx;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim0_max) return;

    scalar_t* base = (comp == 0) ? u : v;

    int64_t dst_lin, src_lin = 0;
    if (axis == 0) {
        dst_lin = (int64_t)dst_along * Ny + i;
        if (is_neu) src_lin = (int64_t)src_along * Ny + i;
    } else {
        dst_lin = (int64_t)i * Ny + dst_along;
        if (is_neu) src_lin = (int64_t)i * Ny + src_along;
    }

    if (is_neu) base[dst_lin] = base[src_lin];
    else        base[dst_lin] = value;
}

void apply_bcs_2d_cuda(
    at::Tensor u, at::Tensor v,
    const at::Tensor& shapes,
    const at::Tensor& neu_desc,
    const at::Tensor& dir_desc,
    const at::Tensor& dir_val,
    const int64_t max_line_dim)
{
    TORCH_CHECK(u.is_contiguous() && v.is_contiguous(),
                "apply_bcs_2d_cuda: u/v must be contiguous");
    TORCH_CHECK(u.scalar_type() == v.scalar_type(),
                "apply_bcs_2d_cuda: u/v must share dtype");
    TORCH_CHECK(shapes.scalar_type() == at::kLong &&
                shapes.dim() == 2 && shapes.size(0) == 2 && shapes.size(1) == 2,
                "apply_bcs_2d_cuda: shapes must be int64[2,2]");

    const int N_neu = (int)neu_desc.size(0);
    const int N_dir = (int)dir_desc.size(0);
    const int total = N_neu + N_dir;
    if (total == 0 || max_line_dim <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blockX = 256;
    const int gridX  = (int)((max_line_dim + blockX - 1) / blockX);

    // All ops launched in a single kernel with blockIdx.y as the op index,
    // matching apply_bcs_3d.  Corner ghost cells (e.g. u[0,0]) are never
    // read by interior stencils, so concurrent writes to corners are harmless.
    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_2d_cuda", [&] {
        apply_bcs_2d_kernel<scalar_t>
            <<<dim3(gridX, total, 1), dim3(blockX, 1, 1), 0, stream>>>(
                u.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                shapes.data_ptr<int64_t>(),
                (N_neu > 0) ? neu_desc.data_ptr<int>() : nullptr, N_neu,
                (N_dir > 0) ? dir_desc.data_ptr<int>() : nullptr,
                (N_dir > 0) ? dir_val.data_ptr<scalar_t>() : nullptr, N_dir);
    });
}

// =====================================================================
//  interpolate_2d: scattered-point bilinear / biquadratic sampling
//
//  One thread per query point.  Calls the same sdf_sample_dispatch_2d
//  device function used by streaming_sdf_min_rho_2d_multi (interp_method 0 =
//  bilinear / "linear", 1 = biquadratic / "quadratic").
// =====================================================================
template <typename scalar_t>
__global__ void interpolate_2d_kernel(
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

void interpolate_2d_cuda(
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

    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "interpolate_2d_cuda", [&] {
        interpolate_2d_kernel<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
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

TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("streaming_sdf_min_rho_2d_multi",        &streaming_sdf_min_rho_2d_multi_cuda);
    m.impl("streaming_sdf_forces_post_2d",          &streaming_sdf_forces_post_2d_cuda);
    m.impl("apply_bcs_2d",                          &apply_bcs_2d_cuda);
    m.impl("interpolate_2d",                        &interpolate_2d_cuda);
}

}  // namespace lilytorch_kernels
