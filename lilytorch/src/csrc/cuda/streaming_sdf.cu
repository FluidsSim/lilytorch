// =====================================================================
//  3-D streaming-SDF support kernels (post-2.4: union path removed;
//  post-CL2: _direct path removed)
//
//  The union-AABB packed-key pipeline (init_keys / min_rho_3d_multi /
//  decode_keys, packed_key.cuh) was deleted in cuda_native_port item 2.4;
//  the Regime-A _direct path was deleted in CL2.  The sole production
//  streaming path is streaming_sdf_regime_b.cu (per-body private buffers
//  + resolve).  This file keeps the shared samplers plus: forces-post
//  readout, bdim_coeff, the fused BC kernel, and scattered-point interp.
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

// =====================================================================
//  CUDA dispatch
// =====================================================================

template <typename scalar_t, int BLOCK_SIZE>
__global__ void streaming_sdf_forces_post_3d_kernel(
    const scalar_t* __restrict__ F_flat,
    const int64_t*  __restrict__ F_offsets,
    const int64_t*  __restrict__ body_shapes,
    const scalar_t* __restrict__ body_meta,
    const scalar_t* __restrict__ kin,
    const int64_t*  __restrict__ aabb_lo,
    const int64_t*  __restrict__ aabb_dim,
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const scalar_t* __restrict__ gz,
    const int Ngx, const int Ngy, const int Ngz,
    const scalar_t* __restrict__ sdf_cc,
    const int interp_method,
    const scalar_t* __restrict__ u_prev,
    const scalar_t* __restrict__ v_prev,
    const scalar_t* __restrict__ w_prev,
    const scalar_t* __restrict__ p_prev,
    const scalar_t* __restrict__ nu_rho_field,
    const int64_t   nu_rho_field_size,
    const scalar_t inv_h,
    const scalar_t eps_body,
    const scalar_t eps_solver,
    const scalar_t h3,
    const int delta_order,
    const int with_pressure,
    double* __restrict__ out)
{
    const int b     = blockIdx.y;
    const int local = blockIdx.x * blockDim.x + threadIdx.x;

    const int Ai = (int)aabb_dim[b*3 + 0];
    const int Aj = (int)aabb_dim[b*3 + 1];
    const int Ak = (int)aabb_dim[b*3 + 2];
    const int vol = Ai * Aj * Ak;

    double acc[12];
#pragma unroll
    for (int c = 0; c < 12; ++c) acc[c] = 0.0;

    if (local < vol) {
        const int di  = local / (Aj * Ak);
        const int rem = local - di * (Aj * Ak);
        const int dj  = rem / Ak;
        const int dk  = rem - dj * Ak;
        const int i0 = (int)aabb_lo[b*3 + 0];
        const int j0 = (int)aabb_lo[b*3 + 1];
        const int k0 = (int)aabb_lo[b*3 + 2];
        const int i  = i0 + di;
        const int j  = j0 + dj;
        const int k  = k0 + dk;
        const int g_idx = (i * Ngy + j) * Ngz + k;

        const scalar_t* F  = F_flat + F_offsets[b];
        const int Mx = (int)body_shapes[b*3 + 0];
        const int My = (int)body_shapes[b*3 + 1];
        const int Mz = (int)body_shapes[b*3 + 2];
        const scalar_t* M  = body_meta + b*10;
        const scalar_t bx0 = M[0], by0 = M[1], bz0 = M[2];
        const scalar_t idx_ = M[6], idy_ = M[7], idz_ = M[8];
        const scalar_t* K  = kin + b*21;
        const scalar_t r00 = K[0],  r01 = K[1],  r02 = K[2];
        const scalar_t r10 = K[3],  r11 = K[4],  r12 = K[5];
        const scalar_t r20 = K[6],  r21 = K[7],  r22 = K[8];
        const scalar_t bp_x = K[9],  bp_y = K[10], bp_z = K[11];
        const scalar_t cm_x = K[12], cm_y = K[13], cm_z = K[14];

        const scalar_t xc = gx[i];
        const scalar_t yc = gy[j];
        const scalar_t zc = gz[k];
        const scalar_t dx_w = xc - bp_x, dy_w = yc - bp_y, dz_w = zc - bp_z;
        const scalar_t bxq = r00*dx_w + r01*dy_w + r02*dz_w;
        const scalar_t byq = r10*dx_w + r11*dy_w + r12*dz_w;
        const scalar_t bzq = r20*dx_w + r21*dy_w + r22*dz_w;

        const scalar_t s_cc_body = sdf_sample_dispatch(
            interp_method, F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
            bxq, byq, bzq);

        const scalar_t band_lo = (eps_solver - eps_body) < (-eps_body)
            ? (eps_solver - eps_body) : (-eps_body);
        const scalar_t band_hi = (eps_solver + eps_body) > (eps_body)
            ? (eps_solver + eps_body) : (eps_body);

        if (s_cc_body > band_lo && s_cc_body < band_hi) {
            const scalar_t nu_rho_val = (nu_rho_field_size == 1)
                ? nu_rho_field[0] : nu_rho_field[g_idx];

#define SDF_AT(ii, jj, kk) (sdf_cc[((ii) * Ngy + (jj)) * Ngz + (kk)])
            scalar_t dsdx_union = 0, dsdy_union = 0, dsdz_union = 0;
            if (Ngx >= 3) {
                if (i == 0) dsdx_union = ((scalar_t)(-3)*SDF_AT(0,j,k) + (scalar_t)4*SDF_AT(1,j,k) - SDF_AT(2,j,k)) * (scalar_t)0.5 * inv_h;
                else if (i == Ngx-1) dsdx_union = ((scalar_t)3*SDF_AT(Ngx-1,j,k) - (scalar_t)4*SDF_AT(Ngx-2,j,k) + SDF_AT(Ngx-3,j,k)) * (scalar_t)0.5 * inv_h;
                else dsdx_union = (SDF_AT(i+1,j,k) - SDF_AT(i-1,j,k)) * (scalar_t)0.5 * inv_h;
            } else if (Ngx == 2) dsdx_union = (SDF_AT(1,j,k) - SDF_AT(0,j,k)) * inv_h;
            if (Ngy >= 3) {
                if (j == 0) dsdy_union = ((scalar_t)(-3)*SDF_AT(i,0,k) + (scalar_t)4*SDF_AT(i,1,k) - SDF_AT(i,2,k)) * (scalar_t)0.5 * inv_h;
                else if (j == Ngy-1) dsdy_union = ((scalar_t)3*SDF_AT(i,Ngy-1,k) - (scalar_t)4*SDF_AT(i,Ngy-2,k) + SDF_AT(i,Ngy-3,k)) * (scalar_t)0.5 * inv_h;
                else dsdy_union = (SDF_AT(i,j+1,k) - SDF_AT(i,j-1,k)) * (scalar_t)0.5 * inv_h;
            } else if (Ngy == 2) dsdy_union = (SDF_AT(i,1,k) - SDF_AT(i,0,k)) * inv_h;
            if (Ngz >= 3) {
                if (k == 0) dsdz_union = ((scalar_t)(-3)*SDF_AT(i,j,0) + (scalar_t)4*SDF_AT(i,j,1) - SDF_AT(i,j,2)) * (scalar_t)0.5 * inv_h;
                else if (k == Ngz-1) dsdz_union = ((scalar_t)3*SDF_AT(i,j,Ngz-1) - (scalar_t)4*SDF_AT(i,j,Ngz-2) + SDF_AT(i,j,Ngz-3)) * (scalar_t)0.5 * inv_h;
                else dsdz_union = (SDF_AT(i,j,k+1) - SDF_AT(i,j,k-1)) * (scalar_t)0.5 * inv_h;
            } else if (Ngz == 2) dsdz_union = (SDF_AT(i,j,1) - SDF_AT(i,j,0)) * inv_h;
            const scalar_t union_norm = sqrt(dsdx_union*dsdx_union + dsdy_union*dsdy_union + dsdz_union*dsdz_union);
#undef SDF_AT
            const scalar_t inv_norm = union_norm > (scalar_t)0 ? (scalar_t)1 / union_norm : (scalar_t)0;
            const scalar_t nx = dsdx_union * inv_norm;
            const scalar_t ny = dsdy_union * inv_norm;
            const scalar_t nz = dsdz_union * inv_norm;

            const int im1 = (i > 0)       ? i-1 : 0;
            const int ip1 = (i+1 < Ngx)   ? i+1 : i;
            const int im2 = (i > 1)       ? i-2 : 0;
            const int ip2 = (i+2 < Ngx)   ? i+2 : (Ngx - 1);
            const int jm1 = (j > 0)       ? j-1 : 0;
            const int jp1 = (j+1 < Ngy)   ? j+1 : j;
            const int jm2 = (j > 1)       ? j-2 : 0;
            const int jp2 = (j+2 < Ngy)   ? j+2 : (Ngy - 1);
            const int km1 = (k > 0)       ? k-1 : 0;
            const int kp1 = (k+1 < Ngz)   ? k+1 : k;
            const int km2 = (k > 1)       ? k-2 : 0;
            const int kp2 = (k+2 < Ngz)   ? k+2 : (Ngz - 1);

            // Normal derivatives: forward diff; backward at upper boundary
            scalar_t dudx;
            if (i + 1 < Ngx) dudx = (u_prev[(ip1*Ngy+j)*Ngz+k] - u_prev[(i*Ngy+j)*Ngz+k]) * inv_h;
            else              dudx = (u_prev[(i*Ngy+j)*Ngz+k]   - u_prev[(im1*Ngy+j)*Ngz+k]) * inv_h;
            scalar_t dvdy;
            if (j + 1 < Ngy) dvdy = (v_prev[(i*Ngy+jp1)*Ngz+k] - v_prev[(i*Ngy+j)*Ngz+k]) * inv_h;
            else              dvdy = (v_prev[(i*Ngy+j)*Ngz+k]   - v_prev[(i*Ngy+jm1)*Ngz+k]) * inv_h;
            scalar_t dwdz;
            if (k + 1 < Ngz) dwdz = (w_prev[(i*Ngy+j)*Ngz+kp1] - w_prev[(i*Ngy+j)*Ngz+k]) * inv_h;
            else              dwdz = (w_prev[(i*Ngy+j)*Ngz+k]   - w_prev[(i*Ngy+j)*Ngz+km1]) * inv_h;

            // dudy: O(h²) one-sided at j boundaries; u staggered in x → CC: 0.5*(u[i,j]+u[i+1,j])
            const scalar_t u_cc_jm2 = (scalar_t)0.5 * (u_prev[(i*Ngy+jm2)*Ngz+k] + u_prev[(ip1*Ngy+jm2)*Ngz+k]);
            const scalar_t u_cc_jm1 = (scalar_t)0.5 * (u_prev[(i*Ngy+jm1)*Ngz+k] + u_prev[(ip1*Ngy+jm1)*Ngz+k]);
            const scalar_t u_cc_j0  = (scalar_t)0.5 * (u_prev[(i*Ngy+j  )*Ngz+k] + u_prev[(ip1*Ngy+j  )*Ngz+k]);
            const scalar_t u_cc_jp1 = (scalar_t)0.5 * (u_prev[(i*Ngy+jp1)*Ngz+k] + u_prev[(ip1*Ngy+jp1)*Ngz+k]);
            const scalar_t u_cc_jp2 = (scalar_t)0.5 * (u_prev[(i*Ngy+jp2)*Ngz+k] + u_prev[(ip1*Ngy+jp2)*Ngz+k]);
            scalar_t dudy;
            if (Ngy >= 3) {
                if (j == 0)          dudy = ((scalar_t)(-3)*u_cc_j0 + (scalar_t)4*u_cc_jp1 - u_cc_jp2) * (scalar_t)0.5 * inv_h;
                else if (j == Ngy-1) dudy = ((scalar_t)3*u_cc_j0 - (scalar_t)4*u_cc_jm1 + u_cc_jm2) * (scalar_t)0.5 * inv_h;
                else                 dudy = (u_cc_jp1 - u_cc_jm1) * (scalar_t)0.5 * inv_h;
            } else { dudy = (u_cc_jp1 - u_cc_jm1) * (scalar_t)0.5 * inv_h; }

            // dudz: O(h²) one-sided at k boundaries
            const scalar_t u_cc_km2 = (scalar_t)0.5 * (u_prev[(i*Ngy+j)*Ngz+km2] + u_prev[(ip1*Ngy+j)*Ngz+km2]);
            const scalar_t u_cc_km1 = (scalar_t)0.5 * (u_prev[(i*Ngy+j)*Ngz+km1] + u_prev[(ip1*Ngy+j)*Ngz+km1]);
            const scalar_t u_cc_k0  = (scalar_t)0.5 * (u_prev[(i*Ngy+j)*Ngz+k  ] + u_prev[(ip1*Ngy+j)*Ngz+k  ]);
            const scalar_t u_cc_kp1 = (scalar_t)0.5 * (u_prev[(i*Ngy+j)*Ngz+kp1] + u_prev[(ip1*Ngy+j)*Ngz+kp1]);
            const scalar_t u_cc_kp2 = (scalar_t)0.5 * (u_prev[(i*Ngy+j)*Ngz+kp2] + u_prev[(ip1*Ngy+j)*Ngz+kp2]);
            scalar_t dudz;
            if (Ngz >= 3) {
                if (k == 0)          dudz = ((scalar_t)(-3)*u_cc_k0 + (scalar_t)4*u_cc_kp1 - u_cc_kp2) * (scalar_t)0.5 * inv_h;
                else if (k == Ngz-1) dudz = ((scalar_t)3*u_cc_k0 - (scalar_t)4*u_cc_km1 + u_cc_km2) * (scalar_t)0.5 * inv_h;
                else                 dudz = (u_cc_kp1 - u_cc_km1) * (scalar_t)0.5 * inv_h;
            } else { dudz = (u_cc_kp1 - u_cc_km1) * (scalar_t)0.5 * inv_h; }

            // dvdx: O(h²) one-sided at i boundaries; v staggered in y → CC: 0.5*(v[i,j]+v[i,j+1])
            const scalar_t v_cc_im2 = (scalar_t)0.5 * (v_prev[(im2*Ngy+j)*Ngz+k] + v_prev[(im2*Ngy+jp1)*Ngz+k]);
            const scalar_t v_cc_im1 = (scalar_t)0.5 * (v_prev[(im1*Ngy+j)*Ngz+k] + v_prev[(im1*Ngy+jp1)*Ngz+k]);
            const scalar_t v_cc_i0  = (scalar_t)0.5 * (v_prev[(i  *Ngy+j)*Ngz+k] + v_prev[(i  *Ngy+jp1)*Ngz+k]);
            const scalar_t v_cc_ip1 = (scalar_t)0.5 * (v_prev[(ip1*Ngy+j)*Ngz+k] + v_prev[(ip1*Ngy+jp1)*Ngz+k]);
            const scalar_t v_cc_ip2 = (scalar_t)0.5 * (v_prev[(ip2*Ngy+j)*Ngz+k] + v_prev[(ip2*Ngy+jp1)*Ngz+k]);
            scalar_t dvdx;
            if (Ngx >= 3) {
                if (i == 0)          dvdx = ((scalar_t)(-3)*v_cc_i0 + (scalar_t)4*v_cc_ip1 - v_cc_ip2) * (scalar_t)0.5 * inv_h;
                else if (i == Ngx-1) dvdx = ((scalar_t)3*v_cc_i0 - (scalar_t)4*v_cc_im1 + v_cc_im2) * (scalar_t)0.5 * inv_h;
                else                 dvdx = (v_cc_ip1 - v_cc_im1) * (scalar_t)0.5 * inv_h;
            } else { dvdx = (v_cc_ip1 - v_cc_im1) * (scalar_t)0.5 * inv_h; }

            // dvdz: O(h²) one-sided at k boundaries
            const scalar_t v_cc_km2 = (scalar_t)0.5 * (v_prev[(i*Ngy+j)*Ngz+km2] + v_prev[(i*Ngy+jp1)*Ngz+km2]);
            const scalar_t v_cc_km1 = (scalar_t)0.5 * (v_prev[(i*Ngy+j)*Ngz+km1] + v_prev[(i*Ngy+jp1)*Ngz+km1]);
            const scalar_t v_cc_k0  = (scalar_t)0.5 * (v_prev[(i*Ngy+j)*Ngz+k  ] + v_prev[(i*Ngy+jp1)*Ngz+k  ]);
            const scalar_t v_cc_kp1 = (scalar_t)0.5 * (v_prev[(i*Ngy+j)*Ngz+kp1] + v_prev[(i*Ngy+jp1)*Ngz+kp1]);
            const scalar_t v_cc_kp2 = (scalar_t)0.5 * (v_prev[(i*Ngy+j)*Ngz+kp2] + v_prev[(i*Ngy+jp1)*Ngz+kp2]);
            scalar_t dvdz;
            if (Ngz >= 3) {
                if (k == 0)          dvdz = ((scalar_t)(-3)*v_cc_k0 + (scalar_t)4*v_cc_kp1 - v_cc_kp2) * (scalar_t)0.5 * inv_h;
                else if (k == Ngz-1) dvdz = ((scalar_t)3*v_cc_k0 - (scalar_t)4*v_cc_km1 + v_cc_km2) * (scalar_t)0.5 * inv_h;
                else                 dvdz = (v_cc_kp1 - v_cc_km1) * (scalar_t)0.5 * inv_h;
            } else { dvdz = (v_cc_kp1 - v_cc_km1) * (scalar_t)0.5 * inv_h; }

            // dwdx: O(h²) one-sided at i boundaries; w staggered in z → CC: 0.5*(w[i,j,k]+w[i,j,k+1])
            const scalar_t w_cc_im2 = (scalar_t)0.5 * (w_prev[(im2*Ngy+j)*Ngz+k] + w_prev[(im2*Ngy+j)*Ngz+kp1]);
            const scalar_t w_cc_im1 = (scalar_t)0.5 * (w_prev[(im1*Ngy+j)*Ngz+k] + w_prev[(im1*Ngy+j)*Ngz+kp1]);
            const scalar_t w_cc_i0  = (scalar_t)0.5 * (w_prev[(i  *Ngy+j)*Ngz+k] + w_prev[(i  *Ngy+j)*Ngz+kp1]);
            const scalar_t w_cc_ip1 = (scalar_t)0.5 * (w_prev[(ip1*Ngy+j)*Ngz+k] + w_prev[(ip1*Ngy+j)*Ngz+kp1]);
            const scalar_t w_cc_ip2 = (scalar_t)0.5 * (w_prev[(ip2*Ngy+j)*Ngz+k] + w_prev[(ip2*Ngy+j)*Ngz+kp1]);
            scalar_t dwdx;
            if (Ngx >= 3) {
                if (i == 0)          dwdx = ((scalar_t)(-3)*w_cc_i0 + (scalar_t)4*w_cc_ip1 - w_cc_ip2) * (scalar_t)0.5 * inv_h;
                else if (i == Ngx-1) dwdx = ((scalar_t)3*w_cc_i0 - (scalar_t)4*w_cc_im1 + w_cc_im2) * (scalar_t)0.5 * inv_h;
                else                 dwdx = (w_cc_ip1 - w_cc_im1) * (scalar_t)0.5 * inv_h;
            } else { dwdx = (w_cc_ip1 - w_cc_im1) * (scalar_t)0.5 * inv_h; }

            // dwdy: O(h²) one-sided at j boundaries
            const scalar_t w_cc_jm2 = (scalar_t)0.5 * (w_prev[(i*Ngy+jm2)*Ngz+k] + w_prev[(i*Ngy+jm2)*Ngz+kp1]);
            const scalar_t w_cc_jm1 = (scalar_t)0.5 * (w_prev[(i*Ngy+jm1)*Ngz+k] + w_prev[(i*Ngy+jm1)*Ngz+kp1]);
            const scalar_t w_cc_j0  = (scalar_t)0.5 * (w_prev[(i*Ngy+j  )*Ngz+k] + w_prev[(i*Ngy+j  )*Ngz+kp1]);
            const scalar_t w_cc_jp1 = (scalar_t)0.5 * (w_prev[(i*Ngy+jp1)*Ngz+k] + w_prev[(i*Ngy+jp1)*Ngz+kp1]);
            const scalar_t w_cc_jp2 = (scalar_t)0.5 * (w_prev[(i*Ngy+jp2)*Ngz+k] + w_prev[(i*Ngy+jp2)*Ngz+kp1]);
            scalar_t dwdy;
            if (Ngy >= 3) {
                if (j == 0)          dwdy = ((scalar_t)(-3)*w_cc_j0 + (scalar_t)4*w_cc_jp1 - w_cc_jp2) * (scalar_t)0.5 * inv_h;
                else if (j == Ngy-1) dwdy = ((scalar_t)3*w_cc_j0 - (scalar_t)4*w_cc_jm1 + w_cc_jm2) * (scalar_t)0.5 * inv_h;
                else                 dwdy = (w_cc_jp1 - w_cc_jm1) * (scalar_t)0.5 * inv_h;
            } else { dwdy = (w_cc_jp1 - w_cc_jm1) * (scalar_t)0.5 * inv_h; }

            const scalar_t xs = nu_rho_val * (2*dudx*nx + (dudy+dvdx)*ny + (dudz+dwdx)*nz);
            const scalar_t ys = nu_rho_val * ((dvdx+dudy)*nx + 2*dvdy*ny + (dvdz+dwdy)*nz);
            const scalar_t zs = nu_rho_val * ((dwdx+dudz)*nx + (dwdy+dvdz)*ny + 2*dwdz*nz);
            const scalar_t p_c  = p_prev[g_idx];
            const scalar_t pxv  = -p_c * nx;
            const scalar_t pyv  = -p_c * ny;
            const scalar_t pzv  = -p_c * nz;
            const scalar_t pi_v = (scalar_t)3.141592653589793;
            const scalar_t inv_2eps = (scalar_t)0.5 / eps_body;
            const scalar_t pi_ov_eb = pi_v / eps_body;
            scalar_t delta_visc = 0, delta_pres = 0;
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
                const scalar_t s_xp = sdf_sample_dispatch(interp_method,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq+r00*h_grid,byq+r10*h_grid,bzq+r20*h_grid);
                const scalar_t s_xm = sdf_sample_dispatch(interp_method,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq-r00*h_grid,byq-r10*h_grid,bzq-r20*h_grid);
                const scalar_t s_yp = sdf_sample_dispatch(interp_method,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq+r01*h_grid,byq+r11*h_grid,bzq+r21*h_grid);
                const scalar_t s_ym = sdf_sample_dispatch(interp_method,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq-r01*h_grid,byq-r11*h_grid,bzq-r21*h_grid);
                const scalar_t s_zp = sdf_sample_dispatch(interp_method,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq+r02*h_grid,byq+r12*h_grid,bzq+r22*h_grid);
                const scalar_t s_zm = sdf_sample_dispatch(interp_method,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq-r02*h_grid,byq-r12*h_grid,bzq-r22*h_grid);
                scalar_t grad_mag = sqrt(((s_xp-s_xm)*(s_xp-s_xm) + (s_yp-s_ym)*(s_yp-s_ym) + (s_zp-s_zm)*(s_zp-s_zm)) * (scalar_t)0.25 * inv_h * inv_h);
                if (grad_mag < (scalar_t)1e-3) grad_mag = (scalar_t)1e-3;
                const scalar_t inv_grad = (scalar_t)1.0 / grad_mag;
                delta_visc *= inv_grad;
                delta_pres *= inv_grad;
            }
            const scalar_t arm_x = xc - cm_x, arm_y = yc - cm_y, arm_z = zc - cm_z;
            const double fv_x = (double)(xs * delta_visc), fv_y = (double)(ys * delta_visc), fv_z = (double)(zs * delta_visc);
            const double fp_x = (double)(pxv * delta_pres), fp_y = (double)(pyv * delta_pres), fp_z = (double)(pzv * delta_pres);
            acc[0]=fv_x; acc[1]=fv_y; acc[2]=fv_z;
            acc[3]=(double)arm_y*fv_z-(double)arm_z*fv_y;
            acc[4]=(double)arm_z*fv_x-(double)arm_x*fv_z;
            acc[5]=(double)arm_x*fv_y-(double)arm_y*fv_x;
            acc[6]=fp_x; acc[7]=fp_y; acc[8]=fp_z;
            acc[9]=(double)arm_y*fp_z-(double)arm_z*fp_y;
            acc[10]=(double)arm_z*fp_x-(double)arm_x*fp_z;
            acc[11]=(double)arm_x*fp_y-(double)arm_y*fp_x;
        }
    }

    // Block-wide sum via CUB BlockReduce (warp shuffles + 1 shmem slot per
    // warp).  Replaces the previous 24 KB / block manual reduction; new
    // shmem footprint is the BlockReduce TempStorage (~256 B) regardless
    // of channel count, lifting the 2-block-per-SM occupancy cap.
    using BlockReduceD = cub::BlockReduce<double, BLOCK_SIZE>;
    __shared__ typename BlockReduceD::TempStorage tmp;
    const double h3_d = (double)h3;
#pragma unroll
    for (int c = 0; c < 12; ++c) {
        const double s = BlockReduceD(tmp).Sum(acc[c]);
        if (threadIdx.x == 0)
            atomicAdd(&out[b*12 + c], s * h3_d);
        // CUB requires a sync between successive uses of the same
        // TempStorage in the same kernel invocation.
        __syncthreads();
    }
}

// ----------------------------------------------------------------------
//  Partial-Heaviside (∂H) pressure-force readout  --  deltaH submethod
//
//  Pressure force density is taken from the UNION SDF (one closed surface,
//  no internal inter-link seams) as  f_i = -p ∂_iH_ε(φ_union), where H is the
//  smooth Heaviside (exact antiderivative of the cosine δ the n·δ path uses)
//  and ∂_iH is its edge_order=2 discrete gradient -- matching
//  TwoPhaseSolver._heaviside_smooth + torch.gradient exactly.  The union force
//  density is split back to the individual bodies by a softmin partition of
//  unity  w_b = softmax_b(-φ_b/τ)  (Σ_b w_b ≡ 1 over the bodies covering the
//  cell), so Σ_b F_b == union force while each body keeps its own force/torque.
//
//  Cell-major launch over the union AABB; each thread loops over all B bodies
//  twice (Z normaliser, then distribute).  B is small (link counts), so the
//  re-sampling is cheap and avoids a separate grid-sized partition buffer.
// ----------------------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ scalar_t heaviside_smooth_dev(scalar_t phi, scalar_t inv_eps) {
    const scalar_t pi = (scalar_t)3.141592653589793;
    scalar_t x = phi * inv_eps;
    x = x < (scalar_t)-1 ? (scalar_t)-1 : (x > (scalar_t)1 ? (scalar_t)1 : x);
    return (scalar_t)0.5 * ((scalar_t)1 + x + sin(pi * x) / pi);
}

template <typename scalar_t>
__global__ void forces_post_deltaH_pressure_3d_kernel(
    const scalar_t* __restrict__ F_flat,
    const int64_t*  __restrict__ F_offsets,
    const int64_t*  __restrict__ body_shapes,
    const scalar_t* __restrict__ body_meta,
    const scalar_t* __restrict__ kin,
    const int64_t*  __restrict__ aabb_lo,
    const int64_t*  __restrict__ aabb_dim,
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const scalar_t* __restrict__ gz,
    const int Ngx, const int Ngy, const int Ngz,
    const scalar_t* __restrict__ sdf_cc,
    const int interp_method,
    const scalar_t* __restrict__ p_prev,
    const scalar_t inv_h,
    const scalar_t inv_eps,
    const scalar_t inv_tau,
    const scalar_t h3,
    const int B,
    const int uli0, const int ulj0, const int ulk0,
    const int ULi, const int ULj, const int ULk,
    double* __restrict__ out)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const int uvol = ULi * ULj * ULk;
    if (local >= uvol) return;

    const int di  = local / (ULj * ULk);
    const int rem = local - di * (ULj * ULk);
    const int dj  = rem / ULk;
    const int dk  = rem - dj * ULk;
    const int i = uli0 + di;
    const int j = ulj0 + dj;
    const int k = ulk0 + dk;
    if (i < 0 || i >= Ngx || j < 0 || j >= Ngy || k < 0 || k >= Ngz) return;

#define SDF_AT(ii, jj, kk) (sdf_cc[(((int64_t)(ii)) * Ngy + (jj)) * Ngz + (kk)])
#define HV_AT(ii, jj, kk) heaviside_smooth_dev<scalar_t>(SDF_AT(ii, jj, kk), inv_eps)
    // edge_order=2 discrete gradient of the union smooth Heaviside
    scalar_t gHx = 0, gHy = 0, gHz = 0;
    if (Ngx >= 3) {
        if (i == 0) gHx = ((scalar_t)(-3)*HV_AT(0,j,k) + (scalar_t)4*HV_AT(1,j,k) - HV_AT(2,j,k)) * (scalar_t)0.5 * inv_h;
        else if (i == Ngx-1) gHx = ((scalar_t)3*HV_AT(Ngx-1,j,k) - (scalar_t)4*HV_AT(Ngx-2,j,k) + HV_AT(Ngx-3,j,k)) * (scalar_t)0.5 * inv_h;
        else gHx = (HV_AT(i+1,j,k) - HV_AT(i-1,j,k)) * (scalar_t)0.5 * inv_h;
    } else if (Ngx == 2) gHx = (HV_AT(1,j,k) - HV_AT(0,j,k)) * inv_h;
    if (Ngy >= 3) {
        if (j == 0) gHy = ((scalar_t)(-3)*HV_AT(i,0,k) + (scalar_t)4*HV_AT(i,1,k) - HV_AT(i,2,k)) * (scalar_t)0.5 * inv_h;
        else if (j == Ngy-1) gHy = ((scalar_t)3*HV_AT(i,Ngy-1,k) - (scalar_t)4*HV_AT(i,Ngy-2,k) + HV_AT(i,Ngy-3,k)) * (scalar_t)0.5 * inv_h;
        else gHy = (HV_AT(i,j+1,k) - HV_AT(i,j-1,k)) * (scalar_t)0.5 * inv_h;
    } else if (Ngy == 2) gHy = (HV_AT(i,1,k) - HV_AT(i,0,k)) * inv_h;
    if (Ngz >= 3) {
        if (k == 0) gHz = ((scalar_t)(-3)*HV_AT(i,j,0) + (scalar_t)4*HV_AT(i,j,1) - HV_AT(i,j,2)) * (scalar_t)0.5 * inv_h;
        else if (k == Ngz-1) gHz = ((scalar_t)3*HV_AT(i,j,Ngz-1) - (scalar_t)4*HV_AT(i,j,Ngz-2) + HV_AT(i,j,Ngz-3)) * (scalar_t)0.5 * inv_h;
        else gHz = (HV_AT(i,j,k+1) - HV_AT(i,j,k-1)) * (scalar_t)0.5 * inv_h;
    } else if (Ngz == 2) gHz = (HV_AT(i,j,1) - HV_AT(i,j,0)) * inv_h;
#undef HV_AT
#undef SDF_AT

    if (gHx == (scalar_t)0 && gHy == (scalar_t)0 && gHz == (scalar_t)0) return;

    const int64_t g = ((int64_t)i * Ngy + j) * Ngz + k;
    const scalar_t p_c = p_prev[g];
    const scalar_t fdx = -p_c * gHx;
    const scalar_t fdy = -p_c * gHy;
    const scalar_t fdz = -p_c * gHz;
    const scalar_t xc = gx[i], yc = gy[j], zc = gz[k];
    const scalar_t sdfu = sdf_cc[g];   // = min_b φ_b (numerically-stable shift)

    // ---- per-body softmin weight  w_b = exp(-(φ_b-φ_union)/τ) / Z ----
    scalar_t Z = 0;
    for (int b = 0; b < B; ++b) {
        const int i0 = (int)aabb_lo[b*3+0], j0 = (int)aabb_lo[b*3+1], k0 = (int)aabb_lo[b*3+2];
        const int Ai = (int)aabb_dim[b*3+0], Aj = (int)aabb_dim[b*3+1], Ak = (int)aabb_dim[b*3+2];
        if (i < i0 || i >= i0+Ai || j < j0 || j >= j0+Aj || k < k0 || k >= k0+Ak) continue;
        const scalar_t* F = F_flat + F_offsets[b];
        const int Mx = (int)body_shapes[b*3+0], My = (int)body_shapes[b*3+1], Mz = (int)body_shapes[b*3+2];
        const scalar_t* M = body_meta + b*10;
        const scalar_t* K = kin + b*21;
        const scalar_t dx_w = xc - K[9], dy_w = yc - K[10], dz_w = zc - K[11];
        const scalar_t bxq = K[0]*dx_w + K[1]*dy_w + K[2]*dz_w;
        const scalar_t byq = K[3]*dx_w + K[4]*dy_w + K[5]*dz_w;
        const scalar_t bzq = K[6]*dx_w + K[7]*dy_w + K[8]*dz_w;
        const scalar_t s_b = sdf_sample_dispatch(interp_method, F, Mx, My, Mz,
            M[0], M[1], M[2], M[6], M[7], M[8], bxq, byq, bzq);
        Z += exp(-(s_b - sdfu) * inv_tau);
    }
    if (Z <= (scalar_t)0) return;
    const scalar_t inv_Z = (scalar_t)1 / Z;
    const double h3_d = (double)h3;

    for (int b = 0; b < B; ++b) {
        const int i0 = (int)aabb_lo[b*3+0], j0 = (int)aabb_lo[b*3+1], k0 = (int)aabb_lo[b*3+2];
        const int Ai = (int)aabb_dim[b*3+0], Aj = (int)aabb_dim[b*3+1], Ak = (int)aabb_dim[b*3+2];
        if (i < i0 || i >= i0+Ai || j < j0 || j >= j0+Aj || k < k0 || k >= k0+Ak) continue;
        const scalar_t* F = F_flat + F_offsets[b];
        const int Mx = (int)body_shapes[b*3+0], My = (int)body_shapes[b*3+1], Mz = (int)body_shapes[b*3+2];
        const scalar_t* M = body_meta + b*10;
        const scalar_t* K = kin + b*21;
        const scalar_t dx_w = xc - K[9], dy_w = yc - K[10], dz_w = zc - K[11];
        const scalar_t bxq = K[0]*dx_w + K[1]*dy_w + K[2]*dz_w;
        const scalar_t byq = K[3]*dx_w + K[4]*dy_w + K[5]*dz_w;
        const scalar_t bzq = K[6]*dx_w + K[7]*dy_w + K[8]*dz_w;
        const scalar_t s_b = sdf_sample_dispatch(interp_method, F, Mx, My, Mz,
            M[0], M[1], M[2], M[6], M[7], M[8], bxq, byq, bzq);
        const scalar_t wb = exp(-(s_b - sdfu) * inv_tau) * inv_Z;
        const scalar_t fbx = wb * fdx, fby = wb * fdy, fbz = wb * fdz;
        const scalar_t ax = xc - K[12], ay = yc - K[13], az = zc - K[14];
        atomicAdd(&out[b*12 + 6], (double)fbx * h3_d);
        atomicAdd(&out[b*12 + 7], (double)fby * h3_d);
        atomicAdd(&out[b*12 + 8], (double)fbz * h3_d);
        atomicAdd(&out[b*12 + 9],  ((double)ay*(double)fbz - (double)az*(double)fby) * h3_d);
        atomicAdd(&out[b*12 + 10], ((double)az*(double)fbx - (double)ax*(double)fbz) * h3_d);
        atomicAdd(&out[b*12 + 11], ((double)ax*(double)fby - (double)ay*(double)fbx) * h3_d);
    }
}

// =====================================================================
//  Kernel B (Phase I): fused BDIM2 + variable-density Poisson coefficients
//
//  One thread per cell (i,j,k) inside the dirty AABB.  For each face
//  axis a ∈ {u,v,w}:
//     1) Reads phi = sdf_<a>[i,j,k] and its 6 face-SDF neighbours.
//     2) Computes mu0, mu1 in registers (smoothed Heaviside / delta from
//        ``_mu_normals_batched``).  Never materialized in global memory.
//     3) Computes the FD unit normal n = grad(phi)/|grad(phi)| in
//        registers (central differences from the 6 neighbours).
//     4) Computes the BDIM2 normal derivative
//           nd = n . grad(phi_prime - body_vel)
//        using central differences on (phi_prime - body_vel).
//     5) Writes the BDIM2 meta-equation result
//           u0[i,j,k] = mu0*(u' - body) + body + mu1*nd
//        and the BDIM2 Poisson coefficient (Weymouth & Yue; body enters via
//        mu0 only, NOT its density)
//           c[i,j,k]  = dt * mu0 / rho_f.
//
//  No race conditions: phi_prime tensors are distinct allocations from
//  u0/v0/w0; each thread writes only its own cell.
// =====================================================================
template <typename scalar_t>
__device__ __forceinline__ void bdim_one_axis_3d(
    const scalar_t* __restrict__ phi_prime,
    const scalar_t* __restrict__ sdf,
    const scalar_t* __restrict__ body,
    const scalar_t eps,
    const scalar_t rho_f,
    const scalar_t dt,
    const scalar_t inv_2h,
    const int Ngx, const int Ngy, const int Ngz,
    const int i, const int j, const int k,
    scalar_t* __restrict__ phi_out,
    scalar_t* __restrict__ c_out,
    const int c_stride_i,
    const int c_stride_j,
    const int c_hi_i,
    const int c_hi_j,
    const int c_hi_k,
    const int mu0_proj)
{
    const int stride_i = Ngy * Ngz;
    const int stride_j = Ngz;
    const int g  = i * stride_i + j * stride_j + k;

    // Neighbour indices, clamped to the grid extent so a body at the
    // grid edge cannot read out of bounds.
    const int im = (i > 0)         ? (i - 1) : 0;
    const int ip = (i < Ngx - 1)   ? (i + 1) : Ngx - 1;
    const int jm = (j > 0)         ? (j - 1) : 0;
    const int jp = (j < Ngy - 1)   ? (j + 1) : Ngy - 1;
    const int km = (k > 0)         ? (k - 1) : 0;
    const int kp = (k < Ngz - 1)   ? (k + 1) : Ngz - 1;

    const int g_im = im * stride_i + j  * stride_j + k;
    const int g_ip = ip * stride_i + j  * stride_j + k;
    const int g_jm = i  * stride_i + jm * stride_j + k;
    const int g_jp = i  * stride_i + jp * stride_j + k;
    const int g_km = i  * stride_i + j  * stride_j + km;
    const int g_kp = i  * stride_i + j  * stride_j + kp;

    // ------------------------------------------------------------------
    // mu0, mu1 from the smoothed Heaviside / delta of phi.  Matches
    // _mu_normals_batched in body.py exactly.
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // Unit normal n = grad(phi) / |grad(phi)|, central differences.
    // Boundary cells (clamped above) fall back to one-sided differences.
    // ------------------------------------------------------------------
    scalar_t nx = (sdf[g_ip] - sdf[g_im]) * inv_2h;
    scalar_t ny = (sdf[g_jp] - sdf[g_jm]) * inv_2h;
    scalar_t nz = (sdf[g_kp] - sdf[g_km]) * inv_2h;
    const scalar_t nn = sqrt(nx*nx + ny*ny + nz*nz);
    if (nn > scalar_t(0)) {
        const scalar_t inv_nn = scalar_t(1) / nn;
        nx *= inv_nn;
        ny *= inv_nn;
        nz *= inv_nn;
    }

    // ------------------------------------------------------------------
    // BDIM2 update.  Normal derivative is taken of (phi_prime - body)
    // to match _bdim_meta in solver.py.  Gradient is zero at the grid
    // boundary cells (matches compute_dpd*).
    // ------------------------------------------------------------------
    const scalar_t b_c    = body[g];
    const scalar_t pp_c   = phi_prime[g];
    const scalar_t diff_c = pp_c - b_c;

    scalar_t ddx, ddy, ddz;
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
    if (k > 0 && k < Ngz - 1) {
        ddz = ((phi_prime[g_kp] - body[g_kp]) -
               (phi_prime[g_km] - body[g_km])) * inv_2h;
    } else {
        ddz = scalar_t(0);
    }
    const scalar_t nd = nx * ddx + ny * ddy + nz * ddz;

    phi_out[g] = mu0 * diff_c + b_c + mu1 * nd;
    // Write coefficient into the face-grid-shaped c_out tensor.
    // The face grid excludes ghost cells: valid range is [1, c_hi_*] in
    // padded coordinates; the face-grid index is (i-1, j-1, k-1).
    if (i >= 1 && j >= 1 && k >= 1 && i <= c_hi_i && j <= c_hi_j && k <= c_hi_k) {
        // BDIM2 coefficient dt*mu0/rho_fluid (Weymouth & Yue): the body enters
        // via mu0 only, NOT its density.  mu0_proj == 0 → plain dt/rho (no mu0
        // numerator), non-degenerate for multibody.
        c_out[(i - 1) * c_stride_i + (j - 1) * c_stride_j + (k - 1)] =
            (mu0_proj ? dt * mu0 : dt) / rho_f;
    }
}

template <typename scalar_t>
__global__ void bdim_coeff_3d_kernel(
    const scalar_t* __restrict__ u_prime,
    const scalar_t* __restrict__ v_prime,
    const scalar_t* __restrict__ w_prime,
    const scalar_t* __restrict__ sdf_u,
    const scalar_t* __restrict__ sdf_v,
    const scalar_t* __restrict__ sdf_w,
    const scalar_t* __restrict__ body_u,
    const scalar_t* __restrict__ body_v,
    const scalar_t* __restrict__ body_w,
    scalar_t* __restrict__ u0,
    scalar_t* __restrict__ v0,
    scalar_t* __restrict__ w0,
    scalar_t* __restrict__ ch,
    scalar_t* __restrict__ cv,
    scalar_t* __restrict__ cw,
    const scalar_t eps,
    const scalar_t rho_f,
    const scalar_t dt,
    const scalar_t inv_2h,
    const int Ngx, const int Ngy, const int Ngz,
    const int di0, const int dj0, const int dk0,
    const int dAi, const int dAj, const int dAk,
    const int dirty_vol,
    const int mu0_proj)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= dirty_vol) return;
    const int dk = local % dAk;
    const int rem = local / dAk;
    const int dj = rem % dAj;
    const int di = rem / dAj;
    const int i = di0 + di;
    const int j = dj0 + dj;
    const int k = dk0 + dk;

    // ch: x-face grid (Ngx-1, Ngy-2, Ngz-2), strides ((Ngy-2)*(Ngz-2), Ngz-2, 1)
    bdim_one_axis_3d<scalar_t>(
        u_prime, sdf_u, body_u,
        eps, rho_f, dt, inv_2h,
        Ngx, Ngy, Ngz, i, j, k, u0, ch,
        (Ngy - 2) * (Ngz - 2), (Ngz - 2),
        Ngx - 1, Ngy - 2, Ngz - 2, mu0_proj);
    // cv: y-face grid (Ngx-2, Ngy-1, Ngz-2), strides ((Ngy-1)*(Ngz-2), Ngz-2, 1)
    bdim_one_axis_3d<scalar_t>(
        v_prime, sdf_v, body_v,
        eps, rho_f, dt, inv_2h,
        Ngx, Ngy, Ngz, i, j, k, v0, cv,
        (Ngy - 1) * (Ngz - 2), (Ngz - 2),
        Ngx - 2, Ngy - 1, Ngz - 2, mu0_proj);
    // cw: z-face grid (Ngx-2, Ngy-2, Ngz-1), strides ((Ngy-2)*(Ngz-1), Ngz-1, 1)
    bdim_one_axis_3d<scalar_t>(
        w_prime, sdf_w, body_w,
        eps, rho_f, dt, inv_2h,
        Ngx, Ngy, Ngz, i, j, k, w0, cw,
        (Ngy - 2) * (Ngz - 1), (Ngz - 1),
        Ngx - 2, Ngy - 2, Ngz - 1, mu0_proj);
}

void bdim_coeff_3d_cuda(
    const at::Tensor& u_prime,
    const at::Tensor& v_prime,
    const at::Tensor& w_prime,
    const at::Tensor& sdf_u,
    const at::Tensor& sdf_v,
    const at::Tensor& sdf_w,
    const at::Tensor& body_u,
    const at::Tensor& body_v,
    const at::Tensor& body_w,
    at::Tensor u0, at::Tensor v0, at::Tensor w0,
    at::Tensor ch, at::Tensor cv, at::Tensor cw,
    const double eps,
    const double rho_f,
    const double dt,
    const double h_grid,
    const int64_t dirty_i0, const int64_t dirty_j0, const int64_t dirty_k0,
    const int64_t dirty_Ai, const int64_t dirty_Aj, const int64_t dirty_Ak,
    const int64_t mu0_projection)
{
    const int64_t dirty_vol = dirty_Ai * dirty_Aj * dirty_Ak;
    if (dirty_vol <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngx = (int)u0.size(0);
    const int Ngy = (int)u0.size(1);
    const int Ngz = (int)u0.size(2);

    const int blockSize = 256;
    const int nblocks   = (int)((dirty_vol + blockSize - 1) / blockSize);

    AT_DISPATCH_FLOATING_TYPES(u0.scalar_type(), "bdim_coeff_3d_cuda", [&] {
        bdim_coeff_3d_kernel<scalar_t>
            <<<nblocks, blockSize, 0, stream>>>(
                u_prime.data_ptr<scalar_t>(),
                v_prime.data_ptr<scalar_t>(),
                w_prime.data_ptr<scalar_t>(),
                sdf_u.data_ptr<scalar_t>(),
                sdf_v.data_ptr<scalar_t>(),
                sdf_w.data_ptr<scalar_t>(),
                body_u.data_ptr<scalar_t>(),
                body_v.data_ptr<scalar_t>(),
                body_w.data_ptr<scalar_t>(),
                u0.data_ptr<scalar_t>(),
                v0.data_ptr<scalar_t>(),
                w0.data_ptr<scalar_t>(),
                ch.data_ptr<scalar_t>(),
                cv.data_ptr<scalar_t>(),
                cw.data_ptr<scalar_t>(),
                (scalar_t)eps,
                (scalar_t)rho_f,
                (scalar_t)dt,
                (scalar_t)(0.5 / h_grid),
                Ngx, Ngy, Ngz,
                (int)dirty_i0, (int)dirty_j0, (int)dirty_k0,
                (int)dirty_Ai, (int)dirty_Aj, (int)dirty_Ak,
                (int)dirty_vol,
                (int)mu0_projection);
    });
}

void streaming_sdf_forces_post_3d_cuda(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes,
    const at::Tensor& body_meta,
    const at::Tensor& kin,
    const at::Tensor& aabb_lo,
    const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy, const at::Tensor& gz,
    const double h_grid,
    const int64_t max_vol_per_body,
    const at::Tensor& sdf_cc,
    const int64_t interp_method,
    const at::Tensor& u, const at::Tensor& v,
    const at::Tensor& w, const at::Tensor& p,
    const at::Tensor& nu_rho_field,
    const double eps_body,
    const double eps_solver,
    const double h3,
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
    const int Ngz = (int)gz.numel();
    const int with_pressure = (force_submethod == 0) ? 1 : 0;

    // Adaptive blockSize matching the streaming_sdf_min_rho launcher: small
    // bodies launch with smaller blocks so the masked-off threads at the
    // tail don't cap occupancy unnecessarily.  CUB's BlockReduce is a
    // compile-time-templated type, so we fan out to one of the three
    // configured sizes.
    const int blockSize = (max_vol_per_body <= 128)  ? 32
                        : (max_vol_per_body <= 4096) ? 128
                                                     : 256;
    const int nblocks = (int)((max_vol_per_body + blockSize - 1) / blockSize);

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_forces_post_3d_cuda", [&] {
        auto launch = [&](auto block_size_ic) {
            constexpr int BS = decltype(block_size_ic)::value;
            streaming_sdf_forces_post_3d_kernel<scalar_t, BS>
                <<<dim3(nblocks, B, 1), dim3(BS, 1, 1), 0, stream>>>(
                    F_flat.data_ptr<scalar_t>(), F_offsets.data_ptr<int64_t>(),
                    body_shapes.data_ptr<int64_t>(), body_meta.data_ptr<scalar_t>(),
                    kin.data_ptr<scalar_t>(), aabb_lo.data_ptr<int64_t>(),
                    aabb_dim.data_ptr<int64_t>(), gx.data_ptr<scalar_t>(),
                    gy.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(), Ngx, Ngy, Ngz,
                    sdf_cc.data_ptr<scalar_t>(), (int)interp_method,
                    u.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                    w.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(),
                    nu_rho_field.data_ptr<scalar_t>(), (int64_t)nu_rho_field.numel(),
                    (scalar_t)(1.0 / h_grid), (scalar_t)eps_body,
                    (scalar_t)eps_solver, (scalar_t)h3, (int)delta_order,
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
        // deltaH: second pass fills the pressure force/torque from the
        // union-∂H density distributed by the softmin partition of unity.
        // Launch over the union AABB (union of all per-body AABBs + halo),
        // computed on host from the tiny aabb_lo/aabb_dim tensors.
        auto lo_c  = aabb_lo.to(at::kCPU);
        auto dim_c = aabb_dim.to(at::kCPU);
        const int64_t* loh  = lo_c.data_ptr<int64_t>();
        const int64_t* dimh = dim_c.data_ptr<int64_t>();
        int ulo[3] = {Ngx, Ngy, Ngz};
        int uhi[3] = {0, 0, 0};
        for (int b = 0; b < B; ++b) {
            for (int d = 0; d < 3; ++d) {
                const int a0 = (int)loh[b*3+d];
                const int a1 = a0 + (int)dimh[b*3+d];
                if (a0 < ulo[d]) ulo[d] = a0;
                if (a1 > uhi[d]) uhi[d] = a1;
            }
        }
        const int Ng[3] = {Ngx, Ngy, Ngz};
        const int halo = 2;
        for (int d = 0; d < 3; ++d) {
            ulo[d] = ulo[d] - halo; if (ulo[d] < 0) ulo[d] = 0;
            uhi[d] = uhi[d] + halo; if (uhi[d] > Ng[d]) uhi[d] = Ng[d];
        }
        const int ULi = uhi[0] - ulo[0];
        const int ULj = uhi[1] - ulo[1];
        const int ULk = uhi[2] - ulo[2];
        const int64_t uvol = (int64_t)ULi * ULj * ULk;
        if (ULi > 0 && ULj > 0 && ULk > 0) {
            const double tau = (ph_tau > 0.0) ? ph_tau : 1e-9;
            const int bs = 256;
            const int nb = (int)((uvol + bs - 1) / bs);
            AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "forces_post_deltaH_pressure_3d_cuda", [&] {
                forces_post_deltaH_pressure_3d_kernel<scalar_t>
                    <<<dim3(nb, 1, 1), dim3(bs, 1, 1), 0, stream>>>(
                        F_flat.data_ptr<scalar_t>(), F_offsets.data_ptr<int64_t>(),
                        body_shapes.data_ptr<int64_t>(), body_meta.data_ptr<scalar_t>(),
                        kin.data_ptr<scalar_t>(), aabb_lo.data_ptr<int64_t>(),
                        aabb_dim.data_ptr<int64_t>(), gx.data_ptr<scalar_t>(),
                        gy.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(), Ngx, Ngy, Ngz,
                        sdf_cc.data_ptr<scalar_t>(), (int)interp_method,
                        p.data_ptr<scalar_t>(),
                        (scalar_t)(1.0 / h_grid), (scalar_t)(1.0 / eps_body),
                        (scalar_t)(1.0 / tau), (scalar_t)h3, B,
                        ulo[0], ulo[1], ulo[2], ULi, ULj, ULk,
                        out.data_ptr<double>());
            });
        }
    }
}

// =====================================================================
//  apply_bcs_3d  (Phase H: combined boundary-condition writes)
//
//  Replaces the Python loop in AdvDiffSolver.set_BCs that issues 18
//  ghost-face slice copies (Neumann) plus a handful of Dirichlet
//  overwrites per call.  All ops of one kind are packed into a single
//  kernel launch, eliminating per-op dispatch overhead.
//
//  One launch per op KIND (Neumann → Dirichlet → reflective).  Within a
//  launch the ops of that stage run concurrently, and the write-ownership
//  rule in bc_ops.h keeps edge/corner ghosts race-free and deterministic.
//  See that header for the full argument.
// =====================================================================

template <typename scalar_t>
__global__ void apply_bcs_3d_kernel(
    scalar_t* __restrict__ u,
    scalar_t* __restrict__ v,
    scalar_t* __restrict__ w,
    const int64_t* __restrict__ shapes,
    const int kind,
    const int* __restrict__ desc,
    const scalar_t* __restrict__ vals,   // null for Neumann
    const int nops)
{
    using namespace lilytorch_kernels::bcs;

    const int op = blockIdx.z;
    if (op >= nops) return;

    int comp, axis, dst_along, src_along;
    bc_decode(kind, desc, op, shapes, /*ndim=*/3, comp, axis, dst_along, src_along);

    const int Nx = (int)shapes[comp * 3 + 0];
    const int Ny = (int)shapes[comp * 3 + 1];
    const int Nz = (int)shapes[comp * 3 + 2];

    int dim0_max, dim1_max;
    if      (axis == 0) { dim0_max = Ny; dim1_max = Nz; }
    else if (axis == 1) { dim0_max = Nx; dim1_max = Nz; }
    else                { dim0_max = Nx; dim1_max = Ny; }

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= dim0_max || j >= dim1_max) return;

    // Destination cell, in full 3-D coordinates.
    int c[3];
    if      (axis == 0) { c[0] = dst_along; c[1] = i;         c[2] = j;         }
    else if (axis == 1) { c[0] = i;         c[1] = dst_along; c[2] = j;         }
    else                { c[0] = i;         c[1] = j;         c[2] = dst_along; }

    int s[3];
    if (!bc_own_and_source(kind, desc, nops, op, shapes, /*ndim=*/3,
                           comp, axis, src_along, c, s))
        return;                                   // a lower-indexed op owns it

    scalar_t* base = (comp == 0) ? u : (comp == 1 ? v : w);
    const int64_t s1 = (int64_t)Ny * Nz;
    const int64_t s2 = (int64_t)Nz;

    const int64_t dst_lin = (int64_t)c[0] * s1 + (int64_t)c[1] * s2 + c[2];
    const int64_t src_lin = (int64_t)s[0] * s1 + (int64_t)s[1] * s2 + s[2];

    if      (kind == BC_KIND_NEUMANN)   base[dst_lin] = base[src_lin];
    else if (kind == BC_KIND_DIRICHLET) base[dst_lin] = vals[op];
    else base[dst_lin] = scalar_t(2) * vals[op] - base[src_lin];
}

void apply_bcs_3d_cuda(
    at::Tensor u, at::Tensor v, at::Tensor w,
    const at::Tensor& shapes,
    const at::Tensor& neu_desc,
    const at::Tensor& dir_desc,
    const at::Tensor& dir_val,
    const at::Tensor& ref_desc,
    const at::Tensor& ref_val,
    const int64_t max_dim0,
    const int64_t max_dim1)
{
    TORCH_CHECK(u.is_cuda() && v.is_cuda() && w.is_cuda(),
        "apply_bcs_3d: u/v/w must be CUDA tensors");
    TORCH_CHECK(u.is_contiguous() && v.is_contiguous() && w.is_contiguous(),
        "apply_bcs_3d: u/v/w must be contiguous");
    TORCH_CHECK(u.scalar_type() == v.scalar_type() &&
                u.scalar_type() == w.scalar_type(),
        "apply_bcs_3d: u/v/w must share dtype");
    TORCH_CHECK(shapes.scalar_type() == at::kLong &&
                shapes.dim() == 2 && shapes.size(0) == 3 && shapes.size(1) == 3,
        "apply_bcs_3d: shapes must be int64[3,3]");
    TORCH_CHECK(neu_desc.scalar_type() == at::kInt && neu_desc.dim() == 2 &&
                neu_desc.size(1) == 3,
        "apply_bcs_3d: neu_desc must be int32[N,3]");
    TORCH_CHECK(dir_desc.scalar_type() == at::kInt && dir_desc.dim() == 2 &&
                dir_desc.size(1) == 3,
        "apply_bcs_3d: dir_desc must be int32[N,3]");
    TORCH_CHECK(ref_desc.scalar_type() == at::kInt && ref_desc.dim() == 2 &&
                ref_desc.size(1) == 4,
        "apply_bcs_3d: ref_desc must be int32[N,4]");

    const int N_neu = (int)neu_desc.size(0);
    const int N_dir = (int)dir_desc.size(0);
    const int N_ref = (int)ref_desc.size(0);
    if (N_neu + N_dir + N_ref == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int TILE = 16;
    // Use a rectangular (blocks_x × blocks_y) grid: blocks_x covers the
    // i-dimension (max_dim0) and blocks_y covers the j-dimension (max_dim1).
    // This avoids the (max_plane_dim)² square launch that wastes ~75 % of
    // thread blocks when max_dim0 >> max_dim1 (e.g. Nx-long y/z faces on a
    // 4:1:1 grid where max_dim0 = Nx but max_dim1 = Nz = Nx/4).
    const int blocks_x = (int)((max_dim0 + TILE - 1) / TILE);
    const int blocks_y = (int)((max_dim1 + TILE - 1) / TILE);
    const dim3 block(TILE, TILE, 1);

    // One launch per op kind, in the order the eager reference applies them:
    //   Neumann → Dirichlet → reflective.
    // The stage boundaries are what make the cross-kind overlaps well defined
    // (a reflective op reads the adjacent cell AFTER any Dirichlet wall write
    // to it; a Neumann op reads it BEFORE — as on CPU).  Within a stage,
    // bc_own_and_source() picks a single writer per cell.
    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_3d_cuda", [&] {
        using namespace lilytorch_kernels::bcs;
        auto launch = [&](int kind, const at::Tensor& desc,
                          const scalar_t* vals, int nops) {
            if (nops == 0) return;
            const dim3 grid((unsigned)blocks_x, (unsigned)blocks_y, (unsigned)nops);
            apply_bcs_3d_kernel<scalar_t><<<grid, block, 0, stream>>>(
                u.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
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
    m.impl("bdim_coeff_3d",                &bdim_coeff_3d_cuda);
    m.impl("streaming_sdf_forces_post_3d", &streaming_sdf_forces_post_3d_cuda);
    m.impl("apply_bcs_3d", &apply_bcs_3d_cuda);
    m.impl("interp_3d", &interp_3d_cuda);
}

}  // namespace lilytorch_kernels
