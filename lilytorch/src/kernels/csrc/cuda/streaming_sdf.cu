// =====================================================================
//  streaming_sdf_min_rho_3d_multi
//
//  One-body combined kernel for BDIM SDF / face-velocity update.  For each
//  cell (i,j,k) inside a body's AABB on the fluid grid, this kernel:
//     1) Computes world coords at cc + 3 face locations (u/v/w-stagger).
//     2) Transforms each into the body-local frame using R_T and body_pos.
//     3) Trilinearly samples the body SDF table (border / nearest fill).
//     4) Compare-swaps each value into the running-min full-grid fields
//        (sdf_val_cc, sdf_val_u, sdf_val_v, sdf_val_w).
//     5) When a face SDF wins the min, writes the rigid-body face
//        velocity into body_u / body_v / body_w.
//     6) Stores the cc SDF into a per-body sparse output buffer for
//        downstream force integration.
//
//  Each cell is touched once per launch -> no atomics needed.  Bodies
//  are processed sequentially on the same CUDA stream from Python.
//
//  All small per-step scalars (R_T, body_pos, com_pos, lin_vel, ang_vel)
//  are passed as float[] from Python -- the kinematics already live
//  CPU-side from FARMS cython, so no device-side copy is required.
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <c10/util/ArrayRef.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "packed_key.cuh"

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

// =====================================================================
//  streaming_sdf_min_rho_3d_multi  (Phase C: parallel-B atomicMin64)
//
//  Single Python op call handles ALL bodies in a single launch fanned
//  across ``gridDim.y == B``.  Per-cell, per-field 64-bit atomicMin
//  on packed ``(s, body_id)`` keys eliminates the multi-field
//  compare-swap race that previously forced a host-side sequential
//  per-body for-loop.  See ``packed_key.cuh`` for the encoding
//  rationale.
//
//  Pipeline (all on the same CUDA stream):
//    1. ``streaming_sdf_init_keys_3d_kernel`` — pack the existing
//       ``sdf_*[g]`` values into the per-field key arrays with
//       sentinel ``body_id = B``.
//    2. ``streaming_sdf_min_rho_3d_multi_kernel`` (this kernel) — fanned
//       across B. Each thread does 4 ``atomicMin`` (cc, u, v, w) on
//       its body's contribution.
//    3. ``streaming_sdf_decode_keys_rho_3d_kernel`` — for each cell, if
//       any body won this field, write back the decoded SDF to
//       ``sdf_*[g]`` and recompute ``bU/bV/bW[g]`` from the winning
//       body's kinematics.
// =====================================================================

template <typename scalar_t>
__global__ void streaming_sdf_init_keys_3d_kernel(
    const scalar_t* __restrict__ sdf_cc,
    const scalar_t* __restrict__ sdf_u,
    const scalar_t* __restrict__ sdf_v,
    const scalar_t* __restrict__ sdf_w,
    uint64_t* __restrict__ key_cc,
    uint64_t* __restrict__ key_u,
    uint64_t* __restrict__ key_v,
    uint64_t* __restrict__ key_w,
    const int Ngrid,
    const int B_sentinel)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= Ngrid) return;
    key_cc[g] = pack_sdf_body_key(sdf_cc[g], B_sentinel);
    key_u [g] = pack_sdf_body_key(sdf_u [g], B_sentinel);
    key_v [g] = pack_sdf_body_key(sdf_v [g], B_sentinel);
    key_w [g] = pack_sdf_body_key(sdf_w [g], B_sentinel);
}

// Decode kernel: scatter winning ``(s, body_id)`` back to ``sdf_*`` and
// recompute the linked face velocities ``bU/bV/bW`` from the winning
// body's kinematics.  When the sentinel ``body_id == B`` survived in
// a key (no body touched that cell), ``sdf_*[g]`` and ``bU/bV/bW[g]``
// are left untouched (the init pass already encoded the prior value).
template <typename scalar_t>
__global__ void streaming_sdf_decode_keys_3d_kernel(
    const uint64_t* __restrict__ key_cc,
    const uint64_t* __restrict__ key_u,
    const uint64_t* __restrict__ key_v,
    const uint64_t* __restrict__ key_w,
    const scalar_t* __restrict__ kin,           // [B,21]
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const scalar_t* __restrict__ gz,
    const int Ngx, const int Ngy, const int Ngz,
    const int B_sentinel,
    scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ sdf_u,
    scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ sdf_w,
    scalar_t* __restrict__ bU,
    scalar_t* __restrict__ bV,
    scalar_t* __restrict__ bW)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int Ngrid = Ngx * Ngy * Ngz;
    if (g >= Ngrid) return;

    const int k    = g % Ngz;
    const int rem  = g / Ngz;
    const int j    = rem % Ngy;
    const int i    = rem / Ngy;

    const uint64_t kc = key_cc[g];
    const uint64_t ku = key_u [g];
    const uint64_t kv = key_v [g];
    const uint64_t kw = key_w [g];

    const uint32_t bc = unpack_body_id(kc);
    const uint32_t bu = unpack_body_id(ku);
    const uint32_t bv = unpack_body_id(kv);
    const uint32_t bw = unpack_body_id(kw);

    if ((int)bc < B_sentinel) sdf_cc[g] = unpack_sdf<scalar_t>(kc);
    if ((int)bu < B_sentinel) sdf_u [g] = unpack_sdf<scalar_t>(ku);
    if ((int)bv < B_sentinel) sdf_v [g] = unpack_sdf<scalar_t>(kv);
    if ((int)bw < B_sentinel) sdf_w [g] = unpack_sdf<scalar_t>(kw);

    // Recompute ``bU/bV/bW`` for the winning bodies of u/v/w faces.
    // Face velocity uses the rigid-body kinematics:
    //   bU = lv_x + av_y*(zc - cm_z) - av_z*(yc - cm_y)
    //   bV = lv_y + av_z*(xc - cm_x) - av_x*(zc - cm_z)
    //   bW = lv_z + av_x*(yc - cm_y) - av_y*(xc - cm_x)
    if ((int)bu < B_sentinel) {
        const scalar_t* K = kin + (int)bu * 21;
        const scalar_t cm_y = K[13], cm_z = K[14];
        const scalar_t lv_x = K[15];
        const scalar_t av_y = K[19], av_z = K[20];
        const scalar_t yc = gy[j];
        const scalar_t zc = gz[k];
        bU[g] = lv_x + av_y*(zc - cm_z) - av_z*(yc - cm_y);
    }
    if ((int)bv < B_sentinel) {
        const scalar_t* K = kin + (int)bv * 21;
        const scalar_t cm_x = K[12], cm_z = K[14];
        const scalar_t lv_y = K[16];
        const scalar_t av_x = K[18], av_z = K[20];
        const scalar_t xc = gx[i];
        const scalar_t zc = gz[k];
        bV[g] = lv_y + av_z*(xc - cm_x) - av_x*(zc - cm_z);
    }
    if ((int)bw < B_sentinel) {
        const scalar_t* K = kin + (int)bw * 21;
        const scalar_t cm_x = K[12], cm_y = K[13];
        const scalar_t lv_z = K[17];
        const scalar_t av_x = K[18], av_y = K[19];
        const scalar_t xc = gx[i];
        const scalar_t yc = gy[j];
        bW[g] = lv_z + av_x*(yc - cm_y) - av_y*(xc - cm_x);
    }
}

// =====================================================================
//  streaming_sdf_min_rho_3d_multi  (Phase C only)
// =====================================================================

template <typename scalar_t>
__global__ void streaming_sdf_min_rho_3d_multi_kernel(
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
    const scalar_t half_h,
    uint64_t* __restrict__ key_cc,
    uint64_t* __restrict__ key_u,
    uint64_t* __restrict__ key_v,
    uint64_t* __restrict__ key_w,
    const int interp_method)
{
    const int b     = blockIdx.y;
    const int local = blockIdx.x * blockDim.x + threadIdx.x;

    const int Ai = (int)aabb_dim[b*3 + 0];
    const int Aj = (int)aabb_dim[b*3 + 1];
    const int Ak = (int)aabb_dim[b*3 + 2];
    const int vol = Ai * Aj * Ak;
    if (local >= vol) return;

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
    const scalar_t bp_x = K[9], bp_y = K[10], bp_z = K[11];

    const scalar_t xc = gx[i];
    const scalar_t yc = gy[j];
    const scalar_t zc = gz[k];
    const scalar_t dx_w = xc - bp_x, dy_w = yc - bp_y, dz_w = zc - bp_z;
    const scalar_t bxq = r00*dx_w + r01*dy_w + r02*dz_w;
    const scalar_t byq = r10*dx_w + r11*dy_w + r12*dz_w;
    const scalar_t bzq = r20*dx_w + r21*dy_w + r22*dz_w;

    const scalar_t neg_hh = -half_h;
    const scalar_t du_x = neg_hh*r00, du_y = neg_hh*r10, du_z = neg_hh*r20;
    const scalar_t dv_x = neg_hh*r01, dv_y = neg_hh*r11, dv_z = neg_hh*r21;
    const scalar_t dw_x = neg_hh*r02, dw_y = neg_hh*r12, dw_z = neg_hh*r22;

    const scalar_t s_cc = sdf_sample_dispatch(
        interp_method, F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
        bxq, byq, bzq);
    atomicMin((unsigned long long*)&key_cc[g_idx],
              (unsigned long long)pack_sdf_body_key(s_cc, b));

    const scalar_t s_u = sdf_sample_dispatch(
        interp_method, F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
        bxq+du_x, byq+du_y, bzq+du_z);
    atomicMin((unsigned long long*)&key_u[g_idx],
              (unsigned long long)pack_sdf_body_key(s_u, b));

    const scalar_t s_v = sdf_sample_dispatch(
        interp_method, F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
        bxq+dv_x, byq+dv_y, bzq+dv_z);
    atomicMin((unsigned long long*)&key_v[g_idx],
              (unsigned long long)pack_sdf_body_key(s_v, b));

    const scalar_t s_w = sdf_sample_dispatch(
        interp_method, F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
        bxq+dw_x, byq+dw_y, bzq+dw_z);
    atomicMin((unsigned long long*)&key_w[g_idx],
              (unsigned long long)pack_sdf_body_key(s_w, b));
}

template <typename scalar_t>
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

    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const double h3_d = (double)h3;
#pragma unroll
    for (int c = 0; c < 12; ++c) sdata[c * bdim + tid] = acc[c];
    __syncthreads();
    for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int c = 0; c < 12; ++c) sdata[c * bdim + tid] += sdata[c * bdim + tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
#pragma unroll
        for (int c = 0; c < 12; ++c) atomicAdd(&out[b*12 + c], sdata[c * bdim] * h3_d);
    }
}

// Decode kernel for the memory-saving 3-D pipeline: extends the basic 3-D
// decode by also stamping ``winning_rho_cc[g] = rho_bodies[bc]`` for
// the body that won the cc-SDF at each cell.  When the sentinel
// survived (no body touched the cell), all outputs are left untouched
// (the init pass already encoded the prior value of ``sdf_*[g]``;
// ``winning_rho_cc`` is pre-filled with ``rho_fluid`` by the caller).
template <typename scalar_t>
__global__ void streaming_sdf_decode_keys_rho_3d_kernel(
    const uint64_t* __restrict__ key_cc,
    const uint64_t* __restrict__ key_u,
    const uint64_t* __restrict__ key_v,
    const uint64_t* __restrict__ key_w,
    const scalar_t* __restrict__ kin,           // [B,21]
    const scalar_t* __restrict__ rho_bodies,    // [B]
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const scalar_t* __restrict__ gz,
    const int Ngx, const int Ngy, const int Ngz,
    const int B_sentinel,
    scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ sdf_u,
    scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ sdf_w,
    scalar_t* __restrict__ bU,
    scalar_t* __restrict__ bV,
    scalar_t* __restrict__ bW,
    scalar_t* __restrict__ winning_rho_cc)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int Ngrid = Ngx * Ngy * Ngz;
    if (g >= Ngrid) return;

    const int k    = g % Ngz;
    const int rem  = g / Ngz;
    const int j    = rem % Ngy;
    const int i    = rem / Ngy;

    const uint64_t kc = key_cc[g];
    const uint64_t ku = key_u [g];
    const uint64_t kv = key_v [g];
    const uint64_t kw = key_w [g];

    const uint32_t bc = unpack_body_id(kc);
    const uint32_t bu = unpack_body_id(ku);
    const uint32_t bv = unpack_body_id(kv);
    const uint32_t bw = unpack_body_id(kw);

    if ((int)bc < B_sentinel) {
        sdf_cc[g] = unpack_sdf<scalar_t>(kc);
        winning_rho_cc[g] = rho_bodies[(int)bc];
    }
    if ((int)bu < B_sentinel) sdf_u[g] = unpack_sdf<scalar_t>(ku);
    if ((int)bv < B_sentinel) sdf_v[g] = unpack_sdf<scalar_t>(kv);
    if ((int)bw < B_sentinel) sdf_w[g] = unpack_sdf<scalar_t>(kw);

    if ((int)bu < B_sentinel) {
        const scalar_t* K = kin + (int)bu * 21;
        const scalar_t cm_y = K[13], cm_z = K[14];
        const scalar_t lv_x = K[15];
        const scalar_t av_y = K[19], av_z = K[20];
        const scalar_t yc = gy[j];
        const scalar_t zc = gz[k];
        bU[g] = lv_x + av_y*(zc - cm_z) - av_z*(yc - cm_y);
    }
    if ((int)bv < B_sentinel) {
        const scalar_t* K = kin + (int)bv * 21;
        const scalar_t cm_x = K[12], cm_z = K[14];
        const scalar_t lv_y = K[16];
        const scalar_t av_x = K[18], av_z = K[20];
        const scalar_t xc = gx[i];
        const scalar_t zc = gz[k];
        bV[g] = lv_y + av_z*(xc - cm_x) - av_x*(zc - cm_z);
    }
    if ((int)bw < B_sentinel) {
        const scalar_t* K = kin + (int)bw * 21;
        const scalar_t cm_x = K[12], cm_y = K[13];
        const scalar_t lv_z = K[17];
        const scalar_t av_x = K[18], av_y = K[19];
        const scalar_t xc = gx[i];
        const scalar_t yc = gy[j];
        bW[g] = lv_z + av_x*(yc - cm_y) - av_y*(xc - cm_x);
    }
}

void streaming_sdf_min_rho_3d_multi_cuda(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes,
    const at::Tensor& body_meta,
    const at::Tensor& kin,
    const at::Tensor& aabb_lo,
    const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy, const at::Tensor& gz,
    const double h_grid,
    const int64_t max_vol_per_body,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v, at::Tensor sdf_w,
    at::Tensor body_u, at::Tensor body_v, at::Tensor body_w,
    const int64_t interp_method,
    const at::Tensor& rho_bodies,
    at::Tensor winning_rho_cc)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngx = (int)gx.numel();
    const int Ngy = (int)gy.numel();
    const int Ngz = (int)gz.numel();
    const int64_t Ngrid = (int64_t)Ngx * Ngy * Ngz;
    const int blockSize = (max_vol_per_body <= 128) ? 32
                        : (max_vol_per_body <= 4096) ? 128 : 256;

    auto key_opts = at::TensorOptions().dtype(at::kLong).device(sdf_cc.device());
    auto key_cc_t = at::empty({Ngrid}, key_opts);
    auto key_u_t  = at::empty({Ngrid}, key_opts);
    auto key_v_t  = at::empty({Ngrid}, key_opts);
    auto key_w_t  = at::empty({Ngrid}, key_opts);

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_min_rho_3d_multi_cuda", [&] {
        const int initBlock = 256;
        const int initBlocks = (int)((Ngrid + initBlock - 1) / initBlock);
        streaming_sdf_init_keys_3d_kernel<scalar_t>
            <<<initBlocks, initBlock, 0, stream>>>(
                sdf_cc.data_ptr<scalar_t>(), sdf_u.data_ptr<scalar_t>(),
                sdf_v.data_ptr<scalar_t>(), sdf_w.data_ptr<scalar_t>(),
                (uint64_t*)key_cc_t.data_ptr<int64_t>(),
                (uint64_t*)key_u_t.data_ptr<int64_t>(),
                (uint64_t*)key_v_t.data_ptr<int64_t>(),
                (uint64_t*)key_w_t.data_ptr<int64_t>(),
                (int)Ngrid, B);

        const int blocksPerBody = (int)((max_vol_per_body + blockSize - 1) / blockSize);
        streaming_sdf_min_rho_3d_multi_kernel<scalar_t>
            <<<dim3(blocksPerBody, B, 1), dim3(blockSize, 1, 1), 0, stream>>>(
                F_flat.data_ptr<scalar_t>(), F_offsets.data_ptr<int64_t>(),
                body_shapes.data_ptr<int64_t>(), body_meta.data_ptr<scalar_t>(),
                kin.data_ptr<scalar_t>(), aabb_lo.data_ptr<int64_t>(),
                aabb_dim.data_ptr<int64_t>(), gx.data_ptr<scalar_t>(),
                gy.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(), Ngx, Ngy, Ngz,
                (scalar_t)(0.5 * h_grid),
                (uint64_t*)key_cc_t.data_ptr<int64_t>(),
                (uint64_t*)key_u_t.data_ptr<int64_t>(),
                (uint64_t*)key_v_t.data_ptr<int64_t>(),
                (uint64_t*)key_w_t.data_ptr<int64_t>(),
                (int)interp_method);

        streaming_sdf_decode_keys_rho_3d_kernel<scalar_t>
            <<<initBlocks, initBlock, 0, stream>>>(
                (const uint64_t*)key_cc_t.data_ptr<int64_t>(),
                (const uint64_t*)key_u_t.data_ptr<int64_t>(),
                (const uint64_t*)key_v_t.data_ptr<int64_t>(),
                (const uint64_t*)key_w_t.data_ptr<int64_t>(),
                kin.data_ptr<scalar_t>(), rho_bodies.data_ptr<scalar_t>(),
                gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(),
                Ngx, Ngy, Ngz, B,
                sdf_cc.data_ptr<scalar_t>(), sdf_u.data_ptr<scalar_t>(),
                sdf_v.data_ptr<scalar_t>(), sdf_w.data_ptr<scalar_t>(),
                body_u.data_ptr<scalar_t>(), body_v.data_ptr<scalar_t>(),
                body_w.data_ptr<scalar_t>(), winning_rho_cc.data_ptr<scalar_t>());
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
    at::Tensor out)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngx = (int)gx.numel();
    const int Ngy = (int)gy.numel();
    const int Ngz = (int)gz.numel();
    const int blockSize = 256;
    const int nblocks = (int)((max_vol_per_body + blockSize - 1) / blockSize);
    const size_t shmem = (size_t)blockSize * 12 * sizeof(double);

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_forces_post_3d_cuda", [&] {
        streaming_sdf_forces_post_3d_kernel<scalar_t>
            <<<dim3(nblocks, B, 1), dim3(blockSize, 1, 1), shmem, stream>>>(
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
                out.data_ptr<double>());
    });
}

// =====================================================================
//  apply_bcs_3d  (Phase H: combined boundary-condition writes)
//
//  Replaces the Python loop in AdvDiffSolver.set_BCs that issues 18
//  ghost-face slice copies (Neumann) plus a handful of Dirichlet
//  overwrites per call.  All operations are packed into a single
//  kernel launch, eliminating per-op dispatch overhead.
// =====================================================================

template <typename scalar_t>
__global__ void apply_bcs_3d_kernel(
    scalar_t* __restrict__ u,
    scalar_t* __restrict__ v,
    scalar_t* __restrict__ w,
    const int64_t* __restrict__ shapes,
    const int* __restrict__ neu_desc,
    const int N_neu,
    const int* __restrict__ dir_desc,
    const scalar_t* __restrict__ dir_val,
    const int N_dir)
{
    const int op = blockIdx.z;
    const int total = N_neu + N_dir;
    if (op >= total) return;

    const bool is_neu = (op < N_neu);
    int comp, axis, dst_along, src_along;
    scalar_t value = scalar_t(0);

    if (is_neu) {
        comp = neu_desc[op * 3 + 0];
        axis = neu_desc[op * 3 + 1];
        const int side = neu_desc[op * 3 + 2];
        const int sz = (int)shapes[comp * 3 + axis];
        if (side == 0) { dst_along = 0;      src_along = 1; }
        else           { dst_along = sz - 1; src_along = sz - 2; }
    } else {
        const int d = op - N_neu;
        comp = dir_desc[d * 3 + 0];
        axis = dir_desc[d * 3 + 1];
        const int offset = dir_desc[d * 3 + 2];
        const int sz = (int)shapes[comp * 3 + axis];
        dst_along = (offset >= 0) ? offset : (sz + offset);
        src_along = 0;
        value = dir_val[d];
    }

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

    scalar_t* base = (comp == 0) ? u : (comp == 1 ? v : w);
    const int64_t s1 = (int64_t)Ny * Nz;
    const int64_t s2 = (int64_t)Nz;

    int64_t dst_lin, src_lin = 0;
    if (axis == 0) {
        dst_lin = (int64_t)dst_along * s1 + (int64_t)i * s2 + j;
        if (is_neu) src_lin = (int64_t)src_along * s1 + (int64_t)i * s2 + j;
    } else if (axis == 1) {
        dst_lin = (int64_t)i * s1 + (int64_t)dst_along * s2 + j;
        if (is_neu) src_lin = (int64_t)i * s1 + (int64_t)src_along * s2 + j;
    } else {
        dst_lin = (int64_t)i * s1 + (int64_t)j * s2 + dst_along;
        if (is_neu) src_lin = (int64_t)i * s1 + (int64_t)j * s2 + src_along;
    }

    if (is_neu) base[dst_lin] = base[src_lin];
    else        base[dst_lin] = value;
}

void apply_bcs_3d_cuda(
    at::Tensor u, at::Tensor v, at::Tensor w,
    const at::Tensor& shapes,
    const at::Tensor& neu_desc,
    const at::Tensor& dir_desc,
    const at::Tensor& dir_val,
    const int64_t max_plane_dim)
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

    const int N_neu = (int)neu_desc.size(0);
    const int N_dir = (int)dir_desc.size(0);
    const int total = N_neu + N_dir;
    if (total == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int TILE = 16;
    const int blocks = (int)((max_plane_dim + TILE - 1) / TILE);
    const dim3 grid((unsigned)blocks, (unsigned)blocks, (unsigned)total);
    const dim3 block(TILE, TILE, 1);

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_3d_cuda", [&] {
        apply_bcs_3d_kernel<scalar_t><<<grid, block, 0, stream>>>(
            u.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            shapes.data_ptr<int64_t>(),
            neu_desc.data_ptr<int>(),
            N_neu,
            dir_desc.data_ptr<int>(),
            dir_val.data_ptr<scalar_t>(),
            N_dir);
    });
}


// =====================================================================
//  interpolate_3d: scattered-point trilinear / triquadratic sampling
//
//  One thread per query point.  Calls the same sdf_sample_dispatch
//  device function used by streaming_sdf_min_rho_3d_multi (interp_method 0 =
//  trilinear / "linear", 1 = triquadratic / "quadratic").
// =====================================================================
template <typename scalar_t>
__global__ void interpolate_3d_kernel(
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

void interpolate_3d_cuda(
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

    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "interpolate_3d_cuda", [&] {
        interpolate_3d_kernel<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
            F.contiguous().data_ptr<scalar_t>(),
            xq.contiguous().to(F.scalar_type()).data_ptr<scalar_t>(),
            yq.contiguous().to(F.scalar_type()).data_ptr<scalar_t>(),
            zq.contiguous().to(F.scalar_type()).data_ptr<scalar_t>(),
            N,
            (int)Mx, (int)My, (int)Mz,
            (scalar_t)bx0, (scalar_t)by0, (scalar_t)bz0,
            (scalar_t)inv_dx, (scalar_t)inv_dy, (scalar_t)inv_dz,
            (int)interp_method,
            G.data_ptr<scalar_t>());
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("streaming_sdf_min_rho_3d_multi", &streaming_sdf_min_rho_3d_multi_cuda);
    m.impl("streaming_sdf_forces_post_3d", &streaming_sdf_forces_post_3d_cuda);
    m.impl("apply_bcs_3d", &apply_bcs_3d_cuda);
    m.impl("interpolate_3d", &interpolate_3d_cuda);
}

}  // namespace lilytorch_kernels
