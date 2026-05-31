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
#include <cub/block/block_reduce.cuh>
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

// Initialises key arrays only within the dirty sub-block
// [di0, di0+dAi) x [dj0, dj0+dAj) x [dk0, dk0+dAk), which is the union
// of the previous and current union AABBs.  Cells outside this region
// retain their previous-step SDF values (still correct and far).
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
    const int dirty_vol,
    const int B_sentinel,
    const int di0, const int dj0, const int dk0,
    const int dAi, const int dAj, const int dAk,
    const int Ngy,  const int Ngz)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= dirty_vol) return;
    const int dk = local % dAk;
    const int rem = local / dAk;
    const int dj = rem % dAj;
    const int di = rem / dAj;
    // Global flat index for reading SDF tensors (full-grid layout).
    const int g  = (di0 + di) * Ngy * Ngz + (dj0 + dj) * Ngz + (dk0 + dk);
    // AABB-local flat index for writing key arrays (dirty_vol-sized buffer).
    key_cc[local] = pack_sdf_body_key(sdf_cc[g], B_sentinel);
    key_u [local] = pack_sdf_body_key(sdf_u [g], B_sentinel);
    key_v [local] = pack_sdf_body_key(sdf_v [g], B_sentinel);
    key_w [local] = pack_sdf_body_key(sdf_w [g], B_sentinel);
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
    const int interp_method,
    // Dirty AABB origin and stride for AABB-local key indexing.
    // key arrays are sized dirty_Ai*dirty_Aj*dirty_Ak (not full Ngrid).
    const int dirty_i0, const int dirty_j0, const int dirty_k0,
    const int dirty_Aj, const int dirty_Ak,
    // Smooth velocity-blend accumulators (dirty-vol sized, AABB-local).
    // Active when blend_eps > 0: instead of the running-min winner taking
    // the face velocity, every body contributes  w_i * v_i  with
    // w_i = sigmoid(-s_i/blend_eps), and the decode pass divides by Σ w_i.
    // This replaces the hard velocity switch at inter-link seams with a
    // continuous blend (the SDF/geometry still uses the running-min keys).
    scalar_t* __restrict__ num_u, scalar_t* __restrict__ num_v,
    scalar_t* __restrict__ num_w,
    scalar_t* __restrict__ den_u, scalar_t* __restrict__ den_v,
    scalar_t* __restrict__ den_w,
    const scalar_t blend_eps)
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
    // AABB-local flat index into the dirty_vol-sized key buffers.
    const int g_local = ((i - dirty_i0) * dirty_Aj + (j - dirty_j0)) * dirty_Ak + (k - dirty_k0);

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
    atomicMin((unsigned long long*)&key_cc[g_local],
              (unsigned long long)pack_sdf_body_key(s_cc, b));

    const scalar_t s_u = sdf_sample_dispatch(
        interp_method, F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
        bxq+du_x, byq+du_y, bzq+du_z);
    atomicMin((unsigned long long*)&key_u[g_local],
              (unsigned long long)pack_sdf_body_key(s_u, b));

    const scalar_t s_v = sdf_sample_dispatch(
        interp_method, F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
        bxq+dv_x, byq+dv_y, bzq+dv_z);
    atomicMin((unsigned long long*)&key_v[g_local],
              (unsigned long long)pack_sdf_body_key(s_v, b));

    const scalar_t s_w = sdf_sample_dispatch(
        interp_method, F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
        bxq+dw_x, byq+dw_y, bzq+dw_z);
    atomicMin((unsigned long long*)&key_w[g_local],
              (unsigned long long)pack_sdf_body_key(s_w, b));

    // ---- smooth velocity blend: accumulate Σ w_i v_i and Σ w_i ----
    if (blend_eps > scalar_t(0)) {
        // This body's rigid face velocities (same formulas as the decode
        // pass; u/v/w each independent of their own stagger direction).
        const scalar_t cm_x = K[12], cm_y = K[13], cm_z = K[14];
        const scalar_t lv_x = K[15], lv_y = K[16], lv_z = K[17];
        const scalar_t av_x = K[18], av_y = K[19], av_z = K[20];
        const scalar_t vU = lv_x + av_y*(zc - cm_z) - av_z*(yc - cm_y);
        const scalar_t vV = lv_y + av_z*(xc - cm_x) - av_x*(zc - cm_z);
        const scalar_t vW = lv_z + av_x*(yc - cm_y) - av_y*(xc - cm_x);
        // w_i = sigmoid(-s/eps): ~1 inside body i, 0.5 on its surface, →0
        // a band eps outside.  Evaluated per stagger from that face's SDF.
        const scalar_t wU = scalar_t(1) / (scalar_t(1) + exp(s_u / blend_eps));
        const scalar_t wV = scalar_t(1) / (scalar_t(1) + exp(s_v / blend_eps));
        const scalar_t wW = scalar_t(1) / (scalar_t(1) + exp(s_w / blend_eps));
        atomicAdd(&num_u[g_local], wU * vU); atomicAdd(&den_u[g_local], wU);
        atomicAdd(&num_v[g_local], wV * vV); atomicAdd(&den_v[g_local], wV);
        atomicAdd(&num_w[g_local], wW * vW); atomicAdd(&den_w[g_local], wW);
    }
}

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
    const int Ngy, const int Ngz,
    const int B_sentinel,
    scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ sdf_u,
    scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ sdf_w,
    scalar_t* __restrict__ bU,
    scalar_t* __restrict__ bV,
    scalar_t* __restrict__ bW,
    scalar_t* __restrict__ winning_rho_cc,
    const int dirty_vol,
    const int di0, const int dj0, const int dk0,
    const int dAi, const int dAj, const int dAk)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= dirty_vol) return;
    const int dk = local % dAk;
    const int rem = local / dAk;
    const int dj = rem % dAj;
    const int di = rem / dAj;
    const int i  = di0 + di;
    const int j  = dj0 + dj;
    const int k  = dk0 + dk;
    const int g  = i * Ngy * Ngz + j * Ngz + k;

    // Key buffers use AABB-local flat indexing (size = dirty_vol).
    const uint64_t kc = key_cc[local];
    const uint64_t ku = key_u [local];
    const uint64_t kv = key_v [local];
    const uint64_t kw = key_w [local];

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

// Phase-I memory-reduction variant of the rho decode: drops winning_rho_cc
// and rho_bodies entirely.  Phase-I computes ``rho_eff`` from ``mu0`` in
// register inside Kernel B, so no per-cell winning density tensor is needed.
template <typename scalar_t>
__global__ void streaming_sdf_decode_keys_stag_3d_kernel(
    const uint64_t* __restrict__ key_cc,
    const uint64_t* __restrict__ key_u,
    const uint64_t* __restrict__ key_v,
    const uint64_t* __restrict__ key_w,
    const scalar_t* __restrict__ kin,           // [B,21]
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const scalar_t* __restrict__ gz,
    const int Ngy, const int Ngz,
    const int B_sentinel,
    scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ sdf_u,
    scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ sdf_w,
    scalar_t* __restrict__ bU,
    scalar_t* __restrict__ bV,
    scalar_t* __restrict__ bW,
    const int dirty_vol,
    const int di0, const int dj0, const int dk0,
    const int dAi, const int dAj, const int dAk,
    // Velocity-blend accumulators (see min kernel); when blend_eps > 0 the
    // face velocity is Σ w_i v_i / Σ w_i instead of the single min-winner.
    const scalar_t* __restrict__ num_u, const scalar_t* __restrict__ num_v,
    const scalar_t* __restrict__ num_w,
    const scalar_t* __restrict__ den_u, const scalar_t* __restrict__ den_v,
    const scalar_t* __restrict__ den_w,
    const scalar_t blend_eps)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= dirty_vol) return;
    const int dk = local % dAk;
    const int rem = local / dAk;
    const int dj = rem % dAj;
    const int di = rem / dAj;
    const int i  = di0 + di;
    const int j  = dj0 + dj;
    const int k  = dk0 + dk;
    const int g  = i * Ngy * Ngz + j * Ngz + k;

    // Key buffers use AABB-local flat indexing (size = dirty_vol).
    const uint64_t kc = key_cc[local];
    const uint64_t ku = key_u [local];
    const uint64_t kv = key_v [local];
    const uint64_t kw = key_w [local];

    const uint32_t bc = unpack_body_id(kc);
    const uint32_t bu = unpack_body_id(ku);
    const uint32_t bv = unpack_body_id(kv);
    const uint32_t bw = unpack_body_id(kw);

    if ((int)bc < B_sentinel) sdf_cc[g] = unpack_sdf<scalar_t>(kc);
    if ((int)bu < B_sentinel) sdf_u[g] = unpack_sdf<scalar_t>(ku);
    if ((int)bv < B_sentinel) sdf_v[g] = unpack_sdf<scalar_t>(kv);
    if ((int)bw < B_sentinel) sdf_w[g] = unpack_sdf<scalar_t>(kw);

    const bool blend = blend_eps > scalar_t(0);
    const scalar_t den_tol = scalar_t(1e-6);

    if (blend && den_u[local] > den_tol) {
        bU[g] = num_u[local] / den_u[local];
    } else if ((int)bu < B_sentinel) {
        const scalar_t* K = kin + (int)bu * 21;
        const scalar_t cm_y = K[13], cm_z = K[14];
        const scalar_t lv_x = K[15];
        const scalar_t av_y = K[19], av_z = K[20];
        const scalar_t yc = gy[j];
        const scalar_t zc = gz[k];
        bU[g] = lv_x + av_y*(zc - cm_z) - av_z*(yc - cm_y);
    }
    if (blend && den_v[local] > den_tol) {
        bV[g] = num_v[local] / den_v[local];
    } else if ((int)bv < B_sentinel) {
        const scalar_t* K = kin + (int)bv * 21;
        const scalar_t cm_x = K[12], cm_z = K[14];
        const scalar_t lv_y = K[16];
        const scalar_t av_x = K[18], av_z = K[20];
        const scalar_t xc = gx[i];
        const scalar_t zc = gz[k];
        bV[g] = lv_y + av_z*(xc - cm_x) - av_x*(zc - cm_z);
    }
    if (blend && den_w[local] > den_tol) {
        bW[g] = num_w[local] / den_w[local];
    } else if ((int)bw < B_sentinel) {
        const scalar_t* K = kin + (int)bw * 21;
        const scalar_t cm_x = K[12], cm_y = K[13];
        const scalar_t lv_z = K[17];
        const scalar_t av_x = K[18], av_y = K[19];
        const scalar_t xc = gx[i];
        const scalar_t yc = gy[j];
        bW[g] = lv_z + av_x*(yc - cm_y) - av_y*(xc - cm_x);
    }
}

void streaming_sdf_stag_3d_multi_cuda(
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
    at::Tensor key_cc_t, at::Tensor key_u_t, at::Tensor key_v_t, at::Tensor key_w_t,
    const int64_t interp_method,
    const int64_t dirty_i0, const int64_t dirty_j0, const int64_t dirty_k0,
    const int64_t dirty_Ai, const int64_t dirty_Aj, const int64_t dirty_Ak,
    at::Tensor num_u, at::Tensor num_v, at::Tensor num_w,
    at::Tensor den_u, at::Tensor den_v, at::Tensor den_w,
    const double blend_eps)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngx = (int)gx.numel();
    const int Ngy = (int)gy.numel();
    const int Ngz = (int)gz.numel();
    const int blockSize = (max_vol_per_body <= 128) ? 32
                        : (max_vol_per_body <= 4096) ? 128 : 256;
    // key_cc_t, key_u_t, key_v_t, key_w_t are pre-allocated int64 buffers
    // of size >= Ngx*Ngy*Ngz, passed in from Python to avoid per-call allocs.

    const int64_t dirty_vol = dirty_Ai * dirty_Aj * dirty_Ak;

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_stag_3d_multi_cuda", [&] {
        const int initBlock  = 256;
        const int initBlocks = (int)((dirty_vol + initBlock - 1) / initBlock);
        streaming_sdf_init_keys_3d_kernel<scalar_t>
            <<<initBlocks, initBlock, 0, stream>>>(
                sdf_cc.data_ptr<scalar_t>(), sdf_u.data_ptr<scalar_t>(),
                sdf_v.data_ptr<scalar_t>(), sdf_w.data_ptr<scalar_t>(),
                (uint64_t*)key_cc_t.data_ptr<int64_t>(),
                (uint64_t*)key_u_t.data_ptr<int64_t>(),
                (uint64_t*)key_v_t.data_ptr<int64_t>(),
                (uint64_t*)key_w_t.data_ptr<int64_t>(),
                (int)dirty_vol, B,
                (int)dirty_i0, (int)dirty_j0, (int)dirty_k0,
                (int)dirty_Ai, (int)dirty_Aj, (int)dirty_Ak,
                Ngy, Ngz);

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
                (int)interp_method,
                (int)dirty_i0, (int)dirty_j0, (int)dirty_k0,
                (int)dirty_Aj, (int)dirty_Ak,
                num_u.data_ptr<scalar_t>(), num_v.data_ptr<scalar_t>(),
                num_w.data_ptr<scalar_t>(),
                den_u.data_ptr<scalar_t>(), den_v.data_ptr<scalar_t>(),
                den_w.data_ptr<scalar_t>(),
                (scalar_t)blend_eps);

        streaming_sdf_decode_keys_stag_3d_kernel<scalar_t>
            <<<initBlocks, initBlock, 0, stream>>>(
                (const uint64_t*)key_cc_t.data_ptr<int64_t>(),
                (const uint64_t*)key_u_t.data_ptr<int64_t>(),
                (const uint64_t*)key_v_t.data_ptr<int64_t>(),
                (const uint64_t*)key_w_t.data_ptr<int64_t>(),
                kin.data_ptr<scalar_t>(),
                gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(),
                Ngy, Ngz, B,
                sdf_cc.data_ptr<scalar_t>(), sdf_u.data_ptr<scalar_t>(),
                sdf_v.data_ptr<scalar_t>(), sdf_w.data_ptr<scalar_t>(),
                body_u.data_ptr<scalar_t>(), body_v.data_ptr<scalar_t>(),
                body_w.data_ptr<scalar_t>(),
                (int)dirty_vol,
                (int)dirty_i0, (int)dirty_j0, (int)dirty_k0,
                (int)dirty_Ai, (int)dirty_Aj, (int)dirty_Ak,
                num_u.data_ptr<scalar_t>(), num_v.data_ptr<scalar_t>(),
                num_w.data_ptr<scalar_t>(),
                den_u.data_ptr<scalar_t>(), den_v.data_ptr<scalar_t>(),
                den_w.data_ptr<scalar_t>(),
                (scalar_t)blend_eps);
    });
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
//        and the variable-density Poisson coefficient
//           c[i,j,k]  = dt / (rho_body + (rho_f - rho_body) * mu0).
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
    const scalar_t rho_body,
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
        // mu0_proj != 0 → BDIM2 mu0-weighted coefficient (dt*mu0/rho_eff).
        // mu0_proj == 0 → plain variable-density coefficient (dt/rho_eff),
        // which keeps the projection non-degenerate for multibody swimmers.
        c_out[(i - 1) * c_stride_i + (j - 1) * c_stride_j + (k - 1)] =
            (mu0_proj ? dt * mu0 : dt) / (rho_body + (rho_f - rho_body) * mu0);
    }
}

template <typename scalar_t>
__global__ void bdim_vardens_3d_kernel(
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
    const scalar_t rho_body,
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
        eps, rho_body, rho_f, dt, inv_2h,
        Ngx, Ngy, Ngz, i, j, k, u0, ch,
        (Ngy - 2) * (Ngz - 2), (Ngz - 2),
        Ngx - 1, Ngy - 2, Ngz - 2, mu0_proj);
    // cv: y-face grid (Ngx-2, Ngy-1, Ngz-2), strides ((Ngy-1)*(Ngz-2), Ngz-2, 1)
    bdim_one_axis_3d<scalar_t>(
        v_prime, sdf_v, body_v,
        eps, rho_body, rho_f, dt, inv_2h,
        Ngx, Ngy, Ngz, i, j, k, v0, cv,
        (Ngy - 1) * (Ngz - 2), (Ngz - 2),
        Ngx - 2, Ngy - 1, Ngz - 2, mu0_proj);
    // cw: z-face grid (Ngx-2, Ngy-2, Ngz-1), strides ((Ngy-2)*(Ngz-1), Ngz-1, 1)
    bdim_one_axis_3d<scalar_t>(
        w_prime, sdf_w, body_w,
        eps, rho_body, rho_f, dt, inv_2h,
        Ngx, Ngy, Ngz, i, j, k, w0, cw,
        (Ngy - 2) * (Ngz - 1), (Ngz - 1),
        Ngx - 2, Ngy - 2, Ngz - 1, mu0_proj);
}

void bdim_vardens_3d_cuda(
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
    const double rho_body,
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

    AT_DISPATCH_FLOATING_TYPES(u0.scalar_type(), "bdim_vardens_3d_cuda", [&] {
        bdim_vardens_3d_kernel<scalar_t>
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
                (scalar_t)rho_body,
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

// =====================================================================
//  BDIM-σ variant of bdim_one_axis_3d / bdim_vardens_3d.
//
//  Per-cell Poisson coefficient is evaluated with mu0 of a shifted SDF
//  phi - sigma_shifts[body_id] (body_id decoded from the AABB-local key
//  buffer populated by Kernel A).  The velocity BDIM field (phi_out)
//  uses the unmodified mu0 — only the c_out line changes.
//
//  key buffers are AABB-local (size = dirty_vol), so we recompute the
//  local flat index from (i-di0, j-dj0, k-dk0).  Sentinel body_idx
//  (>= n_sigma) yields a zero shift, leaving thick bodies unchanged.
// =====================================================================
template <typename scalar_t>
__device__ __forceinline__ void bdim_one_axis_sigma_3d(
    const scalar_t* __restrict__ phi_prime,
    const scalar_t* __restrict__ sdf,
    const scalar_t* __restrict__ body,
    const scalar_t eps,
    const scalar_t rho_body,
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
    const int64_t* __restrict__ key,
    const float*   __restrict__ sigma_shifts,
    const int n_sigma,
    const int di0, const int dj0, const int dk0,
    const int dAj, const int dAk,
    const int mu0_proj)
{
    const int stride_i = Ngy * Ngz;
    const int stride_j = Ngz;
    const int g  = i * stride_i + j * stride_j + k;

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

    const scalar_t phi = sdf[g];
    scalar_t mu0, mu1;
    if (phi <= -eps) {
        mu0 = scalar_t(0); mu1 = scalar_t(0);
    } else if (phi >= eps) {
        mu0 = scalar_t(1); mu1 = scalar_t(0);
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

    // BDIM-σ: shifted mu0 used only for the Poisson coefficient.
    const int di = i - di0;
    const int dj = j - dj0;
    const int dk = k - dk0;
    const int local = di * (dAj * dAk) + dj * dAk + dk;
    const int32_t body_idx = (int32_t)((uint32_t)((uint64_t)key[local] & 0xFFFFFFFFull));
    const scalar_t sigma_shift = (body_idx < n_sigma)
        ? (scalar_t)sigma_shifts[body_idx] : scalar_t(0);
    const scalar_t phi_sigma = phi - sigma_shift;
    scalar_t mu0_poisson;
    if      (phi_sigma <= -eps) { mu0_poisson = scalar_t(0); }
    else if (phi_sigma >=  eps) { mu0_poisson = scalar_t(1); }
    else {
        const scalar_t deps_s = phi_sigma / eps;
        const scalar_t pi     = scalar_t(M_PI);
        mu0_poisson = scalar_t(0.5) * (scalar_t(1) + deps_s + sin(pi * deps_s) / pi);
    }

    scalar_t nx = (sdf[g_ip] - sdf[g_im]) * inv_2h;
    scalar_t ny = (sdf[g_jp] - sdf[g_jm]) * inv_2h;
    scalar_t nz = (sdf[g_kp] - sdf[g_km]) * inv_2h;
    const scalar_t nn = sqrt(nx*nx + ny*ny + nz*nz);
    if (nn > scalar_t(0)) {
        const scalar_t inv_nn = scalar_t(1) / nn;
        nx *= inv_nn; ny *= inv_nn; nz *= inv_nn;
    }

    const scalar_t b_c    = body[g];
    const scalar_t pp_c   = phi_prime[g];
    const scalar_t diff_c = pp_c - b_c;

    scalar_t ddx, ddy, ddz;
    if (i > 0 && i < Ngx - 1) {
        ddx = ((phi_prime[g_ip] - body[g_ip]) -
               (phi_prime[g_im] - body[g_im])) * inv_2h;
    } else { ddx = scalar_t(0); }
    if (j > 0 && j < Ngy - 1) {
        ddy = ((phi_prime[g_jp] - body[g_jp]) -
               (phi_prime[g_jm] - body[g_jm])) * inv_2h;
    } else { ddy = scalar_t(0); }
    if (k > 0 && k < Ngz - 1) {
        ddz = ((phi_prime[g_kp] - body[g_kp]) -
               (phi_prime[g_km] - body[g_km])) * inv_2h;
    } else { ddz = scalar_t(0); }
    const scalar_t nd = nx * ddx + ny * ddy + nz * ddz;

    phi_out[g] = mu0 * diff_c + b_c + mu1 * nd;
    if (i >= 1 && j >= 1 && k >= 1 && i <= c_hi_i && j <= c_hi_j && k <= c_hi_k) {
        // mu0_proj == 0 → drop the mu0 numerator (plain dt/rho_eff).
        c_out[(i - 1) * c_stride_i + (j - 1) * c_stride_j + (k - 1)] =
            (mu0_proj ? dt * mu0_poisson : dt)
            / (rho_body + (rho_f - rho_body) * mu0_poisson);
    }
}

template <typename scalar_t>
__global__ void bdim_vardens_sigma_3d_kernel(
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
    const int64_t* __restrict__ key_u,
    const int64_t* __restrict__ key_v,
    const int64_t* __restrict__ key_w,
    const float*   __restrict__ sigma_shifts,
    const int n_sigma,
    const scalar_t eps,
    const scalar_t rho_body,
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

    bdim_one_axis_sigma_3d<scalar_t>(
        u_prime, sdf_u, body_u,
        eps, rho_body, rho_f, dt, inv_2h,
        Ngx, Ngy, Ngz, i, j, k, u0, ch,
        (Ngy - 2) * (Ngz - 2), (Ngz - 2),
        Ngx - 1, Ngy - 2, Ngz - 2,
        key_u, sigma_shifts, n_sigma,
        di0, dj0, dk0, dAj, dAk, mu0_proj);
    bdim_one_axis_sigma_3d<scalar_t>(
        v_prime, sdf_v, body_v,
        eps, rho_body, rho_f, dt, inv_2h,
        Ngx, Ngy, Ngz, i, j, k, v0, cv,
        (Ngy - 1) * (Ngz - 2), (Ngz - 2),
        Ngx - 2, Ngy - 1, Ngz - 2,
        key_v, sigma_shifts, n_sigma,
        di0, dj0, dk0, dAj, dAk, mu0_proj);
    bdim_one_axis_sigma_3d<scalar_t>(
        w_prime, sdf_w, body_w,
        eps, rho_body, rho_f, dt, inv_2h,
        Ngx, Ngy, Ngz, i, j, k, w0, cw,
        (Ngy - 2) * (Ngz - 1), (Ngz - 1),
        Ngx - 2, Ngy - 2, Ngz - 1,
        key_w, sigma_shifts, n_sigma,
        di0, dj0, dk0, dAj, dAk, mu0_proj);
}

void bdim_vardens_sigma_3d_cuda(
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
    const at::Tensor& key_u,
    const at::Tensor& key_v,
    const at::Tensor& key_w,
    const at::Tensor& sigma_shifts,
    const double eps,
    const double rho_body,
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
    const int n_sigma = (int)sigma_shifts.numel();

    const int blockSize = 256;
    const int nblocks   = (int)((dirty_vol + blockSize - 1) / blockSize);

    AT_DISPATCH_FLOATING_TYPES(u0.scalar_type(), "bdim_vardens_sigma_3d_cuda", [&] {
        bdim_vardens_sigma_3d_kernel<scalar_t>
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
                key_u.data_ptr<int64_t>(),
                key_v.data_ptr<int64_t>(),
                key_w.data_ptr<int64_t>(),
                sigma_shifts.data_ptr<float>(),
                n_sigma,
                (scalar_t)eps,
                (scalar_t)rho_body,
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
    at::Tensor winning_rho_cc,
    // Dirty region: union of prev and curr union-AABB.  init/decode kernels
    // only touch this sub-block, making them O(dirty_vol) not O(Ngrid).
    const int64_t dirty_i0, const int64_t dirty_j0, const int64_t dirty_k0,
    const int64_t dirty_Ai, const int64_t dirty_Aj, const int64_t dirty_Ak)
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

    // Key arrays sized to dirty_vol (AABB-local indexing).
    // Only dirty sub-block cells are written; AABB-local flat index avoids
    // allocating O(Ngrid) buffers when dirty_vol << Ngrid.
    auto key_opts = at::TensorOptions().dtype(at::kLong).device(sdf_cc.device());
    const int64_t dirty_vol = dirty_Ai * dirty_Aj * dirty_Ak;
    auto key_cc_t = at::empty({dirty_vol}, key_opts);
    auto key_u_t  = at::empty({dirty_vol}, key_opts);
    auto key_v_t  = at::empty({dirty_vol}, key_opts);
    auto key_w_t  = at::empty({dirty_vol}, key_opts);

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_min_rho_3d_multi_cuda", [&] {
        const int initBlock  = 256;
        const int initBlocks = (int)((dirty_vol + initBlock - 1) / initBlock);
        streaming_sdf_init_keys_3d_kernel<scalar_t>
            <<<initBlocks, initBlock, 0, stream>>>(
                sdf_cc.data_ptr<scalar_t>(), sdf_u.data_ptr<scalar_t>(),
                sdf_v.data_ptr<scalar_t>(), sdf_w.data_ptr<scalar_t>(),
                (uint64_t*)key_cc_t.data_ptr<int64_t>(),
                (uint64_t*)key_u_t.data_ptr<int64_t>(),
                (uint64_t*)key_v_t.data_ptr<int64_t>(),
                (uint64_t*)key_w_t.data_ptr<int64_t>(),
                (int)dirty_vol, B,
                (int)dirty_i0, (int)dirty_j0, (int)dirty_k0,
                (int)dirty_Ai, (int)dirty_Aj, (int)dirty_Ak,
                Ngy, Ngz);

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
                (int)interp_method,
                (int)dirty_i0, (int)dirty_j0, (int)dirty_k0,
                (int)dirty_Aj, (int)dirty_Ak,
                // velocity blend not used by the rho/decode variant
                (scalar_t*)nullptr, (scalar_t*)nullptr, (scalar_t*)nullptr,
                (scalar_t*)nullptr, (scalar_t*)nullptr, (scalar_t*)nullptr,
                (scalar_t)(-1));

        streaming_sdf_decode_keys_rho_3d_kernel<scalar_t>
            <<<initBlocks, initBlock, 0, stream>>>(
                (const uint64_t*)key_cc_t.data_ptr<int64_t>(),
                (const uint64_t*)key_u_t.data_ptr<int64_t>(),
                (const uint64_t*)key_v_t.data_ptr<int64_t>(),
                (const uint64_t*)key_w_t.data_ptr<int64_t>(),
                kin.data_ptr<scalar_t>(), rho_bodies.data_ptr<scalar_t>(),
                gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(),
                Ngy, Ngz, B,
                sdf_cc.data_ptr<scalar_t>(), sdf_u.data_ptr<scalar_t>(),
                sdf_v.data_ptr<scalar_t>(), sdf_w.data_ptr<scalar_t>(),
                body_u.data_ptr<scalar_t>(), body_v.data_ptr<scalar_t>(),
                body_w.data_ptr<scalar_t>(), winning_rho_cc.data_ptr<scalar_t>(),
                (int)dirty_vol,
                (int)dirty_i0, (int)dirty_j0, (int)dirty_k0,
                (int)dirty_Ai, (int)dirty_Aj, (int)dirty_Ak);
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
    const int N_dir,
    const int* __restrict__ ref_desc,
    const scalar_t* __restrict__ ref_val,
    const int N_ref)
{
    const int op = blockIdx.z;
    const int total = N_neu + N_dir + N_ref;
    if (op >= total) return;

    // 0 = Neumann copy, 1 = Dirichlet direct write, 2 = reflective.
    int kind, comp, axis, dst_along, src_along = 0;
    scalar_t value = scalar_t(0);

    if (op < N_neu) {
        kind = 0;
        comp = neu_desc[op * 3 + 0];
        axis = neu_desc[op * 3 + 1];
        const int side = neu_desc[op * 3 + 2];
        const int sz = (int)shapes[comp * 3 + axis];
        if (side == 0) { dst_along = 0;      src_along = 1; }
        else           { dst_along = sz - 1; src_along = sz - 2; }
    } else if (op < N_neu + N_dir) {
        const int d = op - N_neu;
        kind = 1;
        comp = dir_desc[d * 3 + 0];
        axis = dir_desc[d * 3 + 1];
        const int offset = dir_desc[d * 3 + 2];
        const int sz = (int)shapes[comp * 3 + axis];
        dst_along = (offset >= 0) ? offset : (sz + offset);
        value = dir_val[d];
    } else {
        const int r = op - N_neu - N_dir;
        kind = 2;
        comp = ref_desc[r * 4 + 0];
        axis = ref_desc[r * 4 + 1];
        const int dst_off = ref_desc[r * 4 + 2];
        const int src_off = ref_desc[r * 4 + 3];
        const int sz = (int)shapes[comp * 3 + axis];
        dst_along = (dst_off >= 0) ? dst_off : (sz + dst_off);
        src_along = (src_off >= 0) ? src_off : (sz + src_off);
        value = ref_val[r];
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
        if (kind != 1) src_lin = (int64_t)src_along * s1 + (int64_t)i * s2 + j;
    } else if (axis == 1) {
        dst_lin = (int64_t)i * s1 + (int64_t)dst_along * s2 + j;
        if (kind != 1) src_lin = (int64_t)i * s1 + (int64_t)src_along * s2 + j;
    } else {
        dst_lin = (int64_t)i * s1 + (int64_t)j * s2 + dst_along;
        if (kind != 1) src_lin = (int64_t)i * s1 + (int64_t)j * s2 + src_along;
    }

    if      (kind == 0) base[dst_lin] = base[src_lin];
    else if (kind == 1) base[dst_lin] = value;
    else                base[dst_lin] = scalar_t(2) * value - base[src_lin];
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

    // Two-stage launch on the same stream:
    //   Stage 1: Neumann + Direct ops.
    //   Stage 2: Reflective ops (RMW against adjacent interior — must
    //            run AFTER stage 1 so any direct write that touched the
    //            source cell is already committed).
    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_3d_cuda", [&] {
        const int stage1 = N_neu + N_dir;
        if (stage1 > 0) {
            const dim3 grid1((unsigned)blocks_x, (unsigned)blocks_y, (unsigned)stage1);
            apply_bcs_3d_kernel<scalar_t><<<grid1, block, 0, stream>>>(
                u.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
                shapes.data_ptr<int64_t>(),
                (N_neu > 0) ? neu_desc.data_ptr<int>() : nullptr, N_neu,
                (N_dir > 0) ? dir_desc.data_ptr<int>() : nullptr,
                (N_dir > 0) ? dir_val.data_ptr<scalar_t>() : nullptr, N_dir,
                /*ref_desc=*/nullptr, /*ref_val=*/nullptr, /*N_ref=*/0);
        }
        if (N_ref > 0) {
            const dim3 grid2((unsigned)blocks_x, (unsigned)blocks_y, (unsigned)N_ref);
            apply_bcs_3d_kernel<scalar_t><<<grid2, block, 0, stream>>>(
                u.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
                shapes.data_ptr<int64_t>(),
                /*neu_desc=*/nullptr, /*N_neu=*/0,
                /*dir_desc=*/nullptr, /*dir_val=*/nullptr, /*N_dir=*/0,
                ref_desc.data_ptr<int>(),
                ref_val.data_ptr<scalar_t>(), N_ref);
        }
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

    // Bind temporaries to named locals: ``.contiguous().to(...)`` may return
    // a fresh tensor whose storage is freed at the end of the full-expression
    // unless held.  The CUDA launch is asynchronous, so a dangling pointer
    // could be read by the kernel after the storage has been recycled.
    auto F_c  = F.contiguous();
    auto xq_c = xq.contiguous().to(F.scalar_type());
    auto yq_c = yq.contiguous().to(F.scalar_type());
    auto zq_c = zq.contiguous().to(F.scalar_type());

    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "interpolate_3d_cuda", [&] {
        interpolate_3d_kernel<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
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
    m.impl("streaming_sdf_min_rho_3d_multi", &streaming_sdf_min_rho_3d_multi_cuda);
    m.impl("streaming_sdf_stag_3d_multi",    &streaming_sdf_stag_3d_multi_cuda);
    m.impl("bdim_vardens_3d",                &bdim_vardens_3d_cuda);
    m.impl("bdim_vardens_sigma_3d",          &bdim_vardens_sigma_3d_cuda);
    m.impl("streaming_sdf_forces_post_3d", &streaming_sdf_forces_post_3d_cuda);
    m.impl("apply_bcs_3d", &apply_bcs_3d_cuda);
    m.impl("interpolate_3d", &interpolate_3d_cuda);
}

}  // namespace lilytorch_kernels
