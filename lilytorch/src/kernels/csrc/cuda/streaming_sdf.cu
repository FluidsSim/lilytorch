// =====================================================================
//  streaming_sdf_min_3d
//
//  One-body fused kernel for BDIM SDF / face-velocity update.  For each
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

namespace lilytorch_kernels {

template <typename scalar_t>
__device__ __forceinline__ scalar_t trilinear_sample_border(
    const scalar_t* __restrict__ F,
    const scalar_t* __restrict__ bx,
    const scalar_t* __restrict__ by,
    const scalar_t* __restrict__ bz,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t bx_last, const scalar_t by_last, const scalar_t bz_last,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    const scalar_t inv_vol,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    xq = max(bx0, min(xq, bx_last));
    yq = max(by0, min(yq, by_last));
    zq = max(bz0, min(zq, bz_last));

    int ix = (int)floor((xq - bx0) * inv_dx);
    int iy = (int)floor((yq - by0) * inv_dy);
    int iz = (int)floor((zq - bz0) * inv_dz);
    ix = max(0, min(ix, Mx - 2));
    iy = max(0, min(iy, My - 2));
    iz = max(0, min(iz, Mz - 2));
    const int ixp = ix + 1, iyp = iy + 1, izp = iz + 1;

    const scalar_t wx0 = bx[ixp] - xq, wx1 = xq - bx[ix];
    const scalar_t wy0 = by[iyp] - yq, wy1 = yq - by[iy];
    const scalar_t wz0 = bz[izp] - zq, wz1 = zq - bz[iz];

    const int s2 = Mz;
    const int s1 = My * Mz;

    return (
        wx0 * wy0 * wz0 * F[ix  * s1 + iy  * s2 + iz ]
      + wx0 * wy0 * wz1 * F[ix  * s1 + iy  * s2 + izp]
      + wx0 * wy1 * wz0 * F[ix  * s1 + iyp * s2 + iz ]
      + wx0 * wy1 * wz1 * F[ix  * s1 + iyp * s2 + izp]
      + wx1 * wy0 * wz0 * F[ixp * s1 + iy  * s2 + iz ]
      + wx1 * wy0 * wz1 * F[ixp * s1 + iy  * s2 + izp]
      + wx1 * wy1 * wz0 * F[ixp * s1 + iyp * s2 + iz ]
      + wx1 * wy1 * wz1 * F[ixp * s1 + iyp * s2 + izp]
    ) * inv_vol;
}

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

template <typename scalar_t>
__global__ void streaming_sdf_min_3d_kernel(
    const scalar_t* __restrict__ F,
    const scalar_t* __restrict__ /*bx*/,
    const scalar_t* __restrict__ /*by*/,
    const scalar_t* __restrict__ /*bz*/,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t /*bx_last*/, const scalar_t /*by_last*/, const scalar_t /*bz_last*/,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    const scalar_t /*inv_vol*/,
    const scalar_t r00, const scalar_t r01, const scalar_t r02,
    const scalar_t r10, const scalar_t r11, const scalar_t r12,
    const scalar_t r20, const scalar_t r21, const scalar_t r22,
    const scalar_t bp_x, const scalar_t bp_y, const scalar_t bp_z,
    const scalar_t cm_x, const scalar_t cm_y, const scalar_t cm_z,
    const scalar_t lv_x, const scalar_t lv_y, const scalar_t lv_z,
    const scalar_t av_x, const scalar_t av_y, const scalar_t av_z,
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const scalar_t* __restrict__ gz,
    const int Ngy, const int Ngz,
    const scalar_t half_h,
    const int i0, const int j0, const int k0,
    const int Ai, const int Aj, const int Ak,
    scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ sdf_u,
    scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ sdf_w,
    scalar_t* __restrict__ bU,
    scalar_t* __restrict__ bV,
    scalar_t* __restrict__ bW,
    scalar_t* __restrict__ sparse_cc,
    const int interp_method)
{
    const int total = Ai * Aj * Ak;
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int di  = tid / (Aj * Ak);
    const int rem = tid - di * (Aj * Ak);
    const int dj  = rem / Ak;
    const int dk  = rem - dj * Ak;
    const int i   = i0 + di;
    const int j   = j0 + dj;
    const int k   = k0 + dk;

    const int g_idx = (i * Ngy + j) * Ngz + k;

    const scalar_t xc = gx[i];
    const scalar_t yc = gy[j];
    const scalar_t zc = gz[k];

    // Body-frame CC point (single rotation; the 3 face points are derived
    // from it by precomputed deltas Δ_k = -half_h * col_k(R_T)). See the
    // matching CPU code for the rationale.
    const scalar_t dx_w = xc - bp_x, dy_w = yc - bp_y, dz_w = zc - bp_z;
    const scalar_t bxq = r00 * dx_w + r01 * dy_w + r02 * dz_w;
    const scalar_t byq = r10 * dx_w + r11 * dy_w + r12 * dz_w;
    const scalar_t bzq = r20 * dx_w + r21 * dy_w + r22 * dz_w;

    const scalar_t neg_half_h = -half_h;
    const scalar_t du_x = neg_half_h * r00, du_y = neg_half_h * r10, du_z = neg_half_h * r20;
    const scalar_t dv_x = neg_half_h * r01, dv_y = neg_half_h * r11, dv_z = neg_half_h * r21;
    const scalar_t dw_x = neg_half_h * r02, dw_y = neg_half_h * r12, dw_z = neg_half_h * r22;

    // ---------------- cc ----------------
    {
        const scalar_t s = sdf_sample_dispatch(
            interp_method,
            F, Mx, My, Mz, bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
            bxq, byq, bzq);
        sparse_cc[tid] = s;
        if (s < sdf_cc[g_idx]) sdf_cc[g_idx] = s;
    }

    // ---------------- u-face: world (xc - h/2, yc, zc) ----------------
    {
        const scalar_t s = sdf_sample_dispatch(
            interp_method,
            F, Mx, My, Mz, bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
            bxq + du_x, byq + du_y, bzq + du_z);
        if (s < sdf_u[g_idx]) {
            sdf_u[g_idx] = s;
            bU[g_idx] = lv_x + av_y * (zc - cm_z) - av_z * (yc - cm_y);
        }
    }

    // ---------------- v-face: world (xc, yc - h/2, zc) ----------------
    {
        const scalar_t s = sdf_sample_dispatch(
            interp_method,
            F, Mx, My, Mz, bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
            bxq + dv_x, byq + dv_y, bzq + dv_z);
        if (s < sdf_v[g_idx]) {
            sdf_v[g_idx] = s;
            bV[g_idx] = lv_y + av_z * (xc - cm_x) - av_x * (zc - cm_z);
        }
    }

    // ---------------- w-face: world (xc, yc, zc - h/2) ----------------
    {
        const scalar_t s = sdf_sample_dispatch(
            interp_method,
            F, Mx, My, Mz, bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
            bxq + dw_x, byq + dw_y, bzq + dw_z);
        if (s < sdf_w[g_idx]) {
            sdf_w[g_idx] = s;
            bW[g_idx] = lv_z + av_x * (yc - cm_y) - av_y * (xc - cm_x);
        }
    }
}

// =====================================================================
//  CUDA dispatch
// =====================================================================

void streaming_sdf_min_3d_cuda(
    const at::Tensor& F,
    const at::Tensor& bx, const at::Tensor& by, const at::Tensor& bz,
    const double bx0, const double by0, const double bz0,
    const double bx_last, const double by_last, const double bz_last,
    const double inv_dx, const double inv_dy, const double inv_dz,
    const double inv_vol,
    c10::ArrayRef<double> R_T,
    c10::ArrayRef<double> body_pos,
    c10::ArrayRef<double> com_pos,
    c10::ArrayRef<double> lin_vel,
    c10::ArrayRef<double> ang_vel,
    const at::Tensor& gx, const at::Tensor& gy, const at::Tensor& gz,
    const double h_grid,
    const int64_t i0, const int64_t i1,
    const int64_t j0, const int64_t j1,
    const int64_t k0, const int64_t k1,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v, at::Tensor sdf_w,
    at::Tensor body_u, at::Tensor body_v, at::Tensor body_w,
    at::Tensor sparse_cc,
    const int64_t interp_method)
{
    TORCH_CHECK(R_T.size()      == 9, "R_T must have 9 elements");
    TORCH_CHECK(body_pos.size() == 3, "body_pos must have 3 elements");
    TORCH_CHECK(com_pos.size()  == 3, "com_pos must have 3 elements");
    TORCH_CHECK(lin_vel.size()  == 3, "lin_vel must have 3 elements");
    TORCH_CHECK(ang_vel.size()  == 3, "ang_vel must have 3 elements");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int Ai = (int)(i1 - i0);
    const int Aj = (int)(j1 - j0);
    const int Ak = (int)(k1 - k0);
    const int N  = Ai * Aj * Ak;
    if (N <= 0) return;

    const int Mx = (int)bx.numel();
    const int My = (int)by.numel();
    const int Mz = (int)bz.numel();
    const int Ngy = (int)gy.numel();
    const int Ngz = (int)gz.numel();

    const int blockSize = (N <= 128) ? 32 : (N <= 4096) ? 128 : 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(F.scalar_type(), "streaming_sdf_min_3d_cuda", [&] {
        const scalar_t* F_ptr  = F.contiguous().data_ptr<scalar_t>();
        const scalar_t* bx_ptr = bx.contiguous().data_ptr<scalar_t>();
        const scalar_t* by_ptr = by.contiguous().data_ptr<scalar_t>();
        const scalar_t* bz_ptr = bz.contiguous().data_ptr<scalar_t>();
        const scalar_t* gx_ptr = gx.contiguous().data_ptr<scalar_t>();
        const scalar_t* gy_ptr = gy.contiguous().data_ptr<scalar_t>();
        const scalar_t* gz_ptr = gz.contiguous().data_ptr<scalar_t>();

        scalar_t* sdf_cc_ptr = sdf_cc.data_ptr<scalar_t>();
        scalar_t* sdf_u_ptr  = sdf_u.data_ptr<scalar_t>();
        scalar_t* sdf_v_ptr  = sdf_v.data_ptr<scalar_t>();
        scalar_t* sdf_w_ptr  = sdf_w.data_ptr<scalar_t>();
        scalar_t* bU_ptr     = body_u.data_ptr<scalar_t>();
        scalar_t* bV_ptr     = body_v.data_ptr<scalar_t>();
        scalar_t* bW_ptr     = body_w.data_ptr<scalar_t>();
        scalar_t* sp_ptr     = sparse_cc.data_ptr<scalar_t>();

        streaming_sdf_min_3d_kernel<scalar_t>
            <<<numBlocks, blockSize, 0, stream>>>(
                F_ptr, bx_ptr, by_ptr, bz_ptr,
                Mx, My, Mz,
                (scalar_t)bx0, (scalar_t)by0, (scalar_t)bz0,
                (scalar_t)bx_last, (scalar_t)by_last, (scalar_t)bz_last,
                (scalar_t)inv_dx, (scalar_t)inv_dy, (scalar_t)inv_dz,
                (scalar_t)inv_vol,
                (scalar_t)R_T[0], (scalar_t)R_T[1], (scalar_t)R_T[2],
                (scalar_t)R_T[3], (scalar_t)R_T[4], (scalar_t)R_T[5],
                (scalar_t)R_T[6], (scalar_t)R_T[7], (scalar_t)R_T[8],
                (scalar_t)body_pos[0], (scalar_t)body_pos[1], (scalar_t)body_pos[2],
                (scalar_t)com_pos[0],  (scalar_t)com_pos[1],  (scalar_t)com_pos[2],
                (scalar_t)lin_vel[0],  (scalar_t)lin_vel[1],  (scalar_t)lin_vel[2],
                (scalar_t)ang_vel[0],  (scalar_t)ang_vel[1],  (scalar_t)ang_vel[2],
                gx_ptr, gy_ptr, gz_ptr,
                Ngy, Ngz,
                (scalar_t)(0.5 * h_grid),
                (int)i0, (int)j0, (int)k0,
                Ai, Aj, Ak,
                sdf_cc_ptr, sdf_u_ptr, sdf_v_ptr, sdf_w_ptr,
                bU_ptr, bV_ptr, bW_ptr,
                sp_ptr,
                (int)interp_method);
    });
}

// =====================================================================
//  streaming_sdf_min_3d_multi  (Phase C: batched-call over bodies)
//
//  Single Python op call handles ALL bodies, looping body-by-body
//  inside the C++ launcher (still B kernel launches, but no Python /
//  torch.ops dispatch / per-body tensor packing overhead).
//
//  Each per-body kernel touches each fluid cell at most once -- there
//  are no cross-body races, so the simple compare-swap from the single
//  kernel works without atomics.  The kernel reads its (Mx,My,Mz),
//  axis tables, meta and kin via per-body offsets in the packed device
//  tensors that the wrapper sends down.
// =====================================================================

template <typename scalar_t>
__global__ void streaming_sdf_min_3d_multi_kernel(
    const int b,
    const scalar_t* __restrict__ F_flat,
    const int64_t*  __restrict__ F_offsets,
    const scalar_t* __restrict__ bx_flat,
    const int64_t*  __restrict__ bx_offsets,
    const scalar_t* __restrict__ by_flat,
    const int64_t*  __restrict__ by_offsets,
    const scalar_t* __restrict__ bz_flat,
    const int64_t*  __restrict__ bz_offsets,
    const int64_t*  __restrict__ body_shapes,   // [B,3] (Mx,My,Mz)
    const scalar_t* __restrict__ body_meta,     // [B,10]
    const scalar_t* __restrict__ kin,           // [B,21]
    const int64_t*  __restrict__ aabb_lo,       // [B,3] (i0,j0,k0)
    const int64_t*  __restrict__ aabb_dim,      // [B,3] (Ai,Aj,Ak)
    const int64_t*  __restrict__ cell_offsets,  // [B+1] prefix sum of Ai*Aj*Ak
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const scalar_t* __restrict__ gz,
    const int Ngy, const int Ngz,
    const scalar_t half_h,
    scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ sdf_u,
    scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ sdf_w,
    scalar_t* __restrict__ bU,
    scalar_t* __restrict__ bV,
    scalar_t* __restrict__ bW,
    scalar_t* __restrict__ sparse_cc_flat,
    const int interp_method)
{
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

    const scalar_t* F  = F_flat  + F_offsets[b];
    // Body axis tables (bx/by/bz) and per-body inv_vol/bxL/byL/bzL are no
    // longer read: trilinear_sample_uniform handles uniform grids analytically.
    const int Mx = (int)body_shapes[b*3 + 0];
    const int My = (int)body_shapes[b*3 + 1];
    const int Mz = (int)body_shapes[b*3 + 2];

    const scalar_t* M = body_meta + b*10;
    const scalar_t bx0 = M[0], by0 = M[1], bz0 = M[2];
    const scalar_t idx_ = M[6], idy_ = M[7], idz_ = M[8];

    const scalar_t* K = kin + b*21;
    const scalar_t r00 = K[0],  r01 = K[1],  r02 = K[2];
    const scalar_t r10 = K[3],  r11 = K[4],  r12 = K[5];
    const scalar_t r20 = K[6],  r21 = K[7],  r22 = K[8];
    const scalar_t bp_x = K[9],  bp_y = K[10], bp_z = K[11];
    const scalar_t cm_x = K[12], cm_y = K[13], cm_z = K[14];
    const scalar_t lv_x = K[15], lv_y = K[16], lv_z = K[17];
    const scalar_t av_x = K[18], av_y = K[19], av_z = K[20];

    const scalar_t xc = gx[i];
    const scalar_t yc = gy[j];
    const scalar_t zc = gz[k];

    const int64_t sparse_idx = cell_offsets[b] + (int64_t)local;

    // Body-frame CC point (single rotation per cell). Face points = body_cc + Δ_k,
    // where Δ_k = -half_h * col_k(R_T) is a per-body constant.
    const scalar_t dx_w = xc - bp_x, dy_w = yc - bp_y, dz_w = zc - bp_z;
    const scalar_t bxq = r00*dx_w + r01*dy_w + r02*dz_w;
    const scalar_t byq = r10*dx_w + r11*dy_w + r12*dz_w;
    const scalar_t bzq = r20*dx_w + r21*dy_w + r22*dz_w;

    const scalar_t neg_half_h = -half_h;
    const scalar_t du_x = neg_half_h * r00, du_y = neg_half_h * r10, du_z = neg_half_h * r20;
    const scalar_t dv_x = neg_half_h * r01, dv_y = neg_half_h * r11, dv_z = neg_half_h * r21;
    const scalar_t dw_x = neg_half_h * r02, dw_y = neg_half_h * r12, dw_z = neg_half_h * r22;

    {
        const scalar_t s = sdf_sample_dispatch(
            interp_method,
            F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
            bxq, byq, bzq);
        sparse_cc_flat[sparse_idx] = s;
        if (s < sdf_cc[g_idx]) sdf_cc[g_idx] = s;
    }
    {
        const scalar_t s = sdf_sample_dispatch(
            interp_method,
            F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
            bxq + du_x, byq + du_y, bzq + du_z);
        if (s < sdf_u[g_idx]) {
            sdf_u[g_idx] = s;
            bU[g_idx] = lv_x + av_y*(zc - cm_z) - av_z*(yc - cm_y);
        }
    }
    {
        const scalar_t s = sdf_sample_dispatch(
            interp_method,
            F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
            bxq + dv_x, byq + dv_y, bzq + dv_z);
        if (s < sdf_v[g_idx]) {
            sdf_v[g_idx] = s;
            bV[g_idx] = lv_y + av_z*(xc - cm_x) - av_x*(zc - cm_z);
        }
    }
    {
        const scalar_t s = sdf_sample_dispatch(
            interp_method,
            F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
            bxq + dw_x, byq + dw_y, bzq + dw_z);
        if (s < sdf_w[g_idx]) {
            sdf_w[g_idx] = s;
            bW[g_idx] = lv_z + av_x*(yc - cm_y) - av_y*(xc - cm_x);
        }
    }
}

void streaming_sdf_min_3d_multi_cuda(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& bx_flat, const at::Tensor& bx_offsets,
    const at::Tensor& by_flat, const at::Tensor& by_offsets,
    const at::Tensor& bz_flat, const at::Tensor& bz_offsets,
    const at::Tensor& body_shapes,
    const at::Tensor& body_meta,
    const at::Tensor& kin,
    const at::Tensor& aabb_lo,
    const at::Tensor& aabb_dim,
    const at::Tensor& cell_offsets,
    const at::Tensor& gx, const at::Tensor& gy, const at::Tensor& gz,
    const double h_grid,
    const int64_t max_vol_per_body,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v, at::Tensor sdf_w,
    at::Tensor body_u, at::Tensor body_v, at::Tensor body_w,
    at::Tensor sparse_cc_flat,
    const int64_t interp_method)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngy = (int)gy.numel();
    const int Ngz = (int)gz.numel();

    const int blockSize = (max_vol_per_body <= 128) ? 32
                        : (max_vol_per_body <= 4096) ? 128 : 256;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(F_flat.scalar_type(), "streaming_sdf_min_3d_multi_cuda", [&] {
        const auto* aabb_dim_ptr_h = aabb_dim.data_ptr<int64_t>();  // device ptr; can't read host-side
        for (int b = 0; b < B; ++b) {
            // Compute per-body grid size on host; we need vol_b for that.
            // aabb_dim lives on device, so cache an int CPU view at the call site.
            // For simplicity, re-launch with worst-case grid; threads beyond vol early-out.
            const int blocksPerBody = (int)((max_vol_per_body + blockSize - 1) / blockSize);
            streaming_sdf_min_3d_multi_kernel<scalar_t>
                <<<dim3(blocksPerBody, 1, 1), dim3(blockSize, 1, 1), 0, stream>>>(
                    b,
                    F_flat.data_ptr<scalar_t>(),
                    F_offsets.data_ptr<int64_t>(),
                    bx_flat.data_ptr<scalar_t>(), bx_offsets.data_ptr<int64_t>(),
                    by_flat.data_ptr<scalar_t>(), by_offsets.data_ptr<int64_t>(),
                    bz_flat.data_ptr<scalar_t>(), bz_offsets.data_ptr<int64_t>(),
                    body_shapes.data_ptr<int64_t>(),
                    body_meta.data_ptr<scalar_t>(),
                    kin.data_ptr<scalar_t>(),
                    aabb_lo.data_ptr<int64_t>(),
                    aabb_dim.data_ptr<int64_t>(),
                    cell_offsets.data_ptr<int64_t>(),
                    gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(),
                    Ngy, Ngz,
                    (scalar_t)(0.5 * h_grid),
                    sdf_cc.data_ptr<scalar_t>(),
                    sdf_u.data_ptr<scalar_t>(),
                    sdf_v.data_ptr<scalar_t>(),
                    sdf_w.data_ptr<scalar_t>(),
                    body_u.data_ptr<scalar_t>(),
                    body_v.data_ptr<scalar_t>(),
                    body_w.data_ptr<scalar_t>(),
                    sparse_cc_flat.data_ptr<scalar_t>(),
                    (int)interp_method);
        }
        (void)aabb_dim_ptr_h;
    });
}


// =====================================================================
//  bdim_forces_3d_multi  (Phase D: per-body force integration)
//
//  For each body b with AABB (Ai,Aj,Ak), this kernel walks the AABB,
//  reads the cached per-body cc-SDF from `sparse_cc_flat` (already
//  written by streaming_sdf_min_3d_multi — no resampling), reads the
//  union-AABB-relative stress and pressure-force fields, evaluates the
//  smoothed delta (1st-order cosine), and accumulates per-body
//      (fv_x, fv_y, fv_z, tv_x, tv_y, tv_z,
//       fp_x, fp_y, fp_z, tp_x, tp_y, tp_z)
//  with COM-relative arms.  Block-reduces in shared memory and atomic-
//  adds into out[b, 0..11].
//
//  Output is float64 to match the .to(float64).sum() accumulation
//  precision of the PyTorch path.
// =====================================================================

template <typename scalar_t>
__global__ void bdim_forces_3d_multi_kernel(
    const int b,
    const scalar_t* __restrict__ sparse_cc_flat,
    const int64_t*  __restrict__ cell_offsets,
    const scalar_t* __restrict__ kin,
    const int64_t*  __restrict__ aabb_lo,
    const int64_t*  __restrict__ aabb_dim,
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const scalar_t* __restrict__ gz,
    const int u_i0, const int u_j0, const int u_k0,  // union-AABB origin
    const int Sj, const int Sk,                       // union-AABB dim_j, dim_k
    const scalar_t* __restrict__ xs,
    const scalar_t* __restrict__ ys,
    const scalar_t* __restrict__ zs,
    const scalar_t* __restrict__ px,
    const scalar_t* __restrict__ py,
    const scalar_t* __restrict__ pz,
    const scalar_t eps_body,
    const scalar_t eps_solver,
    const scalar_t h3,
    const int delta_order,
    double* __restrict__ out)  // (B, 12)
{
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
        const int i0  = (int)aabb_lo[b*3 + 0];
        const int j0  = (int)aabb_lo[b*3 + 1];
        const int k0  = (int)aabb_lo[b*3 + 2];
        const int i = i0 + di, j = j0 + dj, k = k0 + dk;

        // Read cached per-body cc-SDF written by streaming_sdf_min_3d_multi.
        const int64_t sparse_base = cell_offsets[b];
        const scalar_t sdf = sparse_cc_flat[sparse_base + (int64_t)local];

        // Early-skip optimisation: most cells of the AABB lie far inside
        // or far outside the body so both smoothed deltas are zero — the
        // 12 force/torque contributions vanish.  Bail out early to avoid
        // 6 global-memory loads (xs/ys/zs/px/py/pz) and 2 ``cos`` calls.
        // Threads with skip=true keep ``acc[*] = 0.0`` (already zeroed
        // above) and still participate in the shared-memory reduction
        // below, so the reduction tree remains correct.
        const scalar_t band_lo =
            (eps_solver - eps_body) < (-eps_body)
                ? (eps_solver - eps_body) : (-eps_body);
        const scalar_t band_hi =
            (eps_solver + eps_body) > ( eps_body)
                ? (eps_solver + eps_body) : ( eps_body);
        if (sdf > band_lo && sdf < band_hi) {
            // Only the COM (kin[12..14]) is needed for force/torque arms.
            const scalar_t* K = kin + b*21;
            const scalar_t cm_x = K[12], cm_y = K[13], cm_z = K[14];

            const scalar_t xc = gx[i];
            const scalar_t yc = gy[j];
            const scalar_t zc = gz[k];

            // Smoothed deltas (cosine kernel)
            const scalar_t pi_v = (scalar_t)3.141592653589793;
            const scalar_t inv_2eps = (scalar_t)0.5 / eps_body;
            const scalar_t pi_over_eb = pi_v / eps_body;
            scalar_t delta_visc = 0;
            const scalar_t d_visc = sdf - eps_solver;
            if (d_visc > -eps_body && d_visc < eps_body) {
                delta_visc = ((scalar_t)1 + cos(pi_over_eb * d_visc)) * inv_2eps;
            }
            scalar_t delta_pres = 0;
            if (sdf > -eps_body && sdf < eps_body) {
                delta_pres = ((scalar_t)1 + cos(pi_over_eb * sdf)) * inv_2eps;
            }

            // Towers (2008) 2nd-order correction: δ_S = δ_ε(φ) / |∇φ|
            // Finite-difference gradient of the per-body SDF within its AABB.
            // Central differences in the interior; one-sided at AABB boundaries.
            if (delta_order == 2 && (delta_visc > (scalar_t)0 || delta_pres > (scalar_t)0)) {
                const scalar_t h_grid = gx[1] - gx[0];  // uniform grid spacing
                const scalar_t inv_h  = (scalar_t)1.0 / h_grid;
                const int64_t  AjAk   = (int64_t)Aj * Ak;
                const int64_t  loc64  = (int64_t)local;

                scalar_t sdf_xp = (di < Ai-1) ? sparse_cc_flat[sparse_base + loc64 + AjAk] : sdf;
                scalar_t sdf_xm = (di > 0)    ? sparse_cc_flat[sparse_base + loc64 - AjAk] : sdf;
                scalar_t cx     = (di > 0 && di < Ai-1) ? (scalar_t)0.5 : (scalar_t)1.0;
                scalar_t dsdx   = cx * (sdf_xp - sdf_xm) * inv_h;

                scalar_t sdf_yp = (dj < Aj-1) ? sparse_cc_flat[sparse_base + loc64 + (int64_t)Ak] : sdf;
                scalar_t sdf_ym = (dj > 0)    ? sparse_cc_flat[sparse_base + loc64 - (int64_t)Ak] : sdf;
                scalar_t cy     = (dj > 0 && dj < Aj-1) ? (scalar_t)0.5 : (scalar_t)1.0;
                scalar_t dsdy   = cy * (sdf_yp - sdf_ym) * inv_h;

                scalar_t sdf_zp = (dk < Ak-1) ? sparse_cc_flat[sparse_base + loc64 + 1] : sdf;
                scalar_t sdf_zm = (dk > 0)    ? sparse_cc_flat[sparse_base + loc64 - 1] : sdf;
                scalar_t cz     = (dk > 0 && dk < Ak-1) ? (scalar_t)0.5 : (scalar_t)1.0;
                scalar_t dsdz   = cz * (sdf_zp - sdf_zm) * inv_h;

                scalar_t grad_mag = sqrt(dsdx*dsdx + dsdy*dsdy + dsdz*dsdz);
                const scalar_t min_grad = (scalar_t)1e-3;
                if (grad_mag < min_grad) grad_mag = min_grad;
                const scalar_t inv_grad = (scalar_t)1.0 / grad_mag;
                delta_visc *= inv_grad;
                delta_pres *= inv_grad;
            }

            // Read union-AABB-relative stress (caller guarantees AABB ⊆ union AABB)
            const int sub_i = i - u_i0;
            const int sub_j = j - u_j0;
            const int sub_k = k - u_k0;
            const int s_idx = (sub_i * Sj + sub_j) * Sk + sub_k;
            const scalar_t xs_v = xs[s_idx], ys_v = ys[s_idx], zs_v = zs[s_idx];
            const scalar_t px_v = px[s_idx], py_v = py[s_idx], pz_v = pz[s_idx];

            const scalar_t arm_x = xc - cm_x;
            const scalar_t arm_y = yc - cm_y;
            const scalar_t arm_z = zc - cm_z;

            const double fv_x_d = (double)(xs_v * delta_visc);
            const double fv_y_d = (double)(ys_v * delta_visc);
            const double fv_z_d = (double)(zs_v * delta_visc);
            const double fp_x_d = (double)(px_v * delta_pres);
            const double fp_y_d = (double)(py_v * delta_pres);
            const double fp_z_d = (double)(pz_v * delta_pres);

            acc[0] = fv_x_d;
            acc[1] = fv_y_d;
            acc[2] = fv_z_d;
            acc[3] = (double)arm_y * fv_z_d - (double)arm_z * fv_y_d;
            acc[4] = (double)arm_z * fv_x_d - (double)arm_x * fv_z_d;
            acc[5] = (double)arm_x * fv_y_d - (double)arm_y * fv_x_d;
            acc[6] = fp_x_d;
            acc[7] = fp_y_d;
            acc[8] = fp_z_d;
            acc[9]  = (double)arm_y * fp_z_d - (double)arm_z * fp_y_d;
            acc[10] = (double)arm_z * fp_x_d - (double)arm_x * fp_z_d;
            acc[11] = (double)arm_x * fp_y_d - (double)arm_y * fp_x_d;
        }
    }

    // Sequential 12-channel block reductions (one shared-mem buffer)
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const double h3_d = (double)h3;

#pragma unroll
    for (int c = 0; c < 12; ++c) {
        sdata[tid] = acc[c];
        __syncthreads();
        for (int s = bdim >> 1; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            atomicAdd(&out[b*12 + c], sdata[0] * h3_d);
        }
        __syncthreads();
    }
}

void bdim_forces_3d_multi_cuda(
    const at::Tensor& sparse_cc_flat,
    const at::Tensor& cell_offsets,
    const at::Tensor& kin,
    const at::Tensor& aabb_lo,
    const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy, const at::Tensor& gz,
    const int64_t u_i0, const int64_t u_j0, const int64_t u_k0,
    const int64_t Sj,   const int64_t Sk,
    const at::Tensor& xs, const at::Tensor& ys, const at::Tensor& zs,
    const at::Tensor& px, const at::Tensor& py, const at::Tensor& pz,
    const double eps_body,
    const double eps_solver,
    const double h3,
    const int64_t max_vol_per_body,
    const int64_t delta_order,
    at::Tensor out)  // (B, 12) float64
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blockSize = 256;
    const size_t shmem  = (size_t)blockSize * sizeof(double);

    AT_DISPATCH_FLOATING_TYPES(sparse_cc_flat.scalar_type(), "bdim_forces_3d_multi_cuda", [&] {
        for (int b = 0; b < B; ++b) {
            const int blocksPerBody = (int)((max_vol_per_body + blockSize - 1) / blockSize);
            bdim_forces_3d_multi_kernel<scalar_t>
                <<<dim3(blocksPerBody, 1, 1), dim3(blockSize, 1, 1), shmem, stream>>>(
                    b,
                    sparse_cc_flat.data_ptr<scalar_t>(),
                    cell_offsets.data_ptr<int64_t>(),
                    kin.data_ptr<scalar_t>(),
                    aabb_lo.data_ptr<int64_t>(),
                    aabb_dim.data_ptr<int64_t>(),
                    gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(),
                    (int)u_i0, (int)u_j0, (int)u_k0,
                    (int)Sj, (int)Sk,
                    xs.data_ptr<scalar_t>(), ys.data_ptr<scalar_t>(), zs.data_ptr<scalar_t>(),
                    px.data_ptr<scalar_t>(), py.data_ptr<scalar_t>(), pz.data_ptr<scalar_t>(),
                    (scalar_t)eps_body, (scalar_t)eps_solver, (scalar_t)h3,
                    (int)delta_order,
                    out.data_ptr<double>());
        }
    });
}

// =====================================================================
//  streaming_sdf_forces_fused_3d_multi  (Phase C+D fused, lagged forces)
//
//  Single kernel per body that replaces both streaming_sdf_min_3d_multi
//  (Phase C) and bdim_forces_3d_multi (Phase D).
//
//  Per-cell operations within the body's per-body AABB:
//  1. Sample body SDF at CC + 3 staggered face locations.
//  2. Compare-swap into union SDF fields (same as Phase C).
//  3. Track winning-body density in winning_rho_cc for variable-density
//     FSI: when body b wins the CC min, winning_rho_cc[g_idx] = rho_bodies[b].
//  4. When the cell is within the force band, compute viscous stress and
//     pressure force INLINE from the beginning-of-step velocity/pressure
//     fields (u_prev, v_prev, w_prev, p_prev) and the cached union CC
//     normals from the previous step (nx_cc, ny_cc, nz_cc).  Accumulate
//     12 force/torque channels for body b.
//
//  nu_rho_field: ν·ρ per cell (size=grid) or scalar (size=1).
//  delta_order: 1 (cosine) or 2 (Towers 2008 |∇φ| correction).
//  For delta_order=2 the SDF is re-sampled at 6 body-frame-shifted
//  positions to compute |∇φ|; no sparse_cc_flat needed.
//
//  Memory savings vs. two-phase:
//  • No sparse_cc_flat (per-body CC-SDF cache).
//  • No union-AABB stress/pforce tensors (xs/ys/zs/px/py/pz).
//  Trade-off: forces are one-step lagged (O(dt) for explicit Heun).
//
//  Force output (B,12) float64 must be pre-zeroed by the caller.
//  winning_rho_cc must be pre-filled with rho_fluid by the caller.
// =====================================================================

template <typename scalar_t>
__global__ void streaming_sdf_forces_fused_3d_multi_kernel(
    const int b,
    // Body SDF data
    const scalar_t* __restrict__ F_flat,
    const int64_t*  __restrict__ F_offsets,
    const int64_t*  __restrict__ body_shapes,   // [B,3] (Mx,My,Mz)
    const scalar_t* __restrict__ body_meta,     // [B,10]
    const scalar_t* __restrict__ kin,           // [B,21]
    // Per-body AABB
    const int64_t*  __restrict__ aabb_lo,       // [B,3]
    const int64_t*  __restrict__ aabb_dim,      // [B,3]
    // Full-grid coordinates
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const scalar_t* __restrict__ gz,
    const int Ngx, const int Ngy, const int Ngz,
    const scalar_t half_h,
    // Union SDF / body-velocity outputs (compare-swap)
    scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ sdf_u,
    scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ sdf_w,
    scalar_t* __restrict__ bU,
    scalar_t* __restrict__ bV,
    scalar_t* __restrict__ bW,
    const int interp_method,
    // Variable-density output
    const scalar_t* __restrict__ rho_bodies,    // [B] device pointer
    scalar_t* __restrict__ winning_rho_cc,      // [Ngx*Ngy*Ngz], pre-filled w/ rho_fluid
    // Prev-step fields for force computation (full-grid, read-only)
    const scalar_t* __restrict__ u_prev,
    const scalar_t* __restrict__ v_prev,
    const scalar_t* __restrict__ w_prev,
    const scalar_t* __restrict__ p_prev,
    // Prev-step CC normals (full-grid, read-only)
    const scalar_t* __restrict__ nx_cc,
    const scalar_t* __restrict__ ny_cc,
    const scalar_t* __restrict__ nz_cc,
    // ν·ρ: size=1 → scalar; size=grid → per-cell (variable viscosity)
    const scalar_t* __restrict__ nu_rho_field,
    const int64_t   nu_rho_field_size,
    const scalar_t inv_h,
    // Force parameters
    const scalar_t eps_body,
    const scalar_t eps_solver,
    const scalar_t h3,
    const int delta_order,
    // Force output (B,12) float64, pre-zeroed
    double* __restrict__ out)
{
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

        // ---- Body-frame setup (identical to streaming_sdf_min_3d_multi) ----
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
        const scalar_t lv_x = K[15], lv_y = K[16], lv_z = K[17];
        const scalar_t av_x = K[18], av_y = K[19], av_z = K[20];

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

        // ---- Phase C: SDF sampling and union field updates ----
        const scalar_t s_cc = sdf_sample_dispatch(
            interp_method, F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
            bxq, byq, bzq);

        // CC min update + variable-density winner tracking
        if (s_cc < sdf_cc[g_idx]) {
            sdf_cc[g_idx] = s_cc;
            winning_rho_cc[g_idx] = rho_bodies[b];
        }

        // Staggered face SDFs + body-velocity compare-swap
        {
            const scalar_t s = sdf_sample_dispatch(
                interp_method, F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
                bxq+du_x, byq+du_y, bzq+du_z);
            if (s < sdf_u[g_idx]) {
                sdf_u[g_idx] = s;
                bU[g_idx] = lv_x + av_y*(zc - cm_z) - av_z*(yc - cm_y);
            }
        }
        {
            const scalar_t s = sdf_sample_dispatch(
                interp_method, F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
                bxq+dv_x, byq+dv_y, bzq+dv_z);
            if (s < sdf_v[g_idx]) {
                sdf_v[g_idx] = s;
                bV[g_idx] = lv_y + av_z*(xc - cm_x) - av_x*(zc - cm_z);
            }
        }
        {
            const scalar_t s = sdf_sample_dispatch(
                interp_method, F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_,
                bxq+dw_x, byq+dw_y, bzq+dw_z);
            if (s < sdf_w[g_idx]) {
                sdf_w[g_idx] = s;
                bW[g_idx] = lv_z + av_x*(yc - cm_y) - av_y*(xc - cm_x);
            }
        }

        // ---- Phase D: inline force integration ----
        const scalar_t band_lo = (eps_solver - eps_body) < (-eps_body)
            ? (eps_solver - eps_body) : (-eps_body);
        const scalar_t band_hi = (eps_solver + eps_body) > (eps_body)
            ? (eps_solver + eps_body) : (eps_body);

        if (s_cc > band_lo && s_cc < band_hi) {
            // Variable or constant ν·ρ
            const scalar_t nu_rho_val = (nu_rho_field_size == 1)
                ? nu_rho_field[0] : nu_rho_field[g_idx];

            // Prev-step CC normals
            const scalar_t nx = nx_cc[g_idx];
            const scalar_t ny = ny_cc[g_idx];
            const scalar_t nz = nz_cc[g_idx];

            // Clamped stencil neighbours (boundary-safe)
            const int im1 = (i > 0)       ? i-1 : 0;
            const int ip1 = (i+1 < Ngx)   ? i+1 : i;   // forward-diff; 0 at wall
            const int jm1 = (j > 0)       ? j-1 : 0;
            const int jp1 = (j+1 < Ngy)   ? j+1 : j;
            const int km1 = (k > 0)       ? k-1 : 0;
            const int kp1 = (k+1 < Ngz)   ? k+1 : k;

            // Diagonal MAC derivatives (compact stagger-exact stencil)
            // u staggered along x: dudx[i] = (u[i+1] - u[i]) / h
            const scalar_t u_ijk = u_prev[(i   * Ngy + j) * Ngz + k];
            const scalar_t u_ip1 = u_prev[(ip1 * Ngy + j) * Ngz + k];
            const scalar_t dudx  = (u_ip1 - u_ijk) * inv_h;

            // v staggered along y: dvdy[j] = (v[j+1] - v[j]) / h
            const scalar_t v_ijk = v_prev[(i * Ngy + j   ) * Ngz + k];
            const scalar_t v_jp1 = v_prev[(i * Ngy + jp1 ) * Ngz + k];
            const scalar_t dvdy  = (v_jp1 - v_ijk) * inv_h;

            // w staggered along z: dwdz[k] = (w[k+1] - w[k]) / h
            const scalar_t w_ijk = w_prev[(i * Ngy + j) * Ngz + k   ];
            const scalar_t w_kp1 = w_prev[(i * Ngy + j) * Ngz + kp1 ];
            const scalar_t dwdz  = (w_kp1 - w_ijk) * inv_h;

            // Cross derivatives via CC-interpolated velocities
            // u_cc = 0.5*(u[i] + u[i+1]): interpolate u to cell centre
            // dudy = central diff of u_cc along j
            const scalar_t u_cc_jp1 = (scalar_t)0.5 * (
                u_prev[(i * Ngy + jp1) * Ngz + k] +
                u_prev[(ip1 * Ngy + jp1) * Ngz + k]);
            const scalar_t u_cc_jm1 = (scalar_t)0.5 * (
                u_prev[(i * Ngy + jm1) * Ngz + k] +
                u_prev[(ip1 * Ngy + jm1) * Ngz + k]);
            const scalar_t dudy = (u_cc_jp1 - u_cc_jm1) * (scalar_t)0.5 * inv_h;

            // dudz = central diff of u_cc along k
            const scalar_t u_cc_kp1 = (scalar_t)0.5 * (
                u_prev[(i * Ngy + j) * Ngz + kp1] +
                u_prev[(ip1 * Ngy + j) * Ngz + kp1]);
            const scalar_t u_cc_km1 = (scalar_t)0.5 * (
                u_prev[(i * Ngy + j) * Ngz + km1] +
                u_prev[(ip1 * Ngy + j) * Ngz + km1]);
            const scalar_t dudz = (u_cc_kp1 - u_cc_km1) * (scalar_t)0.5 * inv_h;

            // v_cc = 0.5*(v[j] + v[j+1]): interpolate v to cell centre
            // dvdx = central diff of v_cc along i
            const int jp1v = (j+1 < Ngy) ? j+1 : j;  // same as jp1
            const scalar_t v_cc_ip1 = (scalar_t)0.5 * (
                v_prev[(ip1 * Ngy + j   ) * Ngz + k] +
                v_prev[(ip1 * Ngy + jp1v) * Ngz + k]);
            const scalar_t v_cc_im1 = (scalar_t)0.5 * (
                v_prev[(im1 * Ngy + j   ) * Ngz + k] +
                v_prev[(im1 * Ngy + jp1v) * Ngz + k]);
            const scalar_t dvdx = (v_cc_ip1 - v_cc_im1) * (scalar_t)0.5 * inv_h;

            // dvdz = central diff of v_cc along k
            const scalar_t v_cc_kp1 = (scalar_t)0.5 * (
                v_prev[(i * Ngy + j   ) * Ngz + kp1] +
                v_prev[(i * Ngy + jp1v) * Ngz + kp1]);
            const scalar_t v_cc_km1 = (scalar_t)0.5 * (
                v_prev[(i * Ngy + j   ) * Ngz + km1] +
                v_prev[(i * Ngy + jp1v) * Ngz + km1]);
            const scalar_t dvdz = (v_cc_kp1 - v_cc_km1) * (scalar_t)0.5 * inv_h;

            // w_cc = 0.5*(w[k] + w[k+1]): interpolate w to cell centre
            // dwdx = central diff of w_cc along i
            const int kp1w = (k+1 < Ngz) ? k+1 : k;  // same as kp1
            const scalar_t w_cc_ip1 = (scalar_t)0.5 * (
                w_prev[(ip1 * Ngy + j) * Ngz + k   ] +
                w_prev[(ip1 * Ngy + j) * Ngz + kp1w]);
            const scalar_t w_cc_im1 = (scalar_t)0.5 * (
                w_prev[(im1 * Ngy + j) * Ngz + k   ] +
                w_prev[(im1 * Ngy + j) * Ngz + kp1w]);
            const scalar_t dwdx = (w_cc_ip1 - w_cc_im1) * (scalar_t)0.5 * inv_h;

            // dwdy = central diff of w_cc along j
            const scalar_t w_cc_jp1 = (scalar_t)0.5 * (
                w_prev[(i * Ngy + jp1) * Ngz + k   ] +
                w_prev[(i * Ngy + jp1) * Ngz + kp1w]);
            const scalar_t w_cc_jm1 = (scalar_t)0.5 * (
                w_prev[(i * Ngy + jm1) * Ngz + k   ] +
                w_prev[(i * Ngy + jm1) * Ngz + kp1w]);
            const scalar_t dwdy = (w_cc_jp1 - w_cc_jm1) * (scalar_t)0.5 * inv_h;

            // Viscous stress · normal: σ_{ij} n_j (summed over j)
            const scalar_t xs = nu_rho_val * (2*dudx*nx + (dudy+dvdx)*ny + (dudz+dwdx)*nz);
            const scalar_t ys = nu_rho_val * ((dvdx+dudy)*nx + 2*dvdy*ny + (dvdz+dwdy)*nz);
            const scalar_t zs = nu_rho_val * ((dwdx+dudz)*nx + (dwdy+dvdz)*ny + 2*dwdz*nz);

            // Pressure force density: -p * n
            const scalar_t p_c  = p_prev[g_idx];
            const scalar_t pxv  = -p_c * nx;
            const scalar_t pyv  = -p_c * ny;
            const scalar_t pzv  = -p_c * nz;

            // Smoothed delta kernels (1st-order cosine)
            const scalar_t pi_v      = (scalar_t)3.141592653589793;
            const scalar_t inv_2eps  = (scalar_t)0.5 / eps_body;
            const scalar_t pi_ov_eb  = pi_v / eps_body;

            scalar_t delta_visc = 0, delta_pres = 0;
            const scalar_t d_visc = s_cc - eps_solver;
            if (d_visc > -eps_body && d_visc < eps_body)
                delta_visc = ((scalar_t)1 + cos(pi_ov_eb * d_visc)) * inv_2eps;
            if (s_cc > -eps_body && s_cc < eps_body)
                delta_pres = ((scalar_t)1 + cos(pi_ov_eb * s_cc))   * inv_2eps;

            // Towers (2008) 2nd-order: δ_S = δ_ε(φ) / |∇φ|
            // Re-sample SDF at 6 world-shifted positions using body-frame
            // rotation columns to compute |∇φ| without sparse_cc_flat.
            if (delta_order == 2 && (delta_visc > 0 || delta_pres > 0)) {
                const scalar_t h_grid = (scalar_t)1.0 / inv_h;
                const scalar_t s_xp = sdf_sample_dispatch(
                    interp_method, F, Mx, My, Mz,
                    bx0, by0, bz0, idx_, idy_, idz_,
                    bxq+r00*h_grid, byq+r10*h_grid, bzq+r20*h_grid);
                const scalar_t s_xm = sdf_sample_dispatch(
                    interp_method, F, Mx, My, Mz,
                    bx0, by0, bz0, idx_, idy_, idz_,
                    bxq-r00*h_grid, byq-r10*h_grid, bzq-r20*h_grid);
                const scalar_t s_yp = sdf_sample_dispatch(
                    interp_method, F, Mx, My, Mz,
                    bx0, by0, bz0, idx_, idy_, idz_,
                    bxq+r01*h_grid, byq+r11*h_grid, bzq+r21*h_grid);
                const scalar_t s_ym = sdf_sample_dispatch(
                    interp_method, F, Mx, My, Mz,
                    bx0, by0, bz0, idx_, idy_, idz_,
                    bxq-r01*h_grid, byq-r11*h_grid, bzq-r21*h_grid);
                const scalar_t s_zp = sdf_sample_dispatch(
                    interp_method, F, Mx, My, Mz,
                    bx0, by0, bz0, idx_, idy_, idz_,
                    bxq+r02*h_grid, byq+r12*h_grid, bzq+r22*h_grid);
                const scalar_t s_zm = sdf_sample_dispatch(
                    interp_method, F, Mx, My, Mz,
                    bx0, by0, bz0, idx_, idy_, idz_,
                    bxq-r02*h_grid, byq-r12*h_grid, bzq-r22*h_grid);
                const scalar_t dsdx = (s_xp - s_xm) * (scalar_t)0.5 * inv_h;
                const scalar_t dsdy = (s_yp - s_ym) * (scalar_t)0.5 * inv_h;
                const scalar_t dsdz = (s_zp - s_zm) * (scalar_t)0.5 * inv_h;
                scalar_t grad_mag = sqrt(dsdx*dsdx + dsdy*dsdy + dsdz*dsdz);
                const scalar_t min_grad = (scalar_t)1e-3;
                if (grad_mag < min_grad) grad_mag = min_grad;
                const scalar_t inv_grad = (scalar_t)1.0 / grad_mag;
                delta_visc *= inv_grad;
                delta_pres *= inv_grad;
            }

            // Force / torque accumulation
            const scalar_t arm_x = xc - cm_x;
            const scalar_t arm_y = yc - cm_y;
            const scalar_t arm_z = zc - cm_z;

            const double fv_x = (double)(xs  * delta_visc);
            const double fv_y = (double)(ys  * delta_visc);
            const double fv_z = (double)(zs  * delta_visc);
            const double fp_x = (double)(pxv * delta_pres);
            const double fp_y = (double)(pyv * delta_pres);
            const double fp_z = (double)(pzv * delta_pres);

            acc[0]  = fv_x;
            acc[1]  = fv_y;
            acc[2]  = fv_z;
            acc[3]  = (double)arm_y * fv_z - (double)arm_z * fv_y;
            acc[4]  = (double)arm_z * fv_x - (double)arm_x * fv_z;
            acc[5]  = (double)arm_x * fv_y - (double)arm_y * fv_x;
            acc[6]  = fp_x;
            acc[7]  = fp_y;
            acc[8]  = fp_z;
            acc[9]  = (double)arm_y * fp_z - (double)arm_z * fp_y;
            acc[10] = (double)arm_z * fp_x - (double)arm_x * fp_z;
            acc[11] = (double)arm_x * fp_y - (double)arm_y * fp_x;
        }
    }

    // Block-reduction: 12 channels sequentially into shared memory
    extern __shared__ double sdata[];
    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;
    const double h3_d = (double)h3;

#pragma unroll
    for (int c = 0; c < 12; ++c) {
        sdata[tid] = acc[c];
        __syncthreads();
        for (int s = bdim >> 1; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) atomicAdd(&out[b*12 + c], sdata[0] * h3_d);
        __syncthreads();
    }
}

void streaming_sdf_forces_fused_3d_multi_cuda(
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
    const at::Tensor& rho_bodies,   // [B] float, per-body density
    at::Tensor winning_rho_cc,      // [Ngx*Ngy*Ngz], pre-filled w/ rho_fluid
    const at::Tensor& u_prev, const at::Tensor& v_prev,
    const at::Tensor& w_prev, const at::Tensor& p_prev,
    const at::Tensor& nx_cc,  const at::Tensor& ny_cc, const at::Tensor& nz_cc,
    const at::Tensor& nu_rho_field,   // size=1 scalar or size=grid per-cell
    const double eps_body,
    const double eps_solver,
    const double h3,
    const int64_t delta_order,
    at::Tensor out)   // (B, 12) float64, pre-zeroed
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngy = (int)gy.numel();
    const int Ngz = (int)gz.numel();
    const int Ngx = (int)gx.numel();
    const int blockSize = 256;
    const size_t shmem = (size_t)blockSize * sizeof(double);

    // ``rho_bodies`` stays on the device: the kernel reads
    // ``rho_bodies[b]`` directly, eliminating a per-step D2H copy
    // and the implicit synchronisation it forced on the CUDA stream.
    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(),
        "streaming_sdf_forces_fused_3d_multi_cuda", [&] {
        const scalar_t* rho_bodies_ptr = rho_bodies.data_ptr<scalar_t>();
        for (int b = 0; b < B; ++b) {
            const int nblocks = (int)((max_vol_per_body + blockSize - 1) / blockSize);
            streaming_sdf_forces_fused_3d_multi_kernel<scalar_t>
                <<<dim3(nblocks,1,1), dim3(blockSize,1,1), shmem, stream>>>(
                    b,
                    F_flat.data_ptr<scalar_t>(),
                    F_offsets.data_ptr<int64_t>(),
                    body_shapes.data_ptr<int64_t>(),
                    body_meta.data_ptr<scalar_t>(),
                    kin.data_ptr<scalar_t>(),
                    aabb_lo.data_ptr<int64_t>(),
                    aabb_dim.data_ptr<int64_t>(),
                    gx.data_ptr<scalar_t>(),
                    gy.data_ptr<scalar_t>(),
                    gz.data_ptr<scalar_t>(),
                    Ngx, Ngy, Ngz,
                    (scalar_t)(0.5 * h_grid),
                    sdf_cc.data_ptr<scalar_t>(),
                    sdf_u.data_ptr<scalar_t>(),
                    sdf_v.data_ptr<scalar_t>(),
                    sdf_w.data_ptr<scalar_t>(),
                    body_u.data_ptr<scalar_t>(),
                    body_v.data_ptr<scalar_t>(),
                    body_w.data_ptr<scalar_t>(),
                    (int)interp_method,
                    rho_bodies_ptr,
                    winning_rho_cc.data_ptr<scalar_t>(),
                    u_prev.data_ptr<scalar_t>(),
                    v_prev.data_ptr<scalar_t>(),
                    w_prev.data_ptr<scalar_t>(),
                    p_prev.data_ptr<scalar_t>(),
                    nx_cc.data_ptr<scalar_t>(),
                    ny_cc.data_ptr<scalar_t>(),
                    nz_cc.data_ptr<scalar_t>(),
                    nu_rho_field.data_ptr<scalar_t>(),
                    (int64_t)nu_rho_field.numel(),
                    (scalar_t)(1.0 / h_grid),
                    (scalar_t)eps_body,
                    (scalar_t)eps_solver,
                    (scalar_t)h3,
                    (int)delta_order,
                    out.data_ptr<double>());
        }
    });
}

// =====================================================================
//  apply_bcs_3d  (Phase H: fused boundary-condition writes)
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
//  device function used by streaming_sdf_min_3d (interp_method 0 =
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
    m.impl("streaming_sdf_min_3d", &streaming_sdf_min_3d_cuda);
    m.impl("streaming_sdf_min_3d_multi", &streaming_sdf_min_3d_multi_cuda);
    m.impl("bdim_forces_3d_multi", &bdim_forces_3d_multi_cuda);
    m.impl("streaming_sdf_forces_fused_3d_multi", &streaming_sdf_forces_fused_3d_multi_cuda);
    m.impl("apply_bcs_3d", &apply_bcs_3d_cuda);
    m.impl("interpolate_3d", &interpolate_3d_cuda);
}

}  // namespace lilytorch_kernels
