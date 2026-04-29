// =====================================================================
//  streaming_sdf_cpu.cpp
//
//  CPU implementations of the lilytorch_kernels operators. These mirror
//  the CUDA kernels in cuda/streaming_sdf.cu line-for-line so that the
//  ops can run on hosts without CUDA (testing, debugging, CPU-only
//  builds via LILYTORCH_NO_CUDA=1). The hot loops are parallelised with
//  OpenMP. Float dispatch covers float32 / float64; half precision is
//  intentionally skipped on CPU.
// =====================================================================

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/all.h>
#include <torch/library.h>
#include <c10/util/ArrayRef.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lilytorch_kernels {

template <typename scalar_t>
static inline scalar_t trilinear_sample_border(
    const scalar_t* F,
    const scalar_t* bx, const scalar_t* by, const scalar_t* bz,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t bx_last, const scalar_t by_last, const scalar_t bz_last,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    const scalar_t inv_vol,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    xq = std::max(bx0, std::min(xq, bx_last));
    yq = std::max(by0, std::min(yq, by_last));
    zq = std::max(bz0, std::min(zq, bz_last));

    int ix = (int)std::floor((xq - bx0) * inv_dx);
    int iy = (int)std::floor((yq - by0) * inv_dy);
    int iz = (int)std::floor((zq - bz0) * inv_dz);
    ix = std::max(0, std::min(ix, Mx - 2));
    iy = std::max(0, std::min(iy, My - 2));
    iz = std::max(0, std::min(iz, Mz - 2));
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
//  The streaming_sdf_min_3d* kernels are exclusively used with body SDF
//  tables built on uniform Cartesian grids (the kernel already takes
//  ``inv_dx``, ``inv_dy``, ``inv_dz``).  In that case the corner
//  weights reduce to ``(1 - frac, frac)`` per axis -- no axis-table
//  lookups, no ``floor`` calls, and the trailing ``* inv_vol`` cancels
//  out (because dx*dy*dz*inv_vol == 1).  This saves 6 axis-table reads
//  + 1 mul per sample (24 reads per cell across the 4 face samples)
//  vs. ``trilinear_sample_border`` and avoids the slow ``floor`` calls
//  on x86 (CPU).
//
//  Clamping is done in body-grid space (after the inv_dx scale) which
//  is one fewer subtraction than clamping ``xq`` against ``bx_last``.
//
//  The factored evaluation form (wx0*(... wy0*(...) + wy1*(...))) cuts
//  the multiply count from ~24 to ~14 vs. the fully-expanded sum that
//  ``trilinear_sample_border`` uses.
// =====================================================================
template <typename scalar_t>
static inline scalar_t trilinear_sample_uniform(
    const scalar_t* F,
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
    if (tx < (scalar_t)0) tx = (scalar_t)0; else if (tx > Mx_lim) tx = Mx_lim;
    if (ty < (scalar_t)0) ty = (scalar_t)0; else if (ty > My_lim) ty = My_lim;
    if (tz < (scalar_t)0) tz = (scalar_t)0; else if (tz > Mz_lim) tz = Mz_lim;

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
//  Per-cell update body shared by single- and multi-body launchers.
//
//  Each cell is touched once per launch -> no atomics needed (matches
//  the CUDA kernel).
//
//  Optimisation (rotation CSE): the 4 sample positions (cc + 3 face
//  staggers) differ in world space only by ``±h/2`` along ONE world
//  axis.  In body frame, a world offset ``(δx, 0, 0)`` becomes
//  ``δx * col0(R_T)``.  So the body-frame face points are just
//  ``body_cc + Δ_k`` where ``Δ_k = -half_h * col_k(R_T)`` is a per-body
//  constant.  We compute the body-frame CC point once (one matrix-vector
//  multiply: 9 muls + 6 adds) and derive the 3 face points by adding
//  precomputed body-frame deltas (3 adds each).  Saves 27 muls + 18 adds
//  per cell vs. doing 4 full rotations.
// =====================================================================
template <typename scalar_t>
static inline void update_cell(
    const scalar_t* F,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    const scalar_t r00, const scalar_t r01, const scalar_t r02,
    const scalar_t r10, const scalar_t r11, const scalar_t r12,
    const scalar_t r20, const scalar_t r21, const scalar_t r22,
    const scalar_t bp_x, const scalar_t bp_y, const scalar_t bp_z,
    const scalar_t cm_x, const scalar_t cm_y, const scalar_t cm_z,
    const scalar_t lv_x, const scalar_t lv_y, const scalar_t lv_z,
    const scalar_t av_x, const scalar_t av_y, const scalar_t av_z,
    // Body-frame face deltas (precomputed once per body):
    //   Δ_u = -half_h * col0(R_T), Δ_v = -half_h * col1(R_T), Δ_w = -half_h * col2(R_T)
    const scalar_t du_x, const scalar_t du_y, const scalar_t du_z,
    const scalar_t dv_x, const scalar_t dv_y, const scalar_t dv_z,
    const scalar_t dw_x, const scalar_t dw_y, const scalar_t dw_z,
    const scalar_t xc, const scalar_t yc, const scalar_t zc,
    const std::int64_t g_idx, const std::int64_t sparse_idx,
    scalar_t* sdf_cc, scalar_t* sdf_u, scalar_t* sdf_v, scalar_t* sdf_w,
    scalar_t* bU, scalar_t* bV, scalar_t* bW,
    scalar_t* sparse_cc)
{
    // Body-frame CC point (single rotation; the 3 face points are derived
    // from it by precomputed deltas).
    const scalar_t dxw = xc - bp_x, dyw = yc - bp_y, dzw = zc - bp_z;
    const scalar_t bxq = r00*dxw + r01*dyw + r02*dzw;
    const scalar_t byq = r10*dxw + r11*dyw + r12*dzw;
    const scalar_t bzq = r20*dxw + r21*dyw + r22*dzw;

    // ---- cc ----
    {
        const scalar_t s = trilinear_sample_uniform<scalar_t>(
            F, Mx, My, Mz, bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
            bxq, byq, bzq);
        sparse_cc[sparse_idx] = s;
        if (s < sdf_cc[g_idx]) sdf_cc[g_idx] = s;
    }
    // ---- u-face ----
    {
        const scalar_t s = trilinear_sample_uniform<scalar_t>(
            F, Mx, My, Mz, bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
            bxq + du_x, byq + du_y, bzq + du_z);
        if (s < sdf_u[g_idx]) {
            sdf_u[g_idx] = s;
            bU[g_idx] = lv_x + av_y*(zc - cm_z) - av_z*(yc - cm_y);
        }
    }
    // ---- v-face ----
    {
        const scalar_t s = trilinear_sample_uniform<scalar_t>(
            F, Mx, My, Mz, bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
            bxq + dv_x, byq + dv_y, bzq + dv_z);
        if (s < sdf_v[g_idx]) {
            sdf_v[g_idx] = s;
            bV[g_idx] = lv_y + av_z*(xc - cm_x) - av_x*(zc - cm_z);
        }
    }
    // ---- w-face ----
    {
        const scalar_t s = trilinear_sample_uniform<scalar_t>(
            F, Mx, My, Mz, bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
            bxq + dw_x, byq + dw_y, bzq + dw_z);
        if (s < sdf_w[g_idx]) {
            sdf_w[g_idx] = s;
            bW[g_idx] = lv_z + av_x*(yc - cm_y) - av_y*(xc - cm_x);
        }
    }
}

// =====================================================================
//  streaming_sdf_min_3d (single body)
// =====================================================================

void streaming_sdf_min_3d_cpu(
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
    at::Tensor sparse_cc)
{
    TORCH_CHECK(R_T.size()      == 9, "R_T must have 9 elements");
    TORCH_CHECK(body_pos.size() == 3, "body_pos must have 3 elements");
    TORCH_CHECK(com_pos.size()  == 3, "com_pos must have 3 elements");
    TORCH_CHECK(lin_vel.size()  == 3, "lin_vel must have 3 elements");
    TORCH_CHECK(ang_vel.size()  == 3, "ang_vel must have 3 elements");

    const int Ai = (int)(i1 - i0);
    const int Aj = (int)(j1 - j0);
    const int Ak = (int)(k1 - k0);
    const int N  = Ai * Aj * Ak;
    if (N <= 0) return;

    const int Mx  = (int)bx.numel();
    const int My  = (int)by.numel();
    const int Mz  = (int)bz.numel();
    const int Ngy = (int)gy.numel();
    const int Ngz = (int)gz.numel();

    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "streaming_sdf_min_3d_cpu", [&] {
        auto F_c  = F.contiguous();
        auto gx_c = gx.contiguous();
        auto gy_c = gy.contiguous();
        auto gz_c = gz.contiguous();

        // Body axis tables (bx/by/bz) and ``inv_vol`` / ``bx_last`` are no
        // longer needed: ``trilinear_sample_uniform`` infers corner weights
        // from (bx0, inv_dx) for uniform body grids.
        const scalar_t* F_ptr  = F_c.data_ptr<scalar_t>();
        const scalar_t* gx_ptr = gx_c.data_ptr<scalar_t>();
        const scalar_t* gy_ptr = gy_c.data_ptr<scalar_t>();
        const scalar_t* gz_ptr = gz_c.data_ptr<scalar_t>();

        scalar_t* sdf_cc_ptr = sdf_cc.data_ptr<scalar_t>();
        scalar_t* sdf_u_ptr  = sdf_u.data_ptr<scalar_t>();
        scalar_t* sdf_v_ptr  = sdf_v.data_ptr<scalar_t>();
        scalar_t* sdf_w_ptr  = sdf_w.data_ptr<scalar_t>();
        scalar_t* bU_ptr     = body_u.data_ptr<scalar_t>();
        scalar_t* bV_ptr     = body_v.data_ptr<scalar_t>();
        scalar_t* bW_ptr     = body_w.data_ptr<scalar_t>();
        scalar_t* sp_ptr     = sparse_cc.data_ptr<scalar_t>();

        const scalar_t bx0_s = (scalar_t)bx0, by0_s = (scalar_t)by0, bz0_s = (scalar_t)bz0;
        const scalar_t idx_s = (scalar_t)inv_dx, idy_s = (scalar_t)inv_dy, idz_s = (scalar_t)inv_dz;
        const scalar_t r00 = (scalar_t)R_T[0], r01 = (scalar_t)R_T[1], r02 = (scalar_t)R_T[2];
        const scalar_t r10 = (scalar_t)R_T[3], r11 = (scalar_t)R_T[4], r12 = (scalar_t)R_T[5];
        const scalar_t r20 = (scalar_t)R_T[6], r21 = (scalar_t)R_T[7], r22 = (scalar_t)R_T[8];
        const scalar_t bp_x = (scalar_t)body_pos[0], bp_y = (scalar_t)body_pos[1], bp_z = (scalar_t)body_pos[2];
        const scalar_t cm_x = (scalar_t)com_pos[0],  cm_y = (scalar_t)com_pos[1],  cm_z = (scalar_t)com_pos[2];
        const scalar_t lv_x = (scalar_t)lin_vel[0],  lv_y = (scalar_t)lin_vel[1],  lv_z = (scalar_t)lin_vel[2];
        const scalar_t av_x = (scalar_t)ang_vel[0],  av_y = (scalar_t)ang_vel[1],  av_z = (scalar_t)ang_vel[2];
        const scalar_t neg_half_h = -(scalar_t)(0.5 * h_grid);
        // Body-frame face deltas: Δ_k = -half_h * col_k(R_T).
        const scalar_t du_x = neg_half_h * r00, du_y = neg_half_h * r10, du_z = neg_half_h * r20;
        const scalar_t dv_x = neg_half_h * r01, dv_y = neg_half_h * r11, dv_z = neg_half_h * r21;
        const scalar_t dw_x = neg_half_h * r02, dw_y = neg_half_h * r12, dw_z = neg_half_h * r22;
        const int i0_i = (int)i0, j0_i = (int)j0, k0_i = (int)k0;

        _Pragma("omp parallel for schedule(static)")
        for (int tid = 0; tid < N; ++tid) {
            const int di  = tid / (Aj * Ak);
            const int rem = tid - di * (Aj * Ak);
            const int dj  = rem / Ak;
            const int dk  = rem - dj * Ak;
            const int i   = i0_i + di;
            const int j   = j0_i + dj;
            const int k   = k0_i + dk;
            const std::int64_t g_idx =
                ((std::int64_t)i * Ngy + j) * Ngz + k;

            update_cell<scalar_t>(
                F_ptr, Mx, My, Mz,
                bx0_s, by0_s, bz0_s,
                idx_s, idy_s, idz_s,
                r00, r01, r02, r10, r11, r12, r20, r21, r22,
                bp_x, bp_y, bp_z, cm_x, cm_y, cm_z,
                lv_x, lv_y, lv_z, av_x, av_y, av_z,
                du_x, du_y, du_z, dv_x, dv_y, dv_z, dw_x, dw_y, dw_z,
                gx_ptr[i], gy_ptr[j], gz_ptr[k],
                g_idx, /*sparse_idx=*/(std::int64_t)tid,
                sdf_cc_ptr, sdf_u_ptr, sdf_v_ptr, sdf_w_ptr,
                bU_ptr, bV_ptr, bW_ptr, sp_ptr);
        }
    });
}

// =====================================================================
//  streaming_sdf_min_3d_multi
//
//  Bodies are processed serially (matches CUDA -- no atomics required
//  because each cell is touched once per body, and bodies progress in
//  order); cells within a body are parallelised.
// =====================================================================

void streaming_sdf_min_3d_multi_cpu(
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
    const int64_t /*max_vol_per_body*/,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v, at::Tensor sdf_w,
    at::Tensor body_u, at::Tensor body_v, at::Tensor body_w,
    at::Tensor sparse_cc_flat)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0) return;

    const int Ngy = (int)gy.numel();
    const int Ngz = (int)gz.numel();

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_min_3d_multi_cpu", [&] {
        auto F_c  = F_flat.contiguous();
        auto gx_c = gx.contiguous();
        auto gy_c = gy.contiguous();
        auto gz_c = gz.contiguous();

        // Body axis tables (bx_flat / by_flat / bz_flat) and the per-body
        // ``inv_vol`` / ``bxL`` / ``byL`` / ``bzL`` entries of ``body_meta``
        // are no longer read: ``trilinear_sample_uniform`` infers corner
        // weights analytically from (bx0, inv_dx) on the uniform body
        // grids that BDIM constructs.  The op signature still accepts
        // them for backwards compatibility (callers pre-allocate them
        // anyway) -- they're simply unused by the new kernel.
        const scalar_t* F_ptr   = F_c.data_ptr<scalar_t>();
        const int64_t*  F_off   = F_offsets.data_ptr<int64_t>();
        const int64_t*  shapes  = body_shapes.data_ptr<int64_t>();
        const scalar_t* meta    = body_meta.data_ptr<scalar_t>();
        const scalar_t* kin_ptr = kin.data_ptr<scalar_t>();
        const int64_t*  lo      = aabb_lo.data_ptr<int64_t>();
        const int64_t*  dim_    = aabb_dim.data_ptr<int64_t>();
        const int64_t*  cell_off= cell_offsets.data_ptr<int64_t>();
        const scalar_t* gx_ptr  = gx_c.data_ptr<scalar_t>();
        const scalar_t* gy_ptr  = gy_c.data_ptr<scalar_t>();
        const scalar_t* gz_ptr  = gz_c.data_ptr<scalar_t>();

        scalar_t* sdf_cc_p = sdf_cc.data_ptr<scalar_t>();
        scalar_t* sdf_u_p  = sdf_u.data_ptr<scalar_t>();
        scalar_t* sdf_v_p  = sdf_v.data_ptr<scalar_t>();
        scalar_t* sdf_w_p  = sdf_w.data_ptr<scalar_t>();
        scalar_t* bU_p     = body_u.data_ptr<scalar_t>();
        scalar_t* bV_p     = body_v.data_ptr<scalar_t>();
        scalar_t* bW_p     = body_w.data_ptr<scalar_t>();
        scalar_t* sparse_p = sparse_cc_flat.data_ptr<scalar_t>();

        const scalar_t neg_half_h = -(scalar_t)(0.5 * h_grid);

        for (int b = 0; b < B; ++b) {
            const int Ai = (int)dim_[b*3 + 0];
            const int Aj = (int)dim_[b*3 + 1];
            const int Ak = (int)dim_[b*3 + 2];
            const int vol = Ai * Aj * Ak;
            if (vol <= 0) continue;

            const int i0_b = (int)lo[b*3 + 0];
            const int j0_b = (int)lo[b*3 + 1];
            const int k0_b = (int)lo[b*3 + 2];

            const scalar_t* F_b  = F_ptr  + F_off[b];
            const int Mx = (int)shapes[b*3 + 0];
            const int My = (int)shapes[b*3 + 1];
            const int Mz = (int)shapes[b*3 + 2];
            // bx_ptr/by_ptr/bz_ptr axis tables and bxL/byL/bzL/inv_vol from
            // ``body_meta`` are no longer needed: ``trilinear_sample_uniform``
            // computes weights analytically from (bx0, inv_dx) for uniform
            // body grids.

            const scalar_t* M = meta + b*10;
            const scalar_t bx0 = M[0], by0 = M[1], bz0 = M[2];
            const scalar_t idx_ = M[6], idy_ = M[7], idz_ = M[8];

            const scalar_t* K = kin_ptr + b*21;
            const scalar_t r00 = K[0],  r01 = K[1],  r02 = K[2];
            const scalar_t r10 = K[3],  r11 = K[4],  r12 = K[5];
            const scalar_t r20 = K[6],  r21 = K[7],  r22 = K[8];
            const scalar_t bp_x = K[9],  bp_y = K[10], bp_z = K[11];
            const scalar_t cm_x = K[12], cm_y = K[13], cm_z = K[14];
            const scalar_t lv_x = K[15], lv_y = K[16], lv_z = K[17];
            const scalar_t av_x = K[18], av_y = K[19], av_z = K[20];

            // Body-frame face deltas: Δ_k = -half_h * col_k(R_T).
            const scalar_t du_x = neg_half_h * r00, du_y = neg_half_h * r10, du_z = neg_half_h * r20;
            const scalar_t dv_x = neg_half_h * r01, dv_y = neg_half_h * r11, dv_z = neg_half_h * r21;
            const scalar_t dw_x = neg_half_h * r02, dw_y = neg_half_h * r12, dw_z = neg_half_h * r22;

            const std::int64_t sparse_base = (std::int64_t)cell_off[b];

            _Pragma("omp parallel for schedule(static)")
            for (int local = 0; local < vol; ++local) {
                const int di  = local / (Aj * Ak);
                const int rem = local - di * (Aj * Ak);
                const int dj  = rem / Ak;
                const int dk  = rem - dj * Ak;
                const int i   = i0_b + di;
                const int j   = j0_b + dj;
                const int k   = k0_b + dk;
                const std::int64_t g_idx =
                    ((std::int64_t)i * Ngy + j) * Ngz + k;

                update_cell<scalar_t>(
                    F_b, Mx, My, Mz,
                    bx0, by0, bz0,
                    idx_, idy_, idz_,
                    r00, r01, r02, r10, r11, r12, r20, r21, r22,
                    bp_x, bp_y, bp_z, cm_x, cm_y, cm_z,
                    lv_x, lv_y, lv_z, av_x, av_y, av_z,
                    du_x, du_y, du_z, dv_x, dv_y, dv_z, dw_x, dw_y, dw_z,
                    gx_ptr[i], gy_ptr[j], gz_ptr[k],
                    g_idx, sparse_base + (std::int64_t)local,
                    sdf_cc_p, sdf_u_p, sdf_v_p, sdf_w_p,
                    bU_p, bV_p, bW_p, sparse_p);
            }
        }
    });
}

// =====================================================================
//  bdim_forces_3d_multi
//
//  Per-body: parallel reduction over cells into 12 float64 channels,
//  then accumulated (scaled by h3) into out[b, 0..11]. The CUDA path
//  uses atomicAdd on out (i.e. += into existing storage); we mirror
//  that here so callers may invoke repeatedly with a pre-existing
//  accumulator.
//
//  Reads the per-body cc-SDF directly from the cache populated by
//  streaming_sdf_min_3d_multi (`sparse_cc_flat` + `cell_offsets`),
//  avoiding a second trilinear interpolation per cell.
// =====================================================================

void bdim_forces_3d_multi_cpu(
    const at::Tensor& sparse_cc_flat,
    const at::Tensor& cell_offsets,
    const at::Tensor& kin,
    const at::Tensor& aabb_lo,
    const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy, const at::Tensor& gz,
    const int64_t u_i0, const int64_t u_j0, const int64_t u_k0,
    const int64_t Sj, const int64_t Sk,
    const at::Tensor& xs, const at::Tensor& ys, const at::Tensor& zs,
    const at::Tensor& px, const at::Tensor& py, const at::Tensor& pz,
    const double eps_body, const double eps_solver, const double h3,
    const int64_t /*max_vol_per_body*/,
    at::Tensor out)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0) return;

    TORCH_CHECK(out.scalar_type() == at::kDouble,
                "bdim_forces_3d_multi_cpu: out must be float64");

    AT_DISPATCH_FLOATING_TYPES(sparse_cc_flat.scalar_type(), "bdim_forces_3d_multi_cpu", [&] {
        auto sp_c = sparse_cc_flat.contiguous();
        auto gx_c = gx.contiguous();
        auto gy_c = gy.contiguous();
        auto gz_c = gz.contiguous();
        auto xs_c = xs.contiguous();
        auto ys_c = ys.contiguous();
        auto zs_c = zs.contiguous();
        auto px_c = px.contiguous();
        auto py_c = py.contiguous();
        auto pz_c = pz.contiguous();

        const scalar_t* sp_ptr  = sp_c.data_ptr<scalar_t>();
        const int64_t*  cell_off= cell_offsets.data_ptr<int64_t>();
        const scalar_t* kin_ptr = kin.data_ptr<scalar_t>();
        const int64_t*  lo      = aabb_lo.data_ptr<int64_t>();
        const int64_t*  dim_    = aabb_dim.data_ptr<int64_t>();
        const scalar_t* gx_ptr  = gx_c.data_ptr<scalar_t>();
        const scalar_t* gy_ptr  = gy_c.data_ptr<scalar_t>();
        const scalar_t* gz_ptr  = gz_c.data_ptr<scalar_t>();
        const scalar_t* xs_ptr  = xs_c.data_ptr<scalar_t>();
        const scalar_t* ys_ptr  = ys_c.data_ptr<scalar_t>();
        const scalar_t* zs_ptr  = zs_c.data_ptr<scalar_t>();
        const scalar_t* px_ptr  = px_c.data_ptr<scalar_t>();
        const scalar_t* py_ptr  = py_c.data_ptr<scalar_t>();
        const scalar_t* pz_ptr  = pz_c.data_ptr<scalar_t>();
        double*         out_ptr = out.data_ptr<double>();

        const scalar_t eps_b      = (scalar_t)eps_body;
        const scalar_t eps_s      = (scalar_t)eps_solver;
        const scalar_t pi_v       = (scalar_t)3.141592653589793;
        const scalar_t inv_2eps   = (scalar_t)0.5 / eps_b;
        const scalar_t pi_over_eb = pi_v / eps_b;
        const double   h3_d       = (double)h3;

        // The smoothed deltas vanish unless sdf is in
        //   visc-band:  eps_s - eps_b < sdf < eps_s + eps_b   (delta_visc)
        //   pres-band:  -eps_b        < sdf < eps_b           (delta_pres)
        // Their union is [min(-eps_b, eps_s-eps_b), max(eps_b, eps_s+eps_b)].
        // Outside this union all 12 contributions are zero, so we skip
        // the cell entirely (no stress / pforce reads, no cos calls).
        // This is the dominant cost saving at low-N: typical body AABBs
        // have only ~5–15% of cells in the band; the rest are deep
        // inside / outside the body.
        const scalar_t band_lo = (eps_s - eps_b) < (-eps_b) ? (eps_s - eps_b) : (-eps_b);
        const scalar_t band_hi = (eps_s + eps_b) > ( eps_b) ? (eps_s + eps_b) : ( eps_b);

        const int u_i0_i = (int)u_i0, u_j0_i = (int)u_j0, u_k0_i = (int)u_k0;
        const int Sj_i   = (int)Sj,   Sk_i   = (int)Sk;

        // ------------------------------------------------------------
        //  Single fused OpenMP parallel region across ALL bodies.
        //  Replaces the previous per-body parallel-region + critical
        //  pattern (B× fork/join + B× critical merges).  Each thread
        //  keeps a private (B, 12) accumulator and merges once at the
        //  end of the region.  At low-B this saves O(B) × OpenMP
        //  fork/join overhead (~10–50 µs each) per call.
        // ------------------------------------------------------------
        const int B_local = B;
        std::vector<double> accs(static_cast<size_t>(B_local) * 12, 0.0);

        _Pragma("omp parallel")
        {
            // Per-thread (B, 12) accumulator on the stack-equivalent.
            std::vector<double> local(static_cast<size_t>(B_local) * 12, 0.0);

            for (int b = 0; b < B_local; ++b) {
                const int Ai = (int)dim_[b*3 + 0];
                const int Aj = (int)dim_[b*3 + 1];
                const int Ak = (int)dim_[b*3 + 2];
                const int vol = Ai * Aj * Ak;
                if (vol <= 0) continue;

                const int i0_b = (int)lo[b*3 + 0];
                const int j0_b = (int)lo[b*3 + 1];
                const int k0_b = (int)lo[b*3 + 2];

                // Only the COM (kin[12..14]) is needed for force/torque arms.
                const scalar_t* K = kin_ptr + b*21;
                const scalar_t cm_x = K[12], cm_y = K[13], cm_z = K[14];

                const std::int64_t sparse_base = (std::int64_t)cell_off[b];
                double* lb = local.data() + (size_t)b * 12;

                _Pragma("omp for nowait schedule(static)")
                for (int idx = 0; idx < vol; ++idx) {
                    // Read cached per-body cc-SDF first; band test gates
                    // every other read & flop in the loop body.
                    const scalar_t sdf = sp_ptr[sparse_base + (std::int64_t)idx];

                    // Early skip: outside both delta bands ⇒ contribution = 0.
                    if (sdf <= band_lo || sdf >= band_hi) continue;

                    const int di  = idx / (Aj * Ak);
                    const int rem = idx - di * (Aj * Ak);
                    const int dj  = rem / Ak;
                    const int dk  = rem - dj * Ak;

                    scalar_t delta_visc = 0;
                    const scalar_t d_visc = sdf - eps_s;
                    if (d_visc > -eps_b && d_visc < eps_b) {
                        delta_visc = ((scalar_t)1 + std::cos(pi_over_eb * d_visc)) * inv_2eps;
                    }
                    scalar_t delta_pres = 0;
                    if (sdf > -eps_b && sdf < eps_b) {
                        delta_pres = ((scalar_t)1 + std::cos(pi_over_eb * sdf)) * inv_2eps;
                    }
                    // Both deltas zero (can happen on the band edge) ⇒ skip.
                    if (delta_visc == (scalar_t)0 && delta_pres == (scalar_t)0) continue;

                    const int i = i0_b + di, j = j0_b + dj, k = k0_b + dk;
                    const int sub_i = i - u_i0_i;
                    const int sub_j = j - u_j0_i;
                    const int sub_k = k - u_k0_i;
                    const std::int64_t s_idx =
                        ((std::int64_t)sub_i * Sj_i + sub_j) * Sk_i + sub_k;

                    const scalar_t xc = gx_ptr[i];
                    const scalar_t yc = gy_ptr[j];
                    const scalar_t zc = gz_ptr[k];

                    const scalar_t arm_x = xc - cm_x;
                    const scalar_t arm_y = yc - cm_y;
                    const scalar_t arm_z = zc - cm_z;

                    // Hoisted reads: only loaded for in-band cells.
                    const double fv_x = (double)(xs_ptr[s_idx] * delta_visc);
                    const double fv_y = (double)(ys_ptr[s_idx] * delta_visc);
                    const double fv_z = (double)(zs_ptr[s_idx] * delta_visc);
                    const double fp_x = (double)(px_ptr[s_idx] * delta_pres);
                    const double fp_y = (double)(py_ptr[s_idx] * delta_pres);
                    const double fp_z = (double)(pz_ptr[s_idx] * delta_pres);

                    lb[0]  += fv_x;
                    lb[1]  += fv_y;
                    lb[2]  += fv_z;
                    lb[3]  += (double)arm_y * fv_z - (double)arm_z * fv_y;
                    lb[4]  += (double)arm_z * fv_x - (double)arm_x * fv_z;
                    lb[5]  += (double)arm_x * fv_y - (double)arm_y * fv_x;
                    lb[6]  += fp_x;
                    lb[7]  += fp_y;
                    lb[8]  += fp_z;
                    lb[9]  += (double)arm_y * fp_z - (double)arm_z * fp_y;
                    lb[10] += (double)arm_z * fp_x - (double)arm_x * fp_z;
                    lb[11] += (double)arm_x * fp_y - (double)arm_y * fp_x;
                }
                // Implicit barrier handled by next ``omp for``; ``nowait``
                // applies to the inner loop so threads can move on to the
                // next body without synchronising — only the final reduction
                // (below) needs synchronisation.
            }

            _Pragma("omp critical")
            {
                for (size_t c = 0; c < accs.size(); ++c) accs[c] += local[c];
            }
        }

        for (int b = 0; b < B_local; ++b) {
            for (int c = 0; c < 12; ++c) {
                out_ptr[b*12 + c] += accs[(size_t)b * 12 + c] * h3_d;
            }
        }
    });
}

// =====================================================================
//  bdim_forces_3d_multi_legacy_resample (legacy: re-samples cc-SDF on the fly)
//
//  Per-body: parallel reduction over cells into 12 float64 channels,
//  then accumulated (scaled by h3) into out[b, 0..11]. The CUDA path
//  uses atomicAdd on out (i.e. += into existing storage); we mirror
//  that here so callers may invoke repeatedly with a pre-existing
//  accumulator.
// =====================================================================

void bdim_forces_3d_multi_legacy_resample_cpu(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& bx_flat, const at::Tensor& bx_offsets,
    const at::Tensor& by_flat, const at::Tensor& by_offsets,
    const at::Tensor& bz_flat, const at::Tensor& bz_offsets,
    const at::Tensor& body_shapes,
    const at::Tensor& body_meta,
    const at::Tensor& kin,
    const at::Tensor& aabb_lo,
    const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy, const at::Tensor& gz,
    const int64_t u_i0, const int64_t u_j0, const int64_t u_k0,
    const int64_t Sj, const int64_t Sk,
    const at::Tensor& xs, const at::Tensor& ys, const at::Tensor& zs,
    const at::Tensor& px, const at::Tensor& py, const at::Tensor& pz,
    const double eps_body, const double eps_solver, const double h3,
    const int64_t /*max_vol_per_body*/,
    at::Tensor out)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0) return;

    TORCH_CHECK(out.scalar_type() == at::kDouble,
                "bdim_forces_3d_multi_legacy_resample_cpu: out must be float64");

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "bdim_forces_3d_multi_legacy_resample_cpu", [&] {
        auto F_c  = F_flat.contiguous();
        auto bx_c = bx_flat.contiguous();
        auto by_c = by_flat.contiguous();
        auto bz_c = bz_flat.contiguous();
        auto gx_c = gx.contiguous();
        auto gy_c = gy.contiguous();
        auto gz_c = gz.contiguous();
        auto xs_c = xs.contiguous();
        auto ys_c = ys.contiguous();
        auto zs_c = zs.contiguous();
        auto px_c = px.contiguous();
        auto py_c = py.contiguous();
        auto pz_c = pz.contiguous();

        const scalar_t* F_ptr   = F_c.data_ptr<scalar_t>();
        const scalar_t* bx_ptr  = bx_c.data_ptr<scalar_t>();
        const scalar_t* by_ptr  = by_c.data_ptr<scalar_t>();
        const scalar_t* bz_ptr  = bz_c.data_ptr<scalar_t>();
        const int64_t*  F_off   = F_offsets.data_ptr<int64_t>();
        const int64_t*  bx_off  = bx_offsets.data_ptr<int64_t>();
        const int64_t*  by_off  = by_offsets.data_ptr<int64_t>();
        const int64_t*  bz_off  = bz_offsets.data_ptr<int64_t>();
        const int64_t*  shapes  = body_shapes.data_ptr<int64_t>();
        const scalar_t* meta    = body_meta.data_ptr<scalar_t>();
        const scalar_t* kin_ptr = kin.data_ptr<scalar_t>();
        const int64_t*  lo      = aabb_lo.data_ptr<int64_t>();
        const int64_t*  dim_    = aabb_dim.data_ptr<int64_t>();
        const scalar_t* gx_ptr  = gx_c.data_ptr<scalar_t>();
        const scalar_t* gy_ptr  = gy_c.data_ptr<scalar_t>();
        const scalar_t* gz_ptr  = gz_c.data_ptr<scalar_t>();
        const scalar_t* xs_ptr  = xs_c.data_ptr<scalar_t>();
        const scalar_t* ys_ptr  = ys_c.data_ptr<scalar_t>();
        const scalar_t* zs_ptr  = zs_c.data_ptr<scalar_t>();
        const scalar_t* px_ptr  = px_c.data_ptr<scalar_t>();
        const scalar_t* py_ptr  = py_c.data_ptr<scalar_t>();
        const scalar_t* pz_ptr  = pz_c.data_ptr<scalar_t>();
        double*         out_ptr = out.data_ptr<double>();

        const scalar_t eps_b    = (scalar_t)eps_body;
        const scalar_t eps_s    = (scalar_t)eps_solver;
        const scalar_t pi_v     = (scalar_t)3.141592653589793;
        const scalar_t inv_2eps = (scalar_t)0.5 / eps_b;
        const double   h3_d     = (double)h3;

        const int u_i0_i = (int)u_i0, u_j0_i = (int)u_j0, u_k0_i = (int)u_k0;
        const int Sj_i   = (int)Sj,   Sk_i   = (int)Sk;

        for (int b = 0; b < B; ++b) {
            const int Ai = (int)dim_[b*3 + 0];
            const int Aj = (int)dim_[b*3 + 1];
            const int Ak = (int)dim_[b*3 + 2];
            const int vol = Ai * Aj * Ak;
            if (vol <= 0) continue;

            const int i0_b = (int)lo[b*3 + 0];
            const int j0_b = (int)lo[b*3 + 1];
            const int k0_b = (int)lo[b*3 + 2];

            const scalar_t* F_b  = F_ptr  + F_off[b];
            const scalar_t* bx_b = bx_ptr + bx_off[b];
            const scalar_t* by_b = by_ptr + by_off[b];
            const scalar_t* bz_b = bz_ptr + bz_off[b];
            const int Mx = (int)shapes[b*3 + 0];
            const int My = (int)shapes[b*3 + 1];
            const int Mz = (int)shapes[b*3 + 2];

            const scalar_t* M = meta + b*10;
            const scalar_t bx0 = M[0], by0 = M[1], bz0 = M[2];
            const scalar_t bxL = M[3], byL = M[4], bzL = M[5];
            const scalar_t idx_ = M[6], idy_ = M[7], idz_ = M[8];
            const scalar_t inv_vol = M[9];

            const scalar_t* K = kin_ptr + b*21;
            const scalar_t r00 = K[0],  r01 = K[1],  r02 = K[2];
            const scalar_t r10 = K[3],  r11 = K[4],  r12 = K[5];
            const scalar_t r20 = K[6],  r21 = K[7],  r22 = K[8];
            const scalar_t bp_x = K[9],  bp_y = K[10], bp_z = K[11];
            const scalar_t cm_x = K[12], cm_y = K[13], cm_z = K[14];

            double accs[12] = {0,0,0,0,0,0,0,0,0,0,0,0};

            _Pragma("omp parallel")
            {
                double local[12] = {0,0,0,0,0,0,0,0,0,0,0,0};

                _Pragma("omp for nowait schedule(static)")
                for (int idx = 0; idx < vol; ++idx) {
                    const int di  = idx / (Aj * Ak);
                    const int rem = idx - di * (Aj * Ak);
                    const int dj  = rem / Ak;
                    const int dk  = rem - dj * Ak;
                    const int i = i0_b + di, j = j0_b + dj, k = k0_b + dk;

                    const scalar_t xc = gx_ptr[i];
                    const scalar_t yc = gy_ptr[j];
                    const scalar_t zc = gz_ptr[k];

                    const scalar_t dxw = xc - bp_x, dyw = yc - bp_y, dzw = zc - bp_z;
                    const scalar_t bxq = r00*dxw + r01*dyw + r02*dzw;
                    const scalar_t byq = r10*dxw + r11*dyw + r12*dzw;
                    const scalar_t bzq = r20*dxw + r21*dyw + r22*dzw;
                    const scalar_t sdf = trilinear_sample_border<scalar_t>(
                        F_b, bx_b, by_b, bz_b, Mx, My, Mz,
                        bx0, by0, bz0, bxL, byL, bzL,
                        idx_, idy_, idz_, inv_vol,
                        bxq, byq, bzq);

                    scalar_t delta_visc = 0;
                    const scalar_t d_visc = sdf - eps_s;
                    if (d_visc > -eps_b && d_visc < eps_b) {
                        delta_visc = ((scalar_t)1 + std::cos(pi_v * d_visc / eps_b)) * inv_2eps;
                    }
                    scalar_t delta_pres = 0;
                    if (sdf > -eps_b && sdf < eps_b) {
                        delta_pres = ((scalar_t)1 + std::cos(pi_v * sdf / eps_b)) * inv_2eps;
                    }

                    const int sub_i = i - u_i0_i;
                    const int sub_j = j - u_j0_i;
                    const int sub_k = k - u_k0_i;
                    const std::int64_t s_idx =
                        ((std::int64_t)sub_i * Sj_i + sub_j) * Sk_i + sub_k;
                    const scalar_t xs_v = xs_ptr[s_idx], ys_v = ys_ptr[s_idx], zs_v = zs_ptr[s_idx];
                    const scalar_t px_v = px_ptr[s_idx], py_v = py_ptr[s_idx], pz_v = pz_ptr[s_idx];

                    const scalar_t arm_x = xc - cm_x;
                    const scalar_t arm_y = yc - cm_y;
                    const scalar_t arm_z = zc - cm_z;

                    const double fv_x = (double)(xs_v * delta_visc);
                    const double fv_y = (double)(ys_v * delta_visc);
                    const double fv_z = (double)(zs_v * delta_visc);
                    const double fp_x = (double)(px_v * delta_pres);
                    const double fp_y = (double)(py_v * delta_pres);
                    const double fp_z = (double)(pz_v * delta_pres);

                    local[0]  += fv_x;
                    local[1]  += fv_y;
                    local[2]  += fv_z;
                    local[3]  += (double)arm_y * fv_z - (double)arm_z * fv_y;
                    local[4]  += (double)arm_z * fv_x - (double)arm_x * fv_z;
                    local[5]  += (double)arm_x * fv_y - (double)arm_y * fv_x;
                    local[6]  += fp_x;
                    local[7]  += fp_y;
                    local[8]  += fp_z;
                    local[9]  += (double)arm_y * fp_z - (double)arm_z * fp_y;
                    local[10] += (double)arm_z * fp_x - (double)arm_x * fp_z;
                    local[11] += (double)arm_x * fp_y - (double)arm_y * fp_x;
                }
                _Pragma("omp critical")
                {
                    for (int c = 0; c < 12; ++c) accs[c] += local[c];
                }
            }

            for (int c = 0; c < 12; ++c) {
                out_ptr[b*12 + c] += accs[c] * h3_d;
            }
        }
    });
}



// =====================================================================
//  apply_bcs_3d
//
//  Each op writes a single 2D plane of u / v / w. Ops are independent
//  (caller arranges them so the addressed planes don't overlap), so we
//  loop ops serially and parallelise the plane fill.
// =====================================================================

template <typename scalar_t>
static void apply_bcs_3d_one_plane(
    scalar_t* base,
    const int Ny, const int Nz,
    const int axis,
    const int dst_along, const int src_along,
    const bool is_neu, const scalar_t value,
    const int dim0_max, const int dim1_max)
{
    const std::int64_t s1 = (std::int64_t)Ny * Nz;
    const std::int64_t s2 = (std::int64_t)Nz;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dim0_max; ++i) {
        for (int j = 0; j < dim1_max; ++j) {
            std::int64_t dst_lin, src_lin = 0;
            if (axis == 0) {
                dst_lin = (std::int64_t)dst_along * s1 + (std::int64_t)i * s2 + j;
                if (is_neu) src_lin = (std::int64_t)src_along * s1 + (std::int64_t)i * s2 + j;
            } else if (axis == 1) {
                dst_lin = (std::int64_t)i * s1 + (std::int64_t)dst_along * s2 + j;
                if (is_neu) src_lin = (std::int64_t)i * s1 + (std::int64_t)src_along * s2 + j;
            } else {
                dst_lin = (std::int64_t)i * s1 + (std::int64_t)j * s2 + dst_along;
                if (is_neu) src_lin = (std::int64_t)i * s1 + (std::int64_t)j * s2 + src_along;
            }
            base[dst_lin] = is_neu ? base[src_lin] : value;
        }
    }
}

void apply_bcs_3d_cpu(
    at::Tensor u, at::Tensor v, at::Tensor w,
    const at::Tensor& shapes,
    const at::Tensor& neu_desc,
    const at::Tensor& dir_desc,
    const at::Tensor& dir_val,
    const int64_t /*max_plane_dim*/)
{
    TORCH_CHECK(u.is_contiguous() && v.is_contiguous() && w.is_contiguous(),
                "apply_bcs_3d_cpu: u/v/w must be contiguous");
    TORCH_CHECK(u.scalar_type() == v.scalar_type() &&
                u.scalar_type() == w.scalar_type(),
                "apply_bcs_3d_cpu: u/v/w must share dtype");
    TORCH_CHECK(shapes.scalar_type() == at::kLong &&
                shapes.dim() == 2 && shapes.size(0) == 3 && shapes.size(1) == 3,
                "apply_bcs_3d_cpu: shapes must be int64[3,3]");
    TORCH_CHECK(neu_desc.scalar_type() == at::kInt && neu_desc.dim() == 2 &&
                neu_desc.size(1) == 3,
                "apply_bcs_3d_cpu: neu_desc must be int32[N,3]");
    TORCH_CHECK(dir_desc.scalar_type() == at::kInt && dir_desc.dim() == 2 &&
                dir_desc.size(1) == 3,
                "apply_bcs_3d_cpu: dir_desc must be int32[N,3]");

    const int N_neu = (int)neu_desc.size(0);
    const int N_dir = (int)dir_desc.size(0);
    if (N_neu + N_dir == 0) return;

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_3d_cpu", [&] {
        const int64_t*  shapes_p  = shapes.data_ptr<int64_t>();
        const int*      neu_p     = (N_neu > 0) ? neu_desc.data_ptr<int>() : nullptr;
        const int*      dir_p     = (N_dir > 0) ? dir_desc.data_ptr<int>() : nullptr;
        const scalar_t* dir_val_p = (N_dir > 0) ? dir_val.data_ptr<scalar_t>() : nullptr;

        scalar_t* u_p = u.data_ptr<scalar_t>();
        scalar_t* v_p = v.data_ptr<scalar_t>();
        scalar_t* w_p = w.data_ptr<scalar_t>();

        const int total = N_neu + N_dir;
        for (int op = 0; op < total; ++op) {
            const bool is_neu = (op < N_neu);
            int comp, axis, dst_along, src_along;
            scalar_t value = scalar_t(0);

            if (is_neu) {
                comp = neu_p[op*3 + 0];
                axis = neu_p[op*3 + 1];
                const int side = neu_p[op*3 + 2];
                const int sz = (int)shapes_p[comp*3 + axis];
                if (side == 0) { dst_along = 0;      src_along = 1; }
                else           { dst_along = sz - 1; src_along = sz - 2; }
            } else {
                const int d = op - N_neu;
                comp = dir_p[d*3 + 0];
                axis = dir_p[d*3 + 1];
                const int offset = dir_p[d*3 + 2];
                const int sz = (int)shapes_p[comp*3 + axis];
                dst_along = (offset >= 0) ? offset : (sz + offset);
                src_along = 0;
                value = dir_val_p[d];
            }

            const int Nx = (int)shapes_p[comp*3 + 0];
            const int Ny = (int)shapes_p[comp*3 + 1];
            const int Nz = (int)shapes_p[comp*3 + 2];

            int dim0_max, dim1_max;
            if      (axis == 0) { dim0_max = Ny; dim1_max = Nz; }
            else if (axis == 1) { dim0_max = Nx; dim1_max = Nz; }
            else                { dim0_max = Nx; dim1_max = Ny; }

            scalar_t* base = (comp == 0) ? u_p : (comp == 1 ? v_p : w_p);

            apply_bcs_3d_one_plane<scalar_t>(
                base, Ny, Nz, axis,
                dst_along, src_along, is_neu, value,
                dim0_max, dim1_max);
        }
    });
}

// =====================================================================
//  CPU registration. The schemas live in ops.cpp; ops.cpp no longer
//  registers CPU stubs, so these implementations bind directly.
// =====================================================================

TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("streaming_sdf_min_3d",       &streaming_sdf_min_3d_cpu);
    m.impl("streaming_sdf_min_3d_multi", &streaming_sdf_min_3d_multi_cpu);
    m.impl("bdim_forces_3d_multi",       &bdim_forces_3d_multi_cpu);
    m.impl("bdim_forces_3d_multi_legacy_resample",
                                          &bdim_forces_3d_multi_legacy_resample_cpu);
    m.impl("apply_bcs_3d",               &apply_bcs_3d_cpu);
}

}  // namespace lilytorch_kernels
