// =====================================================================
//  streaming_sdf_cpu_2d.cpp
//
//  CPU implementations of the 2-D ``streaming_sdf_min_2d`` and
//  ``streaming_sdf_min_2d_multi`` ops.  Mirrors
//  ``streaming_sdf_cpu.cpp`` line-for-line with the z-axis stripped:
//    * 3 face samples per cell (cc, u-stagger -h/2 in x, v-stagger -h/2 in y);
//    * rotation R_T is a 2x2 column-major matrix (4 floats);
//    * angular velocity is a scalar omega (out-of-plane);
//    * world->body transform: [bxq;byq] = R_T * [xc-bp_x; yc-bp_y].
//
//  Body SDF tables are uniform Cartesian grids by construction in BDIM,
//  so corner weights reduce to (1-frac, frac) per axis.  Bilinear and
//  biquadratic Lagrange samplers operate directly on (bx0, inv_dx) /
//  (by0, inv_dy) without axis-table lookups.
//
//  Hot loops are parallelised with ``at::parallel_for`` (PyTorch's
//  intra-op thread pool); see the rationale comment at the top of
//  ``streaming_sdf_cpu.cpp``.
// =====================================================================

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/all.h>
#include <torch/library.h>
#include <c10/util/ArrayRef.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lilytorch_kernels {

// =====================================================================
//  Bilinear sample on a UNIFORM body grid (2-D analogue of
//  ``trilinear_sample_uniform``).  Returns the bilinearly-interpolated
//  value of F at world point (xq, yq), with edge clamping.
// =====================================================================
template <typename scalar_t>
static inline scalar_t bilinear_sample_uniform_2d(
    const scalar_t* F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    const scalar_t Mx_lim = (scalar_t)(Mx - 1);
    const scalar_t My_lim = (scalar_t)(My - 1);
    if (tx < (scalar_t)0) tx = (scalar_t)0; else if (tx > Mx_lim) tx = Mx_lim;
    if (ty < (scalar_t)0) ty = (scalar_t)0; else if (ty > My_lim) ty = My_lim;

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

// =====================================================================
//  Biquadratic sample on a UNIFORM body grid (2-D analogue of
//  ``triquadratic_sample_uniform``).  Lagrange interpolation on a
//  3x3 stencil [ix-1, ix, ix+1] x [iy-1, iy, iy+1] with the same
//  lower-bracketing convention as the bilinear sampler.  Falls back to
//  bilinear in any cell whose lower stencil neighbour is out of range.
// =====================================================================
template <typename scalar_t>
static inline scalar_t biquadratic_sample_uniform_2d(
    const scalar_t* F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    const scalar_t Mx_lim = (scalar_t)(Mx - 1);
    const scalar_t My_lim = (scalar_t)(My - 1);
    if (tx < (scalar_t)0) tx = (scalar_t)0; else if (tx > Mx_lim) tx = Mx_lim;
    if (ty < (scalar_t)0) ty = (scalar_t)0; else if (ty > My_lim) ty = My_lim;

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

    auto col = [&](int dx_off) -> scalar_t {
        const int b0 = base + dx_off * s1;
        return wym * F[b0] + wy0 * F[b0 + 1] + wyp * F[b0 + 2];
    };
    return wxm * col(0) + wx0 * col(1) + wxp * col(2);
}

// =====================================================================
//  Per-cell update body for 2-D streaming-min update.  Each cell is
//  touched once per launch -> no atomics needed.
//
//  In 2-D the rigid-body angular velocity is a scalar ``omega``
//  (out-of-plane), and the face velocities at (xc-h/2, yc) and
//  (xc, yc-h/2) are:
//      u_face = lv_x - omega * (yc - cm_y)
//      v_face = lv_y + omega * (xc - cm_x)
//  These are the 3-D formulas with (av_x, av_y, av_z) = (0, 0, omega)
//  and zc - cm_z = 0 collapsed.
// =====================================================================
template <typename scalar_t>
static inline void update_cell_2d(
    const scalar_t* F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    const scalar_t r00, const scalar_t r01,
    const scalar_t r10, const scalar_t r11,
    const scalar_t bp_x, const scalar_t bp_y,
    const scalar_t cm_x, const scalar_t cm_y,
    const scalar_t lv_x, const scalar_t lv_y,
    const scalar_t omega,
    // Body-frame face deltas: Δ_u = -half_h * col0(R_T),
    //                          Δ_v = -half_h * col1(R_T).
    const scalar_t du_x, const scalar_t du_y,
    const scalar_t dv_x, const scalar_t dv_y,
    const scalar_t xc, const scalar_t yc,
    const std::int64_t g_idx, const std::int64_t sparse_idx,
    scalar_t* sdf_cc, scalar_t* sdf_u, scalar_t* sdf_v,
    scalar_t* bU, scalar_t* bV,
    scalar_t* sparse_cc,
    // 0 = bilinear (default), 1 = biquadratic (Lagrange 3x3).
    const int interp_method)
{
    // Body-frame CC point (single 2x2 rotation; the 2 face points are
    // derived from it by precomputed deltas).
    const scalar_t dxw = xc - bp_x, dyw = yc - bp_y;
    const scalar_t bxq = r00 * dxw + r01 * dyw;
    const scalar_t byq = r10 * dxw + r11 * dyw;

    auto sample = [&](scalar_t xqs, scalar_t yqs) -> scalar_t {
        if (interp_method == 1) {
            return biquadratic_sample_uniform_2d<scalar_t>(
                F, Mx, My, bx0, by0, inv_dx, inv_dy, xqs, yqs);
        }
        return bilinear_sample_uniform_2d<scalar_t>(
            F, Mx, My, bx0, by0, inv_dx, inv_dy, xqs, yqs);
    };

    // ---- cc ----
    {
        const scalar_t s = sample(bxq, byq);
        sparse_cc[sparse_idx] = s;
        if (s < sdf_cc[g_idx]) sdf_cc[g_idx] = s;
    }
    // ---- u-face: world (xc - h/2, yc) ----
    {
        const scalar_t s = sample(bxq + du_x, byq + du_y);
        if (s < sdf_u[g_idx]) {
            sdf_u[g_idx] = s;
            bU[g_idx] = lv_x - omega * (yc - cm_y);
        }
    }
    // ---- v-face: world (xc, yc - h/2) ----
    {
        const scalar_t s = sample(bxq + dv_x, byq + dv_y);
        if (s < sdf_v[g_idx]) {
            sdf_v[g_idx] = s;
            bV[g_idx] = lv_y + omega * (xc - cm_x);
        }
    }
}

// =====================================================================
//  streaming_sdf_min_2d (single body)
// =====================================================================

void streaming_sdf_min_2d_cpu(
    const at::Tensor& F,
    const at::Tensor& bx, const at::Tensor& by,
    const double bx0, const double by0,
    const double bx_last, const double by_last,
    const double inv_dx, const double inv_dy,
    const double inv_vol,
    c10::ArrayRef<double> R_T,
    c10::ArrayRef<double> body_pos,
    c10::ArrayRef<double> com_pos,
    c10::ArrayRef<double> lin_vel,
    const double omega,
    const at::Tensor& gx, const at::Tensor& gy,
    const double h_grid,
    const int64_t i0, const int64_t i1,
    const int64_t j0, const int64_t j1,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v,
    at::Tensor body_u, at::Tensor body_v,
    at::Tensor sparse_cc,
    const int64_t interp_method)
{
    TORCH_CHECK(R_T.size()      == 4, "R_T must have 4 elements (2x2)");
    TORCH_CHECK(body_pos.size() == 2, "body_pos must have 2 elements");
    TORCH_CHECK(com_pos.size()  == 2, "com_pos must have 2 elements");
    TORCH_CHECK(lin_vel.size()  == 2, "lin_vel must have 2 elements");
    (void)bx_last; (void)by_last; (void)inv_vol;

    const int Ai = (int)(i1 - i0);
    const int Aj = (int)(j1 - j0);
    const int N  = Ai * Aj;
    if (N <= 0) return;

    const int Mx  = (int)bx.numel();
    const int My  = (int)by.numel();
    const int Ngy = (int)gy.numel();

    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "streaming_sdf_min_2d_cpu", [&] {
        auto F_c  = F.contiguous();
        auto gx_c = gx.contiguous();
        auto gy_c = gy.contiguous();

        const scalar_t* F_ptr  = F_c.data_ptr<scalar_t>();
        const scalar_t* gx_ptr = gx_c.data_ptr<scalar_t>();
        const scalar_t* gy_ptr = gy_c.data_ptr<scalar_t>();

        scalar_t* sdf_cc_ptr = sdf_cc.data_ptr<scalar_t>();
        scalar_t* sdf_u_ptr  = sdf_u.data_ptr<scalar_t>();
        scalar_t* sdf_v_ptr  = sdf_v.data_ptr<scalar_t>();
        scalar_t* bU_ptr     = body_u.data_ptr<scalar_t>();
        scalar_t* bV_ptr     = body_v.data_ptr<scalar_t>();
        scalar_t* sp_ptr     = sparse_cc.data_ptr<scalar_t>();

        const scalar_t bx0_s = (scalar_t)bx0, by0_s = (scalar_t)by0;
        const scalar_t idx_s = (scalar_t)inv_dx, idy_s = (scalar_t)inv_dy;
        const scalar_t r00 = (scalar_t)R_T[0], r01 = (scalar_t)R_T[1];
        const scalar_t r10 = (scalar_t)R_T[2], r11 = (scalar_t)R_T[3];
        const scalar_t bp_x = (scalar_t)body_pos[0], bp_y = (scalar_t)body_pos[1];
        const scalar_t cm_x = (scalar_t)com_pos[0],  cm_y = (scalar_t)com_pos[1];
        const scalar_t lv_x = (scalar_t)lin_vel[0],  lv_y = (scalar_t)lin_vel[1];
        const scalar_t om   = (scalar_t)omega;
        const scalar_t neg_half_h = -(scalar_t)(0.5 * h_grid);
        const scalar_t du_x = neg_half_h * r00, du_y = neg_half_h * r10;
        const scalar_t dv_x = neg_half_h * r01, dv_y = neg_half_h * r11;
        const int i0_i = (int)i0, j0_i = (int)j0;

        at::parallel_for(0, N, /*grain_size=*/1024, [&](int64_t _begin, int64_t _end) {
        for (int tid = (int)_begin; tid < (int)_end; ++tid) {
            const int di = tid / Aj;
            const int dj = tid - di * Aj;
            const int i  = i0_i + di;
            const int j  = j0_i + dj;
            const std::int64_t g_idx = (std::int64_t)i * Ngy + j;

            update_cell_2d<scalar_t>(
                F_ptr, Mx, My,
                bx0_s, by0_s, idx_s, idy_s,
                r00, r01, r10, r11,
                bp_x, bp_y, cm_x, cm_y,
                lv_x, lv_y, om,
                du_x, du_y, dv_x, dv_y,
                gx_ptr[i], gy_ptr[j],
                g_idx, /*sparse_idx=*/(std::int64_t)tid,
                sdf_cc_ptr, sdf_u_ptr, sdf_v_ptr,
                bU_ptr, bV_ptr, sp_ptr,
                (int)interp_method);
        }
        });
    });
}

// =====================================================================
//  streaming_sdf_min_2d_multi
//
//  Per-body packed layouts:
//    body_shapes : int64 [B,2]     -> (Mx, My)
//    body_meta   : float [B,7]     -> (bx0, by0, bxL, byL, inv_dx, inv_dy, inv_vol)
//                                     (only bx0,by0,inv_dx,inv_dy are read)
//    kin         : float [B,11]    -> (R_T[0..3], bp_x, bp_y, cm_x, cm_y,
//                                       lv_x, lv_y, omega)
//    aabb_lo     : int64 [B,2]     -> (i0, j0)
//    aabb_dim    : int64 [B,2]     -> (Ai, Aj)
//    cell_offsets: int64 [B+1]     -> prefix sum of Ai*Aj over bodies
// =====================================================================

void streaming_sdf_min_2d_multi_cpu(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& bx_flat, const at::Tensor& bx_offsets,
    const at::Tensor& by_flat, const at::Tensor& by_offsets,
    const at::Tensor& body_shapes,
    const at::Tensor& body_meta,
    const at::Tensor& kin,
    const at::Tensor& aabb_lo,
    const at::Tensor& aabb_dim,
    const at::Tensor& cell_offsets,
    const at::Tensor& gx, const at::Tensor& gy,
    const double h_grid,
    const int64_t /*max_vol_per_body*/,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v,
    at::Tensor body_u, at::Tensor body_v,
    at::Tensor sparse_cc_flat,
    const int64_t interp_method)
{
    (void)bx_flat; (void)bx_offsets; (void)by_flat; (void)by_offsets;

    const int B = (int)aabb_dim.size(0);
    if (B <= 0) return;

    const int Ngy = (int)gy.numel();

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_min_2d_multi_cpu", [&] {
        auto F_c  = F_flat.contiguous();
        auto gx_c = gx.contiguous();
        auto gy_c = gy.contiguous();

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

        scalar_t* sdf_cc_p = sdf_cc.data_ptr<scalar_t>();
        scalar_t* sdf_u_p  = sdf_u.data_ptr<scalar_t>();
        scalar_t* sdf_v_p  = sdf_v.data_ptr<scalar_t>();
        scalar_t* bU_p     = body_u.data_ptr<scalar_t>();
        scalar_t* bV_p     = body_v.data_ptr<scalar_t>();
        scalar_t* sparse_p = sparse_cc_flat.data_ptr<scalar_t>();

        const scalar_t neg_half_h = -(scalar_t)(0.5 * h_grid);

        for (int b = 0; b < B; ++b) {
            const int Ai = (int)dim_[b*2 + 0];
            const int Aj = (int)dim_[b*2 + 1];
            const int vol = Ai * Aj;
            if (vol <= 0) continue;

            const int i0_b = (int)lo[b*2 + 0];
            const int j0_b = (int)lo[b*2 + 1];

            const scalar_t* F_b  = F_ptr  + F_off[b];
            const int Mx = (int)shapes[b*2 + 0];
            const int My = (int)shapes[b*2 + 1];

            const scalar_t* M = meta + b*7;
            const scalar_t bx0 = M[0], by0 = M[1];
            const scalar_t idx_ = M[4], idy_ = M[5];

            const scalar_t* K = kin_ptr + b*11;
            const scalar_t r00 = K[0], r01 = K[1];
            const scalar_t r10 = K[2], r11 = K[3];
            const scalar_t bp_x = K[4], bp_y = K[5];
            const scalar_t cm_x = K[6], cm_y = K[7];
            const scalar_t lv_x = K[8], lv_y = K[9];
            const scalar_t om   = K[10];

            const scalar_t du_x = neg_half_h * r00, du_y = neg_half_h * r10;
            const scalar_t dv_x = neg_half_h * r01, dv_y = neg_half_h * r11;

            const std::int64_t sparse_base = (std::int64_t)cell_off[b];

            at::parallel_for(0, vol, /*grain_size=*/1024, [&](int64_t _begin, int64_t _end) {
            for (int local = (int)_begin; local < (int)_end; ++local) {
                const int di = local / Aj;
                const int dj = local - di * Aj;
                const int i  = i0_b + di;
                const int j  = j0_b + dj;
                const std::int64_t g_idx = (std::int64_t)i * Ngy + j;

                update_cell_2d<scalar_t>(
                    F_b, Mx, My,
                    bx0, by0, idx_, idy_,
                    r00, r01, r10, r11,
                    bp_x, bp_y, cm_x, cm_y,
                    lv_x, lv_y, om,
                    du_x, du_y, dv_x, dv_y,
                    gx_ptr[i], gy_ptr[j],
                    g_idx, sparse_base + (std::int64_t)local,
                    sdf_cc_p, sdf_u_p, sdf_v_p,
                    bU_p, bV_p, sparse_p,
                    (int)interp_method);
            }
            });
        }
    });
}

// =====================================================================
//  CPU registration. Schemas live in ops.cpp.
// =====================================================================

TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("streaming_sdf_min_2d",       &streaming_sdf_min_2d_cpu);
    m.impl("streaming_sdf_min_2d_multi", &streaming_sdf_min_2d_multi_cpu);
}

}  // namespace lilytorch_kernels
