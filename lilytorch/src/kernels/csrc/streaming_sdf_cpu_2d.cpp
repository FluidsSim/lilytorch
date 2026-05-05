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
#include <mutex>
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
//  bdim_forces_2d_multi
//
//  2-D analogue of ``bdim_forces_3d_multi``.  Per body, walks the cells
//  inside the AABB, reads the cached cc-SDF from ``sparse_cc_flat``,
//  evaluates the smoothed visc / pres deltas, and accumulates 8 float64
//  channels:
//      [fv_x, fv_y, t_v,
//       fp_x, fp_y, t_p,
//       0,    0]
//  where t_v / t_p are the scalar out-of-plane torques
//      t_v = arm_x * fv_y - arm_y * fv_x
//      t_p = arm_x * fp_y - arm_y * fp_x
//  and the trailing two channels are reserved (kept in the layout so
//  the per-body output row maps cleanly onto a 1-padded 2-or-3-component
//  buffer in Python; the kernel writes exactly zero there).
//
//  ``out`` is float64 and the kernel does ``out += accs * h2`` so the
//  caller may invoke repeatedly on a pre-existing accumulator (mirrors
//  the 3-D op).
// =====================================================================

void bdim_forces_2d_multi_cpu(
    const at::Tensor& sparse_cc_flat,
    const at::Tensor& cell_offsets,
    const at::Tensor& kin,
    const at::Tensor& aabb_lo,
    const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy,
    const int64_t u_i0, const int64_t u_j0,
    const int64_t Sj,
    const at::Tensor& xs, const at::Tensor& ys,
    const at::Tensor& px, const at::Tensor& py,
    const double eps_body, const double eps_solver, const double h2,
    const int64_t /*max_vol_per_body*/,
    const int64_t delta_order,
    at::Tensor out)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0) return;

    TORCH_CHECK(out.scalar_type() == at::kDouble,
                "bdim_forces_2d_multi_cpu: out must be float64");
    TORCH_CHECK(out.size(1) == 6,
                "bdim_forces_2d_multi_cpu: out must have 6 channels");

    AT_DISPATCH_FLOATING_TYPES(sparse_cc_flat.scalar_type(), "bdim_forces_2d_multi_cpu", [&] {
        auto sp_c = sparse_cc_flat.contiguous();
        auto gx_c = gx.contiguous();
        auto gy_c = gy.contiguous();
        auto xs_c = xs.contiguous();
        auto ys_c = ys.contiguous();
        auto px_c = px.contiguous();
        auto py_c = py.contiguous();

        const scalar_t* sp_ptr  = sp_c.data_ptr<scalar_t>();
        const int64_t*  cell_off= cell_offsets.data_ptr<int64_t>();
        const scalar_t* kin_ptr = kin.data_ptr<scalar_t>();
        const int64_t*  lo      = aabb_lo.data_ptr<int64_t>();
        const int64_t*  dim_    = aabb_dim.data_ptr<int64_t>();
        const scalar_t* gx_ptr  = gx_c.data_ptr<scalar_t>();
        const scalar_t* gy_ptr  = gy_c.data_ptr<scalar_t>();
        const scalar_t* xs_ptr  = xs_c.data_ptr<scalar_t>();
        const scalar_t* ys_ptr  = ys_c.data_ptr<scalar_t>();
        const scalar_t* px_ptr  = px_c.data_ptr<scalar_t>();
        const scalar_t* py_ptr  = py_c.data_ptr<scalar_t>();
        double*         out_ptr = out.data_ptr<double>();

        const scalar_t eps_b      = (scalar_t)eps_body;
        const scalar_t eps_s      = (scalar_t)eps_solver;
        const scalar_t pi_v       = (scalar_t)3.141592653589793;
        const scalar_t inv_2eps   = (scalar_t)0.5 / eps_b;
        const scalar_t pi_over_eb = pi_v / eps_b;
        const double   h2_d       = (double)h2;

        // Same band-test logic as the 3-D version: deltas are zero
        // outside [min(-eps_b, eps_s-eps_b), max(eps_b, eps_s+eps_b)].
        const scalar_t band_lo = (eps_s - eps_b) < (-eps_b) ? (eps_s - eps_b) : (-eps_b);
        const scalar_t band_hi = (eps_s + eps_b) > ( eps_b) ? (eps_s + eps_b) : ( eps_b);

        const int u_i0_i = (int)u_i0, u_j0_i = (int)u_j0;
        const int Sj_i   = (int)Sj;

        const int B_local = B;
        std::vector<double> accs(static_cast<size_t>(B_local) * 6, 0.0);

        for (int b = 0; b < B_local; ++b) {
            const int Ai = (int)dim_[b*2 + 0];
            const int Aj = (int)dim_[b*2 + 1];
            const int vol = Ai * Aj;
            if (vol <= 0) continue;

            const int i0_b = (int)lo[b*2 + 0];
            const int j0_b = (int)lo[b*2 + 1];

            // 2-D ``kin`` row (11 floats per body):
            //   R_T[0..3], bp_xy(2), cm_xy(2), lv_xy(2), omega(1).
            // Only the COM (kin[6..7]) is needed for the force/torque arms.
            const scalar_t* K = kin_ptr + b*11;
            const scalar_t cm_x = K[6], cm_y = K[7];

            const std::int64_t sparse_base = (std::int64_t)cell_off[b];
            double* lb = accs.data() + (size_t)b * 6;
            std::mutex acc_mtx;

            at::parallel_for(0, vol, /*grain_size=*/2048,
                [&](int64_t _begin, int64_t _end)
            {
                double local8[6] = {0,0,0,0,0,0};
                for (int idx = (int)_begin; idx < (int)_end; ++idx) {
                    const scalar_t sdf = sp_ptr[sparse_base + (std::int64_t)idx];
                    if (sdf <= band_lo || sdf >= band_hi) continue;

                    const int di = idx / Aj;
                    const int dj = idx - di * Aj;

                    scalar_t delta_visc = 0;
                    const scalar_t d_visc = sdf - eps_s;
                    if (d_visc > -eps_b && d_visc < eps_b) {
                        delta_visc = ((scalar_t)1 + std::cos(pi_over_eb * d_visc)) * inv_2eps;
                    }
                    scalar_t delta_pres = 0;
                    if (sdf > -eps_b && sdf < eps_b) {
                        delta_pres = ((scalar_t)1 + std::cos(pi_over_eb * sdf)) * inv_2eps;
                    }
                    if (delta_visc == (scalar_t)0 && delta_pres == (scalar_t)0) continue;

                    // Towers (2008) 2nd-order: δ_S = δ_ε(φ) / |∇φ|
                    if (delta_order == 2) {
                        const scalar_t h_grid = gx_ptr[1] - gx_ptr[0];
                        const scalar_t inv_h  = (scalar_t)1.0 / h_grid;

                        scalar_t sdf_xp = (di < Ai-1) ? sp_ptr[sparse_base + idx + Aj] : sdf;
                        scalar_t sdf_xm = (di > 0)    ? sp_ptr[sparse_base + idx - Aj] : sdf;
                        scalar_t cx     = (di > 0 && di < Ai-1) ? (scalar_t)0.5 : (scalar_t)1.0;
                        scalar_t dsdx   = cx * (sdf_xp - sdf_xm) * inv_h;

                        scalar_t sdf_yp = (dj < Aj-1) ? sp_ptr[sparse_base + idx + 1] : sdf;
                        scalar_t sdf_ym = (dj > 0)    ? sp_ptr[sparse_base + idx - 1] : sdf;
                        scalar_t cy     = (dj > 0 && dj < Aj-1) ? (scalar_t)0.5 : (scalar_t)1.0;
                        scalar_t dsdy   = cy * (sdf_yp - sdf_ym) * inv_h;

                        scalar_t grad_mag = std::sqrt(dsdx*dsdx + dsdy*dsdy);
                        if (grad_mag < (scalar_t)1e-3) grad_mag = (scalar_t)1e-3;
                        const scalar_t inv_grad = (scalar_t)1.0 / grad_mag;
                        delta_visc *= inv_grad;
                        delta_pres *= inv_grad;
                    }

                    const int i = i0_b + di, j = j0_b + dj;
                    const int sub_i = i - u_i0_i;
                    const int sub_j = j - u_j0_i;
                    const std::int64_t s_idx = (std::int64_t)sub_i * Sj_i + sub_j;

                    const scalar_t xc = gx_ptr[i];
                    const scalar_t yc = gy_ptr[j];

                    const scalar_t arm_x = xc - cm_x;
                    const scalar_t arm_y = yc - cm_y;

                    const double fv_x = (double)(xs_ptr[s_idx] * delta_visc);
                    const double fv_y = (double)(ys_ptr[s_idx] * delta_visc);
                    const double fp_x = (double)(px_ptr[s_idx] * delta_pres);
                    const double fp_y = (double)(py_ptr[s_idx] * delta_pres);

                    local8[0] += fv_x;
                    local8[1] += fv_y;
                    local8[2] += (double)arm_x * fv_y - (double)arm_y * fv_x;
                    local8[3] += fp_x;
                    local8[4] += fp_y;
                    local8[5] += (double)arm_x * fp_y - (double)arm_y * fp_x;
                }
                std::lock_guard<std::mutex> lk(acc_mtx);
                for (int c = 0; c < 6; ++c) lb[c] += local8[c];
            });
        }

        for (int b = 0; b < B_local; ++b) {
            for (int c = 0; c < 6; ++c) {
                out_ptr[b*6 + c] += accs[(size_t)b * 6 + c] * h2_d;
            }
        }
    });
}

// =====================================================================
//  apply_bcs_2d  (Phase H 2-D analogue of apply_bcs_3d)
//
//  Each op writes a single 1-D ghost line of u or v.  Ops are
//  independent so we loop ops serially and parallelise the line fill.
//
//  shapes  : int64 [2,2]      -> per-component (Nx, Ny)
//  neu_desc: int32 [N_neu, 3] -> (comp, axis, side)
//                                   comp in {0=u, 1=v}
//                                   axis in {0=x, 1=y}
//                                   side in {0=lo, 1=hi}
//  dir_desc: int32 [N_dir, 3] -> (comp, axis, offset)
//                                offset is signed: dst = offset if >=0
//                                else (sz + offset).
//  dir_val : float[N_dir]
// =====================================================================

template <typename scalar_t>
static void apply_bcs_2d_one_line(
    scalar_t* base,
    const int Ny,
    const int axis,
    const int dst_along, const int src_along,
    const bool is_neu, const scalar_t value,
    const int dim0_max)
{
    // Flat layout: base[i, j] = base[i*Ny + j].
    at::parallel_for(0, dim0_max, /*grain_size=*/1024, [&](int64_t _begin, int64_t _end) {
    for (int64_t t = _begin; t < _end; ++t) {
        const int i = (int)t;
        std::int64_t dst_lin, src_lin = 0;
        if (axis == 0) {
            dst_lin = (std::int64_t)dst_along * Ny + i;
            if (is_neu) src_lin = (std::int64_t)src_along * Ny + i;
        } else {
            dst_lin = (std::int64_t)i * Ny + dst_along;
            if (is_neu) src_lin = (std::int64_t)i * Ny + src_along;
        }
        base[dst_lin] = is_neu ? base[src_lin] : value;
    }
    });
}

void apply_bcs_2d_cpu(
    at::Tensor u, at::Tensor v,
    const at::Tensor& shapes,
    const at::Tensor& neu_desc,
    const at::Tensor& dir_desc,
    const at::Tensor& dir_val,
    const int64_t /*max_line_dim*/)
{
    TORCH_CHECK(u.is_contiguous() && v.is_contiguous(),
                "apply_bcs_2d_cpu: u/v must be contiguous");
    TORCH_CHECK(u.scalar_type() == v.scalar_type(),
                "apply_bcs_2d_cpu: u/v must share dtype");
    TORCH_CHECK(shapes.scalar_type() == at::kLong &&
                shapes.dim() == 2 && shapes.size(0) == 2 && shapes.size(1) == 2,
                "apply_bcs_2d_cpu: shapes must be int64[2,2]");
    TORCH_CHECK(neu_desc.scalar_type() == at::kInt && neu_desc.dim() == 2 &&
                neu_desc.size(1) == 3,
                "apply_bcs_2d_cpu: neu_desc must be int32[N,3]");
    TORCH_CHECK(dir_desc.scalar_type() == at::kInt && dir_desc.dim() == 2 &&
                dir_desc.size(1) == 3,
                "apply_bcs_2d_cpu: dir_desc must be int32[N,3]");

    const int N_neu = (int)neu_desc.size(0);
    const int N_dir = (int)dir_desc.size(0);
    if (N_neu + N_dir == 0) return;

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_2d_cpu", [&] {
        const int64_t*  shapes_p  = shapes.data_ptr<int64_t>();
        const int*      neu_p     = (N_neu > 0) ? neu_desc.data_ptr<int>() : nullptr;
        const int*      dir_p     = (N_dir > 0) ? dir_desc.data_ptr<int>() : nullptr;
        const scalar_t* dir_val_p = (N_dir > 0) ? dir_val.data_ptr<scalar_t>() : nullptr;

        scalar_t* u_p = u.data_ptr<scalar_t>();
        scalar_t* v_p = v.data_ptr<scalar_t>();

        const int total = N_neu + N_dir;
        for (int op = 0; op < total; ++op) {
            const bool is_neu = (op < N_neu);
            int comp, axis, dst_along, src_along;
            scalar_t value = scalar_t(0);

            if (is_neu) {
                comp = neu_p[op*3 + 0];
                axis = neu_p[op*3 + 1];
                const int side = neu_p[op*3 + 2];
                const int sz = (int)shapes_p[comp*2 + axis];
                if (side == 0) { dst_along = 0;      src_along = 1; }
                else           { dst_along = sz - 1; src_along = sz - 2; }
            } else {
                const int d = op - N_neu;
                comp = dir_p[d*3 + 0];
                axis = dir_p[d*3 + 1];
                const int offset = dir_p[d*3 + 2];
                const int sz = (int)shapes_p[comp*2 + axis];
                dst_along = (offset >= 0) ? offset : (sz + offset);
                src_along = 0;
                value = dir_val_p[d];
            }

            const int Nx = (int)shapes_p[comp*2 + 0];
            const int Ny = (int)shapes_p[comp*2 + 1];

            // axis==0 -> sweep along j (size Ny). axis==1 -> sweep along i (size Nx).
            const int dim0_max = (axis == 0) ? Ny : Nx;

            scalar_t* base = (comp == 0) ? u_p : v_p;

            apply_bcs_2d_one_line<scalar_t>(
                base, Ny, axis,
                dst_along, src_along, is_neu, value,
                dim0_max);
        }
    });
}

// =====================================================================
//  interpolate_2d_cpu: scattered-point bilinear / biquadratic sampling
// =====================================================================
static void interpolate_2d_cpu(
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

    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "interpolate_2d_cpu", [&] {
        const scalar_t* Fp  = F.contiguous().data_ptr<scalar_t>();
        const scalar_t* xqp = xq.contiguous().to(F.scalar_type()).data_ptr<scalar_t>();
        const scalar_t* yqp = yq.contiguous().to(F.scalar_type()).data_ptr<scalar_t>();
        scalar_t* Gp = G.data_ptr<scalar_t>();
        const int iMx = (int)Mx, iMy = (int)My;
        const scalar_t bx0s = (scalar_t)bx0, by0s = (scalar_t)by0;
        const scalar_t idx  = (scalar_t)inv_dx, idy = (scalar_t)inv_dy;
        const int method    = (int)interp_method;

        at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                if (method == 1) {
                    Gp[i] = biquadratic_sample_uniform_2d<scalar_t>(
                        Fp, iMx, iMy, bx0s, by0s, idx, idy,
                        xqp[i], yqp[i]);
                } else {
                    Gp[i] = bilinear_sample_uniform_2d<scalar_t>(
                        Fp, iMx, iMy, bx0s, by0s, idx, idy,
                        xqp[i], yqp[i]);
                }
            }
        });
    });
}

// =====================================================================
//  streaming_sdf_forces_fused_2d_multi (CPU fallback)
//
//  Implements Phase C (SDF update) and Phase D (force integration) in
//  two sequential passes over the per-body AABB.  The intermediate
//  sparse_cc_flat buffer is allocated locally and used to communicate
//  CC-SDF values between the two passes.
// =====================================================================

void streaming_sdf_forces_fused_2d_multi_cpu(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes, const at::Tensor& body_meta, const at::Tensor& kin,
    const at::Tensor& aabb_lo, const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy,
    const double h_grid, const int64_t /*max_vol_per_body*/,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v,
    at::Tensor body_u, at::Tensor body_v,
    const int64_t interp_method,
    const at::Tensor& rho_bodies, at::Tensor winning_rho_cc,
    const at::Tensor& u_prev, const at::Tensor& v_prev, const at::Tensor& p_prev,
    const at::Tensor& nx_cc, const at::Tensor& ny_cc,
    const at::Tensor& nu_rho_field,
    const double eps_body, const double eps_solver, const double h2,
    const int64_t delta_order,
    at::Tensor out)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0) return;

    TORCH_CHECK(out.scalar_type() == at::kDouble,
                "streaming_sdf_forces_fused_2d_multi_cpu: out must be float64");
    TORCH_CHECK(out.size(1) == 6,
                "streaming_sdf_forces_fused_2d_multi_cpu: out must have 6 channels");

    const int Ngy = (int)gy.numel();
    const int Ngx = (int)gx.numel();

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_forces_fused_2d_multi_cpu", [&] {
        auto F_c   = F_flat.contiguous();
        auto gx_c  = gx.contiguous();
        auto gy_c  = gy.contiguous();
        auto up_c  = u_prev.contiguous();
        auto vp_c  = v_prev.contiguous();
        auto pp_c  = p_prev.contiguous();
        auto nxc_c = nx_cc.contiguous();
        auto nyc_c = ny_cc.contiguous();
        auto nr_c  = nu_rho_field.contiguous();

        const scalar_t* F_ptr     = F_c.data_ptr<scalar_t>();
        const int64_t*  F_off     = F_offsets.data_ptr<int64_t>();
        const int64_t*  shapes    = body_shapes.data_ptr<int64_t>();
        const scalar_t* meta      = body_meta.data_ptr<scalar_t>();
        const scalar_t* kin_ptr   = kin.data_ptr<scalar_t>();
        const int64_t*  lo        = aabb_lo.data_ptr<int64_t>();
        const int64_t*  dim_      = aabb_dim.data_ptr<int64_t>();
        const scalar_t* gx_ptr    = gx_c.data_ptr<scalar_t>();
        const scalar_t* gy_ptr    = gy_c.data_ptr<scalar_t>();
        const scalar_t* rho_b_ptr = rho_bodies.data_ptr<scalar_t>();

        scalar_t* sdf_cc_p       = sdf_cc.data_ptr<scalar_t>();
        scalar_t* sdf_u_p        = sdf_u.data_ptr<scalar_t>();
        scalar_t* sdf_v_p        = sdf_v.data_ptr<scalar_t>();
        scalar_t* bU_p           = body_u.data_ptr<scalar_t>();
        scalar_t* bV_p           = body_v.data_ptr<scalar_t>();
        scalar_t* w_rho_p        = winning_rho_cc.data_ptr<scalar_t>();

        const scalar_t* u_p      = up_c.data_ptr<scalar_t>();
        const scalar_t* v_p      = vp_c.data_ptr<scalar_t>();
        const scalar_t* p_p      = pp_c.data_ptr<scalar_t>();
        const scalar_t* nx_p     = nxc_c.data_ptr<scalar_t>();
        const scalar_t* ny_p     = nyc_c.data_ptr<scalar_t>();
        const scalar_t* nr_p     = nr_c.data_ptr<scalar_t>();
        const int64_t   nr_size  = nu_rho_field.numel();

        const scalar_t neg_half_h = -(scalar_t)(0.5 * h_grid);
        const scalar_t inv_h_s    = (scalar_t)(1.0 / h_grid);
        const scalar_t eps_b      = (scalar_t)eps_body;
        const scalar_t eps_s      = (scalar_t)eps_solver;
        const scalar_t pi_v       = (scalar_t)3.141592653589793;
        const scalar_t inv_2eps   = (scalar_t)0.5 / eps_b;
        const scalar_t pi_over_eb = pi_v / eps_b;
        const double   h2_d       = (double)h2;

        const scalar_t band_lo = (eps_s - eps_b) < (-eps_b) ? (eps_s - eps_b) : (-eps_b);
        const scalar_t band_hi = (eps_s + eps_b) > ( eps_b) ? (eps_s + eps_b) : ( eps_b);

        std::vector<double> accs(static_cast<size_t>(B) * 6, 0.0);

        for (int b = 0; b < B; ++b) {
            const int Ai = (int)dim_[b*2 + 0];
            const int Aj = (int)dim_[b*2 + 1];
            const int vol = Ai * Aj;
            if (vol <= 0) continue;

            const int i0_b = (int)lo[b*2 + 0];
            const int j0_b = (int)lo[b*2 + 1];

            const scalar_t* F_b = F_ptr + F_off[b];
            const int Mx = (int)shapes[b*2 + 0];
            const int My = (int)shapes[b*2 + 1];

            const scalar_t* M   = meta + b*7;
            const scalar_t bx0  = M[0], by0 = M[1];
            const scalar_t idx_ = M[4], idy_ = M[5];

            const scalar_t* K   = kin_ptr + b*11;
            const scalar_t r00  = K[0], r01 = K[1];
            const scalar_t r10  = K[2], r11 = K[3];
            const scalar_t bp_x = K[4], bp_y = K[5];
            const scalar_t cm_x = K[6], cm_y = K[7];
            const scalar_t lv_x = K[8], lv_y = K[9];
            const scalar_t om   = K[10];

            const scalar_t du_x = neg_half_h * r00, du_y = neg_half_h * r10;
            const scalar_t dv_x = neg_half_h * r01, dv_y = neg_half_h * r11;

            const scalar_t rho_body_b = rho_b_ptr[b];

            std::vector<scalar_t> sparse_buf(vol);

            at::parallel_for(0, vol, 1024, [&](int64_t _begin, int64_t _end) {
            for (int local = (int)_begin; local < (int)_end; ++local) {
                const int di = local / Aj;
                const int dj = local - di * Aj;
                const int i  = i0_b + di;
                const int j  = j0_b + dj;
                const std::int64_t g_idx = (std::int64_t)i * Ngy + j;

                const scalar_t xc = gx_ptr[i];
                const scalar_t yc = gy_ptr[j];
                const scalar_t dxw = xc - bp_x, dyw = yc - bp_y;
                const scalar_t bxq = r00 * dxw + r01 * dyw;
                const scalar_t byq = r10 * dxw + r11 * dyw;

                auto sample = [&](scalar_t xqs, scalar_t yqs) -> scalar_t {
                    if (interp_method == 1) {
                        return biquadratic_sample_uniform_2d<scalar_t>(
                            F_b, Mx, My, bx0, by0, idx_, idy_, xqs, yqs);
                    }
                    return bilinear_sample_uniform_2d<scalar_t>(
                        F_b, Mx, My, bx0, by0, idx_, idy_, xqs, yqs);
                };

                const scalar_t s_cc = sample(bxq, byq);
                sparse_buf[local] = s_cc;
                if (s_cc < sdf_cc_p[g_idx]) {
                    sdf_cc_p[g_idx] = s_cc;
                    w_rho_p[g_idx]  = rho_body_b;
                }
                {
                    const scalar_t s = sample(bxq + du_x, byq + du_y);
                    if (s < sdf_u_p[g_idx]) {
                        sdf_u_p[g_idx] = s;
                        bU_p[g_idx] = lv_x - om * (yc - cm_y);
                    }
                }
                {
                    const scalar_t s = sample(bxq + dv_x, byq + dv_y);
                    if (s < sdf_v_p[g_idx]) {
                        sdf_v_p[g_idx] = s;
                        bV_p[g_idx] = lv_y + om * (xc - cm_x);
                    }
                }
            }
            });

            double* lb = accs.data() + (size_t)b * 6;
            std::mutex acc_mtx;

            at::parallel_for(0, vol, 2048, [&](int64_t _begin, int64_t _end) {
            double local8[6] = {0,0,0,0,0,0};
            for (int local = (int)_begin; local < (int)_end; ++local) {
                const scalar_t sdf = sparse_buf[local];
                if (sdf <= band_lo || sdf >= band_hi) continue;

                const int di = local / Aj;
                const int dj = local - di * Aj;
                const int i  = i0_b + di;
                const int j  = j0_b + dj;
                const std::int64_t g_idx = (std::int64_t)i * Ngy + j;

                scalar_t delta_visc = 0;
                const scalar_t d_visc = sdf - eps_s;
                if (d_visc > -eps_b && d_visc < eps_b)
                    delta_visc = ((scalar_t)1 + std::cos(pi_over_eb * d_visc)) * inv_2eps;
                scalar_t delta_pres = 0;
                if (sdf > -eps_b && sdf < eps_b)
                    delta_pres = ((scalar_t)1 + std::cos(pi_over_eb * sdf)) * inv_2eps;
                if (delta_visc == (scalar_t)0 && delta_pres == (scalar_t)0) continue;

                if (delta_order == 2) {
                    const scalar_t h_gs   = (scalar_t)h_grid;
                    const scalar_t inv_h2 = inv_h_s;
                    scalar_t sdf_xp = (di < Ai-1) ? sparse_buf[local + Aj] : sdf;
                    scalar_t sdf_xm = (di > 0)    ? sparse_buf[local - Aj] : sdf;
                    scalar_t cx     = (di > 0 && di < Ai-1) ? (scalar_t)0.5 : (scalar_t)1.0;
                    scalar_t dsdx   = cx * (sdf_xp - sdf_xm) * inv_h2;
                    scalar_t sdf_yp = (dj < Aj-1) ? sparse_buf[local + 1] : sdf;
                    scalar_t sdf_ym = (dj > 0)    ? sparse_buf[local - 1] : sdf;
                    scalar_t cy     = (dj > 0 && dj < Aj-1) ? (scalar_t)0.5 : (scalar_t)1.0;
                    scalar_t dsdy   = cy * (sdf_yp - sdf_ym) * inv_h2;
                    scalar_t grad_mag = std::sqrt(dsdx*dsdx + dsdy*dsdy);
                    if (grad_mag < (scalar_t)1e-3) grad_mag = (scalar_t)1e-3;
                    const scalar_t inv_grad = (scalar_t)1.0 / grad_mag;
                    delta_visc *= inv_grad;
                    delta_pres *= inv_grad;
                    (void)h_gs;
                }

                const scalar_t nu_rho_val = (nr_size == 1) ? nr_p[0] : nr_p[g_idx];
                const scalar_t nx = nx_p[g_idx];
                const scalar_t ny = ny_p[g_idx];

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
                    dudx = (u_p[ip1 * Ngy + j] - u_p[i * Ngy + j]) * inv_h_s;
                } else {
                    dudx = (u_p[i * Ngy + j] - u_p[im1 * Ngy + j]) * inv_h_s;
                }

                scalar_t dvdy;
                if (j + 1 < Ngy) {
                    dvdy = (v_p[i * Ngy + jp1] - v_p[i * Ngy + j]) * inv_h_s;
                } else {
                    dvdy = (v_p[i * Ngy + j] - v_p[i * Ngy + jm1]) * inv_h_s;
                }

                const scalar_t u_cc_jm2 = (scalar_t)0.5 * (u_p[i * Ngy + jm2] + u_p[ip1 * Ngy + jm2]);
                const scalar_t u_cc_jm1 = (scalar_t)0.5 * (u_p[i * Ngy + jm1] + u_p[ip1 * Ngy + jm1]);
                const scalar_t u_cc_j0  = (scalar_t)0.5 * (u_p[i * Ngy + j  ] + u_p[ip1 * Ngy + j  ]);
                const scalar_t u_cc_jp1 = (scalar_t)0.5 * (u_p[i * Ngy + jp1] + u_p[ip1 * Ngy + jp1]);
                const scalar_t u_cc_jp2 = (scalar_t)0.5 * (u_p[i * Ngy + jp2] + u_p[ip1 * Ngy + jp2]);

                scalar_t dudy;
                if (Ngy >= 3) {
                    if (j == 0) {
                        dudy = ((scalar_t)(-3) * u_cc_j0 + (scalar_t)4 * u_cc_jp1 - u_cc_jp2)
                             * (scalar_t)0.5 * inv_h_s;
                    } else if (j == Ngy - 1) {
                        dudy = ((scalar_t)3 * u_cc_j0 - (scalar_t)4 * u_cc_jm1 + u_cc_jm2)
                             * (scalar_t)0.5 * inv_h_s;
                    } else {
                        dudy = (u_cc_jp1 - u_cc_jm1) * (scalar_t)0.5 * inv_h_s;
                    }
                } else {
                    dudy = (u_cc_jp1 - u_cc_jm1) * (scalar_t)0.5 * inv_h_s;
                }

                const scalar_t v_cc_im2 = (scalar_t)0.5 * (v_p[im2 * Ngy + j] + v_p[im2 * Ngy + jp1]);
                const scalar_t v_cc_im1 = (scalar_t)0.5 * (v_p[im1 * Ngy + j] + v_p[im1 * Ngy + jp1]);
                const scalar_t v_cc_i0  = (scalar_t)0.5 * (v_p[i   * Ngy + j] + v_p[i   * Ngy + jp1]);
                const scalar_t v_cc_ip1 = (scalar_t)0.5 * (v_p[ip1 * Ngy + j] + v_p[ip1 * Ngy + jp1]);
                const scalar_t v_cc_ip2 = (scalar_t)0.5 * (v_p[ip2 * Ngy + j] + v_p[ip2 * Ngy + jp1]);

                scalar_t dvdx;
                if (Ngx >= 3) {
                    if (i == 0) {
                        dvdx = ((scalar_t)(-3) * v_cc_i0 + (scalar_t)4 * v_cc_ip1 - v_cc_ip2)
                             * (scalar_t)0.5 * inv_h_s;
                    } else if (i == Ngx - 1) {
                        dvdx = ((scalar_t)3 * v_cc_i0 - (scalar_t)4 * v_cc_im1 + v_cc_im2)
                             * (scalar_t)0.5 * inv_h_s;
                    } else {
                        dvdx = (v_cc_ip1 - v_cc_im1) * (scalar_t)0.5 * inv_h_s;
                    }
                } else {
                    dvdx = (v_cc_ip1 - v_cc_im1) * (scalar_t)0.5 * inv_h_s;
                }

                const scalar_t xs = nu_rho_val * (2*dudx*nx + (dudy+dvdx)*ny);
                const scalar_t ys = nu_rho_val * ((dvdx+dudy)*nx + 2*dvdy*ny);

                const scalar_t p_c  = p_p[g_idx];
                const scalar_t pxv  = -p_c * nx;
                const scalar_t pyv  = -p_c * ny;

                const scalar_t xc = gx_ptr[i];
                const scalar_t yc = gy_ptr[j];
                const scalar_t arm_x = xc - cm_x;
                const scalar_t arm_y = yc - cm_y;

                const double fv_x = (double)(xs  * delta_visc);
                const double fv_y = (double)(ys  * delta_visc);
                const double fp_x = (double)(pxv * delta_pres);
                const double fp_y = (double)(pyv * delta_pres);

                local8[0] += fv_x;
                local8[1] += fv_y;
                local8[2] += (double)arm_x * fv_y - (double)arm_y * fv_x;
                local8[3] += fp_x;
                local8[4] += fp_y;
                local8[5] += (double)arm_x * fp_y - (double)arm_y * fp_x;
            }
            std::lock_guard<std::mutex> lk(acc_mtx);
            for (int c = 0; c < 6; ++c) lb[c] += local8[c];
            });
        }

        double* out_ptr = out.data_ptr<double>();
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < 6; ++c)
                out_ptr[b*6 + c] += accs[(size_t)b * 6 + c] * h2_d;
    });
}

// =====================================================================
//  CPU registration. Schemas live in ops.cpp.
// =====================================================================

TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("streaming_sdf_min_2d",                  &streaming_sdf_min_2d_cpu);
    m.impl("streaming_sdf_min_2d_multi",            &streaming_sdf_min_2d_multi_cpu);
    m.impl("bdim_forces_2d_multi",                  &bdim_forces_2d_multi_cpu);
    m.impl("streaming_sdf_forces_fused_2d_multi",   &streaming_sdf_forces_fused_2d_multi_cpu);
    m.impl("apply_bcs_2d",                          &apply_bcs_2d_cpu);
    m.impl("interpolate_2d",                        &interpolate_2d_cpu);
}

}  // namespace lilytorch_kernels
