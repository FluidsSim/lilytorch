// =====================================================================
//  streaming_sdf_regime_b_cpu.cpp
//
//  CPU twins for the Regime-B per-body private buffers + resolve pipeline.
//  Mirrors streaming_sdf_regime_b.cu line-for-line using at::parallel_for.
//
//  Two stages:
//  1. Min-stage: per-body at::parallel_for over body_vol[b] cells.
//     Computes raw SDF + staggered body velocity, writes to per-body
//     private flat buffers (single writer per (body, cell) — no atomics).
//  2. Resolve-stage: at::parallel_for over the union dirty AABB.
//     For each cell, loops over all B bodies, checks AABB coverage,
//     reads raw SDF from each covering body's private buffer, picks the
//     minimum, and writes the winner to the global output tensors.
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
#include "common/interp.h"

namespace lilytorch_kernels {


//  2-D min-stage: per-body parallel_for over body_vol[b]
// =====================================================================
template <typename scalar_t>
static void min_stage_2d(
    const scalar_t* F_flat, const int64_t* F_offsets,
    const int64_t* body_shapes, const scalar_t* body_meta,
    const scalar_t* kin,
    const int64_t* aabb_lo, const int64_t* aabb_dim,
    const scalar_t* gx, const scalar_t* gy,
    int Ngy, scalar_t half_h, int n_active, int interp,
    const int64_t* priv_offsets, const int64_t* active_idx,
    scalar_t* priv_sdf_cc, scalar_t* priv_sdf_u, scalar_t* priv_sdf_v,
    scalar_t* priv_body_u, scalar_t* priv_body_v)
{
    for (int a = 0; a < n_active; ++a) {
        const int b = (int)active_idx[a];
        int Ai = (int)aabb_dim[b*2+0], Aj = (int)aabb_dim[b*2+1];
        int body_vol = Ai * Aj;
        if (body_vol <= 0) continue;
        int i0 = (int)aabb_lo[b*2+0], j0 = (int)aabb_lo[b*2+1];
        int64_t off = priv_offsets[b];

        const scalar_t* F = F_flat + F_offsets[b];
        int Mx = (int)body_shapes[b*2+0], My = (int)body_shapes[b*2+1];
        const scalar_t* M = body_meta + b*7;
        scalar_t bx0 = M[0], by0 = M[1], idx_ = M[4], idy_ = M[5];
        const scalar_t* K = kin + b*11;
        scalar_t r00 = K[0], r01 = K[1], r10 = K[2], r11 = K[3];
        scalar_t bp_x = K[4], bp_y = K[5], cm_x = K[6], cm_y = K[7], lv_x = K[8], lv_y = K[9], om = K[10];

        scalar_t neg_hh = -half_h;
        scalar_t du_x = neg_hh * r00, du_y = neg_hh * r10;
        scalar_t dv_x = neg_hh * r01, dv_y = neg_hh * r11;

        at::parallel_for(0, body_vol, 0, [&](int64_t start, int64_t end) {
            for (int64_t local = start; local < end; ++local) {
                int di = (int)(local / Aj);
                int dj = (int)(local - di * Aj);
                int i = i0 + di, j = j0 + dj;

                scalar_t xc = gx[i], yc = gy[j];
                scalar_t dxw = xc - bp_x, dyw = yc - bp_y;
                scalar_t bxq = r00 * dxw + r01 * dyw;
                scalar_t byq = r10 * dxw + r11 * dyw;

                auto sample = [&](scalar_t xqs, scalar_t yqs) -> scalar_t {
                    if (interp == 1) return biquadratic_sample_uniform_2d<scalar_t>(F, Mx, My, bx0, by0, idx_, idy_, xqs, yqs);
                    return bilinear_sample_uniform_2d<scalar_t>(F, Mx, My, bx0, by0, idx_, idy_, xqs, yqs);
                };

                int64_t idx = off + local;
                priv_sdf_cc[idx] = sample(bxq, byq);
                priv_sdf_u[idx]  = sample(bxq + du_x, byq + du_y);
                priv_body_u[idx] = lv_x - om * (yc - cm_y);
                priv_sdf_v[idx]  = sample(bxq + dv_x, byq + dv_y);
                priv_body_v[idx] = lv_y + om * (xc - cm_x);
            }
        });
    }
}


// =====================================================================
//  2-D resolve-stage: parallel_for over union dirty AABB
// =====================================================================
template <typename scalar_t>
static void resolve_stage_2d(
    const int64_t* aabb_lo, const int64_t* aabb_dim,
    int Ngy, int B,
    int dirty_i0, int dirty_j0, int dirty_Ai, int dirty_Aj,
    const int64_t* priv_offsets,
    const scalar_t* priv_sdf_cc, const scalar_t* priv_sdf_u, const scalar_t* priv_sdf_v,
    const scalar_t* priv_body_u, const scalar_t* priv_body_v,
    scalar_t* sdf_cc, scalar_t* sdf_u, scalar_t* sdf_v,
    scalar_t* body_u, scalar_t* body_v,
    int32_t* owner_cc,
    scalar_t blend_eps)
{
    int64_t dirty_vol = (int64_t)dirty_Ai * dirty_Aj;
    const bool blend = blend_eps > scalar_t(0);
    at::parallel_for(0, dirty_vol, 0, [&](int64_t start, int64_t end) {
        for (int64_t local = start; local < end; ++local) {
            int di = (int)(local / dirty_Aj);
            int dj = (int)(local - di * dirty_Aj);
            if (di >= dirty_Ai) continue;
            int i = dirty_i0 + di, j = dirty_j0 + dj;
            int64_t g_idx = (int64_t)i * Ngy + j;

            // Per-stagger winner selection, mirroring _multi's independent
            // atomicMin on key_cc/key_u/key_v (see the CUDA resolve kernel).
            scalar_t best_cc = (scalar_t)1e4, best_u = (scalar_t)1e4, best_v = (scalar_t)1e4;
            int64_t win_cc = -1, win_u = -1, win_v = -1;
            int win_cc_body = -1;   // the cell-centre argmin, published as owner_cc

            // Softmin velocity blend (mirrors the _multi min/decode pair):
            // w_i = sigmoid(-s_i/blend_eps) per stagger, accumulated in
            // ascending-b order — deterministic, no num/den scratch buffers.
            scalar_t num_u = 0, den_u = 0, num_v = 0, den_v = 0;

            for (int b = 0; b < B; ++b) {
                int Ai = (int)aabb_dim[b*2+0], Aj = (int)aabb_dim[b*2+1];
                int i0 = (int)aabb_lo[b*2+0], j0 = (int)aabb_lo[b*2+1];
                if (i < i0 || i >= i0 + Ai || j < j0 || j >= j0 + Aj) continue;
                int di_b = i - i0, dj_b = j - j0;
                int loc = di_b * Aj + dj_b;
                int64_t idx = priv_offsets[b] + loc;
                const scalar_t s_cc = priv_sdf_cc[idx];
                const scalar_t s_u  = priv_sdf_u[idx];
                const scalar_t s_v  = priv_sdf_v[idx];
                if (s_cc < best_cc) { best_cc = s_cc; win_cc = idx; win_cc_body = b; }
                if (s_u  < best_u)  { best_u  = s_u;  win_u  = idx; }
                if (s_v  < best_v)  { best_v  = s_v;  win_v  = idx; }
                if (blend) {
                    const scalar_t wU = scalar_t(1) / (scalar_t(1) + std::exp(s_u / blend_eps));
                    const scalar_t wV = scalar_t(1) / (scalar_t(1) + std::exp(s_v / blend_eps));
                    num_u += wU * priv_body_u[idx]; den_u += wU;
                    num_v += wV * priv_body_v[idx]; den_v += wV;
                }
            }

            if (win_cc >= 0) {
                sdf_cc[g_idx] = best_cc;
                sdf_u[g_idx]  = best_u;
                sdf_v[g_idx]  = best_v;
                const scalar_t den_tol = scalar_t(1e-6);
                body_u[g_idx] = (blend && den_u > den_tol) ? num_u / den_u
                                                           : priv_body_u[win_u];
                body_v[g_idx] = (blend && den_v > den_tol) ? num_v / den_v
                                                           : priv_body_v[win_v];
                if (owner_cc) owner_cc[g_idx] = win_cc_body;
            }
        }
    });
}


// =====================================================================
//  3-D min-stage: per-body parallel_for over body_vol[b]
// =====================================================================
template <typename scalar_t>
static void min_stage_3d(
    const scalar_t* F_flat, const int64_t* F_offsets,
    const int64_t* body_shapes, const scalar_t* body_meta,
    const scalar_t* kin,
    const int64_t* aabb_lo, const int64_t* aabb_dim,
    const scalar_t* gx, const scalar_t* gy, const scalar_t* gz,
    int Ngy, int Ngz, scalar_t half_h, int n_active, int interp,
    const int64_t* priv_offsets, const int64_t* active_idx,
    scalar_t* priv_sdf_cc, scalar_t* priv_sdf_u,
    scalar_t* priv_sdf_v, scalar_t* priv_sdf_w,
    scalar_t* priv_body_u, scalar_t* priv_body_v, scalar_t* priv_body_w)
{
    for (int a = 0; a < n_active; ++a) {
        const int b = (int)active_idx[a];
        int Ai = (int)aabb_dim[b*3+0], Aj = (int)aabb_dim[b*3+1], Ak = (int)aabb_dim[b*3+2];
        int body_vol = Ai * Aj * Ak;
        if (body_vol <= 0) continue;
        int i0 = (int)aabb_lo[b*3+0], j0 = (int)aabb_lo[b*3+1], k0 = (int)aabb_lo[b*3+2];
        int64_t off = priv_offsets[b];

        const scalar_t* F = F_flat + F_offsets[b];
        int Mx = (int)body_shapes[b*3+0], My = (int)body_shapes[b*3+1], Mz = (int)body_shapes[b*3+2];
        const scalar_t* M = body_meta + b*10;
        scalar_t bx0 = M[0], by0 = M[1], bz0 = M[2], idx_ = M[6], idy_ = M[7], idz_ = M[8];
        const scalar_t* K = kin + b*21;
        scalar_t r00=K[0], r01=K[1], r02=K[2], r10=K[3], r11=K[4], r12=K[5], r20=K[6], r21=K[7], r22=K[8];
        scalar_t bp_x=K[9], bp_y=K[10], bp_z=K[11], cm_x=K[12], cm_y=K[13], cm_z=K[14];
        scalar_t lv_x=K[15], lv_y=K[16], lv_z=K[17], av_x=K[18], av_y=K[19], av_z=K[20];

        scalar_t neg_hh = -half_h;
        scalar_t du_x = neg_hh*r00, du_y = neg_hh*r10, du_z = neg_hh*r20;
        scalar_t dv_x = neg_hh*r01, dv_y = neg_hh*r11, dv_z = neg_hh*r21;
        scalar_t dw_x = neg_hh*r02, dw_y = neg_hh*r12, dw_z = neg_hh*r22;

        int stride_y = Ak;
        int stride_x = Aj * Ak;

        at::parallel_for(0, body_vol, 0, [&](int64_t start, int64_t end) {
            for (int64_t local = start; local < end; ++local) {
                int di = (int)(local / stride_x);
                int rem = (int)(local - di * stride_x);
                int dj = rem / stride_y;
                int dk = rem - dj * stride_y;
                int i = i0 + di, j = j0 + dj, k = k0 + dk;

                scalar_t xc = gx[i], yc = gy[j], zc = gz[k];
                scalar_t dxw = xc - bp_x, dyw = yc - bp_y, bzw = zc - bp_z;
                scalar_t bxq = r00*dxw + r01*dyw + r02*bzw;
                scalar_t byq = r10*dxw + r11*dyw + r12*bzw;
                scalar_t bzq = r20*dxw + r21*dyw + r22*bzw;

                auto sample = [&](scalar_t xqs, scalar_t yqs, scalar_t zqs) -> scalar_t {
                    if (interp == 1) return triquadratic_sample_uniform<scalar_t>(F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_, xqs, yqs, zqs);
                    return trilinear_sample_uniform<scalar_t>(F, Mx, My, Mz, bx0, by0, bz0, idx_, idy_, idz_, xqs, yqs, zqs);
                };

                int64_t idx = off + local;
                priv_sdf_cc[idx] = sample(bxq, byq, bzq);
                priv_sdf_u[idx]  = sample(bxq+du_x, byq+du_y, bzq+du_z);
                priv_body_u[idx] = lv_x + av_y*(zc - cm_z) - av_z*(yc - cm_y);
                priv_sdf_v[idx]  = sample(bxq+dv_x, byq+dv_y, bzq+dv_z);
                priv_body_v[idx] = lv_y + av_z*(xc - cm_x) - av_x*(zc - cm_z);
                priv_sdf_w[idx]  = sample(bxq+dw_x, byq+dw_y, bzq+dw_z);
                priv_body_w[idx] = lv_z + av_x*(yc - cm_y) - av_y*(xc - cm_x);
            }
        });
    }
}


// =====================================================================
//  3-D resolve-stage: parallel_for over union dirty AABB
// =====================================================================
template <typename scalar_t>
static void resolve_stage_3d(
    const int64_t* aabb_lo, const int64_t* aabb_dim,
    int Ngy, int Ngz, int B,
    int dirty_i0, int dirty_j0, int dirty_k0, int dirty_Ai, int dirty_Aj, int dirty_Ak,
    const int64_t* priv_offsets,
    const scalar_t* priv_sdf_cc, const scalar_t* priv_sdf_u,
    const scalar_t* priv_sdf_v, const scalar_t* priv_sdf_w,
    const scalar_t* priv_body_u, const scalar_t* priv_body_v, const scalar_t* priv_body_w,
    scalar_t* sdf_cc, scalar_t* sdf_u, scalar_t* sdf_v, scalar_t* sdf_w,
    scalar_t* body_u, scalar_t* body_v, scalar_t* body_w,
    int32_t* owner_cc,
    scalar_t blend_eps)
{
    int64_t dirty_vol = (int64_t)dirty_Ai * dirty_Aj * dirty_Ak;
    const bool blend = blend_eps > scalar_t(0);
    at::parallel_for(0, dirty_vol, 0, [&](int64_t start, int64_t end) {
        for (int64_t local = start; local < end; ++local) {
            int rem = (int)(local / dirty_Ak);
            int dk = (int)(local - rem * dirty_Ak);
            int di = rem / dirty_Aj;
            int dj = rem - di * dirty_Aj;
            if (di >= dirty_Ai) continue;
            int i = dirty_i0 + di, j = dirty_j0 + dj, k = dirty_k0 + dk;
            int64_t g_idx = ((int64_t)i * Ngy + j) * Ngz + k;

            // Per-stagger winner selection — see the 2-D resolve-stage note.
            scalar_t best_cc = (scalar_t)1e4, best_u = (scalar_t)1e4,
                     best_v = (scalar_t)1e4,  best_w = (scalar_t)1e4;
            int64_t win_cc = -1, win_u = -1, win_v = -1, win_w = -1;
            int win_cc_body = -1;   // the cell-centre argmin, published as owner_cc

            // Softmin velocity blend — see the 2-D resolve-stage note.
            scalar_t num_u = 0, den_u = 0, num_v = 0, den_v = 0, num_w = 0, den_w = 0;

            for (int b = 0; b < B; ++b) {
                int Ai = (int)aabb_dim[b*3+0], Aj = (int)aabb_dim[b*3+1], Ak = (int)aabb_dim[b*3+2];
                int i0 = (int)aabb_lo[b*3+0], j0 = (int)aabb_lo[b*3+1], k0 = (int)aabb_lo[b*3+2];
                if (i < i0 || i >= i0 + Ai || j < j0 || j >= j0 + Aj || k < k0 || k >= k0 + Ak) continue;
                int di_b = i - i0, dj_b = j - j0, dk_b = k - k0;
                int loc = (di_b * Aj + dj_b) * Ak + dk_b;
                int64_t idx = priv_offsets[b] + loc;
                const scalar_t s_cc = priv_sdf_cc[idx];
                const scalar_t s_u  = priv_sdf_u[idx];
                const scalar_t s_v  = priv_sdf_v[idx];
                const scalar_t s_w  = priv_sdf_w[idx];
                if (s_cc < best_cc) { best_cc = s_cc; win_cc = idx; win_cc_body = b; }
                if (s_u  < best_u)  { best_u  = s_u;  win_u  = idx; }
                if (s_v  < best_v)  { best_v  = s_v;  win_v  = idx; }
                if (s_w  < best_w)  { best_w  = s_w;  win_w  = idx; }
                if (blend) {
                    const scalar_t wU = scalar_t(1) / (scalar_t(1) + std::exp(s_u / blend_eps));
                    const scalar_t wV = scalar_t(1) / (scalar_t(1) + std::exp(s_v / blend_eps));
                    const scalar_t wW = scalar_t(1) / (scalar_t(1) + std::exp(s_w / blend_eps));
                    num_u += wU * priv_body_u[idx]; den_u += wU;
                    num_v += wV * priv_body_v[idx]; den_v += wV;
                    num_w += wW * priv_body_w[idx]; den_w += wW;
                }
            }

            if (win_cc >= 0) {
                sdf_cc[g_idx] = best_cc;
                sdf_u[g_idx]  = best_u;
                sdf_v[g_idx]  = best_v;
                sdf_w[g_idx]  = best_w;
                const scalar_t den_tol = scalar_t(1e-6);
                body_u[g_idx] = (blend && den_u > den_tol) ? num_u / den_u
                                                           : priv_body_u[win_u];
                body_v[g_idx] = (blend && den_v > den_tol) ? num_v / den_v
                                                           : priv_body_v[win_v];
                body_w[g_idx] = (blend && den_w > den_tol) ? num_w / den_w
                                                           : priv_body_w[win_w];
                if (owner_cc) owner_cc[g_idx] = win_cc_body;
            }
        }
    });
}


// =====================================================================
//  Launcher functions (CPU)
// =====================================================================

void streaming_sdf_stag_2d_resolve_cpu(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes, const at::Tensor& body_meta, const at::Tensor& kin,
    const at::Tensor& aabb_lo, const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy,
    double h_grid, int64_t /*max_vol_per_body*/,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v,
    at::Tensor body_u, at::Tensor body_v, int64_t interp,
    int64_t dirty_i0, int64_t dirty_j0, int64_t dirty_Ai, int64_t dirty_Aj,
    const at::Tensor& priv_offsets,
    at::Tensor priv_sdf_cc, at::Tensor priv_sdf_u, at::Tensor priv_sdf_v,
    at::Tensor priv_body_u, at::Tensor priv_body_v,
    const at::Tensor& active_idx,
    at::Tensor owner_cc,
    double blend_eps)
{
    int B = (int)aabb_dim.size(0);
    if (B <= 0 || dirty_Ai <= 0 || dirty_Aj <= 0) return;
    int n_active = (int)active_idx.numel();
    int Ngy = (int)gy.numel();
    // A size<=1 tensor is the caller's "owner field not wanted" placeholder.
    int32_t* owner_ptr = (owner_cc.numel() > 1) ? owner_cc.data_ptr<int32_t>()
                                                : nullptr;

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "sdf_stag_2d_resolve_cpu", [&] {
        // Min-stage
        min_stage_2d<scalar_t>(
            F_flat.data_ptr<scalar_t>(), F_offsets.data_ptr<int64_t>(),
            body_shapes.data_ptr<int64_t>(), body_meta.data_ptr<scalar_t>(),
            kin.data_ptr<scalar_t>(),
            aabb_lo.data_ptr<int64_t>(), aabb_dim.data_ptr<int64_t>(),
            gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(),
            Ngy, (scalar_t)(0.5 * h_grid), n_active, (int)interp,
            priv_offsets.data_ptr<int64_t>(), active_idx.data_ptr<int64_t>(),
            priv_sdf_cc.data_ptr<scalar_t>(), priv_sdf_u.data_ptr<scalar_t>(), priv_sdf_v.data_ptr<scalar_t>(),
            priv_body_u.data_ptr<scalar_t>(), priv_body_v.data_ptr<scalar_t>());

        // Resolve-stage
        resolve_stage_2d<scalar_t>(
            aabb_lo.data_ptr<int64_t>(), aabb_dim.data_ptr<int64_t>(),
            Ngy, B,
            (int)dirty_i0, (int)dirty_j0, (int)dirty_Ai, (int)dirty_Aj,
            priv_offsets.data_ptr<int64_t>(),
            priv_sdf_cc.data_ptr<scalar_t>(), priv_sdf_u.data_ptr<scalar_t>(), priv_sdf_v.data_ptr<scalar_t>(),
            priv_body_u.data_ptr<scalar_t>(), priv_body_v.data_ptr<scalar_t>(),
            sdf_cc.data_ptr<scalar_t>(), sdf_u.data_ptr<scalar_t>(), sdf_v.data_ptr<scalar_t>(),
            body_u.data_ptr<scalar_t>(), body_v.data_ptr<scalar_t>(),
            owner_ptr,
            (scalar_t)blend_eps);
    });
}

void streaming_sdf_stag_3d_resolve_cpu(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes, const at::Tensor& body_meta, const at::Tensor& kin,
    const at::Tensor& aabb_lo, const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy, const at::Tensor& gz,
    double h_grid, int64_t /*max_vol_per_body*/,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v, at::Tensor sdf_w,
    at::Tensor body_u, at::Tensor body_v, at::Tensor body_w, int64_t interp,
    int64_t dirty_i0, int64_t dirty_j0, int64_t dirty_k0,
    int64_t dirty_Ai, int64_t dirty_Aj, int64_t dirty_Ak,
    const at::Tensor& priv_offsets,
    at::Tensor priv_sdf_cc, at::Tensor priv_sdf_u, at::Tensor priv_sdf_v, at::Tensor priv_sdf_w,
    at::Tensor priv_body_u, at::Tensor priv_body_v, at::Tensor priv_body_w,
    const at::Tensor& active_idx,
    at::Tensor owner_cc,
    double blend_eps)
{
    int B = (int)aabb_dim.size(0);
    int n_active = (int)active_idx.numel();
    if (B <= 0 || dirty_Ai <= 0 || dirty_Aj <= 0 || dirty_Ak <= 0) return;
    int Ngy = (int)gy.numel(), Ngz = (int)gz.numel();
    // A size<=1 tensor is the caller's "owner field not wanted" placeholder.
    int32_t* owner_ptr = (owner_cc.numel() > 1) ? owner_cc.data_ptr<int32_t>()
                                                : nullptr;

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "sdf_stag_3d_resolve_cpu", [&] {
        // Min-stage
        min_stage_3d<scalar_t>(
            F_flat.data_ptr<scalar_t>(), F_offsets.data_ptr<int64_t>(),
            body_shapes.data_ptr<int64_t>(), body_meta.data_ptr<scalar_t>(),
            kin.data_ptr<scalar_t>(),
            aabb_lo.data_ptr<int64_t>(), aabb_dim.data_ptr<int64_t>(),
            gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(),
            Ngy, Ngz, (scalar_t)(0.5 * h_grid), n_active, (int)interp,
            priv_offsets.data_ptr<int64_t>(), active_idx.data_ptr<int64_t>(),
            priv_sdf_cc.data_ptr<scalar_t>(), priv_sdf_u.data_ptr<scalar_t>(),
            priv_sdf_v.data_ptr<scalar_t>(), priv_sdf_w.data_ptr<scalar_t>(),
            priv_body_u.data_ptr<scalar_t>(), priv_body_v.data_ptr<scalar_t>(), priv_body_w.data_ptr<scalar_t>());

        // Resolve-stage
        resolve_stage_3d<scalar_t>(
            aabb_lo.data_ptr<int64_t>(), aabb_dim.data_ptr<int64_t>(),
            Ngy, Ngz, B,
            (int)dirty_i0, (int)dirty_j0, (int)dirty_k0,
            (int)dirty_Ai, (int)dirty_Aj, (int)dirty_Ak,
            priv_offsets.data_ptr<int64_t>(),
            priv_sdf_cc.data_ptr<scalar_t>(), priv_sdf_u.data_ptr<scalar_t>(),
            priv_sdf_v.data_ptr<scalar_t>(), priv_sdf_w.data_ptr<scalar_t>(),
            priv_body_u.data_ptr<scalar_t>(), priv_body_v.data_ptr<scalar_t>(), priv_body_w.data_ptr<scalar_t>(),
            sdf_cc.data_ptr<scalar_t>(), sdf_u.data_ptr<scalar_t>(), sdf_v.data_ptr<scalar_t>(), sdf_w.data_ptr<scalar_t>(),
            body_u.data_ptr<scalar_t>(), body_v.data_ptr<scalar_t>(), body_w.data_ptr<scalar_t>(),
            owner_ptr,
            (scalar_t)blend_eps);
    });
}


TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("streaming_sdf_stag_2d_resolve", &streaming_sdf_stag_2d_resolve_cpu);
    m.impl("streaming_sdf_stag_3d_resolve", &streaming_sdf_stag_3d_resolve_cpu);
}

}  // namespace lilytorch_kernels
