// =====================================================================
//  streaming.cu — Per-body private buffers + resolve kernels (2-D and 3-D).
//
//  Replaces the union-AABB packed-key ``atomicMin`` pipeline for
//  overlapping-body regimes.  Two kernels:
//
//  1. **Min-kernel** (fanned per-body, gridDim.y = B): each block-row
//     processes one body's AABB cells.  Computes raw SDF + staggered
//     body velocity and writes them into per-body linear private buffers.
//     Single writer per (body, cell) — no atomics, no packed key.
//
//  2. **Resolve kernel** (over the full union dirty AABB): one thread per
//     cell; loops over all B bodies, checks AABB coverage, reads the raw
//     SDF from each covering body's private buffer, picks the minimum
//     (no atomics — single thread per global cell), and writes the winner
//     to the global output tensors.  Full fp64 precision.
//
//  Private buffers are flat tensors with per-body cumulative offsets:
//    priv_offsets[b] = sum_{i < b} cap[i]      (cap = grow-only body_vol)
//    priv_sdf_cc[priv_offsets[b] + local]  ←  body b, local idx
//
//  The min-kernel restreams only the bodies listed in ``active_idx``; the
//  resolve kernel always sweeps all B.  A body left out of active_idx keeps
//  contributing the slab it wrote on an earlier step, which is how a STATIC
//  body is skipped without changing the union.  Because the capacities are
//  grow-only, priv_offsets is stable across steps and those slabs stay put.
//
//  The resolve kernel also publishes ``owner_cc``: the int32 body index that
//  won the CELL-CENTRE min, or -1 where no body covers the cell.  It is the
//  argmin the kernel already computes for ``win_cc``, so recording it is free,
//  and it lets the post-step force readout split the union force by table
//  lookup instead of recomputing the same argmin per (body, cell).  Because it
//  is written exactly where ``sdf_cc`` is written, it inherits sdf_cc's dirty-
//  region staleness semantics unchanged.  Pass a size<=1 tensor to skip it.
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <c10/util/ArrayRef.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "../common/interp.h"

namespace lilytorch_kernels {


//  2-D min kernel — fanned per-body, writes raw SDF + vel to private bufs
// =====================================================================
template <typename scalar_t>
__global__ void regime_b_min_kernel_2d(
    const scalar_t* __restrict__ F_flat, const int64_t* __restrict__ F_offsets,
    const int64_t* __restrict__ body_shapes, const scalar_t* __restrict__ body_meta,
    const scalar_t* __restrict__ kin,
    const int64_t* __restrict__ aabb_lo, const int64_t* __restrict__ aabb_dim,
    const scalar_t* __restrict__ gx, const scalar_t* __restrict__ gy,
    int Ngy, scalar_t half_h, int n_active, int interp,
    const int64_t* __restrict__ priv_offsets,
    const int64_t* __restrict__ active_idx,
    scalar_t* __restrict__ priv_sdf_cc, scalar_t* __restrict__ priv_sdf_u, scalar_t* __restrict__ priv_sdf_v,
    scalar_t* __restrict__ priv_body_u, scalar_t* __restrict__ priv_body_v)
{
    if (blockIdx.y >= n_active) return;
    int b = (int)active_idx[blockIdx.y];  // body index

    int Ai = (int)aabb_dim[b*2+0], Aj = (int)aabb_dim[b*2+1];
    int body_vol = Ai * Aj;
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

    for (int local = blockIdx.x * blockDim.x + threadIdx.x; local < body_vol; local += gridDim.x * blockDim.x) {
        int di = local / Aj;
        int dj = local - di * Aj;
        int i = i0 + di;
        int j = j0 + dj;

        scalar_t xc = gx[i], yc = gy[j];
        scalar_t dxw = xc - bp_x, dyw = yc - bp_y;
        scalar_t bxq = r00 * dxw + r01 * dyw;
        scalar_t byq = r10 * dxw + r11 * dyw;

        int64_t idx = off + local;
        priv_sdf_cc[idx] = sdf_sample_dispatch_2d<scalar_t>(interp, F, Mx, My, bx0, by0, idx_, idy_, bxq, byq);
        priv_sdf_u[idx]  = sdf_sample_dispatch_2d<scalar_t>(interp, F, Mx, My, bx0, by0, idx_, idy_, bxq + du_x, byq + du_y);
        priv_body_u[idx] = lv_x - om * (yc - cm_y);
        priv_sdf_v[idx]  = sdf_sample_dispatch_2d<scalar_t>(interp, F, Mx, My, bx0, by0, idx_, idy_, bxq + dv_x, byq + dv_y);
        priv_body_v[idx] = lv_y + om * (xc - cm_x);
    }
}


// =====================================================================
//  2-D resolve kernel — one thread per cell in union dirty AABB
// =====================================================================
template <typename scalar_t>
__global__ void regime_b_resolve_kernel_2d(
    const int64_t* __restrict__ aabb_lo, const int64_t* __restrict__ aabb_dim,
    int Ngy, int B,
    int dirty_i0, int dirty_j0, int dirty_Ai, int dirty_Aj,
    const int64_t* __restrict__ priv_offsets,
    const scalar_t* __restrict__ priv_sdf_cc, const scalar_t* __restrict__ priv_sdf_u, const scalar_t* __restrict__ priv_sdf_v,
    const scalar_t* __restrict__ priv_body_u, const scalar_t* __restrict__ priv_body_v,
    scalar_t* __restrict__ sdf_cc, scalar_t* __restrict__ sdf_u, scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ body_u, scalar_t* __restrict__ body_v,
    int32_t* __restrict__ owner_cc,
    const scalar_t blend_eps)
{
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    int di = local / dirty_Aj;
    int dj = local - di * dirty_Aj;
    if (di >= dirty_Ai) return;
    int i = dirty_i0 + di;
    int j = dirty_j0 + dj;
    int64_t g_idx = (int64_t)i * Ngy + j;

    // Winner selection is PER STAGGER, mirroring _multi's independent
    // atomicMin on key_cc/key_u/key_v: the union SDF at the u-face is
    // min_b s_u(b), whose winner can differ from the cell-centre winner
    // near inter-body seams (up to ~h/2 bias if the cc winner were reused).
    // Velocity follows each stagger's own winner (or the softmin blend).
    scalar_t best_cc = (scalar_t)1e4, best_u = (scalar_t)1e4, best_v = (scalar_t)1e4;
    int64_t win_cc = -1, win_u = -1, win_v = -1;
    int win_cc_body = -1;   // the cell-centre argmin, published as owner_cc

    // Softmin velocity blend (mirrors the _multi min-kernel/decode pair):
    // w_i = sigmoid(-s_i/blend_eps) per stagger; accumulated in registers
    // over the covering bodies in ascending-b order — deterministic, no
    // atomics, no num/den scratch buffers.
    const bool blend = blend_eps > scalar_t(0);
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
            const scalar_t wU = scalar_t(1) / (scalar_t(1) + exp(s_u / blend_eps));
            const scalar_t wV = scalar_t(1) / (scalar_t(1) + exp(s_v / blend_eps));
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
    // else: cell not covered by any body → keep pre-existing values (FAR/0/-1)
}


// =====================================================================
//  3-D min kernel — fanned per-body
// =====================================================================
template <typename scalar_t>
__global__ void regime_b_min_kernel_3d(
    const scalar_t* __restrict__ F_flat, const int64_t* __restrict__ F_offsets,
    const int64_t* __restrict__ body_shapes, const scalar_t* __restrict__ body_meta,
    const scalar_t* __restrict__ kin,
    const int64_t* __restrict__ aabb_lo, const int64_t* __restrict__ aabb_dim,
    const scalar_t* __restrict__ gx, const scalar_t* __restrict__ gy, const scalar_t* __restrict__ gz,
    int Ngy, int Ngz, scalar_t half_h, int n_active, int interp,
    const int64_t* __restrict__ priv_offsets,
    const int64_t* __restrict__ active_idx,
    scalar_t* __restrict__ priv_sdf_cc, scalar_t* __restrict__ priv_sdf_u,
    scalar_t* __restrict__ priv_sdf_v, scalar_t* __restrict__ priv_sdf_w,
    scalar_t* __restrict__ priv_body_u, scalar_t* __restrict__ priv_body_v, scalar_t* __restrict__ priv_body_w)
{
    if (blockIdx.y >= n_active) return;
    int b = (int)active_idx[blockIdx.y];

    int Ai = (int)aabb_dim[b*3+0], Aj = (int)aabb_dim[b*3+1], Ak = (int)aabb_dim[b*3+2];
    int body_vol = Ai * Aj * Ak;
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

    for (int local = blockIdx.x * blockDim.x + threadIdx.x; local < body_vol; local += gridDim.x * blockDim.x) {
        int di = local / stride_x;
        int rem = local - di * stride_x;
        int dj = rem / stride_y;
        int dk = rem - dj * stride_y;
        int i = i0 + di, j = j0 + dj, k = k0 + dk;

        scalar_t xc = gx[i], yc = gy[j], zc = gz[k];
        scalar_t dxw = xc - bp_x, dyw = yc - bp_y, bzw = zc - bp_z;
        scalar_t bxq = r00*dxw + r01*dyw + r02*bzw;
        scalar_t byq = r10*dxw + r11*dyw + r12*bzw;
        scalar_t bzq = r20*dxw + r21*dyw + r22*bzw;

        int64_t idx = off + local;
        priv_sdf_cc[idx] = sdf_sample_dispatch<scalar_t>(interp,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq,byq,bzq);
        priv_sdf_u[idx]  = sdf_sample_dispatch<scalar_t>(interp,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq+du_x,byq+du_y,bzq+du_z);
        priv_body_u[idx] = lv_x + av_y*(zc - cm_z) - av_z*(yc - cm_y);
        priv_sdf_v[idx]  = sdf_sample_dispatch<scalar_t>(interp,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq+dv_x,byq+dv_y,bzq+dv_z);
        priv_body_v[idx] = lv_y + av_z*(xc - cm_x) - av_x*(zc - cm_z);
        priv_sdf_w[idx]  = sdf_sample_dispatch<scalar_t>(interp,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq+dw_x,byq+dw_y,bzq+dw_z);
        priv_body_w[idx] = lv_z + av_x*(yc - cm_y) - av_y*(xc - cm_x);
    }
}


// =====================================================================
//  3-D resolve kernel — one thread per cell in union dirty AABB
// =====================================================================
template <typename scalar_t>
__global__ void regime_b_resolve_kernel_3d(
    const int64_t* __restrict__ aabb_lo, const int64_t* __restrict__ aabb_dim,
    int Ngy, int Ngz, int B,
    int dirty_i0, int dirty_j0, int dirty_k0, int dirty_Ai, int dirty_Aj, int dirty_Ak,
    const int64_t* __restrict__ priv_offsets,
    const scalar_t* __restrict__ priv_sdf_cc, const scalar_t* __restrict__ priv_sdf_u,
    const scalar_t* __restrict__ priv_sdf_v, const scalar_t* __restrict__ priv_sdf_w,
    const scalar_t* __restrict__ priv_body_u, const scalar_t* __restrict__ priv_body_v, const scalar_t* __restrict__ priv_body_w,
    scalar_t* __restrict__ sdf_cc, scalar_t* __restrict__ sdf_u, scalar_t* __restrict__ sdf_v, scalar_t* __restrict__ sdf_w,
    scalar_t* __restrict__ body_u, scalar_t* __restrict__ body_v, scalar_t* __restrict__ body_w,
    int32_t* __restrict__ owner_cc,
    const scalar_t blend_eps)
{
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    int rem = local / dirty_Ak;
    int dk = local - rem * dirty_Ak;
    int di = rem / dirty_Aj;
    int dj = rem - di * dirty_Aj;
    if (di >= dirty_Ai) return;
    int i = dirty_i0 + di, j = dirty_j0 + dj, k = dirty_k0 + dk;
    int64_t g_idx = ((int64_t)i * Ngy + j) * Ngz + k;

    // Per-stagger winner selection — see the 2-D resolve kernel note.
    scalar_t best_cc = (scalar_t)1e4, best_u = (scalar_t)1e4,
             best_v = (scalar_t)1e4,  best_w = (scalar_t)1e4;
    int64_t win_cc = -1, win_u = -1, win_v = -1, win_w = -1;
    int win_cc_body = -1;   // the cell-centre argmin, published as owner_cc

    // Softmin velocity blend — see the 2-D resolve kernel note.
    const bool blend = blend_eps > scalar_t(0);
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
            const scalar_t wU = scalar_t(1) / (scalar_t(1) + exp(s_u / blend_eps));
            const scalar_t wV = scalar_t(1) / (scalar_t(1) + exp(s_v / blend_eps));
            const scalar_t wW = scalar_t(1) / (scalar_t(1) + exp(s_w / blend_eps));
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


// =====================================================================
//  Launcher functions
// =====================================================================

void streaming_sdf_stag_2d_resolve_cuda(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes, const at::Tensor& body_meta, const at::Tensor& kin,
    const at::Tensor& aabb_lo, const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy,
    double h_grid, int64_t max_vol_per_body,
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
    int n_active = (int)active_idx.numel();
    if (B <= 0 || dirty_Ai <= 0 || dirty_Aj <= 0) return;
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    int Ngy = (int)gy.numel();
    int64_t dirty_vol = dirty_Ai * dirty_Aj;
    // A size<=1 tensor is the caller's "owner field not wanted" placeholder.
    int32_t* owner_ptr = (owner_cc.numel() > 1) ? owner_cc.data_ptr<int32_t>()
                                                : nullptr;

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "sdf_stag_2d_resolve", [&] {
        // ----- min-kernel: fanned per-body (active bodies only) -----
        // gridDim.y == 0 is an invalid launch, and with every body static
        // there is nothing to restream — the resolve stage below still runs
        // and rebuilds the outputs from the retained per-body slabs.
        if (n_active > 0) {
            int bs = 256;
            // Grid sizing from the caller-supplied max ACTIVE body volume.
            // (Do NOT host-deref aabb_dim here: it is a CUDA tensor, so
            // reading its data_ptr on the host is an illegal device-memory
            // access — the former crash.  The caller computes the max from its
            // own host-side copy of the volumes.)
            int64_t max_vol = max_vol_per_body;
            int blocks_per_body = (int)((max_vol + bs - 1) / bs);
            if (blocks_per_body < 1) blocks_per_body = 1;
            regime_b_min_kernel_2d<scalar_t>
                <<<dim3(blocks_per_body, n_active, 1), dim3(bs, 1, 1), 0, s>>>(
                    F_flat.data_ptr<scalar_t>(), F_offsets.data_ptr<int64_t>(),
                    body_shapes.data_ptr<int64_t>(), body_meta.data_ptr<scalar_t>(),
                    kin.data_ptr<scalar_t>(),
                    aabb_lo.data_ptr<int64_t>(), aabb_dim.data_ptr<int64_t>(),
                    gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(),
                    Ngy, (scalar_t)(0.5 * h_grid), n_active, (int)interp,
                    priv_offsets.data_ptr<int64_t>(),
                    active_idx.data_ptr<int64_t>(),
                    priv_sdf_cc.data_ptr<scalar_t>(), priv_sdf_u.data_ptr<scalar_t>(), priv_sdf_v.data_ptr<scalar_t>(),
                    priv_body_u.data_ptr<scalar_t>(), priv_body_v.data_ptr<scalar_t>());
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                AT_ERROR("regime_b_min_kernel_2d launch failed: ", cudaGetErrorString(err));
            }
        }

        // ----- resolve kernel: over union dirty AABB -----
        {
            int bs = (dirty_vol <= 256) ? 32 : (dirty_vol <= 65536) ? 128 : 256;
            regime_b_resolve_kernel_2d<scalar_t>
                <<<dim3((dirty_vol + bs - 1) / bs, 1, 1), dim3(bs, 1, 1), 0, s>>>(
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
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                AT_ERROR("regime_b_resolve_kernel_2d launch failed: ", cudaGetErrorString(err));
            }
        }
    });
}

void streaming_sdf_stag_3d_resolve_cuda(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes, const at::Tensor& body_meta, const at::Tensor& kin,
    const at::Tensor& aabb_lo, const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy, const at::Tensor& gz,
    double h_grid, int64_t max_vol_per_body,
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
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    int Ngy = (int)gy.numel(), Ngz = (int)gz.numel();
    int64_t dirty_vol = dirty_Ai * dirty_Aj * dirty_Ak;
    // A size<=1 tensor is the caller's "owner field not wanted" placeholder.
    int32_t* owner_ptr = (owner_cc.numel() > 1) ? owner_cc.data_ptr<int32_t>()
                                                : nullptr;

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "sdf_stag_3d_resolve", [&] {
        // ----- min-kernel: fanned per-body (active bodies only) -----
        // gridDim.y == 0 is an invalid launch, and with every body static
        // there is nothing to restream — the resolve stage below still runs
        // and rebuilds the outputs from the retained per-body slabs.
        if (n_active > 0) {
            int bs = 256;
            // Grid sizing from the caller-supplied max body volume (see the
            // 2-D launcher note — host-deref of the CUDA aabb_dim was the crash).
            int64_t max_vol = max_vol_per_body;
            int blocks_per_body = (int)((max_vol + bs - 1) / bs);
            if (blocks_per_body < 1) blocks_per_body = 1;
            regime_b_min_kernel_3d<scalar_t>
                <<<dim3(blocks_per_body, n_active, 1), dim3(bs, 1, 1), 0, s>>>(
                    F_flat.data_ptr<scalar_t>(), F_offsets.data_ptr<int64_t>(),
                    body_shapes.data_ptr<int64_t>(), body_meta.data_ptr<scalar_t>(),
                    kin.data_ptr<scalar_t>(),
                    aabb_lo.data_ptr<int64_t>(), aabb_dim.data_ptr<int64_t>(),
                    gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(),
                    Ngy, Ngz, (scalar_t)(0.5 * h_grid), n_active, (int)interp,
                    priv_offsets.data_ptr<int64_t>(),
                    active_idx.data_ptr<int64_t>(),
                    priv_sdf_cc.data_ptr<scalar_t>(), priv_sdf_u.data_ptr<scalar_t>(),
                    priv_sdf_v.data_ptr<scalar_t>(), priv_sdf_w.data_ptr<scalar_t>(),
                    priv_body_u.data_ptr<scalar_t>(), priv_body_v.data_ptr<scalar_t>(), priv_body_w.data_ptr<scalar_t>());
        }
        AT_CUDA_CHECK(cudaGetLastError());

        // ----- resolve kernel: over union dirty AABB -----
        {
            int bs = (dirty_vol <= 256) ? 32 : (dirty_vol <= 65536) ? 128 : 256;
            regime_b_resolve_kernel_3d<scalar_t>
                <<<dim3((dirty_vol + bs - 1) / bs, 1, 1), dim3(bs, 1, 1), 0, s>>>(
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
        }
        AT_CUDA_CHECK(cudaGetLastError());
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("streaming_sdf_stag_2d_resolve", &streaming_sdf_stag_2d_resolve_cuda);
    m.impl("streaming_sdf_stag_3d_resolve", &streaming_sdf_stag_3d_resolve_cuda);
}
}  // namespace lilytorch_kernels
