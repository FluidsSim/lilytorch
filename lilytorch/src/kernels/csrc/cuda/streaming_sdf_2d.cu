// =====================================================================
//  streaming_sdf_2d.cu
//
//  CUDA implementations of ``streaming_sdf_min_2d`` and
//  ``streaming_sdf_min_2d_multi``.  Mirrors ``streaming_sdf.cu``
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
//  streaming_sdf_min_2d (single body) kernel
// =====================================================================
template <typename scalar_t>
__global__ void streaming_sdf_min_2d_kernel(
    const scalar_t* __restrict__ F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    const scalar_t r00, const scalar_t r01,
    const scalar_t r10, const scalar_t r11,
    const scalar_t bp_x, const scalar_t bp_y,
    const scalar_t cm_x, const scalar_t cm_y,
    const scalar_t lv_x, const scalar_t lv_y,
    const scalar_t omega,
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const int Ngy,
    const scalar_t half_h,
    const int i0, const int j0,
    const int Ai, const int Aj,
    scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ sdf_u,
    scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ bU,
    scalar_t* __restrict__ bV,
    scalar_t* __restrict__ sparse_cc,
    const int interp_method)
{
    const int total = Ai * Aj;
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int di = tid / Aj;
    const int dj = tid - di * Aj;
    const int i  = i0 + di;
    const int j  = j0 + dj;
    const int g_idx = i * Ngy + j;

    const scalar_t xc = gx[i];
    const scalar_t yc = gy[j];

    const scalar_t dx_w = xc - bp_x, dy_w = yc - bp_y;
    const scalar_t bxq = r00 * dx_w + r01 * dy_w;
    const scalar_t byq = r10 * dx_w + r11 * dy_w;

    const scalar_t neg_half_h = -half_h;
    const scalar_t du_x = neg_half_h * r00, du_y = neg_half_h * r10;
    const scalar_t dv_x = neg_half_h * r01, dv_y = neg_half_h * r11;

    {
        const scalar_t s = sdf_sample_dispatch_2d(
            interp_method, F, Mx, My, bx0, by0, inv_dx, inv_dy, bxq, byq);
        sparse_cc[tid] = s;
        if (s < sdf_cc[g_idx]) sdf_cc[g_idx] = s;
    }
    {
        const scalar_t s = sdf_sample_dispatch_2d(
            interp_method, F, Mx, My, bx0, by0, inv_dx, inv_dy,
            bxq + du_x, byq + du_y);
        if (s < sdf_u[g_idx]) {
            sdf_u[g_idx] = s;
            bU[g_idx] = lv_x - omega * (yc - cm_y);
        }
    }
    {
        const scalar_t s = sdf_sample_dispatch_2d(
            interp_method, F, Mx, My, bx0, by0, inv_dx, inv_dy,
            bxq + dv_x, byq + dv_y);
        if (s < sdf_v[g_idx]) {
            sdf_v[g_idx] = s;
            bV[g_idx] = lv_y + omega * (xc - cm_x);
        }
    }
}

// =====================================================================
//  CUDA dispatch (single body)
// =====================================================================
void streaming_sdf_min_2d_cuda(
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

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int Ai = (int)(i1 - i0);
    const int Aj = (int)(j1 - j0);
    const int N  = Ai * Aj;
    if (N <= 0) return;

    const int Mx = (int)bx.numel();
    const int My = (int)by.numel();
    const int Ngy = (int)gy.numel();

    const int blockSize = (N <= 128) ? 32 : (N <= 4096) ? 128 : 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(F.scalar_type(), "streaming_sdf_min_2d_cuda", [&] {
        const scalar_t* F_ptr  = F.contiguous().data_ptr<scalar_t>();
        const scalar_t* gx_ptr = gx.contiguous().data_ptr<scalar_t>();
        const scalar_t* gy_ptr = gy.contiguous().data_ptr<scalar_t>();

        scalar_t* sdf_cc_ptr = sdf_cc.data_ptr<scalar_t>();
        scalar_t* sdf_u_ptr  = sdf_u.data_ptr<scalar_t>();
        scalar_t* sdf_v_ptr  = sdf_v.data_ptr<scalar_t>();
        scalar_t* bU_ptr     = body_u.data_ptr<scalar_t>();
        scalar_t* bV_ptr     = body_v.data_ptr<scalar_t>();
        scalar_t* sp_ptr     = sparse_cc.data_ptr<scalar_t>();

        streaming_sdf_min_2d_kernel<scalar_t>
            <<<numBlocks, blockSize, 0, stream>>>(
                F_ptr,
                Mx, My,
                (scalar_t)bx0, (scalar_t)by0,
                (scalar_t)inv_dx, (scalar_t)inv_dy,
                (scalar_t)R_T[0], (scalar_t)R_T[1],
                (scalar_t)R_T[2], (scalar_t)R_T[3],
                (scalar_t)body_pos[0], (scalar_t)body_pos[1],
                (scalar_t)com_pos[0],  (scalar_t)com_pos[1],
                (scalar_t)lin_vel[0],  (scalar_t)lin_vel[1],
                (scalar_t)omega,
                gx_ptr, gy_ptr,
                Ngy,
                (scalar_t)(0.5 * h_grid),
                (int)i0, (int)j0,
                Ai, Aj,
                sdf_cc_ptr, sdf_u_ptr, sdf_v_ptr,
                bU_ptr, bV_ptr,
                sp_ptr,
                (int)interp_method);
    });
}

// =====================================================================
//  streaming_sdf_min_2d_multi  (Phase C: batched-call over bodies)
// =====================================================================

template <typename scalar_t>
__global__ void streaming_sdf_min_2d_multi_kernel(
    const int b,
    const scalar_t* __restrict__ F_flat,
    const int64_t*  __restrict__ F_offsets,
    const int64_t*  __restrict__ body_shapes,   // [B,2] (Mx,My)
    const scalar_t* __restrict__ body_meta,     // [B,7]
    const scalar_t* __restrict__ kin,           // [B,11]
    const int64_t*  __restrict__ aabb_lo,       // [B,2] (i0,j0)
    const int64_t*  __restrict__ aabb_dim,      // [B,2] (Ai,Aj)
    const int64_t*  __restrict__ cell_offsets,  // [B+1] prefix sum
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const int Ngy,
    const scalar_t half_h,
    scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ sdf_u,
    scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ bU,
    scalar_t* __restrict__ bV,
    scalar_t* __restrict__ sparse_cc_flat,
    const int interp_method)
{
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

    const scalar_t* F  = F_flat  + F_offsets[b];
    const int Mx = (int)body_shapes[b*2 + 0];
    const int My = (int)body_shapes[b*2 + 1];

    const scalar_t* M = body_meta + b*7;
    const scalar_t bx0 = M[0], by0 = M[1];
    const scalar_t idx_ = M[4], idy_ = M[5];

    const scalar_t* K = kin + b*11;
    const scalar_t r00 = K[0], r01 = K[1];
    const scalar_t r10 = K[2], r11 = K[3];
    const scalar_t bp_x = K[4], bp_y = K[5];
    const scalar_t cm_x = K[6], cm_y = K[7];
    const scalar_t lv_x = K[8], lv_y = K[9];
    const scalar_t om   = K[10];

    const scalar_t xc = gx[i];
    const scalar_t yc = gy[j];

    const int64_t sparse_idx = cell_offsets[b] + (int64_t)local;

    const scalar_t dx_w = xc - bp_x, dy_w = yc - bp_y;
    const scalar_t bxq = r00 * dx_w + r01 * dy_w;
    const scalar_t byq = r10 * dx_w + r11 * dy_w;

    const scalar_t neg_half_h = -half_h;
    const scalar_t du_x = neg_half_h * r00, du_y = neg_half_h * r10;
    const scalar_t dv_x = neg_half_h * r01, dv_y = neg_half_h * r11;

    {
        const scalar_t s = sdf_sample_dispatch_2d(
            interp_method, F, Mx, My, bx0, by0, idx_, idy_, bxq, byq);
        sparse_cc_flat[sparse_idx] = s;
        if (s < sdf_cc[g_idx]) sdf_cc[g_idx] = s;
    }
    {
        const scalar_t s = sdf_sample_dispatch_2d(
            interp_method, F, Mx, My, bx0, by0, idx_, idy_,
            bxq + du_x, byq + du_y);
        if (s < sdf_u[g_idx]) {
            sdf_u[g_idx] = s;
            bU[g_idx] = lv_x - om * (yc - cm_y);
        }
    }
    {
        const scalar_t s = sdf_sample_dispatch_2d(
            interp_method, F, Mx, My, bx0, by0, idx_, idy_,
            bxq + dv_x, byq + dv_y);
        if (s < sdf_v[g_idx]) {
            sdf_v[g_idx] = s;
            bV[g_idx] = lv_y + om * (xc - cm_x);
        }
    }
}

void streaming_sdf_min_2d_multi_cuda(
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
    const int64_t max_vol_per_body,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v,
    at::Tensor body_u, at::Tensor body_v,
    at::Tensor sparse_cc_flat,
    const int64_t interp_method)
{
    (void)bx_flat; (void)bx_offsets; (void)by_flat; (void)by_offsets;

    const int B = (int)aabb_dim.size(0);
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngy = (int)gy.numel();

    const int blockSize = (max_vol_per_body <= 128) ? 32
                        : (max_vol_per_body <= 4096) ? 128 : 256;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(F_flat.scalar_type(), "streaming_sdf_min_2d_multi_cuda", [&] {
        for (int b = 0; b < B; ++b) {
            const int blocksPerBody = (int)((max_vol_per_body + blockSize - 1) / blockSize);
            streaming_sdf_min_2d_multi_kernel<scalar_t>
                <<<dim3(blocksPerBody, 1, 1), dim3(blockSize, 1, 1), 0, stream>>>(
                    b,
                    F_flat.data_ptr<scalar_t>(),
                    F_offsets.data_ptr<int64_t>(),
                    body_shapes.data_ptr<int64_t>(),
                    body_meta.data_ptr<scalar_t>(),
                    kin.data_ptr<scalar_t>(),
                    aabb_lo.data_ptr<int64_t>(),
                    aabb_dim.data_ptr<int64_t>(),
                    cell_offsets.data_ptr<int64_t>(),
                    gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(),
                    Ngy,
                    (scalar_t)(0.5 * h_grid),
                    sdf_cc.data_ptr<scalar_t>(),
                    sdf_u.data_ptr<scalar_t>(),
                    sdf_v.data_ptr<scalar_t>(),
                    body_u.data_ptr<scalar_t>(),
                    body_v.data_ptr<scalar_t>(),
                    sparse_cc_flat.data_ptr<scalar_t>(),
                    (int)interp_method);
        }
    });
}

// =====================================================================
//  bdim_forces_2d_multi (CUDA)
//
//  2-D analogue of ``bdim_forces_3d_multi``.  Per-body grid: walks the
//  AABB cells, reads cached cc-SDF from ``sparse_cc_flat``, evaluates
//  smoothed visc/pres deltas, accumulates 8 float64 channels:
//      [fv_x, fv_y, t_v, fp_x, fp_y, t_p, 0, 0]
//  via shared-memory reduction + atomicAdd into out[b, 0..7].
// =====================================================================

template <typename scalar_t>
__global__ void bdim_forces_2d_multi_kernel(
    const int b,
    const scalar_t* __restrict__ sparse_cc_flat,
    const int64_t*  __restrict__ cell_offsets,
    const scalar_t* __restrict__ kin,
    const int64_t*  __restrict__ aabb_lo,
    const int64_t*  __restrict__ aabb_dim,
    const scalar_t* __restrict__ gx,
    const scalar_t* __restrict__ gy,
    const int u_i0, const int u_j0,
    const int Sj,
    const scalar_t* __restrict__ xs,
    const scalar_t* __restrict__ ys,
    const scalar_t* __restrict__ px,
    const scalar_t* __restrict__ py,
    const scalar_t eps_body,
    const scalar_t eps_solver,
    const scalar_t h2,
    const int delta_order,
    double* __restrict__ out)  // (B, 8)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;

    const int Ai = (int)aabb_dim[b*2 + 0];
    const int Aj = (int)aabb_dim[b*2 + 1];
    const int vol = Ai * Aj;

    double acc[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) acc[c] = 0.0;

    if (local < vol) {
        const int di = local / Aj;
        const int dj = local - di * Aj;
        const int i0 = (int)aabb_lo[b*2 + 0];
        const int j0 = (int)aabb_lo[b*2 + 1];
        const int i = i0 + di, j = j0 + dj;

        const int64_t sparse_base = cell_offsets[b];
        const scalar_t sdf = sparse_cc_flat[sparse_base + (int64_t)local];

        const scalar_t band_lo =
            (eps_solver - eps_body) < (-eps_body)
                ? (eps_solver - eps_body) : (-eps_body);
        const scalar_t band_hi =
            (eps_solver + eps_body) > ( eps_body)
                ? (eps_solver + eps_body) : ( eps_body);
        if (sdf > band_lo && sdf < band_hi) {
            // 2-D kin row (11 floats); cm_xy at offsets 6..7.
            const scalar_t* K = kin + b*11;
            const scalar_t cm_x = K[6], cm_y = K[7];

            const scalar_t xc = gx[i];
            const scalar_t yc = gy[j];

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
            if (delta_order == 2 && (delta_visc > (scalar_t)0 || delta_pres > (scalar_t)0)) {
                const scalar_t h_grid = gx[1] - gx[0];
                const scalar_t inv_h  = (scalar_t)1.0 / h_grid;
                const int64_t  loc64  = (int64_t)local;

                scalar_t sdf_xp = (di < Ai-1) ? sparse_cc_flat[sparse_base + loc64 + (int64_t)Aj] : sdf;
                scalar_t sdf_xm = (di > 0)    ? sparse_cc_flat[sparse_base + loc64 - (int64_t)Aj] : sdf;
                scalar_t cx     = (di > 0 && di < Ai-1) ? (scalar_t)0.5 : (scalar_t)1.0;
                scalar_t dsdx   = cx * (sdf_xp - sdf_xm) * inv_h;

                scalar_t sdf_yp = (dj < Aj-1) ? sparse_cc_flat[sparse_base + loc64 + 1] : sdf;
                scalar_t sdf_ym = (dj > 0)    ? sparse_cc_flat[sparse_base + loc64 - 1] : sdf;
                scalar_t cy     = (dj > 0 && dj < Aj-1) ? (scalar_t)0.5 : (scalar_t)1.0;
                scalar_t dsdy   = cy * (sdf_yp - sdf_ym) * inv_h;

                scalar_t grad_mag = sqrt(dsdx*dsdx + dsdy*dsdy);
                const scalar_t min_grad = (scalar_t)1e-3;
                if (grad_mag < min_grad) grad_mag = min_grad;
                const scalar_t inv_grad = (scalar_t)1.0 / grad_mag;
                delta_visc *= inv_grad;
                delta_pres *= inv_grad;
            }

            const int sub_i = i - u_i0;
            const int sub_j = j - u_j0;
            const int s_idx = sub_i * Sj + sub_j;
            const scalar_t xs_v = xs[s_idx], ys_v = ys[s_idx];
            const scalar_t px_v = px[s_idx], py_v = py[s_idx];

            const scalar_t arm_x = xc - cm_x;
            const scalar_t arm_y = yc - cm_y;

            const double fv_x_d = (double)(xs_v * delta_visc);
            const double fv_y_d = (double)(ys_v * delta_visc);
            const double fp_x_d = (double)(px_v * delta_pres);
            const double fp_y_d = (double)(py_v * delta_pres);

            acc[0] = fv_x_d;
            acc[1] = fv_y_d;
            acc[2] = (double)arm_x * fv_y_d - (double)arm_y * fv_x_d;
            acc[3] = fp_x_d;
            acc[4] = fp_y_d;
            acc[5] = (double)arm_x * fp_y_d - (double)arm_y * fp_x_d;
            // acc[6], acc[7] reserved (always 0)
        }
    }

    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const double h2_d = (double)h2;

#pragma unroll
    for (int c = 0; c < 8; ++c) {
        sdata[tid] = acc[c];
        __syncthreads();
        for (int s = bdim >> 1; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            atomicAdd(&out[b*8 + c], sdata[0] * h2_d);
        }
        __syncthreads();
    }
}

void bdim_forces_2d_multi_cuda(
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
    const double eps_body,
    const double eps_solver,
    const double h2,
    const int64_t max_vol_per_body,
    const int64_t delta_order,
    at::Tensor out)  // (B, 8) float64
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blockSize = 256;
    const size_t shmem  = (size_t)blockSize * sizeof(double);

    AT_DISPATCH_FLOATING_TYPES(sparse_cc_flat.scalar_type(), "bdim_forces_2d_multi_cuda", [&] {
        for (int b = 0; b < B; ++b) {
            const int blocksPerBody = (int)((max_vol_per_body + blockSize - 1) / blockSize);
            bdim_forces_2d_multi_kernel<scalar_t>
                <<<dim3(blocksPerBody, 1, 1), dim3(blockSize, 1, 1), shmem, stream>>>(
                    b,
                    sparse_cc_flat.data_ptr<scalar_t>(),
                    cell_offsets.data_ptr<int64_t>(),
                    kin.data_ptr<scalar_t>(),
                    aabb_lo.data_ptr<int64_t>(),
                    aabb_dim.data_ptr<int64_t>(),
                    gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(),
                    (int)u_i0, (int)u_j0,
                    (int)Sj,
                    xs.data_ptr<scalar_t>(), ys.data_ptr<scalar_t>(),
                    px.data_ptr<scalar_t>(), py.data_ptr<scalar_t>(),
                    (scalar_t)eps_body, (scalar_t)eps_solver, (scalar_t)h2,
                    (int)delta_order,
                    out.data_ptr<double>());
        }
    });
}

// =====================================================================
//  streaming_sdf_forces_fused_2d_multi  (Phase C+D fused, lagged forces)
// =====================================================================

template <typename scalar_t>
__global__ void streaming_sdf_forces_fused_2d_multi_kernel(
    const int b,
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
    scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ sdf_u,
    scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ bU,
    scalar_t* __restrict__ bV,
    const int interp_method,
    const scalar_t* __restrict__ rho_bodies,
    scalar_t* __restrict__ winning_rho_cc,
    const scalar_t* __restrict__ u_prev,
    const scalar_t* __restrict__ v_prev,
    const scalar_t* __restrict__ p_prev,
    const scalar_t* __restrict__ nx_cc,
    const scalar_t* __restrict__ ny_cc,
    const scalar_t* __restrict__ nu_rho_field,
    const int64_t   nu_rho_field_size,
    const scalar_t inv_h,
    const scalar_t eps_body,
    const scalar_t eps_solver,
    const scalar_t h2,
    const int delta_order,
    double* __restrict__ out)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;

    const int Ai = (int)aabb_dim[b*2 + 0];
    const int Aj = (int)aabb_dim[b*2 + 1];
    const int vol = Ai * Aj;

    double acc[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) acc[c] = 0.0;

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
        const scalar_t lv_x = K[8], lv_y = K[9];
        const scalar_t om   = K[10];

        const scalar_t xc = gx[i];
        const scalar_t yc = gy[j];

        const scalar_t dx_w = xc - bp_x, dy_w = yc - bp_y;
        const scalar_t bxq = r00 * dx_w + r01 * dy_w;
        const scalar_t byq = r10 * dx_w + r11 * dy_w;

        const scalar_t neg_hh = -half_h;
        const scalar_t du_x = neg_hh * r00, du_y = neg_hh * r10;
        const scalar_t dv_x = neg_hh * r01, dv_y = neg_hh * r11;

        const scalar_t s_cc = sdf_sample_dispatch_2d(
            interp_method, F, Mx, My, bx0, by0, idx_, idy_, bxq, byq);

        if (s_cc < sdf_cc[g_idx]) {
            sdf_cc[g_idx] = s_cc;
            winning_rho_cc[g_idx] = rho_bodies[b];
        }

        {
            const scalar_t s = sdf_sample_dispatch_2d(
                interp_method, F, Mx, My, bx0, by0, idx_, idy_,
                bxq + du_x, byq + du_y);
            if (s < sdf_u[g_idx]) {
                sdf_u[g_idx] = s;
                bU[g_idx] = lv_x - om * (yc - cm_y);
            }
        }
        {
            const scalar_t s = sdf_sample_dispatch_2d(
                interp_method, F, Mx, My, bx0, by0, idx_, idy_,
                bxq + dv_x, byq + dv_y);
            if (s < sdf_v[g_idx]) {
                sdf_v[g_idx] = s;
                bV[g_idx] = lv_y + om * (xc - cm_x);
            }
        }

        const scalar_t band_lo = (eps_solver - eps_body) < (-eps_body)
            ? (eps_solver - eps_body) : (-eps_body);
        const scalar_t band_hi = (eps_solver + eps_body) > (eps_body)
            ? (eps_solver + eps_body) : (eps_body);

        if (s_cc > band_lo && s_cc < band_hi) {
            const scalar_t nu_rho_val = (nu_rho_field_size == 1)
                ? nu_rho_field[0] : nu_rho_field[g_idx];

            const scalar_t nx = nx_cc[g_idx];
            const scalar_t ny = ny_cc[g_idx];

            const int im1 = (i > 0)     ? i-1 : 0;
            const int ip1 = (i+1 < Ngx) ? i+1 : i;
            const int jm1 = (j > 0)     ? j-1 : 0;
            const int jp1 = (j+1 < Ngy) ? j+1 : j;

            const scalar_t dudx = (u_prev[ip1 * Ngy + j] - u_prev[i * Ngy + j]) * inv_h;
            const scalar_t dvdy = (v_prev[i * Ngy + jp1] - v_prev[i * Ngy + j]) * inv_h;

            const scalar_t u_cc_jp1 = (scalar_t)0.5 * (u_prev[i * Ngy + jp1] + u_prev[ip1 * Ngy + jp1]);
            const scalar_t u_cc_jm1 = (scalar_t)0.5 * (u_prev[i * Ngy + jm1] + u_prev[ip1 * Ngy + jm1]);
            const scalar_t dudy = (u_cc_jp1 - u_cc_jm1) * (scalar_t)0.5 * inv_h;

            const scalar_t v_cc_ip1 = (scalar_t)0.5 * (v_prev[ip1 * Ngy + j] + v_prev[ip1 * Ngy + jp1]);
            const scalar_t v_cc_im1 = (scalar_t)0.5 * (v_prev[im1 * Ngy + j] + v_prev[im1 * Ngy + jp1]);
            const scalar_t dvdx = (v_cc_ip1 - v_cc_im1) * (scalar_t)0.5 * inv_h;

            const scalar_t xs = nu_rho_val * (2*dudx*nx + (dudy+dvdx)*ny);
            const scalar_t ys = nu_rho_val * ((dvdx+dudy)*nx + 2*dvdy*ny);

            const scalar_t p_c = p_prev[g_idx];
            const scalar_t pxv = -p_c * nx;
            const scalar_t pyv = -p_c * ny;

            const scalar_t pi_v     = (scalar_t)3.141592653589793;
            const scalar_t inv_2eps = (scalar_t)0.5 / eps_body;
            const scalar_t pi_ov_eb = pi_v / eps_body;

            scalar_t delta_visc = 0, delta_pres = 0;
            const scalar_t d_visc = s_cc - eps_solver;
            if (d_visc > -eps_body && d_visc < eps_body)
                delta_visc = ((scalar_t)1 + cos(pi_ov_eb * d_visc)) * inv_2eps;
            if (s_cc > -eps_body && s_cc < eps_body)
                delta_pres = ((scalar_t)1 + cos(pi_ov_eb * s_cc)) * inv_2eps;

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

    extern __shared__ double sdata[];
    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;
    const double h2_d = (double)h2;

#pragma unroll
    for (int c = 0; c < 8; ++c) {
        sdata[tid] = acc[c];
        __syncthreads();
        for (int s = bdim >> 1; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) atomicAdd(&out[b*8 + c], sdata[0] * h2_d);
        __syncthreads();
    }
}

void streaming_sdf_forces_fused_2d_multi_cuda(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes, const at::Tensor& body_meta, const at::Tensor& kin,
    const at::Tensor& aabb_lo, const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy,
    const double h_grid, const int64_t max_vol_per_body,
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
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngx = (int)gx.numel();
    const int Ngy = (int)gy.numel();
    const int blockSize = 256;
    const size_t shmem = (size_t)blockSize * sizeof(double);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(F_flat.scalar_type(),
        "streaming_sdf_forces_fused_2d_multi_cuda", [&] {
        const scalar_t* rho_bodies_ptr = rho_bodies.data_ptr<scalar_t>();
        for (int b = 0; b < B; ++b) {
            const int nblocks = (int)((max_vol_per_body + blockSize - 1) / blockSize);
            streaming_sdf_forces_fused_2d_multi_kernel<scalar_t>
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
                    Ngx, Ngy,
                    (scalar_t)(0.5 * h_grid),
                    sdf_cc.data_ptr<scalar_t>(),
                    sdf_u.data_ptr<scalar_t>(),
                    sdf_v.data_ptr<scalar_t>(),
                    body_u.data_ptr<scalar_t>(),
                    body_v.data_ptr<scalar_t>(),
                    (int)interp_method,
                    rho_bodies_ptr,
                    winning_rho_cc.data_ptr<scalar_t>(),
                    u_prev.data_ptr<scalar_t>(),
                    v_prev.data_ptr<scalar_t>(),
                    p_prev.data_ptr<scalar_t>(),
                    nx_cc.data_ptr<scalar_t>(),
                    ny_cc.data_ptr<scalar_t>(),
                    nu_rho_field.data_ptr<scalar_t>(),
                    (int64_t)nu_rho_field.numel(),
                    (scalar_t)(1.0 / h_grid),
                    (scalar_t)eps_body,
                    (scalar_t)eps_solver,
                    (scalar_t)h2,
                    (int)delta_order,
                    out.data_ptr<double>());
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
    const int N_dir,
    const int op_offset)
{
    const int op = blockIdx.y + op_offset;
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

    // Ops are serialized (one kernel launch per op on the same CUDA stream)
    // because they can share ghost cells.  Two failure modes if launched in
    // parallel via blockIdx.y:
    //
    //   Neumann–Neumann corner race: the x-axis op writes u[0,:] = u[1,:] and
    //   the y-axis op writes u[:,0] = u[:,1] simultaneously, so u[0,0] has an
    //   undefined winner.  Under sequential application the second op reads the
    //   already-updated slice, so u[0,0] ends up holding the diagonal interior
    //   value u_orig[1,1] (axis-0 runs first: u[0,1] ← u_orig[1,1]; axis-1
    //   then reads that: u[0,0] ← u[0,1] = u_orig[1,1]).
    //
    //   Neumann–Dirichlet override race: a Neumann op and a subsequent Dirichlet
    //   op that targets the same face must not overlap — the Dirichlet value
    //   must win.
    //
    // Corner ghost cells (e.g. u[0,0]) are never read by interior 5-point
    // stencils, so their exact value is physically irrelevant.  Many solvers
    // leave corners untouched; this implementation writes a diagonal-interior
    // value as a side-effect of the sequential application order, which is kept
    // for exact reproducibility of the CPU serial-loop reference.
    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_2d_cuda", [&] {
        const int* neu_p = (N_neu > 0) ? neu_desc.data_ptr<int>() : nullptr;
        const int* dir_p = (N_dir > 0) ? dir_desc.data_ptr<int>() : nullptr;
        const scalar_t* dir_val_p =
            (N_dir > 0) ? dir_val.data_ptr<scalar_t>() : nullptr;

        scalar_t* u_p = u.data_ptr<scalar_t>();
        scalar_t* v_p = v.data_ptr<scalar_t>();
        const int64_t* sh_p = shapes.data_ptr<int64_t>();

        for (int op = 0; op < total; ++op) {
            apply_bcs_2d_kernel<scalar_t>
                <<<dim3(gridX, 1, 1), dim3(blockX, 1, 1), 0, stream>>>(
                    u_p, v_p, sh_p,
                    neu_p, N_neu,
                    dir_p, dir_val_p, N_dir,
                    op);
        }
    });
}

// =====================================================================
//  interpolate_2d: scattered-point bilinear / biquadratic sampling
//
//  One thread per query point.  Calls the same sdf_sample_dispatch_2d
//  device function used by streaming_sdf_min_2d (interp_method 0 =
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

    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "interpolate_2d_cuda", [&] {
        interpolate_2d_kernel<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
            F.contiguous().data_ptr<scalar_t>(),
            xq.contiguous().to(F.scalar_type()).data_ptr<scalar_t>(),
            yq.contiguous().to(F.scalar_type()).data_ptr<scalar_t>(),
            N,
            (int)Mx, (int)My,
            (scalar_t)bx0, (scalar_t)by0,
            (scalar_t)inv_dx, (scalar_t)inv_dy,
            (int)interp_method,
            G.data_ptr<scalar_t>());
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("streaming_sdf_min_2d",                  &streaming_sdf_min_2d_cuda);
    m.impl("streaming_sdf_min_2d_multi",            &streaming_sdf_min_2d_multi_cuda);
    m.impl("bdim_forces_2d_multi",                  &bdim_forces_2d_multi_cuda);
    m.impl("streaming_sdf_forces_fused_2d_multi",   &streaming_sdf_forces_fused_2d_multi_cuda);
    m.impl("apply_bcs_2d",                          &apply_bcs_2d_cuda);
    m.impl("interpolate_2d",                        &interpolate_2d_cuda);
}

}  // namespace lilytorch_kernels
