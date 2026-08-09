// =====================================================================
//  eulerian_forces.cu — Eulerian immersed-boundary force integration
//  kernels (post-CL2 refactor).  Contains 2-D and 3-D variants of
//  streaming_sdf_forces_post_{2,3}d.
//
//  The sole streaming-SDF body-update path is streaming.cu (per-body
//  private buffers + resolve).
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

#include "../common/bc_ops.h"
#include "../common/interp.h"

namespace lilytorch_kernels {


// ---- 2-D forces-post -------------------------------------------------------
// Smoothed Heaviside device helper, shared by the union ndelta readout's
// heaviside_grad_2d (below).
template <typename scalar_t>
__device__ __forceinline__ scalar_t heaviside_smooth_dev_2d(scalar_t phi, scalar_t inv_eps) {
    const scalar_t pi = (scalar_t)3.141592653589793;
    scalar_t x = phi * inv_eps;
    x = x < (scalar_t)-1 ? (scalar_t)-1 : (x > (scalar_t)1 ? (scalar_t)1 : x);
    return (scalar_t)0.5 * ((scalar_t)1 + x + sin(pi * x) / pi);
}


// ---- 2-D unified union-measure + partition readout (the fixed ndelta) -------
// Mirrors forces_post_union_blend_3d_kernel; see it for the rationale.
template <typename scalar_t>
__device__ __forceinline__ scalar_t bilinear_sample_grad_uniform_2d(
    const scalar_t* __restrict__ F, const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    scalar_t xq, scalar_t yq, scalar_t& gbx, scalar_t& gby)
{
    scalar_t tx = (xq - bx0) * inv_dx, ty = (yq - by0) * inv_dy;
    tx = max((scalar_t)0, min(tx, (scalar_t)(Mx - 1)));
    ty = max((scalar_t)0, min(ty, (scalar_t)(My - 1)));
    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;
    const scalar_t fx = tx - (scalar_t)ix, fy = ty - (scalar_t)iy;
    const int s1 = My, base = ix*s1 + iy;
    const scalar_t c00 = F[base], c01 = F[base+1], c10 = F[base+s1], c11 = F[base+s1+1];
    const scalar_t omfx = (scalar_t)1-fx, omfy = (scalar_t)1-fy;
    const scalar_t val = omfx*(omfy*c00 + fy*c01) + fx*(omfy*c10 + fy*c11);
    const scalar_t dfx = omfy*(c10-c00) + fy*(c11-c01);
    const scalar_t dfy = omfx*(c01-c00) + fx*(c11-c10);
    gbx = dfx * inv_dx; gby = dfy * inv_dy;
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ void heaviside_grad_2d(
    const scalar_t* __restrict__ sdf_cc, const int Ngx, const int Ngy,
    const int i, const int j, const scalar_t off, const scalar_t inv_eps,
    const scalar_t inv_h, scalar_t& gHx, scalar_t& gHy)
{
#define S2_(ii,jj) (sdf_cc[(int64_t)(ii)*Ngy + (jj)])
#define H2_(ii,jj) heaviside_smooth_dev_2d<scalar_t>(S2_(ii,jj) - off, inv_eps)
    gHx = 0; gHy = 0;
    if (Ngx >= 3) {
        if (i == 0) gHx = ((scalar_t)(-3)*H2_(0,j) + (scalar_t)4*H2_(1,j) - H2_(2,j)) * (scalar_t)0.5 * inv_h;
        else if (i == Ngx-1) gHx = ((scalar_t)3*H2_(Ngx-1,j) - (scalar_t)4*H2_(Ngx-2,j) + H2_(Ngx-3,j)) * (scalar_t)0.5 * inv_h;
        else gHx = (H2_(i+1,j) - H2_(i-1,j)) * (scalar_t)0.5 * inv_h;
    } else if (Ngx == 2) gHx = (H2_(1,j) - H2_(0,j)) * inv_h;
    if (Ngy >= 3) {
        if (j == 0) gHy = ((scalar_t)(-3)*H2_(i,0) + (scalar_t)4*H2_(i,1) - H2_(i,2)) * (scalar_t)0.5 * inv_h;
        else if (j == Ngy-1) gHy = ((scalar_t)3*H2_(i,Ngy-1) - (scalar_t)4*H2_(i,Ngy-2) + H2_(i,Ngy-3)) * (scalar_t)0.5 * inv_h;
        else gHy = (H2_(i,j+1) - H2_(i,j-1)) * (scalar_t)0.5 * inv_h;
    } else if (Ngy == 2) gHy = (H2_(i,1) - H2_(i,0)) * inv_h;
#undef H2_
#undef S2_
}

template <typename scalar_t>
__device__ __forceinline__ void sdf_grad_2d(
    const scalar_t* __restrict__ sdf_cc, const int Ngx, const int Ngy,
    const int i, const int j, const scalar_t inv_h, scalar_t& gsx, scalar_t& gsy)
{
#define S2_(ii,jj) (sdf_cc[(int64_t)(ii)*Ngy + (jj)])
    gsx = 0; gsy = 0;
    if (Ngx >= 3) {
        if (i == 0) gsx = ((scalar_t)(-3)*S2_(0,j) + (scalar_t)4*S2_(1,j) - S2_(2,j)) * (scalar_t)0.5 * inv_h;
        else if (i == Ngx-1) gsx = ((scalar_t)3*S2_(Ngx-1,j) - (scalar_t)4*S2_(Ngx-2,j) + S2_(Ngx-3,j)) * (scalar_t)0.5 * inv_h;
        else gsx = (S2_(i+1,j) - S2_(i-1,j)) * (scalar_t)0.5 * inv_h;
    } else if (Ngx == 2) gsx = (S2_(1,j) - S2_(0,j)) * inv_h;
    if (Ngy >= 3) {
        if (j == 0) gsy = ((scalar_t)(-3)*S2_(i,0) + (scalar_t)4*S2_(i,1) - S2_(i,2)) * (scalar_t)0.5 * inv_h;
        else if (j == Ngy-1) gsy = ((scalar_t)3*S2_(i,Ngy-1) - (scalar_t)4*S2_(i,Ngy-2) + S2_(i,Ngy-3)) * (scalar_t)0.5 * inv_h;
        else gsy = (S2_(i,j+1) - S2_(i,j-1)) * (scalar_t)0.5 * inv_h;
    } else if (Ngy == 2) gsy = (S2_(i,1) - S2_(i,0)) * inv_h;
#undef S2_
}

template <typename scalar_t, int BLOCK_SIZE, bool BODY_NORMAL>
__global__ void forces_post_union_blend_2d_kernel(
    const scalar_t* __restrict__ F_flat, const int64_t* __restrict__ F_offsets,
    const int64_t* __restrict__ body_shapes, const scalar_t* __restrict__ body_meta,
    const scalar_t* __restrict__ kin,
    const int64_t* __restrict__ aabb_lo, const int64_t* __restrict__ aabb_dim,
    const scalar_t* __restrict__ gx, const scalar_t* __restrict__ gy,
    const int Ngx, const int Ngy,
    const scalar_t* __restrict__ sdf_cc,
    const int32_t* __restrict__ owner_cc, const int interp_method,
    const scalar_t* __restrict__ u_prev, const scalar_t* __restrict__ v_prev,
    const scalar_t* __restrict__ p_prev,
    const scalar_t* __restrict__ nu_rho_field, const int64_t nu_rho_field_size,
    const scalar_t inv_h, const scalar_t inv_eps,
    const scalar_t off_pres, const scalar_t off_visc, const scalar_t blend_eps,
    const scalar_t h2, const int B, double* __restrict__ out)
{
    const int b = blockIdx.y;
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const int Ai = (int)aabb_dim[b*2 + 0], Aj = (int)aabb_dim[b*2 + 1];
    const int vol = Ai * Aj;
    double acc[6];
#pragma unroll
    for (int c = 0; c < 6; ++c) acc[c] = 0.0;

    if (local < vol) {
        const int di = local / Aj, dj = local - di * Aj;
        const int i = (int)aabb_lo[b*2+0] + di, j = (int)aabb_lo[b*2+1] + dj;
        const int g_idx = i * Ngy + j;

        scalar_t gpx = 0, gpy = 0, gvx = 0, gvy = 0, meas_p = 0, meas_v = 0;
        bool has_p = false, has_v = false;
        // Ownership gate, FIRST — see the 3-D kernel for why the order matters.
        const bool owned = (owner_cc == nullptr) || (blend_eps > (scalar_t)0)
                           || (owner_cc[g_idx] == b);
        if (!owned) {
            // nothing to do for this (body, cell)
        } else if (!BODY_NORMAL) {
            heaviside_grad_2d<scalar_t>(sdf_cc, Ngx, Ngy, i, j, off_pres, inv_eps, inv_h, gpx, gpy);
            heaviside_grad_2d<scalar_t>(sdf_cc, Ngx, Ngy, i, j, off_visc, inv_eps, inv_h, gvx, gvy);
            has_p = (gpx != (scalar_t)0 || gpy != (scalar_t)0);
            has_v = (gvx != (scalar_t)0 || gvy != (scalar_t)0);
        } else {
            scalar_t gux, guy; sdf_grad_2d<scalar_t>(sdf_cc, Ngx, Ngy, i, j, inv_h, gux, guy);
            const scalar_t gmag = sqrt(gux*gux + guy*guy);
            const scalar_t phi_u = sdf_cc[g_idx];
            const scalar_t pi_v = (scalar_t)3.141592653589793;
            const scalar_t dp = (phi_u - off_pres) * inv_eps, dv = (phi_u - off_visc) * inv_eps;
            const scalar_t hie = (scalar_t)0.5 * inv_eps;
            const scalar_t delp = (dp > (scalar_t)-1 && dp < (scalar_t)1) ? ((scalar_t)1 + cos(pi_v*dp))*hie : (scalar_t)0;
            const scalar_t delv = (dv > (scalar_t)-1 && dv < (scalar_t)1) ? ((scalar_t)1 + cos(pi_v*dv))*hie : (scalar_t)0;
            meas_p = delp * gmag; meas_v = delv * gmag;
            has_p = (meas_p != (scalar_t)0); has_v = (meas_v != (scalar_t)0);
        }

        if (has_p || has_v) {
            const scalar_t xc = gx[i], yc = gy[j];
            const scalar_t* Fb = F_flat + F_offsets[b];
            const int Mxb = (int)body_shapes[b*2+0], Myb = (int)body_shapes[b*2+1];
            const scalar_t* Mb = body_meta + b*7;
            const scalar_t* Kb = kin + b*11;
            const scalar_t dxw = xc - Kb[4], dyw = yc - Kb[5];
            const scalar_t bxq = Kb[0]*dxw + Kb[1]*dyw, byq = Kb[2]*dxw + Kb[3]*dyw;
            const bool soft = blend_eps > (scalar_t)0;
            // See the 3-D kernel: with a hard partition the winner is read
            // from owner_cc, so this body's own phi is only needed for the
            // soft weight (BODY_NORMAL computes it regardless, for the normal).
            const bool need_self = soft || (owner_cc == nullptr);
            scalar_t s_bself = 0;
            if (!BODY_NORMAL) {
                if (need_self)
                    s_bself = sdf_sample_dispatch_2d(interp_method, Fb, Mxb, Myb, Mb[0], Mb[1], Mb[4], Mb[5], bxq, byq);
            } else {
                scalar_t gbx, gby;
                s_bself = bilinear_sample_grad_uniform_2d(Fb, Mxb, Myb, Mb[0], Mb[1], Mb[4], Mb[5], bxq, byq, gbx, gby);
                scalar_t nbx = gbx*Kb[0] + gby*Kb[2], nby = gbx*Kb[1] + gby*Kb[3];
                const scalar_t nlen = sqrt(nbx*nbx + nby*nby);
                const scalar_t invn = nlen > (scalar_t)0 ? ((scalar_t)1/nlen) : (scalar_t)0;
                nbx *= invn; nby *= invn;
                gpx = meas_p*nbx; gpy = meas_p*nby; gvx = meas_v*nbx; gvy = meas_v*nby;
            }
            const scalar_t inv_blend = soft ? ((scalar_t)1/blend_eps) : (scalar_t)0;
            scalar_t wb = 0;
            if (soft) {
                scalar_t Z = 0;
                for (int c = 0; c < B; ++c) {
                    const int i0 = (int)aabb_lo[c*2+0], j0 = (int)aabb_lo[c*2+1];
                    const int Ci = (int)aabb_dim[c*2+0], Cj = (int)aabb_dim[c*2+1];
                    if (i < i0 || i >= i0+Ci || j < j0 || j >= j0+Cj) continue;
                    scalar_t s_c;
                    if (c == b) s_c = s_bself;
                    else {
                        const scalar_t* Fc = F_flat + F_offsets[c];
                        const int Mxc = (int)body_shapes[c*2+0], Myc = (int)body_shapes[c*2+1];
                        const scalar_t* Mc = body_meta + c*7; const scalar_t* Kc = kin + c*11;
                        const scalar_t dxc = xc - Kc[4], dyc = yc - Kc[5];
                        s_c = sdf_sample_dispatch_2d(interp_method, Fc, Mxc, Myc, Mc[0], Mc[1], Mc[4], Mc[5], Kc[0]*dxc+Kc[1]*dyc, Kc[2]*dxc+Kc[3]*dyc);
                    }
                    Z += (scalar_t)1 / ((scalar_t)1 + exp(s_c * inv_blend));
                }
                if (Z > (scalar_t)0) wb = ((scalar_t)1 / ((scalar_t)1 + exp(s_bself * inv_blend))) / Z;
            } else if (owner_cc != nullptr) {
                // hard winner, read from the union the resolve kernel built
                wb = (owner_cc[g_idx] == b) ? (scalar_t)1 : (scalar_t)0;
            } else {
                scalar_t smin = (scalar_t)1e30; int bmin = -1;
                for (int c = 0; c < B; ++c) {
                    const int i0 = (int)aabb_lo[c*2+0], j0 = (int)aabb_lo[c*2+1];
                    const int Ci = (int)aabb_dim[c*2+0], Cj = (int)aabb_dim[c*2+1];
                    if (i < i0 || i >= i0+Ci || j < j0 || j >= j0+Cj) continue;
                    scalar_t s_c;
                    if (c == b) s_c = s_bself;
                    else {
                        const scalar_t* Fc = F_flat + F_offsets[c];
                        const int Mxc = (int)body_shapes[c*2+0], Myc = (int)body_shapes[c*2+1];
                        const scalar_t* Mc = body_meta + c*7; const scalar_t* Kc = kin + c*11;
                        const scalar_t dxc = xc - Kc[4], dyc = yc - Kc[5];
                        s_c = sdf_sample_dispatch_2d(interp_method, Fc, Mxc, Myc, Mc[0], Mc[1], Mc[4], Mc[5], Kc[0]*dxc+Kc[1]*dyc, Kc[2]*dxc+Kc[3]*dyc);
                    }
                    if (s_c < smin) { smin = s_c; bmin = c; }
                }
                wb = (bmin == b) ? (scalar_t)1 : (scalar_t)0;
            }

            if (wb != (scalar_t)0) {
                scalar_t fp_x = 0, fp_y = 0;
                if (has_p) { const scalar_t p_c = p_prev[g_idx]; fp_x = -p_c*gpx; fp_y = -p_c*gpy; }
                scalar_t fv_x = 0, fv_y = 0;
                if (has_v) {
                    const scalar_t nu_rho_val = (nu_rho_field_size == 1) ? nu_rho_field[0] : nu_rho_field[g_idx];
                    const int im1 = (i>0)?i-1:0, ip1 = (i+1<Ngx)?i+1:i, im2 = (i>1)?i-2:0, ip2 = (i+2<Ngx)?i+2:(Ngx-1);
                    const int jm1 = (j>0)?j-1:0, jp1 = (j+1<Ngy)?j+1:j, jm2 = (j>1)?j-2:0, jp2 = (j+2<Ngy)?j+2:(Ngy-1);
                    scalar_t dudx;
                    if (i+1 < Ngx) dudx = (u_prev[ip1*Ngy+j] - u_prev[i*Ngy+j]) * inv_h;
                    else            dudx = (u_prev[i*Ngy+j]   - u_prev[im1*Ngy+j]) * inv_h;
                    scalar_t dvdy;
                    if (j+1 < Ngy) dvdy = (v_prev[i*Ngy+jp1] - v_prev[i*Ngy+j]) * inv_h;
                    else            dvdy = (v_prev[i*Ngy+j]   - v_prev[i*Ngy+jm1]) * inv_h;
                    const scalar_t u_cc_jm2 = (scalar_t)0.5*(u_prev[i*Ngy+jm2]+u_prev[ip1*Ngy+jm2]);
                    const scalar_t u_cc_jm1 = (scalar_t)0.5*(u_prev[i*Ngy+jm1]+u_prev[ip1*Ngy+jm1]);
                    const scalar_t u_cc_j0  = (scalar_t)0.5*(u_prev[i*Ngy+j  ]+u_prev[ip1*Ngy+j  ]);
                    const scalar_t u_cc_jp1 = (scalar_t)0.5*(u_prev[i*Ngy+jp1]+u_prev[ip1*Ngy+jp1]);
                    const scalar_t u_cc_jp2 = (scalar_t)0.5*(u_prev[i*Ngy+jp2]+u_prev[ip1*Ngy+jp2]);
                    scalar_t dudy;
                    if (Ngy >= 3) {
                        if (j == 0) dudy = ((scalar_t)(-3)*u_cc_j0 + (scalar_t)4*u_cc_jp1 - u_cc_jp2) * (scalar_t)0.5 * inv_h;
                        else if (j == Ngy-1) dudy = ((scalar_t)3*u_cc_j0 - (scalar_t)4*u_cc_jm1 + u_cc_jm2) * (scalar_t)0.5 * inv_h;
                        else dudy = (u_cc_jp1 - u_cc_jm1) * (scalar_t)0.5 * inv_h;
                    } else dudy = (u_cc_jp1 - u_cc_jm1) * (scalar_t)0.5 * inv_h;
                    const scalar_t v_cc_im2 = (scalar_t)0.5*(v_prev[im2*Ngy+j]+v_prev[im2*Ngy+jp1]);
                    const scalar_t v_cc_im1 = (scalar_t)0.5*(v_prev[im1*Ngy+j]+v_prev[im1*Ngy+jp1]);
                    const scalar_t v_cc_i0  = (scalar_t)0.5*(v_prev[i  *Ngy+j]+v_prev[i  *Ngy+jp1]);
                    const scalar_t v_cc_ip1 = (scalar_t)0.5*(v_prev[ip1*Ngy+j]+v_prev[ip1*Ngy+jp1]);
                    const scalar_t v_cc_ip2 = (scalar_t)0.5*(v_prev[ip2*Ngy+j]+v_prev[ip2*Ngy+jp1]);
                    scalar_t dvdx;
                    if (Ngx >= 3) {
                        if (i == 0) dvdx = ((scalar_t)(-3)*v_cc_i0 + (scalar_t)4*v_cc_ip1 - v_cc_ip2) * (scalar_t)0.5 * inv_h;
                        else if (i == Ngx-1) dvdx = ((scalar_t)3*v_cc_i0 - (scalar_t)4*v_cc_im1 + v_cc_im2) * (scalar_t)0.5 * inv_h;
                        else dvdx = (v_cc_ip1 - v_cc_im1) * (scalar_t)0.5 * inv_h;
                    } else dvdx = (v_cc_ip1 - v_cc_im1) * (scalar_t)0.5 * inv_h;
                    const scalar_t sxx = nu_rho_val * (scalar_t)2 * dudx;
                    const scalar_t syy = nu_rho_val * (scalar_t)2 * dvdy;
                    const scalar_t sxy = nu_rho_val * (dudy + dvdx);
                    fv_x = sxx*gvx + sxy*gvy;
                    fv_y = sxy*gvx + syy*gvy;
                }
                const scalar_t arm_x = xc - Kb[6], arm_y = yc - Kb[7];
                const scalar_t fvx = wb*fv_x, fvy = wb*fv_y, fpx = wb*fp_x, fpy = wb*fp_y;
                acc[0] = (double)fvx; acc[1] = (double)fvy;
                acc[2] = (double)arm_x*(double)fvy - (double)arm_y*(double)fvx;
                acc[3] = (double)fpx; acc[4] = (double)fpy;
                acc[5] = (double)arm_x*(double)fpy - (double)arm_y*(double)fpx;
            }
        }
    }

    using BlockReduceD = cub::BlockReduce<double, BLOCK_SIZE>;
    __shared__ typename BlockReduceD::TempStorage tmp;
    const double h2_d = (double)h2;
#pragma unroll
    for (int c = 0; c < 6; ++c) {
        const double s = BlockReduceD(tmp).Sum(acc[c]);
        if (threadIdx.x == 0) atomicAdd(&out[b*6 + c], s * h2_d);
        __syncthreads();
    }
}

void streaming_sdf_forces_post_2d_cuda(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes, const at::Tensor& body_meta,
    const at::Tensor& kin,
    const at::Tensor& aabb_lo, const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy,
    const double h_grid, const int64_t max_vol_per_body,
    const at::Tensor& sdf_cc,
    const int64_t interp_method,
    const at::Tensor& u_prev, const at::Tensor& v_prev, const at::Tensor& p_prev,
    const at::Tensor& nu_rho_field,
    const double eps_body,
    const double sample_offset_pressure,
    const double sample_offset_friction,
    const double h2,
    const int64_t delta_order,
    const int64_t force_submethod,
    const double ph_tau,
    const at::Tensor& owner_cc,
    at::Tensor out)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngx = (int)gx.numel();
    const int Ngy = (int)gy.numel();
    // A size<=1 tensor means "no owner field" -> recompute the argmin here.
    const int32_t* owner_ptr = (owner_cc.numel() > 1)
        ? owner_cc.data_ptr<int32_t>() : nullptr;

    if (force_submethod == 0 || force_submethod == 2) {
        // UNIFIED union-measure + partition readout (2-D twin of the 3-D path):
        //   submethod 0 : union grad-H direction (exact net by SBP)
        //   submethod 2 : union coarea magnitude x per-body analytic normal
        const int bs0 = (max_vol_per_body <= 128)  ? 32
                      : (max_vol_per_body <= 4096) ? 128 : 256;
        const int nb0 = (int)((max_vol_per_body + bs0 - 1) / bs0);
        const bool body_normal = (force_submethod == 2);
        AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "forces_post_union_blend_2d_cuda", [&] {
            auto launch = [&](auto block_size_ic, auto bn_ic) {
                constexpr int BS = decltype(block_size_ic)::value;
                constexpr bool BN = decltype(bn_ic)::value;
                forces_post_union_blend_2d_kernel<scalar_t, BS, BN>
                    <<<dim3(nb0, B, 1), dim3(BS, 1, 1), 0, stream>>>(
                        F_flat.data_ptr<scalar_t>(), F_offsets.data_ptr<int64_t>(),
                        body_shapes.data_ptr<int64_t>(), body_meta.data_ptr<scalar_t>(),
                        kin.data_ptr<scalar_t>(), aabb_lo.data_ptr<int64_t>(),
                        aabb_dim.data_ptr<int64_t>(), gx.data_ptr<scalar_t>(),
                        gy.data_ptr<scalar_t>(), Ngx, Ngy,
                        sdf_cc.data_ptr<scalar_t>(), owner_ptr, (int)interp_method,
                        u_prev.data_ptr<scalar_t>(), v_prev.data_ptr<scalar_t>(),
                        p_prev.data_ptr<scalar_t>(),
                        nu_rho_field.data_ptr<scalar_t>(), (int64_t)nu_rho_field.numel(),
                        (scalar_t)(1.0 / h_grid), (scalar_t)(1.0 / eps_body),
                        (scalar_t)sample_offset_pressure, (scalar_t)sample_offset_friction,
                        (scalar_t)ph_tau, (scalar_t)h2, B, out.data_ptr<double>());
            };
            auto pick_bn = [&](auto block_size_ic) {
                if (body_normal) launch(block_size_ic, std::true_type{});
                else             launch(block_size_ic, std::false_type{});
            };
            switch (bs0) {
                case 32:  pick_bn(std::integral_constant<int, 32>{}); break;
                case 128: pick_bn(std::integral_constant<int, 128>{}); break;
                default:  pick_bn(std::integral_constant<int, 256>{}); break;
            }
        });
        return;
    }
}


// Smoothed Heaviside device helper, shared by the union ndelta readout's
// heaviside_grad_3d (below).
template <typename scalar_t>
__device__ __forceinline__ scalar_t heaviside_smooth_dev(scalar_t phi, scalar_t inv_eps) {
    const scalar_t pi = (scalar_t)3.141592653589793;
    scalar_t x = phi * inv_eps;
    x = x < (scalar_t)-1 ? (scalar_t)-1 : (x > (scalar_t)1 ? (scalar_t)1 : x);
    return (scalar_t)0.5 * ((scalar_t)1 + x + sin(pi * x) / pi);
}


// ----------------------------------------------------------------------
//  edge_order=2 discrete gradient of the union smooth Heaviside H_eps(phi-off).
//  The offset shifts the iso-surface the delta rides on (off_pres / off_visc).
// ----------------------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ void heaviside_grad_3d(
    const scalar_t* __restrict__ sdf_cc,
    const int Ngx, const int Ngy, const int Ngz,
    const int i, const int j, const int k,
    const scalar_t off, const scalar_t inv_eps, const scalar_t inv_h,
    scalar_t& gHx, scalar_t& gHy, scalar_t& gHz)
{
#define S_(ii,jj,kk) (sdf_cc[(((int64_t)(ii))*Ngy + (jj))*Ngz + (kk)])
#define H_(ii,jj,kk) heaviside_smooth_dev<scalar_t>(S_(ii,jj,kk) - off, inv_eps)
    gHx = 0; gHy = 0; gHz = 0;
    if (Ngx >= 3) {
        if (i == 0) gHx = ((scalar_t)(-3)*H_(0,j,k) + (scalar_t)4*H_(1,j,k) - H_(2,j,k)) * (scalar_t)0.5 * inv_h;
        else if (i == Ngx-1) gHx = ((scalar_t)3*H_(Ngx-1,j,k) - (scalar_t)4*H_(Ngx-2,j,k) + H_(Ngx-3,j,k)) * (scalar_t)0.5 * inv_h;
        else gHx = (H_(i+1,j,k) - H_(i-1,j,k)) * (scalar_t)0.5 * inv_h;
    } else if (Ngx == 2) gHx = (H_(1,j,k) - H_(0,j,k)) * inv_h;
    if (Ngy >= 3) {
        if (j == 0) gHy = ((scalar_t)(-3)*H_(i,0,k) + (scalar_t)4*H_(i,1,k) - H_(i,2,k)) * (scalar_t)0.5 * inv_h;
        else if (j == Ngy-1) gHy = ((scalar_t)3*H_(i,Ngy-1,k) - (scalar_t)4*H_(i,Ngy-2,k) + H_(i,Ngy-3,k)) * (scalar_t)0.5 * inv_h;
        else gHy = (H_(i,j+1,k) - H_(i,j-1,k)) * (scalar_t)0.5 * inv_h;
    } else if (Ngy == 2) gHy = (H_(i,1,k) - H_(i,0,k)) * inv_h;
    if (Ngz >= 3) {
        if (k == 0) gHz = ((scalar_t)(-3)*H_(i,j,0) + (scalar_t)4*H_(i,j,1) - H_(i,j,2)) * (scalar_t)0.5 * inv_h;
        else if (k == Ngz-1) gHz = ((scalar_t)3*H_(i,j,Ngz-1) - (scalar_t)4*H_(i,j,Ngz-2) + H_(i,j,Ngz-3)) * (scalar_t)0.5 * inv_h;
        else gHz = (H_(i,j,k+1) - H_(i,j,k-1)) * (scalar_t)0.5 * inv_h;
    } else if (Ngz == 2) gHz = (H_(i,j,1) - H_(i,j,0)) * inv_h;
#undef H_
#undef S_
}

// edge_order=2 central-difference gradient of the raw union SDF (used by the
// per-body-normal readout to weight the coarea |grad phi_union| area element).
template <typename scalar_t>
__device__ __forceinline__ void sdf_grad_3d(
    const scalar_t* __restrict__ sdf_cc,
    const int Ngx, const int Ngy, const int Ngz,
    const int i, const int j, const int k, const scalar_t inv_h,
    scalar_t& gsx, scalar_t& gsy, scalar_t& gsz)
{
#define S_(ii,jj,kk) (sdf_cc[(((int64_t)(ii))*Ngy + (jj))*Ngz + (kk)])
    gsx = 0; gsy = 0; gsz = 0;
    if (Ngx >= 3) {
        if (i == 0) gsx = ((scalar_t)(-3)*S_(0,j,k) + (scalar_t)4*S_(1,j,k) - S_(2,j,k)) * (scalar_t)0.5 * inv_h;
        else if (i == Ngx-1) gsx = ((scalar_t)3*S_(Ngx-1,j,k) - (scalar_t)4*S_(Ngx-2,j,k) + S_(Ngx-3,j,k)) * (scalar_t)0.5 * inv_h;
        else gsx = (S_(i+1,j,k) - S_(i-1,j,k)) * (scalar_t)0.5 * inv_h;
    } else if (Ngx == 2) gsx = (S_(1,j,k) - S_(0,j,k)) * inv_h;
    if (Ngy >= 3) {
        if (j == 0) gsy = ((scalar_t)(-3)*S_(i,0,k) + (scalar_t)4*S_(i,1,k) - S_(i,2,k)) * (scalar_t)0.5 * inv_h;
        else if (j == Ngy-1) gsy = ((scalar_t)3*S_(i,Ngy-1,k) - (scalar_t)4*S_(i,Ngy-2,k) + S_(i,Ngy-3,k)) * (scalar_t)0.5 * inv_h;
        else gsy = (S_(i,j+1,k) - S_(i,j-1,k)) * (scalar_t)0.5 * inv_h;
    } else if (Ngy == 2) gsy = (S_(i,1,k) - S_(i,0,k)) * inv_h;
    if (Ngz >= 3) {
        if (k == 0) gsz = ((scalar_t)(-3)*S_(i,j,0) + (scalar_t)4*S_(i,j,1) - S_(i,j,2)) * (scalar_t)0.5 * inv_h;
        else if (k == Ngz-1) gsz = ((scalar_t)3*S_(i,j,Ngz-1) - (scalar_t)4*S_(i,j,Ngz-2) + S_(i,j,Ngz-3)) * (scalar_t)0.5 * inv_h;
        else gsz = (S_(i,j,k+1) - S_(i,j,k-1)) * (scalar_t)0.5 * inv_h;
    } else if (Ngz == 2) gsz = (S_(i,j,1) - S_(i,j,0)) * inv_h;
#undef S_
}

// ----------------------------------------------------------------------
//  Fused trilinear sample + ANALYTIC gradient of a per-body SDF table.
//  Returns phi_b(q) and writes d phi/d(body-coord) into (gbx,gby,gbz) from the
//  SAME 8-corner stencil the value uses -- no finite difference, no stencil
//  width to choose.  The analytic interpolant gradient is the d->0 limit, so it
//  recovers the true surface normal (validated: 1.3 deg vs the marching-cubes
//  normal, per-link force error 135% -> 5%) at the cost of a few extra FLOPs
//  and ZERO extra memory reads.
// ----------------------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ scalar_t trilinear_sample_grad_uniform(
    const scalar_t* __restrict__ F,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq,
    scalar_t& gbx, scalar_t& gby, scalar_t& gbz)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    scalar_t tz = (zq - bz0) * inv_dz;
    tx = max((scalar_t)0, min(tx, (scalar_t)(Mx - 1)));
    ty = max((scalar_t)0, min(ty, (scalar_t)(My - 1)));
    tz = max((scalar_t)0, min(tz, (scalar_t)(Mz - 1)));
    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;
    int iz = (int)tz; if (iz > Mz - 2) iz = Mz - 2;
    const scalar_t fx = tx - (scalar_t)ix, fy = ty - (scalar_t)iy, fz = tz - (scalar_t)iz;
    const int s2 = Mz, s1 = My * Mz, base = ix*s1 + iy*s2 + iz;
    const scalar_t c000 = F[base],          c001 = F[base+1];
    const scalar_t c010 = F[base+s2],       c011 = F[base+s2+1];
    const scalar_t c100 = F[base+s1],       c101 = F[base+s1+1];
    const scalar_t c110 = F[base+s1+s2],    c111 = F[base+s1+s2+1];
    const scalar_t omfx = (scalar_t)1-fx, omfy = (scalar_t)1-fy, omfz = (scalar_t)1-fz;
    const scalar_t val =
        omfx*(omfy*(omfz*c000+fz*c001)+fy*(omfz*c010+fz*c011)) +
        fx  *(omfy*(omfz*c100+fz*c101)+fy*(omfz*c110+fz*c111));
    // analytic gradient in table-index space, scaled to body-coord space
    const scalar_t dfx = omfy*(omfz*(c100-c000)+fz*(c101-c001)) + fy*(omfz*(c110-c010)+fz*(c111-c011));
    const scalar_t dfy = omfx*(omfz*(c010-c000)+fz*(c011-c001)) + fx*(omfz*(c110-c100)+fz*(c111-c101));
    const scalar_t dfz = omfx*(omfy*(c001-c000)+fy*(c011-c010)) + fx*(omfy*(c101-c100)+fy*(c111-c110));
    gbx = dfx * inv_dx; gby = dfy * inv_dy; gbz = dfz * inv_dz;
    return val;
}

// ----------------------------------------------------------------------
//  UNIFIED union-measure + partition readout  --  the fixed ndelta.
//
//  Both channels are integrated against the UNION smoothed-Heaviside gradient
//  d_iH_eps(phi_union - off), which bundles (delta * |grad phi| * normal) into
//  one discrete gradient: no per-link joint caps enter the measure, and the
//  coarea |grad phi| is automatic (force_delta_order is irrelevant here).
//    pressure:  f_i = -p         d_iH(phi_union - off_pres)
//    viscous:   f_i =  sigma_ij  d_jH(phi_union - off_visc)
//  The union force density is split to the individual links by the SAME smooth
//  partition of unity the streaming body-velocity blend uses,
//  w_b = sigmoid(-phi_b/blend_eps) normalised (blend_eps<=0 -> hard nearest-
//  body winner), so Sum_b F_b == union force exactly.
//
//  Launch geometry: dim3(nblocks, B), one thread per (body b, cell in b's
//  AABB), grow-only max_vol watermark, so it is CUDA-graph-safe under the
//  whole-step capture.
//  A union cell covered by several bodies is visited by each covering body's
//  thread; every thread computes the SAME normaliser Z over the SAME covering
//  set, and body b's thread adds only its own share w_b*f -> Sum over threads
//  reconstructs the full cell force with no double counting.  Strain is
//  recomputed per (body,cell).
//
//  HARD partition (blend_eps <= 0): the winner is READ from ``owner_cc``, the
//  argmin the streaming resolve kernel already computed when it built the
//  union.  Recomputing it here was both redundant (a per-cell sweep re-sampling
//  every candidate body's SDF brick) and WRONG: the candidate list was the
//  force-window ``aabb_dim``, which has force-opted-out bodies zeroed, so a
//  cell whose interface force belongs to a fixed body (an immersed ramp) could
//  not be claimed by it and was handed in full to whichever body still covered
//  it -- an 11 N step onto a link 0.166 m outside the fluid box, in air.  With
//  owner_cc the opted-out body owns its own cells and no one else can claim
//  them.  Pass owner_cc = nullptr to fall back to the recompute loop (the
//  analytical body path, which never runs the resolve kernel).
//  The SOFT partition is a sigmoid-weighted sum over the whole covering set,
//  which one index cannot express, so it keeps the loop unconditionally.
// ----------------------------------------------------------------------
template <typename scalar_t, int BLOCK_SIZE, bool BODY_NORMAL>
__global__ void forces_post_union_blend_3d_kernel(
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
    const int32_t* __restrict__ owner_cc,
    const int interp_method,
    const scalar_t* __restrict__ u_prev,
    const scalar_t* __restrict__ v_prev,
    const scalar_t* __restrict__ w_prev,
    const scalar_t* __restrict__ p_prev,
    const scalar_t* __restrict__ nu_rho_field,
    const int64_t   nu_rho_field_size,
    const scalar_t inv_h,
    const scalar_t inv_eps,
    const scalar_t off_pres,
    const scalar_t off_visc,
    const scalar_t blend_eps,
    const scalar_t h3,
    const int B,
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
        const int i = (int)aabb_lo[b*3 + 0] + di;
        const int j = (int)aabb_lo[b*3 + 1] + dj;
        const int k = (int)aabb_lo[b*3 + 2] + dk;
        const int g_idx = (i * Ngy + j) * Ngz + k;

        // ---- the per-channel "measure vector" the traction dots with ----
        // BODY_NORMAL=false: the union Heaviside gradient d_iH(phi-off) directly
        //   (delta * |grad phi| * union-normal folded into one discrete gradient,
        //   exact net by summation-by-parts).
        // BODY_NORMAL=true : the scalar coarea band measure delta(phi-off)*|grad
        //   phi_union| only; the DIRECTION comes from the owning body's analytic
        //   SDF normal n_b (computed below), giving accurate per-link forces.
        scalar_t gpx = 0, gpy = 0, gpz = 0, gvx = 0, gvy = 0, gvz = 0;
        scalar_t meas_p = 0, meas_v = 0;   // BODY_NORMAL scalar measures
        bool has_p = false, has_v = false;
        // ---- ownership gate, FIRST -----------------------------------
        // Under the hard partition a cell this body does not own contributes
        // exactly zero, and owner_cc answers that in one int32 load.  Testing
        // it here rather than after the measure is what makes the field a
        // speed-up and not just a correctness fix: it skips the two Heaviside
        // gradients (~14 sdf_cc loads + 2 sin each) and, downstream, the whole
        // strain-rate block (~40 velocity loads).  With B overlapping links
        // most (body, cell) pairs are NOT owned, so this is the common case.
        // Leaving has_p/has_v false makes every branch below fall through with
        // acc = 0; the block reduce still runs, as it must.
        const bool owned = (owner_cc == nullptr) || (blend_eps > (scalar_t)0)
                           || (owner_cc[g_idx] == b);
        if (!owned) {
            // nothing to do for this (body, cell)
        } else if (!BODY_NORMAL) {
            heaviside_grad_3d<scalar_t>(sdf_cc, Ngx, Ngy, Ngz, i, j, k,
                                        off_pres, inv_eps, inv_h, gpx, gpy, gpz);
            heaviside_grad_3d<scalar_t>(sdf_cc, Ngx, Ngy, Ngz, i, j, k,
                                        off_visc, inv_eps, inv_h, gvx, gvy, gvz);
            has_p = (gpx != (scalar_t)0 || gpy != (scalar_t)0 || gpz != (scalar_t)0);
            has_v = (gvx != (scalar_t)0 || gvy != (scalar_t)0 || gvz != (scalar_t)0);
        } else {
            scalar_t gux, guy, guz;
            sdf_grad_3d<scalar_t>(sdf_cc, Ngx, Ngy, Ngz, i, j, k, inv_h, gux, guy, guz);
            const scalar_t gmag = sqrt(gux*gux + guy*guy + guz*guz);
            const scalar_t phi_u = sdf_cc[g_idx];
            const scalar_t pi_v = (scalar_t)3.141592653589793;
            const scalar_t dp = (phi_u - off_pres) * inv_eps;
            const scalar_t dv = (phi_u - off_visc) * inv_eps;
            const scalar_t half_ie = (scalar_t)0.5 * inv_eps;
            const scalar_t delp = (dp > (scalar_t)-1 && dp < (scalar_t)1)
                ? ((scalar_t)1 + cos(pi_v * dp)) * half_ie : (scalar_t)0;
            const scalar_t delv = (dv > (scalar_t)-1 && dv < (scalar_t)1)
                ? ((scalar_t)1 + cos(pi_v * dv)) * half_ie : (scalar_t)0;
            meas_p = delp * gmag;
            meas_v = delv * gmag;
            has_p = (meas_p != (scalar_t)0);
            has_v = (meas_v != (scalar_t)0);
        }

        if (has_p || has_v) {
            const scalar_t xc = gx[i], yc = gy[j], zc = gz[k];

            // this body's phi at the cell (reused for the partition weight)
            const scalar_t* Fb  = F_flat + F_offsets[b];
            const int Mxb = (int)body_shapes[b*3 + 0];
            const int Myb = (int)body_shapes[b*3 + 1];
            const int Mzb = (int)body_shapes[b*3 + 2];
            const scalar_t* Mb  = body_meta + b*10;
            const scalar_t* Kb  = kin + b*21;
            {
                const scalar_t dxw = xc - Kb[9], dyw = yc - Kb[10], dzw = zc - Kb[11];
                const scalar_t bxq = Kb[0]*dxw + Kb[1]*dyw + Kb[2]*dzw;
                const scalar_t byq = Kb[3]*dxw + Kb[4]*dyw + Kb[5]*dzw;
                const scalar_t bzq = Kb[6]*dxw + Kb[7]*dyw + Kb[8]*dzw;
                // ---- partition weight for THIS body over the covering set ----
                const bool soft = blend_eps > (scalar_t)0;
                // With a hard partition the winner comes from owner_cc, so
                // this body's own phi is needed only for the soft weight (and
                // for the per-body normal, which BODY_NORMAL computes anyway).
                const bool need_self = soft || (owner_cc == nullptr);
                scalar_t s_bself = 0;
                if (!BODY_NORMAL) {
                    if (need_self)
                        s_bself = sdf_sample_dispatch(interp_method, Fb,
                            Mxb, Myb, Mzb, Mb[0], Mb[1], Mb[2], Mb[6], Mb[7], Mb[8],
                            bxq, byq, bzq);
                } else {
                    // fused value + analytic body-frame SDF gradient (one stencil load)
                    scalar_t gbx, gby, gbz;
                    s_bself = trilinear_sample_grad_uniform(Fb,
                        Mxb, Myb, Mzb, Mb[0], Mb[1], Mb[2], Mb[6], Mb[7], Mb[8],
                        bxq, byq, bzq, gbx, gby, gbz);
                    // rotate body-frame gradient to world: n_world = R^T grad_body
                    scalar_t nbx = gbx*Kb[0] + gby*Kb[3] + gbz*Kb[6];
                    scalar_t nby = gbx*Kb[1] + gby*Kb[4] + gbz*Kb[7];
                    scalar_t nbz = gbx*Kb[2] + gby*Kb[5] + gbz*Kb[8];
                    const scalar_t nlen = sqrt(nbx*nbx + nby*nby + nbz*nbz);
                    const scalar_t inv_nlen = nlen > (scalar_t)0 ? ((scalar_t)1 / nlen) : (scalar_t)0;
                    nbx *= inv_nlen; nby *= inv_nlen; nbz *= inv_nlen;
                    // measure direction = per-body normal, magnitude = union coarea band
                    gpx = meas_p * nbx; gpy = meas_p * nby; gpz = meas_p * nbz;
                    gvx = meas_v * nbx; gvy = meas_v * nby; gvz = meas_v * nbz;
                }

                const scalar_t inv_blend = soft ? ((scalar_t)1 / blend_eps) : (scalar_t)0;
                scalar_t wb = 0;
                if (soft) {
                    scalar_t Z = 0;
                    for (int c = 0; c < B; ++c) {
                        const int i0 = (int)aabb_lo[c*3+0], j0 = (int)aabb_lo[c*3+1], k0 = (int)aabb_lo[c*3+2];
                        const int Ci = (int)aabb_dim[c*3+0], Cj = (int)aabb_dim[c*3+1], Ck = (int)aabb_dim[c*3+2];
                        if (i < i0 || i >= i0+Ci || j < j0 || j >= j0+Cj || k < k0 || k >= k0+Ck) continue;
                        scalar_t s_c;
                        if (c == b) s_c = s_bself;
                        else {
                            const scalar_t* Fc = F_flat + F_offsets[c];
                            const int Mxc = (int)body_shapes[c*3+0], Myc = (int)body_shapes[c*3+1], Mzc = (int)body_shapes[c*3+2];
                            const scalar_t* Mc = body_meta + c*10;
                            const scalar_t* Kc = kin + c*21;
                            const scalar_t dxc = xc - Kc[9], dyc = yc - Kc[10], dzc = zc - Kc[11];
                            const scalar_t bxc = Kc[0]*dxc + Kc[1]*dyc + Kc[2]*dzc;
                            const scalar_t byc = Kc[3]*dxc + Kc[4]*dyc + Kc[5]*dzc;
                            const scalar_t bzc = Kc[6]*dxc + Kc[7]*dyc + Kc[8]*dzc;
                            s_c = sdf_sample_dispatch(interp_method, Fc, Mxc, Myc, Mzc,
                                Mc[0], Mc[1], Mc[2], Mc[6], Mc[7], Mc[8], bxc, byc, bzc);
                        }
                        Z += (scalar_t)1 / ((scalar_t)1 + exp(s_c * inv_blend));
                    }
                    if (Z > (scalar_t)0)
                        wb = ((scalar_t)1 / ((scalar_t)1 + exp(s_bself * inv_blend))) / Z;
                } else if (owner_cc != nullptr) {
                    // hard winner, read from the union the resolve kernel built
                    wb = (owner_cc[g_idx] == b) ? (scalar_t)1 : (scalar_t)0;
                } else {
                    // hard nearest-body winner over the covering set
                    scalar_t smin = (scalar_t)1e30; int bmin = -1;
                    for (int c = 0; c < B; ++c) {
                        const int i0 = (int)aabb_lo[c*3+0], j0 = (int)aabb_lo[c*3+1], k0 = (int)aabb_lo[c*3+2];
                        const int Ci = (int)aabb_dim[c*3+0], Cj = (int)aabb_dim[c*3+1], Ck = (int)aabb_dim[c*3+2];
                        if (i < i0 || i >= i0+Ci || j < j0 || j >= j0+Cj || k < k0 || k >= k0+Ck) continue;
                        scalar_t s_c;
                        if (c == b) s_c = s_bself;
                        else {
                            const scalar_t* Fc = F_flat + F_offsets[c];
                            const int Mxc = (int)body_shapes[c*3+0], Myc = (int)body_shapes[c*3+1], Mzc = (int)body_shapes[c*3+2];
                            const scalar_t* Mc = body_meta + c*10;
                            const scalar_t* Kc = kin + c*21;
                            const scalar_t dxc = xc - Kc[9], dyc = yc - Kc[10], dzc = zc - Kc[11];
                            const scalar_t bxc = Kc[0]*dxc + Kc[1]*dyc + Kc[2]*dzc;
                            const scalar_t byc = Kc[3]*dxc + Kc[4]*dyc + Kc[5]*dzc;
                            const scalar_t bzc = Kc[6]*dxc + Kc[7]*dyc + Kc[8]*dzc;
                            s_c = sdf_sample_dispatch(interp_method, Fc, Mxc, Myc, Mzc,
                                Mc[0], Mc[1], Mc[2], Mc[6], Mc[7], Mc[8], bxc, byc, bzc);
                        }
                        if (s_c < smin) { smin = s_c; bmin = c; }
                    }
                    wb = (bmin == b) ? (scalar_t)1 : (scalar_t)0;
                }

                if (wb != (scalar_t)0) {
                    // ---- pressure force density  f = -p grad H(phi-off_pres) ----
                    scalar_t fp_x = 0, fp_y = 0, fp_z = 0;
                    if (has_p) {
                        const scalar_t p_c = p_prev[g_idx];
                        fp_x = -p_c * gpx; fp_y = -p_c * gpy; fp_z = -p_c * gpz;
                    }
                    // ---- viscous force density  f_i = sigma_ij grad_j H(phi-off_visc) ----
                    scalar_t fv_x = 0, fv_y = 0, fv_z = 0;
                    if (has_v) {
                        const scalar_t nu_rho_val = (nu_rho_field_size == 1)
                            ? nu_rho_field[0] : nu_rho_field[g_idx];
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
                        scalar_t dudx;
                        if (i + 1 < Ngx) dudx = (u_prev[(ip1*Ngy+j)*Ngz+k] - u_prev[(i*Ngy+j)*Ngz+k]) * inv_h;
                        else              dudx = (u_prev[(i*Ngy+j)*Ngz+k]   - u_prev[(im1*Ngy+j)*Ngz+k]) * inv_h;
                        scalar_t dvdy;
                        if (j + 1 < Ngy) dvdy = (v_prev[(i*Ngy+jp1)*Ngz+k] - v_prev[(i*Ngy+j)*Ngz+k]) * inv_h;
                        else              dvdy = (v_prev[(i*Ngy+j)*Ngz+k]   - v_prev[(i*Ngy+jm1)*Ngz+k]) * inv_h;
                        scalar_t dwdz;
                        if (k + 1 < Ngz) dwdz = (w_prev[(i*Ngy+j)*Ngz+kp1] - w_prev[(i*Ngy+j)*Ngz+k]) * inv_h;
                        else              dwdz = (w_prev[(i*Ngy+j)*Ngz+k]   - w_prev[(i*Ngy+j)*Ngz+km1]) * inv_h;
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
                        const scalar_t sxx = nu_rho_val * (scalar_t)2 * dudx;
                        const scalar_t syy = nu_rho_val * (scalar_t)2 * dvdy;
                        const scalar_t szz = nu_rho_val * (scalar_t)2 * dwdz;
                        const scalar_t sxy = nu_rho_val * (dudy + dvdx);
                        const scalar_t sxz = nu_rho_val * (dudz + dwdx);
                        const scalar_t syz = nu_rho_val * (dvdz + dwdy);
                        fv_x = sxx*gvx + sxy*gvy + sxz*gvz;
                        fv_y = sxy*gvx + syy*gvy + syz*gvz;
                        fv_z = sxz*gvx + syz*gvy + szz*gvz;
                    }
                    const scalar_t ax = xc - Kb[12], ay = yc - Kb[13], az = zc - Kb[14];
                    const scalar_t fvx = wb*fv_x, fvy = wb*fv_y, fvz = wb*fv_z;
                    const scalar_t fpx = wb*fp_x, fpy = wb*fp_y, fpz = wb*fp_z;
                    acc[0] = (double)fvx; acc[1] = (double)fvy; acc[2] = (double)fvz;
                    acc[3] = (double)ay*(double)fvz - (double)az*(double)fvy;
                    acc[4] = (double)az*(double)fvx - (double)ax*(double)fvz;
                    acc[5] = (double)ax*(double)fvy - (double)ay*(double)fvx;
                    acc[6] = (double)fpx; acc[7] = (double)fpy; acc[8] = (double)fpz;
                    acc[9]  = (double)ay*(double)fpz - (double)az*(double)fpy;
                    acc[10] = (double)az*(double)fpx - (double)ax*(double)fpz;
                    acc[11] = (double)ax*(double)fpy - (double)ay*(double)fpx;
                }
            }
        }
    }

    using BlockReduceD = cub::BlockReduce<double, BLOCK_SIZE>;
    __shared__ typename BlockReduceD::TempStorage tmp;
    const double h3_d = (double)h3;
#pragma unroll
    for (int c = 0; c < 12; ++c) {
        const double s = BlockReduceD(tmp).Sum(acc[c]);
        if (threadIdx.x == 0)
            atomicAdd(&out[b*12 + c], s * h3_d);
        __syncthreads();
    }
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
    const double sample_offset_pressure,
    const double sample_offset_friction,
    const double h3,
    const int64_t delta_order,
    const int64_t force_submethod,
    const double ph_tau,
    const at::Tensor& owner_cc,
    at::Tensor out)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0 || max_vol_per_body <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int Ngx = (int)gx.numel();
    const int Ngy = (int)gy.numel();
    const int Ngz = (int)gz.numel();
    // A size<=1 tensor means "no owner field" -> recompute the argmin here.
    const int32_t* owner_ptr = (owner_cc.numel() > 1)
        ? owner_cc.data_ptr<int32_t>() : nullptr;

    if (force_submethod == 0 || force_submethod == 2) {
        // UNIFIED union-measure + partition readout (the fixed ndelta): BOTH
        // channels via the union band measure, split to links by the streaming
        // body-velocity blend weight (ph_tau slot carries the blend eps in
        // metres; <=0 -> hard nearest-body winner).  Launch geometry
        // dim3(nblocks, B) with a grow-only max_vol watermark, so it is
        // CUDA-graph-safe under the whole-step capture.
        //   submethod 0 : union smooth-Heaviside gradient direction
        //                 (exact net by summation-by-parts; per-link normal crude).
        //   submethod 2 : union coarea magnitude x per-body ANALYTIC normal
        //                 (accurate per-link forces on thin/multi-link bodies).
        const int blockSize0 = (max_vol_per_body <= 128)  ? 32
                             : (max_vol_per_body <= 4096) ? 128
                                                          : 256;
        const int nblocks0 = (int)((max_vol_per_body + blockSize0 - 1) / blockSize0);
        const bool body_normal = (force_submethod == 2);
        AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "forces_post_union_blend_3d_cuda", [&] {
            auto launch = [&](auto block_size_ic, auto bn_ic) {
                constexpr int BS = decltype(block_size_ic)::value;
                constexpr bool BN = decltype(bn_ic)::value;
                forces_post_union_blend_3d_kernel<scalar_t, BS, BN>
                    <<<dim3(nblocks0, B, 1), dim3(BS, 1, 1), 0, stream>>>(
                        F_flat.data_ptr<scalar_t>(), F_offsets.data_ptr<int64_t>(),
                        body_shapes.data_ptr<int64_t>(), body_meta.data_ptr<scalar_t>(),
                        kin.data_ptr<scalar_t>(), aabb_lo.data_ptr<int64_t>(),
                        aabb_dim.data_ptr<int64_t>(), gx.data_ptr<scalar_t>(),
                        gy.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(), Ngx, Ngy, Ngz,
                        sdf_cc.data_ptr<scalar_t>(), owner_ptr, (int)interp_method,
                        u.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                        w.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(),
                        nu_rho_field.data_ptr<scalar_t>(), (int64_t)nu_rho_field.numel(),
                        (scalar_t)(1.0 / h_grid), (scalar_t)(1.0 / eps_body),
                        (scalar_t)sample_offset_pressure,
                        (scalar_t)sample_offset_friction,
                        (scalar_t)ph_tau, (scalar_t)h3, B,
                        out.data_ptr<double>());
            };
            auto pick_bn = [&](auto block_size_ic) {
                if (body_normal) launch(block_size_ic, std::true_type{});
                else             launch(block_size_ic, std::false_type{});
            };
            switch (blockSize0) {
                case 32:  pick_bn(std::integral_constant<int, 32>{}); break;
                case 128: pick_bn(std::integral_constant<int, 128>{}); break;
                default:  pick_bn(std::integral_constant<int, 256>{}); break;
            }
        });
        return;
    }
}


TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("streaming_sdf_forces_post_3d", &streaming_sdf_forces_post_3d_cuda);
    m.impl("streaming_sdf_forces_post_2d", &streaming_sdf_forces_post_2d_cuda);
}

}  // namespace lilytorch_kernels
