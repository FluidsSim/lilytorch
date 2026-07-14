// =====================================================================
//  bdim_forcing_{2d,3d} — static full-grid BDIM2 + Poisson coefficient
//  + Maertens–Weymouth body-divergence correction.
//
//  Static full-grid BDIM forcing kernel
//  (bdim_forcing_2d_kernel / bdim_forcing_3d_kernel), which is itself a
//  faithful port of the original native bdim_one_axis in streaming_sdf.cu.
//
//  Key difference from bdim_coeff_{2,3}d: the launch dim is the FULL grid
//  (pose-independent → CUDA-graph-capturable); the per-step dirty AABB is
//  staged into a device-resident int32 rect tensor outside the graph.
//  Threads inside the rect do the BDIM2 math; outside they pass the
//  advected velocity straight through (u0 = u_prime) and leave ch/cv[/cw]
//  at their persistent dt/rho prefill.
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace lilytorch_kernels {

// =====================================================================
//  bdim_one_axis_3d — device helper (included for separate-compilation
//  compatibility; identical to the same-named helper in streaming_sdf.cu).
// =====================================================================
template <typename scalar_t>
__device__ __forceinline__ void bdim_one_axis_3d(
    const scalar_t* __restrict__ phi_prime,
    const scalar_t* __restrict__ sdf,
    const scalar_t* __restrict__ body,
    const scalar_t eps,
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
    if (i > 0 && i < Ngx - 1)
        ddx = ((phi_prime[g_ip] - body[g_ip]) - (phi_prime[g_im] - body[g_im])) * inv_2h;
    else ddx = scalar_t(0);
    if (j > 0 && j < Ngy - 1)
        ddy = ((phi_prime[g_jp] - body[g_jp]) - (phi_prime[g_jm] - body[g_jm])) * inv_2h;
    else ddy = scalar_t(0);
    if (k > 0 && k < Ngz - 1)
        ddz = ((phi_prime[g_kp] - body[g_kp]) - (phi_prime[g_km] - body[g_km])) * inv_2h;
    else ddz = scalar_t(0);
    const scalar_t nd = nx * ddx + ny * ddy + nz * ddz;

    phi_out[g] = mu0 * diff_c + b_c + mu1 * nd;
    if (i >= 1 && j >= 1 && k >= 1 && i <= c_hi_i && j <= c_hi_j && k <= c_hi_k) {
        c_out[(i - 1) * c_stride_i + (j - 1) * c_stride_j + (k - 1)] =
            (mu0_proj ? dt * mu0 : dt) / rho_f;
    }
}

// =====================================================================
//  2-D bdim_one_axis device helper
// =====================================================================
template <typename scalar_t>
__device__ void bdim_one_axis_2d(
    const scalar_t* __restrict__ phi_prime,
    const scalar_t* __restrict__ sdf,
    const scalar_t* __restrict__ body,
    const scalar_t eps,
    const scalar_t rho_f,
    const scalar_t dt,
    const scalar_t inv_2h,
    const int Ngx, const int Ngy,
    const int i, const int j,
    scalar_t* __restrict__ phi_out,
    scalar_t* __restrict__ c_out,
    const int mu0_proj)
{
    const int stride_i = Ngy;
    const int g = i * stride_i + j;

    const int im = (i > 0)       ? (i - 1) : 0;
    const int ip = (i < Ngx - 1) ? (i + 1) : Ngx - 1;
    const int jm = (j > 0)       ? (j - 1) : 0;
    const int jp = (j < Ngy - 1) ? (j + 1) : Ngy - 1;

    const int g_im = im * stride_i + j;
    const int g_ip = ip * stride_i + j;
    const int g_jm = i  * stride_i + jm;
    const int g_jp = i  * stride_i + jp;

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

    scalar_t nx = (sdf[g_ip] - sdf[g_im]) * inv_2h;
    scalar_t ny = (sdf[g_jp] - sdf[g_jm]) * inv_2h;
    const scalar_t nn = sqrt(nx*nx + ny*ny);
    if (nn > scalar_t(0)) {
        const scalar_t inv_nn = scalar_t(1) / nn;
        nx *= inv_nn;
        ny *= inv_nn;
    }

    const scalar_t b_c    = body[g];
    const scalar_t pp_c   = phi_prime[g];
    const scalar_t diff_c = pp_c - b_c;

    scalar_t ddx, ddy;
    if (i > 0 && i < Ngx - 1) {
        ddx = ((phi_prime[g_ip] - body[g_ip]) -
               (phi_prime[g_im] - body[g_im])) * inv_2h;
    } else { ddx = scalar_t(0); }
    if (j > 0 && j < Ngy - 1) {
        ddy = ((phi_prime[g_jp] - body[g_jp]) -
               (phi_prime[g_jm] - body[g_jm])) * inv_2h;
    } else { ddy = scalar_t(0); }
    const scalar_t nd = nx * ddx + ny * ddy;

    phi_out[g] = mu0 * diff_c + b_c + mu1 * nd;
    // 2-D: Poisson coefficient at the SAME full-grid index g (no face-grid offset).
    c_out[g] = (mu0_proj ? dt * mu0 : dt) / rho_f;
}

// =====================================================================
//  3-D bdim_forcing CUDA kernel (static full-grid launch)
// =====================================================================
template <typename scalar_t>
__global__ void bdim_forcing_3d_kernel(
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
    const scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ div_corr,
    const int32_t* __restrict__ rect,
    const scalar_t eps,
    const scalar_t rho_f,
    const scalar_t dt,
    const scalar_t inv_2h,
    const scalar_t eps_mw,
    const scalar_t inv_dx,
    const scalar_t inv_dy,
    const scalar_t inv_dz,
    const int Ngx, const int Ngy, const int Ngz,
    const int mu0_proj,
    const int mw_on)
{
    // Static FULL-GRID 1-D launch, k fastest → coalesced global stores.
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Ngx * Ngy * Ngz;
    if (g >= total) return;

    const int k = g % Ngz;
    const int rem = g / Ngz;
    const int j = rem % Ngy;
    const int i = rem / Ngy;

    const int di = i - rect[0];
    const int dj = j - rect[1];
    const int dk = k - rect[2];
    if (di >= 0 && di < rect[3] && dj >= 0 && dj < rect[4] && dk >= 0 && dk < rect[5]) {
        // ch: x-face grid (Ngx-1, Ngy-2, Ngz-2)
        bdim_one_axis_3d<scalar_t>(
            u_prime, sdf_u, body_u,
            eps, rho_f, dt, inv_2h,
            Ngx, Ngy, Ngz, i, j, k, u0, ch,
            (Ngy - 2) * (Ngz - 2), (Ngz - 2),
            Ngx - 1, Ngy - 2, Ngz - 2, mu0_proj);
        // cv: y-face grid (Ngx-2, Ngy-1, Ngz-2)
        bdim_one_axis_3d<scalar_t>(
            v_prime, sdf_v, body_v,
            eps, rho_f, dt, inv_2h,
            Ngx, Ngy, Ngz, i, j, k, v0, cv,
            (Ngy - 1) * (Ngz - 2), (Ngz - 2),
            Ngx - 2, Ngy - 1, Ngz - 2, mu0_proj);
        // cw: z-face grid (Ngx-2, Ngy-2, Ngz-1)
        bdim_one_axis_3d<scalar_t>(
            w_prime, sdf_w, body_w,
            eps, rho_f, dt, inv_2h,
            Ngx, Ngy, Ngz, i, j, k, w0, cw,
            (Ngy - 2) * (Ngz - 1), (Ngz - 1),
            Ngx - 2, Ngy - 2, Ngz - 1, mu0_proj);
    } else {
        u0[g] = u_prime[g];
        v0[g] = v_prime[g];
        w0[g] = w_prime[g];
    }

    // Maertens–Weymouth body-divergence correction (full grid).
    if (mw_on) {
        scalar_t db = scalar_t(0);
        if (i > 0 && i < Ngx - 1 && j > 0 && j < Ngy - 1 && k > 0 && k < Ngz - 1) {
            const int stride_i = Ngy * Ngz;
            const int stride_j = Ngz;
            db = ((body_u[g + stride_i] - body_u[g]) * inv_dx
                  + (body_v[g + stride_j] - body_v[g]) * inv_dy
                  + (body_w[g + 1] - body_w[g]) * inv_dz);
        }
        const scalar_t deps = max(scalar_t(-1), min(sdf_cc[g] / eps_mw, scalar_t(1)));
        const scalar_t pi = scalar_t(M_PI);
        const scalar_t mu0c = scalar_t(0.5) * (scalar_t(1) + deps + sin(pi * deps) / pi);
        div_corr[g] = (scalar_t(1) - mu0c) * db;
    }
}

// =====================================================================
//  2-D bdim_forcing CUDA kernel (static full-grid launch)
// =====================================================================
template <typename scalar_t>
__global__ void bdim_forcing_2d_kernel(
    const scalar_t* __restrict__ u_prime,
    const scalar_t* __restrict__ v_prime,
    const scalar_t* __restrict__ sdf_u,
    const scalar_t* __restrict__ sdf_v,
    const scalar_t* __restrict__ body_u,
    const scalar_t* __restrict__ body_v,
    scalar_t* __restrict__ u0,
    scalar_t* __restrict__ v0,
    scalar_t* __restrict__ ch,
    scalar_t* __restrict__ cv,
    const scalar_t* __restrict__ sdf_cc,
    scalar_t* __restrict__ div_corr,
    const int32_t* __restrict__ rect,
    const scalar_t eps,
    const scalar_t rho_f,
    const scalar_t dt,
    const scalar_t inv_2h,
    const scalar_t eps_mw,
    const scalar_t inv_dx,
    const scalar_t inv_dy,
    const int Ngx, const int Ngy,
    const int mu0_proj,
    const int mw_on)
{
    // Static FULL-GRID 1-D launch (j fastest → coalesced).
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Ngx * Ngy;
    if (g >= total) return;

    const int j = g % Ngy;
    const int i = g / Ngy;

    const int di = i - rect[0];
    const int dj = j - rect[1];
    if (di >= 0 && di < rect[2] && dj >= 0 && dj < rect[3]) {
        bdim_one_axis_2d<scalar_t>(
            u_prime, sdf_u, body_u,
            eps, rho_f, dt, inv_2h,
            Ngx, Ngy, i, j, u0, ch, mu0_proj);
        bdim_one_axis_2d<scalar_t>(
            v_prime, sdf_v, body_v,
            eps, rho_f, dt, inv_2h,
            Ngx, Ngy, i, j, v0, cv, mu0_proj);
    } else {
        u0[g] = u_prime[g];
        v0[g] = v_prime[g];
    }

    // Maertens–Weymouth body-divergence correction (full grid).
    if (mw_on) {
        scalar_t db = scalar_t(0);
        if (i > 0 && i < Ngx - 1 && j > 0 && j < Ngy - 1) {
            const int stride_i = Ngy;
            db = ((body_u[g + stride_i] - body_u[g]) * inv_dx
                  + (body_v[g + 1] - body_v[g]) * inv_dy);
        }
        const scalar_t deps = max(scalar_t(-1), min(sdf_cc[g] / eps_mw, scalar_t(1)));
        const scalar_t pi = scalar_t(M_PI);
        const scalar_t mu0c = scalar_t(0.5) * (scalar_t(1) + deps + sin(pi * deps) / pi);
        div_corr[g] = (scalar_t(1) - mu0c) * db;
    }
}

// =====================================================================
//  C++ launcher wrappers (called by TORCH_LIBRARY_IMPL)
// =====================================================================

static void bdim_forcing_3d_cuda(
    const at::Tensor& u_prime,
    const at::Tensor& v_prime,
    const at::Tensor& w_prime,
    const at::Tensor& sdf_u,
    const at::Tensor& sdf_v,
    const at::Tensor& sdf_w,
    const at::Tensor& body_u,
    const at::Tensor& body_v,
    const at::Tensor& body_w,
    at::Tensor& u0, at::Tensor& v0, at::Tensor& w0,
    at::Tensor& ch, at::Tensor& cv, at::Tensor& cw,
    const at::Tensor& sdf_cc,
    at::Tensor& div_corr,
    const at::Tensor& rect,
    double eps, double rho_f, double dt,
    double h_grid,
    double eps_mw, double inv_dx, double inv_dy, double inv_dz,
    int64_t mu0_projection, int64_t mw_on)
{
    const int Ngx = (int)u0.size(0);
    const int Ngy = (int)u0.size(1);
    const int Ngz = (int)u0.size(2);
    const int total = Ngx * Ngy * Ngz;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(u0.scalar_type(), "bdim_forcing_3d_cuda", [&] {
        bdim_forcing_3d_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
                sdf_cc.data_ptr<scalar_t>(),
                div_corr.data_ptr<scalar_t>(),
                rect.data_ptr<int32_t>(),
                (scalar_t)eps, (scalar_t)rho_f, (scalar_t)dt,
                (scalar_t)(0.5 / h_grid),
                (scalar_t)eps_mw, (scalar_t)inv_dx, (scalar_t)inv_dy, (scalar_t)inv_dz,
                Ngx, Ngy, Ngz,
                (int)mu0_projection, (int)mw_on);
    });
}

static void bdim_forcing_2d_cuda(
    const at::Tensor& u_prime,
    const at::Tensor& v_prime,
    const at::Tensor& sdf_u,
    const at::Tensor& sdf_v,
    const at::Tensor& body_u,
    const at::Tensor& body_v,
    at::Tensor& u0, at::Tensor& v0,
    at::Tensor& ch, at::Tensor& cv,
    const at::Tensor& sdf_cc,
    at::Tensor& div_corr,
    const at::Tensor& rect,
    double eps, double rho_f, double dt,
    double h_grid,
    double eps_mw, double inv_dx, double inv_dy,
    int64_t mu0_projection, int64_t mw_on)
{
    const int Ngx = (int)u0.size(0);
    const int Ngy = (int)u0.size(1);
    const int total = Ngx * Ngy;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(u0.scalar_type(), "bdim_forcing_2d_cuda", [&] {
        bdim_forcing_2d_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                u_prime.data_ptr<scalar_t>(),
                v_prime.data_ptr<scalar_t>(),
                sdf_u.data_ptr<scalar_t>(),
                sdf_v.data_ptr<scalar_t>(),
                body_u.data_ptr<scalar_t>(),
                body_v.data_ptr<scalar_t>(),
                u0.data_ptr<scalar_t>(),
                v0.data_ptr<scalar_t>(),
                ch.data_ptr<scalar_t>(),
                cv.data_ptr<scalar_t>(),
                sdf_cc.data_ptr<scalar_t>(),
                div_corr.data_ptr<scalar_t>(),
                rect.data_ptr<int32_t>(),
                (scalar_t)eps, (scalar_t)rho_f, (scalar_t)dt,
                (scalar_t)(0.5 / h_grid),
                (scalar_t)eps_mw, (scalar_t)inv_dx, (scalar_t)inv_dy,
                Ngx, Ngy,
                (int)mu0_projection, (int)mw_on);
    });
}

// =====================================================================
//  Register with the CUDA backend
// =====================================================================
TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("bdim_forcing_3d", &bdim_forcing_3d_cuda);
    m.impl("bdim_forcing_2d", &bdim_forcing_2d_cuda);
}

}  // namespace lilytorch_kernels
