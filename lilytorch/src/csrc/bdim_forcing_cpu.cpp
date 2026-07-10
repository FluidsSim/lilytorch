// =====================================================================
//  bdim_forcing_{2d,3d} — CPU twins (at::parallel_for).
//
//  Faithful line-for-line port of the CUDA kernel in bdim_forcing.cu,
//  which is a faithful port of the Warp kernel in bdim.py.
// =====================================================================

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <torch/all.h>
#include <torch/library.h>

namespace lilytorch_kernels {

// =====================================================================
//  CPU mu0/mu1 helper (shared between 2-D and 3-D)
// =====================================================================
template <typename scalar_t>
static inline void mu0_mu1_cpu(scalar_t phi, scalar_t eps, scalar_t& mu0, scalar_t& mu1) {
    if (phi <= -eps) {
        mu0 = scalar_t(0); mu1 = scalar_t(0);
    } else if (phi >= eps) {
        mu0 = scalar_t(1); mu1 = scalar_t(0);
    } else {
        const scalar_t deps = phi / eps;
        const scalar_t pi   = scalar_t(M_PI);
        const scalar_t s    = std::sin(pi * deps);
        const scalar_t c    = std::cos(pi * deps);
        mu0 = scalar_t(0.5) * (scalar_t(1) + deps + s / pi);
        mu1 = eps * (
            scalar_t(0.25)
            - scalar_t(0.25) * deps * deps
            - (s * deps + (scalar_t(1) + c) / pi) / (scalar_t(2) * pi)
        );
    }
}

// =====================================================================
//  CPU bdim_one_axis_2d helper (same logic as the CUDA device function)
// =====================================================================
template <typename scalar_t>
static inline void bdim_one_axis_2d_cpu(
    const scalar_t* phi_prime,
    const scalar_t* sdf,
    const scalar_t* body,
    scalar_t eps, scalar_t rho_f, scalar_t dt, scalar_t inv_2h,
    int Ngx, int Ngy, int i, int j,
    scalar_t* phi_out, scalar_t* c_out, int mu0_proj)
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
    mu0_mu1_cpu(phi, eps, mu0, mu1);

    scalar_t nx = (sdf[g_ip] - sdf[g_im]) * inv_2h;
    scalar_t ny = (sdf[g_jp] - sdf[g_jm]) * inv_2h;
    const scalar_t nn = std::sqrt(nx*nx + ny*ny);
    if (nn > scalar_t(0)) {
        const scalar_t inv_nn = scalar_t(1) / nn;
        nx *= inv_nn;
        ny *= inv_nn;
    }

    const scalar_t b_c    = body[g];
    const scalar_t pp_c   = phi_prime[g];
    const scalar_t diff_c = pp_c - b_c;

    scalar_t ddx = scalar_t(0), ddy = scalar_t(0);
    if (i > 0 && i < Ngx - 1)
        ddx = ((phi_prime[g_ip] - body[g_ip]) - (phi_prime[g_im] - body[g_im])) * inv_2h;
    if (j > 0 && j < Ngy - 1)
        ddy = ((phi_prime[g_jp] - body[g_jp]) - (phi_prime[g_jm] - body[g_jm])) * inv_2h;
    const scalar_t nd = nx * ddx + ny * ddy;

    phi_out[g] = mu0 * diff_c + b_c + mu1 * nd;
    // 2-D: Poisson coefficient at the SAME full-grid index g.
    c_out[g] = (mu0_proj ? dt * mu0 : dt) / rho_f;
}

// =====================================================================
//  CPU bdim_one_axis_3d helper
// =====================================================================
template <typename scalar_t>
static inline void bdim_one_axis_3d_cpu(
    const scalar_t* phi_prime,
    const scalar_t* sdf,
    const scalar_t* body,
    scalar_t eps, scalar_t rho_f, scalar_t dt, scalar_t inv_2h,
    int Ngx, int Ngy, int Ngz, int i, int j, int k,
    scalar_t* phi_out, scalar_t* c_out,
    int c_stride_i, int c_stride_j,
    int c_hi_i, int c_hi_j, int c_hi_k,
    int mu0_proj)
{
    const int stride_i = Ngy * Ngz;
    const int stride_j = Ngz;
    const int g = i * stride_i + j * stride_j + k;

    const int im = (i > 0)       ? (i - 1) : 0;
    const int ip = (i < Ngx - 1) ? (i + 1) : Ngx - 1;
    const int jm = (j > 0)       ? (j - 1) : 0;
    const int jp = (j < Ngy - 1) ? (j + 1) : Ngy - 1;
    const int km = (k > 0)       ? (k - 1) : 0;
    const int kp = (k < Ngz - 1) ? (k + 1) : Ngz - 1;

    const int g_im = im * stride_i + j  * stride_j + k;
    const int g_ip = ip * stride_i + j  * stride_j + k;
    const int g_jm = i  * stride_i + jm * stride_j + k;
    const int g_jp = i  * stride_i + jp * stride_j + k;
    const int g_km = i  * stride_i + j  * stride_j + km;
    const int g_kp = i  * stride_i + j  * stride_j + kp;

    const scalar_t phi = sdf[g];
    scalar_t mu0, mu1;
    mu0_mu1_cpu(phi, eps, mu0, mu1);

    scalar_t nx = (sdf[g_ip] - sdf[g_im]) * inv_2h;
    scalar_t ny = (sdf[g_jp] - sdf[g_jm]) * inv_2h;
    scalar_t nz = (sdf[g_kp] - sdf[g_km]) * inv_2h;
    const scalar_t nn = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (nn > scalar_t(0)) {
        const scalar_t inv_nn = scalar_t(1) / nn;
        nx *= inv_nn;
        ny *= inv_nn;
        nz *= inv_nn;
    }

    const scalar_t b_c    = body[g];
    const scalar_t pp_c   = phi_prime[g];
    const scalar_t diff_c = pp_c - b_c;

    scalar_t ddx = scalar_t(0), ddy = scalar_t(0), ddz = scalar_t(0);
    if (i > 0 && i < Ngx - 1)
        ddx = ((phi_prime[g_ip] - body[g_ip]) - (phi_prime[g_im] - body[g_im])) * inv_2h;
    if (j > 0 && j < Ngy - 1)
        ddy = ((phi_prime[g_jp] - body[g_jp]) - (phi_prime[g_jm] - body[g_jm])) * inv_2h;
    if (k > 0 && k < Ngz - 1)
        ddz = ((phi_prime[g_kp] - body[g_kp]) - (phi_prime[g_km] - body[g_km])) * inv_2h;
    const scalar_t nd = nx * ddx + ny * ddy + nz * ddz;

    phi_out[g] = mu0 * diff_c + b_c + mu1 * nd;
    if (i >= 1 && j >= 1 && k >= 1 && i <= c_hi_i && j <= c_hi_j && k <= c_hi_k) {
        c_out[(i - 1) * c_stride_i + (j - 1) * c_stride_j + (k - 1)] =
            (mu0_proj ? dt * mu0 : dt) / rho_f;
    }
}

// =====================================================================
//  CPU launcher: bdim_forcing_3d
// =====================================================================
static void bdim_forcing_3d_cpu(
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

    AT_DISPATCH_FLOATING_TYPES(u0.scalar_type(), "bdim_forcing_3d_cpu", [&] {
        const scalar_t eps_s   = (scalar_t)eps;
        const scalar_t rho_s   = (scalar_t)rho_f;
        const scalar_t dt_s    = (scalar_t)dt;
        const scalar_t inv_2h  = (scalar_t)(0.5 / h_grid);
        const scalar_t eps_mw_s = (scalar_t)eps_mw;
        const scalar_t idx_s   = (scalar_t)inv_dx;
        const scalar_t idy_s   = (scalar_t)inv_dy;
        const scalar_t idz_s   = (scalar_t)inv_dz;

        at::parallel_for(0, total, 1024, [&](int64_t begin, int64_t end) {
            const scalar_t* u_p  = u_prime.data_ptr<scalar_t>();
            const scalar_t* v_p  = v_prime.data_ptr<scalar_t>();
            const scalar_t* w_p  = w_prime.data_ptr<scalar_t>();
            const scalar_t* su   = sdf_u.data_ptr<scalar_t>();
            const scalar_t* sv   = sdf_v.data_ptr<scalar_t>();
            const scalar_t* sw   = sdf_w.data_ptr<scalar_t>();
            const scalar_t* bu   = body_u.data_ptr<scalar_t>();
            const scalar_t* bv   = body_v.data_ptr<scalar_t>();
            const scalar_t* bw   = body_w.data_ptr<scalar_t>();
            scalar_t* u0p        = u0.data_ptr<scalar_t>();
            scalar_t* v0p        = v0.data_ptr<scalar_t>();
            scalar_t* w0p        = w0.data_ptr<scalar_t>();
            scalar_t* chp        = ch.data_ptr<scalar_t>();
            scalar_t* cvp        = cv.data_ptr<scalar_t>();
            scalar_t* cwp        = cw.data_ptr<scalar_t>();
            const scalar_t* scc  = sdf_cc.data_ptr<scalar_t>();
            scalar_t* dcp        = div_corr.data_ptr<scalar_t>();
            const int32_t* rp    = rect.data_ptr<int32_t>();

            const int stride_i = Ngy * Ngz;
            const int stride_j = Ngz;

            for (int64_t g = begin; g < end; ++g) {
                const int k = g % Ngz;
                const int rem = (int)(g / Ngz);
                const int j = rem % Ngy;
                const int i = rem / Ngy;

                const int di = i - rp[0];
                const int dj = j - rp[1];
                const int dk = k - rp[2];
                if (di >= 0 && di < rp[3] && dj >= 0 && dj < rp[4] && dk >= 0 && dk < rp[5]) {
                    bdim_one_axis_3d_cpu<scalar_t>(
                        u_p, su, bu, eps_s, rho_s, dt_s, inv_2h,
                        Ngx, Ngy, Ngz, i, j, k, u0p, chp,
                        (Ngy - 2) * (Ngz - 2), (Ngz - 2),
                        Ngx - 1, Ngy - 2, Ngz - 2, (int)mu0_projection);
                    bdim_one_axis_3d_cpu<scalar_t>(
                        v_p, sv, bv, eps_s, rho_s, dt_s, inv_2h,
                        Ngx, Ngy, Ngz, i, j, k, v0p, cvp,
                        (Ngy - 1) * (Ngz - 2), (Ngz - 2),
                        Ngx - 2, Ngy - 1, Ngz - 2, (int)mu0_projection);
                    bdim_one_axis_3d_cpu<scalar_t>(
                        w_p, sw, bw, eps_s, rho_s, dt_s, inv_2h,
                        Ngx, Ngy, Ngz, i, j, k, w0p, cwp,
                        (Ngy - 2) * (Ngz - 1), (Ngz - 1),
                        Ngx - 2, Ngy - 2, Ngz - 1, (int)mu0_projection);
                } else {
                    u0p[g] = u_p[g];
                    v0p[g] = v_p[g];
                    w0p[g] = w_p[g];
                }

                // MW correction
                if (mw_on) {
                    scalar_t db = scalar_t(0);
                    if (i > 0 && i < Ngx - 1 && j > 0 && j < Ngy - 1 && k > 0 && k < Ngz - 1) {
                        db = ((bu[g + stride_i] - bu[g]) * idx_s
                              + (bv[g + stride_j] - bv[g]) * idy_s
                              + (bw[g + 1] - bw[g]) * idz_s);
                    }
                    const scalar_t deps = std::max(scalar_t(-1),
                        std::min(scc[g] / eps_mw_s, scalar_t(1)));
                    const scalar_t pi = scalar_t(M_PI);
                    const scalar_t mu0c = scalar_t(0.5) * (scalar_t(1) + deps
                        + std::sin(pi * deps) / pi);
                    dcp[g] = (scalar_t(1) - mu0c) * db;
                }
            }
        });
    });
}

// =====================================================================
//  CPU launcher: bdim_forcing_2d
// =====================================================================
static void bdim_forcing_2d_cpu(
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

    AT_DISPATCH_FLOATING_TYPES(u0.scalar_type(), "bdim_forcing_2d_cpu", [&] {
        const scalar_t eps_s   = (scalar_t)eps;
        const scalar_t rho_s   = (scalar_t)rho_f;
        const scalar_t dt_s    = (scalar_t)dt;
        const scalar_t inv_2h  = (scalar_t)(0.5 / h_grid);
        const scalar_t eps_mw_s = (scalar_t)eps_mw;
        const scalar_t idx_s   = (scalar_t)inv_dx;
        const scalar_t idy_s   = (scalar_t)inv_dy;

        at::parallel_for(0, total, 1024, [&](int64_t begin, int64_t end) {
            const scalar_t* u_p  = u_prime.data_ptr<scalar_t>();
            const scalar_t* v_p  = v_prime.data_ptr<scalar_t>();
            const scalar_t* su   = sdf_u.data_ptr<scalar_t>();
            const scalar_t* sv   = sdf_v.data_ptr<scalar_t>();
            const scalar_t* bu   = body_u.data_ptr<scalar_t>();
            const scalar_t* bv   = body_v.data_ptr<scalar_t>();
            scalar_t* u0p        = u0.data_ptr<scalar_t>();
            scalar_t* v0p        = v0.data_ptr<scalar_t>();
            scalar_t* chp        = ch.data_ptr<scalar_t>();
            scalar_t* cvp        = cv.data_ptr<scalar_t>();
            const scalar_t* scc  = sdf_cc.data_ptr<scalar_t>();
            scalar_t* dcp        = div_corr.data_ptr<scalar_t>();
            const int32_t* rp    = rect.data_ptr<int32_t>();

            const int stride_i = Ngy;

            for (int64_t g = begin; g < end; ++g) {
                const int j = g % Ngy;
                const int i = (int)(g / Ngy);

                const int di = i - rp[0];
                const int dj = j - rp[1];
                if (di >= 0 && di < rp[2] && dj >= 0 && dj < rp[3]) {
                    bdim_one_axis_2d_cpu<scalar_t>(
                        u_p, su, bu, eps_s, rho_s, dt_s, inv_2h,
                        Ngx, Ngy, i, j, u0p, chp, (int)mu0_projection);
                    bdim_one_axis_2d_cpu<scalar_t>(
                        v_p, sv, bv, eps_s, rho_s, dt_s, inv_2h,
                        Ngx, Ngy, i, j, v0p, cvp, (int)mu0_projection);
                } else {
                    u0p[g] = u_p[g];
                    v0p[g] = v_p[g];
                }

                // MW correction
                if (mw_on) {
                    scalar_t db = scalar_t(0);
                    if (i > 0 && i < Ngx - 1 && j > 0 && j < Ngy - 1) {
                        db = ((bu[g + stride_i] - bu[g]) * idx_s
                              + (bv[g + 1] - bv[g]) * idy_s);
                    }
                    const scalar_t deps = std::max(scalar_t(-1),
                        std::min(scc[g] / eps_mw_s, scalar_t(1)));
                    const scalar_t pi = scalar_t(M_PI);
                    const scalar_t mu0c = scalar_t(0.5) * (scalar_t(1) + deps
                        + std::sin(pi * deps) / pi);
                    dcp[g] = (scalar_t(1) - mu0c) * db;
                }
            }
        });
    });
}

// =====================================================================
//  Register with the CPU backend
// =====================================================================
TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("bdim_forcing_3d", &bdim_forcing_3d_cpu);
    m.impl("bdim_forcing_2d", &bdim_forcing_2d_cpu);
}

}  // namespace lilytorch_kernels
