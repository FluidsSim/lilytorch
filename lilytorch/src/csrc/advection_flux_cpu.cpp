// =====================================================================
//  advection_flux_cpu.cpp — CPU twin of advect_flux_accumulate
//
//  Mirrors csrc/cuda/advection_flux.cu::advect_flux_accumulate_kernel
//  cell-for-cell (same shared scheme functions from
//  advection_schemes.h, same operation order), parallelised over
//  interior cells with at::parallel_for.
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/Parallel.h>

#include "advection_schemes.h"

namespace lilytorch_kernels {

using schemes::apply_scheme;

// Both face fluxes along one direction — identical to the CUDA
// face_flux_diff device helper.
template <typename scalar_t, int scheme_id>
static inline scalar_t face_flux_diff_cpu(
    const scalar_t* phi, int64_t c, int64_t s_d,
    scalar_t fv_L, scalar_t fv_R, int g, int N, scalar_t C)
{
    scalar_t pc = phi[c - s_d];
    scalar_t pd = phi[c];
    scalar_t F_L;
    if (fv_L > scalar_t(0)) {
        if (g == 1) {
            F_L = fv_L * scalar_t(0.5) * (pc + pd);
        } else {
            F_L = fv_L * apply_scheme<scalar_t, scheme_id>(
                      phi[c - 2 * s_d], pc, pd, C);
        }
    } else {
        F_L = fv_L * apply_scheme<scalar_t, scheme_id>(
                  phi[c + s_d], pd, pc, C);
    }

    scalar_t pc2 = phi[c];
    scalar_t pd2 = phi[c + s_d];
    scalar_t F_R;
    if (fv_R > scalar_t(0)) {
        F_R = fv_R * apply_scheme<scalar_t, scheme_id>(
                  phi[c - s_d], pc2, pd2, C);
    } else {
        if (g == N - 2) {
            F_R = fv_R * scalar_t(0.5) * (pc2 + pd2);
        } else {
            F_R = fv_R * apply_scheme<scalar_t, scheme_id>(
                      phi[c + 2 * s_d], pd2, pc2, C);
        }
    }
    return F_L - F_R;
}

template <typename scalar_t, int scheme_id>
static void advect_flux_accumulate_cpu_impl(
    const scalar_t* phi_src,
    scalar_t*       dst,
    const scalar_t* u_ptr,
    const scalar_t* v_ptr,
    const scalar_t* w_ptr,
    int comp_i,
    int Nx, int Ny, int Nz,
    int Nix, int Niy, int Niz,
    int64_t s_x, int64_t s_y, int64_t s_z,
    scalar_t dt_dh_0, scalar_t dt_dh_1, scalar_t dt_dh_2,
    scalar_t C)
{
    const int64_t total = (int64_t)Nix * Niy * Niz;
    const int64_t s_ci = (comp_i == 0) ? s_x : (comp_i == 1) ? s_y : s_z;
    const scalar_t half = scalar_t(0.5);

    at::parallel_for(0, total, 1024, [&](int64_t start, int64_t end) {
        for (int64_t tid = start; tid < end; ++tid) {
            int iz  = (int)(tid % Niz);
            int64_t ixy = tid / Niz;
            int iy  = (int)(ixy % Niy);
            int ix  = (int)(ixy / Niy);

            int gx = ix + 1, gy = iy + 1, gz = iz + 1;
            int64_t c = (int64_t)gx * s_x + (int64_t)gy * s_y
                      + (int64_t)gz * s_z;

            scalar_t acc = scalar_t(0);

            // ---- direction 0 (x) ----
            {
                scalar_t fv_L, fv_R;
                if (comp_i == 0) {
                    fv_L = half * (u_ptr[c - s_x] + u_ptr[c]);
                    fv_R = half * (u_ptr[c] + u_ptr[c + s_x]);
                } else {
                    fv_L = half * (u_ptr[c - s_ci] + u_ptr[c]);
                    fv_R = half * (u_ptr[c + s_x - s_ci] + u_ptr[c + s_x]);
                }
                acc += dt_dh_0 * face_flux_diff_cpu<scalar_t, scheme_id>(
                           phi_src, c, s_x, fv_L, fv_R, gx, Nx, C);
            }

            // ---- direction 1 (y) ----
            {
                scalar_t fv_L, fv_R;
                if (comp_i == 1) {
                    fv_L = half * (v_ptr[c - s_y] + v_ptr[c]);
                    fv_R = half * (v_ptr[c] + v_ptr[c + s_y]);
                } else {
                    fv_L = half * (v_ptr[c - s_ci] + v_ptr[c]);
                    fv_R = half * (v_ptr[c + s_y - s_ci] + v_ptr[c + s_y]);
                }
                acc += dt_dh_1 * face_flux_diff_cpu<scalar_t, scheme_id>(
                           phi_src, c, s_y, fv_L, fv_R, gy, Ny, C);
            }

            // ---- direction 2 (z) — only if 3-D ----
            if (Nz > 1) {
                scalar_t fv_L, fv_R;
                if (comp_i == 2) {
                    fv_L = half * (w_ptr[c - s_z] + w_ptr[c]);
                    fv_R = half * (w_ptr[c] + w_ptr[c + s_z]);
                } else {
                    fv_L = half * (w_ptr[c - s_ci] + w_ptr[c]);
                    fv_R = half * (w_ptr[c + s_z - s_ci] + w_ptr[c + s_z]);
                }
                acc += dt_dh_2 * face_flux_diff_cpu<scalar_t, scheme_id>(
                           phi_src, c, s_z, fv_L, fv_R, gz, Nz, C);
            }

            dst[c] += acc;
        }
    });
}

static void advect_flux_accumulate_cpu(
    const at::Tensor& phi_src,
    at::Tensor&       dst,
    const at::Tensor& u,
    const at::Tensor& v,
    const at::Tensor& w,
    int64_t           comp_i,
    double            dt_dh0,
    double            dt_dh1,
    double            dt_dh2,
    double            C_courant,
    int64_t           scheme_id
) {
    TORCH_CHECK(!phi_src.device().is_cuda(),
                "advect_flux_accumulate CPU: expected CPU tensors");
    int ndim = (int)phi_src.dim();
    TORCH_CHECK(ndim == 2 || ndim == 3,
                "advect_flux_accumulate: expected 2-D or 3-D, got ",
                ndim, "-D");
    TORCH_CHECK(comp_i >= 0 && comp_i < ndim,
                "advect_flux_accumulate: comp_i=", comp_i,
                " out of range for ndim=", ndim);
    const at::Tensor* like[3] = {&dst, &u, &v};
    for (const at::Tensor* t : like) {
        TORCH_CHECK(t->sizes() == phi_src.sizes()
                        && t->strides() == phi_src.strides(),
                    "advect_flux_accumulate: all tensors must share "
                    "phi_src's shape and strides");
    }
    if (ndim == 3) {
        TORCH_CHECK(w.sizes() == phi_src.sizes()
                        && w.strides() == phi_src.strides(),
                    "advect_flux_accumulate: w must share phi_src's "
                    "shape and strides");
    }

    int Nx = (int)phi_src.size(0);
    int Ny = (int)phi_src.size(1);
    int Nz = (ndim == 3) ? (int)phi_src.size(2) : 1;
    int Nix = Nx - 2, Niy = Ny - 2;
    int Niz = (ndim == 3) ? Nz - 2 : 1;
    if ((int64_t)Nix * Niy * Niz <= 0) return;

    int64_t s_x = phi_src.stride(0);
    int64_t s_y = phi_src.stride(1);
    int64_t s_z = (ndim == 3) ? phi_src.stride(2) : 0;

#define LAUNCH_ACC_CPU(SID)                                                     \
    AT_DISPATCH_FLOATING_TYPES(phi_src.scalar_type(),                           \
                               "advect_flux_accumulate_cpu", [&] {              \
        advect_flux_accumulate_cpu_impl<scalar_t, SID>(                         \
            phi_src.data_ptr<scalar_t>(),                                       \
            dst.data_ptr<scalar_t>(),                                           \
            u.data_ptr<scalar_t>(),                                             \
            v.data_ptr<scalar_t>(),                                             \
            (ndim == 3 ? w.data_ptr<scalar_t>() : u.data_ptr<scalar_t>()),      \
            (int)comp_i,                                                        \
            Nx, Ny, Nz, Nix, Niy, Niz,                                         \
            s_x, s_y, s_z,                                                      \
            static_cast<scalar_t>(dt_dh0),                                      \
            static_cast<scalar_t>(dt_dh1),                                      \
            static_cast<scalar_t>(dt_dh2),                                      \
            static_cast<scalar_t>(C_courant));                                  \
    });

    switch ((int)scheme_id) {
        case 0: LAUNCH_ACC_CPU(0); break;
        case 1: LAUNCH_ACC_CPU(1); break;
        case 2: LAUNCH_ACC_CPU(2); break;
        case 3: LAUNCH_ACC_CPU(3); break;
        case 4: LAUNCH_ACC_CPU(4); break;
        default:
            TORCH_CHECK(false,
                        "advect_flux_accumulate: unknown scheme_id=",
                        scheme_id);
    }
#undef LAUNCH_ACC_CPU
}

// =====================================================================
//  advect_flux_add — CPU twin
//
//  The single-direction op (one (velocity component, spatial direction)
//  pair at a time), superseded in production by advect_flux_accumulate but
//  still the op the CPU scheme-regression anchor pins.
//
//  Mirrors cuda/advection_flux.cu::advect_flux_add_kernel by reusing the
//  SAME face_flux_diff_cpu helper as the accumulate twin above — the
//  previous CPU twin was a hand-rolled second copy of the stencil (in
//  multigrid_cpu.cpp, of all places) and it disagreed with CUDA.
//
//  Index mapping: for interior rhs cell i_fd, the left face is p-index i_fd
//  and the right face p-index i_fd+1, so face_flux_diff_cpu is called with
//  the cell offset c = (i_fd + 1) * s_fd and g = i_fd + 1, N = Nfd — which
//  puts its two CDS-fallback branches (g == 1 and g == N-2) exactly on the
//  lo/hi boundary faces the CUDA kernel special-cases.
// =====================================================================

template <typename scalar_t, int scheme_id>
static void advect_flux_add_cpu_impl(
    const scalar_t* p_ptr, const scalar_t* fv_ptr, scalar_t* rhs_ptr,
    int Nfd, int Nt1, int Nt2,
    int64_t p_s_fd,   int64_t p_s_t1,   int64_t p_s_t2,
    int64_t fv_s_fd,  int64_t fv_s_t1,  int64_t fv_s_t2,
    int64_t rhs_s_fd, int64_t rhs_s_t1, int64_t rhs_s_t2,
    scalar_t dt_dh, scalar_t C)
{
    const int Ni_fd = Nfd - 2;
    const int64_t total = (int64_t)Ni_fd * Nt1 * Nt2;

    at::parallel_for(0, total, 1024, [&](int64_t start, int64_t end) {
        for (int64_t idx = start; idx < end; ++idx) {
            const int i_t2 = (int)(idx % Nt2);
            const int64_t r1 = idx / Nt2;
            const int i_t1 = (int)(r1 % Nt1);
            const int i_fd = (int)(r1 / Nt1);

            const int64_t tp  = (int64_t)i_t1 * p_s_t1  + (int64_t)i_t2 * p_s_t2;
            const int64_t tfv = (int64_t)i_t1 * fv_s_t1 + (int64_t)i_t2 * fv_s_t2;

            const scalar_t fv_L = fv_ptr[tfv + (int64_t)i_fd       * fv_s_fd];
            const scalar_t fv_R = fv_ptr[tfv + (int64_t)(i_fd + 1) * fv_s_fd];

            const int64_t c = tp + (int64_t)(i_fd + 1) * p_s_fd;
            const scalar_t d = face_flux_diff_cpu<scalar_t, scheme_id>(
                p_ptr, c, p_s_fd, fv_L, fv_R, i_fd + 1, Nfd, C);

            const int64_t ridx = (int64_t)i_fd * rhs_s_fd
                               + (int64_t)i_t1 * rhs_s_t1
                               + (int64_t)i_t2 * rhs_s_t2;
            rhs_ptr[ridx] += dt_dh * d;
        }
    });
}

static void advect_flux_add_cpu(
        const at::Tensor& fv, const at::Tensor& p, at::Tensor& rhs,
        double dt_dh, double C_courant, int64_t scheme_id, int64_t face_dim)
{
    TORCH_CHECK(!p.device().is_cuda(), "advect_flux_add CPU: expected CPU tensors");
    const int ndim = (int)p.dim();
    TORCH_CHECK(ndim == 2 || ndim == 3, "advect_flux_add: p must be 2-D or 3-D");
    TORCH_CHECK(face_dim >= 0 && face_dim < ndim,
                "advect_flux_add: face_dim out of range");

    const int Nfd = (int)p.size((int)face_dim);
    if (Nfd - 2 <= 0) return;

    // Transverse dims in ORIGINAL grid-dim order, skipping face_dim — the same
    // convention the CUDA launcher and the rhs strides use.
    int t_dims[2] = {-1, -1};
    int nt = 0;
    for (int d = 0; d < ndim; ++d)
        if (d != (int)face_dim) t_dims[nt++] = d;

    int Nt1 = 1, Nt2 = 1;
    int64_t p_s_t1 = 0, p_s_t2 = 0, fv_s_t1 = 0, fv_s_t2 = 0;
    int64_t rhs_s_t1 = 0, rhs_s_t2 = 0;

    if (nt > 0) {
        Nt1      = (int)rhs.size(t_dims[0]);
        p_s_t1   = p.stride(t_dims[0]);
        fv_s_t1  = fv.stride(t_dims[0]);
        rhs_s_t1 = rhs.stride(t_dims[0]);
    }
    if (nt > 1) {
        Nt2      = (int)rhs.size(t_dims[1]);
        p_s_t2   = p.stride(t_dims[1]);
        fv_s_t2  = fv.stride(t_dims[1]);
        rhs_s_t2 = rhs.stride(t_dims[1]);
    }
    if (Nt1 <= 0 || Nt2 <= 0) return;

#define LAUNCH_ADD_CPU(SID)                                                     \
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "advect_flux_add_cpu", [&] {    \
        advect_flux_add_cpu_impl<scalar_t, SID>(                                \
            p.data_ptr<scalar_t>(),                                             \
            fv.data_ptr<scalar_t>(),                                            \
            rhs.data_ptr<scalar_t>(),                                           \
            Nfd, Nt1, Nt2,                                                      \
            p.stride((int)face_dim),   p_s_t1,   p_s_t2,                        \
            fv.stride((int)face_dim),  fv_s_t1,  fv_s_t2,                       \
            rhs.stride((int)face_dim), rhs_s_t1, rhs_s_t2,                      \
            static_cast<scalar_t>(dt_dh),                                       \
            static_cast<scalar_t>(C_courant));                                  \
    });

    switch ((int)scheme_id) {
        case 0: LAUNCH_ADD_CPU(0); break;
        case 1: LAUNCH_ADD_CPU(1); break;
        case 2: LAUNCH_ADD_CPU(2); break;
        case 3: LAUNCH_ADD_CPU(3); break;
        case 4: LAUNCH_ADD_CPU(4); break;
        default:
            TORCH_CHECK(false, "advect_flux_add: unknown scheme_id=", scheme_id);
    }
#undef LAUNCH_ADD_CPU
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("advect_flux_add", &advect_flux_add_cpu);
    m.impl("advect_flux_accumulate", &advect_flux_accumulate_cpu);
}

}  // namespace lilytorch_kernels
