// =====================================================================
//  advection_flux.cu — fused per-cell flux accumulate kernel (13c)
//
//  The sole production convective-flux kernel: computes face velocities
//  from the original staggered fields on the fly and accumulates
//  dst[cell] += Σ_d dt_dh_d*(F_L - F_R) over the interior in one launch
//  per velocity component.
//
//  Op: ``lilytorch_kernels::advect_flux_accumulate``
//  Signature (see ops.cpp):
//      Tensor phi_src, Tensor(a!) dst,
//      Tensor u, Tensor v, Tensor w,
//      int comp_i,
//      float dt_dh0, float dt_dh1, float dt_dh2,
//      float C, int scheme_id
//
//  Scheme IDs (must match _CUDA_SCHEME_IDS in advection.py):
//      0 = QUICK
//      1 = ABDQUICKEST
//      2 = vanLeer
//      3 = CDS
//      4 = CUBISTA
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../common/advection_schemes.h"

namespace lilytorch_kernels {

// ── helpers ───────────────────────────────────────────────────────────
static __host__ __device__ __forceinline__ int cdiv_af(int a, int b) {
    return (a + b - 1) / b;
}

// The five convective schemes live in ../advection_schemes.h (shared
// with the CPU twin in advection_flux_cpu.cpp).
using schemes::apply_scheme;

// =====================================================================
//  advect_flux_accumulate — fused per-cell flux kernel (13c)
//
//  Fused flux accumulate: one launch per velocity
//  component.  Each thread owns one interior cell, computes face
//  velocities from the original staggered fields on the fly, evaluates
//  both face fluxes per spatial direction, and accumulates
//      dst[cell] += Σ_d dt_dh_d * (F_L - F_R)
//  directly into the FULL-GRID output (which already holds the
//  velocity + diffusion increment) — no interior rhs buffer, no
//  separate zero/accumulate passes.
//
//  All tensors share one shape/stride set (asserted in the wrapper).
//  2-D runs with Nz = 1, s_z = 0 and w = u (dummy, never read).
// =====================================================================

// Both face fluxes along one direction: stride s_d, boundary index g in
// [1, N-2].
template <typename scalar_t, int scheme_id>
__device__ __forceinline__ scalar_t face_flux_diff(
    const scalar_t* __restrict__ phi, int64_t c, int64_t s_d,
    scalar_t fv_L, scalar_t fv_R, int g, int N, scalar_t C)
{
    // ---- left face (face g-1) ----
    scalar_t pc = phi[c - s_d];
    scalar_t pd = phi[c];
    scalar_t F_L;
    if (fv_L > scalar_t(0)) {
        if (g == 1) {
            // lo boundary: no upstream → CDS fallback
            F_L = fv_L * scalar_t(0.5) * (pc + pd);
        } else {
            F_L = fv_L * apply_scheme<scalar_t, scheme_id>(
                      phi[c - 2 * s_d], pc, pd, C);
        }
    } else {
        // negative flow; g is never the hi boundary for the left face
        F_L = fv_L * apply_scheme<scalar_t, scheme_id>(
                  phi[c + s_d], pd, pc, C);
    }

    // ---- right face (face g) ----
    scalar_t pc2 = phi[c];
    scalar_t pd2 = phi[c + s_d];
    scalar_t F_R;
    if (fv_R > scalar_t(0)) {
        // positive flow; g is never the lo boundary for the right face
        F_R = fv_R * apply_scheme<scalar_t, scheme_id>(
                  phi[c - s_d], pc2, pd2, C);
    } else {
        if (g == N - 2) {
            // hi boundary: no downstream → CDS fallback
            F_R = fv_R * scalar_t(0.5) * (pc2 + pd2);
        } else {
            F_R = fv_R * apply_scheme<scalar_t, scheme_id>(
                      phi[c + 2 * s_d], pd2, pc2, C);
        }
    }
    return F_L - F_R;
}

template <typename scalar_t, int scheme_id>
__global__ void advect_flux_accumulate_kernel(
    const scalar_t* __restrict__ phi_src,   // full-grid copy of vel[comp_i]
    scalar_t*       __restrict__ dst,       // full-grid output (accumulated)
    const scalar_t* __restrict__ u_ptr,
    const scalar_t* __restrict__ v_ptr,
    const scalar_t* __restrict__ w_ptr,     // dummy in 2-D (never read)
    int comp_i,
    int Nx, int Ny, int Nz,                 // full dims (Nz = 1 → 2-D)
    int Nix, int Niy, int Niz,             // interior dims
    int64_t s_x, int64_t s_y, int64_t s_z, // strides (shared by all arrays)
    scalar_t dt_dh_0, scalar_t dt_dh_1, scalar_t dt_dh_2,
    scalar_t C)
{
    int tid = (int)(blockIdx.x * blockDim.x) + (int)threadIdx.x;
    int total = Nix * Niy * Niz;
    if (tid >= total) return;

    // Decode flat tid → interior (ix, iy, iz); z fastest-varying
    // (matches memory layout: stride-1 dim innermost → coalesced).
    int iz  = tid % Niz;
    int ixy = tid / Niz;
    int iy  = ixy % Niy;
    int ix  = ixy / Niy;

    int gx = ix + 1, gy = iy + 1, gz = iz + 1;
    int64_t c = (int64_t)gx * s_x + (int64_t)gy * s_y + (int64_t)gz * s_z;

    // Stride of the component we are solving for (cross-advection face
    // velocities are averaged along comp_i's axis: the "+1 offset").
    int64_t s_ci = (comp_i == 0) ? s_x : (comp_i == 1) ? s_y : s_z;

    const scalar_t half = scalar_t(0.5);
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
        acc += dt_dh_0 * face_flux_diff<scalar_t, scheme_id>(
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
        acc += dt_dh_1 * face_flux_diff<scalar_t, scheme_id>(
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
        acc += dt_dh_2 * face_flux_diff<scalar_t, scheme_id>(
                   phi_src, c, s_z, fv_L, fv_R, gz, Nz, C);
    }

    dst[c] += acc;
}

// ── C++ wrapper ───────────────────────────────────────────────────────

static void advect_flux_accumulate_cuda(
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
    int ndim = (int)phi_src.dim();
    TORCH_CHECK(ndim == 2 || ndim == 3,
                "advect_flux_accumulate: expected 2-D or 3-D, got ", ndim, "-D");
    TORCH_CHECK(comp_i >= 0 && comp_i < ndim,
                "advect_flux_accumulate: comp_i=", comp_i,
                " out of range for ndim=", ndim);
    // All arrays must share one shape/stride set (kernel uses one set).
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
    int64_t total = (int64_t)Nix * Niy * Niz;
    if (total <= 0) return;

    int64_t s_x = phi_src.stride(0);
    int64_t s_y = phi_src.stride(1);
    int64_t s_z = (ndim == 3) ? phi_src.stride(2) : 0;

    auto stream = at::cuda::getCurrentCUDAStream();
    const int BLK = 256;
    int grd = (int)((total + BLK - 1) / BLK);

#define LAUNCH_ACC(SID)                                                          \
    AT_DISPATCH_FLOATING_TYPES(phi_src.scalar_type(),                            \
                               "advect_flux_accumulate", [&] {                   \
        advect_flux_accumulate_kernel<scalar_t, SID><<<grd, BLK, 0, stream>>>(   \
            phi_src.data_ptr<scalar_t>(),                                        \
            dst.data_ptr<scalar_t>(),                                            \
            u.data_ptr<scalar_t>(),                                              \
            v.data_ptr<scalar_t>(),                                              \
            (ndim == 3 ? w.data_ptr<scalar_t>() : u.data_ptr<scalar_t>()),       \
            (int)comp_i,                                                         \
            Nx, Ny, Nz, Nix, Niy, Niz,                                          \
            s_x, s_y, s_z,                                                       \
            static_cast<scalar_t>(dt_dh0),                                       \
            static_cast<scalar_t>(dt_dh1),                                       \
            static_cast<scalar_t>(dt_dh2),                                       \
            static_cast<scalar_t>(C_courant)                                     \
        );                                                                       \
    });

    switch ((int)scheme_id) {
        case 0: LAUNCH_ACC(0); break;
        case 1: LAUNCH_ACC(1); break;
        case 2: LAUNCH_ACC(2); break;
        case 3: LAUNCH_ACC(3); break;
        case 4: LAUNCH_ACC(4); break;
        default:
            TORCH_CHECK(false,
                        "advect_flux_accumulate: unknown scheme_id=", scheme_id);
    }
#undef LAUNCH_ACC
}

// ── dispatch registration ─────────────────────────────────────────────
TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("advect_flux_accumulate", &advect_flux_accumulate_cuda);
}

}  // namespace lilytorch_kernels
