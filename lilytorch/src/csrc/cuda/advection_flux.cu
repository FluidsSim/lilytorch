// =====================================================================
//  advection_flux.cu — fused CUDA kernel for the T2a advection flux
//
//  Replaces the Python ``_flux`` → ``F[:-1] - F[1:]`` → ``rhs.add_``
//  chain in ``AdvDiffSolver._solve_convective`` with a single kernel
//  launch per (velocity component i, spatial direction d) pair.
//
//  Key savings vs the ATen path:
//    * No intermediate F tensor allocated (~full-grid per call, up to
//      9× per step in 3-D).
//    * No B1/B2/flux_in/flux_lo/flux_hi temporaries.
//    * Stencil values read from L1/L2 rather than global; QUICK needs
//      at most 4 reads along the face dimension per thread.
//
//  Op: ``lilytorch_kernels::advect_flux_add``
//  Signature (see ops.cpp):
//      Tensor fv,  Tensor p,  Tensor(a!) rhs,
//      float dt_dh,  float C,  int scheme_id,  int face_dim
//
//  Scheme IDs (must match _CUDA_SCHEME_IDS in advection.py):
//      0 = QUICK
//      1 = ABDQUICKEST
//      2 = vanLeer
//      3 = CDS
//      4 = CUBISTA
//
//  Works for 2-D and 3-D tensors.  Handles non-contiguous fv / p
//  tensors (which arise from PyTorch slice views of vel arrays) via
//  explicit stride parameters extracted in the C++ wrapper.
//
//  Thread layout  (rhs-optimal coalescing):
//      threadIdx.x → i_t2  (innermost dim of rhs, stride-1 writes)
//      threadIdx.y → i_t1  (middle dim of rhs)
//      blockIdx.z  → i_fd  (face-dim interior index)
//
//  Memory layout convention (from _field_for_flux / _face_vel):
//      p  : shape (Nfd, Nt1[, Nt2])  — full on face_dim, interior on rest
//      fv : shape (Nfd-1, Nt1[, Nt2])
//      rhs: shape (Nfd-2, Nt1[, Nt2])  — always C-contiguous (fresh tensor)
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../advection_schemes.h"

namespace lilytorch_kernels {

// ── helpers ───────────────────────────────────────────────────────────
static __host__ __device__ __forceinline__ int cdiv_af(int a, int b) {
    return (a + b - 1) / b;
}

// The five convective schemes live in ../advection_schemes.h (shared
// with the CPU twin in advection_flux_cpu.cpp).
using schemes::apply_scheme;

// ── main kernel ───────────────────────────────────────────────────────
//
// Each thread handles one interior cell (i_fd, i_t1, i_t2) and
// computes two adjacent face fluxes:
//   F_left  at global face index  i_fd       (in [0, Nfd-3])
//   F_right at global face index  i_fd + 1   (in [1, Nfd-2])
//
// Then it accumulates:  rhs[i_fd, i_t1, i_t2] += dt_dh*(F_left - F_right)
//
// Boundary treatment:
//   lo face  (i_fd == 0):      positive flow → CDS(p[0],p[1]);
//                               negative flow → scheme(p[2],p[1],p[0])
//   hi face  (i_fd+1==Nfd-2):  positive flow → scheme(p[-3],p[-2],p[-1]);
//                               negative flow → CDS(p[-2],p[-1])
//   interior faces: full 3-point scheme in both flow directions
//
// Note: for the LEFT face, the hi-boundary case NEVER occurs (max
// i_fd = Nfd-3 < Nfd-2). For the RIGHT face, the lo-boundary case
// NEVER occurs (min i_fd+1 = 1 > 0). This simplifies the logic below.
//
// Strides p_s_fd / p_s_t1 / p_s_t2 etc. handle non-contiguous views.
// For 2-D, pass Nt2=1 and p_s_t2=fv_s_t2=0.

template <typename scalar_t, int scheme_id>
__global__ void advect_flux_add_kernel(
    const scalar_t* __restrict__ p_ptr,
    const scalar_t* __restrict__ fv_ptr,
    scalar_t*       __restrict__ rhs_ptr,
    int    Nfd,                              // full p size along face_dim
    int    Nt1, int Nt2,                    // interior transverse sizes
    int64_t p_s_fd,   int64_t p_s_t1,  int64_t p_s_t2,
    int64_t fv_s_fd,  int64_t fv_s_t1, int64_t fv_s_t2,
    int64_t rhs_s_fd, int64_t rhs_s_t1, int64_t rhs_s_t2,
    scalar_t dt_dh, scalar_t C_courant
) {
    // Thread-to-cell mapping
    int i_t2 = (int)(blockIdx.x * blockDim.x) + (int)threadIdx.x;
    int i_t1 = (int)(blockIdx.y * blockDim.y) + (int)threadIdx.y;
    int i_fd = (int)(blockIdx.z * blockDim.z) + (int)threadIdx.z;
    int Ni_fd = Nfd - 2;

    if (i_t2 >= Nt2 || i_t1 >= Nt1 || i_fd >= Ni_fd) return;

    // Transverse offsets (same for p and fv)
    int64_t tp  = (int64_t)i_t1 * p_s_t1  + (int64_t)i_t2 * p_s_t2;
    int64_t tfv = (int64_t)i_t1 * fv_s_t1 + (int64_t)i_t2 * fv_s_t2;

    const scalar_t* p_row  = p_ptr  + tp;
    const scalar_t* fv_row = fv_ptr + tfv;

    int f_L = i_fd;         // left face global index  [0, Ni_fd-1]
    int f_R = i_fd + 1;    // right face global index [1, Ni_fd]

    scalar_t fv_L = fv_row[(int64_t)f_L * fv_s_fd];
    scalar_t fv_R = fv_row[(int64_t)f_R * fv_s_fd];

    // ---- left face flux ----
    scalar_t F_L;
    {
        scalar_t pc = p_row[(int64_t) f_L      * p_s_fd];
        scalar_t pd = p_row[(int64_t)(f_L + 1) * p_s_fd];
        if (fv_L > scalar_t(0)) {
            if (f_L == 0) {
                // lo boundary: no upstream → CDS fallback
                F_L = fv_L * scalar_t(0.5) * (pc + pd);
            } else {
                scalar_t pu = p_row[(int64_t)(f_L - 1) * p_s_fd];
                F_L = fv_L * apply_scheme<scalar_t, scheme_id>(pu, pc, pd, C_courant);
            }
        } else {
            // negative flow: f_L is never the hi boundary (max = Nfd-3 < Nfd-2)
            scalar_t pdd = p_row[(int64_t)(f_L + 2) * p_s_fd];
            F_L = fv_L * apply_scheme<scalar_t, scheme_id>(pdd, pd, pc, C_courant);
        }
    }

    // ---- right face flux ----
    scalar_t F_R;
    {
        scalar_t pc = p_row[(int64_t) f_R      * p_s_fd];
        scalar_t pd = p_row[(int64_t)(f_R + 1) * p_s_fd];
        if (fv_R > scalar_t(0)) {
            // positive flow: f_R is never the lo boundary (min = 1 > 0)
            scalar_t pu = p_row[(int64_t)(f_R - 1) * p_s_fd];
            F_R = fv_R * apply_scheme<scalar_t, scheme_id>(pu, pc, pd, C_courant);
        } else {
            if (f_R == Nfd - 2) {
                // hi boundary: no downstream → CDS fallback
                F_R = fv_R * scalar_t(0.5) * (pc + pd);
            } else {
                scalar_t pdd = p_row[(int64_t)(f_R + 2) * p_s_fd];
                F_R = fv_R * apply_scheme<scalar_t, scheme_id>(pdd, pd, pc, C_courant);
            }
        }
    }

    // Accumulate divergence into rhs.
    // rhs has shape (Ni_0, Ni_1, [Ni_2]) in ORIGINAL grid dim order, so we
    // must use the rhs strides (which depend on face_dim) rather than
    // assuming face_dim is the outermost dim.
    int64_t ridx = (int64_t)i_fd * rhs_s_fd
                 + (int64_t)i_t1 * rhs_s_t1
                 + (int64_t)i_t2 * rhs_s_t2;
    rhs_ptr[ridx] += dt_dh * (F_L - F_R);
}

// ── C++ wrapper ───────────────────────────────────────────────────────

static void advect_flux_add_cuda(
    const at::Tensor& fv,
    const at::Tensor& p,
    at::Tensor&       rhs,
    double            dt_dh,
    double            C_courant,
    int64_t           scheme_id,
    int64_t           face_dim
) {
    int ndim = (int)p.dim();
    TORCH_CHECK(ndim == 2 || ndim == 3,
                "advect_flux_add: p must be 2-D or 3-D, got ", ndim, "-D");
    TORCH_CHECK(face_dim >= 0 && face_dim < ndim,
                "advect_flux_add: face_dim=", face_dim, " out of range for ndim=", ndim);

    int Nfd = (int)p.size((int)face_dim);

    // Gather transverse dimensions in order (skip face_dim)
    int Nt1 = 1, Nt2 = 1;
    int64_t p_s_fd  = p.stride((int)face_dim);
    int64_t p_s_t1  = 0, p_s_t2  = 0;
    int64_t fv_s_fd = fv.stride((int)face_dim);
    int64_t fv_s_t1 = 0, fv_s_t2 = 0;

    int t_count = 0;
    int t_dims[2] = {-1, -1};
    for (int d = 0; d < ndim; ++d) {
        if (d != (int)face_dim) t_dims[t_count++] = d;
    }
    if (t_count > 0) {
        Nt1     = (int)p.size(t_dims[0]);
        p_s_t1  = p.stride(t_dims[0]);
        fv_s_t1 = fv.stride(t_dims[0]);
    }
    if (t_count > 1) {
        Nt2     = (int)p.size(t_dims[1]);
        p_s_t2  = p.stride(t_dims[1]);
        fv_s_t2 = fv.stride(t_dims[1]);
    }

    // rhs strides — rhs has shape (Ni_0, Ni_1, [Ni_2]) in ORIGINAL dim order,
    // so rhs.stride(face_dim) != Nt1*Nt2 in general.
    int64_t rhs_s_fd = rhs.stride((int)face_dim);
    int64_t rhs_s_t1 = (t_count > 0) ? rhs.stride(t_dims[0]) : 0;
    int64_t rhs_s_t2 = (t_count > 1) ? rhs.stride(t_dims[1]) : 0;

    int Ni_fd = Nfd - 2;
    if (Ni_fd <= 0 || Nt1 <= 0 || Nt2 <= 0) return;

    auto stream = at::cuda::getCurrentCUDAStream();

    // Block layout: x→t2 (coalesced rhs writes), y→t1, z→fd
    // 2-D: (32, 8, 1) = 256 threads
    // 3-D: (16, 8, 2) = 256 threads
    const int BT2 = (ndim == 3) ? 16 : 32;
    const int BT1 = 8;
    const int BFD = (ndim == 3) ? 2  : 1;
    dim3 blk(BT2, BT1, BFD);
    dim3 grd(cdiv_af(Nt2, BT2), cdiv_af(Nt1, BT1), cdiv_af(Ni_fd, BFD));

#define LAUNCH(SID)                                                             \
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "advect_flux_add", [&] {        \
        advect_flux_add_kernel<scalar_t, SID><<<grd, blk, 0, stream>>>(         \
            p.data_ptr<scalar_t>(),                                             \
            fv.data_ptr<scalar_t>(),                                            \
            rhs.data_ptr<scalar_t>(),                                           \
            Nfd, Nt1, Nt2,                                                      \
            p_s_fd,  p_s_t1,  p_s_t2,                                          \
            fv_s_fd, fv_s_t1, fv_s_t2,                                         \
            rhs_s_fd, rhs_s_t1, rhs_s_t2,                                      \
            static_cast<scalar_t>(dt_dh),                                       \
            static_cast<scalar_t>(C_courant)                                    \
        );                                                                      \
    });

    switch ((int)scheme_id) {
        case 0: LAUNCH(0); break;
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 3: LAUNCH(3); break;
        case 4: LAUNCH(4); break;
        default:
            TORCH_CHECK(false, "advect_flux_add: unknown scheme_id=", scheme_id);
    }
#undef LAUNCH
}

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
    m.impl("advect_flux_add", &advect_flux_add_cuda);
    m.impl("advect_flux_accumulate", &advect_flux_accumulate_cuda);
}

}  // namespace lilytorch_kernels
