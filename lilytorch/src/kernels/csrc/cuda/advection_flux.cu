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

namespace lilytorch_kernels {

// ── helpers ───────────────────────────────────────────────────────────
static __host__ __device__ __forceinline__ int cdiv_af(int a, int b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __forceinline__ T d_abs(T x) { return x < T(0) ? -x : x; }

template <typename T>
__device__ __forceinline__ T d_max(T a, T b) { return a > b ? a : b; }

template <typename T>
__device__ __forceinline__ T d_min(T a, T b) { return a < b ? a : b; }

template <typename T>
__device__ __forceinline__ T median3(T a, T b, T c) {
    return d_max(d_min(a, b), d_min(d_max(a, b), c));
}

// ── scheme device functions ───────────────────────────────────────────
// Convention: u = upstream (far), c = center (upwind), d = downstream.
// The caller handles the positive/negative flow direction swap.

template <typename T>
__device__ __forceinline__ T scheme_quick(T u, T c, T d) {
    T inner = median3(T(10) * c - T(9) * u, c, d);
    T outer = (T(5) * c + T(2) * d - u) / T(6);
    return median3(outer, c, inner);
}

template <typename T>
__device__ __forceinline__ T scheme_abdquickest(T u, T c, T d, T C) {
    T denom = d - c;
    if (d_abs(denom) < T(1e-30)) return c;
    T rf      = (c - u) / denom;
    T C2      = C * C;
    T C_upper = T(2) * (T(1) - C);
    T scale   = (T(1) - C2) / (T(3) - T(3) * C);
    T offset  = (T(2) + C2 - T(3) * C) / (T(3) - T(3) * C);
    T psi     = d_min(rf * scale + offset, C_upper);
    psi       = d_min(psi, rf * C_upper);
    psi       = d_max(psi, T(0));
    return c + T(0.5) * denom * psi;
}

template <typename T>
__device__ __forceinline__ T scheme_van_leer(T u, T c, T d) {
    T denom = d - c;
    if (d_abs(denom) < T(1e-30)) return c;
    T rf     = (c - u) / denom;
    T abs_rf = d_abs(rf);
    T psi    = (rf + abs_rf) / (T(1) + abs_rf);
    return c + T(0.5) * denom * psi;
}

template <typename T>
__device__ __forceinline__ T scheme_cds(T /*u*/, T c, T d) {
    return T(0.5) * (c + d);
}

template <typename T>
__device__ __forceinline__ T scheme_cubista(T u, T c, T d) {
    T denom = d - c;
    if (d_abs(denom) < T(1e-30)) return c;
    T rf  = (c - u) / denom;
    T psi = d_min(T(0.75) * rf + T(0.25), T(1.5));
    psi   = d_min(psi, rf * T(1.5));
    psi   = d_max(psi, T(0));
    return c + T(0.5) * denom * psi;
}

// Dispatch to the appropriate scheme at compile time.
template <typename T, int sid>
__device__ __forceinline__ T apply_scheme(T u, T c, T d, T C) {
    if constexpr (sid == 0) return scheme_quick<T>(u, c, d);
    if constexpr (sid == 1) return scheme_abdquickest<T>(u, c, d, C);
    if constexpr (sid == 2) return scheme_van_leer<T>(u, c, d);
    if constexpr (sid == 3) return scheme_cds<T>(u, c, d);
    if constexpr (sid == 4) return scheme_cubista<T>(u, c, d);
    return c;
}

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

// ── dispatch registration ─────────────────────────────────────────────
TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("advect_flux_add", &advect_flux_add_cuda);
}

}  // namespace lilytorch_kernels
