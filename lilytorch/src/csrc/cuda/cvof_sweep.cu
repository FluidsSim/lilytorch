// =====================================================================
//  cvof_sweep.cu — fused CUDA kernel for the Weymouth & Yue (2010)
//  conservative-VOF directional sweep (MP10 / T2d).
//
//  Replaces the Python ``_cvof_sweep`` chain in ``TwoPhase._cvof_sweep``
//  (lilytorch/src/two_phase.py): the three edge-clamped neighbour shifts
//  (a_m1, a_m2, a_p1), the two van-Leer limited slopes, the two donor
//  face extrapolations, the flux tensor F, and the divergence-corrected
//  interior update — all materialised as ~8 full-grid temporaries per
//  (sweep, direction) — collapse into a single kernel launch that keeps
//  every intermediate in registers and writes the new alpha directly.
//
//  Op: ``lilytorch_kernels::cvof_sweep``
//  Signature (see ops.cpp):
//      Tensor a, Tensor u_d, float cfl, int face_dim, Tensor(a!) out
//
//  Semantics (mirrors the Python reference, per interior cell i along
//  ``face_dim``; transverse dims FULL — the Python ``out[S(slice(1,-1))]``
//  slices only the sweep direction):
//
//      out[i] = a[i] + cfl * ( F(i) - F(i+1) + a[i]*(u[i+1] - u[i]) )
//
//  where F(k) is the upwind flux at the d-face left of cell k:
//
//      C       = u[k] * cfl                       (face Courant number)
//      s_pos   = vleer(a[k-1]-a[k-2],  a[k]-a[k-1])
//      f_pos   = a[k-1] + 0.5*(1-C)*s_pos         (donor = cell k-1)
//      s_neg   = vleer(a[k]-a[k-1],    a[k+1]-a[k])
//      f_neg   = a[k]   - 0.5*(1+C)*s_neg         (donor = cell k)
//      F(k)    = u[k] * (C >= 0 ? f_pos : f_neg)
//
//  Neighbour reads use EDGE-CLAMP (Neumann-consistent) exactly like the
//  Python ``_shift``: a[k-1]→a[max(k-1,0)], a[k-2]→a[max(k-2,0)],
//  a[k+1]→a[min(k+1,N-1)].
//
//  ``out`` MUST be preallocated as ``a.clone()`` by the caller — the
//  kernel only overwrites the interior cells along ``face_dim`` (the
//  boundary cells keep the cloned values, matching ``out = a.clone()``
//  followed by the interior assignment in Python).
//
//  Works for 2-D and 3-D.  Handles non-contiguous a / u_d (velocity
//  components are row views of the stacked _vel tensor → strided) via
//  explicit per-tensor strides extracted in the C++ wrapper.
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace lilytorch_kernels {

static __host__ __device__ __forceinline__ int cdiv_cv(int a, int b) {
    return (a + b - 1) / b;
}

// van-Leer (harmonic) limited slope; 0 at extrema / sign changes.
// Mirrors the Python order: s = 2.0*db*df/denom, denom protected at 0.
template <typename T>
__device__ __forceinline__ T cv_vleer(T db, T df) {
    T sum   = db + df;
    T denom = (sum == T(0)) ? T(1) : sum;
    T s     = T(2) * db * df / denom;
    return (db * df > T(0)) ? s : T(0);
}

// Upwind W&Y flux at the d-face left of cell k (global index k along
// face_dim).  a_row / u_row already point at the transverse slice; the
// only varying index is the face-dim coordinate, scaled by its stride.
template <typename T>
__device__ __forceinline__ T cv_face(
    const T* __restrict__ a_row, const T* __restrict__ u_row,
    int k, int N, int64_t a_s_fd, int64_t u_s_fd, T cfl)
{
    int km1 = k - 1 > 0     ? k - 1 : 0;        // max(k-1, 0)
    int km2 = k - 2 > 0     ? k - 2 : 0;        // max(k-2, 0)
    int kp1 = k + 1 < N - 1 ? k + 1 : N - 1;    // min(k+1, N-1)

    T ak  = a_row[(int64_t)k   * a_s_fd];
    T am1 = a_row[(int64_t)km1 * a_s_fd];
    T am2 = a_row[(int64_t)km2 * a_s_fd];
    T ap1 = a_row[(int64_t)kp1 * a_s_fd];
    T ud  = u_row[(int64_t)k   * u_s_fd];
    T C   = ud * cfl;

    T s_pos = cv_vleer<T>(am1 - am2, ak - am1);
    T f_pos = am1 + T(0.5) * (T(1) - C) * s_pos;

    T s_neg = cv_vleer<T>(ak - am1, ap1 - ak);
    T f_neg = ak - T(0.5) * (T(1) + C) * s_neg;

    T face = (C >= T(0)) ? f_pos : f_neg;
    return ud * face;
}

template <typename scalar_t>
__global__ void cvof_sweep_kernel(
    const scalar_t* __restrict__ a_ptr,
    const scalar_t* __restrict__ u_ptr,
    scalar_t*       __restrict__ out_ptr,
    int    Nfd,                              // size along face_dim
    int    Nt1, int Nt2,                    // transverse sizes (FULL)
    int64_t a_s_fd,   int64_t a_s_t1,   int64_t a_s_t2,
    int64_t u_s_fd,   int64_t u_s_t1,   int64_t u_s_t2,
    int64_t out_s_fd, int64_t out_s_t1, int64_t out_s_t2,
    scalar_t cfl
) {
    int i_t2 = (int)(blockIdx.x * blockDim.x) + (int)threadIdx.x;
    int i_t1 = (int)(blockIdx.y * blockDim.y) + (int)threadIdx.y;
    int i_in = (int)(blockIdx.z * blockDim.z) + (int)threadIdx.z;  // interior index
    int Ni   = Nfd - 2;                        // # interior cells along face_dim

    if (i_t2 >= Nt2 || i_t1 >= Nt1 || i_in >= Ni) return;

    int i = i_in + 1;                          // global cell index along face_dim

    int64_t a_t   = (int64_t)i_t1 * a_s_t1   + (int64_t)i_t2 * a_s_t2;
    int64_t u_t   = (int64_t)i_t1 * u_s_t1   + (int64_t)i_t2 * u_s_t2;
    int64_t out_t = (int64_t)i_t1 * out_s_t1 + (int64_t)i_t2 * out_s_t2;

    const scalar_t* a_row = a_ptr + a_t;
    const scalar_t* u_row = u_ptr + u_t;

    scalar_t ai = a_row[(int64_t)i * a_s_fd];
    scalar_t FL = cv_face<scalar_t>(a_row, u_row, i,     Nfd, a_s_fd, u_s_fd, cfl);
    scalar_t FR = cv_face<scalar_t>(a_row, u_row, i + 1, Nfd, a_s_fd, u_s_fd, cfl);
    scalar_t uL = u_row[(int64_t)i       * u_s_fd];
    scalar_t uR = u_row[(int64_t)(i + 1) * u_s_fd];

    scalar_t res = ai + cfl * ((FL - FR) + ai * (uR - uL));
    out_ptr[out_t + (int64_t)i * out_s_fd] = res;
}

// ── C++ wrapper ───────────────────────────────────────────────────────

static void cvof_sweep_cuda(
    const at::Tensor& a,
    const at::Tensor& u_d,
    double            cfl,
    int64_t           face_dim,
    at::Tensor&       out
) {
    int ndim = (int)a.dim();
    TORCH_CHECK(ndim == 2 || ndim == 3,
                "cvof_sweep: a must be 2-D or 3-D, got ", ndim, "-D");
    TORCH_CHECK(face_dim >= 0 && face_dim < ndim,
                "cvof_sweep: face_dim=", face_dim, " out of range for ndim=", ndim);
    TORCH_CHECK(a.is_cuda() && u_d.is_cuda() && out.is_cuda(),
                "cvof_sweep: all tensors must be CUDA");
    TORCH_CHECK(u_d.dim() == ndim && out.dim() == ndim,
                "cvof_sweep: a, u_d, out must have the same ndim");

    int Nfd = (int)a.size((int)face_dim);

    int Nt1 = 1, Nt2 = 1;
    int64_t a_s_fd = a.stride((int)face_dim);
    int64_t a_s_t1 = 0, a_s_t2 = 0;
    int64_t u_s_fd = u_d.stride((int)face_dim);
    int64_t u_s_t1 = 0, u_s_t2 = 0;
    int64_t out_s_fd = out.stride((int)face_dim);
    int64_t out_s_t1 = 0, out_s_t2 = 0;

    int t_count = 0;
    int t_dims[2] = {-1, -1};
    for (int d = 0; d < ndim; ++d) {
        if (d != (int)face_dim) t_dims[t_count++] = d;
    }
    if (t_count > 0) {
        Nt1      = (int)a.size(t_dims[0]);
        a_s_t1   = a.stride(t_dims[0]);
        u_s_t1   = u_d.stride(t_dims[0]);
        out_s_t1 = out.stride(t_dims[0]);
    }
    if (t_count > 1) {
        Nt2      = (int)a.size(t_dims[1]);
        a_s_t2   = a.stride(t_dims[1]);
        u_s_t2   = u_d.stride(t_dims[1]);
        out_s_t2 = out.stride(t_dims[1]);
    }

    int Ni = Nfd - 2;
    if (Ni <= 0 || Nt1 <= 0 || Nt2 <= 0) return;

    auto stream = at::cuda::getCurrentCUDAStream();

    const int BT2 = (ndim == 3) ? 16 : 32;
    const int BT1 = 8;
    const int BFD = (ndim == 3) ? 2  : 1;
    dim3 blk(BT2, BT1, BFD);
    dim3 grd(cdiv_cv(Nt2, BT2), cdiv_cv(Nt1, BT1), cdiv_cv(Ni, BFD));

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "cvof_sweep", [&] {
        cvof_sweep_kernel<scalar_t><<<grd, blk, 0, stream>>>(
            a.data_ptr<scalar_t>(),
            u_d.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            Nfd, Nt1, Nt2,
            a_s_fd,   a_s_t1,   a_s_t2,
            u_s_fd,   u_s_t1,   u_s_t2,
            out_s_fd, out_s_t1, out_s_t2,
            static_cast<scalar_t>(cfl)
        );
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("cvof_sweep", &cvof_sweep_cuda);
}

}  // namespace lilytorch_kernels
