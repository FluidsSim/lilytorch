// =====================================================================
//  rbgs.cu — CUDA kernels for multigrid Red-Black Gauss-Seidel smoother
//
//  2-D: ``rbgs_sweep_2d``
//    Tiled kernel: BSIZE_I=8 rows × BSIZE_J=32 cols per block (256 threads).
//    Shared memory holds the p tile + 1-cell halo (10×34 floats) plus all
//    five interior coefficient/RHS arrays (5×8×32 floats) — ~6.3 KB/block.
//    All ``nsmoothing`` red-black sweeps are fused inside a single kernel
//    launch, so global memory is read ONCE and written ONCE per smoother
//    call (vs. 2×nsmoothing passes in the PyTorch reference path).
//    Neumann BCs (ghost-row copy) are applied in CUDA before and after the
//    tiled sweep; no Python round-trips are needed between half-sweeps.
//
//  3-D: ``rbgs_sweep_3d``
//    Thread-per-cell kernel, launched separately for color=0 (red) and
//    color=1 (black).  Eliminates the ``torch.where`` double-read of p_old
//    and the boolean mask array.  The C++ wrapper loops ``nsmoothing``
//    times, interleaving with BC updates, entirely on the CUDA stream.
//
//  Memory layout (PyTorch C-contiguous):
//    2-D: p[i,j] = p_ptr[i*(Ny+2) + j],  f/cp/cm[i,j] = ptr[i*Ny + j]
//    3-D: p[i,j,k] = p_ptr[i*(Ny+2)*(Nz+2) + j*(Nz+2) + k]
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace lilytorch_kernels {

// ── Helper ────────────────────────────────────────────────────────────
static __host__ __device__ __forceinline__ int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

// =====================================================================
// Neumann BC helpers (ghost-row = interior-edge copy)
// =====================================================================

// 2-D: two kernel launches — one per axis pair
template <typename scalar_t>
__global__ void neumann_bc_2d_xfaces(scalar_t* __restrict__ p,
                                      int Nx, int Ny) {
    // Handles j = 0..Ny+1  (full column width including ghost cols)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Ny + 2) return;
    const int stride = Ny + 2;
    p[j]                    = p[stride + j];          // p[0,j] = p[1,j]
    p[(Nx + 1) * stride + j] = p[Nx * stride + j];   // p[Nx+1,j] = p[Nx,j]
}

template <typename scalar_t>
__global__ void neumann_bc_2d_yfaces(scalar_t* __restrict__ p,
                                      int Nx, int Ny) {
    // Handles i = 0..Nx+1  (full row height including ghost rows)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nx + 2) return;
    const int stride = Ny + 2;
    const int base   = i * stride;
    p[base]          = p[base + 1];           // p[i,0] = p[i,1]
    p[base + Ny + 1] = p[base + Ny];         // p[i,Ny+1] = p[i,Ny]
}

template <typename scalar_t>
static void apply_neumann_bc_2d(scalar_t* p,
                                  int Nx, int Ny,
                                  cudaStream_t stream) {
    constexpr int BLOCK = 256;
    neumann_bc_2d_xfaces<scalar_t>
        <<<cdiv(Ny + 2, BLOCK), BLOCK, 0, stream>>>(p, Nx, Ny);
    neumann_bc_2d_yfaces<scalar_t>
        <<<cdiv(Nx + 2, BLOCK), BLOCK, 0, stream>>>(p, Nx, Ny);
}

// 3-D: three kernel launches — one per axis pair
template <typename scalar_t>
__global__ void neumann_bc_3d_xfaces(scalar_t* __restrict__ p,
                                      int Nx, int Ny, int Nz) {
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Ny + 2 || k >= Nz + 2) return;
    const int si = (Ny + 2) * (Nz + 2);
    const int jk = j * (Nz + 2) + k;
    p[jk]               = p[si + jk];              // p[0,j,k]  = p[1,j,k]
    p[(Nx+1)*si + jk]   = p[Nx*si + jk];           // p[Nx+1,j,k] = p[Nx,j,k]
}

template <typename scalar_t>
__global__ void neumann_bc_3d_yfaces(scalar_t* __restrict__ p,
                                      int Nx, int Ny, int Nz) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nx + 2 || k >= Nz + 2) return;
    const int si = (Ny + 2) * (Nz + 2);
    const int sj = Nz + 2;
    const int base = i * si + k;
    p[base]               = p[base + sj];             // p[i,0,k] = p[i,1,k]
    p[base + (Ny+1)*sj]   = p[base + Ny*sj];         // p[i,Ny+1,k] = p[i,Ny,k]
}

template <typename scalar_t>
__global__ void neumann_bc_3d_zfaces(scalar_t* __restrict__ p,
                                      int Nx, int Ny, int Nz) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nx + 2 || j >= Ny + 2) return;
    const int si   = (Ny + 2) * (Nz + 2);
    const int sj   = Nz + 2;
    const int base = i * si + j * sj;
    p[base]          = p[base + 1];           // p[i,j,0] = p[i,j,1]
    p[base + Nz + 1] = p[base + Nz];         // p[i,j,Nz+1] = p[i,j,Nz]
}

template <typename scalar_t>
static void apply_neumann_bc_3d(scalar_t* p,
                                  int Nx, int Ny, int Nz,
                                  cudaStream_t stream) {
    const dim3 blk(16, 8);
    neumann_bc_3d_xfaces<scalar_t>
        <<<dim3(cdiv(Nz+2,16), cdiv(Ny+2,8)), blk, 0, stream>>>(p, Nx, Ny, Nz);
    neumann_bc_3d_yfaces<scalar_t>
        <<<dim3(cdiv(Nz+2,16), cdiv(Nx+2,8)), blk, 0, stream>>>(p, Nx, Ny, Nz);
    neumann_bc_3d_zfaces<scalar_t>
        <<<dim3(cdiv(Ny+2,16), cdiv(Nx+2,8)), blk, 0, stream>>>(p, Nx, Ny, Nz);
}

// =====================================================================
// 2-D tiled RBGS kernel
//
//   Block dim:   (BSIZE_J=32, BSIZE_I=8)  →  threadIdx.x = j (fast/coalesced)
//                                             threadIdx.y = i
//   Grid  dim:   (ceil(Ny/32), ceil(Nx/8))
//   SMEM layout: p_s  [BSIZE_I+2][BSIZE_J+2]   (tile + 1-cell halo)
//                cp0_s,cm0_s,cp1_s,cm1_s,f_s  each [BSIZE_I][BSIZE_J]
//   Total SMEM ≈ 6.3 KB/block → 6 blocks/SM on Ada, 100% occupancy.
//
//   All nsmoothing red–black sweeps are executed inside the kernel.
//   Halo cells are loaded ONCE from global memory; interior cells are
//   written back ONCE.  Inter-tile halos use the pre-sweep global values
//   (block-boundary approximation), which is acceptable for multigrid.
// =====================================================================

#define RBGS_2D_I  8
#define RBGS_2D_J  32

template <typename scalar_t>
__global__ void rbgs_2d_tiled_kernel(
        scalar_t* __restrict__ p,
        const scalar_t* __restrict__ f,
        const scalar_t* __restrict__ cp0,
        const scalar_t* __restrict__ cm0,
        const scalar_t* __restrict__ cp1,
        const scalar_t* __restrict__ cm1,
        int Nx, int Ny,
        scalar_t jcap_tol,
        int nsmoothing)
{
    // Interior cell indices (0-based)
    const int gi = blockIdx.y * RBGS_2D_I + threadIdx.y;   // row
    const int gj = blockIdx.x * RBGS_2D_J + threadIdx.x;   // col (coalesced)
    const int ti = (int)threadIdx.y;
    const int tj = (int)threadIdx.x;
    const int stride_p = Ny + 2;   // row stride in padded p
    const int stride_c = Ny;       // row stride in interior coeff arrays

    // ── Shared memory ─────────────────────────────────────────────────
    __shared__ scalar_t p_s  [RBGS_2D_I + 2][RBGS_2D_J + 2];
    __shared__ scalar_t cp0_s[RBGS_2D_I    ][RBGS_2D_J    ];
    __shared__ scalar_t cm0_s[RBGS_2D_I    ][RBGS_2D_J    ];
    __shared__ scalar_t cp1_s[RBGS_2D_I    ][RBGS_2D_J    ];
    __shared__ scalar_t cm1_s[RBGS_2D_I    ][RBGS_2D_J    ];
    __shared__ scalar_t f_s  [RBGS_2D_I    ][RBGS_2D_J    ];

    // ── Load coefficient tile (interior only) ─────────────────────────
    if (gi < Nx && gj < Ny) {
        const int idx  = gi * stride_c + gj;
        cp0_s[ti][tj] = cp0[idx];
        cm0_s[ti][tj] = cm0[idx];
        cp1_s[ti][tj] = cp1[idx];
        cm1_s[ti][tj] = cm1[idx];
        f_s  [ti][tj] = f  [idx];
    }

    // ── Load p_s (tile + 1-cell halo) via linearized loop ─────────────
    // Uses clamped global indices so partial blocks (coarse MG levels
    // where Ny < BSIZE_J) automatically receive ghost-cell values rather
    // than uninitialised shared memory.
    //
    // Mapping: p_s[si][sj] = p[clamp(gi_base+si, 0, Nx+1) * stride_p
    //                           + clamp(gj_base+sj, 0, Ny+1)]
    {
        const int gi_base  = (int)(blockIdx.y) * RBGS_2D_I;
        const int gj_base  = (int)(blockIdx.x) * RBGS_2D_J;
        const int smem_tot = (RBGS_2D_I + 2) * (RBGS_2D_J + 2);  // 340
        const int stride_s = RBGS_2D_J + 2;                       // 34
        const int tid_lin  = ti * RBGS_2D_J + tj;                 // 0..255
        for (int k = tid_lin; k < smem_tot; k += RBGS_2D_I * RBGS_2D_J) {
            const int si = k / stride_s;
            const int sj = k % stride_s;
            const int pi = max(0, min(gi_base + si, Nx + 1));
            const int pj = max(0, min(gj_base + sj, Ny + 1));
            p_s[si][sj] = p[pi * stride_p + pj];
        }
    }

    __syncthreads();

    // Out-of-bounds threads do no work after this point
    if (gi >= Nx || gj >= Ny) return;

    // ── Precompute Jinv (constant across sweeps) ──────────────────────
    const scalar_t J = (cp0_s[ti][tj] + cm0_s[ti][tj]
                      + cp1_s[ti][tj] + cm1_s[ti][tj]);
    const scalar_t Jinv = ((J >= jcap_tol) || (J <= -jcap_tol))
                          ? ((scalar_t)1 / J) : (scalar_t)0;
    const scalar_t neg_f = -f_s[ti][tj];

    // Color: 0 = red (updates on red half-sweep), 1 = black
    const int color = (gi + gj) & 1;

    // ── nsmoothing full RBGS sweeps in shared memory ──────────────────
    for (int s = 0; s < nsmoothing; s++) {

        // --- red half-sweep ---
        if (color == 0) {
            const scalar_t sum =
                  cp0_s[ti][tj] * p_s[ti + 2][tj + 1]   // i+1 neighbor
                + cm0_s[ti][tj] * p_s[ti    ][tj + 1]   // i-1 neighbor
                + cp1_s[ti][tj] * p_s[ti + 1][tj + 2]   // j+1 neighbor
                + cm1_s[ti][tj] * p_s[ti + 1][tj    ];  // j-1 neighbor
            p_s[ti + 1][tj + 1] = (neg_f + sum) * Jinv;
        }
        __syncthreads();

        // --- black half-sweep ---
        if (color == 1) {
            const scalar_t sum =
                  cp0_s[ti][tj] * p_s[ti + 2][tj + 1]
                + cm0_s[ti][tj] * p_s[ti    ][tj + 1]
                + cp1_s[ti][tj] * p_s[ti + 1][tj + 2]
                + cm1_s[ti][tj] * p_s[ti + 1][tj    ];
            p_s[ti + 1][tj + 1] = (neg_f + sum) * Jinv;
        }
        __syncthreads();
    }

    // ── Write result back to global memory ────────────────────────────
    p[(gi + 1) * stride_p + (gj + 1)] = p_s[ti + 1][tj + 1];
}

// ── C++ wrapper ───────────────────────────────────────────────────────
void rbgs_sweep_2d_cuda(
        at::Tensor p,
        at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        double jcap_tol,
        int64_t nsmoothing)
{
    TORCH_CHECK(p.is_contiguous(), "rbgs_sweep_2d: p must be contiguous");
    TORCH_CHECK(f.is_contiguous(), "rbgs_sweep_2d: f must be contiguous");
    TORCH_CHECK(cp0.is_contiguous(), "rbgs_sweep_2d: cp0 must be contiguous");
    TORCH_CHECK(cm0.is_contiguous(), "rbgs_sweep_2d: cm0 must be contiguous");
    TORCH_CHECK(cp1.is_contiguous(), "rbgs_sweep_2d: cp1 must be contiguous");
    TORCH_CHECK(cm1.is_contiguous(), "rbgs_sweep_2d: cm1 must be contiguous");
    TORCH_CHECK(p.device().is_cuda(), "rbgs_sweep_2d: tensors must be on CUDA");

    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);
    auto stream  = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "rbgs_sweep_2d", [&] {
        scalar_t* pp = p.data_ptr<scalar_t>();

        // Neumann BC before sweeps
        apply_neumann_bc_2d<scalar_t>(pp, Nx, Ny, stream);

        // Tiled RBGS sweep (all nsmoothing iterations fused)
        const dim3 blk(RBGS_2D_J, RBGS_2D_I);
        const dim3 grd(cdiv(Ny, RBGS_2D_J), cdiv(Nx, RBGS_2D_I));
        rbgs_2d_tiled_kernel<scalar_t><<<grd, blk, 0, stream>>>(
            pp,
            f.data_ptr<scalar_t>(),
            cp0.data_ptr<scalar_t>(),
            cm0.data_ptr<scalar_t>(),
            cp1.data_ptr<scalar_t>(),
            cm1.data_ptr<scalar_t>(),
            Nx, Ny,
            static_cast<scalar_t>(jcap_tol),
            (int)nsmoothing
        );

        // Neumann BC after sweeps (refresh ghost rows for residual / next call)
        apply_neumann_bc_2d<scalar_t>(pp, Nx, Ny, stream);
    });
}

// =====================================================================
// 3-D RBGS half-sweep kernel (thread-per-cell, red OR black)
//
//   Block dim: (16, 8, 4) = 512 threads  (k=fast/coalesced, j, i)
//   Grid  dim: (ceil(Nz/16), ceil(Ny/8), ceil(Nx/4))
//   No shared memory — maximises occupancy and keeps code simple.
//   The main gain over PyTorch is eliminating torch.where's full read
//   of p_old and the boolean mask, saving ~1 extra pass over p per
//   half-sweep.
// =====================================================================

template <typename scalar_t>
__global__ void rbgs_3d_halfsweep_kernel(
        scalar_t* __restrict__ p,
        const scalar_t* __restrict__ f,
        const scalar_t* __restrict__ cp0,
        const scalar_t* __restrict__ cm0,
        const scalar_t* __restrict__ cp1,
        const scalar_t* __restrict__ cm1,
        const scalar_t* __restrict__ cp2,
        const scalar_t* __restrict__ cm2,
        int Nx, int Ny, int Nz,
        scalar_t jcap_tol,
        int color)
{
    const int gk = blockIdx.x * blockDim.x + threadIdx.x;   // k (fast)
    const int gj = blockIdx.y * blockDim.y + threadIdx.y;   // j
    const int gi = blockIdx.z * blockDim.z + threadIdx.z;   // i (slow)

    if (gi >= Nx || gj >= Ny || gk >= Nz) return;
    if (((gi + gj + gk) & 1) != color) return;

    const int si  = (Ny + 2) * (Nz + 2);   // stride along i in padded p
    const int sj  = Nz + 2;                 // stride along j in padded p
    // stride along k = 1 (implicit)

    const int pi = gi + 1, pj = gj + 1, pk = gk + 1;
    const int p_base = pi * si + pj * sj + pk;

    const int idx_c = gi * (Ny * Nz) + gj * Nz + gk;

    const scalar_t J = (cp0[idx_c] + cm0[idx_c]
                      + cp1[idx_c] + cm1[idx_c]
                      + cp2[idx_c] + cm2[idx_c]);
    if (J < jcap_tol && J > -jcap_tol) return;
    const scalar_t Jinv = (scalar_t)1 / J;

    const scalar_t sum =
          cp0[idx_c] * p[p_base + si]    // p[i+1, j, k]
        + cm0[idx_c] * p[p_base - si]    // p[i-1, j, k]
        + cp1[idx_c] * p[p_base + sj]    // p[i, j+1, k]
        + cm1[idx_c] * p[p_base - sj]    // p[i, j-1, k]
        + cp2[idx_c] * p[p_base + 1]     // p[i, j, k+1]
        + cm2[idx_c] * p[p_base - 1];    // p[i, j, k-1]

    p[p_base] = (-f[idx_c] + sum) * Jinv;
}

// ── C++ wrapper ───────────────────────────────────────────────────────
void rbgs_sweep_3d_cuda(
        at::Tensor p,
        at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        at::Tensor cp2, at::Tensor cm2,
        double jcap_tol,
        int64_t nsmoothing)
{
    TORCH_CHECK(p.is_contiguous(), "rbgs_sweep_3d: p must be contiguous");
    TORCH_CHECK(f.is_contiguous(), "rbgs_sweep_3d: f must be contiguous");
    TORCH_CHECK(cp0.is_contiguous(), "rbgs_sweep_3d: cp0 must be contiguous");
    TORCH_CHECK(cm0.is_contiguous(), "rbgs_sweep_3d: cm0 must be contiguous");
    TORCH_CHECK(cp1.is_contiguous(), "rbgs_sweep_3d: cp1 must be contiguous");
    TORCH_CHECK(cm1.is_contiguous(), "rbgs_sweep_3d: cm1 must be contiguous");
    TORCH_CHECK(cp2.is_contiguous(), "rbgs_sweep_3d: cp2 must be contiguous");
    TORCH_CHECK(cm2.is_contiguous(), "rbgs_sweep_3d: cm2 must be contiguous");
    TORCH_CHECK(p.device().is_cuda(), "rbgs_sweep_3d: tensors must be on CUDA");

    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);
    const int Nz = (int)f.size(2);
    auto stream  = at::cuda::getCurrentCUDAStream();

    const dim3 blk(16, 8, 4);
    const dim3 grd(cdiv(Nz, 16), cdiv(Ny, 8), cdiv(Nx, 4));

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "rbgs_sweep_3d", [&] {
        scalar_t* pp = p.data_ptr<scalar_t>();

        // Neumann BC before all sweeps
        apply_neumann_bc_3d<scalar_t>(pp, Nx, Ny, Nz, stream);

        for (int s = 0; s < (int)nsmoothing; s++) {
            // Red half-sweep (color=0)
            rbgs_3d_halfsweep_kernel<scalar_t><<<grd, blk, 0, stream>>>(
                pp,
                f.data_ptr<scalar_t>(),
                cp0.data_ptr<scalar_t>(), cm0.data_ptr<scalar_t>(),
                cp1.data_ptr<scalar_t>(), cm1.data_ptr<scalar_t>(),
                cp2.data_ptr<scalar_t>(), cm2.data_ptr<scalar_t>(),
                Nx, Ny, Nz,
                static_cast<scalar_t>(jcap_tol),
                0
            );
            apply_neumann_bc_3d<scalar_t>(pp, Nx, Ny, Nz, stream);

            // Black half-sweep (color=1)
            rbgs_3d_halfsweep_kernel<scalar_t><<<grd, blk, 0, stream>>>(
                pp,
                f.data_ptr<scalar_t>(),
                cp0.data_ptr<scalar_t>(), cm0.data_ptr<scalar_t>(),
                cp1.data_ptr<scalar_t>(), cm1.data_ptr<scalar_t>(),
                cp2.data_ptr<scalar_t>(), cm2.data_ptr<scalar_t>(),
                Nx, Ny, Nz,
                static_cast<scalar_t>(jcap_tol),
                1
            );
            apply_neumann_bc_3d<scalar_t>(pp, Nx, Ny, Nz, stream);
        }
    });
}

// ── CUDA dispatch registration ─────────────────────────────────────────
TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("rbgs_sweep_2d", &rbgs_sweep_2d_cuda);
    m.impl("rbgs_sweep_3d", &rbgs_sweep_3d_cuda);
    m.impl("jacobi_sweep_2d", &jacobi_sweep_2d_cuda);
    m.impl("jacobi_sweep_3d", &jacobi_sweep_3d_cuda);
}

}  // namespace lilytorch_kernels
