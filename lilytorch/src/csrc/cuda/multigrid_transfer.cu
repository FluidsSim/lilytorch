// =====================================================================
//  multigrid_transfer.cu — CUDA kernels for multigrid grid transfer
//
//  Provides:
//    - restrict_residual_{2,3}d : fine -> coarse residual restriction
//        r_c[I,J] = sum over 4 (or 8) children   (sum-only, no scaling)
//      Bit-exact with the existing slicing chain
//      ``_restrict_residual_{2,3}d`` in poisson_mult.py.
//
//    - restrict_face_{2,3}d : fine -> coarse face coefficient restriction
//        WaterLily convention: stride-2 in the face direction, SUM in the
//        transverse directions, single 0.5 factor applied at the end.
//        For face along dim 0 (2-D):
//          ch_c[I,J] = 0.5 * (ch[2I, 2J] + ch[2I, 2J+1])
//        For face along dim 1 (2-D):
//          cv_c[I,J] = 0.5 * (cv[2I, 2J] + cv[2I+1, 2J])
//
//    - prolongate_add_{2,3}d : coarse error -> fine correction, ADDED
//      in-place to p[interior]. Implements exact
//      F.interpolate(mode='bilinear/trilinear', align_corners=False)
//      with cell-centred mapping
//          src = (dst + 0.5) * Nc/Nf - 0.5
//      and clamped-edge weights. The fused add saves one fine-grid
//      tensor allocation per V-cycle level.
//
//  Memory layout (C-contiguous, row-major):
//    Interior tensors (r, ch, cv, cw): shape (Nx, Ny[, Nz])
//    Ghost-padded p:                   shape (Nx+2, Ny+2[, Nz+2])
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace lilytorch_kernels {

static __host__ __device__ __forceinline__ int cdiv_t(int a, int b) {
    return (a + b - 1) / b;
}

// =====================================================================
// Residual restriction (sum of 4 / 8 children)
// =====================================================================

template <typename scalar_t>
__global__ void restrict_residual_2d_kernel(
        const scalar_t* __restrict__ r,
        scalar_t* __restrict__ rc,
        int Nx_f, int Ny_f,
        int Nx_c, int Ny_c)
{
    const int J = blockIdx.x * blockDim.x + threadIdx.x;
    const int I = blockIdx.y * blockDim.y + threadIdx.y;
    if (I >= Nx_c || J >= Ny_c) return;

    // Bit-exact with: e0,o0 = r[::2], r[1::2]; rc = e0[:m0] + o0[:m0]; then dim 1.
    // m0 = min(ceil(Nx_f/2), floor(Nx_f/2)) = Nx_c (we use floor(Nx_f/2) here).
    const int i0 = 2 * I;
    const int i1 = i0 + 1;
    const int j0 = 2 * J;
    const int j1 = j0 + 1;

    const scalar_t a = r[i0 * Ny_f + j0];
    const scalar_t b = (i1 < Nx_f) ? r[i1 * Ny_f + j0] : (scalar_t)0;
    const scalar_t c = (j1 < Ny_f) ? r[i0 * Ny_f + j1] : (scalar_t)0;
    const scalar_t d = ((i1 < Nx_f) && (j1 < Ny_f))
                       ? r[i1 * Ny_f + j1] : (scalar_t)0;

    rc[I * Ny_c + J] = a + b + c + d;
}

template <typename scalar_t>
__global__ void restrict_residual_3d_kernel(
        const scalar_t* __restrict__ r,
        scalar_t* __restrict__ rc,
        int Nx_f, int Ny_f, int Nz_f,
        int Nx_c, int Ny_c, int Nz_c)
{
    const int K = blockIdx.x * blockDim.x + threadIdx.x;
    const int J = blockIdx.y * blockDim.y + threadIdx.y;
    const int I = blockIdx.z * blockDim.z + threadIdx.z;
    if (I >= Nx_c || J >= Ny_c || K >= Nz_c) return;

    const int sj_f = Nz_f;
    const int si_f = Ny_f * Nz_f;
    const int sj_c = Nz_c;
    const int si_c = Ny_c * Nz_c;

    scalar_t s = 0;
    for (int di = 0; di < 2; ++di) {
        const int ii = 2*I + di;
        if (ii >= Nx_f) continue;
        for (int dj = 0; dj < 2; ++dj) {
            const int jj = 2*J + dj;
            if (jj >= Ny_f) continue;
            for (int dk = 0; dk < 2; ++dk) {
                const int kk = 2*K + dk;
                if (kk >= Nz_f) continue;
                s += r[ii*si_f + jj*sj_f + kk];
            }
        }
    }
    rc[I*si_c + J*sj_c + K] = s;
}

// ---- C++ wrappers ---------------------------------------------------

void restrict_residual_2d_cuda(at::Tensor r, at::Tensor rc) {
    TORCH_CHECK(r.is_contiguous(),  "restrict_residual_2d: r must be contiguous");
    TORCH_CHECK(rc.is_contiguous(), "restrict_residual_2d: rc must be contiguous");
    TORCH_CHECK(r.device().is_cuda(),  "restrict_residual_2d: tensors must be on CUDA");
    TORCH_CHECK(r.dim() == 2 && rc.dim() == 2, "restrict_residual_2d: tensors must be 2-D");
    const int Nx_f = (int)r.size(0), Ny_f = (int)r.size(1);
    const int Nx_c = (int)rc.size(0), Ny_c = (int)rc.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blk(16, 16);
    const dim3 grd(cdiv_t(Ny_c, 16), cdiv_t(Nx_c, 16));
    AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "restrict_residual_2d", [&] {
        restrict_residual_2d_kernel<scalar_t><<<grd, blk, 0, stream>>>(
            r.data_ptr<scalar_t>(), rc.data_ptr<scalar_t>(),
            Nx_f, Ny_f, Nx_c, Ny_c);
    });
}

void restrict_residual_3d_cuda(at::Tensor r, at::Tensor rc) {
    TORCH_CHECK(r.is_contiguous(),  "restrict_residual_3d: r must be contiguous");
    TORCH_CHECK(rc.is_contiguous(), "restrict_residual_3d: rc must be contiguous");
    TORCH_CHECK(r.device().is_cuda(),  "restrict_residual_3d: tensors must be on CUDA");
    TORCH_CHECK(r.dim() == 3 && rc.dim() == 3, "restrict_residual_3d: tensors must be 3-D");
    const int Nx_f = (int)r.size(0), Ny_f = (int)r.size(1), Nz_f = (int)r.size(2);
    const int Nx_c = (int)rc.size(0), Ny_c = (int)rc.size(1), Nz_c = (int)rc.size(2);
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blk(8, 8, 8);
    const dim3 grd(cdiv_t(Nz_c, 8), cdiv_t(Ny_c, 8), cdiv_t(Nx_c, 8));
    AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "restrict_residual_3d", [&] {
        restrict_residual_3d_kernel<scalar_t><<<grd, blk, 0, stream>>>(
            r.data_ptr<scalar_t>(), rc.data_ptr<scalar_t>(),
            Nx_f, Ny_f, Nz_f, Nx_c, Ny_c, Nz_c);
    });
}

// =====================================================================
// Face restriction (WaterLily convention)
//
// Fine-face array along dim d has shape:
//   2-D: dim 0 face  -> (Nx_f+1, Ny_f)  but stored as (NF0, NT1)
//   2-D: dim 1 face  -> (Nx_f, Ny_f+1)
//   3-D: dim 0 face  -> (NF0, NT1, NT2)
//   3-D: dim 1 face  -> (NT0, NF1, NT2)
//   3-D: dim 2 face  -> (NT0, NT1, NF2)
// where NFd is "face count along d" and NTd is "transverse cell count along d".
//
// Coarse-face shape per axis:
//   face axis d: NFc_d = ceil(NFf_d / 2)  (stride-2 keeps every other face)
//   transverse:  NTc_t = NTf_t / 2        (after sum-of-pairs; uses min())
//
// Concrete coarse-cell formula (FACE_DIM=0, 2-D):
//   ch_c[I, J] = 0.5 * (ch[2I, 2J] + ch[2I, 2J+1])
//
// We pass coarse shape explicitly to allow clamping on odd inputs.
// =====================================================================

template <typename scalar_t, int FACE_DIM>
__global__ void restrict_face_2d_kernel(
        const scalar_t* __restrict__ src,
        scalar_t* __restrict__ dst,
        int Nf0, int Nf1,     // fine src shape
        int Nc0, int Nc1)     // coarse dst shape
{
    const int J = blockIdx.x * blockDim.x + threadIdx.x;
    const int I = blockIdx.y * blockDim.y + threadIdx.y;
    if (I >= Nc0 || J >= Nc1) return;

    // FACE_DIM = face direction (the dim that uses stride-2, NOT sum).
    int i_src, j_src_lo, j_src_hi;
    int i_src_other;
    scalar_t s;
    if (FACE_DIM == 0) {
        // face axis is dim 0 (rows). dim 0: stride-2. dim 1: sum pairs.
        i_src = 2 * I;
        j_src_lo = 2 * J;
        j_src_hi = 2 * J + 1;
        const scalar_t a = src[i_src * Nf1 + j_src_lo];
        const scalar_t b = (j_src_hi < Nf1) ? src[i_src * Nf1 + j_src_hi] : (scalar_t)0;
        s = a + b;
        (void)i_src_other;
    } else {
        // FACE_DIM == 1. dim 0: sum pairs. dim 1: stride-2.
        const int i_lo = 2 * I;
        const int i_hi = 2 * I + 1;
        const int j = 2 * J;
        const scalar_t a = src[i_lo * Nf1 + j];
        const scalar_t b = (i_hi < Nf0) ? src[i_hi * Nf1 + j] : (scalar_t)0;
        s = a + b;
    }

    dst[I * Nc1 + J] = (scalar_t)0.5 * s;
}

// 3-D face restriction. FACE_DIM = which axis is "face direction" (stride-2 only).
// Other two axes do sum-of-pairs.
template <typename scalar_t, int FACE_DIM>
__global__ void restrict_face_3d_kernel(
        const scalar_t* __restrict__ src,
        scalar_t* __restrict__ dst,
        int Nf0, int Nf1, int Nf2,    // fine shape
        int Nc0, int Nc1, int Nc2)    // coarse shape
{
    const int K = blockIdx.x * blockDim.x + threadIdx.x;
    const int J = blockIdx.y * blockDim.y + threadIdx.y;
    const int I = blockIdx.z * blockDim.z + threadIdx.z;
    if (I >= Nc0 || J >= Nc1 || K >= Nc2) return;

    const int sj_f = Nf2;
    const int si_f = Nf1 * Nf2;
    const int sj_c = Nc2;
    const int si_c = Nc1 * Nc2;

    // Children offsets: stride-2 in FACE_DIM, sum {0,1} in others.
    int i_lo = 2*I, j_lo = 2*J, k_lo = 2*K;
    int ni = 2, nj = 2, nk = 2;
    if (FACE_DIM == 0) { ni = 1; }
    else if (FACE_DIM == 1) { nj = 1; }
    else { nk = 1; }

    scalar_t s = 0;
    for (int di = 0; di < ni; ++di) {
        const int ii = i_lo + di;
        if (ii >= Nf0) continue;
        for (int dj = 0; dj < nj; ++dj) {
            const int jj = j_lo + dj;
            if (jj >= Nf1) continue;
            for (int dk = 0; dk < nk; ++dk) {
                const int kk = k_lo + dk;
                if (kk >= Nf2) continue;
                s += src[ii*si_f + jj*sj_f + kk];
            }
        }
    }
    dst[I*si_c + J*sj_c + K] = (scalar_t)0.5 * s;
}

// ---- C++ wrappers ---------------------------------------------------

void restrict_face_2d_cuda(at::Tensor src, at::Tensor dst, int64_t face_dim) {
    TORCH_CHECK(src.is_contiguous(), "restrict_face_2d: src must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "restrict_face_2d: dst must be contiguous");
    TORCH_CHECK(src.device().is_cuda(), "restrict_face_2d: tensors must be on CUDA");
    TORCH_CHECK(src.dim() == 2 && dst.dim() == 2, "restrict_face_2d: tensors must be 2-D");
    TORCH_CHECK(face_dim == 0 || face_dim == 1, "restrict_face_2d: face_dim must be 0 or 1");
    const int Nf0 = (int)src.size(0), Nf1 = (int)src.size(1);
    const int Nc0 = (int)dst.size(0), Nc1 = (int)dst.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blk(16, 16);
    const dim3 grd(cdiv_t(Nc1, 16), cdiv_t(Nc0, 16));
    AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "restrict_face_2d", [&] {
        if (face_dim == 0) {
            restrict_face_2d_kernel<scalar_t, 0><<<grd, blk, 0, stream>>>(
                src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(),
                Nf0, Nf1, Nc0, Nc1);
        } else {
            restrict_face_2d_kernel<scalar_t, 1><<<grd, blk, 0, stream>>>(
                src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(),
                Nf0, Nf1, Nc0, Nc1);
        }
    });
}

void restrict_face_3d_cuda(at::Tensor src, at::Tensor dst, int64_t face_dim) {
    TORCH_CHECK(src.is_contiguous(), "restrict_face_3d: src must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "restrict_face_3d: dst must be contiguous");
    TORCH_CHECK(src.device().is_cuda(), "restrict_face_3d: tensors must be on CUDA");
    TORCH_CHECK(src.dim() == 3 && dst.dim() == 3, "restrict_face_3d: tensors must be 3-D");
    TORCH_CHECK(face_dim == 0 || face_dim == 1 || face_dim == 2,
                "restrict_face_3d: face_dim must be 0, 1 or 2");
    const int Nf0 = (int)src.size(0), Nf1 = (int)src.size(1), Nf2 = (int)src.size(2);
    const int Nc0 = (int)dst.size(0), Nc1 = (int)dst.size(1), Nc2 = (int)dst.size(2);
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blk(8, 8, 8);
    const dim3 grd(cdiv_t(Nc2, 8), cdiv_t(Nc1, 8), cdiv_t(Nc0, 8));
    AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "restrict_face_3d", [&] {
        if (face_dim == 0) {
            restrict_face_3d_kernel<scalar_t, 0><<<grd, blk, 0, stream>>>(
                src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(),
                Nf0, Nf1, Nf2, Nc0, Nc1, Nc2);
        } else if (face_dim == 1) {
            restrict_face_3d_kernel<scalar_t, 1><<<grd, blk, 0, stream>>>(
                src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(),
                Nf0, Nf1, Nf2, Nc0, Nc1, Nc2);
        } else {
            restrict_face_3d_kernel<scalar_t, 2><<<grd, blk, 0, stream>>>(
                src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(),
                Nf0, Nf1, Nf2, Nc0, Nc1, Nc2);
        }
    });
}

// =====================================================================
// Prolongation + correction (fused add into ghost-padded p)
//
//   F.interpolate(align_corners=False, bilinear/trilinear) on
//   err_c[1:-1, 1:-1[, 1:-1]] to shape (Nx_f, Ny_f[, Nz_f]), then
//   p[1+i, 1+j[, 1+k]] += interp.
//
// Cell-centred mapping for each dim:
//   src = (dst + 0.5) * Nc/Nf - 0.5
//   src_l = clamp(floor(src), 0, Nc-1)
//   src_r = clamp(src_l + 1, 0, Nc-1)
//   w_r   = clamp(src - floor(src), 0, 1)   (== src - src_l if src in [0,Nc-1])
//   w_l   = 1 - w_r
//
// Here Nc = interior coarse size (= err_coarse.shape[d] - 2), Nf = fine
// interior size. Source tensor is the ghost-padded err_coarse (so we read
// at [1 + src_l/r, ...]).
// =====================================================================

// Computed in scalar_t -- the tensor dtype the user selected -- like every other
// kernel here.  This used to be hardcoded to double to bit-match torch's
// align_corners=False interpolation (which uses opmath_t = double for f32 src).
// That made a 4-tap bilinear interp cost more than an rbgs sweep on consumer
// GPUs, where FP64 runs at 1/64 of FP32 (measured: 200 us vs 11 us at 1024^2).
// It is also safe to drop: for multigrid coarsening Nc/Nf == 1/2 exactly, so
// `src` and the weights are exact binary fractions (0.25 / 0.75) in fp32 too --
// only the 4-tap accumulation rounds differently, at ~1 ulp on a V-cycle
// *correction*.
template <typename scalar_t>
__device__ __forceinline__ void linear_weights(
        int dst, int Nc, int Nf,
        int& il, int& ir, scalar_t& wl, scalar_t& wr)
{
    const scalar_t src = ((scalar_t)dst + (scalar_t)0.5)
                       * ((scalar_t)Nc / (scalar_t)Nf) - (scalar_t)0.5;
    scalar_t sf = ::floor(src);
    il = (int)sf;
    ir = il + 1;
    if (il < 0) { il = 0; }
    if (ir < 0) { ir = 0; }
    if (il > Nc - 1) { il = Nc - 1; }
    if (ir > Nc - 1) { ir = Nc - 1; }
    scalar_t w = src - sf;     // fractional part in [0,1)
    if (w < (scalar_t)0) w = (scalar_t)0;
    if (w > (scalar_t)1) w = (scalar_t)1;
    // If src is below domain or above, weights collapse to corner cell.
    if (src <= (scalar_t)0) { wl = (scalar_t)1; wr = (scalar_t)0; }
    else if (src >= (scalar_t)Nc - (scalar_t)1) { wl = (scalar_t)0; wr = (scalar_t)1; }
    else { wl = (scalar_t)1 - w; wr = w; }
}

template <typename scalar_t>
__global__ void prolongate_add_2d_kernel(
        const scalar_t* __restrict__ ec,    // err_coarse, ghost-padded (Nx_c+2, Ny_c+2)
        scalar_t* __restrict__ p,            // ghost-padded fine p (Nx_f+2, Ny_f+2)
        int Nx_c, int Ny_c,                  // interior coarse sizes
        int Nx_f, int Ny_f)                  // interior fine sizes
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx_f || j >= Ny_f) return;

    int il, ir, jl, jr;
    scalar_t wil, wir, wjl, wjr;
    linear_weights<scalar_t>(i, Nx_c, Nx_f, il, ir, wil, wir);
    linear_weights<scalar_t>(j, Ny_c, Ny_f, jl, jr, wjl, wjr);

    // Read ec at [1+il, 1+jl], [1+il, 1+jr], [1+ir, 1+jl], [1+ir, 1+jr]
    const int se = Ny_c + 2;   // ghost-padded coarse row stride
    const scalar_t v_ll = ec[(il + 1) * se + (jl + 1)];
    const scalar_t v_lr = ec[(il + 1) * se + (jr + 1)];
    const scalar_t v_rl = ec[(ir + 1) * se + (jl + 1)];
    const scalar_t v_rr = ec[(ir + 1) * se + (jr + 1)];

    const scalar_t interp = wil * wjl * v_ll
                          + wil * wjr * v_lr
                          + wir * wjl * v_rl
                          + wir * wjr * v_rr;

    const int sp = Ny_f + 2;   // fine padded row stride
    p[(i + 1) * sp + (j + 1)] += interp;
}

template <typename scalar_t>
__global__ void prolongate_add_3d_kernel(
        const scalar_t* __restrict__ ec,
        scalar_t* __restrict__ p,
        int Nx_c, int Ny_c, int Nz_c,
        int Nx_f, int Ny_f, int Nz_f)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= Nx_f || j >= Ny_f || k >= Nz_f) return;

    int il, ir, jl, jr, kl, kr;
    scalar_t wil, wir, wjl, wjr, wkl, wkr;
    linear_weights<scalar_t>(i, Nx_c, Nx_f, il, ir, wil, wir);
    linear_weights<scalar_t>(j, Ny_c, Ny_f, jl, jr, wjl, wjr);
    linear_weights<scalar_t>(k, Nz_c, Nz_f, kl, kr, wkl, wkr);

    const int sje = Nz_c + 2;
    const int sie = (Ny_c + 2) * (Nz_c + 2);

    scalar_t interp = (scalar_t)0;
    #pragma unroll
    for (int di = 0; di < 2; ++di) {
        const int ii = (di == 0) ? il : ir;
        const scalar_t wi = (di == 0) ? wil : wir;
        #pragma unroll
        for (int dj = 0; dj < 2; ++dj) {
            const int jj = (dj == 0) ? jl : jr;
            const scalar_t wj = (dj == 0) ? wjl : wjr;
            #pragma unroll
            for (int dk = 0; dk < 2; ++dk) {
                const int kk = (dk == 0) ? kl : kr;
                const scalar_t wk = (dk == 0) ? wkl : wkr;
                interp += wi * wj * wk
                        * ec[(ii + 1) * sie + (jj + 1) * sje + (kk + 1)];
            }
        }
    }

    const int sjp = Nz_f + 2;
    const int sip = (Ny_f + 2) * (Nz_f + 2);
    p[(i + 1) * sip + (j + 1) * sjp + (k + 1)] += interp;
}

// ---- C++ wrappers ---------------------------------------------------

void prolongate_add_2d_cuda(at::Tensor ec, at::Tensor p) {
    TORCH_CHECK(ec.is_contiguous(), "prolongate_add_2d: ec must be contiguous");
    TORCH_CHECK(p.is_contiguous(),  "prolongate_add_2d: p must be contiguous");
    TORCH_CHECK(ec.device().is_cuda(), "prolongate_add_2d: tensors must be on CUDA");
    TORCH_CHECK(ec.dim() == 2 && p.dim() == 2, "prolongate_add_2d: tensors must be 2-D");
    const int Nx_c = (int)ec.size(0) - 2;
    const int Ny_c = (int)ec.size(1) - 2;
    const int Nx_f = (int)p.size(0) - 2;
    const int Ny_f = (int)p.size(1) - 2;
    TORCH_CHECK(Nx_c >= 1 && Ny_c >= 1, "prolongate_add_2d: coarse interior must be >= 1");
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blk(32, 8);
    const dim3 grd(cdiv_t(Ny_f, 32), cdiv_t(Nx_f, 8));
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "prolongate_add_2d", [&] {
        prolongate_add_2d_kernel<scalar_t><<<grd, blk, 0, stream>>>(
            ec.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(),
            Nx_c, Ny_c, Nx_f, Ny_f);
    });
}

void prolongate_add_3d_cuda(at::Tensor ec, at::Tensor p) {
    TORCH_CHECK(ec.is_contiguous(), "prolongate_add_3d: ec must be contiguous");
    TORCH_CHECK(p.is_contiguous(),  "prolongate_add_3d: p must be contiguous");
    TORCH_CHECK(ec.device().is_cuda(), "prolongate_add_3d: tensors must be on CUDA");
    TORCH_CHECK(ec.dim() == 3 && p.dim() == 3, "prolongate_add_3d: tensors must be 3-D");
    const int Nx_c = (int)ec.size(0) - 2;
    const int Ny_c = (int)ec.size(1) - 2;
    const int Nz_c = (int)ec.size(2) - 2;
    const int Nx_f = (int)p.size(0) - 2;
    const int Ny_f = (int)p.size(1) - 2;
    const int Nz_f = (int)p.size(2) - 2;
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blk(16, 8, 4);
    const dim3 grd(cdiv_t(Nz_f, 16), cdiv_t(Ny_f, 8), cdiv_t(Nx_f, 4));
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "prolongate_add_3d", [&] {
        prolongate_add_3d_kernel<scalar_t><<<grd, blk, 0, stream>>>(
            ec.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(),
            Nx_c, Ny_c, Nz_c, Nx_f, Ny_f, Nz_f);
    });
}

// =====================================================================
// Full-weighting residual restriction  =  transpose of prolongate_add
//
// The plain sum-of-children restriction above is NOT the transpose of the
// (bi/tri)linear prolongation, so the multigrid V-cycle built from that pair
// is a NON-SYMMETRIC operator — fine for a stationary V-cycle, but INVALID as
// a preconditioner for CG (mgcg/rmgcg), which needs an SPD M.  This kernel is
// the EXACT adjoint P^T: every fine cell scatters its residual to the same 8
// (4 in 2-D) coarse corners with the SAME linear_weights() the prolongation
// gathers with, so R = P^T holds to rounding, INCLUDING the align_corners=False
// edge clamping (il==ir collapse contributes wl+wr=1, matching P reading the
// same coarse cell twice).  ``rc`` (coarse interior) is zeroed by the wrapper;
// the scatter accumulates with atomicAdd.
// =====================================================================

template <typename scalar_t>
__global__ void restrict_fw_2d_kernel(
        const scalar_t* __restrict__ r,     // fine interior (Nx_f, Ny_f)
        scalar_t* __restrict__ rc,           // coarse interior (Nx_c, Ny_c), pre-zeroed
        int Nx_c, int Ny_c,
        int Nx_f, int Ny_f)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx_f || j >= Ny_f) return;

    int il, ir, jl, jr;
    scalar_t wil, wir, wjl, wjr;
    linear_weights<scalar_t>(i, Nx_c, Nx_f, il, ir, wil, wir);
    linear_weights<scalar_t>(j, Ny_c, Ny_f, jl, jr, wjl, wjr);

    const scalar_t val = r[i * Ny_f + j];
    const int ii[2] = {il, ir};  const scalar_t wi[2] = {wil, wir};
    const int jj[2] = {jl, jr};  const scalar_t wj[2] = {wjl, wjr};
    #pragma unroll
    for (int a = 0; a < 2; ++a)
        #pragma unroll
        for (int b = 0; b < 2; ++b)
            atomicAdd(&rc[ii[a] * Ny_c + jj[b]], wi[a] * wj[b] * val);
}

template <typename scalar_t>
__global__ void restrict_fw_3d_kernel(
        const scalar_t* __restrict__ r,
        scalar_t* __restrict__ rc,
        int Nx_c, int Ny_c, int Nz_c,
        int Nx_f, int Ny_f, int Nz_f)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= Nx_f || j >= Ny_f || k >= Nz_f) return;

    int il, ir, jl, jr, kl, kr;
    scalar_t wil, wir, wjl, wjr, wkl, wkr;
    linear_weights<scalar_t>(i, Nx_c, Nx_f, il, ir, wil, wir);
    linear_weights<scalar_t>(j, Ny_c, Ny_f, jl, jr, wjl, wjr);
    linear_weights<scalar_t>(k, Nz_c, Nz_f, kl, kr, wkl, wkr);

    const int sj_c = Nz_c;
    const int si_c = Ny_c * Nz_c;
    const scalar_t val = r[(i * Ny_f + j) * Nz_f + k];

    const int ii[2] = {il, ir};  const scalar_t wi[2] = {wil, wir};
    const int jj[2] = {jl, jr};  const scalar_t wj[2] = {wjl, wjr};
    const int kk[2] = {kl, kr};  const scalar_t wk[2] = {wkl, wkr};
    #pragma unroll
    for (int a = 0; a < 2; ++a)
        #pragma unroll
        for (int b = 0; b < 2; ++b)
            #pragma unroll
            for (int c = 0; c < 2; ++c)
                atomicAdd(&rc[ii[a] * si_c + jj[b] * sj_c + kk[c]],
                          wi[a] * wj[b] * wk[c] * val);
}

void restrict_fw_2d_cuda(at::Tensor r, at::Tensor rc) {
    TORCH_CHECK(r.is_contiguous() && rc.is_contiguous(),
                "restrict_fw_2d: tensors must be contiguous");
    TORCH_CHECK(r.device().is_cuda(), "restrict_fw_2d: tensors must be on CUDA");
    TORCH_CHECK(r.dim() == 2 && rc.dim() == 2, "restrict_fw_2d: tensors must be 2-D");
    const int Nx_f = (int)r.size(0),  Ny_f = (int)r.size(1);
    const int Nx_c = (int)rc.size(0), Ny_c = (int)rc.size(1);
    rc.zero_();
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blk(32, 8);
    const dim3 grd(cdiv_t(Ny_f, 32), cdiv_t(Nx_f, 8));
    AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "restrict_fw_2d", [&] {
        restrict_fw_2d_kernel<scalar_t><<<grd, blk, 0, stream>>>(
            r.data_ptr<scalar_t>(), rc.data_ptr<scalar_t>(),
            Nx_c, Ny_c, Nx_f, Ny_f);
    });
}

void restrict_fw_3d_cuda(at::Tensor r, at::Tensor rc) {
    TORCH_CHECK(r.is_contiguous() && rc.is_contiguous(),
                "restrict_fw_3d: tensors must be contiguous");
    TORCH_CHECK(r.device().is_cuda(), "restrict_fw_3d: tensors must be on CUDA");
    TORCH_CHECK(r.dim() == 3 && rc.dim() == 3, "restrict_fw_3d: tensors must be 3-D");
    const int Nx_f = (int)r.size(0),  Ny_f = (int)r.size(1),  Nz_f = (int)r.size(2);
    const int Nx_c = (int)rc.size(0), Ny_c = (int)rc.size(1), Nz_c = (int)rc.size(2);
    rc.zero_();
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blk(16, 8, 4);
    const dim3 grd(cdiv_t(Nz_f, 16), cdiv_t(Ny_f, 8), cdiv_t(Nx_f, 4));
    AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "restrict_fw_3d", [&] {
        restrict_fw_3d_kernel<scalar_t><<<grd, blk, 0, stream>>>(
            r.data_ptr<scalar_t>(), rc.data_ptr<scalar_t>(),
            Nx_c, Ny_c, Nz_c, Nx_f, Ny_f, Nz_f);
    });
}

// ---- CUDA dispatch registration -------------------------------------
TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("restrict_residual_2d", &restrict_residual_2d_cuda);
    m.impl("restrict_residual_3d", &restrict_residual_3d_cuda);
    m.impl("restrict_face_2d",     &restrict_face_2d_cuda);
    m.impl("restrict_face_3d",     &restrict_face_3d_cuda);
    m.impl("prolongate_add_2d",    &prolongate_add_2d_cuda);
    m.impl("prolongate_add_3d",    &prolongate_add_3d_cuda);
}

}  // namespace lilytorch_kernels
