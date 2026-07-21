// =====================================================================
//  bcs.cu — Fused boundary-condition kernels (2-D and 3-D).
//
//  Contains apply_bcs_{2,3}d: Neumann copies, Dirichlet direct writes,
//  and reflective (tangential Dirichlet) writes, all in a single fused
//  kernel per dimension.
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

// ---- 2-D apply_bcs kernel + launcher ----------------------------------------
// =====================================================================
//  apply_bcs_2d (CUDA)
//
//  2-D analogue of ``apply_bcs_3d``: writes one ghost line per op
//  (Neumann copy, Dirichlet constant, or reflective).  One launch per op
//  kind; the write-ownership rule in bc_ops.h keeps the corner ghosts
//  race-free and deterministic.  See CPU impl for the argument layout.
// =====================================================================

template <typename scalar_t>
__global__ void apply_bcs_2d_kernel(
    scalar_t* __restrict__ u,
    scalar_t* __restrict__ v,
    const int64_t* __restrict__ shapes,
    const int kind,
    const int* __restrict__ desc,
    const scalar_t* __restrict__ vals,   // null for Neumann
    const int nops)
{
    using namespace lilytorch_kernels::bcs;

    const int op = blockIdx.y;
    if (op >= nops) return;

    int comp, axis, dst_along, src_along;
    bc_decode(kind, desc, op, shapes, /*ndim=*/2, comp, axis, dst_along, src_along);

    const int Nx = (int)shapes[comp * 2 + 0];
    const int Ny = (int)shapes[comp * 2 + 1];
    const int dim0_max = (axis == 0) ? Ny : Nx;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim0_max) return;

    int c[2];
    if (axis == 0) { c[0] = dst_along; c[1] = i;         }
    else           { c[0] = i;         c[1] = dst_along; }

    int s[2];
    if (!bc_own_and_source(kind, desc, nops, op, shapes, /*ndim=*/2,
                           comp, axis, src_along, c, s))
        return;                                   // a lower-indexed op owns it

    scalar_t* base = (comp == 0) ? u : v;
    const int64_t dst_lin = (int64_t)c[0] * Ny + c[1];
    const int64_t src_lin = (int64_t)s[0] * Ny + s[1];

    if      (kind == BC_KIND_NEUMANN)   base[dst_lin] = base[src_lin];
    else if (kind == BC_KIND_DIRICHLET) base[dst_lin] = vals[op];
    else base[dst_lin] = scalar_t(2) * vals[op] - base[src_lin];
}

void apply_bcs_2d_cuda(
    at::Tensor u, at::Tensor v,
    const at::Tensor& shapes,
    const at::Tensor& neu_desc,
    const at::Tensor& dir_desc,
    const at::Tensor& dir_val,
    const at::Tensor& ref_desc,
    const at::Tensor& ref_val,
    const int64_t max_line_dim)
{
    TORCH_CHECK(u.is_cuda() && v.is_cuda(),
                "apply_bcs_2d_cuda: u/v must be CUDA tensors");
    TORCH_CHECK(u.is_contiguous() && v.is_contiguous(),
                "apply_bcs_2d_cuda: u/v must be contiguous");
    TORCH_CHECK(u.scalar_type() == v.scalar_type(),
                "apply_bcs_2d_cuda: u/v must share dtype");
    TORCH_CHECK(shapes.scalar_type() == at::kLong &&
                shapes.dim() == 2 && shapes.size(0) == 2 && shapes.size(1) == 2,
                "apply_bcs_2d_cuda: shapes must be int64[2,2]");
    TORCH_CHECK(ref_desc.scalar_type() == at::kInt && ref_desc.dim() == 2 &&
                ref_desc.size(1) == 4,
                "apply_bcs_2d_cuda: ref_desc must be int32[N,4]");

    const int N_neu = (int)neu_desc.size(0);
    const int N_dir = (int)dir_desc.size(0);
    const int N_ref = (int)ref_desc.size(0);
    if (N_neu + N_dir + N_ref == 0 || max_line_dim <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blockX = 256;
    const int gridX  = (int)((max_line_dim + blockX - 1) / blockX);

    // One launch per op kind, in the order the eager reference applies them:
    //   Neumann → Dirichlet → reflective.
    // The stage boundaries define the cross-kind overlaps (a reflective op
    // reads the adjacent cell AFTER any Dirichlet wall write to it, a Neumann
    // op BEFORE — as on CPU); within a stage, bc_own_and_source() picks a
    // single writer per cell.
    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_2d_cuda", [&] {
        using namespace lilytorch_kernels::bcs;
        auto launch = [&](int kind, const at::Tensor& desc,
                          const scalar_t* vals, int nops) {
            if (nops == 0) return;
            apply_bcs_2d_kernel<scalar_t>
                <<<dim3(gridX, nops, 1), dim3(blockX, 1, 1), 0, stream>>>(
                    u.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                    shapes.data_ptr<int64_t>(),
                    kind, desc.data_ptr<int>(), vals, nops);
        };
        launch(BC_KIND_NEUMANN, neu_desc, nullptr, N_neu);
        launch(BC_KIND_DIRICHLET, dir_desc,
               (N_dir > 0) ? dir_val.data_ptr<scalar_t>() : nullptr, N_dir);
        launch(BC_KIND_REFLECTIVE, ref_desc,
               (N_ref > 0) ? ref_val.data_ptr<scalar_t>() : nullptr, N_ref);
    });
}


// ---- 3-D apply_bcs kernel + launcher ----------------------------------------
// =====================================================================
//  apply_bcs_3d  (Phase H: combined boundary-condition writes)
//
//  Replaces the Python loop in AdvDiffSolver.set_BCs that issues 18
//  ghost-face slice copies (Neumann) plus a handful of Dirichlet
//  overwrites per call.  All ops of one kind are packed into a single
//  kernel launch, eliminating per-op dispatch overhead.
//
//  One launch per op KIND (Neumann → Dirichlet → reflective).  Within a
//  launch the ops of that stage run concurrently, and the write-ownership
//  rule in bc_ops.h keeps edge/corner ghosts race-free and deterministic.
//  See that header for the full argument.
// =====================================================================

template <typename scalar_t>
__global__ void apply_bcs_3d_kernel(
    scalar_t* __restrict__ u,
    scalar_t* __restrict__ v,
    scalar_t* __restrict__ w,
    const int64_t* __restrict__ shapes,
    const int kind,
    const int* __restrict__ desc,
    const scalar_t* __restrict__ vals,   // null for Neumann
    const int nops)
{
    using namespace lilytorch_kernels::bcs;

    const int op = blockIdx.z;
    if (op >= nops) return;

    int comp, axis, dst_along, src_along;
    bc_decode(kind, desc, op, shapes, /*ndim=*/3, comp, axis, dst_along, src_along);

    const int Nx = (int)shapes[comp * 3 + 0];
    const int Ny = (int)shapes[comp * 3 + 1];
    const int Nz = (int)shapes[comp * 3 + 2];

    int dim0_max, dim1_max;
    if      (axis == 0) { dim0_max = Ny; dim1_max = Nz; }
    else if (axis == 1) { dim0_max = Nx; dim1_max = Nz; }
    else                { dim0_max = Nx; dim1_max = Ny; }

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= dim0_max || j >= dim1_max) return;

    // Destination cell, in full 3-D coordinates.
    int c[3];
    if      (axis == 0) { c[0] = dst_along; c[1] = i;         c[2] = j;         }
    else if (axis == 1) { c[0] = i;         c[1] = dst_along; c[2] = j;         }
    else                { c[0] = i;         c[1] = j;         c[2] = dst_along; }

    int s[3];
    if (!bc_own_and_source(kind, desc, nops, op, shapes, /*ndim=*/3,
                           comp, axis, src_along, c, s))
        return;                                   // a lower-indexed op owns it

    scalar_t* base = (comp == 0) ? u : (comp == 1 ? v : w);
    const int64_t s1 = (int64_t)Ny * Nz;
    const int64_t s2 = (int64_t)Nz;

    const int64_t dst_lin = (int64_t)c[0] * s1 + (int64_t)c[1] * s2 + c[2];
    const int64_t src_lin = (int64_t)s[0] * s1 + (int64_t)s[1] * s2 + s[2];

    if      (kind == BC_KIND_NEUMANN)   base[dst_lin] = base[src_lin];
    else if (kind == BC_KIND_DIRICHLET) base[dst_lin] = vals[op];
    else base[dst_lin] = scalar_t(2) * vals[op] - base[src_lin];
}

void apply_bcs_3d_cuda(
    at::Tensor u, at::Tensor v, at::Tensor w,
    const at::Tensor& shapes,
    const at::Tensor& neu_desc,
    const at::Tensor& dir_desc,
    const at::Tensor& dir_val,
    const at::Tensor& ref_desc,
    const at::Tensor& ref_val,
    const int64_t max_dim0,
    const int64_t max_dim1)
{
    TORCH_CHECK(u.is_cuda() && v.is_cuda() && w.is_cuda(),
        "apply_bcs_3d: u/v/w must be CUDA tensors");
    TORCH_CHECK(u.is_contiguous() && v.is_contiguous() && w.is_contiguous(),
        "apply_bcs_3d: u/v/w must be contiguous");
    TORCH_CHECK(u.scalar_type() == v.scalar_type() &&
                u.scalar_type() == w.scalar_type(),
        "apply_bcs_3d: u/v/w must share dtype");
    TORCH_CHECK(shapes.scalar_type() == at::kLong &&
                shapes.dim() == 2 && shapes.size(0) == 3 && shapes.size(1) == 3,
        "apply_bcs_3d: shapes must be int64[3,3]");
    TORCH_CHECK(neu_desc.scalar_type() == at::kInt && neu_desc.dim() == 2 &&
                neu_desc.size(1) == 3,
        "apply_bcs_3d: neu_desc must be int32[N,3]");
    TORCH_CHECK(dir_desc.scalar_type() == at::kInt && dir_desc.dim() == 2 &&
                dir_desc.size(1) == 3,
        "apply_bcs_3d: dir_desc must be int32[N,3]");
    TORCH_CHECK(ref_desc.scalar_type() == at::kInt && ref_desc.dim() == 2 &&
                ref_desc.size(1) == 4,
        "apply_bcs_3d: ref_desc must be int32[N,4]");

    const int N_neu = (int)neu_desc.size(0);
    const int N_dir = (int)dir_desc.size(0);
    const int N_ref = (int)ref_desc.size(0);
    if (N_neu + N_dir + N_ref == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int TILE = 16;
    // Use a rectangular (blocks_x × blocks_y) grid: blocks_x covers the
    // i-dimension (max_dim0) and blocks_y covers the j-dimension (max_dim1).
    // This avoids the (max_plane_dim)² square launch that wastes ~75 % of
    // thread blocks when max_dim0 >> max_dim1 (e.g. Nx-long y/z faces on a
    // 4:1:1 grid where max_dim0 = Nx but max_dim1 = Nz = Nx/4).
    const int blocks_x = (int)((max_dim0 + TILE - 1) / TILE);
    const int blocks_y = (int)((max_dim1 + TILE - 1) / TILE);
    const dim3 block(TILE, TILE, 1);

    // One launch per op kind, in the order the eager reference applies them:
    //   Neumann → Dirichlet → reflective.
    // The stage boundaries are what make the cross-kind overlaps well defined
    // (a reflective op reads the adjacent cell AFTER any Dirichlet wall write
    // to it; a Neumann op reads it BEFORE — as on CPU).  Within a stage,
    // bc_own_and_source() picks a single writer per cell.
    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_3d_cuda", [&] {
        using namespace lilytorch_kernels::bcs;
        auto launch = [&](int kind, const at::Tensor& desc,
                          const scalar_t* vals, int nops) {
            if (nops == 0) return;
            const dim3 grid((unsigned)blocks_x, (unsigned)blocks_y, (unsigned)nops);
            apply_bcs_3d_kernel<scalar_t><<<grid, block, 0, stream>>>(
                u.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
                shapes.data_ptr<int64_t>(),
                kind, desc.data_ptr<int>(), vals, nops);
        };
        launch(BC_KIND_NEUMANN, neu_desc, nullptr, N_neu);
        launch(BC_KIND_DIRICHLET, dir_desc,
               (N_dir > 0) ? dir_val.data_ptr<scalar_t>() : nullptr, N_dir);
        launch(BC_KIND_REFLECTIVE, ref_desc,
               (N_ref > 0) ? ref_val.data_ptr<scalar_t>() : nullptr, N_ref);
    });
}



TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("apply_bcs_3d", &apply_bcs_3d_cuda);
    m.impl("apply_bcs_2d", &apply_bcs_2d_cuda);
}

}  // namespace lilytorch_kernels
