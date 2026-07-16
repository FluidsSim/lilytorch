// =====================================================================
//  streaming_sdf_cpu.cpp
//
//  CPU implementations of the lilytorch_kernels operators. These mirror
//  the CUDA kernels in cuda/streaming_sdf.cu line-for-line so that the
//  ops can run on hosts without CUDA (testing, debugging, CPU-only
//  builds via LILYTORCH_NO_CUDA=1). The hot loops are parallelised with
//  ``at::parallel_for`` (PyTorch's intra-op thread pool) instead of
//  OpenMP. This keeps the source compatible across PyTorch versions —
//  some GCC + OpenMP combinations rejected ``_Pragma("omp ...")`` when
//  it appears inside an ``AT_DISPATCH_FLOATING_TYPES`` lambda body
//  ("'#pragma' is not allowed here"), and the exact expansion of that
//  macro varies by PyTorch release. Using ``at::parallel_for`` avoids
//  putting any pragmas inside macro-expanded lambdas and is part of
//  ATen's long-stable public API. Float dispatch covers float32 /
//  float64; half precision is intentionally skipped on CPU.
// =====================================================================

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/all.h>
#include <torch/library.h>
#include <c10/util/ArrayRef.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

#include "bc_ops.h"

namespace lilytorch_kernels {
// scattered-point interpolation kernels (CPU)
// ==== 3-D kernels =============================================================



template <typename scalar_t>
static inline scalar_t trilinear_sample_border(
    const scalar_t* F,
    const scalar_t* bx, const scalar_t* by, const scalar_t* bz,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t bx_last, const scalar_t by_last, const scalar_t bz_last,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    const scalar_t inv_vol,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    xq = std::max(bx0, std::min(xq, bx_last));
    yq = std::max(by0, std::min(yq, by_last));
    zq = std::max(bz0, std::min(zq, bz_last));

    int ix = (int)std::floor((xq - bx0) * inv_dx);
    int iy = (int)std::floor((yq - by0) * inv_dy);
    int iz = (int)std::floor((zq - bz0) * inv_dz);
    ix = std::max(0, std::min(ix, Mx - 2));
    iy = std::max(0, std::min(iy, My - 2));
    iz = std::max(0, std::min(iz, Mz - 2));
    const int ixp = ix + 1, iyp = iy + 1, izp = iz + 1;

    const scalar_t wx0 = bx[ixp] - xq, wx1 = xq - bx[ix];
    const scalar_t wy0 = by[iyp] - yq, wy1 = yq - by[iy];
    const scalar_t wz0 = bz[izp] - zq, wz1 = zq - bz[iz];

    const int s2 = Mz;
    const int s1 = My * Mz;

    return (
        wx0 * wy0 * wz0 * F[ix  * s1 + iy  * s2 + iz ]
      + wx0 * wy0 * wz1 * F[ix  * s1 + iy  * s2 + izp]
      + wx0 * wy1 * wz0 * F[ix  * s1 + iyp * s2 + iz ]
      + wx0 * wy1 * wz1 * F[ix  * s1 + iyp * s2 + izp]
      + wx1 * wy0 * wz0 * F[ixp * s1 + iy  * s2 + iz ]
      + wx1 * wy0 * wz1 * F[ixp * s1 + iy  * s2 + izp]
      + wx1 * wy1 * wz0 * F[ixp * s1 + iyp * s2 + iz ]
      + wx1 * wy1 * wz1 * F[ixp * s1 + iyp * s2 + izp]
    ) * inv_vol;
}

// =====================================================================
//  Trilinear sample on a UNIFORM body grid.
//
//  The streaming_sdf_stag_3d_* kernels are exclusively used with body SDF
//  tables built on uniform Cartesian grids (the kernel already takes
//  ``inv_dx``, ``inv_dy``, ``inv_dz``).  In that case the corner
//  weights reduce to ``(1 - frac, frac)`` per axis -- no axis-table
//  lookups, no ``floor`` calls, and the trailing ``* inv_vol`` cancels
//  out (because dx*dy*dz*inv_vol == 1).  This saves 6 axis-table reads
//  + 1 mul per sample (24 reads per cell across the 4 face samples)
//  vs. ``trilinear_sample_border`` and avoids the slow ``floor`` calls
//  on x86 (CPU).
//
//  Clamping is done in body-grid space (after the inv_dx scale) which
//  is one fewer subtraction than clamping ``xq`` against ``bx_last``.
//
//  The factored evaluation form (wx0*(... wy0*(...) + wy1*(...))) cuts
//  the multiply count from ~24 to ~14 vs. the fully-expanded sum that
//  ``trilinear_sample_border`` uses.
// =====================================================================
template <typename scalar_t>
static inline scalar_t trilinear_sample_uniform(
    const scalar_t* F,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    scalar_t tz = (zq - bz0) * inv_dz;
    const scalar_t Mx_lim = (scalar_t)(Mx - 1);
    const scalar_t My_lim = (scalar_t)(My - 1);
    const scalar_t Mz_lim = (scalar_t)(Mz - 1);
    if (tx < (scalar_t)0) tx = (scalar_t)0; else if (tx > Mx_lim) tx = Mx_lim;
    if (ty < (scalar_t)0) ty = (scalar_t)0; else if (ty > My_lim) ty = My_lim;
    if (tz < (scalar_t)0) tz = (scalar_t)0; else if (tz > Mz_lim) tz = Mz_lim;

    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;
    int iz = (int)tz; if (iz > Mz - 2) iz = Mz - 2;

    const scalar_t fx = tx - (scalar_t)ix;
    const scalar_t fy = ty - (scalar_t)iy;
    const scalar_t fz = tz - (scalar_t)iz;
    const scalar_t wx0 = (scalar_t)1 - fx, wx1 = fx;
    const scalar_t wy0 = (scalar_t)1 - fy, wy1 = fy;
    const scalar_t wz0 = (scalar_t)1 - fz, wz1 = fz;

    const int s2   = Mz;
    const int s1   = My * Mz;
    const int base = ix*s1 + iy*s2 + iz;

    return (
        wx0 * (
          wy0 * (wz0 * F[base]                + wz1 * F[base + 1]) +
          wy1 * (wz0 * F[base + s2]           + wz1 * F[base + s2 + 1])
        ) +
        wx1 * (
          wy0 * (wz0 * F[base + s1]           + wz1 * F[base + s1 + 1]) +
          wy1 * (wz0 * F[base + s1 + s2]      + wz1 * F[base + s1 + s2 + 1])
        )
    );
}

// =====================================================================
//  Triquadratic sample on a UNIFORM body grid.
//
//  Lagrange interpolation on a 3x3x3 stencil [ix-1, ix, ix+1]^3 with the
//  same lower-bracketing convention as the trilinear sampler:
//     ix = floor((xq - bx0) * inv_dx)   (clamped to [0, M-2])
//     f  = (xq - bx0) * inv_dx - ix     (in [0, 1])
//  The Lagrange weights for centered grid points x_{ix-1}, x_{ix}, x_{ix+1}
//  reduce on a uniform grid to:
//     w_-1(f) = 0.5 * f * (f - 1)
//     w_ 0(f) = 1 - f * f
//     w_+1(f) = 0.5 * f * (f + 1)
//  and the trilinear corner case (f -> 0 or 1) becomes the trilinear value.
//
//  When the lower stencil neighbour falls outside the body grid on any
//  axis (ix == 0), we fall back to trilinear on that sample to mirror
//  the pytorch_interpolation biquadratic behaviour and avoid sampling
//  invalid memory.  In BDIM SDF tables this only happens for queries
//  that already sit beyond the body's far-field padding, where the SDF
//  is monotone and trilinear is fine.
// =====================================================================
template <typename scalar_t>
static inline scalar_t triquadratic_sample_uniform(
    const scalar_t* F,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    scalar_t tz = (zq - bz0) * inv_dz;
    const scalar_t Mx_lim = (scalar_t)(Mx - 1);
    const scalar_t My_lim = (scalar_t)(My - 1);
    const scalar_t Mz_lim = (scalar_t)(Mz - 1);
    tx = std::max((scalar_t)0, std::min(tx, Mx_lim));
    ty = std::max((scalar_t)0, std::min(ty, My_lim));
    tz = std::max((scalar_t)0, std::min(tz, Mz_lim));

    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;
    int iz = (int)tz; if (iz > Mz - 2) iz = Mz - 2;

    if (ix < 1 || iy < 1 || iz < 1 ||
        Mx < 3 || My < 3 || Mz < 3) {
        return trilinear_sample_uniform<scalar_t>(
            F, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, xq, yq, zq);
    }

    const scalar_t fx = tx - (scalar_t)ix;
    const scalar_t fy = ty - (scalar_t)iy;
    const scalar_t fz = tz - (scalar_t)iz;

    const scalar_t half = (scalar_t)0.5;
    const scalar_t wxm = half * fx * (fx - (scalar_t)1);
    const scalar_t wx0 = (scalar_t)1 - fx * fx;
    const scalar_t wxp = half * fx * (fx + (scalar_t)1);
    const scalar_t wym = half * fy * (fy - (scalar_t)1);
    const scalar_t wy0 = (scalar_t)1 - fy * fy;
    const scalar_t wyp = half * fy * (fy + (scalar_t)1);
    const scalar_t wzm = half * fz * (fz - (scalar_t)1);
    const scalar_t wz0 = (scalar_t)1 - fz * fz;
    const scalar_t wzp = half * fz * (fz + (scalar_t)1);

    const int s2 = Mz;
    const int s1 = My * Mz;
    const int base = (ix - 1) * s1 + (iy - 1) * s2 + (iz - 1);

    // Identical accumulation order as the CUDA kernel — running +=
    // accumulators so FMA contraction produces the same rounding.
    scalar_t out = (scalar_t)0;
    for (int dx = 0; dx < 3; ++dx) {
        const scalar_t wx = (dx == 0) ? wxm : (dx == 1 ? wx0 : wxp);
        const int b0 = base + dx * s1;
        scalar_t plane = (scalar_t)0;
        for (int dy = 0; dy < 3; ++dy) {
            const scalar_t wy = (dy == 0) ? wym : (dy == 1 ? wy0 : wyp);
            const int b1 = b0 + dy * s2;
            const scalar_t row =
                wzm * F[b1]     + wz0 * F[b1 + 1] + wzp * F[b1 + 2];
            plane += wy * row;
        }
        out += wx * plane;
    }
    return out;
}

// =====================================================================
//  Per-cell update body shared by single- and multi-body launchers.
//
//  Each cell is touched once per launch -> no atomics needed (matches
//  the CUDA kernel).
//
//  Optimisation (rotation CSE): the 4 sample positions (cc + 3 face
//  staggers) differ in world space only by ``±h/2`` along ONE world
//  axis.  In body frame, a world offset ``(δx, 0, 0)`` becomes
//  ``δx * col0(R_T)``.  So the body-frame face points are just
//  ``body_cc + Δ_k`` where ``Δ_k = -half_h * col_k(R_T)`` is a per-body
//  constant.  We compute the body-frame CC point once (one matrix-vector
//  multiply: 9 muls + 6 adds) and derive the 3 face points by adding
//  precomputed body-frame deltas (3 adds each).  Saves 27 muls + 18 adds
//  per cell vs. doing 4 full rotations.
// =====================================================================
template <typename scalar_t>
static inline void update_cell(
    const scalar_t* F,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    const scalar_t r00, const scalar_t r01, const scalar_t r02,
    const scalar_t r10, const scalar_t r11, const scalar_t r12,
    const scalar_t r20, const scalar_t r21, const scalar_t r22,
    const scalar_t bp_x, const scalar_t bp_y, const scalar_t bp_z,
    const scalar_t cm_x, const scalar_t cm_y, const scalar_t cm_z,
    const scalar_t lv_x, const scalar_t lv_y, const scalar_t lv_z,
    const scalar_t av_x, const scalar_t av_y, const scalar_t av_z,
    // Body-frame face deltas (precomputed once per body):
    //   Δ_u = -half_h * col0(R_T), Δ_v = -half_h * col1(R_T), Δ_w = -half_h * col2(R_T)
    const scalar_t du_x, const scalar_t du_y, const scalar_t du_z,
    const scalar_t dv_x, const scalar_t dv_y, const scalar_t dv_z,
    const scalar_t dw_x, const scalar_t dw_y, const scalar_t dw_z,
    const scalar_t xc, const scalar_t yc, const scalar_t zc,
    const std::int64_t g_idx, const std::int64_t sparse_idx,
    scalar_t* sdf_cc, scalar_t* sdf_u, scalar_t* sdf_v, scalar_t* sdf_w,
    scalar_t* bU, scalar_t* bV, scalar_t* bW,
    scalar_t* sparse_cc,
    // 0 = trilinear (default), 1 = triquadratic (Lagrange 3x3x3).
    const int interp_method)
{
    // Body-frame CC point (single rotation; the 3 face points are derived
    // from it by precomputed deltas).
    const scalar_t dxw = xc - bp_x, dyw = yc - bp_y, dzw = zc - bp_z;
    const scalar_t bxq = r00*dxw + r01*dyw + r02*dzw;
    const scalar_t byq = r10*dxw + r11*dyw + r12*dzw;
    const scalar_t bzq = r20*dxw + r21*dyw + r22*dzw;

    // Sampler dispatch: chosen once per cell, branch is highly predictable
    // across a single op call (interp_method is constant per launch).
    auto sample = [&](scalar_t xqs, scalar_t yqs, scalar_t zqs) -> scalar_t {
        if (interp_method == 1) {
            return triquadratic_sample_uniform<scalar_t>(
                F, Mx, My, Mz, bx0, by0, bz0,
                inv_dx, inv_dy, inv_dz, xqs, yqs, zqs);
        }
        return trilinear_sample_uniform<scalar_t>(
            F, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, xqs, yqs, zqs);
    };

    // ---- cc ----
    {
        const scalar_t s = sample(bxq, byq, bzq);
        sparse_cc[sparse_idx] = s;
        if (s < sdf_cc[g_idx]) sdf_cc[g_idx] = s;
    }
    // ---- u-face ----
    {
        const scalar_t s = sample(bxq + du_x, byq + du_y, bzq + du_z);
        if (s < sdf_u[g_idx]) {
            sdf_u[g_idx] = s;
            bU[g_idx] = lv_x + av_y*(zc - cm_z) - av_z*(yc - cm_y);
        }
    }
    // ---- v-face ----
    {
        const scalar_t s = sample(bxq + dv_x, byq + dv_y, bzq + dv_z);
        if (s < sdf_v[g_idx]) {
            sdf_v[g_idx] = s;
            bV[g_idx] = lv_y + av_z*(xc - cm_x) - av_x*(zc - cm_z);
        }
    }
    // ---- w-face ----
    {
        const scalar_t s = sample(bxq + dw_x, byq + dw_y, bzq + dw_z);
        if (s < sdf_w[g_idx]) {
            sdf_w[g_idx] = s;
            bW[g_idx] = lv_z + av_x*(yc - cm_y) - av_y*(xc - cm_x);
        }
    }
}


// =====================================================================
//  apply_bcs_3d
//
//  Each op writes a single 2-D plane of u / v / w.  Ops run in three
//  ordered stages (Neumann → Dirichlet → reflective); within a stage the
//  write-ownership rule in bc_ops.h gives every cell — edge/corner ghosts
//  included — exactly one writer and a source that no same-stage op is
//  writing, so the result is identical to the CUDA twin cell for cell.
//  See bc_ops.h for the full argument.
// =====================================================================

template <typename scalar_t>
static void apply_bcs_3d_one_plane(
    scalar_t* base,
    const std::int64_t* shapes_p,
    const int kind, const int* desc, const int nops, const int op,
    const scalar_t* vals,
    const int comp, const int axis,
    const int dst_along, const int src_along,
    const int Ny, const int Nz,
    const int dim0_max, const int dim1_max)
{
    using namespace lilytorch_kernels::bcs;

    const std::int64_t s1 = (std::int64_t)Ny * Nz;
    const std::int64_t s2 = (std::int64_t)Nz;

    const int64_t total = (int64_t)dim0_max * (int64_t)dim1_max;
    at::parallel_for(0, total, /*grain_size=*/1024, [&](int64_t _begin, int64_t _end) {
    for (int64_t t = _begin; t < _end; ++t) {
        const int i = (int)(t / dim1_max);
        const int j = (int)(t % dim1_max);

        int c[3];
        if      (axis == 0) { c[0] = dst_along; c[1] = i;         c[2] = j;         }
        else if (axis == 1) { c[0] = i;         c[1] = dst_along; c[2] = j;         }
        else                { c[0] = i;         c[1] = j;         c[2] = dst_along; }

        int s[3];
        if (!bc_own_and_source(kind, desc, nops, op, shapes_p, /*ndim=*/3,
                               comp, axis, src_along, c, s))
            continue;                              // a lower-indexed op owns it

        const std::int64_t dst_lin = (std::int64_t)c[0] * s1 + (std::int64_t)c[1] * s2 + c[2];
        const std::int64_t src_lin = (std::int64_t)s[0] * s1 + (std::int64_t)s[1] * s2 + s[2];

        if      (kind == BC_KIND_NEUMANN)   base[dst_lin] = base[src_lin];
        else if (kind == BC_KIND_DIRICHLET) base[dst_lin] = vals[op];
        else base[dst_lin] = scalar_t(2) * vals[op] - base[src_lin];
    }
    });
}

void apply_bcs_3d_cpu(
    at::Tensor u, at::Tensor v, at::Tensor w,
    const at::Tensor& shapes,
    const at::Tensor& neu_desc,
    const at::Tensor& dir_desc,
    const at::Tensor& dir_val,
    const at::Tensor& ref_desc,
    const at::Tensor& ref_val,
    const int64_t /*max_dim0*/,
    const int64_t /*max_dim1*/)
{
    TORCH_CHECK(u.is_contiguous() && v.is_contiguous() && w.is_contiguous(),
                "apply_bcs_3d_cpu: u/v/w must be contiguous");
    TORCH_CHECK(u.scalar_type() == v.scalar_type() &&
                u.scalar_type() == w.scalar_type(),
                "apply_bcs_3d_cpu: u/v/w must share dtype");
    TORCH_CHECK(shapes.scalar_type() == at::kLong &&
                shapes.dim() == 2 && shapes.size(0) == 3 && shapes.size(1) == 3,
                "apply_bcs_3d_cpu: shapes must be int64[3,3]");
    TORCH_CHECK(neu_desc.scalar_type() == at::kInt && neu_desc.dim() == 2 &&
                neu_desc.size(1) == 3,
                "apply_bcs_3d_cpu: neu_desc must be int32[N,3]");
    TORCH_CHECK(dir_desc.scalar_type() == at::kInt && dir_desc.dim() == 2 &&
                dir_desc.size(1) == 3,
                "apply_bcs_3d_cpu: dir_desc must be int32[N,3]");
    TORCH_CHECK(ref_desc.scalar_type() == at::kInt && ref_desc.dim() == 2 &&
                ref_desc.size(1) == 4,
                "apply_bcs_3d_cpu: ref_desc must be int32[N,4]");

    const int N_neu = (int)neu_desc.size(0);
    const int N_dir = (int)dir_desc.size(0);
    const int N_ref = (int)ref_desc.size(0);
    if (N_neu + N_dir + N_ref == 0) return;

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_3d_cpu", [&] {
        using namespace lilytorch_kernels::bcs;

        const int64_t*  shapes_p  = shapes.data_ptr<int64_t>();
        const int*      neu_p     = (N_neu > 0) ? neu_desc.data_ptr<int>() : nullptr;
        const int*      dir_p     = (N_dir > 0) ? dir_desc.data_ptr<int>() : nullptr;
        const scalar_t* dir_val_p = (N_dir > 0) ? dir_val.data_ptr<scalar_t>() : nullptr;
        const int*      ref_p     = (N_ref > 0) ? ref_desc.data_ptr<int>() : nullptr;
        const scalar_t* ref_val_p = (N_ref > 0) ? ref_val.data_ptr<scalar_t>() : nullptr;

        scalar_t* u_p = u.data_ptr<scalar_t>();
        scalar_t* v_p = v.data_ptr<scalar_t>();
        scalar_t* w_p = w.data_ptr<scalar_t>();

        // One stage per op kind, in the same order as the CUDA launches.
        const int   kinds[3] = {BC_KIND_NEUMANN, BC_KIND_DIRICHLET, BC_KIND_REFLECTIVE};
        const int*  descs[3] = {neu_p, dir_p, ref_p};
        const scalar_t* valss[3] = {nullptr, dir_val_p, ref_val_p};
        const int   counts[3] = {N_neu, N_dir, N_ref};

        for (int st = 0; st < 3; ++st) {
            const int kind = kinds[st];
            const int* desc = descs[st];
            const int nops = counts[st];

            for (int op = 0; op < nops; ++op) {
                int comp, axis, dst_along, src_along;
                bc_decode(kind, desc, op, shapes_p, /*ndim=*/3,
                          comp, axis, dst_along, src_along);

                const int Nx = (int)shapes_p[comp*3 + 0];
                const int Ny = (int)shapes_p[comp*3 + 1];
                const int Nz = (int)shapes_p[comp*3 + 2];

                int dim0_max, dim1_max;
                if      (axis == 0) { dim0_max = Ny; dim1_max = Nz; }
                else if (axis == 1) { dim0_max = Nx; dim1_max = Nz; }
                else                { dim0_max = Nx; dim1_max = Ny; }

                scalar_t* base = (comp == 0) ? u_p : (comp == 1 ? v_p : w_p);

                apply_bcs_3d_one_plane<scalar_t>(
                    base, shapes_p, kind, desc, nops, op, valss[st],
                    comp, axis, dst_along, src_along,
                    Ny, Nz, dim0_max, dim1_max);
            }
        }
    });
}

// =====================================================================
//  interp_3d_cpu: scattered-point trilinear / triquadratic sampling
// =====================================================================
static void interp_3d_cpu(
    const at::Tensor& F,
    const at::Tensor& xq, const at::Tensor& yq, const at::Tensor& zq,
    const double bx0, const double by0, const double bz0,
    const double inv_dx, const double inv_dy, const double inv_dz,
    const int64_t Mx, const int64_t My, const int64_t Mz,
    const int64_t interp_method,
    at::Tensor& G)
{
    const int N = (int)xq.numel();
    if (N == 0) return;

    // Bind temporaries to named locals: ``.contiguous().to(...)`` returns a
    // fresh tensor whose storage is freed at the end of the full-expression
    // unless held.  Without this the raw ``data_ptr`` would dangle for
    // non-contiguous or differently-dtyped inputs and the parallel_for loop
    // below would read freed memory.
    auto F_c  = F.contiguous();
    auto xq_c = xq.contiguous().to(F.scalar_type());
    auto yq_c = yq.contiguous().to(F.scalar_type());
    auto zq_c = zq.contiguous().to(F.scalar_type());

    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "interp_3d_cpu", [&] {
        const scalar_t* Fp  = F_c.data_ptr<scalar_t>();
        const scalar_t* xqp = xq_c.data_ptr<scalar_t>();
        const scalar_t* yqp = yq_c.data_ptr<scalar_t>();
        const scalar_t* zqp = zq_c.data_ptr<scalar_t>();
        scalar_t* Gp = G.data_ptr<scalar_t>();
        const int iMx = (int)Mx, iMy = (int)My, iMz = (int)Mz;
        const scalar_t bx0s = (scalar_t)bx0, by0s = (scalar_t)by0, bz0s = (scalar_t)bz0;
        const scalar_t idx  = (scalar_t)inv_dx, idy = (scalar_t)inv_dy, idz = (scalar_t)inv_dz;
        const int method    = (int)interp_method;

        at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                if (method == 1) {
                    Gp[i] = triquadratic_sample_uniform<scalar_t>(
                        Fp, iMx, iMy, iMz,
                        bx0s, by0s, bz0s, idx, idy, idz,
                        xqp[i], yqp[i], zqp[i]);
                } else {
                    Gp[i] = trilinear_sample_uniform<scalar_t>(
                        Fp, iMx, iMy, iMz,
                        bx0s, by0s, bz0s, idx, idy, idz,
                        xqp[i], yqp[i], zqp[i]);
                }
            }
        });
    });
}

// =====================================================================
//  removed inline force kernel  (removed inline-force path)
//
//  CPU port of the CUDA kernel in cuda/streaming_sdf.cu (line-for-line).
//  Same memory contract:
//    * per-body AABB iteration; all per-body state lives on the stack
//      (registers / locals) and is released between bodies.
//    * No per-body grid-sized scratch. Accumulates 12 force/torque
//      channels into ``out`` (atomically combined across at::parallel_for
//      workers via a final per-body merge).
//    * Updates the union fields (sdf_cc / sdf_u / sdf_v / sdf_w / bU /
//      bV / bW) in place.
//
//  Forces are one-step lagged: viscous stress and pressure force are
//  computed from the beginning-of-step velocity / pressure fields
//  (u_prev, v_prev, w_prev, p_prev) and the cached union CC normals
//  from the previous step (nx_cc, ny_cc, nz_cc).
// =====================================================================

template <typename scalar_t>
static inline scalar_t sample_dispatch_cpu(
    const int interp_method,
    const scalar_t* F, const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    const scalar_t xq, const scalar_t yq, const scalar_t zq)
{
    if (interp_method == 1) {
        return triquadratic_sample_uniform<scalar_t>(
            F, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, xq, yq, zq);
    }
    return trilinear_sample_uniform<scalar_t>(
        F, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, xq, yq, zq);
}

// =====================================================================
//  CPU registration. The schemas live in ops.cpp; ops.cpp no longer
//  registers CPU stubs, so these implementations bind directly.
// =====================================================================

void streaming_sdf_forces_post_3d_cpu(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes,
    const at::Tensor& body_meta,
    const at::Tensor& kin,
    const at::Tensor& aabb_lo,
    const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy, const at::Tensor& gz,
    const double h_grid,
    const int64_t /*max_vol_per_body*/,
    const at::Tensor& sdf_cc,
    const int64_t interp_method,
    const at::Tensor& u, const at::Tensor& v,
    const at::Tensor& w, const at::Tensor& p,
    const at::Tensor& nu_rho_field,
    const double eps_body,
    const double eps_solver,
    const double h3,
    const int64_t delta_order,
    const int64_t force_submethod,
    const double ph_tau,
    at::Tensor out)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0) return;
    TORCH_CHECK(out.scalar_type() == at::kDouble, "streaming_sdf_forces_post_3d_cpu: out must be float64");
    const int Ngx = (int)gx.numel(), Ngy = (int)gy.numel(), Ngz = (int)gz.numel();
    const int64_t nr_size = nu_rho_field.numel();

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_forces_post_3d_cpu", [&] {
        const scalar_t* F_ptr=F_flat.data_ptr<scalar_t>();
        const int64_t* F_off=F_offsets.data_ptr<int64_t>();
        const int64_t* shapes=body_shapes.data_ptr<int64_t>();
        const scalar_t* meta=body_meta.data_ptr<scalar_t>();
        const scalar_t* kin_ptr=kin.data_ptr<scalar_t>();
        const int64_t* lo=aabb_lo.data_ptr<int64_t>();
        const int64_t* dim=aabb_dim.data_ptr<int64_t>();
        const scalar_t* gx_ptr=gx.data_ptr<scalar_t>();
        const scalar_t* gy_ptr=gy.data_ptr<scalar_t>();
        const scalar_t* gz_ptr=gz.data_ptr<scalar_t>();
        const scalar_t* sdf=sdf_cc.data_ptr<scalar_t>();
        const scalar_t* up=u.data_ptr<scalar_t>();
        const scalar_t* vp=v.data_ptr<scalar_t>();
        const scalar_t* wp=w.data_ptr<scalar_t>();
        const scalar_t* pp=p.data_ptr<scalar_t>();
        const scalar_t* nr=nu_rho_field.data_ptr<scalar_t>();
        double* outp=out.data_ptr<double>();
        const scalar_t inv_h=(scalar_t)(1.0/h_grid), eps_b=(scalar_t)eps_body, eps_s=(scalar_t)eps_solver;
        const scalar_t pi=(scalar_t)3.141592653589793, inv_2eps=(scalar_t)0.5/eps_b, pi_eb=pi/eps_b;
        const scalar_t band_lo=(eps_s-eps_b)<(-eps_b)?(eps_s-eps_b):(-eps_b);
        const scalar_t band_hi=(eps_s+eps_b)>(eps_b)?(eps_s+eps_b):(eps_b);
        const double h3d=(double)h3;
        auto S=[&](int i,int j,int k)->scalar_t{return sdf[((int64_t)i*Ngy+j)*Ngz+k];};
        for(int b=0;b<B;++b){
            const int Ai=(int)dim[b*3], Aj=(int)dim[b*3+1], Ak=(int)dim[b*3+2];
            const int vol=Ai*Aj*Ak; if(vol<=0) continue;
            const int i0=(int)lo[b*3], j0=(int)lo[b*3+1], k0=(int)lo[b*3+2];
            const scalar_t* Fb=F_ptr+F_off[b];
            const int Mx=(int)shapes[b*3], My=(int)shapes[b*3+1], Mz=(int)shapes[b*3+2];
            const scalar_t* M=meta+b*10; const scalar_t bx0=M[0],by0=M[1],bz0=M[2],idx=M[6],idy=M[7],idz=M[8];
            const scalar_t* K=kin_ptr+b*21;
            const scalar_t r00=K[0],r01=K[1],r02=K[2],r10=K[3],r11=K[4],r12=K[5],r20=K[6],r21=K[7],r22=K[8];
            const scalar_t bp_x=K[9],bp_y=K[10],bp_z=K[11],cm_x=K[12],cm_y=K[13],cm_z=K[14];
            // Cells of one body's AABB are independent; parallelise across
            // them and accumulate into a per-thread 12-channel slice of a
            // shared scratch buffer (no locking).  Final reduction across
            // threads runs single-threaded after the parallel region.
            const int nT = at::get_num_threads();
            std::vector<double> tls((size_t)nT * 12, 0.0);
            at::parallel_for(0, vol, /*grain_size=*/2048, [&](int64_t _begin, int64_t _end) {
            double local12[12]={0,0,0,0,0,0,0,0,0,0,0,0};
            for(int local=(int)_begin;local<(int)_end;++local){
                const int di=local/(Aj*Ak), rem=local-di*(Aj*Ak), dj=rem/Ak, dk=rem-dj*Ak;
                const int i=i0+di,j=j0+dj,k=k0+dk; const int64_t g=((int64_t)i*Ngy+j)*Ngz+k;
                const scalar_t xc=gx_ptr[i], yc=gy_ptr[j], zc=gz_ptr[k];
                const scalar_t dxw=xc-bp_x,dyw=yc-bp_y,dzw=zc-bp_z;
                const scalar_t bxq=r00*dxw+r01*dyw+r02*dzw, byq=r10*dxw+r11*dyw+r12*dzw, bzq=r20*dxw+r21*dyw+r22*dzw;
                const scalar_t sbody=sample_dispatch_cpu<scalar_t>((int)interp_method,Fb,Mx,My,Mz,bx0,by0,bz0,idx,idy,idz,bxq,byq,bzq);
                if(!(sbody>band_lo && sbody<band_hi)) continue;
                scalar_t dx=0,dy=0,dz=0;
                if(Ngx>=3){ if(i==0) dx=((-3)*S(0,j,k)+4*S(1,j,k)-S(2,j,k))*0.5*inv_h; else if(i==Ngx-1) dx=(3*S(Ngx-1,j,k)-4*S(Ngx-2,j,k)+S(Ngx-3,j,k))*0.5*inv_h; else dx=(S(i+1,j,k)-S(i-1,j,k))*0.5*inv_h; } else if(Ngx==2) dx=(S(1,j,k)-S(0,j,k))*inv_h;
                if(Ngy>=3){ if(j==0) dy=((-3)*S(i,0,k)+4*S(i,1,k)-S(i,2,k))*0.5*inv_h; else if(j==Ngy-1) dy=(3*S(i,Ngy-1,k)-4*S(i,Ngy-2,k)+S(i,Ngy-3,k))*0.5*inv_h; else dy=(S(i,j+1,k)-S(i,j-1,k))*0.5*inv_h; } else if(Ngy==2) dy=(S(i,1,k)-S(i,0,k))*inv_h;
                if(Ngz>=3){ if(k==0) dz=((-3)*S(i,j,0)+4*S(i,j,1)-S(i,j,2))*0.5*inv_h; else if(k==Ngz-1) dz=(3*S(i,j,Ngz-1)-4*S(i,j,Ngz-2)+S(i,j,Ngz-3))*0.5*inv_h; else dz=(S(i,j,k+1)-S(i,j,k-1))*0.5*inv_h; } else if(Ngz==2) dz=(S(i,j,1)-S(i,j,0))*inv_h;
                scalar_t norm=std::sqrt(dx*dx+dy*dy+dz*dz); scalar_t invn=norm>0?(scalar_t)1/norm:(scalar_t)0; const scalar_t nx=dx*invn,ny=dy*invn,nz=dz*invn;
                // Velocity-gradient stencils — must mirror the CUDA kernel
                // (cuda/streaming_sdf.cu) EXACTLY.  Normal derivatives are
                // forward differences (backward on the upper boundary); the
                // cross derivatives are O(h²) central on the CC-interpolated
                // component, with 3-point one-sided formulas on the boundary
                // planes.  (A clamped-index central difference is NOT the same
                // thing: it silently degrades to first order at the boundary,
                // and a clamped forward difference for dudx collapses to zero
                // on the last plane.)
                const int im1=i>0?i-1:0,        ip1=i+1<Ngx?i+1:i;
                const int im2=i>1?i-2:0,        ip2=i+2<Ngx?i+2:(Ngx-1);
                const int jm1=j>0?j-1:0,        jp1=j+1<Ngy?j+1:j;
                const int jm2=j>1?j-2:0,        jp2=j+2<Ngy?j+2:(Ngy-1);
                const int km1=k>0?k-1:0,        kp1=k+1<Ngz?k+1:k;
                const int km2=k>1?k-2:0,        kp2=k+2<Ngz?k+2:(Ngz-1);

                const scalar_t dudx = (i+1<Ngx)
                    ? (up[(ip1*Ngy+j)*Ngz+k] - up[(i*Ngy+j)*Ngz+k]) * inv_h
                    : (up[(i*Ngy+j)*Ngz+k]   - up[(im1*Ngy+j)*Ngz+k]) * inv_h;
                const scalar_t dvdy = (j+1<Ngy)
                    ? (vp[(i*Ngy+jp1)*Ngz+k] - vp[(i*Ngy+j)*Ngz+k]) * inv_h
                    : (vp[(i*Ngy+j)*Ngz+k]   - vp[(i*Ngy+jm1)*Ngz+k]) * inv_h;
                const scalar_t dwdz = (k+1<Ngz)
                    ? (wp[(i*Ngy+j)*Ngz+kp1] - wp[(i*Ngy+j)*Ngz+k]) * inv_h
                    : (wp[(i*Ngy+j)*Ngz+k]   - wp[(i*Ngy+j)*Ngz+km1]) * inv_h;

                // 3-point O(h²) derivative of a CC-interpolated component along
                // one axis: central in the interior, one-sided on the boundary.
                auto d_cc=[&](scalar_t am2,scalar_t am1,scalar_t a0,
                              scalar_t ap1,scalar_t ap2,int idxa,int Na)->scalar_t{
                    if(Na>=3){
                        if(idxa==0)      return ((scalar_t)(-3)*a0 + (scalar_t)4*ap1 - ap2)*(scalar_t)0.5*inv_h;
                        if(idxa==Na-1)   return ((scalar_t)3*a0 - (scalar_t)4*am1 + am2)*(scalar_t)0.5*inv_h;
                    }
                    return (ap1 - am1)*(scalar_t)0.5*inv_h;
                };
                // u staggered in x → CC: 0.5*(u[i,..]+u[i+1,..])
                auto u_cc=[&](int jj,int kk)->scalar_t{
                    return (scalar_t)0.5*(up[(i*Ngy+jj)*Ngz+kk] + up[(ip1*Ngy+jj)*Ngz+kk]); };
                // v staggered in y → CC: 0.5*(v[..,j,..]+v[..,j+1,..])
                auto v_cc=[&](int ii,int kk)->scalar_t{
                    return (scalar_t)0.5*(vp[(ii*Ngy+j)*Ngz+kk] + vp[(ii*Ngy+jp1)*Ngz+kk]); };
                // w staggered in z → CC: 0.5*(w[..,k]+w[..,k+1])
                auto w_cc=[&](int ii,int jj)->scalar_t{
                    return (scalar_t)0.5*(wp[(ii*Ngy+jj)*Ngz+k] + wp[(ii*Ngy+jj)*Ngz+kp1]); };

                const scalar_t dudy=d_cc(u_cc(jm2,k),u_cc(jm1,k),u_cc(j,k),u_cc(jp1,k),u_cc(jp2,k), j,Ngy);
                const scalar_t dudz=d_cc(u_cc(j,km2),u_cc(j,km1),u_cc(j,k),u_cc(j,kp1),u_cc(j,kp2), k,Ngz);
                const scalar_t dvdx=d_cc(v_cc(im2,k),v_cc(im1,k),v_cc(i,k),v_cc(ip1,k),v_cc(ip2,k), i,Ngx);
                const scalar_t dvdz=d_cc(v_cc(i,km2),v_cc(i,km1),v_cc(i,k),v_cc(i,kp1),v_cc(i,kp2), k,Ngz);
                const scalar_t dwdx=d_cc(w_cc(im2,j),w_cc(im1,j),w_cc(i,j),w_cc(ip1,j),w_cc(ip2,j), i,Ngx);
                const scalar_t dwdy=d_cc(w_cc(i,jm2),w_cc(i,jm1),w_cc(i,j),w_cc(i,jp1),w_cc(i,jp2), j,Ngy);
                const scalar_t nrv=nr_size==1?nr[0]:nr[g];
                const scalar_t xs=nrv*(2*dudx*nx+(dudy+dvdx)*ny+(dudz+dwdx)*nz), ys=nrv*((dvdx+dudy)*nx+2*dvdy*ny+(dvdz+dwdy)*nz), zs=nrv*((dwdx+dudz)*nx+(dwdy+dvdz)*ny+2*dwdz*nz);
                scalar_t dv=0,dp=0; const scalar_t sd=sbody-eps_s; if(sd>-eps_b&&sd<eps_b) dv=(1+std::cos(pi_eb*sd))*inv_2eps; if(sbody>-eps_b&&sbody<eps_b) dp=(1+std::cos(pi_eb*sbody))*inv_2eps;
                // deltaH: pressure force/torque come from the union-∂H pass below.
                if(force_submethod!=0) dp=0;
                // delta_order==2: divide both deltas by |grad(sdf_body)| evaluated by
                // re-sampling at world-aligned ±h offsets.  Matches the CUDA-3D path
                // at streaming_sdf.cu:644-657; without this the CPU path silently
                // skipped the gradient correction and disagreed with CUDA forces.
                if(delta_order==2 && (dv>0 || dp>0)){
                    const scalar_t hg=(scalar_t)1.0/inv_h;
                    const scalar_t s_xp=sample_dispatch_cpu<scalar_t>((int)interp_method,Fb,Mx,My,Mz,bx0,by0,bz0,idx,idy,idz,bxq+r00*hg,byq+r10*hg,bzq+r20*hg);
                    const scalar_t s_xm=sample_dispatch_cpu<scalar_t>((int)interp_method,Fb,Mx,My,Mz,bx0,by0,bz0,idx,idy,idz,bxq-r00*hg,byq-r10*hg,bzq-r20*hg);
                    const scalar_t s_yp=sample_dispatch_cpu<scalar_t>((int)interp_method,Fb,Mx,My,Mz,bx0,by0,bz0,idx,idy,idz,bxq+r01*hg,byq+r11*hg,bzq+r21*hg);
                    const scalar_t s_ym=sample_dispatch_cpu<scalar_t>((int)interp_method,Fb,Mx,My,Mz,bx0,by0,bz0,idx,idy,idz,bxq-r01*hg,byq-r11*hg,bzq-r21*hg);
                    const scalar_t s_zp=sample_dispatch_cpu<scalar_t>((int)interp_method,Fb,Mx,My,Mz,bx0,by0,bz0,idx,idy,idz,bxq+r02*hg,byq+r12*hg,bzq+r22*hg);
                    const scalar_t s_zm=sample_dispatch_cpu<scalar_t>((int)interp_method,Fb,Mx,My,Mz,bx0,by0,bz0,idx,idy,idz,bxq-r02*hg,byq-r12*hg,bzq-r22*hg);
                    scalar_t gm=std::sqrt(((s_xp-s_xm)*(s_xp-s_xm)+(s_yp-s_ym)*(s_yp-s_ym)+(s_zp-s_zm)*(s_zp-s_zm))*(scalar_t)0.25*inv_h*inv_h);
                    if(gm<(scalar_t)1e-3) gm=(scalar_t)1e-3;
                    const scalar_t ig=(scalar_t)1.0/gm;
                    dv*=ig; dp*=ig;
                }
                const scalar_t px=-pp[g]*nx, py=-pp[g]*ny, pz=-pp[g]*nz, ax=xc-cm_x, ay=yc-cm_y, az=zc-cm_z;
                const double fvx=xs*dv, fvy=ys*dv, fvz=zs*dv, fpx=px*dp, fpy=py*dp, fpz=pz*dp;
                local12[0]+=fvx; local12[1]+=fvy; local12[2]+=fvz; local12[3]+=ay*fvz-az*fvy; local12[4]+=az*fvx-ax*fvz; local12[5]+=ax*fvy-ay*fvx;
                local12[6]+=fpx; local12[7]+=fpy; local12[8]+=fpz; local12[9]+=ay*fpz-az*fpy; local12[10]+=az*fpx-ax*fpz; local12[11]+=ax*fpy-ay*fpx;
            }
            // Per-thread slice; no race because each thread t writes only
            // tls[t*12 + c..c+11].  Multiple chunks assigned to the same
            // thread accumulate sequentially.
            const int t = at::get_thread_num();
            for(int c=0;c<12;++c) tls[(size_t)t*12+c]+=local12[c];
            });
            double acc[12]={0,0,0,0,0,0,0,0,0,0,0,0};
            for(int t=0;t<nT;++t)
                for(int c=0;c<12;++c) acc[c]+=tls[(size_t)t*12+c];
            for(int c=0;c<12;++c) outp[b*12+c]+=acc[c]*h3d;
        }
        // ---- deltaH: union-∂H pressure force density, softmin partition ----
        if(force_submethod!=0){
            const scalar_t inv_eps=(scalar_t)(1.0/eps_body);
            const double tau=(ph_tau>0.0)?ph_tau:1e-9;
            const scalar_t inv_tau=(scalar_t)(1.0/tau);
            int ulo[3]={Ngx,Ngy,Ngz}, uhi[3]={0,0,0};
            for(int b=0;b<B;++b) for(int d=0;d<3;++d){int a0=(int)lo[b*3+d];int a1=a0+(int)dim[b*3+d];if(a0<ulo[d])ulo[d]=a0;if(a1>uhi[d])uhi[d]=a1;}
            const int Ng[3]={Ngx,Ngy,Ngz}; const int halo=2;
            for(int d=0;d<3;++d){ulo[d]-=halo;if(ulo[d]<0)ulo[d]=0;uhi[d]+=halo;if(uhi[d]>Ng[d])uhi[d]=Ng[d];}
            auto Hs=[&](int i,int j,int k)->scalar_t{ scalar_t x=S(i,j,k)*inv_eps; x=x<(scalar_t)-1?(scalar_t)-1:(x>(scalar_t)1?(scalar_t)1:x); return (scalar_t)0.5*((scalar_t)1+x+std::sin(pi*x)/pi); };
            for(int i=ulo[0];i<uhi[0];++i)for(int j=ulo[1];j<uhi[1];++j)for(int k=ulo[2];k<uhi[2];++k){
                scalar_t gHx=0,gHy=0,gHz=0;
                if(Ngx>=3){ if(i==0) gHx=((-3)*Hs(0,j,k)+4*Hs(1,j,k)-Hs(2,j,k))*(scalar_t)0.5*inv_h; else if(i==Ngx-1) gHx=(3*Hs(Ngx-1,j,k)-4*Hs(Ngx-2,j,k)+Hs(Ngx-3,j,k))*(scalar_t)0.5*inv_h; else gHx=(Hs(i+1,j,k)-Hs(i-1,j,k))*(scalar_t)0.5*inv_h; } else if(Ngx==2) gHx=(Hs(1,j,k)-Hs(0,j,k))*inv_h;
                if(Ngy>=3){ if(j==0) gHy=((-3)*Hs(i,0,k)+4*Hs(i,1,k)-Hs(i,2,k))*(scalar_t)0.5*inv_h; else if(j==Ngy-1) gHy=(3*Hs(i,Ngy-1,k)-4*Hs(i,Ngy-2,k)+Hs(i,Ngy-3,k))*(scalar_t)0.5*inv_h; else gHy=(Hs(i,j+1,k)-Hs(i,j-1,k))*(scalar_t)0.5*inv_h; } else if(Ngy==2) gHy=(Hs(i,1,k)-Hs(i,0,k))*inv_h;
                if(Ngz>=3){ if(k==0) gHz=((-3)*Hs(i,j,0)+4*Hs(i,j,1)-Hs(i,j,2))*(scalar_t)0.5*inv_h; else if(k==Ngz-1) gHz=(3*Hs(i,j,Ngz-1)-4*Hs(i,j,Ngz-2)+Hs(i,j,Ngz-3))*(scalar_t)0.5*inv_h; else gHz=(Hs(i,j,k+1)-Hs(i,j,k-1))*(scalar_t)0.5*inv_h; } else if(Ngz==2) gHz=(Hs(i,j,1)-Hs(i,j,0))*inv_h;
                if(gHx==(scalar_t)0&&gHy==(scalar_t)0&&gHz==(scalar_t)0) continue;
                const int64_t g=((int64_t)i*Ngy+j)*Ngz+k;
                const scalar_t p_c=pp[g];
                const scalar_t fdx=-p_c*gHx, fdy=-p_c*gHy, fdz=-p_c*gHz;
                const scalar_t xc=gx_ptr[i],yc=gy_ptr[j],zc=gz_ptr[k];
                const scalar_t sdfu=sdf[g];
                scalar_t Z=0;
                for(int b=0;b<B;++b){
                    const int i0=(int)lo[b*3],j0=(int)lo[b*3+1],k0=(int)lo[b*3+2];
                    const int Ai=(int)dim[b*3],Aj=(int)dim[b*3+1],Ak=(int)dim[b*3+2];
                    if(i<i0||i>=i0+Ai||j<j0||j>=j0+Aj||k<k0||k>=k0+Ak) continue;
                    const scalar_t* Fb=F_ptr+F_off[b]; const int Mx=(int)shapes[b*3],My=(int)shapes[b*3+1],Mz=(int)shapes[b*3+2];
                    const scalar_t* M=meta+b*10; const scalar_t* K=kin_ptr+b*21;
                    const scalar_t dxw=xc-K[9],dyw=yc-K[10],dzw=zc-K[11];
                    const scalar_t bxq=K[0]*dxw+K[1]*dyw+K[2]*dzw, byq=K[3]*dxw+K[4]*dyw+K[5]*dzw, bzq=K[6]*dxw+K[7]*dyw+K[8]*dzw;
                    const scalar_t s_b=sample_dispatch_cpu<scalar_t>((int)interp_method,Fb,Mx,My,Mz,M[0],M[1],M[2],M[6],M[7],M[8],bxq,byq,bzq);
                    Z+=std::exp(-(s_b-sdfu)*inv_tau);
                }
                if(Z<=(scalar_t)0) continue;
                const scalar_t invZ=(scalar_t)1/Z;
                for(int b=0;b<B;++b){
                    const int i0=(int)lo[b*3],j0=(int)lo[b*3+1],k0=(int)lo[b*3+2];
                    const int Ai=(int)dim[b*3],Aj=(int)dim[b*3+1],Ak=(int)dim[b*3+2];
                    if(i<i0||i>=i0+Ai||j<j0||j>=j0+Aj||k<k0||k>=k0+Ak) continue;
                    const scalar_t* Fb=F_ptr+F_off[b]; const int Mx=(int)shapes[b*3],My=(int)shapes[b*3+1],Mz=(int)shapes[b*3+2];
                    const scalar_t* M=meta+b*10; const scalar_t* K=kin_ptr+b*21;
                    const scalar_t dxw=xc-K[9],dyw=yc-K[10],dzw=zc-K[11];
                    const scalar_t bxq=K[0]*dxw+K[1]*dyw+K[2]*dzw, byq=K[3]*dxw+K[4]*dyw+K[5]*dzw, bzq=K[6]*dxw+K[7]*dyw+K[8]*dzw;
                    const scalar_t s_b=sample_dispatch_cpu<scalar_t>((int)interp_method,Fb,Mx,My,Mz,M[0],M[1],M[2],M[6],M[7],M[8],bxq,byq,bzq);
                    const scalar_t wb=std::exp(-(s_b-sdfu)*inv_tau)*invZ;
                    const scalar_t fbx=wb*fdx,fby=wb*fdy,fbz=wb*fdz;
                    const scalar_t ax=xc-K[12],ay=yc-K[13],az=zc-K[14];
                    outp[b*12+6]+=(double)fbx*h3d; outp[b*12+7]+=(double)fby*h3d; outp[b*12+8]+=(double)fbz*h3d;
                    outp[b*12+9]+=((double)ay*fbz-(double)az*fby)*h3d;
                    outp[b*12+10]+=((double)az*fbx-(double)ax*fbz)*h3d;
                    outp[b*12+11]+=((double)ax*fby-(double)ay*fbx)*h3d;
                }
            }
        }
    });
}
// ==== 2-D kernels =============================================================



// =====================================================================
//  Bilinear sample on a UNIFORM body grid (2-D analogue of
//  ``trilinear_sample_uniform``).  Returns the bilinearly-interpolated
//  value of F at world point (xq, yq), with edge clamping.
// =====================================================================
template <typename scalar_t>
static inline scalar_t bilinear_sample_uniform_2d(
    const scalar_t* F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    const scalar_t Mx_lim = (scalar_t)(Mx - 1);
    const scalar_t My_lim = (scalar_t)(My - 1);
    if (tx < (scalar_t)0) tx = (scalar_t)0; else if (tx > Mx_lim) tx = Mx_lim;
    if (ty < (scalar_t)0) ty = (scalar_t)0; else if (ty > My_lim) ty = My_lim;

    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;

    const scalar_t fx = tx - (scalar_t)ix;
    const scalar_t fy = ty - (scalar_t)iy;
    const scalar_t wx0 = (scalar_t)1 - fx, wx1 = fx;
    const scalar_t wy0 = (scalar_t)1 - fy, wy1 = fy;

    const int s1   = My;
    const int base = ix * s1 + iy;

    return (
        wx0 * (wy0 * F[base]      + wy1 * F[base + 1]) +
        wx1 * (wy0 * F[base + s1] + wy1 * F[base + s1 + 1])
    );
}

// =====================================================================
//  Biquadratic sample on a UNIFORM body grid (2-D analogue of
//  ``triquadratic_sample_uniform``).  Lagrange interpolation on a
//  3x3 stencil [ix-1, ix, ix+1] x [iy-1, iy, iy+1] with the same
//  lower-bracketing convention as the bilinear sampler.  Falls back to
//  bilinear in any cell whose lower stencil neighbour is out of range.
// =====================================================================
template <typename scalar_t>
static inline scalar_t biquadratic_sample_uniform_2d(
    const scalar_t* F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    const scalar_t Mx_lim = (scalar_t)(Mx - 1);
    const scalar_t My_lim = (scalar_t)(My - 1);
    tx = std::max((scalar_t)0, std::min(tx, Mx_lim));
    ty = std::max((scalar_t)0, std::min(ty, My_lim));

    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;

    if (ix < 1 || iy < 1 || Mx < 3 || My < 3) {
        return bilinear_sample_uniform_2d<scalar_t>(
            F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
    }

    const scalar_t fx = tx - (scalar_t)ix;
    const scalar_t fy = ty - (scalar_t)iy;

    const scalar_t half = (scalar_t)0.5;
    const scalar_t wxm = half * fx * (fx - (scalar_t)1);
    const scalar_t wx0 = (scalar_t)1 - fx * fx;
    const scalar_t wxp = half * fx * (fx + (scalar_t)1);
    const scalar_t wym = half * fy * (fy - (scalar_t)1);
    const scalar_t wy0 = (scalar_t)1 - fy * fy;
    const scalar_t wyp = half * fy * (fy + (scalar_t)1);

    const int s1   = My;
    const int base = (ix - 1) * s1 + (iy - 1);

    // Identical accumulation order as the CUDA kernel — running +=
    // accumulator so FMA contraction produces the same rounding.
    scalar_t out = (scalar_t)0;
    for (int dx = 0; dx < 3; ++dx) {
        const scalar_t wx = (dx == 0) ? wxm : (dx == 1 ? wx0 : wxp);
        const int b0 = base + dx * s1;
        const scalar_t col =
            wym * F[b0] + wy0 * F[b0 + 1] + wyp * F[b0 + 2];
        out += wx * col;
    }
    return out;
}

// =====================================================================
//  Per-cell update body for 2-D streaming-min update.  Each cell is
//  touched once per launch -> no atomics needed.
//
//  In 2-D the rigid-body angular velocity is a scalar ``omega``
//  (out-of-plane), and the face velocities at (xc-h/2, yc) and
//  (xc, yc-h/2) are:
//      u_face = lv_x - omega * (yc - cm_y)
//      v_face = lv_y + omega * (xc - cm_x)
//  These are the 3-D formulas with (av_x, av_y, av_z) = (0, 0, omega)
//  and zc - cm_z = 0 collapsed.
// =====================================================================
template <typename scalar_t>
static inline void update_cell_2d(
    const scalar_t* F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    const scalar_t r00, const scalar_t r01,
    const scalar_t r10, const scalar_t r11,
    const scalar_t bp_x, const scalar_t bp_y,
    const scalar_t cm_x, const scalar_t cm_y,
    const scalar_t lv_x, const scalar_t lv_y,
    const scalar_t omega,
    // Body-frame face deltas: Δ_u = -half_h * col0(R_T),
    //                          Δ_v = -half_h * col1(R_T).
    const scalar_t du_x, const scalar_t du_y,
    const scalar_t dv_x, const scalar_t dv_y,
    const scalar_t xc, const scalar_t yc,
    const std::int64_t g_idx, const std::int64_t sparse_idx,
    scalar_t* sdf_cc, scalar_t* sdf_u, scalar_t* sdf_v,
    scalar_t* bU, scalar_t* bV,
    scalar_t* sparse_cc,
    // 0 = bilinear (default), 1 = biquadratic (Lagrange 3x3).
    const int interp_method)
{
    // Body-frame CC point (single 2x2 rotation; the 2 face points are
    // derived from it by precomputed deltas).
    const scalar_t dxw = xc - bp_x, dyw = yc - bp_y;
    const scalar_t bxq = r00 * dxw + r01 * dyw;
    const scalar_t byq = r10 * dxw + r11 * dyw;

    auto sample = [&](scalar_t xqs, scalar_t yqs) -> scalar_t {
        if (interp_method == 1) {
            return biquadratic_sample_uniform_2d<scalar_t>(
                F, Mx, My, bx0, by0, inv_dx, inv_dy, xqs, yqs);
        }
        return bilinear_sample_uniform_2d<scalar_t>(
            F, Mx, My, bx0, by0, inv_dx, inv_dy, xqs, yqs);
    };

    // ---- cc ----
    {
        const scalar_t s = sample(bxq, byq);
        sparse_cc[sparse_idx] = s;
        if (s < sdf_cc[g_idx]) sdf_cc[g_idx] = s;
    }
    // ---- u-face: world (xc - h/2, yc) ----
    {
        const scalar_t s = sample(bxq + du_x, byq + du_y);
        if (s < sdf_u[g_idx]) {
            sdf_u[g_idx] = s;
            bU[g_idx] = lv_x - omega * (yc - cm_y);
        }
    }
    // ---- v-face: world (xc, yc - h/2) ----
    {
        const scalar_t s = sample(bxq + dv_x, byq + dv_y);
        if (s < sdf_v[g_idx]) {
            sdf_v[g_idx] = s;
            bV[g_idx] = lv_y + omega * (xc - cm_x);
        }
    }
}


// =====================================================================
//  apply_bcs_2d  (Phase H 2-D analogue of apply_bcs_3d)
//
//  Each op writes a single 1-D ghost line of u or v.  Ops run in three
//  ordered stages (Neumann → Dirichlet → reflective); within a stage the
//  write-ownership rule in bc_ops.h gives every cell — corner ghosts
//  included — one writer and a race-free source, so CPU and CUDA agree
//  cell for cell.  See bc_ops.h.
//
//  shapes  : int64 [2,2]      -> per-component (Nx, Ny)
//  neu_desc: int32 [N_neu, 3] -> (comp, axis, side)
//                                   comp in {0=u, 1=v}
//                                   axis in {0=x, 1=y}
//                                   side in {0=lo, 1=hi}
//  dir_desc: int32 [N_dir, 3] -> (comp, axis, offset)
//                                offset is signed: dst = offset if >=0
//                                else (sz + offset).
//  dir_val : float[N_dir]
// =====================================================================

template <typename scalar_t>
static void apply_bcs_2d_one_line(
    scalar_t* base,
    const std::int64_t* shapes_p,
    const int kind, const int* desc, const int nops, const int op,
    const scalar_t* vals,
    const int comp, const int axis,
    const int dst_along, const int src_along,
    const int Ny, const int dim0_max)
{
    using namespace lilytorch_kernels::bcs;

    // Flat layout: base[i, j] = base[i*Ny + j].
    at::parallel_for(0, dim0_max, /*grain_size=*/1024, [&](int64_t _begin, int64_t _end) {
    for (int64_t t = _begin; t < _end; ++t) {
        const int i = (int)t;

        int c[2];
        if (axis == 0) { c[0] = dst_along; c[1] = i;         }
        else           { c[0] = i;         c[1] = dst_along; }

        int s[2];
        if (!bc_own_and_source(kind, desc, nops, op, shapes_p, /*ndim=*/2,
                               comp, axis, src_along, c, s))
            continue;                              // a lower-indexed op owns it

        const std::int64_t dst_lin = (std::int64_t)c[0] * Ny + c[1];
        const std::int64_t src_lin = (std::int64_t)s[0] * Ny + s[1];

        if      (kind == BC_KIND_NEUMANN)   base[dst_lin] = base[src_lin];
        else if (kind == BC_KIND_DIRICHLET) base[dst_lin] = vals[op];
        else base[dst_lin] = scalar_t(2) * vals[op] - base[src_lin];
    }
    });
}

void apply_bcs_2d_cpu(
    at::Tensor u, at::Tensor v,
    const at::Tensor& shapes,
    const at::Tensor& neu_desc,
    const at::Tensor& dir_desc,
    const at::Tensor& dir_val,
    const at::Tensor& ref_desc,
    const at::Tensor& ref_val,
    const int64_t /*max_line_dim*/)
{
    TORCH_CHECK(u.is_contiguous() && v.is_contiguous(),
                "apply_bcs_2d_cpu: u/v must be contiguous");
    TORCH_CHECK(u.scalar_type() == v.scalar_type(),
                "apply_bcs_2d_cpu: u/v must share dtype");
    TORCH_CHECK(shapes.scalar_type() == at::kLong &&
                shapes.dim() == 2 && shapes.size(0) == 2 && shapes.size(1) == 2,
                "apply_bcs_2d_cpu: shapes must be int64[2,2]");
    TORCH_CHECK(neu_desc.scalar_type() == at::kInt && neu_desc.dim() == 2 &&
                neu_desc.size(1) == 3,
                "apply_bcs_2d_cpu: neu_desc must be int32[N,3]");
    TORCH_CHECK(dir_desc.scalar_type() == at::kInt && dir_desc.dim() == 2 &&
                dir_desc.size(1) == 3,
                "apply_bcs_2d_cpu: dir_desc must be int32[N,3]");
    TORCH_CHECK(ref_desc.scalar_type() == at::kInt && ref_desc.dim() == 2 &&
                ref_desc.size(1) == 4,
                "apply_bcs_2d_cpu: ref_desc must be int32[N,4]");

    const int N_neu = (int)neu_desc.size(0);
    const int N_dir = (int)dir_desc.size(0);
    const int N_ref = (int)ref_desc.size(0);
    if (N_neu + N_dir + N_ref == 0) return;

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_2d_cpu", [&] {
        using namespace lilytorch_kernels::bcs;

        const int64_t*  shapes_p  = shapes.data_ptr<int64_t>();
        const int*      neu_p     = (N_neu > 0) ? neu_desc.data_ptr<int>() : nullptr;
        const int*      dir_p     = (N_dir > 0) ? dir_desc.data_ptr<int>() : nullptr;
        const scalar_t* dir_val_p = (N_dir > 0) ? dir_val.data_ptr<scalar_t>() : nullptr;
        const int*      ref_p     = (N_ref > 0) ? ref_desc.data_ptr<int>() : nullptr;
        const scalar_t* ref_val_p = (N_ref > 0) ? ref_val.data_ptr<scalar_t>() : nullptr;

        scalar_t* u_p = u.data_ptr<scalar_t>();
        scalar_t* v_p = v.data_ptr<scalar_t>();

        // One stage per op kind, in the same order as the CUDA launches.
        const int   kinds[3] = {BC_KIND_NEUMANN, BC_KIND_DIRICHLET, BC_KIND_REFLECTIVE};
        const int*  descs[3] = {neu_p, dir_p, ref_p};
        const scalar_t* valss[3] = {nullptr, dir_val_p, ref_val_p};
        const int   counts[3] = {N_neu, N_dir, N_ref};

        for (int st = 0; st < 3; ++st) {
            const int kind = kinds[st];
            const int* desc = descs[st];
            const int nops = counts[st];

            for (int op = 0; op < nops; ++op) {
                int comp, axis, dst_along, src_along;
                bc_decode(kind, desc, op, shapes_p, /*ndim=*/2,
                          comp, axis, dst_along, src_along);

                const int Nx = (int)shapes_p[comp*2 + 0];
                const int Ny = (int)shapes_p[comp*2 + 1];

                // axis==0 -> sweep along j (size Ny). axis==1 -> sweep along i (size Nx).
                const int dim0_max = (axis == 0) ? Ny : Nx;

                scalar_t* base = (comp == 0) ? u_p : v_p;

                apply_bcs_2d_one_line<scalar_t>(
                    base, shapes_p, kind, desc, nops, op, valss[st],
                    comp, axis, dst_along, src_along,
                    Ny, dim0_max);
            }
        }
    });
}

// =====================================================================
//  interp_2d_cpu: scattered-point bilinear / biquadratic sampling
// =====================================================================
static void interp_2d_cpu(
    const at::Tensor& F,
    const at::Tensor& xq, const at::Tensor& yq,
    const double bx0, const double by0,
    const double inv_dx, const double inv_dy,
    const int64_t Mx, const int64_t My,
    const int64_t interp_method,
    at::Tensor& G)
{
    const int N = (int)xq.numel();
    if (N == 0) return;

    // Bind temporaries to named locals: ``.contiguous().to(...)`` returns a
    // fresh tensor whose storage is freed at the end of the full-expression
    // unless held.  Without this the raw ``data_ptr`` would dangle for
    // non-contiguous or differently-dtyped inputs and the parallel_for loop
    // below would read freed memory.
    auto F_c  = F.contiguous();
    auto xq_c = xq.contiguous().to(F.scalar_type());
    auto yq_c = yq.contiguous().to(F.scalar_type());

    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "interp_2d_cpu", [&] {
        const scalar_t* Fp  = F_c.data_ptr<scalar_t>();
        const scalar_t* xqp = xq_c.data_ptr<scalar_t>();
        const scalar_t* yqp = yq_c.data_ptr<scalar_t>();
        scalar_t* Gp = G.data_ptr<scalar_t>();
        const int iMx = (int)Mx, iMy = (int)My;
        const scalar_t bx0s = (scalar_t)bx0, by0s = (scalar_t)by0;
        const scalar_t idx  = (scalar_t)inv_dx, idy = (scalar_t)inv_dy;
        const int method    = (int)interp_method;

        at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                if (method == 1) {
                    Gp[i] = biquadratic_sample_uniform_2d<scalar_t>(
                        Fp, iMx, iMy, bx0s, by0s, idx, idy,
                        xqp[i], yqp[i]);
                } else {
                    Gp[i] = bilinear_sample_uniform_2d<scalar_t>(
                        Fp, iMx, iMy, bx0s, by0s, idx, idy,
                        xqp[i], yqp[i]);
                }
            }
        });
    });
}

// =====================================================================
void streaming_sdf_forces_post_2d_cpu(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes, const at::Tensor& body_meta, const at::Tensor& kin,
    const at::Tensor& aabb_lo, const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy,
    const double h_grid, const int64_t /*max_vol_per_body*/,
    const at::Tensor& sdf_cc,
    const int64_t interp_method,
    const at::Tensor& u_prev, const at::Tensor& v_prev, const at::Tensor& p_prev,
    const at::Tensor& nu_rho_field,
    const double eps_body, const double eps_solver, const double h2,
    const int64_t delta_order,
    const int64_t force_submethod,
    const double ph_tau,
    at::Tensor out)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0) return;

    TORCH_CHECK(out.scalar_type() == at::kDouble,
                "streaming_sdf_forces_post_2d_cpu: out must be float64");
    TORCH_CHECK(out.size(1) == 6,
                "streaming_sdf_forces_post_2d_cpu: out must have 6 channels");

    const int Ngy = (int)gy.numel();
    const int Ngx = (int)gx.numel();

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_forces_post_2d_cpu", [&] {
        auto F_c   = F_flat.contiguous();
        auto gx_c  = gx.contiguous();
        auto gy_c  = gy.contiguous();
        auto sdf_c = sdf_cc.contiguous();
        auto up_c  = u_prev.contiguous();
        auto vp_c  = v_prev.contiguous();
        auto pp_c  = p_prev.contiguous();
        auto nr_c  = nu_rho_field.contiguous();

        const scalar_t* F_ptr   = F_c.data_ptr<scalar_t>();
        const int64_t*  F_off   = F_offsets.data_ptr<int64_t>();
        const int64_t*  shapes  = body_shapes.data_ptr<int64_t>();
        const scalar_t* meta    = body_meta.data_ptr<scalar_t>();
        const scalar_t* kin_ptr = kin.data_ptr<scalar_t>();
        const int64_t*  lo      = aabb_lo.data_ptr<int64_t>();
        const int64_t*  dim_    = aabb_dim.data_ptr<int64_t>();
        const scalar_t* gx_ptr  = gx_c.data_ptr<scalar_t>();
        const scalar_t* gy_ptr  = gy_c.data_ptr<scalar_t>();
        const scalar_t* sdf_cc_p= sdf_c.data_ptr<scalar_t>();
        const scalar_t* u_p     = up_c.data_ptr<scalar_t>();
        const scalar_t* v_p     = vp_c.data_ptr<scalar_t>();
        const scalar_t* p_p     = pp_c.data_ptr<scalar_t>();
        const scalar_t* nr_p    = nr_c.data_ptr<scalar_t>();
        const int64_t   nr_size = nu_rho_field.numel();

        const scalar_t inv_h_s    = (scalar_t)(1.0 / h_grid);
        const scalar_t eps_b      = (scalar_t)eps_body;
        const scalar_t eps_s      = (scalar_t)eps_solver;
        const scalar_t pi_v       = (scalar_t)3.141592653589793;
        const scalar_t inv_2eps   = (scalar_t)0.5 / eps_b;
        const scalar_t pi_over_eb = pi_v / eps_b;
        const double   h2_d       = (double)h2;

        const scalar_t band_lo = (eps_s - eps_b) < (-eps_b) ? (eps_s - eps_b) : (-eps_b);
        const scalar_t band_hi = (eps_s + eps_b) > ( eps_b) ? (eps_s + eps_b) : ( eps_b);

        std::vector<double> accs(static_cast<size_t>(B) * 6, 0.0);

        for (int b = 0; b < B; ++b) {
            const int Ai = (int)dim_[b*2 + 0];
            const int Aj = (int)dim_[b*2 + 1];
            const int vol = Ai * Aj;
            if (vol <= 0) continue;

            const int i0_b = (int)lo[b*2 + 0];
            const int j0_b = (int)lo[b*2 + 1];

            const scalar_t* F_b = F_ptr + F_off[b];
            const int Mx = (int)shapes[b*2 + 0];
            const int My = (int)shapes[b*2 + 1];

            const scalar_t* M   = meta + b*7;
            const scalar_t bx0  = M[0], by0 = M[1];
            const scalar_t idx_ = M[4], idy_ = M[5];

            const scalar_t* K   = kin_ptr + b*11;
            const scalar_t r00  = K[0], r01 = K[1];
            const scalar_t r10  = K[2], r11 = K[3];
            const scalar_t bp_x = K[4], bp_y = K[5];
            const scalar_t cm_x = K[6], cm_y = K[7];

            // Single fused pass: sample the body cc-SDF on the fly inside the
            // parallel_for, do the band check, and (for delta_order==2)
            // re-sample at world-aligned ±h offsets to recover |grad(sdf_body)|.
            // This replaces the previous two-pass design with a sparse_buf
            // scratch + AABB-edge one-sided diffs, which disagreed with CUDA-2D
            // at AABB-boundary cells.  Now matches streaming_sdf_2d.cu:524-543.
            //
            // Per-thread accumulator slice (no locking): each worker writes
            // only its own 6-channel stripe of ``tls``, merged single-thread
            // into ``lb`` after the parallel region.  Replaces the previous
            // std::mutex/lock_guard that serialised every chunk's epilogue.
            double* lb = accs.data() + (size_t)b * 6;
            const int nT = at::get_num_threads();
            std::vector<double> tls((size_t)nT * 6, 0.0);

            at::parallel_for(0, vol, 2048, [&](int64_t _begin, int64_t _end) {
            double local8[6] = {0,0,0,0,0,0};
            for (int local = (int)_begin; local < (int)_end; ++local) {
                const int di = local / Aj;
                const int dj = local - di * Aj;
                const int i  = i0_b + di;
                const int j  = j0_b + dj;
                const std::int64_t g_idx = (std::int64_t)i * Ngy + j;

                const scalar_t xc = gx_ptr[i];
                const scalar_t yc = gy_ptr[j];
                const scalar_t dxw = xc - bp_x, dyw = yc - bp_y;
                const scalar_t bxq = r00 * dxw + r01 * dyw;
                const scalar_t byq = r10 * dxw + r11 * dyw;

                const scalar_t sdf = (interp_method == 1)
                    ? biquadratic_sample_uniform_2d<scalar_t>(
                          F_b, Mx, My, bx0, by0, idx_, idy_, bxq, byq)
                    : bilinear_sample_uniform_2d<scalar_t>(
                          F_b, Mx, My, bx0, by0, idx_, idy_, bxq, byq);
                if (sdf <= band_lo || sdf >= band_hi) continue;

                scalar_t delta_visc = 0;
                const scalar_t d_visc = sdf - eps_s;
                if (d_visc > -eps_b && d_visc < eps_b)
                    delta_visc = ((scalar_t)1 + std::cos(pi_over_eb * d_visc)) * inv_2eps;
                scalar_t delta_pres = 0;
                if (sdf > -eps_b && sdf < eps_b)
                    delta_pres = ((scalar_t)1 + std::cos(pi_over_eb * sdf)) * inv_2eps;
                // deltaH: pressure force/torque come from the union-∂H pass below.
                if (force_submethod != 0) delta_pres = 0;
                if (delta_visc == (scalar_t)0 && delta_pres == (scalar_t)0) continue;

                if (delta_order == 2 && (delta_visc > 0 || delta_pres > 0)) {
                    const scalar_t hg = (scalar_t)1.0 / inv_h_s;
                    auto smp = [&](scalar_t xqs, scalar_t yqs) -> scalar_t {
                        return (interp_method == 1)
                            ? biquadratic_sample_uniform_2d<scalar_t>(
                                  F_b, Mx, My, bx0, by0, idx_, idy_, xqs, yqs)
                            : bilinear_sample_uniform_2d<scalar_t>(
                                  F_b, Mx, My, bx0, by0, idx_, idy_, xqs, yqs);
                    };
                    const scalar_t s_xp = smp(bxq + r00*hg, byq + r10*hg);
                    const scalar_t s_xm = smp(bxq - r00*hg, byq - r10*hg);
                    const scalar_t s_yp = smp(bxq + r01*hg, byq + r11*hg);
                    const scalar_t s_ym = smp(bxq - r01*hg, byq - r11*hg);
                    const scalar_t dsdx = (s_xp - s_xm) * (scalar_t)0.5 * inv_h_s;
                    const scalar_t dsdy = (s_yp - s_ym) * (scalar_t)0.5 * inv_h_s;
                    scalar_t grad_mag = std::sqrt(dsdx*dsdx + dsdy*dsdy);
                    if (grad_mag < (scalar_t)1e-3) grad_mag = (scalar_t)1e-3;
                    const scalar_t inv_grad = (scalar_t)1.0 / grad_mag;
                    delta_visc *= inv_grad;
                    delta_pres *= inv_grad;
                }

                const scalar_t nu_rho_val = (nr_size == 1) ? nr_p[0] : nr_p[g_idx];

                scalar_t dsdx_union = 0;
                if (Ngx >= 3) {
                    if (i == 0) {
                        dsdx_union = (
                            (scalar_t)(-3) * sdf_cc_p[j]
                            + (scalar_t)4 * sdf_cc_p[Ngy + j]
                            - sdf_cc_p[2 * Ngy + j]
                        ) * (scalar_t)0.5 * inv_h_s;
                    } else if (i == Ngx - 1) {
                        dsdx_union = (
                            (scalar_t)3 * sdf_cc_p[(Ngx - 1) * Ngy + j]
                            - (scalar_t)4 * sdf_cc_p[(Ngx - 2) * Ngy + j]
                            + sdf_cc_p[(Ngx - 3) * Ngy + j]
                        ) * (scalar_t)0.5 * inv_h_s;
                    } else {
                        dsdx_union = (
                            sdf_cc_p[(i + 1) * Ngy + j]
                            - sdf_cc_p[(i - 1) * Ngy + j]
                        ) * (scalar_t)0.5 * inv_h_s;
                    }
                } else if (Ngx == 2) {
                    dsdx_union = (sdf_cc_p[Ngy + j] - sdf_cc_p[j]) * inv_h_s;
                }

                scalar_t dsdy_union = 0;
                if (Ngy >= 3) {
                    const int row = i * Ngy;
                    if (j == 0) {
                        dsdy_union = (
                            (scalar_t)(-3) * sdf_cc_p[row]
                            + (scalar_t)4 * sdf_cc_p[row + 1]
                            - sdf_cc_p[row + 2]
                        ) * (scalar_t)0.5 * inv_h_s;
                    } else if (j == Ngy - 1) {
                        dsdy_union = (
                            (scalar_t)3 * sdf_cc_p[row + (Ngy - 1)]
                            - (scalar_t)4 * sdf_cc_p[row + (Ngy - 2)]
                            + sdf_cc_p[row + (Ngy - 3)]
                        ) * (scalar_t)0.5 * inv_h_s;
                    } else {
                        dsdy_union = (
                            sdf_cc_p[row + (j + 1)]
                            - sdf_cc_p[row + (j - 1)]
                        ) * (scalar_t)0.5 * inv_h_s;
                    }
                } else if (Ngy == 2) {
                    dsdy_union = (sdf_cc_p[i * Ngy + 1] - sdf_cc_p[i * Ngy]) * inv_h_s;
                }

                const scalar_t union_norm = std::sqrt(dsdx_union*dsdx_union + dsdy_union*dsdy_union);
                const scalar_t union_inv_norm = union_norm > (scalar_t)0
                    ? ((scalar_t)1.0 / union_norm)
                    : (scalar_t)0;
                const scalar_t nx = dsdx_union * union_inv_norm;
                const scalar_t ny = dsdy_union * union_inv_norm;

                const int im1 = (i > 0)         ? i-1 : 0;
                const int ip1 = (i+1 < Ngx)     ? i+1 : i;
                const int im2 = (i > 1)         ? i-2 : 0;
                const int ip2 = (i+2 < Ngx)     ? i+2 : (Ngx - 1);
                const int jm1 = (j > 0)         ? j-1 : 0;
                const int jp1 = (j+1 < Ngy)     ? j+1 : j;
                const int jm2 = (j > 1)         ? j-2 : 0;
                const int jp2 = (j+2 < Ngy)     ? j+2 : (Ngy - 1);

                scalar_t dudx;
                if (i + 1 < Ngx) {
                    dudx = (u_p[ip1 * Ngy + j] - u_p[i * Ngy + j]) * inv_h_s;
                } else {
                    dudx = (u_p[i * Ngy + j] - u_p[im1 * Ngy + j]) * inv_h_s;
                }

                scalar_t dvdy;
                if (j + 1 < Ngy) {
                    dvdy = (v_p[i * Ngy + jp1] - v_p[i * Ngy + j]) * inv_h_s;
                } else {
                    dvdy = (v_p[i * Ngy + j] - v_p[i * Ngy + jm1]) * inv_h_s;
                }

                const scalar_t u_cc_jm2 = (scalar_t)0.5 * (u_p[i * Ngy + jm2] + u_p[ip1 * Ngy + jm2]);
                const scalar_t u_cc_jm1 = (scalar_t)0.5 * (u_p[i * Ngy + jm1] + u_p[ip1 * Ngy + jm1]);
                const scalar_t u_cc_j0  = (scalar_t)0.5 * (u_p[i * Ngy + j  ] + u_p[ip1 * Ngy + j  ]);
                const scalar_t u_cc_jp1 = (scalar_t)0.5 * (u_p[i * Ngy + jp1] + u_p[ip1 * Ngy + jp1]);
                const scalar_t u_cc_jp2 = (scalar_t)0.5 * (u_p[i * Ngy + jp2] + u_p[ip1 * Ngy + jp2]);

                scalar_t dudy;
                if (Ngy >= 3) {
                    if (j == 0) {
                        dudy = ((scalar_t)(-3) * u_cc_j0 + (scalar_t)4 * u_cc_jp1 - u_cc_jp2)
                             * (scalar_t)0.5 * inv_h_s;
                    } else if (j == Ngy - 1) {
                        dudy = ((scalar_t)3 * u_cc_j0 - (scalar_t)4 * u_cc_jm1 + u_cc_jm2)
                             * (scalar_t)0.5 * inv_h_s;
                    } else {
                        dudy = (u_cc_jp1 - u_cc_jm1) * (scalar_t)0.5 * inv_h_s;
                    }
                } else {
                    dudy = (u_cc_jp1 - u_cc_jm1) * (scalar_t)0.5 * inv_h_s;
                }

                const scalar_t v_cc_im2 = (scalar_t)0.5 * (v_p[im2 * Ngy + j] + v_p[im2 * Ngy + jp1]);
                const scalar_t v_cc_im1 = (scalar_t)0.5 * (v_p[im1 * Ngy + j] + v_p[im1 * Ngy + jp1]);
                const scalar_t v_cc_i0  = (scalar_t)0.5 * (v_p[i   * Ngy + j] + v_p[i   * Ngy + jp1]);
                const scalar_t v_cc_ip1 = (scalar_t)0.5 * (v_p[ip1 * Ngy + j] + v_p[ip1 * Ngy + jp1]);
                const scalar_t v_cc_ip2 = (scalar_t)0.5 * (v_p[ip2 * Ngy + j] + v_p[ip2 * Ngy + jp1]);

                scalar_t dvdx;
                if (Ngx >= 3) {
                    if (i == 0) {
                        dvdx = ((scalar_t)(-3) * v_cc_i0 + (scalar_t)4 * v_cc_ip1 - v_cc_ip2)
                             * (scalar_t)0.5 * inv_h_s;
                    } else if (i == Ngx - 1) {
                        dvdx = ((scalar_t)3 * v_cc_i0 - (scalar_t)4 * v_cc_im1 + v_cc_im2)
                             * (scalar_t)0.5 * inv_h_s;
                    } else {
                        dvdx = (v_cc_ip1 - v_cc_im1) * (scalar_t)0.5 * inv_h_s;
                    }
                } else {
                    dvdx = (v_cc_ip1 - v_cc_im1) * (scalar_t)0.5 * inv_h_s;
                }

                const scalar_t xs = nu_rho_val * (2*dudx*nx + (dudy+dvdx)*ny);
                const scalar_t ys = nu_rho_val * ((dvdx+dudy)*nx + 2*dvdy*ny);

                const scalar_t p_c  = p_p[g_idx];
                const scalar_t pxv  = -p_c * nx;
                const scalar_t pyv  = -p_c * ny;

                const scalar_t arm_x = xc - cm_x;
                const scalar_t arm_y = yc - cm_y;

                const double fv_x = (double)(xs  * delta_visc);
                const double fv_y = (double)(ys  * delta_visc);
                const double fp_x = (double)(pxv * delta_pres);
                const double fp_y = (double)(pyv * delta_pres);

                local8[0] += fv_x;
                local8[1] += fv_y;
                local8[2] += (double)arm_x * fv_y - (double)arm_y * fv_x;
                local8[3] += fp_x;
                local8[4] += fp_y;
                local8[5] += (double)arm_x * fp_y - (double)arm_y * fp_x;
            }
            const int t = at::get_thread_num();
            for (int c = 0; c < 6; ++c) tls[(size_t)t*6 + c] += local8[c];
            });
            for (int t = 0; t < nT; ++t)
                for (int c = 0; c < 6; ++c) lb[c] += tls[(size_t)t*6 + c];
        }

        // ---- deltaH: union-∂H pressure force density, softmin partition ----
        if (force_submethod != 0) {
            const scalar_t inv_eps = (scalar_t)(1.0 / eps_body);
            const double tau = (ph_tau > 0.0) ? ph_tau : 1e-9;
            const scalar_t inv_tau = (scalar_t)(1.0 / tau);
            auto sample_b = [&](const scalar_t* Fb, int Mx, int My, scalar_t bx0,
                                scalar_t by0, scalar_t idx_, scalar_t idy_,
                                scalar_t xq, scalar_t yq) -> scalar_t {
                return (interp_method == 1)
                    ? biquadratic_sample_uniform_2d<scalar_t>(Fb, Mx, My, bx0, by0, idx_, idy_, xq, yq)
                    : bilinear_sample_uniform_2d<scalar_t>(Fb, Mx, My, bx0, by0, idx_, idy_, xq, yq);
            };
            int ulo[2] = {Ngx, Ngy}, uhi[2] = {0, 0};
            for (int b = 0; b < B; ++b)
                for (int d = 0; d < 2; ++d) { int a0=(int)lo[b*2+d]; int a1=a0+(int)dim_[b*2+d]; if(a0<ulo[d])ulo[d]=a0; if(a1>uhi[d])uhi[d]=a1; }
            const int Ng[2] = {Ngx, Ngy}; const int halo = 2;
            for (int d = 0; d < 2; ++d) { ulo[d]-=halo; if(ulo[d]<0)ulo[d]=0; uhi[d]+=halo; if(uhi[d]>Ng[d])uhi[d]=Ng[d]; }
            auto Hs = [&](int i, int j) -> scalar_t {
                scalar_t x = sdf_cc_p[(std::int64_t)i*Ngy+j] * inv_eps;
                x = x < (scalar_t)-1 ? (scalar_t)-1 : (x > (scalar_t)1 ? (scalar_t)1 : x);
                return (scalar_t)0.5 * ((scalar_t)1 + x + std::sin(pi_v * x) / pi_v);
            };
            for (int i = ulo[0]; i < uhi[0]; ++i)
            for (int j = ulo[1]; j < uhi[1]; ++j) {
                scalar_t gHx = 0, gHy = 0;
                if (Ngx >= 3) { if(i==0) gHx=((-3)*Hs(0,j)+4*Hs(1,j)-Hs(2,j))*(scalar_t)0.5*inv_h_s; else if(i==Ngx-1) gHx=(3*Hs(Ngx-1,j)-4*Hs(Ngx-2,j)+Hs(Ngx-3,j))*(scalar_t)0.5*inv_h_s; else gHx=(Hs(i+1,j)-Hs(i-1,j))*(scalar_t)0.5*inv_h_s; } else if(Ngx==2) gHx=(Hs(1,j)-Hs(0,j))*inv_h_s;
                if (Ngy >= 3) { if(j==0) gHy=((-3)*Hs(i,0)+4*Hs(i,1)-Hs(i,2))*(scalar_t)0.5*inv_h_s; else if(j==Ngy-1) gHy=(3*Hs(i,Ngy-1)-4*Hs(i,Ngy-2)+Hs(i,Ngy-3))*(scalar_t)0.5*inv_h_s; else gHy=(Hs(i,j+1)-Hs(i,j-1))*(scalar_t)0.5*inv_h_s; } else if(Ngy==2) gHy=(Hs(i,1)-Hs(i,0))*inv_h_s;
                if (gHx == (scalar_t)0 && gHy == (scalar_t)0) continue;
                const std::int64_t g = (std::int64_t)i*Ngy+j;
                const scalar_t p_c = p_p[g];
                const scalar_t fdx = -p_c*gHx, fdy = -p_c*gHy;
                const scalar_t xc = gx_ptr[i], yc = gy_ptr[j];
                const scalar_t sdfu = sdf_cc_p[g];
                scalar_t Z = 0;
                for (int b = 0; b < B; ++b) {
                    const int i0=(int)lo[b*2],j0=(int)lo[b*2+1],Ai=(int)dim_[b*2],Aj=(int)dim_[b*2+1];
                    if (i<i0||i>=i0+Ai||j<j0||j>=j0+Aj) continue;
                    const scalar_t* Fb=F_ptr+F_off[b]; const int Mx=(int)shapes[b*2],My=(int)shapes[b*2+1];
                    const scalar_t* M=meta+b*7; const scalar_t* K=kin_ptr+b*11;
                    const scalar_t dxw=xc-K[4],dyw=yc-K[5];
                    const scalar_t bxq=K[0]*dxw+K[1]*dyw, byq=K[2]*dxw+K[3]*dyw;
                    Z += std::exp(-(sample_b(Fb,Mx,My,M[0],M[1],M[4],M[5],bxq,byq)-sdfu)*inv_tau);
                }
                if (Z <= (scalar_t)0) continue;
                const scalar_t invZ = (scalar_t)1/Z;
                for (int b = 0; b < B; ++b) {
                    const int i0=(int)lo[b*2],j0=(int)lo[b*2+1],Ai=(int)dim_[b*2],Aj=(int)dim_[b*2+1];
                    if (i<i0||i>=i0+Ai||j<j0||j>=j0+Aj) continue;
                    const scalar_t* Fb=F_ptr+F_off[b]; const int Mx=(int)shapes[b*2],My=(int)shapes[b*2+1];
                    const scalar_t* M=meta+b*7; const scalar_t* K=kin_ptr+b*11;
                    const scalar_t dxw=xc-K[4],dyw=yc-K[5];
                    const scalar_t bxq=K[0]*dxw+K[1]*dyw, byq=K[2]*dxw+K[3]*dyw;
                    const scalar_t s_b=sample_b(Fb,Mx,My,M[0],M[1],M[4],M[5],bxq,byq);
                    const scalar_t wb=std::exp(-(s_b-sdfu)*inv_tau)*invZ;
                    const scalar_t fbx=wb*fdx,fby=wb*fdy;
                    const scalar_t ax=xc-K[6],ay=yc-K[7];
                    double* lb=accs.data()+(size_t)b*6;
                    lb[3]+=(double)fbx; lb[4]+=(double)fby; lb[5]+=(double)ax*fby-(double)ay*fbx;
                }
            }
        }

        double* out_ptr = out.data_ptr<double>();
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < 6; ++c)
                out_ptr[b*6 + c] += accs[(size_t)b * 6 + c] * h2_d;
    });
}

// =====================================================================
//  CPU registration. Schemas live in ops.cpp.
// =====================================================================
TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("interp_3d", &interp_3d_cpu);
    m.impl("interp_2d", &interp_2d_cpu);
}

}  // namespace lilytorch_kernels
