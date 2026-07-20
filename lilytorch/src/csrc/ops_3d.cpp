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

#include "common/bc_ops.h"

namespace lilytorch_kernels {

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
    const double sample_offset_pressure,
    const double sample_offset_friction,
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
        const scalar_t inv_h=(scalar_t)(1.0/h_grid);
        const scalar_t off_pres=(scalar_t)sample_offset_pressure;
        const scalar_t off_visc=(scalar_t)sample_offset_friction;
        const scalar_t pi=(scalar_t)3.141592653589793;
        const double h3d=(double)h3;
        auto S=[&](int i,int j,int k)->scalar_t{return sdf[((int64_t)i*Ngy+j)*Ngz+k];};
        // ---- submethod 0 (union grad-H direction) / 2 (per-body analytic
        //      normal): the UNIFIED union-measure + partition readout.  Mirrors
        //      forces_post_union_blend_3d_kernel in cuda/eulerian_forces.cu. ----
        const bool body_normal = (force_submethod == 2);
        const scalar_t inv_eps = (scalar_t)(1.0/eps_body);
        const scalar_t blend_eps = (scalar_t)ph_tau;
        const bool soft = blend_eps > (scalar_t)0;
        const scalar_t inv_blend = soft ? ((scalar_t)1/blend_eps) : (scalar_t)0;
        auto Hoff=[&](int i,int j,int k,scalar_t off)->scalar_t{
            scalar_t x=(S(i,j,k)-off)*inv_eps; x=x<(scalar_t)-1?(scalar_t)-1:(x>(scalar_t)1?(scalar_t)1:x);
            return (scalar_t)0.5*((scalar_t)1+x+std::sin(pi*x)/pi); };
        auto hgrad=[&](int i,int j,int k,scalar_t off,scalar_t& gX,scalar_t& gY,scalar_t& gZ){
            gX=0;gY=0;gZ=0;
            if(Ngx>=3){ if(i==0) gX=((scalar_t)(-3)*Hoff(0,j,k,off)+(scalar_t)4*Hoff(1,j,k,off)-Hoff(2,j,k,off))*(scalar_t)0.5*inv_h; else if(i==Ngx-1) gX=((scalar_t)3*Hoff(Ngx-1,j,k,off)-(scalar_t)4*Hoff(Ngx-2,j,k,off)+Hoff(Ngx-3,j,k,off))*(scalar_t)0.5*inv_h; else gX=(Hoff(i+1,j,k,off)-Hoff(i-1,j,k,off))*(scalar_t)0.5*inv_h; } else if(Ngx==2) gX=(Hoff(1,j,k,off)-Hoff(0,j,k,off))*inv_h;
            if(Ngy>=3){ if(j==0) gY=((scalar_t)(-3)*Hoff(i,0,k,off)+(scalar_t)4*Hoff(i,1,k,off)-Hoff(i,2,k,off))*(scalar_t)0.5*inv_h; else if(j==Ngy-1) gY=((scalar_t)3*Hoff(i,Ngy-1,k,off)-(scalar_t)4*Hoff(i,Ngy-2,k,off)+Hoff(i,Ngy-3,k,off))*(scalar_t)0.5*inv_h; else gY=(Hoff(i,j+1,k,off)-Hoff(i,j-1,k,off))*(scalar_t)0.5*inv_h; } else if(Ngy==2) gY=(Hoff(i,1,k,off)-Hoff(i,0,k,off))*inv_h;
            if(Ngz>=3){ if(k==0) gZ=((scalar_t)(-3)*Hoff(i,j,0,off)+(scalar_t)4*Hoff(i,j,1,off)-Hoff(i,j,2,off))*(scalar_t)0.5*inv_h; else if(k==Ngz-1) gZ=((scalar_t)3*Hoff(i,j,Ngz-1,off)-(scalar_t)4*Hoff(i,j,Ngz-2,off)+Hoff(i,j,Ngz-3,off))*(scalar_t)0.5*inv_h; else gZ=(Hoff(i,j,k+1,off)-Hoff(i,j,k-1,off))*(scalar_t)0.5*inv_h; } else if(Ngz==2) gZ=(Hoff(i,j,1,off)-Hoff(i,j,0,off))*inv_h;
        };
        auto sgrad=[&](int i,int j,int k,scalar_t& gX,scalar_t& gY,scalar_t& gZ){
            gX=0;gY=0;gZ=0;
            if(Ngx>=3){ if(i==0) gX=((scalar_t)(-3)*S(0,j,k)+(scalar_t)4*S(1,j,k)-S(2,j,k))*(scalar_t)0.5*inv_h; else if(i==Ngx-1) gX=((scalar_t)3*S(Ngx-1,j,k)-(scalar_t)4*S(Ngx-2,j,k)+S(Ngx-3,j,k))*(scalar_t)0.5*inv_h; else gX=(S(i+1,j,k)-S(i-1,j,k))*(scalar_t)0.5*inv_h; } else if(Ngx==2) gX=(S(1,j,k)-S(0,j,k))*inv_h;
            if(Ngy>=3){ if(j==0) gY=((scalar_t)(-3)*S(i,0,k)+(scalar_t)4*S(i,1,k)-S(i,2,k))*(scalar_t)0.5*inv_h; else if(j==Ngy-1) gY=((scalar_t)3*S(i,Ngy-1,k)-(scalar_t)4*S(i,Ngy-2,k)+S(i,Ngy-3,k))*(scalar_t)0.5*inv_h; else gY=(S(i,j+1,k)-S(i,j-1,k))*(scalar_t)0.5*inv_h; } else if(Ngy==2) gY=(S(i,1,k)-S(i,0,k))*inv_h;
            if(Ngz>=3){ if(k==0) gZ=((scalar_t)(-3)*S(i,j,0)+(scalar_t)4*S(i,j,1)-S(i,j,2))*(scalar_t)0.5*inv_h; else if(k==Ngz-1) gZ=((scalar_t)3*S(i,j,Ngz-1)-(scalar_t)4*S(i,j,Ngz-2)+S(i,j,Ngz-3))*(scalar_t)0.5*inv_h; else gZ=(S(i,j,k+1)-S(i,j,k-1))*(scalar_t)0.5*inv_h; } else if(Ngz==2) gZ=(S(i,j,1)-S(i,j,0))*inv_h;
        };
        auto trigrad=[&](const scalar_t* F,int Mx,int My,int Mz,scalar_t bx0,scalar_t by0,scalar_t bz0,scalar_t idx,scalar_t idy,scalar_t idz,scalar_t xq,scalar_t yq,scalar_t zq,scalar_t& gbx,scalar_t& gby,scalar_t& gbz)->scalar_t{
            scalar_t tx=(xq-bx0)*idx, ty=(yq-by0)*idy, tz=(zq-bz0)*idz;
            tx=std::max((scalar_t)0,std::min(tx,(scalar_t)(Mx-1))); ty=std::max((scalar_t)0,std::min(ty,(scalar_t)(My-1))); tz=std::max((scalar_t)0,std::min(tz,(scalar_t)(Mz-1)));
            int ix=(int)tx; if(ix>Mx-2) ix=Mx-2; int iy=(int)ty; if(iy>My-2) iy=My-2; int iz=(int)tz; if(iz>Mz-2) iz=Mz-2;
            const scalar_t fx=tx-(scalar_t)ix, fy=ty-(scalar_t)iy, fz=tz-(scalar_t)iz;
            const int s2=Mz, s1=My*Mz, base=ix*s1+iy*s2+iz;
            const scalar_t c000=F[base],c001=F[base+1],c010=F[base+s2],c011=F[base+s2+1],c100=F[base+s1],c101=F[base+s1+1],c110=F[base+s1+s2],c111=F[base+s1+s2+1];
            const scalar_t omfx=(scalar_t)1-fx,omfy=(scalar_t)1-fy,omfz=(scalar_t)1-fz;
            const scalar_t val=omfx*(omfy*(omfz*c000+fz*c001)+fy*(omfz*c010+fz*c011))+fx*(omfy*(omfz*c100+fz*c101)+fy*(omfz*c110+fz*c111));
            const scalar_t dfx=omfy*(omfz*(c100-c000)+fz*(c101-c001))+fy*(omfz*(c110-c010)+fz*(c111-c011));
            const scalar_t dfy=omfx*(omfz*(c010-c000)+fz*(c011-c001))+fx*(omfz*(c110-c100)+fz*(c111-c101));
            const scalar_t dfz=omfx*(omfy*(c001-c000)+fy*(c011-c010))+fx*(omfy*(c101-c100)+fy*(c111-c110));
            gbx=dfx*idx; gby=dfy*idy; gbz=dfz*idz; return val;
        };
        auto phi_c=[&](int c,int i,int j,int k)->scalar_t{
            const scalar_t* Fc=F_ptr+F_off[c]; const int Mxc=(int)shapes[c*3],Myc=(int)shapes[c*3+1],Mzc=(int)shapes[c*3+2];
            const scalar_t* Mc=meta+c*10; const scalar_t* Kc=kin_ptr+c*21;
            const scalar_t dxc=gx_ptr[i]-Kc[9],dyc=gy_ptr[j]-Kc[10],dzc=gz_ptr[k]-Kc[11];
            const scalar_t bxc=Kc[0]*dxc+Kc[1]*dyc+Kc[2]*dzc, byc=Kc[3]*dxc+Kc[4]*dyc+Kc[5]*dzc, bzc=Kc[6]*dxc+Kc[7]*dyc+Kc[8]*dzc;
            return sample_dispatch_cpu<scalar_t>((int)interp_method,Fc,Mxc,Myc,Mzc,Mc[0],Mc[1],Mc[2],Mc[6],Mc[7],Mc[8],bxc,byc,bzc); };
        auto covers=[&](int c,int i,int j,int k)->bool{
            const int ci0=(int)lo[c*3],cj0=(int)lo[c*3+1],ck0=(int)lo[c*3+2];
            const int Ci=(int)dim[c*3],Cj=(int)dim[c*3+1],Ck=(int)dim[c*3+2];
            return !(i<ci0||i>=ci0+Ci||j<cj0||j>=cj0+Cj||k<ck0||k>=ck0+Ck); };
        for(int b=0;b<B;++b){
            const int Ai=(int)dim[b*3], Aj=(int)dim[b*3+1], Ak=(int)dim[b*3+2];
            const int vol=Ai*Aj*Ak; if(vol<=0) continue;
            const int i0=(int)lo[b*3], j0=(int)lo[b*3+1], k0=(int)lo[b*3+2];
            const scalar_t* Fb=F_ptr+F_off[b];
            const int Mxb=(int)shapes[b*3], Myb=(int)shapes[b*3+1], Mzb=(int)shapes[b*3+2];
            const scalar_t* Mb=meta+b*10; const scalar_t* Kb=kin_ptr+b*21;
            double acc[12]={0,0,0,0,0,0,0,0,0,0,0,0};
            for(int local=0;local<vol;++local){
                const int di=local/(Aj*Ak), rem=local-di*(Aj*Ak), dj=rem/Ak, dk=rem-dj*Ak;
                const int i=i0+di,j=j0+dj,k=k0+dk; const int64_t g=((int64_t)i*Ngy+j)*Ngz+k;
                const scalar_t xc=gx_ptr[i], yc=gy_ptr[j], zc=gz_ptr[k];
                scalar_t gpx=0,gpy=0,gpz=0,gvx=0,gvy=0,gvz=0,meas_p=0,meas_v=0; bool has_p,has_v;
                if(!body_normal){
                    hgrad(i,j,k,off_pres,gpx,gpy,gpz); hgrad(i,j,k,off_visc,gvx,gvy,gvz);
                    has_p=(gpx!=(scalar_t)0||gpy!=(scalar_t)0||gpz!=(scalar_t)0);
                    has_v=(gvx!=(scalar_t)0||gvy!=(scalar_t)0||gvz!=(scalar_t)0);
                } else {
                    scalar_t gux,guy,guz; sgrad(i,j,k,gux,guy,guz);
                    const scalar_t gmag=std::sqrt(gux*gux+guy*guy+guz*guz); const scalar_t phi_u=sdf[g];
                    const scalar_t ddp=(phi_u-off_pres)*inv_eps, ddv=(phi_u-off_visc)*inv_eps, hie=(scalar_t)0.5*inv_eps;
                    const scalar_t delp=(ddp>(scalar_t)-1&&ddp<(scalar_t)1)?((scalar_t)1+std::cos(pi*ddp))*hie:(scalar_t)0;
                    const scalar_t delv=(ddv>(scalar_t)-1&&ddv<(scalar_t)1)?((scalar_t)1+std::cos(pi*ddv))*hie:(scalar_t)0;
                    meas_p=delp*gmag; meas_v=delv*gmag; has_p=(meas_p!=(scalar_t)0); has_v=(meas_v!=(scalar_t)0);
                }
                if(!(has_p||has_v)) continue;
                const scalar_t dxw=xc-Kb[9],dyw=yc-Kb[10],dzw=zc-Kb[11];
                const scalar_t bxq=Kb[0]*dxw+Kb[1]*dyw+Kb[2]*dzw, byq=Kb[3]*dxw+Kb[4]*dyw+Kb[5]*dzw, bzq=Kb[6]*dxw+Kb[7]*dyw+Kb[8]*dzw;
                scalar_t s_bself;
                if(!body_normal){
                    s_bself=sample_dispatch_cpu<scalar_t>((int)interp_method,Fb,Mxb,Myb,Mzb,Mb[0],Mb[1],Mb[2],Mb[6],Mb[7],Mb[8],bxq,byq,bzq);
                } else {
                    scalar_t gbx,gby,gbz; s_bself=trigrad(Fb,Mxb,Myb,Mzb,Mb[0],Mb[1],Mb[2],Mb[6],Mb[7],Mb[8],bxq,byq,bzq,gbx,gby,gbz);
                    scalar_t nbx=gbx*Kb[0]+gby*Kb[3]+gbz*Kb[6], nby=gbx*Kb[1]+gby*Kb[4]+gbz*Kb[7], nbz=gbx*Kb[2]+gby*Kb[5]+gbz*Kb[8];
                    const scalar_t nlen=std::sqrt(nbx*nbx+nby*nby+nbz*nbz); const scalar_t invn=nlen>(scalar_t)0?(scalar_t)1/nlen:(scalar_t)0;
                    nbx*=invn;nby*=invn;nbz*=invn;
                    gpx=meas_p*nbx;gpy=meas_p*nby;gpz=meas_p*nbz; gvx=meas_v*nbx;gvy=meas_v*nby;gvz=meas_v*nbz;
                }
                scalar_t wb=0;
                if(soft){
                    scalar_t Z=0;
                    for(int c=0;c<B;++c){ if(!covers(c,i,j,k)) continue; const scalar_t s_c=(c==b)?s_bself:phi_c(c,i,j,k); Z+=(scalar_t)1/((scalar_t)1+std::exp(s_c*inv_blend)); }
                    if(Z>(scalar_t)0) wb=((scalar_t)1/((scalar_t)1+std::exp(s_bself*inv_blend)))/Z;
                } else {
                    scalar_t smin=(scalar_t)1e30; int bmin=-1;
                    for(int c=0;c<B;++c){ if(!covers(c,i,j,k)) continue; const scalar_t s_c=(c==b)?s_bself:phi_c(c,i,j,k); if(s_c<smin){smin=s_c;bmin=c;} }
                    wb=(bmin==b)?(scalar_t)1:(scalar_t)0;
                }
                if(wb==(scalar_t)0) continue;
                scalar_t fp_x=0,fp_y=0,fp_z=0;
                if(has_p){ const scalar_t p_c=pp[g]; fp_x=-p_c*gpx;fp_y=-p_c*gpy;fp_z=-p_c*gpz; }
                scalar_t fv_x=0,fv_y=0,fv_z=0;
                if(has_v){
                    const int im1=i>0?i-1:0, ip1=i+1<Ngx?i+1:i, im2=i>1?i-2:0, ip2=i+2<Ngx?i+2:(Ngx-1);
                    const int jm1=j>0?j-1:0, jp1=j+1<Ngy?j+1:j, jm2=j>1?j-2:0, jp2=j+2<Ngy?j+2:(Ngy-1);
                    const int km1=k>0?k-1:0, kp1=k+1<Ngz?k+1:k, km2=k>1?k-2:0, kp2=k+2<Ngz?k+2:(Ngz-1);
                    const scalar_t dudx=(i+1<Ngx)?(up[(ip1*Ngy+j)*Ngz+k]-up[(i*Ngy+j)*Ngz+k])*inv_h:(up[(i*Ngy+j)*Ngz+k]-up[(im1*Ngy+j)*Ngz+k])*inv_h;
                    const scalar_t dvdy=(j+1<Ngy)?(vp[(i*Ngy+jp1)*Ngz+k]-vp[(i*Ngy+j)*Ngz+k])*inv_h:(vp[(i*Ngy+j)*Ngz+k]-vp[(i*Ngy+jm1)*Ngz+k])*inv_h;
                    const scalar_t dwdz=(k+1<Ngz)?(wp[(i*Ngy+j)*Ngz+kp1]-wp[(i*Ngy+j)*Ngz+k])*inv_h:(wp[(i*Ngy+j)*Ngz+k]-wp[(i*Ngy+j)*Ngz+km1])*inv_h;
                    auto d_cc=[&](scalar_t am2,scalar_t am1,scalar_t a0,scalar_t ap1,scalar_t ap2,int idxa,int Na)->scalar_t{ if(Na>=3){ if(idxa==0) return ((scalar_t)(-3)*a0+(scalar_t)4*ap1-ap2)*(scalar_t)0.5*inv_h; if(idxa==Na-1) return ((scalar_t)3*a0-(scalar_t)4*am1+am2)*(scalar_t)0.5*inv_h; } return (ap1-am1)*(scalar_t)0.5*inv_h; };
                    auto u_cc=[&](int jj,int kk)->scalar_t{ return (scalar_t)0.5*(up[(i*Ngy+jj)*Ngz+kk]+up[(ip1*Ngy+jj)*Ngz+kk]); };
                    auto v_cc=[&](int ii,int kk)->scalar_t{ return (scalar_t)0.5*(vp[(ii*Ngy+j)*Ngz+kk]+vp[(ii*Ngy+jp1)*Ngz+kk]); };
                    auto w_cc=[&](int ii,int jj)->scalar_t{ return (scalar_t)0.5*(wp[(ii*Ngy+jj)*Ngz+k]+wp[(ii*Ngy+jj)*Ngz+kp1]); };
                    const scalar_t dudy=d_cc(u_cc(jm2,k),u_cc(jm1,k),u_cc(j,k),u_cc(jp1,k),u_cc(jp2,k),j,Ngy);
                    const scalar_t dudz=d_cc(u_cc(j,km2),u_cc(j,km1),u_cc(j,k),u_cc(j,kp1),u_cc(j,kp2),k,Ngz);
                    const scalar_t dvdx=d_cc(v_cc(im2,k),v_cc(im1,k),v_cc(i,k),v_cc(ip1,k),v_cc(ip2,k),i,Ngx);
                    const scalar_t dvdz=d_cc(v_cc(i,km2),v_cc(i,km1),v_cc(i,k),v_cc(i,kp1),v_cc(i,kp2),k,Ngz);
                    const scalar_t dwdx=d_cc(w_cc(im2,j),w_cc(im1,j),w_cc(i,j),w_cc(ip1,j),w_cc(ip2,j),i,Ngx);
                    const scalar_t dwdy=d_cc(w_cc(i,jm2),w_cc(i,jm1),w_cc(i,j),w_cc(i,jp1),w_cc(i,jp2),j,Ngy);
                    const scalar_t nrv=nr_size==1?nr[0]:nr[g];
                    const scalar_t sxx=nrv*(scalar_t)2*dudx, syy=nrv*(scalar_t)2*dvdy, szz=nrv*(scalar_t)2*dwdz;
                    const scalar_t sxy=nrv*(dudy+dvdx), sxz=nrv*(dudz+dwdx), syz=nrv*(dvdz+dwdy);
                    fv_x=sxx*gvx+sxy*gvy+sxz*gvz; fv_y=sxy*gvx+syy*gvy+syz*gvz; fv_z=sxz*gvx+syz*gvy+szz*gvz;
                }
                const scalar_t ax=xc-Kb[12],ay=yc-Kb[13],az=zc-Kb[14];
                const scalar_t fvx=wb*fv_x,fvy=wb*fv_y,fvz=wb*fv_z, fpx=wb*fp_x,fpy=wb*fp_y,fpz=wb*fp_z;
                acc[0]+=fvx;acc[1]+=fvy;acc[2]+=fvz; acc[3]+=ay*fvz-az*fvy;acc[4]+=az*fvx-ax*fvz;acc[5]+=ax*fvy-ay*fvx;
                acc[6]+=fpx;acc[7]+=fpy;acc[8]+=fpz; acc[9]+=ay*fpz-az*fpy;acc[10]+=az*fpx-ax*fpz;acc[11]+=ax*fpy-ay*fpx;
            }
            for(int c=0;c<12;++c) outp[b*12+c]+=acc[c]*h3d;
        }
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("streaming_sdf_forces_post_3d",   &streaming_sdf_forces_post_3d_cpu);
    m.impl("apply_bcs_3d",                   &apply_bcs_3d_cpu);
    m.impl("interp_3d",                 &interp_3d_cpu);
}

}  // namespace lilytorch_kernels
