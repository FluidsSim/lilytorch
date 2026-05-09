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
//  The streaming_sdf_min_rho_3d_multi* kernels are exclusively used with body SDF
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
    if (tx < (scalar_t)0) tx = (scalar_t)0; else if (tx > Mx_lim) tx = Mx_lim;
    if (ty < (scalar_t)0) ty = (scalar_t)0; else if (ty > My_lim) ty = My_lim;
    if (tz < (scalar_t)0) tz = (scalar_t)0; else if (tz > Mz_lim) tz = Mz_lim;

    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;
    int iz = (int)tz; if (iz > Mz - 2) iz = Mz - 2;

    // Boundary fallback to trilinear on any axis whose lower stencil
    // neighbour (ix-1) is out of range.  The body grid is at least 2 wide
    // on each axis (otherwise the trilinear sampler would be invalid as
    // well), but ix-1 < 0 for queries in the first cell.
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
    // Each F lookup = base + dx*s1 + dy*s2 + dz with dx,dy,dz in {0,1,2}.
    // Factor by x: 3 inner-y reductions; then sum.
    auto plane = [&](int dx_off) -> scalar_t {
        const int b0 = base + dx_off * s1;
        // y = -1
        const scalar_t r_m =
            wzm * F[b0 + 0]      + wz0 * F[b0 + 1]      + wzp * F[b0 + 2];
        // y =  0
        const scalar_t r_0 =
            wzm * F[b0 + s2]     + wz0 * F[b0 + s2 + 1] + wzp * F[b0 + s2 + 2];
        // y = +1
        const scalar_t r_p =
            wzm * F[b0 + 2*s2]   + wz0 * F[b0 + 2*s2 + 1] + wzp * F[b0 + 2*s2 + 2];
        return wym * r_m + wy0 * r_0 + wyp * r_p;
    };
    return wxm * plane(0) + wx0 * plane(1) + wxp * plane(2);
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
//  streaming_sdf_min_rho_3d_multi
//
//  Bodies are processed serially (matches CUDA -- no atomics required
//  because each cell is touched once per body, and bodies progress in
//  order); cells within a body are parallelised.
// =====================================================================


// =====================================================================
//  apply_bcs_3d
//
//  Each op writes a single 2D plane of u / v / w. Ops are independent
//  (caller arranges them so the addressed planes don't overlap), so we
//  loop ops serially and parallelise the plane fill.
// =====================================================================

template <typename scalar_t>
static void apply_bcs_3d_one_plane(
    scalar_t* base,
    const int Ny, const int Nz,
    const int axis,
    const int dst_along, const int src_along,
    const bool is_neu, const scalar_t value,
    const int dim0_max, const int dim1_max)
{
    const std::int64_t s1 = (std::int64_t)Ny * Nz;
    const std::int64_t s2 = (std::int64_t)Nz;

    const int64_t total = (int64_t)dim0_max * (int64_t)dim1_max;
    at::parallel_for(0, total, /*grain_size=*/1024, [&](int64_t _begin, int64_t _end) {
    for (int64_t t = _begin; t < _end; ++t) {
        const int i = (int)(t / dim1_max);
        const int j = (int)(t % dim1_max);
        std::int64_t dst_lin, src_lin = 0;
        if (axis == 0) {
            dst_lin = (std::int64_t)dst_along * s1 + (std::int64_t)i * s2 + j;
            if (is_neu) src_lin = (std::int64_t)src_along * s1 + (std::int64_t)i * s2 + j;
        } else if (axis == 1) {
            dst_lin = (std::int64_t)i * s1 + (std::int64_t)dst_along * s2 + j;
            if (is_neu) src_lin = (std::int64_t)i * s1 + (std::int64_t)src_along * s2 + j;
        } else {
            dst_lin = (std::int64_t)i * s1 + (std::int64_t)j * s2 + dst_along;
            if (is_neu) src_lin = (std::int64_t)i * s1 + (std::int64_t)j * s2 + src_along;
        }
        base[dst_lin] = is_neu ? base[src_lin] : value;
    }
    });
}

void apply_bcs_3d_cpu(
    at::Tensor u, at::Tensor v, at::Tensor w,
    const at::Tensor& shapes,
    const at::Tensor& neu_desc,
    const at::Tensor& dir_desc,
    const at::Tensor& dir_val,
    const int64_t /*max_plane_dim*/)
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

    const int N_neu = (int)neu_desc.size(0);
    const int N_dir = (int)dir_desc.size(0);
    if (N_neu + N_dir == 0) return;

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_3d_cpu", [&] {
        const int64_t*  shapes_p  = shapes.data_ptr<int64_t>();
        const int*      neu_p     = (N_neu > 0) ? neu_desc.data_ptr<int>() : nullptr;
        const int*      dir_p     = (N_dir > 0) ? dir_desc.data_ptr<int>() : nullptr;
        const scalar_t* dir_val_p = (N_dir > 0) ? dir_val.data_ptr<scalar_t>() : nullptr;

        scalar_t* u_p = u.data_ptr<scalar_t>();
        scalar_t* v_p = v.data_ptr<scalar_t>();
        scalar_t* w_p = w.data_ptr<scalar_t>();

        const int total = N_neu + N_dir;
        for (int op = 0; op < total; ++op) {
            const bool is_neu = (op < N_neu);
            int comp, axis, dst_along, src_along;
            scalar_t value = scalar_t(0);

            if (is_neu) {
                comp = neu_p[op*3 + 0];
                axis = neu_p[op*3 + 1];
                const int side = neu_p[op*3 + 2];
                const int sz = (int)shapes_p[comp*3 + axis];
                if (side == 0) { dst_along = 0;      src_along = 1; }
                else           { dst_along = sz - 1; src_along = sz - 2; }
            } else {
                const int d = op - N_neu;
                comp = dir_p[d*3 + 0];
                axis = dir_p[d*3 + 1];
                const int offset = dir_p[d*3 + 2];
                const int sz = (int)shapes_p[comp*3 + axis];
                dst_along = (offset >= 0) ? offset : (sz + offset);
                src_along = 0;
                value = dir_val_p[d];
            }

            const int Nx = (int)shapes_p[comp*3 + 0];
            const int Ny = (int)shapes_p[comp*3 + 1];
            const int Nz = (int)shapes_p[comp*3 + 2];

            int dim0_max, dim1_max;
            if      (axis == 0) { dim0_max = Ny; dim1_max = Nz; }
            else if (axis == 1) { dim0_max = Nx; dim1_max = Nz; }
            else                { dim0_max = Nx; dim1_max = Ny; }

            scalar_t* base = (comp == 0) ? u_p : (comp == 1 ? v_p : w_p);

            apply_bcs_3d_one_plane<scalar_t>(
                base, Ny, Nz, axis,
                dst_along, src_along, is_neu, value,
                dim0_max, dim1_max);
        }
    });
}

// =====================================================================
//  interpolate_3d_cpu: scattered-point trilinear / triquadratic sampling
// =====================================================================
static void interpolate_3d_cpu(
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

    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "interpolate_3d_cpu", [&] {
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
//      bV / bW) and ``winning_rho_cc`` in place.
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

void streaming_sdf_min_rho_3d_multi_cpu(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes,
    const at::Tensor& body_meta,
    const at::Tensor& kin,
    const at::Tensor& aabb_lo,
    const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy, const at::Tensor& gz,
    const double h_grid,
    const int64_t /*max_vol_per_body*/,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v, at::Tensor sdf_w,
    at::Tensor body_u, at::Tensor body_v, at::Tensor body_w,
    const int64_t interp_method,
    const at::Tensor& rho_bodies,
    at::Tensor winning_rho_cc)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0) return;
    const int Ngy = (int)gy.numel();
    const int Ngz = (int)gz.numel();

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_min_rho_3d_multi_cpu", [&] {
        const scalar_t* F_ptr = F_flat.data_ptr<scalar_t>();
        const int64_t* F_off = F_offsets.data_ptr<int64_t>();
        const int64_t* shapes = body_shapes.data_ptr<int64_t>();
        const scalar_t* meta = body_meta.data_ptr<scalar_t>();
        const scalar_t* kin_ptr = kin.data_ptr<scalar_t>();
        const int64_t* lo = aabb_lo.data_ptr<int64_t>();
        const int64_t* dim = aabb_dim.data_ptr<int64_t>();
        const scalar_t* gx_ptr = gx.data_ptr<scalar_t>();
        const scalar_t* gy_ptr = gy.data_ptr<scalar_t>();
        const scalar_t* gz_ptr = gz.data_ptr<scalar_t>();
        const scalar_t* rho_ptr = rho_bodies.data_ptr<scalar_t>();
        scalar_t* sdf_cc_p = sdf_cc.data_ptr<scalar_t>();
        scalar_t* sdf_u_p = sdf_u.data_ptr<scalar_t>();
        scalar_t* sdf_v_p = sdf_v.data_ptr<scalar_t>();
        scalar_t* sdf_w_p = sdf_w.data_ptr<scalar_t>();
        scalar_t* bU_p = body_u.data_ptr<scalar_t>();
        scalar_t* bV_p = body_v.data_ptr<scalar_t>();
        scalar_t* bW_p = body_w.data_ptr<scalar_t>();
        scalar_t* wrho_p = winning_rho_cc.data_ptr<scalar_t>();
        const scalar_t half_h = (scalar_t)(0.5 * h_grid);
        const int interp = (int)interp_method;

        for (int b = 0; b < B; ++b) {
            const int Ai = (int)dim[b*3+0], Aj = (int)dim[b*3+1], Ak = (int)dim[b*3+2];
            const int vol = Ai * Aj * Ak;
            if (vol <= 0) continue;
            const int i0 = (int)lo[b*3+0], j0 = (int)lo[b*3+1], k0 = (int)lo[b*3+2];
            const scalar_t* F_b = F_ptr + F_off[b];
            const int Mx = (int)shapes[b*3+0], My = (int)shapes[b*3+1], Mz = (int)shapes[b*3+2];
            const scalar_t* M = meta + b*10;
            const scalar_t bx0=M[0], by0=M[1], bz0=M[2], idx=M[6], idy=M[7], idz=M[8];
            const scalar_t* K = kin_ptr + b*21;
            const scalar_t r00=K[0], r01=K[1], r02=K[2], r10=K[3], r11=K[4], r12=K[5], r20=K[6], r21=K[7], r22=K[8];
            const scalar_t bp_x=K[9], bp_y=K[10], bp_z=K[11], cm_x=K[12], cm_y=K[13], cm_z=K[14];
            const scalar_t lv_x=K[15], lv_y=K[16], lv_z=K[17], av_x=K[18], av_y=K[19], av_z=K[20];
            const scalar_t neg_hh = -half_h;
            const scalar_t du_x=neg_hh*r00, du_y=neg_hh*r10, du_z=neg_hh*r20;
            const scalar_t dv_x=neg_hh*r01, dv_y=neg_hh*r11, dv_z=neg_hh*r21;
            const scalar_t dw_x=neg_hh*r02, dw_y=neg_hh*r12, dw_z=neg_hh*r22;
            const scalar_t rho_b = rho_ptr[b];
            // Cells of a single body's AABB are disjoint from one another in
            // ``g`` index space, so concurrent compare-swaps into ``sdf_*[g]``
            // are race-free within one body.  Bodies are still serialised by
            // the outer ``for (b)`` loop, matching the 2-D path.
            at::parallel_for(0, vol, /*grain_size=*/1024, [&](int64_t _begin, int64_t _end) {
            for (int local = (int)_begin; local < (int)_end; ++local) {
                const int di = local / (Aj*Ak);
                const int rem = local - di*(Aj*Ak);
                const int dj = rem / Ak;
                const int dk = rem - dj*Ak;
                const int i = i0+di, j = j0+dj, k = k0+dk;
                const int64_t g = ((int64_t)i * Ngy + j) * Ngz + k;
                const scalar_t xc=gx_ptr[i], yc=gy_ptr[j], zc=gz_ptr[k];
                const scalar_t dxw=xc-bp_x, dyw=yc-bp_y, bzw=zc-bp_z;
                const scalar_t bxq=r00*dxw+r01*dyw+r02*bzw;
                const scalar_t byq=r10*dxw+r11*dyw+r12*bzw;
                const scalar_t bzq=r20*dxw+r21*dyw+r22*bzw;
                const scalar_t scc = sample_dispatch_cpu<scalar_t>(interp,F_b,Mx,My,Mz,bx0,by0,bz0,idx,idy,idz,bxq,byq,bzq);
                if (scc < sdf_cc_p[g]) { sdf_cc_p[g]=scc; wrho_p[g]=rho_b; }
                const scalar_t su = sample_dispatch_cpu<scalar_t>(interp,F_b,Mx,My,Mz,bx0,by0,bz0,idx,idy,idz,bxq+du_x,byq+du_y,bzq+du_z);
                if (su < sdf_u_p[g]) { sdf_u_p[g]=su; bU_p[g]=lv_x + av_y*(zc-cm_z) - av_z*(yc-cm_y); }
                const scalar_t sv = sample_dispatch_cpu<scalar_t>(interp,F_b,Mx,My,Mz,bx0,by0,bz0,idx,idy,idz,bxq+dv_x,byq+dv_y,bzq+dv_z);
                if (sv < sdf_v_p[g]) { sdf_v_p[g]=sv; bV_p[g]=lv_y + av_z*(xc-cm_x) - av_x*(zc-cm_z); }
                const scalar_t sw = sample_dispatch_cpu<scalar_t>(interp,F_b,Mx,My,Mz,bx0,by0,bz0,idx,idy,idz,bxq+dw_x,byq+dw_y,bzq+dw_z);
                if (sw < sdf_w_p[g]) { sdf_w_p[g]=sw; bW_p[g]=lv_z + av_x*(yc-cm_y) - av_y*(xc-cm_x); }
            }
            });
        }
    });
}

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
                const int im1=i>0?i-1:0, ip1=i+1<Ngx?i+1:i, jm1=j>0?j-1:0, jp1=j+1<Ngy?j+1:j, km1=k>0?k-1:0, kp1=k+1<Ngz?k+1:k;
                const scalar_t dudx=(up[(ip1*Ngy+j)*Ngz+k]-up[(i*Ngy+j)*Ngz+k])*inv_h;
                const scalar_t dvdy=(vp[(i*Ngy+jp1)*Ngz+k]-vp[(i*Ngy+j)*Ngz+k])*inv_h;
                const scalar_t dwdz=(wp[(i*Ngy+j)*Ngz+kp1]-wp[(i*Ngy+j)*Ngz+k])*inv_h;
                const scalar_t dudy=((up[(i*Ngy+jp1)*Ngz+k]+up[(ip1*Ngy+jp1)*Ngz+k])-(up[(i*Ngy+jm1)*Ngz+k]+up[(ip1*Ngy+jm1)*Ngz+k]))*(scalar_t)0.25*inv_h;
                const scalar_t dudz=((up[(i*Ngy+j)*Ngz+kp1]+up[(ip1*Ngy+j)*Ngz+kp1])-(up[(i*Ngy+j)*Ngz+km1]+up[(ip1*Ngy+j)*Ngz+km1]))*(scalar_t)0.25*inv_h;
                const scalar_t dvdx=((vp[(ip1*Ngy+j)*Ngz+k]+vp[(ip1*Ngy+jp1)*Ngz+k])-(vp[(im1*Ngy+j)*Ngz+k]+vp[(im1*Ngy+jp1)*Ngz+k]))*(scalar_t)0.25*inv_h;
                const scalar_t dvdz=((vp[(i*Ngy+j)*Ngz+kp1]+vp[(i*Ngy+jp1)*Ngz+kp1])-(vp[(i*Ngy+j)*Ngz+km1]+vp[(i*Ngy+jp1)*Ngz+km1]))*(scalar_t)0.25*inv_h;
                const scalar_t dwdx=((wp[(ip1*Ngy+j)*Ngz+k]+wp[(ip1*Ngy+j)*Ngz+kp1])-(wp[(im1*Ngy+j)*Ngz+k]+wp[(im1*Ngy+j)*Ngz+kp1]))*(scalar_t)0.25*inv_h;
                const scalar_t dwdy=((wp[(i*Ngy+jp1)*Ngz+k]+wp[(i*Ngy+jp1)*Ngz+kp1])-(wp[(i*Ngy+jm1)*Ngz+k]+wp[(i*Ngy+jm1)*Ngz+kp1]))*(scalar_t)0.25*inv_h;
                const scalar_t nrv=nr_size==1?nr[0]:nr[g];
                const scalar_t xs=nrv*(2*dudx*nx+(dudy+dvdx)*ny+(dudz+dwdx)*nz), ys=nrv*((dvdx+dudy)*nx+2*dvdy*ny+(dvdz+dwdy)*nz), zs=nrv*((dwdx+dudz)*nx+(dwdy+dvdz)*ny+2*dwdz*nz);
                scalar_t dv=0,dp=0; const scalar_t sd=sbody-eps_s; if(sd>-eps_b&&sd<eps_b) dv=(1+std::cos(pi_eb*sd))*inv_2eps; if(sbody>-eps_b&&sbody<eps_b) dp=(1+std::cos(pi_eb*sbody))*inv_2eps;
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
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("streaming_sdf_min_rho_3d_multi", &streaming_sdf_min_rho_3d_multi_cpu);
    m.impl("streaming_sdf_forces_post_3d",   &streaming_sdf_forces_post_3d_cpu);
    m.impl("apply_bcs_3d",                   &apply_bcs_3d_cpu);
    m.impl("interpolate_3d",                 &interpolate_3d_cpu);
}

}  // namespace lilytorch_kernels
