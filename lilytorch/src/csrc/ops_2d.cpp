// =====================================================================
//  ops_2d.cpp — 2-D CPU implementations (at::parallel_for).
//
//  Contains forces_post_2d, apply_bcs_2d, interp_2d CPU twins.
//  See ops_3d.cpp for the 3-D counterparts.
// =====================================================================
//
//  Hot loops are parallelised with ``at::parallel_for`` (PyTorch's
//  intra-op thread pool); see the rationale comment at the top of
//  ``streaming_sdf_cpu.cpp``.
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
    m.impl("streaming_sdf_forces_post_2d",          &streaming_sdf_forces_post_2d_cpu);
    m.impl("apply_bcs_2d",                          &apply_bcs_2d_cpu);
    m.impl("interp_2d",                        &interp_2d_cpu);
}

}  // namespace lilytorch_kernels
