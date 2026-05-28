// =====================================================================
//  lagrangian_forces_cpu_2d.cpp
//
//  CPU implementation of ``lagrangian_forces_2d``: fused per-body
//  surface integration of
//
//      F = ∮_∂Ω (ν·ρ·ε·n - p n) dS
//      τ = ∮_∂Ω (r - x_com) × (ν·ρ·ε·n - p n) dS
//
//  Mirrors the reference PyTorch ``forces.forces_lagrangian_2d``:
//    * Bilinear / biquadratic sampling of ε_xx, ε_xy, ε_yy and p at
//      every contour marker (interp_method = 0 / 1).
//    * Outward unit normal computed from central differences on the
//      (CCW-closed) marker list: n = (t_y, -t_x).
//    * Midpoint-rule line integration: each segment ``i → i+1``
//      contributes ``0.5*(f_i + f_{i+1}) * ||cnt[i+1] - cnt[i]||``.
//    * Per-body reduction → ``out[b, 0..5] = [fv_x, fv_y, t_v,
//      fp_x, fp_y, t_p]``.
//
//  This op is callable from CPU only.  The CUDA implementation lives
//  in ``cuda/lagrangian_forces_2d.cu``.
// =====================================================================

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>

namespace lilytorch_kernels {

// Bilinear sampler on a UNIFORM 2-D grid laid out as (Mx, My)
// row-major.  Mirrors the helper in streaming_sdf_cpu_2d.cpp but is
// kept self-contained so this TU does not depend on it.
template <typename scalar_t>
static inline scalar_t lf_bilinear_2d(
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

// Biquadratic sampler on the same uniform grid, identical to
// ``biquadratic_sample_uniform_2d`` in streaming_sdf_cpu_2d.cpp.
template <typename scalar_t>
static inline scalar_t lf_biquadratic_2d(
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

    if (ix < 1 || iy < 1 || Mx < 3 || My < 3) {
        return lf_bilinear_2d<scalar_t>(
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

    const int s1 = My;
    const int base = (ix - 1) * s1 + (iy - 1);

    scalar_t out = (scalar_t)0;
    for (int dx = 0; dx < 3; ++dx) {
        const scalar_t wx = (dx == 0) ? wxm : (dx == 1 ? wx0 : wxp);
        const int b0 = base + dx * s1;
        const scalar_t row = wym * F[b0] + wy0 * F[b0 + 1] + wyp * F[b0 + 2];
        out += wx * row;
    }
    return out;
}

template <typename scalar_t>
static inline scalar_t lf_sample_2d(
    const int method,
    const scalar_t* F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    if (method == 1) {
        return lf_biquadratic_2d<scalar_t>(
            F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
    }
    return lf_bilinear_2d<scalar_t>(
        F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
}

// =====================================================================
void lagrangian_forces_2d_cpu(
    const at::Tensor& eps_xx, const at::Tensor& eps_xy, const at::Tensor& eps_yy,
    const at::Tensor& p, const at::Tensor& nu_rho_field,
    const at::Tensor& cnt_flat, const at::Tensor& cnt_offsets,
    const at::Tensor& com_pos,
    const double bx0, const double by0,
    const double inv_dx, const double inv_dy,
    const int64_t Mx, const int64_t My,
    const int64_t interp_method,
    at::Tensor out)
{
    const int B = (int)com_pos.size(0);
    TORCH_CHECK(out.dim() == 2 && out.size(0) == B && out.size(1) == 6,
                "lagrangian_forces_2d_cpu: out must be (B, 6); got ",
                out.sizes());
    TORCH_CHECK(out.scalar_type() == at::kDouble,
                "lagrangian_forces_2d_cpu: out must be float64");
    TORCH_CHECK(cnt_offsets.numel() == B + 1,
                "lagrangian_forces_2d_cpu: cnt_offsets must have B+1 entries");

    out.zero_();
    if (B <= 0) return;

    // Ensure contiguous, then pin storage to keep raw pointers alive
    // for the duration of the AT_DISPATCH lambda.
    auto eps_xx_c = eps_xx.contiguous();
    auto eps_xy_c = eps_xy.contiguous();
    auto eps_yy_c = eps_yy.contiguous();
    auto p_c      = p.contiguous();
    auto nrho_c   = nu_rho_field.contiguous();
    auto cnt_c    = cnt_flat.contiguous();
    auto offs_c   = cnt_offsets.contiguous().to(at::kLong);
    auto com_c    = com_pos.contiguous();

    const bool nrho_scalar = (nrho_c.numel() == 1);

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "lagrangian_forces_2d_cpu", [&] {
        const scalar_t* exx  = eps_xx_c.data_ptr<scalar_t>();
        const scalar_t* exy  = eps_xy_c.data_ptr<scalar_t>();
        const scalar_t* eyy  = eps_yy_c.data_ptr<scalar_t>();
        const scalar_t* pp   = p_c.data_ptr<scalar_t>();
        const scalar_t* nrho = nrho_c.data_ptr<scalar_t>();
        const int64_t* offs  = offs_c.data_ptr<int64_t>();
        double* outp         = out.data_ptr<double>();

        // cnt_flat is (2, M_total) row-major; row 0 = x, row 1 = y.
        const int64_t M_total = cnt_c.size(1);
        const scalar_t* cnt_x = cnt_c.data_ptr<scalar_t>();
        const scalar_t* cnt_y = cnt_x + M_total;

        // com_pos is (B, 2) row-major.
        const scalar_t* com = com_c.data_ptr<scalar_t>();

        const int iMx = (int)Mx, iMy = (int)My;
        const scalar_t bx0s = (scalar_t)bx0, by0s = (scalar_t)by0;
        const scalar_t idx  = (scalar_t)inv_dx, idy = (scalar_t)inv_dy;
        const int method = (int)interp_method;

        at::parallel_for(0, B, 0, [&](int64_t b_start, int64_t b_end) {
            for (int64_t b = b_start; b < b_end; ++b) {
                const int64_t i0 = offs[b];
                const int64_t i1 = offs[b + 1];
                const int64_t M  = i1 - i0;
                if (M <= 1) continue;

                const scalar_t com_x = com[b * 2 + 0];
                const scalar_t com_y = com[b * 2 + 1];

                double acc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

                // Sample at every marker once; cache per-marker tractions
                // so the segment loop can sum trapezoidal contributions
                // (f_i + f_{i+1})/2 * ds_i without resampling.
                // For the rigid closed contour the wrap segment is
                // i = M-1 → 0.
                //
                // Use a single stack-allocated scratch via std::vector
                // since M is unknown.  Quadrature is exact in fp64.
                std::vector<scalar_t> tvx(M), tvy(M), tpx(M), tpy(M);

                for (int64_t k = 0; k < M; ++k) {
                    const int64_t g = i0 + k;
                    const scalar_t qx = cnt_x[g];
                    const scalar_t qy = cnt_y[g];

                    // Tangent via central diff on the closed contour
                    const int64_t km = (k == 0)     ? (M - 1) : (k - 1);
                    const int64_t kp = (k == M - 1) ? 0       : (k + 1);
                    scalar_t tx = (cnt_x[i0 + kp] - cnt_x[i0 + km]) * (scalar_t)0.5;
                    scalar_t ty = (cnt_y[i0 + kp] - cnt_y[i0 + km]) * (scalar_t)0.5;
                    scalar_t L = std::sqrt(tx * tx + ty * ty);
                    if (L < (scalar_t)1e-30) L = (scalar_t)1e-30;
                    tx /= L; ty /= L;
                    // CCW outward normal = (+t_y, -t_x)
                    const scalar_t nx =  ty;
                    const scalar_t ny = -tx;

                    const scalar_t e_xx = lf_sample_2d<scalar_t>(
                        method, exx, iMx, iMy, bx0s, by0s, idx, idy, qx, qy);
                    const scalar_t e_xy = lf_sample_2d<scalar_t>(
                        method, exy, iMx, iMy, bx0s, by0s, idx, idy, qx, qy);
                    const scalar_t e_yy = lf_sample_2d<scalar_t>(
                        method, eyy, iMx, iMy, bx0s, by0s, idx, idy, qx, qy);

                    const scalar_t nu_rho_m = nrho_scalar
                        ? nrho[0]
                        : lf_sample_2d<scalar_t>(
                            method, nrho, iMx, iMy, bx0s, by0s, idx, idy, qx, qy);

                    tvx[k] = nu_rho_m * (e_xx * nx + e_xy * ny);
                    tvy[k] = nu_rho_m * (e_xy * nx + e_yy * ny);

                    const scalar_t p_m = lf_sample_2d<scalar_t>(
                        method, pp, iMx, iMy, bx0s, by0s, idx, idy, qx, qy);
                    tpx[k] = -p_m * nx;
                    tpy[k] = -p_m * ny;
                }

                // Trapezoidal line integral.  Each segment i → (i+1)%M
                // contributes ds * 0.5 * (f_i + f_{i+1}) to the total.
                for (int64_t k = 0; k < M; ++k) {
                    const int64_t kp = (k == M - 1) ? 0 : (k + 1);
                    const scalar_t xk  = cnt_x[i0 + k];
                    const scalar_t yk  = cnt_y[i0 + k];
                    const scalar_t xkp = cnt_x[i0 + kp];
                    const scalar_t ykp = cnt_y[i0 + kp];
                    const scalar_t dx = xkp - xk;
                    const scalar_t dy = ykp - yk;
                    const scalar_t ds = std::sqrt(dx * dx + dy * dy);
                    const scalar_t w  = (scalar_t)0.5 * ds;

                    // Per-marker arms (r - com)
                    const scalar_t rx  = xk  - com_x;
                    const scalar_t ry  = yk  - com_y;
                    const scalar_t rxp = xkp - com_x;
                    const scalar_t ryp = ykp - com_y;

                    acc[0] += (double)(w * (tvx[k] + tvx[kp]));
                    acc[1] += (double)(w * (tvy[k] + tvy[kp]));
                    acc[2] += (double)(w * (
                        (rx * tvy[k]  - ry * tvx[k]) +
                        (rxp * tvy[kp] - ryp * tvx[kp])));
                    acc[3] += (double)(w * (tpx[k] + tpx[kp]));
                    acc[4] += (double)(w * (tpy[k] + tpy[kp]));
                    acc[5] += (double)(w * (
                        (rx * tpy[k]  - ry * tpx[k]) +
                        (rxp * tpy[kp] - ryp * tpx[kp])));
                }

                outp[b * 6 + 0] = acc[0];
                outp[b * 6 + 1] = acc[1];
                outp[b * 6 + 2] = acc[2];
                outp[b * 6 + 3] = acc[3];
                outp[b * 6 + 4] = acc[4];
                outp[b * 6 + 5] = acc[5];
            }
        });
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("lagrangian_forces_2d", &lagrangian_forces_2d_cpu);
}

}  // namespace lilytorch_kernels
