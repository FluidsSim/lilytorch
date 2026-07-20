// =====================================================================
//  lagrangian_forces_cpu.cpp
//
//  CPU implementation of ``lagrangian_forces_3d``: fused per-body
//  surface integration over a precomputed triangulation:
//
//      F = Σ_T (ν·ρ·ε·n - p n) * A_T
//      τ = Σ_T (c_T - x_com) × (ν·ρ·ε·n - p n) * A_T
//
//  Mirrors the reference PyTorch ``forces.forces_lagrangian_3d``.
//
//  CUDA implementation lives in ``cuda/lagrangian_forces.cu``.
// =====================================================================

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace lilytorch_kernels {

template <typename scalar_t>
static inline scalar_t lf_trilinear_3d(
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

    const int s2 = Mz;
    const int s1 = My * Mz;
    const int base = ix * s1 + iy * s2 + iz;

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

template <typename scalar_t>
static inline scalar_t lf_triquadratic_3d(
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

    if (ix < 1 || iy < 1 || iz < 1 ||
        Mx < 3 || My < 3 || Mz < 3) {
        return lf_trilinear_3d<scalar_t>(
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

    scalar_t out = (scalar_t)0;
    for (int dx = 0; dx < 3; ++dx) {
        const scalar_t wx = (dx == 0) ? wxm : (dx == 1 ? wx0 : wxp);
        const int b0 = base + dx * s1;
        scalar_t plane = (scalar_t)0;
        for (int dy = 0; dy < 3; ++dy) {
            const scalar_t wy = (dy == 0) ? wym : (dy == 1 ? wy0 : wyp);
            const int b1 = b0 + dy * s2;
            const scalar_t row =
                wzm * F[b1] + wz0 * F[b1 + 1] + wzp * F[b1 + 2];
            plane += wy * row;
        }
        out += wx * plane;
    }
    return out;
}

template <typename scalar_t>
static inline scalar_t lf_sample_3d(
    const int method,
    const scalar_t* F,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    if (method == 1) {
        return lf_triquadratic_3d<scalar_t>(
            F, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, xq, yq, zq);
    }
    return lf_trilinear_3d<scalar_t>(
        F, Mx, My, Mz, bx0, by0, bz0,
        inv_dx, inv_dy, inv_dz, xq, yq, zq);
}

// =====================================================================
void lagrangian_forces_3d_cpu(
    const at::Tensor& eps_xx, const at::Tensor& eps_yy, const at::Tensor& eps_zz,
    const at::Tensor& eps_xy, const at::Tensor& eps_xz, const at::Tensor& eps_yz,
    const at::Tensor& p, const at::Tensor& nu_rho_field,
    const at::Tensor& tri_centroid, const at::Tensor& tri_normal,
    const at::Tensor& tri_area,
    const at::Tensor& tri_offsets, const at::Tensor& com_pos,
    const double bx0, const double by0, const double bz0,
    const double inv_dx, const double inv_dy, const double inv_dz,
    const int64_t Mx, const int64_t My, const int64_t Mz,
    const int64_t interp_method,
    const double sample_offset_pressure,
    const double sample_offset_friction,
    at::Tensor out)
{
    const int B = (int)com_pos.size(0);
    TORCH_CHECK(out.dim() == 2 && out.size(0) == B && out.size(1) == 12,
                "lagrangian_forces_3d_cpu: out must be (B, 12); got ",
                out.sizes());
    TORCH_CHECK(out.scalar_type() == at::kDouble,
                "lagrangian_forces_3d_cpu: out must be float64");
    TORCH_CHECK(tri_offsets.numel() == B + 1,
                "lagrangian_forces_3d_cpu: tri_offsets must have B+1 entries");

    out.zero_();
    if (B <= 0) return;

    auto exx_c  = eps_xx.contiguous();
    auto eyy_c  = eps_yy.contiguous();
    auto ezz_c  = eps_zz.contiguous();
    auto exy_c  = eps_xy.contiguous();
    auto exz_c  = eps_xz.contiguous();
    auto eyz_c  = eps_yz.contiguous();
    auto p_c    = p.contiguous();
    auto nrho_c = nu_rho_field.contiguous();
    auto ct_c   = tri_centroid.contiguous();   // (3, T_total)
    auto nm_c   = tri_normal.contiguous();     // (3, T_total)
    auto ar_c   = tri_area.contiguous();       // (T_total,)
    auto offs_c = tri_offsets.contiguous().to(at::kLong);
    auto com_c  = com_pos.contiguous();

    const bool nrho_scalar = (nrho_c.numel() == 1);

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "lagrangian_forces_3d_cpu", [&] {
        const scalar_t* exx  = exx_c.data_ptr<scalar_t>();
        const scalar_t* eyy  = eyy_c.data_ptr<scalar_t>();
        const scalar_t* ezz  = ezz_c.data_ptr<scalar_t>();
        const scalar_t* exy  = exy_c.data_ptr<scalar_t>();
        const scalar_t* exz  = exz_c.data_ptr<scalar_t>();
        const scalar_t* eyz  = eyz_c.data_ptr<scalar_t>();
        const scalar_t* pp   = p_c.data_ptr<scalar_t>();
        const scalar_t* nrho = nrho_c.data_ptr<scalar_t>();
        const int64_t* offs  = offs_c.data_ptr<int64_t>();
        double* outp         = out.data_ptr<double>();

        const int64_t T_total = ct_c.size(1);
        const scalar_t* cx = ct_c.data_ptr<scalar_t>();
        const scalar_t* cy = cx + T_total;
        const scalar_t* cz = cy + T_total;
        const scalar_t* nx_p = nm_c.data_ptr<scalar_t>();
        const scalar_t* ny_p = nx_p + T_total;
        const scalar_t* nz_p = ny_p + T_total;
        const scalar_t* area = ar_c.data_ptr<scalar_t>();

        const scalar_t* com = com_c.data_ptr<scalar_t>();

        const int iMx = (int)Mx, iMy = (int)My, iMz = (int)Mz;
        const scalar_t bx0s = (scalar_t)bx0, by0s = (scalar_t)by0, bz0s = (scalar_t)bz0;
        const scalar_t idx = (scalar_t)inv_dx, idy = (scalar_t)inv_dy, idz = (scalar_t)inv_dz;
        const int method = (int)interp_method;
        // See lagrangian_forces.cu — sample fields offset away from the
        // surface to escape the BDIM band, independently per channel.
        const scalar_t off_pres = (scalar_t)sample_offset_pressure;
        const scalar_t off_visc = (scalar_t)sample_offset_friction;

        at::parallel_for(0, B, 0, [&](int64_t b_start, int64_t b_end) {
            for (int64_t b = b_start; b < b_end; ++b) {
                const int64_t t0 = offs[b];
                const int64_t t1 = offs[b + 1];
                if (t1 <= t0) continue;

                const scalar_t com_x = com[b * 3 + 0];
                const scalar_t com_y = com[b * 3 + 1];
                const scalar_t com_z = com[b * 3 + 2];

                double acc[12] = {0,0,0,0,0,0,0,0,0,0,0,0};

                for (int64_t t = t0; t < t1; ++t) {
                    const scalar_t qx = cx[t];
                    const scalar_t qy = cy[t];
                    const scalar_t qz = cz[t];
                    const scalar_t nxv = nx_p[t];
                    const scalar_t nyv = ny_p[t];
                    const scalar_t nzv = nz_p[t];
                    const scalar_t A   = area[t];
                    const scalar_t qxv = qx + off_visc * nxv;
                    const scalar_t qyv = qy + off_visc * nyv;
                    const scalar_t qzv = qz + off_visc * nzv;

                    const scalar_t e_xx = lf_sample_3d<scalar_t>(
                        method, exx, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxv, qyv, qzv);
                    const scalar_t e_yy = lf_sample_3d<scalar_t>(
                        method, eyy, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxv, qyv, qzv);
                    const scalar_t e_zz = lf_sample_3d<scalar_t>(
                        method, ezz, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxv, qyv, qzv);
                    const scalar_t e_xy = lf_sample_3d<scalar_t>(
                        method, exy, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxv, qyv, qzv);
                    const scalar_t e_xz = lf_sample_3d<scalar_t>(
                        method, exz, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxv, qyv, qzv);
                    const scalar_t e_yz = lf_sample_3d<scalar_t>(
                        method, eyz, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxv, qyv, qzv);

                    // nu_rho rides with the viscous channel.
                    const scalar_t nu_rho_m = nrho_scalar
                        ? nrho[0]
                        : lf_sample_3d<scalar_t>(
                            method, nrho, iMx, iMy, iMz, bx0s, by0s, bz0s,
                            idx, idy, idz, qxv, qyv, qzv);

                    const scalar_t tvx = nu_rho_m * (e_xx * nxv + e_xy * nyv + e_xz * nzv);
                    const scalar_t tvy = nu_rho_m * (e_xy * nxv + e_yy * nyv + e_yz * nzv);
                    const scalar_t tvz = nu_rho_m * (e_xz * nxv + e_yz * nyv + e_zz * nzv);

                    const scalar_t qxp = qx + off_pres * nxv;
                    const scalar_t qyp = qy + off_pres * nyv;
                    const scalar_t qzp = qz + off_pres * nzv;

                    const scalar_t p_m = lf_sample_3d<scalar_t>(
                        method, pp, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxp, qyp, qzp);
                    const scalar_t tpx = -p_m * nxv;
                    const scalar_t tpy = -p_m * nyv;
                    const scalar_t tpz = -p_m * nzv;

                    const scalar_t rx = qx - com_x;
                    const scalar_t ry = qy - com_y;
                    const scalar_t rz = qz - com_z;

                    acc[0]  += (double)(tvx * A);
                    acc[1]  += (double)(tvy * A);
                    acc[2]  += (double)(tvz * A);
                    acc[3]  += (double)((ry * tvz - rz * tvy) * A);
                    acc[4]  += (double)((rz * tvx - rx * tvz) * A);
                    acc[5]  += (double)((rx * tvy - ry * tvx) * A);
                    acc[6]  += (double)(tpx * A);
                    acc[7]  += (double)(tpy * A);
                    acc[8]  += (double)(tpz * A);
                    acc[9]  += (double)((ry * tpz - rz * tpy) * A);
                    acc[10] += (double)((rz * tpx - rx * tpz) * A);
                    acc[11] += (double)((rx * tpy - ry * tpx) * A);
                }

                for (int c = 0; c < 12; ++c) outp[b * 12 + c] = acc[c];
            }
        });
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("lagrangian_forces_3d", &lagrangian_forces_3d_cpu);
}

// =====================================================================
//  2-D Lagrangian forces (merged from lagrangian_forces_cpu_2d.cpp)
// =====================================================================

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
    const double sample_offset_pressure,
    const double sample_offset_friction,
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
        // See lagrangian_forces.cu — sample fields offset away from the
        // surface to escape the BDIM band, independently per channel.
        const scalar_t off_pres = (scalar_t)sample_offset_pressure;
        const scalar_t off_visc = (scalar_t)sample_offset_friction;

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
                    const scalar_t qxv = qx + off_visc * nx;
                    const scalar_t qyv = qy + off_visc * ny;

                    const scalar_t e_xx = lf_sample_2d<scalar_t>(
                        method, exx, iMx, iMy, bx0s, by0s, idx, idy, qxv, qyv);
                    const scalar_t e_xy = lf_sample_2d<scalar_t>(
                        method, exy, iMx, iMy, bx0s, by0s, idx, idy, qxv, qyv);
                    const scalar_t e_yy = lf_sample_2d<scalar_t>(
                        method, eyy, iMx, iMy, bx0s, by0s, idx, idy, qxv, qyv);

                    // nu_rho rides with the viscous channel.
                    const scalar_t nu_rho_m = nrho_scalar
                        ? nrho[0]
                        : lf_sample_2d<scalar_t>(
                            method, nrho, iMx, iMy, bx0s, by0s, idx, idy, qxv, qyv);

                    tvx[k] = nu_rho_m * (e_xx * nx + e_xy * ny);
                    tvy[k] = nu_rho_m * (e_xy * nx + e_yy * ny);

                    const scalar_t qxp = qx + off_pres * nx;
                    const scalar_t qyp = qy + off_pres * ny;

                    const scalar_t p_m = lf_sample_2d<scalar_t>(
                        method, pp, iMx, iMy, bx0s, by0s, idx, idy, qxp, qyp);
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
