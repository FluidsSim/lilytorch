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
    const double sample_offset,
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
        // surface to escape the BDIM band.
        const scalar_t soff = (scalar_t)sample_offset;

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
                    const scalar_t qxs = qx + soff * nxv;
                    const scalar_t qys = qy + soff * nyv;
                    const scalar_t qzs = qz + soff * nzv;

                    const scalar_t e_xx = lf_sample_3d<scalar_t>(
                        method, exx, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxs, qys, qzs);
                    const scalar_t e_yy = lf_sample_3d<scalar_t>(
                        method, eyy, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxs, qys, qzs);
                    const scalar_t e_zz = lf_sample_3d<scalar_t>(
                        method, ezz, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxs, qys, qzs);
                    const scalar_t e_xy = lf_sample_3d<scalar_t>(
                        method, exy, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxs, qys, qzs);
                    const scalar_t e_xz = lf_sample_3d<scalar_t>(
                        method, exz, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxs, qys, qzs);
                    const scalar_t e_yz = lf_sample_3d<scalar_t>(
                        method, eyz, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxs, qys, qzs);

                    const scalar_t nu_rho_m = nrho_scalar
                        ? nrho[0]
                        : lf_sample_3d<scalar_t>(
                            method, nrho, iMx, iMy, iMz, bx0s, by0s, bz0s,
                            idx, idy, idz, qxs, qys, qzs);

                    const scalar_t tvx = nu_rho_m * (e_xx * nxv + e_xy * nyv + e_xz * nzv);
                    const scalar_t tvy = nu_rho_m * (e_xy * nxv + e_yy * nyv + e_yz * nzv);
                    const scalar_t tvz = nu_rho_m * (e_xz * nxv + e_yz * nyv + e_zz * nzv);

                    const scalar_t p_m = lf_sample_3d<scalar_t>(
                        method, pp, iMx, iMy, iMz, bx0s, by0s, bz0s,
                        idx, idy, idz, qxs, qys, qzs);
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

}  // namespace lilytorch_kernels
