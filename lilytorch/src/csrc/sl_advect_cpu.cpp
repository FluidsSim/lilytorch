// =====================================================================
//  sl_advect_{2d,3d} + diffuse_add — CPU twins (at::parallel_for).
//  Interpolation functions mirror the CUDA bilinear/biquadratic/trilinear/
//  triquadratic exactly — same clamping, same boundary-fallback, same
//  quadratic-B-spline weights.
// =====================================================================

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <torch/all.h>
#include <torch/library.h>

namespace lilytorch_kernels {

// =====================================================================
//  CPU bilinear sample (2-D, off-grid indexing)
//  Mirrors interpolation.bilinear_sample_off_2d exactly.
// =====================================================================
template <typename scalar_t>
static inline scalar_t bilinear_sample_off_2d_cpu(
    const scalar_t* F, int offset, int Mx, int My,
    scalar_t bx0, scalar_t by0, scalar_t inv_dx, scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    tx = std::max(scalar_t(0), std::min(tx, scalar_t(Mx - 1)));
    ty = std::max(scalar_t(0), std::min(ty, scalar_t(My - 1)));
    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;

    scalar_t fx = tx - scalar_t(ix);
    scalar_t fy = ty - scalar_t(iy);
    scalar_t wx0 = scalar_t(1) - fx;
    scalar_t wy0 = scalar_t(1) - fy;

    int s1 = My;
    int base = offset + ix * s1 + iy;
    return wx0 * (wy0 * F[base] + fy * F[base + 1])
         +  fx * (wy0 * F[base + s1] + fy * F[base + s1 + 1]);
}

// =====================================================================
//  CPU biquadratic sample (2-D, off-grid indexing)
//  Mirrors interpolation.biquadratic_sample_off_2d exactly:
//  boundary cells → bilinear fallback; interior → quadratic B-spline.
// =====================================================================
template <typename scalar_t>
static inline scalar_t biquadratic_sample_off_2d_cpu(
    const scalar_t* F, int offset, int Mx, int My,
    scalar_t bx0, scalar_t by0, scalar_t inv_dx, scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    tx = std::max(scalar_t(0), std::min(tx, scalar_t(Mx - 1)));
    ty = std::max(scalar_t(0), std::min(ty, scalar_t(My - 1)));
    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;

    // Fall back to bilinear when the stencil would cross the border.
    if (ix < 1 || iy < 1 || Mx < 3 || My < 3) {
        return bilinear_sample_off_2d_cpu(F, offset, Mx, My,
            bx0, by0, inv_dx, inv_dy, xq, yq);
    }

    scalar_t fx = tx - scalar_t(ix);
    scalar_t fy = ty - scalar_t(iy);
    scalar_t half = scalar_t(0.5);
    scalar_t one = scalar_t(1);

    scalar_t wxm = half * fx * (fx - one);
    scalar_t wx0 = one - fx * fx;
    scalar_t wxp = half * fx * (fx + one);
    scalar_t wym = half * fy * (fy - one);
    scalar_t wy0 = one - fy * fy;
    scalar_t wyp = half * fy * (fy + one);

    int s1 = My;
    int base = offset + (ix - 1) * s1 + (iy - 1);

    scalar_t result = scalar_t(0);
    result += wxm * (wym * F[base] + wy0 * F[base + 1] + wyp * F[base + 2]);
    int b1 = base + s1;
    result += wx0 * (wym * F[b1] + wy0 * F[b1 + 1] + wyp * F[b1 + 2]);
    int b2 = base + 2 * s1;
    result += wxp * (wym * F[b2] + wy0 * F[b2 + 1] + wyp * F[b2 + 2]);
    return result;
}

// =====================================================================
//  CPU trilinear sample (3-D, off-grid indexing)
//  Mirrors interpolation.trilinear_sample_off exactly.
// =====================================================================
template <typename scalar_t>
static inline scalar_t trilinear_sample_off_3d_cpu(
    const scalar_t* F, int offset, int Mx, int My, int Mz,
    scalar_t bx0, scalar_t by0, scalar_t bz0,
    scalar_t inv_dx, scalar_t inv_dy, scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    scalar_t tz = (zq - bz0) * inv_dz;
    tx = std::max(scalar_t(0), std::min(tx, scalar_t(Mx - 1)));
    ty = std::max(scalar_t(0), std::min(ty, scalar_t(My - 1)));
    tz = std::max(scalar_t(0), std::min(tz, scalar_t(Mz - 1)));
    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;
    int iz = (int)tz; if (iz > Mz - 2) iz = Mz - 2;

    scalar_t fx = tx - scalar_t(ix);
    scalar_t fy = ty - scalar_t(iy);
    scalar_t fz = tz - scalar_t(iz);
    scalar_t wx0 = scalar_t(1) - fx;
    scalar_t wy0 = scalar_t(1) - fy;
    scalar_t wz0 = scalar_t(1) - fz;

    int s2 = Mz;
    int s1 = My * Mz;
    int base = offset + ix * s1 + iy * s2 + iz;

    return wx0 * (wy0 * (wz0 * F[base]                 + fz * F[base + 1]) +
                  fy  * (wz0 * F[base + s2]             + fz * F[base + s2 + 1]))
         + fx  * (wy0 * (wz0 * F[base + s1]             + fz * F[base + s1 + 1]) +
                  fy  * (wz0 * F[base + s1 + s2]        + fz * F[base + s1 + s2 + 1]));
}

// =====================================================================
//  CPU triquadratic sample (3-D, off-grid indexing)
//  Mirrors interpolation.triquadratic_sample_off exactly:
//  boundary cells → trilinear fallback; interior → quadratic B-spline.
// =====================================================================
template <typename scalar_t>
static inline scalar_t triquadratic_sample_off_3d_cpu(
    const scalar_t* F, int offset, int Mx, int My, int Mz,
    scalar_t bx0, scalar_t by0, scalar_t bz0,
    scalar_t inv_dx, scalar_t inv_dy, scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    scalar_t tx = (xq - bx0) * inv_dx;
    scalar_t ty = (yq - by0) * inv_dy;
    scalar_t tz = (zq - bz0) * inv_dz;
    tx = std::max(scalar_t(0), std::min(tx, scalar_t(Mx - 1)));
    ty = std::max(scalar_t(0), std::min(ty, scalar_t(My - 1)));
    tz = std::max(scalar_t(0), std::min(tz, scalar_t(Mz - 1)));
    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;
    int iz = (int)tz; if (iz > Mz - 2) iz = Mz - 2;

    // Fall back to trilinear when the stencil would cross the border.
    if (ix < 1 || iy < 1 || iz < 1 || Mx < 3 || My < 3 || Mz < 3) {
        return trilinear_sample_off_3d_cpu(F, offset, Mx, My, Mz,
            bx0, by0, bz0, inv_dx, inv_dy, inv_dz, xq, yq, zq);
    }

    scalar_t fx = tx - scalar_t(ix);
    scalar_t fy = ty - scalar_t(iy);
    scalar_t fz = tz - scalar_t(iz);
    scalar_t half = scalar_t(0.5);
    scalar_t one = scalar_t(1);

    scalar_t wxm = half * fx * (fx - one);
    scalar_t wx0 = one - fx * fx;
    scalar_t wxp = half * fx * (fx + one);
    scalar_t wym = half * fy * (fy - one);
    scalar_t wy0 = one - fy * fy;
    scalar_t wyp = half * fy * (fy + one);
    scalar_t wzm = half * fz * (fz - one);
    scalar_t wz0 = one - fz * fz;
    scalar_t wzp = half * fz * (fz + one);

    int s2 = Mz;
    int s1 = My * Mz;
    int base = offset + (ix - 1) * s1 + (iy - 1) * s2 + (iz - 1);

    scalar_t result = scalar_t(0);
    for (int dx = 0; dx < 3; ++dx) {
        scalar_t wx = (dx == 0) ? wxm : ((dx == 1) ? wx0 : wxp);
        int b0 = base + dx * s1;
        scalar_t plane = scalar_t(0);
        for (int dy = 0; dy < 3; ++dy) {
            scalar_t wy = (dy == 0) ? wym : ((dy == 1) ? wy0 : wyp);
            int b1 = b0 + dy * s2;
            scalar_t row = wzm * F[b1] + wz0 * F[b1 + 1] + wzp * F[b1 + 2];
            plane += wy * row;
        }
        result += wx * plane;
    }
    return result;
}

// =====================================================================
//  CPU launchers
// =====================================================================

static void sl_advect_2d_cpu(
    const at::Tensor& u, const at::Tensor& v,
    const at::Tensor& gxu, const at::Tensor& gyu,
    const at::Tensor& gxv, const at::Tensor& gyv,
    at::Tensor& out_u, at::Tensor& out_v,
    double u_bx0, double u_by0, double u_idx, double u_idy,
    double v_bx0, double v_by0, double v_idx, double v_idy,
    double dt)
{
    int Mx = (int)u.size(0), My = (int)u.size(1);
    int total = 2 * Mx * My;
    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "sl_advect_2d_cpu", [&] {
        const scalar_t* up = u.data_ptr<scalar_t>();
        const scalar_t* vp = v.data_ptr<scalar_t>();
        const scalar_t* gxup = gxu.data_ptr<scalar_t>();
        const scalar_t* gyup = gyu.data_ptr<scalar_t>();
        const scalar_t* gxvp = gxv.data_ptr<scalar_t>();
        const scalar_t* gyvp = gyv.data_ptr<scalar_t>();
        scalar_t* oup = out_u.data_ptr<scalar_t>();
        scalar_t* ovp = out_v.data_ptr<scalar_t>();
        scalar_t dt_s = (scalar_t)dt;
        scalar_t ubx0 = (scalar_t)u_bx0, uby0 = (scalar_t)u_by0;
        scalar_t uidx = (scalar_t)u_idx, uidy = (scalar_t)u_idy;
        scalar_t vbx0 = (scalar_t)v_bx0, vby0 = (scalar_t)v_by0;
        scalar_t vidx = (scalar_t)v_idx, vidy = (scalar_t)v_idy;
        scalar_t half = scalar_t(0.5);

        at::parallel_for(0, total, 256, [&](int64_t begin, int64_t end) {
            for (int64_t tid = begin; tid < end; ++tid) {
                if (tid >= total) break;
                int comp = (int)(tid / (Mx * My));
                int lin = (int)(tid - comp * Mx * My);
                int ixq = lin / My;
                int iyq = lin - ixq * My;
                scalar_t X = gxup[ixq], Y = gyup[iyq];
                if (comp == 1) { X = gxvp[ixq]; Y = gyvp[iyq]; }

                scalar_t u1 = biquadratic_sample_off_2d_cpu(up, 0, Mx, My, ubx0, uby0, uidx, uidy, X, Y);
                scalar_t v1 = biquadratic_sample_off_2d_cpu(vp, 0, Mx, My, vbx0, vby0, vidx, vidy, X, Y);
                scalar_t xm = X - half * dt_s * u1;
                scalar_t ym = Y - half * dt_s * v1;
                scalar_t u2 = biquadratic_sample_off_2d_cpu(up, 0, Mx, My, ubx0, uby0, uidx, uidy, xm, ym);
                scalar_t v2 = biquadratic_sample_off_2d_cpu(vp, 0, Mx, My, vbx0, vby0, vidx, vidy, xm, ym);
                scalar_t xd = X - dt_s * u2;
                scalar_t yd = Y - dt_s * v2;
                if (comp == 0)
                    oup[lin] = biquadratic_sample_off_2d_cpu(up, 0, Mx, My, ubx0, uby0, uidx, uidy, xd, yd);
                else
                    ovp[lin] = biquadratic_sample_off_2d_cpu(vp, 0, Mx, My, vbx0, vby0, vidx, vidy, xd, yd);
            }
        });
    });
}

static void sl_advect_3d_cpu(
    const at::Tensor& u, const at::Tensor& v, const at::Tensor& w,
    const at::Tensor& gxu, const at::Tensor& gyu, const at::Tensor& gzu,
    const at::Tensor& gxv, const at::Tensor& gyv, const at::Tensor& gzv,
    const at::Tensor& gxw, const at::Tensor& gyw, const at::Tensor& gzw,
    at::Tensor& out_u, at::Tensor& out_v, at::Tensor& out_w,
    double u_bx0, double u_by0, double u_bz0, double u_idx, double u_idy, double u_idz,
    double v_bx0, double v_by0, double v_bz0, double v_idx, double v_idy, double v_idz,
    double w_bx0, double w_by0, double w_bz0, double w_idx, double w_idy, double w_idz,
    double dt)
{
    int Mx = (int)u.size(0), My = (int)u.size(1), Mz = (int)u.size(2);
    int total = 3 * Mx * My * Mz;
    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "sl_advect_3d_cpu", [&] {
        const scalar_t* up = u.data_ptr<scalar_t>();
        const scalar_t* vp = v.data_ptr<scalar_t>();
        const scalar_t* wp = w.data_ptr<scalar_t>();
        const scalar_t* gxup = gxu.data_ptr<scalar_t>();
        const scalar_t* gyup = gyu.data_ptr<scalar_t>();
        const scalar_t* gzup = gzu.data_ptr<scalar_t>();
        const scalar_t* gxvp = gxv.data_ptr<scalar_t>();
        const scalar_t* gyvp = gyv.data_ptr<scalar_t>();
        const scalar_t* gzvp = gzv.data_ptr<scalar_t>();
        const scalar_t* gxwp = gxw.data_ptr<scalar_t>();
        const scalar_t* gywp = gyw.data_ptr<scalar_t>();
        const scalar_t* gzwp = gzw.data_ptr<scalar_t>();
        scalar_t* oup = out_u.data_ptr<scalar_t>();
        scalar_t* ovp = out_v.data_ptr<scalar_t>();
        scalar_t* owp = out_w.data_ptr<scalar_t>();
        scalar_t dt_s = (scalar_t)dt;
        scalar_t ubx0 = (scalar_t)u_bx0, uby0 = (scalar_t)u_by0, ubz0 = (scalar_t)u_bz0;
        scalar_t uidx = (scalar_t)u_idx, uidy = (scalar_t)u_idy, uidz = (scalar_t)u_idz;
        scalar_t vbx0 = (scalar_t)v_bx0, vby0 = (scalar_t)v_by0, vbz0 = (scalar_t)v_bz0;
        scalar_t vidx = (scalar_t)v_idx, vidy = (scalar_t)v_idy, vidz = (scalar_t)v_idz;
        scalar_t wbx0 = (scalar_t)w_bx0, wby0 = (scalar_t)w_by0, wbz0 = (scalar_t)w_bz0;
        scalar_t widx = (scalar_t)w_idx, widy = (scalar_t)w_idy, widz = (scalar_t)w_idz;
        scalar_t half = scalar_t(0.5);

        at::parallel_for(0, total, 256, [&](int64_t begin, int64_t end) {
            for (int64_t tid = begin; tid < end; ++tid) {
                if (tid >= total) break;
                int per_comp = Mx * My * Mz;
                int comp = (int)(tid / per_comp);
                int lin = (int)(tid - comp * per_comp);
                int ixy = lin / Mz;
                int izq = lin - ixy * Mz;
                int ixq = ixy / My;
                int iyq = ixy - ixq * My;

                scalar_t X, Y, Z;
                if (comp == 0) {
                    X = gxup[ixq]; Y = gyup[iyq]; Z = gzup[izq];
                } else if (comp == 1) {
                    X = gxvp[ixq]; Y = gyvp[iyq]; Z = gzvp[izq];
                } else {
                    X = gxwp[ixq]; Y = gywp[iyq]; Z = gzwp[izq];
                }

                // Stage 1: velocity at the node → midpoint.
                scalar_t u1 = triquadratic_sample_off_3d_cpu(up, 0, Mx, My, Mz,
                    ubx0, uby0, ubz0, uidx, uidy, uidz, X, Y, Z);
                scalar_t v1 = triquadratic_sample_off_3d_cpu(vp, 0, Mx, My, Mz,
                    vbx0, vby0, vbz0, vidx, vidy, vidz, X, Y, Z);
                scalar_t w1 = triquadratic_sample_off_3d_cpu(wp, 0, Mx, My, Mz,
                    wbx0, wby0, wbz0, widx, widy, widz, X, Y, Z);
                scalar_t xm = X - half * dt_s * u1;
                scalar_t ym = Y - half * dt_s * v1;
                scalar_t zm = Z - half * dt_s * w1;

                // Stage 2: velocity at the midpoint → departure.
                scalar_t u2 = triquadratic_sample_off_3d_cpu(up, 0, Mx, My, Mz,
                    ubx0, uby0, ubz0, uidx, uidy, uidz, xm, ym, zm);
                scalar_t v2 = triquadratic_sample_off_3d_cpu(vp, 0, Mx, My, Mz,
                    vbx0, vby0, vbz0, vidx, vidy, vidz, xm, ym, zm);
                scalar_t w2 = triquadratic_sample_off_3d_cpu(wp, 0, Mx, My, Mz,
                    wbx0, wby0, wbz0, widx, widy, widz, xm, ym, zm);
                scalar_t xd = X - dt_s * u2;
                scalar_t yd = Y - dt_s * v2;
                scalar_t zd = Z - dt_s * w2;

                // Sample the advected component at the departure point.
                if (comp == 0) {
                    oup[lin] = triquadratic_sample_off_3d_cpu(up, 0, Mx, My, Mz,
                        ubx0, uby0, ubz0, uidx, uidy, uidz, xd, yd, zd);
                } else if (comp == 1) {
                    ovp[lin] = triquadratic_sample_off_3d_cpu(vp, 0, Mx, My, Mz,
                        vbx0, vby0, vbz0, vidx, vidy, vidz, xd, yd, zd);
                } else {
                    owp[lin] = triquadratic_sample_off_3d_cpu(wp, 0, Mx, My, Mz,
                        wbx0, wby0, wbz0, widx, widy, widz, xd, yd, zd);
                }
            }
        });
    });
}

static void diffuse_add_cpu(
    at::Tensor& target,
    const at::Tensor& copy_buf,
    const at::Tensor& nu_eff,
    double dt,
    int64_t ndim,
    double dh0, double dh1, double dh2,
    double scale_constant)
{
    int Nx, Ny, Nz, Nix, Niy, Niz, s_x, s_y, s_z;
    if (ndim == 2) {
        Nx = (int)target.size(0); Ny = (int)target.size(1); Nz = 1;
        Nix = Nx - 2; Niy = Ny - 2; Niz = 1;
        s_x = (int)target.stride(0); s_y = (int)target.stride(1); s_z = 0;
    } else {
        Nx = (int)target.size(0); Ny = (int)target.size(1); Nz = (int)target.size(2);
        Nix = Nx - 2; Niy = Ny - 2; Niz = Nz - 2;
        s_x = (int)target.stride(0); s_y = (int)target.stride(1); s_z = (int)target.stride(2);
    }
    int total = Nix * Niy * Niz;
    bool is_variable = nu_eff.numel() > 1;
    // Use pre-computed scale_constant for constant case (avoids .item()
    // GPU→CPU sync which would be illegal during CUDA graph capture).
    double scale = is_variable ? dt : scale_constant;

    AT_DISPATCH_FLOATING_TYPES(target.scalar_type(), "diffuse_add_cpu", [&] {
        const scalar_t* ps = copy_buf.data_ptr<scalar_t>();
        scalar_t* pd = target.data_ptr<scalar_t>();
        const scalar_t* nu = nu_eff.data_ptr<scalar_t>();
        scalar_t sc = (scalar_t)scale;
        scalar_t idx2 = (scalar_t)(1.0 / (dh0 * dh0));
        scalar_t idy2 = (scalar_t)(1.0 / (dh1 * dh1));
        scalar_t idz2 = (ndim == 3 ? (scalar_t)(1.0 / (dh2 * dh2)) : scalar_t(0));
        scalar_t two = scalar_t(2);

        at::parallel_for(0, total, 512, [&](int64_t begin, int64_t end) {
            for (int64_t tid = begin; tid < end; ++tid) {
                int iz = (int)(tid % Niz);
                int ixy = (int)(tid / Niz);
                int iy = ixy % Niy;
                int ix = ixy / Niy;
                int gx = ix + 1, gy = iy + 1, gz = iz + 1;
                int c  = gx * s_x + gy * s_y + gz * s_z;
                int xp = (gx + 1) * s_x + gy * s_y + gz * s_z;
                int xm = (gx - 1) * s_x + gy * s_y + gz * s_z;
                int yp = gx * s_x + (gy + 1) * s_y + gz * s_z;
                int ym = gx * s_x + (gy - 1) * s_y + gz * s_z;
                scalar_t lap;
                if (is_variable) {
                    scalar_t tiny = scalar_t(1e-30);
                    scalar_t nu_c = nu[c];
                    scalar_t nu_f = two * nu_c * nu[xp] / (nu_c + nu[xp] + tiny);
                    scalar_t nu_b = two * nu_c * nu[xm] / (nu_c + nu[xm] + tiny);
                    lap = (nu_f * (ps[xp] - ps[c]) - nu_b * (ps[c] - ps[xm])) * idx2;
                    nu_f = two * nu_c * nu[yp] / (nu_c + nu[yp] + tiny);
                    nu_b = two * nu_c * nu[ym] / (nu_c + nu[ym] + tiny);
                    lap += (nu_f * (ps[yp] - ps[c]) - nu_b * (ps[c] - ps[ym])) * idy2;
                    if (Nz > 1) {
                        int zp = gx * s_x + gy * s_y + (gz + 1) * s_z;
                        int zm = gx * s_x + gy * s_y + (gz - 1) * s_z;
                        nu_f = two * nu_c * nu[zp] / (nu_c + nu[zp] + tiny);
                        nu_b = two * nu_c * nu[zm] / (nu_c + nu[zm] + tiny);
                        lap += (nu_f * (ps[zp] - ps[c]) - nu_b * (ps[c] - ps[zm])) * idz2;
                    }
                } else {
                    lap = (ps[xp] - two * ps[c] + ps[xm]) * idx2;
                    lap += (ps[yp] - two * ps[c] + ps[ym]) * idy2;
                    if (Nz > 1) {
                        int zp = gx * s_x + gy * s_y + (gz + 1) * s_z;
                        int zm = gx * s_x + gy * s_y + (gz - 1) * s_z;
                        lap += (ps[zp] - two * ps[c] + ps[zm]) * idz2;
                    }
                }
                pd[c] = pd[c] + sc * lap;
            }
        });
    });
}

// =====================================================================
//  Register with CPU backend
// =====================================================================
TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("sl_advect_2d", &sl_advect_2d_cpu);
    m.impl("sl_advect_3d", &sl_advect_3d_cpu);
    m.impl("diffuse_add", &diffuse_add_cpu);
}

}  // namespace lilytorch_kernels
