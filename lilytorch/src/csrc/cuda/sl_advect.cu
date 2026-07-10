// =====================================================================
//  sl_advect_{2d,3d} + diffuse_add — native CUDA kernels for the
//  pre-Poisson step region (item 8.D).
//
//  Faithful line-for-line port of the Warp kernels in advection.py
//  (sl_advect_{2,3}d_kernel) and diffusion.py (fused_laplacian_accumulate).
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace lilytorch_kernels {

// =====================================================================
//  Biquadratic sample — 2-D, off-grid indexing (EXACT Warp port).
// =====================================================================
template <typename scalar_t>
__device__ __forceinline__ scalar_t biquadratic_sample_off_2d(
    const scalar_t* __restrict__ F,
    int F_off, int Mx, int My,
    scalar_t bx0, scalar_t by0,
    scalar_t inv_dx, scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    scalar_t tx = max(scalar_t(0), min((xq - bx0) * inv_dx, scalar_t(Mx - 1)));
    scalar_t ty = max(scalar_t(0), min((yq - by0) * inv_dy, scalar_t(My - 1)));

    int ix = min((int)tx, Mx - 2);
    int iy = min((int)ty, My - 2);

    // Fall back to bilinear at the table border.
    if (ix < 1 || iy < 1 || Mx < 3 || My < 3) {
        // Simple bilinear fallback
        int ix0 = min(ix, Mx - 2);
        int iy0 = min(iy, My - 2);
        scalar_t fx = tx - scalar_t(ix0);
        scalar_t fy = ty - scalar_t(iy0);
        int stride_i = My;
        int b = F_off + ix0 * stride_i + iy0;
        return (scalar_t(1) - fx) * (scalar_t(1) - fy) * F[b]
             + fx * (scalar_t(1) - fy) * F[b + stride_i]
             + (scalar_t(1) - fx) * fy * F[b + 1]
             + fx * fy * F[b + stride_i + 1];
    }

    scalar_t fx = tx - scalar_t(ix);
    scalar_t fy = ty - scalar_t(iy);

    scalar_t half = scalar_t(0.5);
    scalar_t one  = scalar_t(1);
    scalar_t wxm = half * fx * (fx - one);
    scalar_t wx0 = one - fx * fx;
    scalar_t wxp = half * fx * (fx + one);
    scalar_t wym = half * fy * (fy - one);
    scalar_t wy0 = one - fy * fy;
    scalar_t wyp = half * fy * (fy + one);

    int s1 = My;
    int base = F_off + (ix - 1) * s1 + (iy - 1);

    // dx = 0
    scalar_t col0 = wym * F[base] + wy0 * F[base + 1] + wyp * F[base + 2];
    // dx = 1
    int b1 = base + s1;
    scalar_t col1 = wym * F[b1] + wy0 * F[b1 + 1] + wyp * F[b1 + 2];
    // dx = 2
    int b2 = base + 2 * s1;
    scalar_t col2 = wym * F[b2] + wy0 * F[b2 + 1] + wyp * F[b2 + 2];

    return wxm * col0 + wx0 * col1 + wxp * col2;
}

// =====================================================================
//  Triquadratic sample — 3-D, off-grid indexing (EXACT Warp port).
// =====================================================================
template <typename scalar_t>
__device__ __forceinline__ scalar_t triquadratic_sample_off(
    const scalar_t* __restrict__ F,
    int F_off, int Mx, int My, int Mz,
    scalar_t bx0, scalar_t by0, scalar_t bz0,
    scalar_t inv_dx, scalar_t inv_dy, scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    scalar_t zero = scalar_t(0), one = scalar_t(1);
    scalar_t tx = max(zero, min((xq - bx0) * inv_dx, scalar_t(Mx - 1)));
    scalar_t ty = max(zero, min((yq - by0) * inv_dy, scalar_t(My - 1)));
    scalar_t tz = max(zero, min((zq - bz0) * inv_dz, scalar_t(Mz - 1)));

    int ix = min((int)tx, Mx - 2);
    int iy = min((int)ty, My - 2);
    int iz = min((int)tz, Mz - 2);

    // Fall back to trilinear at the table border.
    if (ix < 1 || iy < 1 || iz < 1 || Mx < 3 || My < 3 || Mz < 3) {
        int ix0 = max(0, min(ix, Mx - 2));
        int iy0 = max(0, min(iy, My - 2));
        int iz0 = max(0, min(iz, Mz - 2));
        scalar_t fx = tx - scalar_t(ix0);
        scalar_t fy = ty - scalar_t(iy0);
        scalar_t fz = tz - scalar_t(iz0);
        scalar_t w000 = (one - fx) * (one - fy) * (one - fz);
        scalar_t w100 = fx * (one - fy) * (one - fz);
        scalar_t w010 = (one - fx) * fy * (one - fz);
        scalar_t w110 = fx * fy * (one - fz);
        scalar_t w001 = (one - fx) * (one - fy) * fz;
        scalar_t w101 = fx * (one - fy) * fz;
        scalar_t w011 = (one - fx) * fy * fz;
        scalar_t w111 = fx * fy * fz;
        int s2 = Mz, s1 = My * Mz;
        int b = F_off + ix0 * s1 + iy0 * s2 + iz0;
        return w000 * F[b] + w100 * F[b + s1] + w010 * F[b + s2] + w110 * F[b + s1 + s2]
             + w001 * F[b + 1] + w101 * F[b + s1 + 1] + w011 * F[b + s2 + 1] + w111 * F[b + s1 + s2 + 1];
    }

    scalar_t fx = tx - scalar_t(ix);
    scalar_t fy = ty - scalar_t(iy);
    scalar_t fz = tz - scalar_t(iz);

    scalar_t half = scalar_t(0.5);
    scalar_t wxm = half * fx * (fx - one), wx0 = one - fx * fx, wxp = half * fx * (fx + one);
    scalar_t wym = half * fy * (fy - one), wy0 = one - fy * fy, wyp = half * fy * (fy + one);
    scalar_t wzm = half * fz * (fz - one), wz0 = one - fz * fz, wzp = half * fz * (fz + one);

    int s2 = Mz, s1 = My * Mz;
    int base = F_off + (ix - 1) * s1 + (iy - 1) * s2 + (iz - 1);

    scalar_t out = scalar_t(0);
    for (int dx = 0; dx < 3; ++dx) {
        scalar_t wx = wxm;
        if (dx == 1) wx = wx0;
        if (dx == 2) wx = wxp;
        int b0 = base + dx * s1;
        scalar_t plane = scalar_t(0);
        for (int dy = 0; dy < 3; ++dy) {
            scalar_t wy = wym;
            if (dy == 1) wy = wy0;
            if (dy == 2) wy = wyp;
            int b1 = b0 + dy * s2;
            scalar_t row = wzm * F[b1] + wz0 * F[b1 + 1] + wzp * F[b1 + 2];
            plane = plane + wy * row;
        }
        out = out + wx * plane;
    }
    return out;
}

// =====================================================================
//  sl_advect_2d CUDA kernel — fused RK2 semi-Lagrangian advection
// =====================================================================
template <typename scalar_t>
__global__ void sl_advect_2d_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ gxu,
    const scalar_t* __restrict__ gyu,
    const scalar_t* __restrict__ gxv,
    const scalar_t* __restrict__ gyv,
    int Mx, int My,
    scalar_t u_bx0, scalar_t u_by0, scalar_t u_idx, scalar_t u_idy,
    scalar_t v_bx0, scalar_t v_by0, scalar_t v_idx, scalar_t v_idy,
    scalar_t dt,
    scalar_t* __restrict__ out_u,
    scalar_t* __restrict__ out_v)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Mx * My;
    if (tid >= 2 * total) return;
    int comp = tid / total;
    int lin = tid - comp * total;
    int ixq = lin / My;
    int iyq = lin - ixq * My;

    scalar_t X = gxu[ixq];
    scalar_t Y = gyu[iyq];
    if (comp == 1) {
        X = gxv[ixq];
        Y = gyv[iyq];
    }

    scalar_t half = scalar_t(0.5);

    // Stage 1: velocity at node → midpoint
    scalar_t u1 = biquadratic_sample_off_2d(u, 0, Mx, My, u_bx0, u_by0, u_idx, u_idy, X, Y);
    scalar_t v1 = biquadratic_sample_off_2d(v, 0, Mx, My, v_bx0, v_by0, v_idx, v_idy, X, Y);
    scalar_t xm = X - half * dt * u1;
    scalar_t ym = Y - half * dt * v1;

    // Stage 2: velocity at midpoint → departure
    scalar_t u2 = biquadratic_sample_off_2d(u, 0, Mx, My, u_bx0, u_by0, u_idx, u_idy, xm, ym);
    scalar_t v2 = biquadratic_sample_off_2d(v, 0, Mx, My, v_bx0, v_by0, v_idx, v_idy, xm, ym);
    scalar_t xd = X - dt * u2;
    scalar_t yd = Y - dt * v2;

    if (comp == 0) {
        out_u[lin] = biquadratic_sample_off_2d(u, 0, Mx, My, u_bx0, u_by0, u_idx, u_idy, xd, yd);
    } else {
        out_v[lin] = biquadratic_sample_off_2d(v, 0, Mx, My, v_bx0, v_by0, v_idx, v_idy, xd, yd);
    }
}

// =====================================================================
//  sl_advect_3d CUDA kernel
// =====================================================================
template <typename scalar_t>
__global__ void sl_advect_3d_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ gxu,
    const scalar_t* __restrict__ gyu,
    const scalar_t* __restrict__ gzu,
    const scalar_t* __restrict__ gxv,
    const scalar_t* __restrict__ gyv,
    const scalar_t* __restrict__ gzv,
    const scalar_t* __restrict__ gxw,
    const scalar_t* __restrict__ gyw,
    const scalar_t* __restrict__ gzw,
    int Mx, int My, int Mz,
    scalar_t u_bx0, scalar_t u_by0, scalar_t u_bz0, scalar_t u_idx, scalar_t u_idy, scalar_t u_idz,
    scalar_t v_bx0, scalar_t v_by0, scalar_t v_bz0, scalar_t v_idx, scalar_t v_idy, scalar_t v_idz,
    scalar_t w_bx0, scalar_t w_by0, scalar_t w_bz0, scalar_t w_idx, scalar_t w_idy, scalar_t w_idz,
    scalar_t dt,
    scalar_t* __restrict__ out_u,
    scalar_t* __restrict__ out_v,
    scalar_t* __restrict__ out_w)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Mx * My * Mz;
    if (tid >= 3 * total) return;
    int comp = tid / total;
    int lin = tid - comp * total;
    int ixy = lin / Mz;
    int izq = lin - ixy * Mz;
    int ixq = ixy / My;
    int iyq = ixy - ixq * My;

    scalar_t X = gxu[ixq], Y = gyu[iyq], Z = gzu[izq];
    if (comp == 1) { X = gxv[ixq]; Y = gyv[iyq]; Z = gzv[izq]; }
    else if (comp == 2) { X = gxw[ixq]; Y = gyw[iyq]; Z = gzw[izq]; }

    scalar_t half = scalar_t(0.5);

    scalar_t u1 = triquadratic_sample_off(u, 0, Mx, My, Mz, u_bx0, u_by0, u_bz0, u_idx, u_idy, u_idz, X, Y, Z);
    scalar_t v1 = triquadratic_sample_off(v, 0, Mx, My, Mz, v_bx0, v_by0, v_bz0, v_idx, v_idy, v_idz, X, Y, Z);
    scalar_t w1 = triquadratic_sample_off(w, 0, Mx, My, Mz, w_bx0, w_by0, w_bz0, w_idx, w_idy, w_idz, X, Y, Z);
    scalar_t xm = X - half * dt * u1;
    scalar_t ym = Y - half * dt * v1;
    scalar_t zm = Z - half * dt * w1;

    scalar_t u2 = triquadratic_sample_off(u, 0, Mx, My, Mz, u_bx0, u_by0, u_bz0, u_idx, u_idy, u_idz, xm, ym, zm);
    scalar_t v2 = triquadratic_sample_off(v, 0, Mx, My, Mz, v_bx0, v_by0, v_bz0, v_idx, v_idy, v_idz, xm, ym, zm);
    scalar_t w2 = triquadratic_sample_off(w, 0, Mx, My, Mz, w_bx0, w_by0, w_bz0, w_idx, w_idy, w_idz, xm, ym, zm);
    scalar_t xd = X - dt * u2;
    scalar_t yd = Y - dt * v2;
    scalar_t zd = Z - dt * w2;

    if (comp == 0) {
        out_u[lin] = triquadratic_sample_off(u, 0, Mx, My, Mz, u_bx0, u_by0, u_bz0, u_idx, u_idy, u_idz, xd, yd, zd);
    } else if (comp == 1) {
        out_v[lin] = triquadratic_sample_off(v, 0, Mx, My, Mz, v_bx0, v_by0, v_bz0, v_idx, v_idy, v_idz, xd, yd, zd);
    } else {
        out_w[lin] = triquadratic_sample_off(w, 0, Mx, My, Mz, w_bx0, w_by0, w_bz0, w_idx, w_idy, w_idz, xd, yd, zd);
    }
}

// =====================================================================
//  diffuse_add CUDA kernel — fused Laplacian-accumulate
//  2-D handled as degenerate 3-D (Nz=1, Niz=1, s_z=0, inv_dh2_z=0)
// =====================================================================
template <typename scalar_t>
__global__ void diffuse_add_kernel(
    const scalar_t* __restrict__ phi_src,
    scalar_t* __restrict__ phi_dst,
    const scalar_t* __restrict__ nu_eff,
    scalar_t scale,
    int is_variable,
    int Nx, int Ny, int Nz,
    int Nix, int Niy, int Niz,
    int s_x, int s_y, int s_z,
    scalar_t inv_dh2_x, scalar_t inv_dh2_y, scalar_t inv_dh2_z)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nix * Niy * Niz;
    if (tid >= total) return;
    int iz = tid % Niz;
    int ixy = tid / Niz;
    int iy = ixy % Niy;
    int ix = ixy / Niy;
    int gx = ix + 1;
    int gy = iy + 1;
    int gz = iz + 1;
    int c  = gx * s_x + gy * s_y + gz * s_z;
    int xp = (gx + 1) * s_x + gy * s_y + gz * s_z;
    int xm = (gx - 1) * s_x + gy * s_y + gz * s_z;
    int yp = gx * s_x + (gy + 1) * s_y + gz * s_z;
    int ym = gx * s_x + (gy - 1) * s_y + gz * s_z;
    scalar_t two = scalar_t(2);
    scalar_t lap;
    if (is_variable) {
        scalar_t tiny = scalar_t(1e-30);
        scalar_t nu_c = nu_eff[c];
        // x-direction
        scalar_t nu_f = two * nu_c * nu_eff[xp] / (nu_c + nu_eff[xp] + tiny);
        scalar_t nu_b = two * nu_c * nu_eff[xm] / (nu_c + nu_eff[xm] + tiny);
        lap = (nu_f * (phi_src[xp] - phi_src[c]) - nu_b * (phi_src[c] - phi_src[xm])) * inv_dh2_x;
        // y-direction
        nu_f = two * nu_c * nu_eff[yp] / (nu_c + nu_eff[yp] + tiny);
        nu_b = two * nu_c * nu_eff[ym] / (nu_c + nu_eff[ym] + tiny);
        lap += (nu_f * (phi_src[yp] - phi_src[c]) - nu_b * (phi_src[c] - phi_src[ym])) * inv_dh2_y;
        if (Nz > 1) {
            int zp = gx * s_x + gy * s_y + (gz + 1) * s_z;
            int zm = gx * s_x + gy * s_y + (gz - 1) * s_z;
            nu_f = two * nu_c * nu_eff[zp] / (nu_c + nu_eff[zp] + tiny);
            nu_b = two * nu_c * nu_eff[zm] / (nu_c + nu_eff[zm] + tiny);
            lap += (nu_f * (phi_src[zp] - phi_src[c]) - nu_b * (phi_src[c] - phi_src[zm])) * inv_dh2_z;
        }
    } else {
        lap = (phi_src[xp] - two * phi_src[c] + phi_src[xm]) * inv_dh2_x;
        lap += (phi_src[yp] - two * phi_src[c] + phi_src[ym]) * inv_dh2_y;
        if (Nz > 1) {
            int zp = gx * s_x + gy * s_y + (gz + 1) * s_z;
            int zm = gx * s_x + gy * s_y + (gz - 1) * s_z;
            lap += (phi_src[zp] - two * phi_src[c] + phi_src[zm]) * inv_dh2_z;
        }
    }
    phi_dst[c] = phi_dst[c] + scale * lap;
}

// =====================================================================
//  C++ launcher wrappers
// =====================================================================

static void sl_advect_2d_cuda(
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
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "sl_advect_2d_cuda", [&] {
        sl_advect_2d_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            u.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            gxu.data_ptr<scalar_t>(), gyu.data_ptr<scalar_t>(),
            gxv.data_ptr<scalar_t>(), gyv.data_ptr<scalar_t>(),
            Mx, My,
            (scalar_t)u_bx0, (scalar_t)u_by0, (scalar_t)u_idx, (scalar_t)u_idy,
            (scalar_t)v_bx0, (scalar_t)v_by0, (scalar_t)v_idx, (scalar_t)v_idy,
            (scalar_t)dt,
            out_u.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>());
    });
}

static void sl_advect_3d_cuda(
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
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "sl_advect_3d_cuda", [&] {
        sl_advect_3d_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            u.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
            gxu.data_ptr<scalar_t>(), gyu.data_ptr<scalar_t>(), gzu.data_ptr<scalar_t>(),
            gxv.data_ptr<scalar_t>(), gyv.data_ptr<scalar_t>(), gzv.data_ptr<scalar_t>(),
            gxw.data_ptr<scalar_t>(), gyw.data_ptr<scalar_t>(), gzw.data_ptr<scalar_t>(),
            Mx, My, Mz,
            (scalar_t)u_bx0, (scalar_t)u_by0, (scalar_t)u_bz0, (scalar_t)u_idx, (scalar_t)u_idy, (scalar_t)u_idz,
            (scalar_t)v_bx0, (scalar_t)v_by0, (scalar_t)v_bz0, (scalar_t)v_idx, (scalar_t)v_idy, (scalar_t)v_idz,
            (scalar_t)w_bx0, (scalar_t)w_by0, (scalar_t)w_bz0, (scalar_t)w_idx, (scalar_t)w_idy, (scalar_t)w_idz,
            (scalar_t)dt,
            out_u.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>(), out_w.data_ptr<scalar_t>());
    });
}

static void diffuse_add_cuda(
    at::Tensor& target,
    const at::Tensor& copy_buf,
    const at::Tensor& nu_eff,
    double dt,
    int64_t ndim,
    double dh0, double dh1, double dh2)
{
    // Note: copy_buf is target's data snapshot (caller does copy first).
    // This kernel is the accumulate pass only.
    // We use the unified kernel with 2D mapped as Nz=1.
    int Nx, Ny, Nz;
    int Nix, Niy, Niz;
    int s_x, s_y, s_z;
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
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    bool is_variable = nu_eff.numel() > 1;
    // nu_eff is nu*dt for constant case (1-element), full nu field for variable.
    // Kernel scale = dt for variable, nu_eff value for constant.
    double scale = is_variable ? dt : (nu_eff.numel() >= 1 ? nu_eff.item<double>() : 0.0);

    AT_DISPATCH_FLOATING_TYPES(target.scalar_type(), "diffuse_add_cuda", [&] {
        diffuse_add_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            copy_buf.data_ptr<scalar_t>(),
            target.data_ptr<scalar_t>(),
            nu_eff.data_ptr<scalar_t>(),
            (scalar_t)scale,
            is_variable ? 1 : 0,
            Nx, Ny, Nz, Nix, Niy, Niz,
            s_x, s_y, s_z,
            (scalar_t)(1.0 / (dh0 * dh0)),
            (scalar_t)(1.0 / (dh1 * dh1)),
            (ndim == 3 ? (scalar_t)(1.0 / (dh2 * dh2)) : scalar_t(0)));
    });
}

// =====================================================================
//  Register with CUDA backend
// =====================================================================
TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("sl_advect_2d", &sl_advect_2d_cuda);
    m.impl("sl_advect_3d", &sl_advect_3d_cuda);
    m.impl("diffuse_add", &diffuse_add_cuda);
}

}  // namespace lilytorch_kernels
