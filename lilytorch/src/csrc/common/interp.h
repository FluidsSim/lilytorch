// =====================================================================
//  interp.h — Unified interpolation / sampling functions for uniform
//  body grids.  Shared by all CUDA kernels (cuda/*.cu) and their CPU
//  twins (*.cpp).
//
//  Families unified here (formerly duplicated across 14 translation units):
//
//  1.  *_sample_uniform*  (2‑D bilinear/biquadratic, 3‑D trilinear/triquadratic)
//      + sdf_sample_dispatch* dispatchers.
//      Was copy‑pasted into: streaming.cu, interp.cu, bcs.cu,
//      eulerian_forces.cu (CUDA) + ops_2d.cpp, ops_3d.cpp, streaming.cpp (CPU).
//
//  2.  lf_*  (lagrangian‑forces samplers) — algorithmically identical to
//      *_sample_uniform*; existed under a different prefix solely to avoid
//      cross‑TU symbol collisions.  The translation units that need them
//      now include this header and keep thin lf_* wrappers that delegate
//      to the unified functions.
//
//  3.  *_sample_off_*  (semi‑Lagrangian advection, off‑grid indexing).
//      Was duplicated across sl_advect.cu + sl_advect_cpu.cpp.
//      Uses an explicit flat‑buffer offset (F_off) instead of a stride‑based
//      base pointer, but the interpolation weights are identical.
//
//  All functions are ``__host__ __device__`` when compiled with nvcc,
//  plain ``inline`` otherwise, so a single source serves both back‑ends.
// =====================================================================
#pragma once

#include <algorithm>
#include <cmath>

#if defined(__CUDACC__)
#define LT_INTERP_FN __host__ __device__ __forceinline__
#else
#define LT_INTERP_FN inline
#endif

namespace lilytorch_kernels {

// =====================================================================
//  2‑D bilinear sample — uniform body grid
// =====================================================================
template <typename scalar_t>
LT_INTERP_FN scalar_t bilinear_sample_uniform_2d(
    const scalar_t* __restrict__ F,
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
//  2‑D biquadratic sample — uniform body grid
//
//  Lagrange interpolation on a 3×3 stencil [ix‑1, ix, ix+1] × [iy‑1, iy, iy+1]
//  with the same lower‑bracketing convention as the bilinear sampler.
//  Falls back to bilinear when the stencil would cross the grid border.
// =====================================================================
template <typename scalar_t>
LT_INTERP_FN scalar_t biquadratic_sample_uniform_2d(
    const scalar_t* __restrict__ F,
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

    // Identical accumulation order as the former per‑file copies so that
    // FMA contraction produces the same rounding on every platform.
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
//  3‑D trilinear sample — uniform body grid
//
//  Body SDF tables are uniform‑grid by construction in BDIM, so corner
//  weights reduce to (1‑frac, frac) per axis — this saves the slow
//  ``floor`` calls, the 24 axis‑table loads per cell across the 4 face
//  samples, and the trailing ``* inv_vol`` multiply.
// =====================================================================
template <typename scalar_t>
LT_INTERP_FN scalar_t trilinear_sample_uniform(
    const scalar_t* __restrict__ F,
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

    const scalar_t fx = tx - (scalar_t)ix;
    const scalar_t fy = ty - (scalar_t)iy;
    const scalar_t fz = tz - (scalar_t)iz;
    const scalar_t wx0 = (scalar_t)1 - fx, wx1 = fx;
    const scalar_t wy0 = (scalar_t)1 - fy, wy1 = fy;
    const scalar_t wz0 = (scalar_t)1 - fz, wz1 = fz;

    const int s2   = Mz;
    const int s1   = My * Mz;
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

// =====================================================================
//  3‑D triquadratic sample — uniform body grid
//
//  Lagrange interpolation on a 3×3×3 stencil [ix‑1, ix, ix+1]³ with the
//  same lower‑bracketing convention as ``trilinear_sample_uniform``.
//  Falls back to trilinear when the stencil would cross the grid border.
// =====================================================================
template <typename scalar_t>
LT_INTERP_FN scalar_t triquadratic_sample_uniform(
    const scalar_t* __restrict__ F,
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
//  2‑D dispatch helper
// =====================================================================
template <typename scalar_t>
LT_INTERP_FN scalar_t sdf_sample_dispatch_2d(
    const int interp_method,
    const scalar_t* __restrict__ F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    if (interp_method == 1) {
        return biquadratic_sample_uniform_2d<scalar_t>(
            F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
    }
    return bilinear_sample_uniform_2d<scalar_t>(
        F, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
}

// =====================================================================
//  3‑D dispatch helper
// =====================================================================
template <typename scalar_t>
LT_INTERP_FN scalar_t sdf_sample_dispatch(
    const int interp_method,
    const scalar_t* __restrict__ F,
    const int Mx, const int My, const int Mz,
    const scalar_t bx0, const scalar_t by0, const scalar_t bz0,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
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
//  Off‑grid indexing family (semi‑Lagrangian advection)
//
//  These differ from the *_sample_uniform* family in that the base
//  pointer is passed as an explicit flat offset ``F_off`` rather than
//  having the caller pre‑offset the pointer.  This matches the calling
//  convention of the semi‑Lagrangian kernels where each thread reads
//  from a different scalar field stored in a packed multi‑field buffer.
// =====================================================================

// -------------------------------------------------------------------
//  2‑D bilinear (off‑grid) — used as fallback by biquadratic_off_2d
// -------------------------------------------------------------------
template <typename scalar_t>
LT_INTERP_FN scalar_t bilinear_sample_off_2d(
    const scalar_t* __restrict__ F,
    int F_off, int Mx, int My,
    scalar_t bx0, scalar_t by0,
    scalar_t inv_dx, scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    scalar_t tx = std::max((scalar_t)0, std::min((xq - bx0) * inv_dx, (scalar_t)(Mx - 1)));
    scalar_t ty = std::max((scalar_t)0, std::min((yq - by0) * inv_dy, (scalar_t)(My - 1)));

    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;

    scalar_t fx = tx - (scalar_t)ix;
    scalar_t fy = ty - (scalar_t)iy;
    scalar_t wx0 = (scalar_t)1 - fx;
    scalar_t wy0 = (scalar_t)1 - fy;

    int s1 = My;
    int base = F_off + ix * s1 + iy;
    return wx0 * (wy0 * F[base] + fy * F[base + 1])
         +  fx * (wy0 * F[base + s1] + fy * F[base + s1 + 1]);
}

// -------------------------------------------------------------------
//  2‑D biquadratic (off‑grid)
// -------------------------------------------------------------------
template <typename scalar_t>
LT_INTERP_FN scalar_t biquadratic_sample_off_2d(
    const scalar_t* __restrict__ F,
    int F_off, int Mx, int My,
    scalar_t bx0, scalar_t by0,
    scalar_t inv_dx, scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    scalar_t tx = std::max((scalar_t)0, std::min((xq - bx0) * inv_dx, (scalar_t)(Mx - 1)));
    scalar_t ty = std::max((scalar_t)0, std::min((yq - by0) * inv_dy, (scalar_t)(My - 1)));

    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;

    // Fall back to bilinear at the table border.
    if (ix < 1 || iy < 1 || Mx < 3 || My < 3) {
        return bilinear_sample_off_2d<scalar_t>(
            F, F_off, Mx, My, bx0, by0, inv_dx, inv_dy, xq, yq);
    }

    scalar_t fx = tx - (scalar_t)ix;
    scalar_t fy = ty - (scalar_t)iy;

    scalar_t half = (scalar_t)0.5;
    scalar_t one  = (scalar_t)1;
    scalar_t wxm = half * fx * (fx - one);
    scalar_t wx0 = one - fx * fx;
    scalar_t wxp = half * fx * (fx + one);
    scalar_t wym = half * fy * (fy - one);
    scalar_t wy0 = one - fy * fy;
    scalar_t wyp = half * fy * (fy + one);

    int s1 = My;
    int base = F_off + (ix - 1) * s1 + (iy - 1);

    scalar_t col0 = wym * F[base] + wy0 * F[base + 1] + wyp * F[base + 2];
    int b1 = base + s1;
    scalar_t col1 = wym * F[b1] + wy0 * F[b1 + 1] + wyp * F[b1 + 2];
    int b2 = base + 2 * s1;
    scalar_t col2 = wym * F[b2] + wy0 * F[b2 + 1] + wyp * F[b2 + 2];

    return wxm * col0 + wx0 * col1 + wxp * col2;
}

// -------------------------------------------------------------------
//  3‑D trilinear (off‑grid) — used as fallback by triquadratic_off
// -------------------------------------------------------------------
template <typename scalar_t>
LT_INTERP_FN scalar_t trilinear_sample_off_3d(
    const scalar_t* __restrict__ F,
    int F_off, int Mx, int My, int Mz,
    scalar_t bx0, scalar_t by0, scalar_t bz0,
    scalar_t inv_dx, scalar_t inv_dy, scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    scalar_t zero = (scalar_t)0, one = (scalar_t)1;
    scalar_t tx = std::max(zero, std::min((xq - bx0) * inv_dx, (scalar_t)(Mx - 1)));
    scalar_t ty = std::max(zero, std::min((yq - by0) * inv_dy, (scalar_t)(My - 1)));
    scalar_t tz = std::max(zero, std::min((zq - bz0) * inv_dz, (scalar_t)(Mz - 1)));

    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;
    int iz = (int)tz; if (iz > Mz - 2) iz = Mz - 2;

    scalar_t fx = tx - (scalar_t)ix;
    scalar_t fy = ty - (scalar_t)iy;
    scalar_t fz = tz - (scalar_t)iz;
    scalar_t wx0 = one - fx, wy0 = one - fy, wz0 = one - fz;

    int s2 = Mz;
    int s1 = My * Mz;
    int base = F_off + ix * s1 + iy * s2 + iz;

    return wx0 * (wy0 * (wz0 * F[base]                 + fz * F[base + 1]) +
                  fy  * (wz0 * F[base + s2]             + fz * F[base + s2 + 1]))
         + fx  * (wy0 * (wz0 * F[base + s1]             + fz * F[base + s1 + 1]) +
                  fy  * (wz0 * F[base + s1 + s2]        + fz * F[base + s1 + s2 + 1]));
}

// -------------------------------------------------------------------
//  3‑D triquadratic (off‑grid)
// -------------------------------------------------------------------
template <typename scalar_t>
LT_INTERP_FN scalar_t triquadratic_sample_off_3d(
    const scalar_t* __restrict__ F,
    int F_off, int Mx, int My, int Mz,
    scalar_t bx0, scalar_t by0, scalar_t bz0,
    scalar_t inv_dx, scalar_t inv_dy, scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    scalar_t zero = (scalar_t)0, one = (scalar_t)1;
    scalar_t tx = std::max(zero, std::min((xq - bx0) * inv_dx, (scalar_t)(Mx - 1)));
    scalar_t ty = std::max(zero, std::min((yq - by0) * inv_dy, (scalar_t)(My - 1)));
    scalar_t tz = std::max(zero, std::min((zq - bz0) * inv_dz, (scalar_t)(Mz - 1)));

    int ix = (int)tx; if (ix > Mx - 2) ix = Mx - 2;
    int iy = (int)ty; if (iy > My - 2) iy = My - 2;
    int iz = (int)tz; if (iz > Mz - 2) iz = Mz - 2;

    // Fall back to trilinear at the table border.
    if (ix < 1 || iy < 1 || iz < 1 || Mx < 3 || My < 3 || Mz < 3) {
        return trilinear_sample_off_3d<scalar_t>(
            F, F_off, Mx, My, Mz, bx0, by0, bz0,
            inv_dx, inv_dy, inv_dz, xq, yq, zq);
    }

    scalar_t fx = tx - (scalar_t)ix;
    scalar_t fy = ty - (scalar_t)iy;
    scalar_t fz = tz - (scalar_t)iz;

    scalar_t half = (scalar_t)0.5;
    scalar_t wxm = half * fx * (fx - one), wx0 = one - fx * fx, wxp = half * fx * (fx + one);
    scalar_t wym = half * fy * (fy - one), wy0 = one - fy * fy, wyp = half * fy * (fy + one);
    scalar_t wzm = half * fz * (fz - one), wz0 = one - fz * fz, wzp = half * fz * (fz + one);

    int s2 = Mz, s1 = My * Mz;
    int base = F_off + (ix - 1) * s1 + (iy - 1) * s2 + (iz - 1);

    scalar_t out = (scalar_t)0;
    for (int dx = 0; dx < 3; ++dx) {
        scalar_t wx = wxm;
        if (dx == 1) wx = wx0;
        if (dx == 2) wx = wxp;
        int b0 = base + dx * s1;
        scalar_t plane = (scalar_t)0;
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

}  // namespace lilytorch_kernels
