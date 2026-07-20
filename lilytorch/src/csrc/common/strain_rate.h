// =====================================================================
//  strain_rate.h — single-source strain-rate magnitude |S̄| = sqrt(2·S_ij·S_ij)
//  on a MAC staggered grid, shared by the CUDA kernel (cuda/strain_rate.cu)
//  and the CPU twin (strain_rate_cpu.cpp).
//
//  Reproduces the legacy reference path exactly:
//      torch.gradient(..., edge_order=2)  +  _stag_to_cc
//
//  i.e. a 2nd-order central difference in the interior, a 3-point one-sided
//  difference on each boundary face, and — for the CROSS derivatives, which
//  land on a staggered location rather than the cell centre — an average of
//  the two neighbouring raw gradients along the staggering direction, with
//  the last cell replicating the pair from its inward neighbour.
//
//  Every intermediate stays in registers: the reference path materialised
//  4 (2-D) / 9 (3-D) full-grid gradient temporaries per call.
// =====================================================================

#pragma once

#if defined(__CUDACC__)
#define LT_SR_HD __host__ __device__ __forceinline__
#else
#define LT_SR_HD inline
#include <cmath>
#endif

namespace lilytorch_kernels {

// 1-D gradient of `f` along a dim with stride `s`, at flat offset `idx`.
// `i` is the index ALONG THAT DIM and `N` its extent, selecting the
// edge_order=2 one-sided stencil on the two boundary planes.
template <typename scalar_t>
LT_SR_HD scalar_t lt_grad1d(const scalar_t* f, int64_t idx, int64_t s,
                            int i, int N, scalar_t inv_2h) {
    if (i == 0) {
        return (-scalar_t(3) * f[idx]
                + scalar_t(4) * f[idx + s]
                - f[idx + 2 * s]) * inv_2h;
    } else if (i == N - 1) {
        return (scalar_t(3) * f[idx]
                - scalar_t(4) * f[idx - s]
                + f[idx - 2 * s]) * inv_2h;
    }
    return (f[idx + s] - f[idx - s]) * inv_2h;
}

// Cross derivative: gradient of `f` along dim `g` (stride `sg`, index `ig`,
// extent `Ng`), averaged to the cell centre along the staggering dim `a`
// (stride `sa`, index `ia`, extent `Na`).
//
// The reference `_stag_to_cc` averages consecutive pairs along `a` and
// replicates the final value, so cell `ia` averages the raw gradients at
// (ia, ia+1) — except at the last cell, which reuses the (Na-2, Na-1) pair.
// Both cases are the pair rooted at `lo`.  Shifting along `a` never changes
// `ig`, so the one-sided/central branch inside lt_grad1d is unaffected.
template <typename scalar_t>
LT_SR_HD scalar_t lt_grad1d_cc(const scalar_t* f, int64_t idx,
                               int64_t sg, int ig, int Ng,
                               int64_t sa, int ia, int Na,
                               scalar_t inv_2h) {
    const int64_t lo = (ia < Na - 1) ? idx : (idx - sa);
    return scalar_t(0.5) * (lt_grad1d(f, lo,      sg, ig, Ng, inv_2h)
                          + lt_grad1d(f, lo + sa, sg, ig, Ng, inv_2h));
}

// |S̄| at one grid point.  `w`/`sw_*` are ignored (and may be null) when
// DIM == 2; k is then 0 and Nz is 1.
template <typename scalar_t, int DIM>
LT_SR_HD scalar_t lt_strain_rate_at(
        const scalar_t* u, const scalar_t* v, const scalar_t* w,
        int i, int j, int k, int Nx, int Ny, int Nz,
        int64_t su_x, int64_t su_y, int64_t su_z,
        int64_t sv_x, int64_t sv_y, int64_t sv_z,
        int64_t sw_x, int64_t sw_y, int64_t sw_z,
        scalar_t inv_2h) {

    const int64_t idx_u = i * su_x + j * su_y + k * su_z;
    const int64_t idx_v = i * sv_x + j * sv_y + k * sv_z;

    const scalar_t half = scalar_t(0.5);

    const scalar_t dudx = lt_grad1d(u, idx_u, su_x, i, Nx, inv_2h);
    const scalar_t dvdy = lt_grad1d(v, idx_v, sv_y, j, Ny, inv_2h);
    // du/dy lives on x-faces → average to CC along x; dv/dx on y-faces → along y.
    const scalar_t dudy = lt_grad1d_cc(u, idx_u, su_y, j, Ny, su_x, i, Nx, inv_2h);
    const scalar_t dvdx = lt_grad1d_cc(v, idx_v, sv_x, i, Nx, sv_y, j, Ny, inv_2h);

    // S_ij S_ij = S11² + S22² + 2·S12²,  S12 = ½(dudy + dvdx)
    scalar_t S2 = dudx * dudx + dvdy * dvdy
                + half * (dudy + dvdx) * (dudy + dvdx);

    if (DIM == 3) {
        const int64_t idx_w = i * sw_x + j * sw_y + k * sw_z;

        const scalar_t dwdz = lt_grad1d(w, idx_w, sw_z, k, Nz, inv_2h);
        const scalar_t dudz = lt_grad1d_cc(u, idx_u, su_z, k, Nz, su_x, i, Nx, inv_2h);
        const scalar_t dwdx = lt_grad1d_cc(w, idx_w, sw_x, i, Nx, sw_z, k, Nz, inv_2h);
        const scalar_t dvdz = lt_grad1d_cc(v, idx_v, sv_z, k, Nz, sv_y, j, Ny, inv_2h);
        const scalar_t dwdy = lt_grad1d_cc(w, idx_w, sw_y, j, Ny, sw_z, k, Nz, inv_2h);

        // Accumulated as three separate adds to match the reference summation
        // order (fp32 is not associative).
        S2 += dwdz * dwdz;
        S2 += half * (dudz + dwdx) * (dudz + dwdx);
        S2 += half * (dvdz + dwdy) * (dvdz + dwdy);
    }

#if defined(__CUDACC__)
    return sqrt(scalar_t(2) * S2);      // CUDA device float/double overloads
#else
    return std::sqrt(scalar_t(2) * S2);
#endif
}

}  // namespace lilytorch_kernels
