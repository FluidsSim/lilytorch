// =====================================================================
//  multigrid_cpu.cpp — CPU (at::parallel_for) twins for the CUDA-only
//  multigrid / advection / cvof / Poisson-solve kernels.
//
//  Replaces the old stub-only rbgs_cpu.cpp.  Registers CPU dispatch for
//  every op that previously had no CPU backend (or had only a TORCH_CHECK
//  stub), satisfying ground rule 4.
//
//  Float dispatch covers float32 / float64; half precision is skipped on
//  CPU.  All hot loops use ``at::parallel_for`` (PyTorch's intra-op thread
//  pool) instead of raw OpenMP pragmas — see streaming_sdf_cpu.cpp for the
//  rationale.
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
#include <tuple>
#include <vector>

#include "common/poisson_gauge.h"

namespace lilytorch_kernels {

// =====================================================================
//  Neumann BC helpers (mirror the CUDA fused versions)
// =====================================================================

template <typename scalar_t>
static inline void apply_neumann_bc_2d_cpu(scalar_t* p, int Nx, int Ny) {
    // Copy interior-adjacent rows into ghost rows on all 4 faces.
    // Dim order: x-faces then y-faces so edge/corner ghosts get filled.
    const int stride = Ny + 2;
    // x-faces
    at::parallel_for(0, Ny, 1, [&](int64_t start, int64_t end) {
        for (int64_t j = start; j < end; ++j) {
            const int jj = (int)j + 1;
            p[jj]                     = p[stride + jj];        // p[0,j]   = p[1,j]
            p[(Nx+1)*stride + jj]    = p[Nx*stride + jj];      // p[Nx+1,j]= p[Nx,j]
        }
    });
    // y-faces
    at::parallel_for(0, Nx, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            const int ii = (int)i + 1;
            const int base = ii * stride;
            p[base]          = p[base + 1];          // p[i,0]    = p[i,1]
            p[base + Ny + 1] = p[base + Ny];         // p[i,Ny+1] = p[i,Ny]
        }
    });
}

template <typename scalar_t>
static inline void apply_neumann_bc_3d_cpu(scalar_t* p, int Nx, int Ny, int Nz) {
    const int si = (Ny + 2) * (Nz + 2);
    const int sj = Nz + 2;
    // x-faces
    at::parallel_for(0, Ny * Nz, 1, [&](int64_t start, int64_t end) {
        for (int64_t idx = start; idx < end; ++idx) {
            const int j = (int)(idx / Nz) + 1;
            const int k = (int)(idx % Nz) + 1;
            const int jk = j * sj + k;
            p[jk]              = p[si + jk];
            p[(Nx+1)*si + jk]  = p[Nx*si + jk];
        }
    });
    // y-faces
    at::parallel_for(0, Nx * Nz, 1, [&](int64_t start, int64_t end) {
        for (int64_t idx = start; idx < end; ++idx) {
            const int i = (int)(idx / Nz) + 1;
            const int k = (int)(idx % Nz) + 1;
            const int base = i * si + k;
            p[base]              = p[base + sj];
            p[base + (Ny+1)*sj]  = p[base + Ny*sj];
        }
    });
    // z-faces
    at::parallel_for(0, Nx * Ny, 1, [&](int64_t start, int64_t end) {
        for (int64_t idx = start; idx < end; ++idx) {
            const int i = (int)(idx / Ny) + 1;
            const int j = (int)(idx % Ny) + 1;
            const int base = i * si + j * sj;
            p[base]          = p[base + 1];
            p[base + Nz + 1] = p[base + Nz];
        }
    });
}

// =====================================================================
//  RBGS smoother (2-D) — serial red-black Gauss-Seidel, interior only.
//  The CUDA kernel is tiled shared-memory; the CPU twin is a straightforward
//  serial loop over red then black, repeated nsmoothing times.
// =====================================================================

template <typename scalar_t>
// ``reverse`` swaps the half-sweep order (red-black -> black-red).  Gauss-
// Seidel is not self-adjoint, so a V-cycle that pre- and post-smooths in the
// same colour order is NON-SYMMETRIC and invalid as a CG preconditioner; the
// post-smooth of a variational V-cycle passes reverse=true to make the pair
// A-adjoint.  See lilytorch/tests/check_poisson_symmetry.py.
static void rbgs_sweep_2d_cpu_impl(
        scalar_t* p, const scalar_t* f,
        const scalar_t* cp0, const scalar_t* cm0,
        const scalar_t* cp1, const scalar_t* cm1,
        int Nx, int Ny, scalar_t jcap_tol, int nsmoothing,
        bool reverse = false)
{
    const int stride_p = Ny + 2;
    const int stride_c = Ny;
    const int c_first  = reverse ? 1 : 0;
    const int c_second = reverse ? 0 : 1;

    // Neumann BC before sweeps
    apply_neumann_bc_2d_cpu(p, Nx, Ny);

    for (int s = 0; s < nsmoothing; ++s) {
        // First half-sweep: red, or black when reversed
        at::parallel_for(0, Nx, 1, [&](int64_t istart, int64_t iend) {
            for (int64_t i = istart; i < iend; ++i) {
                const int pi = (int)i + 1;
                const scalar_t* cp0_row = cp0 + (int)i * stride_c;
                const scalar_t* cm0_row = cm0 + (int)i * stride_c;
                const scalar_t* cp1_row = cp1 + (int)i * stride_c;
                const scalar_t* cm1_row = cm1 + (int)i * stride_c;
                const scalar_t* f_row   = f   + (int)i * stride_c;
                scalar_t* p_row = p + pi * stride_p;
                for (int j = 0; j < Ny; ++j) {
                    if ((((int)i + j) & 1) != c_first) continue;
                    const scalar_t J = cp0_row[j] + cm0_row[j] + cp1_row[j] + cm1_row[j];
                    if (J < jcap_tol && J > -jcap_tol) continue;
                    const int pj = j + 1;
                    const scalar_t sum =
                        cp0_row[j] * p_row[pj + stride_p]   // i+1
                      + cm0_row[j] * p_row[pj - stride_p]   // i-1
                      + cp1_row[j] * p_row[pj + 1]          // j+1
                      + cm1_row[j] * p_row[pj - 1];         // j-1
                    p_row[pj] = (-f_row[j] + sum) / J;
                }
            }
        });

        // Second half-sweep: black, or red when reversed
        at::parallel_for(0, Nx, 1, [&](int64_t istart, int64_t iend) {
            for (int64_t i = istart; i < iend; ++i) {
                const int pi = (int)i + 1;
                const scalar_t* cp0_row = cp0 + (int)i * stride_c;
                const scalar_t* cm0_row = cm0 + (int)i * stride_c;
                const scalar_t* cp1_row = cp1 + (int)i * stride_c;
                const scalar_t* cm1_row = cm1 + (int)i * stride_c;
                const scalar_t* f_row   = f   + (int)i * stride_c;
                scalar_t* p_row = p + pi * stride_p;
                for (int j = 0; j < Ny; ++j) {
                    if ((((int)i + j) & 1) != c_second) continue;
                    const scalar_t J = cp0_row[j] + cm0_row[j] + cp1_row[j] + cm1_row[j];
                    if (J < jcap_tol && J > -jcap_tol) continue;
                    const int pj = j + 1;
                    const scalar_t sum =
                        cp0_row[j] * p_row[pj + stride_p]
                      + cm0_row[j] * p_row[pj - stride_p]
                      + cp1_row[j] * p_row[pj + 1]
                      + cm1_row[j] * p_row[pj - 1];
                    p_row[pj] = (-f_row[j] + sum) / J;
                }
            }
        });
    }

    // Neumann BC after sweeps
    apply_neumann_bc_2d_cpu(p, Nx, Ny);
}

// =====================================================================
//  RBGS smoother (3-D) — thread-per-cell, red then black half-sweeps.
// =====================================================================

template <typename scalar_t>
static void rbgs_sweep_3d_cpu_impl(
        scalar_t* p, const scalar_t* f,
        const scalar_t* cp0, const scalar_t* cm0,
        const scalar_t* cp1, const scalar_t* cm1,
        const scalar_t* cp2, const scalar_t* cm2,
        int Nx, int Ny, int Nz, scalar_t jcap_tol, int nsmoothing,
        bool reverse = false)
{
    const int si = (Ny + 2) * (Nz + 2);
    const int sj = Nz + 2;
    // See the note on ``reverse`` above rbgs_sweep_2d_cpu_impl.
    const int c_first  = reverse ? 1 : 0;
    const int c_second = reverse ? 0 : 1;

    apply_neumann_bc_3d_cpu(p, Nx, Ny, Nz);

    for (int s = 0; s < nsmoothing; ++s) {
        // First half-sweep: red, or black when reversed
        at::parallel_for(0, Nx, 1, [&](int64_t istart, int64_t iend) {
            for (int64_t i = istart; i < iend; ++i) {
                const int pi = (int)i + 1;
                for (int j = 0; j < Ny; ++j) {
                    const int pj = j + 1;
                    const int cbase = (int)i * (Ny * Nz) + j * Nz;
                    const int pbase = pi * si + pj * sj + 1;
                    for (int k = 0; k < Nz; ++k) {
                        if ((((int)i + j + k) & 1) != c_first) continue;
                        const scalar_t J = cp0[cbase+k] + cm0[cbase+k]
                                         + cp1[cbase+k] + cm1[cbase+k]
                                         + cp2[cbase+k] + cm2[cbase+k];
                        if (J < jcap_tol && J > -jcap_tol) continue;
                        const scalar_t sum =
                              cp0[cbase+k] * p[pbase + k + si]
                            + cm0[cbase+k] * p[pbase + k - si]
                            + cp1[cbase+k] * p[pbase + k + sj]
                            + cm1[cbase+k] * p[pbase + k - sj]
                            + cp2[cbase+k] * p[pbase + k + 1]
                            + cm2[cbase+k] * p[pbase + k - 1];
                        p[pbase + k] = (-f[cbase+k] + sum) / J;
                    }
                }
            }
        });
        // Refresh the ghost ring so the black cells read up-to-date Neumann
        // mirrors of the red cells just written (the CUDA twin does the same
        // between half-sweeps; skipping it desynchronises the two backends).
        apply_neumann_bc_3d_cpu(p, Nx, Ny, Nz);

        // Second half-sweep: black, or red when reversed
        at::parallel_for(0, Nx, 1, [&](int64_t istart, int64_t iend) {
            for (int64_t i = istart; i < iend; ++i) {
                const int pi = (int)i + 1;
                for (int j = 0; j < Ny; ++j) {
                    const int pj = j + 1;
                    const int cbase = (int)i * (Ny * Nz) + j * Nz;
                    const int pbase = pi * si + pj * sj + 1;
                    for (int k = 0; k < Nz; ++k) {
                        if ((((int)i + j + k) & 1) != c_second) continue;
                        const scalar_t J = cp0[cbase+k] + cm0[cbase+k]
                                         + cp1[cbase+k] + cm1[cbase+k]
                                         + cp2[cbase+k] + cm2[cbase+k];
                        if (J < jcap_tol && J > -jcap_tol) continue;
                        const scalar_t sum =
                              cp0[cbase+k] * p[pbase + k + si]
                            + cm0[cbase+k] * p[pbase + k - si]
                            + cp1[cbase+k] * p[pbase + k + sj]
                            + cm1[cbase+k] * p[pbase + k - sj]
                            + cp2[cbase+k] * p[pbase + k + 1]
                            + cm2[cbase+k] * p[pbase + k - 1];
                        p[pbase + k] = (-f[cbase+k] + sum) / J;
                    }
                }
            }
        });
        apply_neumann_bc_3d_cpu(p, Nx, Ny, Nz);
    }
}

// =====================================================================
//  Weighted Jacobi smoother (2-D)
// =====================================================================

template <typename scalar_t>
static void jacobi_sweep_2d_cpu_impl(
        scalar_t* p, const scalar_t* f,
        const scalar_t* cp0, const scalar_t* cm0,
        const scalar_t* cp1, const scalar_t* cm1,
        int Nx, int Ny, scalar_t jcap_tol, scalar_t w, int nsmoothing)
{
    const int stride_p = Ny + 2;
    const int stride_c = Ny;

    apply_neumann_bc_2d_cpu(p, Nx, Ny);

    // Use a temporary buffer for synchronous Jacobi (Gauss-Jacobi)
    std::vector<scalar_t> tmp((size_t)(Nx + 2) * (Ny + 2));
    scalar_t* p_tmp = tmp.data();

    for (int s = 0; s < nsmoothing; ++s) {
        // Copy p → p_tmp (ghost-padded)
        at::parallel_for(0, Nx + 2, 1, [&](int64_t istart, int64_t iend) {
            for (int64_t i = istart; i < iend; ++i) {
                const int row = (int)i * stride_p;
                for (int j = 0; j < Ny + 2; ++j)
                    p_tmp[row + j] = p[row + j];
            }
        });

        // Jacobi sweep: read from p_tmp, write to p
        at::parallel_for(0, Nx, 1, [&](int64_t istart, int64_t iend) {
            for (int64_t i = istart; i < iend; ++i) {
                const int pi = (int)i + 1;
                const scalar_t* cp0_row = cp0 + (int)i * stride_c;
                const scalar_t* cm0_row = cm0 + (int)i * stride_c;
                const scalar_t* cp1_row = cp1 + (int)i * stride_c;
                const scalar_t* cm1_row = cm1 + (int)i * stride_c;
                const scalar_t* f_row   = f   + (int)i * stride_c;
                const int p_row_base = pi * stride_p;
                for (int j = 0; j < Ny; ++j) {
                    const scalar_t J = cp0_row[j] + cm0_row[j] + cp1_row[j] + cm1_row[j];
                    if (J < jcap_tol && J > -jcap_tol) continue;
                    const int pj = j + 1;
                    const scalar_t sum =
                        cp0_row[j] * p_tmp[p_row_base + pj + stride_p]
                      + cm0_row[j] * p_tmp[p_row_base + pj - stride_p]
                      + cp1_row[j] * p_tmp[p_row_base + pj + 1]
                      + cm1_row[j] * p_tmp[p_row_base + pj - 1];
                    const scalar_t p_new = (-f_row[j] + sum) / J;
                    p[p_row_base + pj] = w * p_new + ((scalar_t)1 - w) * p_tmp[p_row_base + pj];
                }
            }
        });

        apply_neumann_bc_2d_cpu(p, Nx, Ny);
    }
}

// =====================================================================
//  Weighted Jacobi smoother (3-D)
// =====================================================================

template <typename scalar_t>
static void jacobi_sweep_3d_cpu_impl(
        scalar_t* p, const scalar_t* f,
        const scalar_t* cp0, const scalar_t* cm0,
        const scalar_t* cp1, const scalar_t* cm1,
        const scalar_t* cp2, const scalar_t* cm2,
        int Nx, int Ny, int Nz, scalar_t jcap_tol, scalar_t w, int nsmoothing)
{
    const int si = (Ny + 2) * (Nz + 2);
    const int sj = Nz + 2;
    const size_t total = (size_t)(Nx + 2) * (Ny + 2) * (Nz + 2);

    apply_neumann_bc_3d_cpu(p, Nx, Ny, Nz);

    std::vector<scalar_t> tmp(total);
    scalar_t* p_tmp = tmp.data();

    for (int s = 0; s < nsmoothing; ++s) {
        // Copy p → p_tmp
        at::parallel_for(0, (int64_t)total, 1, [&](int64_t start, int64_t end) {
            for (int64_t idx = start; idx < end; ++idx)
                p_tmp[idx] = p[idx];
        });

        at::parallel_for(0, Nx, 1, [&](int64_t istart, int64_t iend) {
            for (int64_t i = istart; i < iend; ++i) {
                const int pi = (int)i + 1;
                for (int j = 0; j < Ny; ++j) {
                    const int pj = j + 1;
                    const int cbase = (int)i * (Ny * Nz) + j * Nz;
                    const int pbase = pi * si + pj * sj + 1;
                    for (int k = 0; k < Nz; ++k) {
                        const scalar_t J = cp0[cbase+k] + cm0[cbase+k]
                                         + cp1[cbase+k] + cm1[cbase+k]
                                         + cp2[cbase+k] + cm2[cbase+k];
                        if (J < jcap_tol && J > -jcap_tol) continue;
                        const int pk = k;
                        const scalar_t sum =
                              cp0[cbase+k] * p_tmp[pbase + pk + si]
                            + cm0[cbase+k] * p_tmp[pbase + pk - si]
                            + cp1[cbase+k] * p_tmp[pbase + pk + sj]
                            + cm1[cbase+k] * p_tmp[pbase + pk - sj]
                            + cp2[cbase+k] * p_tmp[pbase + pk + 1]
                            + cm2[cbase+k] * p_tmp[pbase + pk - 1];
                        const scalar_t p_new = (-f[cbase+k] + sum) / J;
                        p[pbase + pk] = w * p_new + ((scalar_t)1 - w) * p_tmp[pbase + pk];
                    }
                }
            }
        });

        apply_neumann_bc_3d_cpu(p, Nx, Ny, Nz);
    }
}

// =====================================================================
//  mg_residual (2-D / 3-D)
// =====================================================================

template <typename scalar_t>
static void mg_residual_2d_cpu_impl(
        const scalar_t* p, const scalar_t* f,
        const scalar_t* cp0, const scalar_t* cm0,
        const scalar_t* cp1, const scalar_t* cm1,
        int Nx, int Ny, scalar_t jcap_tol, scalar_t* r)
{
    const int stride_p = Ny + 2;
    const int stride_c = Ny;

    at::parallel_for(0, Nx, 1, [&](int64_t istart, int64_t iend) {
        for (int64_t i = istart; i < iend; ++i) {
            const int pi = (int)i + 1;
            const scalar_t* cp0_row = cp0 + (int)i * stride_c;
            const scalar_t* cm0_row = cm0 + (int)i * stride_c;
            const scalar_t* cp1_row = cp1 + (int)i * stride_c;
            const scalar_t* cm1_row = cm1 + (int)i * stride_c;
            const scalar_t* f_row   = f   + (int)i * stride_c;
            const int p_row_base = pi * stride_p;
            scalar_t* r_row = r + (int)i * stride_c;
            for (int j = 0; j < Ny; ++j) {
                const scalar_t J = cp0_row[j] + cm0_row[j] + cp1_row[j] + cm1_row[j];
                if (J < jcap_tol && J > -jcap_tol) {
                    r_row[j] = (scalar_t)0;
                    continue;
                }
                const int pj = j + 1;
                // Clamp ghosts (homogeneous-Neumann fold) — identical to CUDA smoother
                const scalar_t pc = p[p_row_base + pj];
                const scalar_t pip = (i < Nx - 1) ? p[p_row_base + pj + stride_p] : pc;
                const scalar_t pim = ((int)i > 0)      ? p[p_row_base + pj - stride_p] : pc;
                const scalar_t pjp = (j < Ny - 1) ? p[p_row_base + pj + 1] : pc;
                const scalar_t pjm = (j > 0)      ? p[p_row_base + pj - 1] : pc;
                const scalar_t s = cp0_row[j] * pip + cm0_row[j] * pim
                                 + cp1_row[j] * pjp + cm1_row[j] * pjm;
                r_row[j] = f_row[j] - s + J * pc;
            }
        }
    });
}

template <typename scalar_t>
static void mg_residual_3d_cpu_impl(
        const scalar_t* p, const scalar_t* f,
        const scalar_t* cp0, const scalar_t* cm0,
        const scalar_t* cp1, const scalar_t* cm1,
        const scalar_t* cp2, const scalar_t* cm2,
        int Nx, int Ny, int Nz, scalar_t jcap_tol, scalar_t* r)
{
    const int si = (Ny + 2) * (Nz + 2);
    const int sj = Nz + 2;

    at::parallel_for(0, Nx, 1, [&](int64_t istart, int64_t iend) {
        for (int64_t i = istart; i < iend; ++i) {
            const int pi = (int)i + 1;
            for (int j = 0; j < Ny; ++j) {
                const int pj = j + 1;
                const int cbase = (int)i * (Ny * Nz) + j * Nz;
                const int pbase = pi * si + pj * sj + 1;
                for (int k = 0; k < Nz; ++k) {
                    const scalar_t J = cp0[cbase+k] + cm0[cbase+k]
                                     + cp1[cbase+k] + cm1[cbase+k]
                                     + cp2[cbase+k] + cm2[cbase+k];
                    if (J < jcap_tol && J > -jcap_tol) {
                        r[cbase+k] = (scalar_t)0;
                        continue;
                    }
                    const scalar_t pc = p[pbase + k];
                    const scalar_t pip = (i < Nx - 1) ? p[pbase + k + si] : pc;
                    const scalar_t pim = ((int)i > 0) ? p[pbase + k - si] : pc;
                    const scalar_t pjp = (j < Ny - 1) ? p[pbase + k + sj] : pc;
                    const scalar_t pjm = (j > 0) ? p[pbase + k - sj] : pc;
                    const scalar_t pkp = (k < Nz - 1) ? p[pbase + k + 1] : pc;
                    const scalar_t pkm = (k > 0) ? p[pbase + k - 1] : pc;
                    const scalar_t s = cp0[cbase+k] * pip + cm0[cbase+k] * pim
                                     + cp1[cbase+k] * pjp + cm1[cbase+k] * pjm
                                     + cp2[cbase+k] * pkp + cm2[cbase+k] * pkm;
                    r[cbase+k] = f[cbase+k] - s + J * pc;
                }
            }
        }
    });
}

// =====================================================================
//  restrict_residual (2-D / 3-D) — sum-of-children, no scaling
// =====================================================================

template <typename scalar_t>
static void restrict_residual_2d_cpu_impl(
        const scalar_t* r, scalar_t* rc,
        int Nx_f, int Ny_f, int Nx_c, int Ny_c)
{
    at::parallel_for(0, Nx_c, 1, [&](int64_t Istart, int64_t Iend) {
        for (int64_t I = Istart; I < Iend; ++I) {
            for (int J = 0; J < Ny_c; ++J) {
                scalar_t s = (scalar_t)0;
                const int i0 = (int)I * 2, j0 = (int)J * 2;
                if (i0 < Nx_f && j0 < Ny_f)     s += r[i0 * Ny_f + j0];
                if (i0+1 < Nx_f && j0 < Ny_f)   s += r[(i0+1) * Ny_f + j0];
                if (i0 < Nx_f && j0+1 < Ny_f)   s += r[i0 * Ny_f + (j0+1)];
                if (i0+1 < Nx_f && j0+1 < Ny_f) s += r[(i0+1) * Ny_f + (j0+1)];
                rc[(int)I * Ny_c + (int)J] = s;
            }
        }
    });
}

template <typename scalar_t>
static void restrict_residual_3d_cpu_impl(
        const scalar_t* r, scalar_t* rc,
        int Nx_f, int Ny_f, int Nz_f,
        int Nx_c, int Ny_c, int Nz_c)
{
    const int sif = Ny_f * Nz_f, sjf = Nz_f;
    const int sic = Ny_c * Nz_c, sjc = Nz_c;

    at::parallel_for(0, Nx_c, 1, [&](int64_t Istart, int64_t Iend) {
        for (int64_t I = Istart; I < Iend; ++I) {
            const int i0 = (int)I * 2;
            for (int J = 0; J < Ny_c; ++J) {
                const int j0 = J * 2;
                for (int K = 0; K < Nz_c; ++K) {
                    const int k0 = K * 2;
                    scalar_t s = (scalar_t)0;
                    for (int di = 0; di < 2; ++di) {
                        int ii = i0 + di; if (ii >= Nx_f) continue;
                        for (int dj = 0; dj < 2; ++dj) {
                            int jj = j0 + dj; if (jj >= Ny_f) continue;
                            for (int dk = 0; dk < 2; ++dk) {
                                int kk = k0 + dk; if (kk >= Nz_f) continue;
                                s += r[ii * sif + jj * sjf + kk];
                            }
                        }
                    }
                    rc[(int)I * sic + J * sjc + K] = s;
                }
            }
        }
    });
}

// =====================================================================
//  restrict_face (2-D / 3-D)
// =====================================================================

template <typename scalar_t, int FACE_DIM>
static void restrict_face_2d_cpu_impl(
        const scalar_t* src, scalar_t* dst,
        int Nf0, int Nf1, int Nc0, int Nc1)
{
    if (FACE_DIM == 0) {
        // Face dim 0: src(Nf0+1, Nf1), dst(Nc0+1, Nc1)
        at::parallel_for(0, Nc0 + 1, 1, [&](int64_t Istart, int64_t Iend) {
            for (int64_t I = Istart; I < Iend; ++I) {
                for (int J = 0; J < Nc1; ++J) {
                    const int i0 = (int)I * 2, j0 = J * 2;
                    scalar_t s = (scalar_t)0;
                    if (i0 < Nf0 + 1) {
                        if (j0 < Nf1)       s += src[i0 * Nf1 + j0];
                        if (j0 + 1 < Nf1)   s += src[i0 * Nf1 + (j0+1)];
                    }
                    dst[(int)I * Nc1 + J] = s * (scalar_t)0.5;
                }
            }
        });
    } else {  // FACE_DIM == 1
        at::parallel_for(0, Nc0, 1, [&](int64_t Istart, int64_t Iend) {
            for (int64_t I = Istart; I < Iend; ++I) {
                for (int J = 0; J < Nc1 + 1; ++J) {
                    const int i0 = (int)I * 2, j0 = J * 2;
                    scalar_t s = (scalar_t)0;
                    if (j0 < Nf1 + 1) {
                        if (i0 < Nf0)       s += src[i0 * (Nf1 + 1) + j0];
                        if (i0 + 1 < Nf0)   s += src[(i0+1) * (Nf1 + 1) + j0];
                    }
                    dst[(int)I * (Nc1 + 1) + J] = s * (scalar_t)0.5;
                }
            }
        });
    }
}

template <typename scalar_t, int FACE_DIM>
static void restrict_face_3d_cpu_impl(
        const scalar_t* src, scalar_t* dst,
        int Nf0, int Nf1, int Nf2,
        int Nc0, int Nc1, int Nc2)
{
    int ni = (FACE_DIM == 0) ? 1 : 2;
    int nj = (FACE_DIM == 1) ? 1 : 2;
    int nk = (FACE_DIM == 2) ? 1 : 2;

    if (FACE_DIM == 0) {
        // src: (Nf0+1, Nf1, Nf2), dst: (Nc0+1, Nc1, Nc2)
        const int sif = Nf1 * Nf2, sjf = Nf2;
        const int sic = Nc1 * Nc2, sjc = Nc2;
        at::parallel_for(0, Nc0 + 1, 1, [&](int64_t Istart, int64_t Iend) {
            for (int64_t I = Istart; I < Iend; ++I) {
                for (int J = 0; J < Nc1; ++J) {
                    for (int K = 0; K < Nc2; ++K) {
                        const int i0 = (int)I * 2, j0 = J * 2, k0 = K * 2;
                        scalar_t s = (scalar_t)0;
                        if (i0 < Nf0 + 1) {
                            for (int dj = 0; dj < nj; ++dj) {
                                int jj = j0 + dj; if (jj >= Nf1) continue;
                                for (int dk = 0; dk < nk; ++dk) {
                                    int kk = k0 + dk; if (kk >= Nf2) continue;
                                    s += src[i0 * sif + jj * sjf + kk];
                                }
                            }
                        }
                        dst[(int)I * sic + J * sjc + K] = s * (scalar_t)0.5;
                    }
                }
            }
        });
    } else if (FACE_DIM == 1) {
        // Face dim 1: src(Nf0, Nf1+1, Nf2), dst(Nc0, Nc1+1, Nc2)
        const int sif_src = (Nf1 + 1) * Nf2, sjf_src = Nf2;
        const int sif_dst = (Nc1 + 1) * Nc2, sjf_dst = Nc2;
        at::parallel_for(0, Nc0, 1, [&](int64_t Istart, int64_t Iend) {
            for (int64_t I = Istart; I < Iend; ++I) {
                for (int J = 0; J < Nc1 + 1; ++J) {
                    for (int K = 0; K < Nc2; ++K) {
                        const int i0 = (int)I * 2, j0 = J * 2, k0 = K * 2;
                        scalar_t s = (scalar_t)0;
                        if (j0 < Nf1 + 1) {
                            for (int di = 0; di < ni; ++di) {
                                int ii = i0 + di; if (ii >= Nf0) continue;
                                for (int dk = 0; dk < nk; ++dk) {
                                    int kk = k0 + dk; if (kk >= Nf2) continue;
                                    s += src[ii * sif_src + j0 * sjf_src + kk];
                                }
                            }
                        }
                        dst[(int)I * sif_dst + J * sjf_dst + K] = s * (scalar_t)0.5;
                    }
                }
            }
        });
    } else {  // FACE_DIM == 2
        const int sif_src = Nf1 * (Nf2 + 1), sjf_src = Nf2 + 1;
        const int sif_dst = Nc1 * (Nc2 + 1), sjf_dst = Nc2 + 1;
        at::parallel_for(0, Nc0, 1, [&](int64_t Istart, int64_t Iend) {
            for (int64_t I = Istart; I < Iend; ++I) {
                for (int J = 0; J < Nc1; ++J) {
                    for (int K = 0; K < Nc2 + 1; ++K) {
                        const int i0 = (int)I * 2, j0 = J * 2, k0 = K * 2;
                        scalar_t s = (scalar_t)0;
                        if (k0 < Nf2 + 1) {
                            for (int di = 0; di < ni; ++di) {
                                int ii = i0 + di; if (ii >= Nf0) continue;
                                for (int dj = 0; dj < nj; ++dj) {
                                    int jj = j0 + dj; if (jj >= Nf1) continue;
                                    s += src[ii * sif_src + jj * sjf_src + k0];
                                }
                            }
                        }
                        dst[(int)I * sif_dst + J * sjf_dst + K] = s * (scalar_t)0.5;
                    }
                }
            }
        });
    }
}

// =====================================================================
//  prolongate_add (2-D / 3-D) — bilinear/trilinear, align_corners=False
// =====================================================================

// Helper: compute linear-interpolation weights (identical to CUDA linear_weights).
// Computed in scalar_t -- the user's dtype -- not double; see the note on the
// CUDA twin in cuda/multigrid_transfer.cu.
template <typename scalar_t>
static inline void linear_weights(
    int dst, int Nc, int Nf,
    int& il, int& ir, scalar_t& wl, scalar_t& wr)
{
    scalar_t src = ((scalar_t)dst + (scalar_t)0.5)
                 * ((scalar_t)Nc / (scalar_t)Nf) - (scalar_t)0.5;
    int sf = (int)std::floor(src);
    il = sf; ir = il + 1;
    if (il < 0) il = 0;
    if (ir < 0) ir = 0;
    if (il > Nc - 1) il = Nc - 1;
    if (ir > Nc - 1) ir = Nc - 1;
    scalar_t w = src - (scalar_t)sf;
    if (src <= (scalar_t)0)       { wl = (scalar_t)1; wr = (scalar_t)0; }
    else if (src >= (scalar_t)Nc - (scalar_t)1) { wl = (scalar_t)0; wr = (scalar_t)1; }
    else                  { wl = (scalar_t)1 - w; wr = w; }
}

template <typename scalar_t>
static void prolongate_add_2d_cpu_impl(
        const scalar_t* ec, scalar_t* p,
        int Nx_c, int Ny_c, int Nx_f, int Ny_f)
{
    const int sec = Ny_c + 2;  // stride in ec (ghost-padded)
    const int sp  = Ny_f + 2;  // stride in p  (ghost-padded)

    at::parallel_for(0, Nx_f, 1, [&](int64_t istart, int64_t iend) {
        for (int64_t i = istart; i < iend; ++i) {
            int il, ir; scalar_t wil, wir;
            linear_weights<scalar_t>((int)i, Nx_c, Nx_f, il, ir, wil, wir);
            const int i_p = (int)i + 1;
            for (int j = 0; j < Ny_f; ++j) {
                int jl, jr; scalar_t wjl, wjr;
                linear_weights<scalar_t>(j, Ny_c, Ny_f, jl, jr, wjl, wjr);
                const int j_p = j + 1;
                // ec indices in ghost-padded array: +1 for ghost offset
                const int e_ll = (il + 1) * sec + (jl + 1);
                const int e_lr = (il + 1) * sec + (jr + 1);
                const int e_rl = (ir + 1) * sec + (jl + 1);
                const int e_rr = (ir + 1) * sec + (jr + 1);
                scalar_t interp = wil * wjl * ec[e_ll]
                                + wil * wjr * ec[e_lr]
                                + wir * wjl * ec[e_rl]
                                + wir * wjr * ec[e_rr];
                p[i_p * sp + j_p] += interp;
            }
        }
    });
}

template <typename scalar_t>
static void prolongate_add_3d_cpu_impl(
        const scalar_t* ec, scalar_t* p,
        int Nx_c, int Ny_c, int Nz_c,
        int Nx_f, int Ny_f, int Nz_f)
{
    const int sec = (Ny_c + 2) * (Nz_c + 2);
    const int sjc = Nz_c + 2;
    const int sp  = (Ny_f + 2) * (Nz_f + 2);
    const int sjp = Nz_f + 2;

    at::parallel_for(0, Nx_f, 1, [&](int64_t istart, int64_t iend) {
        for (int64_t i = istart; i < iend; ++i) {
            int il, ir; scalar_t wil, wir;
            linear_weights<scalar_t>((int)i, Nx_c, Nx_f, il, ir, wil, wir);
            const int i_p = (int)i + 1;
            for (int j = 0; j < Ny_f; ++j) {
                int jl, jr; scalar_t wjl, wjr;
                linear_weights<scalar_t>(j, Ny_c, Ny_f, jl, jr, wjl, wjr);
                const int j_p = j + 1;
                for (int k = 0; k < Nz_f; ++k) {
                    int kl, kr; scalar_t wkl, wkr;
                    linear_weights<scalar_t>(k, Nz_c, Nz_f, kl, kr, wkl, wkr);
                    const int k_p = k + 1;
                    const int e_lll = (il+1)*sec + (jl+1)*sjc + (kl+1);
                    const int e_llr = (il+1)*sec + (jl+1)*sjc + (kr+1);
                    const int e_lrl = (il+1)*sec + (jr+1)*sjc + (kl+1);
                    const int e_lrr = (il+1)*sec + (jr+1)*sjc + (kr+1);
                    const int e_rll = (ir+1)*sec + (jl+1)*sjc + (kl+1);
                    const int e_rlr = (ir+1)*sec + (jl+1)*sjc + (kr+1);
                    const int e_rrl = (ir+1)*sec + (jr+1)*sjc + (kl+1);
                    const int e_rrr = (ir+1)*sec + (jr+1)*sjc + (kr+1);
                    scalar_t interp = wil * wjl * wkl * ec[e_lll]
                                    + wil * wjl * wkr * ec[e_llr]
                                    + wil * wjr * wkl * ec[e_lrl]
                                    + wil * wjr * wkr * ec[e_lrr]
                                    + wir * wjl * wkl * ec[e_rll]
                                    + wir * wjl * wkr * ec[e_rlr]
                                    + wir * wjr * wkl * ec[e_rrl]
                                    + wir * wjr * wkr * ec[e_rrr];
                    p[i_p*sp + j_p*sjp + k_p] += interp;
                }
            }
        }
    });
}

// =====================================================================
//  restrict_fw (2-D / 3-D) — full-weighting = transpose of prolongate_add.
//  Serial scatter with the IDENTICAL linear_weights() the prolongation gathers
//  with, so R = P^T exactly (matching the CUDA restrict_fw_* kernels).  This is
//  what the V-cycle uses so that the cycle is a SYMMETRIC operator, required for
//  the mgcg/rmgcg CG preconditioner to be valid.  ``rc`` (coarse interior) is
//  zeroed first, then accumulated; serial so no scatter race (the CPU MGCG path
//  is a fallback, and these grids are small).
// =====================================================================

template <typename scalar_t>
static void restrict_fw_2d_cpu_impl(
        const scalar_t* r, scalar_t* rc,
        int Nx_f, int Ny_f, int Nx_c, int Ny_c)
{
    std::fill(rc, rc + (size_t)Nx_c * Ny_c, (scalar_t)0);
    for (int i = 0; i < Nx_f; ++i) {
        int il, ir; scalar_t wil, wir;
        linear_weights<scalar_t>(i, Nx_c, Nx_f, il, ir, wil, wir);
        const int ii[2] = {il, ir}; const scalar_t wi[2] = {wil, wir};
        for (int j = 0; j < Ny_f; ++j) {
            int jl, jr; scalar_t wjl, wjr;
            linear_weights<scalar_t>(j, Ny_c, Ny_f, jl, jr, wjl, wjr);
            const int jj[2] = {jl, jr}; const scalar_t wj[2] = {wjl, wjr};
            const scalar_t val = r[i * Ny_f + j];
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                    rc[ii[a] * Ny_c + jj[b]] += wi[a] * wj[b] * val;
        }
    }
}

template <typename scalar_t>
static void restrict_fw_3d_cpu_impl(
        const scalar_t* r, scalar_t* rc,
        int Nx_f, int Ny_f, int Nz_f,
        int Nx_c, int Ny_c, int Nz_c)
{
    const int sif = Ny_f * Nz_f, sjf = Nz_f;
    const int sic = Ny_c * Nz_c, sjc = Nz_c;
    std::fill(rc, rc + (size_t)Nx_c * sic, (scalar_t)0);
    for (int i = 0; i < Nx_f; ++i) {
        int il, ir; scalar_t wil, wir;
        linear_weights<scalar_t>(i, Nx_c, Nx_f, il, ir, wil, wir);
        const int ii[2] = {il, ir}; const scalar_t wi[2] = {wil, wir};
        for (int j = 0; j < Ny_f; ++j) {
            int jl, jr; scalar_t wjl, wjr;
            linear_weights<scalar_t>(j, Ny_c, Ny_f, jl, jr, wjl, wjr);
            const int jj[2] = {jl, jr}; const scalar_t wj[2] = {wjl, wjr};
            for (int k = 0; k < Nz_f; ++k) {
                int kl, kr; scalar_t wkl, wkr;
                linear_weights<scalar_t>(k, Nz_c, Nz_f, kl, kr, wkl, wkr);
                const int kk[2] = {kl, kr}; const scalar_t wk[2] = {wkl, wkr};
                const scalar_t val = r[i * sif + j * sjf + k];
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 2; ++b)
                        for (int c = 0; c < 2; ++c)
                            rc[ii[a] * sic + jj[b] * sjc + kk[c]]
                                += wi[a] * wj[b] * wk[c] * val;
            }
        }
    }
}


template <typename scalar_t>
static inline scalar_t cv_vleer(scalar_t db, scalar_t df) {
    if (db * df <= (scalar_t)0) return (scalar_t)0;
    scalar_t denom = db + df;
    if (denom > -(scalar_t)1e-20 && denom < (scalar_t)1e-20)
        return (scalar_t)0.5 * db * df;
    return (scalar_t)2 * db * df / denom;
}

template <typename scalar_t>
static inline scalar_t cv_face(
    const scalar_t* a_ptr, int Nfd,
    int64_t a_s_fd, int64_t a_s_t1, int64_t a_s_t2,
    int i_fd, int i_t1, int i_t2,
    scalar_t u_val, scalar_t cfl)
{
    // Clamped indices for a (Neumann edge condition)
    auto read_a = [&](int idx) -> scalar_t {
        if (idx < 0) idx = 0;
        if (idx >= Nfd) idx = Nfd - 1;
        return a_ptr[idx * a_s_fd + i_t1 * a_s_t1 + i_t2 * a_s_t2];
    };
    scalar_t a_k   = read_a(i_fd);
    scalar_t a_km1 = read_a(i_fd - 1);
    scalar_t a_km2 = read_a(i_fd - 2);
    scalar_t a_kp1 = read_a(i_fd + 1);

    scalar_t C = u_val * cfl;
    if (C >= (scalar_t)0) {
        scalar_t s_pos = cv_vleer(a_km1 - a_km2, a_k - a_km1);
        return u_val * (a_km1 + (scalar_t)0.5 * ((scalar_t)1 - C) * s_pos);
    } else {
        scalar_t s_neg = cv_vleer(a_k - a_km1, a_kp1 - a_k);
        return u_val * (a_k - (scalar_t)0.5 * ((scalar_t)1 + C) * s_neg);
    }
}

template <typename scalar_t>
static void cvof_sweep_cpu_impl(
        const scalar_t* a, const scalar_t* u_d,
        int Nfd, int Nt1, int Nt2,
        int64_t a_s_fd, int64_t a_s_t1, int64_t a_s_t2,
        int64_t u_s_fd, int64_t u_s_t1, int64_t u_s_t2,
        int64_t out_s_fd, int64_t out_s_t1, int64_t out_s_t2,
        scalar_t cfl, scalar_t* out)
{
    const int Ni = Nfd - 2;  // interior cells along face_dim
    const int N_tot = Ni * Nt1 * ((Nt2 > 0) ? Nt2 : 1);
    const bool is3d = (Nt2 > 0);

    at::parallel_for(0, N_tot, 1, [&](int64_t start, int64_t end) {
        for (int64_t idx = start; idx < end; ++idx) {
            int i_in, i_t1, i_t2;
            if (is3d) {
                i_in = (int)(idx / (Nt1 * Nt2));
                int rem = (int)(idx % (Nt1 * Nt2));
                i_t1 = rem / Nt2;
                i_t2 = rem % Nt2;
            } else {
                i_in = (int)(idx / Nt1);
                i_t1 = (int)(idx % Nt1);
                i_t2 = 0;
            }
            const int i_fd = i_in + 1;  // global index along face_dim (interior)

            const scalar_t u_i   = u_d[i_fd * u_s_fd + i_t1 * u_s_t1 + i_t2 * u_s_t2];
            const scalar_t u_ip1 = u_d[(i_fd + 1) * u_s_fd + i_t1 * u_s_t1 + i_t2 * u_s_t2];

            // Left face flux F(i_fd) and right face flux F(i_fd+1)
            scalar_t F_i   = cv_face(a, Nfd, a_s_fd, a_s_t1, a_s_t2,
                                      i_fd, i_t1, i_t2, u_i, cfl);
            scalar_t F_ip1 = cv_face(a, Nfd, a_s_fd, a_s_t1, a_s_t2,
                                      i_fd + 1, i_t1, i_t2, u_ip1, cfl);

            const scalar_t a_val = a[i_fd * a_s_fd + i_t1 * a_s_t1 + i_t2 * a_s_t2];
            out[i_fd * out_s_fd + i_t1 * out_s_t1 + i_t2 * out_s_t2] =
                a_val + cfl * (F_i - F_ip1 + a_val * (u_ip1 - u_i));
        }
    });
}

// =====================================================================
//  Poisson whole-solve driver (2-D multigrid)
// =====================================================================

// ``variational`` selects the residual restriction (see the CUDA note in
// cuda/poisson_solve.cu): false = sum-of-children (robust for the stationary
// multigrid iteration); true = full-weighting = P^T (symmetric V-cycle for the
// mgcg/rmgcg CG preconditioner).
template <typename scalar_t>
static void vcycle_2d_cpu(
        scalar_t* p, const scalar_t* f,
        const scalar_t* ch, const scalar_t* cv,
        int Nx, int Ny, scalar_t jcap_tol, scalar_t w,
        int nsmoothing, int smoother_id, scalar_t* r_out,
        bool variational = false)
{
    // Extract face coefficient slices
    std::vector<scalar_t> cp0_buf(Nx * Ny), cm0_buf(Nx * Ny);
    std::vector<scalar_t> cp1_buf(Nx * Ny), cm1_buf(Nx * Ny);

    // ch is (Nx+1, Ny): cp0 = ch[1:Nx+1, :], cm0 = ch[0:Nx, :]
    // cv is (Nx, Ny+1): cp1 = cv[:, 1:Ny+1], cm1 = cv[:, 0:Ny]
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            cp0_buf[i*Ny + j] = ch[(i+1)*(Ny) + j];
            cm0_buf[i*Ny + j] = ch[i*Ny + j];
            cp1_buf[i*Ny + j] = cv[i*(Ny+1) + (j+1)];
            cm1_buf[i*Ny + j] = cv[i*(Ny+1) + j];
        }
    }

    // Pre-smooth
    if (smoother_id == 0) {
        rbgs_sweep_2d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                cp1_buf.data(), cm1_buf.data(),
                                Nx, Ny, jcap_tol, nsmoothing);
    } else {
        jacobi_sweep_2d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                  cp1_buf.data(), cm1_buf.data(),
                                  Nx, Ny, jcap_tol, w, nsmoothing);
    }

    // Residual
    mg_residual_2d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                             cp1_buf.data(), cm1_buf.data(),
                             Nx, Ny, jcap_tol, r_out);

    if (Nx > 2 && Ny > 2) {
        const int Nx_c = Nx / 2;
        const int Ny_c = Ny / 2;

        // Restrict face arrays
        std::vector<scalar_t> ch_c((Nx_c + 1) * Ny_c);
        std::vector<scalar_t> cv_c(Nx_c * (Ny_c + 1));
        restrict_face_2d_cpu_impl<scalar_t, 0>(ch, ch_c.data(), Nx, Ny, Nx_c, Ny_c);
        restrict_face_2d_cpu_impl<scalar_t, 1>(cv, cv_c.data(), Nx, Ny, Nx_c, Ny_c);

        // Restrict residual
        std::vector<scalar_t> r_c(Nx_c * Ny_c);
        if (variational)
            restrict_fw_2d_cpu_impl(r_out, r_c.data(), Nx, Ny, Nx_c, Ny_c);       // P^T (symmetric)
        else
            restrict_residual_2d_cpu_impl(r_out, r_c.data(), Nx, Ny, Nx_c, Ny_c); // sum-of-children

        // Coarse solve
        std::vector<scalar_t> p_c((Nx_c + 2) * (Ny_c + 2), (scalar_t)0);
        std::vector<scalar_t> r_c2(Nx_c * Ny_c);
        vcycle_2d_cpu<scalar_t>(p_c.data(), r_c.data(), ch_c.data(), cv_c.data(),
                                 Nx_c, Ny_c, jcap_tol, w, nsmoothing, smoother_id,
                                 r_c2.data(), variational);

        // Prolongate + add
        prolongate_add_2d_cpu_impl(p_c.data(), p, Nx_c, Ny_c, Nx, Ny);

        // Post-smooth.  ``variational`` also reverses the RBGS half-sweep
        // order, making the post-smooth the A-adjoint of the pre-smooth —
        // without it the cycle is non-symmetric under RBGS even with R = P^T,
        // and invalid as a CG preconditioner.
        if (smoother_id == 0) {
            rbgs_sweep_2d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                    cp1_buf.data(), cm1_buf.data(),
                                    Nx, Ny, jcap_tol, nsmoothing,
                                    /*reverse=*/variational);
        } else {
            jacobi_sweep_2d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                      cp1_buf.data(), cm1_buf.data(),
                                      Nx, Ny, jcap_tol, w, nsmoothing);
        }

        // Post-smooth residual
        mg_residual_2d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                 cp1_buf.data(), cm1_buf.data(),
                                 Nx, Ny, jcap_tol, r_out);
    } else if (variational && smoother_id == 0) {
        // COARSEST LEVEL.  Every other level is symmetric because its
        // post-smooth is the reversed twin of its pre-smooth, but the bottom
        // of the V has no post-smooth at all — its "solve" is a bare red-black
        // sweep, which is NOT self-adjoint, and that alone keeps the whole
        // cycle asymmetric.  Weighted Jacobi needs nothing here: a bare Jacobi
        // solve already is self-adjoint.
        rbgs_sweep_2d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                cp1_buf.data(), cm1_buf.data(),
                                Nx, Ny, jcap_tol, nsmoothing, /*reverse=*/true);
        mg_residual_2d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                 cp1_buf.data(), cm1_buf.data(),
                                 Nx, Ny, jcap_tol, r_out);
    }
}

template <typename scalar_t>
static std::tuple<at::Tensor, int64_t> poisson_solve_multigrid_2d_cpu_impl(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv,
        double h2, double jcap_tol, double w,
        int64_t nsmoothing, int64_t max_vcycles,
        double tol, int64_t smoother_id)
{
    TORCH_CHECK(p.is_contiguous() && f.is_contiguous(),
                "poisson_solve_multigrid_2d: tensors must be contiguous");
    TORCH_CHECK(!p.device().is_cuda(),
                "poisson_solve_multigrid_2d CPU: tensors must be on CPU");

    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);

    // f_scaled = h2 * f
    auto f_scaled = f.mul(h2);
    auto r = at::empty_like(f_scaled);

    scalar_t* pp = p.data_ptr<scalar_t>();
    const scalar_t* ff = f_scaled.data_ptr<scalar_t>();
    const scalar_t* cch = ch.data_ptr<scalar_t>();
    const scalar_t* ccv = cv.data_ptr<scalar_t>();
    scalar_t* rr = r.data_ptr<scalar_t>();

    int64_t niter = 0;
    for (int64_t i = 0; i < max_vcycles; ++i) {
        vcycle_2d_cpu<scalar_t>(pp, ff, cch, ccv,
                                 Nx, Ny,
                                 static_cast<scalar_t>(jcap_tol),
                                 static_cast<scalar_t>(w),
                                 (int)nsmoothing, (int)smoother_id, rr);
        niter = i + 1;
        if (tol >= 0.0) {
            double rnorm = r.abs().max().item<double>();
            if (rnorm < tol) break;
        }
    }

    // Full ghost-ring Neumann BC (corners included — apply_neumann_bc_2d_cpu is
    // the smoothers' face-only pass and leaves them stale) + interior gauge fix.
    // Same two helpers the CUDA driver calls, so the returned p — dead ghosts
    // included — is bit-comparable across backends.
    lilytorch_kernels::poisson::apply_neumann_bc_full(p);
    lilytorch_kernels::poisson::gauge_fix(p);
    return std::make_tuple(r, niter);
}

// =====================================================================
//  Poisson whole-solve driver (3-D multigrid)
// =====================================================================

template <typename scalar_t>
static void vcycle_3d_cpu(
        scalar_t* p, const scalar_t* f,
        const scalar_t* ch, const scalar_t* cv, const scalar_t* cw,
        int Nx, int Ny, int Nz, scalar_t jcap_tol, scalar_t w,
        int nsmoothing, int smoother_id, scalar_t* r_out,
        bool variational = false)
{
    // Extract face coefficient slices
    const size_t ncell = (size_t)Nx * Ny * Nz;
    std::vector<scalar_t> cp0_buf(ncell), cm0_buf(ncell);
    std::vector<scalar_t> cp1_buf(ncell), cm1_buf(ncell);
    std::vector<scalar_t> cp2_buf(ncell), cm2_buf(ncell);

    const int s_ch = Ny * Nz;   // ch stride: (Nx+1, Ny, Nz)
    const int s_cv = (Ny + 1) * Nz;  // cv stride: (Nx, Ny+1, Nz)
    const int s_cw = Ny * (Nz + 1);  // cw stride: (Nx, Ny, Nz+1)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                cp0_buf[i*Ny*Nz + j*Nz + k] = ch[(i+1)*s_ch + j*Nz + k];
                cm0_buf[i*Ny*Nz + j*Nz + k] = ch[i*s_ch + j*Nz + k];
                cp1_buf[i*Ny*Nz + j*Nz + k] = cv[i*s_cv + (j+1)*Nz + k];
                cm1_buf[i*Ny*Nz + j*Nz + k] = cv[i*s_cv + j*Nz + k];
                cp2_buf[i*Ny*Nz + j*Nz + k] = cw[i*s_cw + j*(Nz+1) + (k+1)];
                cm2_buf[i*Ny*Nz + j*Nz + k] = cw[i*s_cw + j*(Nz+1) + k];
            }
        }
    }

    // Pre-smooth
    if (smoother_id == 0) {
        rbgs_sweep_3d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                cp1_buf.data(), cm1_buf.data(),
                                cp2_buf.data(), cm2_buf.data(),
                                Nx, Ny, Nz, jcap_tol, nsmoothing);
    } else {
        jacobi_sweep_3d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                  cp1_buf.data(), cm1_buf.data(),
                                  cp2_buf.data(), cm2_buf.data(),
                                  Nx, Ny, Nz, jcap_tol, w, nsmoothing);
    }

    mg_residual_3d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                             cp1_buf.data(), cm1_buf.data(),
                             cp2_buf.data(), cm2_buf.data(),
                             Nx, Ny, Nz, jcap_tol, r_out);

    if (Nx > 2 && Ny > 2 && Nz > 2) {
        const int Nx_c = Nx / 2, Ny_c = Ny / 2, Nz_c = Nz / 2;

        std::vector<scalar_t> ch_c((Nx_c + 1) * Ny_c * Nz_c);
        std::vector<scalar_t> cv_c(Nx_c * (Ny_c + 1) * Nz_c);
        std::vector<scalar_t> cw_c(Nx_c * Ny_c * (Nz_c + 1));
        restrict_face_3d_cpu_impl<scalar_t, 0>(ch, ch_c.data(), Nx, Ny, Nz, Nx_c, Ny_c, Nz_c);
        restrict_face_3d_cpu_impl<scalar_t, 1>(cv, cv_c.data(), Nx, Ny, Nz, Nx_c, Ny_c, Nz_c);
        restrict_face_3d_cpu_impl<scalar_t, 2>(cw, cw_c.data(), Nx, Ny, Nz, Nx_c, Ny_c, Nz_c);

        std::vector<scalar_t> r_c(Nx_c * Ny_c * Nz_c);
        if (variational)
            restrict_fw_3d_cpu_impl(r_out, r_c.data(), Nx, Ny, Nz, Nx_c, Ny_c, Nz_c);       // P^T (symmetric)
        else
            restrict_residual_3d_cpu_impl(r_out, r_c.data(), Nx, Ny, Nz, Nx_c, Ny_c, Nz_c); // sum-of-children

        std::vector<scalar_t> p_c((Nx_c + 2) * (Ny_c + 2) * (Nz_c + 2), (scalar_t)0);
        std::vector<scalar_t> r_c2(Nx_c * Ny_c * Nz_c);
        vcycle_3d_cpu<scalar_t>(p_c.data(), r_c.data(), ch_c.data(), cv_c.data(), cw_c.data(),
                                 Nx_c, Ny_c, Nz_c, jcap_tol, w, nsmoothing, smoother_id,
                                 r_c2.data(), variational);

        prolongate_add_3d_cpu_impl(p_c.data(), p, Nx_c, Ny_c, Nz_c, Nx, Ny, Nz);

        // Post-smooth: ``variational`` reverses the RBGS half-sweep order so
        // it is the A-adjoint of the pre-smooth (see vcycle_2d_cpu).
        if (smoother_id == 0) {
            rbgs_sweep_3d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                    cp1_buf.data(), cm1_buf.data(),
                                    cp2_buf.data(), cm2_buf.data(),
                                    Nx, Ny, Nz, jcap_tol, nsmoothing,
                                    /*reverse=*/variational);
        } else {
            jacobi_sweep_3d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                      cp1_buf.data(), cm1_buf.data(),
                                      cp2_buf.data(), cm2_buf.data(),
                                      Nx, Ny, Nz, jcap_tol, w, nsmoothing);
        }

        mg_residual_3d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                 cp1_buf.data(), cm1_buf.data(),
                                 cp2_buf.data(), cm2_buf.data(),
                                 Nx, Ny, Nz, jcap_tol, r_out);
    } else if (variational && smoother_id == 0) {
        // Coarsest level — see the matching note in vcycle_2d_cpu.
        rbgs_sweep_3d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                cp1_buf.data(), cm1_buf.data(),
                                cp2_buf.data(), cm2_buf.data(),
                                Nx, Ny, Nz, jcap_tol, nsmoothing,
                                /*reverse=*/true);
        mg_residual_3d_cpu_impl(p, f, cp0_buf.data(), cm0_buf.data(),
                                 cp1_buf.data(), cm1_buf.data(),
                                 cp2_buf.data(), cm2_buf.data(),
                                 Nx, Ny, Nz, jcap_tol, r_out);
    }
}

template <typename scalar_t>
static std::tuple<at::Tensor, int64_t> poisson_solve_multigrid_3d_cpu_impl(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv, at::Tensor cw,
        double h2, double jcap_tol, double w,
        int64_t nsmoothing, int64_t max_vcycles,
        double tol, int64_t smoother_id)
{
    TORCH_CHECK(p.is_contiguous() && f.is_contiguous(),
                "poisson_solve_multigrid_3d: tensors must be contiguous");
    TORCH_CHECK(!p.device().is_cuda(),
                "poisson_solve_multigrid_3d CPU: tensors must be on CPU");

    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);
    const int Nz = (int)f.size(2);

    auto f_scaled = f.mul(h2);
    auto r = at::empty_like(f_scaled);

    scalar_t* pp = p.data_ptr<scalar_t>();
    const scalar_t* ff = f_scaled.data_ptr<scalar_t>();
    const scalar_t* cch = ch.data_ptr<scalar_t>();
    const scalar_t* ccv = cv.data_ptr<scalar_t>();
    const scalar_t* ccw = cw.data_ptr<scalar_t>();
    scalar_t* rr = r.data_ptr<scalar_t>();

    int64_t niter = 0;
    for (int64_t i = 0; i < max_vcycles; ++i) {
        vcycle_3d_cpu<scalar_t>(pp, ff, cch, ccv, ccw,
                                 Nx, Ny, Nz,
                                 static_cast<scalar_t>(jcap_tol),
                                 static_cast<scalar_t>(w),
                                 (int)nsmoothing, (int)smoother_id, rr);
        niter = i + 1;
        if (tol >= 0.0) {
            double rnorm = r.abs().max().item<double>();
            if (rnorm < tol) break;
        }
    }

    // Full ghost-ring Neumann BC + interior gauge fix — see the 2-D driver.
    lilytorch_kernels::poisson::apply_neumann_bc_full(p);
    lilytorch_kernels::poisson::gauge_fix(p);
    return std::make_tuple(r, niter);
}

// =====================================================================
//  Dispatch stubs (CPU → calls impl)
// =====================================================================

static void rbgs_sweep_2d_cpu(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        double jcap_tol, int64_t nsmoothing)
{
    TORCH_CHECK(!p.device().is_cuda(), "rbgs_sweep_2d CPU: expected CPU tensors");
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "rbgs_sweep_2d_cpu", [&] {
        rbgs_sweep_2d_cpu_impl<scalar_t>(
            p.data_ptr<scalar_t>(), f.data_ptr<scalar_t>(),
            cp0.data_ptr<scalar_t>(), cm0.data_ptr<scalar_t>(),
            cp1.data_ptr<scalar_t>(), cm1.data_ptr<scalar_t>(),
            (int)f.size(0), (int)f.size(1),
            static_cast<scalar_t>(jcap_tol), (int)nsmoothing);
    });
}

static void rbgs_sweep_3d_cpu(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        at::Tensor cp2, at::Tensor cm2,
        double jcap_tol, int64_t nsmoothing)
{
    TORCH_CHECK(!p.device().is_cuda(), "rbgs_sweep_3d CPU: expected CPU tensors");
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "rbgs_sweep_3d_cpu", [&] {
        rbgs_sweep_3d_cpu_impl<scalar_t>(
            p.data_ptr<scalar_t>(), f.data_ptr<scalar_t>(),
            cp0.data_ptr<scalar_t>(), cm0.data_ptr<scalar_t>(),
            cp1.data_ptr<scalar_t>(), cm1.data_ptr<scalar_t>(),
            cp2.data_ptr<scalar_t>(), cm2.data_ptr<scalar_t>(),
            (int)f.size(0), (int)f.size(1), (int)f.size(2),
            static_cast<scalar_t>(jcap_tol), (int)nsmoothing);
    });
}

static void jacobi_sweep_2d_cpu(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        double jcap_tol, double w, int64_t nsmoothing)
{
    TORCH_CHECK(!p.device().is_cuda(), "jacobi_sweep_2d CPU: expected CPU tensors");
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "jacobi_sweep_2d_cpu", [&] {
        jacobi_sweep_2d_cpu_impl<scalar_t>(
            p.data_ptr<scalar_t>(), f.data_ptr<scalar_t>(),
            cp0.data_ptr<scalar_t>(), cm0.data_ptr<scalar_t>(),
            cp1.data_ptr<scalar_t>(), cm1.data_ptr<scalar_t>(),
            (int)f.size(0), (int)f.size(1),
            static_cast<scalar_t>(jcap_tol), static_cast<scalar_t>(w), (int)nsmoothing);
    });
}

static void jacobi_sweep_3d_cpu(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        at::Tensor cp2, at::Tensor cm2,
        double jcap_tol, double w, int64_t nsmoothing)
{
    TORCH_CHECK(!p.device().is_cuda(), "jacobi_sweep_3d CPU: expected CPU tensors");
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "jacobi_sweep_3d_cpu", [&] {
        jacobi_sweep_3d_cpu_impl<scalar_t>(
            p.data_ptr<scalar_t>(), f.data_ptr<scalar_t>(),
            cp0.data_ptr<scalar_t>(), cm0.data_ptr<scalar_t>(),
            cp1.data_ptr<scalar_t>(), cm1.data_ptr<scalar_t>(),
            cp2.data_ptr<scalar_t>(), cm2.data_ptr<scalar_t>(),
            (int)f.size(0), (int)f.size(1), (int)f.size(2),
            static_cast<scalar_t>(jcap_tol), static_cast<scalar_t>(w), (int)nsmoothing);
    });
}

static void mg_residual_2d_cpu_dispatch(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        double jcap_tol, at::Tensor r)
{
    TORCH_CHECK(!p.device().is_cuda(), "mg_residual_2d CPU: expected CPU tensors");
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "mg_residual_2d_cpu", [&] {
        mg_residual_2d_cpu_impl<scalar_t>(
            p.data_ptr<scalar_t>(), f.data_ptr<scalar_t>(),
            cp0.data_ptr<scalar_t>(), cm0.data_ptr<scalar_t>(),
            cp1.data_ptr<scalar_t>(), cm1.data_ptr<scalar_t>(),
            (int)f.size(0), (int)f.size(1),
            static_cast<scalar_t>(jcap_tol), r.data_ptr<scalar_t>());
    });
}

static void mg_residual_3d_cpu_dispatch(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        at::Tensor cp2, at::Tensor cm2,
        double jcap_tol, at::Tensor r)
{
    TORCH_CHECK(!p.device().is_cuda(), "mg_residual_3d CPU: expected CPU tensors");
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "mg_residual_3d_cpu", [&] {
        mg_residual_3d_cpu_impl<scalar_t>(
            p.data_ptr<scalar_t>(), f.data_ptr<scalar_t>(),
            cp0.data_ptr<scalar_t>(), cm0.data_ptr<scalar_t>(),
            cp1.data_ptr<scalar_t>(), cm1.data_ptr<scalar_t>(),
            cp2.data_ptr<scalar_t>(), cm2.data_ptr<scalar_t>(),
            (int)f.size(0), (int)f.size(1), (int)f.size(2),
            static_cast<scalar_t>(jcap_tol), r.data_ptr<scalar_t>());
    });
}

static void restrict_residual_2d_cpu_dispatch(at::Tensor r, at::Tensor rc) {
    TORCH_CHECK(!r.device().is_cuda(), "restrict_residual_2d CPU: expected CPU tensors");
    AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "restrict_residual_2d_cpu", [&] {
        restrict_residual_2d_cpu_impl<scalar_t>(
            r.data_ptr<scalar_t>(), rc.data_ptr<scalar_t>(),
            (int)r.size(0), (int)r.size(1),
            (int)rc.size(0), (int)rc.size(1));
    });
}

static void restrict_residual_3d_cpu_dispatch(at::Tensor r, at::Tensor rc) {
    TORCH_CHECK(!r.device().is_cuda(), "restrict_residual_3d CPU: expected CPU tensors");
    AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "restrict_residual_3d_cpu", [&] {
        restrict_residual_3d_cpu_impl<scalar_t>(
            r.data_ptr<scalar_t>(), rc.data_ptr<scalar_t>(),
            (int)r.size(0), (int)r.size(1), (int)r.size(2),
            (int)rc.size(0), (int)rc.size(1), (int)rc.size(2));
    });
}

static void restrict_face_2d_cpu_dispatch(
        at::Tensor src, at::Tensor dst, int64_t face_dim)
{
    TORCH_CHECK(!src.device().is_cuda(), "restrict_face_2d CPU: expected CPU tensors");
    AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "restrict_face_2d_cpu", [&] {
        if (face_dim == 0) {
            // src: (N0+1, N1) — face_dim=0 has +1 on dim 0
            int Nf0 = (int)src.size(0) - 1, Nf1 = (int)src.size(1);
            int Nc0 = (int)dst.size(0) - 1, Nc1 = (int)dst.size(1);
            restrict_face_2d_cpu_impl<scalar_t, 0>(
                src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(),
                Nf0, Nf1, Nc0, Nc1);
        } else {
            int Nf0 = (int)src.size(0), Nf1 = (int)src.size(1) - 1;
            int Nc0 = (int)dst.size(0), Nc1 = (int)dst.size(1) - 1;
            restrict_face_2d_cpu_impl<scalar_t, 1>(
                src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(),
                Nf0, Nf1, Nc0, Nc1);
        }
    });
}

static void restrict_face_3d_cpu_dispatch(
        at::Tensor src, at::Tensor dst, int64_t face_dim)
{
    TORCH_CHECK(!src.device().is_cuda(), "restrict_face_3d CPU: expected CPU tensors");
    AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "restrict_face_3d_cpu", [&] {
        if (face_dim == 0) {
            int Nf0=(int)src.size(0)-1, Nf1=(int)src.size(1), Nf2=(int)src.size(2);
            int Nc0=(int)dst.size(0)-1, Nc1=(int)dst.size(1), Nc2=(int)dst.size(2);
            restrict_face_3d_cpu_impl<scalar_t, 0>(
                src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(),
                Nf0,Nf1,Nf2, Nc0,Nc1,Nc2);
        } else if (face_dim == 1) {
            int Nf0=(int)src.size(0), Nf1=(int)src.size(1)-1, Nf2=(int)src.size(2);
            int Nc0=(int)dst.size(0), Nc1=(int)dst.size(1)-1, Nc2=(int)dst.size(2);
            restrict_face_3d_cpu_impl<scalar_t, 1>(
                src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(),
                Nf0,Nf1,Nf2, Nc0,Nc1,Nc2);
        } else {
            int Nf0=(int)src.size(0), Nf1=(int)src.size(1), Nf2=(int)src.size(2)-1;
            int Nc0=(int)dst.size(0), Nc1=(int)dst.size(1), Nc2=(int)dst.size(2)-1;
            restrict_face_3d_cpu_impl<scalar_t, 2>(
                src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(),
                Nf0,Nf1,Nf2, Nc0,Nc1,Nc2);
        }
    });
}

static void prolongate_add_2d_cpu_dispatch(at::Tensor ec, at::Tensor p) {
    TORCH_CHECK(!ec.device().is_cuda(), "prolongate_add_2d CPU: expected CPU tensors");
    AT_DISPATCH_FLOATING_TYPES(ec.scalar_type(), "prolongate_add_2d_cpu", [&] {
        prolongate_add_2d_cpu_impl<scalar_t>(
            ec.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(),
            (int)ec.size(0) - 2, (int)ec.size(1) - 2,
            (int)p.size(0) - 2, (int)p.size(1) - 2);
    });
}

static void prolongate_add_3d_cpu_dispatch(at::Tensor ec, at::Tensor p) {
    TORCH_CHECK(!ec.device().is_cuda(), "prolongate_add_3d CPU: expected CPU tensors");
    AT_DISPATCH_FLOATING_TYPES(ec.scalar_type(), "prolongate_add_3d_cpu", [&] {
        prolongate_add_3d_cpu_impl<scalar_t>(
            ec.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(),
            (int)ec.size(0) - 2, (int)ec.size(1) - 2, (int)ec.size(2) - 2,
            (int)p.size(0) - 2, (int)p.size(1) - 2, (int)p.size(2) - 2);
    });
}

static void cvof_sweep_cpu_dispatch(
        const at::Tensor& a, const at::Tensor& u_d, double cfl,
        int64_t face_dim, at::Tensor& out)
{
    TORCH_CHECK(!a.device().is_cuda(), "cvof_sweep CPU: expected CPU tensors");
    const int ndim = (int)a.dim();
    const int Nfd = (int)a.size(face_dim);

    int Nt1 = 1, Nt2 = 1;
    int64_t a_s_fd, a_s_t1, a_s_t2;
    int64_t u_s_fd, u_s_t1, u_s_t2;
    int64_t out_s_fd, out_s_t1, out_s_t2;

    if (ndim == 2) {
        Nt1 = (face_dim == 0) ? (int)a.size(1) : (int)a.size(0);
        a_s_fd = (face_dim == 0) ? a.stride(0) : a.stride(1);
        a_s_t1 = (face_dim == 0) ? a.stride(1) : a.stride(0);
        a_s_t2 = 1;
        u_s_fd = (face_dim == 0) ? u_d.stride(0) : u_d.stride(1);
        u_s_t1 = (face_dim == 0) ? u_d.stride(1) : u_d.stride(0);
        u_s_t2 = 1;
        out_s_fd = out.stride(face_dim);
        out_s_t1 = (face_dim == 0) ? out.stride(1) : out.stride(0);
        out_s_t2 = 1;
    } else {
        if (face_dim == 0) {
            Nt1=(int)a.size(1); Nt2=(int)a.size(2);
            a_s_fd=a.stride(0); a_s_t1=a.stride(1); a_s_t2=a.stride(2);
            u_s_fd=u_d.stride(0); u_s_t1=u_d.stride(1); u_s_t2=u_d.stride(2);
            out_s_fd=out.stride(0); out_s_t1=out.stride(1); out_s_t2=out.stride(2);
        } else if (face_dim == 1) {
            Nt1=(int)a.size(0); Nt2=(int)a.size(2);
            a_s_fd=a.stride(1); a_s_t1=a.stride(0); a_s_t2=a.stride(2);
            u_s_fd=u_d.stride(1); u_s_t1=u_d.stride(0); u_s_t2=u_d.stride(2);
            out_s_fd=out.stride(1); out_s_t1=out.stride(0); out_s_t2=out.stride(2);
        } else {
            Nt1=(int)a.size(0); Nt2=(int)a.size(1);
            a_s_fd=a.stride(2); a_s_t1=a.stride(0); a_s_t2=a.stride(1);
            u_s_fd=u_d.stride(2); u_s_t1=u_d.stride(0); u_s_t2=u_d.stride(1);
            out_s_fd=out.stride(2); out_s_t1=out.stride(0); out_s_t2=out.stride(1);
        }
    }

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "cvof_sweep_cpu", [&] {
        cvof_sweep_cpu_impl<scalar_t>(
            a.data_ptr<scalar_t>(), u_d.data_ptr<scalar_t>(),
            Nfd, Nt1, Nt2,
            a_s_fd, a_s_t1, a_s_t2,
            u_s_fd, u_s_t1, u_s_t2,
            out_s_fd, out_s_t1, out_s_t2,
            static_cast<scalar_t>(cfl), out.data_ptr<scalar_t>());
    });
}

static std::tuple<at::Tensor, int64_t> poisson_solve_multigrid_2d_cpu(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv,
        double h2, double jcap_tol, double w,
        int64_t nsmoothing, int64_t max_vcycles,
        double tol, int64_t smoother_id)
{
    return AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "poisson_solve_multigrid_2d_cpu", [&] {
        return poisson_solve_multigrid_2d_cpu_impl<scalar_t>(
            p, f, ch, cv, h2, jcap_tol, w, nsmoothing, max_vcycles, tol, smoother_id);
    });
}

static std::tuple<at::Tensor, int64_t> poisson_solve_multigrid_3d_cpu(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv, at::Tensor cw,
        double h2, double jcap_tol, double w,
        int64_t nsmoothing, int64_t max_vcycles,
        double tol, int64_t smoother_id)
{
    return AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "poisson_solve_multigrid_3d_cpu", [&] {
        return poisson_solve_multigrid_3d_cpu_impl<scalar_t>(
            p, f, ch, cv, cw, h2, jcap_tol, w, nsmoothing, max_vcycles, tol, smoother_id);
    });
}

// =====================================================================
//  Raw V-cycle op — the MGCG preconditioner primitive (CPU twin).
//
//  ``n_vcycles`` V-cycles with NO gauge fix (no ghost-ring Neumann pass, no
//  mean subtraction), unlike the whole-solve driver above.  ``f`` is the raw
//  smoother RHS (already h²-scaled by the caller); ``p`` (ghost-padded) is
//  mutated in place and the interior residual is returned.
//
//  This is what keeps MGCG / RMGCG alive on the CPU: the whole-solve twins
//  below are stubs, so ``PoissonSolver._cg_core`` runs the CG driver in Python
//  and calls THIS for the preconditioner — the same V-cycle the native CUDA
//  MGCG driver applies to ``z`` internally.
// =====================================================================

template <typename scalar_t>
static at::Tensor mg_vcycle_2d_cpu_impl(
        at::Tensor p, at::Tensor f, at::Tensor ch, at::Tensor cv,
        double jcap_tol, double w,
        int64_t nsmoothing, int64_t n_vcycles, int64_t smoother_id)
{
    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);
    auto r = at::empty_like(f);

    scalar_t* pp = p.data_ptr<scalar_t>();
    const scalar_t* ff = f.data_ptr<scalar_t>();
    const scalar_t* cch = ch.data_ptr<scalar_t>();
    const scalar_t* ccv = cv.data_ptr<scalar_t>();
    scalar_t* rr = r.data_ptr<scalar_t>();

    for (int64_t i = 0; i < n_vcycles; ++i) {
        vcycle_2d_cpu<scalar_t>(pp, ff, cch, ccv, Nx, Ny,
                                static_cast<scalar_t>(jcap_tol),
                                static_cast<scalar_t>(w),
                                (int)nsmoothing, (int)smoother_id, rr,
                                /*variational=*/true);  // CG preconditioner: symmetric V-cycle
    }
    return r;
}

static at::Tensor mg_vcycle_2d_cpu(
        at::Tensor p, at::Tensor f, at::Tensor ch, at::Tensor cv,
        double jcap_tol, double w,
        int64_t nsmoothing, int64_t n_vcycles, int64_t smoother_id)
{
    TORCH_CHECK(p.is_contiguous() && f.is_contiguous()
                && ch.is_contiguous() && cv.is_contiguous(),
                "mg_vcycle_2d: tensors must be contiguous");
    return AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "mg_vcycle_2d_cpu", [&] {
        return mg_vcycle_2d_cpu_impl<scalar_t>(
            p, f, ch, cv, jcap_tol, w, nsmoothing, n_vcycles, smoother_id);
    });
}

template <typename scalar_t>
static at::Tensor mg_vcycle_3d_cpu_impl(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv, at::Tensor cw,
        double jcap_tol, double w,
        int64_t nsmoothing, int64_t n_vcycles, int64_t smoother_id)
{
    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);
    const int Nz = (int)f.size(2);
    auto r = at::empty_like(f);

    scalar_t* pp = p.data_ptr<scalar_t>();
    const scalar_t* ff = f.data_ptr<scalar_t>();
    const scalar_t* cch = ch.data_ptr<scalar_t>();
    const scalar_t* ccv = cv.data_ptr<scalar_t>();
    const scalar_t* ccw = cw.data_ptr<scalar_t>();
    scalar_t* rr = r.data_ptr<scalar_t>();

    for (int64_t i = 0; i < n_vcycles; ++i) {
        vcycle_3d_cpu<scalar_t>(pp, ff, cch, ccv, ccw, Nx, Ny, Nz,
                                static_cast<scalar_t>(jcap_tol),
                                static_cast<scalar_t>(w),
                                (int)nsmoothing, (int)smoother_id, rr,
                                /*variational=*/true);  // CG preconditioner: symmetric V-cycle
    }
    return r;
}

static at::Tensor mg_vcycle_3d_cpu(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv, at::Tensor cw,
        double jcap_tol, double w,
        int64_t nsmoothing, int64_t n_vcycles, int64_t smoother_id)
{
    TORCH_CHECK(p.is_contiguous() && f.is_contiguous() && ch.is_contiguous()
                && cv.is_contiguous() && cw.is_contiguous(),
                "mg_vcycle_3d: tensors must be contiguous");
    return AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "mg_vcycle_3d_cpu", [&] {
        return mg_vcycle_3d_cpu_impl<scalar_t>(
            p, f, ch, cv, cw, jcap_tol, w, nsmoothing, n_vcycles, smoother_id);
    });
}

// MGCG and RMGCG CPU stubs — the whole-solve drivers.  On the CPU the CG loop
// runs in Python (``PoissonSolver._cg_core``) against ``mg_vcycle_{2,3}d``
// above, so these raise rather than duplicating that driver in C++.
static std::tuple<at::Tensor, int64_t> poisson_solve_mgcg_2d_cpu(
        at::Tensor /*p*/, at::Tensor /*f*/,
        at::Tensor /*ch*/, at::Tensor /*cv*/,
        double /*h2*/, double /*jcap_tol*/, double /*w*/,
        int64_t /*nsmoothing*/, int64_t /*max_cycles*/, int64_t /*precond_vcycles*/,
        double /*tol*/, int64_t /*smoother_id*/)
{
    TORCH_CHECK(false,
        "poisson_solve_mgcg_2d CPU twin is not yet implemented. "
        "Use the Python MGCG path (PoissonSolver.solve_mgcg) on CPU.");
    return std::make_tuple(at::Tensor{}, (int64_t)0);
}

static std::tuple<at::Tensor, int64_t> poisson_solve_mgcg_3d_cpu(
        at::Tensor /*p*/, at::Tensor /*f*/,
        at::Tensor /*ch*/, at::Tensor /*cv*/, at::Tensor /*cw*/,
        double /*h2*/, double /*jcap_tol*/, double /*w*/,
        int64_t /*nsmoothing*/, int64_t /*max_cycles*/, int64_t /*precond_vcycles*/,
        double /*tol*/, int64_t /*smoother_id*/)
{
    TORCH_CHECK(false,
        "poisson_solve_mgcg_3d CPU twin is not yet implemented. "
        "Use the Python MGCG path (PoissonSolver.solve_mgcg) on CPU.");
    return std::make_tuple(at::Tensor{}, (int64_t)0);
}

static std::tuple<at::Tensor, at::Tensor, int64_t> poisson_solve_rmgcg_2d_cpu(
        at::Tensor /*p*/, at::Tensor /*f*/,
        at::Tensor /*ch*/, at::Tensor /*cv*/,
        at::Tensor /*U*/, at::Tensor /*W*/, int64_t /*harvest_k*/,
        double /*h2*/, double /*jcap_tol*/, double /*w*/,
        int64_t /*nsmoothing*/, int64_t /*max_cycles*/, int64_t /*precond_vcycles*/,
        double /*tol*/, int64_t /*smoother_id*/)
{
    TORCH_CHECK(false,
        "poisson_solve_rmgcg_2d CPU twin is not yet implemented. "
        "Use the Python RMGCG path (PoissonSolver.solve_rmgcg) on CPU.");
    return {};
}

static std::tuple<at::Tensor, at::Tensor, int64_t> poisson_solve_rmgcg_3d_cpu(
        at::Tensor /*p*/, at::Tensor /*f*/,
        at::Tensor /*ch*/, at::Tensor /*cv*/, at::Tensor /*cw*/,
        at::Tensor /*U*/, at::Tensor /*W*/, int64_t /*harvest_k*/,
        double /*h2*/, double /*jcap_tol*/, double /*w*/,
        int64_t /*nsmoothing*/, int64_t /*max_cycles*/, int64_t /*precond_vcycles*/,
        double /*tol*/, int64_t /*smoother_id*/)
{
    TORCH_CHECK(false,
        "poisson_solve_rmgcg_3d CPU twin is not yet implemented. "
        "Use the Python RMGCG path (PoissonSolver.solve_rmgcg) on CPU.");
    return {};
}

// =====================================================================
//  CPU dispatch registration
// =====================================================================

TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    // Smoothers
    m.impl("rbgs_sweep_2d", &rbgs_sweep_2d_cpu);
    m.impl("rbgs_sweep_3d", &rbgs_sweep_3d_cpu);
    m.impl("jacobi_sweep_2d", &jacobi_sweep_2d_cpu);
    m.impl("jacobi_sweep_3d", &jacobi_sweep_3d_cpu);
    m.impl("mg_residual_2d", &mg_residual_2d_cpu_dispatch);
    m.impl("mg_residual_3d", &mg_residual_3d_cpu_dispatch);
    // Transfer ops
    m.impl("restrict_residual_2d", &restrict_residual_2d_cpu_dispatch);
    m.impl("restrict_residual_3d", &restrict_residual_3d_cpu_dispatch);
    m.impl("restrict_face_2d", &restrict_face_2d_cpu_dispatch);
    m.impl("restrict_face_3d", &restrict_face_3d_cpu_dispatch);
    m.impl("prolongate_add_2d", &prolongate_add_2d_cpu_dispatch);
    m.impl("prolongate_add_3d", &prolongate_add_3d_cpu_dispatch);
    // Advection / cvof
    m.impl("cvof_sweep", &cvof_sweep_cpu_dispatch);
    // Raw V-cycle (MGCG preconditioner primitive)
    m.impl("mg_vcycle_2d", &mg_vcycle_2d_cpu);
    m.impl("mg_vcycle_3d", &mg_vcycle_3d_cpu);
    // Poisson whole-solve drivers
    m.impl("poisson_solve_multigrid_2d", &poisson_solve_multigrid_2d_cpu);
    m.impl("poisson_solve_multigrid_3d", &poisson_solve_multigrid_3d_cpu);
    m.impl("poisson_solve_mgcg_2d", &poisson_solve_mgcg_2d_cpu);
    m.impl("poisson_solve_mgcg_3d", &poisson_solve_mgcg_3d_cpu);
    m.impl("poisson_solve_rmgcg_2d", &poisson_solve_rmgcg_2d_cpu);
    m.impl("poisson_solve_rmgcg_3d", &poisson_solve_rmgcg_3d_cpu);
}

}  // namespace lilytorch_kernels
