// =====================================================================
//  poisson_solve.cu — native C++ driver for variable-coefficient
//  multigrid Poisson solve.  Eliminates all Python-level orchestration.
//
//  Ops registered:
//    poisson_solve_multigrid_2d(p, f, ch, cv, h2, jcap_tol, w,
//        nsmoothing, max_vcycles, tol, smoother_id) -> Tensor
//    poisson_solve_multigrid_3d(p, f, ch, cv, cw, ...)            -> Tensor
//
//  ``p`` is ghost-padded (Nx+2, Ny+2[, Nz+2]); written in place.
//  ``f`` is interior-shape (no h² applied yet — driver multiplies).
//  Returns (final residual r (interior shape), iterations performed).
//
//  Smoother id: 0 = RBGS, 1 = weighted Jacobi.
//
//  Matches the Python ``PoissonSolver.solve_multigrid`` algorithm:
//    f_scaled = h² * f
//    for i in [0, max_vcycles):
//        p, r = vcycle(f_scaled, p, faces)
//        if ||r||_∞ < tol: break
//    p -= mean(p)           # float64 reduction, even for f32 p
// =====================================================================

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "../common/poisson_gauge.h"
#include "../common/poisson_scratch.h"
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <tuple>
#include <algorithm>

namespace lilytorch_kernels {

// ---- Forward declarations of kernel wrappers in other TUs -----------
void rbgs_sweep_2d_cuda(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        double jcap_tol, int64_t nsmoothing);
void rbgs_sweep_3d_cuda(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        at::Tensor cp2, at::Tensor cm2,
        double jcap_tol, int64_t nsmoothing);
// ``_ex`` variants take the half-sweep-order flag used by the symmetric
// V-cycle's post-smooth (see smooth_2d / smooth_3d below).
void rbgs_sweep_2d_cuda_ex(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        double jcap_tol, int64_t nsmoothing, bool reverse);
void rbgs_sweep_3d_cuda_ex(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        at::Tensor cp2, at::Tensor cm2,
        double jcap_tol, int64_t nsmoothing, bool reverse);
void jacobi_sweep_2d_cuda(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        double jcap_tol, double w, int64_t nsmoothing);
void jacobi_sweep_3d_cuda(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        at::Tensor cp2, at::Tensor cm2,
        double jcap_tol, double w, int64_t nsmoothing);
void mg_residual_2d_cuda(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        double jcap_tol, at::Tensor r);
void mg_residual_3d_cuda(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        at::Tensor cp2, at::Tensor cm2,
        double jcap_tol, at::Tensor r);

void restrict_residual_2d_cuda(at::Tensor r, at::Tensor rc);
void restrict_residual_3d_cuda(at::Tensor r, at::Tensor rc);
// Full-weighting restriction = transpose of prolongate_add (see
// multigrid_transfer.cu).  The V-cycle uses THIS, not the sum-of-children
// restrict_residual above, so that R = P^T and the cycle is a SYMMETRIC
// operator — required for the mgcg/rmgcg CG preconditioner to be valid.
void restrict_fw_2d_cuda(at::Tensor r, at::Tensor rc);
void restrict_fw_3d_cuda(at::Tensor r, at::Tensor rc);
void restrict_face_2d_cuda(at::Tensor src, at::Tensor dst, int64_t face_dim);
void restrict_face_3d_cuda(at::Tensor src, at::Tensor dst, int64_t face_dim);
void prolongate_add_2d_cuda(at::Tensor ec, at::Tensor p);
void prolongate_add_3d_cuda(at::Tensor ec, at::Tensor p);

// ---------------------------------------------------------------------
// Smooth dispatch (RBGS vs Jacobi).  ``cp0/cm0`` etc are pre-extracted
// non-contiguous slices of ch/cv/cw; the wrappers call .contiguous()
// internally (they're already contiguous when ch/cv/cw are contiguous,
// which they are in our driver — we materialise them once per level).
// ---------------------------------------------------------------------

// ``reverse`` flips the RBGS half-sweep order (red-black -> black-red).  It is
// set on the POST-smooth of a symmetric (CG-preconditioner) V-cycle so the two
// smoothers are A-adjoint; weighted Jacobi is already self-adjoint and ignores
// it.  See the ``color`` note in multigrid_smoothers.cu.
static inline void smooth_2d(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        double jcap_tol, double w, int64_t nsmoothing,
        int64_t smoother_id, bool reverse = false)
{
    if (smoother_id == 0)
        rbgs_sweep_2d_cuda_ex(p, f, cp0, cm0, cp1, cm1, jcap_tol, nsmoothing,
                              reverse);
    else
        jacobi_sweep_2d_cuda(p, f, cp0, cm0, cp1, cm1, jcap_tol, w, nsmoothing);
}

static inline void smooth_3d(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        at::Tensor cp2, at::Tensor cm2,
        double jcap_tol, double w, int64_t nsmoothing,
        int64_t smoother_id, bool reverse = false)
{
    if (smoother_id == 0)
        rbgs_sweep_3d_cuda_ex(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                              jcap_tol, nsmoothing, reverse);
    else
        jacobi_sweep_3d_cuda(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                             jcap_tol, w, nsmoothing);
}

// =====================================================================
// 2-D recursive V-cycle
//
//   Input  p : ghost-padded (Nx+2, Ny+2) — mutated in place
//          f : interior (Nx, Ny)
//          ch, cv : face arrays at fine level (contiguous)
//   Output (in p) : smoothed pressure;  r filled in caller-provided buffer
// =====================================================================

// ``variational`` selects the residual restriction:
//   false  -> sum-of-children (restrict_residual): robust for the STATIONARY
//             multigrid iteration, which is sensitive to coarse-correction
//             scaling and DIVERGES with full-weighting on stiff (833:1) operators.
//   true   -> full-weighting = P^T (restrict_fw): makes the V-cycle a SYMMETRIC
//             operator, required for the mgcg/rmgcg CG preconditioner.  CG is
//             immune to the scaling issue (it computes the optimal step length),
//             so this is both valid AND far more effective there.
static void vcycle_2d(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv,
        at::Tensor r_out,                 // pre-allocated (Nx, Ny)
        double jcap_tol, double w, int64_t nsmoothing,
        int64_t smoother_id, bool variational = false)
{
    auto opts = p.options();
    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);

    // Face coefficient slices.  ch is (Nx+1, Ny), cv is (Nx, Ny+1).
    auto cp0 = ch.slice(0, 1, Nx + 1).contiguous();
    auto cm0 = ch.slice(0, 0, Nx    ).contiguous();
    auto cp1 = cv.slice(1, 1, Ny + 1).contiguous();
    auto cm1 = cv.slice(1, 0, Ny    ).contiguous();

    // Pre-smooth + residual at this level.
    smooth_2d(p, f, cp0, cm0, cp1, cm1, jcap_tol, w, nsmoothing, smoother_id);
    mg_residual_2d_cuda(p, f, cp0, cm0, cp1, cm1, jcap_tol, r_out);

    if (Nx > 2 && Ny > 2) {
        const int Nx_c = Nx / 2;
        const int Ny_c = Ny / 2;

        using namespace poisson_scratch;

        // Restricted face arrays (persistent, reused across solves)
        auto& ch_c = scratch("vcycle_ch_c", {Nx_c + 1, Ny_c}, opts);
        auto& cv_c = scratch("vcycle_cv_c", {Nx_c, Ny_c + 1}, opts);
        restrict_face_2d_cuda(ch, ch_c, /*face_dim=*/0);
        restrict_face_2d_cuda(cv, cv_c, /*face_dim=*/1);

        // Restricted residual → coarse RHS
        auto& r_c  = scratch("vcycle_r_c", {Nx_c, Ny_c}, opts);
        if (variational) restrict_fw_2d_cuda(r_out, r_c);       // P^T (symmetric)
        else             restrict_residual_2d_cuda(r_out, r_c); // sum-of-children

        // Coarse correction starts from zero, ghost-padded.
        auto& p_c  = scratch("vcycle_p_c", {Nx_c + 2, Ny_c + 2}, opts);
        p_c.zero_();
        auto& r_c2 = scratch("vcycle_r_c2", {Nx_c, Ny_c}, opts);  // coarse-level residual buffer

        vcycle_2d(p_c, r_c, ch_c, cv_c, r_c2,
                  jcap_tol, w, nsmoothing, smoother_id, variational);

        // Prolongate + add into fine p (in place).
        prolongate_add_2d_cuda(p_c, p);

        // Post-smooth + residual.  ``variational`` also reverses the RBGS
        // half-sweep order here, making the post-smooth the A-adjoint of the
        // pre-smooth — without it the cycle is non-symmetric under RBGS even
        // with R = P^T, and invalid as a CG preconditioner.
        smooth_2d(p, f, cp0, cm0, cp1, cm1, jcap_tol, w, nsmoothing,
                  smoother_id, /*reverse=*/variational);
        mg_residual_2d_cuda(p, f, cp0, cm0, cp1, cm1, jcap_tol, r_out);
    } else if (variational && smoother_id == 0) {
        // COARSEST LEVEL.  Every other level is symmetric because its
        // post-smooth is the reversed twin of its pre-smooth, but the bottom
        // of the V has no post-smooth at all — its "solve" is a bare
        // red-black sweep, which is NOT self-adjoint, and that alone keeps the
        // whole cycle asymmetric (measured ~5e-2).  Weighted Jacobi needs
        // nothing here: a bare Jacobi solve already is self-adjoint.
        smooth_2d(p, f, cp0, cm0, cp1, cm1, jcap_tol, w, nsmoothing,
                  smoother_id, /*reverse=*/true);
        mg_residual_2d_cuda(p, f, cp0, cm0, cp1, cm1, jcap_tol, r_out);
    }
}

// =====================================================================
// 3-D recursive V-cycle
// =====================================================================

// See the note on ``variational`` above vcycle_2d.
static void vcycle_3d(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv, at::Tensor cw,
        at::Tensor r_out,
        double jcap_tol, double w, int64_t nsmoothing,
        int64_t smoother_id, bool variational = false)
{
    auto opts = p.options();
    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);
    const int Nz = (int)f.size(2);

    auto cp0 = ch.slice(0, 1, Nx + 1).contiguous();
    auto cm0 = ch.slice(0, 0, Nx    ).contiguous();
    auto cp1 = cv.slice(1, 1, Ny + 1).contiguous();
    auto cm1 = cv.slice(1, 0, Ny    ).contiguous();
    auto cp2 = cw.slice(2, 1, Nz + 1).contiguous();
    auto cm2 = cw.slice(2, 0, Nz    ).contiguous();

    smooth_3d(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
              jcap_tol, w, nsmoothing, smoother_id);
    mg_residual_3d_cuda(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                        jcap_tol, r_out);

    if (Nx > 2 && Ny > 2 && Nz > 2) {
        const int Nx_c = Nx / 2;
        const int Ny_c = Ny / 2;
        const int Nz_c = Nz / 2;

        using namespace poisson_scratch;

        auto& ch_c = scratch("vcycle_ch_c", {Nx_c + 1, Ny_c, Nz_c}, opts);
        auto& cv_c = scratch("vcycle_cv_c", {Nx_c, Ny_c + 1, Nz_c}, opts);
        auto& cw_c = scratch("vcycle_cw_c", {Nx_c, Ny_c, Nz_c + 1}, opts);
        restrict_face_3d_cuda(ch, ch_c, 0);
        restrict_face_3d_cuda(cv, cv_c, 1);
        restrict_face_3d_cuda(cw, cw_c, 2);

        auto& r_c  = scratch("vcycle_r_c", {Nx_c, Ny_c, Nz_c}, opts);
        if (variational) restrict_fw_3d_cuda(r_out, r_c);       // P^T (symmetric)
        else             restrict_residual_3d_cuda(r_out, r_c); // sum-of-children

        auto& p_c  = scratch("vcycle_p_c", {Nx_c + 2, Ny_c + 2, Nz_c + 2}, opts);
        p_c.zero_();
        auto& r_c2 = scratch("vcycle_r_c2", {Nx_c, Ny_c, Nz_c}, opts);

        vcycle_3d(p_c, r_c, ch_c, cv_c, cw_c, r_c2,
                  jcap_tol, w, nsmoothing, smoother_id, variational);

        prolongate_add_3d_cuda(p_c, p);

        // Post-smooth: ``variational`` reverses the RBGS half-sweep order so
        // it is the A-adjoint of the pre-smooth (see vcycle_2d).
        smooth_3d(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                  jcap_tol, w, nsmoothing, smoother_id, /*reverse=*/variational);
        mg_residual_3d_cuda(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                            jcap_tol, r_out);
    } else if (variational && smoother_id == 0) {
        // Coarsest level — see the matching note in vcycle_2d.
        smooth_3d(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                  jcap_tol, w, nsmoothing, smoother_id, /*reverse=*/true);
        mg_residual_3d_cuda(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                            jcap_tol, r_out);
    }
}

// =====================================================================
// Neumann BC helper:  for each dim d, copy interior-adjacent slab into
// the ghost slab on both sides.  Matches PoissonSolver.BC().  Applied in
// dim order so the sequential copies fill the edge/corner ghosts too
// (the fused per-sweep BC in the smoother only refreshes the face ghosts
// the stencil reads, leaving corners stale).
// =====================================================================
using lilytorch_kernels::poisson::apply_neumann_bc_full;
using lilytorch_kernels::poisson::gauge_fix;

static inline void apply_neumann_bc(at::Tensor p) { apply_neumann_bc_full(p); }

// =====================================================================
// Range / null-space handling for the CG drivers (mgcg, rmgcg)
//
// ``mg_residual_*`` zeroes degenerate cells (|J| < jcap_tol), so the operator
// B has identically zero ROWS AND COLUMNS there: the indicator vector of every
// degenerate cell lies in null(B), alongside the usual Neumann constant.  The
// stationary multigrid driver never notices — it masks the residual too, so
// the null-space component enters neither an inner product nor the convergence
// test.  CG is not so lucky: it forms r·M⁻¹r and d·Bd every iteration, and a
// RHS component outside range(B) poisons alpha and beta, making the residual
// GROW with iteration count.  b = −h²·div(u*) has exactly such a component,
// because div(u*) does not vanish inside a BDIM solid.
//
// ``build_active_mask`` fills ``mask`` with 1 on live cells and 0 on
// degenerate ones (same |J| >= jcap_tol test the smoother and residual use).
// ``project_range`` applies it to b and then removes the mean over the LIVE
// cells.  Both steps are needed: masking alone leaves the constant, which the
// CG recurrence amplifies instead of ignoring.
//
// All device-side — the reductions accumulate in float64 via ``sum(kDouble)``
// without materialising a float64 copy, and the mean stays a 0-dim device
// tensor, so the drivers remain host-sync-free and CUDA-graph capturable.
//
// Caveat: if the live region is split into disconnected pools by a solid,
// null(B) holds one constant PER COMPONENT and removing the global mean only
// kills the aggregate.  The stationary driver has the same limitation.
// =====================================================================
static inline void build_active_mask(std::vector<at::Tensor> const& cf,
                                     at::Tensor mask, double jcap_tol)
{
    mask.copy_(cf[0]);
    for (size_t i = 1; i < cf.size(); ++i) mask.add_(cf[i]);
    mask.abs_().ge_(jcap_tol);          // -> 1.0 on live cells, 0.0 elsewhere
}

static inline void project_range(at::Tensor b, at::Tensor const& mask)
{
    b.mul_(mask);
    auto mean = (b.sum(at::kDouble)
                 / mask.sum(at::kDouble).clamp_min(1.0)).to(b.scalar_type());
    b.addcmul_(mask, mean, -1.0);
}

// =====================================================================
// Top-level entry points
// =====================================================================

// The drivers below return (residual, iterations performed).  NOTE: with
// ``tol < 0`` the early-exit test is skipped entirely (that is what keeps the
// solve host-sync-free and CUDA-graph capturable), so the loop always runs its
// full budget and the returned count is just ``max_vcycles`` / ``max_cycles``.
// It is informative only when ``tol >= 0``.
static std::tuple<at::Tensor, int64_t> poisson_solve_multigrid_2d_cuda(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv,
        double h2, double jcap_tol, double w,
        int64_t nsmoothing, int64_t max_vcycles,
        double tol, int64_t smoother_id)
{
    TORCH_CHECK(p.is_contiguous() && f.is_contiguous(),
                "poisson_solve_multigrid_2d: p and f must be contiguous");
    TORCH_CHECK(p.device().is_cuda(),
                "poisson_solve_multigrid_2d: tensors must be on CUDA");
    TORCH_CHECK(p.dim() == 2 && f.dim() == 2,
                "poisson_solve_multigrid_2d: p and f must be 2-D");
    TORCH_CHECK(ch.is_contiguous() && cv.is_contiguous(),
                "poisson_solve_multigrid_2d: ch/cv must be contiguous");

    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);
    TORCH_CHECK(p.size(0) == Nx + 2 && p.size(1) == Ny + 2,
                "poisson_solve_multigrid_2d: p must be (Nx+2, Ny+2)");

    using namespace poisson_scratch;
    auto opts = p.options();

    auto& f_scaled = scratch("mg2d_f_scaled", f.sizes(), opts);
    f_scaled.copy_(f).mul_(h2);
    auto& r = scratch("mg2d_r", f.sizes(), opts);

    int64_t niter = 0;
    for (int64_t i = 0; i < max_vcycles; ++i) {
        vcycle_2d(p, f_scaled, ch, cv, r,
                  jcap_tol, w, nsmoothing, smoother_id);
        niter = i + 1;
        // L∞ early-exit (one D→H sync per cycle — matches Python).
        const double rnorm = (tol >= 0.0) ? r.abs().max().item<double>() : 1.0;
        if (tol >= 0.0 && rnorm < tol) break;
    }

    // Ghost-ring gauge fix
    apply_neumann_bc(p);
    gauge_fix(p);
    return std::make_tuple(r, niter);
}

static std::tuple<at::Tensor, int64_t> poisson_solve_multigrid_3d_cuda(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv, at::Tensor cw,
        double h2, double jcap_tol, double w,
        int64_t nsmoothing, int64_t max_vcycles,
        double tol, int64_t smoother_id)
{
    TORCH_CHECK(p.is_contiguous() && f.is_contiguous(),
                "poisson_solve_multigrid_3d: p and f must be contiguous");
    TORCH_CHECK(p.device().is_cuda(),
                "poisson_solve_multigrid_3d: tensors must be on CUDA");
    TORCH_CHECK(p.dim() == 3 && f.dim() == 3,
                "poisson_solve_multigrid_3d: p and f must be 3-D");
    TORCH_CHECK(ch.is_contiguous() && cv.is_contiguous() && cw.is_contiguous(),
                "poisson_solve_multigrid_3d: ch/cv/cw must be contiguous");

    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);
    const int Nz = (int)f.size(2);
    TORCH_CHECK(p.size(0) == Nx + 2 && p.size(1) == Ny + 2 && p.size(2) == Nz + 2,
                "poisson_solve_multigrid_3d: p must be (Nx+2, Ny+2, Nz+2)");

    using namespace poisson_scratch;
    auto opts = p.options();

    auto& f_scaled = scratch("mg3d_f_scaled", f.sizes(), opts);
    f_scaled.copy_(f).mul_(h2);
    auto& r = scratch("mg3d_r", f.sizes(), opts);

    int64_t niter = 0;
    for (int64_t i = 0; i < max_vcycles; ++i) {
        vcycle_3d(p, f_scaled, ch, cv, cw, r,
                  jcap_tol, w, nsmoothing, smoother_id);
        niter = i + 1;
        const double rnorm = (tol >= 0.0) ? r.abs().max().item<double>() : 1.0;
        if (tol >= 0.0 && rnorm < tol) break;
    }

    // Ghost-ring gauge fix (see 2-D driver): refresh the full ghost ring incl.
    // corners before the gauge mean, which the per-sweep face-only BC leaves stale.
    apply_neumann_bc(p);
    gauge_fix(p);
    return std::make_tuple(r, niter);
}

// =====================================================================
// MGCG (multigrid-preconditioned conjugate gradient) driver
//
// SPD system:  B(x) = b  where  b = -h²·f,  B(p) = (J·p − sum) · active.
// B(p) is evaluated by re-using mg_residual_*_cuda with a zero-f buffer:
//   mg_residual: r = (f − sum + J·p)·active  ⇒  with f=0: r = B(p).
// (mg_residual itself calls BC internally — see multigrid_smoothers.cu —
//  so we don't need to apply BC before calling it.)
// =====================================================================

static std::tuple<at::Tensor, int64_t> poisson_solve_mgcg_2d_cuda(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv,
        double h2, double jcap_tol, double w,
        int64_t nsmoothing, int64_t max_cycles, int64_t precond_vcycles,
        double tol, int64_t smoother_id)
{
    TORCH_CHECK(p.is_contiguous() && f.is_contiguous(),
                "poisson_solve_mgcg_2d: p and f must be contiguous");
    TORCH_CHECK(p.device().is_cuda(), "poisson_solve_mgcg_2d: tensors must be on CUDA");
    TORCH_CHECK(p.dim() == 2 && f.dim() == 2, "poisson_solve_mgcg_2d: p and f must be 2-D");

    auto opts = p.options();
    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);
    TORCH_CHECK(p.size(0) == Nx + 2 && p.size(1) == Ny + 2,
                "poisson_solve_mgcg_2d: p must be (Nx+2, Ny+2)");

    // Face coefficient slices (fine level).
    auto cp0 = ch.slice(0, 1, Nx + 1).contiguous();
    auto cm0 = ch.slice(0, 0, Nx    ).contiguous();
    auto cp1 = cv.slice(1, 1, Ny + 1).contiguous();
    auto cm1 = cv.slice(1, 0, Ny    ).contiguous();

    using namespace poisson_scratch;

    // b = −h²·f  (interior) — persistent scratch, filled every solve
    auto& b = scratch("mgcg2d_b", {Nx, Ny}, opts);
    b.copy_(f).mul_(-h2);

    // Project b onto range(B) and keep the mask for the preconditioner.
    auto& mask = scratch("mgcg2d_mask", {Nx, Ny}, opts);
    build_active_mask({cp0, cm0, cp1, cm1}, mask, jcap_tol);
    project_range(b, mask);

    // x = p (already passed in), apply BC
    apply_neumann_bc(p);

    // Persistent zero-f buffer — allocated once, never written (read-only).
    auto& f_zero = zero_buffer({Nx, Ny}, opts);

    // r = b − B(x).  mg_residual with f_zero gives B(x) directly.
    auto& Bx = scratch("mgcg2d_Bx", {Nx, Ny}, opts);
    mg_residual_2d_cuda(p, f_zero, cp0, cm0, cp1, cm1, jcap_tol, Bx);
    auto& r = scratch("mgcg2d_r", {Nx, Ny}, opts);
    r.copy_(b).sub_(Bx);

    double r_norm = (tol >= 0.0) ? r.abs().max().item<double>() : 1.0;
    if (r_norm < tol) {
        gauge_fix(p);
        return std::make_tuple(r, (int64_t)0);
    }

    // Preconditioner buffers (persistent).
    auto& z     = scratch("mgcg2d_z", {Nx + 2, Ny + 2}, opts);
    z.zero_();
    auto& r_buf = scratch("mgcg2d_r_buf", {Nx, Ny}, opts);   // V-cycle residual scratch

    using namespace torch::indexing;
    auto z_in = z.index({Slice(1, -1), Slice(1, -1)});

    // V-cycle solves (S − Jp) p = f_arg, i.e. −B(p) = f_arg, so pass −r
    // to get B(z) ≈ r.  Use in-place neg to avoid a temporary allocation.
    // Masking z afterwards is what makes M symmetric: the V-cycle zeroes the
    // residual on degenerate cells on the way down (mg_residual_*) but
    // prolongate_add writes the coarse correction into EVERY fine cell on the
    // way up — a zero column against a non-zero row.
    r.neg_();
    for (int64_t i = 0; i < precond_vcycles; ++i)
        vcycle_2d(z, r, ch, cv, r_buf, jcap_tol, w, nsmoothing, smoother_id, /*variational=*/true);
    r.neg_();  // restore
    z_in.mul_(mask);

    auto& d = scratch("mgcg2d_d", {Nx + 2, Ny + 2}, opts);
    d.copy_(z);
    apply_neumann_bc(d);

    auto d_in = d.index({Slice(1, -1), Slice(1, -1)});
    auto x_in = p.index({Slice(1, -1), Slice(1, -1)});

    auto rz = (r * z_in).to(at::kDouble).sum();   // scalar tensor (f64)
    auto& q  = scratch("mgcg2d_q", {Nx, Ny}, opts);

    int64_t niter = 0;
    for (int64_t k = 0; k < max_cycles; ++k) {
        // q = B(d).  mg_residual applies its own BC to d (read-only inside).
        // d has BC applied above / at end of previous iter.
        mg_residual_2d_cuda(d, f_zero, cp0, cm0, cp1, cm1, jcap_tol, q);

        auto dq = (d_in * q).to(at::kDouble).sum();
        auto alpha = (rz / dq).to(p.scalar_type());

        // Pipelined CG: keep alpha/beta as 0-dim device scalars and fuse the
        // axpy updates with addcmul_/mul_ (broadcast, no extra temps).  This
        // removes the per-iter alpha/beta D→H syncs; the only host sync left
        // is the residual-norm convergence check below (~1 sync/iter).
        x_in.addcmul_(d_in, alpha);            // x += alpha·d   (no D→H copy)
        apply_neumann_bc(p);
        r.addcmul_(q, alpha, -1.0);            // r -= alpha·q   (no D→H copy)
        niter = k + 1;

        // tol < 0  ⇒  no early-exit: skip the residual-norm D→H sync entirely
        // (short-circuit avoids .item()) so the whole solve is host-sync-free
        // and CUDA-graph capturable.  Runs the full max_cycles budget.
        double rn = (tol >= 0.0) ? r.abs().max().item<double>() : 1.0;
        if (tol >= 0.0 && rn < tol) break;

        z.zero_();
        r.neg_(); // reuse buffer: r → −r
        for (int64_t i = 0; i < precond_vcycles; ++i)
            vcycle_2d(z, r, ch, cv, r_buf, jcap_tol, w, nsmoothing, smoother_id, /*variational=*/true);
        r.neg_(); // restore
        z_in.mul_(mask);

        auto rz_new = (r * z_in).to(at::kDouble).sum();
        auto beta = (rz_new / rz).to(p.scalar_type());
        // d[in] = z[in] + beta * d[in]
        d_in.mul_(beta).add_(z_in);
        apply_neumann_bc(d);
        rz = rz_new;
    }

    gauge_fix(p);
    return std::make_tuple(r, niter);
}

static std::tuple<at::Tensor, int64_t> poisson_solve_mgcg_3d_cuda(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv, at::Tensor cw,
        double h2, double jcap_tol, double w,
        int64_t nsmoothing, int64_t max_cycles, int64_t precond_vcycles,
        double tol, int64_t smoother_id)
{
    TORCH_CHECK(p.is_contiguous() && f.is_contiguous(),
                "poisson_solve_mgcg_3d: p and f must be contiguous");
    TORCH_CHECK(p.device().is_cuda(), "poisson_solve_mgcg_3d: tensors must be on CUDA");
    TORCH_CHECK(p.dim() == 3 && f.dim() == 3, "poisson_solve_mgcg_3d: p and f must be 3-D");

    auto opts = p.options();
    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);
    const int Nz = (int)f.size(2);
    TORCH_CHECK(p.size(0) == Nx + 2 && p.size(1) == Ny + 2 && p.size(2) == Nz + 2,
                "poisson_solve_mgcg_3d: p must be (Nx+2, Ny+2, Nz+2)");

    auto cp0 = ch.slice(0, 1, Nx + 1).contiguous();
    auto cm0 = ch.slice(0, 0, Nx    ).contiguous();
    auto cp1 = cv.slice(1, 1, Ny + 1).contiguous();
    auto cm1 = cv.slice(1, 0, Ny    ).contiguous();
    auto cp2 = cw.slice(2, 1, Nz + 1).contiguous();
    auto cm2 = cw.slice(2, 0, Nz    ).contiguous();

    using namespace poisson_scratch;

    // b = −h²·f  (interior) — persistent scratch, filled every solve
    auto& b = scratch("mgcg3d_b", {Nx, Ny, Nz}, opts);
    b.copy_(f).mul_(-h2);

    // Project b onto range(B) and keep the mask for the preconditioner.
    auto& mask = scratch("mgcg3d_mask", {Nx, Ny, Nz}, opts);
    build_active_mask({cp0, cm0, cp1, cm1, cp2, cm2}, mask, jcap_tol);
    project_range(b, mask);

    apply_neumann_bc(p);

    // Persistent zero-f buffer — allocated once, never written (read-only).
    auto& f_zero = zero_buffer({Nx, Ny, Nz}, opts);
    auto& Bx = scratch("mgcg3d_Bx", {Nx, Ny, Nz}, opts);
    mg_residual_3d_cuda(p, f_zero, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, Bx);
    auto& r = scratch("mgcg3d_r", {Nx, Ny, Nz}, opts);
    r.copy_(b).sub_(Bx);

    double r_norm = (tol >= 0.0) ? r.abs().max().item<double>() : 1.0;
    if (r_norm < tol) {
        gauge_fix(p);
        return std::make_tuple(r, (int64_t)0);
    }

    auto& z     = scratch("mgcg3d_z", {Nx + 2, Ny + 2, Nz + 2}, opts);
    z.zero_();
    auto& r_buf = scratch("mgcg3d_r_buf", {Nx, Ny, Nz}, opts);

    using namespace torch::indexing;
    auto z_in = z.index({Slice(1, -1), Slice(1, -1), Slice(1, -1)});

    // In-place neg to avoid temporary allocation (same pattern as inner loop).
    // z is masked afterwards to keep M symmetric — see build_active_mask.
    r.neg_();
    for (int64_t i = 0; i < precond_vcycles; ++i)
        vcycle_3d(z, r, ch, cv, cw, r_buf, jcap_tol, w, nsmoothing, smoother_id, /*variational=*/true);
    r.neg_();  // restore
    z_in.mul_(mask);

    auto& d = scratch("mgcg3d_d", {Nx + 2, Ny + 2, Nz + 2}, opts);
    d.copy_(z);
    apply_neumann_bc(d);

    auto d_in = d.index({Slice(1, -1), Slice(1, -1), Slice(1, -1)});
    auto x_in = p.index({Slice(1, -1), Slice(1, -1), Slice(1, -1)});

    auto rz = (r * z_in).to(at::kDouble).sum();
    auto& q  = scratch("mgcg3d_q", {Nx, Ny, Nz}, opts);

    int64_t niter = 0;
    for (int64_t k = 0; k < max_cycles; ++k) {
        mg_residual_3d_cuda(d, f_zero, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, q);

        auto dq = (d_in * q).to(at::kDouble).sum();
        auto alpha = (rz / dq).to(p.scalar_type());

        x_in.addcmul_(d_in, alpha);
        apply_neumann_bc(p);
        r.addcmul_(q, alpha, -1.0);
        niter = k + 1;

        // tol < 0  ⇒  no early-exit: skip the residual-norm D→H sync entirely
        // (short-circuit avoids .item()) so the whole solve is host-sync-free
        // and CUDA-graph capturable.  Runs the full max_cycles budget.
        double rn = (tol >= 0.0) ? r.abs().max().item<double>() : 1.0;
        if (tol >= 0.0 && rn < tol) break;

        z.zero_();
        r.neg_();
        for (int64_t i = 0; i < precond_vcycles; ++i)
            vcycle_3d(z, r, ch, cv, cw, r_buf, jcap_tol, w, nsmoothing, smoother_id, /*variational=*/true);
        r.neg_();
        z_in.mul_(mask);

        auto rz_new = (r * z_in).to(at::kDouble).sum();
        auto beta = (rz_new / rz).to(p.scalar_type());
        d_in.mul_(beta).add_(z_in);
        apply_neumann_bc(d);
        rz = rz_new;
    }

    gauge_fix(p);
    return std::make_tuple(r, niter);
}

// =====================================================================
// RMGCG — recycled (deflated) MGCG.  Identical to the MGCG drivers above
// plus:
//   * deflation: a B-orthonormal recycle basis U (kdef, full grid) with
//     W = B·U (kdef, interior) is projected out of the initial residual
//     (Galerkin) and out of every search direction (B-orthogonalisation).
//     kdef == 0 → behaves exactly like plain MGCG.
//   * harvesting: the last ``harvest_k`` search directions are written
//     into D (harvest_k, full grid) as a ring buffer, for the Python
//     driver to refresh the recycle space.
// Returns (r, D, niter).  All deflation math is batched ATen (no per-vector
// host syncs); alpha/beta stay on-device (fused addcmul_/mul_ updates), so the
// only host sync per iteration is the residual-norm check — exactly as MGCG.
// =====================================================================
static std::tuple<at::Tensor, at::Tensor, int64_t> poisson_solve_rmgcg_2d_cuda(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv,
        at::Tensor U, at::Tensor W,
        int64_t harvest_k,
        double h2, double jcap_tol, double w,
        int64_t nsmoothing, int64_t max_cycles, int64_t precond_vcycles,
        double tol, int64_t smoother_id)
{
    TORCH_CHECK(p.is_contiguous() && f.is_contiguous(),
                "poisson_solve_rmgcg_2d: p and f must be contiguous");
    TORCH_CHECK(p.device().is_cuda(), "poisson_solve_rmgcg_2d: tensors must be on CUDA");
    TORCH_CHECK(p.dim() == 2 && f.dim() == 2, "poisson_solve_rmgcg_2d: p and f must be 2-D");

    auto opts = p.options();
    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);
    const int64_t kdef = U.size(0);
    using namespace torch::indexing;

    auto cp0 = ch.slice(0, 1, Nx + 1).contiguous();
    auto cm0 = ch.slice(0, 0, Nx    ).contiguous();
    auto cp1 = cv.slice(1, 1, Ny + 1).contiguous();
    auto cm1 = cv.slice(1, 0, Ny    ).contiguous();

    using namespace poisson_scratch;

    // b = −h²·f  (interior) — persistent scratch, filled every solve
    auto& b = scratch("rmgcg2d_b", {Nx, Ny}, opts);
    b.copy_(f).mul_(-h2);

    // Project b onto range(B) and keep the mask for the preconditioner.
    auto& mask = scratch("rmgcg2d_mask", {Nx, Ny}, opts);
    build_active_mask({cp0, cm0, cp1, cm1}, mask, jcap_tol);
    project_range(b, mask);

    apply_neumann_bc(p);

    // Persistent zero-f buffer — allocated once, never written (read-only).
    auto& f_zero = zero_buffer({Nx, Ny}, opts);
    auto& Bx = scratch("rmgcg2d_Bx", {Nx, Ny}, opts);
    mg_residual_2d_cuda(p, f_zero, cp0, cm0, cp1, cm1, jcap_tol, Bx);
    auto& r = scratch("rmgcg2d_r", {Nx, Ny}, opts);
    r.copy_(b).sub_(Bx);

    auto x_in = p.index({Slice(1, -1), Slice(1, -1)});
    auto U_in = (kdef > 0) ? U.index({Slice(), Slice(1, -1), Slice(1, -1)}) : U;

    // ---- deflation init:  x += U Uᵀr ;  r -= W Uᵀr   (C = I) ----
    if (kdef > 0) {
        auto c = (U_in * r.unsqueeze(0)).to(at::kDouble).sum({1, 2})
                     .to(p.scalar_type());               // (kdef,)
        x_in.add_((c.view({kdef, 1, 1}) * U_in).sum(0));
        apply_neumann_bc(p);
        r.sub_((c.view({kdef, 1, 1}) * W).sum(0));
    }

    // Harvest buffer — key includes harvest_k to distinguish configs
    int64_t dk = std::max<int64_t>(harvest_k, 1);
    auto& D = scratch("rmgcg2d_D_hk" + std::to_string(dk),
                      {dk, Nx + 2, Ny + 2}, opts);
    D.zero_();
    int64_t niter = 0;

    double r_norm = (tol >= 0.0) ? r.abs().max().item<double>() : 1.0;
    if (r_norm < tol) {
        gauge_fix(p);
        return std::make_tuple(r, D, niter);
    }

    auto& z     = scratch("rmgcg2d_z", {Nx + 2, Ny + 2}, opts);
    z.zero_();
    auto& r_buf = scratch("rmgcg2d_r_buf", {Nx, Ny}, opts);

    auto z_in = z.index({Slice(1, -1), Slice(1, -1)});

    // In-place neg to avoid temporary allocation.  z is masked afterwards to
    // keep M symmetric — see build_active_mask.
    r.neg_();
    for (int64_t i = 0; i < precond_vcycles; ++i)
        vcycle_2d(z, r, ch, cv, r_buf, jcap_tol, w, nsmoothing, smoother_id, /*variational=*/true);
    r.neg_();  // restore
    z_in.mul_(mask);

    auto& d    = scratch("rmgcg2d_d", {Nx + 2, Ny + 2}, opts);
    d.copy_(z);
    auto d_in = d.index({Slice(1, -1), Slice(1, -1)});

    // proj d B-orthogonal to U:  d -= U (Wᵀz)
    if (kdef > 0) {
        auto nu = (W * z_in.unsqueeze(0)).to(at::kDouble).sum({1, 2})
                      .to(p.scalar_type());
        d_in.sub_((nu.view({kdef, 1, 1}) * U_in).sum(0));
    }
    apply_neumann_bc(d);

    auto rz = (r * z_in).to(at::kDouble).sum();
    auto& q  = scratch("rmgcg2d_q", {Nx, Ny}, opts);

    for (int64_t k = 0; k < max_cycles; ++k) {
        mg_residual_2d_cuda(d, f_zero, cp0, cm0, cp1, cm1, jcap_tol, q);
        auto dq = (d_in * q).to(at::kDouble).sum();
        auto alpha = (rz / dq).to(p.scalar_type());

        x_in.addcmul_(d_in, alpha);
        apply_neumann_bc(p);
        r.addcmul_(q, alpha, -1.0);

        if (harvest_k > 0) D.index({k % harvest_k}).copy_(d);
        niter = k + 1;

        // tol < 0  ⇒  no early-exit: skip the residual-norm D→H sync entirely
        // (short-circuit avoids .item()) so the whole solve is host-sync-free
        // and CUDA-graph capturable.  Runs the full max_cycles budget.
        double rn = (tol >= 0.0) ? r.abs().max().item<double>() : 1.0;
        if (tol >= 0.0 && rn < tol) break;

        z.zero_();
        r.neg_();
        for (int64_t i = 0; i < precond_vcycles; ++i)
            vcycle_2d(z, r, ch, cv, r_buf, jcap_tol, w, nsmoothing, smoother_id, /*variational=*/true);
        r.neg_();
        z_in.mul_(mask);

        auto rz_new = (r * z_in).to(at::kDouble).sum();
        auto beta = (rz_new / rz).to(p.scalar_type());
        d_in.mul_(beta).add_(z_in);
        if (kdef > 0) {
            auto nu = (W * z_in.unsqueeze(0)).to(at::kDouble).sum({1, 2})
                          .to(p.scalar_type());
            d_in.sub_((nu.view({kdef, 1, 1}) * U_in).sum(0));
        }
        apply_neumann_bc(d);
        rz = rz_new;
    }

    gauge_fix(p);
    return std::make_tuple(r, D, niter);
}

static std::tuple<at::Tensor, at::Tensor, int64_t> poisson_solve_rmgcg_3d_cuda(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv, at::Tensor cw,
        at::Tensor U, at::Tensor W,
        int64_t harvest_k,
        double h2, double jcap_tol, double w,
        int64_t nsmoothing, int64_t max_cycles, int64_t precond_vcycles,
        double tol, int64_t smoother_id)
{
    TORCH_CHECK(p.is_contiguous() && f.is_contiguous(),
                "poisson_solve_rmgcg_3d: p and f must be contiguous");
    TORCH_CHECK(p.device().is_cuda(), "poisson_solve_rmgcg_3d: tensors must be on CUDA");
    TORCH_CHECK(p.dim() == 3 && f.dim() == 3, "poisson_solve_rmgcg_3d: p and f must be 3-D");

    auto opts = p.options();
    const int Nx = (int)f.size(0);
    const int Ny = (int)f.size(1);
    const int Nz = (int)f.size(2);
    const int64_t kdef = U.size(0);
    using namespace torch::indexing;

    auto cp0 = ch.slice(0, 1, Nx + 1).contiguous();
    auto cm0 = ch.slice(0, 0, Nx    ).contiguous();
    auto cp1 = cv.slice(1, 1, Ny + 1).contiguous();
    auto cm1 = cv.slice(1, 0, Ny    ).contiguous();
    auto cp2 = cw.slice(2, 1, Nz + 1).contiguous();
    auto cm2 = cw.slice(2, 0, Nz    ).contiguous();

    using namespace poisson_scratch;

    // b = −h²·f  (interior) — persistent scratch, filled every solve
    auto& b = scratch("rmgcg3d_b", {Nx, Ny, Nz}, opts);
    b.copy_(f).mul_(-h2);

    // Project b onto range(B) and keep the mask for the preconditioner.
    auto& mask = scratch("rmgcg3d_mask", {Nx, Ny, Nz}, opts);
    build_active_mask({cp0, cm0, cp1, cm1, cp2, cm2}, mask, jcap_tol);
    project_range(b, mask);

    apply_neumann_bc(p);

    // Persistent zero-f buffer — allocated once, never written (read-only).
    auto& f_zero = zero_buffer({Nx, Ny, Nz}, opts);
    auto& Bx = scratch("rmgcg3d_Bx", {Nx, Ny, Nz}, opts);
    mg_residual_3d_cuda(p, f_zero, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, Bx);
    auto& r = scratch("rmgcg3d_r", {Nx, Ny, Nz}, opts);
    r.copy_(b).sub_(Bx);

    auto x_in = p.index({Slice(1, -1), Slice(1, -1), Slice(1, -1)});
    auto U_in = (kdef > 0)
        ? U.index({Slice(), Slice(1, -1), Slice(1, -1), Slice(1, -1)}) : U;

    if (kdef > 0) {
        auto c = (U_in * r.unsqueeze(0)).to(at::kDouble).sum({1, 2, 3})
                     .to(p.scalar_type());
        x_in.add_((c.view({kdef, 1, 1, 1}) * U_in).sum(0));
        apply_neumann_bc(p);
        r.sub_((c.view({kdef, 1, 1, 1}) * W).sum(0));
    }

    // Harvest buffer — key includes harvest_k to distinguish configs
    int64_t dk = std::max<int64_t>(harvest_k, 1);
    auto& D = scratch("rmgcg3d_D_hk" + std::to_string(dk),
                      {dk, Nx + 2, Ny + 2, Nz + 2}, opts);
    D.zero_();
    int64_t niter = 0;

    double r_norm = (tol >= 0.0) ? r.abs().max().item<double>() : 1.0;
    if (r_norm < tol) {
        gauge_fix(p);
        return std::make_tuple(r, D, niter);
    }

    auto& z     = scratch("rmgcg3d_z", {Nx + 2, Ny + 2, Nz + 2}, opts);
    z.zero_();
    auto& r_buf = scratch("rmgcg3d_r_buf", {Nx, Ny, Nz}, opts);

    auto z_in = z.index({Slice(1, -1), Slice(1, -1), Slice(1, -1)});

    // In-place neg to avoid temporary allocation.  z is masked afterwards to
    // keep M symmetric — see build_active_mask.
    r.neg_();
    for (int64_t i = 0; i < precond_vcycles; ++i)
        vcycle_3d(z, r, ch, cv, cw, r_buf, jcap_tol, w, nsmoothing, smoother_id, /*variational=*/true);
    r.neg_();  // restore
    z_in.mul_(mask);

    auto& d    = scratch("rmgcg3d_d", {Nx + 2, Ny + 2, Nz + 2}, opts);
    d.copy_(z);
    auto d_in = d.index({Slice(1, -1), Slice(1, -1), Slice(1, -1)});

    if (kdef > 0) {
        auto nu = (W * z_in.unsqueeze(0)).to(at::kDouble).sum({1, 2, 3})
                      .to(p.scalar_type());
        d_in.sub_((nu.view({kdef, 1, 1, 1}) * U_in).sum(0));
    }
    apply_neumann_bc(d);

    auto rz = (r * z_in).to(at::kDouble).sum();
    auto& q  = scratch("rmgcg3d_q", {Nx, Ny, Nz}, opts);

    for (int64_t k = 0; k < max_cycles; ++k) {
        mg_residual_3d_cuda(d, f_zero, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, q);
        auto dq = (d_in * q).to(at::kDouble).sum();
        auto alpha = (rz / dq).to(p.scalar_type());

        x_in.addcmul_(d_in, alpha);
        apply_neumann_bc(p);
        r.addcmul_(q, alpha, -1.0);

        if (harvest_k > 0) D.index({k % harvest_k}).copy_(d);
        niter = k + 1;

        // tol < 0  ⇒  no early-exit: skip the residual-norm D→H sync entirely
        // (short-circuit avoids .item()) so the whole solve is host-sync-free
        // and CUDA-graph capturable.  Runs the full max_cycles budget.
        double rn = (tol >= 0.0) ? r.abs().max().item<double>() : 1.0;
        if (tol >= 0.0 && rn < tol) break;

        z.zero_();
        r.neg_();
        for (int64_t i = 0; i < precond_vcycles; ++i)
            vcycle_3d(z, r, ch, cv, cw, r_buf, jcap_tol, w, nsmoothing, smoother_id, /*variational=*/true);
        r.neg_();
        z_in.mul_(mask);

        auto rz_new = (r * z_in).to(at::kDouble).sum();
        auto beta = (rz_new / rz).to(p.scalar_type());
        d_in.mul_(beta).add_(z_in);
        if (kdef > 0) {
            auto nu = (W * z_in.unsqueeze(0)).to(at::kDouble).sum({1, 2, 3})
                          .to(p.scalar_type());
            d_in.sub_((nu.view({kdef, 1, 1, 1}) * U_in).sum(0));
        }
        apply_neumann_bc(d);
        rz = rz_new;
    }

    gauge_fix(p);
    return std::make_tuple(r, D, niter);
}

// =====================================================================
// Raw V-cycle op — the MGCG preconditioner primitive.
//
// ``n_vcycles`` V-cycles on (p, f, faces) with NO gauge fix: no ghost-ring
// Neumann pass and no mean subtraction, unlike the whole-solve drivers above.
// That is exactly what a PCG preconditioner needs — the internal
// ``vcycle_{2,3}d`` the native MGCG driver already applies to ``z`` — and it
// is what the Python CG driver (``PoissonSolver._cg_core``) calls on the CPU,
// where the MGCG/RMGCG whole-solve C++ twins do not exist.
//
// ``f`` is consumed as the raw smoother RHS (already h²-scaled by the caller).
// ``p`` (ghost-padded) is mutated in place; the interior residual is returned.
// =====================================================================

static at::Tensor mg_vcycle_2d_cuda(
        at::Tensor p, at::Tensor f, at::Tensor ch, at::Tensor cv,
        double jcap_tol, double w,
        int64_t nsmoothing, int64_t n_vcycles, int64_t smoother_id)
{
    using namespace poisson_scratch;
    auto opts = f.options();
    auto& r = scratch("mg_vcycle2d_r", f.sizes(), opts);
    for (int64_t i = 0; i < n_vcycles; ++i)
        vcycle_2d(p, f, ch, cv, r, jcap_tol, w, nsmoothing, smoother_id, /*variational=*/true);
    return r;
}

static at::Tensor mg_vcycle_3d_cuda(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv, at::Tensor cw,
        double jcap_tol, double w,
        int64_t nsmoothing, int64_t n_vcycles, int64_t smoother_id)
{
    using namespace poisson_scratch;
    auto opts = f.options();
    auto& r = scratch("mg_vcycle3d_r", f.sizes(), opts);
    for (int64_t i = 0; i < n_vcycles; ++i)
        vcycle_3d(p, f, ch, cv, cw, r, jcap_tol, w, nsmoothing, smoother_id, /*variational=*/true);
    return r;
}

// ---- CUDA dispatch registration -------------------------------------
TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("mg_vcycle_2d", &mg_vcycle_2d_cuda);
    m.impl("mg_vcycle_3d", &mg_vcycle_3d_cuda);
    m.impl("poisson_solve_multigrid_2d", &poisson_solve_multigrid_2d_cuda);
    m.impl("poisson_solve_multigrid_3d", &poisson_solve_multigrid_3d_cuda);
    m.impl("poisson_solve_mgcg_2d", &poisson_solve_mgcg_2d_cuda);
    m.impl("poisson_solve_mgcg_3d", &poisson_solve_mgcg_3d_cuda);
    m.impl("poisson_solve_rmgcg_2d", &poisson_solve_rmgcg_2d_cuda);
    m.impl("poisson_solve_rmgcg_3d", &poisson_solve_rmgcg_3d_cuda);
}

}  // namespace lilytorch_kernels
