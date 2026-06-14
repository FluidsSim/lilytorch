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
//  Returns the final residual r (interior shape).
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

static inline void smooth_2d(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        double jcap_tol, double w, int64_t nsmoothing,
        int64_t smoother_id)
{
    if (smoother_id == 0)
        rbgs_sweep_2d_cuda(p, f, cp0, cm0, cp1, cm1, jcap_tol, nsmoothing);
    else
        jacobi_sweep_2d_cuda(p, f, cp0, cm0, cp1, cm1, jcap_tol, w, nsmoothing);
}

static inline void smooth_3d(
        at::Tensor p, at::Tensor f,
        at::Tensor cp0, at::Tensor cm0,
        at::Tensor cp1, at::Tensor cm1,
        at::Tensor cp2, at::Tensor cm2,
        double jcap_tol, double w, int64_t nsmoothing,
        int64_t smoother_id)
{
    if (smoother_id == 0)
        rbgs_sweep_3d_cuda(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                           jcap_tol, nsmoothing);
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

static void vcycle_2d(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv,
        at::Tensor r_out,                 // pre-allocated (Nx, Ny)
        double jcap_tol, double w, int64_t nsmoothing,
        int64_t smoother_id)
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
        // Restricted face arrays
        auto ch_c = at::empty({Nx_c + 1, Ny_c}, opts);
        auto cv_c = at::empty({Nx_c, Ny_c + 1}, opts);
        restrict_face_2d_cuda(ch, ch_c, /*face_dim=*/0);
        restrict_face_2d_cuda(cv, cv_c, /*face_dim=*/1);

        // Restricted residual → coarse RHS
        auto r_c = at::empty({Nx_c, Ny_c}, opts);
        restrict_residual_2d_cuda(r_out, r_c);

        // Coarse correction starts from zero, ghost-padded.
        auto p_c = at::zeros({Nx_c + 2, Ny_c + 2}, opts);
        auto r_c2 = at::empty({Nx_c, Ny_c}, opts);  // coarse-level residual buffer

        vcycle_2d(p_c, r_c, ch_c, cv_c, r_c2,
                  jcap_tol, w, nsmoothing, smoother_id);

        // Prolongate + add into fine p (in place).
        prolongate_add_2d_cuda(p_c, p);

        // Post-smooth + residual.
        smooth_2d(p, f, cp0, cm0, cp1, cm1, jcap_tol, w, nsmoothing, smoother_id);
        mg_residual_2d_cuda(p, f, cp0, cm0, cp1, cm1, jcap_tol, r_out);
    }
}

// =====================================================================
// 3-D recursive V-cycle
// =====================================================================

static void vcycle_3d(
        at::Tensor p, at::Tensor f,
        at::Tensor ch, at::Tensor cv, at::Tensor cw,
        at::Tensor r_out,
        double jcap_tol, double w, int64_t nsmoothing,
        int64_t smoother_id)
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

        auto ch_c = at::empty({Nx_c + 1, Ny_c, Nz_c}, opts);
        auto cv_c = at::empty({Nx_c, Ny_c + 1, Nz_c}, opts);
        auto cw_c = at::empty({Nx_c, Ny_c, Nz_c + 1}, opts);
        restrict_face_3d_cuda(ch, ch_c, 0);
        restrict_face_3d_cuda(cv, cv_c, 1);
        restrict_face_3d_cuda(cw, cw_c, 2);

        auto r_c  = at::empty({Nx_c, Ny_c, Nz_c}, opts);
        restrict_residual_3d_cuda(r_out, r_c);

        auto p_c  = at::zeros({Nx_c + 2, Ny_c + 2, Nz_c + 2}, opts);
        auto r_c2 = at::empty({Nx_c, Ny_c, Nz_c}, opts);

        vcycle_3d(p_c, r_c, ch_c, cv_c, cw_c, r_c2,
                  jcap_tol, w, nsmoothing, smoother_id);

        prolongate_add_3d_cuda(p_c, p);

        smooth_3d(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                  jcap_tol, w, nsmoothing, smoother_id);
        mg_residual_3d_cuda(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                            jcap_tol, r_out);
    }
}

// =====================================================================
// Top-level entry points
// =====================================================================

static at::Tensor poisson_solve_multigrid_2d_cuda(
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

    auto f_scaled = f.mul(h2);                   // single allocation, one launch
    auto r = at::empty_like(f_scaled);

    for (int64_t i = 0; i < max_vcycles; ++i) {
        vcycle_2d(p, f_scaled, ch, cv, r,
                  jcap_tol, w, nsmoothing, smoother_id);
        // L∞ early-exit (one D→H sync per cycle — matches Python).
        const double rnorm = r.abs().max().item<double>();
        if (rnorm < tol) break;
    }

    // float64 mean subtraction (matches Python: p -= p.to(f64).mean().to(p.dtype)).
    auto pmean = p.to(at::kDouble).mean();
    p.sub_(pmean.to(p.scalar_type()));
    return r;
}

static at::Tensor poisson_solve_multigrid_3d_cuda(
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

    auto f_scaled = f.mul(h2);
    auto r = at::empty_like(f_scaled);

    for (int64_t i = 0; i < max_vcycles; ++i) {
        vcycle_3d(p, f_scaled, ch, cv, cw, r,
                  jcap_tol, w, nsmoothing, smoother_id);
        const double rnorm = r.abs().max().item<double>();
        if (rnorm < tol) break;
    }

    auto pmean = p.to(at::kDouble).mean();
    p.sub_(pmean.to(p.scalar_type()));
    return r;
}

// =====================================================================
// Neumann BC helper:  for each dim d, copy interior-adjacent slab into
// the ghost slab on both sides.  Matches PoissonSolver.BC().
// =====================================================================
static inline void apply_neumann_bc(at::Tensor p)
{
    const int64_t nd = p.dim();
    for (int64_t d = 0; d < nd; ++d) {
        const int64_t L = p.size(d);
        p.select(d, 0).copy_(p.select(d, 1));
        p.select(d, L - 1).copy_(p.select(d, L - 2));
    }
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

static at::Tensor poisson_solve_mgcg_2d_cuda(
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

    // b = −h²·f  (interior)
    auto b = f.mul(-h2);

    // x = p (already passed in), apply BC
    apply_neumann_bc(p);

    // Zero-f buffer used to evaluate B(·) via mg_residual.
    auto f_zero = at::zeros({Nx, Ny}, opts);

    // r = b − B(x).  mg_residual with f_zero gives B(x) directly.
    auto Bx = at::empty({Nx, Ny}, opts);
    mg_residual_2d_cuda(p, f_zero, cp0, cm0, cp1, cm1, jcap_tol, Bx);
    auto r = b.sub(Bx);

    double r_norm = r.abs().max().item<double>();
    if (r_norm < tol) {
        auto pmean = p.to(at::kDouble).mean();
        p.sub_(pmean.to(p.scalar_type()));
        return r;
    }

    // Preconditioner buffers.
    auto z      = at::zeros({Nx + 2, Ny + 2}, opts);
    auto r_buf  = at::empty({Nx, Ny}, opts);   // V-cycle residual scratch

    // V-cycle solves (S − Jp) p = f_arg, i.e. −B(p) = f_arg, so pass −r
    // to get B(z) ≈ r.
    auto neg_r = r.neg();
    for (int64_t i = 0; i < precond_vcycles; ++i)
        vcycle_2d(z, neg_r, ch, cv, r_buf, jcap_tol, w, nsmoothing, smoother_id);

    auto d = z.clone();
    apply_neumann_bc(d);

    using namespace torch::indexing;
    auto d_in = d.index({Slice(1, -1), Slice(1, -1)});
    auto x_in = p.index({Slice(1, -1), Slice(1, -1)});
    auto z_in = z.index({Slice(1, -1), Slice(1, -1)});

    auto rz = (r * z_in).to(at::kDouble).sum();   // scalar tensor (f64)
    auto q  = at::empty({Nx, Ny}, opts);

    for (int64_t k = 0; k < max_cycles; ++k) {
        // q = B(d).  mg_residual applies its own BC to d (read-only inside).
        // d has BC applied above / at end of previous iter.
        mg_residual_2d_cuda(d, f_zero, cp0, cm0, cp1, cm1, jcap_tol, q);

        auto dq = (d_in * q).to(at::kDouble).sum();
        auto alpha = (rz / dq).to(p.scalar_type());

        x_in.add_(d_in, alpha.item());            // host scalar avoids extra kernels
        apply_neumann_bc(p);
        r.sub_(q, alpha.item());

        double rn = r.abs().max().item<double>();
        if (rn < tol) break;

        z.zero_();
        r.neg_(); // reuse buffer: r → −r
        for (int64_t i = 0; i < precond_vcycles; ++i)
            vcycle_2d(z, r, ch, cv, r_buf, jcap_tol, w, nsmoothing, smoother_id);
        r.neg_(); // restore

        auto rz_new = (r * z_in).to(at::kDouble).sum();
        auto beta = (rz_new / rz).to(p.scalar_type());
        // d[in] = z[in] + beta * d[in]
        d_in.mul_(beta.item()).add_(z_in);
        apply_neumann_bc(d);
        rz = rz_new;
    }

    auto pmean = p.to(at::kDouble).mean();
    p.sub_(pmean.to(p.scalar_type()));
    return r;
}

static at::Tensor poisson_solve_mgcg_3d_cuda(
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

    auto b = f.mul(-h2);
    apply_neumann_bc(p);

    auto f_zero = at::zeros({Nx, Ny, Nz}, opts);
    auto Bx = at::empty({Nx, Ny, Nz}, opts);
    mg_residual_3d_cuda(p, f_zero, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, Bx);
    auto r = b.sub(Bx);

    double r_norm = r.abs().max().item<double>();
    if (r_norm < tol) {
        auto pmean = p.to(at::kDouble).mean();
        p.sub_(pmean.to(p.scalar_type()));
        return r;
    }

    auto z     = at::zeros({Nx + 2, Ny + 2, Nz + 2}, opts);
    auto r_buf = at::empty({Nx, Ny, Nz}, opts);

    auto neg_r = r.neg();
    for (int64_t i = 0; i < precond_vcycles; ++i)
        vcycle_3d(z, neg_r, ch, cv, cw, r_buf, jcap_tol, w, nsmoothing, smoother_id);

    auto d = z.clone();
    apply_neumann_bc(d);

    using namespace torch::indexing;
    auto d_in = d.index({Slice(1, -1), Slice(1, -1), Slice(1, -1)});
    auto x_in = p.index({Slice(1, -1), Slice(1, -1), Slice(1, -1)});
    auto z_in = z.index({Slice(1, -1), Slice(1, -1), Slice(1, -1)});

    auto rz = (r * z_in).to(at::kDouble).sum();
    auto q  = at::empty({Nx, Ny, Nz}, opts);

    for (int64_t k = 0; k < max_cycles; ++k) {
        mg_residual_3d_cuda(d, f_zero, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, q);

        auto dq = (d_in * q).to(at::kDouble).sum();
        auto alpha = (rz / dq).to(p.scalar_type());

        x_in.add_(d_in, alpha.item());
        apply_neumann_bc(p);
        r.sub_(q, alpha.item());

        double rn = r.abs().max().item<double>();
        if (rn < tol) break;

        z.zero_();
        r.neg_();
        for (int64_t i = 0; i < precond_vcycles; ++i)
            vcycle_3d(z, r, ch, cv, cw, r_buf, jcap_tol, w, nsmoothing, smoother_id);
        r.neg_();

        auto rz_new = (r * z_in).to(at::kDouble).sum();
        auto beta = (rz_new / rz).to(p.scalar_type());
        d_in.mul_(beta.item()).add_(z_in);
        apply_neumann_bc(d);
        rz = rz_new;
    }

    auto pmean = p.to(at::kDouble).mean();
    p.sub_(pmean.to(p.scalar_type()));
    return r;
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
// host syncs); only alpha/beta/residual-norm sync, exactly as MGCG.
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

    auto b = f.mul(-h2);
    apply_neumann_bc(p);

    auto f_zero = at::zeros({Nx, Ny}, opts);
    auto Bx = at::empty({Nx, Ny}, opts);
    mg_residual_2d_cuda(p, f_zero, cp0, cm0, cp1, cm1, jcap_tol, Bx);
    auto r = b.sub(Bx);

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

    auto D = at::zeros({std::max<int64_t>(harvest_k, 1), Nx + 2, Ny + 2}, opts);
    int64_t niter = 0;

    double r_norm = r.abs().max().item<double>();
    if (r_norm < tol) {
        auto pmean = p.to(at::kDouble).mean();
        p.sub_(pmean.to(p.scalar_type()));
        return std::make_tuple(r, D, niter);
    }

    auto z     = at::zeros({Nx + 2, Ny + 2}, opts);
    auto r_buf = at::empty({Nx, Ny}, opts);
    auto neg_r = r.neg();
    for (int64_t i = 0; i < precond_vcycles; ++i)
        vcycle_2d(z, neg_r, ch, cv, r_buf, jcap_tol, w, nsmoothing, smoother_id);

    auto d    = z.clone();
    auto d_in = d.index({Slice(1, -1), Slice(1, -1)});
    auto z_in = z.index({Slice(1, -1), Slice(1, -1)});

    // proj d B-orthogonal to U:  d -= U (Wᵀz)
    if (kdef > 0) {
        auto nu = (W * z_in.unsqueeze(0)).to(at::kDouble).sum({1, 2})
                      .to(p.scalar_type());
        d_in.sub_((nu.view({kdef, 1, 1}) * U_in).sum(0));
    }
    apply_neumann_bc(d);

    auto rz = (r * z_in).to(at::kDouble).sum();
    auto q  = at::empty({Nx, Ny}, opts);

    for (int64_t k = 0; k < max_cycles; ++k) {
        mg_residual_2d_cuda(d, f_zero, cp0, cm0, cp1, cm1, jcap_tol, q);
        auto dq = (d_in * q).to(at::kDouble).sum();
        auto alpha = (rz / dq).to(p.scalar_type());

        x_in.add_(d_in, alpha.item());
        apply_neumann_bc(p);
        r.sub_(q, alpha.item());

        if (harvest_k > 0) D.index({k % harvest_k}).copy_(d);
        niter = k + 1;

        double rn = r.abs().max().item<double>();
        if (rn < tol) break;

        z.zero_();
        r.neg_();
        for (int64_t i = 0; i < precond_vcycles; ++i)
            vcycle_2d(z, r, ch, cv, r_buf, jcap_tol, w, nsmoothing, smoother_id);
        r.neg_();

        auto rz_new = (r * z_in).to(at::kDouble).sum();
        auto beta = (rz_new / rz).to(p.scalar_type());
        d_in.mul_(beta.item()).add_(z_in);
        if (kdef > 0) {
            auto nu = (W * z_in.unsqueeze(0)).to(at::kDouble).sum({1, 2})
                          .to(p.scalar_type());
            d_in.sub_((nu.view({kdef, 1, 1}) * U_in).sum(0));
        }
        apply_neumann_bc(d);
        rz = rz_new;
    }

    auto pmean = p.to(at::kDouble).mean();
    p.sub_(pmean.to(p.scalar_type()));
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

    auto b = f.mul(-h2);
    apply_neumann_bc(p);

    auto f_zero = at::zeros({Nx, Ny, Nz}, opts);
    auto Bx = at::empty({Nx, Ny, Nz}, opts);
    mg_residual_3d_cuda(p, f_zero, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, Bx);
    auto r = b.sub(Bx);

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

    auto D = at::zeros({std::max<int64_t>(harvest_k, 1), Nx + 2, Ny + 2, Nz + 2}, opts);
    int64_t niter = 0;

    double r_norm = r.abs().max().item<double>();
    if (r_norm < tol) {
        auto pmean = p.to(at::kDouble).mean();
        p.sub_(pmean.to(p.scalar_type()));
        return std::make_tuple(r, D, niter);
    }

    auto z     = at::zeros({Nx + 2, Ny + 2, Nz + 2}, opts);
    auto r_buf = at::empty({Nx, Ny, Nz}, opts);
    auto neg_r = r.neg();
    for (int64_t i = 0; i < precond_vcycles; ++i)
        vcycle_3d(z, neg_r, ch, cv, cw, r_buf, jcap_tol, w, nsmoothing, smoother_id);

    auto d    = z.clone();
    auto d_in = d.index({Slice(1, -1), Slice(1, -1), Slice(1, -1)});
    auto z_in = z.index({Slice(1, -1), Slice(1, -1), Slice(1, -1)});

    if (kdef > 0) {
        auto nu = (W * z_in.unsqueeze(0)).to(at::kDouble).sum({1, 2, 3})
                      .to(p.scalar_type());
        d_in.sub_((nu.view({kdef, 1, 1, 1}) * U_in).sum(0));
    }
    apply_neumann_bc(d);

    auto rz = (r * z_in).to(at::kDouble).sum();
    auto q  = at::empty({Nx, Ny, Nz}, opts);

    for (int64_t k = 0; k < max_cycles; ++k) {
        mg_residual_3d_cuda(d, f_zero, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, q);
        auto dq = (d_in * q).to(at::kDouble).sum();
        auto alpha = (rz / dq).to(p.scalar_type());

        x_in.add_(d_in, alpha.item());
        apply_neumann_bc(p);
        r.sub_(q, alpha.item());

        if (harvest_k > 0) D.index({k % harvest_k}).copy_(d);
        niter = k + 1;

        double rn = r.abs().max().item<double>();
        if (rn < tol) break;

        z.zero_();
        r.neg_();
        for (int64_t i = 0; i < precond_vcycles; ++i)
            vcycle_3d(z, r, ch, cv, cw, r_buf, jcap_tol, w, nsmoothing, smoother_id);
        r.neg_();

        auto rz_new = (r * z_in).to(at::kDouble).sum();
        auto beta = (rz_new / rz).to(p.scalar_type());
        d_in.mul_(beta.item()).add_(z_in);
        if (kdef > 0) {
            auto nu = (W * z_in.unsqueeze(0)).to(at::kDouble).sum({1, 2, 3})
                          .to(p.scalar_type());
            d_in.sub_((nu.view({kdef, 1, 1, 1}) * U_in).sum(0));
        }
        apply_neumann_bc(d);
        rz = rz_new;
    }

    auto pmean = p.to(at::kDouble).mean();
    p.sub_(pmean.to(p.scalar_type()));
    return std::make_tuple(r, D, niter);
}

// ---- CUDA dispatch registration -------------------------------------
TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("poisson_solve_multigrid_2d", &poisson_solve_multigrid_2d_cuda);
    m.impl("poisson_solve_multigrid_3d", &poisson_solve_multigrid_3d_cuda);
    m.impl("poisson_solve_mgcg_2d", &poisson_solve_mgcg_2d_cuda);
    m.impl("poisson_solve_mgcg_3d", &poisson_solve_mgcg_3d_cuda);
    m.impl("poisson_solve_rmgcg_2d", &poisson_solve_rmgcg_2d_cuda);
    m.impl("poisson_solve_rmgcg_3d", &poisson_solve_rmgcg_3d_cuda);
}

}  // namespace lilytorch_kernels
