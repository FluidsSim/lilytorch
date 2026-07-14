// =====================================================================
//  poisson_gauge.h — the ghost-ring refresh and the gauge fix that every
//  whole-solve Poisson driver ends with, shared by the CUDA drivers
//  (cuda/poisson_solve.cu) and the CPU twins (multigrid_cpu.cpp).
//
//  Both are at::Tensor-level (once per solve, not hot), so a single
//  definition is used by both backends — they cannot drift apart.
//
//  ── Why the gauge is interior-only ─────────────────────────────────
//  The drivers used to gauge on ``p.mean()`` over the FULL ghost-padded
//  tensor.  Nothing reads p's edge/corner ghosts — the Poisson stencil is
//  5/7-point and the projection only ever takes face differences of p, so
//  unlike the *velocity* ghosts (which the wide advection stencil does
//  read — see bc_ops.h) these really are dead.  But they hold
//  backend-dependent garbage: the CUDA
//  Jacobi's ping-pong memcpys a zeroed scratch buffer over p when
//  nsmoothing is odd, zeroing them, while the CPU twin leaves whatever
//  was there.  So the gauge constant itself was backend-dependent and
//  CPU and CUDA returned pressures differing by a CONSTANT.  Harmless
//  while only ∇p is consumed, but it forced every cross-backend p
//  comparison to special-case it and it is a landmine for anyone who
//  later reads p absolutely (a pressure probe, a Bernoulli check, a Cp).
//
//  Gauging on the interior makes the constant a function of the solution
//  alone.  The subtraction is applied to the WHOLE tensor, which leaves
//  the Neumann ring consistent (a ghost and its interior neighbour shift
//  by the same constant) — so no re-BC afterwards, provided the caller
//  refreshed the ring first.
// =====================================================================
#pragma once

#include <ATen/ATen.h>

namespace lilytorch_kernels {
namespace poisson {

// Full ghost-ring Neumann refresh: for each dim, copy the interior-adjacent
// slab into the ghost slab on both sides.  Applied in dim order over FULL
// slabs, so the later dims' copies pick up the earlier dims' writes and the
// edge/corner ghosts get filled too (the per-sweep BC inside the smoothers
// only refreshes the face ghosts the stencil reads, and leaves them stale).
// Matches poisson_mult.PoissonSolver.BC().
inline void apply_neumann_bc_full(at::Tensor p)
{
    const int64_t nd = p.dim();
    for (int64_t d = 0; d < nd; ++d) {
        const int64_t L = p.size(d);
        p.select(d, 0).copy_(p.select(d, 1));
        p.select(d, L - 1).copy_(p.select(d, L - 2));
    }
}

// Neumann gauge fix: subtract the INTERIOR mean (float64 accumulation, as in
// Python: ``p -= p.to(f64).mean().to(p.dtype)``).  Call with a consistent
// ghost ring — i.e. after apply_neumann_bc_full() or an equivalent BC pass.
inline void gauge_fix(at::Tensor p)
{
    at::Tensor interior = p;
    for (int64_t d = 0; d < p.dim(); ++d)
        interior = interior.slice(d, 1, interior.size(d) - 1);
    auto pmean = interior.to(at::kDouble).mean();
    p.sub_(pmean.to(p.scalar_type()));
}

}  // namespace poisson
}  // namespace lilytorch_kernels
