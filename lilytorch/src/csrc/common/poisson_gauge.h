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

// ── Per-face pressure BC ───────────────────────────────────────────────
// Face f = 2*d + side, side 0 = lo, 1 = hi, so the order is
// [xmin, xmax, ymin, ymax, zmin, zmax] — the SAME convention the velocity
// BCs use (advection.py::_build_bc_ops).  Bit f of the mask set => that
// face is homogeneous DIRICHLET (p = 0 on the face) instead of Neumann.
//
// Both are one ghost write, differing only in sign:
//     Neumann   (dp/dn = 0):  p_ghost = +p_interior
//     Dirichlet (p    = 0):   p_ghost = -p_interior
// because the face sits half a cell outside the last interior centre, so
// (p_ghost + p_interior)/2 = 0 is exactly p = 0 ON the face.
//
// The mask is a run-constant carried in a process-global rather than
// threaded through ~12 op schemas (multigrid / mgcg / rmgcg / residual /
// smoother, 2-D and 3-D).  It is set once from
// PoissonSolverMultigrid.__init__ via the ``poisson_set_pressure_bc_mask``
// op, read on the host at launch time, and baked into kernel args — so it
// is CUDA-graph safe as long as it does not change mid-run, which config
// cannot do.  The SAME mask applies at every multigrid level: coarsening
// preserves the domain faces, and the coarse-grid correction inherits the
// homogeneous form of the fine BC (Dirichlet -> correction 0 on the face).
inline int& pressure_bc_mask()
{
    static int mask = 0;
    return mask;
}

// +1 for a Neumann face, -1 for a Dirichlet one.
inline double pressure_bc_sign(int face)
{
    return (pressure_bc_mask() >> face) & 1 ? -1.0 : 1.0;
}

// Full ghost-ring refresh: for each dim, copy the interior-adjacent slab into
// the ghost slab on both sides, negating it on Dirichlet faces.  Applied in
// dim order over FULL slabs, so the later dims' copies pick up the earlier
// dims' writes and the edge/corner ghosts get filled too (the per-sweep BC
// inside the smoothers only refreshes the face ghosts the stencil reads, and
// leaves them stale).  Matches poisson_mult.PoissonSolver.BC().
inline void apply_neumann_bc_full(at::Tensor p)
{
    const int64_t nd = p.dim();
    const int m = pressure_bc_mask();
    for (int64_t d = 0; d < nd; ++d) {
        const int64_t L = p.size(d);
        const bool dlo = (m >> (2 * d + 0)) & 1;
        const bool dhi = (m >> (2 * d + 1)) & 1;
        if (dlo) p.select(d, 0).copy_(p.select(d, 1).neg());
        else     p.select(d, 0).copy_(p.select(d, 1));
        if (dhi) p.select(d, L - 1).copy_(p.select(d, L - 2).neg());
        else     p.select(d, L - 1).copy_(p.select(d, L - 2));
    }
}

// Gauge fix: subtract the INTERIOR mean (float64 accumulation, as in Python:
// ``p -= p.to(f64).mean().to(p.dtype)``).  Call with a consistent ghost ring —
// i.e. after apply_neumann_bc_full() or an equivalent BC pass.
//
// ONLY valid for an all-Neumann problem, where the constant is a genuine null
// direction.  With any Dirichlet face the constant is NOT in the null space —
// the solution is unique and already anchored — and subtracting the mean would
// shift it off the boundary condition it was just solved against.  Callers must
// skip this whenever pressure_bc_mask() != 0.
inline void gauge_fix(at::Tensor p)
{
    // Self-guarding: every driver calls this unconditionally at the end of a
    // solve, so the Dirichlet check lives here rather than at ~12 call sites.
    if (pressure_bc_mask() != 0) return;
    at::Tensor interior = p;
    for (int64_t d = 0; d < p.dim(); ++d)
        interior = interior.slice(d, 1, interior.size(d) - 1);
    auto pmean = interior.to(at::kDouble).mean();
    p.sub_(pmean.to(p.scalar_type()));
}

}  // namespace poisson
}  // namespace lilytorch_kernels
