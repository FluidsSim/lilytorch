// =====================================================================
//  bc_ops.h — descriptor decode + write-ownership rule for
//  ``apply_bcs_2d`` / ``apply_bcs_3d``, shared by the CUDA kernels
//  (cuda/streaming_sdf{,_2d}.cu) and their CPU twins
//  (streaming_sdf_cpu{,_2d}.cpp).
//
//  ── Why this exists ────────────────────────────────────────────────
//  Each BC op writes one ghost plane (3-D) / line (2-D) of one velocity
//  component.  A cell that lies on TWO boundary planes (an edge/corner
//  ghost) is therefore in the destination set of two different ops.  The
//  CUDA kernel used to run every op concurrently (one per blockIdx.z),
//  so those cells took two concurrent writes from two different source
//  cells and the winner was whichever block landed last — the solver was
//  bit-irreproducible from step 1, and disagreed with the (sequential,
//  deterministic) CPU twin on exactly those cells.
//
//  This is NOT confined to unread cells.  The wide / cross-term advection
//  stencil DOES read edge and corner ghosts: perturb only those cells and
//  a single step moves the INTERIOR velocity by ~8e-3 (3-D).  So the race
//  fed schedule-dependent garbage into the interior — CUDA's 3-D
//  all-Neumann interior disagreed with the CPU twin by 2.2e-3 over 50
//  steps; with the rule below they agree to 3e-15.
//
//  ── The rule ───────────────────────────────────────────────────────
//  Ops are executed in three ORDERED stages — Neumann, then Dirichlet,
//  then reflective (separate kernel launches on CUDA, separate loops on
//  CPU).  That is the order the eager Python reference applies them in,
//  so every cross-kind overlap keeps the value it has today.  Within a
//  stage:
//
//    * OWNERSHIP.  A cell claimed by several ops of the stage is written
//      only by the LOWEST-INDEXED of them (which, with the packing in
//      AdvDiffSolver._pack_bc_descriptors_3d, is the one on the lowest
//      axis).  One writer per cell ⇒ no write-write race.
//
//    * COMPOSED SOURCE.  The owner reads its source cell stepped inward
//      along its own axis AND along every other axis whose same-stage op
//      also claims the cell.  Stepping inward on every claimed axis lands
//      on a cell no op of this stage writes (destinations are ghost planes
//      0 / n-1; sources are 1 / n-2), so the read cannot race a write
//      either.  For the all-Neumann case this reproduces exactly what the
//      old sequential CPU loop produced (the last op read a cell the
//      earlier ops had already stepped inward), so those cells do not move.
//
//  Cross-stage reads (a reflective op reading a cell the Dirichlet stage
//  wrote) are ordered by the stage boundary, as before.
//
//  Both backends run this same rule, so they now agree cell-for-cell,
//  dead ghosts included.
// =====================================================================
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define LT_BC_FN __host__ __device__ __forceinline__
#else
#define LT_BC_FN inline
#endif

namespace lilytorch_kernels {
namespace bcs {

// Op kinds.  The stages run in this order.
enum : int {
    BC_KIND_NEUMANN    = 0,   // desc stride 3: (comp, axis, side)
    BC_KIND_DIRICHLET  = 1,   // desc stride 3: (comp, axis, signed offset)
    BC_KIND_REFLECTIVE = 2,   // desc stride 4: (comp, axis, dst_off, src_off)
};

LT_BC_FN int bc_desc_stride(const int kind) {
    return (kind == BC_KIND_REFLECTIVE) ? 4 : 3;
}

// Decode op ``op`` of a single-kind descriptor array.
//   shapes: int64[n_comp, ndim] — per-component extents.
//   dst / src: the plane index along ``axis`` (src unused for Dirichlet).
LT_BC_FN void bc_decode(
    const int kind, const int* __restrict__ desc, const int op,
    const std::int64_t* __restrict__ shapes, const int ndim,
    int& comp, int& axis, int& dst, int& src)
{
    const int s = bc_desc_stride(kind);
    comp = desc[op * s + 0];
    axis = desc[op * s + 1];
    const int sz = (int)shapes[comp * ndim + axis];

    if (kind == BC_KIND_NEUMANN) {
        const int side = desc[op * s + 2];          // 0 = lo, 1 = hi
        if (side == 0) { dst = 0;      src = 1;      }
        else           { dst = sz - 1; src = sz - 2; }
    } else if (kind == BC_KIND_DIRICHLET) {
        const int off = desc[op * s + 2];           // signed
        dst = (off >= 0) ? off : (sz + off);
        src = dst;                                   // unused
    } else {
        const int doff = desc[op * s + 2];
        const int soff = desc[op * s + 3];
        dst = (doff >= 0) ? doff : (sz + doff);
        src = (soff >= 0) ? soff : (sz + soff);
    }
}

// Ownership + composed source for cell ``c`` (which satisfies c[axis] == dst).
//
// Returns false if a lower-indexed op of the same stage also claims ``c`` —
// that op owns it and this one must not write.  On true, ``src_c`` is filled
// with the source cell: ``c`` stepped inward along ``axis`` and along every
// other axis whose same-stage op also claims ``c``.
LT_BC_FN bool bc_own_and_source(
    const int kind, const int* __restrict__ desc, const int nops, const int op,
    const std::int64_t* __restrict__ shapes, const int ndim,
    const int comp, const int axis, const int src,
    const int* __restrict__ c, int* __restrict__ src_c)
{
    for (int d = 0; d < ndim; ++d) src_c[d] = c[d];

    for (int o = 0; o < nops; ++o) {
        if (o == op) continue;
        int comp2, axis2, dst2, src2;
        bc_decode(kind, desc, o, shapes, ndim, comp2, axis2, dst2, src2);
        if (comp2 != comp) continue;          // different field
        if (c[axis2] != dst2) continue;       // op o does not touch this cell
        if (o < op) return false;             // lower-indexed op owns it
        src_c[axis2] = src2;                  // compose the inward step
    }
    src_c[axis] = src;                        // own axis is authoritative
    return true;
}

}  // namespace bcs
}  // namespace lilytorch_kernels
