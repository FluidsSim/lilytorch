# TODO — decide the corner/edge ghost BC rule on purpose

**Status**: open, low priority. Not a known bug — a known *unexamined choice*.
**Prereq**: done. `apply_bcs_{2,3}d` is now deterministic and backend-identical
(commit `c919d6c`, `csrc/bc_ops.h`); the values below are stable and testable.

## Why this is now a question at all

Closing the `apply_bcs` race turned up something the codebase had assumed away:
**the wide / cross-term advection stencil reads edge and corner velocity ghosts.**
Perturb *only* those cells and one step moves the **interior** velocity by ~8e-3
(3-D) / ~3e-3 (2-D). See `ghost_cell_issues_handoff.md` and
`csrc/bc_ops.h`. (Pressure corners really are dead — the Poisson stencil is
5/7-point and the projection only takes face differences.)

So the value in a corner ghost is part of the discretisation, not a don't-care.
And nobody ever chose it.

## What the value currently is, and where it came from

It is whatever the old **sequential CPU BC loop** happened to leave there, which
the new ownership rule reproduces deliberately (so that the race fix moved no
physics). Concretely:

* **All-Neumann.** `v[0,j,0]` ends up as `v[1,j,1]` — the diagonal interior
  neighbour. This is the composition of the two face ops: the z-face op ran last
  and copied from a cell the x-face op had already overwritten. A defensible
  zero-gradient extrapolation, but an accident of op ordering, not a decision.
* **Mixed walls.** With x-lo a Dirichlet wall (value `g`) and z-lo a tangential
  wall enforced reflectively (value `w`), the corner comes out as `2w − g`: the
  z-wall's reflection rule applied to a source cell that lies inside the x-wall.

A corner ghost is **over-determined** — it is asked to satisfy two wall conditions
at once — so *some* arbitration is unavoidable. The point is to pick it, and to
know what it costs.

## The task

1. Work out what the advection scheme actually needs from that cell (which term
   reaches it, and with what weight — it is the transverse-velocity interpolation
   to a face near the corner).
2. Bound the impact before investing: perturb the corner rule and measure how far
   into the interior the change propagates, and how it scales with resolution. If
   it stays a one-cell corner effect that converges away, close this as "the
   current rule is fine, and here is why".
3. If it does matter, choose the rule deliberately (candidates: keep the composed
   inward step; average the competing ops; extrapolate along the diagonal) and
   pin it in `bc_ops.h` with the reasoning.
4. A lid-driven cavity is the natural benchmark — its singular corners are exactly
   where it is sensitive.

## Expected outcome

Probably "no change, now justified". The affected region is a one-cell-thick
diagonal at the domain corners, the current value is a sensible extrapolation, and
it is the value every validated CPU result to date has used — so nothing already
validated is in question. This is a *know why* item, not a *fix* item.
