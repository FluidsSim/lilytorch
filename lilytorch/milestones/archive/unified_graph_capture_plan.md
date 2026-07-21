# Unified CUDA Graph Capture — Implementation Plan

**Goal:** Replace all individual per-kernel graph captures with **two** graphs:
1. **Whole-step pre-Poisson graph** (SL advection + diffusion + bdim_forcing + set_BCs)
2. **Poisson V-cycle graph** (already independent — no change needed)

This covers 2-D + 3-D, constant + variable viscosity. The individual graph runners
are then simplified to pure launch wrappers (no capture/replay logic).

**Status today:**
- 2-D constant ν: ✅ whole-step graph exists (`WholeStepGraphRunner`)
- 2-D variable ν: ✅ whole-step graph (Tasks 2 + 3)
- 3-D constant ν: ✅ whole-step graph (Task 1)
- 3-D variable ν: ✅ whole-step graph (Tasks 2 + 4)
- Forces: ✅ unified graph for constant + variable ν (Task 5)
- Individual runners: ✅ simplified to pure launch wrappers (Task 6)
- Convective schemes: ✅ capture-safe + whole-step graph enabled (Task 8)
- ABDQUICKEST: ✅ whole-step graph enabled (Task 6 extension — 2026-07-07)
- Body update: independent graph (stays as-is)
- Poisson: ✅ independent graph (stays as-is)

**Remaining:**
- None — all tasks complete ✅

---

## Architecture Pattern (used by all tasks)

```
┌── OUTSIDE GRAPH (eager, torch + Warp allowed) ──────────┐
│ 1. Compute torch tensors (nu_t, nu_rho, etc.)           │
│ 2. Copy into persistent device buffers (stable ptrs)    │
│ 3. torch.cuda.synchronize() if cross-stream             │
└─────────────────────────────────────────────────────────┘
                         ↓
┌── INSIDE wp.ScopedCapture (Warp-only) ──────────────────┐
│ 4. Warp kernel launches reading from persistent buffers │
│ 5. Single wp.capture_launch(graph) on replay            │
└─────────────────────────────────────────────────────────┘
```

This is the same pattern the bdim runner already uses (`_stage_rect` → persistent `_rect_dev`)
and the Poisson MG uses (copy inputs into level-0 buffers before replay).

---

## TASK 1: 3-D Whole-Step Graph (constant viscosity)

**Complexity:** Low · **Risk:** Low · **Depends on:** nothing

Add a `WholeStepGraphRunner` for `_fluid_step_fused_3d`, exactly mirroring the 2-D pattern
in `_fluid_step_fused_2d`.

**What exists today:** `_fluid_step_fused_3d` (solver.py ~line 1993) runs eagerly —
`_compute_nu_t(...)` → `adv_diff_solver.solve(...)` → `bdim_forcing_3d(...)` →
`set_BCs(...)`. All individual runners use their own graph capture/replay.

**What to do:**
1. Add `self._preproj_graph_3d = WholeStepGraphRunner()` in solver `__init__` or lazily.
2. In `_fluid_step_fused_3d`, gate on `u.is_cuda and not self.use_variable_viscosity`:
   - `stage()`: copy dirty rect into `bdim_runner._stage_rect(...)` + `torch.cuda.synchronize()`
   - `issue()`: call `adv_diff_solver.solve(u, v, w_vel, nu_t=None)` → bdim_forcing_3d →
     set_BCs, all inside `capturing()` context so individual runners issue raw launches.
3. Key on pointer signature of u/v/w/u0/v0/w0/sdf_u/sdf_v/sdf_w/bU/bV/bW/ch_persist/
   cv_persist/cw_persist + dt + dtype (same pattern as 2-D).
4. Validate: run 3-D 1guilla/jellyfish, check `runner.replays > 0` and forces unchanged.

**Files touched:** `solver.py` only.

---

## TASK 2: Warp Diffusion Kernel with Variable Coefficient (nu_t staging)

**Complexity:** Medium · **Risk:** Medium · **Depends on:** nothing (enabler for Tasks 3, 4)

The variable-viscosity diffusion step currently uses `_DiffusionGraphRunner._launch_variable`
which reads `nu_eff` from a per-step-allocated tensor → pointer churn prevents graph capture.

**What to do:**
1. Add a persistent `_nu_t_persist` buffer to the diffusion runner (or solver).
2. Add a `_stage_nu_t(nu_t)` method: `_nu_t_persist.copy_(nu_t.reshape(-1))` — stable pointer.
3. The Warp kernel `variable_laplacian_warp_kernel` already exists and reads `nu_eff` via
   pointer. The graph captures it reading from `_nu_t_persist` (stable ptr).
4. The `diffuse_add_` variant (used by SL path) needs the same staging for variable ν.
   Add `_launch_add_variable` or extend `_launch_add` with an optional `nu_t` staging buffer.
5. The returning `diffuse()` (used by convective path) is separate — stage its nu_t too.

**Key constraint:** `diffuse_add_` is called from `_solve_semi_lagrangian_warp` which
currently passes `nu_t` as a tensor. The SL solver itself is pure Warp (capturable),
but `diffuse_add_` must read the staged buffer. The SL solver path needs a small refactor:
compute `nu_t * dt / h²` scale outside, stage it, and have the Warp kernel read it.

**Files touched:** `diffusion.py`, `advection.py` (SL solve call site).

---

## TASK 3: 2-D Whole-Step Graph for Variable Viscosity

**Complexity:** Medium · **Risk:** Medium · **Depends on:** Task 2

Extend the existing 2-D `WholeStepGraphRunner` to also work when `use_variable_viscosity=True`.

**Today's blocker:** `_compute_nu_t(u, v)` uses `torch.gradient`, `torch.sqrt` — can't live
inside `wp.ScopedCapture`. But it CAN run OUTSIDE the graph in `stage()`.

**What to do:**
1. In `stage()`: call `nu_t = self._compute_nu_t(u, v)`, then stage it:
   `diff_runner._stage_nu_t(nu_t)` (from Task 2) + `bdim_runner._stage_rect(...)`.
2. In `issue()`: call `adv_diff_solver.solve(u, v, nu_t=nu_t)` — the SL path detects
   `_gc.in_capture()` and issues raw launches reading the staged nu_t buffer.
3. Remove the `not self.use_variable_viscosity` gate for the whole-step graph.
4. The variable-viscosity diffusion uses `diffuse_add_` with a staged nu_t buffer —
   the raw launch reads it via stable pointer. Same for the returning `diffuse()` path
   if that code path is used (check: with SL advection, the path is `diffuse_add_`;
   with convective advection, it's the returning `diffuse()`).
5. Validate: run Smagorinsky/Carreau 2-D cases, check graph replays > 0, forces match
   the pre-refactor eager path.

**Files touched:** `solver.py`, `diffusion.py`, `advection.py` (SL solve).

---

## TASK 4: 3-D Whole-Step Graph for Variable Viscosity

**Complexity:** Medium · **Risk:** Medium · **Depends on:** Tasks 1, 2, 3

The 3-D analogue of Task 3. Combine the 3-D whole-step graph (Task 1) with
variable-viscosity staging (Task 2).

**What to do:**
1. In `stage()`: call `nu_t = self._compute_nu_t(u, v, w_vel)`, stage it +
   stage the bdim dirty rect.
2. In `issue()`: `adv_diff_solver.solve(u, v, w_vel, nu_t=nu_t)` with
   raw launches reading staged nu_t.
3. Remove the `use_variable_viscosity` gate (or keep it for the eager fallback).
4. Validate: 3-D Smagorinsky/Carreau + BDIM, check graphs engage, forces match.

**Files touched:** `solver.py`, `diffusion.py`, `advection.py`.

---

## TASK 5: Forces Graph for Variable Viscosity  ✅ DONE (2026-07-07)

**Complexity:** Low–Medium · **Risk:** Low · **Depends on:** Task 2 (nu_t staging pattern)

The `ForcesPostGraph` currently gates on `not self.use_variable_viscosity` because
`_compute_nu_rho_for_forces` allocates a fresh tensor each step.

**What was done:**
1. Added a persistent `_nu_rho_staging` buffer to `ForcesPostGraph.__init__`.
2. Extended `_stage()` to accept an optional `nu_rho_field` parameter and copy it
   into `_nu_rho_staging` (with graph-cache invalidation on shape/dtype/device change).
3. `run()` now uses `self._nu_rho_staging` (the staged buffer) for both the pointer
   signature and the kernel call, instead of the caller's transient `nu_rho_field`.
   This unifies constant and variable viscosity behind the same stable pointer.
4. Removed the `not self.use_variable_viscosity` gate at both 2-D and 3-D call sites
   in `force_method2_{2,3}d`: `_use_fgraph = u.is_cuda` now covers both paths.
5. Updated the class docstring to remove the outdated "variable-viscosity branch
   stays eager" constraint.
6. Validated: all 8 graph-replay tests pass (2-D/3-D × both submethods × both
   dtypes). Benchmarks show 6.5× (2-D) / 3.4× (3-D) submit speedup, bit-identical
   forces (|ΔF|max ≤ 1e-15).

**Files touched:** `forces.py`.

---

## TASK 6: Clean Up Individual Graph Runners ✅ DONE (2026-07-07)

**Complexity:** Medium · **Risk:** Medium · **Depends on:** Tasks 1, 3, 4 (whole-step graphs
must cover ALL paths)

Once the whole-step graph covers ALL pre-Poisson paths (2-D/3-D, constant/variable ν),
the individual graph caches are dead code. Simplify each runner to a pure launch wrapper.

**What was done:**
1. Simplified `_WarpGraphRunner` (advection.py): removed `_graphs`, `_seen`, key-building
   logic, `wp.capture_launch`, `wp.ScopedCapture`, and `replays`/`captures`/`eager`
   counters.  Now a pure launch wrapper that calls `eager_fn` directly (raw `wp.launch`
   recorded by the outer whole-step graph when inside `ScopedCapture`, otherwise
   standalone eager).
2. Simplified `ApplyBcs2DGraphRunner` and `ApplyBcs3DGraphRunner` (advection.py): same
   pattern — removed all graph-cache state, key building, and counters.  Now pure
   launch dispatchers to `apply_bcs_{2,3}d_warp`.
3. Simplified `_DiffusionGraphRunner` (diffusion.py): removed `_graphs`, `_add_graphs`,
   `_seen`, `_add_seen`, and all per-kernel capture/replay logic.  The `_gc.in_capture()`
   paths (persistent buffers + raw Warp launches) are kept; the standalone path now
   does raw launches + torch `mul_` (no per-kernel graph acceleration).
4. Simplified `_BdimForcingGraphBase` (bdim.py): removed `_graphs`, `_seen`,
   `replays`/`captures`/`eager_calls` counters, and all capture/replay logic.  The
   dirty-rect staging pattern is kept; `_dispatch` now does raw launch in both paths.
5. Updated all affected tests to remove assertions on the removed counters.

---

## TASK 7 (Optional): Remove `_gc.in_capture()` Re-entrancy Flag ✅ DONE (2026-07-07)

**Complexity:** Low · **Risk:** Low · **Depends on:** Task 6

Once individual runners no longer have their own graph capture/replay, they don't need
to check `_gc.in_capture()` — they ALWAYS issue raw `wp.launch`. The whole-step graph
calls them inside `wp.ScopedCapture` and the launches are recorded. Outside the graph
(eager path), they just launch normally.

**What was done:**
1. Simplified `_WarpGraphRunner` (advection.py) to a pure launch wrapper: removed
   `_graphs`, `_seen`, `_build_key`, `_skip_graph`, and all capture/replay logic.
   Now always calls `eager_fn` directly — raw `wp.launch`.
2. Simplified `ApplyBcs2DGraphRunner` and `ApplyBcs3DGraphRunner` (advection.py):
   removed `_graphs`, `_seen`, `replays`/`captures`/`eager` counters, and all
   per-kernel graph capture/replay. Now always delegates to `apply_bcs_{2,3}d_warp`.
3. Removed `if _gc.in_capture()` branches from `_DiffusionGraphRunner`
   (diffusion.py): `_launch_constant`, `_launch_variable`, and `_launch_add`
   always use persistent output buffers + pure Warp kernels (no torch compute
   ops). The standalone torch `mul_` path is replaced by Warp `_scale_interior_eager`.
4. Removed `if _gc.in_capture()` branch from `_BdimForcingGraphBase._dispatch`
   (bdim.py): always self-stages the dirty rect (pinned-host → async device copy)
   then raw-launches. The caller's `stage()` callback still pre-stages for
   correctness on replay (the pinned-host update must happen outside the graph).
5. Removed `in_capture()` function, `capturing` context manager, and the
   `_DEPTH` global from `graph_capture.py`. Removed `with capturing():` from
   `WholeStepGraphRunner.run()` — Warp's `ScopedCapture` handles recording
   raw launches automatically.
6. Removed `from lilytorch.src import graph_capture as _gc` imports from
   `advection.py`, `diffusion.py`, and `bdim.py`.
7. Updated `solver.py` comments to reflect the simplified architecture.
8. Updated `test_whole_step_capture.py` to remove `capturing` import and usage.
9. Removed dead code: `_sl2d_key`, `_sl3d_key`, `_flux_skip_graph`, `_flux_key`,
   `_wp_device` from advection.py.
10. Validated: 120/120 tests pass across test_advection, test_bdim, test_forces,
    and test_whole_step_capture.

**Files touched:** `graph_capture.py`, `advection.py`, `diffusion.py`, `bdim.py`,
`solver.py`, `test_whole_step_capture.py`.

---

## TASK 8 (Follow-up): Enable Whole-Step Graph for Convective + Variable ν ✅ DONE (2026-07-07)

**Complexity:** Low · **Risk:** Low · **Depends on:** Tasks 5, 6

Once all individual runners are pure launch wrappers (Task 6) and `_launch_variable`
is capture-safe (Task 2 done in parallel with Tasks 3–4), the only thing preventing
the whole-step graph from covering *all* paths is the `_sl_schemes` gate in
`_fluid_step_fused_{2,3}d`.

**What was done:**
1. Made `_solve_convective` capture-safe: added a `_clone_eager` helper (using
   `wp.copy`) and used persistent output buffers (`_conv_out`, `_conv_rhs`) with
   pure-Warp laplacian/scale/accumulate kernels inside `wp.ScopedCapture`.  The
   torch ops (`vel.clone()`, `+= rhs`) are replaced by Warp equivalents when
   `_gc.in_capture()` is True; the eager path is unchanged.
2. Removed the `_sl_schemes` gate in both `_fluid_step_fused_2d` and
   `_fluid_step_fused_3d`.  The whole-step graph now activates for ALL CUDA
   advection schemes: semi-Lagrangian, implicit, QUICK, CUBISTA, van Leer, CDS.
3. Kept ABDQUICKEST excluded (`scheme_name != 'abdquickest'`) because its live
   Courant number (|u|·dt/h) varies per step and would be frozen at capture
   time — the existing `_flux_skip_graph` mechanism handles it correctly in
   the eager path.

**Files touched:** `advection.py`, `solver.py`.

---

## Execution Order & Dependencies

```
Task 2 (nu_t staging) ✅
  ├── Task 3 (2-D variable ν whole-step) ✅
  ├── Task 4 (3-D variable ν whole-step) ✅
  └── Task 5 ✅ (forces variable ν graph)

Task 1 (3-D constant ν whole-step) ✅

Task 6 (cleanup individual runners) ✅
Task 8 (convective capture-safe + gate removal) ✅

Task 7 (remove re-entrancy flag) — remaining (optional)
```

**Recommended order:** Task 1 + Task 2 in parallel → Task 3 → Task 4 → Task 5 → Task 6 → Task 7.

**Status:** Tasks 1–6, 8 ✅ complete.  Task 7 remains (optional cleanup).

---

## Risk Mitigation

- **Every task keeps the eager path as fallback** — if `use_graph` gate is false
  (CPU, or the graph cache is full), the eager path runs identically to today.
- **Bit-identical forces** is the acceptance test for every task.
- **Individual runner graph caches are NOT deleted until Task 6** — they remain
  as a safety net. If a whole-step graph fails to capture, the individual runner's
  own capture/replay still works (it's just not used when `_gc.in_capture()` is True).
- **Each task is one PR** — small, reviewable, revertible.
