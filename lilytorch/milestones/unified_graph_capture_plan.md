# Unified CUDA Graph Capture — Implementation Plan

**Goal:** Replace all individual per-kernel graph captures with **two** graphs:
1. **Whole-step pre-Poisson graph** (SL advection + diffusion + bdim_forcing + set_BCs)
2. **Poisson V-cycle graph** (already independent — no change needed)

This covers 2-D + 3-D, constant + variable viscosity. The individual graph runners
are then simplified to pure launch wrappers (no capture/replay logic).

**Status today:**
- 2-D constant ν: ✅ whole-step graph exists (`WholeStepGraphRunner`)
- 2-D variable ν: ❌ individual per-kernel graphs only
- 3-D constant ν: ❌ individual per-kernel graphs only
- 3-D variable ν: ❌ individual per-kernel graphs only
- Forces: ❌ individual graph only (constant ν), eager (variable ν)
- Body update: independent graph (stays as-is)
- Poisson: ✅ independent graph (stays as-is)

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

## TASK 5: Forces Graph for Variable Viscosity

**Complexity:** Low–Medium · **Risk:** Low · **Depends on:** Task 2 (nu_t staging pattern)

The `ForcesPostGraph` currently gates on `not self.use_variable_viscosity` because
`_compute_nu_rho_for_forces` allocates a fresh tensor each step.

**What to do:**
1. Add a persistent `_nu_rho_staging` buffer to `ForcesPostGraph`.
2. In `_stage()` (already exists for pose staging): compute and copy `nu_rho` into
   the persistent buffer when variable viscosity is active.
3. The `forces_post_{2,3}d_kernel` already reads `nu_rho` via pointer — it just
   needs a stable one. If currently it reads a scalar when constant, a tensor when
   variable: unify to always pass a tensor pointer (a 1-element tensor for scalar).
4. Remove the `use_variable_viscosity` gate.
5. Validate: forces with Smagorinsky/Carreau match eager path.

**Files touched:** `forces.py`.

---

## TASK 6: Clean Up Individual Graph Runners

**Complexity:** Medium · **Risk:** Medium · **Depends on:** Tasks 1, 3, 4 (whole-step graphs
must cover ALL paths)

Once the whole-step graph covers ALL pre-Poisson paths (2-D/3-D, constant/variable ν),
the individual graph caches are dead code. Simplify each runner to a pure launch wrapper.

**Runners to simplify:**

| Runner | File | What to remove |
|--------|------|----------------|
| `_WarpGraphRunner` (SL, Flux) | `advection.py` | `_graphs`, `_seen`, `replays`/`captures` counters, key-building logic, `wp.capture_launch` call. Keep: raw `wp.launch` + `_gc.in_capture()` check. |
| `ApplyBcs2DGraphRunner` | `advection.py` | Same — remove graph cache, keep raw launch. |
| `ApplyBcs3DGraphRunner` | `advection.py` | Same. |
| `_DiffusionGraphRunner` | `diffusion.py` | Remove `_graphs`, `_add_graphs`, `_seen`, `_add_seen`, all capture/replay logic. Keep: `_launch_add` → raw launch, `_launch_constant` → raw launch, `_launch_variable` → raw launch. Add `_stage_nu_t` from Task 2. |
| `BdimForcing2DGraph` | `bdim.py` | Remove `_graphs`, key logic, capture/replay. Keep: `_stage_rect`, raw `_bdim_forcing_2d_launch`. |
| `BdimForcing3DGraph` | `bdim.py` | Same. |
| `ForcesPostGraph` | `forces.py` | Keep its OWN graph (forces are outside whole-step region). But if variable ν staging is added (Task 5), simplify the constant-ν path similarly. |

**What stays independent (not part of whole-step graph):**
- `_BodyUpdate2DBridge` / `_BodyUpdate3DBridge` (body update — runs before fluid step)
- `WarpMG2D` / `WarpMG3D` (Poisson — runs after pre-Poisson region)
- `ForcesPostGraph` (force readout — runs after Poisson)
- `WarpStreamingSDF` captures (used by body bridges)

**Validation gate:** All existing tests pass. Forces bit-identical to pre-refactor baseline
for ALL paths (2-D/3-D, constant/variable ν, all advection schemes).

**Files touched:** `advection.py`, `diffusion.py`, `bdim.py`, `forces.py`.

---

## TASK 7 (Optional): Remove `_gc.in_capture()` Re-entrancy Flag

**Complexity:** Low · **Risk:** Low · **Depends on:** Task 6

Once individual runners no longer have their own graph capture/replay, they don't need
to check `_gc.in_capture()` — they ALWAYS issue raw `wp.launch`. The whole-step graph
calls them inside `wp.ScopedCapture` and the launches are recorded. Outside the graph
(eager path), they just launch normally.

**What to do:**
1. Remove all `if _gc.in_capture(): ... else: ...` branches from runners.
2. Simplify to always call raw `wp.launch`.
3. Remove `graph_capture.py` `in_capture()` and `capturing()` (or keep `capturing()`
   for the `WholeStepGraphRunner` internal use but remove the global flag).
4. `WholeStepGraphRunner.run()` no longer needs the `with capturing():` context —
   Warp's `ScopedCapture` handles recording raw launches automatically.

**Files touched:** `graph_capture.py`, `advection.py`, `diffusion.py`, `bdim.py`.

---

## Execution Order & Dependencies

```
Task 2 (nu_t staging)
  ├── Task 3 (2-D variable ν whole-step)
  ├── Task 4 (3-D variable ν whole-step)
  └── Task 5 (forces variable ν graph)

Task 1 (3-D constant ν whole-step) —— independent, can run in parallel with Task 2

Task 6 (cleanup individual runners) —— after Tasks 1+3+4 (all paths covered)
Task 7 (remove re-entrancy flag) ——— after Task 6
```

**Recommended order:** Task 1 + Task 2 in parallel → Task 3 → Task 4 → Task 5 → Task 6 → Task 7.

Tasks 3 and 4 can also be parallelized if desired (different files, same pattern).

---

## Risk Mitigation

- **Every task keeps the eager path as fallback** — if `use_graph` gate is false
  (CPU, or the graph cache is full), the eager path runs identically to today.
- **Bit-identical forces** is the acceptance test for every task.
- **Individual runner graph caches are NOT deleted until Task 6** — they remain
  as a safety net. If a whole-step graph fails to capture, the individual runner's
  own capture/replay still works (it's just not used when `_gc.in_capture()` is True).
- **Each task is one PR** — small, reviewable, revertible.
