# Unified Single-Graph Capture — Implementation Plan

**Date:** 2026-07-08 · **Branch:** `warp_port`

## Goal

A single `WholeStepGraphRunner` captures the entire pre-Poisson region for **all**
simulation configurations: 2-D / 3-D, constant / variable viscosity, all
advection methods (semi-Lagrangian, QUICK, ABDQUICKEST, CUBISTA, van Leer, CDS).

## Current State

| Configuration                                      | Whole-step graph? | Blocker                                         |
|----------------------------------------------------|:-----------------:|-------------------------------------------------|
| 2-D SL + constant viscosity                        | ✅                | —                                               |
| 2-D convective + constant viscosity                | ❌                | torch ops in `_solve_convective`                |
| 2-D any + variable viscosity                       | ❌                | `torch.gradient`/`sqrt`/`clamp` in strain-rate  |
| 3-D all methods                                    | ❌                | No `WholeStepGraphRunner` at all                |
| Two-phase kernel-mode repairs                      | ❌                | `torch.lerp_`, `torch.where`, `clone`           |
| Post-BDIM damping (sponge/yield)                   | ❌                | `torch.Tensor.__mul__`, `torch.clamp`           |
| Eager BC fallback on GPU                           | ❌                | torch slice assignment                          |

## Architecture Principle

Follow the **`diffuse_add_` pattern**: double-buffer copy → fused
compute+accumulate Warp kernel with persistent, pointer-stable buffers.
**Zero per-step torch ops.** `WholeStepGraphRunner` is the only graph capture
class — no per-kernel graph runners.

---

## Phase 1: Warp-ify Strain-Rate Computation  ✅ COMPLETE

**Unlocks:** variable viscosity (Smagorinsky, Carreau) for graph capture.

**Files:** `lilytorch/src/operations.py`, `lilytorch/src/solver.py`, `lilytorch/src/diffusion.py`, `lilytorch/src/advection.py`

### Tasks

- [x] **1.1** Write `strain_rate_magnitude_warp` kernel
  - Reads staggered velocities (`u`, `v`, [`w`]), strides, grid spacing `dh`
  - Computes central differences on staggered grids → face-average (`_stag_to_cc`)
  - Assembles `S_ij S_ij` → `sqrt(2*S2)` at cell centres
  - 2-D + 3-D in one unified kernel (degenerate 3-D with `Nz=1`, like `laplacian_warp_kernel`)
  - f32 + f64 via `wp.overload`
  - Returns persistent cc buffer (reallocated only on shape/dtype/device change)

- [x] **1.2** Write eager launch wrapper `_strain_rate_magnitude_warp_eager(u, v, w, h)`

- [x] **1.3** Update `smagorinsky_viscosity()` to use Warp `S_mag` instead of torch ops
  - (automatic — `strain_rate_magnitude()` is now a thin Warp wrapper)

- [x] **1.4** Update `carreau_viscosity()` to use Warp `S_mag` instead of torch ops
  - (automatic — same as 1.3)

- [x] **1.5** Remove `not self.use_variable_viscosity` gate from `use_graph` in `solver.py`
  - Graph path now pre-computes `nu_t` + `nu_eff` outside the capture and passes `nu_eff` through the solve chain.
  - `diffuse_add_` gained an optional `nu_eff` parameter (graph-safe, no torch add inside capture).
  - `_solve_semi_lagrangian_warp` / `_solve_semi_lagrangian` forward `nu_eff`.

- [x] **1.6** Test: `torch.testing.assert_close` against current eager output (CPU + CUDA)
  - f64: max error ≤ 5.7e-14 (machine precision) across 2-D/3-D, CPU/CUDA.
  - f32: passes `rtol=1e-5, atol=1e-7`.
  - Full graph-path pipeline (strain → smagorinsky → nu_eff → diffuse_add_) matches eager path to 1.2e-7 (f32 CUDA).
  - Constant-viscosity path unaffected (bit-identical).
  - Removed: `_stag_to_cc` helper, old torch-based `strain_rate_magnitude`.

---

## Phase 2: Warp-ify the Convective Solve  ✅ COMPLETE

**Unlocks:** all flux schemes (QUICK, ABDQUICKEST, CUBISTA, van Leer, CDS) for graph capture.

**Files:** `lilytorch/src/advection.py`

### Tasks

- [x] **2.1** Factor `copy_full_grid_kernel` from `diffusion.py` into a shared utility module
  - Re-used via `diffusion._copy_full_grid_eager` (already public)

- [x] **2.2** Write `advect_flux_accumulate_kernel`
  - Reads stencil from read-only copy buffer (`phi_src`)
  - Computes face velocities internally (no torch `_face_vel` slicing)
  - Computes fluxes using existing scheme logic (port from `advect_flux_add_warp`)
  - Accumulates `phi_dst[c] += Σ_d flux_contribution` in a single pass
  - One launch per velocity component, loops over spatial directions internally
  - 2-D + 3-D unified, f32 + f64, compile-time `scheme_id` to eliminate branch overhead

- [x] **2.3** Add persistent double-buffers to `AdvDiffSolver` (like `_sl_out` / `_diff_out`)
  - `_conv_copy_buf`: per-component full-grid copy buffers
  - `_conv_out_buf`: per-component output buffers (pointer-stable)
  - Initialized once in `_init_convective_buffers()`, called during `solve`

- [x] **2.4** Rewrite `_solve_convective` to use the fused kernel:
  ```
  for i in range(ndim):
      copy_full_grid(vel[i] → copy_buf[i])            # Warp, graph-safe
      rhs = diffusion.diffuse(vel[i], ...)             # already graph-safe
      advect_flux_accumulate(copy_buf[i], rhs, ...)    # Warp, graph-safe
      copy_full_grid(vel[i] → out_buf[i]) + accumulate_interior(out_buf[i], rhs)
  ```

- [x] **2.5** Keep `_face_vel`, `_field_for_flux` (still used in tests, no longer in production path)

- [x] **2.6** Removed `vel[i].clone()` and `vel_new[i][inner] += rhs` lines (replaced by pure-Warp `accumulate_interior_kernel`)

- [x] **2.7** Test: bit-identical output (max diff = 0.0 for f64) against current eager path
  - All 5 schemes × 2-D/3-D × f32/f64 × const/var viscosity: BIT-IDENTICAL
  - All 53 existing tests in `test_advection.py` pass

---

## Phase 3: Add 3-D Whole-Step Graph + Unify Dispatch  ✅ COMPLETE

**Unlocks:** 3-D graph capture, all methods in both 2-D and 3-D.

**Files:** `lilytorch/src/solver.py`, `lilytorch/src/advection.py`, `lilytorch/src/diffusion.py`

### Tasks

- [x] **3.1** Add `self._preproj_graph_3d = WholeStepGraphRunner()` in `_fluid_step_fused_3d`

- [x] **3.2** Add `stage()` closure for 3-D:
  - `bdim_runner._stage_rect(rect_vals, u.device)` (already works for 3-D)
  - `torch.cuda.synchronize(u.device)`

- [x] **3.3** Add `issue()` closure for 3-D:
  - `primes = self.adv_diff_solver.solve(u, v, w_vel, nu_eff=nu_eff)`
  - `bdim_forcing_3d(..., runner=bdim_runner, skip_stage=True)`
  - `self.adv_diff_solver.set_BCs(self.u0, self.v0, self.w0)`

- [x] **3.4** Key includes `w_vel.data_ptr()`, `self.w0.data_ptr()`, and 3-D rect `(k0, Ak)`

- [x] **3.5** Remove `not self.use_variable_viscosity` gate (fixed in Phase 1)

- [x] **3.6** Remove 2-D-only restriction — the 2-D graph path now works for convective schemes
  - Added `nu_eff` parameter to `_solve_convective()` (advection.py)
  - Added `nu_eff` parameter to `diffusion.diffuse()` (diffusion.py, matching `diffuse_add_` pattern)
  - Graph-safe: `nu_eff` pre-computed outside capture, forwarded without torch ops

- [x] **3.7** Eager `else` branch triggers only for CPU or `force_eager=True`

- [x] **3.8** Add `_force_eager: bool = False` debug flag to `FluidSolver.__init__`
  - Configurable via `solver.force_eager` in YAML config

- [x] **3.9** Test: all 245 existing tests pass (forces, bdim, advection, whole-step capture, poisson, pose_source)

---

## Phase 4: Warp-ify Two-Phase Repairs

**Unlocks:** two-phase kernel-mode path for graph capture.

**Files:** `lilytorch/src/two_phase_solver.py`

### Tasks

- [ ] **4.1** Write `face_average_warp` kernel
  - Reads cell-centred field `q`, writes `0.5*(q[lo] + q[hi])` into face-centred buffer
  - Dimension-agnostic (2-D / 3-D), f32 + f64
  - Replaces `_face_mean` (which does `clone()` + slice assignment)

- [ ] **4.2** Write `velocity_blend_warp` kernel
  - Elementwise `dst[i] = src[i] + w[i] * (prime[i] - src[i])`
  - Replaces `torch.Tensor.lerp_`

- [ ] **4.3** Write `rescale_coeffs_warp` kernel
  - Fuses face-averaging + `torch.where(sdf >= 0, mu0_t, mu0)` + arithmetic
  - One launch per spatial direction

- [ ] **4.4** Manage persistent face-averaged buffers (same pattern as `_diff_out`)

- [ ] **4.5** Rewrite `_kernel_blend_velocities` to use Warp kernels only

- [ ] **4.6** Rewrite `_rescale_kernel_coeffs_two_phase` to use Warp kernels only

- [ ] **4.7** Test: two-phase simulations produce bit-identical results

---

## Phase 5: Post-BDIM Cleanup + Eager BC Fallback Hardening

**Files:** `lilytorch/src/solver.py`, `lilytorch/src/advection.py`, `lilytorch/src/extras.py`

### Tasks

- [ ] **5.1** Make `apply_sponge_damping` graph-safe:
  - Option A: Warp elementwise-multiply kernel `damp_multiply_kernel(dst, damp_factor)`
  - Option B (preferred): bake damping into BDIM forcing kernel as a post-pass

- [ ] **5.2** Make `apply_yield_damping` graph-safe:
  - Option A: Warp elementwise kernel for `clamp(1 - S/gamma, 0) * strength * ratio²`
  - Option B (preferred): bake into BDIM forcing kernel

- [ ] **5.3** Harden eager BC fallback in `set_BCs` (advection.py lines 840–845):
  - Assert `u.is_cuda` so the fused Warp path is always taken on GPU
  - Remove the torch slice-assignment fallback for CUDA devices

- [ ] **5.4** Investigate extending `WholeStepGraphRunner` to capture the **full step**
  (pre-Poisson + Poisson projection + post-Poisson):
  - The MG Poisson solver has its own graph management via WarpMG
  - Document blockers or implement if feasible

---

## Execution Order

```
Phase 1 (strain-rate) ──┐
                         ├──→ Phase 3 (unify + 3-D graph) ──→ Phase 5 (cleanup)
Phase 2 (convective) ───┘

Phase 4 (two-phase) ──── independent, can run in parallel with 1+2
```

Phases 1 and 2 are the critical path — they unlock the building blocks.
Phase 3 wires them together. Phase 4 is independent (separate code path).
Phase 5 is polish.

---

## Key Design Principles

1. **Persistent buffers, never allocate per step** — follow `_const_out_persist` / `_var_out_persist` pattern
2. **Double-buffer for any in-place mutate** — copy source, read from copy, write to target (like `diffuse_add_`)
3. **One unified kernel per operation (2-D + 3-D, f32 + f64)** — degenerate 3-D with `Nz=1` like `laplacian_warp_kernel`
4. **Compile-time flags for dispatch** — `is_variable`, `is_3d`, `scheme_id` as `int` args; Warp eliminates dead branches
5. **Eager path must produce bit-identical results** — test with `torch.testing.assert_close` on CPU and CUDA
6. **No new graph runner classes** — `WholeStepGraphRunner` is the only graph capture; all kernels are pure `wp.launch` wrappers

---

## References

- `diffuse_add_` reference implementation: `lilytorch/src/diffusion.py` lines 481–512
- `_solve_semi_lagrangian_warp` reference implementation: `lilytorch/src/advection.py` lines 400–475
- `WholeStepGraphRunner`: `lilytorch/src/graph_capture.py`
- `laplacian_warp_kernel` (unified 2-D/3-D pattern): `lilytorch/src/diffusion.py` lines 98–150
- `strain_rate_magnitude` (torch ops to replace): `lilytorch/src/operations.py` lines 223–273
- `_solve_convective` (torch ops to replace): `lilytorch/src/advection.py` lines 323–372
- `_fluid_step_fused_2d` (2-D graph path): `lilytorch/src/solver.py` lines 1798–1991
- `_fluid_step_fused_3d` (3-D, no graph path): `lilytorch/src/solver.py` lines 1993–2105
- `_kernel_blend_velocities` (two-phase torch ops): `lilytorch/src/two_phase_solver.py` lines 594–611
- `_rescale_kernel_coeffs_two_phase`: `lilytorch/src/two_phase_solver.py` lines 613–644
