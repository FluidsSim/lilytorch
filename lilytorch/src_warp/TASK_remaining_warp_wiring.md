# Task brief — make `src_warp/` a TOTALLY native-independent Warp backend

**Owner:** (next agent) · **Prereq reading (in order):**
`src_warp/README.md`, this file, `warp_poc/VALIDATION_STATUS.md` §F,
`warp_poc/HANDOFF.md` (lessons 2/3/9/13/14/15/17 especially).

## THE GOAL (changed — read carefully)
The Warp backend tree must be **totally independent of the native CUDA/C++ kernel
path** (`torch.ops.lilytorch_kernels.*` / `src/kernels`).  That means:
- **Every** custom-kernel op the live solver step touches runs on Warp.
- **No native fallback** is acceptable as an end state — not for float32, not for
  the σ path, not for 3-D.  (A fallback only for "Warp failed to import" is fine.)
- **Both dtypes** (float32 AND float64) run on Warp, in **2-D AND 3-D**.

Pure-torch ops are NOT part of this — `diffusion.diffuse`, the FFT Poisson
(`poisson_fft`), sponge/yield damping, projection arithmetic are plain torch and
dtype-agnostic; leave them.  The target is the `lilytorch_kernels` custom ops only.

## THE HARD GUARDRAIL
- **No edits to `lilytorch/src/`.**  After every change:
  `git diff --name-only -- lilytorch/src` must show **only** pre-existing
  modifications (a `diagnostics.py`/`solver.py` diagnostics change that predates
  this work) — your work adds **nothing** to `src/`.  All wiring lives in
  `src_warp/` (subclass + override), all kernels in `warp_poc/`.
- Keep `src_warp` runnable at every step (parity-test before flipping to Warp).
- `python -m pytest warp_poc/ -q` after each change (currently **169 pass**).
- Update `src_warp/README.md` (WARP_BACKED table) + `VALIDATION_STATUS.md` §F per op.

---

## STATE: what is on Warp vs still native
`WARP_BACKED` (authoritative) = `{advect_flux_add, cvof_sweep,
streaming_sdf_stag_2d_multi, bdim_coeff_2d}`.

| Native op | Warp port exists? | dtype-generic? | Wired into solver? | Independence gap |
|---|---|---|---|---|
| `advect_flux_add` | yes | **yes (f32+f64)** | yes | ✅ done |
| `cvof_sweep` | yes | **yes (f32+f64)** | yes | ✅ done |
| `streaming_sdf_stag_2d_multi` (Kernel A 2-D) | yes | **yes** | yes (bridge) | ✅ done |
| `bdim_coeff_2d` (Kernel B 2-D) | yes | **yes** | yes | ✅ done |
| `bdim_coeff_sigma_2d` (σ 2-D) | yes (same kernel) | yes | **no** — needs Kernel-A keys | **σ keys** |
| `streaming_sdf_stag_3d_multi` (Kernel A 3-D) | yes (`warp_kernels.py`) | f32 only | no | Item 1-3D |
| `bdim_coeff_3d` / `_sigma_3d` (Kernel B 3-D) | yes (`warp_bdim.py`) | **f64 only** | no | Item 1-3D |
| Poisson `poisson_solve_{mgcg,multigrid,rmgcg}_{2,3}d` | composition (`WarpVCycle`) | make generic | no | Item 2 |
| `apply_bcs_2d` / `apply_bcs_3d` | yes (warp_poc) | check | no | Item 5 |
| `interp_2d` / `interp_3d` | yes (warp_poc) | check | no | Item 5 |
| `lagrangian_forces_2d/3d` | yes (`warp_lagrangian.py`) | check | no | Item 3 |
| `streaming_sdf_forces_post_2d/3d` (Eulerian) | **NOT ported** | — | no | Item 3 |

So even the wired 2-D f64 step still calls native **Poisson**, **apply_bcs**
(`set_BCs`), **forces**, and **σ**.  Independence needs ALL rows ✅.

## THE dtype-generic recipe (REUSE for every port below)
Warp 1.14 has no implicit cross-dtype respecialisation, so:
1. Value arrays → `wp.array(dtype=Any)`; float scalar args → `Any`
   (`from typing import Any`).  Keep int/int64 args and incidental f32 tables
   (e.g. `sigma_shifts`) concrete, cast them inside via `type(x)(arr[i])`.
2. Materialise float literals in the bound type: bind a **value**
   `half = type(x)(0.5)` (NOT a type — `t = type(x); t(0.5)` is unproven).
3. Pre-register both specialisations after the kernel def:
   `for dt in (wp.float32, wp.float64): wp.overload(K, {"arr": wp.array(dtype=dt), "scal": dt, ...})`
   (only the generic args).  `@wp.func` helpers auto-specialise.
4. Host wrapper: `wpf = wp.float64 if t.dtype==torch.float64 else wp.float32`;
   build flat views with `wpf`; pass scalars `wpf(value)`.
Worked examples (copy verbatim): `warp_kernels_2d.py`, `warp_bdim_2d.py`,
`warp_advection.py`, `warp_cvof.py`.  f64 codegen stays bit-identical (existing
f64 parity holds); f32 matches native f32 to single precision.

## The marshalling-bridge pattern (REUSE for Kernel A/B 3-D)
`src_warp/kernel/_KernelA2DBridge` adapts the native positional
`streaming_sdf_stag_2d_multi(...)` call into `WarpStreamingSDF2D.setup/
update_kinematics/run_fanned_eager` — caches the static body table, per-step
`update_kinematics`, wraps the caller's torch outputs zero-copy.  The override
`FluidSolver._fluid_step_kernel_2d` copies the native body verbatim and swaps ONLY
the kernel calls to `kernel.*`, pre-fills `comp.sdf_val` to `+FAR` (Warp
`atomic_min` needs it).  3-D mirrors this.

---

## ITEMS (do in this order)

### Item 1 (3-D) — Kernel A/B 3-D
3-D solvers run **float32** (e.g. `_1guillasim/gen_configs_one_pinned_3d.py`).
1. Make `warp_bdim.py` (3-D Kernel B) dtype-generic (recipe).  **HANDOFF lesson 9
   is 3-D-specific:** native writes the Poisson coeff at a *face-grid* offset on
   CUDA but full-grid on CPU — honour whatever the existing 3-D parity test
   asserts.  Add f32 parity to `warp_poc/test_bdim.py` (mirror `test_bdim_2d.py`).
2. Make `warp_kernels.py` (3-D Kernel A) dtype-generic too (it is f32-only now;
   needed if any 3-D case runs f64).  Add f64 parity to `test_parity.py`.
3. Write `_KernelA3DBridge` in `src_warp/kernel/__init__.py` mirroring the 2-D one
   (z axis: `aabb_*`=`[B*3]`, `kin`=`[B*16]`, add `gz`, `sdf_w/bW/key_w`, dirty
   `*_k`).  Route `bdim_coeff_3d` straight through (signature-compatible).
4. Override `_fluid_step_kernel_3d` in `src_warp/solver.py` (copy native
   `src/solver.py` ~L2017–end; swap only the two kernel calls; keep `_chk`,
   temps, `del`, projection; pre-fill `comp.sdf_val`; σ → handled by Item 6).
5. 3-D chain parity test in `test_src_trees.py` (synthetic 3-D scene; assert
   `u0/v0/w0` + `ch/cv/cw`).  Add the two ops to `WARP_BACKED`.

### Item 2 — Poisson driver (native `poisson_solve_*`)
Native is a monolithic C++ driver (`src/poisson_mult.py` → `self._K.poisson_solve_*`)
with no injectable smoother seam.  The Warp smoother + 4 transfer ops +
`WarpVCycle`/`WarpVCycle2D` are DONE (`warp_poc/warp_multigrid{,_2d}.py`).
1. Make the Warp multigrid kernels dtype-generic (recipe) so f32+f64 both run.
2. In `src_warp/poisson_mult.py`, subclass `PoissonSolver`; assemble
   `solve_multigrid`/`solve_mgcg`/`solve_rmgcg` outer drivers in Python around
   `WarpVCycle` (mirror `src/poisson_mult.py::_vcycle_rbgs_*` + Krylov/Aitken
   loop), host-side driver + CUDA-graphed inner kernels.
3. Parity: converge to the same residual as native on a manufactured Neumann
   Poisson (NOT bit-exact — different solver; assert residual + trajectory tol).
This subclass is already injected via the `__init__` module-name swap in
`src_warp/solver.py` — just make it actually use Warp instead of inheriting native.

### Item 3 — Forces (native `lagrangian_forces_*` + `streaming_sdf_forces_post_*`)
Both readouts must run on Warp for independence:
- **Lagrangian:** `warp_lagrangian.py` IS ported (parity ≤1.7e-16).  Make it
  dtype-generic; wire behind the `forces_lagrangian_{2,3}d` call site in a
  `src_warp/forces.py` subclass/override (the native call builds the decomposed
  args — `eps_xx…`, `tri_centroid/normal/area`, `com_pos` — which
  `lagrangian_forces_*_warp` takes directly).  Add to `WARP_BACKED`.
- **Eulerian:** `streaming_sdf_forces_post_{2,3}d` is the **one native op with no
  Warp port** — the volumetric n·δ viscous+pressure band integral (+ deltaH ∂H
  pass).  Write `warp_poc/warp_forces.py` (block reduction + `wp.atomic_add`
  scatter into the per-body force row, same class as `warp_lagrangian.py`),
  dtype-generic, `test_forces.py` parity vs native; wire + add to `WARP_BACKED`.
  Read `streaming_sdf_forces_post_*_kernel` in
  `src/kernels/csrc/cuda/streaming_sdf{,_2d}.cu`.

### Item 4 — BCs + interp (native `apply_bcs_*`, `interp_*`)
- `apply_bcs_*`: the live `AdvDiffSolver.set_BCs` calls native
  `torch.ops.lilytorch_kernels.apply_bcs_{2,3}d`.  The Warp wrapper exists but uses
  one `max_face_dim` vs native `(max_dim0, max_dim1)` — safe only on cubic faces;
  pass both face dims through.  Make dtype-generic.  Wire by overriding `set_BCs`
  in `src_warp/advection.py` to dispatch through `kernel.apply_bcs_*`; add to
  `WARP_BACKED`.
- `interp_*`: gather op (used on the marker/semi-Lagrangian path).  Make the Warp
  port dtype-generic and route `kernel.interp_*` to it; add to `WARP_BACKED`.
  (Lower priority — only fires on cases that use it.)

### Item 5 — σ path keys (native `bdim_coeff_sigma_{2,3}d`)
The σ Kernel B needs the packed `key_*` (body-id + SDF) arrays that the native
Kernel A emits but the Warp streaming bridge does not.  Extend the Warp streaming
kernels (`warp_kernels_2d.py` / `warp_kernels.py`) to write the same packed-key
arrays (or recompute the winning body-id in the σ Kernel B), so the σ branch of
`_fluid_step_kernel_{2,3}d` runs on Warp instead of `super()` (native).  Then the
override's `if self.apply_bdim_sigma: return super()...` gate can be removed.
Parity vs native σ; add `bdim_coeff_sigma_{2,3}d` to `WARP_BACKED`.

---

## DEFINITION OF DONE — enforce independence
1. Add a test (e.g. `test_src_trees.py::test_no_native_kernel_calls`) that runs a
   small **coupled step on the `src_warp` solver** at BOTH f32 and f64, in 2-D and
   3-D, with `torch.ops.lilytorch_kernels` monkeypatched to raise on call — the
   step must complete using only Warp.  (Or assert every `lilytorch_kernels` op
   the step would call is in `WARP_BACKED`.)
2. `WARP_BACKED` ⊇ every custom-kernel op in the table above.
3. **Item 6 — End-to-end validation (the deliverable):**
   - SU1 trajectory match vs native (`src_cuda`): 2-D `_1guillasim` pinned (f64)
     and 3-D jellyfish (f32); body trajectory / key fields within tolerance
     (document f32/FMA ULP drift).  Build the problem once, reuse (lesson 5).
   - Perf: <5% wall-clock regression vs native per step (CUDA-graph the per-step
     kernel sequence; precompute torch slice views before capture — lesson 15).
   - CPU end-to-end: a small case fully on CPU Warp kernels (one source, CPU+GPU),
     retiring the `.cpp` twins.  Tiled smoothers are GPU-only — use the
     thread-per-cell `warp_poisson*` smoother on CPU.

When the independence test passes at both dtypes in 2-D+3-D and Item 6 is green,
the port is COMPLETE.
