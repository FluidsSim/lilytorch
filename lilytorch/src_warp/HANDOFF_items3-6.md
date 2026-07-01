# Handoff — finishing the native-independent Warp backend (Items 3–6)

This supersedes the *status* sections of `TASK_remaining_warp_wiring.md` (the
original brief).  **Items 1 and 2 are DONE, tested, and documented.**  Read this
first, then the original brief for the overarching goal + guardrails, then
`warp_poc/HANDOFF.md` lessons 2/3/9/13/14/15/17.

## Prereq reading (in order)
1. This file.
2. `src_warp/README.md` (WARP_BACKED table — current truth).
3. `TASK_remaining_warp_wiring.md` (goal + dtype-generic recipe + guardrails).
4. `warp_poc/VALIDATION_STATUS.md` §F (per-op state).

---

## GUARDRAILS (unchanged — do not violate)
- **No edits to `lilytorch/src/`.**  After every change:
  `git diff --name-only -- lilytorch/src` must show **only** `diagnostics.py`
  (a pre-existing change).  All wiring lives in `src_warp/`, all kernels in `warp_poc/`.
- Keep `src_warp` runnable at every step; **parity-test before flipping** a dispatch.
- `python -m pytest lilytorch/warp_poc/ -q` after each change. **Currently 197 pass.**
- Update `src_warp/README.md` (WARP_BACKED table) + `VALIDATION_STATUS.md` §F per op.
- Branch: `optimize_speed_memory`.  Env: warp 1.14, torch 2.6+cu124, CUDA (RTX 4080S).
- Repo path nesting: the package is at `/data/andreaferrario/lilytorch/lilytorch/…`
  (note the doubled dir).  Run pytest from `/data/andreaferrario/lilytorch`.

## The dtype-generic recipe (REUSE for every kernel port)
Warp 1.14 doesn't implicitly respecialise. For each kernel:
1. Value arrays → `wp.array(dtype=Any)`; float scalar args → `Any` (`from typing import Any`).
   Keep `int`/`int64` and incidental f32 tables concrete; cast inside via `type(x)(arr[i])`.
2. Float literals in the bound type: `half = type(x)(0.5)` (bind a *value*, not a type).
3. After the kernel def: `for dt in (wp.float32, wp.float64): wp.overload(K, {"arr": wp.array(dtype=dt), "scal": dt, ...})` (generic args only; `@wp.func` helpers auto-specialise).
4. Host wrapper: `wpf = wp.float64 if t.dtype==torch.float64 else wp.float32`; build flat
   views with `wpf`; pass scalars `wpf(value)`.
Worked examples done this session: `warp_bdim.py`, `warp_kernels.py`, `warp_poisson{,_2d}.py`.

---

## DONE THIS SESSION (do not redo)

### Item 1 — 3-D Kernel A/B on Warp ✅
- `warp_poc/warp_bdim.py` (Kernel B 3-D): f64→dtype-generic; keeps the **face-grid**
  coeff layout (HANDOFF lesson 9, 3-D-specific). f64 bit-identical, f32 to single prec.
- `warp_poc/warp_kernels.py` (Kernel A 3-D): f32→dtype-generic **and** added the
  velocity-blend (num/den softmin) path it was missing. `WarpStreamingSDF` takes `dtype`.
- `src_warp/kernel/__init__.py`: `_KernelA3DBridge` (z axis: `B*3` shapes/aabb, `B*21`
  kin, `gz`/`sdf_w`/`bW`); `bdim_coeff_3d` routes straight through. Both in `WARP_BACKED`.
- `src_warp/solver.py`: `_fluid_step_kernel_3d` override (σ path gated to native,
  `comp.sdf_val` pre-filled to `+FAR`).
- Tests: `test_bdim.py` (f32 Kernel B), `test_parity.py` (f64 Kernel A),
  `test_src_trees.py::test_kernelAB_3d_bridge_matches_native` (f32+f64 chain).
- **Perf/mem** (vs native CUDA): memory identical (0.0 MiB both); f64 at parity
  (0.97–1.23×); f32 1.06–1.34× on 128³ (Kernel A eager has Python-launch overhead,
  competitive under graph capture). No regression.

### Item 2 — Poisson driver on Warp ✅ (correct + independent; PERF GAP, see below)
- `warp_poc/warp_poisson_2d.py` + `warp_poisson.py`: RBGS/Jacobi smoother + residual
  dtype-generic (f32+f64); **added a 3-D Jacobi kernel**; native-signature host wrappers
  `rbgs_sweep_{2,3}d_warp`, `jacobi_sweep_{2,3}d_warp`, `mg_residual_{2,3}d_warp`.
  Neumann is folded into the stencil (ghost = self at boundary) — no separate BC launch.
- `src_warp/poisson_mult.py`: `PoissonSolver` subclass forces the three `solve_*` onto
  the **Python outer driver** (`use_kernels=False` inside each call → bypasses the C++
  `poisson_solve_*`) and overrides `_dispatch_vcycle` to run the fine-level
  smoother+residual on Warp; restriction/prolongation/coarse recursion + CG/Aitken loops
  are reused **pure-torch** from `src.poisson_mult` (no `lilytorch_kernels` op).
- **GOTCHA (cost me a debug cycle):** native `mg_residual_*` returns `+f + A·p`, which is
  the **negative** of the POC `residual_{2,3}d` kernel (`-f - A·p`). The wrapper returns
  `r.neg_()`. The coarse `_vcycle_*` expects the native sign or it diverges.
- **GOTCHA:** native residual reads p's actual ghost cells; the Warp residual clamps
  (Neumann). They agree only on a Neumann-BC'd p — when validating in isolation, BC the
  ghosts first or you'll see spurious diffs.
- Tests: `test_poisson_driver.py` — residual parity (multigrid+mgcg × rbgs+jacobi ×
  2D+3D × f32+f64), monkeypatch-`lilytorch_kernels`-to-raise independence, CPU solve.
- `WARP_BACKED` += `rbgs_sweep_{2,3}d`, `jacobi_sweep_{2,3}d`, `mg_residual_{2,3}d`.

#### Item 2 PERF — measured vs the native CUDA *kernel* path; graph closes it (3-D MG)
**Benchmark baseline = the native CUDA kernel path** (`use_kernels=False`: the Python
multigrid driver + native CUDA `rbgs_sweep`/`mg_residual` `.cu` kernels — the per-kernel
path Warp replaces), **NOT** the monolithic C++ `poisson_solve_*` fused driver (that's a
separate hand-fused optimisation).
- Item-2 Python-loop Warp ≈ native CUDA kernel path (1.0–1.1×) → the Warp smoother is on
  par with the native CUDA smoother kernel-for-kernel.  Both are dominated by the
  **Python multigrid driver** (per-launch overhead + the pure-torch coarse recursion with
  `interpolate`): ~57–62 ms / 6 V-cycles at 96–128³.
- `warp_poc/warp_mg_var.py::WarpMG3D` — an all-Warp, variable-coefficient, anisotropic,
  dtype-generic multigrid whose fixed-cycle (no early-exit, sync-free) V-cycle captures
  into ONE Warp CUDA graph — is **3–9× faster than the native CUDA kernel path**
  (96³ f32 7.1 ms vs 57 ms; 128³ f32 18.9 ms vs 62 ms; 64³ f64 5.3 ms vs 47 ms) with
  **0 MiB per-step allocation** (persistent level buffers).  It wins because the cycle is
  Warp-only → graph-capturable end-to-end, which the native per-kernel path (torch coarse
  recursion) is not.  Wired into `solve_multigrid` behind `cuda_graph=True` (lazy
  per-shape cache).  Tests: `test_poisson_driver.py::test_warp_poisson_graphed_multigrid`
  + `::test_warp_poisson_graphed_independent`.
- For *absolute* parity with the monolithic C++ fused driver (~4 ms at 128³, still faster
  than the graphed MG there) you need the sync-free graphed **mgcg** below.

**STILL TO DO (Item 6 perf):**
- **2-D graphed multigrid** — mirror `WarpMG3D` as `WarpMG2D` (use the 2-D transfer
  kernels in `warp_multigrid_2d.py`; `_graphed_mg` already returns `None` for ndim=2 so
  it falls through to the Python path). Needed for the 2-D `_1guillasim` (f64) target.
- **mgcg / rmgcg graphing** — the CG outer loop has dot-product `.item()` syncs that the
  graphed V-cycle doesn't remove. To match native C++ mgcg you need a sync-free
  fixed-iteration CG (dot products accumulated on-GPU, no `.item()`), with the V-cycle
  preconditioner above. Jellyfish uses mgcg → this is the one that matters for it.
  GOTCHA discovered building `WarpMG3D`: the smoother folds Neumann by clamping ghosts
  (never writes them), so `mg_residual` MUST clamp ghosts identically (it now does) or
  the operator/residual disagree at the boundary when domain-boundary face coeffs are
  non-zero → divergence. Also feed it the SAME `f_scaled` (h²-scaled) as the native
  outer driver. Face-pair slices (cv[:,1:], cw[:,:,1:]) are non-contiguous → can't be
  flat-viewed for a persistent graph pointer; `extract_pairs_3d` materialises the six
  contiguous pair buffers in-graph each step.

---

## REMAINING WORK

### Item 3 — Forces ✅ DONE (both readouts on Warp)
3a (Lagrangian) + 3b (Eulerian) ported, wired, parity-tested. Wiring: the inherited
`forces_lagrangian_{2,3}d` / `forces_method2{,_3d}` call module-globals in
`lilytorch.src.forces`; the `src_warp.solver.FluidSolver` subclass overrides those four
methods and swaps the global for the Warp facade shim for the call (localized injection,
same as the `__init__` sub-solver swap — no `lilytorch.src` edits). Facade shims
(`lagrangian_forces_{2,3}d`, `streaming_sdf_forces_post_{2,3}d`) added to `WARP_BACKED`.
3b's `warp_forces.py` replaces the native CUB `BlockReduce` with one `wp.atomic_add` per
cell into the float64 accumulator (identical sum, ~1e-9 reduction-order noise only),
reuses the Kernel-A `sdf_sample_off_*` samplers (+ a 3-D triquadratic-with-offset), and
ports the deltaH ∂H pressure pass. Parity: `test_lagrangian.py` (3a, f32+f64) +
`test_forces.py` (3b, 2-D+3-D × ndelta+deltaH × delta_order 1/2 × f32+f64); routing:
`test_src_trees.py::test_{lagrangian,eulerian}_force_override_routes_to_warp`. **242
pytest pass; `git diff --name-only -- lilytorch/src` still only `diagnostics.py`.**
The original 3a/3b briefs are kept below for reference.

#### (historical brief)
`src_warp/forces.py` originally re-exported native (`from lilytorch.src.forces import *`).
The native force methods are module-level functions bound as `FluidSolver` methods.

**3a — Lagrangian (tractable; port EXISTS).** `warp_poc/warp_lagrangian.py` has
`lagrangian_forces_{2,3}d_warp` (parity ≤1.7e-16) but is **f64-only** → make it
dtype-generic (recipe). The native call sites are `forces_lagrangian_2d` (src/forces.py
~L1120) and `forces_lagrangian_3d` (~L1348); they already build the decomposed args
(`eps_ij[..]`, `tri_centroid/normal/area`, `tri_offsets`, `com_pos`, `bx0..`,
`inv_dx..`, `Mx..`, `method`, `sample_offset`, `out=(B,12 or analogous)`). Override
those two methods in `src_warp` to call `kernel.lagrangian_forces_{2,3}d` (route the
Warp wrapper through the facade). Add to `WARP_BACKED`; parity test in `test_forces.py`.

**3b — Eulerian (the ONE unported kernel — write it).** `streaming_sdf_forces_post_{2,3}d`
is the n·δ viscous+pressure band integral (+ deltaH ∂H pass), used by `forces_method2`
(src/forces.py ~L413) and `forces_method2_3d` (~L634). Read the CUDA source
`src/kernels/csrc/cuda/streaming_sdf{,_2d}.cu` (`streaming_sdf_forces_post_*_kernel`).
Write `warp_poc/warp_forces.py` — a block-reduction + `wp.atomic_add` scatter into the
per-body force row, **same class as `warp_lagrangian.py`** (use `wp.atomic_add` on a
float64 accumulator; mind the f32/f64 atomic note at the top of `warp_lagrangian.py`).
Make it dtype-generic, parity vs native in `test_forces.py`, wire behind the method2
call sites, add to `WARP_BACKED`.
**Note on submethods:** `force_submethod` ∈ {`ndelta`, `deltaH`}; deltaH adds a union-∂H
pass split by a softmin partition. Match whichever the validation case uses (check
`forces_method2_3d` for the deltaH branch). Start with `ndelta`, add `deltaH` after.

### Item 4 — BCs + interp  (ports EXIST; only wiring left)
The Warp kernels are already written + parity-tested in `warp_poc/warp_misc_2d.py` /
`warp_misc_3d.py` (`apply_bcs_{2,3}d_warp`, `interp_{2,3}d_warp`; tests
`warp_poc/test_misc_{2,3}d.py`).  The facade still points `apply_bcs_*`/`interp_*` at
native (`src_warp/kernel/__init__.py`).  Remaining:
- `apply_bcs_{2,3}d`: **WIRING NUANCE** — unlike forces/lagrangian, the native
  `AdvDiffSolver.set_BCs` (src/advection.py ~L1124) calls
  `torch.ops.lilytorch_kernels.apply_bcs_2d(...)` **directly** (not a module global), so
  the module-global-swap trick does NOT apply.  Instead **override `set_BCs` in
  `src_warp/advection.py`** (the subclass already exists) to dispatch the same cached
  descriptors through `kernel.apply_bcs_*`.  Before flipping, confirm the Warp wrapper
  handles non-cubic faces: the 2-D wrapper takes one `max_line_dim` (fine, 2-D faces are
  1-D); the **3-D** wrapper must pass both face dims (`max_dim0, max_dim1`) — check
  `apply_bcs_3d_warp` and the native `apply_bcs_3d_kernel` face-dim layout.  Make the
  Warp port dtype-generic if it isn't (recipe).  Route `kernel.apply_bcs_*` → Warp in the
  facade; add to `WARP_BACKED`.
- `interp_{2,3}d`: gather op (marker / semi-Lagrangian path only — lower priority; the
  rigid-streaming step never calls it).  Make the Warp port dtype-generic, route
  `kernel.interp_*` → Warp, add to `WARP_BACKED`.  (No `set_BCs`-style call-site nuance;
  `interp_*` IS reached through the `kernel.*` facade where it's used.)

### Item 5 — σ path keys (removes the last native gate in the step)
Both `_fluid_step_kernel_{2,3}d` overrides currently do
`if self.apply_bdim_sigma: return super()...` (native) because the Warp streaming
bridge does **not** emit the packed `key_*` (body-id + SDF) arrays that the σ Kernel B
(`bdim_coeff_sigma_{2,3}d`) reads. Fix: extend the Warp streaming kernels
(`warp_kernels_2d.py` / `warp_kernels.py`) to write the same packed-key arrays (the
native Kernel A emits them; mirror the bit layout — body-id in low 32 bits, SDF f32 in
high bits, see the native `.cu`), OR recompute the winning body-id inside σ Kernel B.
Then route `bdim_coeff_sigma_{2,3}d` through the Warp σ kernel (already exists in
`warp_bdim{,_2d}.py` behind the `sigma_shifts=` kwarg — now dtype-generic), remove the
native gate in both overrides, add `bdim_coeff_sigma_{2,3}d` to `WARP_BACKED`. Parity vs
native σ.

### Item 6 — Definition of done
1. **Independence test** `test_src_trees.py::test_no_native_kernel_calls`: run a small
   coupled step on the `src_warp` solver at f32 AND f64, 2-D AND 3-D, with
   `torch.ops.lilytorch_kernels` monkeypatched to raise — the step must complete on Warp
   only. (Build the BDIMhandler+solver minimal scene; the per-op independence pattern is
   shown in `test_poisson_driver.py::test_warp_poisson_independent_of_native_ops`.)
2. `WARP_BACKED` ⊇ every custom-kernel op the step calls.
3. **End-to-end trajectory match** vs `src_cuda`: 2-D `_1guillasim` pinned (f64) +
   3-D jellyfish (f32); body trajectory / key fields within tol (document f32/FMA ULP
   drift). Build the problem once, reuse.
4. **Perf <5%/step** vs native: **the Poisson CUDA-graph (sync-free fixed-cycle Warp
   V-cycle) is the critical piece** — see the Item-2 perf gap above.  Also CUDA-graph the
   remaining eager paths (precompute torch views before capture — lesson 15): Kernel A,
   **and the two force kernels** (single launch for ndelta, two for ndelta+deltaH).
   Force-kernel perf is already characterised (README "Force-kernel perf/mem"): GPU
   compute is at native parity (3-D f64 64³: native 0.069 ms, Warp raw 0.071 ms, Warp
   graph 0.069 ms) and memory is at parity (0 global scratch); the only gap is the eager
   *wrapper* host floor (per-call `wp.synchronize()` + flat-view rebuild + int64 casts).
   So the force fix is: drop the wrapper `wp.synchronize()`, cache the flat `wp.from_torch`
   views, capture into a graph — exactly the lever this item already calls for.
5. **CPU end-to-end**: a small case fully on CPU Warp kernels (thread-per-cell
   `warp_poisson*` smoother on CPU — tiled smoothers are GPU-only). Per-op CPU is proven
   (`test_cvof_warp_cpu_matches_python_reference`, `test_warp_poisson_cpu`).

---

## PROVEN WIRING PATTERN (use it for Item 5; it does NOT fit Item 4 apply_bcs)
For a native method that calls a **module-global** kernel name, route to Warp by a
subclass method override that swaps the global for the call, then restores it — no
`lilytorch.src` edits, no body duplication.  Used 4× and tested:
`src_warp/solver.py` `forces_lagrangian_{2,3}d` + `forces_method2{,_3d}` swap
`lilytorch.src.forces.{_lagrangian_forces_*_kernel, streaming_sdf_forces_post_*}`; the
`__init__` does the same for `AdvDiffSolver`/`PoissonSolver`.  Template:
```python
def forces_method2(self, u, v, p, it):
    _save = _forces_mod.streaming_sdf_forces_post_2d
    _forces_mod.streaming_sdf_forces_post_2d = kernel.streaming_sdf_forces_post_2d
    try:    return super().forces_method2(u, v, p, it)
    finally: _forces_mod.streaming_sdf_forces_post_2d = _save
```
Routing test template: `test_src_trees.py::test_{lagrangian,eulerian}_force_override_routes_to_warp`
(monkeypatch the base method to capture the bound global; assert it's the Warp shim mid-call
and restored after).  **Exception:** `set_BCs` calls `torch.ops...apply_bcs` *directly*
(not a global) → Item 4 needs a real `set_BCs` override, not this swap.

## STATUS UPDATE — Items 4, 5, 6 (this session)
- **Item 4 (BCs + interp) — DONE.** `apply_bcs_{2,3}d`/`interp_{2,3}d` made
  dtype-generic; 3-D `apply_bcs` wrapper takes both face dims `(max_dim0,
  max_dim1)` (non-cubic safe); `set_BCs` overridden in `src_warp/advection.py`
  dispatching through `kernel.apply_bcs_*`; facade routed; `WARP_BACKED` += 4.
  Tests: `test_misc_{2,3}d` (f32 + non-cubic), `test_src_trees::
  test_set_bcs_override_routes_to_warp`.
- **Item 5 (σ keys) — DONE.** The Warp streaming Kernel A emits the winning
  body-id into `key_{u,v[,w]}` on an `emit_keys` path (Pass-C int64 `atomic_min`,
  lowest-id-wins, sentinel=B; 2-D full-grid / 3-D dirty-local — matching
  `bdim_coeff_sigma_*`'s read).  Both `_fluid_step_kernel_{2,3}d` overrides drop
  the native σ gate and route the σ Kernel B with keys + `sigma_shifts`.
  `WARP_BACKED` += `bdim_coeff_sigma_{2,3}d`.  Parity:
  `test_src_trees::test_kernelAB_{2,3}d_sigma_chain_matches_native` (f32).
- **Item 6 — independence + CPU DONE; trajectory + perf-graph REMAINING.**
  `test_warp_backed_covers_step_custom_ops` (static) +
  `test_no_native_kernel_calls_{2,3}d` (dynamic, monkeypatch-to-raise, f32+f64) +
  `test_kernelAB_2d_chain_cpu_eq_gpu` (CPU==GPU, plain+σ).  **Remaining:** (3)
  full coupled trajectory match vs `src_cuda` (eel pinned f64 / jellyfish f32 —
  needs the FARMS/MuJoCo bridge); (4) perf <5%/step CUDA-graph capture (set_BCs
  128³ ≈ 131 µs warp vs 25 µs native = eager-wrapper floor, ~1% of a
  Poisson-bound step; levers = WarpMG2D + sync-free graphed mgcg + graph-capture
  of eager Kernel-A/forces/apply_bcs).
- Guardrail held throughout: `git diff --name-only -- lilytorch/src` → only
  `diagnostics.py`.

## Test baseline & guardrail (re-verify after every change)
- `python -m pytest lilytorch/warp_poc/ -q` → **264 pass** (run from `/data/andreaferrario/lilytorch`).
- `git diff --name-only -- lilytorch/src` → **only `diagnostics.py`** (+ no untracked under `src/`).
- Branch `optimize_speed_memory`; warp 1.14, torch 2.6+cu124, RTX 4080S; doubled path
  `/data/andreaferrario/lilytorch/lilytorch/…`.

## Quick map of what's wired where
- Dispatch facade + WARP_BACKED: `src_warp/kernel/__init__.py` (22 ops backed).
- Step overrides + force overrides: `src_warp/solver.py`.
- Poisson: `src_warp/poisson_mult.py` + `warp_poc/warp_poisson{,_2d}.py` + `warp_mg_var.py`.
- Advection/BCs: `src_warp/advection.py` (set_BCs override = Item 4). Two-phase: `src_warp/two_phase.py`.
- Forces: `warp_poc/warp_forces.py` + `warp_lagrangian.py` (Item 3 ✅, routed from `src_warp/solver.py`).
- BCs/interp ports: `warp_poc/warp_misc_{2,3}d.py` (Item 4 — wiring left).
- Kernels live in `warp_poc/`; parity tests are `warp_poc/test_*.py`.
