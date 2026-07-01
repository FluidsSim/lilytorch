# `src_warp/` — Warp single-source kernel backend

Parallel backend tree to [`src_cuda/`](../src_cuda/README.md).  Kernel-dispatching
solver modules here call the single-source `@wp.kernel` ports (in
[`warp_poc/`](../warp_poc/)) through the unified backend API in
[`kernel/`](kernel/__init__.py).  **One kernel source runs on CPU *and* GPU**,
retiring the hand-written `.cu`/`.cpp` twins for the ops that are wired in.

This is the §F "end-to-end swap-in" of the Warp port, realized as a parallel
backend tree (rather than an in-`src/` `use_warp_kernels` flag), per the repo
owner's request.  `lilytorch/src/` is **untouched**; both backend trees import
the kernel-agnostic modules (`body`, `plotting`, `poisson_fft`, `operations`,
`diagnostics`, …) from there.

## Layout
```
src_warp/
  kernel/__init__.py     # unified backend API: Warp where wired, native fallback
  solver.py              # FluidSolver: injects Warp sub-solvers at __init__
  two_phase_solver.py    # TwoPhaseSolver: Warp sub-solvers + Warp VOF + forces
  backend.py             # resolve_solver_class(backend, two_phase) — config seam
  advection.py           # AdvDiffSolver: flux on Warp advect_flux_add
  two_phase.py           # TwoPhase: VOF sweep on Warp cvof_sweep
  poisson_mult.py        # PoissonSolver: multigrid/MGCG on Warp smoother+residual
  forces.py              # lagrangian on Warp (override); Eulerian still native
```

## Selecting the Warp backend from a config (`solver.backend`)
Opt-in, default `"native"` (existing configs unaffected).  Set in the YAML /
config `solver` block:
```yaml
solver:
  backend: warp        # "native" (default) | "warp"
  solver_method: kernel   # unchanged — "kernel" | "python" (the STEP path, NOT
                          # the backend; "warp" is NOT a valid solver_method)
```
`solver.backend` and `solver_method` are orthogonal: `backend` picks native-CUDA
vs the single-source Warp tree; `solver_method` picks the streaming Kernel-A/B
step (`kernel`) vs the python reference loop (`python`) *within* that backend.
The key is resolved by `src_warp.backend.resolve_solver_class` and honored by
`integration/BDIMhandler.py` (coupled FARMS scenes) and the standalone example
drivers (e.g. `farms_examples/jellyfish/run_jellyfish_fluid.py`).  For a script
that builds the solver directly, just import from `src_warp` instead
(`from lilytorch.src_warp.solver import FluidSolver`).

## What runs on Warp today (`kernel.WARP_BACKED`)
| op | path | status |
|----|------|--------|
| `advect_flux_add` | `AdvDiffSolver._solve_convective` fused-flux | **Warp**, dtype-generic (f64 bit-exact vs native all 5 schemes 2-D+3-D; f32 to single precision), CPU==GPU |
| `cvof_sweep` | `TwoPhase._cvof_sweep` | **Warp**, dtype-generic (f64 bit-exact 2-D+3-D; f32 to single precision), CPU==GPU |
| `streaming_sdf_stag_2d_multi` (Kernel A, 2-D) | `_fluid_step_kernel_2d` via marshalling bridge | **Warp**, dtype-generic (f32 bit-exact, f64 to ~1e-7 — native interpolates the SDF in f32), CPU==GPU |
| `bdim_coeff_2d` (Kernel B, 2-D) | `_fluid_step_kernel_2d` | **Warp**, dtype-generic (f64 bit-exact, f32 to single precision) — no native fallback |
| `streaming_sdf_stag_3d_multi` (Kernel A, 3-D) | `_fluid_step_kernel_3d` via marshalling bridge | **Warp**, dtype-generic (f32 bit-exact, f64 to ~1e-7), + velocity-blend path, CPU==GPU |
| `bdim_coeff_3d` (Kernel B, 3-D) | `_fluid_step_kernel_3d` | **Warp**, dtype-generic (f64 bit-exact, f32 to single precision) — no native fallback |
| `bdim_coeff_sigma_{2,3}d` (σ Kernel B, thin bodies) | `_fluid_step_kernel_{2,3}d` σ branch | **Warp** (Item 5): the streaming Kernel A emits the winning body-id into `key_{u,v,w}` (2-D full-grid / 3-D dirty-local, lowest-id-wins via int64 `atomic_min`, sentinel = B); the σ Kernel B masks `key & 0xffffffff`. No native gate; parity vs native in `test_src_trees::test_kernelAB_{2,3}d_sigma_chain_matches_native` |
| `rbgs_sweep_{2,3}d` / `jacobi_sweep_{2,3}d` / `mg_residual_{2,3}d` (Poisson) | `PoissonSolver._dispatch_vcycle` | **Warp**, dtype-generic (f32+f64), CPU+GPU; multigrid/MGCG/RMGCG outer driver in Python, native C++ `poisson_solve_*` bypassed; residual-level parity vs native |
| `lagrangian_forces_{2,3}d` (Lagrangian surface force) | `forces_lagrangian_{2,3}d` override (`src_warp.solver`) | **Warp**, dtype-generic (f64 bit-exact, f32 to single precision; per-element math in field dtype, float64 atomic accumulator), CPU==GPU |
| `streaming_sdf_forces_post_{2,3}d` (Eulerian n·δ band integral + deltaH) | `forces_method2{,_3d}` override (`src_warp.solver`) | **Warp**, dtype-generic (f64/f32; per-element math in field dtype, float64 atomic accumulator); ndelta + deltaH ∂H pass; parity vs native to atomic-reduction noise |
| `apply_bcs_{2,3}d` (fused Neumann/Dirichlet/reflective ghost-line writes) | `AdvDiffSolver.set_BCs` override (`src_warp.advection`) | **Warp**, dtype-generic (f32+f64, bit-exact per-op on disjoint ops), 3-D wrapper takes both face dims `(max_dim0, max_dim1)` for non-cubic grids; CPU+GPU. The native `set_BCs` calls `torch.ops…` directly, so this is a real method override (not the global swap). **CUDA-graph-cached** (`ApplyBcs{2,3}DGraphRunner`, default on, memory-free): 128³ f32 `set_BCs` **131 µs eager → 11.3 µs = native parity (1.00×)** |
| `interp_{2,3}d` (scattered bilinear/trilinear/quadratic gather) | `kernel.interp_*` facade (marker / semi-Lagrangian path; not in the rigid-streaming step) | **Warp**, dtype-generic (f32+f64; CPU bit-exact, GPU to ~1 ULP from FMA order), reuses the Kernel-A samplers |

**Force-kernel perf/mem (vs native CUDA).** *Memory:* at parity — both allocate **0
global scratch** (zero-copy views; register/atomic reduction replaces native's CUB
block-reduce); the Lagrangian `elem_body` map is now **cached** on the offsets
buffer (it was rebuilt per call via a `repeat_interleave` that forced a D2H sync).
*Compute:* the GPU kernel is native-equivalent.  The per-call **`wp.synchronize()`
is removed** from all four force wrappers (the caller reads `out` through torch,
which orders after the launch on the legacy null default stream — the sync was a
pure latency floor, not a correctness need).  Result (3-D f64, `force_submethod=ndelta`):
- **Eulerian `streaming_sdf_forces_post_3d` BEATS native at realistic grids:**
  96³ 267 µs vs 278 µs (**0.96×**), 128³ 417 µs vs 489 µs (**0.85×**).  At small
  grids the fixed ~144 µs view-wrapping floor shows (48³ 2.4×, 64³ 1.3×).
- **Lagrangian `lagrangian_forces_3d`:** the `elem_body` cache cut the per-call
  floor 231 µs → 160 µs; native scales with triangle count, so Warp is competitive
  at real mesh sizes (the 244-triangle micro-bench is floor-bound).
Remaining force lever (not blocking): CUDA-graph the ndelta launch (cache the flat
views, fold the `out.zero_()` into the graph) to also win at small grids.

`advect_flux_add` / `cvof_sweep` are *signature-identical drop-ins* (pure
dispatch-target swap).  The 2-D Kernel A/B are wired through a **marshalling
bridge** (`kernel._KernelA2DBridge`) re-expressing the live flat-table tensors
into `WarpStreamingSDF2D`'s setup/update/run API, plus an overridden
`FluidSolver._fluid_step_kernel_2d`.  Kernel A and Kernel B are both
**dtype-generic** (one `@wp.kernel` source, `wp.overload` f32+f64).  The σ path
also runs on Warp (Item 5): the bridge emits the winning body-id into `key_*`
on its `emit_keys` path, and the σ Kernel B reads it.  Verified in
`warp_poc/test_src_trees.py` (incl. the **CPU single-source payoff**, the f64
Kernel A/B chain-vs-native parity, and the σ chain parity).

## What still falls back to native (the §F remaining work)
The Warp kernels for all of these are **already ported and parity-clean** in
`warp_poc/` (202 tests); what remains is the *in-solver wiring*, not the kernels:

1. ~~**Kernel A/B (3-D)**~~ — **DONE** (Item 1): `warp_kernels.py`/`warp_bdim.py`
   are dtype-generic (f32+f64), wired via `kernel._KernelA3DBridge` + an overridden
   `_fluid_step_kernel_3d`; parity in `test_parity.py` (f64), `test_bdim.py` (f32),
   and `test_src_trees.py::test_kernelAB_3d_bridge_matches_native` (f32+f64 chain).
   The σ path also runs on Warp (Item 5 — body-id key emission).
2. ~~**Poisson driver**~~ — **DONE** (Item 2).  `poisson_mult.PoissonSolver`
   subclasses the native solver, forces the three `solve_*` entry points onto the
   Python outer driver (bypassing the C++ `poisson_solve_*`), and overrides
   `_dispatch_vcycle` to run the fine-level smoother + residual on the
   single-source Warp kernels (`warp_poisson{,_2d}`, dtype-generic f32+f64, CPU+GPU;
   Neumann folded into the stencil).  Restriction/prolongation/coarse recursion and
   the CG/Aitken loops are reused pure-torch (no `lilytorch_kernels` op).  Residual
   parity vs native in `test_poisson_driver.py` (incl. monkeypatch-to-raise
   independence + a CPU solve).
   **Perf (vs the native CUDA *kernel* path — `use_kernels=False`, the per-kernel
   `.cu` `rbgs_sweep`/`mg_residual` ops Warp replaces; NOT the monolithic C++
   `poisson_solve_*` fused driver):** the Item-2 Python-loop Warp ≈ the native CUDA
   kernel path (1.0–1.1×, confirming the Warp smoother ≈ native kernel-for-kernel) —
   both are dominated by the **Python multigrid driver** (per-launch + pure-torch
   coarse recursion), ~57–62 ms / 6 V-cycles at 96–128³.  The all-Warp,
   variable-coefficient, CUDA-graph-captured fixed-cycle multigrid
   `warp_mg_var.WarpMG3D` (opt-in `cuda_graph=True`) is **3–9× faster than the native
   CUDA kernel path** (96³ f32 7.1 ms vs 57 ms; 128³ f32 18.9 ms vs 62 ms) with
   **0 MiB per-step allocation** (persistent level buffers) — the cycle is Warp-only
   so it captures end-to-end into one graph, which the native per-kernel path (torch
   coarse recursion) cannot.  The monolithic C++ `poisson_solve_*` fused driver is
   still faster in absolute terms at small grids (~4 ms at 128³).
   **Sync-free graphed MGCG — DONE** (`_dispatch_vcycle` routes the CG
   preconditioner through a CUDA-graphed `WarpMG{2,3}D`, one captured V-cycle per
   `precond_vcycles` step, opt-in `cuda_graph=True`): **4.3–14× over the Item-2
   Python-driver MGCG** (3-D f32: 64³ 24.4→1.7 ms, 96³ 27.1→2.8 ms, 128³ 27.7→6.4 ms),
   same residual.  `-r` is passed un-rescaled (already h²-scaled in the SPD units).
   Optional periodic residual check (`poisson_cg_check_every`, default 1 = native)
   only helps at high CG-iteration counts.  2-D graphed MG (`WarpMG2D`) also wired.
3. ~~**`apply_bcs_*`**~~ — **DONE** (Item 4).  `warp_misc_{2,3}d` are
   dtype-generic (f32+f64); the 3-D wrapper takes both face dims
   `(max_dim0, max_dim1)` (non-cubic safe), wired behind a `set_BCs` override in
   `src_warp.advection.AdvDiffSolver` (the native `set_BCs` calls `torch.ops…`
   directly).  `interp_{2,3}d` (marker/semi-Lagrangian gather) is also made
   dtype-generic and routed.  Parity in `test_misc_{2,3}d` (incl. f32 + a
   non-cubic dual-face-dim case); routing in
   `test_src_trees.py::test_set_bcs_override_routes_to_warp`.
4. **Forces** — **DONE** (Item 3).  *Lagrangian* `lagrangian_forces_{2,3}d` (3a):
   `warp_lagrangian.py` is dtype-generic (f32+f64); the Warp shims have the native
   `ops` arg list (incl. `out=`), wired behind the inherited `forces_lagrangian_{2,3}d`
   by an `src_warp.solver` method override (localized module-global swap, no
   `lilytorch.src` edits).  Parity in `test_lagrangian.py`.  *Eulerian*
   `streaming_sdf_forces_post_{2,3}d` (3b): newly written `warp_forces.py` — the n·δ
   viscous+pressure band integral (+ deltaH ∂H pressure pass), block-reduction
   replaced by per-cell `wp.atomic_add` into the float64 accumulator (same sum,
   reduction-order noise only); dtype-generic, wired behind `forces_method2{,_3d}`
   by the same override pattern.  Parity in `test_forces.py` (2-D+3-D, ndelta+deltaH,
   delta_order 1/2, f32+f64, scalar/field nu_rho); routing in
   `test_src_trees.py::test_{lagrangian,eulerian}_force_override_routes_to_warp`.
   **No native custom force kernel remains.**

## Validate end-to-end
- **2-D `_1guillasim` pinned (f64) — DONE** (C2, 2026-06-30).  Real headless
  FARMS/MuJoCo coupled run, native vs Warp backend, via
  `lilytorch/validation/warp_e2e/run_c2.py` (backend swapped by monkeypatching
  `BDIMhandler.FluidSolver` through `_extra_run_patch` — no core edits).
  Warp ≡ native to **residual level** (field rel-L2 p ~1e-8, qpos ~1e-11) for
  ~325 steps, then a deterministic discrete divergence (the documented Kernel-A
  f32-SDF interp difference crossing a discretization threshold — proven not
  chaos by a perturbation sweep; both runs stay stable, final qpos Δ=3.4e-5 in
  band).  **Perf: end-to-end Warp 1.45× (eager) / 1.68× (Kernel-A graph) faster
  than native** (3.73 / 3.22 vs 5.42 ms/step), past the "~5 %, ideally faster"
  target.  See `HANDOFF_perf_remaining.md` §C2.
- **3-D jellyfish (f32, two-phase, python path) — DONE** (C2(b),
  `validation/warp_e2e/run_c2_jelly.py`).  Standalone driver; Warp injected by
  rebinding the `AdvDiffSolver`/`PoissonSolver`/`TwoPhase` module globals (+
  `src.forces` kernels) before the native `TwoPhaseSolver` builds (no Warp
  two-phase subclass; no `src/` edits).  Exercises Warp advection / var-density
  Poisson / `cvof_sweep` / `apply_bcs` / forces (NOT Kernel A/B).  **Eager MGCG:
  clean f32 parity over 120 steps** (com 2.5e-9, linvel 3.4e-6, alpha_sum 1.7e-7,
  fields at f32 round-off, bounded).  Perf 128³: warp-eager 0.94× native; C1
  **graphed** MGCG 1.18× but under-converges the stiff 1000:1 two-phase Poisson at
  default cycles → eager is the parity path.  See `HANDOFF_perf_remaining.md`
  §C2(b).
- Full CPU run on Warp kernels — the per-op CPU single-source is proven
  (`test_src_trees.py::test_cvof_warp_cpu_matches_python_reference`); the
  end-to-end CPU run follows once the step path is fully Warp.

## Tests
```
python -m pytest warp_poc/test_src_trees.py -q   # structure/parity/CPU + force routing
python -m pytest warp_poc/ -q                     # 277 total
```
