# Unify 2D/3D solver implementations

## Status

- [x] **Step 1** — dispatcher table (commit `33b8b09`).
- [x] **Step 2** — `_mu_normals_batched_2d/_3d` (already unified upstream
      in commit `9d67357`; verified at body.py:233 + thin per-D wrappers).
- [x] **Step 3** — forces leaves (commit `3233a44`, -135 lines).
- [x] **Step 4** — shared BDIM-apply loop + per-instance FREE dicts in
      `_fluid_step` (commit `1fa83ed`).  Note: did NOT collapse the two
      `_fluid_step_*d` method bodies into one — Heun (2-D-only) and
      FFT/multigrid bifurcation (3-D-only) make a single-body merge noisy
      without proportionate gain.  Helpers are now shared.
- [ ] **Step 5** — stacked-tensor storage refactor.  **Not started.**
      Per todo.md guidance, this is the deepest refactor (touches every
      callsite, FARMS bridge, kernels, plotting, HDF5 schemas) and needs
      explicit user sign-off before starting.
- [x] **Step 6 (partial)** — `_apply_forces_2d`/`_apply_forces_3d`
      collapsed into one `_apply_forces` (commit `d47ebf2`, -41 lines).
- [ ] **Step 6 (remainder)** — `_update_2d` / `_update_3d` /
      `_update_2d_streaming_multi` / `_update_3d_streaming_multi`
      (~1000 lines combined).  Deferred — needs full FSI regression
      coverage and FARMS axis bookkeeping cleanup.

Total LOC removed across Steps 3, 4, 6 (apply_forces): ~310 net.

## For the agent picking this up

**Base branch:** check out from `optimize_speed_memory` (the current dev branch with the latest 3D-solver speed/memory optimizations and kernel fixes).

```
git fetch origin
git checkout optimize_speed_memory
git pull --ff-only
git checkout -b unify-2d-3d/<step-name>
```

**Do NOT branch from `main` or `3d_solver`** — `optimize_speed_memory` already contains the kernel post-process paths, streaming-SDF multi-body changes, and the latest forces refactor. Branching from anywhere else will produce noisy merge conflicts in `solver.py`, `body.py`, `forces.py`, and `BDIMhandler.py`.

**One PR per step.** Each step is independently mergeable, regression-checked against existing benchmarks before moving on. Do not bundle steps. If a step turns out to be larger than expected, split it further rather than merging it with the next.

**Validation per step (mandatory):**
- 2D: `lilytorch/farms_examples/_1guillasim/gen_configs_one_pinned_2d.py` — short run, compare integrated forces / kinetic energy / vorticity snapshot vs. baseline on `optimize_speed_memory`.
- 3D: `lilytorch/farms_examples/jellyfish/run_jellyfish_fluid.py` (or any 3D example currently passing) — same comparison.
- Cost analysis: `lilytorch/validation/cost_analysis_free_swimming_2d/run_cost_analysis.py` and `..._3d/run_cost_analysis.py` should not regress wall-clock by more than ~5% (target: improvement).
- Bit-for-bit equivalence is NOT required (refactors may change op order); aim for relative error <1e-6 on integrated quantities.

**Do not change semantics.** Every step here is a refactor. No new physics, no new BC types, no new solver options. If you find a bug along the way, file it separately and keep the refactor PR clean.

**`torch.compile` caveat.** `torch.compile` specializes on shape and dtype. Do NOT collapse two specialized compiled leaves into one Python function in a way that forces re-tracing every step. The pattern is: a single Python dispatcher that routes by `ndim` to one of two compiled artifacts. Preserve `mode="reduce-overhead"` settings and the `_FS_FREE_AFTER_*` dicts.

---

## Step 1 — Dispatcher table (cheap, do first)

**Goal:** remove runtime `if self.ndim == 2:` branches in the hot path by binding the right method once at `__init__`.

**Files:**
- [lilytorch/src/solver.py](lilytorch/src/solver.py)
- [lilytorch/integration/BDIMhandler.py](lilytorch/integration/BDIMhandler.py)

**Tasks:**
1. In `FluidSolver.__init__`, after `self.ndim` is known, bind:
   - `self._fluid_step = self._fluid_step_3d if self.ndim == 3 else self._fluid_step_2d`
   - `self._recompute_mu_normals = self._recompute_mu_normals_3d if self.ndim == 3 else self._recompute_mu_normals_2d`
   - `self._bdim_apply` — only the 3D version exists today ([solver.py:1077](lilytorch/src/solver.py#L1077)). For 2D, factor the inline BDIM application out of `_fluid_step_2d` into `_bdim_apply_2d` first, then bind.
2. Remove the `step()` dispatcher branch at [solver.py:1796-1797](lilytorch/src/solver.py#L1796) — call `self._fluid_step(*args)` directly.
3. In `BDIMhandler.__init__` ([BDIMhandler.py:204-213](lilytorch/integration/BDIMhandler.py#L204-L213)), `self.update` is already dispatched; do the same for `self._apply_forces` so callers do not branch.
4. Remove every remaining `if self.ndim == 2:` / `if self.ndim == 3:` from per-step methods, EXCEPT in `__init__` (where it must stay to choose the dispatch).

**Acceptance:** zero `if self.ndim` branches outside `__init__` in the four target files. Diff should be ~50 lines net deletion. Run validation, commit, PR.

---

## Step 2 — Unify `_mu_normals_batched_2d` / `_3d`

**Goal:** one Python function with two compiled wrappers.

**File:** [lilytorch/src/body.py:276-301](lilytorch/src/body.py#L276-L301)

**Tasks:**
1. Define `_mu_normals_batched(sdf_stag, sdf_cc, h, eps)` where `sdf_stag` is either a tuple/list of `(sdf_u, sdf_v)` or `(sdf_u, sdf_v, sdf_w)`. Loop over the axis dimension; the math is identical.
2. Keep two compiled artifacts: `_mu_normals_batched_2d_compiled = torch.compile(lambda *a: _mu_normals_batched(...))` etc. so `torch.compile` still specializes per-D.
3. The standalone `compute_normals_3d_batched` at [body.py:924](lilytorch/src/body.py#L924) is already dim-agnostic via input — point its 2D callers at the unified function.
4. Update imports in `solver.py` ([solver.py:17-20](lilytorch/src/solver.py#L17-L20)) — keep the two compiled symbol names so `solver.py` does not change.

**Acceptance:** body.py shrinks by ~25 lines, no behavior change.

---

## Step 3 — Unify forces leaves

**Files:** [lilytorch/src/forces.py](lilytorch/src/forces.py)

Functions to merge:
- `_forces_shared_2d` ([forces.py:294](lilytorch/src/forces.py#L294)) + `_forces_shared_3d` ([forces.py:24](lilytorch/src/forces.py#L24))
- `_forces_body_batch_2d` ([forces.py:357](lilytorch/src/forces.py#L357)) + `_forces_body_batch_3d` ([forces.py:175](lilytorch/src/forces.py#L175))
- `_forces_body_integrate_3d` ([forces.py:96](lilytorch/src/forces.py#L96)) — has no 2D counterpart; leave alone.

**Tasks:**
1. Express viscous stress and pressure-gradient projection in terms of stacked component lists; use `torch.stack` for cross-products. The 2D "torque is a scalar" case becomes `cross_2d(a, b) = a[0]*b[1] - a[1]*b[0]` — keep a small helper.
2. Replace the two compiled-symbol bindings in [solver.py:566-608](lilytorch/src/solver.py#L566-L608) with calls into the unified function compiled twice (once per D).
3. `forces_method2_3d` at [forces.py:783](lilytorch/src/forces.py#L783) — rename to `forces_method2` once the 2D path (`forces_method2` in the same file) collapses into the same code. Currently the 2D path is the longer code at [forces.py:559-781](lilytorch/src/forces.py#L559-L781). They are NOT structurally identical — the 2D path has the legacy sparse fallback and the kernel-post path. Keep those branches but merge everything outside them.

**Acceptance:** forces.py shrinks by ~300-400 lines. Same numerical results to <1e-6.

---

## Step 4 — Unify `_fluid_step_2d` / `_3d`

**File:** [lilytorch/src/solver.py:1799 and 1859](lilytorch/src/solver.py#L1799)

**Prereq:** Steps 1-3 done. Reason: once mu/normals and forces are unified, the two `_fluid_step` methods differ only in `(u, v)` vs `(u, v, w)` plumbing.

**Tasks:**
1. Adopt a stacked velocity convention internally to `_fluid_step`: a tuple `vels = (u, v)` or `(u, v, w)`. Loop over components for advection-diffusion, BDIM, and pressure correction.
2. Pressure projection is already dim-agnostic ([poisson_fft.py](lilytorch/src/poisson_fft.py), [poisson_mult.py](lilytorch/src/poisson_mult.py)) — no change there.
3. Extend the `_FS_FREE_AFTER_BDIM_3D` / `_FS_FREE_AFTER_VAR_DENS_3D` memory-free dicts ([solver.py:184-191](lilytorch/src/solver.py#L184)) to a single dict keyed by component count, OR build them from a list comprehension over component names. The 2D path currently does NOT free intermediates between substeps — fix that as part of this step (stand-alone perf win).

**Acceptance:** one `_fluid_step` method, ~200 lines deleted from solver.py, 2D memory footprint reduced.

---

## Step 5 — Stacked-tensor storage (the deep refactor)

**Goal:** replace `(u0, v0, w0)`, `(nx, ny, nz)`, `(mu0_u, mu0_v, mu0_w)`, etc. with single tensors of shape `(D, *grid)`.

**Files:** all four target files, plus [lilytorch/src/operations.py](lilytorch/src/operations.py), [lilytorch/src/adv_diff.py](lilytorch/src/adv_diff.py), kernels.

**Risk:** highest. Touches every callsite, the FARMS bridge, kernel signatures, plotting, diagnostics, and HDF5 schemas. Coordinate with the user before starting — this is multi-day work.

**Tasks:**
1. Audit storage layout: list every attribute on `FluidSolver` and `CompositeBody` that has `_u`/`_v`/`_w` siblings. There are roughly 12 groups (`u/v/w`, `sdf_val_u/v/w`, `body_u/v/w`, `mu0_u/v/w`, `mu1_u/v/w`, `diff_u/v/w`, `nx/ny/nz`, etc.).
2. For each group, decide: stack into `(D, *grid)` tensor, or keep tuple? Tuple is fine where the kernel hits the symbol directly (avoids unnecessary indexing); stack is better where we currently loop over axes.
3. Update `_compute_sdfs_2d` / `_3d` ([body.py:1948, 2063](lilytorch/src/body.py#L1948)) to write into the new layout.
4. Update plotting ([plotting.py](lilytorch/src/plotting.py)) and diagnostics; HDF5 keys should keep `u/v/w` for backward compatibility — translate at the I/O layer.
5. Kernel signatures ([lilytorch/src/kernels/](lilytorch/src/kernels/)) — leave alone if they are bandwidth-bound and per-component; the unification is at the Python layer.

**Acceptance:** body.py and solver.py each shrink ~15-20%. Numerical equivalence within 1e-6. Memory profile (peak allocator stat) decreases on 3D runs by >5%.

---

## Step 6 — Unify BDIMhandler updates

**File:** [lilytorch/integration/BDIMhandler.py](lilytorch/integration/BDIMhandler.py)

Functions to merge:
- `_update_2d` ([line 708](lilytorch/integration/BDIMhandler.py#L708)) + `_update_3d` ([line 841](lilytorch/integration/BDIMhandler.py#L841))
- `_update_2d_streaming_multi` ([line 1315](lilytorch/integration/BDIMhandler.py#L1315)) + `_update_3d_streaming_multi` ([line 962](lilytorch/integration/BDIMhandler.py#L962))
- `_apply_forces_2d` ([line 1652](lilytorch/integration/BDIMhandler.py#L1652)) + `_apply_forces_3d` ([line 1708](lilytorch/integration/BDIMhandler.py#L1708))

**Why last:** the FARMS axis bookkeeping in 2D (`_2d_plane`, `_2d_ang_ax`, `_2d_force_axes`, `_2d_has_buoyancy` at [BDIMhandler.py:103-117](lilytorch/integration/BDIMhandler.py#L103-L117)) is the trickiest dim-dependent code in the project. Easier to refactor once everything underneath is dim-agnostic.

**Tasks:**
1. Replace the per-plane `if self._2d_plane == "xz":` / `else:` branches with a single index array `self._sim_axes` of length `ndim` (e.g. `[0, 2]` for xz-plane 2D, `[0, 1]` for xy-plane 2D, `[0, 1, 2]` for 3D). Read MuJoCo state through the index, write `xfrc` through it. Buoyancy axis becomes `self._buoyancy_axis = 1 if (ndim == 2 and "xz") else None`.
2. Merge the two `_apply_forces_*d` into one — they are nearly identical aside from torque dimensionality, which is already gated by `n_torque_comp = 1 if self.ndim == 2 else 3` at [BDIMhandler.py:709](lilytorch/integration/BDIMhandler.py#L709).
3. Merge the two `_update_*` and the two `_update_*_streaming_multi`. The streaming-SDF metadata (`comp._kernel_static_2d` / `_3d`) keeps separate names — the dispatcher selects which one to call.

**Acceptance:** BDIMhandler.py shrinks ~40%. All FARMS examples (2D pinned, 2D free, 3D jellyfish, 3D salamander) pass.

---

## Notes for the agent

- The user's preferred working branch is `optimize_speed_memory`. After each step's PR is merged, rebase the next step branch onto the updated `optimize_speed_memory`.
- After each PR merge, update [MEMORY.md](.) (project memory) with the architectural change and remove the corresponding TODO from this file.
- If a step blows up (e.g. `torch.compile` retracing every step, perf regression, numerical drift), STOP, write a `BLOCKER-stepN.md` next to this file describing the failure mode, and wait for the user. Do not work around it silently.
- Do not modify [FARMS_V2/](lilytorch/FARMS_V2/) — those are git submodules.
- `forces_method2_3d` is bound onto `FluidSolver` via `forces_method2_3d = forces.forces_method2_3d` at [solver.py:914](lilytorch/src/solver.py#L914). Same pattern for the 2D method. Keep that binding mechanism after unification (rename target if needed).

---

## Kernel parity audit (2D vs 3D, CPU + CUDA)

Audit of [streaming_sdf.cu](lilytorch/src/kernels/csrc/cuda/streaming_sdf.cu), [streaming_sdf_2d.cu](lilytorch/src/kernels/csrc/cuda/streaming_sdf_2d.cu), [streaming_sdf_cpu.cpp](lilytorch/src/kernels/csrc/streaming_sdf_cpu.cpp), [streaming_sdf_cpu_2d.cpp](lilytorch/src/kernels/csrc/streaming_sdf_cpu_2d.cpp). Operators (`min_rho`, `forces_post`, `apply_bcs`, `interpolate`) match in shape/intent across 2D and 3D, but several correctness and performance gaps remain. Items are listed in fix-order.

### Correctness (must fix before perf work)

- [x] **K1 — 3D CPU `forces_post` ignores `delta_order`.** *(fixed; CPU↔CUDA parity verified at ~1e-15 rel-err on delta_order∈{1,2}.)*
  [streaming_sdf_cpu.cpp:647](lilytorch/src/kernels/csrc/streaming_sdf_cpu.cpp#L647) declared the parameter; the function body never branched on it. CUDA-3D, CUDA-2D, and CPU-2D all apply the `(1/|∇sdf|)` correction when `delta_order == 2`. CPU-3D and CUDA-3D therefore returned different forces on the same input.
  **Fix landed:** ported the `delta_order==2` block from [streaming_sdf.cu:644-657](lilytorch/src/kernels/csrc/cuda/streaming_sdf.cu#L644-L657) — 6 extra body-SDF samples at `bxq ± r··*h` then divide both deltas by clamped `|∇|`.

- [x] **K2 — 2D CPU vs 2D CUDA `delta_order==2` use different gradient sources.** *(fixed; CPU↔CUDA parity verified at ~1e-15 rel-err.)*
  CUDA-2D ([streaming_sdf_2d.cu:524-543](lilytorch/src/kernels/csrc/cuda/streaming_sdf_2d.cu#L524-L543)) re-sampled body SDF at `bxq ± r··*h`. CPU-2D used cached `sparse_buf` neighbors with one-sided diffs at AABB edges (`cx = 1.0` when `di==0` or `di==Ai-1`). Disagreed at AABB-edge cells when bodies are tightly fitted.
  **Fix landed:** switched CPU-2D to resampling and dropped the two-pass `sparse_buf` design — single fused `at::parallel_for` now does cc-sample, band check, and (if `delta_order==2`) 4 extra resamples for `|∇|`. Closes K7 as well.

- [x] **K3 — `interpolate_*` (CPU and CUDA) dangling-pointer pattern.** *(fixed; verified with non-contiguous fp32 strided views into a fp64 grid on both CPU and CUDA, max|diff|=0 vs eager-cast reference.)*
  [streaming_sdf.cu:1083-1085](lilytorch/src/kernels/csrc/cuda/streaming_sdf.cu#L1083-L1085), [streaming_sdf_2d.cu:939-941](lilytorch/src/kernels/csrc/cuda/streaming_sdf_2d.cu#L939-L941), and the CPU equivalents called `xq.contiguous().to(F.scalar_type()).data_ptr<scalar_t>()`. The temporary tensor returned by `.to(...)` was destroyed at the next semicolon; the pointer dangled unless input was already contiguous and dtype-matched. CUDA worsened this — async kernel can read freed memory.
  **Fix landed:** bound temporaries to named locals before reading `data_ptr` in all four sites (CPU 2D/3D + CUDA 2D/3D).

### Performance parity

- [x] **K4 — 3D CPU paths are not parallelized.** *(fixed; 8-thread bench shows forces_post ~5× faster, min_rho ~1.6× faster vs 1-thread; CPU↔CUDA parity preserved at ~1e-15 rel-err.)*
  `streaming_sdf_min_rho_3d_multi_cpu` ([streaming_sdf_cpu.cpp:614](lilytorch/src/kernels/csrc/streaming_sdf_cpu.cpp#L614)) and `streaming_sdf_forces_post_3d_cpu` ([:711](lilytorch/src/kernels/csrc/streaming_sdf_cpu.cpp#L711)) iterated cells with plain `for`. Largest CPU-side perf gap.
  **Fix landed:** wrapped both per-body cell loops in `at::parallel_for` (grain 1024 for min_rho, 2048 for forces). Forces uses a per-body `acc[12]` + `std::mutex` with thread-local `local12[12]` chunks — matches the 2-D pattern. min_rho needs no merge because each cell touches its own `g`. K8 (replace mutex with thread-local indexed array) tracked separately.

- [x] **K5 — 3D CUDA `forces_post` shared-memory pressure.** *(fixed via CUB BlockReduce; CPU↔CUDA parity preserved at ~1e-15 rel-err.)*
  Replaced the hand-rolled 24 KB / block reduction with `cub::BlockReduce<double, BLOCK_SIZE>::Sum` called once per channel ([streaming_sdf.cu:672-686](lilytorch/src/kernels/csrc/cuda/streaming_sdf.cu#L672-L686), [streaming_sdf_2d.cu:564-577](lilytorch/src/kernels/csrc/cuda/streaming_sdf_2d.cu#L564-L577)). Internally CUB does warp shuffles + ~one shmem slot per warp; net per-block shmem drops from 24 KB → ~256 B (3D) and 12 KB → ~128 B (2D). Lifts the 2-block-per-SM occupancy ceiling on 48 KB-shmem consumer SMs. Direct kernel-level A/B not measured in-session — relies on parity check + CUB's well-known reduction perf.

- [x] **K6 — Forces kernels use fixed `blockSize = 256`.** *(fixed; bench confirms small-body launches now use bs=32, large bs=256.)*
  Forces launchers now mirror the `min_rho` adaptive sizing: `<=128 → 32`, `<=4096 → 128`, else 256. Required templating the kernels on `BLOCK_SIZE` (CUB's `BlockReduce` is compile-time-typed) and a 3-way `switch` in the launcher to dispatch to the right instantiation. Sites: [streaming_sdf.cu:885-907](lilytorch/src/kernels/csrc/cuda/streaming_sdf.cu#L885-L907), [streaming_sdf_2d.cu:744-783](lilytorch/src/kernels/csrc/cuda/streaming_sdf_2d.cu#L744-L783).

- [x] **K7 — CPU-2D `sparse_buf` allocated/freed per body per call.** *(fixed as a side effect of K2 — the entire two-pass design is gone.)*

- [x] **K8 — CPU-2D forces uses `std::mutex` for per-body merge.** *(fixed; bench shows ~30% additional speedup at 8 threads on top of K4 for A=32 bodies — 4.5× → 6.4× vs single-thread.)*
  Replaced the per-body `std::mutex` + `lock_guard` with a per-thread accumulator stripe `tls[t * N]` indexed by `at::get_thread_num()` ([streaming_sdf_cpu.cpp:707-769](lilytorch/src/kernels/csrc/streaming_sdf_cpu.cpp#L707-L769) for 3D 12-ch, [streaming_sdf_cpu_2d.cpp:601-810](lilytorch/src/kernels/csrc/streaming_sdf_cpu_2d.cpp#L601-L810) for 2D 6-ch). Each worker writes only its own stripe, so no locking is needed; final reduction across threads runs single-thread after the `parallel_for`.

### Minor

- [ ] **K9 — 2D CUDA `apply_bcs_2d` lacks `is_cuda` TORCH_CHECK.**
  3D version checks at [streaming_sdf.cu:991](lilytorch/src/kernels/csrc/cuda/streaming_sdf.cu#L991); 2D ([streaming_sdf_2d.cu:861-867](lilytorch/src/kernels/csrc/cuda/streaming_sdf_2d.cu#L861-L867)) checks contiguity / dtype only.

- [ ] **K10 — Dirichlet branch in `apply_bcs_*_kernel` writes dead `src_lin = 0`.**
  Harmless. Can drop or `[[maybe_unused]]`.

### Out of scope (noted, not actionable without restructuring)

- 3D `min_rho` allocates 4 × `Ngrid` × int64 of scratch ([streaming_sdf.cu:803-807](lilytorch/src/kernels/csrc/cuda/streaming_sdf.cu#L803-L807)) — 512 MB at 256³. Inherent to the parallel-B atomicMin design.
- `s_cc` is sampled twice per body per cell (once in `min_rho`, again in `forces_post`). True fusion would require restructuring the pipeline because `forces_post` needs the **decoded union** `sdf_cc` from all bodies.

### Recommended order

K1 → K3 → K4 → K2 (kills K7) → K5 + K6 → K8 → K9, K10. Each is a self-contained patch; K1 and K3 are correctness and should land first.
