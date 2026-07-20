# Status: removing the python force path + deltaH  (COMPLETE 2026-07-19)

> ✅ **DONE 2026-07-19.** Extension rebuilt; all §2.8 validation passed:
> `test_forces.py` 50 passed · `fish_geometry_oracle.py` 0.955/0.955 ·
> `oracle_native_three_way.py` runs (ndelta≈sm2 single sphere, lagr exact) ·
> `band_treatment_check.py::kernel_ndelta_gauge` PASSED (const-p net≈1e-16 rel,
> hydrostatic→vertical-only; single/dumbbell/3-link) · live amphibious_pool
> two-phase headless smoke (80 steps, union default) clean, no NaN/over-thrust.
> Test edits (§Remaining) done; oracle docstring fixed; docs (README,
> HANDOFF_NEXT_AGENT) note deltaH retired; memory updated. Two extra fixes to
> `band_treatment_check.py` were needed (never-run harness had drifted): mock
> `fluid_solver` needed device/dtype/blend attrs, and the const-p gauge is now
> normalised by buoyancy (a single closed body's whole force is ~0, so it can't
> be the scale). Committed as `fbba817` on `cuda_native_port`.

---

# Status: removing the python force path + deltaH  (mid-task handoff)

**Branch:** `cuda_native_port` · nothing committed. Read this alongside the
original spec `HANDOFF_REMOVE_PYTHON_AND_DELTAH.md`. This file records exactly
what is done, the non-obvious decisions, and what remains.

## ⚠️ FIRST THING THE NEXT AGENT MUST DO

**The extension has NOT been rebuilt.** All the C++/CUDA edits below are
UNCOMPILED — the current `_C.so` is stale and still contains deltaH. Before any
validation:

```
cd /data/andreaferrario/lilytorch && python setup.py build_ext --inplace
```

Then finish the two small test edits in §"Remaining" and run Part 2.8.

---

## PART 1 — python eulerian force path: DONE and VALIDATED ✅

- `src/forces.py`: deleted module helpers `_forces_shared`,
  `_forces_body_integrate_3d`, `_forces_body_batch`. `forces_method2` /
  `forces_method2_3d` now run the native `_use_kernel_post` branch and `return`;
  the python fallback is replaced by a `RuntimeError` ("eulerian forces require
  the native streaming path…"). Removed the now-unused `import math`? — no, math
  is still used by the lagrangian `_marker_aabb_slab`; removed `operations as ops`
  (unused). `_viscous_stress_tensor` / `_marker_aabb_slab` KEPT (lagrangian uses them).
- `src/solver.py`: dropped the `_forces_shared, _forces_body_batch,
  _forces_body_integrate_3d` import and the four `self._forces_*_compiled`
  assignments.
- `tests/test_forces.py`: deleted `test_python_eulerian_force_path_cpu_regression`,
  `test_python_eulerian_force_path_cpu_eq_gpu`, the `_run_python_eulerian` fixture
  and their comment block. **These were the 2 chronic pre-existing failures**
  (`project_python_eulerian_cpu_gpu_divergence`) — now legitimately retired (path
  is gone). Mark that memory entry resolved when committing.

**Validation done (before the C++ edits, so still on the old `_C.so`):**
- `pytest lilytorch/tests/test_forces.py -q` → **50 passed** (the 2 failures gone).
- `fish_geometry_oracle.py` → union ndelta **0.955 / 0.955** unchanged.
- Live FARMS eulerian smoke (`gen_zfish_readout_arbitration.py`,
  `ZFISH_FORCE_METHOD=eulerian ZFISH_NITER=8`) → **exit 0, no NaN/error**.

---

## PART 2 — wipe deltaH: code edits DONE, NOT YET BUILT/VALIDATED

### 2.1 Numbering decision (done)
Kept ints **0 = union ndelta, 2 = sm2 per-body normal**; gap at 1. Added a new
config knob **`solver.force_link_normal = "union" | "body"`** (default
`"union"`→0, `"body"`→2). The string `"deltaH"` is retired. sm2 is guarded:
`force_link_normal="body"` **raises if `solver.two_phase` is set** (gauge-unsafe).

### 2.2 CUDA `src/csrc/cuda/eulerian_forces.cu` (done, ~997 lines removed)
Deleted: `forces_post_deltaH_pressure_{2,3}d_kernel`, the legacy
`streaming_sdf_forces_post_{2,3}d_kernel` templates, and in both `_cuda`
launchers the `with_pressure` local + the legacy launch + the `force_submethod!=0`
deltaH second pass. Both launchers reduce to the `force_submethod==0||==2` unified
branch (`forces_post_union_blend_{2,3}d_kernel`) with its early `return`.
Fixed 3 stale comments that referenced the deleted kernel.

**NON-OBVIOUS / CONTRADICTS THE SPEC:** the spec said to delete
`heaviside_smooth_dev_2d` and `heaviside_smooth_dev` as "used only by deltaH".
**That is WRONG — they are load-bearing for the KEPT union path** (used by
`heaviside_grad_{2,3}d` → `forces_post_union_blend_*`). I **kept both**, with
corrected comments. (Classic "python duplicate was actually load-bearing" trap.)

Brace balance verified (open==close). `delta_order` is now an unused param in the
two `_cuda` launchers (union kernel doesn't need it) — harmless `-Wunused` only;
signature kept intentionally.

### 2.3 CPU twins `src/csrc/ops_{2d,3d}.cpp` (done)
Each had `if(force_submethod==1){legacy+deltaH} else {unified}`. Removed the
`if` block + `else` wrapper, kept the unified body unconditionally. Pruned the
now-dead locals (`eps_b`, `inv_2eps`, `pi_over_eb`/`pi_eb`, `band_lo`, `band_hi`).
Kept the shared `S` lambda in 3-D (unified uses it). Braces balanced. No `-Werror`
in setup.py, so leftover warnings are non-fatal.

### 2.4 Op docs (done)
`native.py`: both `streaming_sdf_forces_post_{2,3}d` docstrings now describe
0=ndelta / 2=sm2 with the numbering-gap note. `ops.cpp` schema unchanged (still
carries the `int force_submethod` arg — correct, sm2 needs it).

### 2.5 Plumbing (done)
- `solver.py` ~386: replaced the `force_submethod ∈ (ndelta,deltaH)` block with
  the `force_link_normal ∈ (union,body)` knob → sets `self.force_submethod` int
  (0/2) + the two-phase guard. Removed `force_ph_blend_cells`.
- `forces.py` (2-D ~87, 3-D ~235): `_fsm = int(getattr(self,'force_submethod',0))`;
  `_ph_tau` now always the body-velocity-blend width × h (both submethods use it).
  Fixed the `ForcesPostGraph` docstrings/comments that named deltaH.
- `base_sim_config.py`: replaced `force_submethod`/`force_ph_blend_cells` attrs
  (and their config tuples ~895) with `force_link_normal` (default None→solver default).

### 2.6 Python two-phase deltaH (done)
`src/two_phase_solver.py`: deleted the `partial_heaviside_forces` /
`partial_heaviside_blend_cells` config parse, the flags-list entry, the dispatch
in `_two_phase_forces`, and the `_apply_partition_heaviside` + `_heaviside_smooth`
methods. (`_heaviside_t` in the validation harness is a different function.)
File was also auto-reformatted by a linter — parses clean, deletions intact.

### 2.7 Configs / harness / validation (MOSTLY done)
Done: `examples/amphibious_pool/gen_config_amphibious.py` +
`farms_examples/amphibious_pool/gen_config_amphibious.py` +
`examples/_1guillasim/experiments/gen_config_surface_pool.py` — dropped
`self.force_submethod="deltaH"` (now default union). `gen_config_submerged_diag.py`
and `_run_keflow.py` — removed the dead python-path / `partial_heaviside_forces`
opt-ins. `oracle_native_three_way.py` — repointed the middle column from
deltaH(1) to **sm2(2)** (header/keys updated; **but the top-of-file module
docstring lines 4/11/12 still say "deltaH" — FIX THESE**).
`zfish_snapshot_hook.py` — `force_submethod` now int, `ph_blend_cells` reads
`_body_vel_blend_cells`. `shift_sweep_3d.py` — all callers already pass
submethod=0, no change needed.

**`validation/two_phase_3d/band_treatment_check.py`** — the vacuous
`kernel_deltaH_parity` (it compared against the deleted python
`_apply_partition_heaviside`) was **repurposed → `kernel_ndelta_gauge`**: on
single/dumbbell/3-link scenes it asserts the union gauge property with
submethod 0 — **constant p → Σ_b F_p ≈ 0** (SBP), **hydrostatic p → horizontal
net ≈ 0, vertical ≠ 0**. Renamed the `main()` call too. NOTE: the OLD harness's
op call was against a pre-two-offset signature (would have been broken anyway);
the rewrite uses the CURRENT signature
`(…, eps_body, off_pres, off_friction, h3, delta_order, out, submethod, ph_tau)`.
`_heaviside_t` and the `interp_3d` import are now unused there (harmless).

### ❗ REMAINING (test_forces.py) — I was mid-edit here
`tests/test_forces.py` still parametrizes several tests on the now-invalid
**submethod 1**. On the new code, submethod 1 is a no-op on CUDA (falls through
the `==0||==2` guard → zeros) but runs the union path on CPU → they will DIVERGE.
Repoint every `1` to `2` (sm2) and rename:
- `test_forces_2d_deltaH_cpu_eq_gpu` (~131) → `_sm2_`, `_run_2d(..., 1, ...)` → `2`.
- `test_forces_3d_deltaH_cpu_eq_gpu` (~206) → same.
- `@pytest.mark.parametrize("submethod", [0, 1])` at ~289, ~356, ~647, ~672 → `[0, 2]`.
- Comments at lines 10, 55, 261 (say "deltaH"/"n·δ") → update to union/sm2 wording.
Note `ph_tau = 0.5*h if submethod else 0.0` is fine (2 is truthy).

### 2.8 VALIDATION — NONE DONE YET
1. **BUILD** (see top). 2. `pytest lilytorch/tests/test_forces.py -q` all pass.
3. `fish_geometry_oracle.py` still 0.955/0.955; `oracle_native_three_way.py` runs.
4. **Gauge**: `python -m lilytorch.validation.two_phase_3d.band_treatment_check`
   → `kernel_ndelta_gauge` PASSED (const-p net≈0). This replaces the SESSION-8
   gauge script as the frozen proxy.
5. **Live two-phase** (the real acceptance test): run `amphibious_pool` (and
   `surface_pool` if live) now defaulting to `force_link_normal='union'` for
   enough steps — confirm no over-thrust / gauge leak / blow-up.
6. `grep -rn "deltaH\|force_submethod == 1\|partial_heaviside\|_apply_partition_heaviside\|forces_post_deltaH\|streaming_sdf_forces_post_._kernel" lilytorch/`
   → only historical mentions in `*.md` remain. (Currently still ~a few code
   comments to clean: oracle_native_three_way.py header, test_forces.py above.)
7. Docs: note deltaH retired in `force_benchmarks/README.md` and
   `HANDOFF_NEXT_AGENT.md`. Update the memory entry
   `project_force_readout_eulerian_viscous_bias.md` (deltaH gone; union is the
   sole eulerian gauge; sm2=submethod 2 is analysis-only, two-phase-forbidden).

## Gotchas that already bit / would bite
- Rebuild after EVERY C++ edit; a stale `_C.so` hides everything.
- Don't "restore" `heaviside_smooth_dev*` deletion — they are used by the union
  kernel (see 2.2).
- sm2 (force_link_normal="body") is gauge-unsafe; the solver.py guard forbids it
  with two-phase. Don't remove that guard.
- `force_submethod` int has a gap (0, then 2) — intentional, documented.
