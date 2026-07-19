# Handoff: remove the legacy python force path, then wipe out deltaH

**Branch:** `cuda_native_port` · **Prereq reading:** the SESSION 6–8 blocks in the memory file
`project_force_readout_eulerian_viscous_bias.md`, and `HANDOFF_NEXT_AGENT.md` §B1c.

## 0. Why this is safe to do now (already established — do not re-litigate)

The **unified `ndelta`** readout (union smoothed-Heaviside gradient `∂_iH(φ_union)` for BOTH the
pressure and viscous channels, split to links by the streaming body-velocity blend partition) is
**already implemented and validated** on the native path:

- **CUDA 3-D + 2-D + CPU 3-D + 2-D** all ported. CPU==GPU parity to ~1e-16; `test_forces.py`
  16 `*_cpu_eq_gpu` pass.
- **Oracle** (`fish_geometry_oracle.py`): union `ndelta` = **0.955 / 0.955** (pressure/viscous),
  `delta_order`-independent, per-link signs clean.
- **GATE PASSED — the two-phase gauge fix (the reason deltaH exists) is preserved by union-∇H:**
  on the fish snapshot, constant `p` → net force **1.3e-10 N ≈ 0**; hydrostatic `p=−ρgz` →
  horizontal **8e-14 ≈ 0**, vertical = **0.955× buoyancy** (exact band factor). So `ndelta` (union)
  **subsumes** deltaH everywhere deltaH was used.

The readout is selected by the op's `force_submethod` int argument:
- `0` = **ndelta** = union-∇H (the default, and the only real eulerian readout going forward),
- `1` = **deltaH** ← THIS IS WHAT YOU ARE DELETING,
- `2` = **sm2** = union coarea magnitude × per-body analytic normal (per-link-accuracy variant;
  **gauge-UNSAFE — 32× spurious force under constant p — analysis-only, must never reach two-phase**).

Production (zebrafish/salamander/amphibious/1guilla, all FARMS-coupled) uses the **native kernel
path** (`BDIMhandler` sets `comp._kernel_step` / `_kernel_static_{2,3}d`). The **python force branch
is legacy and bypassed in production** — it is reached only by (a) the 2 `test_python_eulerian_*`
tests, which build a deliberately non-streaming solver, and (b) at most one experimental config,
`_1guillasim/experiments/gen_config_submerged_diag.py`.

**Do BOTH removals in the order below** (python path first — deleting it shrinks what deltaH removal
must touch). Rebuild (`python setup.py build_ext --inplace` from repo root) after each C++ change and
re-run `pytest lilytorch/tests/test_forces.py -q` frequently. Nothing is committed; work on
`cuda_native_port`.

---

## PART 1 — Remove the legacy python (non-kernel) eulerian force path

**Goal:** `forces_method2` / `forces_method2_3d` become native-only (the `_use_kernel_post` branch),
and everything reachable only from the python fallback is deleted.

### 1.1 Confirm expendability first (BLOCKER)
- Open `examples/_1guillasim/experiments/gen_config_submerged_diag.py`. If it (or anything you
  care about) relies on a **standalone (non-FARMS) python-body eulerian run** or on
  `partial_heaviside_forces`, decide whether to **migrate it to the native streaming path** (which
  now carries the union gauge fix) or drop it. If it is expendable, note that and proceed. **Do not
  delete the python path while a live example still needs it.**
- Grep for any other standalone (non-BDIMhandler) `FluidSolver` + eulerian usage:
  `grep -rn "force_method.*eulerian" examples farms_examples validation` and check each hit is
  FARMS-coupled (goes through `BDIMhandler`). Coupled → safe. Standalone → must migrate or drop.

### 1.2 Delete the python branch of the eulerian force functions (`src/forces.py`)
- In `forces_method2` (2-D, ~line 274) and `forces_method2_3d` (3-D, ~line 540): the top of each is
  the `_use_kernel_post` branch that runs the native op and `return`s. **Keep that.** **Delete
  everything after it** (the CC-normals recompute, the batched/sparse `_forces_shared_*` +
  `_forces_body_integrate_*` python integration, ~lines 404–535 in 2-D and ~650–900 in 3-D). After
  the kernel branch, if `_use_kernel_post` is False, raise a clear `RuntimeError` ("eulerian forces
  require the native streaming path; standalone python bodies are no longer supported — use the
  FARMS/BDIMhandler path").
- Delete the now-unreferenced module-level helpers in `forces.py`:
  `_forces_shared`, `_forces_body_integrate_2d`, `_forces_body_integrate_3d`, `_forces_body_batch`
  (and any `_forces_shared_dyn` / narrow-band python-only helpers). **Verify they are not used by the
  lagrangian path** (`forces_lagrangian_2d/3d`, ~lines 1027/1145) — they are not today, but confirm
  with `grep -n "_forces_shared\|_forces_body" src/forces.py` returning only the eulerian sites.

### 1.3 Remove the compile wiring (`src/solver.py`)
- Line 26 import: drop `_forces_shared, _forces_body_batch`.
- Lines ~517–519: delete `self._forces_shared_compiled`, `self._forces_body_batch_compiled`,
  `self._forces_shared_dyn_compiled` assignments and any `torch.compile` of them.
- Grep `grep -n "_forces_shared\|_forces_body_batch" src/solver.py` → 0 hits when done.

### 1.4 Delete the 2 python-path tests
- In `tests/test_forces.py` remove `test_python_eulerian_force_path_cpu_regression` and
  `test_python_eulerian_force_path_cpu_eq_gpu` (~lines 212–290) and their non-streaming solver
  fixture. **These are the 2 chronic pre-existing failures** (the python CPU/GPU divergence bug,
  `project_python_eulerian_cpu_gpu_divergence`) — deleting the path retires them legitimately. Note
  in the commit that this bug is now moot (the path is gone), and mark the memory entry resolved.

### 1.5 Validate Part 1
- `pytest lilytorch/tests/test_forces.py -q` → **all pass** (the 2 python failures are gone; nothing
  else should regress).
- `python -m lilytorch.examples.force_benchmarks.fish_geometry_oracle` → still 0.955 / 0.955.
- Run one live FARMS eulerian sim (zebrafish arbitration harness, `ZFISH_FORCE_METHOD=eulerian`) for
  a few steps → no crash, forces sane (it never used the python branch, so this is a smoke test).

---

## PART 2 — Wipe out deltaH (`force_submethod == 1`)

After Part 1, deltaH lives only on the native path + configs + one validation harness. Removing it
also makes the **old per-body CUDA kernels and the wrapped CPU `submethod==1` blocks DEAD** — delete
those too (a large, satisfying simplification: sm0/sm2 use the *new* `forces_post_union_blend_*`
kernels exclusively).

### 2.1 Decide sm2's numbering FIRST (design decision)
Valid `force_submethod` values become `{0 = ndelta-union, 2 = sm2-per-body-normal}`. Either:
- **(recommended)** keep the ints as-is (0 and 2, gap where deltaH was) and add a clear config knob
  that maps to 2, e.g. `solver.force_link_normal = "union" | "body"` (default "union"→0). Do NOT
  reuse the string `"deltaH"` for anything.
- or renumber sm2 → 1. Riskier (a stale config passing `1` would now silently get sm2, which is
  **gauge-unsafe**); if you do this, add a hard guard (see 2.5).
Whichever you pick, **sm2 must raise if used with the two-phase solver** (gauge-unsafe).

### 2.2 CUDA (`src/csrc/cuda/eulerian_forces.cu`)
- Delete the deltaH kernels: `forces_post_deltaH_pressure_2d_kernel` (~579) and
  `forces_post_deltaH_pressure_3d_kernel` (~1397), plus the device helpers used only by them:
  `heaviside_smooth_dev_2d` (~571) and `heaviside_smooth_dev` (~1389).
- In `streaming_sdf_forces_post_2d_cuda` and `_3d_cuda`: delete the `if (force_submethod != 0) { …
  deltaH ∂H pass … }` blocks (2-D ~1032, 3-D analogue) **and** the legacy per-body launch that now
  only serves deltaH's viscous channel (the `streaming_sdf_forces_post_{2,3}d_kernel<...>` launch +
  its `with_pressure`/`blockSize`/`nblocks` scaffolding). The launcher body reduces to: the
  `force_submethod == 0 || force_submethod == 2` unified branch (already present, ~946) — make that
  the whole function (drop the `|| == 2` special-casing only if you renumber; otherwise keep it).
- Delete the now-orphaned `streaming_sdf_forces_post_2d_kernel` (~279) and
  `streaming_sdf_forces_post_3d_kernel` (~1086) template kernels themselves. `grep -n
  "streaming_sdf_forces_post_._kernel\|deltaH\|heaviside_smooth_dev" eulerian_forces.cu` → only the
  comment reference at ~1637 should remain (fix the comment).

### 2.3 CPU twins (`src/csrc/ops_2d.cpp`, `src/csrc/ops_3d.cpp`)
- Each has `if (force_submethod == 1) { <old per-body loop + deltaH ∂H pass> } else { <unified> }`.
  **Delete the `if (force_submethod == 1) { … }` block and the `else` wrapper**, keeping the unified
  body unconditionally. Remove the now-unused `band_lo/band_hi`, the deltaH `Hs` lambda, `inv_tau`,
  `ph_tau`-as-tau usage, etc. (compiler `-Wunused` will flag leftovers).

### 2.4 Op schema / signature (`src/csrc/ops.cpp`, `src/native.py`, `src/forces.py`)
- The op keeps a `force_submethod` int (0/2) — do **not** remove the argument (sm2 needs it), but
  update the doc comment to drop deltaH. `ph_tau` stays (it now only carries the blend eps for the
  partition — used by sm0/sm2). If you renumber sm2, update `native.py` / call sites accordingly.

### 2.5 Plumbing (`src/solver.py`, `examples/base_sim_config.py`, `src/forces.py`)
- `solver.py` ~407: `force_submethod` validation currently allows `("ndelta","deltaH")`. Replace with
  the new knob (2.1). Remove `force_ph_blend_cells` **only if** deltaH was its sole consumer — CHECK:
  the unified partition reuses `body_velocity_blend_eps_cells` (see `forces.py` `_ph_tau` for `_fsm==0`),
  so `force_ph_blend_cells` is deltaH-only → remove it. Grep to confirm.
- `forces.py` ~325 and ~592: the `_fsm = 1 if force_submethod=='deltaH' else 0` lines → replace with
  the new 0/2 selector; the `_ph_tau` for the unified path already reads `_body_vel_blend_cells` — keep.
- `base_sim_config.py` ~190–193 and ~895: drop `force_submethod`/deltaH plumbing (or repoint to the
  new knob).

### 2.6 Python two-phase deltaH (`src/two_phase_solver.py`)
- Delete `_apply_partition_heaviside` and the `partial_heaviside_forces` / `partial_heaviside_blend_cells`
  config handling (~382–395, 780, 808–812, 922–927 dispatch). Standalone python two-phase is removed
  with the python path (Part 1); any two-phase run now goes native and inherits the union gauge fix.
  Keep this change **confined to `two_phase*.py`** (repo rule: no two-phase logic in core
  `forces/solver/body.py`).

### 2.7 Configs / examples / validation / harness
- Remove `self.force_submethod = "deltaH"` from: `examples/amphibious_pool/gen_config_amphibious.py`
  (~91) **and** its twin `farms_examples/amphibious_pool/gen_config_amphibious.py`;
  `examples/_1guillasim/experiments/gen_config_surface_pool.py` (~36), `gen_config_submerged_diag.py`,
  `_run_keflow.py`. They then default to `ndelta` (union) — which is the point.
- `validation/two_phase_3d/band_treatment_check.py`: `kernel_deltaH_parity` becomes vacuous — delete
  that function (and its `__main__` call) or repurpose the harness to assert the unified `ndelta`
  gauge property (constant-p → net≈0) instead.
- `tests/test_forces.py`: remove any remaining deltaH-referencing tests/assertions. The
  `force_benchmarks/` helpers that take a `submethod` arg (`shift_sweep_3d.py`,
  `oracle_native_three_way.py`, `zfish_snapshot_hook.py`) — update so `submethod=1` is no longer a
  valid deltaH selection (map to the new scheme or drop the option).
- Docs: `force_benchmarks/README.md`, `HANDOFF_NEXT_AGENT.md` — note deltaH is retired.

### 2.8 Validate Part 2 (do ALL of these)
1. Build clean; `pytest lilytorch/tests/test_forces.py -q` → all pass.
2. `fish_geometry_oracle.py` → union `ndelta` still 0.955 / 0.955; sm2 unchanged.
3. **Gauge re-check** (the load-bearing one): constant-`p` net force ≈ 0 and hydrostatic → exact
   buoyancy, for `ndelta` on the snapshot (reproduce the SESSION-8 gauge script from memory). sm2
   must still show the 32× leak (proving the guard in 2.1/2.5 matters).
4. **Live two-phase re-run** — the original reason deltaH existed: run `amphibious_pool` (and
   `surface_pool` if still live) now defaulting to `ndelta`, for enough steps to see the waterline/
   buoyancy behave. Confirm **no over-thrust / no gauge leak** and no blow-up. This is the real
   acceptance test that union-∇H replaces deltaH in two-phase, live (the frozen gauge test is a proxy).
5. `grep -rn "deltaH\|force_submethod == 1\|partial_heaviside\|_apply_partition_heaviside\|
   forces_post_deltaH\|streaming_sdf_forces_post_._kernel" lilytorch/` → only historical mentions in
   memory/handoff docs remain.

---

## Gotchas (each has cost a session before)

- **Rebuild the extension after every C++ edit** — a stale `_C.so` silently hides changes.
- **Do not write "bit-exact" of any 3-D force-kernel output** — the kernels accumulate per-link with
  atomics; quote a tolerance vs an identical-call repeat instead.
- **sm2 is gauge-unsafe (32× spurious force under constant p).** Never let it reach the two-phase
  solver; guard it (2.1/2.5). It exists only for per-link-accuracy analysis, and even then does not
  beat the lagrangian.
- **Keep two-phase changes inside `two_phase*.py`** (repo rule).
- **The union-∇H net exactness is an SBP property of the discrete Heaviside gradient, not of normal
  accuracy** — do not "improve" it by swapping in per-body normals (that is sm2, and it breaks the
  net + gauge).
- After removal, `force_submethod` has a **gap** (0, then 2) unless you renumber — document it so the
  gap doesn't look like a bug.
