# Proposal: Merging `solver.py` and `BDIMhandler.py`

**TODO reference (HIGH PRIORITY #3):**
> Combine `solver.py` and `BDIMhandler` in a single simulation file (just
> `solver.py`). `BDIMhandler` should only keep whatever is necessary for
> handling the coupling with FARMS, if possible. **Review options and
> propose what to do.**

This document is the requested *review and proposal*, not a code change:
the merge itself is a multi-day refactor that touches every example in
`farms_examples/` and every script in `validation/`, and it should be
landed on its own branch with full integration testing rather than
piggy-backed onto the present session.

---

## 1. What each file actually does today

### `lilytorch/src/solver.py` (2 032 lines)

Three logical layers, currently bolted into one class:

1. **Pure-fluid Navier–Stokes core (≈ 1 100 lines).**
   Constructor (grid, BCs, dt/dx/eps), `solve_euler`, `solve_heun`,
   `fluid_step`, `set_BCs`, `divergence`, the four `_compile_*`
   torch-compile branches, FFT/multigrid Poisson dispatch, sponge,
   yield damping, Carreau / Smagorinsky dispatchers (already extracted
   to `extras.py`), and the diagnostics class.
   *Has zero dependency on FARMS.*

2. **Body / immersed-boundary plumbing (≈ 600 lines).**
   `_recompute_mu_normals_2d/3d`, `_mu_pack`, the union-AABB BDIM meta-
   equation, the `streaming_*` kernel dispatch, force computation
   (delegated to `forces.py`).  *Does not call FARMS, but does assume
   `composite_body.update(t, iteration, dt)` is provided by someone.*

3. **Standalone driver (≈ 300 lines).**
   `step_`, file IO (`save_drags_h5`, frame dumping), diagnostics
   recording.  Used by `run_*` validation scripts and by the
   non-FARMS `BodyFish` analytical-kinematics path.

### `lilytorch/integration/BDIMhandler.py` (2 270 lines)

Five logical pieces:

1. **FARMS sensor → torch translation (≈ 250 lines).**
   `cython2numpy`, `_compose_body_frame_3d`, `_init_buoyancy_params`,
   `_init_body_neighbors_2d`, `apply_forces`/`_apply_forces_2d`/
   `_apply_forces_3d` (writes `xfrc_applied` back into MuJoCo).
   *Genuinely FARMS-specific.*

2. **`update()` driver (≈ 350 lines).**
   `_update_2d` and `_update_3d` — read FARMS sensors, build per-body
   rotation matrices, push them into the SDF + body-velocity grids.
   *Uses FARMS at the top, but the bottom two-thirds is just a
   per-body grid-sample / rotation pipeline that reads `comp.bodies`
   and writes the union SDF/body-velocity fields.*

3. **Streaming-kernel staging (≈ 700 lines).**
   `_update_3d_streaming`, `_update_3d_streaming_multi`,
   `_update_2d_streaming_multi`, `_init_custom_trilinear_2d/3d`,
   `_body_aabb_indices`, the per-body kinematics packing
   (`F_flat`, `F_offsets`, `body_meta`), the fused C+D streaming
   call, and the post-call composite-min handling.
   ***Has nothing to do with FARMS** — it is the C++/CUDA-side
   scaffolding for the kernels in `lilytorch/src/kernels/`.*

4. **`step()` orchestrator (≈ 50 lines).**
   The public per-step entry point: `update → mu_normals → fluid_step
   → forces → apply_forces`.  Mirrors `FluidSolver.step_` exactly,
   except for replacing the analytical body update with FARMS sensors
   and adding `apply_forces`.

5. **Initialisation (≈ 120 lines).**
   `__init__`, `_init_batched_sdf_2d`, the dtype/device boilerplate.
   *Trivial wrapper around `FluidSolver(...)`.*

---

## 2. Why the current split is awkward

* **No clean interface**: `BDIMhandler` reaches into ~80 attributes of
  `FluidSolver` (grep `self.fluid_solver.<name>` — there are far more
  read/write touch-points than any reasonable API).  Symmetrically,
  `FluidSolver._recompute_mu_normals_*` reads `composite_body.sdf_val_*`
  buffers that are *only* populated correctly by `BDIMhandler.update()`
  in coupled mode.

* **Streaming-kernel staging lives in the wrong module.** Sections 3
  of `BDIMhandler.py` (~700 lines) are pure C++/CUDA staging.  They
  exist there only because *someone* needed an object that knows the
  body list, the device, and the dtype.  But that "someone" is the
  fluid solver, not the FARMS bridge.

* **Reading flow is hard.** `step()` jumps `BDIMhandler →
  FluidSolver._recompute_mu_normals_3d → composite_body.update (which
  is monkey-patched onto `BDIMhandler.update`!) → _update_3d_streaming
  → fluid_step → forces_method2_3d → _apply_forces_3d`.  The control
  flow zig-zags across two files four times per step.

* **The `composite_body.update = self.update` monkey-patch in
  `BDIMhandler.__init__` is genuinely fragile.** Any subclass that
  overrides `update()` only fixes the `BDIMhandler` view — the
  `composite_body.update` callable still points to the parent.

---

## 3. Three options

### Option A — full merge into `solver.py` (literal reading of TODO #3)

Move every line of `BDIMhandler.py` into `solver.py`.  Deprecate
`integration/BDIMhandler.py` to a thin re-export shim.

* **Pros:** maximally simple from a top-down reading; one place to
  look for everything.
* **Cons:**
  - `solver.py` grows from 2 032 to ~4 100 lines; that is *worse* for
    navigation, not better, and it adds a hard dependency on FARMS
    sensor types into a module that today imports zero FARMS symbols.
  - Validation scripts that import `FluidSolver` (e.g.
    `run_cylinder_drag.py`, `run_error_analysis_*`) suddenly transitively
    import `farms_amphibious`, breaking pure-PyTorch installs.
  - No improvement to the awkward 80-attribute interface — it's just
    folded inside one class.
  - Conflicts directly with the README's two-mode architecture
    diagram, which positions standalone (no FARMS) and coupled (with
    FARMS) as peer modes.
* **Verdict:** ❌ Reject.  Misreads the spirit of TODO #3.

### Option B — three-file split (recommended)

Read TODO #3 as: *"the only thing that should be in `BDIMhandler.py`
is FARMS-coupling logic; everything else belongs with the solver."*
Concretely:

```
lilytorch/src/solver.py              ← unchanged scope: pure NS + IBM
lilytorch/src/streaming_setup.py     ← NEW: kernel staging from BDIMhandler §3
lilytorch/src/composite_update.py    ← NEW: the per-body grid_sample /
                                       rotation pipeline from BDIMhandler §2
                                       (reads comp.bodies, writes union SDF)
lilytorch/integration/BDIMhandler.py ← shrunk to ~350 lines:
                                       FARMS sensor reads, apply_forces,
                                       buoyancy, step orchestrator, dtype
                                       boilerplate.  Imports the new modules.
```

Migration steps (each step is a self-contained PR):

1. Lift `_init_custom_trilinear_2d`, `_init_custom_trilinear_3d`,
   `_init_batched_sdf_2d`, `_body_aabb_indices`,
   `_update_3d_streaming_multi`'s kinematics-staging helper, and the
   `F_flat / F_offsets / body_meta` builders into
   `lilytorch/src/streaming_setup.py`.  Wire them as
   `FluidSolver._init_streaming(...)` (run once after
   `composite_body` is built).  No behaviour change — just code
   relocation; existing tests in `lilytorch/src/kernels/test_*.py`
   already cover the kernels themselves.
2. Lift the FARMS-free portion of `_update_2d` /
   `_update_3d` (the rotation + grid-sample + union-min loop) into
   `lilytorch/src/composite_update.py` as
   `update_composite_2d(comp, kin)` / `update_composite_3d(comp, kin)`.
   `BDIMhandler._update_*d` becomes just the FARMS sensor unpacking
   followed by the new shared call.  The analytical
   `BodyFish`-driven update path can also be migrated to call the same
   shared function (kinematics built from `body.update_kinematics`
   instead of FARMS sensors), so it is exercised even in the standalone
   tests.
3. Promote `step()` to `FluidSolver.step_with_external_update(kin)`.
   `BDIMhandler.step` becomes a 15-line wrapper that builds `kin` from
   FARMS sensors and calls the solver.  Apply forces back to MuJoCo
   stays in `BDIMhandler`.
4. Remove the `composite_body.update = self.update` monkey-patch.
   Replace with explicit dependency injection: `FluidSolver` holds an
   `update_callback` slot that defaults to `composite_body.update` and
   that `BDIMhandler` overrides through a setter.
5. Delete now-dead `BDIMhandler` private helpers; final size ~350
   lines.

* **Pros:** matches the *spirit* of TODO #3 (BDIMhandler keeps only the
  FARMS coupling); separates concerns; preserves the standalone /
  coupled split in the README; each step is independently testable.
* **Cons:** still a non-trivial refactor; touches every example in
  `farms_examples/` (mostly `from lilytorch.integration.BDIMhandler
  import BDIMhandler` import paths, which can stay stable).
* **Test impact:** existing kernel tests still cover step 1 verbatim;
  steps 2–4 need a new analytical-kinematics integration test
  (`test_bdim_step_against_FARMS_recording.py`) that records one FARMS
  step and replays it via the new `update_composite_3d` to confirm
  bit-equality.  Feasible.

### Option C — leave the structure, fix only the awkwardness

Cheapest possible delivery:

1. Replace the `composite_body.update = self.update` monkey-patch with
   an explicit `FluidSolver.set_update_callback(fn)` (1-day fix).
2. Add a single docstring section at the top of `solver.py` listing
   the ~80 attributes that `BDIMhandler` reads, marking each as
   `# public` or `# stable`.
3. Move the streaming-kernel staging into a private mixin class in
   `lilytorch/src/streaming_setup.py` to relieve `BDIMhandler.py`
   without changing the user-facing API.

* **Pros:** very low risk; can be done in one PR.
* **Cons:** does not literally satisfy TODO #3 — `BDIMhandler.py` still
  contains non-FARMS code (the streaming staging).  But it removes the
  worst foot-guns.

---

## 4. Recommendation

**Adopt Option B, executed as five small PRs in the order listed.**

It is the only option that *honestly* satisfies TODO #3 ("BDIMhandler
should only keep whatever is necessary for handling the coupling with
FARMS") without growing `solver.py` past readability or imposing FARMS
as a hard dependency on the standalone solver.

**Suggested timeline:** one PR per step, each independently mergeable
and revertible.  Estimate 2–4 days of focused work per step including
review + integration testing.  Steps 1 and 5 are pure code motion (no
runtime behaviour change) and can be reviewed by diff alone.  Steps
2–4 each need one regression run on a 3-D salamander config to
confirm bit-equality of forces/torques against the current branch.

**Out of scope for the present session.** This proposal is the
deliverable for TODO #3 in this PR; the actual refactor should be
opened as `copilot/solver-bdim-merge` against the same
`optimize_speed_memory` base.

---

## 5. What changes for users in `farms_examples/`

In Option B, **the public API surface does not change**:

* `from lilytorch.integration.BDIMhandler import BDIMhandler` keeps
  working.
* `BDIMhandler(yaml_file, data, physics, dtype=None)` keeps the same
  constructor signature.
* `handler.step(task, physics)` keeps the same semantics.
* The yaml schema is unchanged.

So *no* example in `farms_examples/` needs editing.  The validation
scripts and the analytical `BodyFish` driver only benefit (step 2 means
they share the same composite-update code path as the FARMS coupling,
making bugs in either show up in both test suites).
