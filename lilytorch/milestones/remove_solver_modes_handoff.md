# Handoff: remove `solver_method` python/kernel modes → single fused path

Branch **warp_port**. Task from the user: *"can you now safely remove python and
kernel modes from the codebase?"* — answer is yes, the enabler landed in the
previous session (uncommitted, verified). This document is the plan + survey
for the removal itself.

## State you inherit (uncommitted on warp_port, all verified)

The "body-agnostic solver" refactor is DONE and green:

- **Renames**: Kernel A → `body_update_{2,3}d`, Kernel B → `bdim_forcing_{2,3}d`
  (+ `_sigma` variants, `bdim_forcing_*_warp` inner wrappers,
  `_BodyUpdate{2,3}DBridge` facade bridges, `test_kernelA_2d.py` →
  `test_body_update_2d.py`). All call sites updated.
- **Ownership**: `body_update` is launched from
  `BDIMhandler._launch_body_update` (end of `_update_streaming_multi`),
  including graph-mode persistent buffers (`_bu_bufs`), lazy σ-shift compute,
  and blend. solver.py has NO Kernel-A knowledge left.
- **Contract** (data bus = composite body; any `composite_body.update()` impl):
  after update, comp exposes `sdf_val`, `sdf_val_u/v[/w]`, `body_u/v[/w]`;
  optional `bdim_dirty` (dict `i0,j0[,k0],Ai,Aj[,Ak]`; None/absent → full
  grid), `bdim_keys` (σ body-id keys tuple), `bdim_fields_scratch` (True →
  the fused step sets the staggered fields to None before the projection —
  preserves the old memory profile).
- **solver.py**: `_fluid_step_fused_{2,3}d` (formerly `_fluid_step_kernel_*`)
  consume ONLY the contract fields + call `bdim_forcing_*`. The legacy python
  BDIM branch in `fluid_step` still exists (that is what this task removes).
- **Verified** (`milestones/verify_body_update_refactor.py`, run from repo root):
  - Test A: python path vs fused-step-fed-by-python-provider,
    5 steps, 96² cylinder → max|du|=1.6e-9 (f32 machine precision).
    **This is the jellyfish enabler: deforming bodies that publish the contract
    fields run the fused path with full-grid dirty rect.**
  - Test B: FARMS-free fake-handler drives streaming update → fused step;
    body velocity imposed exactly, scratch released, dirty rect correct.
  - Suites: 100 passed (kernels + test_two_phase + test_air_transparent_body);
    integration pose_source/strong_coupling/checkpoint 12 passed.
    (164 skips = pre-existing native-`_C.so` parity skips, not regressions.)

## The removal task

Collapse `solver_method` ("python" | "kernel") into the single fused path.
`_use_kernels` disappears as a mode; some capabilities must be RELOCATED, not
deleted.

### Survey (complete)

`_use_kernels` / `solver_method` consumers:
- `lilytorch/src/solver.py` — 16 refs: __init__ parse (~line 463-482),
  `PoissonSolver(use_kernels=...)`, `body_from_yaml(use_kernels=...)`,
  eager coeff-persist init, `fluid_step` dispatch + whole python branch
  (steps 3–7: `_apply_bdim_all_axes`, `_recompute_mu_normals` gating,
  `_compute_bdim_coefficients`, `_FS_FREE_AFTER_BDIM*`), `_recompute_mu_normals`
  union-crop gate, `_compute_bdim_coefficients` narrow-band gate,
  `advance_and_compute_loads` recompute gate.
- `lilytorch/src/two_phase_solver.py` — 12 refs: init guards (σ, mu0_proj),
  adv-solve stash wrapper, `consistent_momentum` python-only check,
  `project` kernel-branch (blend identity + coeff rescale),
  `_apply_partition_heaviside` python-only check, gravity suppress,
  `fluid_step` consistent branch, `advance_and_compute_loads` recompute gate.
- `lilytorch/src/forces.py` — 3 refs (lines ~379, ~608, ~696): gates choosing
  streaming post-step force kernels vs python force integration, and the
  union-AABB crop for the python shared kernel. Already half duck-typed on
  `_kernel_step`/`_kernel_static_*` presence.
- `lilytorch/integration/BDIMhandler.py` — 5 refs: `_init_update`
  python/streaming branch, `_init_interp` gate, `_stream_static_pack`
  `_use_combined_cache`, `_update_streaming_multi` guard, dtype comment.
- `lilytorch/src/body.py` — `use_kernels` kwarg (line ~896 kwargs.pop,
  MultiAnimatBodies `_setup_grids` skip line ~2801, staggered-field alloc skip
  line ~3173).
- `lilytorch/src/poisson_mult.py` — `use_kernels` param (python outer driver vs
  warp fine-smoother path).
- Standalone scripts: `integration/fsi_rigid_body.py` (docstring says configs
  should select python; injects `SingleBodyComposite` that ALIASES
  `sdf_val_u = b.sdf_u` etc. — publishes the contract, should work fused),
  `integration/demo_real_fsi.py` (solver_method="python"),
  `validation/cylinder_drag_2d/run_cylinder_drag.py` (SOLVER_METHOD toggle),
  `validation/cost_analysis` (`resolve_solver_mode`),
  `validation/memory_comparison_3d` (compares the two modes — becomes moot),
  `validation/error_analysis_cylinder_2d/run.py`,
  gazzola/coquerelle single_sphere_drop controllers,
  several `examples/**/gen_config*` set `solver_method` (submerged_diag,
  _run_keflow, study_overlap python variants, _test_spheroid_consistent).

Capability edges found:
1. **two-phase `consistent_momentum`** — ACTIVE users:
   `examples/boat/_verify_run_full.py`, `_verify_run_small.py`,
   `_diag_hullonly_full.py`, `_test_spheroid_consistent.py`. Its
   `TwoPhaseSolver.fluid_step` override reuses the python BDIM helpers
   (`_apply_bdim_all_axes`, `_bdim_apply`, `_bdim_meta`,
   `_compute_bdim_coefficients`, `_FS_FREE_AFTER_BDIM*`) inside a fixed-point
   loop. KEEP the capability.
2. **`contour_mask`** — set True in salamander/pleurodeles/jellyfish-drag/
   submarine gen_configs, but ONLY applied inside `BDIMhandler._update_python`
   (`_apply_contour_mask_2d` → `body.mask`), consumed by the python 2-D
   Lagrangian-ish force loop in forces.py (~line 297 `mask = body.mask`).
   In streaming mode it is ALREADY silently ignored today, so configs setting
   it under kernel mode lose nothing. Decide: port it or (default) drop the
   python-only application and leave `body.mask` at its full-contour default
   (body.py initialises `self.mask = torch.arange(...)`).
3. **python force readout for standalone bodies** — forces.py python branch
   needs `self.mu0_all`/normals (from `_recompute_mu_normals`) + per-body SDF
   stacks (`comp.sdf_vals`/`_sdf_sparse`). Standalone sims (cylinder drag,
   gazzola, fsi_rigid_body, jellyfish) have no `_kernel_static_*`, so they MUST
   keep this readout. KEEP `_recompute_mu_normals` + the python force path.
4. **`partial_heaviside_forces`** (two-phase python) — needs per-body SDFs;
   streaming equivalent is `force_submethod: deltaH`. Re-key its guard on
   metadata presence instead of `_use_kernels`.
5. **jellyfish** — config says `solver_method: "python"` + fft. After removal
   it runs the fused path via the contract (Test A proves parity). Check
   JellyfishBody's update publishes all contract fields (memory says its
   sub-bodies are SimpleNamespace proxies; comp-level fields are what matter).
   NOTE stale memory: an old note says "user prefers jellyfish on python path —
   do not switch"; the user's CURRENT instruction (this task) supersedes it.

### Plan (agreed defaults)

1. **solver.py**
   - Parse `solver_method` but only to emit a DeprecationWarning that it is
     ignored (dozens of configs set it; don't break them). Delete
     `_use_kernels`; `fluid_step` = dispatch to `_fluid_step_fused_{2,3}d`
     only; delete the legacy python branch.
   - Move `_bdim_meta`, `_bdim_apply`, `_apply_bdim_all_axes`,
     `_compute_bdim_coefficients` (python variant), `_compute_sigma_mu_grids`,
     `_build_fs_free_dicts` + `_FS_FREE_AFTER_BDIM*` into
     `two_phase_solver.py` (sole remaining consumer = consistent_momentum).
     This also honours the standing rule "keep two-phase changes out of core".
   - KEEP `_recompute_mu_normals` (+ `_compute_union_aabb`, `_mu_pack` crop) in
     solver.py — needed by the python force readout and the consistent path.
     In `advance_and_compute_loads`, gate the call on
     "no streaming metadata" (e.g. `getattr(comp, '_kernel_static_*', None) is
     None`) AND (compute_forces or two-phase-consistent) — measure: kernel-mode
     FARMS runs must NOT start paying the mu/normals peak
     (see memory `project_3d_memory_benchmark`).
   - `PoissonSolver(use_kernels=True)` always. `body_from_yaml(use_kernels=...)`:
     see body.py below.
   - Keep the eager `_init_bdim_coeff_persist_*` call (now unconditional).
2. **two_phase_solver.py** — collapse `_use_kernels` branches: the
   project-time blend identity + coeff rescale run whenever the coefficients
   came from the fused step (`isinstance(ch, torch.Tensor)` already
   distinguishes); consistent_momentum keeps the relocated helpers and now
   requires a python-style (persistent-field) body update — raise if
   `comp.bdim_fields_scratch`. Re-key `partial_heaviside` guard on per-body
   SDF availability.
3. **BDIMhandler** — `_init_update` always streaming; DELETE `_update_python`
   (~180 lines) + `_apply_contour_mask_2d` + `_blend_den` python bookkeeping +
   `_init_body_neighbors_2d` if now unused. `_init_interp` +
   `_init_static_body_metadata` run unconditionally. `_stream_static_pack`:
   `_use_combined_cache` → always True.
4. **body.py** — drop the `use_kernels` kwarg: `MultiAnimatBodies` always
   skips `_setup_grids` + staggered-field allocation (BDIMhandler streams);
   standalone composite classes keep allocating (they are the python-style
   providers). Search for other `use_kernels` readers first.
5. **forces.py** — replace the three `self._use_kernels` gates with pure
   metadata-presence checks (`_kernel_step`/`_kernel_static_*`/`_sdf_sparse`),
   which they already mostly are.
6. **Scripts** — fsi_rigid_body/demo_real_fsi: remove the python-mode notes
   (they work via the contract); cylinder drag SOLVER_METHOD toggle → delete;
   cost_analysis `resolve_solver_mode` → single mode; memory_comparison_3d:
   leave but note the python column is gone (or park it — ask user if unsure).
7. **Tests/verification** (must all be green before finishing):
   - `python3 milestones/verify_body_update_refactor.py` (repo root; Test A
     will need updating — "python" mode no longer exists, so re-purpose it:
     construct one solver whose comp update is python-style and check the
     fused path still matches a saved reference, or simply keep both runs and
     assert the deprecation warning fires + results identical).
   - `python3 -m pytest lilytorch/src/kernels/ lilytorch/src/test_two_phase.py
     lilytorch/src/test_air_transparent_body.py -q`
   - `cd lilytorch/integration && python3 -m pytest test_pose_source.py
     test_strong_coupling.py test_mujoco_checkpoint.py -q`
   - test_two_phase.py constructs solvers with `solver_method` — they will go
     through the deprecation path; fix fixtures if they assert on mode.
   - A boat consistent-momentum smoke: `examples/boat/_verify_run_small.py`
     (needs FARMS; if too heavy, at least import-check + a few steps).
   - grep clean: `grep -rn "_use_kernels\|_update_python" lilytorch --include=*.py`
     (excluding `/old/`) should return nothing (except the deprecation shim).

### Gotchas
- Machine: RTX 4080 SUPER, warp 1.14, everything runs on this box. Sims write
  to `/data/andreaferrario/ns_data/` (user browses there). No Co-Authored-By
  trailer in commits. Do NOT commit unless the user asks — the previous
  session's work is also still uncommitted; keep it that way unless told.
- `zero_pressure_inside`, diagnostics, two-phase repairs read only the CC
  `comp.sdf_val` (persistent) — safe with scratch release.
- The num/den args of `body_update_*` are dead (Warp class blends internally);
  handler passes size-1 dummies.
- 2-D σ keys are full-grid sized; 3-D σ keys are dirty-local (see
  `_launch_body_update`).
- FFT path: fused step passes `ch_cc=getattr(self,'_ch_cc_persist',None)` →
  None in fused mode → scalar-coefficient FFT projection (correct; FFT is
  constant-coefficient by design).

## Autonomy contract (user-approved)

The user will NOT be watching. Work fully autonomously:
- Make every decision yourself using the defaults in the Plan section; when the
  plan is silent, choose the least-destructive option that keeps a capability
  working and note it in the final report. Do NOT use AskUserQuestion; do not
  pause for confirmation mid-task.
- Front-load anything requiring judgement; batch all code changes; verify with
  the checklist; iterate until green.
- Never `git commit`, never push, never delete files outside the repo, never
  write outside the repo / ns_data / the scratchpad.
- If truly blocked (e.g. a test needs hardware/data that does not exist), park
  that item with a clear note in the final report and continue with the rest.
- End with a single review-ready report: what changed (file list), decisions
  taken, test results (exact pass/fail counts), parked items, and suggested
  manual spot-checks. The user reviews everything at the end via `git diff`.
