# Task brief — END-TO-END Warp swap-in (`use_warp_kernels`), the FINAL port item

**Owner:** (next agent)  ·  **This is §F / P5 — the capstone of the Warp port** (the payoff that
retires the `.cpp` twins AND the python path with one `@wp.kernel` source running on CPU+GPU).
All per-kernel ports are DONE and parity-clean (147/147 tests).  This task WIRES them in.

## ⛔ AUTHORIZATION GATE — read before writing any code
This is the **one task that requires editing CORE SOURCE** (`solver.py` / `BDIMhandler.py`,
possibly `two_phase.py`).  The standing constraint everywhere else is **warp_poc-only**, and the
user has previously DEFERRED this item.  **Do NOT edit core source until the user explicitly
authorizes the scope.**  Open with a short plan + the exact files/functions you'll touch and the
toggle name, get a yes, THEN proceed.  Keep the edit surgical (a guarded dispatch behind a flag,
no behavior change when the flag is off).  After: `git diff` on core must be ONLY your toggle +
the pre-existing diagnostics edit (diagnostics.py ~36 lines + solver.py 3 lines — leave that).

## 1. Read first
- `lilytorch/warp_poc/VALIDATION_STATUS.md` — the per-step kernel-mode PIPELINE table (stages
  1–6 + two-phase/interp) and the coverage audit.  This is the map of what to route.
- `lilytorch/warp_poc/HANDOFF.md` — lessons, esp. 3 (CUDA-graph capture is essential), 5
  (per-device RNG), 12 (measuring Warp memory), 14 (zero-copy strided views), 15 (no torch ops
  inside `wp.ScopedCapture`), 25 (tile API).
- The Warp wrappers you'll call (all in `warp_poc/`): `warp_kernels.py`/`warp_kernels_2d.py`
  (Kernel A), `warp_bdim.py`/`warp_bdim_2d.py` (Kernel B), `warp_advection.py` (advect_flux_add),
  `warp_poisson*.py` + `warp_multigrid*.py` + `warp_poisson_tiled_*` (smoother/transfer/V-cycle),
  `warp_lagrangian.py` (forces), `warp_cvof.py` (two-phase), `warp_misc_2d.py`/`warp_misc_3d.py`
  (interp + apply_bcs).
- Core dispatch sites (where native ops are called today):
  - `solver.py` imports at top (`from lilytorch.src.kernels import …`, ~line 12).
  - 2-D step: `solver.py:_fluid_step_kernel_2d` (~`streaming_sdf_stag_2d_multi` @1939,
    `bdim_coeff_2d`/`_sigma_2d` @1958/1973).
  - 3-D step: `solver.py:_fluid_step_kernel_3d` (~`streaming_sdf_stag_3d_multi` @2125,
    `bdim_coeff_3d`/`_sigma_3d` @2148/2163).
  - Poisson: the `poisson_solve_{mgcg,multigrid,rmgcg}_{2,3}d` driver calls (grep `poisson_solve_`
    in solver.py / poisson_mult.py).
  - Forces: `streaming_sdf_forces_post_*` (Eulerian) or `lagrangian_forces_*` call sites.
  - BCs: `apply_bcs_*` call sites.
  - Two-phase: `two_phase.py:_cvof_sweep` (@242 dispatches `torch.ops.…cvof_sweep`).

## 2. Prerequisite decision — the one UNPORTED kernel
`streaming_sdf_forces_post_2d/3d` (the **Eulerian** n·δ viscous+pressure force readout + deltaH ∂H
pass) is the ONLY native op without a Warp equivalent (the *lagrangian* force readout IS ported in
`warp_lagrangian.py`).  Two options — pick per the validation case:
  - **Port it first** (recommended for full coverage): it's a Warp-class scatter kernel (block
    reduction + `wp.atomic_add` into the per-body force row), same pattern as `warp_lagrangian.py`.
    Read `streaming_sdf_forces_post_{2,3}d_kernel` in `streaming_sdf*.cu` (+ the `forces_post_deltaH_
    pressure_*` second pass).  New `warp_forces.py` + `test_forces.py`, parity vs native.
  - **Route around it**: the SU1 validation cases — 2-D `_1guillasim` pinned + 3-D jellyfish — can
    use `force_method=lagrangian` (already ported) so the toggle needn't touch Eulerian forces for
    the trajectory match.  Then port Eulerian forces as a follow-up for full kernel-mode coverage.

## 3. Design the toggle
- Config flag `use_warp_kernels: bool` on `BaseSimConfig` / solver config (default **False** →
  zero behaviour change, all existing runs unaffected).
- A thin dispatch layer in solver.py: where it calls a native op, branch on the flag to call the
  Warp wrapper instead.  Prefer a small adapter (e.g. a `_kdispatch` object holding either the
  native callables or the Warp ones, selected once at init) over scattered `if` branches.
- **CUDA-graph capture is essential** (lesson 3): the Warp per-step kernel sequence must be captured
  once and replayed, with persistent arrays updated in place (the wrappers already support this —
  `update_kinematics`, `capture_*`, `run_graph*`).  Precompute any torch-side slice views BEFORE
  the capture (lesson 15).
- Keep the Poisson OUTER driver (mgcg/rmgcg control flow) in Python/native; route only the
  kernel-level smoother/residual/transfer ops to Warp (or use `WarpVCycle*` where it composes).

## 4. Validate (the actual deliverable)
1. **SU1 trajectory match** vs native on:
   - 2-D `_1guillasim` pinned (find under `farms_examples/_1guillasim/`),
   - 3-D jellyfish (python+fft path — see memory `project_jellyfish_kernel_mode`).
   Run N steps native vs `use_warp_kernels=True`; assert body trajectory / key fields match within
   tolerance (bit-where-deterministic; document any float32/FMA ULP drift — cf. the kernel-level
   parity caveats already recorded).  Build the problem once and reuse (lesson 5).
2. **Perf**: <5% wall-clock regression vs native per step (CUDA-graph, with-reset timing).
3. **CPU end-to-end** (THE payoff): run a SMALL case fully on CPU Warp kernels — proves one kernel
   source serves the CPU fallback AND GPU, retiring the `.cpp` twins + the python-vs-kernel split.
   (Kernel A/B/smoother/cvof/interp/apply_bcs are all single-source CPU==GPU verified; the tiled
   smoothers are GPU-only — use the thread-per-cell `warp_poisson*` smoother on the CPU path.)

## 5. Guardrails
- Core diff stays minimal + behind the flag (off = identical to today).  Re-run the FULL existing
  suite (native paths) to prove no regression, plus `python -m pytest lilytorch/warp_poc/ -q` (147+).
- Don't break the pre-existing diagnostics edit in solver.py/diagnostics.py.
- Report parity (trajectory) + perf (≤5% regression) + the CPU-end-to-end result explicitly.
- Update VALIDATION_STATUS §F + HANDOFF when done; this is the line that lets the port be declared
  COMPLETE.

## 6. Exit criteria
- `use_warp_kernels=True` reproduces the SU1 2-D + 3-D trajectories within tolerance at <5%
  wall-clock regression, AND a small case runs fully on CPU Warp kernels → **port complete**; OR
- a documented blocker (e.g. a kernel that can't be graph-captured in the step loop, or a tolerance
  gap traced to a specific op) with the analysis + the narrowed remaining work.

Ship the flag OFF by default; this is an opt-in capability, not a default-path change.
