# Lilytorch — TODO

Memory vars: `sdf_val_{u,v,w}`, `{u,v,w,p}0`, `n{x,y,z}_{u,v,w}`, `body_{u,v,w}`,
`mu{0,1}_{u,v,w}`, `diff_{u,v,w}`.

---

# HIGH PRIORITY

- **Polish repo & docs** — review/correct outdated documentation, including `docs/`.

- **Wire in `FlowDiagnostics`.** The class (solver.py:44) already computes max-divergence,
  energy, enstrophy, CFL and warns on blow-up — but it is **never instantiated or called**
  anywhere. Wire it into the step loop on the post-projection field (every N steps) so
  Poisson under-convergence is caught before it cascades to NaN. Set a `diagnostics_every`
  default in `base_sim_config.py` (the class default `check_every=1` is too expensive — use
  `save_every` or ≥10). This subsumes the old F4 item.

---

# MEMORY / PERF (optimize_speed_memory branch)

Target: ~8 GiB peak alloc on 3D runs. Do in sequence; remeasure after each stage.

- **T1a** Inline `_tvd_face` into `abdquickest`/`cubista`, chain temps in-place
  (`adv_diff.py`). ~0.5 GiB. Bit-exact testable.
- **T1b** Free `self.div` right after `_poisson_solve` returns (`solver.py:project()`).
  ~0.5 GiB transient.
- **T3a** Eliminate the `div` field — inline `divergence()` into the multigrid RHS.
  ~543 MB persistent + ~0.5 GiB transient.
- **T3b** Preallocate the V-cycle coarse-level pyramid at `__init__` instead of
  `torch.zeros` inside the recursion. ~0.5-1 GiB transient.
- **T2a** Fused CUDA `_flux` kernel (QUICK/ABDQUICKEST stencil in registers).
  ~3 GiB + 5-10× adv-diff speedup. Only pays off after multigrid (T3) is fixed.
- **T2b** Dirty-AABB-sized Kernel-A temps (`sdf_*_tmp`, `b*_tmp`: full-grid → AABB+halo).
  Needs `streaming_sdf.cu` changes; no peak movement until T2a.
- **T2c** Two-pass Kernel B for `primes` elimination (write to AABB scratch, copy back).
  ~1.5 GiB; no peak movement until T2a.
- **T4** (architectural, 1+ wk each, measure first): fp16 SDF+body-vel temps;
  mixed-precision velocity fields (fp16 storage, fp32 compute); `--poisson_compile`
  CUDA-graph capture.

---

# 2D/3D SOLVER UNIFICATION (remaining)

Steps 1-4 + apply_forces merge DONE. Remaining:

- **Step 5 — stacked-tensor storage.** Replace `(u0,v0,w0)`, `(nx,ny,nz)`,
  `(mu0_{u,v,w})` etc. with `(D, *grid)` tensors. Deepest refactor (every callsite,
  FARMS bridge, kernels, plotting, HDF5). Needs explicit user sign-off.
- **Step 6 remainder — merge BDIMhandler `_update_2d/_3d` +
  `_update_*_streaming_multi`** (~1000 lines). Replace per-plane branches with a
  `self._sim_axes` index array; needs full FSI regression coverage.

Per-step rules: branch from `optimize_speed_memory`, one PR per step, validate 2D
(`_1guillasim` pinned) + 3D (jellyfish) + cost_analysis (<5% wall-clock regression),
rel-err <1e-6 on integrated quantities. No semantics changes.

### Kernel parity (remaining minor)
- **K9** 2D CUDA `apply_bcs_2d` lacks `is_cuda` TORCH_CHECK (3D has it).
- **K10** Dirichlet branch in `apply_bcs_*_kernel` writes dead `src_lin = 0` — drop it.

---

# LOW PRIORITY

- Analytical 2D salamander swimmer sim (via `control.py` + `gamepad.py`).
- Crank-Nicolson diffusion — current explicit limit `dt < h²/(2ν·ndim)` is not a
  bottleneck now, relevant only if dt is pushed aggressively.
- **eps configurable** — BDIM transition thickness is hardcoded `2h`; add `eps_cells`
  config key (3h-4h smoother on coarse grids).
- **F1 AABB cull force integration** — δ(sdf−ε) is evaluated over the whole domain per
  body but is nonzero only within ε. Slice to each body's AABB+ε. 10-100× for small
  swimmers in big pools.
- **F3 cache CC normals** — recomputed via `torch.gradient` every force call; cache
  alongside staggered normals at body update.
- **F2** drag records: CPU pinned memory + async copy instead of GPU `nt` pre-alloc.

---

# LONG TERM

- **LES for high-Reynolds** — extend the existing Smagorinsky SGS model into a full LES
  workflow (WALE/dynamic-Smagorinsky options, wall treatment) for turbulent high-Re
  regimes where DNS is intractable.
- **AMR (Adaptive Mesh Refinement)** — refine the grid only near bodies and in the wake;
  the enabler for high-Re cases at tractable cost (pairs with LES).
- **Near-boundary stress stencils** — velocity gradients use central differences,
  degrading to 1st-order near immersed bodies. One-sided / ghost-cell stencils would
  improve force accuracy and reduce oscillations.
- **5th-order Hermite smoothstep** for the BDIM delta — `0.5*(1+d/ε+sin(πd/ε)/π)` has
  cancellation at `d≈±ε`; Hermite is more robust and drops sin/cos.
- **2nd-order body coupling** — body SDF/velocity are updated once per step, so Heun's
  corrector uses body state at *t* not *t+dt/2* → coupling is effectively 1st-order.
  Update body to *t+dt* after the predictor and feed the corrector.
- **Checkpoint/restart** — periodic full-state save (iteration, drag records,
  Adams-Bashforth flux, body poses) so a crash at iter 999k of a 1M run isn't fatal.
  Current `_load_initial_conditions` restores only `u,v,[w],p` → warm restart diverges
  from a continuous run.
- SPH simulation support (?).
- Monolithic strongly-coupled fluid + multi-rigid-body solver (?) — hard, would require dropping MuJoCo.
- Refactor: extract `FluidSolver.__init__` (~500 lines) into `_setup_grid/_models/
  _poisson/_output`; add `BaseSimConfig.generate_config()` (dry-run YAML without launch);
  add type hints.
