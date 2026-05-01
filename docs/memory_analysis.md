# Systematic Memory Cost Analysis

**Branch:** `optimize_speed_memory` (commit `9b58889` and the streaming SDF + fused
force kernel work that follows it).
**Scope:** every persistent (lifetime ≥ one solver step) and the dominant
transient (lifetime within a step) GPU/CPU allocation owned by the
`FluidSolver` / `BDIMhandler` pair, in the recommended `use_kernels=True`
configuration.

The goal is to confirm the headline claim of the recent kernel work — that
*per-body SDFs and body velocities are no longer materialised on the full
fluid grid, only a single composite "union" copy is* — and to flag what
remains as the next biggest memory lever.

Notation: `N = Nx·Ny·Nz` (3-D) or `Nx·Ny` (2-D), `B = number of bodies`,
`s = sizeof(dtype) ∈ {4, 8}` bytes.

---

## 1. Persistent (lifetime = entire run)

| Block                       | Where                                                | Shape                                | Bytes (3-D)              | Notes |
|-----------------------------|------------------------------------------------------|--------------------------------------|--------------------------|-------|
| **Velocity / pressure**     | `solver.py: u0, v0, w0, p0` (and current `u, v, w, p`) | 8 × `(Nx, Ny, Nz)`                | `8·N·s`                  | Two copies per field (predictor + corrector at the start of `solve_*`); current step often reuses `u0` etc. via aliasing. |
| **Composite mu / normals**  | `solver.py: _mu_pack` (lazy)                         | `(21, Nx, Ny, Nz)`                   | `21·N·s`                 | Pre-allocated once when `_mu_normals_union` is on (default with `use_kernels=True`). Holds `mu0_{u,v,w,cc}`, `mu1_{...}`, `n_{x,y,z}_{u,v,w,cc}`, `1−mu0_cc` in a single contiguous tensor. |
| **Variable-density coeffs** | `_ch_persist, _cv_persist, _cw_persist, _ch_cc_persist` | 4 × `(Nx, Ny, Nz)`                | `4·N·s`                  | Allocated only when FSI variable density is active. |
| **Body SDF batch** (3-D)    | BDIM streaming setup: `F_flat`, `bx_off`, `by_off`, `bz_off` | `Σ_b (Mx_b · My_b · Mz_b)`     | `(Σ_b Mx·My·Mz)·s`       | This is the body-local SDF grid — *not* the fluid grid. For a typical swimmer with 9–13 segments at 64³ each: `B·64³·s ≈ 2–3 MB` at fp32. Negligible relative to fluid-grid buffers. |
| **Sparse cc-SDF slabs**     | `sparse_cc_flat` + `cell_offsets`                    | `Σ_b Ai·Aj·Ak` (per-body AABB cells) | `(Σ_b vol_AABB)·s` + `(B+1)·8` | Fully obsoleted when `fused_sdf_forces=True`: the fused C+D kernel inlines the lagged force loop and never writes the cell-centred SDF slabs. **Setting `fused_sdf_forces=True` saves `Σ_b vol_AABB · s` bytes**, which on a small swimmer in a large pool is `~0.05·N·s` and on a tight-fit pool can rise to `0.3·N·s`. |
| **Drag / torque records**   | `*_drag_record`, `*_torque_record`                   | 4 × `(B, 3, nt)`                     | `12·B·nt·s`              | This dominates **long runs**. At `nt = 10⁶`, `B = 13`, fp32 → 624 MB on device. **Move to CPU pinned memory (TODO F2 in `to_do_list.md`).** |
| **Diagnostics**             | `FlowDiagnostics.kinetic_energy, enstrophy, max_divergence, cfl_number` | 4 × `(nt,)`            | `32·nt`                  | Always present, tiny (4 MB at `nt=10⁶`). |
| **Sponge fields**           | `_sponge_sigma_{u,v,w}` (optional)                   | up to 3 × `N`                        | up to `3·N·s`            | Allocated only with `solver.sponge`. |
| **Compiled CUDA-graph buffers** | `torch.compile` of `_bdim_meta`, `_forces_*`, etc. | implementation-defined            | several × `N·s`          | These appear in `nvidia-smi` but are not counted by `torch.cuda.memory_allocated()`; they are the main reason `reduce-overhead` mode shows ~1.5–2× the "logical" peak in practice. |

### Persistent total (3-D, swimmer-in-pool, `fused_sdf_forces=True`, no Carreau, no sponge)

```
  velocity/pressure :   8·N·s
  mu_pack           :  21·N·s
  variable-density  :   4·N·s   (only with FSI variable density)
  body SDF batch    :  ~B · 64³ · s          (independent of N)
  drag records      :  12·B·nt·s             (independent of N)
  ─────────────────────────────────────
  cell-volume term  :  ≈ 33 · N · s
```

So **before any per-step transients, the fluid-grid memory pressure is
≈ 33·N·s bytes** in the recommended configuration.  At `N = 256³, s = 4`
that is `33 · 16.78 M · 4 ≈ 2.2 GB`, which matches what `estimate_mem.py`
reports.

---

## 2. Per-step transient (peak inside one solver step)

These are the big short-lived tensors allocated and freed inside `solve_euler` /
`solve_heun`.  Peak transient memory is usually what determines the maximum
problem size that fits on a given GPU.

| Block                         | Origin / lifetime                                                                                       | Shape                | Bytes        |
|-------------------------------|---------------------------------------------------------------------------------------------------------|----------------------|--------------|
| **Stress tensors**            | `xstress, ystress, zstress` from `forces.compute_stress_tensor` until the force integral is reduced. | 3 × `(Nx, Ny, Nz)`   | `3·N·s`      |
| **Pressure-force fields**     | `pforce_x, pforce_y, pforce_z` (`= ∇p / ρ` on the staggered grid).                                    | 3 × `(Nx, Ny, Nz)`   | `3·N·s`      |
| **Composite SDF / body vel**  | `comp.sdf_val`, `comp.sdf_val_{u,v,w}`, `comp.body_{u,v,w}` (running-min from each body's kernel).    | 7 × `(Nx, Ny, Nz)`   | `7·N·s`      |
| **Adv-diff scratch**          | `AdvDiffSolver.solve` allocates RHS & intermediate fields (RK2 inside `solve_heun` doubles this).      | ~6 × `N·s` (Heun)    | `6·N·s`      |
| **Poisson workspace**         | `PoissonSolverFFT` keeps complex coefficient + frequency tensors; `PoissonSolver` (multigrid) keeps a V-cycle hierarchy. | varies | `~3·N·s` (FFT) / `~2.4·N·s` (multigrid hierarchy ∑ 1/8ⁿ) |
| **Phase D force accumulator** | `_phaseD_out_buf_3d` — `(B, 12) float64`.                                                              | `12·B·8`             | tiny         |

### Transient peak

```
  3 (xstress/ystress/zstress)
+ 3 (pforce_x/y/z)
+ 7 (sdf + sdf_uvw + body_uvw)            ← these are the union, NOT per-body
+ 6 (Heun RK2 buffers)
+ 3 (Poisson FFT)
+ 1 (residual / divergence)
≈ 23·N·s peak transient
```

**Combined peak (persistent + transient) ≈ 33·N·s + 23·N·s ≈ 56·N·s**, again
matching `estimate_mem.py` to within 10–15 %.

### What the streaming-SDF kernel work eliminated

Before the streaming kernel work, every body wrote its own
`sdf_val_b, sdf_val_u_b, sdf_val_v_b, sdf_val_w_b, body_u_b, body_v_b, body_w_b`
to **the full fluid grid**.  For `B = 13` bodies that was an extra
`7·B·N·s ≈ 91·N·s` of per-step transient memory.

The streaming kernel now keeps only the *running-minimum* composite (7·N·s)
plus the *body-local* SDF batch (`Σ_b Mx·My·Mz · s`, which is independent of
`N`).  At `N = 256³` and `B = 13`, that is a **savings of ~91·N·s ≈ 6 GB
fp32** at peak — i.e. the streaming work converts an out-of-memory run into
one that fits.

Crucially: the kernel still needs the *composite* SDF/body fields on the
fluid grid because (a) the BDIM meta-equation, (b) the variable-coefficient
update, and (c) the force integration all read them.  Eliminating those 7
union fields would require fusing all three downstream consumers into a
per-AABB kernel — see *§3. Remaining levers* and the proposal in
`docs/solver_bdim_merge_proposal.md`.

---

## 3. Remaining memory levers (in order of expected ROI)

1. **Drag-record CPU offload (TODO F2).**
   `12·B·nt·s` is independent of `N` and dominates long runs.  Move
   `viscous_drag_record`, `pressure_drag_record` and the torque equivalents
   to pinned host memory and stream the per-step write asynchronously with
   `tensor.pin_memory()` + `non_blocking=True`.  Saves up to several
   hundred MB on multi-million-step runs.  Effort: low; risk: low.

2. **`fused_sdf_forces=True` everywhere.**
   Already the default in `solver.py` (`use_kernels and
   solver.get("fused_sdf_forces", True)`).  Just make sure no
   `farms_examples/*` config sets `fused_sdf_forces=False` on inertia.
   Saves the `sparse_cc_flat` slabs.

3. **Kill the union mu-pack channels that are unused per step.**
   `_mu_pack` carries 21 channels but a single Euler step never reads more
   than ~13 of them (`mu0_{u,v,w,cc}`, `mu1_{u,v,w,cc}`, `m_m0_all`).  The
   normal fields are only used for the diffusive force, which in
   `use_kernels=True` mode is folded into `streaming_sdf_forces_fused_3d`
   inside the AABB.  Reorganising `_mu_pack` to (8, Nx, Ny, Nz) and
   computing CC normals on demand would save **13·N·s per resident
   instance** (~875 MB at 256³ fp32).  Effort: medium; risk: moderate
   (touches `_mu_normals_batched_3d` + every site that aliases through
   `pack[0..20]`).

4. **Half-precision (bf16/fp16) velocity transport.**
   The kernels already dispatch over `AT_DISPATCH_FLOATING_TYPES_AND_HALF`
   for the streaming SDF path.  Running advection–diffusion in bf16 with
   fp32 master copies for divergence + Poisson would halve `N·s` for the
   ~14·N transient cluster (`{u,v,w,p} × 2` + Heun copies + adv-diff
   scratch).  This is genuinely speculative — divergence-free constraint
   accumulates error fast in bf16 — and not part of the HIGH PRIORITY
   list.

5. **Per-AABB BDIM/mu/forces kernel.**
   The fundamentally largest remaining lever.  This is exactly the
   suggestion at the bottom of the **DONE** list in `to_do_list.md`
   ("…bdim update, setting of the variable coefficients, mu/normals could
   in my view also be computed in the same kernel function locally…").
   If executed, the 7 composite union fields plus the 21-channel
   `_mu_pack` plus the variable-coeff `_c?_persist` quartet become
   per-body AABB-local — saving up to **`(7+21+4)·(N − Σ_b vol_AABB)·s`**.
   For a small swimmer in a 256³ pool that approaches `30·N·s ≈ 2 GB` at
   fp32.  This is a several-week refactor and almost certainly belongs
   on its own branch with full validation against the current path.

---

## 4. How to validate these numbers on hardware

There are two validation paths in the repo:

* **Live, single-snapshot:** `estimate_mem.py` allocates a scaled-down
  problem and prints the per-buffer breakdown.  Run with the exact config
  you intend to use; compare with `nvidia-smi` after `solver.step_(0)`.

* **Time-resolved:** `run_memory_profile_free_3d.py` runs a full free-
  swimming step loop with `torch.cuda.memory._record_memory_history`
  enabled and dumps a chrome-trace pickle.  Open it in
  <https://pytorch.org/memory_viz> to see allocation lifetimes.  Use this
  to confirm the **transient peak** rather than just the resident
  footprint — it is the transient that tends to OOM the GPU.

A consistency check across the three is shown in the testing guide
appended to this PR description.
