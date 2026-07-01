# 3D peak-memory baseline (optimize_speed_memory)

Captured with `bench_memory.py` on the standalone python path
(`flow_past_sphere_3d`, multigrid Poisson, quick convection, float32),
RTX 4080 SUPER, warmup then peak reset. Re-run after each T-stage and update.

| grid (Nx×Ny×Nz) | cells | persistent (GiB) | **peak (GiB)** | leak? |
|-----------------|-------|------------------|----------------|-------|
| 256×128×128     | 4.2M  | 0.187            | **1.097**      | none  |
| 448×224×224     | 22.5M | 0.954            | **5.749**      | none  |

Use the 448³-class grid for measuring T3a/T3b/T2a (savings are ~0.5 GiB each,
visible here; negligible at the small grid). The small grid is the fast
bit-exact regression check.

## Where the peak actually is (resolved with per-substep peak-reset probes)
**IMPORTANT: this benchmark runs the standalone PYTHON path** (`solver_method=python`)
because the kernel path needs the FARMS/BDIMhandler bridge. The python path's memory
profile is NOT the production (kernel) profile — see the big caveat below.

Per-substep isolation (reset `max_memory_allocated` immediately before each phase):

| phase (python path, per step)        | internal peak | note |
|--------------------------------------|---------------|------|
| `composite_body.update` (SDF build)  | ~2.24 GiB     | transient, frees back |
| **`_recompute_mu_normals`**          | **5.749 GiB** | ← the peak; settles to 2.667 resident |
| `adv_diff_solver.solve` (advection)  | ~3.36 GiB     | only ~0.7 GiB above resident |
| multigrid `project()` solve          | ~3.58 GiB     | only ~1.0 GiB above resident |
| `finalize_step`                      | ≤3.58 GiB     | no extra spike |

⇒ **On the python path the peak is `_recompute_mu_normals`** (builds mu0/mu1 + 9 normal
fields via `torch.gradient`), NOT advection and NOT the multigrid solve. (An earlier note
here wrongly attributed it to advection — that was carryover from not resetting the peak
counter per substep. Corrected.)

### ⚠️ Big caveat — python path ≠ production kernel path
The KERNEL path (`solver_method=kernel`, the production default) computes mu0/mu1 and
normals **in CUDA thread registers inside Kernel B during `fluid_step`** — it allocates
**no persistent mu/normal buffers** (see solver.py `advance_and_compute_loads`, the
`if not self._use_kernels: self._recompute_mu_normals()` guard + comment). So the 5.749
GiB `_recompute_mu_normals` peak **does not exist in kernel mode**. The TODO's memory
targets (T2a/T2b/T2c referencing `streaming_sdf.cu`, Kernel A/B) are about the KERNEL
path. **To measure the real production memory we must benchmark the kernel path**, which
requires driving the solver through BDIMhandler (FARMS). This standalone bench is only
valid for python-path work.

## Stage log
- **Baseline (after T1a/T1b/diagnostics):** python-path peak 5.749 GiB @ 448×224×224,
  1.097 GiB @ 256×128×128. No leak. Peak = `_recompute_mu_normals` (python path only).
  - command: `python lilytorch/validation/cost_analysis/bench_memory.py --nx 448 --ny 224 --nz 224 --nt 5 --warmup 3`
  - NOTE: not representative of kernel-mode production; see caveat above.
