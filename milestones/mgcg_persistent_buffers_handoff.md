# Handoff: persistent scratch buffers for the native mgcg/rmgcg Poisson driver

**Goal (one line):** Eliminate the per-step GPU allocation churn in the native
`mgcg`/`rmgcg` Poisson drivers so they stop forcing a `cudaMalloc` during the
pre-projection CUDA-graph capture — removing the need for the
`expandable_segments` mitigation and shaving per-step allocator overhead.

**Branch:** `cuda_native_port`. **Effort:** ~1 focused day. **Risk:** medium
(touches core Poisson C++; must not regress `multigrid` or CPU/CUDA parity).

---

## 1. Why this exists (background)

`gen_config_surface_pool.py` with `poisson_method: rmgcg` crashed intermittently
(iter 15 one run, clean past 60 another) with:

```
RuntimeError: CUDA error: operation failed due to a previous error during capture
  File ".../solver.py" in _run_preproj -> advection.py _conv_copy[i].copy_(vel[i])
```

Diagnosed 2026-07-15 (see memory `project_mgcg_graph_capture_cudamalloc.md`):

- **Not** a compute bug, **not** an illegal access. The rmgcg solve is
  `compute-sanitizer`-clean, and the `restrict_fw` symmetric-restriction change
  (`project_mgcg_nonsymmetric_preconditioner.md`) adds no allocations.
- The mgcg/rmgcg solve runs **eager**, separate from the captured pre-projection
  region. It allocates ~8 full-grid transient tensors every step. That churn
  occasionally forces PyTorch's caching allocator to grow the pool (`cudaMalloc`)
  **during** the next step's graph capture — illegal → capture invalidated.
  Intermittent because it depends on exactly when the pool needs to grow.
- `multigrid` uses a tight driver (2 top-level tensors) and does not trip it.

**Current mitigation (already shipped):** `base_sim_config.gen_sh_config` exports
`PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"`
in the generated `run.sh`, which makes pool growth capture-safe. That is a
process-wide band-aid. **This task is the root fix:** make the driver allocate
its scratch once and reuse it (pointer-stable), so after warmup a solve performs
**zero allocations** and the pool never grows mid-capture.

---

## 2. Root cause, precisely

The caching allocator serves repeated same-size allocations from its pool without
`cudaMalloc` — but only once the pool is large enough for the working set's PEAK.
mgcg/rmgcg's peak transient footprint is much larger than multigrid's (~6 extra
full-grid f64 tensors ≈ 670 MB at 900×300×52), so the pool must grow over the
first several steps; if a growth lands inside a capture, the capture dies.

Make the driver's scratch **persistent and pointer-stable** → no growth after the
first (eager, pre-capture) solve → no `cudaMalloc` during capture. This also
mirrors the codebase's established pattern (pointer-stable persistent buffers +
in-place ops — see `project_projection_fresh_tensor_graph_leak.md`,
`project_forces_cuda_graph_spike.md`).

---

## 3. What allocates today (the targets)

All in `lilytorch/src/csrc/cuda/poisson_solve.cu`:

**`poisson_solve_mgcg_3d_cuda` (and 2d)** — per call:
`b = f.mul(-h2)`, `f_zero = at::zeros`, `Bx = at::empty`, `z = at::zeros`,
`r_buf = at::empty`, `neg_r = r.neg()`, `d = z.clone()`, `q = at::empty`.
(`r` aliases `b.sub(Bx)`.) Note `f_zero` is a pure waste: it exists only so
`mg_residual_*` can compute `B(p)` with a zero RHS.

**`poisson_solve_rmgcg_3d_cuda` (and 2d)** — same set plus the deflation basis
`U`/`W` (empty when `recycle_k == 0`, which is the surface_pool case).

**`vcycle_2d/3d` (static, shared by ALL drivers incl. multigrid)** — per level of
the recursion: `ch_c, cv_c, cw_c = at::empty`, `r_c = at::empty`,
`p_c = at::zeros`, `r_c2 = at::empty`. multigrid also pays this, so it is
tolerated (same sizes every step, cached) — but converting it too is the complete
fix and removes ALL Poisson allocation churn.

`restrict_fw_*` already takes `rc` as an out-param and only does `rc.zero_()`
(no alloc) — no change needed there.

---

## 4. Recommended design

**A C++-side persistent scratch cache keyed by `(device, dtype, shape)`.** The grid
shape is constant across steps, so the cache is populated on the first (eager)
solve and reused forever after. This is the cleanest fit because the vcycle
recursion allocates a *tree* of level-dependent sizes that is awkward to pass
down from Python.

Sketch (put in an anonymous namespace in `poisson_solve.cu`, or a small
`poisson_scratch.h`):

```cpp
// Returns a pointer-stable scratch tensor of the given shape/opts, allocated
// once and reused. NOT zeroed — caller must .zero_() if it needs zeros.
static at::Tensor& scratch(const std::string& tag, at::IntArrayRef shape,
                           const at::TensorOptions& opts) {
    static std::unordered_map<std::string, at::Tensor> pool;   // process-lifetime
    // key on tag + shape + dtype + device index
    std::string key = tag + "|" + shape_dtype_dev_string(shape, opts);
    auto it = pool.find(key);
    if (it == pool.end() || !it->second.defined()
        || it->second.sizes() != shape) {
        pool[key] = at::empty(shape, opts);
        it = pool.find(key);
    }
    return it->second;
}
```

Then replace each `auto Bx = at::empty(...)` with
`auto& Bx = scratch("mgcg3d_Bx", {Nx,Ny,Nz}, opts)`, and each `at::zeros(...)`
with `scratch(...).zero_()`. Give the vcycle a per-level tag that includes the
level size in the key (the shape already distinguishes levels, so a single tag
`"vcycle_r_c"` etc. suffices — different sizes get different cache slots).

**Why a `tag`:** several buffers at the same level share a shape but must be
distinct live tensors (e.g. `r_c`, `r_c2`, `p_c` interior). The tag prevents them
aliasing to the same slot.

**Alternative (rejected as primary):** pass scratch from Python
(`PoissonSolver` holds buffers, ops take them as args). Matches the existing
`_persist` pattern but explodes the op signatures and cannot express the vcycle
recursion tree cleanly. Consider only for the top-level CG vectors if you want to
avoid C++ global state; not worth it for the recursion.

**Cheap independent win (do this too):** kill `f_zero`. Add a tiny apply-operator
path so `B(p)` is computed without a zero-RHS buffer — either a
`mg_apply_operator_*` kernel, or reuse a single process-wide persistent zero
buffer via the same `scratch()` (zeroed once, never written). Removes one
full-grid alloc + the memset per solve.

---

## 5. Gotchas / constraints

1. **Zeroing.** `at::zeros` sites (`f_zero`, `z`, `p_c`) become
   `scratch(...).zero_()`. Miss one and you get wrong results from stale data —
   the parity tests will catch it, but check each.
2. **Distinct live buffers.** Do NOT alias two simultaneously-live tensors to the
   same cache slot. Audit the data flow (`r = b.sub(Bx)` makes `r` a fresh
   tensor aliasing neither; `d = z.clone()` — decide whether `d` gets its own
   persistent slot and an explicit copy).
3. **Shape/dtype/device in the key.** f32 and f64 both run (parity tests use
   both); 2-D and 3-D; and different configs use different grids. The cache must
   not hand an f32 solve an f64 buffer.
4. **Pointer stability is the whole point.** Once cached, a slot's `data_ptr`
   must never change across steps. Never `resize_`/reassign a cached tensor
   except on a genuine shape change (new grid).
5. **First solve is cold.** The first solve (allocates the cache) must happen
   BEFORE any graph capture — it already does (capture starts after warmup steps
   run eager). No action needed, but don't move allocation into a captured path.
6. **CUDA-graph capture safety.** After warmup, a captured region that calls the
   driver must hit zero allocations. If the pre-Poisson graph ever grows to
   include the Poisson solve (`poisson_cuda_graph`, GU6), the persistent buffers
   make that capturable; verify.
7. **Don't regress `multigrid`.** It shares `vcycle_*`. If you convert the vcycle
   buffers, re-run the multigrid parity + convergence checks.
8. **CPU twin.** `multigrid_cpu.cpp` uses `std::vector` scratch (`vcycle_*_cpu`,
   `mg_vcycle_*_cpu`). The graph bug is CUDA-only, so CPU is optional — but if you
   want symmetry, cache the vectors there too. Keep CPU/CUDA numerics identical.
9. **Thread-safety.** The `static` cache is process-global. If multiple solvers
   or threads ever share it, key must include enough to disambiguate, or make it
   `thread_local`. Current usage is single-threaded per process.
10. **Memory is held for process life.** Acceptable (it's reused scratch), but note
    it in a comment; `empty_cache` won't reclaim it.

---

## 6. Validation plan (must all pass)

**A. Reproduce the bug is GONE without the mitigation.** Run the real config with
the `expandable_segments` export removed/overridden and the graph ON:

```bash
# temporarily: run with the default allocator to prove the root fix stands alone
env PYTORCH_CUDA_ALLOC_CONF="" <venv>/bin/python - <<'PY'
from lilytorch.examples._1guillasim.experiments.gen_config_surface_pool import SimConfig
class C(SimConfig):
    def __init__(self):
        super().__init__(); self.headless=True
        self.n_iterations=600; self.bdim_nt=601
C().run()
PY
```
Before the fix this crashes within ~tens of steps; after, it must run 600 clean.
(venv: `/data/andreaferrario/venv_ns_312/bin/python`.)

**B. Steady-state does zero allocations.** After ~20 warmup steps, assert the
allocator makes no new segments during a solve. Isolated harness:

```python
import torch
from lilytorch.src.poisson_mult import PoissonSolver
# build a 900x300x52 rmgcg solve (see repro below), warm up 20 solves, then:
torch.cuda.synchronize(); s0 = torch.cuda.memory_stats()
# ... one more solve ...
torch.cuda.synchronize(); s1 = torch.cuda.memory_stats()
assert s1["segment.all.allocated"] == s0["segment.all.allocated"]  # no growth
assert s1["num_device_alloc"] == s0["num_device_alloc"]            # no cudaMalloc
```

Isolated rmgcg solve (matches the config: tol 1e-5, max_vcycles 30,
max_mgcg_cycles 10, nsmoothing 5, jacobi, recycle_k 0):

```python
dev, dt, (Nx,Ny,Nz) = "cuda", torch.float64, (900,300,52)
s = PoissonSolver(dt, dev, 2.0/Ny, tol=1e-5, max_vcycles=30, max_cycles=10,
                  nsmoothing=5, w=1, smoother="jacobi", verbose=False,
                  precond_vcycles=1, recycle_k=0)
faces = dict(ch=torch.full((Nx+1,Ny,Nz),1.,dtype=dt,device=dev),
             cv=torch.full((Nx,Ny+1,Nz),1.,dtype=dt,device=dev),
             cw=torch.full((Nx,Ny,Nz+1),1.,dtype=dt,device=dev))
p = torch.zeros(Nx+2,Ny+2,Nz+2,dtype=dt,device=dev)
for _ in range(25):
    f = torch.randn(Nx,Ny,Nz,dtype=dt,device=dev); f-=f.mean()
    p,_ = s.solve_rmgcg(f, p, **{k:v.clone() for k,v in faces.items()})
```

**C. Numerics unchanged.** The solve must be bit-identical (or within existing
tolerances) to pre-change. Reuse the session bench idea: on a stiff z-jump
operator (80:1 and 833:1), mgcg-30 residual must match pre-change
(80:1 ≈ 1.4e-6, 833:1 ≈ 9.2e-6) and the V-cycle preconditioner must stay
symmetric (rel-asym ~1e-15). See `test_poisson_driver.py::test_mgcg_*`.

**D. Full suite green:**
```bash
<venv>/bin/python -m pytest lilytorch/tests/test_poisson_driver.py \
     lilytorch/tests/test_two_phase.py -q      # then the whole suite
```
Baseline this session: 372 passed, 1 skipped. Must stay green (both f32/f64,
both smoothers, both devices, CPU↔CUDA parity).

**E. `compute-sanitizer --tool memcheck`** on the isolated rmgcg solve: 0 errors
(the cache must not introduce OOB or use-after-scope).

**F. Rebuild between edits:** `PYTHON=<venv>/bin/python bash lilytorch/src/build.sh`.

---

## 7. Out of scope / do not touch

- Do not remove the `expandable_segments` export from `gen_sh_config` in the same
  change — leave it as belt-and-suspenders until B/D prove the root fix, then
  optionally remove in a separate commit.
- Do not alter `restrict_fw_*` or the `variational` gating — that is the
  (already-fixed) convergence work, orthogonal to this.
- Do not change convergence behavior (cycle counts, tol, gauge) — allocation only.

---

## 8. References

- Memory: `project_mgcg_graph_capture_cudamalloc.md` (this bug),
  `project_mgcg_nonsymmetric_preconditioner.md` (the restrict_fw fix, same files),
  `project_projection_fresh_tensor_graph_leak.md` /
  `project_forces_cuda_graph_spike.md` (the pointer-stable-buffer idiom to imitate).
- Code: `lilytorch/src/csrc/cuda/poisson_solve.cu`
  (`poisson_solve_{mgcg,rmgcg}_{2,3}d_cuda`, `vcycle_{2,3}d`,
  `poisson_solve_multigrid_*` as the low-churn reference);
  `lilytorch/src/csrc/multigrid_cpu.cpp` (CPU twins);
  `lilytorch/examples/base_sim_config.py::gen_sh_config` (current mitigation).
- Driver dispatch: `lilytorch/src/poisson_mult.py` (`solve_mgcg`, `solve_rmgcg`,
  `_cg_core`, `_dispatch_vcycle`).
