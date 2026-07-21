# Handoff: CUDA graph capture failure in fluid step pre-projection

**Branch:** `cuda_native_port`. **Status:** ✅ RESOLVED (2026-07-15).

## Resolution

The CUDA graph was NEVER being captured — not because of a `freeAsync` crash
during capture, but because the **graph key changed every step**, so no key
was ever seen 4 times to trigger capture.  Two root causes:

### Root cause 1: `_cached_float` bypassed cache for non-tensor scalars

`solver.py:_cached_float` had an early-return `if not torch.is_tensor(value):
return float(value)` that skipped the cache for plain Python floats.  The
BDIMhandler init path passed `self.fluid_solver.dt` (a 0-d GPU tensor →
cached as ~`0.0010000000474974513`), but the per-step path passed
`self.pars["solver"]["dt"]` (a Python `float(0.001)` → returned `0.001`
WITHOUT checking the cache).  The resulting `_dt_over_rhofluid` values
differed at the 16th decimal digit, making the `_ch_outside_val` guard
fire every step → persistent BDIM coefficient buffers (`_ch_persist`,
`_cv_persist`, `_cw_persist`) were reallocated every step → their
`data_ptr()` values changed → the graph key changed.

**Fix (solver.py):** removed the early-return; `_cached_float` now ALWAYS
checks the cache first, regardless of whether the value is a tensor.

### Root cause 2: `device` comparison `cuda:0 != cuda`

The solver initialises `self.device = torch.device("cuda")`, but tensors
allocated on it report `.device` as `cuda:0`.  The `needs_realloc` guards in
both `BDIMhandler._init_bdim_coeff_persist_{2,3}d` and
`FluidSolver._init_bdim_coeff_persist_{2,3}d` compared `.device != device`
directly, which is always True → reallocation every step.

**Fix (solver.py + BDIMhandler.py):** changed to `.device.type != device.type`
in all four `needs_realloc` checks and the `_mw_div_corr_persist` guards.

### Files changed
| File | Change |
|------|--------|
| `lilytorch/src/solver.py` | `_cached_float`: always check cache first. `_init_bdim_coeff_persist_{2,3}d`: use `.device.type` comparison. |
| `lilytorch/integration/BDIMhandler.py` | `_init_bdim_coeff_persist_{2,3}d`: use `.device.type` comparison (both coefficient and div_corr guards). |
| `lilytorch/src/graph_capture.py` | Added lightweight "First graph captured" notification on first successful capture. |

### Validation
- 200-step `gen_config_surface_pool.py` (900×300×52 grid) runs successfully
  with `backend:cudaMallocAsync`.  Graph captured and replaying.
- Minimal standalone test confirms capture succeeds.

### Workaround (no longer needed)
The `solver["graph_capture_debug"] = True` line in
`gen_config_surface_pool.py:_bdim_extension` can now be removed/commented out.

---

## Background

`NativeWholeStepGraphRunner` (in `graph_capture.py`) captures the fluid step's
pre-projection region — `adv_diff_solver.solve()` + `bdim_forcing_3d()` +
`set_BCs()` — as a single `torch.cuda.CUDAGraph`.  On capture, it fails with:

```
RuntimeError: CUDA error: operation failed due to a previous error during capture
```

Enabling `backend:cudaMallocAsync` revealed the root cause:

```
Warning: freeAsync() was called on an uncaptured allocation during graph capture
```

A Python CUDA tensor allocated **outside** the `with torch.cuda.graph():` block
is being freed (refcount → 0) **inside** the block.  This is illegal during
CUDA stream capture.

## What we know

### Confirmed NOT the cause
- **Poisson solver allocations** — now zero (persistent scratch cache,
  `poisson_scratch.h`).  The Poisson solve runs eagerly, outside the graph.
- **Multiple concurrent captures** — only ONE graph capture is active:
  `_preproj_graph_3d` in `solver.py:2031`.  Two-phase CVOF and forces graphs
  are not active in the failing config (`gen_config_surface_pool.py`).
- **`expandable_segments`** — doesn't help (the bug is a `freeAsync`, not a
  `cudaMalloc`).

### What was tried (none fixed it)
| Attempt | File | Result |
|---------|------|--------|
| Side-stream warmup → default-stream warmup | `graph_capture.py` | Same crash |
| `gc.collect()` before capture | `graph_capture.py` | Same crash |
| `gc.disable()` during capture | `graph_capture.py` | Same crash |
| `empty_cache()` before capture | `graph_capture.py` | PyTorch assertion: `captures_underway.empty()` |
| 3→4 warmup steps before capture | `graph_capture.py` | Same crash |
| `backend:cudaMallocAsync` | `base_sim_config.py` | Revealed `freeAsync` warning, still crashed |
| `garbage_collection_threshold:0` | `base_sim_config.py` | Same crash |
| `RuntimeError` fallback (discard graph, run eager) | `graph_capture.py` | Works but reverts to eager forever |

### What works
- **Eager execution** (`graph_capture_debug: True`) — simulation completes
  normally (tested 1440+ steps, 659s).
- **Isolated captures** — `copy_` alone and `diffuse_add` alone both capture
  successfully in a minimal test script.
- **Poisson persistent buffers** — 0 allocs/solve, 72/72 tests pass.

## Investigation plan

### 1. Bisect the captured operations (highest priority)

Modify `_run_preproj()` in `solver.py:_fluid_step_fused_3d` to capture only ONE
operation at a time:

```python
# Test A: advection+diffusion only
def _run_preproj():
    self.adv_diff_solver.solve(u, v, w_vel, nu_eff=_nu_eff)

# Test B: bdim_forcing only
def _run_preproj():
    bdim_forcing_3d(...)

# Test C: set_BCs only
def _run_preproj():
    self.adv_diff_solver.set_BCs(self.u0, self.v0, self.w0)

# Test D: advection+diffusion + bdim (no BCs)
# Test E: bdim + BCs (no advection)
```

Run each with `gen_config_surface_pool.py` (n_iterations=50).  Whichever
combination first triggers the crash is the culprit.

### 2. Check the `stage()` function

The `stage()` closure runs **before** the capture (eager), doing:
```python
self._bdim_rect_dev.copy_(cpu)   # CPU→GPU copy
```

This `copy_` on the default stream might create an internal temporary whose
lifetime overlaps with the capture.  Try: move the `stage()` work inside
`_run_preproj()` so it's captured too, or ensure `cpu`/`_bdim_rect_dev` are
explicitly kept alive.

### 3. Remove the `nu_eff_t` view creation

In `native.py:diffuse_add`, the constant-viscosity path creates a tensor view:
```python
nu_eff_t = copy_buf[0:1, 0:1, 0:1]  # view — new TensorImpl, shared storage
```

Even though views don't own storage, their TensorImpl allocation/free might
interact badly with graph capture.  Pre-create a persistent 1-element dummy
tensor in `AdvDiffSolver` and pass it instead.

### 4. PyTorch version check

PyTorch 2.6.0+cu124.  Known issues:
- Search https://github.com/pytorch/pytorch/issues for "freeAsync during capture"
  or "CUDAGraph freeAsync uncaptured"
- Try PyTorch 2.5.1 or a recent 2.6 nightly to see if fixed upstream

### 5. Minimal reproduction

Extract `_solve_convective` + `bdim_forcing_3d` + `set_BCs` into a standalone
script with the same tensor shapes as the failing config.  Submit as a PyTorch
bug report if it reproduces without LilyTorch-specific code.

### 6. Alternative: per-op graphs

Instead of one graph for all 3 operations, capture 3 separate smaller graphs.
Each would have a smaller memory footprint and might avoid the `freeAsync`
trigger.  The overhead of 3 replays vs 1 is negligible.

## Files to focus on

| File | Role |
|------|------|
| `lilytorch/src/graph_capture.py` | Graph capture/replay logic |
| `lilytorch/src/solver.py:2031-2120` | `_fluid_step_fused_3d` — where the graph is used |
| `lilytorch/src/solver.py:2055-2090` | `_run_preproj()` — the captured closure |
| `lilytorch/src/advection.py:360-420` | `_solve_convective` — advection+diffusion |
| `lilytorch/src/native.py:452-490` | `diffuse_add` — creates `nu_eff_t` view |
| `lilytorch/src/csrc/cuda/sl_advect.cu` | `diffuse_add_cuda` — native kernel |
| `lilytorch/src/csrc/cuda/bdim_forcing.cu` | `bdim_forcing_3d` — native kernel |

## Current workaround

Add to any config's `_bdim_extension()`:
```python
solver["graph_capture_debug"] = True
```
This disables the graph capture entirely.  Performance impact is small because
the heavy Poisson solve was never captured.

## Changes already on the branch (should be KEPT)

- `lilytorch/src/csrc/poisson_scratch.h` — persistent scratch buffer cache
- `lilytorch/src/csrc/cuda/poisson_solve.cu` — zero-allocation Poisson drivers
- `lilytorch/src/graph_capture.py` — has `gc` import and `RuntimeError` fallback
  (the fallback can be removed if the root cause is fixed)
