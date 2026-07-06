# Phase I — Fused BDIM Kernel: Memory Reduction Plan

## Objective

Eliminate ~3.9 GB of persistent GPU allocations in 3-D kernel mode by fusing the
SDF-update, mu/normals computation, BDIM velocity update, and variable-density
coefficient computation into a single Python-level sub-step that writes only the
minimum set of persistently stored tensors: **u, v, w, p, ch, cv, cw**.

All other tensors that are currently stored across steps (sdf_val_u/v/w,
body_u/v/w, _winning_rho_cc, and the 20-channel _mu_pack) become temporary
register-level or AABB-scoped intermediates that are never materialized as
full-grid global tensors.

### Speed constraint (non-negotiable)
The refactoring must not increase the total wall-clock time per step on the
kernel path. Every eliminated tensor write is O(N) bandwidth: removing them
should give a small speedup, not a regression.

### Complexity constraint
Do not add new abstractions or intermediate layers. The goal is to remove code
and tensors, not rearrange them.

---

## Background: Current step pipeline (kernel mode, 3-D)

```
BDIMhandler.step()
  ├─ _update_3d_streaming_multi()          [BDIMhandler.py ~1015]
  │    ├─ reset sdf_val, sdf_val_u/v/w, body_u/v/w in dirty sub-block
  │    └─ streaming_sdf_min_rho_3d_multi() → writes 8 full-grid tensors
  │         sdf_val (CC), sdf_val_u, sdf_val_v, sdf_val_w  (4×131MB)
  │         body_u, body_v, body_w                          (3×131MB)
  │         winning_rho_cc                                  (1×131MB)
  │
  ├─ FluidSolver._recompute_mu_normals()   [solver.py ~1170]
  │    └─ reads 4 SDF fields → writes _mu_pack[20,…]       (20×131MB = 2625MB)
  │         channels 0-3:  mu0 [u,v,w,CC]
  │         channels 4-7:  mu1 [u,v,w,CC]   ← channel 7 (mu1_CC) never used downstream
  │         channels 8-19: normals_x/y/z [u,v,w,CC]
  │
  └─ FluidSolver.fluid_step()              [solver.py ~1374]
       ├─ adv_diff_solver.solve()
       ├─ _apply_bdim_all_axes()           reads _mu_pack[0:3,4:7,8:19] for u,v,w axes
       ├─ set_BCs()
       ├─ _FS_FREE_AFTER_BDIM             sets mu1_all_* and normals_*_* to None
       ├─ _compute_variable_density_coefficients()
       │    reads _mu_pack[0:4] (mu0 channels) → writes ch, cv, cw, ch_cc_persist
       │    (narrow-band: only union AABB sub-block written)
       ├─ _FS_FREE_AFTER_VAR_DENS        sets mu0_all_* to None
       └─ project()
            └─ uses ch, cv, cw (multigrid path; ch_cc_persist unused)

  [after fluid_step, in step()]
  └─ forces_method2_3d()                  [forces.py]
       reads normal_x/y/z (CC) from self.normal_x/y/z
       (these are still alive because _release_bdim_fields keeps them in kernel mode)
```

### Persistent tensors at step boundary (currently):
| Tensor | Shape | MB |
|---|---|---|
| u, v, w, p | [N,N,N] | 524 |
| ch, cv, cw | [N,N,N] | 393 |
| sdf_val (CC) | [N,N,N] | 131 |
| sdf_val_u/v/w | [N,N,N] | 393 |
| body_u/v/w | [N,N,N] | 393 |
| _winning_rho_cc | [N,N,N] | 131 |
| _mu_pack[20,N,N,N] | — | 2625 |
| inv_eig (FFT only) | [N,N,N] | 131 |
| **Total** | | **~4721 MB** |

### Target persistent tensors after this phase:
| Tensor | Shape | MB |
|---|---|---|
| u, v, w, p | [N,N,N] | 524 |
| ch, cv, cw | [N,N,N] | 393 |
| sdf_val (CC) | [N,N,N] | 131 |
| inv_eig (FFT only) | [N,N,N] | 131 |
| **Total** | | **~1048 MB** |

**Savings: ~3.7 GB** at 1024×256×128 (3-D kernel mode).

Within a step, two sets of full-grid tensors are allocated transiently (between
Kernel A and Kernel B) and then freed:
- `sdf_val_u/v/w`: 3 × 131 MB = 393 MB per-step temporary
- `body_u/v/w`: 3 × 131 MB = 393 MB per-step temporary

Peak within a step (advdiff output + Kernel A temporaries + multigrid overhead)
is higher, but nvidia-smi between steps will show only ~1048 MB.

---

## Design: Two new CUDA kernels

The new kernel-mode step replaces the current chain of four separate passes
(streaming SDF update → `_recompute_mu_normals` → `_apply_bdim_all_axes` →
`_compute_variable_density_coefficients`) with **two CUDA kernels**. No
Python-level intermediate tensors are created. mu0, mu1, and normals are
computed entirely inside CUDA thread registers and are **never stored to global
GPU memory**.

### Kernel A — `streaming_sdf_stag_3d_multi` (modified from existing)

Nearly identical to the current `streaming_sdf_min_rho_3d_multi`.

**Outputs (written to full-grid tensors):**
- `sdf_val` (CC) → **persistent** (needed by `streaming_sdf_forces_post_3d`)
- `sdf_val_u`, `sdf_val_v`, `sdf_val_w` → **per-step temporaries**, freed after Kernel B
- `body_u`, `body_v`, `body_w` → **per-step temporaries**, freed after Kernel B

**Removed from current kernel:**
- `winning_rho_cc` — no longer written; Kernel B computes `rho_eff` from `mu0`
  in registers and writes `ch/cv/cw` directly

The 3-pass structure (init_keys → min_rho_kernel → decode_keys) is kept.
In the decode pass, remove the `winning_rho_cc` write.

### Kernel B — `bdim_vardens_3d_multi` (new kernel)

One CUDA thread per cell `(i,j,k)` inside the dirty union AABB.

**Reads (from global memory):**
- `u_prime[i,j,k]` and its 6 neighbors (advdiff output tensor, read-only)
- `sdf_val_u[i,j,k]` and its 6 face-SDF neighbors (for FD normal)
- `body_u[i,j,k]`
- Repeat for v-face and w-face

**Computes (in thread registers, NEVER stored globally):**
```
phi_u   = sdf_val_u[i,j,k]
mu0_u   = smooth_heaviside(phi_u / eps)
mu1_u   = smooth_delta(phi_u / eps) / eps

nx_u = (sdf_u[i+1,j,k] - sdf_u[i-1,j,k]) / (2h)
ny_u = (sdf_u[i,j+1,k] - sdf_u[i,j-1,k]) / (2h)
nz_u = (sdf_u[i,j,k+1] - sdf_u[i,j,k-1]) / (2h)
# normalize: nn = sqrt(nx²+ny²+nz²); nx/=nn, ny/=nn, nz/=nn

# BDIM2 normal derivative (reads u_prime neighbors):
du_dx = (u_prime[i+1,j,k] - u_prime[i-1,j,k]) / (2h)
du_dy = (u_prime[i,j+1,k] - u_prime[i,j-1,k]) / (2h)
du_dz = (u_prime[i,j,k+1] - u_prime[i,j,k-1]) / (2h)
nd_u  = nx_u*du_dx + ny_u*du_dy + nz_u*du_dz

u_new = mu0_u*(u_prime[i,j,k] - body_u[i,j,k]) + body_u[i,j,k] + mu1_u*nd_u
ch_val = dt / (rho_f + (rho_body - rho_f) * mu0_u)
```
(repeat identically for v-face and w-face within the same thread)

**Writes (to global memory, one cell per thread):**
```
u0[i,j,k]  = u_new
ch[i,j,k]  = ch_val
v0[i,j,k]  = v_new
cv[i,j,k]  = cv_val
w0[i,j,k]  = w_new
cw[i,j,k]  = cw_val
```

**No race conditions:** `u_prime / v_prime / w_prime` (inputs) are separate tensor
allocations from `u0 / v0 / w0` (outputs). The advdiff solver returns new tensors.
Kernel B reads from the advdiff output and writes to the persistent velocity fields
— different memory regions, no WAR hazard.

**CC quantities:** Not computed in this kernel. `sdf_val` CC is already written
by Kernel A and kept for the forces pass. CC normals are computed internally by
`streaming_sdf_forces_post_3d` (which has access to the body SDF table).

### New fluid_step (kernel path)

```python
# Step 1: advection-diffusion (unchanged)
primes = self.adv_diff_solver.solve(u0, v0, w0, nu_t=nu_t)  # returns new tensors

# Step 2a: SDF at staggered faces + body velocities (Kernel A)
sdf_u = torch.full(grid_shape, _FAR, ...)   # per-step temporary
sdf_v = torch.full(grid_shape, _FAR, ...)
sdf_w = torch.full(grid_shape, _FAR, ...)
body_u_tmp = torch.zeros(grid_shape, ...)   # per-step temporary
body_v_tmp = torch.zeros(grid_shape, ...)
body_w_tmp = torch.zeros(grid_shape, ...)
streaming_sdf_stag_3d_multi(
    ..., sdf_val, sdf_u, sdf_v, sdf_w, body_u_tmp, body_v_tmp, body_w_tmp
)

# Step 2b: BDIM2 update + var-dens coefficients (Kernel B, in-place on u0/v0/w0 and ch/cv/cw)
bdim_vardens_3d_multi(
    primes[0], primes[1], primes[2],        # u'/v'/w' inputs (read-only)
    sdf_u, sdf_v, sdf_w,                   # face SDFs (read-only)
    body_u_tmp, body_v_tmp, body_w_tmp,    # body velocities (read-only)
    u0, v0, w0,                             # velocity outputs (written)
    ch_persist, cv_persist, cw_persist,    # Poisson coeffs (written)
    eps, rho_body, rho_f, dt,
    dirty_i0, dirty_j0, dirty_k0,
    dirty_Ai, dirty_Aj, dirty_Ak,
)

# Step 2c: free per-step temporaries
del sdf_u, sdf_v, sdf_w, body_u_tmp, body_v_tmp, body_w_tmp

# Step 3-6: set_BCs, project, sponge, set_BCs (unchanged)
self.adv_diff_solver.set_BCs(u0, v0, w0)
(u0, v0, w0, p0) = self.project(u0, v0, w0, p0,
                                  ch=ch_persist, cv=cv_persist, cw=cw_persist)
# Step 7: forces (unchanged; streaming_sdf_forces_post_3d reads sdf_val CC internally)
```

---

## Detailed changes required

### 1. `lilytorch/src/body.py` — remove persistent staggered SDF/body-vel allocations

**File**: `lilytorch/src/body.py`

In `MultiAnimatBodies.__init__` / `Body._setup_grids()`, the following six
tensors are currently allocated as persistent fields in kernel mode.  After
this refactor they become per-step temporaries allocated in `fluid_step` and
freed immediately after Kernel B.  Remove them from the body allocation entirely
when `use_kernels=True`:

```python
# Guard with use_kernels — these no longer exist as persistent body fields
if not use_kernels:
    self.sdf_val_u = torch.full(self.grid_shape, _FAR, ...)
    self.sdf_val_v = torch.full(self.grid_shape, _FAR, ...)
    self.sdf_val_w = torch.full(self.grid_shape, _FAR, ...)
    self.body_u    = torch.zeros(self.grid_shape, ...)
    self.body_v    = torch.zeros(self.grid_shape, ...)
    self.body_w    = torch.zeros(self.grid_shape, ...)
```

Also remove `_winning_rho_cc` from the body allocation — Kernel B computes
`rho_eff` from `mu0` directly in registers and writes `ch/cv/cw` without ever
storing a per-cell winning density.

The `use_kernels` flag is already threaded through `body_from_yaml` and
`MultiAnimatBodies.__init__`.

### 2. `lilytorch/integration/BDIMhandler.py` — gut `_update_3d_streaming_multi`

**File**: `lilytorch/integration/BDIMhandler.py`

`_update_3d_streaming_multi` currently:
1. Resets dirty sub-blocks of `sdf_val`, `sdf_val_u/v/w`, `body_u/v/w`, `winning_rho_cc`
2. Calls `streaming_sdf_min_rho_3d_multi` with all 8 full-grid output tensors

After the refactor this function only prepares the kinematic state that
`fluid_step` will pass to the new kernels.  The dirty-block fills and the
kernel call are moved into `fluid_step` (see §4).  Remove:

```python
# REMOVE: dirty-block fills for staggered fields (no longer persistent)
comp.sdf_val_u[...].fill_(_FAR)
comp.sdf_val_v[...].fill_(_FAR)
comp.sdf_val_w[...].fill_(_FAR)
comp.body_u[...].zero_()
comp.body_v[...].zero_()
comp.body_w[...].zero_()
comp._winning_rho_cc[...].zero_()

# REMOVE: the streaming_sdf_min_rho_3d_multi() call that writes all 8 fields
```

Keep: the `sdf_val` CC dirty-block fill (it is still written by Kernel A) and
the AABB/kinematics bookkeeping (`_kernel_step`, `_combined_union_aabb`) which
`fluid_step` still reads.

### 3. `lilytorch/src/kernels/csrc/cuda/streaming_sdf.cu` — two new kernel functions

**File**: `lilytorch/src/kernels/csrc/cuda/streaming_sdf.cu`

#### 3a. Kernel A — `streaming_sdf_stag_3d_multi_cuda`

Clone `streaming_sdf_min_rho_3d_multi_cuda` and rename it.  Change the decode
pass:
- Keep writes of `sdf_cc` (CC SDF, persistent output needed by forces)
- Keep writes of `sdf_u/v/w` and `body_u/v/w` (per-step temporaries, but
  caller-allocated full-grid tensors passed in)
- **Remove** the `winning_rho_cc` output argument and all writes to it — it
  is no longer needed

Signature change (remove last argument):
```cpp
void streaming_sdf_stag_3d_multi_cuda(
    ...same as current minus winning_rho_cc...
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v, at::Tensor sdf_w,
    at::Tensor body_u, at::Tensor body_v, at::Tensor body_w,
    ...dirty AABB args...);
```

Register under `torch.ops.lilytorch_kernels.streaming_sdf_stag_3d_multi`.

Do **NOT** modify or remove the existing `streaming_sdf_min_rho_3d_multi_cuda` —
it remains the python-path implementation.

#### 3b. Kernel B — `bdim_vardens_3d_cuda` (new kernel)

One CUDA thread per cell `(i,j,k)` inside the dirty union AABB.
All mu0/mu1/normal values are computed in thread registers and **never written
to global memory**.

```cpp
template <typename scalar_t>
__global__ void bdim_vardens_3d_kernel(
    // advdiff outputs (read-only, separate allocations from u0/v0/w0)
    const scalar_t* __restrict__ u_prime,
    const scalar_t* __restrict__ v_prime,
    const scalar_t* __restrict__ w_prime,
    // staggered face SDFs from Kernel A (read-only)
    const scalar_t* __restrict__ sdf_u,
    const scalar_t* __restrict__ sdf_v,
    const scalar_t* __restrict__ sdf_w,
    // rigid body face velocities from Kernel A (read-only)
    const scalar_t* __restrict__ body_u,
    const scalar_t* __restrict__ body_v,
    const scalar_t* __restrict__ body_w,
    // persistent velocity fields (written, one cell per thread)
    scalar_t* __restrict__ u0,
    scalar_t* __restrict__ v0,
    scalar_t* __restrict__ w0,
    // persistent Poisson coefficients (written, one cell per thread)
    scalar_t* __restrict__ ch,
    scalar_t* __restrict__ cv,
    scalar_t* __restrict__ cw,
    // BDIM parameters
    scalar_t eps, scalar_t rho_body, scalar_t rho_f, scalar_t dt,
    scalar_t inv_h,   // 1/(2h) for FD normals and BDIM2 gradient
    int Ngx, int Ngy, int Ngz,
    // dirty AABB bounds
    int di0, int dj0, int dk0,
    int dAi, int dAj, int dAk)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= dAi * dAj * dAk) return;
    // reconstruct (i,j,k) from local index + dirty offsets
    const int dk = local % dAk;
    const int dj = (local / dAk) % dAj;
    const int di = local / (dAk * dAj);
    const int i = di0 + di, j = dj0 + dj, k = dk0 + dk;
    const int g  = i * Ngy * Ngz + j * Ngz + k;

    // --- u-face ---
    scalar_t phi_u = sdf_u[g];
    scalar_t mu0_u = smooth_heaviside(phi_u / eps);
    scalar_t mu1_u = smooth_delta(phi_u / eps) / eps;
    // finite-difference normal from face-SDF field (6 neighbour reads)
    scalar_t nx_u = (sdf_u[(i+1)*Ngy*Ngz + j*Ngz + k] -
                     sdf_u[(i-1)*Ngy*Ngz + j*Ngz + k]) * inv_h;
    scalar_t ny_u = (sdf_u[i*Ngy*Ngz + (j+1)*Ngz + k] -
                     sdf_u[i*Ngy*Ngz + (j-1)*Ngz + k]) * inv_h;
    scalar_t nz_u = (sdf_u[i*Ngy*Ngz + j*Ngz + (k+1)] -
                     sdf_u[i*Ngy*Ngz + j*Ngz + (k-1)]) * inv_h;
    scalar_t nn = sqrtf(nx_u*nx_u + ny_u*ny_u + nz_u*nz_u);
    if (nn > 1e-8f) { nx_u /= nn; ny_u /= nn; nz_u /= nn; }
    // BDIM2 normal derivative (reads u_prime neighbours — different tensor from u0)
    scalar_t du_dx = (u_prime[(i+1)*Ngy*Ngz + j*Ngz + k] -
                      u_prime[(i-1)*Ngy*Ngz + j*Ngz + k]) * inv_h;
    scalar_t du_dy = (u_prime[i*Ngy*Ngz + (j+1)*Ngz + k] -
                      u_prime[i*Ngy*Ngz + (j-1)*Ngz + k]) * inv_h;
    scalar_t du_dz = (u_prime[i*Ngy*Ngz + j*Ngz + (k+1)] -
                      u_prime[i*Ngy*Ngz + j*Ngz + (k-1)]) * inv_h;
    scalar_t nd_u = nx_u*du_dx + ny_u*du_dy + nz_u*du_dz;
    scalar_t ub = body_u[g];
    u0[g] = mu0_u*(u_prime[g] - ub) + ub + mu1_u*nd_u;
    ch[g] = dt / (rho_f + (rho_body - rho_f)*mu0_u);

    // --- v-face (same pattern with sdf_v, v_prime, body_v → v0, cv) ---
    // ... [analogous code] ...

    // --- w-face (same pattern with sdf_w, w_prime, body_w → w0, cw) ---
    // ... [analogous code] ...
}
```

**No race conditions**: `u_prime/v_prime/w_prime` are the advdiff output tensors
(separate allocations from `u0/v0/w0`).  Each thread writes only its own global
index `g`.  Reads of `sdf_u[g±...]` and `u_prime[g±...]` are from read-only
tensors — no WAR hazard.

**Cells outside the dirty AABB**: `u0/v0/w0` and `ch/cv/cw` are already
correct outside the AABB (from the previous step) so no reset is needed for
cells not covered by this kernel.  Only the AABB is processed.

Register under `torch.ops.lilytorch_kernels.bdim_vardens_3d`.

**Rebuild**: `pip install -e . --no-build-isolation` after adding both kernels.

### 4. `lilytorch/src/kernels/ops.py` — Python wrappers for Kernel A and B

Add two new wrapper functions in `ops.py`:

```python
def streaming_sdf_stag_3d_multi(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim, gx, gy, gz, h_grid, max_vol_per_body,
        sdf_cc, sdf_u, sdf_v, sdf_w,
        body_u, body_v, body_w,
        interp_method,
        dirty_i0, dirty_j0, dirty_k0,
        dirty_Ai, dirty_Aj, dirty_Ak) -> None: ...

def bdim_vardens_3d(
        u_prime, v_prime, w_prime,
        sdf_u, sdf_v, sdf_w,
        body_u, body_v, body_w,
        u0, v0, w0,
        ch, cv, cw,
        eps: float, rho_body: float, rho_f: float, dt: float,
        dirty_i0: int, dirty_j0: int, dirty_k0: int,
        dirty_Ai: int, dirty_Aj: int, dirty_Ak: int) -> None: ...
```

Re-export both from `kernels/__init__.py`.

### 5. `lilytorch/src/solver.py` — hook into `fluid_step`, remove old passes

**File**: `lilytorch/src/solver.py`

In `fluid_step()` (kernel path, 3-D only), replace:
```python
# OLD — remove these three calls on the kernel path:
primes = self._apply_bdim_all_axes(primes)
self.__dict__.update(self._FS_FREE_AFTER_BDIM)
coeffs = self._compute_variable_density_coefficients(timestep)
```

With:
```python
# NEW (kernel path only, gated on self._use_kernels):
comp = self.composite_body
_opts = dict(device=self.device, dtype=self.dtype)
sdf_u_tmp  = torch.full(self.grid_shape, _FAR, **_opts)
sdf_v_tmp  = torch.full(self.grid_shape, _FAR, **_opts)
sdf_w_tmp  = torch.full(self.grid_shape, _FAR, **_opts)
bU_tmp     = torch.zeros(self.grid_shape, **_opts)
bV_tmp     = torch.zeros(self.grid_shape, **_opts)
bW_tmp     = torch.zeros(self.grid_shape, **_opts)
# Kernel A: compute staggered face SDFs + body velocities
streaming_sdf_stag_3d_multi(
    ...,  # body table, kin, AABB, grid coords (same as before, minus winning_rho_cc)
    comp.sdf_val, sdf_u_tmp, sdf_v_tmp, sdf_w_tmp,
    bU_tmp, bV_tmp, bW_tmp,
    interp_method=...,
    dirty_i0=..., dirty_j0=..., dirty_k0=...,
    dirty_Ai=..., dirty_Aj=..., dirty_Ak=...,
)
# Kernel B: fused BDIM2 + variable-density coefficients
bdim_vardens_3d(
    primes[0], primes[1], primes[2],    # advdiff outputs (read-only)
    sdf_u_tmp, sdf_v_tmp, sdf_w_tmp,   # staggered face SDFs (read-only)
    bU_tmp, bV_tmp, bW_tmp,            # body velocities (read-only)
    self.u0, self.v0, self.w0,         # persistent velocity fields (written)
    self._ch_persist, self._cv_persist, self._cw_persist,  # Poisson coeffs (written)
    eps=float(comp.eps),
    rho_body=float(self.rho_body), rho_f=float(self.rho),
    dt=float(timestep),
    dirty_i0=..., dirty_j0=..., dirty_k0=...,
    dirty_Ai=..., dirty_Aj=..., dirty_Ak=...,
)
# Free per-step temporaries immediately
del sdf_u_tmp, sdf_v_tmp, sdf_w_tmp, bU_tmp, bV_tmp, bW_tmp
primes = (self.u0, self.v0, self.w0)
```

Also:
- Remove `_recompute_mu_normals()` call on the kernel path (no longer exists)
- Remove `_mu_pack` allocation from `__init__`
- Remove all `mu0_all_u/v/w`, `mu1_all_u/v/w`, `normal_x_u/y_u/z_u`, etc. fields
- Remove `_FS_FREE_AFTER_BDIM` and `_FS_FREE_AFTER_VAR_DENS` dict entries for
  staggered mu/normals (they no longer exist as persistent fields)
- Keep `_ch_persist / _cv_persist / _cw_persist` as pre-allocated full-grid tensors
  (multigrid reads `ch[1:, 1:-1, 1:-1]` etc.)
- Keep the python path (`_use_kernels=False`) completely **unchanged**

### 6. `lilytorch/src/forces.py` — CC normals no longer cached

**File**: `lilytorch/src/forces.py`, function `forces_method2_3d`

`streaming_sdf_forces_post_3d` already computes CC normals internally from the
body SDF table at query time.  `forces_method2_3d` must NOT read
`self.normal_x/y/z` or `self.mu0_all` in the kernel path.  Verify by checking
whether the CUDA forces kernel uses these as inputs or computes them itself.

If the CUDA kernel computes normals from the body SDF table internally:
- Remove `self.normal_x/y/z` from `_release_bdim_fields` keep-set
- Remove `self.normal_x/y/z` full-grid allocations (saves 3 × 131 MB = 393 MB more)

If `forces_method2_3d` has a Python preamble that reads `self.normal_x/y/z`
(check `lilytorch/src/forces.py`), verify whether this is used only for
diagnostics/plotting (safe to gate on a flag) or for the kernel call itself
(must be checked more carefully).

This verification step can be done during implementation — it is not a
blocker for the memory savings from Kernels A and B.

---

## Testing plan (NO FARMS required — use standalone jellyfish)

### Test harness

Use `lilytorch/examples/jellyfish/run_jellyfish_fluid.py`.
The jellyfish is a pure analytical body — no FARMS, no MuJoCo, no mesh files.
It exercises the full 3-D kernel path including SDF streaming, forces, and
the variable-density projection.

### Test 1: Correctness (regression)

Run 10 steps with the **old** code, save `u0`, `v0`, `w0`, `p0` at each step
to disk. Then run the **new** code and verify field agreement to within float32
tolerance (max error < 1e-4). Script:

```bash
cd /data/andreaferrario/lilytorch
PYTHON=/data/andreaferrario/venv_ns_312/bin/python

# Save reference outputs from old code (before any changes — run on optimize_speed_memory)
$PYTHON - << 'EOF'
import torch
from lilytorch.examples.jellyfish.run_jellyfish_fluid import build_solver
solver = build_solver("lilytorch/examples/jellyfish/config_fluid.yaml")
solver.composite_body.update(0.0, 0, dt=float(solver.dt))
refs = []
for i in range(10):
    solver.step_(solver.u0, solver.v0, solver.w0, solver.p0, i)
    refs.append((solver.u0.clone(), solver.v0.clone(), solver.w0.clone(), solver.p0.clone()))
torch.save(refs, "/tmp/jellyfish_ref.pt")
print("Reference saved.")
EOF

# After implementing changes, verify:
$PYTHON - << 'EOF'
import torch
refs = torch.load("/tmp/jellyfish_ref.pt")
from lilytorch.examples.jellyfish.run_jellyfish_fluid import build_solver
solver = build_solver("lilytorch/examples/jellyfish/config_fluid.yaml")
solver.composite_body.update(0.0, 0, dt=float(solver.dt))
for i, (ru, rv, rw, rp) in enumerate(refs):
    solver.step_(solver.u0, solver.v0, solver.w0, solver.p0, i)
    err_u = (solver.u0 - ru).abs().max().item()
    err_p = (solver.p0 - rp).abs().max().item()
    print(f"step {i:2d}  max|Δu|={err_u:.2e}  max|Δp|={err_p:.2e}")
    assert err_u < 1e-4, f"u diverged at step {i}"
EOF
```

### Test 2: Memory reduction

```bash
$PYTHON - << 'EOF'
import torch
from lilytorch.examples.jellyfish.run_jellyfish_fluid import build_solver
torch.cuda.reset_peak_memory_stats()
solver = build_solver("lilytorch/examples/jellyfish/config_fluid.yaml")
solver.composite_body.update(0.0, 0, dt=float(solver.dt))
for i in range(5):
    solver.step_(solver.u0, solver.v0, solver.w0, solver.p0, i)
alloc = torch.cuda.memory_allocated() / (1024**2)
peak  = torch.cuda.max_memory_allocated() / (1024**2)
rsrv  = torch.cuda.memory_reserved() / (1024**2)
print(f"alloc={alloc:.0f} MB  peak={peak:.0f} MB  reserved={rsrv:.0f} MB")
# At jellyfish default grid (~256x64x64) expect alloc < 700 MB (was ~1800 MB)
EOF
```

### Test 3: Speed (no regression)

```bash
$PYTHON - << 'EOF'
import torch, time
from lilytorch.examples.jellyfish.run_jellyfish_fluid import build_solver
solver = build_solver("lilytorch/examples/jellyfish/config_fluid.yaml")
solver.composite_body.update(0.0, 0, dt=float(solver.dt))
# warmup
for i in range(5):
    solver.step_(solver.u0, solver.v0, solver.w0, solver.p0, i)
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(20):
    solver.step_(solver.u0, solver.v0, solver.w0, solver.p0, i+5)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"20 steps: {dt*1000:.0f} ms  ({dt/20*1000:.1f} ms/step)")
EOF
```

---

## Implementation order (recommended)

1. **Save reference outputs** (Test 1 first half) on the current
   `optimize_speed_memory` code before making any changes.

2. **CUDA Kernel A** (§3a): add `streaming_sdf_stag_3d_multi_cuda` — clone of
   existing kernel with `winning_rho_cc` removed.  Register in `ops.cpp`.
   Add Python wrapper (§4).  Rebuild and smoke-test.

3. **CUDA Kernel B** (§3b): add `bdim_vardens_3d_cuda` — the fused BDIM+vardens
   kernel.  Register in `ops.cpp`.  Add Python wrapper (§4).  Rebuild.

4. **Body allocations** (§1): gate `sdf_val_u/v/w`, `body_u/v/w`, `_winning_rho_cc`
   on `not use_kernels`.

5. **BDIMhandler** (§2): remove dirty-block fills and old kernel call.

6. **`fluid_step`** (§5): allocate per-step temporaries, call Kernel A then B,
   free temporaries, remove old passes.

7. **Solver cleanup** (§5 continued): remove `_mu_pack`, staggered mu/normals
   allocations, `_recompute_mu_normals`, `_apply_bdim_all_axes` calls from
   kernel path.

8. **Forces** (§6): verify whether CC normals are needed; remove if possible.

9. **Run all three tests** (§ Testing). Fix any remaining attribute references
   to removed tensors.

---

## Key invariants to preserve

- The **python path** (`solver_method="python"`, `use_kernels=False`) must be
  completely unchanged.  Every new code path must be gated on
  `self._use_kernels`.

- `comp.sdf_val` (CC) must still be filled by Kernel A because
  `streaming_sdf_forces_post_3d` reads it.

- `_ch_persist / _cv_persist / _cw_persist` must be pre-allocated as full-grid
  tensors (multigrid reads `ch[1:, 1:-1, 1:-1]` etc. from them).  Always pass
  them explicitly to `project()` so the `if ch is None: ch = coeff * mu0_all_u`
  fallback inside `project()` is never reached on the kernel path.

- Kernel B reads `u_prime/v_prime/w_prime` (advdiff outputs) and writes to
  `u0/v0/w0` (persistent fields).  These are **different tensor objects** — do
  not inadvertently alias them (e.g. do not do `u0 = u_prime` before calling
  Kernel B).

- Boundary cells of the dirty AABB: Kernel B reads `sdf_u[i±1,j,k]` and
  `u_prime[i±1,j,k]` for cells at the AABB boundary.  These neighbors must
  exist in the full-grid tensors (they always do since both are full-grid).
  Add a 1-cell safety margin to the dirty AABB bounds passed to Kernel B to
  avoid out-of-bounds reads at the grid edges.



---

## What this does NOT touch

- 2-D kernel path: identical analysis applies, but handle in a separate phase
  to reduce risk.
- FFT Poisson path: `_ch_cc_persist` and `inv_eig` are untouched.
- Multigrid solver internals: unchanged.
- Python path (`use_kernels=False`): unchanged.
- Advection-diffusion solver: unchanged.
- Force computation: unchanged for Phase I; `normal_x/y/z` are still written
  full-grid (but only populated in the AABB region). Phase II can move this
  into `streaming_sdf_forces_post_3d`.
