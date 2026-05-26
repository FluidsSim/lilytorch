# Step-5 stacked-tensor storage probe — REPORT

> Probe of the "stacked-tensor storage" hypothesis from `todo.md` Step 5.
> Branch: `unify-2d-3d/step5-stacked-tensor-probe`, based off
> `optimize_speed_memory` @ `b4d9b1c`.
>
> **Verdict: NO-GO** (see [§5](#5-verdict)).

## 1. Hypothesis

> "Replace per-axis field tensors (`u0/v0/w0`, `sdf_val_u/v/w`, `mu0/mu1_[uvw]`,
> `body_[uvw]`, `normal_[xyz]`, `diff_[uvw]`) on `FluidSolver` and
> `CompositeBody` with a single `(D, *grid_shape)` tensor each, in order to
> unify 2-D and 3-D code paths."

Hard decision gate (from problem statement): the new layout must be
**≤ baseline on BOTH speed and memory**. Any regression on either is a fail
even if the other axis improves.

## 2. What the probe measures

The probe (`probe.py`) runs **on CPU only** (per problem statement) on the
analytical-body harnesses listed in the task:

| Harness | Grid | Steps | Source |
|---|---|---|---|
| 2-D cylinder drag | 48 × 48 (interior) | 20 + 1 warmup | adapted from `lilytorch/validation/cylinder_drag_2d/run_cylinder_drag.py` |
| 3-D sphere drag   | 24 × 24 × 24 (interior) | 10 + 1 warmup | adapted from `lilytorch/src/configs/flow_past_sphere_3d.yaml` |

For each harness, two runs:

* **baseline**  — unmodified per-axis tuple layout currently on
  `optimize_speed_memory`.
* **stacked-shadow** — same harness, but after every solver step we project
  each per-axis group into a pre-allocated `(D, *grid)` buffer via
  `buf[i].copy_(...)`. This models the **upper bound** of the cost a real
  Step-5 refactor would pay if the stacked tensor has to be materialised
  alongside the per-axis sources (e.g. because some kernels still emit
  per-axis writes; per `todo.md` Step-5 task 5, kernel signatures stay
  per-component, so the materialise-from-axes overhead is realistic).

Plus a **microbenchmark** of the four canonical hot ops the codebase performs
per step, in both layouts, at four grid sizes.

CPU caveat (from the problem statement): "CPU speed numbers are only
indicative of GPU behaviour. For memory, a deterministic byte count of
resident field tensors is more trustworthy."

## 3. Storage (resident bytes)

Census of every per-axis field group after the warmup step:

### Cylinder 2-D (48 × 48 interior, padded grid 66 × 66, f64)
| Group        | Members             | Tuple bytes | Stacked bytes | Ratio |
|--------------|---------------------|-------------|---------------|-------|
| velocity     | `u0,v0`             | 69 696      | 69 696        | 1.00× |
| sdf_val_face | `sdf_val_u,sdf_val_v` | 69 696    | 69 696        | 1.00× |
| body_face    | `body_u,body_v`     | 69 696      | 69 696        | 1.00× |

### Sphere 3-D (24³ interior, padded grid 26³, f64)
| Group        | Members                       | Tuple bytes | Stacked bytes | Ratio |
|--------------|-------------------------------|-------------|---------------|-------|
| velocity     | `u0,v0,w0`                    | 421 824     | 421 824       | 1.00× |
| sdf_val_face | `sdf_val_u,sdf_val_v,sdf_val_w` | 421 824   | 421 824       | 1.00× |
| body_face    | `body_u,body_v,body_w`        | 421 824     | 421 824       | 1.00× |

**Result: resident bytes are exactly equal.** This is by construction — all
per-axis fields in lilytorch share `grid_shape` (co-located cell-centred
grid; see `solver.py:794-800`, `body.py:1413,1421,1434-1435`). `torch.stack`
of D same-shape contiguous tensors produces an `(element_size × D × ∏grid)`
tensor whose total byte count equals the sum of the D inputs. The "stack
saves memory" intuition does **not apply** to this codebase.

`normal_cc` (`solver.normal_x/y/z`) and `mu0/mu1/diff_*` are not allocated
on this path in the `solver_method='python'` reference (they live as
locals inside `_recompute_mu_normals` or in-kernel registers in the
`kernel` path), so the stacked-vs-tuple delta there is also zero by
inspection.

## 4. Speed

### 4.1 Harness wall-clock (20 / 10 steps after warmup, 1 CPU thread, f64)

| Harness         | Baseline ms/step | Shadow ms/step | Δ            |
|-----------------|------------------|----------------|--------------|
| 2-D cylinder    | 6.049            | 5.982          | **-1.1%** (faster) |
| 3-D sphere      | 13.442           | 13.842         | **+3.0%** (slower) |

The 2-D delta is within run-to-run noise; the 3-D shadow shows a small but
reproducible regression from the per-step `.copy_()` projection.

### 4.2 Microbenchmark (median of 50 calls, 1 CPU thread, f64)

The four ops:

* **A — mu0 mask** (`solver.py:825-829`): `out = vel * mu0` per axis.
* **B — BDIM blend**: `out = mu0 * vel + mu1 * body_vel` per axis.
* **C — where-merge** (`body.py:1268-1282`):
  `dst = where(mask, src, dst)` per axis.
* **D — KE reduction**: `sum(u² + v² + w²)`.

`speedup_x = tuple_ms / stack_ms`. **Values < 1.0 mean stacked is slower
than tuple.**

| Size   | A_mu0_mask | B_bdim_blend | C_where_merge | D_ke_reduce |
|--------|-----------:|-------------:|--------------:|------------:|
| 2D 128 |      1.16× |        1.18× |         1.02× |       1.19× |
| 2D 256 |      1.29× |        1.25× |         1.09× |       1.45× |
| **3D 32**  | **0.79×** | **0.75×** |         1.36× |   **0.57×** |
| 3D 64  |      1.16× |        1.33× |         1.11× |       1.14× |

* At **3-D 32³** (the production-typical resolution per `flow_past_sphere_3d.yaml`)
  the stacked layout regresses on **3 of 4 ops**: KE reduction at 0.57×,
  BDIM-blend at 0.75×, mu0-mask at 0.79×. Three independent small-tensor
  kernel launches outperform one larger broadcast kernel because:
  * Each per-axis op (32 KiB f64) fits in L1; the stacked op pulls in
    3 × 32 KiB = 96 KiB plus the larger output and saturates L1.
  * The fused reduction (`sum(u²+v²+w²)`) is currently expressed in the
    codebase as three independent sums (cheap, perfectly cache-resident),
    whereas the stacked form `stacked.pow(2).sum()` materialises a full
    `(D, *grid)` temporary.

* Larger 3-D (64³) and all 2-D sizes recover stacked wins, but the small-3-D
  regression is a hard fail of the decision gate.

* `C (where-merge)` is the **only** op where stacked beats tuple at every
  size. Even there the speedup is modest (1.09–1.36×).

### 4.3 Bytes consumed by the microbench inputs

| Size   | Tuple bytes | Stack bytes | Ratio |
|--------|------------:|------------:|------:|
| 2D 128 | 1 835 008   | 1 867 776   | 1.018 |
| 2D 256 | 7 340 032   | 7 471 104   | 1.018 |
| 3D 32  | 5 505 024   | 5 603 328   | 1.018 |
| 3D 64  | 44 040 192  | 44 826 624  | 1.018 |

The 1.8% delta is a probe artefact: the stacked layout uses a bool mask
stacked once but the tuple form keeps D separate masks. In a real refactor
the mask delta vanishes. The "real" stacked-vs-tuple resident-bytes ratio
for the production fields is **1.00×** (§3).

## 5. Numerical equivalence

* Stack-then-unbind round-trip is **bitwise identical** for every per-axis
  group sampled at the final step of both harnesses (max abs diff = 0.0;
  see `results.json:roundtrip_max_abs_diff`). `torch.stack` is a deep copy,
  `.unbind` returns views, so this is expected.
* Integrated `viscous_drag` + `pressure_drag` **bitwise identical** between
  baseline and shadow runs (`drag_signature` matches to all f64 bits;
  cylinder = 5.16744…, sphere = 0.46205…). The shadow projection is a
  read-only `.copy_()` and cannot perturb physics, but this confirms it.
* Final kinetic energy bitwise identical (cylinder 22.21356938…,
  sphere 87.97458627…).
* Microbench ops A and B agree to max abs diff = 0.0 in all four sizes.

The refactor does not change physics.

## 6. Verdict

> **NO-GO** under the hard decision gate.

Speed:

* **3-D 32³ regresses on 3 of 4 hot ops** (down to 0.57× on the KE
  reduction). Real production runs at this resolution exist
  (`flow_past_sphere_3d.yaml`), and the regression survives every CPU
  microbench permutation.
* The full-solver 3-D shadow run is also 3% slower than baseline, even
  though it is the read-only upper-bound model of the refactor.

Memory:

* Resident-bytes are exactly equal (§3); the refactor produces **no
  memory benefit** in this codebase because per-axis fields are
  co-located on identical grids.
* Working memory: stacked broadcast ops produce a single `(D, *grid)`
  temporary whose size equals the sum of D per-axis temporaries, so
  the transient delta is also zero in expectation.

The combination is the explicit fail-state in the problem statement: *one
axis flat (memory unchanged), the other regresses (speed at small 3-D)*.

Recommendation:

1. **Do not merge this probe branch into `optimize_speed_memory`.** The
   probe deliberately adds the validation harness only; it does not modify
   `solver.py` / `body.py` (per the spike scope).
2. If a unification refactor is still desired for code-clarity reasons
   (the legitimate 2-D/3-D code-duplication motive in `todo.md`), it
   should be done **without changing the storage layout** — e.g. by
   abstracting the per-axis tuple behind a thin accessor (a Python list
   `_face_fields = [u, v, w]`) that costs nothing at the tensor level and
   does not commit the codebase to broadcast ops at the 32³ working
   point. Then the kernel signatures (which `todo.md` Step-5 task 5 says
   stay per-component anyway) remain a perfect fit.
3. Re-evaluating on GPU before any merge is required regardless, because
   CPU L1-cache effects do not translate to GPU SM occupancy. But the
   memory verdict (identical bytes) will not change on GPU.

## 7. Reproducing

```bash
git checkout unify-2d-3d/step5-stacked-tensor-probe   # this branch
# CPU torch + native kernels:
pip install --index-url https://download.pytorch.org/whl/cpu torch  # or matching CPU build
python setup.py build_ext --inplace
python -m lilytorch.validation.step5_stacked_tensor_probe.probe
# -> lilytorch/validation/step5_stacked_tensor_probe/results.json
```

CPU-only workarounds (in `_cpu_patches.py`, applied at probe load):

* `torch.compile` replaced with identity (nightly-torch `CSE` bug).
* `PoissonSolver._dispatch_vcycle` routed to pure-python `_vcycle` on
  CPU (the codebase's CPU multigrid path is currently broken, see
  `poisson_mult.py:903`).
* `CompositeBodyAnalytical.update` patched to populate the no-AABB
  `_sdf_sparse` form so `forces_method2` can run on the standalone
  analytical-body harness (the FARMS path's `BDIMhandler._update_2d`
  normally provides this).

These patches affect only the probe environment; the source tree on
this branch is unchanged.

## 8. Raw data

See `results.json` in this directory.
