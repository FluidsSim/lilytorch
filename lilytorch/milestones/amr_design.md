# Adaptive Mesh Refinement (AMR) for LilyTorch — Design Notes

Status: **design proposal / review**, no code yet.
Branch context: written against `optimize_speed_memory`.
Author: planning assistant, May 2026.

This document is a roadmap for adding *spatial* adaptive mesh refinement to
LilyTorch's BDIM2 incompressible Navier–Stokes solver while keeping the
**time step `dt` global and fixed** (so the FARMS/MuJoCo coupling stays
unchanged). It explains the design trade-offs, points at the most relevant
literature, sketches the equations that change relative to the current
uniform-grid solver, and lists the concrete work items in the order they
should be tackled.

---

## 1. Why AMR, and what we want from it

Today every field in LilyTorch lives on a single uniform MAC grid with a
single scalar spacing `h = dx = dy [= dz]`
(`lilytorch/src/solver.py` lines 274–310, asserted equal). All discrete
operators in `lilytorch/src/operations.py`, the FFT/multigrid Poisson
solvers, the BDIM streaming-SDF kernels
(`lilytorch/src/kernels/csrc/cuda/streaming_sdf*.cu`), and the
multigrid transfer kernels are hard-wired to this single `h`.

For fish/robot swimming we mostly need fine resolution in a thin
*boundary layer* around the swimmer and in its near-wake (vortex
shedding region); the far field is essentially quiescent and could
tolerate 4×–16× coarser cells. With the current uniform grid the cost
scales as `O(N_fine^d)` everywhere; with AMR the *memory and compute
of advection/diffusion/projection* drop to roughly the volume actually
covered by the finest level — typically a 5×–20× speed-up at 3‑D
Reynolds numbers we care about, with the same `dt` and the same wake
fidelity.

Concrete success criteria for this work:

1. **Global synchronous `dt`** — no temporal sub-cycling, identical
   integration cadence to the current code. This is non‑negotiable
   because MuJoCo advances at a fixed `dt` and the BDIM coupling reads
   body poses once per fluid step
   (`integration/BDIMhandler.py`, `BDIMhandler.step`).
2. **BDIM2 semantics preserved** — `mu0`, `mu1`, the σ-shift, the
   `bdim_mu0_projection` flag, and the variable-density Poisson all
   keep their current meaning (`solver.py` lines 322–337).
3. **Forces unchanged in interface** — `forces_method2` /
   `forces_lagrangian_3d` still return one tensor per body; they read
   from whatever level a triangle centroid lands on.
4. **GPU-resident, kernel-driven** — no CPU↔GPU per-step copies, no
   Python-level loops over patches in the hot path; all per-cell work
   stays inside CUDA kernels using the same `packed_key` /
   `gridDim.y`‑over‑bodies idioms already in use
   (`csrc/cuda/packed_key.cuh`, `streaming_sdf.cu`).

---

## 2. Choice of AMR flavour

There are three viable families. The recommendation is **(B)**.

### (A) Octree / quadtree (cell-based)
Examples: Losasso, Gibou & Fedkiw 2004 (octree level-set);
Min & Gibou 2006; Popinet's *Gerris* (2003) and *Basilisk* (2015).
Pros: very high geometric flexibility, naturally tracks thin
boundary layers. Cons: pointer-heavy data structures, hard to
vectorise on GPU, non-trivial MAC-staggering across T-junctions,
multigrid coarsening is awkward. Several GPU ports exist but none
match LilyTorch's PyTorch+CUDA-extension style well.

### (B) **Block-structured AMR (BSAMR)** — recommended
Berger & Oliger 1984; Berger & Colella 1989; Martin & Colella 2000
(projection on locally-refined MAC grids); the AMReX framework
(Zhang et al. 2019, *J. Open Source Softw.*) and the IAMR /
incflo / Nyx applications; the recent **WaterLily-derived BSAMR
paper on Wiley FLD (`fld.5283`, 2024)** which is the closest direct
template — same BDIM family, same staggered MAC layout, same
projection structure as LilyTorch.
Pros: each level is a small set of *uniform* logically-rectangular
patches, so every per-cell kernel we already have can be reused
unchanged on each patch; multigrid (which we already use) extends
naturally to a *composite* multigrid across levels (Martin &
Colella's MLMG); the GPU pattern is "launch one kernel grid per
patch, batch patches via `gridDim.y`" — exactly the
`packed_key`/multi-body pattern in `streaming_sdf*.cu`.
Cons: coarse-fine *flux corrections* and *ghost filling* at level
boundaries are the only real algorithmic novelty; everything else
is reuse.

### (C) Static, body-fitted nested grids (no dynamic regrid)
A degenerate case of (B) with a hand-placed refinement region that
follows the body's bounding box. Trivial to implement, useful as
the *first milestone* of any AMR project, and already enough for
a fish that doesn't move very far during one simulation. Worth
shipping first as a stepping stone before turning on dynamic
regridding.

---

## 3. Reference papers (read in this order)

1. **Wiley FLD `fld.5283` (2024)** — block-structured AMR with BDIM /
   immersed boundaries on a staggered Cartesian grid. Most directly
   applicable; mirror its data layout and projection. (The user-supplied
   reference.)
2. **Martin & Colella, *J. Comput. Phys.* 163 (2000), pp. 271–312** —
   "A cell-centred adaptive projection method for the incompressible
   Euler equations." The canonical reference for projection on
   locally-refined MAC grids; defines the composite divergence,
   composite gradient, and the synchronisation projection that fixes
   coarse-fine divergence after the level solve.
3. **Almgren, Bell, Colella, Howell & Welcome, *J. Comput. Phys.* 142
   (1998)** — variable-density adaptive projection (matches our
   variable-density Poisson with BDIM's `mu0`).
4. **AMReX papers**: Zhang et al. 2019, JOSS; Almgren et al. 2020
   (incflo). The MLMG (multi-level multigrid) algorithm there is the
   target template for our Poisson solve. Open-source C++/CUDA code
   to crib data structures from.
5. **Berger & Colella, *J. Comput. Phys.* 82 (1989)** — the original
   BSAMR with subcycling; we deliberately *drop* the subcycling
   (Section 4).
6. **Sussman, Almgren, Bell, Colella, Howell & Welcome 1999** —
   level-set + projection AMR; useful for the BDIM SDF analogue.
7. **Popinet 2003 / Basilisk** — for the regridding/tagging heuristics
   (vorticity-based, wavelet-error-based).
8. **WaterLily.jl source** (Weymouth & coauthors) — LilyTorch already
   mirrors its `mom_step!`; check the open `WaterLily-AMR` branch /
   the Wiley FLD paper above for the BDIM extension to AMR.
9. **Min & Gibou 2006**, *J. Comput. Phys.* 219 — second-order finite
   differences on non-graded adaptive grids; useful if we later go
   tree-based.
10. **Guittet, Lepilliez, Tanguy & Gibou 2015** — variable-coefficient
    Poisson on adaptive grids with sharp interfaces.

---

## 4. Time stepping with a global, fixed `dt`

Berger–Colella subcycle by a factor of 2 between levels so that the
CFL number is the same on every level. We **do not** subcycle. Instead:

- Pick a single `dt` (the MuJoCo step) and require it to satisfy the
  CFL on the **finest** level `ℓ_max`:

      dt * max|u| / h_{ℓ_max}        ≤ C_adv   (≈ 0.4 with QUICK/Heun)
      dt * 2*ndim*ν_max / h_{ℓ_max}² ≤ C_diff  (≈ 0.5 for explicit diffusion)

  This is what the existing solver already does on the (single) grid;
  for AMR with refinement ratio `r=2` and `L` levels the finest spacing
  is `h_{ℓ_max} = h_0 / 2^{L-1}`, so users dial `L` until the desired
  resolution is reached and accept the corresponding `dt`.

- The same `dt` is used on every level. This wastes a factor `r^L`
  of advection work on coarse levels relative to subcycling, but
  (a) the coarse-level work is tiny anyway because the volume is
  small, and (b) it eliminates all coarse-fine *time* interpolation
  — the only thing left to reconcile at coarse-fine *boundaries* is
  *space*, which is the Martin–Colella synchronisation step. This is
  what makes the FARMS/MuJoCo coupling drop-in.

- Heun's predictor–corrector (`docs/numerical_schemes.rst`,
  `solver.step_`) stays exactly as it is; each of its sub-stages
  (adv-diff, BDIM, projection) is replaced by its composite-grid
  counterpart.

---

## 5. Data layout

The single most important design decision after "no subcycling" is:
**store fields as a list of uniform patches per level, each patch a
contiguous PyTorch tensor with its own ghost layer.**

Proposed structures (Python side, would live in a new
`lilytorch/src/amr.py`):

```
Level ℓ:
    h_ℓ             scalar spacing                  (= h_0 / 2^ℓ)
    patches[ℓ] = list of Patch
        Patch:
            origin_ijk        (3,) int, lower corner in level-ℓ index space
            shape             (Nx, Ny, Nz) interior shape (no ghosts)
            u, v, w           face-centred tensors (Nx+1, Ny, Nz), etc.
            p                 cell-centred  (Nx+2, Ny+2, Nz+2) incl. ghosts
            mu0, mu1_x/y/z    BDIM coefficients on this patch
            sdf_val_cc        signed-distance, cell-centred
            level_mask        (Nx, Ny, Nz) uint8  — 1 if covered by a finer
                              patch (composite operators must skip these)
    coarse_fine_iface[ℓ]  list of CFInterface descriptors (Section 7)
```

Implementation tip: keep all patches at a given level **batched into a
single 4‑D tensor** `(P, Nx, Ny, Nz)` whenever they share `(Nx,Ny,Nz)`
— which they will if we always cut along power-of-two block boundaries
(typical block size 16³ or 32³). This lets us reuse the existing
`gridDim.y`-over-bodies kernel pattern as `gridDim.y`-over-patches.
Variable-shape patches can be handled with a CSR-style "patch
descriptor" buffer and one launch per (Nx,Ny,Nz) bucket.

Recommended block size: **16³ in 3-D, 32² in 2-D**. Large enough that
ghost-cell overhead is small (a 16³ block has 18³ with one ghost layer
= 41% overhead per coarse-fine face; with two ghost layers needed for
QUICK we go to 20³ ≈ 95% overhead — so prefer 32³ if memory allows, or
introduce a thinner ghost layer for QUICK by switching to one-sided
fluxes at patch boundaries).

---

## 6. Discrete operators on the composite grid

Let `ℓ` index refinement levels (`ℓ=0` coarsest), with refinement
ratio `r=2` and `h_ℓ = h_0 / 2^ℓ`.

### 6.1 Advection–diffusion (per level, independent)

On each patch the existing kernels in `adv_diff.py` are applied
**unchanged** with the patch's own `h_ℓ`. The convective stencils
(QUICK, ADBQUICKEST, CUBISTA, …) need a 2-cell ghost layer; the
diffusion stencil needs 1. Those ghosts are filled by:

- Same-level neighbours: straight copy from the neighbour patch.
- Coarser neighbour (`ℓ-1`): **space- and time-coincident** linear or
  quadratic interpolation in the tangential directions, with the
  coarse value placed at the geometric centre of the four (or eight)
  fine ghost cells. Since `dt` is the same, no time interpolation is
  needed.
- Domain boundary: existing BC code, unchanged.

For face-centred quantities (`u, v, w`), refluxing of the conservative
advection flux across a coarse-fine face is required to keep the
discrete divergence consistent — see Section 7.

### 6.2 BDIM streaming SDF and `mu0/mu1`

The streaming-SDF kernels already loop over a body's *AABB* on the
fluid grid and atomically compare-swap into the global SDF/face fields
(`streaming_sdf_min_*_multi`, packed-key 64-bit atomic with body-id).
For AMR:

- For each body, intersect its world-space AABB with each patch's
  world-space extent (cheap O(#patches) test in Python; cache per
  regrid).
- Launch the existing kernel per (level, patch, body) — packed into
  `gridDim.y` over `(patch, body)` pairs so the launch count is
  level-bounded.
- The SDF table is sampled trilinearly in body-local coordinates as
  today, so the sampler is *resolution-agnostic*; only the per-cell
  world coordinates change (use the patch's `h_ℓ` and `origin`).
- `mu0`, `mu1`, and the σ-shift are computed per patch with `eps_ℓ =
  eps_multiplier * h_ℓ` so that the BDIM kernel width scales with the
  local cell — preserving the "≈2 cells" smoothing irrespective of
  level. (Reference: the σ-shift fact in repository memory; same
  formula, replace `eps` by `eps_ℓ`.)
- BDIM is only meaningfully applied where `mu0 < 1`; therefore the
  refinement criterion (Section 8) **must** guarantee that every cell
  with `|φ| < k·eps_max` (k≈4) sits on the finest level. This is the
  hard correctness constraint of BDIM-AMR.

### 6.3 Pressure projection — composite variable-coefficient solve

The Poisson equation today is

    ∇·( (w·dt/ρ) · mu0 · ∇p ) = ∇·u*       (eq. 'poisson' in docs)

On a composite AMR hierarchy this becomes, in **Martin–Colella
composite form** (`L_comp p = ∇·u*`):

- *Interior of a level patch:* the usual 5-/7-point variable-
  coefficient stencil with face coefficient
  `c_face = (w·dt/ρ) · harmonic_mean(mu0)` and spacing `h_ℓ`.
- *Coarse-fine face F separating level ℓ (fine, F⁻) from level ℓ-1
  (coarse, F⁺):* replace the coarse-side flux of the standard
  coarse-cell stencil by the **sum of fine-face fluxes** that tile
  the coarse face:

      flux_coarse_to_fine = (1/r^{d-1}) · Σ_{fine faces tiling F}
                                  c_face · (p_fine_+ − p_coarse) / (1.5·h_ℓ)

  i.e. a "ghost" fine pressure is reconstructed from the coarse
  pressure by quadratic interpolation (Martin–Colella eq. 2.13).
  This is the only stencil that differs from the existing
  `poisson_mult.py` operator.

- Solve with **MLMG** (multi-level multigrid):
  *V-cycle on each level* (the existing geometric multigrid in
  `poisson_mult.py` works unmodified for a single level), then
  *exchange* corrections across coarse-fine faces using the bilinear
  prolongation that the codebase already has
  (`csrc/cuda/multigrid_transfer.cu`, `prolongate_add_*d`).
  Wrap that into the standard "MLMG V-cycle":
  bottom-up restrict residual through patches *and* across levels;
  coarsest-level bottom solve = existing FFT (`poisson_fft.py`) on
  the level-0 covering rectangle; top-down prolongate the
  composite correction back.

  AMReX's MLMG is the most directly portable reference; the existing
  LilyTorch smoother (`rbgs_sweep_3d`, `jacobi_sweep_3d`) is reused
  unchanged on each patch as the level smoother.

### 6.4 Velocity correction and **refluxing**

After the projection, correct each face velocity by `−grad(p)·c_face`
on its own level. Then, at every coarse-fine face, replace the
**coarse face velocity** by the area-weighted average of the fine
face velocities that tile it:

      u_coarse_face = (1/r^{d-1}) · Σ u_fine_face

This is the *averaging-down* (synchronisation) step and is what makes
the composite divergence zero to machine precision on the coarse side
too. Identical idea for the advected scalar fluxes if any scalar
transport is added later.

### 6.5 Free surface, Carreau, Smagorinsky, …

These are all pure pointwise / stencil operations on a single grid in
the current code (`free_surface.py`, the carreau/smagorinsky blocks in
`solver.py`). They lift to AMR trivially: apply per patch with `h_ℓ`,
fill ghosts as in 6.1, no special treatment.

---

## 7. Coarse-fine interface bookkeeping

The single class that pays for the whole framework. Proposed minimal
descriptor (computed at each regrid, valid for many time steps):

```
CFInterface(level ℓ):
    direction          (0,1,2) and sign (±)
    fine_patch_id      index into patches[ℓ]
    fine_face_slab     2-D slab of face indices on the fine patch
    coarse_patch_id    index into patches[ℓ-1]
    coarse_face_idx    1 face per (r^{d-1}) fine faces
    ghost_fill_op      enum {linear, quadratic} for adv-diff ghosts
    reflux_register    pre-allocated tensor for refluxing
```

All of these are *index-only* tensors that live on GPU and never
change between regrids — every per-step kernel just gathers/scatters
through them. This is the same pattern as the per-body sparse output
buffer in `streaming_sdf*.cu`.

Three GPU primitives suffice (write them once, reuse for adv-diff
ghosts, refluxing, and projection-stencil modification):

1. **`amr_fill_ghosts_from_coarse(level)`** — for each ghost cell
   listed in the CF interface, gather one (1‑D) or four/eight
   (2‑D/3‑D) coarse values and write an interpolated value. Block size
   = #ghost cells per CF face slab; one CUDA block per slab; trivially
   parallel.
2. **`amr_average_down_face(level)`** — for each coarse face in the
   CF list, sum the `r^{d-1}` fine faces and divide. One thread per
   coarse face.
3. **`amr_reflux(level)`** — adds the mismatch
   `(flux_fine_sum − flux_coarse)` back into the coarse cell on the
   coarse side of the interface. One thread per coarse face,
   `atomicAdd` into the coarse RHS / coarse velocity (only one writer
   per coarse cell if the CF list is built right, so atomics are
   only needed if patches overlap in the index-space sense at corners).

All three are independent across patches and across the two CF sides
→ map cleanly onto `gridDim.y = #CF interfaces` launches.

---

## 8. Refinement criteria and regridding

When to refine a cell? Two criteria, OR-ed together:

1. **Geometric (BDIM correctness)** — refine every cell within a
   distance `k·eps_max = k·eps_multiplier·h_{ℓ_max}` of any body
   surface, with `k≈4`. Cheap: a single SDF table lookup per cell.
   This is the *mandatory* criterion.
2. **Flow-feature (efficiency)** — refine where the local *vorticity
   magnitude* or *Q-criterion* or *λ₂* exceeds a user-set threshold;
   alternatively a **Richardson-extrapolation error** estimate
   (compare a 1-step prediction on level ℓ versus level ℓ-1) à la
   Berger–Colella. Vorticity-based tagging is the cheapest and works
   well for swimmers (vortex rings dominate the wake).

How often to regrid? Every `N_regrid` steps, where `N_regrid` ≈
`block_size / (CFL · 2)` ≈ 16–32. The body moves at most a few cells
between regrids, and tagging includes a `n_grow` buffer of 2–4 cells.

Regridding algorithm (Berger–Rigoutsos 1991, the standard):

1. On each level top-down, *tag* cells failing either criterion.
2. *Buffer* tags by `n_grow` cells (dilation) so the refined region
   has a margin around features.
3. *Cluster* tagged cells into rectangular patches with the
   Berger–Rigoutsos signature-cut algorithm (≥ ~70% tag-density per
   patch). This is O(N_tagged) per level and runs on CPU once per
   regrid; the patches themselves are then allocated as GPU tensors.
4. *Proper nesting*: every fine patch must be at least 1 (typically
   2) coarse cells inside its parent. Enforce in the clustering step.
5. *Transfer state* from the old hierarchy to the new one: copy
   overlapping fine cells, fill newly refined cells by
   conservative prolongation from the coarse parent. The existing
   `prolongate_add_*d` kernels do exactly this; just wrap them with
   a "fill, don't add" variant.

Regrid cost is `O(N_total)` and ≪ one time step's compute, so it is
not on the GPU critical path.

---

## 9. Mapping to CUDA — parallelisation strategy

The repository already uses three GPU patterns that *every* AMR kernel
can reuse:

- **`gridDim.y` over independent units** (here: patches and CF
  interfaces) — see `streaming_sdf*.cu` multi-body launches.
- **Packed-key 64-bit `atomicMin`** for compare-and-swap with an id
  — `packed_key.cuh`. Use the same trick if two bodies share refined
  regions, and to break ties when several CF averages target the
  same coarse cell.
- **CUB block reduce** — already used in `streaming_sdf.cu`; reuse
  for per-patch divergence / vorticity reductions in the tagging
  pass.

Concrete kernel inventory we need to add (CUDA + CPU parity, mirroring
the existing dual-target pattern in `csrc/cpu/...` and `csrc/cuda/...`):

| Kernel                                | Launch dimensionality              | Notes |
|---------------------------------------|------------------------------------|-------|
| `amr_fill_ghosts_from_coarse_{2,3}d`  | (slab_pts, n_iface, 1)             | linear / quadratic |
| `amr_fill_ghosts_same_level_{2,3}d`   | (slab_pts, n_iface, 1)             | straight copy |
| `amr_average_down_face_{2,3}d`        | (coarse_faces, n_iface, 1)         | divide by r^{d-1} |
| `amr_reflux_{2,3}d`                   | (coarse_faces, n_iface, 1)         | atomicAdd |
| `amr_prolongate_fill_{2,3}d`          | (fine_cells, n_patch, 1)           | "fill" variant of existing add |
| `amr_tag_cells_{2,3}d`                | (cells, n_patch, 1)                | SDF + vorticity criterion |
| `amr_mlmg_residual_{2,3}d`            | (cells, n_patch_level, 1)          | extends existing `mg_residual_*d` with composite-stencil correction at CF faces |
| `amr_mlmg_smoother_{2,3}d`            | (cells, n_patch_level, 1)          | wraps existing rbgs/jacobi sweep |
| `amr_mlmg_correct_at_cf_{2,3}d`       | (slab_pts, n_iface, 1)             | quadratic ghost reconstruction for projection stencil |

Pure-PyTorch fallbacks (slicing / `F.interpolate`) should exist for
each, mirroring the existing CPU op definitions in
`csrc/streaming_sdf_cpu*.cpp` — required by the parity tests
(see repository memory `testing`).

### 9.1 Per-level fusion opportunities

- Fuse `streaming_sdf_min_*` + `bdim_vardens_*` per patch (the
  fused-2-D/3-D path you already have): this is a per-patch
  operation and lifts unchanged.
- Fuse `amr_average_down_face` with the velocity correction kernel
  on the coarse side to avoid one global read/write.
- Fuse `amr_fill_ghosts_from_coarse` for `u`, `v`, `w` into a single
  kernel taking the 3 face tensors as arrays (analogous to the
  3-channel face packing used by `streaming_sdf_stag_3d_multi`).

### 9.2 Streams and overlap

Per-level work on disjoint patches is embarrassingly parallel. Place
patches into a small number of CUDA streams (e.g. one per level) so
that ghost-fill from level ℓ-1 can overlap with the smoother on
level ℓ. The existing solver uses the default stream; introduce
`torch.cuda.Stream` per level only after the single-stream version is
correct.

### 9.3 Memory

Each level's storage = Σ (patch_volume + ghost_volume) × n_fields.
With block size 16³ and a 2-ghost layer the overhead is ≈+95 % per
patch; with block size 32³ it's ≈+45 %. Budget accordingly. Recycle
patch tensors across regrids by keeping a per-shape free-list (PyTorch
caching allocator does most of this already, but explicit `torch.empty`
reuse cuts allocator fragmentation in long runs).

---

## 10. Test strategy

Mirror the existing CPU/GPU parity tests
(`test_streaming_sdf_forces_fused_self.py`,
`test_streaming_sdf_forces_fused_2d_self.py`):

1. **Operator-level**: for a known analytical `p`, check that
   composite gradient on a manually-built 2-level patch hierarchy
   agrees with the uniform-grid gradient on the projected uniform
   grid to `O(h²)` (or to machine precision when the two coincide).
   Same for divergence, refluxing, average-down.
2. **Projection-level**: pick a divergence field with localised
   support, solve with MLMG on a 3-level hierarchy and on a uniform
   grid at the finest resolution; the two pressures should agree to
   the multigrid tolerance away from CF faces and to `O(h_{ℓ_max}²)`
   at them.
3. **End-to-end**: Taylor–Green vortex on a 2-level hierarchy with
   the refined patch covering one vortex; compare the energy decay
   curve to the uniform-fine reference. Acceptable error < 1 %
   relative at `t = 2π/ω`.
4. **Coupled**: a stationary sphere in uniform flow at `Re=200`;
   integrated drag coefficient must match the uniform-fine reference
   to < 2 %.
5. **Moving body**: a 1-link 1guilla animat with body-tracking
   refinement; check that the wake structure (vorticity isosurfaces)
   matches the uniform-fine simulation to eye, and that the per-step
   wallclock improves by the expected factor.

Each test goes in `lilytorch/src/kernels/test_amr_*.py` and runs at
`fp64` for parity.

---

## 11. Recommended implementation order (milestones)

A new agent should tackle this in the order below; each milestone is
independently shippable and useful.

**M0 — Plumbing.** Introduce `Level`, `Patch`, `CFInterface` Python
dataclasses in a new `lilytorch/src/amr.py`. Add a "uniform AMR" mode
that is a single level with one patch covering the whole domain. All
existing tests must still pass when this mode is selected. *No new
kernels.*

**M1 — Static 2-level nested grid.** Allow the YAML config to declare
a single refined sub-box (e.g. tracking the bounding box of all bodies
+ a wake margin). Implement same-level + coarse-to-fine ghost fills
(linear), average-down, and a *naive* MLMG that does N V-cycles per
level then one extra V-cycle on the composite for synchronisation.
First measurable speed-up on a quiescent-tank fish simulation.

**M2 — BDIM lift.** Make `streaming_sdf_*_multi` patch-aware (one
extra `gridDim.y` over patches). Validate that drag on a static sphere
at Re=200 matches the uniform-fine reference within tolerance.

**M3 — Refluxing + quadratic CF projection stencil.** This is the
Martin–Colella step; required for true second-order accuracy at CF
faces. Validate with the Taylor–Green test (M3-quality means the
energy-decay curve overlays the uniform-fine reference).

**M4 — Dynamic regridding.** Berger–Rigoutsos clustering + vorticity
tagging + state transfer. Trigger every 16–32 steps. Validate on a
swimming 1guilla.

**M5 — Multi-level (≥3 levels).** MLMG cleanup, proper nesting
enforcement, performance tuning (block size sweep, stream overlap).

**M6 — Production polish.** YAML config schema for AMR parameters
(block size, max levels, n_grow, regrid frequency, vorticity
threshold), HDF5 logging per-level statistics, visualisation hooks
(the existing `flow_viewer_*` integrations need a "downsample fine
levels to coarse for display" mode).

The first three milestones are by far the most labour-intensive
(~70 % of the total effort) because they introduce all the new data
structures; M4–M6 are mostly incremental.

---

## 12. Risks and open questions

- **BDIM-σ + CF interfaces.** The σ-shift relies on per-body global
  `phi_min` (repository memory `BDIM-σ correction`). On AMR the
  body's surface may straddle a CF face. *Mitigation:* the geometric
  refinement criterion (Section 8.1) already forces the entire
  `|φ| < k·eps_max` neighbourhood onto the finest level, so σ-shift
  becomes a single-level operation. Add an assertion at regrid time.
- **Multibody Lagrangian forces sampling across patches.** Today,
  triangle centroids sample `p, eps_ij` at `centroid + sample_offset
  * normal` (repository memory `lagrangian forces`). With AMR this
  sample point may land in a different patch from the centroid.
  *Mitigation:* the existing geometric tagging guarantees the whole
  body neighbourhood is on the finest level; the sampler then just
  reads from the finest level — but explicitly assert this in the
  Lagrangian-force kernel (and unit-test it).
- **2-D fused lagged normals.** The fix in repository memory
  `2-D fused lagged normals` (seed `fs.normal_x/y` from
  `comp.sdf_val` *before* the `_FAR` reset) must be applied per
  patch. Easy, but worth a regression test.
- **Free surface.** Free-surface markers are stored as separate
  Lagrangian particles in `free_surface.py`; their advection only
  reads cell-centred fields and so lifts unchanged — but moving a
  marker across a CF face means its interpolation stencil straddles
  two levels and must fall back to the coarser of the two (worst
  case: one tangential averaging step). Defer to a post-M5 polish
  pass.
- **FFT bottom solve.** `PoissonSolverFFT` only works on a uniform
  rectangle. Use it as the *coarsest-level* solver in MLMG (always a
  single covering patch by construction), not on intermediate levels.
- **Determinism.** Refluxing uses atomic adds; the CF-list builder
  should arrange that each coarse cell has at most one writer
  (geometrically true if CF faces are non-overlapping). Make this an
  invariant of `CFInterface` construction and avoid atomics
  entirely.
- **MuJoCo `dt` fixing.** Verify in `BDIMhandler.__init__` that the
  fluid `dt` declared in YAML is exactly the MuJoCo `physics.timestep`;
  the current code asserts equality of `dx` and `dy` only. Add an
  assertion `dt_fluid == dt_mujoco` when AMR is enabled (it is a
  silent correctness requirement of "no subcycling").

---

## 13. Out-of-scope (for now)

- Temporal AMR / subcycling between levels (incompatible with the
  fixed-`dt` MuJoCo coupling requirement).
- Non-power-of-two refinement ratios (would break the FFT bottom
  solver and the existing `prolongate_add_*d` kernels).
- Anisotropic refinement (refining only in `x` etc.).
- AMR-aware load balancing across multiple GPUs.

These can all be revisited once the synchronous 2× AMR is in production.

---

## 14. TL;DR for the next agent

1. Read Wiley FLD `fld.5283`, Martin–Colella 2000, and the AMReX MLMG
   paper. They cover ~95 % of what is needed.
2. Add `lilytorch/src/amr.py` with `Level`/`Patch`/`CFInterface`
   dataclasses and a "single-level AMR" mode that wraps the existing
   solver with zero behaviour change.
3. Implement the 9 CUDA kernels listed in Section 9 with CPU parity
   stubs, following the multi-body kernel template
   (`streaming_sdf*.cu`, `packed_key.cuh`).
4. Keep `dt` global and fixed; enforce CFL on `h_{ℓ_max}` only.
5. Refine *geometrically* around bodies (mandatory for BDIM
   correctness) and *featurally* on vorticity (for efficiency).
6. Validate via the test ladder in Section 10 before moving the next
   milestone.
