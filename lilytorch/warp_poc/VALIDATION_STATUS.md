# Warp viability — validation status & testing plan

Scope of Part 2 (to_do_list AP6/AP7/AP8): can Warp single-source kernels replace
the hand-written `.cpp`/`.cu` pairs **and** the speed/memory-tuned "kernel mode",
collapsing everything into one codebase?

This file tracks what the POC has actually validated vs. what remains.

---

## The full per-step kernel-mode pipeline (3-D; 2-D mirrors it)

From `solver.py: _fluid_step_kernel_3d`, in order:

| # | Stage            | Native op(s)                                            | Kind                         |
|---|------------------|---------------------------------------------------------|------------------------------|
| 1 | Advection-diff   | `advect_flux_add` (+ diffusion)                         | fusible stencil/pointwise    |
| 2 | **Kernel A**: staggered SDF + body vel | `streaming_sdf_stag_3d_multi`          | irregular scatter/gather     |
| 3 | **Kernel B**: BDIM coeff + FD **normals** (registers) | `bdim_coeff_3d`, `bdim_coeff_sigma_3d` | gather + stencil (register-fused) |
| 4 | Pressure projection / **Poisson** | `poisson_solve_mgcg_3d` (or `multigrid`/`rmgcg`) → `rbgs_sweep`, `jacobi_sweep`, `mg_residual`, `restrict_residual`, `restrict_face`, `prolongate_add` | driver + fusible stencils |
| 5 | **Forces**       | `streaming_sdf_forces_post_3d` (Eulerian) **or** `lagrangian_forces_3d` | scatter (atomicAdd)          |
| 6 | Boundary cond.   | `apply_bcs_3d`                                          | pointwise                    |
| – | Two-phase VOF    | `cvof_sweep`                                            | directional-split stencil    |
| – | Interpolation    | `interp_2d/3d`                                          | gather                       |

Strategy split (AP6/AP7):
- **Warp targets** (irregular scatter/gather): **Kernel A**, **lagrangian_forces**.
- **torch.compile/Inductor targets** (fusible stencil/pointwise): advection,
  rbgs, jacobi, residual, restrict, prolongate, apply_bcs, cvof, interp — and
  arguably **Kernel B**.
- **Stays `.cu` / CUDA-graph** (driver-style control flow): the Poisson
  `mgcg/multigrid/rmgcg` outer loops.

---

## ✅ DONE — validated in this POC (`warp_poc/`)  — 147/147 pytest pass

### Native-op COVERAGE (audit)
Every native `lilytorch_kernels` op now has a Warp equivalent **except the Eulerian force
readout**:
- **Ported:** streaming_sdf_stag (A) 2D+3D · bdim_coeff(+σ) (B) 2D+3D · lagrangian_forces 2D+3D ·
  rbgs_sweep & jacobi_sweep 2D+3D (thread-per-cell **and** fused-tiled) · mg_residual /
  restrict_residual / restrict_face / prolongate_add 2D+3D · advect_flux_add · cvof_sweep ·
  interp 2D+3D · apply_bcs 2D+3D.  Poisson DRIVERS (multigrid/mgcg/rmgcg) = composition shown via
  WarpVCycle 2D+3D (kernel-level Warp + Python control flow).
- **NOT ported (1 kernel family):** `streaming_sdf_forces_post_2d/3d` — the Eulerian n·δ viscous+
  pressure force integration (+ deltaH ∂H second pass).  The *lagrangian* force readout IS ported;
  this Eulerian one (block-reduction + atomicAdd scatter + the force discretization) is the sole
  remaining op.  Tractable (same Warp-class scatter pattern as lagrangian_forces) — open.

### Fused TILED smoother — 2D **and 3D** (Jacobi + RBGS)
`warp_poisson_tiled_2d.py` / `warp_poisson_tiled_3d.py` (+ `test_poisson_tiled_{2,3}d.py`):
- **2D** (native smoothers ARE shared-mem-fused): tiled Jacobi **~1.1×** of native, tiled RBGS
  **~1.75–2×** (masked two-pass: the cooperative model updates all cells then masks by colour,
  vs native doing only the active colour).  ns=1 matches native to f32 ULP; multi-sweep converges.
- **3D** (native smoothers are thread-per-cell, NOT fused): tiled Jacobi/RBGS **BEAT native** —
  Jacobi 0.24× and RBGS 0.32–0.33× at 128³ (3–4× faster), 1.1× at 64³ (launch-bound).  Single-tile
  matches the global red-black/Jacobi reference to f32 ULP; converges.
  **CAVEAT (do not over-read the 3-D RBGS win):** the 3-D win is **FUSION** against an unfused native
  baseline, NOT the tile model handling red-black colour better than in 2-D.  Even Warp **Jacobi**
  wins 3-D ~4× (pure fusion, no colour).  The RBGS-vs-Jacobi colour-masking penalty is the SAME in
  both dims (3-D 0.32/0.24 = **1.33×**, 2-D ~1.5–1.8×).  A *fused* native 3-D RBGS would likely beat
  Warp 3-D RBGS too — see the RBGS detail + compressed-storage note below and HANDOFF lesson 26.
⇒ The smoother is NOT structurally limited in Warp 1.14 — shared-mem fusion via the tile API
reaches parity (2D) or wins (3D, where native isn't fused).  The one thing the tile model CANNOT
match is a fused **colour-selective** RBGS (native 2-D); tiled **Jacobi** (no colour redundancy)
is the smoother that reaches native parity and is the right MG smoother choice in both dims.

---

## (legacy) ✅ DONE — validated in this POC (`warp_poc/`)  — 127/127 pytest pass

**2-D variants + remaining stencil/gather kernels + a 2-D V-cycle (this round)**
— see "(7) 2-D parity round" below; the §A–F boxes are updated in place.

**(1) Kernel A — `streaming_sdf_stag_3d_multi`** (irregular scatter/gather, AP7):
- Two Warp designs: sequential-per-body, and fanned all-body (2 launches, const in B).
- **Numerical parity vs native** (`test_parity.py`, 14/14): B=1/3/9, eager + graph +
  fanned, 64×32×32 + 128×64×32; SDF rel < 5e-4 (abs < 2 ULP at overlap), body-vel rel < 1e-4.
- **Performance vs native** (`bench_viability.py`, RTX 4080 Super): fanned+graph
  0.59–1.16× of native across 64³→384³, all B; advantage grows with grid.
- **Single-source CPU+GPU confirmed**: the *same* `@wp.kernel` runs on Warp `device="cpu"`
  (→ C++/OpenMP) and `"cuda:0"`; CPU vs GPU agree to ~1e-8 (float32 reduction noise).
- **CUDA-graph capture** removes the per-launch Python overhead (eager is 3–14× slower).

**(2) Poisson RBGS smoother — `rbgs_sweep_3d`** (fusible stencil; the plan slated this for
torch.compile — this POC shows Warp does it too, *faster*) (`warp_poisson.py`, `test_poisson.py`):
- Faithful port of the native variable-coefficient 7-point red-black GS. **Parity vs native**
  to 1.3e-7 (1 sweep, interior); residual converges (215→0.4 over 100 sweeps); **CPU == GPU**.
- **Perf** (`bench_poisson.py`, 10 sweeps, graph): **Warp BEATS native at every grid —
  0.51× / 0.80× / 0.98× / 0.95× at 32³ / 64³ / 96³ / 128³.**
- **The ~1.5× gap was closed (and reversed) by two opts**, NOT `wp.tile`:
  1. **Flat 1-D array addressing** (precomputed base + ±stride offsets) instead of 3-D
     `wp.array` indexing (which recomputes strides per access) → ~1.5×→1.0–1.2×.
  2. **Fold homogeneous Neumann into the half-sweep via index clamp** (ghost = self at the
     boundary) → eliminates ALL BC kernel launches (native pays 2/sweep) → 1.0–1.2×→0.5–1.0×.
  Both stay bit-parity with native.  `wp.tile` shared-mem would be the *next* lever but is
  no longer needed to match/beat the (untiled) native 3-D rbgs.

**(3) Advection — upwind flux-divergence + HIGH-ORDER LIMITER** (fusible stencil; faithful
replica of `advect_flux_add`) (`warp_advection.py`, `test_advection.py`, `bench_advection.py`):
- **First-order upwind**: single fused `@wp.kernel`; parity vs a PyTorch reference 1.4e-7, CPU+GPU.
- **High-order limiter `advect_flux_add_warp`** — faithful float64 port of the native fused
  `advect_flux_add` CUDA op: per-(component i, direction d), two adjacent face fluxes accumulated
  in place `rhs[i_fd] += dt_dh·(F_L − F_R)`, all five schemes (QUICK / ABDQUICKEST / vanLeer / CDS
  / CUBISTA — `median3` + the exact `rf`/`psi` limiter algebra), lo/hi-boundary CDS fallbacks.
  Flat 1-D storage views (built from `data_ptr` honouring `storage_offset`, zero-copy) + explicit
  element strides mirror the native pointer arithmetic; **the rhs strides are passed separately**
  (face_dim ≠ outermost in the original-dim-order rhs) — the documented caveat, exercised by every
  d>0 case.  One kernel serves 2-D (Nt2=1, s_t2=0) and 3-D, like the `.cu`.
- **Parity vs native CUDA = 0.0 (bit-exact)** over {2-D,48²; 3-D,24³} × {5 schemes} × every (i,d)
  pair, built from the genuine strided slice views of `advection.py`; plus an in-place-accumulation
  (pre-seeded rhs) check.  Native op is CUDA-only → CPU validated by **Warp CPU == Warp GPU** (<1e-12),
  all schemes 2-D+3-D (HANDOFF lesson 10).
- **Perf** (`bench_advection.py`, full momentum step = ndim² launches, graph): **2-D Warp BEATS
  native 0.05–0.38×** (native pays per-op torch-dispatch overhead the graph removes).  **3-D: at/
  below native after COMPILE-TIME scheme specialization** — QUICK 0.98–1.03×, ABDQUICKEST
  **0.58–0.62× (Warp ~1.6× faster)**.  The first cut ran a runtime `if scheme_id==…` branch (all
  five scheme paths live in one kernel) → 1.3–1.4× of native; replacing it with one kernel per
  scheme via a factory that closes over the scheme `@wp.func` (`_make_flux_kernel`, the Warp
  analogue of native's `template<int scheme_id>`) closed the gap, bit-exact (HANDOFF lesson 17).
  Launch map (flat-1-D vs 3-D-tid) was confirmed irrelevant (graph≈eager → not launch-bound).
- **MEMORY** (`bench_advection_memory.py`, full momentum step, flux views precomputed as baseline):
  native and **warp both add 0.0 MiB** above baseline (torch *and* driver-level), vs the python
  `_flux`→`cat`→`add_` path's **27.7 / 65.9 / 131.9 MiB** (torch; 42 / 86 / 150 driver) at 96³ /
  128³ / 160³ — the intermediate `F` + B1/B2/flux_* temporaries (~grid each) the fused kernel
  removes (exactly the `.cu` header's rationale).  All three agree to ~4e-17.  ⇒ Warp accumulates
  `rhs += dt·(F_L−F_R)` in registers, no global spill — same register-residency story as Kernel B.

**(4) Lagrangian forces — `lagrangian_forces_{2d,3d}`** (the 2nd Warp-class / atomicAdd-scatter
kernel, AP7) (`warp_lagrangian.py`, `test_lagrangian.py`, `bench_lagrangian.py`):
- Faithful float64 port: clamped tri/biquadratic field sample a `sample_offset` along the
  outward normal, traction `nu*rho*(eps.n) - p n`, per-element area (3-D) / trapezoid contour
  (2-D) weight, `wp.atomic_add` scatter into the per-body (B,12)/(B,6) row.
- **Parity vs native ≤1.7e-16** (machine precision) over the full matrix {2D,3D}×{linear,
  quadratic}×{scalar,field nu_rho}×{offset 0,≠0}, CPU **and** GPU; **CPU==GPU** to 1e-16.
- **Perf** (graph): 0.91×→1.26× of native; the >1× tail is pure atomicAdd contention at
  B=2/huge-T (same double-atomic contention native pays), which dilutes with more bodies.
- Both Warp-class kernels (Kernel A + lagrangian_forces) now validated for correctness,
  single-source CPU+GPU, and competitive perf.

**(5) Kernel B — `bdim_coeff_3d` + FD normals** (gather + register-fused stencil; the
MEMORY-critical kernel) (`warp_bdim.py`, `test_bdim.py`, `bench_bdim_memory.py`):
- Faithful float64 port of `bdim_one_axis_3d` (+ the σ variant `bdim_one_axis_sigma_3d`):
  smoothed Heaviside/delta `mu0,mu1`, FD unit normals `n=∇φ/|∇φ|` (clamped edges), BDIM2
  normal derivative, persistent `u0/v0/w0` + Weymouth-&-Yue Poisson coeff `(mu0_proj? dt·mu0:dt)/ρ`.
- **Parity vs native CUDA = 0.0 (bit-exact)** over {mu0_proj 0/1} × {full grid, interior
  AABB sub-block} × {plain, BDIM-σ shifted-coeff}.  CPU: velocity fields `u0/v0/w0` bit-exact
  vs native CPU (the native CPU op writes `c_out` at the *padded-grid* index — a different
  `ch/cv/cw` layout from CUDA — so coefficient parity is taken vs CUDA, the production path).
  **Warp CPU == Warp GPU** to 4.4e-16.
- **MEMORY (the load-bearing claim — `bench_bdim_memory.py`, RTX 4080S, float64):** peak GPU
  memory *above baseline* for the SAME computation —
  | grid | python path | native | **warp** |
  |------|-------------|--------|----------|
  | 96³  | 128 MiB     | 0.0    | **0.0**  |
  | 128³ | 304 MiB     | 0.0    | **0.0**  |
  | 160³ | 608 MiB     | 0.0    | **0.0**  |
  Measured both by `torch.cuda.max_memory_allocated` AND driver-level `torch.cuda.mem_get_info`
  (which also sees Warp's separate allocator) — **both 0.0 for Warp**.  All three paths agree to
  8.9e-16.  ⇒ **Warp keeps mu0/mu1/normals register-resident exactly like native; it does NOT
  spill to globals.** The python path's full-grid mu/normal intermediates (what kernel mode
  removes, cf. the 5.749 GiB `_recompute_mu_normals` peak) scale with the grid; Warp/native don't.
- **Perf** (`bench_bdim.py`, graph): **0.96–1.07× of native** at 96³–160³ (faster than native at
  128³).  The first cut was 1.12–1.19×; switching from a 3-D `wp.tid()` launch to a **flat 1-D
  launch with native's exact decode** (`dk = local % dAk` → k fastest → coalesced stores) closed
  it to ~1.05× — flat addressing (lesson 2) applies to the LAUNCH MAP too, not just array reads.
  Single fused launch, so graph ≈ eager.  Tiling would be the next lever but isn't needed.

**(6) Multigrid transfer ops + a full Warp V-cycle** (fusible stencils, AP6)
(`warp_multigrid.py`, `test_multigrid.py`):
- Faithful float64 ports of `jacobi_3d` (ping-pong weighted Jacobi), `mg_residual_3d`
  (register-J residual, reads padded-p ghosts directly), `restrict_residual_3d` (sum-of-8),
  `restrict_face_3d` (0.5·sum WaterLily, face_dim 0/1/2), `prolongate_add_3d` (trilinear
  align_corners=False, native double-precision `linear_weights`).
- **Parity vs native CUDA = 0.0 (bit-exact)** for residual/restrict-residual/restrict-face/
  prolongate; Jacobi interior < 1e-13 (atomic-free, the ε is just clamp-fold-Neumann vs explicit
  ghost on boundary cells).  Native mg/transfer ops are **CUDA-only**, so CPU is validated by
  **Warp CPU == Warp GPU** (< 1e-12).
- **Assembled `WarpVCycle`** (smoother = Warp RBGS-f64 + the 4 transfer ops, built like
  `poisson_mult._vcycle_rbgs_3d`: face-coeff slices → `cp0..cm2`, coarse op via `restrict_face`,
  graph-capturable) **converges geometrically** on a manufactured Neumann-Laplacian Poisson:
  32³, 4 levels, residual 1.8e2 → 4.9e-10 in 10 cycles, **~0.07 contraction factor/cycle**.
- **Perf** (`bench_multigrid.py`, 128³, graph): every op **0.98–1.04× of native** —
  mg_residual 1.01×, jacobi×2 0.98×, restrict_residual 1.03×, restrict_face 1.04×,
  prolongate_add 1.01×.  (Eager is 3–4× on the *tiny* restriction kernels — pure per-launch
  overhead — which graph capture removes; lesson 3.)  WarpVCycle 2.75 ms/cycle (6 levels).
  **Memory matches native**: mg_residual 16.0 MiB = native (register-J, no global J/active);
  jacobi allocates one ping-pong buffer like native (via Warp's pool; same physical size).

**(7) 2-D parity round — Kernel A/B, smoother, transfer ops, interp, apply_bcs, cvof**
(`warp_kernels_2d.py`, `warp_bdim_2d.py`, `warp_poisson_2d.py`, `warp_misc_2d.py`,
`warp_cvof.py`, `warp_multigrid_2d.py` + `test_*_2d.py` / `test_cvof.py`; `scene_2d.py`):
- **Kernel A 2-D** (`streaming_sdf_stag_2d_multi`): fanned + sequential designs, parity vs
  native (SDF rel<5e-4, body-vel rel<1e-4) for B=1/3/5, eager+graph; **also covers the two
  §B gaps** — `interp_method=1` (biquadratic) and the **blend-eps softmin body-velocity path**
  (num/den accumulators, overlapping AABBs) — both parity-clean.  CPU==GPU.
- **Kernel B 2-D** (`bdim_coeff_2d` + σ): parity vs native **bit-exact (0.0)** over {mu0_proj
  0/1}×{full,sub-block}×{plain,σ}.  NOTE: unlike 3-D, the 2-D native writes the Poisson coeff
  at the **full-grid** index `c_out[g]` in BOTH the CUDA and CPU ops → **no CPU/CUDA layout
  discrepancy** (HANDOFF lesson 9 is 3-D-only), so coeff parity holds vs CPU *and* CUDA.  Warp
  CPU==GPU < 1e-12.
- **RBGS/Jacobi 2-D smoother**: the native 2-D smoothers are **block-tiled (8×32)** and fuse
  all sweeps with stale inter-tile halos for sweeps 2..n (multigrid block-boundary approx),
  unlike the thread-per-cell 3-D ones.  Consequences (validated): **Jacobi nsmoothing=1 is
  bit-exact on any grid**; **RBGS nsmoothing=1 is bit-exact single-tile** (Nx≤8,Ny≤32) and on
  multi-tile grids differs ONLY at the 8×32 seams (every red cell + every off-seam black cell
  bit-exact — `test_rbgs_multiblock_diff_only_at_tile_seams`).  Global Warp RBGS converges; CPU==GPU.
- **interp_2d** (gather): parity vs native — **CPU bit-exact**, GPU ~1 float32 ULP (FMA-order),
  linear+quadratic.  **apply_bcs_2d** (Neumann/Dirichlet/reflective ghost writes): **bit-exact
  CPU+GPU** (disjoint stage-1 ops; overlapping corners are order-undefined per native's own note).
- **cvof_sweep** (two-phase W&Y, `warp_cvof.py`): **bit-exact vs native CUDA (0.0)** over
  {2-D,3-D}×every face_dim×{contiguous, strided velocity row view} via the flat-pointer+stride
  path (lesson 14); Warp CPU==GPU.  Native is CUDA-only.  Confined to the two-phase path.
- **2-D multigrid transfer ops + V-cycle** (`warp_multigrid_2d.py`): `mg_residual_2d`,
  `restrict_residual_2d` (sum-of-4, native add order → bit-exact), `restrict_face_2d`,
  `prolongate_add_2d` (1-ULP FMA on the bilinear sum) — parity vs native CUDA; assembled into
  `WarpVCycle2D` (RBGS-f64 + the 4 ops, like `poisson_mult._vcycle_rbgs_2d`) that **converges
  geometrically** (residual → <1e-6·r0 in 10 cycles).  Driver stays Python/CUDA-graph; Warp is
  kernel-level only → the **Poisson outer-driver composition** (§E) is demonstrated in 2-D.

**PERF + MEMORY (2-D round, `bench_2d.py`, RTX 4080S, CUDA-graph, ratio vs native; <1=Warp faster):**
| kernel | 512² | 1024² | note |
|--------|------|-------|------|
| Kernel A 2-D | 0.57× | 0.55× | **Warp faster** — native pays per-launch dispatch the graph amortises (cf. 2-D advection) |
| Kernel B 2-D | 1.20× | 1.13× | ~native (matches 3-D 0.96–1.07×); single fused launch |
| RBGS smoother | see ↓ | see ↓ | the gap is ENTIRELY native's shared-mem multi-sweep FUSION — see the dedicated table |
| cvof 2-D | **= native-3D** | **= native-3D** | Warp-2D **4–5 Mcells/ms ≈ native cvof's real (3-D) throughput 3.8 Mcells/ms**.  The apparent "25× vs native-2D" is native-2D being **~26× under-occupied** (32-wide block × transverse size 1 → ~97% idle threads); Warp hits native's genuine per-cell efficiency.  NOT a like-for-like Warp speedup — a native-2D-cvof inefficiency surfaced |

**RBGS smoother detail** — the native advantage is shared-memory multi-sweep fusion.  The simple
thread-per-cell Warp port (color-compacted flat launch, ~10% over the naive 2-D launch) **matches
native UNFUSED (0.79–1.11×)** but trails native-FUSED by the fusion factor (1.6× at the realistic
V-cycle nu=2, up to 4.2× at nu=10).  The FUSED tiled Warp RBGS (`WarpTiledRBGS2D`) closes most of
that; the residual gap is native's **color-selective** half-sweeps (it updates only the active
colour; the cooperative kernel evaluates the stencil on all cells then masks).

**RBGS op-fusion (HANDOFF lesson 27) — SHIPPED, the penalty was mostly OVERHEAD not colour FLOPs.**
The original masked kernel built each half-sweep from ~6–9 chained binary `tile_map` ops (each a
cooperative pass + barrier + intermediate whole-tile temp).  Collapsing to **`tile_map(_gs8,…)` (raw
5-pt sum, 8 tile args) + `tile_map(_sel_red/_blk,…)` (fused divide + colour select) + assign** = 3
ops/half-sweep (enabled by `tile_map` actually supporting ≤8 tile args, not ≤3).  Measured
(kernel-only, RTX 4080S): **2-D fused vs old masked 0.74–0.95× (10–26% faster)**; vs tiled Jacobi
**1.14–1.38× at ns=2** (the realistic V-cycle smoothing — down from 1.5–1.9×), ≈ Jacobi parity; the
~2× at ns=10 is the irreducible compute-saturated colour redundancy.  **3-D fused 1.15× Jacobi at
128³/ns=2** (was ~1.33×), still ~3× over unfused native.  Bit-faithful (`(s-ff)/Jt` matches native
op order) → parity tests stay green (ns=1 ULP + convergence).
**KNOWN RESIDUAL (to improve later — see HANDOFF "Next work" item 7):** 2-D RBGS is still
**~1.5–1.6× native at high nsmoothing** (1024²/ns=10: native 79 µs vs fused 122 µs).  This is the
irreducible whole-tile colour redundancy (both half-sweeps evaluate the stencil on all cells); the
Warp tile model can't do native's per-thread colour-selective update.  Accepted for now — tiled
Jacobi (~1.1× native) is the recommended fast smoother; not a blocker for the end-to-end swap-in.

**RBGS 2-D — red/black-COMPRESSED storage: TRIED, NEGATIVE RESULT (keep the masked kernel).**
`experiment_rbgs_compressed_2d.py` (brief `TASK_rbgs_compressed_2d.md`) tested compact `(TILE,TILE/2)`
red/black tiles so each half-sweep is a dense N/2 update (native's work profile).  It is **correct**
(bit-close to numpy red-black, f32 ULP) and fully cooperative, but **not faster** — 1024², ns=10:
native 75–80 µs, masked **1.71×**, compressed-kernel-only **1.88×** (+ pack/unpack → 2.2×).  Cause:
Warp 1.14 `tile_view` is **affine-only** (no stride / no per-row offset), and a checkerboard's
left/right neighbours **shear by row parity** → must be assembled as a 2-view parity-mask blend that
costs ~what the compression saves (~15N vs ~16N per sweep).  Native sidesteps this with per-thread
**scalar** shared indexing, which the whole-tile model can't express.  See HANDOFF lesson 26.
⇒ **The practical fast smoother (both dims) is tiled JACOBI** (~1.1× in 2-D, ~0.24× in 3-D; no
red-black dependency → no shear → no colour redundancy).  The colour-masking penalty (~1.3–1.8×)
applies to RBGS in **both** dims; the 3-D RBGS "win" is a fusion win over an unfused native baseline,
not the tile model solving the colour problem in 3-D (a fused native 3-D RBGS would expose the same
gap).  Net: the compressed approach can't help in 2-D OR 3-D — use tiled Jacobi.

**⚠️ CORRECTION (was previously reported as a hard block — it is NOT):** Warp 1.14 **does** ship
the full tile / shared-memory API (`tile_load`/`tile_store`/`tile_view`/`tile_map`,
`storage="shared"`, cooperative ops with implicit barriers — the earlier "absent" finding was an
`hasattr` mistake: they're kernel builtins, not module attrs; and 1.14 IS the latest PyPI release).
A **fused TILED Jacobi** (`warp_poisson_tiled_2d.py`: per block load p-halo+coeffs+f to shared
once, run N sweeps in shared via `tile_view` neighbour shifts + `tile_map`, store once — the native
trick) **CLOSES the gap**:
| | native-fused jacobi | **warp TILED-fused** | ratio |
|--|------|------|------|
| 512², nsmoothing=10 | 0.016ms | 0.020ms | **1.23×** |
| 1024², nsmoothing=10 | 0.046ms | 0.050ms | **1.09×** |
⇒ from 3–4× (untiled) down to **~1.1× of native**.  Parity: at nsmoothing=1 it matches native
`jacobi_sweep_2d` to float32 ULP; multi-sweep converges (`test_poisson_tiled_2d.py`).  Caveats:
Jacobi (the cooperative tile model expresses it cleanly; RBGS red-black needs masking — follow-up),
`nsmoothing` compile-time (kernel factory), grid a multiple of TILE=32 (coarse-MG partial tiles =
follow-up).  Net: the smoother is NOT structurally limited in Warp 1.14 — tiling reaches ~parity.

**Peak GPU memory above the pre-allocated outputs (torch / driver MiB) — the register-residency claim:**
| | 256² | 512² | 1024² |
|--|------|------|-------|
| Kernel B 2-D: native | 0/0 | 0/0 | 0/0 |
| Kernel B 2-D: **warp** | **0/0** | **0/0** | **0/0** |
| Kernel B 2-D: python path | 8/8 | 32/42 | 128/182 |
| cvof: warp & native | 0/0 | 0/0 | 0/0 |
| cvof: python path | 6/8 | 24/22 | 97/102 |

⇒ Warp keeps mu0/mu1/normals (Kernel B) and the ~8 W&Y temporaries (cvof) **register-resident**
(0 MiB extra, torch AND driver), identical to native, vs the PyTorch path's grid-scaling
intermediates — the kernel-mode memory story reproduced in 2-D.

**Takeaway on the AP6/AP7 split:** the stencil kernels (Poisson smoother, advection) are NOT
blocked from Warp — they work, are single-source, and parity-clean.  The only reason the plan
leaned toward torch.compile for them is **performance vs hand-tiled native** (~1.5× gap), which
is closeable with `wp.tile` shared-memory tiling.  So "everything in Warp" is feasible; the
choice is Warp-everywhere (one tool, needs tiling for stencils) vs Warp-for-scatter +
Inductor-for-stencils (Inductor auto-tiles).

---

## ❌ NOT YET TESTED — required before declaring kernel-mode replaceable

### A. Other Warp-class kernel (AP7)
- [x] **`lagrangian_forces_3d` / `_2d`** in Warp — atomicAdd scatter to triangle
      (3-D) / contour (2-D) markers. Simpler than Kernel A (no argmin).
      `warp_lagrangian.py` + `test_lagrangian.py` (21 tests) + `bench_lagrangian.py`.
      **Parity vs native** (float64) to **≤1.7e-16** — machine precision, well below
      the native self-test's 1e-12 — across {2-D,3-D} × {linear,quadratic} ×
      {scalar,field nu_rho} × {sample_offset 0, nonzero}, on **CPU and GPU**.
      **CPU == GPU** (single source) to 1e-16. **Perf** (RTX 4080S, graph): 0.91× @
      T=4.8k → 1.16–1.26× @ T=20k–80k; the >1× tail is atomicAdd contention with
      only B=2 bodies (12 slots each) — identical double-atomicAdd contention as
      native; spreads with more bodies. Faithful port of the `.cu` discretization
      (clamped tri/biquadratic sample along outward normal, trapezoid contour
      weight in 2-D / area weight in 3-D).

### B. Kernel A — coverage gaps in what's "done"
- [x] **2-D variant** (`streaming_sdf_stag_2d_multi`) — **DONE** (`warp_kernels_2d.py`,
      `test_kernelA_2d.py`; fanned+sequential, parity vs native, CPU==GPU).
- [x] **blend-eps softmin path** (`num_*`/`den_*`, `body_velocity_blend_eps_cells`) —
      **DONE in 2-D** (num/den accumulators + softmin decode, parity vs native 2-D on
      overlapping AABBs, `test_kernelA_2d_blend`).  3-D Kernel A still hardcodes blend off;
      the capability is proven (the 2-D path is the native algorithm with z stripped).
- [x] **interp_method ≠ 0** — **DONE in 2-D** (biquadratic, `test_kernelA_2d_quadratic`;
      and the standalone `interp_2d` gather, both modes).  3-D Kernel A still hardcodes trilinear.
- [ ] **AABB-sized output buffers** (MP8/T2b) — POC writes full-grid; the memory win
      needs AABB+halo scratch.  (Kernel B's register-residency memory win IS proven, DONE (5);
      this is the separate transient-scratch sizing for Kernel A's union-SDF outputs.)
- [ ] **Real coupled scene** — parity is vs synthetic spheres/discs only, not a FARMS/MuJoCo
      trajectory (rotations, real link SDF tables, dirty-AABB churn).  Tied to §F end-to-end.
- [x] **MEMORY measurement** — **DONE** (`bench_bdim_memory.py`, on Kernel B; see DONE (5)).
      Warp peak GPU memory above baseline = **0.0** (torch *and* driver-level), identical to
      native, vs 128–608 MiB for the python path at 96³–160³.  Register-resident confirmed.
      (Kernel-A AABB-scratch transient remains a separate, untested item — see above.)

### C. Kernel B — BDIM coefficient + normals (the big untested piece)
- [x] **`bdim_coeff_3d` / `bdim_coeff_sigma_3d` in Warp** — **DONE** (`warp_bdim.py`,
      `test_bdim.py`, `bench_bdim_memory.py`; see DONE (5)).  Verdict: **Warp** (not
      torch.compile) — parity bit-exact vs native CUDA, single-source CPU==GPU, and the
      peak-memory check confirms it stays register-resident (0.0 extra, no global spill).

### D. Stencil-class kernels (AP6) — Warp feasibility now SHOWN, perf/coverage remain
- [x] **Advection (first-order upwind)** in Warp — parity vs PyTorch ref 1e-7, CPU+GPU.
      [x] **high-order limiter** (QUICK/ABDQUICKEST/vanLeer/CDS/CUBISTA) — **DONE**
      (`advect_flux_add_warp`): parity vs native `advect_flux_add` **bit-exact (0.0)**, all 5
      schemes × 2-D+3-D × every (i,d) pair; Warp CPU==GPU; rhs-stride caveat handled.  Perf
      (after compile-time scheme specialization): 2-D 0.05–0.38× (faster), 3-D 0.98–1.03× QUICK /
      0.58–0.62× ABDQUICKEST (≤ native).  MEMORY: warp adds 0.0 MiB = native, vs python 28–132 MiB.
- [x] **RBGS smoother** in Warp — parity vs native 1.3e-7; perf **0.5–1.0× native (graph,
      i.e. Warp faster)** after flat-addressing + clamp-Neumann opts.  `wp.tile` NOT needed.
      [x] `jacobi_sweep` variant — **DONE** (`warp_multigrid.py`; interior parity < 1e-13).
- [x] **Multigrid transfer**: `mg_residual_3d`, `restrict_residual_3d`, `restrict_face_3d`,
      `prolongate_add_3d` — **DONE** (`warp_multigrid.py`, `test_multigrid.py`): bit-exact (0.0)
      vs the native CUDA ops; assembled into a graph-capturable **Warp V-cycle** that converges
      geometrically (~0.07/cycle) on a manufactured Neumann Poisson. See DONE (6).
- [x] **`cvof_sweep`** (two-phase Weymouth–Yue) — **DONE** (`warp_cvof.py`, `test_cvof.py`):
      bit-exact vs native CUDA (0.0), 2-D+3-D, every face_dim, contiguous + strided velocity
      view; Warp CPU==GPU.  Confined to the two-phase path (no core edits).
- [x] **`apply_bcs`, `interp`** — **DONE in 2-D** (`warp_misc_2d.py`, `test_misc_2d.py`):
      apply_bcs_2d bit-exact CPU+GPU; interp_2d bit-exact CPU, ~1 float32 ULP GPU (FMA).
- [ ] Decide Warp-everywhere vs Warp(scatter)+Inductor(stencil) once `wp.tile` perf is known.

### E. Poisson driver
- [x] **`poisson_solve_{mgcg,multigrid,rmgcg}` outer loop** — composition **DEMONSTRATED**
      in both 3-D (`WarpVCycle`, DONE (6)) and 2-D (`WarpVCycle2D`, DONE (7)): a Python /
      CUDA-graph multigrid driver built from the Warp smoother + the 4 transfer ops converges
      geometrically (the native `poisson_mult._vcycle_rbgs_{2,3}d` control flow with Warp
      kernels swapped in).  The mgcg/rmgcg *outer* Krylov loop (Aitken/recycling) is itself
      pure host control flow over these same kernel-level ops; not separately assembled here.

### F. End-to-end — STRUCTURE SHIPPED; clean drop-in ops wired; bridge remains
**Realized as a parallel backend tree (`lilytorch/src_cuda/` + `lilytorch/src_warp/`),
not an in-`src/` `use_warp_kernels` flag** (repo owner's request).  `lilytorch/src/`
is untouched (core diff = 0); both trees share the kernel-agnostic modules and each
has a `kernel/` package exposing one unified op API.  See `src_warp/README.md`.

- [x] **Structure + clean-drop-in swap-in** — `src_warp` wires the two
      signature-identical native ops into the live solver sub-classes:
      `advect_flux_add` (`AdvDiffSolver._solve_convective` fused flux) and
      `cvof_sweep` (`TwoPhase._cvof_sweep`).  Both **bit-exact vs native through
      the solver subclasses**, plus the **CPU single-source payoff** verified
      (`warp_poc/test_src_trees.py`, 11 tests; 165 total in `warp_poc/`).
      `src_warp/solver.py` injects the Warp sub-solvers at `__init__` via a
      localized, leak-free module-name swap (no `src/` edits).
- [x] **Kernel A/B (2-D) in-solver wiring — DONE** (Item 1, 2-D).  A
      **marshalling bridge** (`src_warp/kernel/_KernelA2DBridge`) re-expresses the
      live flat-table tensors (`F_flat`/`F_offsets`/`body_meta`/`kin`, AABB) into
      `WarpStreamingSDF2D`'s setup/update/run API (the class already consumed the
      flat layout — the "per-body scene layout" premise was stale); `bdim_coeff_2d`
      routes straight through (already signature-compatible).
      `FluidSolver._fluid_step_kernel_2d` is overridden in `src_warp/solver.py`
      (kernel calls swapped, `comp.sdf_val` pre-filled for `atomic_min`, σ path →
      native).  **Kernel A made dtype-generic** (one `@wp.kernel` source,
      `wp.overload` f32+f64) so an f64 solver streams on Warp; Kernel B's port is
      f64-only (f32 solver keeps native Kernel B — an f32 Kernel B variant is the
      3-D follow-up).  Parity: f32 Kernel A bit-exact, f64 chain vs native to the
      documented Kernel-A SDF gate (~1e-7; native interpolates the SDF in f32).
      `WARP_BACKED` += `streaming_sdf_stag_2d_multi`, `bdim_coeff_2d`.
- [ ] **Kernel A/B (3-D) in-solver wiring** — needs the 3-D marshalling bridge +
      an **f32 Kernel B Warp variant** (3-D solvers run f32; mirror the 2-D
      dtype-generic treatment) + override `_fluid_step_kernel_3d`.
- [ ] **Poisson driver assembly** — smoother + transfer ops + `WarpVCycle`
      composition all DONE (6, 7); remaining: wrap the mgcg/multigrid **outer
      driver** as a `poisson_solve_*`-compatible `PoissonSolver` backend (the
      native op is a monolithic C++ driver with no injectable smoother seam).
- [ ] **Full coupled trajectory match** (2-D `_1guillasim` pinned + 3-D
      jellyfish, <5% wall-clock) — needs Kernel A/B (+Poisson) wired.  Forces
      routed via `force_method=lagrangian` (native; Eulerian readout unported).
- [ ] **CPU end-to-end run** — per-op CPU single-source proven
      (`test_cvof_warp_cpu_matches_python_reference`); the full-step CPU run
      follows once the step path is fully Warp.

---

## Bottom line so far
**Both Warp-class scatter kernels (Kernel A, lagrangian_forces), Kernel B
(BDIM coeff + FD normals), the Poisson smoother, and the full multigrid
transfer set + a converging Warp V-cycle are now validated** — parity to machine
precision vs the native CUDA oracle, single-source CPU==GPU, competitive-to-faster
perf where benched.  Critically, **Kernel B's headline peak-MEMORY claim is now
proven**: Warp keeps mu0/mu1/normals register-resident (0.0 extra GPU memory,
torch- and driver-level), identical to native, vs the python path's 128–608 MiB
of full-grid intermediates.

The **high-order advection limiter** parity is now also proven (all 5 schemes,
2-D+3-D, bit-exact vs native).  This round closed most of what remained: the
**2-D variants** of Kernel A / Kernel B / the smoother (+ blend-eps and
interp_method≠0 coverage), the remaining stencil/gather kernels
(**`cvof_sweep`**, **`apply_bcs_2d`**, **`interp_2d`**), and the **Poisson
outer-driver composition** (`WarpVCycle2D` converges, mirroring the native 2-D
multigrid driver).  Every ported kernel is parity-clean vs the native CUDA
oracle (bit-exact where deterministic; documented 1-ULP FMA / block-tiling
exceptions) and single-source CPU==GPU.

**The one remaining item is §F end-to-end** — the in-solver `use_warp_kernels`
swap-in (SU1 2-D `_1guillasim` pinned + 3-D jellyfish trajectory match, <5%
wall-clock, CPU end-to-end payoff).  It is the ONLY task needing core-source
edits (the toggle in `solver.py`/`BDIMhandler`), so it is **deferred to a
follow-up** by request — the kernel set is otherwise proven kernel-by-kernel and
ready to wire in.
