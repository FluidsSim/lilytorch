# Warp porting — agent handoff brief

## Mission
Continue the Warp single-source kernel port (to_do_list Part 2, AP6/AP7/AP8):
prove Warp can replace the hand-written `.cpp`/`.cu` kernel pairs AND the
PyTorch "python path", collapsing the triple-write (python + CPU `.cpp` + CUDA
`.cu`) into one `@wp.kernel` source that runs on CPU **and** GPU.

## Where everything lives
All work is isolated in **`lilytorch/warp_poc/`** (do not touch core source).
Read **`lilytorch/warp_poc/VALIDATION_STATUS.md` first** — it has the full
per-step kernel-mode pipeline, what is validated, and the prioritized testing
checklist (sections A–F).  Existing files:
- `warp_kernels.py` + `test_parity.py` + `bench_viability.py` — **Kernel A**
  (`streaming_sdf_stag_3d_multi`): sequential-per-body + fanned all-body designs.
- `warp_poisson.py` + `test_poisson.py` + `bench_poisson.py` — **RBGS Poisson smoother**.
- `warp_advection.py` + `test_advection.py` — **upwind advection**.
- `warp_lagrangian.py` + `test_lagrangian.py` + `bench_lagrangian.py` —
  **lagrangian_forces 2-D/3-D** (the 2nd Warp-class atomicAdd-scatter kernel).

- `warp_bdim.py` + `test_bdim.py` + `bench_bdim_memory.py` + `bench_bdim.py` —
  **Kernel B** (`bdim_coeff_3d` + `bdim_coeff_sigma_3d`): fused BDIM mu0/mu1 + FD
  normals + the peak-MEMORY experiment + the wall-clock bench.
- `warp_multigrid.py` + `test_multigrid.py` + `bench_multigrid.py` — **multigrid
  transfer ops** (`jacobi_3d`, `mg_residual_3d`, `restrict_residual_3d`,
  `restrict_face_3d`, `prolongate_add_3d`) + an assembled, converging `WarpVCycle`.

- **2-D round** (this handoff): `scene_2d.py` (shared disc-SDF scene builder);
  `warp_kernels_2d.py` + `test_kernelA_2d.py` — **Kernel A 2-D** (fanned+sequential,
  blend-eps + interp_method≠0); `warp_bdim_2d.py` + `test_bdim_2d.py` — **Kernel B 2-D**;
  `warp_poisson_2d.py` + `test_poisson_2d.py` — **2-D RBGS/Jacobi smoother**;
  `warp_misc_2d.py` + `test_misc_2d.py` — **interp_2d + apply_bcs_2d**;
  `warp_cvof.py` + `test_cvof.py` — **cvof_sweep** (two-phase, 2-D+3-D);
  `warp_multigrid_2d.py` + `test_multigrid_2d.py` — **2-D transfer ops + `WarpVCycle2D`**.

Run tests: `python -m pytest lilytorch/warp_poc/ -q` (127 pass today).
Run a bench: `python -m lilytorch.warp_poc.bench_poisson --grids 32 64 96 128`.
Env: Warp 1.14.0, torch CUDA, RTX 4080 Super; native `_C.so` is built/importable.

## What is DONE (don't redo)
1. **Kernel A** — parity vs native (SDF rel<5e-4, vel rel<1e-4), fanned design
   0.59–1.16× of native at all B and grid sizes, single-source CPU+GPU verified.
2. **RBGS Poisson smoother** — bit-parity vs native 1.3e-7, converges, CPU==GPU,
   and **FASTER than native** (0.5–1.0× under CUDA graph) after two opts.
3. **Advection (1st-order upwind)** — parity 1.4e-7 vs a PyTorch reference, CPU+GPU.
   **High-order limiter (`advect_flux_add_warp`)** — faithful float64 port of native
   `advect_flux_add`; all 5 schemes (QUICK/ABDQUICKEST/vanLeer/CDS/CUBISTA), per-(i,d)
   in-place `rhs += dt_dh·(F_L−F_R)`, lo/hi CDS fallbacks. **Parity vs native CUDA bit-exact
   (0.0)** over {2-D,3-D}×{5 schemes}×every (i,d) pair (built from advection.py's real strided
   slice views; rhs strides passed separately — the face_dim≠outermost caveat). Native is
   CUDA-only → CPU via Warp CPU==GPU (<1e-12). Perf (`bench_advection.py`, graph, AFTER
   compile-time scheme specialization — lesson 17): 2-D 0.05–0.38× (faster — native pays
   torch-dispatch overhead), 3-D 0.98–1.03× QUICK / 0.58–0.62× ABDQUICKEST (≤ native).
   MEMORY (`bench_advection_memory.py`): warp adds **0.0 MiB** above baseline (torch+driver),
   = native, vs the python `_flux` path's 28–132 MiB of F/B1/B2/flux_* temporaries at 96³–160³
   (the `.cu`'s whole rationale) — same register-residency story as Kernel B.
4. **lagrangian_forces 2-D/3-D** — both Warp-class scatter kernels now done. Parity
   vs native ≤1.7e-16 (machine precision) over the full {2D,3D}×{linear,quadratic}×
   {scalar,field nu_rho}×{offset} matrix, CPU+GPU; CPU==GPU; perf 0.91–1.26× (graph).
5. **Kernel B** (`bdim_coeff_3d` + `bdim_coeff_sigma_3d` + FD normals) — parity vs
   native CUDA **bit-exact (0.0)** over {mu0_proj 0/1}×{full,sub-block}×{plain,σ};
   CPU velocity bit-exact, CPU==GPU 4.4e-16. **MEMORY (the load-bearing experiment):
   Warp peak GPU memory above baseline = 0.0** (torch *and* driver-level), identical to
   native, vs 128–608 MiB for the python path at 96³–160³ → mu/normals stay register-
   resident, no global spill.  **Perf** (`bench_bdim.py`, graph): 0.96–1.07× of native
   (faster at 128³) after switching to a flat 1-D launch (lesson 13); single fused launch
   → graph ≈ eager.  NOTE: native CPU `bdim_coeff_3d` writes `c_out` at the
   *padded-grid* index (different `ch/cv/cw` layout than CUDA) — coeff parity is vs CUDA,
   the production path; only velocities are compared on CPU.
6. **Multigrid transfer ops + a Warp V-cycle** — `jacobi_3d`/`mg_residual_3d`/
   `restrict_residual_3d`/`restrict_face_3d`/`prolongate_add_3d` parity vs native CUDA
   **bit-exact (0.0)** (Jacobi interior < 1e-13); native ops are CUDA-only so CPU is
   CPU==GPU-validated. `WarpVCycle` (Warp RBGS-f64 + the 4 transfer ops, face-coeff →
   `cp0..cm2`, coarse op via `restrict_face`, like `poisson_mult._vcycle_rbgs_3d`)
   **converges ~0.07/cycle** (1.8e2→4.9e-10 in 10 cycles, 32³/4 levels).  **Perf**
   (`bench_multigrid.py`, 128³, graph): every op 0.98–1.04× of native; memory matches
   native (mg_residual register-J = 16 MiB; jacobi one ping-pong buffer, same as native).

## What is DONE — 2-D round (this handoff)
7. **2-D Kernel A / Kernel B / smoother / interp / apply_bcs / cvof / multigrid + V-cycle**
   — all parity-clean vs native, single-source CPU==GPU.  Highlights & gotchas folded into
   lessons 18–22 below.  Kernel B 2-D is **bit-exact (0.0)** incl. the σ variant; the 2-D
   native writes the Poisson coeff at the FULL-grid index `c_out[g]` in BOTH CUDA and CPU →
   **no CPU/CUDA layout discrepancy** (lesson 9 is 3-D-only), so coeff parity holds vs both.
   Kernel A 2-D closes the §B blend-eps + interp_method≠0 gaps.  `WarpVCycle2D` converges
   geometrically → the Poisson outer-driver composition (§E) is shown in 2-D.

## Hard-won lessons (apply these — they cost real debugging time)
1. **`wp.bit_cast` is NOT in Warp 1.14.** Can't replicate the native uint64
   packed-key atomicMin. Use float `wp.atomic_min` + an equality-decode pass
   (see fanned `streaming_sdf`), or a sequential-per-body design.
2. **Flat 1-D `wp.array` addressing beats 3-D `wp.array` indexing by ~1.5×.**
   3-D indexing recomputes strides per access. Precompute a base index and add
   `±stride` offsets, mirroring the native CUDA (`p_base ± si/sj/1`).
3. **CUDA-graph capture is essential.** Eager Warp is 3–14× slower (per-launch
   Python overhead). Always capture the per-step kernel sequence with
   `wp.ScopedCapture` and replay via `wp.capture_launch(graph)` (NOTE: no
   `device=` kwarg on `capture_launch` in 1.14).
4. **Fold homogeneous-Neumann BC into the stencil via index clamp** (ghost =
   nearest interior = self at the boundary) → eliminates separate BC kernel
   launches and is bit-identical to explicit ghosts. Dirichlet/other BCs still
   need an explicit ghost kernel (kept as `apply_neumann`).
5. **Parity-test trap:** `torch.Generator(device=...)` yields DIFFERENT random
   sequences on CPU vs CUDA for the same seed. For CPU-vs-GPU tests, build the
   problem ONCE and `.to(dev)` it — else you compare two different problems.
6. **Single-source works:** the same `@wp.kernel` runs on Warp `device="cpu"`
   (→ C++/OpenMP) and `"cuda:0"`; results agree to ~1e-8 (float32 reduction noise).
7. **Always keep the native self-test as the parity oracle** before claiming a port.
8. **No native self-test exists for `bdim_coeff` / multigrid transfer ops** — build a
   manufactured-field oracle (sphere SDF + random fields), call the native op directly.
9. **Native CPU and CUDA can disagree on output layout.** `bdim_coeff_3d_cpu` writes the
   Poisson coeff at the *padded-grid* flat index `c_out[g]` (full Ngx·Ngy·Ngz), but the
   CUDA kernel writes the *compact face grid* `c_out[(i-1)·…]`.  Feeding the CPU op the
   compact `ch/cv/cw` that `solver.py` allocates corrupts the heap (`free(): invalid size`).
   The production path is CUDA + compact, so port to that; on CPU only compare the velocity
   fields (padded-`g` write, identical in both impls).
10. **`mg_residual_3d` and the multigrid transfer ops are CUDA-only** (no CPU `m.impl`).
    Validate Warp-on-CPU via Warp-CPU==Warp-GPU, not against a native CPU oracle.
11. **`wp.func` can return a tuple and write to array args** (used for `_mu0_mu1` and
    `linear_weights`); a void func that only writes arrays is fine too.
12. **Measuring Warp memory:** `torch.cuda.max_memory_allocated` does NOT see Warp's own
    allocator — wrap outputs as torch tensors via `wp.from_torch` (zero-copy) and also read
    driver-level `torch.cuda.mem_get_info` to catch any Warp-allocated global spill.
13. **Flat 1-D *launch* beats 3-D `wp.tid()` launch** (extends lesson 2 from addressing to the
    launch map). A 3-D `dim=(Ai,Aj,Ak)` launch maps thread IDs to cells with worse coalescing
    than the native flat decode. Launch `dim=Ai*Aj*Ak` and decode `dk=local%dAk` (k fastest,
    matching the native `.cu`) — this took Kernel B from 1.15× to ~1.0× of native, bit-exact.
    Bench the two mappings whenever a ported stencil lags native by ~15%.
14. **Zero-copy flat Warp view over an arbitrary strided/offset torch tensor:**
    `wp.array(ptr=t.data_ptr(), dtype=…, shape=(remaining,), device=…)` where
    `remaining = (t.untyped_storage().nbytes() − t.storage_offset()*elem)//elem`.
    `t.data_ptr()` already includes the storage offset, so flat index 0 == t's logical
    `[0,0,..]`; pass `t.stride(dim)` (in elements) as kernel int args and do the native
    pointer arithmetic. This is how the advection port honours `_field_for_flux`/`_face_vel`
    slice views AND the rhs-stride caveat (rhs is contiguous in *original* dim order, so
    `rhs.stride(face_dim) ≠ Nt1·Nt2` for d>0 — pass all three rhs strides separately).
    NB: `wp.array(..., owner=…)` is NOT a kwarg in 1.14 (it errored); just omit it.
15. **Torch ops inside `wp.ScopedCapture` break the capture** with "operation would make
    the legacy stream depend on a capturing blocking stream" — the face-velocity / field
    slicing (`0.5*(…)`, advanced indexing) are torch kernels on the default stream.
    Precompute all the slice views (and any torch-side scratch) BEFORE the capture; capture
    only the `wp.launch` calls. (Same cost for native and warp, so excluding it is also the
    fair kernel-isolating bench.)
16. **2-D WINS big (0.05–0.4×) because native pays per-(i,d) torch-dispatch overhead** that the
    Warp graph amortises (ndim²=4 launches/step). The graph replays one captured sequence;
    native's `torch.ops.…` dispatch per call is pure host overhead on the small 2-D kernels.
17. **Specialize a compile-time-templated native kernel with a Warp KERNEL FACTORY, not a runtime
    branch.** The advection 3-D port first ran a runtime `if scheme_id==…` dispatch (all 5 scheme
    code paths compiled into one kernel) → ~1.3–1.4× of native (which uses `template<int
    scheme_id>`), *despite* the branch being fully thread-uniform — the cost is register pressure /
    occupancy from the 4 dead paths, not divergence. Fix: a Python factory `_make_flux_kernel(scheme)`
    that defines an `@wp.kernel` **closing over a scheme `@wp.func`** and returns it; build one per
    scheme (`_FLUX_KERNELS = {sid: _make_flux_kernel(fn)}`) and launch the right one. Warp resolves
    the closed-over func at kernel-creation and inlines it → one lean kernel per scheme, exactly
    like the native template. This took 3-D QUICK from 1.39× to ~1.0× and ABDQUICKEST to 0.6×
    (Warp faster), still bit-exact. Give all the closed-over funcs a UNIFORM signature (here
    `(u,c,d,C)`, C ignored by 4 of 5) so one kernel body works for any scheme. Launch map
    (flat-1-D vs 3-D-tid) was confirmed irrelevant here (graph≈eager → not launch-bound).

18. **`wp.tid()` returns a TUPLE for multi-dim launches — never `wp.tid(0)`.** A 2-D launch
    kernel must do `op, line = wp.tid()`; `wp.tid(0)` raises "Couldn't find function overload
    for 'tid' that matched inputs with types: [int32]" and (worse) fails the WHOLE module's
    codegen, so an unrelated kernel in the same file (e.g. `interp_2d_kernel`) also "can't be
    found" on load.  When a Warp module won't load, check every kernel's `wp.tid()` arity first.
19. **The native 2-D Poisson smoothers are BLOCK-TILED (8×32), not thread-per-cell like 3-D.**
    They fuse all `nsmoothing` sweeps in shared memory using STALE inter-tile halos for sweeps
    2..n (a multigrid block-boundary approximation).  So a clean thread-per-cell Warp port is
    NOT bit-exact to native 2-D in general: **Jacobi nsmoothing=1 IS bit-exact on any grid**
    (one load, write `w·p_new+(1-w)·p_old` from pre-sweep neighbours); **RBGS nsmoothing=1 is
    bit-exact only single-tile** (Nx≤8,Ny≤32).  On multi-tile grids the RBGS difference is
    EXACTLY localized: the red half-sweep reads only pre-sweep values (every red cell matches);
    only black cells with a red neighbour ACROSS a tile seam differ.  Assert bit-exactness on
    off-seam cells (`gi%8∉{0,7} ∧ gj%32∉{0,31}`), not a loose global tolerance.  Also: native
    caps `Jinv=0` (writes 0) when `|J|<jcap_tol`; the 3-D Warp port instead skipped the cell —
    a benign divergence (J never caps for positive face coeffs) but match the cap to be faithful.
20. **2-D Kernel B has NO CPU/CUDA coeff-layout discrepancy (lesson 9 is 3-D-only).** The 2-D
    native `bdim_one_axis_2d` writes `c_out[g]` at the full-grid index g in BOTH the CUDA and
    CPU ops (`streaming_sdf_2d.cu` and `streaming_sdf_cpu_2d.cpp`) — there's no compact face-grid
    offset.  So `ch`/`cv` are full-grid `(Ngx,Ngy)` and coeff parity holds vs CPU *and* CUDA,
    and the launch decode is just `dj=local%dAj; di=local/dAj` (no per-face strides).
21. **GPU float-blend gathers (interp / bilinear prolong) are ~1 ULP off native, NOT bit-exact,
    due to FMA contraction** — nvcc and Warp codegen contract `Σ w·v` into different FMA trees.
    The CPU path (no FMA contraction) IS bit-exact, and pure-ADD reductions ARE bit-exact if you
    match the native add ORDER (e.g. `restrict_residual_2d` = `a+b+c+d` with a=[i0,j0] b=[i1,j0]
    c=[i0,j1] d=[i1,j1]).  So: take pure-add ops bit-exact (match order), allow ~1e-14 (float64)
    / ~1e-6 (float32) on weighted-sum gathers, and bit-exact on CPU.  Don't chase the GPU ULP.
22. **`apply_bcs` overlapping ghost writes are ORDER-UNDEFINED on GPU (native says so).** Two
    STAGE-1 ops (Neumann/Dirichlet) that touch the same corner race on GPU but run serially on
    CPU → a bit-exact GPU-vs-native test needs DISJOINT stage-1 ops (different field, or
    non-adjacent boundaries).  Reflective ops are stage-2 (a second launch, read-after-write) so
    they DO win the corner deterministically — overlap there is fine.  Build the test descriptors
    accordingly; don't assert equality on a contended corner.

23. **2-D Poisson smoother: the gap to native is shared-mem multi-sweep FUSION, and Warp 1.14
    can't do it.** Native `rbgs_sweep_2d` loads p/coeffs/f into shared memory ONCE and runs all
    `nsmoothing` sweeps there (1 global round-trip); a thread-per-cell Warp port re-reads global
    every half-sweep.  Measured fusion advantage: **1.6× at nsmoothing=2 (the real V-cycle nu),
    up to 4.2× at nsmoothing=10**; at nsmoothing=1 there's NO fusion and Warp matches native.
    **⚠️ CORRECTION — this is NOT a hard block (an earlier version of this lesson said it was):**
    Warp 1.14 ships the FULL tile API (`tile_load`/`tile_store`/`tile_view`/`tile_map`,
    `storage="shared"`, cooperative ops with implicit barriers).  The earlier "absent" finding was
    an `hasattr(wp,'tile_load')` mistake — they're KERNEL builtins (registered in codegen), not
    Python module attrs, so `hasattr` returns False though they work inside `@wp.kernel`.  And 1.14
    IS the latest PyPI `warp-lang`.  A **fused TILED Jacobi** (`warp_poisson_tiled_2d.py`) using it
    closes the gap to **~1.1× of native** (from 3–4×) — see lesson 25.  What ALSO helped the simple
    untiled path (~10%, kept):
    a **color-compacted flat launch** (`rbgs_halfsweep_2d_compact`) that launches only the Nx·Ny/2
    cells of one color (even Ny) so NO thread early-returns on the parity check — a plain 2-D
    `wp.tid()` launch idles ~50% on the color test.  Net: Warp **matches/beats native UNFUSED**
    (0.79–1.11×); report vs BOTH fused and unfused, and bench at the realistic nu, not just nu=10.
    CONFIRMED it's NOT coeff-bandwidth- or divide-bound: packing cp0..cm1 into a `wp.vec4` (one
    16-B coalesced load) + precomputing `Jinv` once gave **0 gain** (within noise) — the only lever
    is the shared-mem cross-sweep reuse (fusion), which the tile API DOES provide (lesson 25).
    When comparing any ported kernel against a native op, **first check the native isn't itself
    mis-occupied** (lesson 24).
24. **Check the native op's occupancy before claiming a Warp speedup.** Native 2-D `cvof_sweep`
    looked 25–33× slower than Warp — but its block map is `(BT2=32, BT1=8, BFD=1)` with the 2-D
    transverse size = 1, so ~97% of each 32-wide block idles.  Native 3-D cvof does the SAME
    262k cells 26× faster (well-occupied).  The fair efficiency reference is **Mcells/ms vs the
    native op at PROPER occupancy** (here native-3D = 3.8 Mcells/ms); Warp-2D hits 4–5 Mcells/ms
    = that, i.e. Warp reaches native's genuine per-cell throughput while native-2D doesn't.  Don't
    report "N× vs native" when native is the under-configured outlier — report cells/ms vs the
    well-occupied native, and flag the native inefficiency.
25. **Warp 1.14 HAS the tile / shared-memory API — use it for fused stencils.** `hasattr(wp, ...)`
    does NOT see tile builtins (they're registered in codegen, not as module attrs) — check the
    `.pyi` / `tests/tile/` instead.  Available: `tile_load`/`tile_store`/`tile_zeros`/`tile_ones`/
    `tile_view`/`tile_map`/`tile_assign`/`untile`/`tile_atomic_add`, `storage="shared"` (up to
    64 KB), launched via `wp.launch_tiled(k, dim=[nbi,nbj], block_dim=…)` where `wp.tid()` gives the
    BLOCK index.  Recipe for a fused multi-sweep smoother (`warp_poisson_tiled_2d.py`, 3–4× → ~1.1×
    of native): per block `tile_load` the (TILE+2)² p-halo + coeff/f tiles to shared ONCE; loop the
    sweeps (compile-time count so it unrolls — use a kernel FACTORY keyed by nsweep), each sweep
    taking neighbour shifts as `tile_view(ps, offset=(2,1)/(0,1)/(1,2)/(1,0), shape=(TILE,TILE))`
    and combining cooperatively; `tile_store` once.  GOTCHAS: tile `shape=` must be a COMPILE-TIME
    literal (`TILE+2` arithmetic fails → precompute `TILEH=34`); kernels must be in a FILE (Warp
    can't read `exec`'d source); `tile_map` takes **up to 8 tile args** (NOT ≤3 — that earlier claim
    was wrong; the C++ overloads go `T1..T8`, confirmed in `native/tile.h`).  PREFER one fused
    `@wp.func` over chained binary `tile_map(wp.mul,…)+…`: each `tile_map` is a cooperative pass with
    a barrier + an intermediate whole-tile temporary, so collapsing the math into ONE `tile_map` with
    a custom func cuts barriers/temporaries and is materially faster (see lesson 27).  GOTCHA: don't
    name a `@wp.func` `fma` (or other CUDA-builtin names) — `tile_map(fma,…)` fails to overload-resolve
    against `::fma`; rename (`fmadd`).  Set
    `@wp.kernel(enable_backward=False)` (tile_map's adjoint failed to compile); manual `for i: for j`
    element loops run REDUNDANTLY on every block thread (only `tile_*` ops are cooperative) — express
    the math in cooperative tile ops, not scalar loops, or you get correct-but-256×-slow.  Jacobi
    maps cleanly (no red-black dep); a tiled RBGS needs color-masked updates — still open.  Grid
    must be a multiple of TILE (32); coarse-MG partial tiles via bounds-checked load/store — open.

26. **Red/black-COMPRESSED tiled 2-D RBGS does NOT beat the masked kernel — Warp's whole-tile
    affine `tile_view` cannot express a checkerboard's sheared neighbour.** (Experiment:
    `experiment_rbgs_compressed_2d.py`, brief `TASK_rbgs_compressed_2d.md`.)  Motivation: native 2-D
    `rbgs_2d_tiled_kernel` is fused + **color-selective** (`if(color==0)`), so the masked whole-tile
    Warp RBGS is ~1.75–2× (it evaluates the stencil on ALL cells then masks).  Idea: store the two
    colours as compact `(TILE, TILE/2)` tiles so each half-sweep is a dense N/2 update.  Result: the
    kernel is **correct** (bit-close to numpy red-black, f32 ULP) and **fully cooperative** (no
    redundant per-thread loops), but **NOT faster** — at 1024², ns=10: native 75–80 µs, masked
    1.71×, compressed-kernel-only **1.88×** (pack/unpack on top → 2.2×).  WHY: in compressed layout a
    red cell's up/down black neighbours are clean (`jb=jr`) but **left/right shear by row parity**
    (`jb=jr-1+(gi&1)` / `jr+(gi&1)`).  `tile_view` is **affine-only** (offset+shape, no stride / no
    per-row offset — confirmed in `__init__.pyi`), so the horizontal neighbour must be assembled as
    `even_mask*view(A) + odd_mask*view(B)` — a 2-view parity-mask blend that costs ~as many flops
    (~15N/sweep) as the redundant whole-tile stencil it removes (~16N/sweep).  Net zero, plus a
    per-call global pack/unpack.  ROOT CAUSE: native uses per-thread **scalar** shared indexing
    (`p_s[ti+1][tj]`, arbitrary per-thread reads + idle threads skip) — strictly more expressive than
    Warp's whole-tile affine ops; the fused 2-D RBGS is not matchable in the tile model.  ⇒ Keep the
    masked `WarpTiledRBGS2D`; the practical fast **2-D** smoother is tiled **Jacobi** (`WarpTiledJacobi2D`,
    ~1.1×, no red-black dep so no shear).

    **The colour-masking penalty is the SAME in 3-D — the "3-D RBGS wins 3×" claim is a FUSION win
    against an unfused native baseline, not the tile model handling colour better.**  Decompose the
    two factors (they are independent and were previously conflated):
    (i) FUSION — native **2-D** RBGS/Jacobi fuse all sweeps in one shared-mem launch; native **3-D**
        does NOT (thread-per-cell, `2·nsmoothing` separate global-pass launches).  This is why Warp
        tiling (which fuses) beats native in 3-D for BOTH smoothers — even Warp **Jacobi** wins 3-D
        ~4× (0.24×), pure fusion, no colour confound.
    (ii) COLOUR REDUNDANCY — the masked whole-tile RBGS does ~2× the stencil of a colour-selective
        kernel.  Measure it as the RBGS-vs-Jacobi penalty WITHIN Warp: **3-D 0.32/0.24 = 1.33×**,
        2-D ~1.5–1.8× — present in BOTH dims.
    So Warp RBGS is ~1.3–1.8× of Warp Jacobi everywhere; the native-relative sign only flips because
    native 3-D is unfused.  A FUSED native 3-D RBGS (the 2-D trick lifted to 3-D) would very likely
    beat Warp 3-D RBGS too, exposing the same gap.  Net: the tile model cannot match a fused
    colour-selective native RBGS in EITHER dimension; tiled **Jacobi** is the smoother that actually
    reaches native parity (it has no colour redundancy), and it is the right Warp MG smoother in both
    2-D and 3-D.  Tiled RBGS only looks good in 3-D because the native 3-D RBGS is itself un-optimised.

27. **FUSE the smoother stencil into one `tile_map(custom_func, …)` — the RBGS-vs-Jacobi penalty was
    mostly per-op OVERHEAD, not the colour redundancy.**  The masked RBGS (lesson 26) built each
    half-sweep from ~6–9 chained binary tile ops (`tile_map(wp.mul,c,nbr)` + `+`/`-` + `tile_map(div)`
    + a 2-mul-1-add blend).  Each is a separate cooperative pass (barrier) writing an intermediate
    whole-tile temp.  Collapsing it to **one `tile_map(_gs8, c0,up,c1,dn,c2,rt,c3,lf)` (raw 5-pt sum,
    8 tiles = the max) + one `tile_map(_sel_red/_sel_blk, m,s,ff,Jt,old)` (fuses the `(s-ff)/Jt`
    divide AND the colour select, 5 tiles) + one assign** = 3 ops/half-sweep.  Only the RED mask is
    needed (`_sel_blk` updates where m==0).  3-D: split the 6-term sum across two passes (`_gs8` first
    4 + `_add2` folds the 2 z-terms).  Measured (kernel-only, RTX 4080S):
    - 2-D fused vs masked: **0.74–0.95×** (10–26% faster).  vs tiled Jacobi: **1.14–1.38× at ns=2**
      (the realistic V-cycle smoothing) — down from masked's 1.5–1.9×; ≈ parity.  At ns=10
      (compute-saturated) still ~2× Jacobi = the irreducible whole-tile colour redundancy (lesson 26).
    - 3-D fused vs Jacobi: **1.15× at 128³/ns=2** (was ~1.33×); still beats native ~3× (unfused).
    Bit-faithful: the fused `(s-ff)/Jt` matches the masked/native op order → existing parity tests
    (`test_poisson_tiled_2d.py`, ns=1 ULP + convergence) stay green.  SHIPPED in
    `warp_poisson_tiled_{2,3}d.py` (`_make_rbgs_kernel`/`_make_rbgs_3d`).  Takeaway: in the tile
    model, MINIMISE the number of `tile_map` passes (barriers) — fuse aggressively with custom funcs
    up to the 8-tile-arg limit; the per-op overhead is real and dominates at low `nsmoothing`.

## Next work, prioritized
1. ~~**`lagrangian_forces_3d`/`_2d` in Warp**~~ — **DONE**.
2. ~~**Kernel B: `bdim_coeff_3d` + FD normals** + peak-MEMORY measurement~~ —
   **DONE** (see DONE 5; the memory experiment is in `bench_bdim_memory.py`).
3. ~~**Multigrid transfer ops** + a full Warp V-cycle~~ — **DONE** (see DONE 6).
4. ~~**High-order advection limiter** (van Leer/QUICK/ABDQUICKEST) and parity vs
   native `advect_flux_add`~~ — **DONE** (`advect_flux_add_warp`, see DONE 3;
   bit-exact all 5 schemes 2-D+3-D). THE NEW TOP OPEN ITEM is the **2-D variants**
   (item 5) and/or the **end-to-end swap-in** (item 6).
5. ~~**2-D variants** of Kernel A / smoother / Kernel B~~ — **DONE** (DONE 7).  Also DONE this
   round: `cvof_sweep`, `interp_2d`, `apply_bcs_2d`, the 2-D multigrid transfer ops, and the
   `WarpVCycle2D` outer-driver composition.
6. **End-to-end** — STRUCTURE SHIPPED as a **parallel backend tree** (`lilytorch/src_cuda/`
   + `lilytorch/src_warp/`, each with a `kernel/` package), per the repo owner's request —
   NOT an in-`src/` `use_warp_kernels` flag.  `lilytorch/src/` is untouched (core diff = 0);
   the kernel-agnostic modules stay shared there.  `src_warp` wires the two **signature-clean
   drop-in** ops into the live solver subclasses — `advect_flux_add`
   (`AdvDiffSolver._solve_convective`) and `cvof_sweep` (`TwoPhase._cvof_sweep`) — both
   bit-exact vs native through the subclasses, plus the **CPU single-source payoff**
   (`warp_poc/test_src_trees.py`, 9 tests; 156 total).  `src_warp/solver.py` injects the Warp
   sub-solvers at `__init__` via a leak-free temporary module-name swap.  See
   `src_warp/README.md` + `VALIDATION_STATUS.md` §F.  **REMAINING** (kernels all ported +
   parity-clean; what's left is in-solver wiring, not kernels):
   (a) **Kernel A/B marshalling bridge** — re-express the live `_kernel_static_*`/`_kernel_step`
   flat-table tensors (`F_flat`/`body_meta`/`kin`) in the POC wrappers' per-body layout, then
   override `_fluid_step_kernel_{2,3}d`.
   (b) **Poisson driver assembly** — wrap the `WarpVCycle` mgcg/multigrid outer driver as a
   `poisson_solve_*`-compatible `PoissonSolver` backend (native op has no injectable smoother seam).
   (c) **forces** — route `force_method=lagrangian` (Warp lagrangian wrapper) or port the Eulerian
   `streaming_sdf_forces_post_*`.
   (d) Then the **coupled SU1 trajectory match** (2-D `_1guillasim` pinned + 3-D jellyfish,
   <5% wall-clock) and the **full-step CPU run**.
   Optional 3-D polish: extend 3-D Kernel A with blend-eps + interp_method≠0 (2-D already proves
   the algorithm) and the Kernel-A AABB-sized output scratch (MP8/T2b transient-memory win).
7. **(LATER) Close the residual 2-D RBGS gap vs native (~1.15–1.6×).** After op-fusion (lesson 27)
   the tiled 2-D RBGS is ~1.15–1.24× native at ns=2 but still **~1.5–1.6× at high nsmoothing**
   (1024²/ns=10: native 79 µs vs fused 122 µs).  This residual is the irreducible whole-tile colour
   redundancy (lesson 26): both half-sweeps evaluate the stencil on ALL cells, vs native's
   colour-selective `if(color==…)` per-thread update.  KNOWN, ACCEPTABLE FOR NOW — tiled **Jacobi**
   is the recommended fast smoother (no colour redundancy, ~1.1× native).  To actually match native
   2-D RBGS would need per-thread colour-selective execution (scalar shared indexing), which the Warp
   whole-tile tile_map model does not expose; candidate avenues to revisit later: `tile_scatter_masked`
   / `tile_store_indexed` for active-colour-only writes, or a streamlined fused-COMPRESSED kernel
   (lesson 26's compressed layout but with the lesson-27 fusion applied to the parity-mask blend, to
   see if N/2 stencil work + minimal ops finally beats the masked N-work at saturation).  NOT a
   blocker for the end-to-end swap-in (item 6).

## Constraints
- **Stay in `lilytorch/warp_poc/`.** Do NOT edit `solver.py`, `body.py`,
  `two_phase*.py`, or `src/kernels/csrc/*`. This is a parallel POC; the native
  kernels are the oracle, not the target of edits.
- Keep each ported kernel's native self-test green as the parity gate.
- Match the native discretization exactly (read the `.cu` before porting).
- Report perf as ratio-vs-native (with-reset / production-relevant timing) and
  always note bit-parity + CPU==GPU.
- Update `VALIDATION_STATUS.md` checkboxes as you go.
