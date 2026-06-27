# Task brief — red/black-COMPRESSED 2-D tiled RBGS (chase native parity)

**Owner:** (next agent)  ·  **Scope:** `lilytorch/warp_poc/` ONLY (no core-source edits)
**Status going in:** POC suite is 147/147 green; this is an OPTIONAL perf experiment with
uncertain payoff — frame results honestly, do not force a win.

---

## 1. Read first (context, ~15 min)
- `lilytorch/warp_poc/HANDOFF.md` — esp. **lesson 25** (the Warp 1.14 tile/shared-memory API +
  every gotcha: compile-time literal shapes, kernels-must-be-in-a-file, `tile_map` ≤3 tile args,
  `enable_backward=False`, manual element loops run REDUNDANTLY, `tile + scalar` unsupported, tile
  arithmetic `*` may be matmul → use `tile_map(wp.mul,…)`, shared budget ~48 KB).
- `lilytorch/warp_poc/VALIDATION_STATUS.md` — the "Fused TILED smoother" section + perf tables.
- Current impl: `lilytorch/warp_poc/warp_poisson_tiled_2d.py`
  (`WarpTiledRBGS2D` / `_make_rbgs_kernel`) and `warp_poisson_tiled_3d.py`.
- Native oracle: `lilytorch/src/kernels/csrc/cuda/multigrid_smoothers.cu`
  → `rbgs_2d_tiled_kernel` (block 8×32, shared-mem, fuses all `nsmoothing` sweeps, **color-
  selective** half-sweeps) and the C++ wrapper `rbgs_sweep_2d_cuda`.  Op = `rbgs_sweep_2d`.
- Tests: `lilytorch/warp_poc/test_poisson_tiled_2d.py`.
- Env: Warp 1.14.0 (latest), torch+CUDA, RTX 4080S, native `_C.so` importable
  (`import lilytorch.src.kernels`).  Run: `python -m pytest lilytorch/warp_poc/ -q`.

## 2. The problem (why the current 2-D tiled RBGS is ~1.75–2× of native)
The current `_make_rbgs_kernel` does a **masked whole-tile two-pass** per sweep: it computes the GS
update `U` for ALL `TILE×TILE` cells, then blends `rm*U + bm*center` (red half) / `bm*U + rm*center`
(black half).  Native `rbgs_2d_tiled_kernel` instead updates **only the active color** each
half-sweep — half the cells, half the stencil arithmetic.  So the cooperative version does ~2× the
flops.  `tile_map` is a whole-tile op; you cannot cheaply tell it "only the red cells", which is the
entire gap.  (Confirmed: precomputing `1/J` and a leaner blend don't help — and the extra tile
overflows the 48 KB shared budget at TILE=32.  The cost is the redundant whole-tile stencil, not the
div/blend.)

NOTE the asymmetry vs 3-D: native **3-D** smoothers are thread-per-cell (NOT fused), so the tiled
Warp version already BEATS them 3×.  Native **2-D** is already fused + color-selective, so 2-D is the
hard case.  Target here: tiled 2-D RBGS **≤ native** (currently 1.75× @1024³… er 1024², ns=10:
native 73 µs, current tiled 128 µs).

## 3. The proposed approach — red/black-COMPRESSED storage
Standard HPC trick for optimal red-black GS: store the two colours as two **compact** arrays so each
half-sweep streams a dense N/2 tile instead of a masked N tile.  In a checkerboard, every 4-neighbour
of a red cell is BLACK and vice-versa, so each half-sweep reads only the opposite colour's compact
tile and writes its own — exactly native's work profile.

Per block, in shared:
- `R` tile and `B` tile, each `(TILE, TILE/2)` (a row of `TILE` cells splits into `TILE/2` red +
  `TILE/2` black; choose row-major compaction `j_red = j//2` for the cells of that row's colour).
- Red half-sweep: `R[i, jr] = (-f_R + Σ c · {B neighbours}) / J_R`  — a dense `(TILE, TILE/2)`
  cooperative update reading shifted views of `B` (+ the halo).
- Black half-sweep symmetric.

**The hard part (resolve this FIRST — it's the feasibility gate):** the red cell's 4 black
neighbours do NOT map to a single affine `tile_view` of `B`.  In compressed layout the left/right
neighbours sit in the same compressed row but the up/down neighbours **shift by the row parity**
(a shear, not a stride).  So you need either:
  (a) a `tile_view` with the right per-axis offset for each of the (≤4) neighbour gathers, IF the
      shear can be folded into the offset by also splitting on row parity (often you need TWO views
      per vertical neighbour — even rows vs odd rows — and a `tile_assign`/`tile_map` blend), or
  (b) a cooperative indexed gather that Warp 1.14 can actually parallelise (NOT a manual `for i,j`
      element loop — those run redundantly on every thread, lesson 25).
Spend the first hour proving ONE neighbour gather (e.g. red←black-up) is expressible cooperatively
and bit-correct on a tiny case.  **If it can't be expressed without a redundant per-thread loop,
STOP and report that — the compressed approach then can't beat the masked one in this tile model.**

## 4. Concrete steps
1. **Feasibility spike** (throwaway, in scratchpad): one block, TILE=8, build `R`/`B` compact tiles
   from a known field, do ONE red half-sweep via cooperative ops, compare to a numpy red-black
   reference.  Confirm the neighbour shear is expressible + parallel (inspect generated `.cu` or just
   trust speed: time it vs the masked kernel — if it's not ~2× fewer flops, the gather isn't free).
2. If feasible: implement `_make_rbgs_compressed_kernel(nsweep)` + `WarpTiledRBGSCompressed2D` in
   `warp_poisson_tiled_2d.py` (kernel factory keyed by nsweep, `enable_backward=False`, TILE a
   multiple constraint, Neumann via the loaded-once halo = block-boundary approx like native).
3. **Validate** in `test_poisson_tiled_2d.py`:
   - ns=1 single-tile (N=TILE): bit-close (f32 ULP, <1e-5) to the existing `_torch_rbgs_singletile`
     reference (already in the test file).
   - multi-sweep: converges (residual drops).
   - (optional) off-seam parity vs native rbgs at ns=1 on a multi-tile grid.
4. **Bench** (extend `bench_2d.py` or a scratch script, CUDA-graph capture, vs `rbgs_sweep_2d`):
   report µs + ratio at 512²/1024², ns∈{2,10}.  Target ≤ native; even ~1.2–1.3× (down from 1.75–2×)
   is a real win — say so plainly.

## 5. Constraints & guardrails
- **warp_poc/ only.** `git diff --stat lilytorch/src lilytorch/integration` must stay EMPTY except
  the pre-existing diagnostics edit (diagnostics.py ~36 lines + solver.py 3 lines) — leave it.
- Keep all existing tests green (`python -m pytest lilytorch/warp_poc/ -q` → 147+).
- Match the native discretization (read `rbgs_2d_tiled_kernel` before coding); block-stale halo +
  Neumann-by-clamp is the accepted approximation (the existing tiled smoothers do this).
- Apply lesson 25 verbatim; add any NEW tile gotcha you hit to HANDOFF lesson 25/append a lesson.
- Report perf as ratio-vs-native under CUDA-graph capture; note ns=2 (the real V-cycle nu) AND ns=10.

## 6. Honest exit criteria (any of these is a valid "done")
- **Win:** compressed RBGS ≤ native (or clearly < the current 1.75–2×) with parity + convergence →
  wire it as the default `WarpTiledRBGS2D` path, update VALIDATION_STATUS perf table + HANDOFF.
- **No-go (expected-possible):** the red-black neighbour gather can't be expressed cooperatively in
  Warp 1.14 without redundant per-thread loops, OR it compiles but isn't faster (gather overhead
  eats the flop savings) → document the finding in HANDOFF (a new lesson) + VALIDATION_STATUS, keep
  the masked kernel, and note that **2-D tiled JACOBI (~1.1×) is the practical fast 2-D smoother**
  and tiled RBGS's home is 3-D (where it wins 3×).  A clean negative result IS a deliverable.

Do NOT ship a "win" that's only faster because it skipped correctness — the parity gate is mandatory.
