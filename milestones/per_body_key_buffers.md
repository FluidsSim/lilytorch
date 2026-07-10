# Per-body key buffers — eliminate union-AABB waste in streaming SDF pipeline

## Motivation

The current streaming SDF pipeline (`streaming_sdf_stag_3d_multi`) processes
the **union AABB** of all bodies (current ∪ previous).  When bodies are widely
separated (e.g. two robots at opposite ends of the domain), the union AABB
spans the entire gap between them, including large regions of empty fluid where
every body SDF is `_FAR` ($10^4$) — no useful work is done but the init and
decode kernel passes still iterate over all of it.

### What is already optimal

The SDF interpolation kernel (`streaming_sdf_min_rho_3d_multi_kernel`)
**already uses per-body AABB fan-out**: `gridDim.y = B`, each block-row
processes only `body_vol[b]` cells.  The interpolation work is exactly
$\sum \text{body\_vol}[b]$ — zero waste on gap cells.  This should NOT be
changed; it is the correct parallelisation strategy.

### Where the waste lives

The waste is in the key-buffer infrastructure:

| Kernel | Launch grid | Work |
|---|---|---|
| `streaming_sdf_init_keys_3d_kernel` | `dirty_vol / 256` blocks | Touches every cell in union AABB, including gaps |
| `streaming_sdf_min_rho_3d_multi_kernel` | `blocksPerBody × B` blocks | **Already optimal**: only `body_vol[b]` per body |
| `streaming_sdf_decode_keys_stag_3d_kernel` | `dirty_vol / 256` blocks | Touches every cell in union AABB, including gaps |

The key buffer itself is sized to `dirty_vol = dirty_Ai × dirty_Aj × dirty_Ak`
(union AABB), and the init/decode passes scan the entire buffer.

```
           Body 1 AABB        gap         Body 2 AABB
           ┌────────┐                    ┌────────┐
           │        │                    │        │
           │  key   │ ←── empty fluid ──→│  key   │
           │ writes │                    │ writes │
           └────────┘                    └────────┘
           ┌─────────────────────────────────────┐
           │     key buffer = dirty_Ai × dirty_Aj│  ← init + decode scan ALL of this
           └─────────────────────────────────────┘
```

For two separated bodies, `dirty_vol` can be 60–90% of the full grid, while
$\sum \text{body\_vol}[b]$ is only 2–5%.

---

## Proposal: per-body key buffers + overlap-only resolve

### Core idea

Instead of one `dirty_vol`-sized key buffer indexed by AABB-local coordinates,
give each body its own key buffer sized to `body_vol[b]`.  The min-kernel
writes directly to `key_b[local]` (no `g_local` mapping needed).  Then a
resolve pass handles only cells that appear in 2+ bodies' AABBs.

### Two regimes

#### Regime A: non-overlapping bodies (the common case for separated robots)

Each body's AABB is disjoint from every other body's AABB.  In this regime:

- **No key buffers needed at all.**  Each body just writes its SDF directly
  into the global `sdf_cc/u/v/w` and `bU/bV/bW` arrays with a plain `<`
  comparison (not atomic), since no other body competes for the same cell.

- **No init pass, no decode pass.**  The SDF and body velocity are written
  directly to the global output tensors inside the min-kernel.

- Launches per step: 1 (the min-kernel only, with `gridDim.y = B`).

Overlap detection is cheap: compare every pair of body AABBs.  For `B ≤ O(10)`
bodies, this is a few hundred integer comparisons on the host.

#### Regime B: overlapping bodies (multi-link robots, convexify seams)

Where two or more bodies overlap (e.g. adjacent links in an articulated
robot), the per-body writes race.  In this regime:

1. **Per-body key buffers** `key_c_b[body_vol[b]]` (cc only; u/v/w handled
   analogously).  Each body's min-kernel thread writes `pack_sdf_body_key(s, b)`
   into `key_c_b[local]` — no atomics, single writer per cell.

2. **Overlap-resolve kernel**.  Launch `overlap_vol` threads, where
   `overlap_vol` is the volume of the intersection region (sum over all
   overlapping pairs).  For each cell `g` in the overlap:
   ```
   best = pack_sdf_body_key(+∞, sentinel)
   for each body b that covers cell g:
       key = key_c_b[body_local_index(b, g)]
       best = atomicMin(best, key)     // or sequential compare-swap if bodies ≤ 4
   write best to global sdf_cc[g], bU/V/W[g]
   ```

3. **Direct write for non-overlap cells.**  Cells covered by exactly one body
   are written directly (no key buffer, no atomics) from the min-kernel.

4. **No init pass.**  Per-body key buffers don't need initialisation because
   every cell is written exactly once per body.

5. **No full decode pass.**  Only the overlap region needs decoding.

### Why `atomicMin` is acceptable on the overlap

The overlap region is typically tiny: a few hundred to a few thousand cells at
the seams between adjacent links.  Even with `B = 20` links and a 4-cell seam
band per interface, the total overlap volume is `O(B × 4³) ~ 1280` cells —
negligible relative to the full grid.  The current code already uses `atomicMin`
in the min-kernel, and it works for the full dirty AABB; restricting it to the
overlap only is strictly better.

---

## What changes

### CUDA side (`streaming_sdf.cu`)

1. **New kernel: `streaming_sdf_direct_write_3d_kernel`** (Regime A).
   - `gridDim.y = B`, each block-row processes `body_vol[b]`.
   - Computes SDF + face velocity for body `b` at its cell.
   - Writes directly to global `sdf_cc/u/v/w` and `bU/bV/bW`:
     `if (my_sdf < global_sdf[g]) { global_sdf[g] = my_sdf; bU[g] = my_vel; }`
   - No atomics — the CUDA stream is the only writer to non-overlap cells.
     (Within a single body's AABB, each cell is touched by exactly one thread.)

2. **Modified kernel: `streaming_sdf_min_rho_3d_multi_kernel`** (Regime B).
   - When overlap exists, writes to per-body key buffers instead of
     `dirty_vol`-sized union key buffer.
   - `g_local` → `local` (no AABB-offset mapping).

3. **New kernel: `streaming_sdf_resolve_overlap_3d_kernel`** (Regime B).
   - Launched over the overlap volume only.
   - Per cell: loops over covering bodies, reads per-body keys, finds winner,
     decodes SDF + body velocity to global tensors.

4. **Remove** `streaming_sdf_init_keys_3d_kernel` (no longer needed).

5. **Remove** `streaming_sdf_decode_keys_stag_3d_kernel` as a full-grid pass.
   Replace with the overlap-only resolve for Regime B.

### Python side (`solver.py`, `kernels/ops.py`, `BDIMhandler.py`)

1. **Overlap detection** in `BDIMhandler.update()`:
   - After computing per-body AABBs, check pairwise overlap.
   - Store `_kernel_step['overlap_pairs']` and `_kernel_step['regime']`
     (`'disjoint'` or `'overlapping'`).

2. **Key buffer allocation** in `_fluid_step_kernel_3d`:
   - Regime A: no key buffers allocated (save `4 × dirty_vol × 8` bytes).
   - Regime B: per-body key buffers sized to `max(body_vol)` or
     `sum(body_vol)` for simplicity; freed before projection as before.

3. **Op dispatch** in `_fluid_step_kernel_3d`:
   - Regime A: call `streaming_sdf_direct_write_3d` (new op).
   - Regime B: call min-kernel (per-body keys) → resolve-overlap kernel.

4. **Kernel B (`bdim_coeff_3d`)** is unaffected: it reads global
   `sdf_u/v/w` and `bU/bV/bW` which are already populated by whichever
   Regime A or B path ran.  The init/decode passes never touched these
   global tensors anyway — they only managed the key buffers.

### Backward compatibility

- Single-body simulations: always Regime A (no overlap).  Same correctness,
  fewer kernel launches.
- Existing multi-link setups (salamander, jellyfish): Regime B with per-body
  keys + tiny overlap resolve.  Byte-identical results (the `atomicMin` resolve
  produces the exact same winner as the current union-AABB `atomicMin`).

---

## Expected impact

| Scenario | Current init+decode cost | Proposed cost |
|---|---|---|
| 1 body, 5% of grid | O(dirty_vol) ≈ 5% grid | O(0) — direct write |
| 2 separated bodies, 60% union AABB | O(60% grid) on gap cells | O(0) — direct write |
| Multi-link robot, overlapping links | O(dirty_vol), full AABB | O(overlap_vol) ≪ dirty_vol |

For the two-robots scenario at 512×128×128:
- Current: init + decode touch ~60% of ~8.4M cells ≈ 5M cells
- Proposed: 0 init, 0 decode, direct writes only
- Saving: ~5M pointless key pack/unpack ops per step

For multi-link robots: the overlap region is `O(B × seam_cells)`, typically
2–3 orders of magnitude smaller than the union AABB.

---

## Risks / open questions

1. **Regime A atomicity**: the direct-write kernel has threads for body 1 and
   body 2 on the same warp potentially writing to the same global cell if
   the bodies' AABBs are disjoint but grid-aligned.  Since the AABBs are
   verified disjoint before launch, the cells are distinct — but warp
   divergence between the two body block-rows may reduce occupancy.  Mitigation:
   keep `gridDim.y = B` as in the current kernel (this is already the pattern).

2. **Overlap detection cost**: pairwise AABB intersection for `B ≤ 50` bodies
   is `O(B²)` integer comparisons on the host — negligible (< 1 µs).

3. **Mixed regime**: what if some bodies overlap and others don't?  The
   simplest approach: if ANY pair overlaps, use Regime B for all bodies.
   The per-body key write is cheap; the only extra cost vs. Regime A is the
   resolve pass over the overlap region.  Alternatively, partition bodies
   into connected components and apply Regime A per component, Regime B
   per overlapping pair within a component — but the engineering complexity
   likely outweighs the benefit for `B ≤ O(10)`.

4. **Kernel B unchanged**: `bdim_coeff_3d_kernel` still iterates over the
   union AABB (`dirty_vol`).  This is a separate concern — see
   `milestones/per_body_bdim_kernel.md` (future).  For now, Kernel B's
   waste on gap cells is small (the bdim_one_axis is cheap identity when
   `mu0=1, mu1=0`) compared to the init/decode waste being eliminated here.

---

## Implementation order

1. Add overlap detection to `BDIMhandler.update()` → `_kernel_step` dict.
2. Write `streaming_sdf_direct_write_3d_kernel` and register the op.
3. Dispatch Regime A vs Regime B in `_fluid_step_kernel_3d`.
4. Benchmark single-body (must be no regression).
5. Write per-body key + overlap-resolve kernels for Regime B.
6. Benchmark multi-link (salamander/jellyfish) — must be byte-identical.
7. Benchmark two separated bodies — expected large speedup on the streaming
   SDF portion.
8. Remove `streaming_sdf_init_keys_3d_kernel` and the full-grid decode pass
   (or keep as fallback for Regime B if the overlap-resolve is buggy).
