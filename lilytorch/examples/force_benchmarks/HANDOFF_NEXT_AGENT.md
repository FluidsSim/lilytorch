# Handoff: establish force-readout parity properly

**Branch:** `cuda_native_port` · **Written:** 2026-07-17 · **Updated:** 2026-07-17 (B1 landed) ·
**Predecessor evidence:** `lilytorch/milestones/force_readout_agreement_handoff.md`
(§§1-11; §10 is the current state)

Read this file first, then §10 of the milestone doc for the raw evidence. **Rebuild the native
extension before anything else** (`python setup.py build_ext --inplace`) — the op set moves with
refactors, and a stale `_C.so` silently hides bugs (it hid two for a whole session).

---

## 0. The situation in one paragraph

The zebrafish (`examples/zebrafish_ki_project/gen_configs_pd_3d_slow_fast.py`) swims **~45% slower**
under `force_method: eulerian` than under `lagrangian`. Prior work established a real, quantified
mechanism for part of this (the eulerian's viscous band is shifted out by `eps` and, on a body only
~2.9 cells thick, integrates over an inflated iso-surface). **But the headline comparisons that
produced the 45%, the "3.96× viscous over-read" and the `lagrangian_sample_offset ≈ h`
recommendation are CONFOUNDED** — see §A. A 35% residual mismatch is not acceptable, and the
comparison must be redone with the two readouts pinned to identical sampling locations before any
conclusion about "which is right" is trustworthy.

**Your job:** make the comparison rigorous, find where parity is actually lost, and validate against
analytic and published references. Not to defend the previous conclusions.

### STATUS — what has been done since this file was written

- **B1 (split the sampling offsets) is DONE and landed.** Both readouts now carry independent
  `sample_offset_pressure` / `sample_offset_friction` knobs, end to end (op schemas → CUDA → both
  CPU twins → python path → solver → config), defaults bit-exact. See §B1 for the API.
- **§B1a steps 0-2 are DONE (2026-07-17, second session). The matched-sampling comparison has now
  been run, and its central prediction is FALSIFIED. READ §B1b BEFORE ANYTHING ELSE** — it
  supersedes large parts of the plan below and of §C. Headlines:
  - **The ~45% zebrafish speed gap is NOT a readout difference. It is sampling location.** At
    matched `(0,0)` the two readouts swim the fish at **0.999×** of each other (65.36 vs 65.44 mm/s).
  - **The `off≈h` plateau is driven by the PRESSURE offset — which no paper does.** Moving *friction*
    out (the M&W/Verma prescription) makes the fish **slower**; moving *pressure* out is what buys
    the plateau. This inverts §10's story.
  - **Verma's `(0, 2h)` has now been run (step 1). It is NOT the answer for this fish**: 79.3%
    closure, 57.9 mm/s — off the plateau, worse than `(h,h)` or `(2h,2h)`.
  - **Step 2's prediction is dead: the pressure ratio at matched `(0,0)` is 1.327, not ≈1** — the
    handoff's own criterion for "something unidentified is live". It is **sheet truncation** of the
    eulerian's delta measure (§B1b), a mechanism not in this document before.
  - **⇒ §B1a step 3 (shift-the-sample) will NOT fix the fish** and should not be started as
    designed: its premise is that the `(0,0)` gap is ~0 and the offset Jacobian is the story. Both
    are false here.
- **A fact from B1, now CORRECTED in magnitude for the fish**: the two readouts differ by an area
  Jacobian `J = (1+sκ₁)(1+sκ₂)` **on a sphere**. §C predicted `J ≈ 2.89` for the fish; **measured
  `A(2h)/A(0) = 1.169`** — the fish is a *sheet* (κ≈0 in the thin direction), not a sphere, so the
  Jacobian is small and explains little. See §B1b.
- Everything else (B2 proximity, B3 references, B4 zebrafish) is untouched. **B2 (proximity) is now
  the leading unexplained candidate** — see §B1b "what is still unexplained".

---

## A. THE FLAW IN THE EXISTING COMPARISON — read before trusting any §10 number

> **The knobs described here have been FIXED (B1 landed).** The diagnosis below is retained because
> it explains why every §10 cross-method number is untrustworthy — the *data* is still confounded
> even though the *code* no longer is. What follows describes the code as it was.

The two readouts did not sample in the same places, and the previous work varied both at once:

|  | pressure sampled at | viscous σ sampled at |
|---|---|---|
| **eulerian** | `φ = 0` — **never shifted** | `φ = eps_solver = eps_multiplier·h` |
| **lagrangian** | `φ = lagrangian_sample_offset` | `φ = lagrangian_sample_offset` — **same knob** |

Verified in code (pre-B1): `cuda/eulerian_forces.cu:494-499` (`d_visc = s_cc_body - eps_solver` for
the viscous delta; `s_cc_body` unshifted for the pressure delta) and
`cuda/lagrangian_forces.cu:208-243` (one `sample_offset` moving the query point for the strain
tensor **and** `p`).

So the configurations compared in §10 were:

| config | pressure at | viscous at |
|---|---|---|
| eulerian, live | 0 | 2h |
| lagrangian, off=0 | 0 | 0 |
| lagrangian, off=h (the recommendation) | h | h |

Consequences — be precise about what this does and does not invalidate:

- **Still valid**: everything measured against an *exact* answer (the analytic oracles: lagrangian is
  exact at every R/h; the eulerian's over-read tracks the enclosed-volume ratio), the
  `delta_order=2` coarea fix, the geometric facts (inscribed radius 2.86h vs `eps_body` 2h; 95% of
  the interior inside the band), and §10.3's **viscous-only** matched-distance table (that one *is*
  like-for-like: eul σ@s vs lag σ@s).
- **Confounded**: the "3.96× viscous / 0.41× net thrust at live settings" (compares σ@2h against
  σ@0), every **net-force** and **swim-speed** comparison between methods, and therefore the
  strength of the `off≈h` recommendation. `off≈h` is supported by closure (74.4% → 107.8%) and by a
  flat speed plateau across `off ∈ [0.5h, 3h]`, but **why** it helps is unknown: it moves pressure
  and friction together, so the benefit could be entirely from one channel.
- **The `eps` shift is not a bug.** It is the Maertens & Weymouth convention, and it exists for a
  real reason: inside the BDIM band the blended velocity gives `ε(u_blend) ≈ μ0·ε(u_fluid)`, so σ
  read there is contaminated. What is unvalidated is its behaviour at `eps/R ~ 1`, and the fact that
  the surface it then integrates over is not corrected for.

---

## B. The work, in order

### B1. Split the sampling offsets — ✅ DONE (2026-07-17)

Both readouts now carry two independent knobs. **Config API** (`base_sim_config.py`, in **CELLS**,
so a value survives a change of grid — trap #5):

```python
self.sample_offset_pressure_cells = None   # where p is read
self.sample_offset_friction_cells = None   # where σ is read
```

`None` → each readout keeps its **legacy** convention, so every existing config is bit-exact:

| | default `(p, f)` | when the knobs are set |
|---|---|---|
| **eulerian** | `(0, eps_multiplier·h)` — Maertens & Weymouth | both readouts take the same values |
| **lagrangian** | `(lagrangian_sample_offset, lagrangian_sample_offset)` | ″ |

Setting either knob applies it to **both** readouts — that is the whole point: it pins them to
identical sampling locations. The legacy `lagrangian_sample_offset` still works and still moves both
channels together.

Internals: `solver.eul_sample_offset_{pressure,friction}` and `solver.lagr_sample_offset_{pressure,
friction}` (metres, resolved once in `FluidSolver.__init__`). Op schemas take
`float sample_offset_pressure, float sample_offset_friction` — the eulerian's old single
`eps_solver` argument is **gone**, so any out-of-tree caller will fail loudly rather than silently
mis-bind (`eps_solver` was always the *friction* offset).

Landed across: `csrc/ops.cpp` (4 schemas), `csrc/cuda/eulerian_forces.cu` (2-D+3-D),
`csrc/ops_{2,3}d.cpp` (CPU twins), `csrc/cuda/lagrangian_forces{,_2d}.cu`,
`csrc/lagrangian_forces_cpu{,_2d}.cpp`, `native.py`, `forces.py` (native + python paths),
`solver.py`, `base_sim_config.py`, and the `force_benchmarks/` harnesses.

**Verified** (not merely compiled):
- Explicit `(0, 2h)` is **bit-exact** with the default on a 5-step coupled 2-D run
  (`fv=0.487350562671`, `fp=28.237438779046`).
- Each knob moves **only its own channel**, in both readouts, at rel 1e-12 (new oracle tests).
- The pressure knob now does something on the eulerian — **it previously could not**, p was pinned
  at `φ=0`.
- Suite: 370 passed, 10 failed — **all 10 verified pre-existing** by re-running with `forces.py` +
  `solver.py` stashed (identical failures, bit-identical numbers). They are
  `test_python_eulerian_force_path_*` (the known CPU/GPU divergence, §C Open),
  `test_two_phase::*uniform*` (3), and `test_whole_step_capture_native` (5). **Do not attribute
  these to B1.**

New tests in `tests/test_forces.py` (all against *exact* answers, ~1 s):
`test_split_offsets_each_channel_moves_only_with_its_own_knob`,
`test_split_offsets_equal_knobs_reproduce_the_legacy_single_knob`,
`test_split_offsets_eulerian_pressure_reads_the_offset_iso_surface`.

### B1a. THE PLAN, REVISED 2026-07-17 after reading Verma et al. — do these in order

The goal is **parity between the readouts AND accurate force estimation**, and §C now shows these
are *not* a trade-off: there is one configuration where both hold. Read §C "Verma et al. settles
the design question" and "the fix" before starting.

0. **Plumb the split knobs into `gen_zfish_readout_arbitration.py` first.** It currently exposes
   only the LEGACY single knob (`ZFISH_LAGR_OFFSET_CELLS` / `ZFISH_LAGR_OFFSET` →
   `self.lagrangian_sample_offset`, lines ~74-78), so **it cannot express `(0, 2h)`** — the whole
   point of step 1. Add `ZFISH_OFF_P_CELLS` / `ZFISH_OFF_F_CELLS` → `sample_offset_{pressure,
   friction}_cells`, keeping the legacy vars working (unset ⇒ unchanged behaviour). Print the
   resolved `(off_p, off_f)` in the `[arb]` banner so every run self-documents which setting it was.
1. **Run the zebrafish at `sample_offset_pressure_cells: 0`, `sample_offset_friction_cells: 2`.**
   One config line in the production config, no solver code. This is **Verma et al.'s published
   method** (p at the true surface,
   `∇u` lifted 2h, both integrated over the true surface) *and* the same sampling as M&W's
   eulerian. **It has never been run**: the old single knob forced `(2h,2h)` and §10's `off≈h` was
   `(h,h)`, both of which move the pressure sample — which no paper does. Cheapest shot at the
   accuracy goal. Arbiter: `zfish_pbdim_closure.py` + `zfish_swim_speed.py`, as in §10.10.
2. **Measure eul-vs-lag at `(0, 2h)`** — same runs, both methods. **Prediction to falsify:**
   *pressure* ratio ≈ 1 (identical sampling, identical measure at φ=0), *viscous* ratio ≈ `J`
   (≈2.89 if the fish's curvature is near the inscribed `R=2.86h`; expect a curvature-weighted
   average, not a clean constant). If that holds, **the parity gap IS the Jacobian and nothing
   else**, and step 3 closes it. If the pressure channel does *not* come out ≈1, something
   unidentified is in play — that would be the real finding.
3. **Fix the eulerian: shift the sample, not the delta** (§C). This is the deliverable: it makes
   the eulerian mathematically identical to Verma/lagrangian, giving parity *and* accuracy. Gate on
   the sphere oracle first (exact answer), then the fish.
4. **Build the `F_penal` arbiter** (§B3a) to decide *which readout is right* independently of the
   oracle and of energy closure.

Then B2 (proximity) / B3 (references) / B4 as originally planned.

### B1b. RESULTS of §B1a steps 0-2 (2026-07-17, session 3) — the prediction is FALSIFIED

**Reproduce everything here in ~4 minutes:**

```bash
python setup.py build_ext --inplace
# live: 4 runs x ~41 s, into the SAME stack as the §10.10 cases
for c in "lagr_p0_f2 lagrangian 0 2 1" "eul_p0_f2 eulerian 0 2 2" \
         "eul_p0_f0 eulerian 0 0 2" "lagr_p0_f0 lagrangian 0 0 1"; do
  set -- $c
  ZFISH_CASE=$1 ZFISH_FORCE_METHOD=$2 ZFISH_OFF_P_CELLS=$3 ZFISH_OFF_F_CELLS=$4 \
    ZFISH_DELTA_ORDER=$5 python -m lilytorch.examples.force_benchmarks.gen_zfish_readout_arbitration
done
python -m lilytorch.examples.force_benchmarks.zfish_pbdim_closure
python -m lilytorch.examples.force_benchmarks.zfish_swim_speed
# frozen field, both readouts, channel-decomposed (needs snap_lagr.pt):
python -m lilytorch.examples.force_benchmarks.eul_vs_lag_matched
```

**Step 0 (plumbing) — done, and VALIDATED against §10.10 by two live controls.**
`gen_zfish_readout_arbitration.py` now takes `ZFISH_OFF_P_CELLS` / `ZFISH_OFF_F_CELLS` and prints
the resolved `(off_p, off_f)` in the `[arb]` banner. The legacy vars still work. Two controls
reproduce §10.10 **exactly**, which is what makes every number below comparable to that table:

| control | closure | speed | vs §10.10 |
|---|---|---|---|
| `eul_p0_f2` vs `eul_order2` | 115.8% = 115.8% | 42.067 = 42.1 | identical |
| `lagr_p0_f0` vs `lagr_off0` | 74.4% = 74.4% | 65.436 = 65.4 | identical |

#### The live table (all at the production grid; `eul_*` = delta_order 2)

| case | `(off_p, off_f)` | closure | speed [mm/s] |
|---|---|---|---|
| `lagr_p0_f0` = `lagr_off0` | (0, 0) | 74.4% | **65.436** |
| `eul_p0_f0` | (0, 0) | 91.2% | **65.357** |
| `lagr_p0_f2` **= VERMA, never run before** | (0, 2h) | 79.3% | 57.940 |
| `eul_p0_f2` = `eul_order2` | (0, 2h) | 115.8% | 42.067 |
| `lagr_off2h` (§10.10) | (2h, 2h) | 104.9% | 78.402 |
| `lagr_off1h` (§10.10) | (h, h) | 107.8% | 79.462 |

**1. The ~45% speed gap is NOT a readout difference — it is sampling location.** At matched `(0,0)`
the two readouts swim the fish at **65.357 vs 65.436 mm/s = 0.999×**. The famous gap compared
`eul(0, 2h)` against `lagr(0, 0)` — i.e. friction@2h against friction@0. **The eulerian's own speed
falls 65.36 → 42.07 (−36%) purely from moving its own friction sample 0 → 2h**, with the readout held
fixed. §A said the 45% was confounded; it is now *quantified*: essentially all of it was the offset.

**2. But "0.999×" does NOT mean the readouts agree — and this is a methodological trap.** Every
*absolute* measure says the eulerian reads **~1.23-1.33×** the lagrangian at matched `(0,0)`:

| measure (matched `(0,0)`) | eul/lag |
|---|---|
| frozen-field pressure `Fp_x` | 1.327 |
| frozen-field viscous `Fv_x` | 1.551 |
| frozen-field **net** `Fx` | 1.249 |
| live `<P_BDIM>` | 1.238 |
| live closure (91.2% / 74.4%) | 1.226 |
| **live swim speed** | **0.999** |

**⚠ CORRECTED — the first explanation of this was WRONG.** It said "swim speed is blind to a uniform
force scale". That argument needs the scaling to *be* uniform, and it is not: pressure 1.327 vs
viscous 1.551 (frozen), 1.126 vs 1.385 (live time-avg). A non-uniform scaling *does* move the
thrust/drag balance point, so it cannot explain a 0.12% speed match.

**The real reason (measured over all 1201 steps of the two `(0,0)` runs, `drags.h5`, window [0,0.3] s,
x-component summed over links):**

| quantity | eulerian | lagrangian | eul/lag |
|---|---|---|---|
| `Fp_x` (thrust) | 2.049e-5 | 1.819e-5 | 1.126 |
| `Fv_x` (drag) | −7.232e-6 | −5.222e-6 | 1.385 |
| **net `Fx`** | **1.326e-5** | **1.297e-5** | **1.022** |

The eulerian over-reads thrust by **+2.3e-6** *and* over-reads drag by **−2.0e-6** — opposite signs,
nearly equal magnitude — so they **cancel in the NET to 2.2%**, and the net is what moves the body.
**This is a COINCIDENTAL near-cancellation of two independent errors, not a structural property.**
The `(0,0)` agreement is therefore *fragile*, not reassuring: nothing protects it on another body,
gait, or setting.

It also explains `(0,2h)` in one move: shifting friction out inflates the viscous over-read (measure
×2.554) while leaving pressure untouched, **breaking the cancellation** ⇒ net thrust collapses ⇒ 36%
slower. Same mechanism, one number changed.

**⇒ Neither closure nor speed arbitrates a readout alone** — closure sees the scale, speed sees only
the net — **and the "45% speed gap" was never a measure of readout scale.** Say which you mean.

**NO TIME-COMPOUNDING (tested, not assumed).** The mismatch is a steady bias, not an accumulating
one: `Fp_x` eul/lag by quarter of [0,0.3] s = **1.168, 1.111, 1.114, 1.114**; `Fv_x` = 1.376, 1.424,
1.387, 1.359. Flat. (Compute the ratio of time-MEANS, never the mean of instantaneous ratios —
`lag Fp_x` crosses zero within the cycle and the pointwise ratio is NaN/unbounded.)

**⚠ The frozen 1.327 and the live 1.126 are DIFFERENT QUANTITIES — do not conflate them.** Frozen =
both readouts on ONE identical field (the clean readout comparison, zero trajectory divergence, one
instant). Live = each readout on ITS OWN field, time-averaged. The snapshot instant is not
representative of the cycle mean.

**And "the speeds agree" is true only of the SWIMMING direction**: `dx` 39.140 vs 39.198 mm (0.15%),
but **`dy` −1.812 vs −1.547 mm — 17% apart**, consistent with the sign-flipped `Fp_y`.

**3. The `off≈h` plateau is bought by the PRESSURE offset — which neither M&W nor Verma does.**
Decomposing §10.10's recommendation, one channel at a time:

| move | speed | closure |
|---|---|---|
| `(0,0) → (0,2h)` — **friction only** (the M&W/Verma prescription) | 65.4 → **57.9** (−11%) | 74.4% → 79.3% |
| `(0,2h) → (2h,2h)` — **pressure only** | 57.9 → **78.4** (+35%) | 79.3% → 104.9% |

**Moving the friction sample out — the thing the literature prescribes — makes the fish SLOWER. The
entire plateau comes from moving the pressure sample, which no paper does.** §B4 anticipated that
`off≈h` "may survive as a number" while dying as a recommendation, and predicted `(0,2h)` would
replace it. The opposite happened: `(0,2h)` is *worse* than `(0,0)` on speed, and `(h,h)`/`(2h,2h)`
survive. **Why the pressure offset helps is now the central open question** — it is not explained by
anything in this document, and it is worth more than every other effect here combined.

**4. STEP 2'S PREDICTION IS FALSIFIED.** Predicted: pressure ratio ≈1 at matched `off_p` (same
points, same iso-surface, same measure). **Measured at `(0,0)`: 1.327.** The handoff's own criterion:
*"If the pressure ratio is not ≈1, something unidentified is live — that is the real finding."*
It is live. The lateral component does not merely differ in scale — **it flips sign**
(`eul Fp_y = +1.437e-5` vs `lag Fp_y = −2.790e-6`).

**Control passed, and here is its exact strength** (the `CONTROL:` block printed by
`eul_vs_lag_matched.py`): with `off_p` pinned at 0, the eulerian pressure channel is invariant under
`off_f` **to within the kernel's own run-to-run non-determinism** — *not* bit-identical, and it
cannot be: `streaming_sdf_forces_post_3d` accumulates per-link forces with **atomics**, so repeating
the *identical call* does not reproduce bitwise.

That repeat is the null. **The null is itself noisy** (5 repeats: 3.9e-14 … 3.0e-13), so a single
sample of it is a bad floor — the script takes the worst of 5. Against that floor, every `off_f` row
over `[0.5h, 4h]` comes in at **0.08-0.93×**, i.e. **entirely below the noise**. Absolute drift
~7e-21 N against a 2.6e-5 N force. So any leak is bounded ~3e-13 relative, against a **0.327**
effect — twelve orders of magnitude of headroom.
**The control is also not vacuous**: over the same sweep the *viscous* channel moves **2.554×**, so
`off_f` is unambiguously live — it simply does not touch pressure.
⇒ the 1.327 is a real readout disagreement, not offset leakage.

> ⚠ **Do not write "bit-exact" of any 3-D force-kernel output.** The atomics make it false by
> construction. Quote a tolerance against the identical-call repeat as the null instead. (B1's
> "bit-exact" defaults were verified on a *2-D 5-step coupled* run — a different, reproducible path;
> do not generalise that phrasing to these kernels.)

**5. THE MECHANISM: SHEET TRUNCATION of the eulerian's delta measure** (new; not in §C).
The eulerian's surface measure on the fish is **0.636× the true area** (`A_coarea = 1.599e-4` vs
`Σ tri_area = 2.513e-4 m²`). Established by falsification, not assertion:

| test | result | verdict |
|---|---|---|
| **buried inter-link faces?** (the retracted 1.57× claim) | 100% of tri area lies within **0.75h** of the union surface; centroid depths span only `[−0.55h, +0.52h]` | **DEAD.** The retraction was right — `A_tri` is a genuine area |
| **generic thinness?** analytic sphere, `R/h = 2.86`, `eps = 2h` | **1.065** — no deficit at all | **DEAD.** On a sphere the inward truncation is compensated by outward area growth (`r²`) |
| **sheet truncation?** analytic slab `φ=|z|−t`, `eps_body = 2h` | `t=0.61h` (fish median depth) → **0.758**; `t=1.43h` → 0.953; **`t ≥ 2h = eps_body` → exactly 1.000** | **CONFIRMED.** The deficit switches off precisely when the body is thicker than the band |

On a body thinner than the delta band, each face's **inward half-delta is cut off by the medial
axis**, and — unlike a sphere — a sheet has no outward area growth to compensate. It is pure
geometry. The fish's median interior depth is **0.61h** against `eps_body = 2h`.

**The retracted "1.57× buried area" was a real number attached to the wrong object**: `1/0.636 =
1.57`. The area deficit is in the eulerian's **coarea measure**, not in the triangulation.

**6. So the eulerian carries TWO large, partially-cancelling errors at zero offset:**

```
eulerian(0,0) / lagrangian(0,0)  =  measure (0.636)  ×  integrand (~2.09)  =  1.327
```

It integrates over 36% too little surface while over-reading the integrand ~2×. The integrand
over-read is consistent with the known **`zero_pressure_inside=True` halves the eulerian pressure
force (0.513×)** result in §C: the pressure delta band spans both sides of the surface, so the
interior half feeds BDIM-contaminated pressure into the readout. **That is the next thing to test**
(cheap: it is a config flag on the frozen snapshot).

**7. ⇒ §B1a step 3 (shift the sample, not the delta) will NOT fix the fish. Do not start it as
designed.** Its premise is that the `(0,0)` gap is ~0 and that the offset-induced Jacobian is the
story. **Both are false here:** the gap at zero offset is already 1.33× (pressure) / 1.55× (viscous),
and the fish's Jacobian is small anyway — §C predicted `J ≈ 2.89`, but **measured `A(2h)/A(0) =
1.169`**, because the fish is a sheet (κ≈0 in the thin direction), not a sphere. The sample-shift
remains correct *in principle* and would still help a well-resolved rigid body, but it addresses an
error that is not the dominant one on this geometry.

#### What is still unexplained (ranked, with a falsification test for each)

1. **Why moving the PRESSURE sample outward buys +35% speed** (result 3). Biggest effect measured;
   no mechanism. Test: it is the one move that changes the thrust/drag *ratio* rather than the
   scale — decompose `Fp_x` per link at `(0,·)` vs `(2h,·)` on the frozen field and find whether the
   gain is localised to the tail (thrust) or the head (drag).
2. **The eulerian's ~2.09× integrand over-read at zero offset** (result 6). Test:
   `zero_pressure_inside` on the frozen snapshot; §C already measured 0.513× for that flag.
3. **The `Fp_y` sign flip** (result 4) — a *qualitative* disagreement, not a scale error. Suggests
   the two readouts see different lateral pressure structure. **This is exactly what B2 (proximity)
   predicts**: 16 links in contact, the lagrangian sampling pointwise and the eulerian smearing over
   a 4h band. B2 is now the leading candidate.
4. **Sheet truncation is not fixable by any sample-shift.** If the eulerian is to be used on thin
   bodies at all, the measure itself needs correcting (divide by the measured `A_coarea/A_true`?
   normalise the delta per surface patch?) — or `eps_body` must shrink below the body half-thickness,
   which the slab table says restores the measure exactly (`t ≥ eps_body → 1.000`). **Cheap test that
   nobody has run: sweep `eps_multiplier` (the delta WIDTH — trap #4 no longer applies, B1 decoupled
   it from the offset) and watch `A_coarea/A_tri` → 1 and the `(0,0)` pressure ratio → 1.**

#### ⚠️ Then redo the core comparison — STILL TO DO, and read §C "the two offset laws" first

Set BOTH methods to `(p_off, f_off) = (0, 2h)`, and again at `(0,0)`, `(h,h)`, `(2h,2h)`, `(0,h)`.
Sampling location is now eliminated as a variable, so any residual difference is **the readout
itself** — surface integral over an explicit triangulation vs a smeared volume integral over an
iso-surface. Nobody has measured that number yet.

**But do not expect matched sampling to make them agree, and do not read a residual gap at nonzero
offset as "the parity gap".** §C shows the two readouts obey *different exact laws* in the offset,
so they provably diverge as the offset grows even with identical knobs. **The only setting at which
"they should agree" is a meaningful prediction is `(0,0)`** — there both reduce to the true surface
integral, and there they do agree (0.999× on the sphere oracle). At `(0,0)` on the *live* fish the
comparison is finally like-for-like; that is the number to get first.

### B2. The proximity hypothesis — two simplified bodies, swept separation

**Hypothesis (from the user, untested):** bodies very close together produce locally high divergence
and sharp pressure variation between them. The **lagrangian samples pointwise** and picks that up;
the **eulerian smears over a 4h-thick band** and averages it away. The zebrafish is 16 links in
contact — if this is real, it is a large part of the residual, and it would be invisible in every
single-body benchmark run so far.

Design (build it in this folder, e.g. `two_body_proximity.py`):
- Two **analytical** bodies (spheres or cylinders — `BodyAnalytical`, so `|∇φ|=1` exactly and
  `force_delta_order` stays out of it). Well-resolved, `R/h ≳ 10`, to keep the §10 thinness effect
  out of the picture — you want to isolate proximity, not re-measure thinness.
- Sweep the gap `d/h ∈ {0.5, 1, 2, 4, 8, 16, ∞}` (∞ = a single isolated body, the control).
- At each gap, compare eulerian vs lagrangian **at matched sampling** (needs B1), per body, split
  into pressure and viscous channels.
- Two flow regimes at least: (i) imposed analytic field with a known answer where possible,
  (ii) a real coupled/impulsive flow.
- **Prediction to falsify**: the gap grows as `d/h → 0`, and it lives in the **pressure** channel.
  If the divergence is instead flat in `d`, the hypothesis is dead — say so and move on.
- Watch for the confound: as `d → 0` the two bodies' bands overlap, so the eulerian's softmin
  partition (`deltaH`) and the union SDF start doing work. Test `ndelta` and `deltaH` both —
  `deltaH` exists precisely to handle union surfaces and may behave very differently here.

### B3. Re-validate against published references, both methods, both channels

Once B1 lands, redo these with matched sampling — they are the only *ground truth* available:
- `validation/cylinder_drag_2d/` (Koumoutsakos & Leonard, Re=550) at 512² and 1024². The
  **decomposed** viscous/pressure plot is the point. Known: eulerian viscous 0.89× lagrangian at
  512², 0.98× at 1024².
- `examples/single_sphere_drop_gazzola/` vs Namkoong `U_t = −0.025`. Known: lagrangian 99.3%,
  eulerian 103.6%. Terminal velocity is a clean integral test.
- `examples/single_sphere_drop_coquerelle_3d/` — a d×ν sweep varying R/h. **Set
  `bdim_mu0_projection: False`** (see `project_coquerelle_regression_f650945`).

### B3a. The `F_penal` arbiter — an INDEPENDENT net-force reference (new, from Verma et al.)

**This is arguably the most valuable thing in that paper, and it is not a readout — it is the
referee.** Verma does not *use* the volume/momentum method as a force readout; he uses it to
**validate** his surface forces. Gazzola et al. 2011's penalization force (Verma Eq. 19)

```
F_penal = ∭ λ·χ·u dV          (u_s = 0)
```

is the net force from the penalty term over the whole domain, and Verma's Figs 6 and 12 validate
the surface readout against it (`a_SF = (F_P + F_ν)/M` vs `a_penal = (u_CM^{n+1} − u_CM^{n−1})/2Δt`;
Fig 13 repeats it across 4 grid resolutions). It gives **only the net force, no distribution** —
which is why Verma couldn't use it as his readout, but it is **exactly what we need**, since the
coupling only consumes the net force/torque.

Why this matters here: **every arbiter we currently have is weak.** Closure is "a screen, not a
scoreboard" (§C — the readouts closed within 2 points while their speeds differed 35%);
`verify_energy_balance.py` cannot discriminate at all; the oracle has no BDIM band (trap #0). A
BDIM analogue of `F_penal` — the momentum imparted by the mu0 blending, integrated over the body —
would settle "which readout is right" **on the live fish, with the band present**, which nothing
else does.

Also note Verma §2.3.2 explicitly acknowledges the control-volume family (refs 38/39 = Mimeau 2015;
**Noca, Shiels & Jeon 1999** — the classic) and rejects it *only* because *"the integral methods
yield just the net force, or net acceleration"*, which is not a problem for us. So the
control-volume readout floated above is a known, published technique — not a speculative idea — and
`F_penal` is its cheapest instance.

### B4. Only then, revisit the zebrafish

With matched sampling, re-run the arbitration (`gen_zfish_readout_arbitration.py` +
`zfish_pbdim_closure.py` + `zfish_swim_speed.py`) and decide whether the residual gap is proximity
(B2), thinness (§10.2), or something still unidentified.

**Note `off≈h` is now a dead recommendation to *defend*, though it may survive as a number.** It was
`(h, h)` — it moved the pressure sample, which neither M&W nor Verma does, and §C shows there is no
common offset at which both channels are "right". The candidate that replaces it is **`(0, 2h)` =
Verma** (§B1a step 1), which has never been run. If `(h,h)` still beats `(0,2h)` empirically, that
is an interesting result demanding an explanation — not a vindication of `(h,h)`.

---

## C. Established, with confidence levels

> **⚠ AMENDED 2026-07-17 (session 3) by §B1b — read that first.** Three items below are now
> corrected by measurement:
> 1. **"Verma settles the design question"** — settles the *design*, but **`(0,2h)` has now been run
>    on the fish and is NOT its best setting** (79.3% closure, 57.9 mm/s, off the plateau). Moving
>    the friction sample out makes this fish *slower*. Verma's own caveat (his §2.3.2: the lift is
>    for rigid bodies, "not necessary … for self-propelled swimmers") may simply apply.
> 2. **"`J ≈ 2.89` for the fish"** — **wrong. Measured `A(2h)/A(0) = 1.169`.** That estimate assumed
>    sphere-like curvature; the fish is a *sheet* (κ≈0 in the thin direction), so the Steiner
>    Jacobian is small and explains little of the gap.
> 3. **"the two readouts differ ONLY by an area Jacobian"** — true **on the sphere oracle**, false on
>    the fish: at matched `(0,0)`, where `J ≡ 1` by construction, they still differ by **1.327×**
>    (pressure) and the lateral component flips sign. A second mechanism — **sheet truncation of the
>    delta measure** (§B1b) — dominates on this geometry.

**Trust — VERMA ET AL. (2017) SETTLES THE DESIGN QUESTION (new, 2026-07-17):**

`Verma, Abbati, Novati, Koumoutsakos, "Computing the force distribution on the surface of complex,
deforming geometries using vortex methods and Brinkman penalization", Int J Numer Meth Fluids
2017;85:484-501` (doi 10.1002/fld.4392). Same lab as Gazzola/Koumoutsakos, same penalization family
as BDIM, and it is **the published statement of the lagrangian readout we implement**. Read §2.3.

Their method, stated explicitly:
- **Pressure** (§2.3.1): *"These exact surface coordinates (x_surf) ... are used as target
  locations ... for computing P(x_surf)"* → **sampled at the true surface, no lift**.
- **Viscous** (§2.3.2): *"∇u is computed on a 'lifted' body surface via nearest-neighbor
  interpolation. The lifted surface is formed by moving the original solid surface outward along
  the surface normal. Empirical tests indicate that a distance of **2h** ... yields the best
  accuracy for determining ∇u."*
- **The integral** (right after Eq. 15): *"**The closed integral in Equation 15 is performed over
  the surface of the solid object.**"*

⇒ **Verma = `(sample_offset_pressure, sample_offset_friction) = (0, 2h)`, integrated over the TRUE
surface.** That is *exactly* the configuration B1 made expressible and **it has never been run
here**: the old single knob forced `(2h, 2h)`, and §10's `off≈h` recommendation was `(h, h)` —
**both move the pressure sample, which no paper does**.

Three consequences:
1. **The lifted sample is an ESTIMATOR FOR THE INTEGRAND, not a relocation of the integral.** The
   objection "an integral over S₀ of a function evaluated off S₀ is not meaningful" is natural but
   answered: we want `∮_{S₀} σ_wall·n dS`, and the grid cannot supply `σ_wall` — Verma's diagnosis
   is *"the velocity magnitude and velocity-derivatives are essentially zero next to the solid
   boundary"*, which he attributes to the smoothing of χ (our BDIM band does the same). So `∇u` is
   read where it is resolved and used **as** the wall value. It is a wall model. The domain never
   moves. **Moving the markers instead changes the domain and computes a different physical
   quantity** (see the REFUTED entry above).
2. **M&W's delta-shift is not wrong, it is out of regime.** Shifting the delta is harmless when
   `J ≈ 1` (cylinder, `eps/R ≈ 0.08` → J ≈ 1.17) and fails at `eps/R ≈ 0.7` (fish → J ≈ 2.89).
   This is the same `eps/R` story §10 already told, now with a mechanism.
3. **⚠ Verma says the lift is for RIGID bodies and is NOT needed for deforming swimmers:** *"such a
   correction is not necessary in the case of temporally deforming shapes (such as self-propelled
   swimmers)"*. Our fish **is** deforming yet behaves like his rigid case (§10: `off=0` is its worst
   setting, 74.4% closure ⇒ under-read, exactly the failure the lift exists to cure). Untested
   hypothesis for the discrepancy: our band is `4h` wide against `R = 2.86h`, whereas his χ smooths
   over **2 cells** on a far better-resolved body. **Do not treat "2h" as transferable without
   checking this.**

**Trust — THE TWO READOUTS DIFFER ONLY BY AN AREA JACOBIAN (new, 2026-07-17; exact-answer tests):**

This is the sharpest statement of the offset problem, and it supersedes the "cubic vs linear"
framing below (that was true but field-specific).

**The lagrangian does not move its triangulation when `sample_offset > 0`.** The mesh stays welded
to the skin: `A` (triangle area), `n` (normal) and the moment arm `r = q_centroid − com` are all
taken at the **unshifted** centroid (`cuda/lagrangian_forces.cu:194-260`). Only the *field lookup*
moves out along `n`. **This is correct and it is what the literature does** — see "Verma et al.
settles the design question" below. The eulerian does not do this: it shifts the *delta*, so the
measure moves with the sample. (That is a choice, not a necessity — see the fix in §B1a.)

So both readouts evaluate the **same traction at the same point**, and differ only by the area
element:

```
lagrangian(s) = ∮_{S₀} T(x+s·n)          dS₀
eulerian(s)   = ∮_{S₀} T(x+s·n) · J(x,s) dS₀      J = (1+s·κ₁)(1+s·κ₂)
```

`J` is the Steiner area Jacobian — **pure geometry, no fluid in it**. On a sphere `J = (1+s/R)²`.

**Measured, and field-independent** (sphere oracle R/h=9.4; pressure and viscous channels agree to
4 d.p., which is the signature of a purely geometric factor):

| s/h | eul/lag pressure | eul/lag viscous | (1+s/R)² |
|---|---|---|---|
| 0 | 1.0044 | 1.0044 | 1.0000 |
| 1 | 1.2320 | 1.2320 | 1.2241 |
| 2 | 1.4758 | 1.4758 | 1.4708 |
| 4 | 2.0374 | 2.0374 | 2.0321 |

**⇒ REFUTED (tested, 2026-07-17): "build the lagrangian triangulation ON the offset surface
instead of sampling outward".** Natural idea, but it *is* the eulerian: a sphere mesh built at
radius `R+s` and sampled at offset 0 reproduces `eulerian(s)` to 0.3-0.6% (0.1361 vs 0.1370 at
s=h; 0.1792 vs 0.1798 at 2h; 0.2911 vs 0.2918 at 4h). It would import the eulerian's error into the
one readout that lacks it: at `s=2h` it is **+78%** vs true, where the current lagrangian is +21%.

Why, physically — **`∮_{S_s} σ·n dS` is NOT the body force**:
```
∮_{S_s} σ·n dS − ∮_{S₀} σ·n dS = ∫_shell ∇·σ dV = ∫_shell ρ·Du/Dt dV
```
i.e. **the inertia (added mass) of the fluid in the shell**. For the fish the shell between R=2.86h
and R+2h holds ~2.9× the body's own volume and is the most accelerated fluid in the domain — it is
what the tail is flapping. (In the oracle `∇·σ = G·x̂` is constant, so the excess is exactly
`G·V_shell` — that is *why* it tracks volume inflation.) The two designs carry **different errors**:
mesh-on-skin costs an extrapolation error `O(s·∂σ/∂n)`; mesh-on-offset costs the shell inertia,
which grows with shell volume and dominates on a body thinner than the offset.

**⇒ The legitimate version of that idea: a CONTROL-VOLUME / momentum-balance readout.** Integrate
over `S_s` but add back what was dropped — the momentum flux `ρu(u·n)` through `S_s` and the
unsteady `d/dt ∫ρu dV` (get the signs from a standard derivation; the point is the terms exist).
This is the standard PIV force-estimation technique. **Its real appeal: it never reads the corrupted
band at all**, which is the entire reason the offset exists. Cost: needs `Du/Dt` in the shell —
expensive and noisy. Worth considering as a third readout rather than a patch to either existing one.

**⇒ THE FIX (untested, but this is the deliverable): SHIFT THE SAMPLE, NOT THE DELTA.**

```
today:  δ_ε(φ − s) × σ(x)         ← shifts the DELTA ⇒ the measure moves ⇒ Jacobian error
fix:    δ_ε(φ)     × σ(x + s·n)   ← shifts the SAMPLE ⇒ the measure stays on S₀
```
The fixed form is **`∮_{S₀} σ(x+s·n)·n dS₀` — mathematically identical to the lagrangian at
`(0, s)`, i.e. to Verma.** So it delivers **parity AND accuracy at once**, which is the actual goal;
they are not a trade-off. The eulerian already has `n` at every band cell (it needs it for the
traction), so this is a sample-point interpolation of the strain tensor, not new machinery. Sites:
`cuda/eulerian_forces.cu` (2-D + 3-D) + both CPU twins; the interpolators already exist
(`sdf_sample_dispatch_*` / `lf_sample_*`). Gate on the sphere oracle, where the answer is exact.

*Superseded lead (do not pursue first):* dividing by `J` via a curvature weight `(1 − s·κ₁ˢ)(1 −
s·κ₂ˢ) ≈ 1 − s·Δφ` also cancels the Jacobian and needs only a grid Laplacian — but it is **singular
exactly where it hurts** (`(1 − s·κˢ) → 0` for features thinner than the offset) and assumes
`|∇φ|=1` (the fish band has `mean|∇φ| = 0.82`). The sample-shift is the same fix without either
weakness.

---

**Trust — the two offset laws (field-specific corollary of the above):**

On the sphere oracle (`R`, linear pressure `p = -G·X`, shear `u = c·Y²`), with `A(s) ≡ (4/3)πR²(R+s)`:

| readout | pressure at offset `s` | viscous at offset `s` | growth |
|---|---|---|---|
| **lagrangian** | `G·A(s)` | `2νρc·A(s)` | **LINEAR** in `s` — fixed surface, only the sample point moves |
| **eulerian** | `G·(4/3)π(R+s)³` | ∝ `(R+s)³` | **CUBIC** in `s` — the iso-surface *itself* moves |

Both reduce to the same exact answer at `s=0` (`A(0) = V`). So:

- **Matched sampling does NOT imply agreement, and cannot.** Pinning both readouts to a common
  nonzero offset `s` makes them diverge by a *known, exactly-predictable* factor

  ```
  eulerian(s) / lagrangian(s)  =  (1 + s/R)²
  ```

  — not by "the parity gap". Measured against the formula on the sphere oracle (R/h=9.4):

  | s/h | measured eul/lag | (1+s/R)² |
  |---|---|---|
  | 0 | 1.004 | 1.000 |
  | 1 | 1.232 | 1.224 |
  | 2 | 1.476 | 1.471 |
  | 4 | 2.037 | 2.032 |

  At `s=2h` on a body with `R≈2.86h` (the fish) that factor alone is **2.89×**. Any matched-offset
  comparison must divide this out or it measures nothing.
  **⚠ REFUTED FOR THE FISH (§B1b, measured): `A(2h)/A(0) = 1.169`, not 2.89.** The `(1+s/R)²` form
  assumes sphere-like curvature in *both* principal directions. The fish is a **sheet**: κ≈0 in the
  thin direction, so its Steiner Jacobian is ~1.17 and the Jacobian explains little of its gap. The
  formula is still exact on the sphere oracle, where it was measured.
- **`s = 0` is the only setting where "they should agree" is a real prediction.** Get that number
  on the live fish first.
- This also **kills a tempting misreading of the `off≈h` recommendation**: `off≈h` cannot be
  "moving both readouts to the right place", because there is no common place where both are right
  except 0.
- The eulerian's *pressure* channel obeys the same cubic inflation the viscous channel does — it
  was simply never measurable before B1 (p was pinned at `φ=0`). This is a **second, independent
  instance of the same defect** (§10's enclosed-volume over-read), not a new one.

**Trust (measured against exact answers, reproducible in ~1 s):**
- The lagrangian *formula* is exact: 0.999× the analytic answer at **every** R/h from 3.0 to 12.6.
  Thinness does not touch it — its area comes from the mesh, not the grid.
- The eulerian's viscous over-read tracks the **enclosed-volume ratio** of `{φ = eps_solver}`. At the
  production setting it is 1.26× at R/h=25 and ~5.8× at R/h=3. Confirmed to 3 decimals against
  `((R+s)/R)³` on a clean sphere.
- `deltaH` viscous is **bit-identical** to `ndelta` viscous — deltaH only replaces the pressure
  readout. It is not a candidate fix for the viscous channel.
- **`zero_pressure_inside=True` HALVES the eulerian pressure force** (measured 0.513×; the pressure
  delta band spans both sides of the surface). **Audit needed**: `salamander/gen_configs_*`,
  `zebrafishsim/gen_configs_pd_3d.py`, `_1guillasim/gen_configs_pd_vary_f.py` set it True.
- `force_delta_order=2` was **inverted** (divided by `|∇φ|` where coarea multiplies) — **fixed**
  (`edf417b`), pinned by `test_delta_order2_applies_the_coarea_factor`. Inert at `|∇φ|=1`.

**Trust (geometry, directly measured on the live snapshot):**
- Fish bbox **87 × 19 × 21 cells**; **max inscribed radius 2.86h**, median interior depth **0.61h**;
  `eps_body = 2h` ⇒ **95.1% of interior cells sit inside the band**. `eps/R ≈ 0.7` (cylinder: 0.08).
- The triangulation is **clean**: `∮x·n dS/3 = 0.997 × V_union` (links tile, no overlap), and 100% of
  174,436 triangle centroids lie within 1h of the union surface. **No buried faces** — an earlier
  claim of 1.57× buried area was an artifact and is retracted.

**Trust (methodological):**
- **Closure is a screen, not a scoreboard.** On the fine grid the readouts closed within 2 points
  (105.3% vs 107.1%) while their speeds differed 35%.
- `verify_energy_balance.py` cannot arbitrate readouts (identical to 5 s.f. for all of them).

**Do NOT trust (confounded — §A):** the 3.96× live viscous ratio, the 0.41× net thrust, the 45%
speed gap as a *readout* comparison, and the mechanism behind `off≈h`. B1 fixed the *code*, not
these *numbers* — they were measured with the old confounded knobs and must be re-measured.

**⇒ RESOLVED 2026-07-17 (§B1b): the 45% speed gap was ENTIRELY sampling location.** At matched
`(0,0)` the readouts swim the fish at 0.999× of each other. The eulerian's *own* speed drops 36%
when only its friction offset moves 0 → 2h. And the mechanism behind `off≈h` is now known to be the
**pressure** channel, not the friction channel — the opposite of what §10 assumed. The 3.96× and
0.41× remain un-re-measured, but they are now known to be offset artifacts, not readout properties.

**Open:**
- **Why the lagrangian is noisier in coupled runs — the ORIGINAL complaint that motivated the
  eulerian, still never measured.** Best remaining hypothesis: `off=0` sits on a steep part of the
  offset curve, so pose jitter maps into force jitter; the plateau at `off≥0.5h` is stationary and
  should be quieter. Cheap to test on snapshots from consecutive steps. The attractive
  "buried joint caps sampling unconstrained interior pressure" explanation is **dead** (no buried faces).
- **A live CPU/CUDA physics divergence** (`test_python_eulerian_force_path_{cpu_regression,cpu_eq_gpu}`,
  ~4e-3 relative). **Pre-existing** — verified bit-identical at 39fb3b4 in a worktree, so the
  `bdim_forcing→bdim_apply` refactor is physics-neutral. It is upstream of the readout (the test
  drives the *python* branch), so look in projection / `bdim_apply` / Poisson. Independent bug.
- `forces_method2_3d` full-grid branch reads `comp.sdf_vals` with no fallback (`forces.py:800`).
- Neither readout is grid-converged on the zebrafish: truth bracketed ~48-74 mm/s, and first-order
  Richardson is inconsistent between the two sides (53.7 vs 68.9) ⇒ not asymptotic. The order could
  not be measured (2h explodes; h/4 needs 537M cells > 16 GB).

---

## D. Traps — every one of these cost a session

0. **THE ORACLE HAS NO BDIM BAND — it measures the COST of the offset, never its BENEFIT.**
   `_oracle_scene` imposes an analytic field, so `σ` at `φ=0` is *clean* there and every oracle
   result says "offset = pure error, use s=0". The live run says the **opposite** (§10: `off=0` is
   the *worst* setting, 74.4% closure) because there `s=0` reads stress corrupted by the band
   (`ε(u_blend) ≈ μ0·ε(u_fluid)`). Both are true. The `[0.5h, 3h]` speed plateau is where the
   offset's cost (geometric + extrapolation, which the oracle measures) trades off against its
   benefit (escaping contamination, which only a live run measures). **Never conclude "use s=0"
   from an oracle number, and never conclude "the offset is free" from a live one.**
1. **Rebuild the extension.** A stale `_C.so` hid two bugs for a session and made a physics change
   look like a refactor regression.
2. **A parity test is only worth its oracle.** Every force test in the repo except
   `test_oracle_*` / `test_delta_order2_*` is CPU-vs-GPU parity or a frozen snapshot — all of them
   pass happily when both sides are wrong.
3. **`_build_surface_3d` leaves `tri_*_world` in a bbox-centred LOCAL frame.** Only
   `BDIMhandler._refresh_lagrangian_tris_3d` moves it to world, and only on lagrangian runs. Capture
   it at the force call site (as `zfish_snapshot_hook.py` does) or you will silently sample the wrong
   locations and get a plausible-looking wrong answer. Validate any triangulation with `∮x·n dS = 3V`.
4. **`eps_multiplier` is not a free knob for the delta WIDTH** — but since B1 it no longer sets the
   eulerian's viscous shift. `eps_multiplier` now sets `eps_body` (the delta width) and the
   *default* value of the friction offset only; `sample_offset_friction_cells` overrides the latter
   independently. Sweep the shift with that, never by moving `eps_multiplier`.
5. **Offsets in metres do not survive a grid change.** Use cells (§B1).
5b. **`solver.eps` no longer feeds the force readout — mutating it at runtime sweeps NOTHING.**
   Pre-B1, `shift_sweep_2d.py` swept the viscous shift by assigning `solver.eps = s*h` between force
   calls (it doubled as `eps_solver`). That now silently does nothing to the readout. Assign
   `solver.eul_sample_offset_friction` / `solver.lagr_sample_offset_{pressure,friction}` instead
   (`shift_sweep_2d.py` has been updated). Note the *native* path caches these floats via
   `_cached_float` for CUDA-graph key stability, so runtime mutation only takes effect on the python
   branch — set them via config for a native run.
6. **Regime B is not the force path.** `streaming_sdf_regime_b.cu` is the body-update/streaming
   stage; forces are `streaming_sdf_forces_post_{2,3}d` in `cuda/eulerian_forces.cu`.
7. **`save=True` is required for `diagnostics.h5`/`drags.h5`** — it is what creates `save_path`
   (`solver.py:641`). It does **not** dump fields in a FARMS-coupled run (nothing calls
   `save_results`), so it is cheap. Without it the closure arbiter has no data.
8. **Coarse grids explode.** At `R/h ≈ 1.4` the fish is ~3 cells thick and MuJoCo dies with
   `mjWARN_BADQACC`. Not a bug.
9. **`pgrep -f <script>` matches your own waiter.** Cost 5 minutes of idle GPU here.

---

## E. Suggested acceptance criteria

The job is done when:
1. Both readouts, at **identical** `(sample_offset_pressure, sample_offset_friction)`, agree to a
   stated tolerance on a well-resolved single body — and the residual is explained.
   **Amended after B1:** state this at `(0,0)`. At nonzero matched offsets the two obey different
   exact laws (§C "the two offset laws") and provably do not agree, so "agreement" is only a
   meaningful acceptance test at zero. At nonzero offsets the testable claim is instead that the
   measured ratio matches the predicted `(1 + s/R)²`; a deviation from *that* is the real residual.
2. The remaining zebrafish gap is **attributed** (proximity / thinness / something else) with a
   falsification test for each.
3. Both readouts reproduce K&L and Namkoong within stated bounds, decomposed by channel.
4. The recommendation for the production zebrafish config is backed by a comparison in which
   sampling location is not a free variable.
