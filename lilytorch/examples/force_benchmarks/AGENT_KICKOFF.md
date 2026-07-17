# Agent kickoff — force-readout parity + accuracy

Paste the block below as the agent's task. Everything it needs beyond that is in
`HANDOFF_NEXT_AGENT.md`, which is the source of truth — this file is only the driver.

---

## TASK

The repo has two hydrodynamic force readouts, `force_method: eulerian` and `force_method:
lagrangian`. They disagree badly on the zebrafish (`examples/zebrafish_ki_project/
gen_configs_pd_3d_slow_fast.py`, which swims ~45% slower under eulerian). **Two goals, and they are
NOT a trade-off — there is one configuration where both hold:**

1. **Parity** — the two readouts should agree.
2. **Accuracy** — both should be right, validated against exact/published answers.

**Read `lilytorch/examples/force_benchmarks/HANDOFF_NEXT_AGENT.md` first: the `STATUS` block, then
§B1b (RESULTS — what is actually true), then §C (established/refuted).**

> **⚠ UPDATED 2026-07-17 (session 3): §B1a steps 0-2 are DONE and step 2's central prediction was
> FALSIFIED. The plan below is now HISTORY — read §B1b for what replaced it.** In short: the ~45%
> speed gap was **entirely sampling location** (at matched `(0,0)` the readouts agree to 0.999× on
> live swim speed); Verma's `(0,2h)` **has now been run and is not this fish's best setting**; the
> `off≈h` plateau turns out to be bought by the **pressure** offset, which no paper does; and the
> eulerian's `(0,0)` gap is **sheet truncation of its delta measure** (0.636× the true area),
> which **step 3 below would not fix**. Do not start step 3 as designed. The open questions and
> their falsification tests are listed at the end of §B1b.

## BEFORE ANY OTHER COMMAND

```
python setup.py build_ext --inplace
```
A stale `_C.so` has hidden bugs for entire sessions here, and made a physics change look like a
refactor regression. This is trap #1 for a reason.

## STATE OF THE TREE — read this or you will misattribute failures

- Branch `cuda_native_port`. **The B1 work (split sampling offsets) is LANDED but UNCOMMITTED**,
  and it sits alongside *other* people's uncommitted work (`native.py`, `facade.py` deleted,
  `advection.py`, `BDIMhandler.py`, …). Do not `git stash`/`checkout` casually. Do not commit
  unless asked.
- **The suite is 370 passed / 10 failed, and all 10 failures are PRE-EXISTING**, verified by
  re-running with `lilytorch/src/forces.py` + `lilytorch/src/solver.py` stashed and getting
  bit-identical numbers. They are `test_python_eulerian_force_path_*` (2), `test_two_phase::
  *uniform*` (3), `test_whole_step_capture_native` (5). **Do not attribute them to your change, and
  do not "fix" them as part of this task.** If you touch force code and a *new* failure appears,
  use the same stash technique to prove whether it is yours.
- `HEAD` is NOT a usable baseline: `test_forces.py` at HEAD cannot even import (it needs the
  uncommitted `native.py`). Compare against the working tree minus your own edits, not against HEAD.

## THE ORIGINAL PLAN — steps 0-2 DONE, step 3 REFUTED (kept for context; details in §B1a/§B1b)

**Step 0 — plumbing.** `gen_zfish_readout_arbitration.py` exposes only the LEGACY single knob
(`ZFISH_LAGR_OFFSET_CELLS` → `lagrangian_sample_offset`, ~lines 74-78). It **cannot express
`(0, 2h)`**, which is the whole point of step 1. Add `ZFISH_OFF_P_CELLS` / `ZFISH_OFF_F_CELLS` →
`sample_offset_{pressure,friction}_cells`; keep the legacy vars working (unset ⇒ unchanged); print
the resolved `(off_p, off_f)` in the `[arb]` banner.

**Step 1 — run the fish at `(off_p, off_f) = (0, 2h)`.** This is Verma et al. 2017's published
method and **has never been run here** (the old single knob forced `(2h,2h)`; §10's `off≈h` was
`(h,h)` — both move the *pressure* sample, which no paper does). Arbiter:
`zfish_pbdim_closure.py` + `zfish_swim_speed.py`, as in §10.10. Compare against the §10.10 table.

**Step 2 — measure eul-vs-lag at `(0, 2h)`, decomposed by channel.** This is the number nobody has.
**Prediction to falsify:** pressure ratio ≈ 1; viscous ratio ≈ the area Jacobian `J`. If the
*pressure* ratio is not ≈1, something unidentified is live — that is the real finding, chase it.

**⇒ CHECKPOINT. Report steps 1-2 before touching any kernel.** Steps 3-4 are expensive and their
design depends on what step 2 says.

**Step 3 — fix the eulerian: shift the SAMPLE, not the DELTA** (§C "the fix"):
`δ_ε(φ−s)×σ(x)` → `δ_ε(φ)×σ(x+s·n)`. That form is mathematically identical to the lagrangian at
`(0,s)`, so it delivers parity *and* accuracy. Gate on the sphere oracle (exact answer) before the
fish. Sites: `cuda/eulerian_forces.cu` 2-D+3-D + both CPU twins — keep them single-source.

**Step 4 — build the `F_penal` arbiter** (§B3a): an independent net-force reference that works on
the live fish with the band present, which nothing else we have does.

## RULES OF ENGAGEMENT

- **Falsify, don't defend.** Your job is not to confirm the prior conclusions — several have already
  been retracted (a "1.57× buried area" claim, and my own "cubic vs linear" framing). §A lists
  numbers that are CONFOUNDED and must not be cited: the 3.96× viscous ratio, the 0.41× net thrust,
  the 45% speed gap *as a readout comparison*, and the mechanism behind `off≈h`.
- **A parity test is only worth its oracle.** Nearly every force test in this repo is CPU-vs-GPU
  parity or a frozen snapshot — they all pass happily when both sides are wrong. Prefer the
  `test_oracle_*` / `test_split_offsets_*` family, which check against exact answers in ~1 s.
- **The oracle has no BDIM band** (trap #0). It measures the offset's *cost*, never its *benefit*.
  Never conclude "use s=0" from an oracle number, nor "the offset is free" from a live one.
- **Report what actually happened.** If a run explodes, say so with the output. If a prediction
  fails, that is a result, not a setback. Do not fabricate or extrapolate numbers you did not
  measure.
- Read §D (traps) before running anything long. Several of those cost a full session each.
- Write sim plots/frames to `/data/andreaferrario/ns_data/`. No `Co-Authored-By` trailer in commits.

## DONE WHEN

§E, as amended: both readouts at identical `(p_off, f_off)` agree to a stated tolerance **at
`(0,0)`** and the residual is explained; at nonzero offsets the measured ratio matches the predicted
`J` (a deviation from *that* is the real residual); the remaining zebrafish gap is **attributed**
with a falsification test for each candidate; and the production recommendation is backed by a
comparison in which sampling location is not a free variable.
