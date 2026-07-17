# Handoff: establish force-readout parity properly

**Branch:** `cuda_native_port` · **Written:** 2026-07-17 · **Predecessor evidence:**
`lilytorch/milestones/force_readout_agreement_handoff.md` (§§1-11; §10 is the current state)

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

---

## A. THE FLAW IN THE EXISTING COMPARISON — read before trusting any §10 number

The two readouts do not sample in the same places, and the previous work varied both at once:

|  | pressure sampled at | viscous σ sampled at |
|---|---|---|
| **eulerian** | `φ = 0` — **never shifted** | `φ = eps_solver = eps_multiplier·h` |
| **lagrangian** | `φ = lagrangian_sample_offset` | `φ = lagrangian_sample_offset` — **same knob** |

Verified in code: `cuda/eulerian_forces.cu:494-499` (`d_visc = s_cc_body - eps_solver` for the
viscous delta; `s_cc_body` unshifted for the pressure delta) and `cuda/lagrangian_forces.cu:208-243`
(one `sample_offset` moves the query point for the strain tensor **and** `p`).

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

### B1. Split the sampling offsets — do this FIRST; everything else depends on it

Give **both** readouts two independent knobs:

```
sample_offset_pressure     # where p is read,  in metres (or cells — see below)
sample_offset_friction     # where σ is read,  in metres
```

- **eulerian**: today `eps_solver` shifts only the viscous band and pressure is pinned at `φ=0`.
  Generalise: `delta_pres` centred at `φ = sample_offset_pressure`, `delta_visc` at
  `φ = sample_offset_friction`. Current behaviour = `(0, eps_multiplier·h)`, which **must remain the
  default** (it is Maertens & Weymouth, and it is what every existing run and reference result used).
- **lagrangian**: split the single `sample_offset` into the two. Current behaviour = `(off, off)`.
  Default `(0, 0)` to preserve today's results.

Sites (keep single-source — CPU twin, CUDA, python path all carry the convention):
`src/csrc/cuda/eulerian_forces.cu` (2-D + 3-D kernels), `src/csrc/ops_2d.cpp`, `src/csrc/ops_3d.cpp`,
`src/csrc/cuda/lagrangian_forces.cu` (+ `lagrangian_forces_2d.cu`), `src/csrc/lagrangian_forces_cpu*.cpp`,
`src/forces.py`, `src/solver.py`, `examples/base_sim_config.py`. Op schemas in `src/csrc/ops.cpp`.

**Gate**: existing `tests/test_forces.py` must pass unchanged (defaults preserved, bit-exact), plus a
new test that `(p_off, f_off)` reproduces the old single-knob behaviour when both are equal.

**Consider expressing offsets in CELLS, not metres.** The right value scales with `h`; a value fixed
in metres means a different thing on every grid, and a grid-convergence study silently changes the
physics. `gen_zfish_readout_arbitration.py` already works around this with
`ZFISH_LAGR_OFFSET_CELLS`.

**Then redo the core comparison**, which is the whole point: set BOTH methods to
`(p_off, f_off) = (0, 2h)`, and again at `(0,0)`, `(h,h)`, `(2h,2h)`, `(0,h)`. Any residual
difference is now **the readout itself** — surface integral over an explicit triangulation vs a
smeared volume integral over an iso-surface — with sampling location eliminated as a variable. That
number is the real "parity gap", and nobody has measured it yet.

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

### B4. Only then, revisit the zebrafish

With matched sampling, re-run the arbitration (`gen_zfish_readout_arbitration.py` +
`zfish_pbdim_closure.py` + `zfish_swim_speed.py`) and decide whether `off≈h` survives and whether
the residual gap is proximity (B2), thinness (§10.2), or something still unidentified.

---

## C. Established, with confidence levels

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
speed gap as a *readout* comparison, and the mechanism behind `off≈h`.

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

1. **Rebuild the extension.** A stale `_C.so` hid two bugs for a session and made a physics change
   look like a refactor regression.
2. **A parity test is only worth its oracle.** Every force test in the repo except
   `test_oracle_*` / `test_delta_order2_*` is CPU-vs-GPU parity or a frozen snapshot — all of them
   pass happily when both sides are wrong.
3. **`_build_surface_3d` leaves `tri_*_world` in a bbox-centred LOCAL frame.** Only
   `BDIMhandler._refresh_lagrangian_tris_3d` moves it to world, and only on lagrangian runs. Capture
   it at the force call site (as `zfish_snapshot_hook.py` does) or you will silently sample the wrong
   locations and get a plausible-looking wrong answer. Validate any triangulation with `∮x·n dS = 3V`.
4. **`eps_multiplier` is not a free knob** — it sets both the delta width and the eulerian's viscous
   shift, and the two readouts want opposite values.
5. **Offsets in metres do not survive a grid change.** Use cells (§B1).
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
2. The remaining zebrafish gap is **attributed** (proximity / thinness / something else) with a
   falsification test for each.
3. Both readouts reproduce K&L and Namkoong within stated bounds, decomposed by channel.
4. The recommendation for the production zebrafish config is backed by a comparison in which
   sampling location is not a free variable.
