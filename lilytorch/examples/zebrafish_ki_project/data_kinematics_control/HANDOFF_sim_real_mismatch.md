# Handoff: zebrafish sim-vs-real speed & turning mismatch (ep248 slow)

Written 2026-07-20. Context: `compare_sim_real.py` (in `keypoints/`) compares the
PD-position-controlled zebrafish sim against real ep248 slow kinematics. Two
issues persist after this session's work. This doc records what was ruled out
so you don't repeat it, and the leading root cause.

## The two issues (user's words)
1. **Speed:** the real experiment speed shows **massive oscillations that the sim
   speed does not**. Mean speeds actually match (~6 BL/s).
2. **Turning:** the **sim turns slightly right; the real is almost straight.**
   Improved ~15% this session but not resolved.

## STATUS 2026-07-21 (SESSION 3) — Issue 2 REFRAMED: it is a SLOW-DRIFT failure, not a bias
Five results. **Issue 1 is SETTLED by a momentum budget — section (0), read that
first.** Then two hypotheses retired for issue 2 (arena walls, model left/right
asymmetry), a reframing of what issue 2 actually is (section (b)), and a sizing
number for the fix.

### (0) ISSUE 1 SETTLED: the TRACKED MIDLINE CHANGES LENGTH, and a fish's cannot
User's objection to every earlier explanation was that none carried a number,
and the band that matters is **20-45 Hz**, where the experiment swings **15.8
BL/s peak-to-peak against the sim's 1.44** (rms 3.15 vs 0.31). Answer:

> **The tracked arc length breathes 3.87% rms / 19.9% peak-to-peak
> (per-segment 5-8.5%). A real fish's midline is inextensible — no muscle
> contraction shortens a body by a fifth. The sim's tracked chain is 16 RIGID
> links, arc length fixed by construction (0.149% rms, 26x less), so it
> structurally cannot produce the signal.**

The metric is `|d/dt (mean of tracked points)|`; a point set whose extent
breathes moves its own centroid, and d/dt reads that out as "speed". In the
20-45 Hz band, `corr(centroid speed, |dL/dt|) = 0.83`, and `|dL/dt|` rms of
4.90 BL/s predicts ~2.45 BL/s of centroid speed against 3.15 observed.

**DISCRETISATION CONTROL (the obvious objection, and it fails).** Some arc
variation is unavoidable: a 9-chord polyline under-measures a bending body's
arc more than a 15-chord one, so the comparison could be unfair to the sim.
Resample the SIM midline to the SAME 10 equally-spaced stations:

| point set | arc length variation |
|---|---|
| REAL, 10 stations / 9 chords | **3.873% rms, 19.9% p2p** |
| SIM, 16 stations / 15 chords | 0.150% rms, 0.633% p2p |
| **SIM resampled to 10 stations** | **0.067% rms, 0.272% p2p** |

Discretisation on an inextensible body costs 0.067% rms; the real tracking
shows **58x more**. The objection is answered — this is not a chord-vs-arc
effect.

**FORWARD TEST (the decisive one).** Take the simulated swimming EXACTLY AS IS
and merely give its rigid midline the tracking's own per-segment length
variation. No physics, gait, force or coupling change:

| centroid speed, 20-45 Hz | rms | peak-to-peak |
|---|---|---|
| sim as simulated (rigid chain) | 0.31 | 1.44 |
| **sim + tracking length variation** | **2.03** | **11.65** |
| REAL experiment | 3.15 | 15.80 |

full trace: sim std 3.09 → 3.96 (real 4.41); cv 0.451 → 0.562 (real 0.728).

**That one artifact reproduces ~2/3 of the entire amplitude gap.** The residual
is a factor ~1.55, not ~10 — and part of it is genuine: the real fish's
centroid-relative shape motion at 20-45 Hz is 2-3x the sim's (0.0158 vs 0.0052
BL at the head), i.e. real body-wave harmonic content the sim under-resolves.
**THAT is the tractable physical residual worth chasing.** The factor of 10 was
never a physics problem.

Likely provenance of the length variation: 2D projection of a body that is not
exactly in the image plane (out-of-plane motion foreshortens the projection)
plus DLC labels sliding along the body. Neither is reproducible by, or a defect
of, the simulation.

### (0b) Momentum budget — a WEAKER, supporting argument (note the caveat)
Independently: if the centroid trace were the CoM, a component of amplitude `v`
at frequency `f` needs `F = M·2πf·v`. The 20-45 Hz experimental swing would
need **536 uN rms / 1346 uN at peak = 2.7 g** of acceleration sustained at
30 Hz during a *slow* 6 BL/s swim — 25-50x this fish's own mean thrust
(11-25 uN, measured, F=ma verified).
**CAVEAT — do not oversell this:** the bound uses the SIM's measured force, so
it presumes the sim does not under-predict oscillating force by ~20-45x. It is
an argument from plausibility, not a theorem, and the user correctly pushed
back on it. The inextensibility argument in (0) is the strong one because it
uses only the experimental data. The momentum budget is airtight only for the
high bands (45-300 Hz, 30-396x over), which are not what anyone cares about.

The metric is `|d/dt (mean of tracked points)|`. IF that centroid trace were the
body's CoM, a narrow-band component of velocity amplitude `v` at frequency `f`
would require a net external force `F = M·2πf·v`. So the MEASURED hydrodynamic
force (`drags.h5`) converts directly into the largest centroid surge Newton
permits at each frequency, for a 50.2 mg / 18 mm fish:

| band [Hz] | REAL | SIM | F_meas [uN] | v_allowed | REAL/allowed |
|---|---|---|---|---|---|
| 2-5 | 1.18 | 0.57 | 7.0 | 0.39 | 3.0 |
| 5-10 | 0.27 | 0.14 | 6.2 | 0.15 | 1.8 |
| 10-20 | 0.53 | 0.13 | 29.6 | 0.37 | 1.4 |
| **20-45** | **3.15** | 0.31 | 29.5 | 0.17 | **18.2** |
| **45-100** | **1.35** | 0.07 | 17.1 | 0.04 | **30.2** |
| **100-300** | **1.22** | 0.01 | 3.0 | 0.00 | **396** |

(centroid-speed rms, BL/s. Real total 4.41, sim total 3.09.)

**Below 20 Hz — the stroke band, real swimming — both sides sit within a small
factor of the bound and of each other. The sim is NOT deficient there.** Above
20 Hz, where **72% of the experimental variance lives** (3.15 of 4.41 BL/s),
the experiment runs 18x / 30x / 396x over the bound. For 45-300 Hz that is
conclusive. For the 20-45 Hz band (18x) treat it as supporting evidence only —
see the caveat in (0b); the inextensibility result in (0) is what settles that
band. The sim sits at or just under the bound in every band.

Method validated where the answer is known independently: on the sim, band by
band, `M·a` of the true mass CoM reproduces the measured hydro force to a few
percent (implied 14.6/25.0/27.9/19.9 uN vs measured 14.6/25.0/27.1/18.9).

`analyze_speed_oscillation_budget.py` -> `ns_data/speed_oscillation_newton_budget.png`.

**Two earlier explanations are now REFUTED, not merely unconvincing:**
- **"~27% of real variance is >40 Hz DLC jitter" is WRONG.** A 2nd-difference
  noise estimator (immune to trend and to FFT leakage) puts the per-keypoint
  white-noise floor at **0.4 um = 0.002 px** — the released data has already
  been smoothed. There is no sample-to-sample jitter to blame.
- **"surface point-cloud foreshortening" is WRONG in framing.** The 10 DLC
  keypoints are evenly spaced MIDLINE stations (0.11 BL apart, spanning 0.987
  BL) — the same kind of object as the sim's link COMs, not a surface cloud.

What the excess actually is: smooth, sub-pixel wobble of the tracked midline at
20-300 Hz (~0.2 px per keypoint at 100-300 Hz, 20-30x the sim's), plus a
structural difference between the two point sets — **the tracked midline is
EXTENSIBLE (arc length 3.87% rms, 19.9% peak-to-peak, per-segment 5-8.5%) while
the sim's tracked chain is a rigid-link chain with arc length fixed by
construction (0.149% rms, 26x less).** The same metric is being fed structurally
different objects.

CAVEAT, stated honestly: the budget bounds how much of the signal CAN be
translation; it does not fully attribute the remainder. At 20-45 Hz the real
fish's centroid-relative shape motion is genuinely 2-3x the sim's (0.0158 vs
0.0052 BL at the head), so part of that band may be real body-wave harmonic
content the sim under-resolves. A keypoint-coherence test did NOT separate
artifact from body wave (real and sim are equally coherent, 0.29-0.44, above
20 Hz). Also do NOT band-pass raw position traces with an FFT brick wall or a
transfer-function Butterworth — the translation ramp leaks into every band and
manufactures 0.086 BL of fake 100-300 Hz motion in the SIM. Use `sosfiltfilt`,
and validate any such pipeline against the sim's known F=ma first.

### (0c) The residual body-wave HARMONIC DEFICIT: fully diagnosed, not yet fixable
This is the ~1.55x that remains in the 20-45 Hz band after the midline-length
artifact of (0) is accounted for. **It is the PD servo's own frequency
response, and it is quantitatively exact.**

Realized/commanded joint-angle ratio per tail-beat harmonic (f0 = 16.4 Hz),
sim joints sensor vs the reference xlsx:

| run | 1f0 (16.4) | 2f0 (32.7) | 3f0 (49.1) | 4f0 (65.5) |
|---|---|---|---|---|
| baseline kp=0.2 | 0.892 | **0.707** | 0.550 | 0.434 |
| + scalar pre-emphasis 1.15x | 1.029 | 0.829 | 0.646 | 0.512 |

Fit `|H| = 1/sqrt(1+(f/fc)^2)` to each harmonic INDEPENDENTLY → **fc = 32.3 /
32.7 / 32.3 / 31.5 Hz**. The servo is a first-order lag whose corner is exactly
**kp/kv = 0.2/0.001 = 200 rad/s = 31.8 Hz**, and 2f0 sits on the −3 dB point
(0.707). That is why **the scalar pre-emphasis did not fix issue 1**: it lifts
every frequency equally, so it corrects the fundamental and leaves 2f0/3f0/4f0
17/35/49% short. The 20-45 Hz band the user cares about *is* 2f0.

**Correction built** (`servo_corner_hz` + `servo_preemphasis_cap` +
`servo_preemphasis_joints` in `zebrafish_ki_project/pd_controller.py`, all
default OFF/inert): a capped, zero-phase inverse-magnitude pre-emphasis,
`boost(f) = min(sqrt(1+(f/fc)^2), cap)`. Verified offline to apply
1.119/1.433/1.500 vs the intended 1.124/1.434/1.500.

**It works at the joint level and CONFIRMS the diagnosis — but NO variant gave a
usable episode.** Config `gen_configs_pd_3d_ep248_slow_lead.py`:

| variant | steps | 1f0 | 2f0 | 3f0 | 4f0 | verdict |
|---|---|---|---|---|---|---|
| all joints, cap 2.5 | 579 | 0.972 | 0.978 | 0.972 | 0.698 | CRASHED |
| all joints, cap 1.5 | 820 | 1.014 | 0.987 | 0.892 | 0.725 | CRASHED |
| skip 3 tail joints, cap 1.5 | 2200 | 0.992 | 0.962 | 0.772 | 0.599 | ran, PATHOLOGICAL |

- All-joint variants die with mjWARN_BADQACC at the posterior DOFs.
- The skip-tail variant completes but the free-body response is wrong: **mean
  CoM speed 32.9 BL/s with 1.05 BL net progress** (baseline 6.67 BL/s / 3.45
  BL) — the fish thrashes in place, and its 20-45 Hz shape amplitude overshoots
  to 2.06x real (baseline 0.46x). Boosting joints 0-11 while leaving 12-14 raw
  kinks the travelling wave. **LESSON: a numerically clean run is not a
  physically valid one — check net displacement, not just completion.**
- Also tried and rejected: the textbook time-domain lead `ref + (1/wc)·dref/dt`
  — an unbounded high-pass, died at iter ~450.

**The blocker is the model, not the compensator.** The posterior DOFs
(~1e-13 kg·m²) cannot absorb any increase in commanded high-frequency content;
this is the same fragility that pins kp at 0.2. **The MJCF has NO `armature`
attribute anywhere (verified) and FARMS never emits one** (no hits in
`FARMS_V2/`). Joint armature is the real cure and is reachable by wrapping
`farms_mujoco.simulation.mjcf.setup_mjcf_xml` exactly as `_offscreen_patch.py`
already does — but the value changes the passive dynamics, so it is a modelling
decision for the user, not a bug fix. **That is the recommended next step.**

### (a) Arena-wall hypothesis: TESTED and REFUTED (do not re-chase)
Noticed that the production config spawns an 18 mm fish at yaw 134.93 deg
(heading = yaw - 180 = -45 deg) inside a 0.10 x 0.05 x 0.0125 m arena — only
**2.8 BL wide** — aimed diagonally at the ymin wall.  Lateral wall clearance
decays 1.0 -> 0.20 BL through the episode and the turn tracks it; pooled over 6
runs the beat-smoothed turn rate is +5.9 deg/s above 0.8 BL clearance and
+87..+310 deg/s below it, with a sustained +34..39 uN lateral force appearing
exactly when clearance drops under 0.33 BL.  It looked like textbook near-wall
parallel alignment.
**It is not.**  The clearance/turn correlation is a TIME confound: clearance
decays monotonically in every run, so "low clearance" just means "late".  New
control run `gen_configs_pd_3d_ep248_slow_openwater.py` (spawn heading +x along
the long axis, on the y centreline; grid, physics, gait, controller, coupling
all untouched) starts at 1.38 BL clearance and holds >1.2 BL for the first
0.30 s — and produces a heading trace **indistinguishable from the baseline**
(RMS error vs real 5.8 deg early / 28.4 deg late, versus baseline 5.8 / 29.5).
The fish turns FIRST, at 1.2 BL clearance, and only then drifts into the wall.
Confinement contributes ~nothing to the turn.

### (b) The real fish does NOT swim straight — and the sim tracks it for 5 beats
Plotting the beat-smoothed heading of BOTH sides against time (same metric:
head-tail chord angle, one-beat moving average) —
`analyze_heading_divergence.py` -> `ns_data/heading_divergence_sim_vs_real.png`:

| t [s]  | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 |
|--------|------|------|------|------|------|------|------|------|------|------|
| REAL   | +0.7 |-13.3 |-13.1 | -7.8 | -3.0 | +4.1 |+10.9 | +3.6 | +1.7 | +7.0 |
| SIM    | +0.3 |-10.7 | -8.0 | -0.3 | +4.0 |+16.3 |+28.6 |+34.2 |+43.1 |+36.9 |

The real fish executes a slow **+-13 deg yaw manoeuvre**, and the sim reproduces
it — dip, phase and amplitude — for the first ~0.30 s (5 tail beats), RMS error
5-8 deg.  The divergence starts at t ~ 0.28 s: the real heading returns toward
zero, the sim's ratchets up.  RMS error 5.8 deg early -> 29.5 deg late.

Decomposing heading into per-beat oscillation and >3-beat slow drift:

| | per-beat osc RMS early/late | slow drift range early / late |
|---|---|---|
| REAL | 4.4 / 5.5 deg | -9.5..+0.8 / +1.5..+10.0 deg |
| sim baseline T12:17 | 4.0 / 6.3 deg | -7.0..+15.4 / **+16.6..+41.1** deg |
| sim openwater T10:00 | 4.0 / 6.2 deg | -7.2..+15.8 / **+17.0..+40.2** deg |
| sim preemph T18:10 | 4.9 / 7.8 deg | -10.7..+18.8 / **+20.5..+52.7** deg |

**The per-beat yaw response is CORRECT (within 15-25%); 100% of the discrepancy
lives in the slow drift**, which the real fish keeps bounded to +-10 deg and the
sim lets ramp to +41 deg.  This kills the "steady curvature bias integrated by an
open-loop body" framing (already suspect after Session 2's debias failure): the
sim is not accumulating a bias from step one, it is failing to ARREST a
low-frequency yaw excursion after ~5 beats.  It also means the hydrodynamics and
coupling are in better shape than assumed — they reproduce a non-trivial real
yaw manoeuvre for 0.3 s.  The target is yaw RESTORING/DAMPING (fin area,
heading-hold control), and the right metric is the slow-drift component, not net
yaw over the whole episode.

### (c) Model + solver are left/right SYMMETRIC — the drift is gait-determined
`gen_configs_pd_3d_ep248_slow_invert.py` (`kinematics_invert: True`, i.e.
`kinematics *= -1`, the exact left/right mirror of the gait; run
`ns_data/2026-07-21T10:10:20.681483`) vs the T12:17 baseline. A symmetric model
must give `heading_inverted(t) = -heading_baseline(t)`:

| t [s] | 0.08 | 0.13 | 0.18 | 0.23 | 0.28 | 0.33 | 0.38 | 0.43 |
|---|---|---|---|---|---|---|---|---|
| baseline | −3.1 | −1.9 | −2.7 | −6.3 | +5.3 | +24.2 | +40.9 | +48.1 |
| inverted | +3.0 | +1.7 | +2.6 | +6.2 | −5.3 | −23.9 | −2.0 | +6.5 |
| sum (0 if perfect) | −0.0 | −0.1 | −0.1 | −0.1 | −0.0 | **+0.3** | +39.0 | +54.7 |

**Mirror symmetry is exact to ≤0.3° (≈1% of a 24° excursion) for the first
0.33 s** → there is NO left/right asymmetry bug in the mesh, the staggered SDF
or the solver, and the yaw drift is a deterministic, reproducible response to
the prescribed gait. The symmetry then breaks — but for an understood reason:
the inverted fish turns the other way, which drives it INTO the ymin wall
(clearance 0.16 BL at t = 0.33), and its heading is bent back
(−23° → −6° → +8°). The arena is not symmetric under reflection about the
fish's initial heading axis, so the two runs stop being mirror images once one
of them is against a wall.

Net: walls DO act, but only inside ~0.2 BL clearance — they are a late-time
perturbation, not the cause of the drift onset (which the open-water control in
(a) shows happens at 1.2 BL clearance). Both wall results are consistent.

### (d) How big a yaw counter-moment would the fix need? ~5-15 nN·m
From `drags.h5` on T12:17, per-link yaw moment about the body CoM, averaged over
3 tail beats (the slow-drift band):

- body yaw inertia `I_zz = 6.2e-10 kg·m²` (point-mass links; link self-inertias
  ~1e-12 are negligible).
- DC yaw moment: **+15.4 nN·m early (t 0.05-0.28), +4.1 nN·m late (0.28-0.52)**.
  Alone that gives 6.6 rad/s² → ~11° over the late window, the right order for
  the observed ~25° late drift (the body is deforming, so it is not an exact
  rigid-body budget).
- Per-link late DC breakdown: link15 (tail tip) **−31 nN·m**, anterior link0/1
  **+8.9/+8.3**, posterior link12/13 **+6.1/+9.6**; net **+4.1**.

So the required yaw restoring moment is **~5-15 nN·m sustained — only ~3% of the
±500 nN·m per-beat yaw moment swing**, i.e. a modest fin area or a weak
heading-hold term should be enough. CAVEAT: that net DC value is a small
residual of much larger opposing per-link contributions, so it is numerically
delicate — treat it as an order of magnitude, not a precise target.

## STATUS 2026-07-20 (SESSION 2) — Open Q1 SETTLED, Open Q2 lever REFUTED
Attacked the two Open Questions with on-disk data + two new coupled runs. Net:
Issue 1 is confirmed a metric/data artifact; the Open-Q2 "de-bias the gait"
lever is TESTED and does NOT straighten the sim (makes turning WORSE), so the
turn is confirmed emergent open-loop hydrodynamics — NOT the prescribed gait
curvature. Do not re-chase gait de-biasing.

### Open Q1 (Issue 1 speed oscillation) = METRIC/DATA ARTIFACT (settled)
Used the sim's OWN mass distribution (`/links/masses`) as ground truth:
| metric (on the SAME sim run T12:17) | mean BL/s | cv |
|---|---|---|
| unweighted centroid of 16 link-COMs (what compare.py uses) | 6.85 | 0.451 |
| TRUE mass-weighted CoM | 6.68 | 0.471 |
Mass-weighting is NOT the difference (0.45 vs 0.47). The real centroid's extra
surge (cv 0.73) is a point-cloud foreshortening + jitter effect: real
corr(centroid-speed, |d chord/dt|) = **0.50** vs sim **0.065**, and the real
body foreshortens 6.2% vs sim 3.8%. There is no evidence the real fish's TRUE
CoM surges more than the sim's. (Real masses still unavailable, but shown moot.)

### Open Q2 (Issue 2 turning): net gait curvature is NOISE, and zeroing it FAILS
- Confirmed the -3.9 deg net curvature is statistically zero: std 42.5 deg over
  9 tail beats -> mean/SE = **-0.28** (indistinguishable from a straight gait).
- BUILT a reversible de-biasing option (`debias_reference` in
  `pd_controller.py`, default OFF) + config `gen_configs_pd_3d_ep248_slow_debias.py`
  (`DEBIAS_MODE=net|true`). Ran two coupled implicit sims vs the T12:17 baseline
  (net body-yaw +53 deg, forward +3.4 BL, max-lateral 0.66 BL):
  | run | realized net curv | body yaw | fwd disp | max lateral |
  |-----|------|------|------|------|
  | baseline (biased gait) | -3.9 deg | +53 deg | +3.4 BL | 0.66 BL |
  | debias PER-JOINT (mean posture straight) | ~0 deg (verified in sim) | **+63 deg** | **-0.7 BL** (curled 142 deg heading) | 2.76 BL |
  | debias NET (uniform -0.26 deg/joint) | ~0 deg | still turning RIGHT to -1.5 BL by step 1600, then BLEW UP @ step 1705 (CFL climbed >0.6) | — | — |
- **KEY NEGATIVE RESULT: zeroing the gait's net curvature does NOT straighten the
  sim — it turns MORE (lateral 0.66 -> 2.76 BL) and loses forward progress.** The
  prescribed -3.9 deg bias is NOT the turn driver; removing it disrupts the
  balanced traveling wave. This CONFIRMS the TL;DR root cause: the turn is an
  emergent open-loop hydrodynamic/recoil response of the finless body, which only
  fins + heading-hold control can fix. Open Q2's "straight-swim reference = 0 deg
  net curvature" idea is refuted as a fix.
- INCIDENTAL (required to run): the uncommitted working tree had
  `BDIMhandler._launch_body_update` passing `use_graph=` to `native.body_update_{2,3}d`,
  whose signatures had just been cleaned to drop that (always-ignored) param ->
  `TypeError` on step 1, blocking EVERY sim on this branch. Fixed by removing the
  two dead kwargs at the call sites (behavior-preserving).

## STATUS 2026-07-20 (SESSION 1) — BOTH ISSUES STILL UNSOLVED
A long recoil/PD-tracking deep-dive this session **did NOT solve either original
issue.** Current numbers (CoM = geometric centroid of link COMs; real from
keypoints):

| run | CoM speed mean / std / cv (BL/s) | net turn |
|-----|----------------------------------|----------|
| REAL | 6.05 / 4.41 / **0.73** | +3° (≈straight) |
| sim kp=0.2 baseline | 6.70 / 3.18 / **0.48** | +15° |
| sim kp=0.2 + pre-emphasis | 7.43 / 3.52 / **0.47** | +20° |

- **Issue 1 (CoM oscillation) NOT solved:** real cv 0.73 vs sim ~0.47. Recovering
  the body-wave amplitude (see below) left the CoM oscillation **unchanged**
  (0.48→0.47). **KEY NEGATIVE RESULT: the CoM-velocity oscillation deficit is NOT
  a body-wave-amplitude problem** — a near-perfect body wave still does not make
  the sim CoM surge like the real one.
- **Issue 2 (turning) NOT solved, slightly WORSE:** sim turns +15° (baseline) /
  **+20°** (pre-emphasis) vs real +3°. Bigger body wave integrates *more* curvature
  bias, not less.

### CAUSE OF ISSUE 1 = THE METRIC (data-backed, filter both sides identically)
Low-pass the CoM *position* then compute speed; cv = std/mean of speed:
| cutoff | REAL cv | SIM cv |
|--------|---------|--------|
| raw (6 kHz) | 0.73 | 0.45 |
| <50 Hz | 0.64 | 0.46 |
| **<20 Hz** | **0.43** | **0.49** |
| <10 Hz | 0.47 | 0.55 |
The whole discrepancy is **>20 Hz**. At stroke level (<20 Hz, the tail-beat
fundamental — the physically meaningful surge) **real ≈ sim (sim even slightly
higher)**. The raw real excess is (a) 6 kHz DLC keypoint JITTER — 27% of real
CoM-speed variance is >40 Hz — the sim has none; and (b) the **2×-tail-beat
foreshortening of the geometric centroid** (~29 Hz peak): the mean of 10 tracked
SURFACE keypoints pulses as the body bends, twice/beat — a property of the point
cloud, NOT true CoM translation. Sim CoM is a clean CENTERLINE mean and doesn't.
**=> Issue 1 is largely a METRIC artifact (noisy surface-keypoint centroid @6kHz
vs clean centerline centroid); there is NO real "fish surges, sim doesn't" gap.
Resolves Open Q1: real true-CoM surge is modest (cv 0.73 would mean the fish
nearly stops each beat — it doesn't).** Issue 2 (turning) remains genuine: gait
curvature bias integrated by the open-loop finless body (pre-emphasis WORSENED it
+15→+20°, confirming steady-bias-integrated-open-loop).

### What the recoil/tracking deep-dive found (real, but orthogonal to the 2 issues)
The sim's body-wave (recoil) amplitude was ~86% of real. Decomposed:
- **Conversion (keypoints→model angles) is FAITHFUL (~100-106%)** in tangent-angle
  terms; the earlier "−5%" was a recoil-metric point-distribution artifact.
  `extrapolate_angles.py` is fine — do NOT chase it.
- The deficit was **PD tracking**: the over-damped servo (kp=0.2, kv=0.001,
  bandwidth ~32 Hz) realizes only ~86% of commanded joint amplitude.
- **Stiffening blows up** (BADQACC at tail DOFs; tail links ~1e-12 kg·m², no
  armature, FARMS doesn't plumb it). kp≥0.5 fails at startup; kp=0.4 fails
  mid-run (t=0.42 s) even at half dt; ceiling ~kp0.4 only reaches 93% recoil.
- **FIX = reference PRE-EMPHASIS** (new `reference_gain` in
  `zebrafish_ki_project/pd_controller.py`, per-joint 1/tracking_ratio ~1.15):
  at the STABLE kp=0.2, tracking 0.86→1.00, recoil **86%→97% of real**, full
  stable run. Config `gen_configs_pd_3d_ep248_slow_preemph.py`
  (run `ns_data/2026-07-20T18:10:10`). Stiff probe: `..._slow_stiff.py`.

**Takeaway for the next agent:** kinematics/controller fidelity is now essentially
maxed (body wave = 97% of real) and the two issues REMAIN. This confirms the
root cause below (finless open-loop free-body hydrodynamic response), NOT the
prescribed kinematics. Do not spend more on gait/tracking/conversion — attack the
morphology/control (fins, heading hold) or re-examine whether the CoM-oscillation
metric is even the right target (Open Q1 still open: real true-CoM surge needs
real masses).

## TL;DR root cause (best current understanding)
Both issues trace to the **same limitation**: the zebrafish model is a
**finless, open-loop body** (16-link midline, `sdfs/zebrafish/zebrafish_v1_triangulated`,
zero fin geometry). Given a *prescribed* gait (position control), the free-body
6-DOF response through the fluid differs from a real fish that stabilizes itself
with fins + closed-loop control. The sim under-anchors its anterior body (head
yaw-wag 3.4° vs real 1.3°, i.e. 2.7x), which:
- makes its tracked-point centroid *not* surge like the real one (issue 1), and
- lets a slow yaw excursion run away instead of being arrested (issue 2).
This is a morphology/control limitation, NOT a solver/coupling/readout bug.

**Refined by Session 3 (see top of file).** The issue-2 half of this — "the body
integrates a steady gait curvature into a turn" — is WRONG and should be read as:
the sim's *per-beat* yaw response matches the real fish within 15-25%, and it
reproduces the real fish's own ±13° yaw manoeuvre for the first 5 tail beats.
What fails is the **slow (>3-beat) drift**, which the real fish holds inside
±10° and the sim lets ramp to +41°. So the missing ingredient is yaw
restoring/damping, not the removal of a bias — which is exactly why Session 2's
de-biasing made things worse.

## What was RULED OUT this session (do not re-chase)
- **Metric-only artifact for issue 1:** partially. The compare speed = `|Δ(mean of
  tracked points)|/dt` = geometric centroid. That centroid surges from body
  **foreshortening** (real: corr +0.74 with chord-shortening rate, ≥55% of surge
  variance). BUT the sim body foreshortens too (chord pulses 3.8% vs real 6.2%),
  so foreshortening alone does NOT explain why real surges (44% band-20-45) and
  sim doesn't (2%). The differentiator is head-anchoring (above), which is real.
- **Kinematics / low-pass:** `pd_controller.py` low-passes the reference at
  `lowpass_cutoff`. Raised 30→100 Hz; joint tracking in the 20-45 Hz surge band
  went to 77% of reference, but the CoM surge stayed 2% and the turn persisted.
  Not the cause.
- **Coupling scheme:** implicit (Aitken) strong coupling ≡ explicit — identical
  trajectory, surge, drift. See `gen_configs_pd_3d_ep248_slow_implicit.py`. Not it.
- **Steady/under-read thrust:** REFUTED by direct per-link force log (`drags.h5`,
  enable with `save_drags=True`). Net forward thrust pulsates (56% band at 32 Hz),
  `F = m·a` holds to the digit (11.1 uN). Tail drives (link15: 21 uN lat, 184 nN·m
  yaw), head/anterior resist (link0: 8 uN, 51 nN·m). Force machinery is healthy.
- **Mass-weighting the speed metric:** even sim-mass-weighting the real body keeps
  real surge ~44% vs sim 2%. Not a weighting artifact. (Real masses unavailable.)

## What was FIXED this session
- **Extraction curvature bug** in `keypoints/extrapolate_angles.py`
  (`compute_coordinates_from_arc_lengths`): it extrapolated the untracked head/tail
  along the **spline endpoint tangent**, injecting −3.56° of phantom net curvature
  (raw keypoints −3.90° → extracted −6.22°). Changed to extend along the **terminal
  keypoint chord** → net curvature now −3.90° = raw, mid-body gait preserved
  (boundary joints 0 & 13 stiffen to ~30%, acceptable — they attach to untracked body).
  `model_angles.xlsx` regenerated (both the `keypoints/` and parent copies;
  `.prefix_bak` backups exist). Validation re-run: turning dropped ~15%
  (net yaw 65→53°, lateral 0.62→0.55 BL) — real bug, but only a small part of the drift.

## Key evidence numbers (implicit run, fixed angles = ns_data/2026-07-20T12:17:59.330717)
- Head yaw-wag (20-45 Hz): real 1.3°, sim 3.4° (2.7x too much).
- Per-station forward surge RMS (20-45 Hz): real head 1.3 / tail 6.9 BL/s;
  sim head 0.6 / tail 2.9 — every station 3-8x weaker, head-to-tail phase
  coherence real +0.78 vs sim +0.20 (real body surges as a coherent unit; sim doesn't).
- Fixed gait net curvature −3.90° = the REAL fish's body curvature, yet real swam
  straight (0.1° net heading) while sim turns 53° → sim lacks heading stabilization.

## Open questions for the next agent
0. **(Session 3, now the main one) What arrests the real fish's slow yaw drift?**
   The sim's per-beat yaw response is right and it follows the real ±13° yaw
   manoeuvre for 5 beats; only the >3-beat drift runs away (+41° vs the real
   fish's ±10° bound). Candidates, in order of cost: (a) yaw damping from the
   missing median/anal/pectoral fin area — estimate the added yaw damping
   coefficient a fin set would contribute before modelling it; (b) a
   heading-hold term in `pd_controller.py` (a real fish's slow-drift correction
   is closed-loop and would not show up in prescribed midline kinematics at
   all); (c) under-resolved anterior body — the head over-wag (3.4° vs 1.3°) is
   the same deficit seen from the recoil side. Measure the sim's slow-drift
   yaw torque budget from `drags.h5` first: which links supply the net DC yaw
   moment after t = 0.28 s, and how big a counter-moment is needed to cap the
   drift at ±10°? That number sizes the fin/control fix.
1. **Is the real 44% centroid surge TRUE CoM motion (physics the sim should match)
   or geometric-centroid-relative-to-CoM (a metric effect)?** Cannot settle without
   the real fish's mass distribution (unavailable). If you can get/estimate real
   masses, compute the real true-CoM surge — if it's small (~5-15%, biologically
   normal), the sim is basically right and issue 1 is mostly the metric.
2. **Does the −3.9° gait curvature even represent a real steady bias?** It's a
   time-mean with std ~42° (the tail-beat swing); over ~9 cycles it's within ~1 SE
   of zero. The real fish swam straight with it. The sim turns on it because it's
   open-loop. Consider whether a straight-swim reference should have 0° net curvature.
3. **Fins / heading control:** the model has no fins and no yaw/heading controller.
   Adding fin surfaces (esp. anterior/median for yaw damping + caudal for tail area)
   and/or a heading-hold controller is the real fix for both the head over-wag and
   the residual turn. This is a modeling project, not a bug.

## Issue-1 visualization + fair-comparison fixes (2026-07-20 SESSION 2)
`keypoints/viz_issue1_foreshortening.py [sim_run]` -> `ns_data/issue1_foreshortening_explained.png`
(6 panels: mechanism, chord foreshortening, arc-length breathing, speed trace,
PSD, cv-vs-cutoff). Two fair-comparison fixes, both implemented in the script:
- FIX 1 denoise the real DLC keypoints (temporal low-pass, `DENOISE_HZ`=40): real
  speed cv 0.73->0.58, chord ±6.2%->±5.9%. NB the chord amplitude barely moves
  because the real arc "breathing" is mostly BEAT-LOCKED (<40 Hz, likely 2D
  projection of a not-perfectly-planar body), NOT white jitter; only the SPEED
  (differentiation amplifies high-freq at 6 kHz) is strongly cleaned.
- FIX 2 extend the sim midline to the true body tips (`extend_to_tips`; link COMs
  span only 0.85 BL vs real keypoints' 0.99 BL): sim chord ±4.8%->±5.4%.
- Net: denoised-real ±5.9% vs tip-extended-sim ±5.4% (gap 1.4->0.5 pts); at
  stroke bandwidth (<20 Hz) the speed cv converges (~0.45). Point/count density is
  negligible (resampling sim to real's 10 stations: 4.8%->4.8%, no change).
- Decomposition of real speed std (4.41 BL/s): <10 Hz surge 2.43 (matches sim),
  20-40 Hz foreshortening 1.56, >40 Hz jitter 1.61. So the big 10-15 BL/s swings
  are foreshortening + genuine surge, NOT the ±3.9% arc jitter (that = ~1.6 BL/s).
- USE the pre-emphasis run (T18:10, ~97% body wave, chord ±4.8%) not baseline
  (T12:17, ±3.8%): better realized kinematics genuinely foreshortens more.

## Fix ideas for issue 1 (speed panel), if a physical match isn't required
Compare **stroke-averaged** speed (low-pass < ~10 Hz) applied identically to both
sides, and/or the sim's **true mass-CoM** (masses ARE logged in
`simulation.hdf5` `/links/masses`). This removes the foreshortening ripple and the
two traces align (~6 BL/s). NOT yet implemented — user was undecided.

Note (2026-07-20): user rejected changing the compare metric. The metric MUST be
identical on both sides (it already is: geometric centroid of tracked points /
|Δ|/dt). The open puzzle the user is now focused on: **the sim runs the exact
real-derived joint kinematics, yet the sim centroid oscillates far less than the
real centroid.** Since the metric is shared, the difference is in the world-frame
body motion, not the metric — see the head-anchoring / foreshortening analysis above.

## Files & artifacts
- Compare script: `keypoints/compare_sim_real.py`
- Extraction (FIXED): `keypoints/extrapolate_angles.py`
- Controller: `pd_controller.py` (`lowpass_cutoff` via config `control_pars`)
- Configs: `gen_configs_pd_3d_ep248_slow.py` (explicit, production),
  `gen_configs_pd_3d_ep248_slow_implicit.py` (implicit + `save_drags=True`)
- Runs (ns_data/): `zebrafish_real_kinematics/slow` (orig);
  `2026-07-20T10:39` (explicit 100Hz); `...T10:59` (implicit 100Hz);
  `...T11:27` (implicit + drags.h5, OLD −6.22° angles);
  `...T12:17` (implicit + drags.h5, FIXED −3.90° angles = latest).
- Per-link hydro force log: `<run>/drags.h5` (viscous/pressure force + torque per link).
- Real data: `keypoints/ep248_Cl2_slow_fish13_XY_BL.csv` (10 DLC keypoints, BL, 6 kHz),
  `..._model_angles.xlsx` (regenerated, fixed).

### Added 2026-07-21 (Session 3)
- `analyze_heading_divergence.py` — beat-smoothed heading, sim vs real, same
  metric both sides; prints early/late RMS error, writes
  `ns_data/heading_divergence_sim_vs_real.png`. **This is the plot that reframes
  issue 2.**
- `analyze_wall_effect.py` — lateral wall clearance vs turn rate, pooled over
  runs. Kept because it documents the refuted wall hypothesis and is the tool
  to re-check confinement if the arena is ever shrunk.
- `gen_configs_pd_3d_ep248_slow_openwater.py` — spawn heading +x on the y
  centreline (1.38 BL clearance instead of 0.73 decaying to 0.20). Run
  `ns_data/2026-07-21T10:00:19.717441`. Result: heading trace matches the
  baseline → **confinement is NOT the turn driver.**
- `gen_configs_pd_3d_ep248_slow_invert.py` — `kinematics_invert: True` mirrors
  the gait (`kinematics *= -1`). Left/right symmetry check on the drift.
  Run `ns_data/2026-07-21T10:10:20.681483`.
- Useful convention discovered: **world heading = spawn yaw − 180°** (production
  yaw 134.93° → heading −45.07°), and `gen_pool_sdf` places the pool walls
  OUTSIDE the fluid box, so their inner faces sit exactly at xmin/xmax/ymin/ymax.
