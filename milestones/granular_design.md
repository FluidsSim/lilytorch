# Granular Flow (sand) via μ(I)-rheology — Design Notes

Status: **design proposal / review**, no code yet.
Branch context: written against `optimize_speed_memory`.
Author: planning assistant, June 2026.

This document is a roadmap for letting LilyTorch simulate **dense dry/immersed
granular flow** (sand, gravel, grains) as a continuum, by treating the granular
medium as an **incompressible fluid with a pressure-dependent, shear-rate-
dependent effective viscosity** — the **μ(I) rheology**. The central claim is
that this is the *single cheapest genuinely-new physics* LilyTorch can add,
because ~80 % of the required machinery already exists:

- the **variable-viscosity dispatcher** `FluidSolver._compute_nu_t`
  (`lilytorch/src/extras.py:193`) and the regularized yield-stress viscosity op
  `ops.carreau_viscosity` (`lilytorch/src/operations.py:295`) — which *already*
  implements `τ_y/(ρ·max(γ̇,γ̇_min)) + ν_∞ + … ` with a CFL clamp `nu_max`;
- the staggered **strain-rate magnitude** `ops.strain_rate_magnitude`
  (`operations.py:223`, with the `_stag_to_cc` cross-derivative fix);
- the **VOF free surface** in `TwoPhase` (`lilytorch/src/two_phase.py`): the
  granular *pile surface* is just the air/granular interface, advected by the
  same Weymouth–Yue scheme already used for water/air;
- the **subclass-and-override** extension pattern proven by
  `TwoPhaseSolver` (`lilytorch/src/two_phase_solver.py:162`), which adds an
  entire new fluid model by overriding ~5 methods of `FluidSolver`.

The deliverable is a new `GranularSolver(TwoPhaseSolver)` that reuses the
projection, BDIM, advection–diffusion, forces, and FARMS/MuJoCo coupling
**unchanged**, and adds only (a) a pressure-dependent μ(I) viscosity closure and
(b) granular-specific stabilisation.

---

## 1. What μ(I) rheology is, and why it fits LilyTorch

Dense granular flow (the "liquid" granular regime — flowing sand, not a
granular gas and not a static pile) is well described by a continuum momentum
balance identical in form to incompressible Navier–Stokes, closed by the
**μ(I) rheology** of GDR MiDi (2004) and Jop, Forterre & Pouliquen (2006):

The deviatoric stress is aligned with the strain-rate tensor,

    τ_ij = η_eff · 2·S_ij ,     S_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i),

with an effective **shear viscosity** that depends on the local granular
pressure `p` and the **inertial number** `I`:

    η_eff(γ̇, p) = μ(I) · p / γ̇ ,         γ̇ = sqrt(2 S_ij S_ij) = |2S|

    I = γ̇ · d / sqrt(p / ρ_s)

    μ(I) = μ_s + (μ_2 − μ_s) / (I_0 / I + 1)

where

| symbol | meaning |
|--------|---------|
| `d`    | grain diameter |
| `ρ_s`  | grain (solid) material density |
| `μ_s`  | static friction coefficient (= tan of repose angle), ≈ 0.32 |
| `μ_2`  | dynamic (high-I) friction coefficient, ≈ 0.6 |
| `I_0`  | material constant, ≈ 0.3 |
| `p`    | granular pressure (the **mechanical pressure from the projection**) |

The kinematic effective viscosity LilyTorch actually needs is
`ν_eff = η_eff / ρ`. Compare this to the **already-implemented**
Herschel–Bulkley/Carreau viscosity (`operations.py:295`):

    ν_carreau = τ_y/(ρ·max(γ̇,γ̇_min)) + ν_∞ + (ν_0−ν_∞)[1+(λγ̇)²]^((n−1)/2)

The granular closure is *structurally the same object* — a `1/γ̇`
yield-like term plus a bounded background — except the "yield stress" is
**pressure-dependent**: `τ_y(x) = μ_s · p(x)` instead of a constant. Everything
else (the `max(γ̇,γ̇_min)` regularisation, the `nu_max` CFL clamp) carries over
verbatim. This is why the increment is small.

### Why a continuum (and what it does NOT model)

A continuum μ(I) model captures **flowing/slumping/avalanching** sand: column
collapse, hopper discharge, a foot or wheel ploughing through a granular bed,
underwater-walking on a sandy bottom. It does **not** capture discrete-grain
effects — segregation by size, force chains/arching, the granular-gas (dilute,
collisional) regime, individual particle trajectories, or fracture/clogging.
If those are required, μ(I) is the wrong model and the project is into
DEM/MPM territory (out of scope; see §11).

---

## 2. Reference papers (read in this order)

1. **Lagrée, Staron & Popinet, *J. Fluid Mech.* 686 (2011), 378–408** —
   "The granular column collapse as a continuum: validity of a 2-D
   Navier–Stokes model with a μ(I)-rheology." **The direct template**: an
   Eulerian incompressible NS solver (Gerris) + VOF free surface + μ(I)
   effective viscosity — i.e. *exactly* the LilyTorch architecture
   (projection + `TwoPhase` VOF + `_compute_nu_t`). Mirror their viscosity
   regularisation and their column-collapse validation.
2. **Jop, Forterre & Pouliquen, *Nature* 441 (2006), 727–730** — the μ(I)
   constitutive law (the `μ(I)` and `I` formulas above).
3. **GDR MiDi, *Eur. Phys. J. E* 14 (2004), 341–365** — phenomenology of dense
   granular flows; where the inertial number `I` comes from.
4. **Pouliquen & Forterre, *J. Fluid Mech.* 453 (2002)** — `h_stop`, the
   stopping angle / angle of repose; how to validate `μ_s`.
5. **Barker, Schaeffer, Bohórquez & Gray, *J. Fluid Mech.* 779 (2015)** —
   μ(I) is **ill-posed** (Hadamard / short-wave instability) for small and
   large `I`. *Mandatory reading before trusting any result.*
6. **Barker & Gray, *J. Fluid Mech.* 828 (2017)** — a regularised μ(I) that
   restores well-posedness. This is the stabilisation recipe to implement
   (see §6).
7. **Lacaze & Kerswell 2009; Lube et al. 2005; Lajeunesse et al. 2005** —
   column-collapse experiments for quantitative validation (runout length vs.
   initial aspect ratio).
8. **Cassar, Nicolas & Pouliquen, *Phys. Fluids* 17 (2005)** — the
   **immersed** (underwater) granular regime: how `I` generalises to a
   viscous inertial number `I_v` when the interstitial fluid matters. Needed
   for the "salamander walking on a submerged sandy bed" use case.

---

## 3. Where this hooks into the existing code

`GranularSolver` subclasses `TwoPhaseSolver`, reusing its two-phase plumbing
to represent the **granular/air free surface**. The *only* new physics is the
viscosity closure; the *only* new numerical concern is stabilising the stiff
`1/γ̇` and pressure feedback.

```
GranularSolver(TwoPhaseSolver)
  ├─ phase α : 1 = granular medium, 0 = air                (reuse TwoPhase VOF)
  │     ρ_cc  = α·ρ_grain_bulk + (1−α)·ρ_air               (recip_density_cc)
  │     advected by Weymouth–Yue                            (TwoPhase.advect)
  │
  ├─ _compute_granular_nu_t(*vel)   ← NEW                   (dispatched by
  │     ν_eff = μ(I)·p / (ρ·max(γ̇,γ̇_min)),  clamped nu_max   _compute_nu_t)
  │     reads the LAGGED pressure self.p0 (previous step)
  │
  ├─ everything else INHERITED UNCHANGED:
  │     advection–diffusion (uses ν_eff as nu_t)            FluidSolver.fluid_step
  │     BDIM no-slip on immersed bodies                     _apply_bdim_all_axes
  │     variable-density pressure projection                project / _compute_bdim_coefficients
  │     forces on bodies (foot/wheel reaction)              forces_method2[_3d]
  │     FARMS/MuJoCo coupling                               BDIMhandler (no change)
  └─
```

### 3.1 The pressure-lag coupling (the one real subtlety)

In an incompressible model the granular pressure `p` is the Lagrange
multiplier that enforces `∇·u = 0` — it is *output* by the projection, but
μ(I) needs it as *input* to the viscosity. This is a genuine implicit coupling.
The standard, robust resolution (Lagrée–Staron–Popinet 2011) is to **lag the
pressure by one step**: compute `ν_eff` from `self.p0` left over from the
previous projection.

This requires **zero new plumbing** because `_compute_nu_t` is already called
at the *top* of `fluid_step` (`solver.py:1748` in 2-D, `1906` in 3-D), before
advection–diffusion, at which point `self.p0` still holds the previous step's
pressure. The granular closure just reads `self.p0`. For the very first step
`p0 = 0` ⇒ `ν_eff = ν_floor` (a safe quiescent start).

Optionally, a `granular_pressure_subiter` config (default 1) re-evaluates
`ν_eff` with the *new* `p0` and repeats the projection N times for tighter
coupling — exactly mirroring the existing `consistent_n_cycles` fixed-point
loop already in `TwoPhaseSolver`.

---

## 4. The new viscosity op

Add `ops.granular_viscosity` next to `ops.carreau_viscosity`
(`operations.py:295`). It is a near-clone with a pressure field argument:

```python
def granular_viscosity(vel, p, h, ndim, *, mu_s, mu_2, I0, d, rho_s, rho,
                       gamma_min=1e-6, p_min=0.0, nu_max=None):
    """Kinematic effective viscosity for the mu(I) granular rheology.

        gamma_dot = strain_rate_magnitude(vel)          # |2S|
        p_eff     = clamp(p, min=p_min)                 # granular p >= 0 (no tension)
        I         = gamma_dot * d / sqrt(p_eff/rho_s + eps)
        mu_I      = mu_s + (mu_2 - mu_s) / (I0/ max(I,eps) + 1)
        eta_eff   = mu_I * p_eff / max(gamma_dot, gamma_min)   # dynamic
        nu_eff    = eta_eff / rho                              # kinematic
        nu_eff    = clamp(nu_eff, max=nu_max)                  # diffusion CFL
    """
    S = strain_rate_magnitude(vel, h, ndim)             # reuse operations.py:223
    p_eff = torch.clamp(p, min=p_min)
    I = S * d / torch.sqrt(p_eff / rho_s + 1e-12)
    mu_I = mu_s + (mu_2 - mu_s) / (I0 / torch.clamp(I, min=1e-9) + 1.0)
    eta = mu_I * p_eff / torch.clamp(S, min=gamma_min)
    nu = eta / rho
    if nu_max is not None:
        nu = torch.clamp(nu, max=nu_max)
    return nu
```

Notes:
- `strain_rate_magnitude` already returns a **cell-centred** `|2S|` with the
  correct cross-derivative staggering (the `_stag_to_cc` fix). `p` is cell-
  centred. So the whole expression is cell-centred and matches how `nu_t`
  is consumed by `adv_diff_solver.solve(..., nu_t=...)`.
- The `nu_max` CFL clamp is **load-bearing**: a static/jammed region has
  `γ̇ → 0` ⇒ `ν_eff → ∞`. Cap it via the same diffusion-CFL bound the Carreau
  path already computes (`carreau_nu_max`): `ν_max ≈ C_diff·h²/(2·ndim·dt)`.
  Below yield the medium is "frozen" by an enormous (but finite) viscosity —
  this is the standard regularised-rigid approximation, not a true elastic
  solid (see §6 and §11 for the honest limitation).

### Dispatch

In `extras._compute_nu_t` (`extras.py:193`) add a branch mirroring the Carreau
one (`extras.py:201`):

```python
if self.use_granular:
    return self._compute_granular_nu_t(*vel)   # NEW, in GranularSolver
```

`GranularSolver._compute_granular_nu_t` calls `ops.granular_viscosity(vel,
self.p0, ...)`. As with Carreau, set the solver baseline `self.nu` to a small
floor so that `ν_total = self.nu + ν_eff` is well-defined and bounded below.

---

## 5. Configuration schema

A single `granular` dict on the sim config (parallel to `self.carreau` and
`self.sponge`), routed through `BaseSimConfig` into the solver:

```python
self.granular = {
    "mu_s":   0.32,     # static friction (tan of repose angle)
    "mu_2":   0.60,     # dynamic friction at high I
    "I0":     0.30,     # inertial-number constant
    "d":      1.0e-3,   # grain diameter [m]
    "rho_s":  2500.0,   # grain material density [kg/m^3]  (quartz)
    "phi":    0.60,     # solid volume fraction (bulk = phi*rho_s)
    "p_min":  0.0,      # clamp granular pressure >= 0 (no tensile stress)
    "regularization": "barker2017",   # "none" | "barker2017"
    "nu_floor": 1.0e-6,
}
```

`ρ_grain_bulk = phi·rho_s` is what feeds the `TwoPhase` dense-phase density
(`rho_water` slot). Air stays the light phase.

---

## 6. Stability and regularisation (do NOT skip)

μ(I) is **mathematically ill-posed** (Barker et al. 2015): the governing
equations are Hadamard-unstable (grid-scale modes grow unboundedly) when `I`
is very small or very large. A naive implementation will produce
mesh-dependent blow-ups that *look* like bugs but are intrinsic to the model.
Three layers of defence, in increasing order of effort:

1. **Regularisation floor/ceiling on `I`** and the `nu_max` CFL clamp (cheap,
   in §4 already). Keeps a run alive; does not cure ill-posedness.
2. **Barker & Gray (2017) regularised μ(I)** — replace `μ(I)` with their
   well-posed functional form (a modified low-`I` branch + a linear high-`I`
   branch). This is the recommended production closure; it is a drop-in change
   to the `mu_I = …` line in `granular_viscosity`.
3. **Partial viscous regularisation** — add a small constant Newtonian
   viscosity `ν_reg` so the deviatoric stress never fully collapses. Pairs
   well with (2).

**Reuse the project's existing instability tooling.** `check_explosion`
(`solver.py`) is already called in `TwoPhaseSolver.finalize_step` before VOF
transport; it will catch granular blow-ups for free. The
`LILYTORCH_UMAX_PROBE` diagnostic in `TwoPhaseSolver` (max-|u| cell tagging)
should be extended to also print the local `I`, `γ̇`, and `p` so that
ill-posedness vs. a coding bug can be told apart.

**Known shared singularity with the open two-phase bug.** The triple-point
instability tracked in `to_do_list.md` (air/water/solid waterline) has a direct
granular analogue: the **air/granular/body corner** and the **static-pile free
surface** (where `γ̇ → 0` and `p → 0` simultaneously, so `I` is `0/0`). It is
worth resolving — or at least definitively characterising — the two-phase
triple-point issue *first*, because the fix (contact-line treatment, or the
Brinkman/DLM multiphase-WSI scheme already cited in the TODO) is reusable here.

---

## 7. Forces and FARMS/MuJoCo coupling — unchanged

A body (wheel, foot, salamander limb, submarine nose) ploughing through sand is
still an immersed BDIM body. The granular medium exerts a reaction load that is
integrated by the **existing** `forces_method2[_3d]` / Lagrangian-marker force
path — no change. The load is the granular analogue of hydrodynamic drag:
pressure (now the granular pressure, which can be large under a static foot) +
"viscous" (now frictional) stress. The FARMS coupling (`BDIMhandler`) is
untouched: it reads body poses, the granular solver advances, loads go back to
MuJoCo. This is the payoff of building on the Eulerian core rather than forking
to a particle method — **granular FSI comes for free** from the FSI you already
have.

One honest caveat: the BDIM band smears the body surface over ~2 cells, and
granular pressure under a quasi-static contact is much stiffer than
hydrodynamic pressure, so contact reaction forces will be lower-fidelity than a
dedicated contact model. Adequate for locomotion-over-sand studies; not a
substitute for a tribology/contact-mechanics solver.

---

## 8. Test / validation ladder

Mirror the existing two-phase validation style
(`lilytorch/src/test_two_phase.py`, `validation/two_phase_2d/`).

1. **Op unit test** — `ops.granular_viscosity`: for a uniform shear with known
   `p`, `γ̇`, check `ν_eff` equals the analytic `μ(I)p/(ργ̇)`; check the
   `nu_max` clamp and the `p_min` clamp activate correctly. fp64.
2. **Pure shear (planar)** — steady simple shear under gravity-loaded `p`;
   compare the velocity profile to the Bagnold profile predicted by μ(I).
3. **Angle of repose** — release a heap; the static surface angle must settle
   to `atan(μ_s)` within a few degrees (Pouliquen–Forterre `h_stop`).
4. **Granular column collapse (2-D)** — the canonical benchmark
   (Lube/Lajeunesse/Lagrée–Staron–Popinet): runout length and final deposit
   height vs. initial aspect ratio `a = H/L`. Target: within ~10–15 % of the
   experimental correlation across `a ∈ [0.5, 10]`. This is the headline
   acceptance test.
5. **Hopper / silo discharge** — Beverloo scaling of mass flow rate vs.
   aperture width `W` (`Q ∝ W^{(2d-1)/2}`); a qualitative arching check.
6. **Coupled** — a prescribed rigid wheel or the existing salamander foot
   driven through a granular bed; check the reaction force is finite, smooth,
   and scales sensibly with depth, and that the FARMS step stays stable.

Each test at `fp64` for the op-level checks; runs in
`lilytorch/src/test_granular.py` and `validation/granular_2d/`.

---

## 9. Recommended implementation order (milestones)

Each milestone is independently shippable; gate everything new on
`use_granular` so the single-phase and two-phase paths are byte-for-byte
unchanged when it is off.

**G0 — Viscosity op + unit tests.** `ops.granular_viscosity` (§4) and tests
1–2 (§8). Pure-Python/PyTorch, no solver wiring. Cheapest, fully testable in
isolation. *(~3–5 days)*

**G1 — `GranularSolver` skeleton.** Subclass `TwoPhaseSolver`; add the
`granular` config schema (§5), the `use_granular` flag, `_compute_granular_nu_t`
(reads lagged `self.p0`), and the `_compute_nu_t` dispatch branch. Reuse VOF
for the granular/air surface. Validate angle of repose (test 3). *(~1 wk)*

**G2 — Regularisation + diagnostics.** Barker & Gray (2017) closure (§6),
extend `LILYTORCH_UMAX_PROBE` to print `I/γ̇/p`. Without this, G3 will be
mesh-noise-limited. *(~1 wk)*

**G3 — Column collapse validation.** The acceptance test (test 4). Tune
`nu_max`/`γ̇_min`/`p_min`, document the stable CFL envelope. This is where the
real debugging lives. *(~1–2 wk)*

**G4 — Coupled body-in-sand.** Wire a prescribed body (then a FARMS animat)
through a granular bed; validate reaction-force smoothness (test 6). *(~1 wk)*

**G5 — Immersed granular (optional).** Generalise `I → I_v` (Cassar et al.
2005) so sand *under water* uses the interstitial-fluid timescale. Lets a
swimmer interact with a settling/eroding sandy bed. *(post-G4 research)*

G0–G3 are ~70 % of the effort (the viscosity closure and its stabilisation);
G4–G5 are incremental because the coupling and forces are inherited.

---

## 10. Risks and open questions

- **Ill-posedness (the big one).** Without §6 regularisation, expect
  grid-dependent blow-ups; do not interpret early instability as a bug before
  ruling out Hadamard instability. Budget G2 accordingly.
- **Pressure-lag accuracy.** One-step lag is standard but couples weakly under
  rapid loading; the optional `granular_pressure_subiter` fixed point is the
  fallback (reuses the `consistent_n_cycles` machinery).
- **Quasi-static / jamming limit.** A continuum μ(I) with a viscous regularised
  "frozen" region is **not** a true elastic/elastoplastic solid — it slowly
  creeps and cannot store static shear stress indefinitely. A genuinely
  static, fully-jammed pile under long-term load is outside μ(I)'s validity;
  for that, an elastoplastic (Drucker–Prager) or MPM treatment is required.
- **Density / dilatancy.** μ(I) as implemented assumes constant solid fraction
  `φ` (incompressible). Real sand dilates/compacts (Roux–Radjai, `μ(I)–φ(I)`
  models). Out of scope for v1; the incompressible projection forbids it
  without a compressible extension.
- **Shared triple-point singularity** with the open two-phase bug (§6) —
  resolve/characterise that first.
- **Validation data.** Column-collapse correlations have ~10–20 % scatter
  across experiments; set acceptance tolerances accordingly, don't chase
  spurious precision.

---

## 11. Out of scope (for now)

- Discrete-element (DEM) grain-scale dynamics; segregation; force chains.
- Granular **gas** / dilute collisional regime (kinetic theory).
- Compressible `μ(I)–φ(I)` dilatancy.
- True elastoplastic static piles (Drucker–Prager / MPM).
- Fracture, clogging, cohesion (wet-sand capillary bridges) — a cohesive yield
  term `τ_c` could be bolted on later as an additive constant in
  `granular_viscosity`, but is unvalidated.

These are the boundary where the continuum model stops being the right tool and
MPM/DEM becomes necessary — a separate strategic decision, not an increment on
this design.

---

## 12. TL;DR for the next agent

1. Read Lagrée–Staron–Popinet 2011 (the template), Jop et al. 2006 (the law),
   and Barker & Gray 2017 (the stabiliser).
2. Add `ops.granular_viscosity` — a clone of `ops.carreau_viscosity`
   (`operations.py:295`) with a **pressure-dependent** yield term
   `μ_s·p` instead of constant `τ_y`. Unit-test it standalone (G0).
3. Add `GranularSolver(TwoPhaseSolver)`; reuse VOF for the granular/air
   surface; compute `ν_eff` from the **lagged** `self.p0` inside the existing
   `_compute_nu_t` dispatch (G1).
4. Keep `dt` and the projection/BDIM/forces/FARMS coupling **unchanged** —
   the only new physics is the viscosity closure.
5. Regularise (Barker & Gray) before trusting any result (G2).
6. Validate on **granular column collapse** before declaring success (G3).
