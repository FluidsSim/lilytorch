# Force benchmarks — eulerian vs lagrangian readouts, vs analytic and published references

Everything for validating and comparing the two hydrodynamic force readouts. Start with
`HANDOFF_NEXT_AGENT.md` — it carries the current state, what is trusted, what is **not**, and the
planned work.

> **Retired (2026-07-19):** the `deltaH` eulerian submethod and the legacy standalone **python**
> eulerian force path are gone. The union `ndelta` readout (`force_submethod=0`, selected by
> `solver.force_link_normal="union"`, the default) inherits deltaH's gauge fix — the two-phase gauge
> tests confirm it — so deltaH is no longer needed. `sm2` (`force_submethod=2`,
> `force_link_normal="body"`) survives as an analysis-only per-link-normal variant, **gauge-unsafe and
> forbidden with the two-phase solver**. `force_submethod` has an intentional numbering gap (0, then 2)
> where deltaH's `1` was. Historical `deltaH` numbers below are kept for the record.

## The two readouts, and the thing that makes comparing them hard

|  | pressure sampled at | viscous (σ) sampled at |
|---|---|---|
| **eulerian** (`force_method: eulerian`) | `φ = 0` (never shifted) | `φ = eps_solver` (= `eps_multiplier·h`) |
| **lagrangian** (`force_method: lagrangian`) | `φ = lagrangian_sample_offset` | `φ = lagrangian_sample_offset` |

**The two channels move independently in one method and together in the other.** The eulerian pins
pressure at the surface and shifts only σ (the Maertens & Weymouth convention: get σ out of the BDIM
band, where the μ0 blend makes `ε(u_blend) ≈ μ0·ε(u_fluid)`). The lagrangian has ONE knob that moves
both. So "eulerian at s" vs "lagrangian at off=s" is **not** an apples-to-apples comparison of the
readouts — it also changes *where pressure is read*. Any comparison of NET force or swim speed
between the two methods to date is confounded by this. See `HANDOFF_NEXT_AGENT.md` §B1, which
proposes splitting both methods into `sample_offset_pressure` / `sample_offset_friction` so the two
can be pinned to identical sampling locations and the *readout* difference isolated.

## What is here

### Analytic oracles — absolute accuracy (no reference run needed)
| script | what it does |
|---|---|
| `oracle_native_three_way.py` | ndelta (union-∇H, `force_submethod=0`) vs sm2 (union coarea × per-body normal, `force_submethod=2`) vs lagrangian on an exact sphere with closed-form answers (divergence theorem). Drives the **native** op — the production path. Prints ratio-to-exact vs R/h and eps. |
| `oracle_python_path.py` | same two analytic cases through a real `FluidSolver` (python path). |

The same cases are pinned as **physics** tests in `lilytorch/tests/test_forces.py::test_oracle_*`
(~1 s; the only force tests in the repo that check physics rather than CPU/GPU parity).

### Frozen-field sweeps — both readouts on ONE identical field
| script | what it does |
|---|---|
| `zfish_snapshot_hook.py` | dumps every argument the live solver hands the force op at one step. Wraps both readouts; captures the world-frame triangulation at the force call site (the only place it is guaranteed world-frame). |
| `gen_zfish_snapshot.py` | runs the production zebrafish config headless and takes that snapshot (~15 s). `ZFISH_SNAP_FORCE_METHOD=lagrangian` also captures the triangulation. |
| `shift_sweep_3d.py` | re-drives the readouts on the frozen field at any band shift / sample offset. Also reports the geometry (inscribed radius vs band width, volume inflation). |
| `shift_sweep_2d.py` | 2-D twin, on a saved cylinder field. |

No trajectory divergence — this is the only way to compare readouts on *identical* fluid state.

### Live coupled runs — does it swim right, and do the books balance
| script | what it does |
|---|---|
| `gen_zfish_readout_arbitration.py` | runs the production zebrafish config headless into a `{stack}/{case}` layout. Env: `ZFISH_CASE`, `ZFISH_FORCE_METHOD`, `ZFISH_LAGR_OFFSET`, `ZFISH_LAGR_OFFSET_CELLS`, `ZFISH_DELTA_ORDER`, `ZFISH_NITER`, `ZFISH_REFINE`, `ZFISH_STACK`. |
| `zfish_pbdim_closure.py` | **the arbiter.** `closure = ∫P_BDIM dt / (ΔE_k + E_diss)`, where `P_BDIM = −Σ(F·v + τ·ω)` comes from the readout and the denominator from the fluid fields alone. |
| `zfish_swim_speed.py` | net COM displacement / mean speed per case. |

**`verify_energy_balance.py` (in `zebrafish_ki_project/`) is NOT an arbiter for readouts** — it
returns identical numbers for every readout, because actuator power is ~5000-12000× the power
reaching the fluid and a position-controlled fish tracks the same gait regardless. Use
`zfish_pbdim_closure.py`.

**Closure is a screen, not a scoreboard.** On a fine grid the two readouts closed within 2 points of
each other (105.3% vs 107.1%) while their swim speeds still differed by 35%.

## Published-reference benchmarks (left in place — reference data and paths live with them)

These were not moved: they carry their own reference CSVs, plotting scripts and configs, and are
referenced from docs and memory. Index only.

| case | reference | where | notes |
|---|---|---|---|
| impulsively started cylinder, Re=550 | Koumoutsakos & Leonard 1995 | `lilytorch/validation/cylinder_drag_2d/` | `run_cylinder_drag.py` has env-overridable `FORCE_METHOD` / `BDIM_MU0_PROJECTION` / `ZERO_PRESSURE_INSIDE` / `NX`. `plot_cylinder_drag.py` gives the **decomposed viscous/pressure** plot — the channel under suspicion. R/h=25.6. |
| settling cylinder, U_t | Namkoong et al. 2008 (`U_t = −0.025 m/s`) | `lilytorch/examples/single_sphere_drop_gazzola/` | terminal velocity is a direct integral test of the readout: at U_t drag balances net weight exactly. Second density ratio in `..._low_density/`. |
| settling sphere, 3-D | Coquerelle | `lilytorch/examples/single_sphere_drop_coquerelle_3d/` | a d×ν sweep → varies R/h. **Trap: set `bdim_mu0_projection: False`.** |
| convergence tooling | — | `lilytorch/validation/error_analysis_cylinder_2d/` | may already do half the sweep work. |

## Quick start

```bash
python setup.py build_ext --inplace          # the op set moves with refactors; rebuild first
python -m pytest lilytorch/tests/test_forces.py -k oracle -q                  # ~1 s
python -m lilytorch.examples.force_benchmarks.oracle_native_three_way

ZFISH_SNAP_FORCE_METHOD=lagrangian \
  ZFISH_SNAP_OUT=/data/andreaferrario/ns_data/zfish_force_snapshot/snap_lagr.pt \
  python -m lilytorch.examples.force_benchmarks.gen_zfish_snapshot            # ~15 s
python -m lilytorch.examples.force_benchmarks.shift_sweep_3d \
  /data/andreaferrario/ns_data/zfish_force_snapshot/snap_lagr.pt

ZFISH_CASE=lagr_off1h ZFISH_FORCE_METHOD=lagrangian ZFISH_LAGR_OFFSET_CELLS=1.0 \
  python -m lilytorch.examples.force_benchmarks.gen_zfish_readout_arbitration # ~40 s
python -m lilytorch.examples.force_benchmarks.zfish_pbdim_closure
python -m lilytorch.examples.force_benchmarks.zfish_swim_speed
```

Full history and evidence: `lilytorch/milestones/force_readout_agreement_handoff.md` (§10 is current).
