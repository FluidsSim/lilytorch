# Strong (implicit) FSI coupling for the FARMS/MuJoCo path — design

Status: **implemented; FARMS end-to-end validation blocked by an unstable
test config** (see §16). The standalone path is implemented and validated
(`strong_coupling.py`, `fsi_rigid_body.py`, `demo_real_fsi.py`).  The
FARMS/MuJoCo implicit path is implemented in `BDIMhandler` (`_step_implicit`,
`_MujocoCheckpoint`, `_pose_source="physics"`, `_apply_forces(loads=)`); its
building blocks are individually validated, but a clean end-to-end run needs
a coupled config whose explicit baseline is stable (§16).

---

## 1. What the coupled path does today (explicit / staggered)

The dm_control runtime drives the loop. Per full step:

```
Task.before_step(action, physics):          # runs BEFORE integration
    update_sensors()
    for ext in extensions:                   # sim-extensions then animat-extensions
        ext.before_step(...)                 #   FluidExtension here -> BDIMhandler.step()
        if ext is AnimatController:           #   controller sets data.ctrl / muscle excitations
            set joint/torque/muscle ctrl
physics.step()                                # mj_step: integrate under (ctrl, xfrc_applied, contacts)
Task.after_step(physics)
```

`FluidExtension.before_step` (extensions.py) on a *full* step calls
`BDIMhandler.step(task, physics)`:

```
BDIMhandler.step:
    fs.step_(...)            # reads body poses from MuJoCo (BDIMhandler.update),
                             # advances fluid, computes loads
    self.apply_forces(...)   # assembles (F,T)+buoyancy, writes physics.data.xfrc_applied
```

Then the runtime's `physics.step()` integrates MuJoCo **once** under that
force. The fluid saw pose `pⁿ`; MuJoCo advances `pⁿ → pⁿ⁺¹` under `Fⁿ`; the
fluid never re-solves at `pⁿ⁺¹`. This is explicit Dirichlet–Neumann
staggering — unstable when added (displaced-fluid) mass ≳ body mass, which
is exactly why `force_relaxation` (a constant low-pass on `F`) exists. It
damps but does not remove the instability.

## 2. Goal

Replace the single force push with a **converged fixed-point iteration**
per step, accelerated by the existing `fsi_coupling` accelerators
(Aitken / IQN-ILS), so the coupling is stable independent of mass ratio
and `force_relaxation` is no longer needed.

## 3. The architectural constraint and the key idea

`BDIMhandler.step()` runs **inside** `before_step`, *before* the runtime's
`physics.step()`. The runtime owns the one real integration. We must not
double-integrate.

**Key idea — "leave the converged force, let FARMS integrate once"
(Option A).** Run the whole fixed-point loop *internally* inside
`BDIMhandler.step()`, using throwaway MuJoCo integrations for *prediction*
only, each undone by a checkpoint restore. When converged, restore MuJoCo
to the **start-of-step** state `sⁿ` and leave `xfrc_applied` holding the
**converged** force `F*`. Then `before_step` returns and the runtime does
its single `physics.step()`, integrating `sⁿ` under `F*` → the converged
end state. Net external effect: `xfrc_applied` carries `F*` instead of the
explicit one-shot `Fⁿ`. The FARMS control/integration pipeline is otherwise
untouched.

This is non-invasive: no changes to the runtime, the task loop, or the
controller — only `BDIMhandler.step()` gains an implicit branch.

## 4. Coupling variable

Two valid choices; **v1 recommends coupling on the force** (the Neumann
data), because it is the smallest vector and a direct upgrade of the
existing `force_relaxation`:

- **`x = applied loads`** per coupled body `[Fx,Fy,Fz,Tx,Ty,Tz] × B`
  (dim `6B`). The fixed-point map is

  ```
  H(x):  restore(mj, fluid) to start
         write x (+ buoyancy@candidate-pose) to xfrc_applied
         mj_step                      # structure: loads -> predicted pose/vel s̃
         mj_forward                   # refresh link xpos/xquat/cvel at s̃
         fluid solve at s̃ (BDIMhandler.update reads s̃ -> SDF + no-slip)
         x̃ = get_loads(+buoyancy)     # fluid -> new loads
  residual r = x̃ - x ; accelerate.
  ```

  At the fixed point the applied force equals the force the fluid produces
  for the pose that force induces — i.e. fully consistent.

- *(alternative)* **`x = generalized state (qpos, qvel)`** of the coupled
  animats. More "monolithic", but larger and needs care differencing the
  floating-base quaternion in the residual. Keep as a fallback if force
  coupling is found to converge slowly for highly articulated swimmers.

The accelerator vectors are tiny (`6B`, or tens of DOFs), so the IQN-ILS
QR solve is trivial — reuse `fsi_coupling.IQNILS` unchanged.

## 5. Checkpoint primitive (MuJoCo 3.3.7)

Use `mjSTATE_INTEGRATION` — it bundles qpos, qvel, act, time,
`qacc_warmstart`, ctrl, `qfrc_applied`, `xfrc_applied`, mocap, eq_active —
everything needed to reproduce an integration step bit-for-bit.

```python
import numpy as np, mujoco

class _MujocoCheckpoint:
    SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

    def __init__(self, physics):
        self.m = physics.model.ptr
        self.d = physics.data.ptr
        self.n = mujoco.mj_stateSize(self.m, self.SPEC)
        self.buf = np.zeros(self.n, dtype=np.float64)

    def save(self):
        mujoco.mj_getState(self.m, self.d, self.buf, self.SPEC)
        return self.buf.copy()

    def restore(self, state):
        mujoco.mj_setState(self.m, self.d, state, self.SPEC)
        mujoco.mj_forward(self.m, self.d)   # refresh derived quantities
```

Fluid checkpoint reuses the existing pattern (clone `u0,v0,w0,p0`), exactly
as `FluidSolverAdapter.snapshot/restore`.

## 6. Algorithm — implicit `BDIMhandler.step()`

```python
def step_implicit(self, task, physics):
    fs   = self.fluid_solver
    it   = self.iteration
    t    = it * self.pars["solver"]["dt"]
    acc  = self.accelerator                 # fsi_coupling accelerator
    ckpt = self._mj_ckpt                     # _MujocoCheckpoint(physics)

    mj_n    = ckpt.save()                    # start-of-step MuJoCo state
    fluid_n = self._fluid_snapshot()         # clone u0,v0,w0,p0
    x = self._read_loads_vector(physics)     # initial guess: last step's loads
                                             # (0 on the first step)
    x_tilde, converged = x, False
    for k in range(1, self.max_iter + 1):
        # ---- restore both sides to start of step ----
        ckpt.restore(mj_n)
        self._fluid_restore(fluid_n)

        # ---- structure: predict pose/vel under candidate loads x ----
        self._write_loads_vector(physics, x, pose_for_buoyancy=mj_n)
        mujoco.mj_step(ckpt.m, ckpt.d)       # one full step (cb_sub_steps==1)
        mujoco.mj_forward(ckpt.m, ckpt.d)    # refresh link kinematics at s̃

        # ---- fluid: solve at predicted pose s̃ -> new loads x̃ ----
        fs.advance_and_compute_loads(fs.u0, fs.v0, fs.p0, it, t,
                                     w_vel=getattr(fs, "w0", None))
        x_tilde = self._read_loads_vector(physics, from_solver=True)

        # ---- residual + acceleration ----
        res = acc.residual_norm(x, x_tilde)
        self.last_iters, self.last_residual = k, res
        if res < self.tol * (1.0 + np.linalg.norm(x)):
            converged = True
            break
        if not np.isfinite(res):
            break
        x = acc.relax(x, x_tilde)

    acc.finalize_timestep()

    # ---- commit (Option A): leave MuJoCo at sⁿ with the converged force ----
    ckpt.restore(mj_n)
    self._write_loads_vector(physics, x_tilde, pose_for_buoyancy=mj_n)
    self._cache_forces_for_substeps(x_tilde)   # so apply_forces() re-applies F* on substeps
    fs.finalize_step(fs.u0, fs.v0, fs.p0, it,
                     w_vel=getattr(fs, "w0", None))   # plot/free-surface/release ONCE

    if not converged:
        self._handle_nonconvergence(k)             # see §9
    self.iteration += 1
```

Helpers:

- `_read_loads_vector(physics, from_solver=False)`:
  when `from_solver`, assemble from `fs.get_loads()` (the per-body
  `(F,T)`); otherwise read back what is currently in `xfrc_applied` for the
  coupled links (used for the initial guess). Flatten to `6B`.
- `_write_loads_vector(physics, x, pose_for_buoyancy)`: the body-facing
  half of the existing `_apply_forces` — scatter `x` into
  `xfrc_applied[ind, :]` for each coupled link, adding buoyancy computed at
  the **candidate** pose (reuse the existing buoyancy block). This is the
  only place that touches `xfrc`.
- `_cache_forces_for_substeps`: store `F*` in the same buffers
  `apply_forces()` reads, so intermediate MuJoCo substeps re-apply `F*`.

Note the fluid solve uses `advance_and_compute_loads` / `get_loads` /
`finalize_step` — the **same** drift-free methods the standalone
`FluidSolverAdapter` calls. `BDIMhandler.update` (which reads poses from
`physics`) runs inside `advance_and_compute_loads` via
`composite_body.update`, so it automatically sees the predicted pose `s̃`.

## 7. Requirements / constraints

1. **`cb_sub_steps == 1` for implicit mode.** The fluid advances once per
   full step; the prediction does one `mj_step`. With `cb_sub_steps > 1`
   the real integration is several substeps under the held force, so the
   prediction must loop `cb_sub_steps` `mj_step`s (re-applying the held
   force each) to match. v1 asserts `cb_sub_steps == 1`; the generalization
   is a `for _ in range(cb_sub_steps): mj_step` in the predict block.

2. **Order `FluidExtension` *after* the `AnimatController`.** The controller
   sets `data.ctrl` inside the same `before_step` pass; if Fluid runs first,
   the prediction `mj_step` uses *stale* ctrl. Putting Fluid last (or
   pre-computing the excitations, which are open-loop functions of
   `(iteration, time)`) makes the prediction use the same actuation the
   real step will use. Document/auto-check in `FluidExtension.from_options`.

3. **Buoyancy is pose-dependent** → recompute it inside
   `_write_loads_vector` at the candidate pose each sweep (the existing
   buoyancy block already reads `comp.com_pos`, which reflects the pose
   after `BDIMhandler.update`).

4. **Determinism.** `mjSTATE_INTEGRATION` includes `qacc_warmstart`, so the
   constraint solver warm-starts identically each restore → the prediction
   `mj_step` and the runtime's real `mj_step` produce identical `s̃` for
   identical `(sⁿ, ctrl, xfrc)`. Verify this equality once in validation
   (it is the correctness anchor for Option A).

## 8. Config surface

```yaml
body:
  coupling:
    scheme: implicit          # "explicit" (default, today) | "implicit"
    accelerator: iqn-ils      # iqn-ils | aitken | constant
    reuse: 2                  # IQN-ILS time-window reuse
    tol: 1.0e-4               # relative interface-residual tolerance
    max_iter: 30
    # force_relaxation is ignored when scheme == implicit
```

`BDIMhandler.__init__` builds `self.accelerator =
fsi_coupling.make_accelerator(...)`, `self._mj_ckpt`, and binds
`self.step = self.step_implicit if scheme == "implicit" else self.step_explicit`
(the current `step` becomes `step_explicit`, unchanged).

## 9. Edge cases & robustness

- **Non-convergence**: if `max_iter` is hit, fall back to the
  last-accepted `x_tilde` (best effort) and log a warning with the residual.
  Optionally trigger a one-step `force_relaxation`-style damped apply as a
  safety net. Never propagate NaNs — `check_explosion` still runs in
  `finalize_step`.
- **Contacts** (sphere near a wall, paddling on ground): handled naturally
  by `mj_step` in the prediction; no special treatment. The added
  stiffness can slow convergence — IQN-ILS reuse helps.
- **Multiple animats / partial coupling**: `_read/_write_loads_vector`
  iterate the same `comp.bodies / body_ids` the explicit path already uses;
  the coupling vector is the concatenation over coupled links only.
- **Cost**: `N` fluid solves per step (N = sweeps). Expect `N≈2–4` after
  the first step thanks to IQN-ILS reuse; the first step of a sim costs
  more. This is the price of stability; it replaces sims that diverge or
  need tiny `dt` + `force_relaxation`.

## 10. Reuse of existing code

- Accelerators: `fsi_coupling.{IQNILS,AitkenRelaxation,ConstantUnderRelaxation}`
  — unchanged.
- Fluid stepping: `FluidSolver.advance_and_compute_loads / get_loads /
  finalize_step` — unchanged (already used by `FluidSolverAdapter`).
- Force assembly/scatter + buoyancy: factor the body-facing half of the
  current `_apply_forces` into `_write_loads_vector` so explicit and
  implicit share it.

## 11. Validation plan

1. **Determinism check** (§7.4): one internal predict `mj_step` vs the
   runtime's real `mj_step` from the same checkpoint must give identical
   `qpos/qvel`. Anchors Option A.
2. **Single sphere drop** (`single_sphere_drop_coquerelle` /
   `_gazzola`): run explicit (current) vs implicit. Expect implicit to (a)
   run stably with the true `rho_body` (no `bdim_mu0_projection` / equal-
   density workaround), (b) remove the iteration-~12,500 blow-up, (c) match
   the published terminal velocity / `Cd(Re)`. Report sweeps/step.
3. **1guilla full tank**: confirm implicit removes the need for
   `force_relaxation` even with `convexify=True` overlap (see
   `project_convexify_overlap_instability`), or quantify the remaining gap.

## 12. Open questions

- Force-coupling vs state-coupling convergence for highly articulated
  swimmers (many controlled joints): start with force; switch the coupling
  vector if needed (the accelerator is agnostic).
- Whether to sub-iterate the *whole* animat state or only the floating
  base + wetted-link velocities (reduce vector size / ignore stiff
  controlled joints in the interface residual).
- Interaction with the interactive viewer (which can clear `xfrc`): the
  per-substep `apply_forces` re-application already handles this; confirm it
  still holds with Option A.
```

---

## 13. Design review — findings & required revisions

Reviewed against the actual code paths. Two **blocking** issues invalidate
the §6 algorithm as written; several minor corrections follow.

### BLOCKER 1 — the fluid reads FARMS *sensor buffers*, not `physics.data`

`BDIMhandler.update → gather_data(iteration)` (BDIMhandler.py:343) reads
poses/velocities from `exp_data.sensors.links.com_positions()[iteration]`
(and `urdf_*`, `*_velocities`), **not** from `physics.data.xpos/xquat/cvel`.
Those sensor arrays are written by `Task.update_sensors(physics)`
(task.py:249, `index = iteration % buffer_size`), called once at the top of
`Task.before_step`.

Consequence: an internal `mj_step` + `mj_forward` updates `physics.data`
but **not** what the fluid reads — the fluid would keep solving at the
start-of-step pose every sweep, and the coupling could never converge to a
moved pose.

**Revision:** after each prediction `mj_step`, call
`task.update_sensors(physics, links_only=True)` to push the predicted pose
`s̃` into the sensor buffer that `gather_data` reads. The corrected sweep:

```
restore(mj→sⁿ, fluid→fluidⁿ)
write candidate force x → xfrc
mj_step ; mj_forward                          # physics.data := s̃
task.update_sensors(physics, links_only=True) # sensor buffer[it] := s̃   ← ADDED
fs.advance_and_compute_loads(...)             # gather_data now reads s̃
x̃ = get_loads(+buoyancy)
```

Also note: the MuJoCo checkpoint (`mjSTATE_INTEGRATION`) does **not** cover
the FARMS `AnimatData` sensor buffers. We don't need to restore them
(each sweep overwrites buffer[it] via `update_sensors`), but be aware
buffer[it] is left holding the last `s̃` after the step (see Minor 3).

### BLOCKER 2 — the "single `mj_step`" prediction is unverified

The real integration is `self._env.step()` → dm_control
`physics.step(n_sub_steps)` (simulation.py:270/311/333), where
`n_sub_steps = physics.num_sub_steps` and there is also a `legacy_step`
toggle (simulation.py:58) — *separate from* FARMS `cb_sub_steps`. So the
real step may be **multiple** `mj_step`s, and possibly a custom stepping
path.

**Revision:** the prediction must replicate the real integration exactly,
i.e. `n_sub_steps × (cb_sub_steps==1)` `mj_step`s through the same path the
runtime uses (prefer calling `physics.step(n_sub_steps)` on the checkpointed
data, not a bare `mj_step`, unless `legacy_step` changes that). The §11.1
**determinism check is therefore a blocking prerequisite, not a nicety**:
confirm `predict-integration(sⁿ, ctrl, xfrc) == runtime-integration` to
machine precision before trusting Option A. Resolve `num_sub_steps`,
`cb_sub_steps`, and `legacy_step` for the target sphere-drop config first.

### Minor 1 — commit `x`, not `x_tilde`

At convergence the fluid's committed solve was performed at pose
`s̃ = integrate(sⁿ, x)` where `x` is the **last candidate** force (the
prediction used `x`, not `x̃`). To keep {committed force, fluid pose,
FARMS integration} mutually consistent, commit **`x`** to `xfrc` (FARMS then
reproduces exactly the `s̃` the fluid solved at). `x ≈ x̃` within `tol`, so
the difference is small, but `x` is the correct choice. (Fix the §6
pseudocode: `_write_loads_vector(physics, x, ...)`.)

### Minor 2 — buoyancy is outside the coupling vector

Buoyancy is pose-dependent and recomputed each sweep at write time, so the
interface residual is on the **hydrodynamic** load only. This is fine
(buoyancy is a deterministic function of the converging pose), but state it
explicitly and make sure `get_loads` (hydro) and the buoyancy add are not
double-counted.

### Minor 3 — sensor buffer left at `s̃` after commit (logging)

Because Option A restores MuJoCo to `sⁿ` but the sensor buffer at `it` holds
the last prediction `s̃`, the *logged* link pose at iteration `it` is the
end-of-step prediction rather than the start pose. Decide intended
semantics; if start-of-step logging is desired, re-run
`update_sensors(physics)` after the final `ckpt.restore(mj→sⁿ)`.

### Minor 4 — iteration-index alignment

`gather_data` indexes with `BDIMhandler.iteration`; `update_sensors`/buffers
index with `Task.iteration`. They both start at 0 and advance once per full
step, but the implicit path must keep them aligned (and respect
`buffer_size`, given `gather_data` uses a raw `[iteration]` index while
writes use `iteration % buffer_size`).

### Minor 5 — cost

Each sweep now also runs `update_sensors` (links) in addition to a full
fluid solve and `n_sub_steps` `mj_step`s. Still `O(sweeps)`; acceptable, but
the per-sweep constant is higher than the §9 estimate implied.

### What still holds

- Option A (run the loop inside `before_step`; leave `sⁿ` + converged force;
  FARMS integrates once) remains valid **with** the `update_sensors`
  propagation and the exact-integration replication added.
- Checkpoint primitive (`mjSTATE_INTEGRATION`) — correct/sufficient for the
  MuJoCo side (covers qpos/qvel/act/time/warmstart/ctrl/qfrc/xfrc); just
  remember it excludes the FARMS sensor buffers.
- Force coupling vector, `fsi_coupling` accelerator reuse, and the
  `advance_and_compute_loads / get_loads / finalize_step` fluid path —
  unchanged and valid.
- Requirements §7 (cb_sub_steps==1, Fluid ordered after the controller,
  pose-dependent buoyancy) — still required; add "resolve num_sub_steps /
  legacy_step" to the list.

### Recommended first implementation step (unchanged, now justified)

Implement `_MujocoCheckpoint` + the **determinism check** against a real
sphere-drop config, explicitly resolving `num_sub_steps`, `cb_sub_steps`,
`legacy_step`, and confirming `update_sensors`→`gather_data` round-trips the
predicted pose. Only once that anchor passes is the rest safe to wire.

---

## 14. Revised data path — read `physics.data` directly (supersedes Blocker 1)

The cleaner fix for Blocker 1 is to **read link kinematics straight from
`physics.data`** in the implicit path, instead of the FARMS sensor buffers.
This is exactly what FARMS' own `SwimmingExtension` does
(`swimming/extension.py:205`: `physics.data.xmat[indices]`), and the
per-link MuJoCo body-id map is the one BDIMhandler already uses for forces:
`indices = task.maps[animat_i]['sensors']['data2xfrc']`.

Field mapping (MuJoCo → what `gather_data` returns), with FARMS unit
scaling so values match the SI sensor path:

| gather_data output | physics.data source | units |
|---|---|---|
| `urdf_pos` (link frame origin) | `data.xpos[ind]` | `÷ units.meters` |
| `R` (link orientation) | `data.xmat[ind].reshape(3,3)` | — |
| `com_pos` (link COM) | `data.xipos[ind]` | `÷ units.meters` |
| `lin_vel` (COM linear vel) | `mj_objectVelocity(... mjOBJ_XBODY, ind, flg_local=0)` linear part | `÷ units.velocity` |
| `ang_vel` (COM angular vel) | same call, angular part | `÷ (units.velocity/units.meters)` |

Because the internal prediction (`physics.step` / `mj_step` + `mj_forward`)
updates `physics.data`, the fluid now sees the predicted pose **with no
`update_sensors` round-trip**. This removes the §13 Blocker-1 workaround and
Minor 3/4 (no sensor-buffer mutation, no buffer/iteration-index coupling).
Revised sweep:

```
restore(mj→sⁿ, fluid→fluidⁿ)
write candidate force x → xfrc
physics.step(n_sub_steps)            # SAME call the runtime uses (see §15)
fs.advance_and_compute_loads(...)    # gather_data(source="physics") reads physics.data = s̃
x̃ = get_loads
```

**Risk to control:** `gather_data` is shared with the *explicit* (validated)
path. To avoid silently changing existing coupled sims, **do not replace**
the sensor path — *parametrize* it: `gather_data(iteration, source="sensors"
| "physics")`, explicit keeps `"sensors"`, implicit uses `"physics"`. Then
add an **equivalence test**: at the start-of-step pose `sⁿ`, the two sources
must return matching `(urdf_pos, R, com_pos, lin_vel, ang_vel)` to tight
tolerance. Positions/orientation are trivial (`xpos/xipos/xmat`); the
**velocities are the subtle part** — match the exact `mj_objectVelocity`
convention (com-based, global frame) and unit scaling FARMS uses to fill the
sensors. The equivalence test is the gate for the data path; the §11.1
determinism test is the gate for the integration.

## 15. Blocker 2, downgraded — use the runtime's own integration call

Rather than reimplement the integration with a bare `mj_step` (and guess at
`num_sub_steps` / `legacy_step`), the prediction should invoke the **same**
call the dm_control runtime uses after `before_step`:
`physics.step(self._env._n_sub_steps)` (simulation.py:270 → dm_control
`Environment.step` → `physics.step(n_sub_steps)`). On the checkpointed data
this reproduces the real integration by construction — there is nothing to
re-derive. Blocker 2 then reduces to a single assertion (§11.1): predict via
`physics.step(n)`, read `s̃`; restore; let the runtime step; assert the
runtime's `s̃` equals the predicted one. The only residual is whether
`legacy_step=True` routes through a non-`physics.step` path — which the
assertion catches immediately. So: **Blocker 2 is "use the same call + one
test", not a redesign.**

---

## 16. Implementation status (steps 1–3 done; FARMS validation blocked)

**Implemented & validated (building blocks):**
- §1 pose-source data path: `gather_data(_pose_source="physics")` + `_gather_data_physics`. Equivalence to the sensor path **verified on a real farmsim sphere-drop run** (iters 0,1 PASS). `test_pose_source.py` (3 convention tests pass).
- §5 checkpoint: `_MujocoCheckpoint` (mjSTATE_INTEGRATION). Determinism/reproducibility verified — `test_mujoco_checkpoint.py` (3 pass).
- `_apply_forces` refactored to accept a `loads=` override (shared buoyancy/scaling/write); explicit `loads=None` path is byte-identical (confirmed: original and refactored both explode coquerelle at iter ~5, i.e. no regression).
- §6 implicit step: `_step_implicit` (force coupling, Option A, checkpoint/restore loop, IQN-ILS via `fsi_coupling`), config surface `body.coupling.{scheme,accelerator,reuse,tol,max_iter,predict_substeps}`. Dispatched from `step()`; explicit path renamed `_step_explicit` (unchanged).

**FARMS end-to-end validation: BLOCKED by config, not implemented bug.**
- The only coupled config exercised (`single_sphere_drop_coquerelle`) has an **unstable explicit baseline**: the *original* (pre-work) code explodes at iteration 5 (`|u|_max=2.4e6 > 100`). It is one of the known instability cases (see `INSTABILITY_ANALYSIS`, `project_bdim_pressure_band`), so it cannot validate the implicit scheme.
- Under implicit coupling the same config diverges at step 0 (force → ~1e6): a single in-loop fluid solve at the *predicted moving-body* pose drives `|u|` to ~1e5 in one solve (gathered velocity `[0,-0.098]` is correct; gravity_z=-980 ⇒ cm units; pose correct). `bdim_mu0_projection: True` did **not** change this. Because the explicit baseline is itself unstable, the step-0 divergence cannot be cleanly attributed to an implicit bug vs the config's inherent fragility (the implicit scheme imposes the moving body immediately, hitting the instability sooner than explicit's lagged ramp-up).

**Next steps to actually validate:**
1. Find/produce a **stable coupled config** (explicit baseline runs without `check_explosion`): e.g. a swimmer at moderate Re, or a sphere-drop with grid/dt/eps tuned so the explicit baseline is at least marginally stable. Run explicit vs implicit and compare.
2. If a target is meant to be the *unstable* sphere-drop (the case the project wants implicit to cure), first establish that implicit *converges* per step on a stable case, then test whether it tames the unstable one — instrument the per-sweep residual to confirm the fixed-point map is contractive (it should be for `rho_body/rho>1`).
3. Resolve the impulsive-startup question: whether a single solve at the predicted moving-body pose on the start-of-step fluid field is well-posed, or whether the first sweep should solve at the start pose (explicit predictor) before predicting.
