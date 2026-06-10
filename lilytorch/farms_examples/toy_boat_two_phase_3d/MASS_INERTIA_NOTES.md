# Toy-Boat Mass / Inertia Rebuild — Notes

## Symptom
The boat "sat on top of the water" (floated far too high) and developed an
extreme, persistent nose-down pitch. Initial suspicion was a wrong longitudinal
mass balance (COM too far forward).

## Root cause
The masses were **derived as if the hull were a nearly-empty shell** (972 kg
total), but the *buoyant* shape BDIM uses is the **convex hull** of each mesh
(`convexify=True`). Key volumes (mesh scale 0.0026970, hull L=6.04 m):

| Shape | Hull volume |
|-------|-------------|
| Real (concave) mesh | 1.614 m³ |
| Convex hull (what BDIM floats on) | 6.638 m³ (4× larger) |

With only 972 kg on the 6.69 m³ convex envelope, the effective density is
**145 kg/m³** → the boat floats at just **14.5 %** submergence ("sits on top").

> A floating body's *average* density (mass ÷ buoyant volume) must be < 1000.
> The "2000–3000 kg/m³" of a real boat is the **material** density and only
> floats because the hull is hollow (air inside). You cannot recover the mass by
> filling any solid hull (convex *or* concave) at material density — it sinks.

## Fix — masses/inertias from the REAL (concave) meshes
Computed with `trimesh.mass_properties` on the **non-convexified** meshes at
physical densities (inertia about each part's COM, mesh frame). BDIM buoyancy is
left on the convex envelope (unchanged).

| Part | ρ [kg/m³] | V (real) | Mass | COM (mesh x, y) |
|------|-----------|----------|------|-----------------|
| Hull | 2000 (material) | 1.614 m³ | 3227.3 kg | (1.919, +0.329) |
| Keel | 8000 (iron ballast) | 0.049 m³ | 393.6 kg | (2.469, −0.229) |
| Rudder | 2000 | 0.005 m³ | 10.2 kg | (−0.203, +0.054) |
| **Total** | | | **3631 kg** | (1.972, +0.268) |

Inertia tensors (about COM, mesh frame):
- Hull:   ixx 1416.6, iyy 8838.5, izz 7870.6, ixy 62.8, ixz −0.1, iyz 0.1
- Keel:   ixx 11.3, iyy 27.0, izz 37.4, ixy −9.5
- Rudder: ixx 0.43, iyy 0.05, izz 0.48, ixy −0.03

## Resulting float / trim / stability (on the convex BDIM envelope)
- Displaces 3.63 m³ → **54 % submerged** (proper waterline, no longer on top).
- **Trim balanced**: COB_x 1.931 vs COM_x 1.972 → +0.042 m → ~0.2° static trim.
- COM at world-z **0.153, below the waterline (0.40)** → statically stable.
- Equilibrium spawn: model-origin world-z **−0.115**.

## MuJoCo gotcha — thin-plate rudder precision
MuJoCo enforces the triangle inequality `A + B >= C` on the **principal** moments.
The rudder is a near-ideal thin plate, where `izz ≈ ixx + iyy` (perpendicular-axis
theorem), so the margin is tiny (A+B−C ≈ 0.0015). Rounding the inertia to 2
decimals collapses it to exactly the boundary and the compiler errors with
`inertia must satisfy A + B >= C`. Fix: emit the inertia at full precision (5
decimals here). All three parts satisfy the inequality at full precision
(margins: hull 447.7, keel 0.90, rudder 0.0015).

## Files changed
- `toy_boat.sdf` — hull/keel/rudder `<inertial>` blocks (mass, COM pose, inertia).
- `gen_configs.py` — `SPAWN_Z 0.256 → −0.115`, `BOAT_MASS 972 → 3631`.

## Why this is self-consistent (not a fudge of the convex-buoyancy choice)
Because BDIM floats on the *larger* convex envelope, a solid-concave-×-material-
density mass (denser than water as a solid, "wants to sink") settles to a sane
54 % draft instead of riding on top. Masses are now traceable to mesh geometry ×
a stated density rather than a hand-picked "empty boat" number.

## Postscript — the masses were a detour; the heavy rebuild was reverted
The user's real problem was an extreme stern-up **pitch**, not the float height.
We reverted to the realistic ~972 kg masses (a 6 m boat is ~1 t, not 3.6 t) after
establishing the pitch is **not** a mass/shape problem:

- **Trim is balanced on the convex envelope** (the masses were tuned to it):
  COM_x 2.254 over COB_x 2.192 → arm −0.062 m → static pitch torque **−594 N·m
  (~0.3°)**. The *concave* hull would be **5× worse** (COB_x 1.942, arm −0.312 m,
  −2979 N·m), so switching the buoyant shape would *increase* the static pitch.
- **The force integration matches the analytic hydrostatic moment** — the
  `_diag_hull_force.py` 11 kN·m figure was an artifact of measuring about the
  bbox-centre and flooding along the beam axis; about the real COM at the real
  waterline it is only −594 N·m.
- The marker-offset pitch-torque bug is already corrected
  (`BDIMhandler._lagr_marker_offset`).

**Conclusion:** the 45° pitch is **dynamic**, not static — a light free body on a
free surface under **explicit** FSI coupling is the textbook added-mass
instability regime (also recorded in `project_toy_boat_two_phase_debug`). The fix
is **implicit coupling** (`coupling.scheme="implicit"`, IQN-ILS), now enabled in
`gen_configs.py`. The mass/shape levers are exhausted.
