# Issue: Propeller-induced pitch & yaw in two-phase boat simulation

## Summary
A 6 m DSYHS yacht hull in a two-phase (water+air) BDIM simulation floats
level and stable with **no propeller torque** (tau=0).  With **any non-zero
propeller torque**, the boat develops a strong nose-down pitch and a yaw to
port (left turn), instead of accelerating straight forward.

## What has been ruled out
- **Static trim / mass distribution**: COM is below the waterline, static
  pitch torque is ~0.3° (verified analytically).  Mass changes from 972 kg
  up to 3422 kg (extreme keel) only marginally reduce the effect.
- **Propeller geometry**: 2-blade, 3-blade, centred on the centreline (Z=0),
  horizontal shaft, tilted shaft, deep (Y=-0.45), shallow (Y=-0.12) — all
  produce the same qualitative behaviour.
- **Air pressure oscillations**: ν_air=0.05 does not fix it.
- **Joint damping**: values from 1.0 to 10.0 change the propeller speed
  but not the pitch/yaw pattern.
- **Force method**: eulerian crashes immediately; lagrangian is the only
  usable option.
- **Grid resolution**: h=0.10 (270×64×64) is the only stable resolution;
  h=0.05 (540×128×64) is unstable (fluid explosion at iter ~93) despite
  smaller Δt, stronger multigrid settings, and isotropic dx=dy=dz.
- **Body-body SDF overlap**: moving propeller deep enough to avoid hull
  intersection did not fix it.

## Observable facts
1. tau=0 → boat floats level, no pitch, no yaw
2. tau=+30 → nose-down pitch + port yaw, marginal forward motion
3. tau=-20 → same behaviour (direction unchanged — **pitch/yaw does NOT
   reverse with torque sign**)
4. The effect appears within the first few hundred iterations

## Key files
- `boat/gen_configs.py` — simulation config
- `boat/toy_boat.sdf` — MuJoCo/SDF model
- `lilytorch/src/two_phase_solver.py` — two-phase solver overrides
- `lilytorch/src/forces.py` — `forces_lagrangian_3d`
- `boat/MASS_INERTIA_NOTES.md` — prior analysis

## Fluid properties
- ρ_water=1000, ρ_air=1.2 (ratio 833:1)
- ν_water=1.0×10⁻⁶, ν_air=1.5×10⁻⁵ or 5.0×10⁻² (both tested)
- face_density: harmonic
- Two-phase VOF, BDIM immersed boundary, variable-density Poisson (multigrid,
  all-Neumann BCs)
- force_method: lagrangian (surface integral via marching-cubes triangulation)
- air_transparent_body: enabled (default)
