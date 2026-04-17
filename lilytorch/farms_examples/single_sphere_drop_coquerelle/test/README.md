Coquerelle test harnesses

This folder contains experimental scripts used to isolate Heun/Euler behavior
outside the FARMS+MuJoCo runtime.

Files
- adhoc_fsi_coquerelle.py: standalone FSI harness with selectable fluid and body integrators.
- prescribed_trajectory_fluid_study.py: fluid-only force comparison on identical prescribed body trajectory.
- analyze_adhoc_fsi_coquerelle.py: plots and CSV summaries for available standalone runs.

Typical runs
- Heun fluid + Euler body:
  /data/andreaferrario/venv_ns_312/bin/python test/adhoc_fsi_coquerelle.py --heun --body-integrator euler --dt 5e-5 --t-end 0.32 --case-name test_heunFluid_bodyEuler_dt_0p00005

- Heun fluid + Heun body:
  /data/andreaferrario/venv_ns_312/bin/python test/adhoc_fsi_coquerelle.py --heun --body-integrator heun --dt 5e-5 --t-end 0.32 --case-name test_heunFluid_bodyHeun_dt_0p00005

- Drag-only isolation (no fluid pressure/viscous body forces):
  /data/andreaferrario/venv_ns_312/bin/python test/adhoc_fsi_coquerelle.py --heun --body-integrator euler --force-model drag_only --drag-coeff 1.8 --dt 5e-5 --t-end 0.16 --case-name test_drag_heunFluid_bodyEuler_dt_0p00005

- Heun body with single vs double force staging:
  /data/andreaferrario/venv_ns_312/bin/python test/adhoc_fsi_coquerelle.py --heun --body-integrator heun --force-model full --heun-force-update single --dt 5e-5 --t-end 0.16 --case-name test_full_heunFluid_bodyHeun_forceSingle_dt_0p00005
  /data/andreaferrario/venv_ns_312/bin/python test/adhoc_fsi_coquerelle.py --heun --body-integrator heun --force-model full --heun-force-update single_bodycorr --dt 5e-5 --t-end 0.16 --case-name test_full_heunFluid_bodyHeun_forceSingleBodyCorr_dt_0p00005
  /data/andreaferrario/venv_ns_312/bin/python test/adhoc_fsi_coquerelle.py --heun --body-integrator heun --force-model full --heun-force-update double --dt 5e-5 --t-end 0.16 --case-name test_full_heunFluid_bodyHeun_forceDouble_dt_0p00005

Notes
- The body Heun mode uses a second force probe at a predicted rigid state.
- `--heun-force-update single` reuses stage-1 force.
- `--heun-force-update single_bodycorr` keeps single-force rigid dynamics but updates body geometry at the correction stage.
- `--heun-force-update double` recomputes force at the predicted stage (applies only when `--body-integrator heun`).
- `--force-model drag_only` bypasses fluid pressure/viscous body forcing and applies linear drag `F = -c * v`.
- Fluid state advancement remains the stage-1 trajectory for continuity between steps.
- Outputs are written under /data/andreaferrario/ns_data/coquerelle_adhoc_study/<case_name>/.
