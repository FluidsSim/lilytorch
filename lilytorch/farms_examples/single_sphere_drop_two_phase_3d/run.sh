#!/usr/bin/env sh
# Two-phase MuJoCo<->BDIM coupled sphere-drop. Runs the FARMS task with the
# FluidExtension; BDIMhandler auto-builds a TwoPhaseSolver from the embedded
# bdim_yaml (it carries a ``solver.two_phase`` block) and the sphere floats via
# emergent buoyancy. Reduce bdim_yaml Nx/Ny/Nz and nt in simulation_config.yaml
# for a quicker look.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

FARMSIM_BIN="${FARMSIM_BIN:-farmsim}"

mkdir -p /tmp/two_phase_sphere_drop/output
"$FARMSIM_BIN" --experiment_config experiment_config.yaml "$@"
