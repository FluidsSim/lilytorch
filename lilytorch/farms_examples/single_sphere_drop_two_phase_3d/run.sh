#!/usr/bin/env sh
# Two-phase MuJoCo<->BDIM coupled sphere-drop. Runs the FARMS task with the
# FluidExtension; BDIMhandler auto-builds a TwoPhaseSolver from the embedded
# bdim_yaml (it carries a ``solver.two_phase`` block) and the sphere floats via
# emergent buoyancy. Reduce bdim_yaml Nx/Ny/Nz and nt in simulation_config.yaml
# for a quicker look.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

FARMSIM_BIN="${FARMSIM_BIN:-farmsim}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p /tmp/two_phase_sphere_drop/output
# (re)generate the open-tank + water SDFs sized to the fluid box
"$PYTHON_BIN" gen_arena.py
# sync the sphere SDFs (mass/inertia, striped mesh + texture)
"$PYTHON_BIN" gen_sphere.py

# FlowIsoGLViewer needs its OpenGL hook LD_PRELOAD-ed BEFORE the sim process
# starts (it intercepts mjr_render to draw the live air/water interface into
# MuJoCo's scene). prepare_iso_gl_hook_env builds the hook and returns the env;
# we export LD_PRELOAD + the hook var here, then launch farmsim.
eval "$("$PYTHON_BIN" - <<'PY'
from lilytorch.integration.flow_iso_gl_viewer import prepare_iso_gl_hook_env, _HOOK_ENV_VAR
env = prepare_iso_gl_hook_env()
print(f'export LD_PRELOAD="{env.get("LD_PRELOAD", "")}"')
print(f'export {_HOOK_ENV_VAR}="{env.get(_HOOK_ENV_VAR, "")}"')
PY
)"

"$FARMSIM_BIN" --experiment_config experiment_config.yaml "$@"
