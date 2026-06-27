#!/usr/bin/env sh
# ── Toy boat two-phase MuJoCo<->BDIM coupled demo ──────────────────────────
# A light box hull (~500 kg/m³) bobs at the air/water interface of a small
# 3-D tank.  The two-phase VOF solver models both water and real air, so
# buoyancy and dynamic pressure forces emerge from the variable-density fluid.
#
# Run with::
#     cd lilytorch/farms_examples/boat
#     bash run.sh
#
# For a quick test, reduce Nx/Ny/Nz and n_iterations in gen_configs.py first.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Generate configs, pool/water SDFs, and launch the simulation.
# gen_configs.py inherits from BaseSimConfig and auto-generates all YAML
# configs in a timestamped output folder, then runs farmsim.
"$PYTHON_BIN" gen_configs.py "$@"
