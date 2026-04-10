#!/usr/bin/env sh

set -eu

if [ "$#" -lt 1 ]; then
  echo "Usage: sh run_heun_euler_case.sh <case_name> [extra farmsim args]"
  echo "Example: sh run_heun_euler_case.sh euler_dt_0p000025"
  exit 1
fi

CASE_NAME="$1"
shift

EXP_FILE="experiment_config_${CASE_NAME}.yaml"

if [ ! -f "$EXP_FILE" ]; then
  echo "Case config not found: $EXP_FILE"
  echo "Run: /data/andreaferrario/venv_ns_312/bin/python prepare_heun_euler_study.py"
  echo "Then retry with one of the generated case names."
  exit 2
fi

echo "[run] farmsim --experiment_config $EXP_FILE $*"
farmsim --experiment_config "$EXP_FILE" "$@"
