#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if [ "$#" -lt 1 ]; then
  echo "Usage: sh run_case.sh <case_name> [extra farmsim args]"
  echo "Example: sh run_case.sh d0p2_nu0p05"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
FARMSIM_BIN="${FARMSIM_BIN:-farmsim}"
CASE_NAME="$1"
shift

EXP_FILE="experiment_config_${CASE_NAME}.yaml"

if [ ! -f "$EXP_FILE" ]; then
  echo "[prepare] case configs not found, generating them first"
  "$PYTHON_BIN" prepare_3d_drop_study.py
fi

if [ ! -f "$EXP_FILE" ]; then
  echo "Case config not found: $EXP_FILE"
  exit 2
fi

echo "[run] $FARMSIM_BIN --experiment_config $EXP_FILE $*"
"$FARMSIM_BIN" --experiment_config "$EXP_FILE" "$@"
