#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
FARMSIM_BIN="${FARMSIM_BIN:-farmsim}"
"$PYTHON_BIN" prepare_3d_drop_study.py

CASES="d0p2_nu0p02 d0p2_nu0p05 d0p2_nu0p1 d0p3_nu0p02 d0p3_nu0p05 d0p3_nu0p1"

for c in $CASES; do
  echo "============================================================"
  echo "Running case: $c"
  echo "============================================================"
  FARMSIM_BIN="$FARMSIM_BIN" sh run_case.sh "$c" "$@"
  echo ""
done

echo "All 3-D dropped-sphere cases completed."