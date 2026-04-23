#!/usr/bin/env sh

set -eu

CASES="heun_dt_0p00005 heun_dt_0p000025 euler_dt_0p00005 euler_dt_0p000025 euler_dt_0p0000125"

for c in $CASES; do
  echo "============================================================"
  echo "Running case: $c"
  echo "============================================================"
  sh run_heun_euler_case.sh "$c" "$@"
  echo ""
done

echo "All study cases completed."
