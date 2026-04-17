#!/usr/bin/env sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

FARMSIM_BIN="${FARMSIM_BIN:-farmsim}"

"$FARMSIM_BIN" --experiment_config experiment_config.yaml "$@"
