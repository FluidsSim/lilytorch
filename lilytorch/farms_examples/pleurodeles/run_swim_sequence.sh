#!/usr/bin/env bash
set -euo pipefail

# Run 6 simulations sequentially with different swim kinematics CSVs.
# Usage:
#   bash lilytorch/farms_examples/pleurodeles/run_swim_sequence.sh 

SCRIPT="gen_configs_swim_3d_batch.py"
CONFIG_FILE="gen_configs_swim_3d_batch.py"
OUTPUT_ROOT="/data/andreaferrario/ns_data"
CSV_LIST=(
  nominal_swim_2.5.csv
  nominal_swim_3.0.csv
  nominal_swim_3.5.csv
  nominal_swim_4.0.csv
  nominal_swim_4.5.csv
  nominal_swim_5.0.csv
)

# Suggested step counts to match the Freq4.5 reference travel distance.
declare -A N_ITERATIONS_BY_CSV=(
  [nominal_swim_2.5.csv]=8098
  [nominal_swim_3.0.csv]=6385
  [nominal_swim_3.5.csv]=5274
  [nominal_swim_4.0.csv]=4511
  [nominal_swim_4.5.csv]=4000
  [nominal_swim_5.0.csv]=3575
)

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  echo "Output root not found: ${OUTPUT_ROOT}" >&2
  exit 1
fi

# Restore original config on exit.
ORIGINAL_CONFIG_CONTENT="$(cat "${CONFIG_FILE}")"
cleanup() {
  printf "%s" "${ORIGINAL_CONFIG_CONTENT}" > "${CONFIG_FILE}"
}
trap cleanup EXIT

for csv in "${CSV_LIST[@]}"; do
  latest_before="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"

  n_iter="${N_ITERATIONS_BY_CSV[${csv}]:-}"
  if [[ -z "${n_iter}" ]]; then
    echo "No n_iterations mapping found for ${csv}" >&2
    exit 1
  fi

  # Patch `self.n_iterations = ...` for this frequency.
  sed -E -i "s/^([[:space:]]*self\.n_iterations[[:space:]]*=[[:space:]]*).*/\1${n_iter}/" "${CONFIG_FILE}"

  echo "==============================================="
  echo "Running with LILYTORCH_SWIM_CSV=${csv} and n_iterations=${n_iter}"
  echo "==============================================="
  LILYTORCH_SWIM_CSV="${csv}" python "${SCRIPT}"

  latest_after="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"

  freq="$(echo "${csv}" | sed -E 's/^nominal_swim_([0-9]+\.[0-9]+)\.csv$/\1/')"
  if [[ -z "${freq}" || "${freq}" == "${csv}" ]]; then
    echo "Could not parse frequency from CSV name: ${csv}" >&2
    exit 1
  fi

  target_dir="${OUTPUT_ROOT}/Freq${freq}_a"

  if [[ -z "${latest_after}" ]]; then
    echo "No output folder found in ${OUTPUT_ROOT} after run for ${csv}" >&2
    exit 1
  fi

  if [[ "${latest_after}" == "${latest_before}" ]]; then
    echo "Warning: newest output folder did not change after ${csv}. Skipping rename."
    continue
  fi

  if [[ "${latest_after}" == "${target_dir}" ]]; then
    echo "Newest folder already named $(basename "${target_dir}")."
    continue
  fi

  if [[ -e "${target_dir}" ]]; then
    echo "Target folder already exists: ${target_dir}" >&2
    echo "Refusing to overwrite; latest run folder left as: ${latest_after}" >&2
    exit 1
  fi

  mv "${latest_after}" "${target_dir}"
  echo "Renamed newest output folder to: ${target_dir}"
done

echo "All sequential runs completed."
