#!/usr/bin/env bash
set -euo pipefail

# Benchmark Phase Test 2 variant outputs after `test/run_phase2_variants.sh`.
#
# Examples:
#   bash test/benchmark_phase2_variants.sh
#   OUTPUT_ROOT=/path/to/outputs bash test/benchmark_phase2_variants.sh
#   bash test/benchmark_phase2_variants.sh baseline_ssr ssr_se_dcn ssr_full

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-test/outputs}"
RUN_PREFIX="${RUN_PREFIX:-ssr_phase2_acdc_224}"
VARIANT_SCRIPT="${VARIANT_SCRIPT:-test/run_phase2_variants.sh}"
OUT_DIR="${OUT_DIR:-}"
STRICT="${STRICT:-0}"

cmd=(
  "${PYTHON_BIN}"
  test/benchmark_phase2_variants.py
  --output_root "${OUTPUT_ROOT}"
  --run_prefix "${RUN_PREFIX}"
  --variant_script "${VARIANT_SCRIPT}"
)

if [[ -n "${OUT_DIR}" ]]; then
  cmd+=(--out_dir "${OUT_DIR}")
fi

if [[ "${STRICT}" == "1" ]]; then
  cmd+=(--strict)
fi

if [[ "$#" -gt 0 ]]; then
  cmd+=(--variants "$@")
fi

printf 'Running benchmark:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
