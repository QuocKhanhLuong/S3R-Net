#!/usr/bin/env bash
set -euo pipefail

# Default server run: full-slice SSR full 2.5D, 3 seeds, 200 epochs, batch 8.
# To run both 2D and 2.5D seeds, use:
#   MODE=train_all bash test/run_ssr_full_suite.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-test/configs/ssr_full_acdc_224.yaml}"
MODE="${MODE:-train_25d}"
OUTPUT_ROOT="${OUTPUT_ROOT:-test/outputs}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FORCE="${FORCE:-0}"
INCLUDE_BASELINES="${INCLUDE_BASELINES:-0}"
RUN_ROBUSTNESS="${RUN_ROBUSTNESS:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"

COMMON_ARGS=(
  --config "${CONFIG}"
  --output_root "${OUTPUT_ROOT}"
  --python "${PYTHON_BIN}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --image_size "${IMAGE_SIZE}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
)

# Leave MAX_SLICES unset for full slices. Set MAX_SLICES=600 or 32 for capped runs.
if [[ -n "${MAX_SLICES:-}" ]]; then
  COMMON_ARGS+=(--max_slices "${MAX_SLICES}")
fi

if [[ "${FORCE}" == "1" ]]; then
  COMMON_ARGS+=(--force)
fi

if [[ "${INCLUDE_BASELINES}" == "1" ]]; then
  COMMON_ARGS+=(--include_baselines)
fi

echo "[SSR suite] training mode=${MODE} output_root=${OUTPUT_ROOT}"
"${PYTHON_BIN}" test/run_ssr_full_suite.py "${COMMON_ARGS[@]}" --mode "${MODE}"

if [[ "${RUN_ROBUSTNESS}" == "1" ]]; then
  echo "[SSR suite] robustness evaluation"
  ROBUST_ARGS=(
    --config "${CONFIG}"
    --output_root "${OUTPUT_ROOT}"
    --python "${PYTHON_BIN}"
  )
  if [[ "${FORCE}" == "1" ]]; then
    ROBUST_ARGS+=(--force)
  fi
  "${PYTHON_BIN}" test/run_ssr_full_suite.py "${ROBUST_ARGS[@]}" --mode robustness
fi

if [[ "${RUN_AGGREGATE}" == "1" ]]; then
  echo "[SSR suite] aggregate benchmark"
  "${PYTHON_BIN}" test/run_ssr_full_suite.py \
    --config "${CONFIG}" \
    --output_root "${OUTPUT_ROOT}" \
    --python "${PYTHON_BIN}" \
    --mode aggregate
fi

echo "[SSR suite] done"
