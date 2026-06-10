#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-preprocessed_data/ACDC}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
DEVICE="${DEVICE:-cuda}"
BASE_CHANNELS="${BASE_CHANNELS:-48}"
OUTPUT_ROOT="${OUTPUT_ROOT:-weights/block_audit}"
WANDB_ARGS=()

if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_ARGS=(--wandb --wandb_project "${WANDB_PROJECT:-s3r-acdc}")
fi

variants=(
  s3r_gamma0
  s3r_fft_identity
  s3r_fixed_band
  s3r_no_suppress
  s3r_simple_spectral
  s3r_full
)

for variant in "${variants[@]}"; do
  python src/training/train_s3r_acdc.py \
    --model s3r_net \
    --block_variant "${variant}" \
    --data_dir "${DATA_DIR}" \
    --input_mode 25d \
    --in_channels 5 \
    --image_size "${IMAGE_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --base_channels "${BASE_CHANNELS}" \
    --device "${DEVICE}" \
    --return_logs \
    --save_dir "${OUTPUT_ROOT}/${variant}" \
    --run_name "block_audit_${variant}" \
    "${WANDB_ARGS[@]}"
done
