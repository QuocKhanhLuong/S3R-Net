# S3R/SSR Block Audit

This audit temporarily stops KD, teacher, and learned-gate experiments and
returns to the original S3R/SSR block. The anchor run is:

```text
s3r_net_acdc_25d_b8_224
```

## What changed

- `SSRFullBlock` now supports `block_variant`:
  - `s3r_full`
  - `s3r_gamma0`
  - `s3r_fft_identity`
  - `s3r_fixed_band`
  - `s3r_no_suppress`
  - `s3r_simple_spectral`
- The trainer accepts `--block_variant`.
- Block logs are saved to both:
  - `ssr_logs.csv`
  - `s3r_block_logs.csv`
- W&B receives block metrics under keys like:
  - `s3r_block/val/stage1_1/high_freq_ratio`
  - `s3r_block/val/stage1_1/update_gate_mean/b0`

## Diagnostics

Each S3R/SSR block logs:

- residual strength: `feature_norm`, `delta_norm`, `gamma_delta_norm`,
  `residual_ratio`
- gamma: `gamma_raw`, `gamma_effective`, `gamma`
- per-band gates: `retain_gate_mean/std`, `update_gate_mean/std`,
  `suppress_gate_mean/std`
- spectral stats: `energy`, `relative_energy`, `log_energy`, `variance`,
  `phase_coherence`
- high-frequency behavior: `high_freq_ratio`, `high_freq_penalty`,
  `boundary_high_density`, `nonboundary_high_density`,
  `boundary_to_nonboundary_high_ratio`
- stability: `block_input_mean/std`, `block_output_mean/std`,
  `output_delta_mean/std`
- FFT sanity: `fft_reconstruction_error`

## Short controlled run

Run all six variants for 50 epochs:

```bash
cd /home/linhdang/workspace/quockhanh_workspace/SpecMamba
conda activate specmamba

DATA_DIR=preprocessed_data/ACDC \
EPOCHS=50 \
BATCH_SIZE=8 \
IMAGE_SIZE=224 \
DEVICE=cuda \
WANDB=1 \
bash scripts/run_s3r_block_audit.sh
```

Single variant example:

```bash
python src/training/train_s3r_acdc.py \
  --model s3r_net \
  --block_variant s3r_gamma0 \
  --data_dir preprocessed_data/ACDC \
  --input_mode 25d \
  --in_channels 5 \
  --image_size 224 \
  --batch_size 8 \
  --epochs 50 \
  --base_channels 48 \
  --device cuda \
  --return_logs \
  --save_dir weights/block_audit/s3r_gamma0 \
  --run_name block_audit_s3r_gamma0
```

## Required analysis

After the 50-epoch runs, compare `summary.json`, `training_log.csv`, and
`s3r_block_logs.csv` for:

1. best DSC;
2. best HD95/ASSD;
3. most stable train/val curve;
4. whether `s3r_gamma0` is close to `s3r_full`;
5. whether `s3r_no_suppress` beats `s3r_full`;
6. whether `s3r_fixed_band` beats learned gates;
7. whether `hf_ratio_penalty` is active or simply below threshold;
8. whether `fft_reconstruction_error` stays near zero;
9. which block should become the new baseline.

Decision rules:

- If `s3r_gamma0` is close to `s3r_full`, the spectral residual is not
  contributing enough.
- If `s3r_no_suppress` beats `s3r_full`, remove or weaken suppress.
- If `s3r_fixed_band` beats learned gates, simplify the gate.
- If `s3r_simple_spectral` is close to full but more stable, promote the simpler
  block.
- If `s3r_full` clearly wins, keep it and focus on logging/regularization.

Generate the first-pass summary:

```bash
python scripts/summarize_s3r_block_audit.py \
  --root weights/block_audit \
  --output weights/block_audit/block_audit_summary.md
```
