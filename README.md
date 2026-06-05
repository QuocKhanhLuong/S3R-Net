# SpecUMamba

SpecUMamba is now centered on **S3R-Net**: Spectral State-Space Retention Network for cardiac MRI segmentation.

The main task is ACDC segmentation with four classes:

| Label | Class |
| --- | --- |
| 0 | Background |
| 1 | RV |
| 2 | MYO |
| 3 | LV |

## Architecture Brief

S3R keeps an explicit pair of signals through the network:

```text
F_t = spatial / latent feature map
S_t = spectral state memory, shaped [B, K, C_s]

(F_t, S_t) -> (F_{t+1}, S_{t+1})
```

The core transition is `S3RTransitionBlock`:

```text
SpectralStateTransition updates S_t
SSRFullBlock updates F_t with FFT radial-band retention gates
StateGuidedModulation injects S_t back into F_t
```

`SSRFullBlock` is the promoted `ssr_full` prototype. It uses radial spectral bands, retain/update/suppress gates, bounded residual strength, large-kernel refinement, high-frequency ratio regularization, and boundary spectral diagnostics.

Two model sizes are exposed:

| Model | Use |
| --- | --- |
| `s3r_mini` | default practical training and smoke tests |
| `s3r_net` | deeper future architecture with staged downsample/reconstruction |
| `s3r` | alias for the recommended default |

S3R outputs segmentation logits, boundary logits, spectral-state logs, gate regularization, and high-frequency ratio penalty.

## Repository Layout

```text
src/models/s3r/        S3R blocks, spectral state, losses, metrics, models
src/training/          supervised S3R ACDC training
src/distillation/      S3R-SCSD dual-teacher distillation
src/data/              ACDC S3R slice dataset
scripts/               data preprocessing and split utilities
docs/                  detailed S3R model and distillation notes
tests/                 S3R and preprocessing tests
```

Detailed docs:

- `docs/S3R_MODEL.md`
- `docs/S3R_DISTILLATION.md`
- `docs/DUAL_TEACHER_KD.md`

## Supervised Training

```bash
python src/training/train_s3r_acdc.py \
  --model s3r_mini \
  --data_dir preprocessed_data/ACDC \
  --image_size 224 \
  --epochs 200 \
  --batch_size 8 \
  --input_mode 2d \
  --save_dir weights/s3r_mini_acdc
```

Smoke test:

```bash
python src/training/train_s3r_acdc.py \
  --model s3r_mini \
  --data_dir preprocessed_data/ACDC \
  --image_size 224 \
  --epochs 2 \
  --batch_size 8 \
  --max_slices 32 \
  --save_dir weights/smoke_s3r_mini
```

Console output is intentionally compact. Training prints the main metric table per class:

- Dice
- HD95
- Precision
- Recall
- ASSD

Useful reporting flags:

```bash
--no_tqdm
--wandb --wandb_project s3r-acdc --wandb_run_name my_run
```

## Distillation

S3R-SCSD is a phased dual-teacher framework:

| Phase | Description |
| --- | --- |
| `phase1_semantic` | supervised S3R + semantic teacher KD |
| `phase2_characteristic` | supervised S3R + boundary/distance/spectral KD |
| `phase3_dual_routing` | both teachers with region-aware routing |
| `phase4_state_kd` | phase 3 plus spectral-state KD |

Generate a GT-derived teacher cache for pipeline smoke testing:

```bash
python src/distillation/generate_gt_teacher_cache.py \
  --data_root preprocessed_data/ACDC \
  --output_dir teacher_cache/acdc_gt_debug \
  --image_size 224 \
  --max_slices 32
```

Run phase 3 distillation:

```bash
python src/distillation/train_s3r_distill.py \
  --config src/distillation/configs/s3r_scsd_phase3_dual_routing.yaml \
  --teacher_cache_dir teacher_cache/acdc_gt_debug \
  --epochs 2 \
  --batch_size 8 \
  --max_slices 32 \
  --run_name smoke_s3r_scsd_phase3
```

Distillation uses the same compact metric table and supports:

```bash
--no_tqdm
--wandb --wandb_project s3r-scsd --wandb_run_name my_distill_run
```

## Agreement-Aware Dual-Teacher KD

Optional training-time KD can use frozen Medical-SAM3 and CineMA teachers:

```bash
python src/training/train_s3r_acdc.py \
  --model s3r_net \
  --data_dir preprocessed_data/ACDC/training \
  --image_size 224 \
  --epochs 250 \
  --batch_size 8 \
  --input_mode 25d \
  --in_channels 5 \
  --base_channels 48 \
  --use_dual_teacher_kd \
  --teacher_cache_dir teacher_cache/acdc \
  --kd_temperature 4.0 \
  --lambda_field 0.3 \
  --lambda_cine_boundary 0.5 \
  --lambda_fuse 0.5 \
  --lambda_spec 0.05 \
  --save_dir weights/s3r_dual_teacher_kd \
  --wandb \
  --wandb_project s3r-acdc \
  --wandb_run_name s3r_net_dual_teacher_kd
```

Debug without real teacher weights:

```bash
python scripts/test_teacher_loading.py \
  --data_dir preprocessed_data/ACDC/training \
  --teacher_stub \
  --num_classes 4 \
  --device cuda
```

Teachers are frozen and used only during training. Inference remains S3R-only.

## Data

Default supervised and distillation commands expect:

```text
preprocessed_data/ACDC
```

Patient-level splitting is handled by:

```bash
python scripts/acdc_split.py
```

Preprocessing utilities are kept under `scripts/`.

## Verification

Compile the active Python code:

```bash
python -m py_compile $(find src scripts tests -name "*.py")
```

Run the S3R smoke commands above when ACDC data is available. If data is missing, the scripts should fail clearly instead of faking success.
