# Agreement-Aware Dual-Teacher KD

This module adds the first S3R implementation of agreement-aware dual-teacher knowledge distillation for cardiac MRI segmentation.

## Roles

| Component | Role |
| --- | --- |
| Student | S3R / S3R-Net, trainable |
| Teacher A | Medical-SAM3, frozen prompt-driven segmentation field teacher |
| Teacher B | CineMA, frozen cardiac anatomy and boundary correction teacher |

Teachers are used during training only. Student inference uses image input and S3R only. Teacher outputs are detached and never become mandatory inference inputs.

## Method

Teacher outputs are normalized to:

```text
P_M3 [B,4,H,W]  Medical-SAM3 class probabilities
C_M3 [B,1,H,W]  Medical-SAM3 confidence
P_C  [B,4,H,W]  CineMA class probabilities
C_C  [B,1,H,W]  CineMA confidence
B_C  [B,1,H,W]  CineMA boundary prior
```

Agreement is computed with Jensen-Shannon divergence:

```text
A = exp(-JS(P_M3, P_C))
```

Medical-SAM3 is weighted more in stable interior regions. CineMA is weighted more around boundaries:

```text
W_boundary = GT boundary band during training, else CineMA boundary
W_interior = 1 - W_boundary

R_M3 = C_M3 * W_interior * (0.5 + 0.5A)
R_C  = C_C  * W_boundary * (0.5 + 0.5A)

P_F = normalize(R_M3 * P_M3 + R_C * P_C)
```

Loss:

```text
L = L_seg
  + lambda_field * KL(S3R, Medical-SAM3)
  + lambda_cine_boundary * KL(S3R, CineMA boundary regions)
  + lambda_fuse * KL(S3R, P_F)
  + lambda_spec * spectral_boundary_loss
```

Defaults:

```text
kd_temperature = 4.0
lambda_field = 0.3
lambda_cine_boundary = 0.5
lambda_fuse = 0.5
lambda_spec = 0.05
```

## Setup

Clone teacher repos:

```bash
bash scripts/setup_teachers.sh
```

Install HF helpers if needed:

```bash
pip install -U huggingface_hub safetensors
```

Download teacher weights:

```bash
python scripts/download_teachers.py \
  --medical_sam3_repo ChongCong/Medical-SAM3 \
  --cinema_repo mathpluscode/CineMA \
  --output_dir checkpoints/teachers
```

If Hugging Face access fails, login:

```bash
huggingface-cli login
```

## Stub Test

Run the full wrapper and visualization path without external weights:

```bash
python scripts/test_teacher_loading.py \
  --data_dir preprocessed_data/ACDC/training \
  --teacher_stub \
  --num_classes 4 \
  --device cuda
```

This writes:

```text
debug_outputs/teacher_kd_preview.png
```

## Precompute Cache

```bash
python scripts/precompute_teacher_outputs.py \
  --data_dir preprocessed_data/ACDC/training \
  --output_dir teacher_cache/acdc \
  --medical_sam3_ckpt_dir checkpoints/teachers/medical_sam3 \
  --cinema_ckpt_dir checkpoints/teachers/cinema \
  --medical_sam3_prompt_mode gt_box \
  --num_classes 4 \
  --device cuda
```

Cache files are `.pt` per sample:

```text
teacher_cache/acdc/<case_id>_<slice_idx>.pt
```

Each item stores `P_M3`, `C_M3`, `P_C`, `C_C`, `B_C`, and metadata.

## Train

S3R full with cached dual-teacher KD:

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

Stub training smoke:

```bash
python src/training/train_s3r_acdc.py \
  --model s3r_mini \
  --data_dir preprocessed_data/ACDC/training \
  --image_size 224 \
  --epochs 1 \
  --batch_size 2 \
  --max_slices 8 \
  --use_dual_teacher_kd \
  --teacher_stub \
  --device cpu \
  --no_tqdm \
  --save_dir weights/smoke_dual_teacher_stub
```

## Ablations

Flags:

```text
--disable_field_kd
--disable_cine_boundary_kd
--disable_fused_kd
--disable_spectral_kd
--disable_agreement_weighting
--use_vanilla_kd_only
```

Suggested ablations:

1. S3R baseline, no KD
2. S3R + vanilla Medical-SAM3 KD
3. S3R + Medical-SAM3 field KD only
4. S3R + CineMA boundary KD only
5. S3R + field KD + boundary KD without agreement
6. S3R + full agreement-aware fusion
7. S3R + full agreement-aware fusion + spectral boundary KD

## Current Limitations

- Real Medical-SAM3 and CineMA forward APIs are not hard-coded yet. The wrappers validate dependencies/checkpoints and fail with actionable errors unless `--teacher_stub` or precomputed cache is used.
- Medical-SAM3 `gt_box` prompts are training-only supervision generation. They are not used at inference.
- CineMA class ordering must be verified for each checkpoint. Use `--cinema_class_map` if upstream class order differs.
- Cache provenance should be recorded for real experiments: HF revision, checkpoint hash, image size, class order, and split metadata.
