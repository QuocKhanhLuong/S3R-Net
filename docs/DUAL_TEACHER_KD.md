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

For a lighter one-teacher setup:

```bash
bash scripts/setup_teachers.sh cinema
```

Install HF helpers if needed:

```bash
pip install -U huggingface_hub safetensors
```

Download teacher weights:

```bash
python scripts/download_teachers.py \
  --teacher both \
  --medical_sam3_repo ChongCong/Medical-SAM3 \
  --cinema_repo mathpluscode/CineMA \
  --output_dir checkpoints/teachers
```

Medical-SAM3 weights are large. To download only CineMA:

```bash
python scripts/download_teachers.py \
  --teacher cinema \
  --cinema_repo mathpluscode/CineMA \
  --output_dir checkpoints/teachers
```

By default this downloads only the ACDC SAX segmentation teacher files:

```text
finetuned/segmentation/acdc_sax/config.yaml
finetuned/segmentation/acdc_sax/acdc_sax_0.safetensors
```

If you accidentally downloaded the full CineMA repository, prune unused weight
files with a dry-run first:

```bash
python scripts/prune_cinema_weights.py \
  --cinema_dir checkpoints/teachers/cinema
```

Then delete the unused checkpoint files:

```bash
python scripts/prune_cinema_weights.py \
  --cinema_dir checkpoints/teachers/cinema \
  --execute
```

Keep all three ACDC SAX seeds only if you plan an ensemble:

```bash
python scripts/prune_cinema_weights.py \
  --cinema_dir checkpoints/teachers/cinema \
  --keep_all_acdc_seeds \
  --execute
```

Medical-SAM3 currently publishes a single large `checkpoint.pt` on Hugging Face.
To download only that latest main-branch weight:

```bash
python scripts/download_teachers.py \
  --teacher medical_sam3 \
  --medical_sam3_repo ChongCong/Medical-SAM3 \
  --output_dir checkpoints/teachers
```

This uses `allow_patterns=["checkpoint.pt"]` by default for Medical-SAM3.
For reproducible experiments, pin a Hugging Face revision:

```bash
python scripts/download_teachers.py \
  --teacher medical_sam3 \
  --medical_sam3_repo ChongCong/Medical-SAM3 \
  --revision 116930dd8feae51790703337c4090691f9c4aa05 \
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

Agreement-gated fused KD smoke:

```bash
python src/training/train_s3r_acdc.py \
  --config configs/s3r_dual_teacher_agreement_gated_smoke.yaml
```

The new fused-KD gate is opt-in. Default behavior remains the unweighted fused
KD loss. Enable it with:

```yaml
dual_teacher_kd:
  fused_kd_weight_mode: agreement
  fused_kd_min_weight: 0.10
  fused_kd_agreement_power: 1.0
```

This downweights fused KD where Medical-SAM3 and CineMA disagree while keeping
field KD, CineMA boundary KD, and spectral KD unchanged.

## CineMA-Only Real Teacher Test

After cloning CineMA and downloading weights, test the real CineMA wrapper without Medical-SAM3:

```bash
python scripts/test_teacher_loading.py \
  --teacher cinema \
  --data_dir preprocessed_data/ACDC \
  --cinema_repo_path external/CineMA \
  --cinema_ckpt_dir checkpoints/teachers/cinema \
  --cinema_ckpt checkpoints/teachers/cinema/finetuned/segmentation/acdc_sax/acdc_sax_0.safetensors \
  --cinema_config checkpoints/teachers/cinema/finetuned/segmentation/acdc_sax/config.yaml \
  --input_mode 25d \
  --num_classes 4 \
  --device cuda
```

Train with CineMA boundary KD only:

```bash
python src/training/train_s3r_acdc.py \
  --model s3r_mini \
  --data_dir preprocessed_data/ACDC \
  --image_size 224 \
  --epochs 1 \
  --batch_size 2 \
  --max_slices 8 \
  --input_mode 25d \
  --in_channels 5 \
  --use_dual_teacher_kd \
  --disable_field_kd \
  --disable_fused_kd \
  --cinema_repo_path external/CineMA \
  --cinema_ckpt_dir checkpoints/teachers/cinema \
  --cinema_ckpt checkpoints/teachers/cinema/finetuned/segmentation/acdc_sax/acdc_sax_0.safetensors \
  --cinema_config checkpoints/teachers/cinema/finetuned/segmentation/acdc_sax/config.yaml \
  --lambda_cine_boundary 0.5 \
  --lambda_spec 0.05 \
  --device cuda \
  --save_dir weights/smoke_cinema_real_kd
```

The trainer prints the effective KD lambdas, fused-KD weight mode, per-epoch KD
components, `teacher_disagreement_mean`, `fuse_weight_mean`, and fuse-weight
range; it saves `kd_loss_components.png` plus `kd_agreement_weights.png`.

## Ablations

Flags:

```text
--disable_field_kd
--disable_cine_boundary_kd
--disable_fused_kd
--disable_spectral_kd
--disable_agreement_weighting
--use_vanilla_kd_only
--fused_kd_weight_mode {none,agreement}
--fused_kd_min_weight 0.10
--fused_kd_agreement_power 1.0
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
