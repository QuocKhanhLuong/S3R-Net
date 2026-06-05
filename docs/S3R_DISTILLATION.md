# S3R-SCSD Distillation

S3R-SCSD means Semantic-Characteristic Spectral-State Distillation.

It is a phased dual-teacher framework:

- Teacher 1: semantic / Dice teacher
- Teacher 2: characteristic / boundary / spectral teacher
- Student: S3R-Net

The first implementation is cache-first. Real teacher wrappers can be plugged in later through `src/distillation/teacher_interfaces.py`.

## Teacher Targets

Teacher 1 provides:

- soft 4-class probabilities
- semantic entropy
- foreground probability

Teacher 2 provides:

- boundary map
- distance map
- foreground probability
- spectral boundary distribution

Cache files are per sample:

```text
teacher_cache/<dataset>/<case_id>_<slice_idx>.npz
```

Required fields:

- `t1_probs`
- `t1_entropy`
- `t1_foreground`
- `t2_boundary`
- `t2_distance`
- `t2_foreground`
- `agreement_weight`

## Region Routing

Region-aware routing computes:

- `W_sem`: high in stable interior regions
- `W_char`: high around boundaries and transition regions
- `W_uncertain`: high where teachers disagree or semantic entropy is high

During supervised training, GT masks may be used to define training-only boundary routing.

## Phases

`phase1_semantic`:

- supervised S3R loss
- Teacher 1 semantic KL

`phase2_characteristic`:

- supervised S3R loss
- Teacher 2 boundary, distance, and spectral boundary KD

`phase3_dual_routing`:

- supervised S3R loss
- Teacher 1 semantic KD
- Teacher 2 characteristic KD
- region-aware routing
- teacher agreement weighting

`phase4_state_kd`:

- all phase 3 losses
- teacher-derived spectral-state KD

## Smoke Test Cache

Generate a procedural GT teacher cache:

```bash
python src/distillation/generate_gt_teacher_cache.py \
  --data_root preprocessed_data/ACDC \
  --output_dir teacher_cache/acdc_gt_debug \
  --image_size 224 \
  --max_slices 32
```

Run phase 3 smoke training:

```bash
python src/distillation/train_s3r_distill.py \
  --config src/distillation/configs/s3r_scsd_phase3_dual_routing.yaml \
  --teacher_cache_dir teacher_cache/acdc_gt_debug \
  --epochs 2 \
  --batch_size 8 \
  --max_slices 32 \
  --run_name smoke_s3r_scsd_phase3
```

Optional reporting flags:

```bash
--no_tqdm
--wandb --wandb_project s3r-scsd --wandb_run_name my_distill_run
```

## Real Teacher Integration

Next steps for real teachers:

- implement a `SemanticTeacher` wrapper for nnU-Net, AsymSpecMambaDCN, or an ensemble
- implement a `CharacteristicTeacher` wrapper for boundary/SDF/spectral outputs
- include checkpoint hashes and split/fold metadata in generated cache metadata
- validate class order, image size, and split provenance before training
- compare against no-KD, single-teacher KD, naive dual-teacher KD, and full S3R-SCSD
