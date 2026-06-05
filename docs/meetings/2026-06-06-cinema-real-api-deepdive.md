# AgenTeam Deepdive: CineMA Real API Binding

Date: 2026-06-06

## External Signals

- CineMA exposes the public package `cinema` and the segmentation class `ConvUNetR`.
- Official SAX segmentation examples call `ConvUNetR.from_finetuned(...)` and then run `model({"sax": tensor})["sax"]`.
- ACDC SAX segmentation weights are published as `finetuned/segmentation/acdc_sax/acdc_sax_<seed>.safetensors` with `finetuned/segmentation/acdc_sax/config.yaml`.
- CineMA label order matches this project: `0=BG`, `1=RV`, `2=MYO`, `3=LV`.

## Internal Decisions

- Bind CineMA through `ConvUNetR` and keep Medical-SAM3 optional.
- Add a local `.safetensors + config.yaml` loader before `from_finetuned` to avoid re-downloading when users used `snapshot_download --local_dir`.
- Support CineMA-only training by skipping Medical-SAM3 when `--disable_field_kd --disable_fused_kd` are enabled.
- Treat `25d` S3R input channels as a local SAX depth window for CineMA.
- Print KD lambdas, enabled/disabled components, teacher source, checkpoint/config, and epoch KD components.

## Verification

- `py_compile` passed for modified CineMA/training/scripts.
- `scripts/test_teacher_loading.py --teacher cinema --teacher_stub --input_mode 25d` passed.
- `train_s3r_acdc.py --use_dual_teacher_kd --teacher_stub --disable_field_kd --disable_fused_kd --input_mode 25d` passed one CPU smoke epoch.

## Remaining Risk

- Real CineMA execution was not verified in this local checkout because `external/CineMA` and real checkpoints are not present here.
- The local loader depends on upstream `cinema.segmentation.convunetr.get_model`; if upstream changes, fallback attempts `ConvUNetR.from_finetuned`.
