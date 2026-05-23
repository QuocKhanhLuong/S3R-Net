# CHANGELOG.md

## 2026-05-23

### Added

- Added an isolated SSR Phase 2 `geometry_refine: dcnv4` path in
  `test/ssr_blocks.py`. It requires a compatible external DCNv4 extension and
  fails clearly when the operator is not installed.
- Added the `ssr_se_deformable` Phase 2 variant for the existing torchvision
  `deform_conv2d` refinement path.

### Changed

- Updated the Phase 2 `ssr_se_dcn` variant to request `geometry_refine: dcnv4`
  instead of the torchvision deformable-convolution path.
- Clarified that `geometry_refine: deformable` uses
  `torchvision.ops.deform_conv2d` and is not a verified DCNv4 operator.

### Validation

- Passed compile checks for the SSR phase-2 files with base Python and
  `conda run -n alvin`.
- Confirmed local environment does not have a DCNv4 extension; `ssr_se_dcn`
  now fails with a clear install/error message instead of silently using
  torchvision deformable convolution.
- Passed a small CPU smoke train for the legacy `ssr_se_deformable` variant:
  `conda run -n alvin python -B test/train_ssr_acdc.py --config test/configs/ssr_phase2_acdc_224.yaml --variant ssr_se_deformable --epochs 1 --max_slices 4 --batch_size 2 --image_size 64 --device cpu --num_workers 0 --output_root /private/tmp/specumamba_ssr_dcn_smoke --run_name deformable_smoke`
- Still pending: run the `ssr_se_dcn` variant on a machine with a compatible
  DCNv4 extension installed.

## 2026-05-11

### Added

- Added `scripts/acdc_split.py` for patient-level ACDC split creation and
  validation.
- Added `splits/acdc_patient_split_seed42.json` with a saved 80/20 patient
  split.
- Added `configs/acdc_asym_v31.yaml` and aligned `configs/acdc_reproducible.yaml`
  for config-driven AsymSpecMambaDCN v3.1 training.
- Added `scripts/train_acdc_reproducible.sh` wrapper for config-driven training.
- Added split/reproducibility tests in `tests/test_acdc_split.py` and
  `tests/test_acdc_training_reproducibility.py`.
- Added `KNOWLEDGE.md` with research critique, external evidence, implementation
  notes, and remaining paper-readiness work.

### Changed

- Updated `src/training/train_acdc.py` to support YAML/JSON configs, seeded
  training setup, patient-level split manifests, seeded DataLoader workers, and
  resolved run-argument export.
- Moved `src/losses/physics_loss.py` demo code behind a `__main__` guard so
  importing training modules no longer prints random demo loss values.
- Updated `report.md` to use the AsymSpecMambaDCN v3.1 paper narrative instead
  of the legacy 3-stream Spec-HRNet narrative.
- Updated `ARCHITECTURE.md` to qualify Mamba/DCNv3/DCNv4/HDC claims, remove
  unsupported SOTA-style language, and point to config-driven training.
- Softened source docstrings in `src/models/specmamba_net.py` so comments match
  the conservative paper terminology.

### Validation

- Generated patient-level split manifest:
  `python scripts/acdc_split.py --data_dir preprocessed_data/ACDC --output splits/acdc_patient_split_seed42.json --seed 42 --train_ratio 0.8`
- Passed focused tests:
  `python -m pytest tests/test_acdc_training_reproducibility.py tests/test_acdc_split.py -q`
- Passed compile check:
  `python -m py_compile src/training/train_acdc.py scripts/acdc_split.py tests/test_acdc_split.py tests/test_acdc_training_reproducibility.py`
- Passed wrapper shell syntax:
  `bash -n scripts/train_acdc_reproducible.sh`
- Confirmed `conda run -n alvin python src/training/train_acdc.py --help`
  reaches argparse successfully. It still reports missing optional
  `albumentations` in that environment.
- Confirmed split manifest has 80 train patients, 20 validation patients, and
  zero patient overlap.

### Not Done

- Full training was not started.
- Baselines and ablations remain to be implemented and rerun.
