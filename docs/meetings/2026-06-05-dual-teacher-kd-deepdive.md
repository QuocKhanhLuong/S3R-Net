# AgenTeam Deepdive: Agreement-Aware Dual-Teacher KD

Date: 2026-06-05

## Health

ON TRACK

## External Signals

- Medical-SAM3 is available through AIM-Research-Lab/Medical-SAM3 and ChongCong/Medical-SAM3, but the integration risk is high because checkpoints are large, API details are not pinned, and dependencies may require a newer SAM3 stack.
- CineMA is a better scoped cardiac teacher candidate. Hugging Face provides ACDC SAX segmentation weights under task-specific fine-tuned paths, including `.safetensors` checkpoints.
- Both teachers should stay outside core dependencies. The first implementation should be cache-first and support deterministic stubs.

## Internal Health

- The active repo is S3R-only after cleanup. The correct train entry point is `src/training/train_s3r_acdc.py`.
- Existing S3R-SCSD distillation modules already support cache-style teacher targets, but agreement weighting needed explicit behavior outside region routing.
- Teacher cache alignment is the main risk: `case_id`, `slice_idx`, image size, class order, checkpoint identity, and split provenance must match.

## Decisions

- Implement frozen teacher wrappers in `src/teachers/` with clear load errors and `--teacher_stub` mode.
- Add agreement-aware KD in `src/losses/agreement_kd.py`.
- Add optional `--use_dual_teacher_kd` to S3R training without changing baseline supervised behavior.
- Use `.pt` per-sample cache for Medical-SAM3/CineMA outputs.
- Keep real upstream forward binding as a future pinned-adapter task.

## Action Items

- Pin Medical-SAM3 and CineMA commits before adding real forward calls.
- Record teacher checkpoint hashes and class order in cache metadata for real experiments.
- Run ablations: baseline, vanilla Medical-SAM3 KD, field KD, CineMA boundary KD, no-agreement dual KD, full agreement-aware fusion, and full fusion plus spectral KD.
