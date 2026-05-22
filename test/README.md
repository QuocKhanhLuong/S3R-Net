# SSRBlockV3 ACDC Debug Experiment

This folder contains isolated Selective Spectral Retention experiments. It does
not modify the main SpecMamba training pipeline, models, or paper-facing code.

The experiment uses the existing preprocessed ACDC data under:

```bash
preprocessed_data/ACDC
```

Supported layouts:

```bash
preprocessed_data/ACDC/volumes/*.npy
preprocessed_data/ACDC/masks/*.npy
```

or:

```bash
preprocessed_data/ACDC/training/volumes/*.npy
preprocessed_data/ACDC/training/masks/*.npy
preprocessed_data/ACDC/testing/volumes/*.npy
preprocessed_data/ACDC/testing/masks/*.npy
```

When `training/` and `testing/` are present, train/val splitting is made from
`training/` only. To point directly at a split directory, pass:

```bash
python test/train_ssr_acdc.py --config test/configs/ssr_v3_acdc.yaml --data_root preprocessed_data/ACDC/training
```

Do not download Kaggle data for this debug harness.

## Train

Run from the repository root:

```bash
python test/train_ssr_acdc.py --config test/configs/ssr_v3_acdc.yaml
```

Quick smoke run:

```bash
python test/train_ssr_acdc.py --config test/configs/ssr_v3_acdc.yaml --epochs 2 --max_slices 32
```

The script supports CLI overrides for `--epochs`, `--max_slices`, `--run_name`,
`--device`, `--data_root`, `--output_root`, `--batch_size`, and `--num_workers`.

## Analyze

```bash
python test/analyze_ssr_logs.py --run_dir test/outputs/ssr_v3_acdc_debug
```

## Logs To Inspect

- `training_log.csv`: loss, validation foreground Dice, and per-class Dice.
- `ssr_logs.csv`: per-block spectral gate and contribution diagnostics.
- update gate collapse: band 0 taking most of the update budget.
- suppress saturation: high-band suppress gates pinned near their configured max.
- high-frequency ratio: collapse below 0.75 or amplification risk above 1.7.
- boundary to non-boundary high ratio: should improve if boundary-relevant high
  frequencies are retained better.
- validation foreground Dice: sanity metric only; do not treat this prototype as
  a final performance claim.

Generated artifacts are written to `test/outputs/<run_name>/`.
