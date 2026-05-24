#!/usr/bin/env python3
"""Launch SSR full validation-suite commands from one entry point."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from exc


SEEDS = [42, 123, 2025]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SSR full ACDC validation suite")
    parser.add_argument("--config", default="test/configs/ssr_full_acdc_224.yaml")
    parser.add_argument(
        "--mode",
        default="train_all",
        choices=["train_all", "train_2d", "train_25d", "robustness", "aggregate"],
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--max_slices", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include_baselines", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_yaml(Path(args.config))
    output_root = Path(args.output_root or cfg.get("output_root", "test/outputs"))

    if args.mode in {"train_all", "train_2d", "train_25d"}:
        for spec in training_specs(args.mode, include_baselines=args.include_baselines):
            run_training(args, output_root, spec)
    elif args.mode == "robustness":
        run_robustness(args, output_root)
    elif args.mode == "aggregate":
        run_aggregate(args, output_root)


def training_specs(mode: str, include_baselines: bool = False) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if mode in {"train_all", "train_2d"}:
        specs.extend(
            {
                "variant": "ssr_full",
                "input_mode": "2d",
                "in_channels": 1,
                "seed": seed,
                "run_name": f"ssr_full_2d_seed{seed}",
            }
            for seed in SEEDS
        )
    if mode in {"train_all", "train_25d"}:
        specs.extend(
            {
                "variant": "ssr_full",
                "input_mode": "25d",
                "in_channels": 5,
                "seed": seed,
                "run_name": f"ssr_full_25d_seed{seed}",
            }
            for seed in SEEDS
        )
    if include_baselines and mode in {"train_all", "train_2d"}:
        specs.extend(
            [
                {
                    "variant": "baseline_ssr",
                    "input_mode": "2d",
                    "in_channels": 1,
                    "seed": 42,
                    "run_name": "baseline_ssr_2d_seed42",
                },
                {
                    "variant": "ssr_se_lk",
                    "input_mode": "2d",
                    "in_channels": 1,
                    "seed": 42,
                    "run_name": "ssr_se_lk_2d_seed42",
                },
            ]
        )
    return specs


def run_training(args: argparse.Namespace, output_root: Path, spec: dict[str, Any]) -> None:
    run_dir = output_root / spec["run_name"]
    if not args.force and (run_dir / "summary.json").exists() and (run_dir / "best_model.pt").exists():
        print(f"Skipping completed training run: {run_dir}")
        return

    cmd = [
        args.python,
        "test/train_ssr_acdc.py",
        "--config",
        args.config,
        "--variant",
        spec["variant"],
        "--input_mode",
        spec["input_mode"],
        "--in_channels",
        str(spec["in_channels"]),
        "--seed",
        str(spec["seed"]),
        "--run_name",
        spec["run_name"],
        "--output_root",
        str(output_root),
    ]
    append_override(cmd, "--epochs", args.epochs)
    append_override(cmd, "--batch_size", args.batch_size)
    append_override(cmd, "--image_size", args.image_size)
    append_override(cmd, "--max_slices", args.max_slices)
    append_override(cmd, "--device", args.device)
    append_override(cmd, "--num_workers", args.num_workers)
    run_command(cmd)


def run_robustness(args: argparse.Namespace, output_root: Path) -> None:
    expected = [
        *(output_root / f"ssr_full_2d_seed{seed}" for seed in SEEDS),
        *(output_root / f"ssr_full_25d_seed{seed}" for seed in SEEDS),
    ]
    for run_dir in expected:
        checkpoint = run_dir / "best_model.pt"
        config = run_dir / "config_resolved.yaml"
        if not checkpoint.exists() or not config.exists():
            print(f"Skipping robustness; missing checkpoint/config: {run_dir}")
            continue
        if not args.force and (run_dir / "robustness_metrics.csv").exists():
            print(f"Skipping completed robustness eval: {run_dir}")
            continue
        cmd = [
            args.python,
            "test/evaluate_robustness.py",
            "--run_dir",
            str(run_dir),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
        ]
        run_command(cmd)


def run_aggregate(args: argparse.Namespace, output_root: Path) -> None:
    cmd = [
        args.python,
        "test/aggregate_ssr_results.py",
        "--output_root",
        str(output_root),
        "--pattern",
        "ssr_full_*",
    ]
    run_command(cmd)


def append_override(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def run_command(cmd: list[str]) -> None:
    print("+ " + shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    main()
