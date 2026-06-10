#!/usr/bin/env python3
"""Summarize S3R block-audit runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize S3R block audit variants.")
    parser.add_argument("--root", default="weights/block_audit")
    parser.add_argument("--output", default="weights/block_audit/block_audit_summary.md")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for summary_path in sorted(root.glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = read_json(summary_path)
        rows = read_csv(run_dir / "training_log.csv")
        block_rows = read_csv(run_dir / "s3r_block_logs.csv")
        variant = str(summary.get("config_flags", {}).get("block_variant") or summary.get("variant") or run_dir.name)
        runs.append(
            {
                "variant": variant,
                "run_dir": run_dir,
                "summary": summary,
                "training_rows": rows,
                "block_rows": block_rows,
            }
        )
    return runs


def metric(run: dict[str, Any], key: str) -> float:
    return as_float(run["summary"].get(key))


def stability_score(run: dict[str, Any]) -> float:
    rows = run["training_rows"]
    if not rows:
        return math.inf
    vals = [as_float(row.get("val_fg_mean")) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) < 2:
        return math.inf
    tail = vals[-min(10, len(vals)) :]
    mean = sum(tail) / len(tail)
    return sum((v - mean) ** 2 for v in tail) / len(tail)


def block_metric_mean(run: dict[str, Any], metric_name: str) -> float:
    values = [as_float(row.get("value")) for row in run["block_rows"] if row.get("metric") == metric_name and row.get("split") == "val"]
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else math.nan


def best_by(runs: list[dict[str, Any]], key: str, lower: bool = False) -> dict[str, Any] | None:
    valid = [run for run in runs if math.isfinite(metric(run, key))]
    if not valid:
        return None
    return min(valid, key=lambda run: metric(run, key)) if lower else max(valid, key=lambda run: metric(run, key))


def fmt(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.4f}"


def write_report(runs: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    best_dsc = best_by(runs, "best_val_fg_mean")
    best_hd95 = best_by(runs, "best_val_hd95_best", lower=True)
    best_assd = best_by(runs, "best_val_assd_fg_mean", lower=True)
    stable = min(runs, key=stability_score) if runs else None
    by_variant = {run["variant"]: run for run in runs}
    full = by_variant.get("s3r_full")
    gamma0 = by_variant.get("s3r_gamma0")
    no_suppress = by_variant.get("s3r_no_suppress")
    fixed_band = by_variant.get("s3r_fixed_band")
    simple = by_variant.get("s3r_simple_spectral")

    lines = [
        "# S3R Block Audit Summary",
        "",
        "## Variant Table",
        "",
        "| Variant | Best DSC | Best HD95 | Best ASSD | Final DSC | Stability | Mean HF Penalty | Mean Residual Ratio | Mean FFT Error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in sorted(runs, key=lambda r: r["variant"]):
        lines.append(
            "| {variant} | {best_dsc} | {hd95} | {assd} | {final_dsc} | {stable} | {hf} | {residual} | {fft} |".format(
                variant=run["variant"],
                best_dsc=fmt(metric(run, "best_val_fg_mean")),
                hd95=fmt(metric(run, "best_val_hd95_best")),
                assd=fmt(metric(run, "best_val_assd_fg_mean")),
                final_dsc=fmt(metric(run, "final_val_fg_mean")),
                stable=fmt(stability_score(run)),
                hf=fmt(block_metric_mean(run, "high_freq_penalty")),
                residual=fmt(block_metric_mean(run, "residual_ratio")),
                fft=fmt(block_metric_mean(run, "fft_reconstruction_error")),
            )
        )

    def name(run: dict[str, Any] | None) -> str:
        return run["variant"] if run is not None else "n/a"

    gamma_close = "n/a"
    if full and gamma0:
        diff = abs(metric(full, "best_val_fg_mean") - metric(gamma0, "best_val_fg_mean"))
        gamma_close = f"yes, diff={diff:.4f}" if diff < 0.01 else f"no, diff={diff:.4f}"

    suppress_help = "n/a"
    if full and no_suppress:
        diff = metric(no_suppress, "best_val_fg_mean") - metric(full, "best_val_fg_mean")
        suppress_help = f"no_suppress better by {diff:.4f}" if diff > 0 else f"full better by {-diff:.4f}"

    fixed_help = "n/a"
    if full and fixed_band:
        diff = metric(fixed_band, "best_val_fg_mean") - metric(full, "best_val_fg_mean")
        fixed_help = f"fixed_band better by {diff:.4f}" if diff > 0 else f"full learned gates better by {-diff:.4f}"

    simple_help = "n/a"
    if full and simple:
        diff = metric(simple, "best_val_fg_mean") - metric(full, "best_val_fg_mean")
        simple_help = f"simple better by {diff:.4f}" if diff > 0 else f"full better by {-diff:.4f}"

    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. Best DSC: {name(best_dsc)}",
            f"2. Best HD95: {name(best_hd95)}; best ASSD: {name(best_assd)}",
            f"3. Most stable curve: {name(stable)}",
            f"4. Gamma=0 close to full S3R: {gamma_close}",
            f"5. Suppress helps or hurts: {suppress_help}",
            f"6. Learned gates vs fixed band: {fixed_help}",
            "7. hf_ratio_penalty active: inspect `Mean HF Penalty`; zero means below threshold, not proof of no spectral learning.",
            "8. FFT reconstruction correct: inspect `Mean FFT Error`; expected near 0.",
            f"9. Simple spectral vs full: {simple_help}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = load_runs(Path(args.root))
    if not runs:
        raise SystemExit(f"No runs found under {args.root}")
    write_report(runs, Path(args.output))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
