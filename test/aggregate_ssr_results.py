#!/usr/bin/env python3
"""Aggregate completed SSR validation runs into CSV, JSON, and markdown."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SSR validation outputs")
    parser.add_argument("--output_root", default="test/outputs")
    parser.add_argument("--pattern", default="ssr_full_*")
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir) if args.out_dir else output_root / "ssr_full_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [
        path for path in sorted(output_root.iterdir())
        if path.is_dir() and fnmatch.fnmatch(path.name, args.pattern) and (path / "summary.json").exists()
    ]
    rows = [summarize_run(path) for path in run_dirs]
    rows = [row for row in rows if row]
    rows.sort(key=lambda row: finite_or_neg(row.get("best_val_fg_mean")), reverse=True)

    write_csv(out_dir / "aggregate_results.csv", rows)
    summary = build_group_summary(rows)
    with open(out_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    write_markdown(out_dir / "aggregate_markdown_report.md", rows, summary)
    print(f"Aggregated {len(rows)} runs into {out_dir}")


def summarize_run(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "summary.json")
    training_rows = read_csv(run_dir / "training_log.csv")
    ssr_rows = read_csv(run_dir / "ssr_logs.csv")
    robustness = read_json(run_dir / "robustness_summary.json") if (run_dir / "robustness_summary.json").exists() else {}

    best_epoch = int(summary.get("best_epoch") or 0)
    best_train_row = find_epoch_row(training_rows, best_epoch) or {}
    final_row = training_rows[-1] if training_rows else {}
    cfg_flags = summary.get("config_flags", {})

    row: dict[str, Any] = {
        "run_name": run_dir.name,
        "variant": summary.get("variant"),
        "input_mode": summary.get("input_mode") or infer_input_mode(run_dir.name),
        "seed": infer_seed(summary, run_dir.name),
        "best_epoch": best_epoch,
        "best_val_fg_mean": summary.get("best_val_fg_mean"),
        "best_val_dice_rv": summary.get("best_val_dice_RV"),
        "best_val_dice_myo": summary.get("best_val_dice_MYO"),
        "best_val_dice_lv": summary.get("best_val_dice_LV"),
        "final_val_fg_mean": coalesce(summary.get("final_val_fg_mean"), maybe_float(final_row.get("val_fg_mean"))),
        "best_val_hd95_fg_mean": coalesce(summary.get("best_val_hd95_fg_mean"), maybe_float(best_train_row.get("val_hd95_fg_mean"))),
        "best_val_assd_fg_mean": coalesce(summary.get("best_val_assd_fg_mean"), maybe_float(best_train_row.get("val_assd_fg_mean"))),
        "best_val_boundary_f1_fg": coalesce(summary.get("best_val_boundary_f1_fg"), maybe_float(best_train_row.get("val_boundary_f1_fg"))),
        "best_val_surface_dice_fg": coalesce(summary.get("best_val_surface_dice_fg"), maybe_float(best_train_row.get("val_surface_dice_fg"))),
        "actual_batch_size": summary.get("actual_batch_size"),
        "image_size": summary.get("image_size"),
        "epochs": summary.get("epochs"),
        "model_parameter_count": summary.get("model_parameter_count"),
        "residual_gate_type": (cfg_flags.get("residual_gate_type") if isinstance(cfg_flags, dict) else None),
        "geometry_refine": (cfg_flags.get("geometry_refine") if isinstance(cfg_flags, dict) else None),
    }
    row.update(summarize_ssr(ssr_rows))
    if robustness:
        clean = robustness.get("clean") or {}
        worst = robustness.get("worst_by_fg_mean") or {}
        row["robust_clean_fg_mean"] = clean.get("fg_mean")
        row["robust_worst_fg_mean"] = worst.get("fg_mean")
        row["robust_worst_perturbation"] = worst.get("perturbation")
    return row


def summarize_ssr(rows: list[dict[str, str]]) -> dict[str, float | None]:
    high_freq = metric_values(rows, "high_freq_ratio", split="val")
    boundary_ratio = metric_values(rows, "boundary_to_nonboundary_high_ratio", split="val")
    gamma = metric_values(rows, "gamma", split="val")
    residual = metric_values(rows, "residual_gate_mean", split="val")
    return {
        "high_freq_ratio_mean": nanmean(high_freq),
        "high_freq_ratio_max": nanmax(high_freq),
        "boundary_ratio_mean": nanmean(boundary_ratio),
        "gamma_mean": nanmean(gamma),
        "residual_gate_mean": nanmean(residual),
    }


def metric_values(rows: list[dict[str, str]], metric: str, split: str | None = None) -> list[float]:
    values = []
    for row in rows:
        if row.get("metric") != metric:
            continue
        if split is not None and row.get("split") != split:
            continue
        value = maybe_float(row.get("value"))
        if value is not None:
            values.append(value)
    return values


def build_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        "ssr_full_2d": [row for row in rows if str(row.get("input_mode")) == "2d" and row.get("variant") == "ssr_full"],
        "ssr_full_25d": [row for row in rows if str(row.get("input_mode")) == "25d" and row.get("variant") == "ssr_full"],
    }
    metrics = [
        "best_val_fg_mean",
        "best_val_dice_rv",
        "best_val_dice_myo",
        "best_val_dice_lv",
        "best_val_hd95_fg_mean",
        "best_val_assd_fg_mean",
        "best_val_boundary_f1_fg",
        "high_freq_ratio_mean",
        "high_freq_ratio_max",
        "boundary_ratio_mean",
        "gamma_mean",
        "residual_gate_mean",
        "robust_clean_fg_mean",
        "robust_worst_fg_mean",
    ]
    out: dict[str, Any] = {"runs": len(rows), "groups": {}}
    for group_name, group_rows in groups.items():
        group_summary: dict[str, Any] = {"count": len(group_rows)}
        for metric in metrics:
            vals = [maybe_float(row.get(metric)) for row in group_rows]
            vals = [v for v in vals if v is not None]
            group_summary[metric] = mean_std(vals)
        out["groups"][group_name] = group_summary
    out["overall_ranking"] = [
        {
            "rank": idx + 1,
            "run_name": row.get("run_name"),
            "input_mode": row.get("input_mode"),
            "seed": row.get("seed"),
            "best_val_fg_mean": row.get("best_val_fg_mean"),
            "best_val_hd95_fg_mean": row.get("best_val_hd95_fg_mean"),
            "best_val_boundary_f1_fg": row.get("best_val_boundary_f1_fg"),
        }
        for idx, row in enumerate(rows)
    ]
    return out


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# SSR Full Aggregate Report")
    lines.append("")
    lines.append("This report aggregates debug-suite runs. Interpret it as experimental evidence, not a final paper-level conclusion.")
    lines.append("")
    lines.append("## Overall Ranking")
    lines.append("")
    lines.append("| Rank | Run | Mode | Seed | FG Dice | HD95 FG | Boundary F1 |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for idx, row in enumerate(rows, 1):
        lines.append(
            f"| {idx} | {row.get('run_name')} | {row.get('input_mode')} | {row.get('seed')} | "
            f"{fmt(row.get('best_val_fg_mean'))} | {fmt(row.get('best_val_hd95_fg_mean'))} | "
            f"{fmt(row.get('best_val_boundary_f1_fg'))} |"
        )
    lines.append("")
    lines.append("## Multi-Seed Results")
    lines.append("")
    for group_name in ("ssr_full_2d", "ssr_full_25d"):
        group = summary.get("groups", {}).get(group_name, {})
        lines.append(f"### {group_name}")
        lines.append("")
        lines.append(f"- runs: {group.get('count', 0)}")
        lines.append(f"- best_val_fg_mean: {fmt_mean_std(group.get('best_val_fg_mean'))}")
        lines.append(f"- RV/MYO/LV Dice: {fmt_mean_std(group.get('best_val_dice_rv'))} / {fmt_mean_std(group.get('best_val_dice_myo'))} / {fmt_mean_std(group.get('best_val_dice_lv'))}")
        lines.append(f"- HD95 FG: {fmt_mean_std(group.get('best_val_hd95_fg_mean'))}")
        lines.append(f"- ASSD FG: {fmt_mean_std(group.get('best_val_assd_fg_mean'))}")
        lines.append(f"- Boundary F1 FG: {fmt_mean_std(group.get('best_val_boundary_f1_fg'))}")
        lines.append(f"- high_freq_ratio mean/max: {fmt_mean_std(group.get('high_freq_ratio_mean'))} / {fmt_mean_std(group.get('high_freq_ratio_max'))}")
        lines.append(f"- boundary_ratio_mean: {fmt_mean_std(group.get('boundary_ratio_mean'))}")
        lines.append("")
    lines.append("## 2D vs 2.5D Comparison")
    lines.append("")
    lines.extend(compare_groups(summary))
    lines.append("")
    lines.append("## Robustness Summary")
    lines.append("")
    lines.extend(robustness_lines(rows))
    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    lines.extend(diagnosis_lines(summary))
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def compare_groups(summary: dict[str, Any]) -> list[str]:
    g2d = summary.get("groups", {}).get("ssr_full_2d", {})
    g25d = summary.get("groups", {}).get("ssr_full_25d", {})
    d2 = maybe_float((g2d.get("best_val_fg_mean") or {}).get("mean"))
    d25 = maybe_float((g25d.get("best_val_fg_mean") or {}).get("mean"))
    if d2 is None or d25 is None:
        return ["- Not enough completed 2D and 2.5D runs for a direct comparison."]
    delta = d25 - d2
    return [f"- 2.5D minus 2D mean FG Dice: {delta:+.4f}. Treat this as a stability signal until all seeds finish."]


def robustness_lines(rows: list[dict[str, Any]]) -> list[str]:
    robust_rows = [row for row in rows if maybe_float(row.get("robust_clean_fg_mean")) is not None]
    if not robust_rows:
        return ["- No robustness summaries found yet."]
    lines = ["| Run | Clean FG | Worst FG | Worst perturbation |", "|---|---:|---:|---|"]
    for row in robust_rows:
        lines.append(
            f"| {row.get('run_name')} | {fmt(row.get('robust_clean_fg_mean'))} | "
            f"{fmt(row.get('robust_worst_fg_mean'))} | {row.get('robust_worst_perturbation')} |"
        )
    return lines


def diagnosis_lines(summary: dict[str, Any]) -> list[str]:
    lines = []
    for group_name in ("ssr_full_2d", "ssr_full_25d"):
        group = summary.get("groups", {}).get(group_name, {})
        count = int(group.get("count", 0) or 0)
        if count == 0:
            lines.append(f"- {group_name}: no completed runs found.")
            continue
        hf_mean = maybe_float((group.get("high_freq_ratio_max") or {}).get("mean"))
        boundary = maybe_float((group.get("boundary_ratio_mean") or {}).get("mean"))
        bf1 = maybe_float((group.get("best_val_boundary_f1_fg") or {}).get("mean"))
        if hf_mean is not None:
            lines.append(f"- {group_name}: spectral max ratio mean is {hf_mean:.3f}; values near or below 4 are more stable for this debug protocol.")
        if boundary is not None:
            lines.append(f"- {group_name}: boundary/non-boundary high-frequency ratio mean is {boundary:.3f}; values above 1.5 support boundary-focused retention.")
        if bf1 is not None:
            lines.append(f"- {group_name}: boundary F1 mean is {bf1:.3f}; compare it with Dice before making geometry claims.")
    return lines


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_epoch_row(rows: list[dict[str, str]], epoch: int) -> dict[str, str] | None:
    for row in rows:
        if int(float(row.get("epoch", 0) or 0)) == epoch:
            return row
    return None


def infer_seed(summary: dict[str, Any], run_name: str) -> int | None:
    seed = summary.get("seed")
    if seed is not None:
        return int(seed)
    if "seed" in run_name:
        try:
            return int(run_name.rsplit("seed", 1)[-1])
        except ValueError:
            return None
    return None


def infer_input_mode(run_name: str) -> str | None:
    if "_25d_" in run_name:
        return "25d"
    if "_2d_" in run_name:
        return "2d"
    return None


def maybe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def finite_or_neg(value: Any) -> float:
    out = maybe_float(value)
    return out if out is not None else -float("inf")


def nanmean(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.isnan(arr).all():
        return None
    return float(np.nanmean(arr))


def nanmax(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.isnan(arr).all():
        return None
    return float(np.nanmax(arr))


def mean_std(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def fmt(value: Any) -> str:
    out = maybe_float(value)
    return "NA" if out is None else f"{out:.4f}"


def fmt_mean_std(value: Any) -> str:
    if not isinstance(value, dict) or value.get("mean") is None:
        return "NA"
    return f"{float(value['mean']):.4f} +/- {float(value.get('std') or 0.0):.4f} (n={value.get('n', 0)})"


if __name__ == "__main__":
    main()
