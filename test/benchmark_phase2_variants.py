#!/usr/bin/env python3
"""Benchmark and rank SSR Phase Test 2 architecture variants.

Run this after `test/run_phase2_variants.sh` finishes. The script reads each
variant run directory, ranks by best validation foreground Dice, and writes a
CSV/JSON/Markdown report with SSR stability diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_VARIANTS = [
    "baseline_ssr",
    "ssr_se",
    "ssr_se_bounded",
    "ssr_se_lk",
    "ssr_se_dcn",
    "ssr_full",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SSR phase-2 baseline variants")
    parser.add_argument("--output_root", default="test/outputs", help="Root directory containing run outputs")
    parser.add_argument("--run_prefix", default="ssr_phase2_acdc_224", help="Run prefix used by run_phase2_variants.sh")
    parser.add_argument("--variant_script", default="test/run_phase2_variants.sh", help="Shell script to parse variant names from")
    parser.add_argument("--variants", nargs="*", default=None, help="Explicit variants to benchmark")
    parser.add_argument("--out_dir", default=None, help="Directory for benchmark report artifacts")
    parser.add_argument("--min_complete_epochs", type=int, default=1, help="Warn if a run has fewer completed epochs")
    parser.add_argument("--high_freq_risk", type=float, default=6.0, help="Warning threshold for max high_freq_ratio")
    parser.add_argument("--overfit_gap", type=float, default=0.03, help="Warning threshold for best-final Dice gap")
    parser.add_argument("--strict", action="store_true", help="Fail if any expected variant output is missing")
    return parser.parse_args()


def parse_variants_from_script(path: Path) -> list[str]:
    if not path.exists():
        return DEFAULT_VARIANTS
    text = path.read_text(encoding="utf-8")
    match = re.search(r"DEFAULT_VARIANTS=\(\s*(.*?)\s*\)", text, flags=re.S)
    if not match:
        return DEFAULT_VARIANTS
    variants = []
    for line in match.group(1).splitlines():
        clean = line.strip().strip('"').strip("'")
        if clean and not clean.startswith("#"):
            variants.append(clean)
    return variants or DEFAULT_VARIANTS


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def find_run_dir(output_root: Path, run_prefix: str, variant: str) -> Path | None:
    expected = output_root / f"{run_prefix}_{variant}"
    if expected.exists():
        return expected

    candidates = []
    for summary in output_root.glob("*/summary.json"):
        data = read_json(summary)
        if data.get("variant") == variant:
            candidates.append(summary.parent)
        elif summary.parent.name.endswith(f"_{variant}") or summary.parent.name == variant:
            candidates.append(summary.parent)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def metric_from_row(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None


def best_training_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: metric_from_row(row, "val_fg_mean", "val_fg_dice", "val_fg") or -1.0,
    )


def ssr_metric_values(rows: list[dict[str, str]], metric: str, *, split: str = "val", band: int | None = None) -> list[float]:
    values = []
    for row in rows:
        if row.get("split") != split or row.get("metric") != metric:
            continue
        if band is not None and row.get("band") not in (str(band), f"{float(band):.1f}"):
            continue
        value = as_float(row.get("value"))
        if value is not None:
            values.append(value)
    return values


def summarize_ssr(ssr_rows: list[dict[str, str]]) -> dict[str, float | None]:
    high = ssr_metric_values(ssr_rows, "high_freq_ratio")
    boundary = ssr_metric_values(ssr_rows, "boundary_to_nonboundary_high_ratio")
    gamma = ssr_metric_values(ssr_rows, "gamma")
    update_b0 = ssr_metric_values(ssr_rows, "update_gate_mean", band=0)
    suppress_b2 = ssr_metric_values(ssr_rows, "suppress_gate_mean", band=2)
    suppress_b3 = ssr_metric_values(ssr_rows, "suppress_gate_mean", band=3)
    suppress_high = suppress_b2 + suppress_b3
    hf_penalty = ssr_metric_values(ssr_rows, "high_freq_penalty")
    residual_gate = ssr_metric_values(ssr_rows, "residual_gate_mean")
    return {
        "ssr_high_freq_ratio_mean": mean(high) if high else None,
        "ssr_high_freq_ratio_max": max(high) if high else None,
        "ssr_boundary_ratio_mean": mean(boundary) if boundary else None,
        "ssr_gamma_mean": mean(gamma) if gamma else None,
        "ssr_update_band0_mean": mean(update_b0) if update_b0 else None,
        "ssr_suppress_high_mean": mean(suppress_high) if suppress_high else None,
        "ssr_high_freq_penalty_mean": mean(hf_penalty) if hf_penalty else None,
        "ssr_residual_gate_mean": mean(residual_gate) if residual_gate else None,
    }


def summarize_run(run_dir: Path, variant: str, args: argparse.Namespace) -> dict[str, Any]:
    summary = read_json(run_dir / "summary.json")
    training_rows = read_csv_rows(run_dir / "training_log.csv")
    ssr_rows = read_csv_rows(run_dir / "ssr_logs.csv")
    best_row = best_training_row(training_rows)
    final_row = training_rows[-1] if training_rows else None

    best_val = as_float(summary.get("best_val_fg_mean"), as_float(summary.get("best_val_fg_dice")))
    if best_row is not None:
        best_val = metric_from_row(best_row, "val_fg_mean", "val_fg_dice", "val_fg") or best_val
    final_val = as_float(summary.get("final_val_fg_mean"))
    if final_row is not None:
        final_val = metric_from_row(final_row, "val_fg_mean", "val_fg_dice", "val_fg") or final_val

    best_epoch = as_int(summary.get("best_epoch"))
    if best_row is not None:
        best_epoch = as_int(best_row.get("epoch"), best_epoch)

    completed_epochs = len(training_rows) or as_int(summary.get("epochs"), 0) or 0
    row: dict[str, Any] = {
        "rank": None,
        "variant": summary.get("variant") or variant,
        "run_name": summary.get("run_name") or run_dir.name,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "best_val_fg_mean": best_val,
        "final_val_fg_mean": final_val,
        "overfit_gap": (best_val - final_val) if best_val is not None and final_val is not None else None,
        "val_loss_at_best": metric_from_row(best_row or {}, "val_loss") if best_row else None,
        "final_val_loss": metric_from_row(final_row or {}, "val_loss") if final_row else None,
        "best_val_dice_RV": as_float(summary.get("best_val_dice_RV")),
        "best_val_dice_MYO": as_float(summary.get("best_val_dice_MYO")),
        "best_val_dice_LV": as_float(summary.get("best_val_dice_LV")),
        "actual_batch_size": as_int(summary.get("actual_batch_size")),
        "image_size": as_int(summary.get("image_size")),
        "model_parameter_count": as_int(summary.get("model_parameter_count")),
    }
    if best_row is not None:
        row["best_val_dice_RV"] = metric_from_row(best_row, "val_dice_RV") or row["best_val_dice_RV"]
        row["best_val_dice_MYO"] = metric_from_row(best_row, "val_dice_MYO") or row["best_val_dice_MYO"]
        row["best_val_dice_LV"] = metric_from_row(best_row, "val_dice_LV") or row["best_val_dice_LV"]

    row.update(summarize_ssr(ssr_rows))
    row["warnings"] = build_warnings(row, args)
    return row


def build_warnings(row: dict[str, Any], args: argparse.Namespace) -> str:
    warnings = []
    if (row.get("completed_epochs") or 0) < args.min_complete_epochs:
        warnings.append(f"incomplete_epochs<{args.min_complete_epochs}")
    if row.get("overfit_gap") is not None and row["overfit_gap"] > args.overfit_gap:
        warnings.append(f"overfit_gap>{args.overfit_gap:.3f}")
    if row.get("ssr_high_freq_ratio_max") is not None and row["ssr_high_freq_ratio_max"] > args.high_freq_risk:
        warnings.append(f"hf_ratio_max>{args.high_freq_risk:.1f}")
    rv = row.get("best_val_dice_RV")
    myo = row.get("best_val_dice_MYO")
    lv = row.get("best_val_dice_LV")
    if any(v is not None and v < 0.05 for v in (rv, myo, lv)):
        warnings.append("class_collapse_risk")
    return ";".join(warnings)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], missing: list[str], args: argparse.Namespace) -> None:
    lines = [
        "# SSR Phase 2 Baseline Benchmark",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Output root: `{args.output_root}`",
        f"- Ranking metric: `best_val_fg_mean` (higher is better)",
        "",
    ]
    if missing:
        lines += ["## Missing Runs", "", *[f"- `{variant}`" for variant in missing], ""]
    lines += [
        "## Ranking",
        "",
        "| Rank | Variant | Best FG | Best Epoch | Final FG | RV | MYO | LV | HF Max | Boundary Mean | Warnings | Run Dir |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rank"]),
                    str(row["variant"]),
                    fmt(row.get("best_val_fg_mean")),
                    fmt(row.get("best_epoch"), 0),
                    fmt(row.get("final_val_fg_mean")),
                    fmt(row.get("best_val_dice_RV")),
                    fmt(row.get("best_val_dice_MYO")),
                    fmt(row.get("best_val_dice_LV")),
                    fmt(row.get("ssr_high_freq_ratio_max")),
                    fmt(row.get("ssr_boundary_ratio_mean")),
                    str(row.get("warnings") or ""),
                    f"`{row['run_dir']}`",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Selection Notes",
        "",
        "- Use `best_model.pt` from the top-ranked run, not `final_model.pt`, when the overfit gap is nonzero.",
        "- If two variants are close, prefer the one without class-collapse warnings and with lower high-frequency amplification.",
        "- Treat runs with different seeds, image sizes, max slices, or completed epochs as non-comparable until rerun under the same protocol.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_rankings(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not rows:
        return
    names = [str(r["variant"]) for r in rows]
    values = [float(r["best_val_fg_mean"] or 0.0) for r in rows]
    plt.figure(figsize=(max(7, len(rows) * 1.2), 4))
    plt.bar(names, values)
    plt.ylabel("best_val_fg_mean")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    variants = args.variants or parse_variants_from_script(Path(args.variant_script))
    out_dir = Path(args.out_dir) if args.out_dir else output_root / f"{args.run_prefix}_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((out_dir / ".plot_cache" / "matplotlib").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str((out_dir / ".plot_cache" / "xdg").resolve()))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for variant in variants:
        run_dir = find_run_dir(output_root, args.run_prefix, variant)
        if run_dir is None:
            missing.append(variant)
            continue
        rows.append(summarize_run(run_dir, variant, args))

    if args.strict and missing:
        raise SystemExit(f"Missing expected variant outputs: {missing}")
    if not rows:
        raise SystemExit(f"No benchmarkable runs found under {output_root}")

    rows.sort(
        key=lambda r: (
            r.get("best_val_fg_mean") if r.get("best_val_fg_mean") is not None else -1.0,
            -(r.get("overfit_gap") or 0.0),
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    write_csv(out_dir / "baseline_benchmark.csv", rows)
    (out_dir / "baseline_benchmark.json").write_text(json.dumps({"rows": rows, "missing": missing}, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_dir / "baseline_benchmark.md", rows, missing, args)
    plot_rankings(out_dir / "baseline_benchmark.png", rows)

    print("SSR Phase 2 baseline ranking")
    for row in rows:
        print(
            f"{row['rank']:>2}. {row['variant']:<18} "
            f"best_fg={fmt(row.get('best_val_fg_mean')):<8} "
            f"epoch={fmt(row.get('best_epoch'), 0):<4} "
            f"final_fg={fmt(row.get('final_val_fg_mean')):<8} "
            f"HFmax={fmt(row.get('ssr_high_freq_ratio_max')):<8} "
            f"warnings={row.get('warnings') or '-'}"
        )
    if missing:
        print("Missing variants:", ", ".join(missing))
    print(f"Reports saved under {out_dir}")


if __name__ == "__main__":
    main()
