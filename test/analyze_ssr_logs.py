#!/usr/bin/env python3
"""Analyze SSRBlockV3 phase-2 diagnostic logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SSR phase-2 logs")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--high_ratio_threshold", type=float, default=6.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("epoch", "band", "value"):
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def load_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config_resolved.yaml"
    if cfg_path.exists() and yaml is not None:
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def prepare_plot_cache(run_dir: Path) -> None:
    cache_root = run_dir / ".plot_cache"
    mpl_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache.resolve()))


def plot_curves(run_dir: Path, ssr_rows: list[dict[str, Any]]) -> None:
    prepare_plot_cache(run_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return

    def plot(metric: str, filename: str, title: str) -> None:
        rows = [r for r in ssr_rows if r["split"] == "val" and r["metric"] == metric]
        if not rows:
            return
        plt.figure(figsize=(9, 5))
        for block in sorted({r["block"] for r in rows}):
            bands = sorted({r["band"] for r in rows if r["block"] == block}, key=lambda x: -1 if x == "" else int(x))
            for band in bands:
                series = [r for r in rows if r["block"] == block and r["band"] == band]
                series.sort(key=lambda r: r["epoch"])
                label = f"{block}:b{int(band)}" if band != "" else block
                plt.plot([r["epoch"] for r in series], [r["value"] for r in series], marker="o", linewidth=1.4, label=label)
        plt.title(title)
        plt.xlabel("epoch")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(run_dir / filename, dpi=160)
        plt.close()

    plot("update_gate_mean", "update_gate.png", "Update gate")
    plot("suppress_gate_mean", "suppress_gate.png", "Suppress gate")
    plot("retain_gate_mean", "retain_gate.png", "Retain gate")
    plot("high_freq_ratio", "high_freq_ratio.png", "High-frequency ratio")
    plot("high_freq_penalty", "high_freq_penalty.png", "High-frequency penalty")
    plot("boundary_to_nonboundary_high_ratio", "boundary_ratio.png", "Boundary/non-boundary high ratio")
    plot("gamma", "gamma.png", "Effective gamma")
    plot("update_contribution", "contribution_update.png", "Update contribution")
    plot("suppress_contribution", "contribution_suppress.png", "Suppress contribution")
    plot("residual_gate_mean", "residual_gate_mean.png", "Residual gate mean")


def values(rows: list[dict[str, Any]], metric: str, *, band: int | None = None, split: str = "val") -> list[float]:
    out = []
    for row in rows:
        if row["split"] != split or row["metric"] != metric:
            continue
        if band is not None:
            if row["band"] == "" or int(row["band"]) != band:
                continue
        out.append(float(row["value"]))
    return out


def block_values(rows: list[dict[str, Any]], metric: str, block: str, *, split: str = "val") -> list[float]:
    return [float(r["value"]) for r in rows if r["split"] == split and r["metric"] == metric and r["block"] == block]


def many(count: int, total: int) -> bool:
    return total > 0 and count > total / 2


def best_final_gap(training_rows: list[dict[str, Any]]) -> tuple[float, float, int]:
    if not training_rows:
        return 0.0, 0.0, 0
    vals = [float(r.get("val_fg_mean", r.get("val_fg_dice", 0.0))) for r in training_rows]
    best = max(vals)
    best_epoch = vals.index(best) + 1
    return best, vals[-1], best_epoch


def print_diagnosis(training_rows: list[dict[str, Any]], ssr_rows: list[dict[str, Any]], cfg: dict[str, Any], high_threshold: float) -> None:
    ssr_cfg = cfg.get("ssr", {})
    update_budget = float(ssr_cfg.get("update_budget", 1.5))
    suppress_max = ssr_cfg.get("suppress_max", [0.05, 0.15, 0.25, 0.25])
    gamma_max = float(ssr_cfg.get("gamma_max", 0.25))

    print("SSR phase-2 diagnosis")
    best, final, best_epoch = best_final_gap(training_rows)
    if training_rows:
        print(f"- Best val fg mean: {best:.4f} at epoch {best_epoch}; final: {final:.4f}")
        if best - final > 0.03:
            print(f"- Overfitting warning: best-final gap is {best - final:.4f}")

    update_b0 = values(ssr_rows, "update_gate_mean", band=0)
    collapse_threshold = 0.75 * update_budget
    collapse_count = sum(v > collapse_threshold for v in update_b0)
    print(f"- Update gate collapse: {'YES' if many(collapse_count, len(update_b0)) else 'no'} ({collapse_count}/{len(update_b0)} band0 snapshots > {collapse_threshold:.3f})")

    suppress_hi = values(ssr_rows, "suppress_gate_mean", band=2) + values(ssr_rows, "suppress_gate_mean", band=3)
    weak_count = sum(v < 0.01 for v in suppress_hi)
    print(f"- Suppress too weak in bands 2/3: {'YES' if many(weak_count, len(suppress_hi)) else 'no'} ({weak_count}/{len(suppress_hi)} < 0.01)")

    sat_vals = []
    for band in (2, 3):
        threshold = 0.9 * float(suppress_max[band])
        sat_vals.extend(v > threshold for v in values(ssr_rows, "suppress_gate_mean", band=band))
    sat_count = sum(bool(v) for v in sat_vals)
    print(f"- Suppress saturation in bands 2/3: {'YES' if many(sat_count, len(sat_vals)) else 'no'} ({sat_count}/{len(sat_vals)} > 0.9*suppress_max)")

    high_ratios = values(ssr_rows, "high_freq_ratio")
    risk_count = sum(v > high_threshold for v in high_ratios)
    severe_count = sum(v > 10.0 for v in high_ratios)
    print(f"- High-frequency amplification risk: {'YES' if many(risk_count, len(high_ratios)) else 'no'} ({risk_count}/{len(high_ratios)} > {high_threshold:.1f}); severe {severe_count}/{len(high_ratios)} > 10.0")

    boundary_ratios = values(ssr_rows, "boundary_to_nonboundary_high_ratio")
    good_count = sum(v > 1.5 for v in boundary_ratios)
    strong_count = sum(v > 2.0 for v in boundary_ratios)
    print(f"- Boundary ratio: good {good_count}/{len(boundary_ratios)} > 1.5; strong {strong_count}/{len(boundary_ratios)} > 2.0")

    blocks = sorted({r["block"] for r in ssr_rows})
    for block in blocks:
        gammas = block_values(ssr_rows, "gamma", block)
        if not gammas:
            continue
        near_count = sum(g > 0.9 * gamma_max for g in gammas)
        print(f"- Gamma {block}: mean={sum(gammas)/len(gammas):.4f}, max={max(gammas):.4f}, near gamma_max={near_count}/{len(gammas)}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    training_rows = read_csv(run_dir / "training_log.csv")
    ssr_rows = read_csv(run_dir / "ssr_logs.csv")
    cfg = load_config(run_dir)
    plot_curves(run_dir, ssr_rows)
    print_diagnosis(training_rows, ssr_rows, cfg, args.high_ratio_threshold)
    print(f"Analysis plots saved under {run_dir}")


if __name__ == "__main__":
    main()
