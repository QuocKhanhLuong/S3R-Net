#!/usr/bin/env python3
"""Evaluate a trained SSR run under simple intensity and blur perturbations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ssr_blocks import boundary_map_from_mask
from ssr_metrics import HAS_SCIPY, segmentation_surface_metrics
from train_ssr_acdc import (
    CLASS_NAMES,
    apply_config_defaults,
    build_model,
    load_config,
    make_loaders,
    prepare_plot_cache,
    resolve_device,
    seed_everything,
    surface_metric_keys,
)


PerturbFn = Callable[[Tensor], Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness evaluation for SSR ACDC runs")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--perturbations",
        nargs="*",
        default=["all"],
        choices=["clean", "gaussian_noise", "blur", "contrast", "gamma", "all"],
    )
    parser.add_argument("--save_predictions", default="true")
    return parser.parse_args()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg_path = Path(args.config) if args.config else run_dir / "config_resolved.yaml"
    ckpt_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not HAS_SCIPY:
        raise ImportError("Robustness boundary metrics require scipy.ndimage. Install scipy first.")

    cfg = apply_config_defaults(load_config(cfg_path))
    cfg["run_name"] = str(run_dir.name)
    cfg["output_root"] = str(run_dir.parent)
    cfg["batch_size"] = int(cfg.get("actual_batch_size") or cfg.get("batch_size", 8))
    seed_everything(int(cfg["seed"]))
    device = resolve_device(str(cfg["device"]))
    if device.type == "cpu":
        cfg["num_workers"] = 0

    _, val_loader, split_info = make_loaders(cfg)
    model = build_model(cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    requested = set(args.perturbations)
    specs = build_perturbations(cfg, requested)
    print(
        f"Robustness eval run={run_dir.name} checkpoint={ckpt_path.name} "
        f"val_slices={split_info['val_slices']} perturbations={len(specs)}"
    )

    rows: list[dict[str, Any]] = []
    for spec in specs:
        label, kind, level, perturb_fn = spec
        row = evaluate_perturbation(model, val_loader, device, cfg, label, kind, level, perturb_fn)
        rows.append(row)
        print(
            f"{label}: fg={row['fg_mean']:.4f} hd95={row['hd95_fg_mean']:.3f} "
            f"assd={row['assd_fg_mean']:.3f} bf1={row['boundary_f1_fg']:.4f}"
        )

    write_csv(run_dir / "robustness_metrics.csv", rows)
    write_summary(run_dir / "robustness_summary.json", rows, cfg, ckpt_path)
    plot_robustness(run_dir, rows)
    if str_to_bool(args.save_predictions):
        save_selected_prediction_grids(model, val_loader, device, cfg, run_dir, specs)


def build_perturbations(
    cfg: dict[str, Any],
    requested: set[str],
) -> list[tuple[str, str, float | str, PerturbFn]]:
    if "all" in requested:
        requested = {"clean", "gaussian_noise", "blur", "contrast", "gamma"}
    robustness = cfg.get("robustness", {})
    specs: list[tuple[str, str, float | str, PerturbFn]] = []
    if "clean" in requested:
        specs.append(("clean", "clean", "clean", lambda x: x))
    if "gaussian_noise" in requested:
        for sigma in robustness.get("gaussian_noise_sigmas", [0.05, 0.10, 0.15]):
            sigma = float(sigma)
            specs.append((f"gaussian_noise_sigma{sigma:.2f}", "gaussian_noise", sigma, lambda x, s=sigma: x + s * torch.randn_like(x)))
    if "blur" in requested:
        for sigma in robustness.get("blur_sigmas", [0.75, 1.25, 1.75]):
            sigma = float(sigma)
            specs.append((f"blur_sigma{sigma:.2f}", "blur", sigma, lambda x, s=sigma: gaussian_blur(x, s)))
    if "contrast" in requested:
        for factor in robustness.get("contrast_factors", [0.75, 1.25, 1.50]):
            factor = float(factor)
            specs.append((f"contrast_factor{factor:.2f}", "contrast", factor, lambda x, f=factor: x * f))
    if "gamma" in requested:
        for gamma in robustness.get("gamma_values", [0.7, 1.5]):
            gamma = float(gamma)
            specs.append((f"gamma{gamma:g}", "gamma", gamma, lambda x, g=gamma: gamma_intensity(x, g)))
    return specs


def evaluate_perturbation(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    label: str,
    kind: str,
    level: float | str,
    perturb_fn: PerturbFn,
) -> dict[str, Any]:
    num_classes = int(cfg["num_classes"])
    inter = torch.zeros(num_classes, device=device)
    denom = torch.zeros(num_classes, device=device)
    surface_values = {key: [] for key in surface_metric_keys()}
    ssr_values: dict[str, list[float]] = {}

    with torch.no_grad():
        for batch in loader:
            images = perturb_fn(batch["image"].to(device, non_blocking=True))
            masks = batch["mask"].to(device, non_blocking=True)
            boundary_target = boundary_map_from_mask(masks).to(device)
            outputs = model(images, boundary_mask=boundary_target, return_logs=True)
            preds = outputs["seg_logits"].argmax(dim=1)

            for cls in range(num_classes):
                pred_c = preds == cls
                target_c = masks == cls
                inter[cls] += (pred_c & target_c).sum()
                denom[cls] += pred_c.sum() + target_c.sum()

            batch_surface = segmentation_surface_metrics(
                preds.detach().cpu().numpy(),
                masks.detach().cpu().numpy(),
                num_classes=num_classes,
                tolerance=float(cfg.get("metrics", {}).get("surface_tolerance", 2)),
                spacing=None,
            )
            for key, value in batch_surface.items():
                if key in surface_values and value is not None:
                    surface_values[key].append(float(value))
            collect_ssr_diagnostics(outputs.get("logs", {}), ssr_values)

    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    row: dict[str, Any] = {"perturbation": label, "kind": kind, "level": level}
    for cls, name in enumerate(CLASS_NAMES[:num_classes]):
        row[f"dice_{name}"] = float(dice[cls].detach().cpu())
    row["fg_mean"] = float(np.mean([row[f"dice_{name}"] for name in CLASS_NAMES[1:num_classes]]))
    for key, vals in surface_values.items():
        row[key] = nanmean(vals)
    for key, vals in ssr_values.items():
        row[key] = nanmean(vals)
    return row


def collect_ssr_diagnostics(logs: dict[str, Any], values: dict[str, list[float]]) -> None:
    for block_name, block_logs in (logs or {}).items():
        for metric in ("high_freq_ratio", "boundary_to_nonboundary_high_ratio", "gamma", "residual_gate_mean"):
            if metric in block_logs:
                values.setdefault(f"{block_name}_{metric}", []).append(float(block_logs[metric]))


def gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    if sigma <= 0:
        return x
    radius = max(int(math.ceil(3.0 * sigma)), 1)
    kernel_size = 2 * radius + 1
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - radius
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-8)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    weight = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, weight, padding=radius, groups=x.shape[1])


def gamma_intensity(x: Tensor, gamma: float) -> Tensor:
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    unit = ((x - x_min) / (x_max - x_min).clamp_min(1e-6)).clamp(0.0, 1.0)
    adjusted = unit.pow(gamma)
    mean = adjusted.mean(dim=(-2, -1), keepdim=True)
    std = adjusted.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
    return (adjusted - mean) / std


def save_selected_prediction_grids(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    run_dir: Path,
    specs: list[tuple[str, str, float | str, PerturbFn]],
) -> None:
    selected = {"clean", "gaussian_noise_sigma0.10", "blur_sigma1.25", "contrast_factor1.50", "gamma1.5"}
    pred_dir = run_dir / "robustness_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    prepare_plot_cache(run_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping robustness prediction grids.")
        return

    batch = next(iter(loader))
    masks = batch["mask"].to(device)
    boundary_target = boundary_map_from_mask(masks).to(device)
    for label, _, _, perturb_fn in specs:
        if label not in selected:
            continue
        images = perturb_fn(batch["image"].to(device))
        with torch.no_grad():
            outputs = model(images, boundary_mask=boundary_target)
            preds = outputs["seg_logits"].argmax(dim=1)
            boundary_prob = torch.sigmoid(outputs["boundary_logits"])
        n = min(4, images.shape[0])
        fig, axes = plt.subplots(n, 5, figsize=(13, 2.6 * n))
        if n == 1:
            axes = np.expand_dims(axes, axis=0)
        for idx in range(n):
            img = images[idx, images.shape[1] // 2].detach().cpu().numpy()
            panels = [
                (img, "image", "gray", None, None),
                (masks[idx].detach().cpu().numpy(), "mask", "viridis", 0, 3),
                (preds[idx].detach().cpu().numpy(), "prediction", "viridis", 0, 3),
                (boundary_prob[idx, 0].detach().cpu().numpy(), "boundary prob", "magma", 0, 1),
                ((preds[idx] != masks[idx]).float().detach().cpu().numpy(), "error", "Reds", 0, 1),
            ]
            for ax, (data, title, cmap, vmin, vmax) in zip(axes[idx], panels):
                ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(title)
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(pred_dir / f"{label}.png", dpi=160)
        plt.close(fig)


def plot_robustness(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    prepare_plot_cache(run_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping robustness plots.")
        return

    labels = [str(row["perturbation"]) for row in rows]
    for key, filename, ylabel in (
        ("fg_mean", "robustness_dice_plot.png", "foreground Dice"),
        ("hd95_fg_mean", "robustness_hd95_plot.png", "HD95 foreground"),
        ("boundary_f1_fg", "robustness_boundary_f1_plot.png", "Boundary F1 foreground"),
    ):
        plt.figure(figsize=(max(8, len(rows) * 0.7), 4))
        plt.plot(labels, [float(row.get(key, math.nan)) for row in rows], marker="o")
        plt.ylabel(ylabel)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(run_dir / filename, dpi=160)
        plt.close()


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


def write_summary(path: Path, rows: list[dict[str, Any]], cfg: dict[str, Any], ckpt_path: Path) -> None:
    clean = next((row for row in rows if row["perturbation"] == "clean"), None)
    worst = min(rows, key=lambda row: finite_or_large(row.get("fg_mean"))) if rows else None
    by_kind: dict[str, dict[str, float | int]] = {}
    for kind in sorted({str(row["kind"]) for row in rows}):
        group = [row for row in rows if row["kind"] == kind]
        by_kind[kind] = {
            "count": len(group),
            "fg_mean": nanmean([float(row.get("fg_mean", math.nan)) for row in group]),
            "hd95_fg_mean": nanmean([float(row.get("hd95_fg_mean", math.nan)) for row in group]),
            "boundary_f1_fg": nanmean([float(row.get("boundary_f1_fg", math.nan)) for row in group]),
        }
    summary = {
        "run_name": cfg.get("run_name"),
        "checkpoint": str(ckpt_path),
        "input_mode": cfg.get("input_mode"),
        "in_channels": cfg.get("in_channels"),
        "surface_metrics": "pixel-based",
        "perturbation_count": len(rows),
        "clean": clean,
        "worst_by_fg_mean": worst,
        "by_kind": by_kind,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.isnan(arr).all():
        return math.nan
    return float(np.nanmean(arr))


def finite_or_large(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return out if math.isfinite(out) else float("inf")


if __name__ == "__main__":
    main()
