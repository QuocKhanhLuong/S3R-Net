#!/usr/bin/env python3
"""Train S3R with phased S3R-SCSD cache-based distillation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
for path in (ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from exc

from data.acdc_s3r_dataset import ACDCSSRSliceDataset, load_or_create_split
from distillation.distill_losses import S3RSCSDLoss
from distillation.teacher_cache import load_teacher_cache
from models.s3r import S3RLoss, build_s3r_model
from models.s3r.losses import boundary_map_from_mask
from models.s3r.metrics import segmentation_surface_metrics
from training.train_s3r_acdc import (
    append_classification_metrics,
    finish_wandb,
    format_class_metric_table,
    init_wandb,
    log_wandb_metrics,
    nanmean_or_nan,
    save_prediction_grid,
    should_compute_surface_metrics,
    surface_metric_keys,
)


CLASS_NAMES = ["BG", "RV", "MYO", "LV"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S3R-SCSD distillation training")
    parser.add_argument("--config", default="src/distillation/configs/s3r_scsd_phase3_dual_routing.yaml")
    parser.add_argument("--phase", default=None)
    parser.add_argument("--teacher_cache_dir", default=None)
    parser.add_argument("--semantic_teacher_checkpoint", default=None)
    parser.add_argument("--characteristic_teacher_checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--input_mode", choices=["2d", "25d"], default=None)
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--max_slices", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--model", choices=["s3r", "s3r_mini", "s3r_net"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_tqdm", action="store_true", default=None)
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_mode", default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("phase", "phase3_dual_routing")
    cfg.setdefault("data_root", "preprocessed_data/ACDC")
    cfg.setdefault("output_root", "weights/s3r_distill")
    cfg.setdefault("model", "s3r_mini")
    cfg.setdefault("image_size", 224)
    cfg.setdefault("input_mode", "2d")
    cfg.setdefault("in_channels", 5 if str(cfg["input_mode"]).lower() == "25d" else 1)
    cfg.setdefault("epochs", 200)
    cfg.setdefault("batch_size", 8)
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "cuda")
    cfg.setdefault("base_channels", 32)
    cfg.setdefault("num_classes", 4)
    cfg.setdefault("num_bands", 4)
    cfg.setdefault("lr", 3e-4)
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("grad_clip", 3.0)
    cfg.setdefault("teacher_cache_dir", None)
    cfg.setdefault("missing_cache_policy", "raise")
    cfg.setdefault("run_name", str(cfg["phase"]))
    cfg.setdefault("split_manifest", "splits/acdc_patient_split_seed42.json")
    cfg.setdefault("foreground_only", True)
    cfg.setdefault("loss_weights", {})
    cfg["loss_weights"].setdefault("supervised", 1.0)
    cfg.setdefault("s3r_loss_weights", {})
    cfg.setdefault("ssr", {})
    cfg.setdefault("use_tqdm", True)

    metrics = cfg.setdefault("metrics", {})
    metrics.setdefault("surface_tolerance", 2)
    metrics.setdefault("compute_hd95", True)
    metrics.setdefault("compute_assd", True)
    metrics.setdefault("compute_boundary_f1", False)
    metrics.setdefault("compute_surface_dice", False)
    metrics.setdefault("full_every_n_epochs", 1)

    wandb_cfg = cfg.setdefault("wandb", {})
    wandb_cfg.setdefault("enabled", False)
    wandb_cfg.setdefault("project", "s3r-scsd")
    wandb_cfg.setdefault("entity", None)
    wandb_cfg.setdefault("run_name", None)
    wandb_cfg.setdefault("mode", "online")
    return cfg


def apply_cli(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key in (
        "phase",
        "teacher_cache_dir",
        "epochs",
        "batch_size",
        "image_size",
        "input_mode",
        "in_channels",
        "max_slices",
        "run_name",
        "model",
        "device",
        "data_root",
        "output_root",
        "num_workers",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.semantic_teacher_checkpoint is not None:
        cfg.setdefault("semantic_teacher", {})["checkpoint"] = args.semantic_teacher_checkpoint
    if args.characteristic_teacher_checkpoint is not None:
        cfg.setdefault("characteristic_teacher", {})["checkpoint"] = args.characteristic_teacher_checkpoint
    if args.input_mode is not None and args.in_channels is None:
        cfg["in_channels"] = 5 if args.input_mode == "25d" else 1
    if args.no_tqdm:
        cfg["use_tqdm"] = False
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    for arg_key, cfg_key in (
        ("wandb_project", "project"),
        ("wandb_entity", "entity"),
        ("wandb_run_name", "run_name"),
        ("wandb_mode", "mode"),
    ):
        value = getattr(args, arg_key)
        if value is not None:
            cfg.setdefault("wandb", {})[cfg_key] = value
    return apply_defaults(cfg)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is not available; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def make_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    train_cases, val_cases, split_manifest = load_or_create_split(
        cfg["data_root"],
        seed=int(cfg["seed"]),
        train_fraction=0.8,
        split_manifest=cfg.get("split_manifest", "splits/acdc_patient_split_seed42.json"),
    )
    common = {
        "data_root": cfg["data_root"],
        "input_mode": cfg["input_mode"],
        "image_size": int(cfg["image_size"]),
        "foreground_only": bool(cfg.get("foreground_only", True)),
        "max_slices": cfg.get("max_slices"),
    }
    train_ds = ACDCSSRSliceDataset(case_ids=train_cases, seed=int(cfg["seed"]), **common)
    val_ds = ACDCSSRSliceDataset(case_ids=val_cases, seed=int(cfg["seed"]) + 1, **common)
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, {
        "train_cases": train_cases,
        "val_cases": val_cases,
        "train_slices": len(train_ds),
        "val_slices": len(val_ds),
        "split_manifest": split_manifest,
    }


def build_model(cfg: dict[str, Any]) -> nn.Module:
    ssr_cfg = dict(cfg.get("ssr", {}))
    ssr_cfg.setdefault("num_bands", int(cfg["num_bands"]))
    return build_s3r_model(
        model=str(cfg["model"]),
        in_channels=int(cfg["in_channels"]),
        base_channels=int(cfg["base_channels"]),
        num_classes=int(cfg["num_classes"]),
        num_bands=int(cfg["num_bands"]),
        state_dim=cfg.get("state_dim"),
        ssr=ssr_cfg,
    )


def build_supervised_loss(cfg: dict[str, Any]) -> S3RLoss:
    weights = cfg.get("s3r_loss_weights", {})
    return S3RLoss(
        num_classes=int(cfg["num_classes"]),
        num_bands=int(cfg["num_bands"]),
        boundary_bce_weight=float(weights.get("boundary_bce", 0.50)),
        boundary_dice_weight=float(weights.get("boundary_dice", 0.30)),
        boundary_freq_weight=float(weights.get("boundary_frequency", 0.20)),
        tv_weight=float(weights.get("tv", 0.05)),
        gate_reg_weight=float(weights.get("gate_reg", 0.03)),
        hf_ratio_weight=float(weights.get("hf_ratio", 0.005)),
        state_reg_weight=float(weights.get("state_reg", 0.0)),
    )


def load_batch_teacher_cache(batch: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor] | None:
    cache_dir = cfg.get("teacher_cache_dir")
    if not cache_dir:
        return None
    samples = []
    for case_id, slice_idx in zip(batch["case_id"], batch["slice_idx"]):
        try:
            samples.append(load_teacher_cache(cache_dir, str(case_id), int(slice_idx), dataset="ACDC"))
        except FileNotFoundError:
            if cfg.get("missing_cache_policy") == "supervised_only":
                return None
            raise
    keys = [key for key in samples[0] if key != "metadata_json"]
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        out[key] = torch.from_numpy(np.stack([sample[key] for sample in samples], axis=0)).float().to(device)
    return out


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    supervised_loss: S3RLoss,
    distill_loss: S3RSCSDLoss,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
    split: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)
    total_samples = 0
    totals: dict[str, float] = {}
    distill_rows: list[dict[str, Any]] = []
    num_classes = int(cfg["num_classes"])
    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)
    surface_values: dict[str, list[float]] = {key: [] for key in surface_metric_keys()}
    compute_surface = (not training) and should_compute_surface_metrics(cfg, epoch)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        progress = tqdm(
            loader,
            desc=f"{split} e{epoch:03d}",
            leave=False,
            disable=not bool(cfg.get("use_tqdm", True)),
        )
        for batch_idx, batch in enumerate(progress):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            boundary_target = boundary_map_from_mask(masks).to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images, boundary_mask=boundary_target, return_logs=(batch_idx == 0))
            sup_loss, sup_parts = supervised_loss(outputs, masks, boundary_target)
            teacher = load_batch_teacher_cache(batch, cfg, device) if training else None
            if training and teacher is not None:
                kd_loss, kd_parts = distill_loss(outputs, masks, teacher)
            else:
                kd_loss = torch.zeros((), device=device, dtype=sup_loss.dtype)
                kd_parts = {
                    "semantic_kd": 0.0,
                    "boundary_kd": 0.0,
                    "distance_kd": 0.0,
                    "spectral_boundary_kd": 0.0,
                    "state_kd": 0.0,
                    "distill_loss": 0.0,
                }
            loss = float(cfg["loss_weights"].get("supervised", 1.0)) * sup_loss + kd_loss
            if training:
                loss.backward()
                grad_clip = float(cfg.get("grad_clip", 0.0) or 0.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

            batch_size = int(images.shape[0])
            total_samples += batch_size
            parts = {**sup_parts, **kd_parts, "total_loss": float(loss.detach().cpu())}
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
            if training:
                distill_rows.append({"epoch": epoch, "split": split, "batch": batch_idx, **kd_parts})

            preds = outputs["seg_logits"].argmax(dim=1)
            for cls in range(num_classes):
                pred_c = preds == cls
                target_c = masks == cls
                tp[cls] += (pred_c & target_c).sum()
                fp[cls] += (pred_c & ~target_c).sum()
                fn[cls] += (~pred_c & target_c).sum()

            if compute_surface:
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

    metrics = {key: value / max(total_samples, 1) for key, value in totals.items()}
    append_classification_metrics(metrics, tp, fp, fn, CLASS_NAMES[:num_classes])
    for key, vals in surface_values.items():
        metrics[key] = nanmean_or_nan(vals)
    return metrics, distill_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_empty_s3r_log(path: Path) -> None:
    if path.exists():
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "split", "block", "metric", "band", "value"])
        writer.writeheader()


def is_cuda_oom(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def print_distill_epoch_report(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
    print(
        f"epoch {epoch:03d} "
        f"train_loss={train_metrics.get('total_loss', math.nan):.4f} "
        f"val_loss={val_metrics.get('total_loss', math.nan):.4f} "
        f"train_fg={train_metrics.get('fg_dice', math.nan):.4f} "
        f"val_fg={val_metrics.get('fg_dice', math.nan):.4f} "
        f"sem_kd={train_metrics.get('semantic_kd', 0.0):.4f} "
        f"bnd_kd={train_metrics.get('boundary_kd', 0.0):.4f} "
        f"state_kd={train_metrics.get('state_kd', 0.0):.4f}"
    )
    print(format_class_metric_table(val_metrics))


def train_once(cfg: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(cfg["seed"]))
    device = resolve_device(str(cfg["device"]))
    if device.type == "cpu" and int(cfg.get("num_workers", 0)) > 0:
        cfg["num_workers"] = 0
    run_dir = Path(cfg["output_root"]) / str(cfg["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["actual_batch_size"] = int(cfg["batch_size"])
    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_loader, val_loader, split_info = make_loaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    wandb_run = init_wandb(cfg, run_dir)
    supervised_loss = build_supervised_loss(cfg)
    distill_loss = S3RSCSDLoss(
        num_classes=int(cfg["num_classes"]),
        num_bands=int(cfg["num_bands"]),
        phase=str(cfg["phase"]),
        loss_weights=cfg.get("loss_weights", {}),
    )
    print(
        f"S3R-SCSD run={cfg['run_name']} phase={cfg['phase']} model={cfg['model']} "
        f"device={device} batch={cfg['actual_batch_size']} train={split_info['train_slices']} val={split_info['val_slices']}"
    )

    training_rows: list[dict[str, Any]] = []
    distill_rows: list[dict[str, Any]] = []
    best_val = -1.0
    best_epoch = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_metrics, batch_distill = run_epoch(model, train_loader, optimizer, supervised_loss, distill_loss, device, cfg, epoch, "train")
        val_metrics, _ = run_epoch(model, val_loader, None, supervised_loss, distill_loss, device, cfg, epoch, "val")
        distill_rows.extend(batch_distill)
        row = {
            "epoch": epoch,
            "actual_batch_size": int(cfg["actual_batch_size"]),
            "train_loss": train_metrics["total_loss"],
            "val_loss": val_metrics["total_loss"],
            "train_fg_mean": train_metrics["fg_dice"],
            "val_fg_mean": val_metrics["fg_dice"],
            "semantic_kd": train_metrics.get("semantic_kd", 0.0),
            "boundary_kd": train_metrics.get("boundary_kd", 0.0),
            "distance_kd": train_metrics.get("distance_kd", 0.0),
            "spectral_boundary_kd": train_metrics.get("spectral_boundary_kd", 0.0),
            "state_kd": train_metrics.get("state_kd", 0.0),
            "val_hd95_fg_mean": val_metrics.get("hd95_fg_mean", math.nan),
            "val_assd_fg_mean": val_metrics.get("assd_fg_mean", math.nan),
        }
        for name in CLASS_NAMES[: int(cfg["num_classes"])]:
            lower = name.lower()
            for metric_name in ("dice", "precision", "recall"):
                row[f"train_{metric_name}_{name}"] = train_metrics.get(f"{metric_name}_{name}", math.nan)
                row[f"val_{metric_name}_{name}"] = val_metrics.get(f"{metric_name}_{name}", math.nan)
            row[f"val_hd95_{lower}"] = val_metrics.get(f"hd95_{lower}", math.nan)
            row[f"val_assd_{lower}"] = val_metrics.get(f"assd_{lower}", math.nan)
        training_rows.append(row)
        if val_metrics["fg_dice"] > best_val:
            best_val = val_metrics["fg_dice"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_fg_mean": best_val, "config": cfg}, run_dir / "best_model.pt")
        write_csv(run_dir / "training_log.csv", training_rows)
        write_csv(run_dir / "distill_log.csv", distill_rows)
        write_empty_s3r_log(run_dir / "s3r_logs.csv")
        print_distill_epoch_report(epoch, train_metrics, val_metrics)
        log_wandb_metrics(wandb_run, epoch, train_metrics, val_metrics, {"lr": optimizer.param_groups[0]["lr"]})

    torch.save({"epoch": int(cfg["epochs"]), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "config": cfg}, run_dir / "final_model.pt")
    save_prediction_grid(model, train_loader, device, run_dir / "train_predictions.png")
    save_prediction_grid(model, val_loader, device, run_dir / "val_predictions.png")
    summary = {
        "run_name": cfg["run_name"],
        "phase": cfg["phase"],
        "model": cfg["model"],
        "best_epoch": best_epoch,
        "best_val_fg_mean": best_val,
        "actual_batch_size": int(cfg["actual_batch_size"]),
        "train_slices": split_info["train_slices"],
        "val_slices": split_info["val_slices"],
        "artifacts": [
            "training_log.csv",
            "distill_log.csv",
            "s3r_logs.csv",
            "summary.json",
            "config_resolved.yaml",
            "best_model.pt",
            "final_model.pt",
            "train_predictions.png",
            "val_predictions.png",
        ],
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    finish_wandb(wandb_run)
    return summary


def train_with_oom_recovery(cfg: dict[str, Any]) -> dict[str, Any]:
    batch_size = int(cfg["batch_size"])
    while batch_size >= 1:
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg["batch_size"] = batch_size
        try:
            return train_once(trial_cfg)
        except RuntimeError as exc:
            if is_cuda_oom(exc) and torch.cuda.is_available():
                print(f"CUDA OOM at batch_size={batch_size}; retrying with batch_size={batch_size // 2}.")
                torch.cuda.empty_cache()
                batch_size //= 2
                if batch_size < 1:
                    raise RuntimeError("CUDA OOM even at batch_size=1.") from exc
                continue
            raise
    raise RuntimeError("CUDA OOM recovery exhausted all batch sizes.")


def main() -> None:
    args = parse_args()
    cfg = apply_cli(apply_defaults(load_config(args.config)), args)
    train_with_oom_recovery(cfg)


if __name__ == "__main__":
    main()
