#!/usr/bin/env python3
"""Train S3R-Mini/S3R-Net on preprocessed ACDC data."""

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
from torch import Tensor, nn
import torch.nn.functional as F
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
from models.s3r import S3RLoss, build_s3r_model
from models.s3r.losses import boundary_map_from_mask
from models.s3r.metrics import HAS_SCIPY, segmentation_surface_metrics
from models.s3r.spectral_utils import build_radial_frequency_masks
from losses.agreement_kd import compute_dual_teacher_kd_loss
from teachers import CineMATeacher, MedSAM2Teacher, TeacherLoadError
from teachers.teacher_utils import load_dual_teacher_cache, resize_teacher_output_to_student


CLASS_NAMES = ["BG", "RV", "MYO", "LV"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S3R ACDC training")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", choices=["s3r", "s3r_mini", "s3r_net"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--max_slices", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input_mode", choices=["2d", "25d"], default=None)
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--num_bands", type=int, default=None)
    parser.add_argument("--use_s3r_state", action="store_true", default=None)
    parser.add_argument("--return_logs", action="store_true", default=None)
    parser.add_argument("--boundary_weight", type=float, default=None)
    parser.add_argument("--boundary_freq_weight", type=float, default=None)
    parser.add_argument("--hf_ratio_weight", type=float, default=None)
    parser.add_argument("--gate_reg_weight", type=float, default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_tqdm", action="store_true", default=None)
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_mode", default=None)
    parser.add_argument("--use_dual_teacher_kd", action="store_true", default=None)
    parser.add_argument("--teacher_stub", action="store_true", default=None)
    parser.add_argument("--medsam2_stub", action="store_true", default=None)
    parser.add_argument("--cinema_stub", action="store_true", default=None)
    parser.add_argument("--teacher_cache_dir", default=None)
    parser.add_argument("--strict_teacher_cache", action="store_true", default=None)
    parser.add_argument("--precompute_teachers", action="store_true", default=None)
    parser.add_argument("--medsam2_repo_path", default=None)
    parser.add_argument("--medsam2_ckpt_dir", default=None)
    parser.add_argument("--medsam2_ckpt", default=None)
    parser.add_argument("--medsam2_config", default=None)
    parser.add_argument("--medsam2_prompt_mode", default=None)
    parser.add_argument("--cinema_repo_path", default=None)
    parser.add_argument("--cinema_ckpt_dir", default=None)
    parser.add_argument("--cinema_ckpt", default=None)
    parser.add_argument("--cinema_config", default=None)
    parser.add_argument("--cinema_dataset", default=None)
    parser.add_argument("--cinema_view", default=None)
    parser.add_argument("--cinema_seed", type=int, default=None)
    parser.add_argument("--cinema_class_map", default=None)
    parser.add_argument("--kd_temperature", type=float, default=None)
    parser.add_argument("--lambda_field", type=float, default=None)
    parser.add_argument("--lambda_cine_boundary", type=float, default=None)
    parser.add_argument("--lambda_fuse", type=float, default=None)
    parser.add_argument("--lambda_spec", type=float, default=None)
    parser.add_argument("--fused_kd_weight_mode", choices=["none", "agreement"], default=None)
    parser.add_argument("--fused_kd_min_weight", type=float, default=None)
    parser.add_argument("--fused_kd_agreement_power", type=float, default=None)
    parser.add_argument("--teacher_amp", action="store_true", default=None)
    parser.add_argument("--teacher_amp_dtype", choices=["bfloat16", "float16"], default=None)
    parser.add_argument("--teacher_device", default=None)
    parser.add_argument("--teacher_eval_every", type=int, default=None)
    parser.add_argument("--disable_field_kd", action="store_true", default=None)
    parser.add_argument("--disable_cine_boundary_kd", action="store_true", default=None)
    parser.add_argument("--disable_fused_kd", action="store_true", default=None)
    parser.add_argument("--disable_spectral_kd", action="store_true", default=None)
    parser.add_argument("--disable_agreement_weighting", action="store_true", default=None)
    parser.add_argument("--use_vanilla_kd_only", action="store_true", default=None)
    parser.add_argument("--medical_sam3_stub", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--medical_sam3_repo_path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--medical_sam3_ckpt_dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--medical_sam3_prompt_mode", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def normalize_medsam2_kd_config(kd: dict[str, Any]) -> dict[str, Any]:
    """Map deprecated Medical-SAM3 config keys onto the MedSAM2 teacher slot."""
    aliases = {
        "medical_sam3_stub": "medsam2_stub",
        "medical_sam3_repo_path": "medsam2_repo_path",
        "medical_sam3_ckpt_dir": "medsam2_ckpt_dir",
        "medical_sam3_prompt_mode": "medsam2_prompt_mode",
    }
    for old_key, new_key in aliases.items():
        if new_key not in kd and old_key in kd:
            kd[new_key] = kd[old_key]
    return kd


def apply_config_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("model", "s3r_mini")
    cfg.setdefault("run_name", "s3r_mini_acdc")
    cfg.setdefault("seed", 42)
    cfg.setdefault("data_root", "preprocessed_data/ACDC")
    cfg.setdefault("output_root", "weights")
    cfg.setdefault("input_mode", "2d")
    cfg.setdefault("in_channels", 5 if str(cfg["input_mode"]).lower() == "25d" else 1)
    cfg.setdefault("image_size", 224)
    cfg.setdefault("foreground_only", True)
    cfg.setdefault("batch_size", 8)
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("epochs", 200)
    cfg.setdefault("lr", 3e-4)
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("base_channels", 32)
    cfg.setdefault("num_classes", 4)
    cfg.setdefault("num_bands", 4)
    cfg.setdefault("grad_clip", 3.0)
    cfg.setdefault("device", "cuda")
    cfg.setdefault("split_manifest", "splits/acdc_patient_split_seed42.json")
    cfg.setdefault("variant", "default")
    cfg.setdefault("use_tqdm", True)

    wandb_cfg = cfg.setdefault("wandb", {})
    wandb_cfg.setdefault("enabled", False)
    wandb_cfg.setdefault("project", "s3r-acdc")
    wandb_cfg.setdefault("entity", None)
    wandb_cfg.setdefault("run_name", None)
    wandb_cfg.setdefault("mode", "online")

    kd = normalize_medsam2_kd_config(cfg.setdefault("dual_teacher_kd", {}))
    kd.setdefault("enabled", False)
    kd.setdefault("teacher_stub", False)
    kd.setdefault("medsam2_stub", False)
    kd.setdefault("cinema_stub", False)
    kd.setdefault("teacher_cache_dir", None)
    kd.setdefault("strict_teacher_cache", False)
    kd.setdefault("precompute_teachers", False)
    kd.setdefault("medsam2_repo_path", "external/MedSAM2")
    kd.setdefault("medsam2_ckpt_dir", "checkpoints/teachers/medsam2")
    kd.setdefault("medsam2_ckpt", "")
    kd.setdefault("medsam2_config", "configs/sam2.1_hiera_t512.yaml")
    kd.setdefault("medsam2_prompt_mode", "gt_box")
    kd.setdefault("cinema_repo_path", "external/CineMA")
    kd.setdefault("cinema_ckpt_dir", "checkpoints/teachers/cinema")
    kd.setdefault("cinema_ckpt", "")
    kd.setdefault("cinema_config", "")
    kd.setdefault("cinema_dataset", "acdc")
    kd.setdefault("cinema_view", "sax")
    kd.setdefault("cinema_seed", 0)
    kd.setdefault("cinema_class_map", "")
    kd.setdefault("kd_temperature", 4.0)
    kd.setdefault("lambda_field", 0.3)
    kd.setdefault("lambda_cine_boundary", 0.5)
    kd.setdefault("lambda_fuse", 0.5)
    kd.setdefault("lambda_spec", 0.05)
    kd.setdefault("fused_kd_weight_mode", "none")
    kd.setdefault("fused_kd_min_weight", 0.10)
    kd.setdefault("fused_kd_agreement_power", 1.0)
    kd.setdefault("teacher_amp", False)
    kd.setdefault("teacher_amp_dtype", "bfloat16")
    kd.setdefault("teacher_device", "cuda")
    kd.setdefault("teacher_eval_every", 0)
    kd.setdefault("disable_field_kd", False)
    kd.setdefault("disable_cine_boundary_kd", False)
    kd.setdefault("disable_fused_kd", False)
    kd.setdefault("disable_spectral_kd", False)
    kd.setdefault("disable_agreement_weighting", False)
    kd.setdefault("use_vanilla_kd_only", False)

    loss_weights = cfg.setdefault("loss_weights", {})
    loss_weights.setdefault("boundary_bce", 0.50)
    loss_weights.setdefault("boundary_dice", 0.30)
    loss_weights.setdefault("boundary_frequency", 0.20)
    loss_weights.setdefault("tv", 0.05)
    loss_weights.setdefault("gate_reg", 0.03)
    loss_weights.setdefault("hf_ratio", 0.005)

    metrics = cfg.setdefault("metrics", {})
    metrics.setdefault("surface_tolerance", 2)
    metrics.setdefault("compute_hd95", True)
    metrics.setdefault("compute_assd", True)
    metrics.setdefault("compute_boundary_f1", True)
    metrics.setdefault("compute_surface_dice", True)
    metrics.setdefault("spacing_aware", False)
    metrics.setdefault("full_every_n_epochs", 1)

    ssr = cfg.setdefault("ssr", {})
    ssr.setdefault("update_budget", 1.5)
    ssr.setdefault("min_update", 0.08)
    ssr.setdefault("noise_strength", 0.04)
    ssr.setdefault("retain_floor", [0.15, 0.18, 0.22, 0.28])
    ssr.setdefault("suppress_min", [0.00, 0.02, 0.03, 0.03])
    ssr.setdefault("suppress_max", [0.05, 0.15, 0.25, 0.25])
    ssr.setdefault("update_target", [0.30, 0.30, 0.25, 0.15])
    ssr.setdefault("noise_aware_suppress", True)
    ssr.setdefault("use_bounded_gamma", True)
    ssr.setdefault("gamma_max", 0.25)
    ssr.setdefault("gamma_init", -2.0)
    ssr.setdefault("residual_gate_type", "residual_channel_gate")
    ssr.setdefault("residual_gate_max", 0.6)
    ssr.setdefault("se_reduction", 4)
    ssr.setdefault("geometry_refine", "large_kernel")
    ssr.setdefault("large_kernel_size", 7)
    ssr.setdefault("use_hf_ratio_penalty", True)
    ssr.setdefault("hf_ratio_threshold", 4.0)
    return cfg


def apply_variant_and_cli(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = apply_config_defaults(cfg)
    requested_variant = args.variant
    if requested_variant:
        variants = cfg.get("variants", {})
        if requested_variant not in variants:
            raise ValueError(f"Unknown variant {requested_variant!r}. Available: {sorted(variants)}")
        cfg = deep_update(cfg, variants[requested_variant])
        cfg["variant"] = requested_variant
        if args.run_name is None:
            cfg["run_name"] = f"{cfg['run_name']}_{requested_variant}"

    for key in (
        "epochs",
        "batch_size",
        "image_size",
        "max_slices",
        "run_name",
        "device",
        "seed",
        "input_mode",
        "in_channels",
        "base_channels",
        "num_bands",
        "data_root",
        "output_root",
        "save_dir",
        "num_workers",
        "model",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.data_dir is not None:
        cfg["data_root"] = args.data_dir
    if args.boundary_weight is not None:
        cfg.setdefault("loss_weights", {})["boundary_bce"] = args.boundary_weight
    if args.boundary_freq_weight is not None:
        cfg.setdefault("loss_weights", {})["boundary_frequency"] = args.boundary_freq_weight
    if args.hf_ratio_weight is not None:
        cfg.setdefault("loss_weights", {})["hf_ratio"] = args.hf_ratio_weight
    if args.gate_reg_weight is not None:
        cfg.setdefault("loss_weights", {})["gate_reg"] = args.gate_reg_weight
    if args.return_logs is not None:
        cfg["return_logs"] = bool(args.return_logs)
    if args.use_s3r_state is not None:
        cfg["use_s3r_state"] = bool(args.use_s3r_state)
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
    if args.input_mode is not None and args.in_channels is None:
        cfg["in_channels"] = 5 if str(args.input_mode).lower() == "25d" else 1
    kd_cli: dict[str, Any] = {}
    if args.use_dual_teacher_kd:
        kd_cli["enabled"] = True
    for arg_key, cfg_key in (
        ("teacher_stub", "teacher_stub"),
        ("medsam2_stub", "medsam2_stub"),
        ("cinema_stub", "cinema_stub"),
        ("teacher_cache_dir", "teacher_cache_dir"),
        ("strict_teacher_cache", "strict_teacher_cache"),
        ("precompute_teachers", "precompute_teachers"),
        ("medsam2_repo_path", "medsam2_repo_path"),
        ("medsam2_ckpt_dir", "medsam2_ckpt_dir"),
        ("medsam2_ckpt", "medsam2_ckpt"),
        ("medsam2_config", "medsam2_config"),
        ("medsam2_prompt_mode", "medsam2_prompt_mode"),
        ("cinema_repo_path", "cinema_repo_path"),
        ("cinema_ckpt_dir", "cinema_ckpt_dir"),
        ("cinema_ckpt", "cinema_ckpt"),
        ("cinema_config", "cinema_config"),
        ("cinema_dataset", "cinema_dataset"),
        ("cinema_view", "cinema_view"),
        ("cinema_seed", "cinema_seed"),
        ("cinema_class_map", "cinema_class_map"),
        ("kd_temperature", "kd_temperature"),
        ("lambda_field", "lambda_field"),
        ("lambda_cine_boundary", "lambda_cine_boundary"),
        ("lambda_fuse", "lambda_fuse"),
        ("lambda_spec", "lambda_spec"),
        ("fused_kd_weight_mode", "fused_kd_weight_mode"),
        ("fused_kd_min_weight", "fused_kd_min_weight"),
        ("fused_kd_agreement_power", "fused_kd_agreement_power"),
        ("teacher_amp", "teacher_amp"),
        ("teacher_amp_dtype", "teacher_amp_dtype"),
        ("teacher_device", "teacher_device"),
        ("teacher_eval_every", "teacher_eval_every"),
        ("disable_field_kd", "disable_field_kd"),
        ("disable_cine_boundary_kd", "disable_cine_boundary_kd"),
        ("disable_fused_kd", "disable_fused_kd"),
        ("disable_spectral_kd", "disable_spectral_kd"),
        ("disable_agreement_weighting", "disable_agreement_weighting"),
        ("use_vanilla_kd_only", "use_vanilla_kd_only"),
    ):
        value = getattr(args, arg_key)
        if value is not None:
            kd_cli[cfg_key] = value
    for arg_key, cfg_key in (
        ("medical_sam3_stub", "medsam2_stub"),
        ("medical_sam3_repo_path", "medsam2_repo_path"),
        ("medical_sam3_ckpt_dir", "medsam2_ckpt_dir"),
        ("medical_sam3_prompt_mode", "medsam2_prompt_mode"),
    ):
        value = getattr(args, arg_key)
        if value is not None:
            kd_cli[cfg_key] = value
    if kd_cli:
        cfg.setdefault("dual_teacher_kd", {}).update(kd_cli)
    return apply_config_defaults(cfg)


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


def _resolve_teacher_amp_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        if device.type == "cuda" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            print("Requested bfloat16 teacher AMP but this CUDA device does not support BF16; using float16.")
            return torch.float16
        return torch.bfloat16
    raise ValueError(f"Unsupported teacher AMP dtype: {name}")


def make_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    train_cases, val_cases, split_manifest = load_or_create_split(
        cfg["data_root"],
        seed=int(cfg["seed"]),
        train_fraction=0.8,
        split_manifest=cfg.get("split_manifest", "splits/acdc_patient_split_seed42.json"),
    )
    train_ds = ACDCSSRSliceDataset(
        cfg["data_root"],
        case_ids=train_cases,
        input_mode=cfg["input_mode"],
        image_size=int(cfg["image_size"]),
        foreground_only=bool(cfg["foreground_only"]),
        max_slices=cfg.get("max_slices"),
        seed=int(cfg["seed"]),
    )
    val_ds = ACDCSSRSliceDataset(
        cfg["data_root"],
        case_ids=val_cases,
        input_mode=cfg["input_mode"],
        image_size=int(cfg["image_size"]),
        foreground_only=bool(cfg["foreground_only"]),
        max_slices=cfg.get("max_slices"),
        seed=int(cfg["seed"]) + 1,
    )

    generator = torch.Generator()
    generator.manual_seed(int(cfg["seed"]))
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
    in_channels = int(cfg.get("in_channels") or (5 if str(cfg["input_mode"]).lower() == "25d" else 1))
    ssr_cfg = dict(cfg.get("ssr", {}))
    ssr_cfg.setdefault("num_bands", int(cfg["num_bands"]))
    return build_s3r_model(
        model=str(cfg.get("model", "s3r_mini")),
        in_channels=in_channels,
        base_channels=int(cfg["base_channels"]),
        num_classes=int(cfg["num_classes"]),
        num_bands=int(cfg["num_bands"]),
        state_dim=cfg.get("state_dim"),
        ssr=ssr_cfg,
        use_s3r_state=bool(cfg.get("use_s3r_state", True)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_profile(model: nn.Module, cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Estimate S3R student params and FLOPs for one forward pass."""
    in_channels = int(cfg.get("in_channels") or (5 if str(cfg["input_mode"]).lower() == "25d" else 1))
    image_size = int(cfg["image_size"])
    example = torch.zeros(1, in_channels, image_size, image_size, device=device)
    boundary = torch.zeros(1, 1, image_size, image_size, device=device)
    profile: dict[str, Any] = {
        "params": count_parameters(model),
        "flops": None,
        "gflops": None,
        "profile_backend": "unavailable",
    }
    was_training = model.training
    model.eval()
    try:
        try:
            from thop import profile as thop_profile  # type: ignore

            flops, params = thop_profile(model, inputs=(example,), kwargs={"boundary_mask": boundary}, verbose=False)
            profile.update({"params": int(params), "flops": int(flops), "gflops": float(flops) / 1e9, "profile_backend": "thop"})
        except Exception:
            flops = estimate_conv_linear_flops(model, example, boundary)
            profile.update({"flops": int(flops), "gflops": float(flops) / 1e9, "profile_backend": "conv_linear_hooks"})
    finally:
        model.train(was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return profile


def estimate_conv_linear_flops(model: nn.Module, example: Tensor, boundary: Tensor) -> int:
    """Fallback FLOP estimate for Conv/Linear layers only."""
    flops = 0
    hooks = []

    def conv_hook(module: nn.Module, inputs: tuple[Any, ...], output: Tensor) -> None:
        nonlocal flops
        if not isinstance(output, Tensor):
            return
        batch = int(output.shape[0])
        out_spatial = int(np.prod(output.shape[2:])) if output.ndim > 2 else 1
        out_channels = int(output.shape[1]) if output.ndim > 1 else 1
        kernel = int(np.prod(getattr(module, "kernel_size", (1, 1))))
        in_channels = int(getattr(module, "in_channels", 1))
        groups = int(getattr(module, "groups", 1))
        flops += batch * out_spatial * out_channels * (in_channels // groups) * kernel * 2

    def linear_hook(module: nn.Module, inputs: tuple[Any, ...], output: Tensor) -> None:
        nonlocal flops
        if not isinstance(output, Tensor):
            return
        batch_items = int(output.numel() // max(int(output.shape[-1]), 1))
        flops += batch_items * int(module.in_features) * int(module.out_features) * 2

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            model(example, boundary_mask=boundary)
    finally:
        for hook in hooks:
            hook.remove()
    return flops


def format_count(value: int | float | None, scale: float, suffix: str, digits: int = 2) -> str:
    if value is None:
        return f"unknown{suffix}"
    return f"{float(value) / scale:.{digits}f}{suffix}"


def format_pre_epoch_config_table(
    cfg: dict[str, Any],
    device: torch.device,
    split_info: dict[str, Any],
    profile: dict[str, Any],
    kd_context: dict[str, Any] | None,
) -> str:
    """Return a compact startup table printed before epoch 1."""
    kd = cfg.get("dual_teacher_kd", {})
    rows: list[tuple[str, str]] = [
        ("Run", str(cfg.get("run_name"))),
        ("Model", f"{cfg.get('model')} variant={cfg.get('variant')}"),
        ("Device", str(device)),
        ("Batch size", str(cfg.get("actual_batch_size", cfg.get("batch_size")))),
        ("Epochs", str(cfg.get("epochs"))),
        ("Input", f"{cfg.get('input_mode')} channels={cfg.get('in_channels')} image={cfg.get('image_size')}"),
        ("Train/val slices", f"{split_info.get('train_slices')} / {split_info.get('val_slices')}"),
        ("Params", format_count(profile.get("params"), 1e6, "M")),
        ("GFLOPs", f"{format_count(profile.get('flops'), 1e9, '')} backend={profile.get('profile_backend')}"),
        (
            "W&B",
            _status(
                bool(cfg.get("wandb", {}).get("enabled", False)),
                f"project={cfg.get('wandb', {}).get('project')} run={cfg.get('wandb', {}).get('run_name') or cfg.get('run_name')}",
            ),
        ),
    ]
    if not bool(kd.get("enabled", False)):
        rows.append(("Dual-teacher KD", "disabled"))
        return _format_ascii_table("Pre-epoch configuration", rows)

    cache_dir = kd_context.get("cache_dir") if kd_context else None
    source = f"cache:{cache_dir}" if cache_dir else "online teachers"
    field_active = not bool(kd.get("disable_field_kd", False)) and float(kd.get("lambda_field", 0.0) or 0.0) > 0
    cine_active = not bool(kd.get("disable_cine_boundary_kd", False)) and float(kd.get("lambda_cine_boundary", 0.0) or 0.0) > 0
    fuse_active = not bool(kd.get("disable_fused_kd", False)) and float(kd.get("lambda_fuse", 0.0) or 0.0) > 0
    spec_active = not bool(kd.get("disable_spectral_kd", False)) and float(kd.get("lambda_spec", 0.0) or 0.0) > 0
    agreement_active = not bool(kd.get("disable_agreement_weighting", False))
    fused_mode = str(kd.get("fused_kd_weight_mode", "none") or "none").lower()

    rows.extend(
        [
            ("Dual-teacher KD", "enabled"),
            ("Teacher source", source),
            ("Teacher stub", str(bool(kd.get("teacher_stub", False)))),
            (
                "Teacher AMP",
                _status(
                    bool(kd.get("teacher_amp", False)),
                    f"dtype={kd.get('teacher_amp_dtype')} device={kd_context.get('teacher_device') if kd_context else kd.get('teacher_device')}",
                ),
            ),
            ("SAM2/MedSAM2 field", _semantic_teacher_status(kd, kd_context, cache_dir)),
            ("CineMA boundary/anatomy", _cinema_teacher_status(kd_context, cache_dir)),
            ("Agreement weighting", _status(agreement_active, "A=exp(-JS(P_M3,P_C))")),
            (
                "Fused KD gate",
                _status(
                    fuse_active and fused_mode == "agreement",
                    f"mode={fused_mode} min={kd.get('fused_kd_min_weight')} power={kd.get('fused_kd_agreement_power')}",
                ),
            ),
            ("Field KD", _status(field_active, f"lambda={kd.get('lambda_field')}")),
            ("Cine boundary KD", _status(cine_active, f"lambda={kd.get('lambda_cine_boundary')}")),
            ("Fused KD", _status(fuse_active, f"lambda={kd.get('lambda_fuse')}")),
            ("Spectral KD", _status(spec_active, f"lambda={kd.get('lambda_spec')}")),
            ("Temperature", str(kd.get("kd_temperature"))),
        ]
    )
    return _format_ascii_table("Pre-epoch configuration", rows)


def _status(enabled: bool, detail: str = "") -> str:
    prefix = "enabled" if enabled else "disabled"
    return f"{prefix} ({detail})" if detail else prefix


def _semantic_teacher_status(kd: dict[str, Any], kd_context: dict[str, Any] | None, cache_dir: Path | str | None) -> str:
    if not needs_medsam2_teacher(kd):
        return "disabled by KD flags"
    if cache_dir is not None:
        return "enabled via cache (P_M3/C_M3)"
    if kd_context is not None and kd_context.get("medsam2") is not None:
        return f"enabled online ({kd.get('medsam2_prompt_mode')})"
    return "missing"


def _cinema_teacher_status(kd_context: dict[str, Any] | None, cache_dir: Path | str | None) -> str:
    if cache_dir is not None:
        return "enabled via cache (P_C/C_C/B_C)"
    if kd_context is not None and kd_context.get("cinema") is not None:
        return "enabled online"
    return "missing"


def _format_ascii_table(title: str, rows: list[tuple[str, str]]) -> str:
    key_width = max([len(title), *(len(key) for key, _ in rows)])
    value_width = max(len(value) for _, value in rows)
    border = f"+-{'-' * key_width}-+-{'-' * value_width}-+"
    lines = [border, f"| {title.ljust(key_width)} | {'value'.ljust(value_width)} |", border]
    lines.extend(f"| {key.ljust(key_width)} | {value.ljust(value_width)} |" for key, value in rows)
    lines.append(border)
    return "\n".join(lines)


def foreground_dice_loss(logits: Tensor, target: Tensor, num_classes: int) -> Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - dice[1:].mean() if num_classes > 1 else 1.0 - dice.mean()


def boundary_dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    pred = torch.sigmoid(logits)
    target = target.float()
    inter = (pred * target).sum(dim=(0, 2, 3))
    denom = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - dice.mean()


def boundary_frequency_loss(logits: Tensor, target: Tensor, num_bands: int = 4) -> Tensor:
    pred = torch.sigmoid(logits).float()
    target = target.float()
    _, _, H, W = pred.shape
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    masks = build_radial_frequency_masks(H, W, num_bands, pred.device)
    high_mask = masks[max(num_bands - 2, 0):].sum(dim=0).clamp(0, 1).view(1, 1, H, W // 2 + 1)
    return F.l1_loss(pred_fft.abs() * high_mask, target_fft.abs() * high_mask)


def total_variation_loss(foreground_prob: Tensor) -> Tensor:
    dx = (foreground_prob[:, :, :, 1:] - foreground_prob[:, :, :, :-1]).abs().mean()
    dy = (foreground_prob[:, :, 1:, :] - foreground_prob[:, :, :-1, :]).abs().mean()
    return dx + dy


def compute_loss(
    outputs: dict[str, Any],
    mask: Tensor,
    boundary_target: Tensor,
    cfg: dict[str, Any],
) -> tuple[Tensor, dict[str, float]]:
    seg_logits = outputs["seg_logits"]
    boundary_logits = outputs["boundary_logits"]
    num_classes = int(cfg["num_classes"])
    weights = cfg.get("loss_weights", {})

    ce = F.cross_entropy(seg_logits, mask.long())
    dice = foreground_dice_loss(seg_logits, mask, num_classes)
    boundary_bce = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target.float())
    boundary_dice = boundary_dice_loss(boundary_logits, boundary_target)
    bfreq = boundary_frequency_loss(boundary_logits, boundary_target, int(cfg["num_bands"]))
    probs = torch.softmax(seg_logits, dim=1)
    tv = total_variation_loss(probs[:, 1:].sum(dim=1, keepdim=True))
    gate_reg = outputs["gate_reg"]
    hf_ratio_penalty = outputs["hf_ratio_penalty"]

    loss = (
        ce
        + dice
        + float(weights.get("boundary_bce", 0.50)) * boundary_bce
        + float(weights.get("boundary_dice", 0.30)) * boundary_dice
        + float(weights.get("boundary_frequency", 0.20)) * bfreq
        + float(weights.get("tv", 0.05)) * tv
        + float(weights.get("gate_reg", 0.03)) * gate_reg
        + float(weights.get("hf_ratio", 0.005)) * hf_ratio_penalty
    )
    return loss, {
        "ce": float(ce.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
        "boundary_bce": float(boundary_bce.detach().cpu()),
        "boundary_dice": float(boundary_dice.detach().cpu()),
        "boundary_frequency": float(bfreq.detach().cpu()),
        "tv": float(tv.detach().cpu()),
        "gate_reg": float(gate_reg.detach().cpu()),
        "hf_ratio_penalty": float(hf_ratio_penalty.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }


def build_teacher_kd_context(cfg: dict[str, Any], student_device: torch.device) -> dict[str, Any] | None:
    """Build optional dual-teacher KD runtime context."""
    kd = cfg.get("dual_teacher_kd", {})
    if not bool(kd.get("enabled", False)):
        return None
    teacher_device_name = str(kd.get("teacher_device") or student_device)
    if teacher_device_name == "cuda" and not torch.cuda.is_available():
        print("Teacher CUDA requested but unavailable; using CPU for teachers.")
        teacher_device_name = "cpu"
    teacher_device = torch.device(teacher_device_name)
    context: dict[str, Any] = {
        "cfg": kd,
        "teacher_device": teacher_device,
        "medsam2": None,
        "cinema": None,
    }
    cache_dir = kd.get("teacher_cache_dir")
    if cache_dir:
        context["cache_dir"] = Path(str(cache_dir))

    needs_online_teacher = not cache_dir or not bool(kd.get("strict_teacher_cache", False))
    if needs_online_teacher:
        need_m3 = needs_medsam2_teacher(kd)
        if need_m3:
            context["medsam2"] = MedSAM2Teacher(
                kd.get("medsam2_ckpt_dir"),
                device=teacher_device,
                num_classes=int(cfg["num_classes"]),
                image_size=int(cfg["image_size"]),
                repo_path=kd.get("medsam2_repo_path", "external/MedSAM2"),
                checkpoint_path=kd.get("medsam2_ckpt") or None,
                config_path=kd.get("medsam2_config", "configs/sam2.1_hiera_t512.yaml"),
                prompt_mode=kd.get("medsam2_prompt_mode", "gt_box"),
                teacher_stub=bool(kd.get("teacher_stub", False)) or bool(kd.get("medsam2_stub", False)),
            )
        context["cinema"] = CineMATeacher(
            kd.get("cinema_ckpt_dir"),
            device=teacher_device,
            num_classes=int(cfg["num_classes"]),
            image_size=int(cfg["image_size"]),
            repo_path=kd.get("cinema_repo_path", "external/CineMA"),
            checkpoint_path=kd.get("cinema_ckpt") or None,
            config_path=kd.get("cinema_config") or None,
            dataset=kd.get("cinema_dataset", "acdc"),
            view=kd.get("cinema_view", "sax"),
            seed=int(kd.get("cinema_seed", 0) or 0),
            class_map=kd.get("cinema_class_map") or None,
            teacher_stub=bool(kd.get("teacher_stub", False)) or bool(kd.get("cinema_stub", False)),
        )
        try:
            if context["medsam2"] is not None:
                context["medsam2"].load()
            context["cinema"].load()
        except TeacherLoadError as exc:
            if cache_dir and not bool(kd.get("strict_teacher_cache", False)):
                raise TeacherLoadError(
                    f"Teacher cache is enabled but online fallback could not load teachers: {exc}. "
                    "Either precompute the missing cache files, use --strict_teacher_cache to fail at the missing cache item, "
                    "or use --teacher_stub for debug."
                ) from exc
            raise

    return context


def needs_medsam2_teacher(kd: dict[str, Any]) -> bool:
    """Return whether current KD settings require MedSAM2 semantic-field outputs."""
    if bool(kd.get("use_vanilla_kd_only", False)):
        return True
    field_enabled = not bool(kd.get("disable_field_kd", False)) and float(kd.get("lambda_field", 0.0) or 0.0) > 0
    fused_enabled = not bool(kd.get("disable_fused_kd", False)) and float(kd.get("lambda_fuse", 0.0) or 0.0) > 0
    return field_enabled or fused_enabled


needs_medical_sam3_teacher = needs_medsam2_teacher


def load_or_compute_teacher_outputs(
    kd_context: dict[str, Any],
    batch: dict[str, Any],
    target_shape: tuple[int, ...],
    student_device: torch.device,
) -> dict[str, Tensor]:
    """Load teacher cache or run frozen teachers for one batch."""
    cache_dir = kd_context.get("cache_dir")
    if cache_dir is not None:
        try:
            return load_teacher_outputs_from_cache(cache_dir, batch, target_shape, student_device)
        except FileNotFoundError:
            if bool(kd_context["cfg"].get("strict_teacher_cache", False)):
                raise

    cinema = kd_context.get("cinema")
    m3 = kd_context.get("medsam2") or kd_context.get("medical_sam3")
    if cinema is None:
        raise FileNotFoundError(
            "Teacher cache item is missing and online teachers are not loaded. "
            "Use --teacher_stub for debug, remove --strict_teacher_cache, or precompute teacher outputs."
        )
    teacher_device = kd_context["teacher_device"]
    teacher_batch = {
        key: value.to(teacher_device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    amp_enabled = bool(kd_context["cfg"].get("teacher_amp", False)) and teacher_device.type == "cuda"
    amp_dtype = _resolve_teacher_amp_dtype(str(kd_context["cfg"].get("teacher_amp_dtype", "bfloat16")), teacher_device)
    with torch.no_grad(), torch.autocast(device_type=teacher_device.type, dtype=amp_dtype, enabled=amp_enabled):
        out_m3 = m3(teacher_batch) if m3 is not None else None
        out_c = cinema(teacher_batch)
    outputs = {
        "P_C": out_c["probs"].detach(),
        "C_C": out_c["confidence"].detach(),
        "B_C": out_c.get("boundary", torch.zeros_like(out_c["confidence"])).detach(),
    }
    if out_m3 is not None:
        outputs["P_M3"] = out_m3["probs"].detach()
        outputs["C_M3"] = out_m3["confidence"].detach()
    outputs = resize_teacher_output_to_student(outputs, target_shape)
    return {key: value.to(student_device, non_blocking=True).detach() for key, value in outputs.items()}


def load_teacher_outputs_from_cache(
    cache_dir: Path,
    batch: dict[str, Any],
    target_shape: tuple[int, ...],
    device: torch.device,
) -> dict[str, Tensor]:
    """Stack per-sample `.pt` teacher cache files into a batch."""
    case_ids = batch["case_id"]
    slice_indices = batch["slice_idx"]
    tensors: dict[str, list[Tensor]] = {"P_C": [], "C_C": [], "B_C": []}
    optional_tensors: dict[str, list[Tensor]] = {"P_M3": [], "C_M3": []}
    for idx, case_id in enumerate(case_ids):
        slice_idx = int(slice_indices[idx])
        payload = load_dual_teacher_cache(cache_dir, str(case_id), slice_idx, map_location="cpu")
        for key in tensors:
            if key not in payload:
                raise ValueError(f"Teacher cache for {case_id}:{slice_idx} is missing required field {key}")
            value = payload[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            tensors[key].append(value.float())
        for key in optional_tensors:
            if key in payload:
                value = payload[key]
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)
                optional_tensors[key].append(value.float())
            elif optional_tensors[key]:
                raise ValueError(f"Teacher cache has inconsistent optional field {key}; missing at {case_id}:{slice_idx}")
    for key, values in optional_tensors.items():
        if values:
            tensors[key] = values
    stacked = {key: torch.stack(values, dim=0).to(device, non_blocking=True).detach() for key, values in tensors.items()}
    stacked = resize_teacher_output_to_student(stacked, target_shape)
    return {key: value.detach() for key, value in stacked.items()}


def append_classification_metrics(
    metrics: dict[str, float],
    tp: Tensor,
    fp: Tensor,
    fn: Tensor,
    class_names: list[str],
) -> None:
    """Add Dice, Precision, and Recall per class from confusion counts."""
    tp_d = tp.detach().double().cpu()
    fp_d = fp.detach().double().cpu()
    fn_d = fn.detach().double().cpu()
    for cls, name in enumerate(class_names):
        dice_den = (2.0 * tp_d[cls] + fp_d[cls] + fn_d[cls]).item()
        precision_den = (tp_d[cls] + fp_d[cls]).item()
        recall_den = (tp_d[cls] + fn_d[cls]).item()
        metrics[f"dice_{name}"] = 1.0 if dice_den == 0.0 else float((2.0 * tp_d[cls]).item() / dice_den)
        metrics[f"precision_{name}"] = 1.0 if precision_den == 0.0 else float(tp_d[cls].item() / precision_den)
        metrics[f"recall_{name}"] = 1.0 if recall_den == 0.0 else float(tp_d[cls].item() / recall_den)

    foreground = [metrics[f"dice_{name}"] for name in class_names[1:] if f"dice_{name}" in metrics]
    metrics["fg_dice"] = float(np.mean(foreground)) if foreground else metrics.get("dice_BG", math.nan)


def format_metric(value: Any, digits: int = 4) -> str:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(scalar):
        return "-"
    return f"{scalar:.{digits}f}"


def format_class_metric_table(metrics: dict[str, float], class_names: list[str] | None = None) -> str:
    """Format the compact validation class table printed each epoch."""
    class_names = class_names or CLASS_NAMES[1:]
    rows = [("Class", "Dice", "HD95", "Precision", "Recall", "ASSD")]
    for name in class_names:
        lower = name.lower()
        rows.append(
            (
                name,
                format_metric(metrics.get(f"dice_{name}")),
                format_metric(metrics.get(f"hd95_{lower}")),
                format_metric(metrics.get(f"precision_{name}")),
                format_metric(metrics.get(f"recall_{name}")),
                format_metric(metrics.get(f"assd_{lower}")),
            )
        )
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(rows[0]))]
    lines = []
    for idx, row in enumerate(rows):
        line = "  ".join(cell.rjust(widths[col]) for col, cell in enumerate(row))
        lines.append(line)
        if idx == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def print_epoch_report(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
    print(
        f"epoch {epoch:03d} "
        f"train_loss={train_metrics.get('loss', math.nan):.4f} "
        f"val_loss={val_metrics.get('loss', math.nan):.4f} "
        f"train_fg={train_metrics.get('fg_dice', math.nan):.4f} "
        f"val_fg={val_metrics.get('fg_dice', math.nan):.4f}"
    )
    if "loss_kd" in train_metrics:
        print(
            "  kd "
            f"seg={train_metrics.get('loss_seg', math.nan):.4f} "
            f"kd={train_metrics.get('loss_kd', math.nan):.4f} "
            f"field={train_metrics.get('loss_field', math.nan):.4f} "
            f"cine_boundary={train_metrics.get('loss_cine_boundary', math.nan):.4f} "
            f"fuse={train_metrics.get('loss_fuse', math.nan):.4f} "
            f"spec={train_metrics.get('loss_spec', math.nan):.4f} "
            f"agree={train_metrics.get('agreement_mean', math.nan):.4f} "
            f"disagree={train_metrics.get('teacher_disagreement_mean', math.nan):.4f} "
            f"fuse_w={train_metrics.get('fuse_weight_mean', math.nan):.4f} "
            f"fuse_w_range=[{train_metrics.get('fuse_weight_min', math.nan):.4f},"
            f"{train_metrics.get('fuse_weight_max', math.nan):.4f}] "
            f"W_M3={train_metrics.get('W_M3_mean', math.nan):.4f} "
            f"W_C={train_metrics.get('W_C_mean', math.nan):.4f}"
        )
    print(format_class_metric_table(val_metrics))


def init_wandb(cfg: dict[str, Any], run_dir: Path) -> Any:
    wandb_cfg = cfg.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb was requested but is not installed; continuing without wandb logging.")
        return None
    return wandb.init(
        project=wandb_cfg.get("project") or "s3r-acdc",
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name") or cfg.get("run_name"),
        mode=wandb_cfg.get("mode") or "online",
        dir=str(run_dir),
        config=cfg,
    )


def flatten_metrics_for_wandb(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(scalar):
            flat[f"{prefix}/{key}"] = scalar
    return flat


def log_wandb_metrics(run: Any, epoch: int, train_metrics: dict[str, Any], val_metrics: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    if run is None:
        return
    payload: dict[str, Any] = {"epoch": epoch}
    payload.update(flatten_metrics_for_wandb("train", train_metrics))
    payload.update(flatten_metrics_for_wandb("val", val_metrics))
    if extra:
        payload.update({key: value for key, value in extra.items() if isinstance(value, (int, float)) and math.isfinite(float(value))})
    run.log(payload, step=epoch)


def finish_wandb(run: Any) -> None:
    if run is not None:
        run.finish()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
    split: str,
    log_ssr: bool,
    kd_context: dict[str, Any] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any] | None]:
    training = optimizer is not None
    model.train(training)
    num_classes = int(cfg["num_classes"])
    totals: dict[str, float] = {}
    total_samples = 0
    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)
    ssr_rows: list[dict[str, Any]] = []
    detailed_logs: dict[str, Any] | None = None
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
            return_logs = log_ssr and batch_idx == 0

            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images, boundary_mask=boundary_target, return_logs=return_logs)
            loss, parts = compute_loss(outputs, masks, boundary_target, cfg)
            parts["loss_seg"] = parts["loss"]
            if training and kd_context is not None:
                eval_every = int(kd_context["cfg"].get("teacher_eval_every", 0) or 0)
                should_eval_teacher = eval_every <= 0 or batch_idx % eval_every == 0
                if should_eval_teacher:
                    teacher_outputs = load_or_compute_teacher_outputs(kd_context, batch, tuple(outputs["seg_logits"].shape), device)
                    kd_loss, kd_parts, _ = compute_dual_teacher_kd_loss(outputs, teacher_outputs, masks, cfg)
                    loss = loss + kd_loss
                    parts.update(kd_parts)
                    parts["loss"] = float(loss.detach().cpu())
                else:
                    parts.update(
                        {
                            "loss_field": 0.0,
                            "loss_cine_boundary": 0.0,
                            "loss_fuse": 0.0,
                            "loss_spec": 0.0,
                            "loss_kd": 0.0,
                            "agreement_mean": math.nan,
                            "teacher_disagreement_mean": math.nan,
                            "fuse_weight_mean": math.nan,
                            "fuse_weight_min": math.nan,
                            "fuse_weight_max": math.nan,
                            "W_M3_mean": math.nan,
                            "W_C_mean": math.nan,
                        }
                    )
            if training:
                loss.backward()
                grad_clip = float(cfg.get("grad_clip", 0.0) or 0.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            progress.set_postfix(loss=f"{parts['loss']:.4f}")

            batch_size = int(images.shape[0])
            total_samples += batch_size
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size

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

            if return_logs:
                detailed_logs = outputs.get("logs")
                ssr_rows.extend(flatten_ssr_logs(epoch, split, detailed_logs))

    metrics = {key: value / max(total_samples, 1) for key, value in totals.items()}
    append_classification_metrics(metrics, tp, fp, fn, CLASS_NAMES[:num_classes])
    for key, vals in surface_values.items():
        metrics[key] = nanmean_or_nan(vals)
    return metrics, ssr_rows, detailed_logs


def surface_metric_keys() -> list[str]:
    return [
        "hd95_rv",
        "hd95_myo",
        "hd95_lv",
        "hd95_fg_mean",
        "assd_rv",
        "assd_myo",
        "assd_lv",
        "assd_fg_mean",
        "boundary_f1_rv",
        "boundary_f1_myo",
        "boundary_f1_lv",
        "boundary_f1_fg",
        "surface_dice_rv",
        "surface_dice_myo",
        "surface_dice_lv",
        "surface_dice_fg",
    ]


def should_compute_surface_metrics(cfg: dict[str, Any], epoch: int) -> bool:
    metrics_cfg = cfg.get("metrics", {})
    enabled = any(
        bool(metrics_cfg.get(key, True))
        for key in ("compute_hd95", "compute_assd", "compute_boundary_f1", "compute_surface_dice")
    )
    if not enabled:
        return False
    if not HAS_SCIPY:
        raise ImportError("Validation surface metrics require scipy.ndimage. Install scipy or disable metrics in config.")
    every = int(metrics_cfg.get("full_every_n_epochs", 1) or 1)
    return epoch == 1 or epoch % every == 0 or epoch == int(cfg.get("epochs", epoch))


def nanmean_or_nan(values: list[float]) -> float:
    if not values:
        return math.nan
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.isnan(arr).all():
        return math.nan
    return float(np.nanmean(arr))


def finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def flatten_ssr_logs(epoch: int, split: str, logs: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not logs:
        return rows

    transitions = logs.get("transitions", logs)
    for block, block_logs in transitions.items():
        if not isinstance(block_logs, dict):
            continue
        if "ssr" in block_logs or "state" in block_logs:
            groups = {
                "ssr": block_logs.get("ssr", {}),
                "state": block_logs.get("state", {}),
            }
        else:
            groups = {"ssr": block_logs}
        for group, group_logs in groups.items():
            if not isinstance(group_logs, dict):
                continue
            for metric, value in group_logs.items():
                metric_name = f"{group}_{metric}" if group != "ssr" else metric
                if isinstance(value, list):
                    for band, band_value in enumerate(value):
                        rows.append({"epoch": epoch, "split": split, "block": block, "metric": metric_name, "band": band, "value": float(band_value)})
                elif isinstance(value, (int, float)):
                    rows.append({"epoch": epoch, "split": split, "block": block, "metric": metric_name, "band": "", "value": float(value)})
    return rows


def print_detailed_logs(logs: dict[str, Any] | None, prefix: str) -> None:
    if not logs:
        return
    metrics = [
        "retain_gate_mean",
        "suppress_gate_mean",
        "update_gate_mean",
        "input_energy",
        "output_energy",
        "phase_coherence",
        "retain_contribution",
        "update_contribution",
        "suppress_contribution",
        "high_freq_ratio",
        "high_freq_penalty",
        "boundary_to_nonboundary_high_ratio",
        "gamma",
        "residual_gate_mean",
        "residual_gate_std",
    ]
    transitions = logs.get("transitions", logs)
    for block, block_logs in transitions.items():
        print(f"  {prefix}/{block}")
        if isinstance(block_logs, dict) and "ssr" in block_logs:
            block_logs = block_logs["ssr"]
        for metric in metrics:
            if isinstance(block_logs, dict) and metric in block_logs:
                print(f"    {metric}: {block_logs[metric]}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(run_dir: Path, rows: list[dict[str, Any]], ssr_rows: list[dict[str, Any]]) -> None:
    prepare_plot_cache(run_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return

    if rows:
        epochs = [row["epoch"] for row in rows]
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [row["train_loss"] for row in rows], label="train")
        plt.plot(epochs, [row["val_loss"] for row in rows], label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "loss_curve.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [row["train_fg_mean"] for row in rows], label="train fg mean")
        plt.plot(epochs, [row["val_fg_mean"] for row in rows], label="val fg mean")
        plt.xlabel("epoch")
        plt.ylabel("foreground Dice")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "val_dice_curve.png", dpi=160)
        plt.close()

        kd_keys = ["train_loss_field", "train_loss_cine_boundary", "train_loss_fuse", "train_loss_spec", "train_loss_kd"]
        if any(key in rows[0] for key in kd_keys):
            plt.figure(figsize=(8, 4.5))
            for key in kd_keys:
                if key in rows[0]:
                    plt.plot(epochs, [row.get(key, math.nan) for row in rows], label=key.replace("train_", ""))
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(run_dir / "kd_loss_components.png", dpi=160)
            plt.close()

        kd_weight_keys = [
            "train_agreement_mean",
            "train_teacher_disagreement_mean",
            "train_fuse_weight_mean",
            "train_fuse_weight_min",
            "train_fuse_weight_max",
            "train_W_M3_mean",
            "train_W_C_mean",
        ]
        if any(key in rows[0] for key in kd_weight_keys):
            plt.figure(figsize=(8, 4.5))
            for key in kd_weight_keys:
                if key in rows[0]:
                    plt.plot(epochs, [row.get(key, math.nan) for row in rows], label=key.replace("train_", ""))
            plt.xlabel("epoch")
            plt.ylim(0, 1)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(run_dir / "kd_agreement_weights.png", dpi=160)
            plt.close()

    _plot_ssr_metrics(run_dir, ssr_rows)


def _plot_ssr_metrics(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    def plot_metric_set(metrics: list[str], filename: str, title: str) -> None:
        selected = [row for row in rows if row["split"] == "val" and row["metric"] in metrics]
        if not selected:
            return
        plt.figure(figsize=(9, 5))
        for metric in metrics:
            for block in sorted({row["block"] for row in selected}):
                bands = sorted({row["band"] for row in selected if row["metric"] == metric and row["block"] == block}, key=lambda x: -1 if x == "" else int(x))
                for band in bands:
                    series = [row for row in selected if row["metric"] == metric and row["block"] == block and row["band"] == band]
                    series.sort(key=lambda row: row["epoch"])
                    label = f"{block}:{metric}:b{band}" if band != "" else f"{block}:{metric}"
                    plt.plot([row["epoch"] for row in series], [row["value"] for row in series], marker="o", linewidth=1.4, label=label)
        plt.title(title)
        plt.xlabel("epoch")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(run_dir / filename, dpi=160)
        plt.close()

    plot_metric_set(["retain_gate_mean", "update_gate_mean", "suppress_gate_mean"], "gate_curves.png", "SSR gate means")
    plot_metric_set(["retain_contribution", "update_contribution", "suppress_contribution"], "contribution_curves.png", "SSR contributions")
    plot_metric_set(["high_freq_ratio"], "high_freq_ratio_curves.png", "High-frequency ratio")
    plot_metric_set(["high_freq_penalty"], "high_freq_penalty_curves.png", "High-frequency penalty")
    plot_metric_set(["boundary_to_nonboundary_high_ratio"], "boundary_ratio_curves.png", "Boundary/non-boundary high ratio")
    plot_metric_set(["gamma"], "gamma_curves.png", "Effective gamma")
    plot_metric_set(["residual_gate_mean"], "residual_gate_mean_curves.png", "Residual gate mean")


def save_prediction_grid(model: nn.Module, loader: DataLoader, device: torch.device, path: Path) -> None:
    prepare_plot_cache(path.parent)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()
    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    boundary_target = boundary_map_from_mask(masks).to(device)
    with torch.no_grad():
        outputs = model(images, boundary_mask=boundary_target)
        preds = outputs["seg_logits"].argmax(dim=1)
        boundary_prob = torch.sigmoid(outputs["boundary_logits"])
        errors = (preds != masks).float()

    n = min(4, images.shape[0])
    fig, axes = plt.subplots(n, 5, figsize=(13, 2.6 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(n):
        img = images[i, images.shape[1] // 2].detach().cpu().numpy()
        panels = [
            (img, "image", "gray", None, None),
            (masks[i].detach().cpu().numpy(), "mask", "viridis", 0, 3),
            (preds[i].detach().cpu().numpy(), "prediction", "viridis", 0, 3),
            (boundary_prob[i, 0].detach().cpu().numpy(), "boundary prob", "magma", 0, 1),
            (errors[i].detach().cpu().numpy(), "error", "Reds", 0, 1),
        ]
        for ax, (data, title, cmap, vmin, vmax) in zip(axes[i], panels):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def prepare_plot_cache(run_dir: Path) -> None:
    cache_root = run_dir / ".plot_cache"
    mpl_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache.resolve()))


def is_cuda_oom(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def train_once(cfg: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(cfg["seed"]))
    device = resolve_device(str(cfg["device"]))
    if device.type == "cpu" and int(cfg.get("num_workers", 0)) > 0:
        print("CPU run detected; using num_workers=0 to avoid local DataLoader worker shared-memory failures.")
        cfg["num_workers"] = 0

    run_dir = Path(cfg["save_dir"]) if cfg.get("save_dir") else Path(cfg["output_root"]) / str(cfg["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["actual_batch_size"] = int(cfg["batch_size"])

    train_loader, val_loader, split_info = make_loaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    profile = estimate_model_profile(model, cfg, device)
    model_params = int(profile["params"])
    wandb_run = init_wandb(cfg, run_dir)
    teacher_kd_context = build_teacher_kd_context(cfg, device)

    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    training_rows: list[dict[str, Any]] = []
    ssr_rows: list[dict[str, Any]] = []
    best_val = -1.0
    best_hd95 = float("inf")
    best_hd95_epoch = 0
    best_epoch = 0
    best_row: dict[str, Any] | None = None

    startup_table = format_pre_epoch_config_table(cfg, device, split_info, profile, teacher_kd_context)
    print(startup_table)
    with open(run_dir / "startup_config_table.txt", "w", encoding="utf-8") as f:
        f.write(startup_table)
        f.write("\n")

    for epoch in range(1, int(cfg["epochs"]) + 1):
        log_ssr = bool(cfg.get("return_logs", False)) and (
            epoch == 1 or epoch % 5 == 0 or epoch == int(cfg["epochs"])
        )
        train_metrics, train_ssr_rows, train_logs = run_epoch(model, train_loader, optimizer, device, cfg, epoch, "train", log_ssr, teacher_kd_context)
        val_metrics, val_ssr_rows, val_logs = run_epoch(model, val_loader, None, device, cfg, epoch, "val", log_ssr)
        ssr_rows.extend(train_ssr_rows)
        ssr_rows.extend(val_ssr_rows)

        row = {
            "epoch": epoch,
            "actual_batch_size": int(cfg["actual_batch_size"]),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_fg_mean": train_metrics["fg_dice"],
            "val_fg_mean": val_metrics["fg_dice"],
            "train_fg_dice": train_metrics["fg_dice"],
            "val_fg_dice": val_metrics["fg_dice"],
            "train_boundary_bce": train_metrics["boundary_bce"],
            "train_boundary_dice": train_metrics["boundary_dice"],
            "train_bfreq": train_metrics["boundary_frequency"],
            "train_tv": train_metrics["tv"],
            "train_gate_reg": train_metrics["gate_reg"],
            "train_hf_ratio_penalty": train_metrics["hf_ratio_penalty"],
            "train_loss_seg": train_metrics.get("loss_seg", math.nan),
            "train_loss_field": train_metrics.get("loss_field", math.nan),
            "train_loss_cine_boundary": train_metrics.get("loss_cine_boundary", math.nan),
            "train_loss_fuse": train_metrics.get("loss_fuse", math.nan),
            "train_loss_spec": train_metrics.get("loss_spec", math.nan),
            "train_loss_kd": train_metrics.get("loss_kd", math.nan),
            "train_agreement_mean": train_metrics.get("agreement_mean", math.nan),
            "train_teacher_disagreement_mean": train_metrics.get("teacher_disagreement_mean", math.nan),
            "train_fuse_weight_mean": train_metrics.get("fuse_weight_mean", math.nan),
            "train_fuse_weight_min": train_metrics.get("fuse_weight_min", math.nan),
            "train_fuse_weight_max": train_metrics.get("fuse_weight_max", math.nan),
            "train_W_M3_mean": train_metrics.get("W_M3_mean", math.nan),
            "train_W_C_mean": train_metrics.get("W_C_mean", math.nan),
            "val_boundary_bce": val_metrics["boundary_bce"],
            "val_boundary_dice": val_metrics["boundary_dice"],
            "val_bfreq": val_metrics["boundary_frequency"],
            "val_tv": val_metrics["tv"],
            "val_gate_reg": val_metrics["gate_reg"],
            "val_hf_ratio_penalty": val_metrics["hf_ratio_penalty"],
            "val_hd95_fg_mean": val_metrics.get("hd95_fg_mean", math.nan),
            "val_assd_fg_mean": val_metrics.get("assd_fg_mean", math.nan),
            "val_boundary_f1_fg": val_metrics.get("boundary_f1_fg", math.nan),
            "val_surface_dice_fg": val_metrics.get("surface_dice_fg", math.nan),
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
            best_row = dict(row)
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_fg_mean": best_val, "config": cfg}, run_dir / "best_model.pt")
        hd95_value = float(row.get("val_hd95_fg_mean", math.nan))
        if math.isfinite(hd95_value) and hd95_value < best_hd95:
            best_hd95 = hd95_value
            best_hd95_epoch = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_hd95_fg_mean": best_hd95, "config": cfg}, run_dir / "best_hd95_model.pt")

        write_csv(run_dir / "training_log.csv", training_rows)
        write_csv(run_dir / "ssr_logs.csv", ssr_rows, ["epoch", "split", "block", "metric", "band", "value"])

        print_epoch_report(epoch, train_metrics, val_metrics)
        log_wandb_metrics(wandb_run, epoch, train_metrics, val_metrics, {"lr": optimizer.param_groups[0]["lr"]})

    torch.save({"epoch": int(cfg["epochs"]), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "config": cfg}, run_dir / "final_model.pt")
    plot_training_curves(run_dir, training_rows, ssr_rows)
    save_prediction_grid(model, train_loader, device, run_dir / "train_predictions.png")
    save_prediction_grid(model, val_loader, device, run_dir / "val_predictions.png")

    final_row = training_rows[-1]
    summary = {
        "run_name": cfg["run_name"],
        "variant": cfg.get("variant", "default"),
        "seed": int(cfg["seed"]),
        "best_epoch": best_epoch,
        "best_val_fg_mean": best_val,
        "best_val_fg_dice": best_val,
        "best_val_dice_RV": (best_row or {}).get("val_dice_RV", 0.0),
        "best_val_dice_MYO": (best_row or {}).get("val_dice_MYO", 0.0),
        "best_val_dice_LV": (best_row or {}).get("val_dice_LV", 0.0),
        "best_val_hd95_fg_mean": finite_or_none((best_row or {}).get("val_hd95_fg_mean")),
        "best_val_assd_fg_mean": finite_or_none((best_row or {}).get("val_assd_fg_mean")),
        "best_val_boundary_f1_fg": finite_or_none((best_row or {}).get("val_boundary_f1_fg")),
        "best_val_surface_dice_fg": finite_or_none((best_row or {}).get("val_surface_dice_fg")),
        "best_val_hd95_best": None if not math.isfinite(best_hd95) else best_hd95,
        "best_val_hd95_epoch": best_hd95_epoch,
        "final_val_fg_mean": final_row["val_fg_mean"],
        "final_val_hd95_fg_mean": finite_or_none(final_row.get("val_hd95_fg_mean")),
        "final_val_assd_fg_mean": finite_or_none(final_row.get("val_assd_fg_mean")),
        "final_val_boundary_f1_fg": finite_or_none(final_row.get("val_boundary_f1_fg")),
        "final_val_surface_dice_fg": finite_or_none(final_row.get("val_surface_dice_fg")),
        "actual_batch_size": int(cfg["actual_batch_size"]),
        "input_mode": cfg.get("input_mode"),
        "in_channels": int(cfg.get("in_channels") or 0),
        "image_size": int(cfg["image_size"]),
        "epochs": int(cfg["epochs"]),
        "model_parameter_count": model_params,
        "model_parameter_millions": model_params / 1e6,
        "model_flops": profile.get("flops"),
        "model_gflops": profile.get("gflops"),
        "model_profile_backend": profile.get("profile_backend"),
        "train_slices": split_info["train_slices"],
        "val_slices": split_info["val_slices"],
        "train_cases": len(split_info["train_cases"]),
        "val_cases": len(split_info["val_cases"]),
        "config_flags": {
            "residual_gate_type": cfg["ssr"].get("residual_gate_type"),
            "residual_gate_max": cfg["ssr"].get("residual_gate_max"),
            "geometry_refine": cfg["ssr"].get("geometry_refine"),
            "use_bounded_gamma": cfg["ssr"].get("use_bounded_gamma"),
            "gamma_max": cfg["ssr"].get("gamma_max"),
            "suppress_min": cfg["ssr"].get("suppress_min"),
            "suppress_max": cfg["ssr"].get("suppress_max"),
            "hf_ratio_threshold": cfg["ssr"].get("hf_ratio_threshold"),
            "loss_weights": cfg.get("loss_weights", {}),
            "metrics": cfg.get("metrics", {}),
            "dual_teacher_kd": cfg.get("dual_teacher_kd", {}),
        },
        "artifacts": [
            "startup_config_table.txt",
            "training_log.csv",
            "ssr_logs.csv",
            "loss_curve.png",
            "val_dice_curve.png",
            "kd_loss_components.png",
            "kd_agreement_weights.png",
            "gate_curves.png",
            "contribution_curves.png",
            "high_freq_ratio_curves.png",
            "high_freq_penalty_curves.png",
            "boundary_ratio_curves.png",
            "gamma_curves.png",
            "train_predictions.png",
            "val_predictions.png",
            "best_model.pt",
            "best_hd95_model.pt",
            "final_model.pt",
            "config_resolved.yaml",
        ],
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Finished S3R run. Best val fg mean={best_val:.4f} at epoch {best_epoch}.")
    print(f"Artifacts saved under {run_dir}")
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
    cfg = apply_variant_and_cli(load_config(args.config), args)
    train_with_oom_recovery(cfg)


if __name__ == "__main__":
    main()
