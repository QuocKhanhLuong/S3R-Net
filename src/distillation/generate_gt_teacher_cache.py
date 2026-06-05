#!/usr/bin/env python3
"""Generate procedural GT teacher cache for S3R-SCSD smoke tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
for path in (ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.acdc_s3r_dataset import ACDCSSRSliceDataset, load_or_create_split
from distillation.teacher_cache import save_teacher_cache
from distillation.teacher_targets import (
    boundary_from_mask,
    distance_map_from_mask_or_boundary,
    semantic_entropy,
    teacher_agreement_weight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GT-derived teacher cache for S3R-SCSD smoke tests")
    parser.add_argument("--data_root", default="preprocessed_data/ACDC")
    parser.add_argument("--output_dir", default="teacher_cache/acdc_gt_debug")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--input_mode", choices=["2d", "25d"], default="2d")
    parser.add_argument("--max_slices", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_manifest", default="splits/acdc_patient_split_seed42.json")
    parser.add_argument("--dataset", default="ACDC")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--smoothing", type=float, default=0.03)
    return parser.parse_args()


def make_datasets(args: argparse.Namespace) -> list[tuple[str, ACDCSSRSliceDataset]]:
    train_cases, val_cases, _ = load_or_create_split(
        args.data_root,
        seed=args.seed,
        train_fraction=0.8,
        split_manifest=args.split_manifest,
    )
    common = {
        "data_root": args.data_root,
        "input_mode": args.input_mode,
        "image_size": args.image_size,
        "foreground_only": True,
        "max_slices": args.max_slices,
    }
    return [
        ("train", ACDCSSRSliceDataset(case_ids=train_cases, seed=args.seed, **common)),
        ("val", ACDCSSRSliceDataset(case_ids=val_cases, seed=args.seed + 1, **common)),
    ]


def build_payload(mask: torch.Tensor, num_classes: int, smoothing: float) -> dict[str, torch.Tensor]:
    one_hot = F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()
    probs = one_hot * (1.0 - smoothing) + smoothing / float(num_classes)
    entropy = semantic_entropy(probs)
    foreground = probs[:, 1:].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    boundary = boundary_from_mask(mask)
    distance = distance_map_from_mask_or_boundary(boundary)
    agreement = teacher_agreement_weight(foreground, (mask > 0).float().unsqueeze(1))
    return {
        "t1_probs": probs,
        "t1_entropy": entropy,
        "t1_foreground": foreground,
        "t2_boundary": boundary,
        "t2_distance": distance,
        "t2_foreground": (mask > 0).float().unsqueeze(1),
        "agreement_weight": agreement,
    }


def save_batch(
    batch: dict[str, Any],
    args: argparse.Namespace,
    split: str,
    num_classes: int = 4,
) -> int:
    mask = batch["mask"]
    payload = build_payload(mask, num_classes, float(args.smoothing))
    count = 0
    for i, case_id in enumerate(batch["case_id"]):
        slice_idx = int(batch["slice_idx"][i])
        sample_payload = {key: value[i].detach().cpu().numpy().astype(np.float32) for key, value in payload.items()}
        save_teacher_cache(
            args.output_dir,
            str(case_id),
            slice_idx,
            sample_payload,
            dataset=args.dataset,
            metadata={
                "source": "gt_debug",
                "split": split,
                "image_size": int(args.image_size),
                "input_mode": args.input_mode,
            },
        )
        count += 1
    return count


def main() -> None:
    args = parse_args()
    total = 0
    for split, dataset in make_datasets(args):
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        for batch in loader:
            total += save_batch(batch, args, split)
    print(f"Generated {total} GT teacher-cache files under {Path(args.output_dir) / args.dataset}")


if __name__ == "__main__":
    main()
