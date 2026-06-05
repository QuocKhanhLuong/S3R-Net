#!/usr/bin/env python3
"""Precompute Medical-SAM3 and CineMA teacher outputs for ACDC slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.acdc_s3r_dataset import ACDCSSRSliceDataset
from teachers import CineMATeacher, MedicalSAM3Teacher
from teachers.teacher_utils import save_dual_teacher_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute dual-teacher outputs")
    parser.add_argument("--teacher", choices=["medical_sam3", "cinema", "both"], default="both")
    parser.add_argument("--data_dir", "--data_root", dest="data_dir", default="preprocessed_data/ACDC/training")
    parser.add_argument("--output_dir", default="teacher_cache/acdc")
    parser.add_argument("--medical_sam3_repo_path", default="external/Medical-SAM3")
    parser.add_argument("--medical_sam3_ckpt_dir", default="checkpoints/teachers/medical_sam3")
    parser.add_argument("--medical_sam3_prompt_mode", default="gt_box")
    parser.add_argument("--cinema_repo_path", default="external/CineMA")
    parser.add_argument("--cinema_ckpt_dir", default="checkpoints/teachers/cinema")
    parser.add_argument("--cinema_ckpt", default="")
    parser.add_argument("--cinema_config", default="")
    parser.add_argument("--cinema_dataset", default="acdc")
    parser.add_argument("--cinema_view", default="sax")
    parser.add_argument("--cinema_seed", type=int, default=0)
    parser.add_argument("--cinema_class_map", default="")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--input_mode", choices=["2d", "25d"], default="2d")
    parser.add_argument("--max_slices", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher_stub", action="store_true")
    parser.add_argument("--medical_sam3_stub", action="store_true")
    parser.add_argument("--cinema_stub", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = ACDCSSRSliceDataset(
        args.data_dir,
        input_mode=args.input_mode,
        image_size=args.image_size,
        foreground_only=False,
        max_slices=args.max_slices,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    m3 = None
    cinema = None
    if args.teacher in {"medical_sam3", "both"}:
        m3 = MedicalSAM3Teacher(
            args.medical_sam3_ckpt_dir,
            device=device,
            num_classes=args.num_classes,
            image_size=args.image_size,
            repo_path=args.medical_sam3_repo_path,
            prompt_mode=args.medical_sam3_prompt_mode,
            teacher_stub=args.teacher_stub or args.medical_sam3_stub,
        )
        m3.load()
    if args.teacher in {"cinema", "both"}:
        cinema = CineMATeacher(
            args.cinema_ckpt_dir,
            device=device,
            num_classes=args.num_classes,
            image_size=args.image_size,
            repo_path=args.cinema_repo_path,
            checkpoint_path=args.cinema_ckpt or None,
            config_path=args.cinema_config or None,
            dataset=args.cinema_dataset,
            view=args.cinema_view,
            seed=args.cinema_seed,
            class_map=args.cinema_class_map or None,
            teacher_stub=args.teacher_stub or args.cinema_stub,
        )
        cinema.load()

    output_dir = Path(args.output_dir)
    saved = 0
    for batch in tqdm(loader, desc="precompute teachers"):
        batch = _move_batch(batch, device)
        out_m3 = m3(batch) if m3 is not None else None
        out_c = cinema(batch) if cinema is not None else None
        case_id = str(batch["case_id"][0])
        slice_idx = int(batch["slice_idx"][0])
        payload: dict[str, Any] = {
            "metadata": {
                "case_id": case_id,
                "slice_idx": slice_idx,
                "shape": list(batch["image"].shape[-2:]),
                "class_order": ["BG", "RV", "MYO", "LV"][: args.num_classes],
                "medical_sam3_prompt_mode": args.medical_sam3_prompt_mode,
                "teacher_stub": bool(args.teacher_stub),
                "teacher": args.teacher,
            },
        }
        if out_m3 is not None:
            payload["P_M3"] = out_m3["probs"][0].detach().cpu()
            payload["C_M3"] = out_m3["confidence"][0].detach().cpu()
            payload["metadata"]["shape"] = list(out_m3["probs"].shape[-2:])
        if out_c is not None:
            payload["P_C"] = out_c["probs"][0].detach().cpu()
            payload["C_C"] = out_c["confidence"][0].detach().cpu()
            payload["B_C"] = out_c.get("boundary", torch.zeros_like(out_c["confidence"]))[0].detach().cpu()
            payload["metadata"]["shape"] = list(out_c["probs"].shape[-2:])
        save_dual_teacher_cache(output_dir, case_id, slice_idx, payload)
        saved += 1

    manifest = {
        "data_dir": args.data_dir,
        "output_dir": str(output_dir),
        "num_items": saved,
        "teacher_stub": bool(args.teacher_stub),
        "num_classes": args.num_classes,
        "image_size": args.image_size,
        "input_mode": args.input_mode,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Saved {saved} teacher cache items under {output_dir}")


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


if __name__ == "__main__":
    main()
