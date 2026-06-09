#!/usr/bin/env python3
"""Precompute MedSAM2 and CineMA teacher outputs for ACDC slices."""

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
from teachers import CineMATeacher, MedSAM2Teacher
from teachers.teacher_utils import save_dual_teacher_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute dual-teacher outputs")
    parser.add_argument("--teacher", choices=["medsam2", "medical_sam3", "cinema", "both"], default="both")
    parser.add_argument("--data_dir", "--data_root", dest="data_dir", default="preprocessed_data/ACDC/training")
    parser.add_argument("--output_dir", default="teacher_cache/acdc")
    parser.add_argument("--medsam2_repo_path", default="external/MedSAM2")
    parser.add_argument("--medsam2_ckpt_dir", default="checkpoints/teachers/medsam2")
    parser.add_argument("--medsam2_ckpt", default="")
    parser.add_argument("--medsam2_config", default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--medsam2_prompt_mode", default="gt_box")
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
    parser.add_argument("--medsam2_stub", action="store_true")
    parser.add_argument("--cinema_stub", action="store_true")
    parser.add_argument("--teacher_amp", action="store_true", help="Enable CUDA autocast around teacher inference.")
    parser.add_argument(
        "--teacher_amp_dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="CUDA autocast dtype for teacher inference when --teacher_amp is enabled.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--medical_sam3_repo_path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--medical_sam3_ckpt_dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--medical_sam3_prompt_mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--medical_sam3_stub", action="store_true", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize_medsam2_args(args)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.teacher_amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(args.teacher_amp_dtype, device)
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
    if args.teacher in {"medsam2", "both"}:
        m3 = MedSAM2Teacher(
            args.medsam2_ckpt_dir,
            device=device,
            num_classes=args.num_classes,
            image_size=args.image_size,
            repo_path=args.medsam2_repo_path,
            checkpoint_path=args.medsam2_ckpt or None,
            config_path=args.medsam2_config,
            prompt_mode=args.medsam2_prompt_mode,
            teacher_stub=args.teacher_stub or args.medsam2_stub,
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
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
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
                "medsam2_prompt_mode": args.medsam2_prompt_mode,
                "teacher_stub": bool(args.teacher_stub),
                "teacher": args.teacher,
            },
        }
        if out_m3 is not None:
            payload["P_M3"] = out_m3["probs"][0].detach().float().cpu()
            payload["C_M3"] = out_m3["confidence"][0].detach().float().cpu()
            payload["metadata"]["shape"] = list(out_m3["probs"].shape[-2:])
        if out_c is not None:
            payload["P_C"] = out_c["probs"][0].detach().float().cpu()
            payload["C_C"] = out_c["confidence"][0].detach().float().cpu()
            payload["B_C"] = out_c.get("boundary", torch.zeros_like(out_c["confidence"]))[0].detach().float().cpu()
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


def normalize_medsam2_args(args: argparse.Namespace) -> None:
    if args.teacher == "medical_sam3":
        print("--teacher medical_sam3 is deprecated; using medsam2.")
        args.teacher = "medsam2"
    if args.medical_sam3_repo_path:
        args.medsam2_repo_path = args.medical_sam3_repo_path
    if args.medical_sam3_ckpt_dir:
        args.medsam2_ckpt_dir = args.medical_sam3_ckpt_dir
    if args.medical_sam3_prompt_mode:
        args.medsam2_prompt_mode = args.medical_sam3_prompt_mode
    if args.medical_sam3_stub is not None:
        args.medsam2_stub = bool(args.medical_sam3_stub)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _resolve_amp_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        if device.type == "cuda" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            print("Requested bfloat16 teacher AMP but this CUDA device does not support BF16; using float16.")
            return torch.float16
        return torch.bfloat16
    raise ValueError(f"Unsupported teacher AMP dtype: {name}")


if __name__ == "__main__":
    main()
