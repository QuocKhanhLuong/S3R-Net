#!/usr/bin/env python3
"""Load teacher wrappers and save a dual-teacher KD preview grid."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.acdc_s3r_dataset import ACDCSSRSliceDataset
from losses.agreement_kd import agreement_aware_fusion
from teachers import CineMATeacher, MedSAM2Teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test teacher wrappers on one ACDC batch")
    parser.add_argument("--teacher", choices=["medsam2", "medical_sam3", "cinema", "both"], default="both")
    parser.add_argument("--data_dir", "--data_root", dest="data_dir", default="preprocessed_data/ACDC/training")
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
    parser.add_argument("--teacher_stub", action="store_true")
    parser.add_argument("--medsam2_stub", action="store_true")
    parser.add_argument("--cinema_stub", action="store_true")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--input_mode", choices=["2d", "25d"], default="2d")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="debug_outputs/teacher_kd_preview.png")
    parser.add_argument("--skip_preview", action="store_true")
    parser.add_argument("--medical_sam3_repo_path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--medical_sam3_ckpt_dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--medical_sam3_prompt_mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--medical_sam3_stub", action="store_true", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize_medsam2_args(args)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = ACDCSSRSliceDataset(
        args.data_dir,
        input_mode=args.input_mode,
        image_size=args.image_size,
        foreground_only=True,
        max_slices=1,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    batch = _move_batch(batch, device)
    out_m3 = None
    out_c = None
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
        out_m3 = m3(batch)
        print("MedSAM2 probs:", tuple(out_m3["probs"].shape))
        print("MedSAM2 confidence:", tuple(out_m3["confidence"].shape))
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
        out_c = cinema(batch)
        print("CineMA probs:", tuple(out_c["probs"].shape))
        print("CineMA boundary:", tuple(out_c.get("boundary", torch.empty(0)).shape))
    if out_m3 is None and out_c is None:
        raise RuntimeError("No teacher output available for preview.")
    if out_m3 is not None and out_c is not None:
        fusion = agreement_aware_fusion(
            out_m3["probs"],
            out_c["probs"],
            out_m3["confidence"],
            out_c["confidence"],
            gt_mask=batch["mask"],
            cinema_boundary=out_c.get("boundary"),
        )
        print("Fused target:", tuple(fusion["P_F"].shape))
        print("Agreement:", tuple(fusion["agreement"].shape))
    else:
        fusion = None
    if args.skip_preview:
        print("Skipped preview: --skip_preview")
    elif save_preview(batch, out_m3, out_c, fusion, Path(args.output)):
        print(f"Saved preview: {args.output}")


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
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def save_preview(
    batch: dict[str, Any],
    out_m3: dict[str, Any] | None,
    out_c: dict[str, Any] | None,
    fusion: dict[str, torch.Tensor] | None,
    path: Path,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping teacher preview.")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    image = batch["image"][0, batch["image"].shape[1] // 2].detach().cpu()
    gt = batch["mask"][0].detach().cpu()
    panels = [
        (image, "image", "gray", None, None),
        (gt, "GT", "viridis", 0, 3),
    ]
    if out_m3 is not None:
        panels.append((out_m3["probs"][0].argmax(dim=0).detach().cpu(), "MedSAM2", "viridis", 0, 3))
    if out_c is not None:
        panels.extend(
            [
                (out_c["probs"][0].argmax(dim=0).detach().cpu(), "CineMA", "viridis", 0, 3),
                (out_c["boundary"][0, 0].detach().cpu(), "CineMA boundary", "magma", 0, 1),
            ]
        )
    if fusion is not None:
        panels.extend(
            [
                (fusion["P_F"][0].argmax(dim=0).detach().cpu(), "fused", "viridis", 0, 3),
                (fusion["agreement"][0, 0].detach().cpu(), "agreement", "magma", 0, 1),
            ]
        )
    fig, axes = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 2.6))
    for ax, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return True


if __name__ == "__main__":
    main()
