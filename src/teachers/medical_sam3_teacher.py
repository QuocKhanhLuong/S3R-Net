"""Medical-SAM3 frozen teacher adapter."""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .base_teacher import FrozenSegmentationTeacher, TeacherLoadError
from .teacher_utils import list_checkpoint_candidates, make_teacher_stub


class MedicalSAM3Teacher(FrozenSegmentationTeacher):
    """Prompt-driven Medical-SAM3 teacher wrapper.

    The public Medical-SAM3 API may change, so this first adapter provides
    robust dependency/checkpoint validation and a deterministic stub path.
    Real forward binding should be added after the external repo is cloned and
    its inference API is pinned.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path | None,
        device: str | torch.device,
        num_classes: int,
        image_size: int | None = None,
        repo_path: str | Path = "external/Medical-SAM3",
        prompt_mode: str = "gt_box",
        teacher_stub: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(checkpoint_dir, device, num_classes, image_size, teacher_stub, **kwargs)
        self.repo_path = Path(repo_path)
        self.prompt_mode = str(prompt_mode)
        self.checkpoint_candidates: list[Path] = []

    def load(self) -> "MedicalSAM3Teacher":
        if self.teacher_stub:
            self.loaded = True
            return self
        if not self.repo_path.exists():
            raise TeacherLoadError(
                f"Medical-SAM3 repo not found at {self.repo_path}. "
                "Run: bash scripts/setup_teachers.sh"
            )
        if self.checkpoint_dir is None or not self.checkpoint_dir.exists():
            raise TeacherLoadError(
                f"Medical-SAM3 checkpoint directory not found: {self.checkpoint_dir}. "
                "Run: python scripts/download_teachers.py --output_dir checkpoints/teachers"
            )
        self.checkpoint_candidates = list_checkpoint_candidates(self.checkpoint_dir)
        if not self.checkpoint_candidates:
            raise TeacherLoadError(
                f"No Medical-SAM3 checkpoint candidates found under {self.checkpoint_dir}. "
                "Expected one of .pt/.pth/.ckpt/.safetensors/.bin. If the Hugging Face repo is gated, run: huggingface-cli login"
            )
        sys.path.insert(0, str(self.repo_path.resolve()))
        import_errors = []
        for module_name in ("medical_sam3", "medsam3", "sam3", "segment_anything"):
            try:
                importlib.import_module(module_name)
                self.loaded = True
                break
            except Exception as exc:  # pragma: no cover - depends on external repo
                import_errors.append(f"{module_name}: {exc}")
        else:
            raise TeacherLoadError(
                "Could not import Medical-SAM3 modules from external/Medical-SAM3. "
                "Install the repo dependencies, then retry. Import attempts: "
                + " | ".join(import_errors[:4])
            )
        raise TeacherLoadError(
            "Medical-SAM3 dependency import succeeded, but this adapter does not yet bind a pinned upstream inference API. "
            "Use --teacher_cache_dir with precomputed outputs or --teacher_stub for pipeline testing."
        )

    @torch.no_grad()
    def predict(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.teacher_stub:
            return make_teacher_stub(batch, self.num_classes, mode="medical_sam3")
        raise TeacherLoadError("Medical-SAM3 real predict is not bound. Use cache/stub or implement the pinned repo API adapter.")

    def boxes_from_gt(self, mask: Tensor) -> Tensor:
        """Return per-class boxes `[B,C,4]` in xyxy order from GT labels."""
        return _boxes_from_labels(mask.long(), self.num_classes)

    def boxes_from_cinema(self, cinema_probs: Tensor) -> Tensor:
        """Return per-class boxes from CineMA predictions for future collaboration."""
        labels = cinema_probs.argmax(dim=1) if cinema_probs.ndim == 4 else cinema_probs.long()
        return _boxes_from_labels(labels, self.num_classes)

    def resolve_prompt_mode(self, batch: dict[str, Any], cinema_output: dict[str, Any] | None = None) -> Tensor | None:
        if self.prompt_mode == "gt_box":
            if "mask" not in batch:
                raise TeacherLoadError("Medical-SAM3 prompt_mode=gt_box requires batch['mask'] during training.")
            return self.boxes_from_gt(batch["mask"])
        if self.prompt_mode == "cinema_box":
            if cinema_output is None or "probs" not in cinema_output:
                raise TeacherLoadError("Medical-SAM3 prompt_mode=cinema_box requires CineMA probabilities.")
            return self.boxes_from_cinema(cinema_output["probs"])
        if self.prompt_mode == "text":
            warnings.warn("Medical-SAM3 text prompts are not bound in this adapter; falling back to gt_box when GT is available.")
            return self.boxes_from_gt(batch["mask"]) if "mask" in batch else None
        raise TeacherLoadError(f"Unknown Medical-SAM3 prompt mode: {self.prompt_mode}")


def _boxes_from_labels(mask: Tensor, num_classes: int) -> Tensor:
    if mask.ndim == 4:
        mask = mask[:, 0]
    B, H, W = mask.shape
    boxes = torch.zeros(B, num_classes, 4, device=mask.device, dtype=torch.float32)
    for b in range(B):
        for cls in range(1, num_classes):
            ys, xs = torch.where(mask[b] == cls)
            if ys.numel() == 0:
                continue
            boxes[b, cls] = torch.tensor(
                [xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item()],
                device=mask.device,
                dtype=torch.float32,
            )
    return boxes
