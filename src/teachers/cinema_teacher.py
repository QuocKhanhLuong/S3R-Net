"""CineMA frozen teacher adapter."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch

from .base_teacher import FrozenSegmentationTeacher, TeacherLoadError
from .teacher_utils import list_checkpoint_candidates, make_teacher_stub


class CineMATeacher(FrozenSegmentationTeacher):
    """Cardiac anatomical / boundary teacher wrapper."""

    AUTOSELECT_KEYWORDS = ("seg", "segment", "ventricle", "myocardium", "acdc", "cine", "sax")

    def __init__(
        self,
        checkpoint_dir: str | Path | None,
        device: str | torch.device,
        num_classes: int,
        image_size: int | None = None,
        repo_path: str | Path = "external/CineMA",
        checkpoint_path: str | Path | None = None,
        class_map: str | dict[int, int] | None = None,
        teacher_stub: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(checkpoint_dir, device, num_classes, image_size, teacher_stub, **kwargs)
        self.repo_path = Path(repo_path)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.class_map = parse_class_map(class_map)

    def load(self) -> "CineMATeacher":
        if self.teacher_stub:
            self.loaded = True
            return self
        if not self.repo_path.exists():
            raise TeacherLoadError(f"CineMA repo not found at {self.repo_path}. Run: bash scripts/setup_teachers.sh")
        ckpt = self.checkpoint_path or self.auto_select_checkpoint()
        if ckpt is None:
            raise TeacherLoadError(
                f"No compatible CineMA checkpoint found under {self.checkpoint_dir}. "
                "Prefer the ACDC SAX segmentation checkpoint such as finetuned/segmentation/acdc_sax/acdc_sax_0.safetensors. "
                "If the Hugging Face repo is gated, run: huggingface-cli login"
            )
        self.checkpoint_path = ckpt
        sys.path.insert(0, str(self.repo_path.resolve()))
        import_errors = []
        for module_name in ("cinema", "cinema.models", "cinema.segmentation"):
            try:
                importlib.import_module(module_name)
                self.loaded = True
                break
            except Exception as exc:  # pragma: no cover - depends on external repo
                import_errors.append(f"{module_name}: {exc}")
        else:
            raise TeacherLoadError(
                "Could not import CineMA modules from external/CineMA. "
                "Install CineMA dependencies, then retry. Import attempts: "
                + " | ".join(import_errors[:4])
            )
        raise TeacherLoadError(
            "CineMA dependency import succeeded, but this adapter does not yet bind a pinned upstream inference API. "
            "Use --teacher_cache_dir with precomputed outputs or --teacher_stub for pipeline testing."
        )

    def auto_select_checkpoint(self) -> Path | None:
        if self.checkpoint_dir is None or not self.checkpoint_dir.exists():
            return None
        candidates = list_checkpoint_candidates(self.checkpoint_dir)
        if not candidates:
            return None
        scored = []
        for path in candidates:
            text = str(path).lower()
            score = sum(1 for key in self.AUTOSELECT_KEYWORDS if key in text)
            scored.append((score, len(str(path)), path))
        scored.sort(key=lambda item: (-item[0], item[1], str(item[2])))
        return scored[0][2] if scored[0][0] > 0 else None

    @torch.no_grad()
    def predict(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.teacher_stub:
            return make_teacher_stub(batch, self.num_classes, mode="cinema")
        raise TeacherLoadError("CineMA real predict is not bound. Use cache/stub or implement the pinned repo API adapter.")


def parse_class_map(value: str | dict[int, int] | None) -> dict[int, int] | None:
    """Parse `source:target,source:target` class mapping."""
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return {int(k): int(v) for k, v in value.items()}
    mapping: dict[int, int] = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        source, target = item.split(":")
        mapping[int(source.strip())] = int(target.strip())
    return mapping
