"""Base interface for frozen segmentation teachers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import nn


class TeacherLoadError(RuntimeError):
    """Raised when an external teacher cannot be loaded actionably."""


class FrozenSegmentationTeacher(ABC):
    """Abstract frozen teacher interface used only during training."""

    def __init__(
        self,
        checkpoint_dir: str | Path | None,
        device: str | torch.device,
        num_classes: int,
        image_size: int | None = None,
        teacher_stub: bool = False,
        **kwargs: Any,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.device = torch.device(device)
        self.num_classes = int(num_classes)
        self.image_size = image_size
        self.teacher_stub = bool(teacher_stub)
        self.kwargs = kwargs
        self.model: nn.Module | None = None
        self.loaded = False

    @abstractmethod
    def load(self) -> "FrozenSegmentationTeacher":
        """Load external weights and freeze the teacher."""

    def freeze_model(self) -> None:
        """Set eval mode and disable gradients."""
        if self.model is None:
            return
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensor batch entries to the teacher device."""
        out: dict[str, Any] = {}
        for key, value in batch.items():
            out[key] = value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
        return out

    @torch.no_grad()
    @abstractmethod
    def predict(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run teacher inference and return a standard output dict."""

    def postprocess(self, raw_outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        """Detach teacher outputs and keep metadata as-is."""
        out: dict[str, Any] = {}
        for key, value in raw_outputs.items():
            out[key] = value.detach() if torch.is_tensor(value) else value
        return out

    @torch.no_grad()
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not self.loaded:
            self.load()
        prepared = self.preprocess(batch)
        raw = self.predict(prepared)
        return self.postprocess(raw, prepared)
