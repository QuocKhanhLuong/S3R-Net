"""Optional teacher interfaces for S3R-SCSD.

The first implementation is intentionally cache-first. Concrete open-weight
teacher wrappers can subclass these interfaces later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .teacher_cache import load_teacher_cache
from .teacher_targets import semantic_entropy


class BaseTeacher:
    """Base wrapper for optional online teachers."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | torch.device = "cpu",
        model: nn.Module | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.device = torch.device(device)
        self.model = model
        if self.model is not None:
            self.model.to(self.device).eval()
            if self.checkpoint is not None:
                state = torch.load(self.checkpoint, map_location=self.device)
                if isinstance(state, dict):
                    state = state.get("model_state", state.get("state_dict", state))
                self.model.load_state_dict(state, strict=False)

    def preprocess(self, x: Tensor) -> Tensor:
        return x.to(self.device)

    def predict(self, x: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError("Subclasses must implement predict(). Use teacher_cache_dir for cache-only mode.")


class SemanticTeacher(BaseTeacher):
    """Semantic/Dice teacher wrapper returning probabilities and entropy."""

    @torch.no_grad()
    def predict(self, x: Tensor) -> dict[str, Tensor]:
        if self.model is None:
            raise RuntimeError("SemanticTeacher requires a model or a teacher cache.")
        x = self.preprocess(x)
        out = self.model(x)
        logits = out["seg_logits"] if isinstance(out, dict) and "seg_logits" in out else out["output"]
        probs = torch.softmax(logits, dim=1)
        return {
            "probs": probs,
            "logits": logits,
            "entropy": semantic_entropy(probs),
            "foreground_prob": probs[:, 1:].sum(dim=1, keepdim=True),
        }


class CharacteristicTeacher(BaseTeacher):
    """Boundary/spectral teacher interface hook."""

    @torch.no_grad()
    def predict(self, x: Tensor) -> dict[str, Tensor]:
        if self.model is None:
            raise RuntimeError("CharacteristicTeacher requires a model or a teacher cache.")
        x = self.preprocess(x)
        out = self.model(x)
        boundary = out.get("boundary_logits") if isinstance(out, dict) else None
        if boundary is None:
            raise RuntimeError("Characteristic teacher output must include boundary_logits.")
        boundary_prob = torch.sigmoid(boundary)
        return {
            "boundary": boundary_prob,
            "distance": out.get("distance", torch.zeros_like(boundary_prob)),
            "foreground_prob": out.get("foreground_prob", boundary_prob),
        }


class CacheOnlyTeacher:
    """Load teacher targets from cache without constructing online teachers."""

    def __init__(self, teacher_cache_dir: str | Path, dataset: str = "ACDC") -> None:
        self.teacher_cache_dir = Path(teacher_cache_dir)
        self.dataset = dataset

    def predict(self, case_id: str, slice_idx: int) -> dict[str, Any]:
        return load_teacher_cache(self.teacher_cache_dir, case_id, slice_idx, self.dataset)
