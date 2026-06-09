"""Medical-SAM3 frozen teacher adapter."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .base_teacher import FrozenSegmentationTeacher, TeacherLoadError
from .teacher_utils import entropy_confidence, list_checkpoint_candidates, make_teacher_stub, normalize_probs, one_hot_to_boundary


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
        self.prompt_mode = str(prompt_mode).lower()
        self.checkpoint_candidates: list[Path] = []
        self.checkpoint_path: Path | None = None
        self.sam3_model: Any | None = None
        self.inference_module: Any | None = None
        self.class_prompts = {
            1: "right ventricle",
            2: "myocardium",
            3: "left ventricle",
        }

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
        self.checkpoint_path = _select_checkpoint(self.checkpoint_candidates)
        self.inference_module = self._load_inference_module()
        sam3_cls = getattr(self.inference_module, "SAM3Model", None)
        if sam3_cls is None:
            raise TeacherLoadError("Medical-SAM3 inference/sam3_inference.py does not expose SAM3Model.")
        self.sam3_model = sam3_cls(
            confidence_threshold=float(self.kwargs.get("confidence_threshold", 0.1)),
            device=str(self.device),
            checkpoint_path=str(self.checkpoint_path),
        )
        self.loaded = True
        return self

    @torch.no_grad()
    def predict(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.teacher_stub:
            return make_teacher_stub(batch, self.num_classes, mode="medical_sam3")
        if self.sam3_model is None:
            self.load()
        if self.sam3_model is None:
            raise TeacherLoadError("Medical-SAM3 model was not initialized.")
        if "image" not in batch:
            raise KeyError("Medical-SAM3 expects batch['image']")

        image = batch["image"].to(self.device).float()
        B, _, H, W = image.shape
        boxes = self.resolve_prompt_mode(batch)
        probs = torch.zeros(B, self.num_classes, H, W, device=self.device, dtype=torch.float32)
        probs[:, 0] = 1.0

        for batch_idx in range(B):
            rgb = _image_tensor_to_rgb_uint8(image[batch_idx])
            try:
                inference_state = self.sam3_model.encode_image(rgb)
            except Exception as exc:  # pragma: no cover - depends on external repo
                raise TeacherLoadError(f"Medical-SAM3 encode_image failed: {exc}") from exc

            for cls in range(1, self.num_classes):
                pred = self._predict_class_mask(inference_state, boxes, batch_idx, cls, (H, W))
                if pred is None:
                    continue
                mask = _numpy_mask_to_tensor(pred, (H, W), self.device)
                probs[batch_idx, cls] = torch.maximum(probs[batch_idx, cls], mask)

            foreground = probs[batch_idx, 1:].amax(dim=0)
            probs[batch_idx, 0] = (1.0 - foreground).clamp(0.0, 1.0)

        probs = normalize_probs(probs)
        confidence = entropy_confidence(probs)
        boundary = one_hot_to_boundary(probs, mode="soft", dilation=3)
        return {
            "logits": probs.clamp_min(1e-8).log(),
            "probs": probs.detach(),
            "mask": probs.argmax(dim=1).detach(),
            "confidence": confidence.detach(),
            "boundary": boundary.detach(),
            "meta": {
                "teacher_stub": False,
                "teacher": "medical_sam3",
                "checkpoint_path": str(self.checkpoint_path),
                "prompt_mode": self.prompt_mode,
                "class_order": ["BG", "RV", "MYO", "LV"][: self.num_classes],
            },
        }

    def _load_inference_module(self) -> Any:
        repo = self.repo_path.resolve()
        inference_dir = repo / "inference"
        inference_file = inference_dir / "sam3_inference.py"
        if not inference_file.exists():
            raise TeacherLoadError(
                f"Medical-SAM3 inference file not found: {inference_file}. "
                "Expected external/Medical-SAM3/inference/sam3_inference.py from AIM-Research-Lab/Medical-SAM3."
            )
        sam3_root = _resolve_sam3_root(repo)
        for path in (repo, inference_dir, sam3_root):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)
        module_name = f"_medical_sam3_inference_{abs(hash(str(inference_file)))}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, inference_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not build import spec for {inference_file}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.SAM3_ROOT = sam3_root
            return module
        except Exception as exc:  # pragma: no cover - depends on external repo
            raise TeacherLoadError(
                "Could not import Medical-SAM3 inference wrapper. Install its dependencies, then retry:\n"
                "  python -m pip install -e external/Medical-SAM3\n"
                "  python -m pip install iopath opencv-python scikit-image\n"
                f"Original import error: {type(exc).__name__}: {exc}"
            ) from exc

    def _predict_class_mask(
        self,
        inference_state: dict[str, Any],
        boxes: Tensor | None,
        batch_idx: int,
        cls: int,
        image_size: tuple[int, int],
    ) -> np.ndarray | None:
        if self.prompt_mode in {"gt_box", "cinema_box"}:
            if boxes is None:
                raise TeacherLoadError(f"Medical-SAM3 prompt_mode={self.prompt_mode} requires boxes.")
            box = boxes[batch_idx, cls].detach().float().cpu()
            x_min, y_min, x_max, y_max = [int(round(float(v))) for v in box]
            if x_max <= x_min or y_max <= y_min:
                return None
            try:
                return self.sam3_model.predict_box(inference_state, (x_min, y_min, x_max, y_max), image_size)
            except Exception as exc:  # pragma: no cover - depends on external repo
                raise TeacherLoadError(f"Medical-SAM3 box prediction failed for class {cls}: {exc}") from exc
        if self.prompt_mode == "text":
            prompt = self.class_prompts.get(cls, f"class {cls}")
            try:
                return self.sam3_model.predict_text(inference_state, prompt)
            except Exception as exc:  # pragma: no cover - depends on external repo
                raise TeacherLoadError(f"Medical-SAM3 text prediction failed for prompt {prompt!r}: {exc}") from exc
        raise TeacherLoadError(f"Unknown Medical-SAM3 prompt mode: {self.prompt_mode}")

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


def _select_checkpoint(candidates: list[Path]) -> Path:
    preferred = sorted(candidates, key=lambda path: (path.name != "checkpoint.pt", len(str(path)), str(path)))
    return preferred[0]


def _resolve_sam3_root(repo_path: Path) -> Path:
    candidates = [repo_path / "sam3", repo_path]
    for path in candidates:
        if (path / "assets").exists() or (path / "sam3").exists():
            return path.resolve()
    return (repo_path / "sam3").resolve()


def _image_tensor_to_rgb_uint8(image: Tensor) -> np.ndarray:
    """Convert `[C,H,W]` tensor to RGB uint8 expected by Medical-SAM3."""
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got {tuple(image.shape)}")
    if image.shape[0] >= 3:
        img = image[:3]
    else:
        center = image[image.shape[0] // 2]
        img = center.unsqueeze(0).repeat(3, 1, 1)
    img = img.detach().float().cpu()
    img_min = float(img.min())
    img_max = float(img.max())
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = torch.zeros_like(img)
    return (img.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _numpy_mask_to_tensor(mask: np.ndarray, spatial: tuple[int, int], device: torch.device) -> Tensor:
    tensor = torch.as_tensor(np.asarray(mask).astype(np.float32), device=device)
    if tensor.ndim > 2:
        tensor = tensor.squeeze()
    tensor = (tensor > 0).float().view(1, 1, *tensor.shape[-2:])
    if tuple(tensor.shape[-2:]) != spatial:
        tensor = F.interpolate(tensor, size=spatial, mode="nearest")
    return tensor[0, 0].clamp(0.0, 1.0)
