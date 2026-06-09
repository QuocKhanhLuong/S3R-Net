"""MedSAM2 frozen teacher adapter."""

from __future__ import annotations

import importlib
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


class MedSAM2Teacher(FrozenSegmentationTeacher):
    """Prompt-driven MedSAM2 teacher wrapper.

    MedSAM2's public repo exposes SAM2 video predictors. This adapter uses the
    3D/volume builder when available and maps ACDC slice training batches to a
    small per-sample video, using GT boxes only for teacher-cache generation.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path | None,
        device: str | torch.device,
        num_classes: int,
        image_size: int | None = None,
        repo_path: str | Path = "external/MedSAM2",
        checkpoint_path: str | Path | None = None,
        config_path: str | Path = "configs/sam2.1_hiera_t512.yaml",
        prompt_mode: str = "gt_box",
        teacher_stub: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(checkpoint_dir, device, num_classes, image_size, teacher_stub, **kwargs)
        self.repo_path = Path(repo_path)
        self.explicit_checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.config_path = Path(config_path)
        self.prompt_mode = str(prompt_mode).lower()
        self.predictor_image_size = int(kwargs.get("predictor_image_size", 512))
        self.checkpoint_candidates: list[Path] = []
        self.checkpoint_path: Path | None = None
        self.predictor: Any | None = None

    def load(self) -> "MedSAM2Teacher":
        if self.teacher_stub:
            self.loaded = True
            return self
        if not self.repo_path.exists():
            raise TeacherLoadError(
                f"MedSAM2 repo not found at {self.repo_path}. "
                "Run: bash scripts/setup_teachers.sh medsam2"
            )
        self.checkpoint_path = self._resolve_checkpoint()
        builder = self._load_predictor_builder()
        try:
            self.predictor = builder(str(self._resolve_config_path()), str(self.checkpoint_path))
        except Exception as exc:  # pragma: no cover - depends on external repo
            raise TeacherLoadError(
                "Could not initialize MedSAM2 predictor. Install the external repo dependencies, then retry:\n"
                "  python -m pip install -e external/MedSAM2\n"
                "  python -m pip install SimpleITK scikit-image opencv-python\n"
                f"Original initialization error: {type(exc).__name__}: {exc}"
            ) from exc
        self.loaded = True
        return self

    @torch.no_grad()
    def predict(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.teacher_stub:
            return make_teacher_stub(batch, self.num_classes, mode="medsam2")
        if self.predictor is None:
            self.load()
        if self.predictor is None:
            raise TeacherLoadError("MedSAM2 predictor was not initialized.")
        if "image" not in batch:
            raise KeyError("MedSAM2 expects batch['image']")

        image = batch["image"].to(self.device).float()
        B, _, H, W = image.shape
        boxes = self.resolve_prompt_mode(batch)
        probs = torch.zeros(B, self.num_classes, H, W, device=self.device, dtype=torch.float32)
        probs[:, 0] = 1.0

        for batch_idx in range(B):
            video, frame_idx = _image_tensor_to_medsam2_video(
                image[batch_idx],
                device=self.device,
                predictor_image_size=self.predictor_image_size,
            )
            try:
                inference_state = self.predictor.init_state(video, H, W)
            except Exception as exc:  # pragma: no cover - depends on external repo
                raise TeacherLoadError(f"MedSAM2 init_state failed: {exc}") from exc

            for cls in range(1, self.num_classes):
                pred = self._predict_class_mask(inference_state, boxes, batch_idx, cls, frame_idx, (H, W))
                if pred is None:
                    continue
                mask = _mask_logits_to_tensor(pred, (H, W), self.device)
                probs[batch_idx, cls] = torch.maximum(probs[batch_idx, cls], mask)

            foreground = probs[batch_idx, 1:].amax(dim=0)
            probs[batch_idx, 0] = (1.0 - foreground).clamp(0.0, 1.0)
            self._reset_state(inference_state)

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
                "teacher": "medsam2",
                "checkpoint_path": str(self.checkpoint_path),
                "config_path": str(self._resolve_config_path()),
                "prompt_mode": self.prompt_mode,
                "class_order": ["BG", "RV", "MYO", "LV"][: self.num_classes],
            },
        }

    def _resolve_checkpoint(self) -> Path:
        if self.explicit_checkpoint_path is not None:
            explicit = self.explicit_checkpoint_path.expanduser()
            if not explicit.is_absolute() and self.checkpoint_dir is not None:
                explicit = self.checkpoint_dir / explicit
            if not explicit.exists():
                raise TeacherLoadError(f"MedSAM2 checkpoint not found: {explicit}")
            return explicit.resolve()
        if self.checkpoint_dir is None or not self.checkpoint_dir.exists():
            raise TeacherLoadError(
                f"MedSAM2 checkpoint directory not found: {self.checkpoint_dir}. "
                "Run: python scripts/download_teachers.py --teacher medsam2 --output_dir checkpoints/teachers"
            )
        self.checkpoint_candidates = list_checkpoint_candidates(self.checkpoint_dir)
        if not self.checkpoint_candidates:
            raise TeacherLoadError(
                f"No MedSAM2 checkpoint candidates found under {self.checkpoint_dir}. "
                "Expected MedSAM2_latest.pt from Hugging Face repo wanglab/MedSAM2."
            )
        return _select_checkpoint(self.checkpoint_candidates)

    def _resolve_config_path(self) -> Path:
        path = self.config_path.expanduser()
        if path.is_absolute():
            return path
        return (self.repo_path / path).resolve()

    def _load_predictor_builder(self) -> Any:
        repo = self.repo_path.resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        for cached_name in ("sam2.build_sam", "sam2"):
            cached = sys.modules.get(cached_name)
            cached_file = getattr(cached, "__file__", None) if cached is not None else None
            if cached_file and not _is_relative_to(Path(cached_file).resolve(), repo):
                del sys.modules[cached_name]
        for module_name in ("sam2.build_sam", "build_sam"):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            builder = getattr(module, "build_sam2_video_predictor_npz", None)
            if builder is None:
                builder = getattr(module, "build_sam2_video_predictor", None)
            if builder is not None:
                return builder
        raise TeacherLoadError(
            "Could not import MedSAM2 predictor builder from external/MedSAM2. "
            "Expected sam2.build_sam.build_sam2_video_predictor_npz from bowang-lab/MedSAM2. "
            "Install with: python -m pip install -e external/MedSAM2"
        )

    def _predict_class_mask(
        self,
        inference_state: Any,
        boxes: Tensor | None,
        batch_idx: int,
        cls: int,
        frame_idx: int,
        image_size: tuple[int, int],
    ) -> Tensor | np.ndarray | None:
        if self.prompt_mode in {"gt_box", "cinema_box"}:
            if boxes is None:
                raise TeacherLoadError(f"MedSAM2 prompt_mode={self.prompt_mode} requires boxes.")
            box = boxes[batch_idx, cls].detach().float().cpu()
            x_min, y_min, x_max, y_max = [int(round(float(v))) for v in box]
            if x_max <= x_min or y_max <= y_min:
                return None
            try:
                _, out_obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=int(cls),
                    box=np.array([x_min, y_min, x_max, y_max], dtype=np.float32),
                )
                return _select_object_logits(mask_logits, out_obj_ids, cls)
            except Exception as exc:  # pragma: no cover - depends on external repo
                raise TeacherLoadError(f"MedSAM2 box prediction failed for class {cls}: {exc}") from exc
        if self.prompt_mode == "text":
            raise TeacherLoadError("MedSAM2 adapter supports box prompts only; use --medsam2_prompt_mode gt_box.")
        raise TeacherLoadError(f"Unknown MedSAM2 prompt mode: {self.prompt_mode}")

    def _reset_state(self, inference_state: Any) -> None:
        reset = getattr(self.predictor, "reset_state", None)
        if callable(reset):
            try:
                reset(inference_state)
            except Exception:  # pragma: no cover - external cleanup should not mask predictions
                pass

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
                raise TeacherLoadError("MedSAM2 prompt_mode=gt_box requires batch['mask'] during training.")
            return self.boxes_from_gt(batch["mask"])
        if self.prompt_mode == "cinema_box":
            if cinema_output is None or "probs" not in cinema_output:
                raise TeacherLoadError("MedSAM2 prompt_mode=cinema_box requires CineMA probabilities.")
            return self.boxes_from_cinema(cinema_output["probs"])
        if self.prompt_mode == "text":
            warnings.warn("MedSAM2 text prompts are not supported in this adapter; falling back to gt_box when GT is available.")
            return self.boxes_from_gt(batch["mask"]) if "mask" in batch else None
        raise TeacherLoadError(f"Unknown MedSAM2 prompt mode: {self.prompt_mode}")


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
    def score(path: Path) -> tuple[int, int, int, str]:
        name = path.name
        return (
            0 if name == "MedSAM2_latest.pt" else 1,
            0 if name.startswith("MedSAM2_") else 1,
            len(str(path)),
            str(path),
        )

    return sorted(candidates, key=score)[0]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _image_tensor_to_medsam2_video(image: Tensor, device: torch.device, predictor_image_size: int) -> tuple[Tensor, int]:
    """Convert `[C,H,W]` ACDC tensor to normalized MedSAM2 video `[D,3,S,S]`."""
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got {tuple(image.shape)}")
    slices = image.detach().float()
    img_min = float(slices.min())
    img_max = float(slices.max())
    if img_max > img_min:
        slices = (slices - img_min) / (img_max - img_min)
    else:
        slices = torch.zeros_like(slices)
    video = slices.unsqueeze(1).repeat(1, 3, 1, 1)
    if tuple(video.shape[-2:]) != (predictor_image_size, predictor_image_size):
        video = F.interpolate(video, size=(predictor_image_size, predictor_image_size), mode="bilinear", align_corners=False)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=video.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=video.device).view(1, 3, 1, 1)
    video = (video.clamp(0.0, 1.0) - mean) / std
    frame_idx = int(slices.shape[0] // 2)
    return video.to(device), frame_idx


def _select_object_logits(mask_logits: Tensor | np.ndarray, out_obj_ids: Any, cls: int) -> Tensor | np.ndarray:
    if out_obj_ids is None:
        return mask_logits
    try:
        ids = [int(item) for item in out_obj_ids]
    except TypeError:
        ids = [int(out_obj_ids)]
    if int(cls) not in ids:
        return mask_logits
    index = ids.index(int(cls))
    if isinstance(mask_logits, Tensor):
        if mask_logits.ndim <= 2:
            return mask_logits
        return mask_logits[index : index + 1]
    array = np.asarray(mask_logits)
    if array.ndim <= 2:
        return mask_logits
    return array[index : index + 1]


def _mask_logits_to_tensor(mask_logits: Tensor | np.ndarray, spatial: tuple[int, int], device: torch.device) -> Tensor:
    tensor = torch.as_tensor(mask_logits, device=device).detach().float()
    while tensor.ndim > 2:
        tensor = tensor[0]
    tensor = tensor.view(1, 1, *tensor.shape[-2:])
    if tuple(tensor.shape[-2:]) != spatial:
        tensor = F.interpolate(tensor, size=spatial, mode="bilinear", align_corners=False)
    if float(tensor.max()) > 1.0 or float(tensor.min()) < 0.0:
        tensor = torch.sigmoid(tensor)
    else:
        tensor = tensor.clamp(0.0, 1.0)
    return (tensor[0, 0] > 0.5).float()
