"""Utilities shared by frozen teacher wrappers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


CHECKPOINT_EXTENSIONS = (".pt", ".pth", ".ckpt", ".safetensors", ".bin")


def list_checkpoint_candidates(root: str | Path) -> list[Path]:
    """Return likely checkpoint files under a teacher checkpoint directory."""
    path = Path(root)
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS)


def teacher_cache_key(case_id: str, slice_idx: int) -> str:
    """Return a deterministic per-slice teacher cache key."""
    safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_id))
    return f"{safe_case}_{int(slice_idx):04d}"


def teacher_cache_path(cache_dir: str | Path, case_id: str, slice_idx: int) -> Path:
    """Return the `.pt` cache path for one sample."""
    return Path(cache_dir) / f"{teacher_cache_key(case_id, slice_idx)}.pt"


def save_dual_teacher_cache(
    cache_dir: str | Path,
    case_id: str,
    slice_idx: int,
    payload: dict[str, Any],
) -> Path:
    """Save one dual-teacher cache item."""
    path = teacher_cache_path(cache_dir, case_id, slice_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_dual_teacher_cache(cache_dir: str | Path, case_id: str, slice_idx: int, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load one dual-teacher cache item."""
    path = teacher_cache_path(cache_dir, case_id, slice_idx)
    if not path.exists():
        raise FileNotFoundError(f"Teacher cache item not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Teacher cache item must be a dict, got {type(payload)!r}: {path}")
    return payload


def entropy_confidence(probs: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert class probabilities to normalized confidence `[B,1,...]`."""
    probs = normalize_probs(probs, eps=eps)
    entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=1, keepdim=True)
    max_entropy = math.log(max(int(probs.shape[1]), 2))
    return (1.0 - entropy / max_entropy).clamp(0.0, 1.0)


def safe_softmax_logits(logits: Tensor, dim: int = 1) -> Tensor:
    """Softmax with finite-value guard."""
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    return torch.softmax(logits, dim=dim)


def normalize_probs(probs: Tensor, eps: float = 1e-8) -> Tensor:
    """Clamp and renormalize class probabilities along channel dim."""
    probs = torch.nan_to_num(probs.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)


def one_hot_to_boundary(mask_or_probs: Tensor, mode: str = "soft", dilation: int = 3) -> Tensor:
    """Extract a foreground boundary map from labels or multiclass probs."""
    if mask_or_probs.ndim == 3:
        labels = mask_or_probs.long()
        foreground = (labels > 0).float().unsqueeze(1)
    elif mask_or_probs.ndim == 4 and mask_or_probs.shape[1] == 1:
        foreground = mask_or_probs.float().clamp(0.0, 1.0)
    elif mask_or_probs.ndim == 4:
        if mode == "hard":
            foreground = (mask_or_probs.argmax(dim=1, keepdim=True) > 0).float()
        else:
            foreground = mask_or_probs[:, 1:].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Expected labels [B,H,W] or probs [B,C,H,W], got {tuple(mask_or_probs.shape)}")

    kernel = max(int(dilation), 1)
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    high = F.max_pool2d(foreground, kernel_size=kernel, stride=1, padding=pad)
    low = -F.max_pool2d(-foreground, kernel_size=kernel, stride=1, padding=pad)
    return (high - low).clamp(0.0, 1.0)


def extract_boundary_band(mask: Tensor, radius: int = 3) -> Tensor:
    """Dilated boundary band around labels or soft foreground."""
    boundary = one_hot_to_boundary(mask, mode="hard" if mask.ndim == 3 else "soft", dilation=3)
    kernel = max(int(radius) * 2 + 1, 1)
    pad = kernel // 2
    return F.max_pool2d(boundary, kernel_size=kernel, stride=1, padding=pad).clamp(0.0, 1.0)


def resize_teacher_output_to_student(output: Any, target_shape: tuple[int, ...]) -> Any:
    """Resize tensors in a teacher output dict to the student spatial shape."""
    spatial = tuple(int(v) for v in target_shape[-2:])
    if isinstance(output, Tensor):
        return _resize_tensor(output, spatial)
    if isinstance(output, Mapping):
        return {key: resize_teacher_output_to_student(value, target_shape) if key != "meta" else value for key, value in output.items()}
    if isinstance(output, list):
        return [resize_teacher_output_to_student(item, target_shape) for item in output]
    if isinstance(output, tuple):
        return tuple(resize_teacher_output_to_student(item, target_shape) for item in output)
    return output


def _resize_tensor(tensor: Tensor, spatial: tuple[int, int]) -> Tensor:
    if tensor.ndim < 4 or tuple(tensor.shape[-2:]) == spatial:
        return tensor
    mode = "nearest" if tensor.dtype in (torch.long, torch.int64, torch.int32, torch.bool) else "bilinear"
    if mode == "nearest":
        return F.interpolate(tensor.float(), size=spatial, mode=mode).to(dtype=tensor.dtype)
    return F.interpolate(tensor.float(), size=spatial, mode=mode, align_corners=False)


def stack_binary_masks_to_multiclass(binary_masks: Tensor | list[Tensor], class_order: list[int] | None = None) -> Tensor:
    """Create `[B,C,H,W]` probs from foreground binary masks.

    `binary_masks` may be `[B,K,H,W]` or a list of `[B,1,H,W]` tensors. The
    background channel is computed as `1 - max(foreground)`.
    """
    if isinstance(binary_masks, list):
        if not binary_masks:
            raise ValueError("binary_masks list is empty")
        fg = torch.cat([m.float() if m.ndim == 4 else m.float().unsqueeze(1) for m in binary_masks], dim=1)
    else:
        fg = binary_masks.float()
        if fg.ndim == 3:
            fg = fg.unsqueeze(1)
    fg = fg.clamp(0.0, 1.0)
    if class_order is not None:
        if len(class_order) != fg.shape[1]:
            raise ValueError(f"class_order length {len(class_order)} does not match foreground masks {fg.shape[1]}")
        ordered = torch.zeros_like(fg)
        for source_idx, target_class in enumerate(class_order):
            if target_class <= 0:
                continue
            target_idx = int(target_class) - 1
            if 0 <= target_idx < ordered.shape[1]:
                ordered[:, target_idx] = fg[:, source_idx]
        fg = ordered
    bg = (1.0 - fg.max(dim=1, keepdim=True).values).clamp(0.0, 1.0)
    return normalize_probs(torch.cat([bg, fg], dim=1))


def make_teacher_stub(batch: dict[str, Any], num_classes: int, mode: str) -> dict[str, Any]:
    """Return deterministic dummy teacher output for pipeline testing."""
    if "image" not in batch:
        raise KeyError("Teacher stub expects batch['image']")
    image = batch["image"]
    device = image.device
    B, _, H, W = image.shape
    if "mask" in batch:
        labels = batch["mask"].to(device).long().clamp(0, num_classes - 1)
    else:
        center = image[:, image.shape[1] // 2]
        thresh = center.flatten(1).median(dim=1).values.view(B, 1, 1)
        labels = (center > thresh).long()
        labels = labels.clamp(0, num_classes - 1)

    probs = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    if mode.lower() in {"medsam2", "medical_sam3", "m3", "field"}:
        probs = 0.92 * probs + 0.08 / float(num_classes)
    elif mode.lower() in {"cinema", "cine", "boundary"}:
        boundary = extract_boundary_band(labels, radius=2)
        smooth = F.avg_pool2d(probs, kernel_size=3, stride=1, padding=1)
        probs = torch.where(boundary > 0.0, 0.85 * smooth + 0.15 / float(num_classes), 0.95 * probs + 0.05 / float(num_classes))
    else:
        probs = 0.90 * probs + 0.10 / float(num_classes)
    probs = normalize_probs(probs)
    confidence = entropy_confidence(probs)
    boundary = one_hot_to_boundary(probs, mode="soft", dilation=3)
    return {
        "logits": probs.clamp_min(1e-8).log(),
        "probs": probs.detach(),
        "mask": probs.argmax(dim=1).detach(),
        "confidence": confidence.detach(),
        "boundary": boundary.detach(),
        "meta": {"teacher_stub": True, "mode": mode, "class_order": ["BG", "RV", "MYO", "LV"][:num_classes]},
    }
