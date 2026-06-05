"""S3R segmentation and boundary metrics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    HAS_SCIPY = True
except Exception:  # pragma: no cover - depends on environment
    binary_erosion = None
    distance_transform_edt = None
    HAS_SCIPY = False


CLASS_METRIC_NAMES = {1: "rv", 2: "myo", 3: "lv"}


def dice_per_class(pred: Tensor, target: Tensor, num_classes: int = 4) -> dict[str, float]:
    """Return label Dice per class with empty classes handled safely."""
    pred = pred.detach()
    target = target.detach()
    out: dict[str, float] = {}
    vals = []
    for cls in range(num_classes):
        pred_c = pred == cls
        target_c = target == cls
        denom = pred_c.sum() + target_c.sum()
        if denom == 0:
            val = math.nan
        else:
            val = float((2 * (pred_c & target_c).sum()).float() / denom.float().clamp_min(1.0))
        name = CLASS_METRIC_NAMES.get(cls, "bg" if cls == 0 else f"class{cls}")
        out[f"dice_{name}"] = val
        if cls > 0 and not math.isnan(val):
            vals.append(val)
    out["mean_foreground_dice"] = float(np.mean(vals)) if vals else math.nan
    return out


def binary_surface(mask: np.ndarray) -> np.ndarray:
    _require_scipy()
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, border_value=0)
    return mask ^ eroded


def binary_surface_distances(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _require_scipy()
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    spacing_tuple = _spacing_tuple(spacing, pred.ndim)
    pred_surface = binary_surface(pred)
    gt_surface = binary_surface(gt)
    pred_count = int(pred_surface.sum())
    gt_count = int(gt_surface.sum())
    if gt_count == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if pred_count == 0:
        penalty = _image_diagonal(pred.shape, spacing_tuple)
        return np.full(1, penalty, dtype=np.float64), np.full(gt_count, penalty, dtype=np.float64)
    dt_to_gt = distance_transform_edt(~gt_surface, sampling=spacing_tuple)
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing_tuple)
    return dt_to_gt[pred_surface].astype(np.float64), dt_to_pred[gt_surface].astype(np.float64)


def hd95_binary(pred: np.ndarray, gt: np.ndarray, spacing: Iterable[float] | None = None) -> float:
    p2g, g2p = binary_surface_distances(pred, gt, spacing)
    distances = np.concatenate([p2g, g2p])
    if distances.size == 0:
        return math.nan
    return float(np.percentile(distances, 95))


def assd_binary(pred: np.ndarray, gt: np.ndarray, spacing: Iterable[float] | None = None) -> float:
    p2g, g2p = binary_surface_distances(pred, gt, spacing)
    distances = np.concatenate([p2g, g2p])
    if distances.size == 0:
        return math.nan
    return float(distances.mean())


def boundary_f1_binary(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance: float = 2.0,
    spacing: Iterable[float] | None = None,
) -> float:
    _require_scipy()
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    pred_surface = binary_surface(pred)
    gt_surface = binary_surface(gt)
    pred_count = int(pred_surface.sum())
    gt_count = int(gt_surface.sum())
    if gt_count == 0:
        return math.nan
    if pred_count == 0:
        return 0.0
    spacing_tuple = _spacing_tuple(spacing, pred.ndim)
    dt_to_gt = distance_transform_edt(~gt_surface, sampling=spacing_tuple)
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing_tuple)
    precision = float((dt_to_gt[pred_surface] <= tolerance).mean())
    recall = float((dt_to_pred[gt_surface] <= tolerance).mean())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def surface_dice_binary(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance: float = 2.0,
    spacing: Iterable[float] | None = None,
) -> float:
    _require_scipy()
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    pred_surface = binary_surface(pred)
    gt_surface = binary_surface(gt)
    pred_count = int(pred_surface.sum())
    gt_count = int(gt_surface.sum())
    if gt_count == 0:
        return math.nan
    if pred_count == 0:
        return 0.0
    spacing_tuple = _spacing_tuple(spacing, pred.ndim)
    dt_to_gt = distance_transform_edt(~gt_surface, sampling=spacing_tuple)
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing_tuple)
    close_pred = int((dt_to_gt[pred_surface] <= tolerance).sum())
    close_gt = int((dt_to_pred[gt_surface] <= tolerance).sum())
    return float((close_pred + close_gt) / max(pred_count + gt_count, 1))


def segmentation_surface_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4,
    tolerance: float = 2.0,
    spacing: Iterable[float] | None = None,
) -> dict[str, float]:
    """Compute foreground class HD95, ASSD, Boundary F1, and Surface Dice."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.ndim == 2:
        pred = pred[None]
        target = target[None]
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}")

    metrics: dict[str, float] = {}
    class_hd95 = []
    class_assd = []
    for cls in range(1, min(num_classes, 4)):
        name = CLASS_METRIC_NAMES.get(cls, f"class{cls}")
        hd_vals = []
        assd_vals = []
        bf1_vals = []
        sd_vals = []
        for idx in range(pred.shape[0]):
            pred_c = pred[idx] == cls
            gt_c = target[idx] == cls
            hd_vals.append(hd95_binary(pred_c, gt_c, spacing))
            assd_vals.append(assd_binary(pred_c, gt_c, spacing))
            bf1_vals.append(boundary_f1_binary(pred_c, gt_c, tolerance, spacing))
            sd_vals.append(surface_dice_binary(pred_c, gt_c, tolerance, spacing))
        metrics[f"hd95_{name}"] = _nanmean(hd_vals)
        metrics[f"assd_{name}"] = _nanmean(assd_vals)
        metrics[f"boundary_f1_{name}"] = _nanmean(bf1_vals)
        metrics[f"surface_dice_{name}"] = _nanmean(sd_vals)
        class_hd95.append(metrics[f"hd95_{name}"])
        class_assd.append(metrics[f"assd_{name}"])
    metrics["hd95_fg_mean"] = _nanmean(class_hd95)
    metrics["assd_fg_mean"] = _nanmean(class_assd)

    union_bf1 = []
    union_sd = []
    for idx in range(pred.shape[0]):
        union_bf1.append(boundary_f1_binary(pred[idx] > 0, target[idx] > 0, tolerance, spacing))
        union_sd.append(surface_dice_binary(pred[idx] > 0, target[idx] > 0, tolerance, spacing))
    metrics["boundary_f1_fg"] = _nanmean(union_bf1)
    metrics["surface_dice_fg"] = _nanmean(union_sd)
    return metrics


def _nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0 or np.isnan(arr).all():
        return math.nan
    return float(np.nanmean(arr))


def _spacing_tuple(spacing: Iterable[float] | None, ndim: int) -> tuple[float, ...] | None:
    if spacing is None:
        return None
    vals = tuple(float(v) for v in spacing)
    if len(vals) != ndim:
        raise ValueError(f"Expected spacing length {ndim}, got {vals}")
    return vals


def _image_diagonal(shape: tuple[int, ...], spacing: tuple[float, ...] | None) -> float:
    if spacing is None:
        spacing = tuple(1.0 for _ in shape)
    extents = [(max(s - 1, 1) * sp) for s, sp in zip(shape, spacing)]
    return float(np.linalg.norm(extents))


def _require_scipy() -> None:
    if not HAS_SCIPY:
        raise ImportError("Surface metrics require scipy.ndimage; install scipy or disable these metrics.")
