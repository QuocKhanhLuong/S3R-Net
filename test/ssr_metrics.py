"""Pixel-space segmentation metrics for SSR debug validation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    HAS_SCIPY = True
except Exception:  # pragma: no cover - depends on environment
    binary_erosion = None
    distance_transform_edt = None
    HAS_SCIPY = False


CLASS_METRIC_NAMES = {1: "rv", 2: "myo", 3: "lv"}


def binary_surface(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel binary surface for a 2D mask."""
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
    """Return pred-to-GT and GT-to-pred surface distances.

    If GT is non-empty but prediction is empty, distances are filled with the
    image diagonal so missed objects are penalized but do not crash logging.
    """
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
        return (
            np.full(1, penalty, dtype=np.float64),
            np.full(gt_count, penalty, dtype=np.float64),
        )

    dt_to_gt = distance_transform_edt(~gt_surface, sampling=spacing_tuple)
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing_tuple)
    return dt_to_gt[pred_surface].astype(np.float64), dt_to_pred[gt_surface].astype(np.float64)


def hd95_binary(pred: np.ndarray, gt: np.ndarray, spacing: Iterable[float] | None = None) -> float:
    """95th percentile symmetric Hausdorff distance for one binary mask."""
    p2g, g2p = binary_surface_distances(pred, gt, spacing)
    distances = np.concatenate([p2g, g2p])
    if distances.size == 0:
        return math.nan
    return float(np.percentile(distances, 95))


def assd_binary(pred: np.ndarray, gt: np.ndarray, spacing: Iterable[float] | None = None) -> float:
    """Average symmetric surface distance for one binary mask."""
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
    """Boundary F1 score for one binary mask."""
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
    """Surface Dice at a pixel/mm tolerance for one binary mask."""
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
    """Compute foreground class surface metrics for a batch of 2D predictions.

    Args:
        pred: Label map `[B,H,W]` or `[H,W]`.
        target: Label map with the same shape.
    """
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
        pred_fg = pred[idx] > 0
        gt_fg = target[idx] > 0
        union_bf1.append(boundary_f1_binary(pred_fg, gt_fg, tolerance, spacing))
        union_sd.append(surface_dice_binary(pred_fg, gt_fg, tolerance, spacing))
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
