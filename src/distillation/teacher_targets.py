"""Teacher-target generation and routing maps for S3R-SCSD."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from src.models.s3r.spectral_utils import radial_band_amplitudes

try:
    from scipy.ndimage import distance_transform_edt

    HAS_SCIPY = True
except Exception:  # pragma: no cover - depends on local environment
    distance_transform_edt = None
    HAS_SCIPY = False


def boundary_from_mask(mask: Tensor) -> Tensor:
    """Create a foreground boundary map from label masks."""
    if mask.ndim == 4:
        mask_2d = mask[:, 0]
    elif mask.ndim == 3:
        mask_2d = mask
    else:
        raise ValueError(f"Expected mask [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}")
    mask_2d = mask_2d.long()
    boundary = torch.zeros_like(mask_2d, dtype=torch.bool)
    boundary[:, :, 1:] |= mask_2d[:, :, 1:] != mask_2d[:, :, :-1]
    boundary[:, :, :-1] |= mask_2d[:, :, 1:] != mask_2d[:, :, :-1]
    boundary[:, 1:, :] |= mask_2d[:, 1:, :] != mask_2d[:, :-1, :]
    boundary[:, :-1, :] |= mask_2d[:, 1:, :] != mask_2d[:, :-1, :]
    boundary &= mask_2d > 0
    return boundary.unsqueeze(1).float()


def soft_boundary_from_prob(prob: Tensor, kernel_size: int = 3) -> Tensor:
    """Approximate a soft boundary from a foreground probability map."""
    if prob.ndim == 3:
        prob = prob.unsqueeze(1)
    if prob.ndim != 4:
        raise ValueError(f"Expected probability [B,1,H,W], got {tuple(prob.shape)}")
    pad = kernel_size // 2
    high = F.max_pool2d(prob, kernel_size=kernel_size, stride=1, padding=pad)
    low = -F.max_pool2d(-prob, kernel_size=kernel_size, stride=1, padding=pad)
    return (high - low).clamp(0.0, 1.0)


def distance_map_from_mask_or_boundary(mask_or_boundary: Tensor) -> Tensor:
    """Return normalized distance-to-boundary maps in `[0,1]`."""
    if mask_or_boundary.ndim == 3:
        boundary = boundary_from_mask(mask_or_boundary)
    elif mask_or_boundary.ndim == 4 and mask_or_boundary.shape[1] == 1:
        boundary = mask_or_boundary.float()
    else:
        raise ValueError(f"Expected mask [B,H,W] or boundary [B,1,H,W], got {tuple(mask_or_boundary.shape)}")

    if not HAS_SCIPY:
        return _torch_distance_fallback(boundary)

    arrays = []
    for item in boundary.detach().cpu().numpy()[:, 0]:
        b = item > 0.5
        if not b.any():
            arrays.append(np.zeros_like(item, dtype=np.float32))
            continue
        dist = distance_transform_edt(~b)
        dist = dist / max(float(dist.max()), 1e-6)
        arrays.append(dist.astype(np.float32))
    return torch.from_numpy(np.stack(arrays, axis=0)).unsqueeze(1).to(boundary.device)


def spectral_boundary_target(boundary: Tensor, num_bands: int = 4) -> Tensor:
    """Return teacher boundary amplitude per spectral band `[B,K]`."""
    return radial_band_amplitudes(boundary.float(), num_bands)


def semantic_entropy(probs: Tensor) -> Tensor:
    """Return normalized semantic entropy `[B,1,H,W]`."""
    if probs.ndim != 4:
        raise ValueError(f"Expected probs [B,C,H,W], got {tuple(probs.shape)}")
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
    return entropy / math.log(max(probs.shape[1], 2))


def teacher_agreement_weight(t1_foreground: Tensor, t2_foreground: Tensor) -> Tensor:
    """High where semantic and characteristic teachers agree on foreground."""
    if t1_foreground.ndim == 3:
        t1_foreground = t1_foreground.unsqueeze(1)
    if t2_foreground.ndim == 3:
        t2_foreground = t2_foreground.unsqueeze(1)
    if t1_foreground.shape[-2:] != t2_foreground.shape[-2:]:
        t2_foreground = F.interpolate(t2_foreground.float(), size=t1_foreground.shape[-2:], mode="bilinear", align_corners=False)
    return (1.0 - (t1_foreground.float() - t2_foreground.float()).abs()).clamp(0.0, 1.0)


def region_routing_weights(
    *,
    t1_probs: Tensor,
    t2_boundary: Tensor,
    gt_mask: Tensor | None = None,
    agreement_weight: Tensor | None = None,
    dilation: int = 5,
) -> dict[str, Tensor]:
    """Compute semantic, characteristic, and uncertainty routing maps."""
    if t2_boundary.ndim == 3:
        t2_boundary = t2_boundary.unsqueeze(1)
    entropy = semantic_entropy(t1_probs)
    if gt_mask is not None:
        boundary_region = boundary_from_mask(gt_mask).to(t1_probs.device)
    else:
        boundary_region = (t2_boundary > 0.25).float().to(t1_probs.device)
    pad = dilation // 2
    boundary_region = F.max_pool2d(boundary_region.float(), kernel_size=dilation, stride=1, padding=pad).clamp(0.0, 1.0)
    if agreement_weight is None:
        agreement_weight = torch.ones_like(boundary_region)
    agreement_weight = agreement_weight.to(t1_probs.device).float()
    if agreement_weight.shape[-2:] != boundary_region.shape[-2:]:
        agreement_weight = F.interpolate(agreement_weight, size=boundary_region.shape[-2:], mode="bilinear", align_corners=False)

    semantic = (1.0 - boundary_region) * (1.0 - entropy).clamp(0.0, 1.0)
    characteristic = boundary_region * agreement_weight
    uncertain = ((1.0 - agreement_weight) + entropy).mul(0.5).clamp(0.0, 1.0)
    return {
        "semantic": semantic.clamp(0.0, 1.0),
        "characteristic": characteristic.clamp(0.0, 1.0),
        "uncertain": uncertain,
    }


def _torch_distance_fallback(boundary: Tensor) -> Tensor:
    """Cheap fallback distance proxy used only when scipy is unavailable."""
    inv = 1.0 - (boundary > 0.5).float()
    smooth = F.avg_pool2d(inv, kernel_size=9, stride=1, padding=4)
    return smooth.clamp(0.0, 1.0)
