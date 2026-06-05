"""Losses for supervised S3R training."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .spectral_utils import build_radial_frequency_masks


def boundary_map_from_mask(mask: Tensor) -> Tensor:
    """Create a one-pixel foreground boundary map from a label mask."""
    if mask.ndim == 4:
        mask_2d = mask[:, 0]
    elif mask.ndim == 3:
        mask_2d = mask
    else:
        raise ValueError(f"Expected mask rank 3 or 4, got shape {tuple(mask.shape)}")
    mask_2d = mask_2d.long()
    boundary = torch.zeros_like(mask_2d, dtype=torch.bool)
    boundary[:, :, 1:] |= mask_2d[:, :, 1:] != mask_2d[:, :, :-1]
    boundary[:, :, :-1] |= mask_2d[:, :, 1:] != mask_2d[:, :, :-1]
    boundary[:, 1:, :] |= mask_2d[:, 1:, :] != mask_2d[:, :-1, :]
    boundary[:, :-1, :] |= mask_2d[:, 1:, :] != mask_2d[:, :-1, :]
    boundary &= mask_2d > 0
    return boundary.unsqueeze(1).float()


def foreground_dice_loss(logits: Tensor, target: Tensor, num_classes: int) -> Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - dice[1:].mean() if num_classes > 1 else 1.0 - dice.mean()


def boundary_dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    pred = torch.sigmoid(logits)
    target = target.float()
    inter = (pred * target).sum(dim=(0, 2, 3))
    denom = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - dice.mean()


def boundary_frequency_loss(logits: Tensor, target: Tensor, num_bands: int = 4) -> Tensor:
    pred = torch.sigmoid(logits).float()
    target = target.float()
    _, _, H, W = pred.shape
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    masks = build_radial_frequency_masks(H, W, num_bands, pred.device)
    high_mask = masks[max(num_bands - 2, 0):].sum(dim=0).clamp(0, 1).view(1, 1, H, W // 2 + 1)
    return F.l1_loss(pred_fft.abs() * high_mask, target_fft.abs() * high_mask)


def total_variation_loss(foreground_prob: Tensor) -> Tensor:
    dx = (foreground_prob[:, :, :, 1:] - foreground_prob[:, :, :, :-1]).abs().mean()
    dy = (foreground_prob[:, :, 1:, :] - foreground_prob[:, :, :-1, :]).abs().mean()
    return dx + dy


def state_regularization(outputs: dict[str, Any]) -> Tensor:
    state = outputs.get("state")
    if state is None:
        logits = outputs["seg_logits"]
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    loss = state.square().mean() * 0.0
    raw = outputs.get("logs", {}).get("state_raw", {})
    for item in raw.values():
        delta = item.get("state_delta") if isinstance(item, dict) else None
        if isinstance(delta, Tensor):
            loss = loss + delta.mean()
    centered = state - state.mean(dim=1, keepdim=True)
    diversity = -centered.square().mean()
    return loss + 0.01 * diversity


class S3RLoss(nn.Module):
    """Default supervised S3R objective."""

    def __init__(
        self,
        num_classes: int = 4,
        num_bands: int = 4,
        boundary_bce_weight: float = 0.50,
        boundary_dice_weight: float = 0.30,
        boundary_freq_weight: float = 0.20,
        tv_weight: float = 0.05,
        gate_reg_weight: float = 0.03,
        hf_ratio_weight: float = 0.005,
        state_reg_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_bands = int(num_bands)
        self.boundary_bce_weight = float(boundary_bce_weight)
        self.boundary_dice_weight = float(boundary_dice_weight)
        self.boundary_freq_weight = float(boundary_freq_weight)
        self.tv_weight = float(tv_weight)
        self.gate_reg_weight = float(gate_reg_weight)
        self.hf_ratio_weight = float(hf_ratio_weight)
        self.state_reg_weight = float(state_reg_weight)

    def forward(
        self,
        outputs: dict[str, Any],
        mask: Tensor,
        boundary_target: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        seg_logits = outputs["seg_logits"]
        boundary_logits = outputs["boundary_logits"]
        if boundary_target is None:
            boundary_target = boundary_map_from_mask(mask).to(seg_logits.device)
        else:
            boundary_target = boundary_target.float().to(seg_logits.device)

        ce = F.cross_entropy(seg_logits, mask.long())
        dice = foreground_dice_loss(seg_logits, mask, self.num_classes)
        boundary_bce = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
        bdice = boundary_dice_loss(boundary_logits, boundary_target)
        bfreq = boundary_frequency_loss(boundary_logits, boundary_target, self.num_bands)
        probs = torch.softmax(seg_logits, dim=1)
        tv = total_variation_loss(probs[:, 1:].sum(dim=1, keepdim=True))
        gate_reg = outputs.get("gate_reg", torch.zeros((), device=seg_logits.device, dtype=seg_logits.dtype))
        hf_penalty = outputs.get("hf_ratio_penalty", torch.zeros((), device=seg_logits.device, dtype=seg_logits.dtype))
        state_reg = state_regularization(outputs)
        loss = (
            ce
            + dice
            + self.boundary_bce_weight * boundary_bce
            + self.boundary_dice_weight * bdice
            + self.boundary_freq_weight * bfreq
            + self.tv_weight * tv
            + self.gate_reg_weight * gate_reg
            + self.hf_ratio_weight * hf_penalty
            + self.state_reg_weight * state_reg
        )
        parts = {
            "ce": float(ce.detach().cpu()),
            "dice": float(dice.detach().cpu()),
            "boundary_bce": float(boundary_bce.detach().cpu()),
            "boundary_dice": float(bdice.detach().cpu()),
            "boundary_frequency": float(bfreq.detach().cpu()),
            "tv": float(tv.detach().cpu()),
            "gate_reg": float(gate_reg.detach().cpu()),
            "hf_ratio_penalty": float(hf_penalty.detach().cpu()),
            "state_reg": float(state_reg.detach().cpu()),
            "loss": float(loss.detach().cpu()),
        }
        return loss, parts
