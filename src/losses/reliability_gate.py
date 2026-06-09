"""Learned pixel-wise reliability gate for dual-teacher KD."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.teachers.teacher_utils import normalize_probs, one_hot_to_boundary


class StudentAwareReliabilityGate(nn.Module):
    """Small CNN that predicts per-pixel teacher reliability maps.

    The gate consumes detached teacher outputs and detached student uncertainty.
    Its outputs remain differentiable with respect to the gate parameters, so KD
    loss can train the gate without backpropagating through teachers or through
    student uncertainty construction.
    """

    def __init__(
        self,
        num_classes: int = 4,
        hidden_channels: int = 32,
        use_student_uncertainty: bool = True,
        student_uncertainty_warmup_epochs: int = 10,
        use_ignore: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.use_student_uncertainty = bool(use_student_uncertainty)
        self.student_uncertainty_warmup_epochs = int(student_uncertainty_warmup_epochs)
        self.use_ignore = bool(use_ignore)
        in_channels = 2 * self.num_classes + 14
        out_channels = 3 if self.use_ignore else 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(self.hidden_channels), self.hidden_channels),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(self.hidden_channels), self.hidden_channels),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, max(self.hidden_channels // 2, 8), kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(max(self.hidden_channels // 2, 8)), max(self.hidden_channels // 2, 8)),
            nn.GELU(),
            nn.Conv2d(max(self.hidden_channels // 2, 8), out_channels, kernel_size=1),
        )

    def forward(
        self,
        *,
        student_logits: Tensor,
        p_semantic_teacher: Tensor,
        p_cinema: Tensor,
        c_semantic_teacher: Tensor,
        c_cinema: Tensor,
        teacher_boundary: Tensor | None,
        agreement: Tensor,
        epoch: int | None = None,
    ) -> dict[str, Tensor | bool]:
        p_m3 = normalize_probs(p_semantic_teacher.detach())
        p_c = normalize_probs(p_cinema.detach())
        spatial = p_m3.shape[-2:]
        p_c = _resize_probs(p_c, spatial)
        c_m3 = _ensure_map(c_semantic_teacher.detach(), spatial, p_m3.device)
        c_c = _ensure_map(c_cinema.detach(), spatial, p_m3.device)
        boundary = (
            _ensure_map(teacher_boundary.detach(), spatial, p_m3.device)
            if teacher_boundary is not None
            else one_hot_to_boundary(p_m3, mode="soft", dilation=3)
        )
        agreement_map = _ensure_map(agreement.detach(), spatial, p_m3.device).clamp(0.0, 1.0)
        disagreement = (1.0 - agreement_map).clamp(0.0, 1.0)

        fg_m3 = p_m3[:, 1:].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        fg_c = p_c[:, 1:].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        abs_fg_diff = (fg_m3 - fg_c).abs()
        entropy_m3 = _normalized_entropy(p_m3)
        entropy_c = _normalized_entropy(p_c)
        margin_m3 = _prediction_margin(p_m3)
        margin_c = _prediction_margin(p_c)

        uncertainty_enabled = bool(self.use_student_uncertainty) and int(epoch or 0) > self.student_uncertainty_warmup_epochs
        if uncertainty_enabled:
            student_probs = torch.softmax(student_logits.detach().float(), dim=1)
            student_entropy = _normalized_entropy(student_probs)
            student_margin = _prediction_margin(student_probs)
            student_boundary = one_hot_to_boundary(student_probs, mode="soft", dilation=3)
        else:
            student_entropy = torch.zeros_like(disagreement)
            student_margin = torch.zeros_like(disagreement)
            student_boundary = torch.zeros_like(disagreement)

        features = torch.cat(
            [
                p_m3,
                p_c,
                fg_m3,
                fg_c,
                entropy_m3,
                entropy_c,
                margin_m3,
                margin_c,
                c_m3,
                c_c,
                boundary,
                disagreement,
                abs_fg_diff,
                student_entropy,
                student_margin,
                student_boundary,
            ],
            dim=1,
        )
        logits = self.net(features)
        weights = torch.softmax(logits, dim=1)
        if self.use_ignore:
            w_sem = weights[:, 0:1]
            w_char = weights[:, 1:2]
            w_ignore = weights[:, 2:3]
        else:
            w_sem = weights[:, 0:1]
            w_char = weights[:, 1:2]
            w_ignore = torch.zeros_like(w_sem)

        gate_entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        return {
            "W_sem": w_sem,
            "W_char": w_char,
            "W_ignore": w_ignore,
            "gate_entropy": gate_entropy,
            "student_entropy": student_entropy.detach(),
            "student_margin": student_margin.detach(),
            "student_uncertainty_enabled": uncertainty_enabled,
        }

    def estimate_flops(self, image_size: int) -> int:
        """Approximate multiply-adds for one square image."""
        h = int(image_size)
        spatial = h * h
        flops = 0
        in_channels = 2 * self.num_classes + 14
        channels = [self.hidden_channels, self.hidden_channels, max(self.hidden_channels // 2, 8)]
        last = in_channels
        for out in channels:
            flops += spatial * last * out * 3 * 3
            last = out
        flops += spatial * last * (3 if self.use_ignore else 2)
        return int(flops)


def gate_regularization_loss(
    gate_output: dict[str, Tensor | bool],
    analytic_fusion: dict[str, Tensor],
    cfg: dict[str, Any],
    *,
    epoch: int | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Return optional regularization loss for learned gate maps."""
    w_sem = gate_output["W_sem"]
    w_char = gate_output["W_char"]
    w_ignore = gate_output["W_ignore"]
    assert isinstance(w_sem, Tensor) and isinstance(w_char, Tensor) and isinstance(w_ignore, Tensor)
    zero = w_sem.sum() * 0.0
    prior_weight = float(cfg.get("lambda_gate_prior", 0.0) or 0.0)
    decay_epochs = max(0, int(cfg.get("gate_prior_decay_epochs", 0) or 0))
    if decay_epochs > 0 and epoch is not None:
        prior_weight *= max(0.0, 1.0 - float(max(epoch - 1, 0)) / float(decay_epochs))
    prior = zero
    if prior_weight > 0:
        target_sem = analytic_fusion["W_C"].to(device=w_sem.device, dtype=w_sem.dtype).detach()
        target_char = analytic_fusion["W_M3"].to(device=w_char.device, dtype=w_char.dtype).detach()
        prior = F.l1_loss(w_sem, target_sem) + F.l1_loss(w_char, target_char)

    ignore = w_ignore.mean()
    tv = _total_variation(w_sem) + _total_variation(w_char) + _total_variation(w_ignore)
    loss = prior_weight * prior + float(cfg.get("lambda_ignore_sparsity", 0.0) or 0.0) * ignore + float(cfg.get("lambda_gate_tv", 0.0) or 0.0) * tv
    parts = {
        "loss_gate_prior": float((prior_weight * prior).detach().cpu()),
        "loss_gate_ignore": float((float(cfg.get("lambda_ignore_sparsity", 0.0) or 0.0) * ignore).detach().cpu()),
        "loss_gate_tv": float((float(cfg.get("lambda_gate_tv", 0.0) or 0.0) * tv).detach().cpu()),
    }
    return loss, parts


def _normalized_entropy(probs: Tensor, eps: float = 1e-8) -> Tensor:
    entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=1, keepdim=True)
    return (entropy / math.log(max(int(probs.shape[1]), 2))).clamp(0.0, 1.0)


def _prediction_margin(probs: Tensor) -> Tensor:
    top2 = torch.topk(probs, k=min(2, int(probs.shape[1])), dim=1).values
    if top2.shape[1] == 1:
        return top2[:, :1]
    return (top2[:, :1] - top2[:, 1:2]).clamp(0.0, 1.0)


def _ensure_map(weight: Tensor, spatial: tuple[int, int], device: torch.device) -> Tensor:
    weight = weight.float().to(device)
    if weight.ndim == 3:
        weight = weight.unsqueeze(1)
    if weight.shape[-2:] != spatial:
        weight = F.interpolate(weight, size=spatial, mode="bilinear", align_corners=False)
    return weight.clamp(0.0, 1.0)


def _resize_probs(probs: Tensor, spatial: tuple[int, int]) -> Tensor:
    if probs.shape[-2:] != spatial:
        probs = F.interpolate(probs, size=spatial, mode="bilinear", align_corners=False)
    return normalize_probs(probs)


def _total_variation(x: Tensor) -> Tensor:
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dx + dy


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
