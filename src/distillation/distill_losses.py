"""S3R-SCSD distillation losses."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.models.s3r.spectral_utils import build_radial_frequency_masks, radial_band_amplitudes

from .teacher_targets import region_routing_weights


DEFAULT_BAND_ALPHA = (
    (0.7, 0.3),
    (0.5, 0.5),
    (0.3, 0.7),
    (0.2, 0.8),
)


class S3RSCSDLoss(nn.Module):
    """Semantic-Characteristic Spectral-State Distillation loss."""

    def __init__(
        self,
        num_classes: int = 4,
        num_bands: int = 4,
        phase: str = "phase3_dual_routing",
        temperature: float = 2.0,
        loss_weights: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_bands = int(num_bands)
        self.phase = str(phase)
        self.temperature = float(temperature)
        self.loss_weights = dict(loss_weights or {})

    def forward(
        self,
        student: dict[str, Any],
        gt_mask: Tensor | None,
        teacher: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, float]]:
        seg_logits = student["seg_logits"]
        device = seg_logits.device
        dtype = seg_logits.dtype
        zero = torch.zeros((), device=device, dtype=dtype)
        routing = self._routing(teacher, gt_mask, device)

        semantic = zero
        if "t1_probs" in teacher and float(self.loss_weights.get("semantic_kd", 0.0)) > 0:
            semantic = self.semantic_kd(seg_logits, teacher["t1_probs"].to(device), routing.get("semantic"))

        boundary = zero
        if "t2_boundary" in teacher and float(self.loss_weights.get("boundary_kd", 0.0)) > 0:
            boundary = self.boundary_kd(student["boundary_logits"], teacher["t2_boundary"].to(device), routing.get("characteristic"))

        distance = zero
        if "t2_distance" in teacher and float(self.loss_weights.get("distance_kd", 0.0)) > 0:
            distance_pred = student.get("distance", student.get("distance_logits"))
            if distance_pred is not None:
                distance = self.distance_kd(distance_pred, teacher["t2_distance"].to(device), routing.get("characteristic"))

        spectral_boundary = zero
        if "t2_boundary" in teacher and float(self.loss_weights.get("spectral_boundary_kd", 0.0)) > 0:
            spectral_boundary = self.spectral_boundary_kd(student["boundary_logits"], teacher["t2_boundary"].to(device))

        state_kd = zero
        if float(self.loss_weights.get("state_kd", 0.0)) > 0 and "state" in student:
            state_kd = self.spectral_state_kd(student["state"], teacher)

        total = (
            float(self.loss_weights.get("semantic_kd", 0.0)) * semantic
            + float(self.loss_weights.get("boundary_kd", 0.0)) * boundary
            + float(self.loss_weights.get("distance_kd", 0.0)) * distance
            + float(self.loss_weights.get("spectral_boundary_kd", 0.0)) * spectral_boundary
            + float(self.loss_weights.get("state_kd", 0.0)) * state_kd
        )
        parts = {
            "semantic_kd": float(semantic.detach().cpu()),
            "boundary_kd": float(boundary.detach().cpu()),
            "distance_kd": float(distance.detach().cpu()),
            "spectral_boundary_kd": float(spectral_boundary.detach().cpu()),
            "state_kd": float(state_kd.detach().cpu()),
            "distill_loss": float(total.detach().cpu()),
            "agreement_mean": float(routing["agreement"].detach().mean().cpu()) if isinstance(routing.get("agreement"), Tensor) else 0.0,
        }
        return total, parts

    def semantic_kd(self, student_logits: Tensor, teacher_probs: Tensor, weight: Tensor | None = None) -> Tensor:
        if teacher_probs.shape[-2:] != student_logits.shape[-2:]:
            teacher_probs = F.interpolate(teacher_probs, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)
        if teacher_probs.shape[1] != self.num_classes:
            raise ValueError(f"Teacher probabilities must have {self.num_classes} classes, got {teacher_probs.shape[1]}")
        T = self.temperature
        log_p = F.log_softmax(student_logits / T, dim=1)
        teacher = teacher_probs.clamp_min(1e-8)
        loss = F.kl_div(log_p, teacher, reduction="none").sum(dim=1, keepdim=True) * (T * T)
        return _weighted_mean(loss, weight)

    def boundary_kd(self, student_boundary_logits: Tensor, teacher_boundary: Tensor, weight: Tensor | None = None) -> Tensor:
        if teacher_boundary.shape[-2:] != student_boundary_logits.shape[-2:]:
            teacher_boundary = F.interpolate(teacher_boundary, size=student_boundary_logits.shape[-2:], mode="bilinear", align_corners=False)
        bce = F.binary_cross_entropy_with_logits(student_boundary_logits, teacher_boundary.float(), reduction="none")
        dice = _soft_binary_dice_loss(torch.sigmoid(student_boundary_logits), teacher_boundary.float(), weight)
        return _weighted_mean(bce, weight) + dice

    def distance_kd(self, student_distance: Tensor, teacher_distance: Tensor, weight: Tensor | None = None) -> Tensor:
        if teacher_distance.shape[-2:] != student_distance.shape[-2:]:
            teacher_distance = F.interpolate(teacher_distance, size=student_distance.shape[-2:], mode="bilinear", align_corners=False)
        return _weighted_mean((student_distance - teacher_distance.float()).abs(), weight)

    def spectral_boundary_kd(self, student_boundary_logits: Tensor, teacher_boundary: Tensor) -> Tensor:
        student = torch.sigmoid(student_boundary_logits).float()
        teacher = teacher_boundary.float()
        if teacher.shape[-2:] != student.shape[-2:]:
            teacher = F.interpolate(teacher, size=student.shape[-2:], mode="bilinear", align_corners=False)
        _, _, H, W = student.shape
        student_fft = torch.fft.rfft2(student, norm="ortho").abs()
        teacher_fft = torch.fft.rfft2(teacher, norm="ortho").abs()
        masks = build_radial_frequency_masks(H, W, self.num_bands, student.device)
        band_loss = []
        for band in range(1, self.num_bands):
            mask = masks[band].view(1, 1, H, W // 2 + 1)
            band_loss.append(F.l1_loss(student_fft * mask, teacher_fft * mask))
        return torch.stack(band_loss).mean() if band_loss else F.l1_loss(student_fft, teacher_fft)

    def spectral_state_kd(self, student_state: Tensor, teacher: dict[str, Tensor]) -> Tensor:
        sem_source = teacher.get("t1_foreground")
        if sem_source is None and "t1_probs" in teacher:
            sem_source = teacher["t1_probs"][:, 1:].sum(dim=1, keepdim=True)
        char_source = teacher.get("t2_boundary")
        if sem_source is None or char_source is None:
            return torch.zeros((), device=student_state.device, dtype=student_state.dtype)
        sem_bands = radial_band_amplitudes(sem_source.to(student_state.device), self.num_bands)
        char_bands = radial_band_amplitudes(char_source.to(student_state.device), self.num_bands)
        alpha = torch.tensor(DEFAULT_BAND_ALPHA[: self.num_bands], device=student_state.device, dtype=student_state.dtype)
        if alpha.shape[0] < self.num_bands:
            tail = alpha[-1:].repeat(self.num_bands - alpha.shape[0], 1)
            alpha = torch.cat([alpha, tail], dim=0)
        target = alpha[:, 0].view(1, -1) * sem_bands + alpha[:, 1].view(1, -1) * char_bands
        student_summary = student_state.norm(dim=-1)
        student_summary = _normalize_bands(student_summary)
        target = _normalize_bands(target)
        return F.l1_loss(student_summary, target)

    def _routing(self, teacher: dict[str, Tensor], gt_mask: Tensor | None, device: torch.device) -> dict[str, Tensor | None]:
        agreement = teacher.get("agreement_weight")
        agreement_map = agreement.to(device).float() if agreement is not None else None
        if not bool(self.loss_weights.get("region_routing", False)):
            if bool(self.loss_weights.get("agreement_weighting", False)) and agreement_map is not None:
                return {"semantic": agreement_map, "characteristic": agreement_map, "uncertain": None, "agreement": agreement_map}
            return {"semantic": None, "characteristic": None, "uncertain": None, "agreement": agreement_map}
        if "t1_probs" not in teacher or "t2_boundary" not in teacher:
            return {"semantic": None, "characteristic": None, "uncertain": None, "agreement": agreement_map}
        routing = region_routing_weights(
            t1_probs=teacher["t1_probs"].to(device),
            t2_boundary=teacher["t2_boundary"].to(device),
            gt_mask=gt_mask.to(device) if gt_mask is not None else None,
            agreement_weight=agreement_map,
        )
        routing["agreement"] = agreement_map
        return routing


def _weighted_mean(loss: Tensor, weight: Tensor | None) -> Tensor:
    if weight is None:
        return loss.mean()
    if weight.shape[-2:] != loss.shape[-2:]:
        weight = F.interpolate(weight.float(), size=loss.shape[-2:], mode="bilinear", align_corners=False)
    while weight.ndim < loss.ndim:
        weight = weight.unsqueeze(1)
    weight = weight.float().to(loss.device)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def _soft_binary_dice_loss(pred: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
    if weight is not None:
        if weight.shape[-2:] != pred.shape[-2:]:
            weight = F.interpolate(weight.float(), size=pred.shape[-2:], mode="bilinear", align_corners=False)
        pred = pred * weight
        target = target * weight
    inter = (pred * target).sum(dim=(0, 2, 3))
    denom = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean()


def _normalize_bands(x: Tensor) -> Tensor:
    return x / x.sum(dim=1, keepdim=True).clamp_min(1e-8)
