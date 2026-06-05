"""Agreement-aware dual-teacher KD losses for S3R."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from src.teachers.teacher_utils import (
    extract_boundary_band,
    normalize_probs,
    one_hot_to_boundary,
    resize_teacher_output_to_student,
)


_MISSING = object()


def js_divergence(p: Tensor, q: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    """Pixelwise Jensen-Shannon divergence between probability maps."""
    p = normalize_probs(p, eps=eps)
    q = normalize_probs(q, eps=eps)
    if p.shape != q.shape:
        raise ValueError(f"JS divergence requires matching shapes, got {tuple(p.shape)} vs {tuple(q.shape)}")
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=dim, keepdim=True)
    kl_qm = (q * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=dim, keepdim=True)
    return 0.5 * (kl_pm + kl_qm).clamp_min(0.0)


def agreement_map(p_medical_sam3: Tensor, p_cinema: Tensor, eps: float = 1e-8, normalize: bool = False) -> Tensor:
    """Return high values where both teachers agree."""
    js = js_divergence(p_medical_sam3, p_cinema, eps=eps)
    if normalize:
        max_js = js.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(eps)
        return (1.0 - js / max_js).clamp(0.0, 1.0)
    return torch.exp(-js).clamp(0.0, 1.0)


def agreement_aware_fusion(
    p_medical_sam3: Tensor,
    p_cinema: Tensor,
    c_medical_sam3: Tensor,
    c_cinema: Tensor,
    *,
    gt_mask: Tensor | None = None,
    cinema_boundary: Tensor | None = None,
    disable_agreement_weighting: bool = False,
    eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Fuse Medical-SAM3 field target and CineMA boundary/anatomy target."""
    p_m3 = normalize_probs(p_medical_sam3.detach(), eps=eps)
    p_c = normalize_probs(p_cinema.detach(), eps=eps)
    if p_c.shape[-2:] != p_m3.shape[-2:]:
        p_c = F.interpolate(p_c, size=p_m3.shape[-2:], mode="bilinear", align_corners=False)
        p_c = normalize_probs(p_c, eps=eps)

    c_m3 = _ensure_weight_map(c_medical_sam3.detach(), p_m3.shape[-2:], p_m3.device)
    c_c = _ensure_weight_map(c_cinema.detach(), p_m3.shape[-2:], p_m3.device)
    if gt_mask is not None:
        w_boundary = extract_boundary_band(gt_mask.to(p_m3.device), radius=3)
    elif cinema_boundary is not None:
        w_boundary = _ensure_weight_map(cinema_boundary.detach(), p_m3.shape[-2:], p_m3.device)
    else:
        w_boundary = one_hot_to_boundary(p_c, mode="soft", dilation=3)
    w_boundary = w_boundary.clamp(0.0, 1.0)
    w_interior = (1.0 - w_boundary).clamp(0.0, 1.0)

    agreement = agreement_map(p_m3, p_c)
    agreement_factor = 1.0 if disable_agreement_weighting else (0.5 + 0.5 * agreement)
    r_m3 = c_m3 * w_interior * agreement_factor
    r_c = c_c * w_boundary * agreement_factor
    denom = (r_m3 + r_c).clamp_min(eps)
    w_m3 = (r_m3 / denom).clamp(0.0, 1.0)
    w_c = (r_c / denom).clamp(0.0, 1.0)
    fused = normalize_probs(w_m3 * p_m3 + w_c * p_c, eps=eps)
    return {
        "P_F": fused.detach(),
        "agreement": agreement.detach(),
        "W_M3": w_m3.detach(),
        "W_C": w_c.detach(),
        "W_boundary": w_boundary.detach(),
        "W_interior": w_interior.detach(),
    }


def soft_kl_loss(
    student_logits: Tensor,
    teacher_probs: Tensor,
    temperature: float = 4.0,
    weight: Tensor | None = None,
) -> Tensor:
    """Temperature-scaled KL from student logits to detached teacher probs."""
    teacher = normalize_probs(teacher_probs.detach()).to(student_logits.device)
    if teacher.shape[-2:] != student_logits.shape[-2:]:
        teacher = F.interpolate(teacher, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)
        teacher = normalize_probs(teacher)
    T = float(temperature)
    log_p = F.log_softmax(student_logits / T, dim=1)
    loss = F.kl_div(log_p, teacher, reduction="none").sum(dim=1, keepdim=True) * (T * T)
    return _weighted_mean(loss, weight)


def segmentation_field_kd_loss(
    student_logits: Tensor,
    p_medical_sam3: Tensor,
    w_interior: Tensor,
    temperature: float = 4.0,
) -> Tensor:
    return soft_kl_loss(student_logits, p_medical_sam3, temperature=temperature, weight=w_interior)


def cinema_boundary_kd_loss(
    student_logits: Tensor,
    p_cinema: Tensor,
    w_boundary: Tensor,
    temperature: float = 4.0,
) -> Tensor:
    return soft_kl_loss(student_logits, p_cinema, temperature=temperature, weight=w_boundary)


def fused_kd_loss(student_logits: Tensor, fused_probs: Tensor, temperature: float = 4.0) -> Tensor:
    return soft_kl_loss(student_logits, fused_probs, temperature=temperature, weight=None)


def spectral_boundary_loss(
    student_probs_or_logits: Tensor,
    cinema_probs_or_boundary: Tensor,
    *,
    threshold: float = 0.35,
) -> Tensor:
    """Compare high-frequency amplitude spectra of student and CineMA boundaries."""
    student_probs = _to_probs(student_probs_or_logits)
    if cinema_probs_or_boundary.shape[1] == 1:
        cinema_boundary = cinema_probs_or_boundary.detach().float()
    else:
        cinema_boundary = one_hot_to_boundary(normalize_probs(cinema_probs_or_boundary.detach()), mode="soft", dilation=3)
    student_boundary = one_hot_to_boundary(student_probs, mode="soft", dilation=3)
    if cinema_boundary.shape[-2:] != student_boundary.shape[-2:]:
        cinema_boundary = F.interpolate(cinema_boundary, size=student_boundary.shape[-2:], mode="bilinear", align_corners=False)

    _, _, H, W = student_boundary.shape
    student_fft = torch.fft.fft2(student_boundary.float(), norm="ortho").abs()
    cinema_fft = torch.fft.fft2(cinema_boundary.float(), norm="ortho").abs()
    yy = torch.fft.fftfreq(H, device=student_boundary.device).view(H, 1)
    xx = torch.fft.fftfreq(W, device=student_boundary.device).view(1, W)
    radius = torch.sqrt(xx.square() + yy.square())
    hf_mask = (radius >= float(threshold)).float().view(1, 1, H, W)
    return F.l1_loss(student_fft * hf_mask, cinema_fft * hf_mask)


def compute_dual_teacher_kd_loss(
    student_outputs: dict[str, Any],
    teacher_outputs: dict[str, Tensor],
    gt_mask: Tensor,
    cfg: dict[str, Any],
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    """Compute optional dual-teacher KD terms and return scalar loss + logs."""
    logits = student_outputs["seg_logits"]
    device = logits.device
    zero = torch.zeros((), device=device, dtype=logits.dtype)
    kd = cfg.get("dual_teacher_kd", cfg)
    p_c = _pick(teacher_outputs, "P_C", "cinema_probs", "cine_probs", "probs_c").to(device)
    need_m3 = (
        bool(kd.get("use_vanilla_kd_only", False))
        or not bool(kd.get("disable_field_kd", False))
        or not bool(kd.get("disable_fused_kd", False))
    )
    p_m3 = _pick(
        teacher_outputs,
        "P_M3",
        "medical_sam3_probs",
        "m3_probs",
        "probs_m3",
        default=None if not need_m3 else _MISSING,
    )
    if p_m3 is None:
        p_m3 = p_c.detach()
    p_m3 = p_m3.to(device)
    c_m3 = _pick(teacher_outputs, "C_M3", "medical_sam3_confidence", "m3_confidence", default=torch.ones_like(p_m3[:, :1])).to(device)
    c_c = _pick(teacher_outputs, "C_C", "cinema_confidence", "cine_confidence", default=torch.ones_like(p_c[:, :1])).to(device)
    b_c = _pick(teacher_outputs, "B_C", "cinema_boundary", "boundary", default=None)
    if b_c is not None:
        b_c = b_c.to(device)

    p_m3 = resize_teacher_output_to_student(p_m3, logits.shape)
    p_c = resize_teacher_output_to_student(p_c, logits.shape)
    c_m3 = resize_teacher_output_to_student(c_m3, logits.shape)
    c_c = resize_teacher_output_to_student(c_c, logits.shape)
    b_c = resize_teacher_output_to_student(b_c, logits.shape) if b_c is not None else None

    fusion = agreement_aware_fusion(
        p_m3,
        p_c,
        c_m3,
        c_c,
        gt_mask=gt_mask,
        cinema_boundary=b_c,
        disable_agreement_weighting=bool(kd.get("disable_agreement_weighting", False)),
    )
    T = float(kd.get("kd_temperature", 4.0))
    field = zero if bool(kd.get("disable_field_kd", False)) else segmentation_field_kd_loss(logits, p_m3, fusion["W_interior"], T)
    cine_boundary = zero if bool(kd.get("disable_cine_boundary_kd", False)) else cinema_boundary_kd_loss(logits, p_c, fusion["W_boundary"], T)
    fuse = zero if bool(kd.get("disable_fused_kd", False)) else fused_kd_loss(logits, fusion["P_F"], T)
    spec = zero
    if not bool(kd.get("disable_spectral_kd", False)) and float(kd.get("lambda_spec", 0.05)) > 0:
        spec = spectral_boundary_loss(logits, b_c if b_c is not None else p_c)

    if bool(kd.get("use_vanilla_kd_only", False)):
        field = soft_kl_loss(logits, p_m3, temperature=T)
        cine_boundary = zero
        fuse = zero
        spec = zero

    total = (
        float(kd.get("lambda_field", 0.3)) * field
        + float(kd.get("lambda_cine_boundary", 0.5)) * cine_boundary
        + float(kd.get("lambda_fuse", 0.5)) * fuse
        + float(kd.get("lambda_spec", 0.05)) * spec
    )
    parts = {
        "loss_field": float(field.detach().cpu()),
        "loss_cine_boundary": float(cine_boundary.detach().cpu()),
        "loss_fuse": float(fuse.detach().cpu()),
        "loss_spec": float(spec.detach().cpu()),
        "loss_kd": float(total.detach().cpu()),
        "agreement_mean": float(fusion["agreement"].mean().detach().cpu()),
        "W_M3_mean": float(fusion["W_M3"].mean().detach().cpu()),
        "W_C_mean": float(fusion["W_C"].mean().detach().cpu()),
    }
    return total, parts, fusion


def _to_probs(x: Tensor) -> Tensor:
    if x.shape[1] == 1:
        fg = x.sigmoid() if x.min() < 0 or x.max() > 1 else x.clamp(0.0, 1.0)
        return torch.cat([1.0 - fg, fg], dim=1)
    if x.min() < 0 or x.max() > 1.0:
        return torch.softmax(x, dim=1)
    return normalize_probs(x)


def _ensure_weight_map(weight: Tensor, spatial: tuple[int, int], device: torch.device) -> Tensor:
    weight = weight.float().to(device)
    if weight.ndim == 3:
        weight = weight.unsqueeze(1)
    if weight.shape[-2:] != spatial:
        weight = F.interpolate(weight, size=spatial, mode="bilinear", align_corners=False)
    return weight.clamp(0.0, 1.0)


def _weighted_mean(loss: Tensor, weight: Tensor | None) -> Tensor:
    if weight is None:
        return loss.mean()
    weight = _ensure_weight_map(weight, loss.shape[-2:], loss.device)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def _pick(payload: dict[str, Tensor], *keys: str, default: Tensor | object = _MISSING) -> Tensor | None:
    for key in keys:
        if key in payload:
            return payload[key].detach()
    if default is not _MISSING:
        if default is None:
            return None
        return default.detach()
    raise KeyError(f"Teacher output missing one of required keys: {keys}")
