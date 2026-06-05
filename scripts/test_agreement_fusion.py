#!/usr/bin/env python3
"""Standalone sanity test for agreement-aware teacher fusion."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses.agreement_kd import agreement_aware_fusion, compute_dual_teacher_kd_loss


def main() -> None:
    torch.manual_seed(7)
    B, C, H, W = 2, 4, 32, 32
    p_m3 = torch.softmax(torch.randn(B, C, H, W), dim=1)
    p_c = torch.softmax(torch.randn(B, C, H, W), dim=1)
    c_m3 = torch.rand(B, 1, H, W)
    c_c = torch.rand(B, 1, H, W)
    gt = torch.randint(0, C, (B, H, W))
    fusion = agreement_aware_fusion(p_m3, p_c, c_m3, c_c, gt_mask=gt)
    assert fusion["P_F"].shape == (B, C, H, W)
    assert torch.allclose(fusion["P_F"].sum(dim=1), torch.ones(B, H, W), atol=1e-5)
    for key in ("agreement", "W_M3", "W_C", "W_boundary", "W_interior"):
        value = fusion[key]
        assert torch.isfinite(value).all(), key
        assert value.min() >= 0.0 and value.max() <= 1.0, key

    student = {"seg_logits": torch.randn(B, C, H, W)}
    teacher = {"P_M3": p_m3, "C_M3": c_m3, "P_C": p_c, "C_C": c_c}
    loss, parts, _ = compute_dual_teacher_kd_loss(student, teacher, gt, {"dual_teacher_kd": {}})
    assert torch.isfinite(loss).all()
    assert parts["loss_kd"] >= 0.0
    print("agreement fusion sanity checks passed")


if __name__ == "__main__":
    main()
