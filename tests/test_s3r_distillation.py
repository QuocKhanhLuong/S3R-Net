from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from src.distillation.distill_losses import S3RSCSDLoss
from src.distillation.teacher_cache import load_teacher_cache, save_teacher_cache, validate_cache
from src.distillation.teacher_targets import (
    boundary_from_mask,
    distance_map_from_mask_or_boundary,
    region_routing_weights,
    semantic_entropy,
    soft_boundary_from_prob,
    spectral_boundary_target,
    teacher_agreement_weight,
)


def test_teacher_cache_round_trip_and_validation(tmp_path: Path) -> None:
    payload = {
        "t1_probs": np.zeros((4, 16, 16), dtype=np.float32),
        "t1_entropy": np.zeros((1, 16, 16), dtype=np.float32),
        "t1_foreground": np.zeros((1, 16, 16), dtype=np.float32),
        "t2_boundary": np.zeros((1, 16, 16), dtype=np.float32),
        "t2_distance": np.zeros((1, 16, 16), dtype=np.float32),
        "t2_foreground": np.zeros((1, 16, 16), dtype=np.float32),
        "agreement_weight": np.ones((1, 16, 16), dtype=np.float32),
    }

    path = save_teacher_cache(tmp_path, "patient001_ED", 3, payload, dataset="ACDC")
    loaded = load_teacher_cache(tmp_path, "patient001_ED", 3, dataset="ACDC")
    report = validate_cache(tmp_path, dataset="ACDC")

    assert path.exists()
    assert loaded["t1_probs"].shape == (4, 16, 16)
    assert report["valid_files"] == 1
    assert report["missing_required_fields"] == {}


def test_teacher_targets_produce_expected_shapes() -> None:
    mask = torch.zeros(2, 16, 16, dtype=torch.long)
    mask[:, 4:10, 4:10] = 1
    probs = torch.nn.functional.one_hot(mask, 4).permute(0, 3, 1, 2).float()
    boundary = boundary_from_mask(mask)
    soft_boundary = soft_boundary_from_prob(probs[:, 1:].sum(dim=1, keepdim=True))
    distance = distance_map_from_mask_or_boundary(mask)
    spectral = spectral_boundary_target(boundary, num_bands=4)
    entropy = semantic_entropy(probs)
    agreement = teacher_agreement_weight(probs[:, 1:].sum(dim=1, keepdim=True), boundary)
    routing = region_routing_weights(
        t1_probs=probs,
        t2_boundary=boundary,
        gt_mask=mask,
        agreement_weight=agreement,
    )

    assert boundary.shape == (2, 1, 16, 16)
    assert soft_boundary.shape == (2, 1, 16, 16)
    assert distance.shape == (2, 1, 16, 16)
    assert spectral.shape == (2, 4)
    assert entropy.shape == (2, 1, 16, 16)
    assert routing["semantic"].shape == (2, 1, 16, 16)
    assert routing["characteristic"].shape == (2, 1, 16, 16)
    assert routing["uncertain"].shape == (2, 1, 16, 16)


def test_s3r_scsd_loss_supports_dual_routing_and_state_kd() -> None:
    student = {
        "seg_logits": torch.randn(2, 4, 16, 16),
        "boundary_logits": torch.randn(2, 1, 16, 16),
        "distance": torch.randn(2, 1, 16, 16),
        "state": torch.randn(2, 4, 8),
    }
    gt = torch.randint(0, 4, (2, 16, 16))
    teacher = {
        "t1_probs": torch.softmax(torch.randn(2, 4, 16, 16), dim=1),
        "t1_entropy": torch.rand(2, 1, 16, 16),
        "t1_foreground": torch.rand(2, 1, 16, 16),
        "t2_boundary": torch.rand(2, 1, 16, 16),
        "t2_distance": torch.rand(2, 1, 16, 16),
        "t2_foreground": torch.rand(2, 1, 16, 16),
        "agreement_weight": torch.rand(2, 1, 16, 16),
    }
    loss_fn = S3RSCSDLoss(
        num_classes=4,
        num_bands=4,
        phase="phase4_state_kd",
        loss_weights={
            "semantic_kd": 0.3,
            "boundary_kd": 0.2,
            "distance_kd": 0.05,
            "spectral_boundary_kd": 0.1,
            "state_kd": 0.05,
            "agreement_weighting": True,
            "region_routing": True,
        },
    )

    loss, parts = loss_fn(student, gt, teacher)

    assert math.isfinite(float(loss.detach()))
    for key in ("semantic_kd", "boundary_kd", "distance_kd", "spectral_boundary_kd", "state_kd"):
        assert key in parts
        assert math.isfinite(parts[key])
