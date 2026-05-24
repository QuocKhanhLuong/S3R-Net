from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from ssr_metrics import (  # noqa: E402
    assd_binary,
    boundary_f1_binary,
    hd95_binary,
    segmentation_surface_metrics,
    surface_dice_binary,
)


def test_surface_metrics_perfect_overlap() -> None:
    pred = np.zeros((16, 16), dtype=bool)
    gt = np.zeros((16, 16), dtype=bool)
    pred[4:10, 5:12] = True
    gt[4:10, 5:12] = True

    assert hd95_binary(pred, gt) == 0.0
    assert assd_binary(pred, gt) == 0.0
    assert boundary_f1_binary(pred, gt, tolerance=1) == 1.0
    assert surface_dice_binary(pred, gt, tolerance=1) == 1.0


def test_surface_metrics_penalize_missing_prediction() -> None:
    pred = np.zeros((16, 16), dtype=bool)
    gt = np.zeros((16, 16), dtype=bool)
    gt[4:10, 5:12] = True

    assert math.isfinite(hd95_binary(pred, gt))
    assert hd95_binary(pred, gt) > 0.0
    assert assd_binary(pred, gt) > 0.0
    assert boundary_f1_binary(pred, gt, tolerance=2) == 0.0
    assert surface_dice_binary(pred, gt, tolerance=2) == 0.0


def test_segmentation_surface_metrics_class_and_foreground_union() -> None:
    pred = np.zeros((1, 8, 8), dtype=np.int64)
    gt = np.zeros((1, 8, 8), dtype=np.int64)
    pred[0, 2:5, 2:5] = 1
    gt[0, 2:5, 2:5] = 1

    metrics = segmentation_surface_metrics(pred, gt, num_classes=4, tolerance=1)

    assert metrics["hd95_rv"] == 0.0
    assert metrics["assd_rv"] == 0.0
    assert metrics["boundary_f1_fg"] == 1.0
    assert metrics["surface_dice_fg"] == 1.0
    assert math.isnan(metrics["hd95_myo"])
