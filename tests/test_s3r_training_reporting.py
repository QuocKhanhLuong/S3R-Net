from __future__ import annotations

import math

import torch

from src.training.train_s3r_acdc import (
    CLASS_NAMES,
    append_classification_metrics,
    format_class_metric_table,
)


def test_append_classification_metrics_adds_dice_precision_and_recall() -> None:
    metrics: dict[str, float] = {}
    tp = torch.tensor([4, 3, 0, 2], dtype=torch.float32)
    fp = torch.tensor([0, 1, 0, 2], dtype=torch.float32)
    fn = torch.tensor([0, 1, 0, 0], dtype=torch.float32)

    append_classification_metrics(metrics, tp, fp, fn, CLASS_NAMES)

    assert metrics["dice_RV"] == 0.75
    assert metrics["precision_RV"] == 0.75
    assert metrics["recall_RV"] == 0.75
    assert metrics["precision_MYO"] == 1.0
    assert metrics["recall_MYO"] == 1.0
    assert metrics["dice_LV"] == 2.0 / 3.0


def test_format_class_metric_table_uses_primary_columns_only() -> None:
    metrics = {
        "dice_RV": 0.8,
        "precision_RV": 0.7,
        "recall_RV": 0.9,
        "hd95_rv": 12.3,
        "assd_rv": 1.2,
        "dice_MYO": math.nan,
        "precision_MYO": math.nan,
        "recall_MYO": math.nan,
        "hd95_myo": math.nan,
        "assd_myo": math.nan,
        "dice_LV": 0.6,
        "precision_LV": 0.5,
        "recall_LV": 0.75,
        "hd95_lv": 9.0,
        "assd_lv": 0.9,
    }

    table = format_class_metric_table(metrics)

    assert "Class" in table
    assert "Dice" in table
    assert "HD95" in table
    assert "Precision" in table
    assert "Recall" in table
    assert "ASSD" in table
    assert "RV" in table
    assert "MYO" in table
    assert "LV" in table
    assert "boundary" not in table.lower()
