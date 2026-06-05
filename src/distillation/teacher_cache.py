"""Offline teacher-cache utilities for S3R-SCSD."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FIELDS = (
    "t1_probs",
    "t1_entropy",
    "t1_foreground",
    "t2_boundary",
    "t2_distance",
    "t2_foreground",
    "agreement_weight",
)

EXPECTED_CHANNELS = {
    "t1_probs": 4,
    "t1_entropy": 1,
    "t1_foreground": 1,
    "t2_boundary": 1,
    "t2_distance": 1,
    "t2_foreground": 1,
    "agreement_weight": 1,
}


def cache_file_path(
    cache_root: str | Path,
    case_id: str,
    slice_idx: int,
    dataset: str = "ACDC",
) -> Path:
    safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_id))
    return Path(cache_root) / str(dataset) / f"{safe_case}_{int(slice_idx):04d}.npz"


def save_teacher_cache(
    cache_root: str | Path,
    case_id: str,
    slice_idx: int,
    payload: dict[str, Any],
    dataset: str = "ACDC",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save one per-sample teacher-cache `.npz` file."""
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Teacher cache payload missing required fields: {missing}")
    shape_errors = _shape_errors(payload)
    if shape_errors:
        raise ValueError(f"Teacher cache payload has invalid shapes: {shape_errors}")
    path = cache_file_path(cache_root, case_id, slice_idx, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in payload.items()}
    if metadata is not None:
        arrays["metadata_json"] = np.asarray(json.dumps(metadata), dtype=object)
    np.savez_compressed(path, **arrays)
    return path


def load_teacher_cache(
    cache_root: str | Path,
    case_id: str,
    slice_idx: int,
    dataset: str = "ACDC",
    require_all: bool = True,
) -> dict[str, np.ndarray]:
    """Load one per-sample teacher-cache file."""
    path = cache_file_path(cache_root, case_id, slice_idx, dataset)
    if not path.exists():
        raise FileNotFoundError(f"Teacher cache file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    if require_all:
        missing = [field for field in REQUIRED_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"Teacher cache file {path} missing required fields: {missing}")
    return payload


def validate_cache(cache_root: str | Path, dataset: str = "ACDC") -> dict[str, Any]:
    """Validate cache files and report missing required fields."""
    root = Path(cache_root) / str(dataset)
    files = sorted(root.glob("*.npz"))
    missing_required: dict[str, list[str]] = {}
    invalid_shapes: dict[str, dict[str, str]] = {}
    invalid_files: dict[str, str] = {}
    valid = 0
    for path in files:
        try:
            with np.load(path, allow_pickle=True) as data:
                payload = {key: data[key] for key in data.files}
                fields = set(payload)
            missing = [field for field in REQUIRED_FIELDS if field not in fields]
            if missing:
                missing_required[path.name] = missing
            shapes = _shape_errors({key: payload[key] for key in payload if key in EXPECTED_CHANNELS})
            if shapes:
                invalid_shapes[path.name] = shapes
            if not missing and not shapes:
                valid += 1
        except Exception as exc:
            invalid_files[path.name] = str(exc)
    return {
        "cache_root": str(root),
        "total_files": len(files),
        "valid_files": valid,
        "missing_required_fields": missing_required,
        "invalid_shapes": invalid_shapes,
        "invalid_files": invalid_files,
    }


def _shape_errors(payload: dict[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    spatial: tuple[int, int] | None = None
    for field, channels in EXPECTED_CHANNELS.items():
        if field not in payload:
            continue
        arr = np.asarray(payload[field])
        if arr.ndim != 3:
            errors[field] = f"expected [C,H,W], got {arr.shape}"
            continue
        if arr.shape[0] != channels:
            errors[field] = f"expected channel {channels}, got {arr.shape[0]}"
            continue
        if spatial is None:
            spatial = (int(arr.shape[-2]), int(arr.shape[-1]))
        elif spatial != (int(arr.shape[-2]), int(arr.shape[-1])):
            errors[field] = f"expected spatial {spatial}, got {arr.shape[-2:]}"
    return errors
