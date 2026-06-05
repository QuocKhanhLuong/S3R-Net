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
    invalid_files: dict[str, str] = {}
    valid = 0
    for path in files:
        try:
            with np.load(path, allow_pickle=True) as data:
                fields = set(data.files)
            missing = [field for field in REQUIRED_FIELDS if field not in fields]
            if missing:
                missing_required[path.name] = missing
            else:
                valid += 1
        except Exception as exc:
            invalid_files[path.name] = str(exc)
    return {
        "cache_root": str(root),
        "total_files": len(files),
        "valid_files": valid,
        "missing_required_fields": missing_required,
        "invalid_files": invalid_files,
    }
