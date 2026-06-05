"""Model registry for S3R experiments."""

from __future__ import annotations

from typing import Any

from torch import nn


S3R_MODELS = {"s3r", "s3r_mini", "s3r_net"}
DEFAULT_MODEL = "s3r"


def available_models() -> list[str]:
    """Return public S3R model names."""
    return sorted(S3R_MODELS)


def build_model(name: str = DEFAULT_MODEL, **kwargs: Any) -> nn.Module:
    """Build an S3R model by registry name."""
    model_name = str(name).lower()
    if model_name in S3R_MODELS:
        from .s3r import build_s3r_model

        return build_s3r_model(model=model_name, **kwargs)
    raise ValueError(f"Unknown model {name!r}. Available: {available_models()}")


__all__ = ["DEFAULT_MODEL", "S3R_MODELS", "available_models", "build_model"]
