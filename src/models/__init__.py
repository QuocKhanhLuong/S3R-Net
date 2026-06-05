"""Model registry for SpecUMamba/S3R experiments."""

from __future__ import annotations

from typing import Any

from torch import nn


LEGACY_MODELS = {"specmamba", "asym_spec_mamba", "hrnet_dcn", "hrnet_resnet34"}
S3R_MODELS = {"s3r", "s3r_mini", "s3r_net"}
DEFAULT_MODEL = "s3r"


def available_models() -> list[str]:
    """Return public model names, preserving legacy options."""
    return sorted(S3R_MODELS | LEGACY_MODELS)


def build_model(name: str = DEFAULT_MODEL, **kwargs: Any) -> nn.Module:
    """Build a model by registry name.

    Legacy models are left in their original modules for backward
    compatibility. New experiments should prefer `s3r`/`s3r_mini`/`s3r_net`.
    """
    model_name = str(name).lower()
    if model_name in S3R_MODELS:
        from .s3r import build_s3r_model

        return build_s3r_model(model=model_name, **kwargs)
    if model_name == "specmamba":
        from .specmamba_net import SpecMambaNet

        return SpecMambaNet(**kwargs)
    if model_name == "asym_spec_mamba":
        from .specmamba_net import AsymSpecMambaDCN

        return AsymSpecMambaDCN(**kwargs)
    if model_name == "hrnet_dcn":
        from .hrnet_dcn import HRNetDCN

        return HRNetDCN(**kwargs)
    if model_name == "hrnet_resnet34":
        from .hrnet_resnet34 import HRNetResNet34

        return HRNetResNet34(**kwargs)
    raise ValueError(f"Unknown model {name!r}. Available: {available_models()}")


__all__ = ["DEFAULT_MODEL", "LEGACY_MODELS", "S3R_MODELS", "available_models", "build_model"]
