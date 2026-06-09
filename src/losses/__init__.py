"""Optional losses outside the core S3R model package."""

from .agreement_kd import (
    agreement_aware_fusion,
    agreement_map,
    cinema_boundary_kd_loss,
    fused_kd_loss,
    js_divergence,
    segmentation_field_kd_loss,
    soft_kl_loss,
    spectral_boundary_loss,
)
from .reliability_gate import StudentAwareReliabilityGate

__all__ = [
    "agreement_aware_fusion",
    "agreement_map",
    "cinema_boundary_kd_loss",
    "fused_kd_loss",
    "js_divergence",
    "segmentation_field_kd_loss",
    "soft_kl_loss",
    "StudentAwareReliabilityGate",
    "spectral_boundary_loss",
]
