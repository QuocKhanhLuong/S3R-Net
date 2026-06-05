"""S3R-SCSD distillation utilities."""

from .distill_losses import S3RSCSDLoss
from .teacher_cache import load_teacher_cache, save_teacher_cache, validate_cache
from .teacher_targets import (
    boundary_from_mask,
    distance_map_from_mask_or_boundary,
    region_routing_weights,
    semantic_entropy,
    soft_boundary_from_prob,
    spectral_boundary_target,
    teacher_agreement_weight,
)

__all__ = [
    "S3RSCSDLoss",
    "save_teacher_cache",
    "load_teacher_cache",
    "validate_cache",
    "boundary_from_mask",
    "soft_boundary_from_prob",
    "distance_map_from_mask_or_boundary",
    "spectral_boundary_target",
    "semantic_entropy",
    "teacher_agreement_weight",
    "region_routing_weights",
]
