"""S3R-Net model family.

Legacy SpecMamba/HRNet models remain in `src/models`; this package is the new
default path for future S3R experiments.
"""

from .losses import S3RLoss, boundary_map_from_mask
from .s3r_blocks import (
    DeformableRefine,
    LargeKernelRefine,
    ResidualChannelGate,
    SEBlock,
    SSRFullBlock,
    build_radial_frequency_masks,
)
from .s3r_model import S3RMini, S3RNet, S3RTransitionBlock, build_s3r_model
from .s3r_state import SpectralStateInitializer, SpectralStateTransition, StateGuidedModulation

__all__ = [
    "S3RLoss",
    "boundary_map_from_mask",
    "build_radial_frequency_masks",
    "SEBlock",
    "ResidualChannelGate",
    "LargeKernelRefine",
    "DeformableRefine",
    "SSRFullBlock",
    "SpectralStateInitializer",
    "SpectralStateTransition",
    "StateGuidedModulation",
    "S3RTransitionBlock",
    "S3RMini",
    "S3RNet",
    "build_s3r_model",
]
