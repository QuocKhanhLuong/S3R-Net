"""Frozen external teacher adapters for S3R dual-teacher KD."""

from .base_teacher import FrozenSegmentationTeacher, TeacherLoadError
from .cinema_teacher import CineMATeacher
from .medical_sam3_teacher import MedicalSAM3Teacher

__all__ = [
    "CineMATeacher",
    "FrozenSegmentationTeacher",
    "MedicalSAM3Teacher",
    "TeacherLoadError",
]
