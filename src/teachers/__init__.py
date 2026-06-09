"""Frozen external teacher adapters for S3R dual-teacher KD."""

from .base_teacher import FrozenSegmentationTeacher, TeacherLoadError
from .cinema_teacher import CineMATeacher
from .medsam2_teacher import MedSAM2Teacher

MedicalSAM3Teacher = MedSAM2Teacher

__all__ = [
    "CineMATeacher",
    "FrozenSegmentationTeacher",
    "MedSAM2Teacher",
    "MedicalSAM3Teacher",
    "TeacherLoadError",
]
