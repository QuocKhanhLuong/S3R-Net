"""Backward-compatible alias for the MedSAM2 semantic teacher."""

from __future__ import annotations

from .medsam2_teacher import MedSAM2Teacher


MedicalSAM3Teacher = MedSAM2Teacher

__all__ = ["MedicalSAM3Teacher"]
