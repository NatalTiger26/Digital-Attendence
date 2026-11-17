"""Face recognition models package."""

from digital_attendance.models.base import BaseFaceClassifier, Prediction
from digital_attendance.models.eigenfaces import EigenfaceClassifier
from digital_attendance.models.euclidean import MeanFaceClassifier

__all__ = [
    "BaseFaceClassifier",
    "Prediction",
    "MeanFaceClassifier",
    "EigenfaceClassifier",
]
