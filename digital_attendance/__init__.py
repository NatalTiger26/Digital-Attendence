"""Digital Attendance: Automated face detection and recognition attendance system.

Modules:
    - config: Configuration schemas and default paths.
    - data: Image loading, normalization, vectorization, and collage synthesis.
    - detection: Haar Cascade face detection and visual annotation.
    - models: Euclidean Mean Face and PCA Eigenface classifiers.
    - pipeline: Integrated end-to-end attendance processing and report generation.
    - evaluation: Benchmark metrics (Top-1, Top-3, Top-5, MRR, latency).
"""

from digital_attendance.detection.detector import (
    DetectedFace,
    HaarCascadeFaceDetector,
)
from digital_attendance.models.base import (
    BaseFaceClassifier,
    Prediction,
)
from digital_attendance.models.eigenfaces import (
    EigenfaceClassifier,
)
from digital_attendance.models.euclidean import (
    MeanFaceClassifier,
)
from digital_attendance.pipeline.attendance import (
    AttendanceRecord,
    AttendanceReport,
    DigitalAttendancePipeline,
)

__version__ = "0.1.0"

__all__ = [
    "DetectedFace",
    "HaarCascadeFaceDetector",
    "BaseFaceClassifier",
    "Prediction",
    "MeanFaceClassifier",
    "EigenfaceClassifier",
    "AttendanceRecord",
    "AttendanceReport",
    "DigitalAttendancePipeline",
]
