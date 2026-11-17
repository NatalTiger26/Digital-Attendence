"""End-to-End Digital Attendance Pipeline.

Orchestrates multi-face detection from group images or classroom collages,
face extraction and normalization, identity classification, and attendance roster generation.
"""

from __future__ import annotations

import csv
import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from digital_attendance.config import DEFAULT_IMAGE_SHAPE
from digital_attendance.data.loader import image_to_vector, resolve_data_path
from digital_attendance.detection.detector import DetectedFace, HaarCascadeFaceDetector
from digital_attendance.models.base import BaseFaceClassifier
from digital_attendance.models.eigenfaces import EigenfaceClassifier


@dataclass
class AttendanceRecord:
    """Individual face detection and recognition record for attendance.

    Attributes:
        person_name: Identified identity.
        confidence: Prediction confidence score.
        distance: Prediction distance metric.
        bounding_box: (x, y, w, h) coordinates in the source image.
        timestamp: ISO-formatted detection timestamp.
    """

    person_name: str
    confidence: float
    distance: float
    bounding_box: tuple[int, int, int, int]
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


@dataclass
class AttendanceReport:
    """Consolidated digital attendance roster report.

    Attributes:
        source_image: Identifier or path of the processed input image.
        total_detected: Total number of faces localized in the frame.
        records: List of individual attendance records.
        present_members: Deduplicated sorted list of recognized subjects marked present.
    """

    source_image: str
    total_detected: int
    records: list[AttendanceRecord]
    present_members: list[str]

    def to_dict(self) -> dict:
        """Convert report to serializable dictionary."""
        return {
            "source_image": self.source_image,
            "total_detected": self.total_detected,
            "present_members": self.present_members,
            "records": [asdict(r) for r in self.records],
        }

    def to_json(self, output_path: str | Path) -> Path:
        """Export report to a formatted JSON file.

        Args:
            output_path: Target filesystem path for the JSON file.

        Returns:
            Resolved Path to the saved file.
        """
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out

    def to_csv(self, output_path: str | Path) -> Path:
        """Export attendance roster to a CSV table.

        Args:
            output_path: Target filesystem path for the CSV file.

        Returns:
            Resolved Path to the saved file.
        """
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "Person Name", "Confidence", "Distance", "BoundingBox (x,y,w,h)", "Timestamp"])
            for idx, r in enumerate(self.records, start=1):
                writer.writerow([
                    idx,
                    r.person_name,
                    f"{r.confidence:.4f}",
                    f"{r.distance:.2f}",
                    f"{r.bounding_box}",
                    r.timestamp,
                ])
        return out


class DigitalAttendancePipeline:
    """Full attendance pipeline coupling Haar Cascade detection with face recognition."""

    def __init__(
        self,
        detector: HaarCascadeFaceDetector | None = None,
        classifier: BaseFaceClassifier | None = None,
        target_face_size: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
    ) -> None:
        """Initialize the attendance pipeline.

        Args:
            detector: Face detector instance (defaults to HaarCascadeFaceDetector).
            classifier: Face classifier instance (defaults to EigenfaceClassifier).
            target_face_size: Resolution to which cropped faces are resized prior to classification.
        """
        self.detector = detector or HaarCascadeFaceDetector()
        self.classifier = classifier or EigenfaceClassifier()
        self.target_face_size = target_face_size

    def fit(self, X: np.ndarray, y: Sequence[str] | np.ndarray) -> DigitalAttendancePipeline:
        """Fit the underlying face recognition classifier.

        Args:
            X: Training face vectors of shape `(num_samples, num_features)`.
            y: Subject labels of length `num_samples`.

        Returns:
            Fitted pipeline instance (`self`).
        """
        self.classifier.fit(X, y)
        return self

    def process_image(
        self,
        image_or_path: str | Path | np.ndarray,
        annotate: bool = True,
    ) -> tuple[AttendanceReport, np.ndarray]:
        """Detect and identify all faces in an image, generating an attendance report.

        Args:
            image_or_path: Path to image file or numpy image array (BGR).
            annotate: Whether to render bounding boxes and labels onto the image.

        Returns:
            Tuple of `(AttendanceReport, annotated_image)`.

        Raises:
            ValueError: If input image is empty or cannot be read.
        """
        if isinstance(image_or_path, (str, Path)):
            resolved = resolve_data_path(image_or_path)
            img = cv2.imread(str(resolved))
            if img is None:
                raise ValueError(f"Could not load image: {resolved}")
            source_name = resolved.name
        elif isinstance(image_or_path, np.ndarray):
            if image_or_path.size == 0:
                raise ValueError("Provided image array is empty.")
            img = image_or_path.copy()
            source_name = "in_memory_image"
        else:
            raise TypeError(f"Expected str, Path, or np.ndarray, got {type(image_or_path)}")

        # 1. Detect faces
        detections = self.detector.detect(img)

        records: list[AttendanceRecord] = []
        annotation_pairs: list[tuple[DetectedFace, str]] = []

        # 2. Extract crops and classify
        if detections:
            vectors: list[np.ndarray] = []
            for face in detections:
                vec = image_to_vector(face.crop, target_size=self.target_face_size)
                vectors.append(vec)

            X_query = np.asarray(vectors, dtype=np.float64)
            top_preds = self.classifier.predict_top_k(X_query, k=1)

            for face, preds in zip(detections, top_preds):
                best = preds[0]
                rec = AttendanceRecord(
                    person_name=best.label,
                    confidence=best.confidence,
                    distance=best.distance,
                    bounding_box=face.bounding_box,
                )
                records.append(rec)
                annotation_pairs.append((face, f"{best.label} ({best.confidence:.2f})"))

        present = sorted(list(dict.fromkeys(r.person_name for r in records)))
        report = AttendanceReport(
            source_image=source_name,
            total_detected=len(detections),
            records=records,
            present_members=present,
        )

        annotated_image = (
            self.detector.annotate(img, annotation_pairs) if annotate else img
        )

        return report, annotated_image
