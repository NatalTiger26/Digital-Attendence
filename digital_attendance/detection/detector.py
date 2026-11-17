"""Face detection module wrapping OpenCV Haar Feature-based Cascade Classifiers.

Provides robust multi-scale frontal face detection, bounding box extraction,
ROI cropping with boundary protection, and visual annotation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from digital_attendance.config import (
    DEFAULT_MIN_FACE_SIZE,
    DEFAULT_MIN_NEIGHBORS,
    DEFAULT_SCALE_FACTOR,
    resolve_cascade_path,
)


@dataclass(frozen=True)
class DetectedFace:
    """Represents a face detected within an image frame.

    Attributes:
        x: X-coordinate (horizontal pixel column) of top-left corner.
        y: Y-coordinate (vertical pixel row) of top-left corner.
        w: Width of the detected bounding box in pixels.
        h: Height of the detected bounding box in pixels.
        crop: Cropped image region of the face (grayscale or BGR).
        confidence: Detection confidence score (defaults to 1.0 for Haar cascade).
    """

    x: int
    y: int
    w: int
    h: int
    crop: np.ndarray
    confidence: float = 1.0

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        """Return (x, y, w, h) bounding box tuple."""
        return (self.x, self.y, self.w, self.h)


class HaarCascadeFaceDetector:
    """Frontal face detector based on OpenCV's Haar Feature-based Cascade Classifier."""

    def __init__(self, cascade_path: str | Path | None = None) -> None:
        """Initialize the detector and load cascade XML weights.

        Args:
            cascade_path: Optional explicit path to cascade XML file.

        Raises:
            RuntimeError: If cascade classifier fails to load.
        """
        self.cascade_path: Path = resolve_cascade_path(cascade_path)
        self.classifier: cv2.CascadeClassifier = cv2.CascadeClassifier(str(self.cascade_path))

        if self.classifier.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from: {self.cascade_path}")

    def detect(
        self,
        image: np.ndarray,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        min_neighbors: int = DEFAULT_MIN_NEIGHBORS,
        min_size: tuple[int, int] = DEFAULT_MIN_FACE_SIZE,
    ) -> list[DetectedFace]:
        """Detect faces in an input image.

        Correctly extracts regions using NumPy image coordinate conventions:
        `image[y : y + h, x : x + w]` with boundary clamping to prevent slicing bugs.

        Args:
            image: Image array (BGR, RGB, or Grayscale).
            scale_factor: Parameter specifying how much the image size is reduced at each scale.
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should retain.
            min_size: Minimum possible object size. Objects smaller than this are ignored.

        Returns:
            List of `DetectedFace` objects sorted by left-to-right (x) position.
        """
        if image is None or image.size == 0:
            return []

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")

        rects = self.classifier.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(rects) == 0:
            return []

        img_h, img_w = image.shape[:2]
        detections: list[DetectedFace] = []

        for (x, y, w, h) in rects:
            # Boundary protection
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, int(x + w))
            y2 = min(img_h, int(y + h))

            if x2 <= x1 or y2 <= y1:
                continue

            # NumPy indexing: [row_start:row_end, col_start:col_end] -> [y1:y2, x1:x2]
            crop = image[y1:y2, x1:x2].copy()

            detections.append(
                DetectedFace(
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    crop=crop,
                    confidence=1.0,
                )
            )

        # Sort spatially: top-to-bottom then left-to-right
        detections.sort(key=lambda d: (d.y // max(1, d.h), d.x))
        return detections

    @staticmethod
    def annotate(
        image: np.ndarray,
        detections: Sequence[tuple[DetectedFace, str]],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes and identity text labels on an image.

        Args:
            image: Original image array.
            detections: List of pairs `(detected_face, label_text)`.
            color: Box and text color in BGR.
            thickness: Line width for bounding box.

        Returns:
            Annotated BGR copy of the image.
        """
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        for face, text in detections:
            x, y, w, h = face.x, face.y, face.w, face.h
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

            label = str(text)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            text_y = max(y - 5, text_h + 5)
            # Background badge for text legibility
            cv2.rectangle(
                canvas,
                (x, text_y - text_h - 2),
                (x + text_w + 4, text_y + baseline),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.putText(
                canvas,
                label,
                (x + 2, text_y),
                font,
                font_scale,
                color,
                text_thickness,
                lineType=cv2.LINE_AA,
            )

        return canvas
