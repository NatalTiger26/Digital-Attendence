"""Unit tests for HaarCascadeFaceDetector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from digital_attendance.detection.detector import DetectedFace, HaarCascadeFaceDetector


def test_detector_initialization_and_resolution() -> None:
    """Verify detector initializes and resolves existing cascade XML."""
    detector = HaarCascadeFaceDetector()
    assert detector.classifier is not None
    assert not detector.classifier.empty()
    assert detector.cascade_path.is_file()


def test_detector_boundary_handling() -> None:
    """Verify detector handles blank/uniform images without crashing."""
    detector = HaarCascadeFaceDetector()
    blank = np.zeros((100, 100), dtype=np.uint8)
    detections = detector.detect(blank)
    assert isinstance(detections, list)
    assert len(detections) == 0


def test_detector_annotate() -> None:
    """Verify annotation creates rectangles and text labels on image copy."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = np.zeros((50, 50), dtype=np.uint8)
    face = DetectedFace(x=20, y=30, w=50, h=50, crop=crop, confidence=0.95)

    labels_and_boxes = [(face, "Test Subject")]
    annotated = HaarCascadeFaceDetector.annotate(img, labels_and_boxes)

    assert annotated.shape == (200, 200, 3)
    # Annotated canvas must have modified non-zero pixels
    assert np.count_nonzero(annotated) > 0
    # Original image must not be mutated
    assert np.count_nonzero(img) == 0
