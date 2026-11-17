"""Unit and integration tests for Digital Attendance Pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from digital_attendance.detection.detector import HaarCascadeFaceDetector
from digital_attendance.models.euclidean import MeanFaceClassifier
from digital_attendance.pipeline.attendance import (
    AttendanceRecord,
    AttendanceReport,
    DigitalAttendancePipeline,
)


def test_attendance_report_json_and_csv(tmp_path: Path) -> None:
    """Verify exporting attendance reports to JSON and CSV formats."""
    rec1 = AttendanceRecord("Alice", 0.95, 12.3, (10, 10, 50, 50))
    rec2 = AttendanceRecord("Bob", 0.88, 15.1, (70, 70, 50, 50))
    report = AttendanceReport("group_1.jpg", 2, [rec1, rec2], ["Alice", "Bob"])

    json_file = report.to_json(tmp_path / "report.json")
    csv_file = report.to_csv(tmp_path / "report.csv")

    assert json_file.is_file()
    assert csv_file.is_file()

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_detected"] == 2
    assert data["present_members"] == ["Alice", "Bob"]
    assert len(data["records"]) == 2

    csv_content = csv_file.read_text(encoding="utf-8")
    assert "Alice" in csv_content
    assert "Bob" in csv_content


def test_pipeline_execution_on_blank_image() -> None:
    """Verify pipeline gracefully handles image without any faces."""
    detector = HaarCascadeFaceDetector()
    classifier = MeanFaceClassifier()

    # Fit classifier with synthetic data
    X = np.ones((4, 128 * 128), dtype=np.float64)
    y = ["A", "B", "C", "D"]
    pipeline = DigitalAttendancePipeline(detector=detector, classifier=classifier)
    pipeline.fit(X, y)

    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    report, annotated = pipeline.process_image(blank, annotate=True)

    assert report.total_detected == 0
    assert len(report.records) == 0
    assert len(report.present_members) == 0
    assert annotated.shape == blank.shape
