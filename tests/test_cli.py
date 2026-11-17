"""Unit tests for the CLI dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from digital_attendance.cli import main


def test_cli_help(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI prints help when invoked without arguments."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Digital Attendance" in captured.out
    assert "benchmark" in captured.out
    assert "attendance" in captured.out


def test_cli_mark_attendance_missing_image() -> None:
    """Verify CLI exits with error code when supplied non-existent image."""
    ret = main(["attendance", "--image", "non_existent_file.jpg"])
    assert ret != 0
