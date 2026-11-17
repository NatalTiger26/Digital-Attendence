"""Configuration defaults and paths for the Digital Attendance system."""

from __future__ import annotations

import os
from pathlib import Path

# Base paths
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATASET_DIR: Path = ROOT_DIR / "Dataset"
FULL_DATA_DIR: Path = ROOT_DIR / "Full Data"
COLLAGES_DIR: Path = ROOT_DIR / "Collages"
TEST_IMAGES_DIR: Path = ROOT_DIR / "test images"
DEFAULT_OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

# Haar Cascade Model Path (with case-insensitive resolution fallback)
DEFAULT_CASCADE_FILENAME: str = "haarcascade_frontalface_default.xml"
DEFAULT_CASCADE_PATH: Path = ROOT_DIR / DEFAULT_CASCADE_FILENAME

# Image and Face Normalization Parameters
DEFAULT_IMAGE_WIDTH: int = 128
DEFAULT_IMAGE_HEIGHT: int = 128
DEFAULT_IMAGE_SHAPE: tuple[int, int] = (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)
TOTAL_PIXELS: int = DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT

# Recognition Model Hyperparameters
DEFAULT_PCA_COMPONENTS: int = 60
DEFAULT_TRAIN_SAMPLES_PER_CLASS: int = 75
DEFAULT_TOTAL_SAMPLES_PER_CLASS: int = 80
DEFAULT_TEST_SAMPLES_PER_CLASS: int = (
    DEFAULT_TOTAL_SAMPLES_PER_CLASS - DEFAULT_TRAIN_SAMPLES_PER_CLASS
)

# Face Detection Hyperparameters
DEFAULT_SCALE_FACTOR: float = 1.05
DEFAULT_MIN_NEIGHBORS: int = 7
DEFAULT_MIN_FACE_SIZE: tuple[int, int] = (30, 30)


def resolve_cascade_path(custom_path: str | Path | None = None) -> Path:
    """Resolve the path to the Haar Cascade frontal face XML classifier.

    Checks the provided path, default repository path, and standard OpenCV data
    directories with case-insensitive fallback.

    Args:
        custom_path: Optional explicit path to the cascade XML file.

    Returns:
        Resolved Path object pointing to an existing cascade XML file.

    Raises:
        FileNotFoundError: If the cascade file cannot be located.
    """
    candidates: list[Path] = []
    if custom_path is not None:
        candidates.append(Path(custom_path))

    candidates.append(DEFAULT_CASCADE_PATH)
    candidates.append(ROOT_DIR / "Haarcascade_frontalface_default.xml")

    # Try OpenCV's built-in data directory if available
    try:
        import cv2

        cv2_data = getattr(cv2, "data", None)
        if cv2_data is not None and hasattr(cv2_data, "haarcascades"):
            candidates.append(Path(cv2_data.haarcascades) / DEFAULT_CASCADE_FILENAME)
    except Exception:
        pass

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Unable to locate face cascade XML. Checked locations: {[str(c) for c in candidates]}"
    )
