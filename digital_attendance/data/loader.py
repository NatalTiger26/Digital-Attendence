"""Data loading, preprocessing, and vectorization module for face recognition.

Handles reading face images from disk, grayscale conversion, dimensional normalization,
flattening into 1D vectors, and train/test dataset partitioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from digital_attendance.config import (
    DATASET_DIR,
    DEFAULT_IMAGE_SHAPE,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_IMAGE_HEIGHT,
    TOTAL_PIXELS,
)


@dataclass(frozen=True)
class FaceSample:
    """Represents an individual face image sample with metadata.

    Attributes:
        path: Filesystem path to the image file.
        label: Class identifier / identity name of the subject.
        index: Numerical index within the subject's sample collection.
    """

    path: Path
    label: str
    index: int


def resolve_data_path(path_or_str: str | Path) -> Path:
    """Resolve a filesystem path, expanding user directory and checking existence.

    Args:
        path_or_str: Input path as string or Path object.

    Returns:
        Absolute resolved Path.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    path = Path(path_or_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def preprocess_image(
    image_or_path: str | Path | np.ndarray,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
) -> np.ndarray:
    """Load and normalize an image into a 2D single-channel grayscale array.

    Args:
        image_or_path: File path or already-loaded numpy image array (BGR or Grayscale).
        target_size: Target (width, height) resolution for spatial normalization.

    Returns:
        Grayscale float64 array of shape `(target_height, target_width)` with values in [0, 255].

    Raises:
        ValueError: If input image is empty or cannot be read.
    """
    if isinstance(image_or_path, (str, Path)):
        resolved = resolve_data_path(image_or_path)
        img = cv2.imread(str(resolved), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image from path: {resolved}")
    elif isinstance(image_or_path, np.ndarray):
        if image_or_path.size == 0:
            raise ValueError("Input image array is empty.")
        if image_or_path.ndim == 3:
            img = cv2.cvtColor(image_or_path, cv2.COLOR_BGR2GRAY)
        elif image_or_path.ndim == 2:
            img = image_or_path
        else:
            raise ValueError(f"Unsupported array dimensions: {image_or_path.ndim}")
    else:
        raise TypeError(f"Expected str, Path, or np.ndarray, got {type(image_or_path)}")

    if img.shape[1] != target_size[0] or img.shape[0] != target_size[1]:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return img.astype(np.float64)


def image_to_vector(
    image: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
) -> np.ndarray:
    """Preprocess and flatten an image into a 1D feature vector.

    Args:
        image: File path or image array.
        target_size: Target (width, height) resolution.

    Returns:
        Flattened 1D array of shape `(width * height,)` with float64 dtype.
    """
    preprocessed = preprocess_image(image, target_size=target_size)
    return preprocessed.reshape(-1)


def vector_to_image(
    vector: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
) -> np.ndarray:
    """Reshape a 1D feature vector back into a 2D grayscale image.

    Args:
        vector: 1D array of length `width * height` (or 2D column vector).
        target_size: Target (width, height) dimensions.

    Returns:
        2D array of shape `(height, width)`.

    Raises:
        ValueError: If vector length does not match `width * height`.
    """
    expected_len = target_size[0] * target_size[1]
    flat = vector.reshape(-1)
    if flat.shape[0] != expected_len:
        raise ValueError(
            f"Vector length {flat.shape[0]} does not match target dimensions {target_size} ({expected_len})."
        )
    return flat.reshape((target_size[1], target_size[0]))


def load_dataset(
    dataset_dir: str | Path = DATASET_DIR,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all subject images from directory structured as `dataset_dir/<class>/<image>.jpg`.

    Safely reads without mutating current working directory (`os.chdir`).

    Args:
        dataset_dir: Root dataset directory containing subdirectories for each person.
        target_size: Target (width, height) resolution.

    Returns:
        Tuple of:
            - `X`: 2D array of shape `(num_samples, num_features)` containing flattened face vectors.
            - `y`: 1D array of shape `(num_samples,)` containing subject label strings.
            - `classes`: Sorted list of unique class label strings.

    Raises:
        FileNotFoundError: If dataset_dir does not exist.
        RuntimeError: If no valid image files are found.
    """
    root = resolve_data_path(dataset_dir)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not class_dirs:
        raise RuntimeError(f"No subdirectories found in dataset path: {root}")

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    classes: list[str] = [d.name for d in class_dirs]

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_dir in class_dirs:
        class_name = class_dir.name
        img_files = sorted(
            [f for f in class_dir.iterdir() if f.suffix.lower() in valid_extensions]
        )
        for img_file in img_files:
            try:
                vec = image_to_vector(img_file, target_size=target_size)
                vectors.append(vec)
                labels.append(class_name)
            except Exception:
                continue

    if not vectors:
        raise RuntimeError(f"No valid images could be loaded from: {root}")

    X = np.asarray(vectors, dtype=np.float64)
    y = np.asarray(labels, dtype=object)
    return X, y, classes


def split_dataset(
    X: np.ndarray,
    y: Sequence[str] | np.ndarray,
    train_per_class: int = 75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition dataset into training and testing sets based on fixed count per class.

    Preserves original class ordering while splitting deterministic prefixes into train
    and suffixes into test.

    Args:
        X: Feature matrix of shape `(num_samples, num_features)`.
        y: Class labels of length `num_samples`.
        train_per_class: Number of samples per class allocated to the training set.

    Returns:
        Tuple of `(X_train, y_train, X_test, y_test)`.
    """
    y_arr = np.asarray(y)
    unique_classes = sorted(list(dict.fromkeys(y_arr)))

    train_indices: list[int] = []
    test_indices: list[int] = []

    for cls in unique_classes:
        indices = np.where(y_arr == cls)[0]
        train_indices.extend(indices[:train_per_class])
        test_indices.extend(indices[train_per_class:])

    X_train = X[train_indices]
    y_train = y_arr[train_indices]
    X_test = X[test_indices]
    y_test = y_arr[test_indices]

    return X_train, y_train, X_test, y_test
