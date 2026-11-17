"""Test fixtures and synthetic data generators for unit tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_face_image() -> np.ndarray:
    """Generate a synthetic 128x128 grayscale face-like image."""
    rng = np.random.default_rng(42)
    # Background
    img = np.full((128, 128), 120, dtype=np.uint8)
    # Oval "face"
    y, x = np.ogrid[:128, :128]
    mask = ((x - 64) ** 2) / (35**2) + ((y - 64) ** 2) / (45**2) <= 1
    img[mask] = 180
    # Two "eyes"
    left_eye = ((x - 50) ** 2) + ((y - 52) ** 2) <= 16
    right_eye = ((x - 78) ** 2) + ((y - 52) ** 2) <= 16
    img[left_eye | right_eye] = 40
    # "Mouth"
    mouth = (abs(x - 64) <= 18) & (abs(y - 85) <= 4)
    img[mouth] = 50
    return img


@pytest.fixture
def synthetic_face_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic train/test dataset for 4 classes in 100-dim feature space.

    Returns:
        Tuple of `(X_train, y_train, X_test, y_test)`.
    """
    rng = np.random.default_rng(123)
    n_features = 256
    n_train_per_class = 20
    n_test_per_class = 5
    classes = ["Alice", "Bob", "Charlie", "Diana"]

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    for idx, cls in enumerate(classes):
        # Distinct centroid for each class
        centroid = rng.normal(loc=float(idx) * 3.0, scale=1.0, size=(n_features,))

        for _ in range(n_train_per_class):
            noise = rng.normal(loc=0.0, scale=0.5, size=(n_features,))
            X_train_list.append(centroid + noise)
            y_train_list.append(cls)

        for _ in range(n_test_per_class):
            noise = rng.normal(loc=0.0, scale=0.5, size=(n_features,))
            X_test_list.append(centroid + noise)
            y_test_list.append(cls)

    return (
        np.array(X_train_list, dtype=np.float64),
        np.array(y_train_list, dtype=object),
        np.array(X_test_list, dtype=np.float64),
        np.array(y_test_list, dtype=object),
    )
