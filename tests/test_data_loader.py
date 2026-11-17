"""Unit tests for data loading and preprocessing modules."""

from __future__ import annotations

import numpy as np
import pytest

from digital_attendance.data.loader import (
    FaceSample,
    image_to_vector,
    preprocess_image,
    split_dataset,
    vector_to_image,
)


def test_preprocess_image(synthetic_face_image: np.ndarray) -> None:
    """Verify image preprocessing resizes and normalizes to 2D float64."""
    processed = preprocess_image(synthetic_face_image, target_size=(64, 64))
    assert processed.shape == (64, 64)
    assert processed.dtype == np.float64
    assert np.min(processed) >= 0.0
    assert np.max(processed) <= 255.0


def test_image_to_vector_and_back(synthetic_face_image: np.ndarray) -> None:
    """Verify round-trip flattening and reshaping of images."""
    vec = image_to_vector(synthetic_face_image, target_size=(128, 128))
    assert vec.ndim == 1
    assert vec.shape[0] == 128 * 128

    reconstructed = vector_to_image(vec, target_size=(128, 128))
    assert reconstructed.shape == (128, 128)
    np.testing.assert_allclose(reconstructed, synthetic_face_image.astype(np.float64))


def test_split_dataset() -> None:
    """Verify deterministic train/test dataset partitioning."""
    n_classes = 3
    samples_per_class = 10
    total = n_classes * samples_per_class

    X = np.arange(total).reshape(total, 1)
    y = np.array([f"Person_{i // samples_per_class}" for i in range(total)])

    X_train, y_train, X_test, y_test = split_dataset(X, y, train_per_class=7)

    assert X_train.shape[0] == n_classes * 7
    assert X_test.shape[0] == n_classes * 3
    assert len(y_train) == 21
    assert len(y_test) == 9

    # Verify per-class counts
    for cls in ["Person_0", "Person_1", "Person_2"]:
        assert np.sum(y_train == cls) == 7
        assert np.sum(y_test == cls) == 3
