"""Unit tests for the PCA Eigenfaces classifier."""

from __future__ import annotations

import numpy as np
import pytest

from digital_attendance.models.eigenfaces import EigenfaceClassifier


def test_eigenface_classifier_fit_and_predict(
    synthetic_face_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Verify eigenspace basis computation, projection, and inference."""
    X_train, y_train, X_test, y_test = synthetic_face_dataset
    k = 10
    clf = EigenfaceClassifier(n_components=k)
    clf.fit(X_train, y_train)

    assert clf.is_fitted
    assert clf.eigenfaces_ is not None
    assert clf.eigenfaces_.shape == (k, X_train.shape[1])
    assert clf.weights_ is not None
    assert clf.weights_.shape == (X_train.shape[0], k)

    # Check orthonormal basis property: U @ U.T ≈ I_k
    ortho = clf.eigenfaces_ @ clf.eigenfaces_.T
    np.testing.assert_allclose(ortho, np.eye(k), atol=1e-5)

    preds = clf.predict(X_test)
    assert len(preds) == len(y_test)

    acc = np.mean(np.array(preds) == y_test)
    assert acc >= 0.85


def test_eigenface_transform_inverse_transform(
    synthetic_face_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Verify transform and inverse_transform reconstruction."""
    X_train, y_train, _, _ = synthetic_face_dataset
    clf = EigenfaceClassifier(n_components=20)
    clf.fit(X_train, y_train)

    query = X_train[:3]
    weights = clf.transform(query)
    assert weights.shape == (3, 20)

    reconstructed = clf.inverse_transform(weights)
    assert reconstructed.shape == query.shape
    # Reconstruction error should be bounded
    rec_error = np.mean((query - reconstructed) ** 2)
    assert rec_error < 5.0


def test_eigenface_predict_top_k(
    synthetic_face_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Verify subspace nearest-neighbor top-k ranking and candidate deduplication."""
    X_train, y_train, X_test, _ = synthetic_face_dataset
    clf = EigenfaceClassifier(n_components=10)
    clf.fit(X_train, y_train)

    top_k = clf.predict_top_k(X_test[:2], k=3)
    assert len(top_k) == 2
    for query_preds in top_k:
        assert len(query_preds) == 3
        labels = [p.label for p in query_preds]
        # Unique classes
        assert len(set(labels)) == 3
        # Ascending distance
        assert query_preds[0].distance <= query_preds[1].distance <= query_preds[2].distance
