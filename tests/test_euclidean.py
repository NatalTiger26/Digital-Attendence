"""Unit tests for the Mean Face Euclidean distance classifier."""

from __future__ import annotations

import numpy as np
import pytest

from digital_attendance.models.euclidean import MeanFaceClassifier


def test_mean_face_classifier_fit_and_predict(
    synthetic_face_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Verify fitting centroids and achieving high accuracy on synthetic clustered data."""
    X_train, y_train, X_test, y_test = synthetic_face_dataset
    clf = MeanFaceClassifier()
    clf.fit(X_train, y_train)

    assert clf.is_fitted
    assert len(clf.classes_) == 4
    assert clf.class_means_ is not None
    assert clf.class_means_.shape == (4, X_train.shape[1])

    preds = clf.predict(X_test)
    assert len(preds) == len(y_test)

    acc = np.mean(np.array(preds) == y_test)
    assert acc >= 0.90


def test_mean_face_predict_top_k(
    synthetic_face_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Verify top-k candidate ranking, distances, and confidence normalization."""
    X_train, y_train, X_test, _ = synthetic_face_dataset
    clf = MeanFaceClassifier()
    clf.fit(X_train, y_train)

    top_k = clf.predict_top_k(X_test[:2], k=3)
    assert len(top_k) == 2
    for query_preds in top_k:
        assert len(query_preds) == 3
        # Ensure monotonic ascending distances
        assert query_preds[0].distance <= query_preds[1].distance <= query_preds[2].distance
        # Ensure rank indexing
        assert [p.rank for p in query_preds] == [1, 2, 3]
        # Ensure positive confidences summing to ~1
        confs = [p.confidence for p in query_preds]
        assert all(c > 0.0 for c in confs)
        assert abs(sum(confs) - 1.0) < 1e-4


def test_unfitted_error() -> None:
    """Verify RuntimeError is raised when predicting before fit."""
    clf = MeanFaceClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(np.zeros((1, 10)))
