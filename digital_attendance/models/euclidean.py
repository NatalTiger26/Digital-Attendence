"""Class-Centroid Euclidean Distance Face Classifier.

Implements baseline face recognition via class mean vectors in raw pixel space.
An unknown face vector is classified based on minimum Euclidean (L2) distance
to the average face vectors of enrolled subjects.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from digital_attendance.models.base import BaseFaceClassifier, Prediction


class MeanFaceClassifier(BaseFaceClassifier):
    """Face classifier matching query vectors to enrolled class centroids via L2 distance."""

    def __init__(self) -> None:
        """Initialize unfitted MeanFaceClassifier."""
        super().__init__()
        self.class_means_: np.ndarray | None = None
        self.class_counts_: dict[str, int] = {}

    def fit(self, X: np.ndarray, y: Sequence[str] | np.ndarray) -> MeanFaceClassifier:
        """Compute the average face vector (centroid) for each unique class.

        Args:
            X: Training face vectors of shape `(num_samples, num_features)`.
            y: Subject labels of length `num_samples`.

        Returns:
            Fitted classifier instance (`self`).

        Raises:
            ValueError: If `X` and `y` lengths mismatch or `X` is empty.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=object)

        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got {X_arr.ndim}D.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"Mismatched sample counts: X has {X_arr.shape[0]}, y has {y_arr.shape[0]}.")
        if X_arr.shape[0] == 0:
            raise ValueError("Training dataset cannot be empty.")

        unique_classes = sorted(list(dict.fromkeys(y_arr)))
        means: list[np.ndarray] = []

        for cls in unique_classes:
            mask = y_arr == cls
            cls_samples = X_arr[mask]
            means.append(np.mean(cls_samples, axis=0))
            self.class_counts_[cls] = int(np.sum(mask))

        self.classes_ = unique_classes
        self.class_means_ = np.asarray(means, dtype=np.float64)  # Shape: (num_classes, num_features)
        self.is_fitted = True
        return self

    def _compute_distances(self, queries: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances from query vectors to all class centroids.

        Vectorized matrix operation:
            `queries`: shape `(Q, D)`
            `class_means_`: shape `(C, D)`
            Result: shape `(Q, C)` where entry `(q, c)` is Euclidean distance.

        Args:
            queries: Array of shape `(Q, D)`.

        Returns:
            Distance matrix of shape `(Q, C)`.
        """
        self._check_is_fitted()
        assert self.class_means_ is not None

        # Vectorized pairwise Euclidean distance: (Q, 1, D) - (1, C, D) -> (Q, C)
        diff = queries[:, np.newaxis, :] - self.class_means_[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=2)

    def predict(self, X: np.ndarray) -> list[str]:
        """Classify query face vectors to closest class centroid.

        Args:
            X: Array of shape `(num_queries, num_features)` or `(num_features,)`.

        Returns:
            List of predicted class names.
        """
        self._check_is_fitted()
        queries = np.asarray(X, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        distances = self._compute_distances(queries)
        best_indices = np.argmin(distances, axis=1)
        return [self.classes_[idx] for idx in best_indices]

    def predict_top_k(self, X: np.ndarray, k: int = 5) -> list[list[Prediction]]:
        """Predict top-k closest class centroids for each query vector.

        Confidence is computed via softmax over negative normalized distances.

        Args:
            X: Array of shape `(num_queries, num_features)` or `(num_features,)`.
            k: Number of candidates to return per query (capped at num_classes).

        Returns:
            List containing top-k `Prediction` records for each query.
        """
        self._check_is_fitted()
        queries = np.asarray(X, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        distances = self._compute_distances(queries)
        num_classes = len(self.classes_)
        k = max(1, min(k, num_classes))

        results: list[list[Prediction]] = []

        for dist_row in distances:
            sorted_indices = np.argsort(dist_row)[:k]
            row_dists = dist_row[sorted_indices]

            # Softmax confidence score
            scale = np.std(dist_row) if np.std(dist_row) > 1e-6 else 1.0
            shifted = -(row_dists - np.min(row_dists)) / scale
            exp_scores = np.exp(shifted)
            confidences = exp_scores / np.sum(exp_scores)

            preds: list[Prediction] = []
            for rank_idx, (c_idx, d_val, conf) in enumerate(
                zip(sorted_indices, row_dists, confidences), start=1
            ):
                preds.append(
                    Prediction(
                        label=self.classes_[c_idx],
                        confidence=float(conf),
                        distance=float(d_val),
                        rank=rank_idx,
                    )
                )
            results.append(preds)

        return results
