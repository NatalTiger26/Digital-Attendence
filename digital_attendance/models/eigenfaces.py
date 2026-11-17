"""Principal Component Analysis (PCA) / Eigenface Face Classifier.

Implements the classical Turk and Pentland (1991) Eigenfaces framework:
1. Calculates global mean face across all training samples.
2. Centers face vectors and constructs the surrogate covariance matrix A A^T / (M - 1).
3. Computes the top K principal eigenvectors and projects them back to full face space.
4. Orthonormalizes the resulting K eigenfaces.
5. Projects face vectors into a compact K-dimensional subspace for nearest-neighbor classification.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from digital_attendance.config import DEFAULT_PCA_COMPONENTS
from digital_attendance.models.base import BaseFaceClassifier, Prediction


class EigenfaceClassifier(BaseFaceClassifier):
    """Face classifier projecting face vectors into a low-dimensional PCA eigenspace."""

    def __init__(self, n_components: int = DEFAULT_PCA_COMPONENTS) -> None:
        """Initialize EigenfaceClassifier.

        Args:
            n_components: Number of principal eigenfaces K to retain.
        """
        super().__init__()
        self.n_components: int = max(1, int(n_components))
        self.mean_face_: np.ndarray | None = None
        self.eigenfaces_: np.ndarray | None = None  # Shape: (K, D)
        self.eigenvalues_: np.ndarray | None = None  # Shape: (K,)
        self.weights_: np.ndarray | None = None  # Shape: (M, K)
        self.y_train_: np.ndarray | None = None  # Shape: (M,)
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Sequence[str] | np.ndarray) -> EigenfaceClassifier:
        r"""Compute the global mean face, eigenface basis, and projection weights.

        Uses the Turk & Pentland snapshot trick:
        Given $M$ images of dimension $D$ ($M \ll D$), compute eigenvectors of $L = A A^T$
        ($M \times M$) instead of $C = A^T A$ ($D \times D$), providing exact eigenvectors
        with orders of magnitude lower memory and computation.

        Args:
            X: Training face vectors of shape `(num_samples, num_features)`.
            y: Subject labels of length `num_samples`.

        Returns:
            Fitted classifier instance (`self`).

        Raises:
            ValueError: If training data dimensions are invalid.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=object)

        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got {X_arr.ndim}D.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"Mismatched sample counts: X={X_arr.shape[0]}, y={y_arr.shape[0]}.")
        if X_arr.shape[0] == 0:
            raise ValueError("Training dataset cannot be empty.")

        n_samples, n_features = X_arr.shape
        k = min(self.n_components, n_samples - 1, n_features)

        # 1. Global Mean Face (Shape: (D,))
        self.mean_face_ = np.mean(X_arr, axis=0)

        # 2. Mean-centered data matrix A (Shape: (M, D))
        centered = X_arr - self.mean_face_

        # 3. Surrogate covariance matrix L = A @ A.T / (M - 1) (Shape: (M, M))
        divisor = max(1, n_samples - 1)
        L = (centered @ centered.T) / divisor

        # 4. Eigendecomposition of real symmetric matrix L
        eigvals, eigvecs = np.linalg.eigh(L)

        # Sort eigenvalues and eigenvectors in descending order
        sort_idx = np.argsort(eigvals)[::-1]
        sorted_vals = eigvals[sort_idx]
        sorted_vecs = eigvecs[:, sort_idx]

        # Select top-k eigenvectors (Shape: (M, K))
        top_k_vecs = sorted_vecs[:, :k]
        top_k_vals = np.maximum(0.0, sorted_vals[:k])

        # 5. Project back to face space: U = top_k_vecs.T @ centered (Shape: (K, D))
        eigenfaces = top_k_vecs.T @ centered

        # Orthonormalize each eigenface to unit L2 norm
        norms = np.linalg.norm(eigenfaces, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        eigenfaces = eigenfaces / norms

        self.eigenfaces_ = eigenfaces
        self.eigenvalues_ = top_k_vals

        total_var = np.sum(np.maximum(0.0, sorted_vals))
        if total_var > 0:
            self.explained_variance_ratio_ = top_k_vals / total_var
        else:
            self.explained_variance_ratio_ = np.ones(k) / k

        # 6. Project all training images into K-dimensional face space (Shape: (M, K))
        self.weights_ = centered @ self.eigenfaces_.T
        self.y_train_ = y_arr
        self.classes_ = sorted(list(dict.fromkeys(y_arr)))
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project query face vectors into the K-dimensional eigenface subspace.

        Args:
            X: Face vector array of shape `(num_queries, num_features)` or `(num_features,)`.

        Returns:
            Subspace coordinates array of shape `(num_queries, K)`.
        """
        self._check_is_fitted()
        assert self.mean_face_ is not None and self.eigenfaces_ is not None

        queries = np.asarray(X, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        centered = queries - self.mean_face_
        return centered @ self.eigenfaces_.T

    def inverse_transform(self, weights: np.ndarray) -> np.ndarray:
        """Reconstruct original pixel-space face vectors from subspace coordinates.

        Args:
            weights: Subspace weights array of shape `(num_queries, K)`.

        Returns:
            Reconstructed face vectors of shape `(num_queries, num_features)`.
        """
        self._check_is_fitted()
        assert self.mean_face_ is not None and self.eigenfaces_ is not None

        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 1:
            w = w.reshape(1, -1)

        reconstructed = (w @ self.eigenfaces_) + self.mean_face_
        return reconstructed

    def predict(self, X: np.ndarray) -> list[str]:
        """Classify query samples by finding the nearest neighbor in eigenface subspace.

        Args:
            X: Query samples of shape `(num_queries, num_features)` or `(num_features,)`.

        Returns:
            List of predicted class names.
        """
        self._check_is_fitted()
        assert self.weights_ is not None and self.y_train_ is not None

        query_weights = self.transform(X)  # Shape: (Q, K)

        # Vectorized Euclidean distance in subspace: (Q, 1, K) - (1, M, K) -> (Q, M)
        diff = query_weights[:, np.newaxis, :] - self.weights_[np.newaxis, :, :]
        subspace_dists = np.linalg.norm(diff, axis=2)

        best_sample_indices = np.argmin(subspace_dists, axis=1)
        return [str(self.y_train_[idx]) for idx in best_sample_indices]

    def predict_top_k(self, X: np.ndarray, k: int = 5) -> list[list[Prediction]]:
        """Predict top-k unique candidate classes for each query sample in subspace.

        Args:
            X: Query samples of shape `(num_queries, num_features)` or `(num_features,)`.
            k: Number of candidate classes to return.

        Returns:
            List of top-k `Prediction` records per query.
        """
        self._check_is_fitted()
        assert self.weights_ is not None and self.y_train_ is not None

        query_weights = self.transform(X)  # Shape: (Q, K)
        diff = query_weights[:, np.newaxis, :] - self.weights_[np.newaxis, :, :]
        subspace_dists = np.linalg.norm(diff, axis=2)

        k_classes = max(1, min(k, len(self.classes_)))
        results: list[list[Prediction]] = []

        for dist_row in subspace_dists:
            sorted_sample_indices = np.argsort(dist_row)

            # Accumulate best distance for each unique class
            seen_classes: dict[str, float] = {}
            for s_idx in sorted_sample_indices:
                cls_name = str(self.y_train_[s_idx])
                d_val = float(dist_row[s_idx])
                if cls_name not in seen_classes:
                    seen_classes[cls_name] = d_val
                if len(seen_classes) >= k_classes:
                    break

            class_items = list(seen_classes.items())
            distances = np.array([d for _, d in class_items], dtype=np.float64)

            # Softmax confidence score
            scale = np.std(distances) if np.std(distances) > 1e-6 else 1.0
            shifted = -(distances - np.min(distances)) / scale
            exp_scores = np.exp(shifted)
            confidences = exp_scores / np.sum(exp_scores)

            preds: list[Prediction] = []
            for rank_idx, ((c_name, d_val), conf) in enumerate(
                zip(class_items, confidences), start=1
            ):
                preds.append(
                    Prediction(
                        label=c_name,
                        confidence=float(conf),
                        distance=float(d_val),
                        rank=rank_idx,
                    )
                )
            results.append(preds)

        return results
