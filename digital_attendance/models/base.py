"""Abstract base class and prediction protocols for face classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Prediction:
    """Class prediction result with associated similarity metrics.

    Attributes:
        label: Predicted identity or class name.
        confidence: Normalized confidence or posterior probability in [0, 1].
        distance: Raw Euclidean distance in feature/subspace coordinates.
        rank: 1-based candidate rank (1 = Top-1 prediction).
    """

    label: str
    confidence: float
    distance: float
    rank: int = 1


class BaseFaceClassifier(ABC):
    """Abstract base class for all face recognition classifiers."""

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self.classes_: list[str] = []

    def _check_is_fitted(self) -> None:
        """Validate that the classifier has already been fitted on training data.

        Raises:
            RuntimeError: If called prior to `fit()`.
        """
        if not self.is_fitted:
            raise RuntimeError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' first.")

    @abstractmethod
    def fit(self, X: np.ndarray, y: Sequence[str] | np.ndarray) -> BaseFaceClassifier:
        """Fit the model according to the given training data.

        Args:
            X: Training vectors of shape `(num_samples, num_features)`.
            y: Target class labels of shape `(num_samples,)`.

        Returns:
            Fitted classifier instance (`self`).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> list[str]:
        """Perform classification on samples in X.

        Args:
            X: Query samples of shape `(num_queries, num_features)` or `(num_features,)`.

        Returns:
            List of predicted class label strings.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_top_k(self, X: np.ndarray, k: int = 5) -> list[list[Prediction]]:
        """Predict top-k candidate classes with distances and confidence scores.

        Args:
            X: Query samples of shape `(num_queries, num_features)` or `(num_features,)`.
            k: Number of top candidate classes to return.

        Returns:
            Nested list where each element contains top-k `Prediction` records for a query.
        """
        raise NotImplementedError
