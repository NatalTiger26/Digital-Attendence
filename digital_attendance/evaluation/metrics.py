"""Evaluation and benchmarking module for face recognition classifiers.

Calculates multi-rank accuracy (Top-1, Top-3, Top-5), Mean Reciprocal Rank (MRR),
inference latency, and formats comparative benchmark tables.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from digital_attendance.models.base import BaseFaceClassifier
from digital_attendance.models.eigenfaces import EigenfaceClassifier
from digital_attendance.models.euclidean import MeanFaceClassifier


@dataclass(frozen=True)
class EvaluationResult:
    """Benchmark metrics summary for an evaluated classifier.

    Attributes:
        model_name: Identifier name of the model.
        top1_accuracy: Fraction of queries where ground truth equals top-1 prediction.
        top3_accuracy: Fraction of queries where ground truth is within top-3 predictions.
        top5_accuracy: Fraction of queries where ground truth is within top-5 predictions.
        mrr: Mean Reciprocal Rank score in [0, 1].
        latency_ms: Average classification latency per query in milliseconds.
        total_samples: Total number of evaluated test queries.
    """

    model_name: str
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    mrr: float
    latency_ms: float
    total_samples: int


def evaluate_classifier(
    classifier: BaseFaceClassifier,
    X_test: np.ndarray,
    y_test: Sequence[str] | np.ndarray,
    model_name: str | None = None,
) -> EvaluationResult:
    """Evaluate a fitted classifier against test samples.

    Args:
        classifier: Fitted BaseFaceClassifier model.
        X_test: Query feature matrix of shape `(num_queries, num_features)`.
        y_test: Ground-truth class labels of length `num_queries`.
        model_name: Optional custom model name.

    Returns:
        EvaluationResult containing Top-1/3/5 accuracy, MRR, and latency.

    Raises:
        ValueError: If test set is empty.
    """
    X_arr = np.asarray(X_test, dtype=np.float64)
    y_arr = np.asarray(y_test, dtype=object)
    n_samples = X_arr.shape[0]

    if n_samples == 0:
        raise ValueError("Cannot evaluate on empty test set.")

    name = model_name or classifier.__class__.__name__

    start_time = time.perf_counter()
    top_k_preds = classifier.predict_top_k(X_arr, k=5)
    total_time_ms = (time.perf_counter() - start_time) * 1000.0
    latency_ms = total_time_ms / n_samples

    top1 = 0
    top3 = 0
    top5 = 0
    reciprocal_ranks = []

    for true_label, preds in zip(y_arr, top_k_preds):
        candidate_labels = [p.label for p in preds]
        if true_label in candidate_labels:
            rank = candidate_labels.index(true_label) + 1
            if rank == 1:
                top1 += 1
            if rank <= 3:
                top3 += 1
            if rank <= 5:
                top5 += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return EvaluationResult(
        model_name=name,
        top1_accuracy=top1 / n_samples,
        top3_accuracy=top3 / n_samples,
        top5_accuracy=top5 / n_samples,
        mrr=float(np.mean(reciprocal_ranks)),
        latency_ms=latency_ms,
        total_samples=n_samples,
    )


def run_benchmark_suite(
    X_train: np.ndarray,
    y_train: Sequence[str] | np.ndarray,
    X_test: np.ndarray,
    y_test: Sequence[str] | np.ndarray,
) -> dict[str, EvaluationResult]:
    """Train and benchmark both Euclidean Mean Face and PCA Eigenface models.

    Args:
        X_train: Training vectors `(N_train, D)`.
        y_train: Training labels `(N_train,)`.
        X_test: Test query vectors `(N_test, D)`.
        y_test: Test query labels `(N_test,)`.

    Returns:
        Dictionary mapping model names to their respective `EvaluationResult`.
    """
    models: dict[str, BaseFaceClassifier] = {
        "Euclidean Mean Face": MeanFaceClassifier(),
        "PCA Eigenfaces (K=60)": EigenfaceClassifier(n_components=60),
    }

    results: dict[str, EvaluationResult] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        res = evaluate_classifier(model, X_test, y_test, model_name=name)
        results[name] = res

    return results


def format_markdown_table(results: dict[str, EvaluationResult]) -> str:
    """Format benchmark results into a clean GitHub Flavored Markdown table.

    Args:
        results: Dictionary of model names to `EvaluationResult`.

    Returns:
        Formatted markdown table string.
    """
    lines = [
        "| Model Architecture | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | MRR | Latency (ms/sample) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for res in results.values():
        lines.append(
            f"| **{res.model_name}** | {res.top1_accuracy:.2%} | {res.top3_accuracy:.2%} | "
            f"{res.top5_accuracy:.2%} | {res.mrr:.4f} | {res.latency_ms:.3f} ms |"
        )
    return "\n".join(lines)
