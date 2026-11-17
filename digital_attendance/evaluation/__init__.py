"""Evaluation and benchmarking package."""

from digital_attendance.evaluation.metrics import (
    EvaluationResult,
    evaluate_classifier,
    format_markdown_table,
    run_benchmark_suite,
)

__all__ = [
    "EvaluationResult",
    "evaluate_classifier",
    "run_benchmark_suite",
    "format_markdown_table",
]
