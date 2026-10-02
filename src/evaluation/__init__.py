"""Evaluation, uncertainty estimation, and operational metrics for SolSight."""

from .bootstrap import BootstrapResult, bootstrap_ci, stratified_bootstrap_ci
from .confusion import (
    ConfusionMetrics,
    compute_confusion_matrix_and_metrics,
    compute_operational_review_burden,
    compute_pr_metrics,
)

__all__ = [
    "BootstrapResult",
    "bootstrap_ci",
    "stratified_bootstrap_ci",
    "ConfusionMetrics",
    "compute_confusion_matrix_and_metrics",
    "compute_operational_review_burden",
    "compute_pr_metrics",
]
