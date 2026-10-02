"""Uncertainty estimation via stratified and standard bootstrap confidence intervals."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np


@dataclass
class BootstrapResult:
    """Container for metric point estimate and bootstrap confidence interval bounds."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    standard_error: float
    confidence: float = 0.95

    def to_dict(self) -> Dict[str, float]:
        return {
            "point_estimate": float(self.point_estimate),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "standard_error": float(self.standard_error),
            "confidence": float(self.confidence),
        }

    def formatted(self, fmt: str = ".3f") -> str:
        """Format as string: estimate [95% CI: lower - upper]."""
        pct = int(round(self.confidence * 100))
        return f"{self.point_estimate:{fmt}} [{pct}% CI: {self.ci_lower:{fmt}} - {self.ci_upper:{fmt}}]"


def bootstrap_ci(
    data: Union[np.ndarray, list],
    metric_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compute percentile bootstrap confidence interval for a 1D dataset.

    Args:
        data: Input array or list of numerical values.
        metric_fn: Callable mapping 1D ndarray -> scalar metric float.
        n_bootstrap: Number of bootstrap resamples (default: 2000).
        confidence: Confidence level in (0, 1) (default: 0.95).
        seed: Random seed for deterministic reproducibility.

    Returns:
        BootstrapResult with point estimate, lower and upper bounds, and standard error.
    """
    arr = np.asarray(data)
    n = len(arr)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, 0.0, confidence)

    point_estimate = float(metric_fn(arr))
    if n == 1 or np.all(arr == arr[0]):
        return BootstrapResult(point_estimate, point_estimate, point_estimate, 0.0, confidence)

    rng = np.random.RandomState(seed)
    # Generate random indices of shape (n_bootstrap, n)
    indices = rng.randint(0, n, size=(n_bootstrap, n))
    bootstrap_samples = arr[indices]

    # Evaluate metric per resample
    bootstrap_estimates = np.array([metric_fn(sample) for sample in bootstrap_samples], dtype=np.float64)

    # Filter out NaNs or non-finite values if any
    valid_estimates = bootstrap_estimates[np.isfinite(bootstrap_estimates)]
    if len(valid_estimates) == 0:
        return BootstrapResult(point_estimate, point_estimate, point_estimate, 0.0, confidence)

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(valid_estimates, lower_pct))
    ci_upper = float(np.percentile(valid_estimates, upper_pct))
    std_error = float(np.std(valid_estimates))

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        standard_error=std_error,
        confidence=confidence,
    )


def stratified_bootstrap_ci(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compute stratified bootstrap confidence interval for binary classification metrics.

    Preserves the exact ratio of positive and negative labels in every bootstrap
    resample, preventing degenerate samples where all draws are from one class.

    Args:
        y_true: Ground truth binary labels (0 = normal, 1 = defect).
        y_scores: Model prediction scores or probabilities.
        metric_fn: Callable mapping (y_true_sample, y_scores_sample) -> scalar metric (e.g. roc_auc_score).
        n_bootstrap: Number of bootstrap iterations (default: 2000).
        confidence: Confidence level in (0, 1) (default: 0.95).
        seed: Random seed for deterministic reproducibility.

    Returns:
        BootstrapResult with point estimate, lower and upper bounds, and standard error.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_scores_arr = np.asarray(y_scores, dtype=np.float64)

    if len(y_true_arr) != len(y_scores_arr):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true_arr)} != len(y_scores)={len(y_scores_arr)}")

    pos_idx = np.where(y_true_arr == 1)[0]
    neg_idx = np.where(y_true_arr == 0)[0]

    point_estimate = float(metric_fn(y_true_arr, y_scores_arr))

    # If only one class is present, stratification cannot resample both classes
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return BootstrapResult(point_estimate, point_estimate, point_estimate, 0.0, confidence)

    rng = np.random.RandomState(seed)
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    pos_draws = pos_idx[rng.randint(0, n_pos, size=(n_bootstrap, n_pos))]
    neg_draws = neg_idx[rng.randint(0, n_neg, size=(n_bootstrap, n_neg))]

    estimates = []
    for i in range(n_bootstrap):
        sample_indices = np.concatenate([pos_draws[i], neg_draws[i]])
        y_t = y_true_arr[sample_indices]
        y_s = y_scores_arr[sample_indices]
        try:
            val = metric_fn(y_t, y_s)
            if np.isfinite(val):
                estimates.append(float(val))
        except Exception:
            continue

    if len(estimates) == 0:
        return BootstrapResult(point_estimate, point_estimate, point_estimate, 0.0, confidence)

    est_arr = np.array(estimates, dtype=np.float64)
    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(est_arr, lower_pct))
    ci_upper = float(np.percentile(est_arr, upper_pct))
    std_error = float(np.std(est_arr))

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        standard_error=std_error,
        confidence=confidence,
    )
