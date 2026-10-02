"""Confusion matrix computation, classification metrics, and operational review burden."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


@dataclass
class ConfusionMetrics:
    """Detailed confusion matrix counts and industrial operational metrics."""
    tp: int
    fp: int
    tn: int
    fn: int
    total: int
    total_positives: int
    total_negatives: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    fpr: float
    fnr: float
    f1_score: float
    npv: float
    operational_burden_per_1k: float
    operational_burden_per_10k: float

    @property
    def matrix(self) -> List[List[int]]:
        """Standard 2x2 confusion matrix: [[TN, FP], [FN, TP]]."""
        return [[self.tn, self.fp], [self.fn, self.tp]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix": self.matrix,
            "counts": {
                "tp": self.tp,
                "fp": self.fp,
                "tn": self.tn,
                "fn": self.fn,
                "total": self.total,
                "total_positives": self.total_positives,
                "total_negatives": self.total_negatives,
            },
            "metrics": {
                "accuracy": float(self.accuracy),
                "precision": float(self.precision),
                "recall": float(self.recall),
                "specificity": float(self.specificity),
                "fpr": float(self.fpr),
                "fnr": float(self.fnr),
                "f1_score": float(self.f1_score),
                "npv": float(self.npv),
            },
            "operational_review_burden": {
                "per_1000_joints": float(self.operational_burden_per_1k),
                "per_10000_joints": float(self.operational_burden_per_10k),
            },
        }


def compute_operational_review_burden(
    fpr: float,
    joint_scales: Tuple[int, ...] = (1000, 10000),
) -> Dict[str, float]:
    """Calculate the expected number of false alarms requiring manual AOI operator inspection.

    Args:
        fpr: False positive rate in [0, 1].
        joint_scales: Joint production volume scales to compute for.

    Returns:
        Dictionary mapping scale string to expected false alarm count.
    """
    burden = {}
    for scale in joint_scales:
        burden[f"per_{scale}_joints"] = float(fpr * scale)
    return burden


def compute_confusion_matrix_and_metrics(
    y_true: Union[np.ndarray, list],
    y_pred_or_scores: Union[np.ndarray, list],
    threshold: Optional[float] = None,
) -> ConfusionMetrics:
    """Compute detailed confusion counts, rates, and operational review burden.

    Args:
        y_true: Ground truth binary labels (0 = normal, 1 = defect).
        y_pred_or_scores: Predicted binary labels or continuous anomaly scores.
        threshold: If provided, binarizes scores as (y_pred_or_scores > threshold).
                   If None, assumes y_pred_or_scores are already binary 0/1.

    Returns:
        ConfusionMetrics object containing counts, rates, and operator burden.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    pred_arr = np.asarray(y_pred_or_scores)

    if threshold is not None:
        y_pred_binary = (pred_arr > threshold).astype(np.int32)
    else:
        y_pred_binary = (pred_arr >= 0.5).astype(np.int32)

    total = len(y_true_arr)
    if total == 0:
        return ConfusionMetrics(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    tp = int(np.sum((y_true_arr == 1) & (y_pred_binary == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_binary == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_binary == 0)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_binary == 0)))

    total_pos = tp + fn
    total_neg = tn + fp

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_pos if total_pos > 0 else 0.0
    specificity = tn / total_neg if total_neg > 0 else 0.0
    fpr = fp / total_neg if total_neg > 0 else 0.0
    fnr = fn / total_pos if total_pos > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    burden = compute_operational_review_burden(fpr, (1000, 10000))

    return ConfusionMetrics(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        total=total,
        total_positives=total_pos,
        total_negatives=total_neg,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
        fpr=float(fpr),
        fnr=float(fnr),
        f1_score=float(f1_score),
        npv=float(npv),
        operational_burden_per_1k=burden["per_1000_joints"],
        operational_burden_per_10k=burden["per_10000_joints"],
    )


def compute_pr_metrics(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
) -> Dict[str, Any]:
    """Compute precision-recall curve coordinates and Average Precision (AP).

    Args:
        y_true: Ground truth binary labels (0 = normal, 1 = defect).
        y_scores: Continuous anomaly scores.

    Returns:
        Dict with precision, recall, thresholds, and average_precision.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_scores_arr = np.asarray(y_scores, dtype=np.float64)

    if len(np.unique(y_true_arr)) < 2:
        return {
            "average_precision": 0.0,
            "precision_curve": [],
            "recall_curve": [],
            "thresholds": [],
        }

    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_scores_arr)
    ap = float(average_precision_score(y_true_arr, y_scores_arr))

    return {
        "average_precision": ap,
        "precision_curve": precision.tolist(),
        "recall_curve": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }
