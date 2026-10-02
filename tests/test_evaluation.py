"""Unit tests for bootstrap confidence intervals and confusion matrix metrics."""

import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
from src.evaluation.bootstrap import bootstrap_ci, stratified_bootstrap_ci
from src.evaluation.confusion import (
    compute_confusion_matrix_and_metrics,
    compute_operational_review_burden,
    compute_pr_metrics,
)


class TestEvaluationSuite(unittest.TestCase):
    """Test bootstrap interval estimation, confusion counts, and operational metrics."""

    def test_bootstrap_ci_mean(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = bootstrap_ci(data, np.mean, n_bootstrap=1000, confidence=0.95, seed=42)

        self.assertAlmostEqual(res.point_estimate, 3.0, places=4)
        self.assertLessEqual(res.ci_lower, res.point_estimate)
        self.assertGreaterEqual(res.ci_upper, res.point_estimate)
        self.assertGreater(res.standard_error, 0.0)

        fmt = res.formatted(".2f")
        self.assertIn("3.00 [95% CI:", fmt)

    def test_stratified_bootstrap_ci_auroc(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.2, 0.4, 0.7, 0.8, 0.9, 0.6, 0.85])

        res = stratified_bootstrap_ci(
            y_true,
            y_scores,
            lambda yt, ys: roc_auc_score(yt, ys),
            n_bootstrap=500,
            confidence=0.95,
            seed=42,
        )

        self.assertAlmostEqual(res.point_estimate, 1.0, places=2)
        self.assertGreaterEqual(res.ci_lower, 0.8)
        self.assertLessEqual(res.ci_upper, 1.0)

    def test_confusion_matrix_metrics(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.2, 0.7, 0.9, 0.95])
        # Threshold at 0.5:
        # Preds: [0, 0, 0, 1, 0, 1, 1, 1]
        # True:  [0, 0, 0, 0, 1, 1, 1, 1]
        # TN=3, FP=1, FN=1, TP=3
        cm = compute_confusion_matrix_and_metrics(y_true, y_scores, threshold=0.5)

        self.assertEqual(cm.tp, 3)
        self.assertEqual(cm.fp, 1)
        self.assertEqual(cm.tn, 3)
        self.assertEqual(cm.fn, 1)
        self.assertEqual(cm.total, 8)
        self.assertAlmostEqual(cm.accuracy, 6 / 8)
        self.assertAlmostEqual(cm.recall, 3 / 4)
        self.assertAlmostEqual(cm.precision, 3 / 4)
        self.assertAlmostEqual(cm.fpr, 1 / 4)

        # Review burden: 1/4 = 25% FPR -> 250 per 1k, 2500 per 10k
        self.assertAlmostEqual(cm.operational_burden_per_1k, 250.0)
        self.assertAlmostEqual(cm.operational_burden_per_10k, 2500.0)

        # Dictionary serialization
        d = cm.to_dict()
        self.assertEqual(d["matrix"], [[3, 1], [1, 3]])
        self.assertIn("operational_review_burden", d)

    def test_pr_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.8])
        pr = compute_pr_metrics(y_true, y_scores)

        self.assertGreater(pr["average_precision"], 0.9)
        self.assertGreater(len(pr["precision_curve"]), 0)
        self.assertGreater(len(pr["recall_curve"]), 0)


if __name__ == "__main__":
    unittest.main()
