"""Comprehensive Benchmark Evaluation Script for SolSight (Hardened Milestone 1).

Evaluates the multi-scale SolderCAE model across all 4 resolution tiers:
- 16px (trained native tier)
- 32px (untrained zero-shot generalization tier)
- 64px (trained native tier)
- 128px (trained native tier)

Features:
1. Strict Held-Out Evaluation: Uses independent test_normal and test_defects splits.
2. Dual-Threshold Policy Reporting: Direct comparison between Global Threshold (T)
   and Per-Resolution Threshold (T_r) at every resolution.
3. Uncertainty Estimation: 95% stratified bootstrap confidence intervals on AUROC,
   overall recall, per-class recall, and False Positive Rate (FPR).
4. Industrial Error Counts: Confusion matrices reporting exact TP, FP, TN, FN counts.
5. Operational Review Burden: Expected false alarms per 1,000 and 10,000 inspected joints.
6. Diagnostic Plots: Multi-tier ROC curves and Precision-Recall (PR) curves.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_ALL_TIERS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SYNTHETIC_DATA_DIR,
    DEFAULT_THRESHOLD_CONFIG,
    FALLBACK_THRESHOLD,
)
from src.data.dataset import SolderPatchDataset
from src.evaluation.bootstrap import BootstrapResult, bootstrap_ci, stratified_bootstrap_ci
from src.evaluation.confusion import (
    ConfusionMetrics,
    compute_confusion_matrix_and_metrics,
    compute_pr_metrics,
)
from src.inference.predict import inspect_patch
from src.model.cae import load_trained_model


DEFECT_CLASSES = ["voids", "bridging", "cold_joints", "solder_amount"]


def evaluate_tier_under_threshold(
    y_true: np.ndarray,
    norm_scores: List[float],
    all_defect_scores: List[float],
    defect_scores_by_class: Dict[str, List[float]],
    threshold: float,
    threshold_name: str,
    n_bootstrap: int = 1500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate performance metrics, uncertainty intervals, and confusion counts for a single threshold."""
    y_scores = np.concatenate([np.array(norm_scores), np.array(all_defect_scores)])

    # Confusion matrix and operational review burden
    cm: ConfusionMetrics = compute_confusion_matrix_and_metrics(y_true, y_scores, threshold=threshold)

    # Uncertainty: 95% Bootstrap Confidence Intervals
    fpr_ci: BootstrapResult = bootstrap_ci(
        norm_scores,
        lambda s: float(np.mean(s > threshold)),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    recall_ci: BootstrapResult = bootstrap_ci(
        all_defect_scores,
        lambda s: float(np.mean(s > threshold)),
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )

    # Per-class recall and CI
    class_recalls: Dict[str, Dict[str, Any]] = {}
    for dclass in DEFECT_CLASSES:
        d_scores = defect_scores_by_class.get(dclass, [])
        if len(d_scores) > 0:
            c_ci = bootstrap_ci(
                d_scores,
                lambda s: float(np.mean(s > threshold)),
                n_bootstrap=n_bootstrap,
                seed=seed + 2,
            )
            class_recalls[dclass] = c_ci.to_dict()
        else:
            class_recalls[dclass] = {"point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    return {
        "threshold": float(threshold),
        "threshold_policy": threshold_name,
        "confusion_matrix": cm.to_dict(),
        "fpr": fpr_ci.to_dict(),
        "overall_recall": recall_ci.to_dict(),
        "class_recalls": class_recalls,
    }


def evaluate_tier_dataset(
    model: torch.nn.Module,
    tier_size: int,
    norm_dir: Path,
    defect_base_dir: Path,
    global_threshold: float,
    per_tier_threshold: float,
    device: torch.device,
    n_bootstrap: int = 1500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate a specific resolution tier with both global and per-tier thresholds on held-out test data."""
    # 1. Load Held-Out Normal Patches
    norm_tier_dir = norm_dir / f"{tier_size}px"
    if not norm_tier_dir.exists():
        raise FileNotFoundError(f"Held-out normal directory not found: {norm_tier_dir}")

    norm_ds = SolderPatchDataset(norm_tier_dir)
    norm_scores = []
    for i in range(len(norm_ds)):
        t, _ = norm_ds[i]
        res = inspect_patch(model, t, device=device)
        norm_scores.append(float(res["anomaly_score"]))

    # 2. Load Held-Out Defect Classes
    defect_scores_by_class: Dict[str, List[float]] = {}
    all_defect_scores: List[float] = []

    for dclass in DEFECT_CLASSES:
        d_dir = defect_base_dir / dclass / f"{tier_size}px"
        if not d_dir.exists():
            continue
        d_ds = SolderPatchDataset(d_dir)
        d_scores = []
        for i in range(len(d_ds)):
            t, _ = d_ds[i]
            res = inspect_patch(model, t, device=device)
            d_scores.append(float(res["anomaly_score"]))

        defect_scores_by_class[dclass] = d_scores
        all_defect_scores.extend(d_scores)

    # 3. Ground truth binary labels (0 = normal, 1 = defect)
    y_true = np.array([0] * len(norm_scores) + [1] * len(all_defect_scores), dtype=np.int32)
    y_scores = np.array(norm_scores + all_defect_scores, dtype=np.float64)

    # 4. Discrimination metrics (Threshold-Independent)
    try:
        auroc_ci: BootstrapResult = stratified_bootstrap_ci(
            y_true,
            y_scores,
            lambda yt, ys: float(roc_auc_score(yt, ys)),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
    except Exception:
        auroc_ci = BootstrapResult(0.0, 0.0, 0.0, 0.0)
        fpr_curve, tpr_curve = np.array([0, 1]), np.array([0, 1])

    # Precision-Recall curve and Average Precision (AP)
    pr_metrics = compute_pr_metrics(y_true, y_scores)

    # 5. Threshold Operating Point A: Global Threshold
    global_eval = evaluate_tier_under_threshold(
        y_true=y_true,
        norm_scores=norm_scores,
        all_defect_scores=all_defect_scores,
        defect_scores_by_class=defect_scores_by_class,
        threshold=global_threshold,
        threshold_name="global_threshold",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    # 6. Threshold Operating Point B: Per-Tier Threshold
    tier_eval = evaluate_tier_under_threshold(
        y_true=y_true,
        norm_scores=norm_scores,
        all_defect_scores=all_defect_scores,
        defect_scores_by_class=defect_scores_by_class,
        threshold=per_tier_threshold,
        threshold_name="per_tier_threshold",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    return {
        "tier": tier_size,
        "is_untrained": (tier_size == 32),
        "sample_counts": {
            "normal_samples": len(norm_scores),
            "defect_samples": len(all_defect_scores),
            "total_samples": len(y_true),
        },
        "score_statistics": {
            "normal_mean": float(np.mean(norm_scores)),
            "normal_std": float(np.std(norm_scores)),
            "defect_mean": float(np.mean(all_defect_scores)),
            "defect_std": float(np.std(all_defect_scores)),
        },
        "auroc": auroc_ci.to_dict(),
        "average_precision": pr_metrics["average_precision"],
        "operating_points": {
            "global_threshold": global_eval,
            "per_tier_threshold": tier_eval,
        },
        "curves": {
            "roc_fpr": fpr_curve.tolist(),
            "roc_tpr": tpr_curve.tolist(),
            "pr_precision": pr_metrics["precision_curve"],
            "pr_recall": pr_metrics["recall_curve"],
        },
    }


def run_full_evaluation(
    checkpoint_path: str = str(DEFAULT_CHECKPOINT_PATH),
    config_path: str = str(DEFAULT_THRESHOLD_CONFIG),
    data_dir: str = str(DEFAULT_SYNTHETIC_DATA_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    use_val_split: bool = False,
    n_bootstrap: int = 1500,
) -> Dict[str, Any]:
    """Execute complete split-safe evaluation with dual-threshold reporting and uncertainty intervals."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105)
    print("SolSight Hardened Benchmark Evaluation (Milestone 1 — Split-Safe & Operational)")
    print(f"Loading checkpoint : {checkpoint_path}")
    print(f"Device             : {device}")
    print(f"Data Root          : {data_dir}")
    print("=" * 105)

    data_base = Path(data_dir)
    # Determine split paths
    if use_val_split:
        print("Note: Evaluating on legacy validation split (--use-val-split specified)")
        norm_dir = data_base / "val_normal"
        defect_dir = data_base / "val_defects"
        split_role_name = "validation_set"
    else:
        norm_dir = data_base / "test_normal"
        defect_dir = data_base / "test_defects"
        # Fallback if test splits are not found
        if not norm_dir.exists() or len(list(norm_dir.glob("*/*.png"))) == 0:
            print("Held-out test split not found; falling back to legacy val split.")
            norm_dir = data_base / "val_normal"
            defect_dir = data_base / "val_defects"
            split_role_name = "legacy_validation_fallback"
        else:
            split_role_name = "independent_held_out_test_set"

    print(f"Evaluation Split   : {split_role_name} ({norm_dir})")

    # Load threshold configuration
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        print(f"Warning: {cfg_file} not found. Running with fallback threshold {FALLBACK_THRESHOLD:.4f}.")
        global_threshold = FALLBACK_THRESHOLD
        per_res_thresholds = {str(t): FALLBACK_THRESHOLD for t in DEFAULT_ALL_TIERS}
        tier_32_policy = {}
    else:
        with open(cfg_file, "r") as f:
            cfg_data = json.load(f)
        global_threshold = float(cfg_data["global_threshold"])
        per_res_thresholds = {str(k): float(v) for k, v in cfg_data.get("per_resolution_thresholds", {}).items()}
        tier_32_policy = cfg_data.get("tier_32_operational_policy", {})

    print(f"Global Anomaly Threshold T: {global_threshold:.4f}")
    print("Per-Resolution Thresholds : " + ", ".join([f"{k}px={v:.4f}" for k, v in per_res_thresholds.items()]))

    # Load model
    model, ckpt_meta = load_trained_model(checkpoint_path, device=device)

    tiers = list(DEFAULT_ALL_TIERS)
    results: Dict[str, Any] = {}

    print("\n" + "=" * 105)
    print("EVALUATION RESULTS: OPERATING POINT COMPARISON (GLOBAL T vs PER-TIER T)")
    print("=" * 105)
    print(
        f"{'Tier':<12} | {'Role':<12} | {'AUROC [95% CI]':<26} | "
        f"{'Global-T (FPR / Recall / Review-10k)':<32} | {'Per-Tier-T (FPR / Recall / Review-10k)'}"
    )
    print("-" * 105)

    for tier in tiers:
        t_tier = per_res_thresholds.get(str(tier), global_threshold)
        tier_res = evaluate_tier_dataset(
            model=model,
            tier_size=tier,
            norm_dir=norm_dir,
            defect_base_dir=defect_dir,
            global_threshold=global_threshold,
            per_tier_threshold=t_tier,
            device=device,
            n_bootstrap=n_bootstrap,
            seed=42 + tier,
        )
        results[f"{tier}px"] = tier_res

        tier_label = f"{tier}px"
        role_label = "UNTRAINED" if tier == 32 else "TRAINED"

        auc_data = tier_res["auroc"]
        auc_str = f"{auc_data['point_estimate']:.3f} [{auc_data['ci_lower']:.3f}-{auc_data['ci_upper']:.3f}]"

        g_op = tier_res["operating_points"]["global_threshold"]
        g_fpr = g_op["fpr"]["point_estimate"]
        g_rec = g_op["overall_recall"]["point_estimate"]
        g_burden = g_op["confusion_matrix"]["operational_review_burden"]["per_10000_joints"]
        g_str = f"{g_fpr:>5.1%} / {g_rec:>5.1%} / {int(g_burden):>4d}"

        t_op = tier_res["operating_points"]["per_tier_threshold"]
        t_fpr = t_op["fpr"]["point_estimate"]
        t_rec = t_op["overall_recall"]["point_estimate"]
        t_burden = t_op["confusion_matrix"]["operational_review_burden"]["per_10000_joints"]
        t_str = f"{t_fpr:>5.1%} / {t_rec:>5.1%} / {int(t_burden):>4d}"

        print(f"{tier_label:<12} | {role_label:<12} | {auc_str:<26} | {g_str:<32} | {t_str}")

    print("-" * 105)

    # Detailed Breakdown Table
    print("\n" + "=" * 105)
    print("DETAILED PER-DEFECT RECALL & CONFUSION COUNTS (Under Operational Per-Tier Threshold)")
    print("=" * 105)
    print(f"{'Tier':<8} | {'T_tier':<8} | {'TP':<4} | {'FP':<4} | {'TN':<4} | {'FN':<4} | {'Precision':<9} | {'Voids Rec':<11} | {'Bridge Rec':<11} | {'Cold-J Rec':<11} | {'Amount Rec'}")
    print("-" * 105)

    for tier in tiers:
        t_op = results[f"{tier}px"]["operating_points"]["per_tier_threshold"]
        counts = t_op["confusion_matrix"]["counts"]
        crec = t_op["class_recalls"]
        prec = t_op["confusion_matrix"]["metrics"]["precision"]
        t_val = t_op["threshold"]

        print(
            f"{tier:>4}px | {t_val:>6.4f} | {counts['tp']:>4d} | {counts['fp']:>4d} | {counts['tn']:>4d} | {counts['fn']:>4d} | "
            f"{prec:>8.1%} | {crec['voids']['point_estimate']:>10.1%} | {crec['bridging']['point_estimate']:>10.1%} | "
            f"{crec['cold_joints']['point_estimate']:>10.1%} | {crec['solder_amount']['point_estimate']:>10.1%}"
        )
    print("=" * 105)

    # Plot 1: Multi-Tier ROC Curves
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8.5, 7.5))
    colors = {16: "#2ca02c", 32: "#ff7f0e", 64: "#1f77b4", 128: "#9467bd"}

    for tier in tiers:
        tr = results[f"{tier}px"]
        auc_val = tr["auroc"]["point_estimate"]
        ci_l = tr["auroc"]["ci_lower"]
        ci_u = tr["auroc"]["ci_upper"]
        tier_name = f"{tier}px (Untrained)" if tier == 32 else f"{tier}px"
        lbl = f"{tier_name} [AUC={auc_val:.3f} (95% CI: {ci_l:.3f}–{ci_u:.3f})]"

        plt.plot(
            tr["curves"]["roc_fpr"],
            tr["curves"]["roc_tpr"],
            label=lbl,
            color=colors.get(tier, "black"),
            linewidth=2.2 if tier == 32 else 1.8,
            linestyle="--" if tier == 32 else "-",
        )

        # Plot Per-Tier operating point marker
        top_fpr = tr["operating_points"]["per_tier_threshold"]["fpr"]["point_estimate"]
        top_rec = tr["operating_points"]["per_tier_threshold"]["overall_recall"]["point_estimate"]
        plt.scatter(top_fpr, top_rec, color=colors.get(tier, "black"), s=50, zorder=5)

    plt.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Chance Level (AUC=0.500)")
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("SolSight Multi-Resolution ROC Curves on Held-Out Test Data", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    roc_plot_path = out_p / "roc_curves.png"
    plt.savefig(roc_plot_path, dpi=160)
    plt.close()
    print(f"\nSaved ROC curves plot to: {roc_plot_path}")

    # Plot 2: Precision-Recall Curves
    plt.figure(figsize=(8.5, 7.5))
    for tier in tiers:
        tr = results[f"{tier}px"]
        ap_val = tr["average_precision"]
        tier_name = f"{tier}px (Untrained)" if tier == 32 else f"{tier}px"
        lbl = f"{tier_name} (AP={ap_val:.3f})"
        plt.plot(
            tr["curves"]["pr_recall"],
            tr["curves"]["pr_precision"],
            label=lbl,
            color=colors.get(tier, "black"),
            linewidth=2.0 if tier == 32 else 1.6,
            linestyle="--" if tier == 32 else "-",
        )

    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("SolSight Precision-Recall Curves on Held-Out Test Data", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    pr_plot_path = out_p / "pr_curves.png"
    plt.savefig(pr_plot_path, dpi=160)
    plt.close()
    print(f"Saved Precision-Recall curves plot to: {pr_plot_path}")

    # Save Complete Structured Summary JSON
    summary_path = out_p / "evaluation_metrics.json"
    full_output = {
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "split_role": split_role_name,
            "normal_dir": str(norm_dir),
            "defect_dir": str(defect_dir),
            "global_threshold": global_threshold,
            "per_resolution_thresholds": per_res_thresholds,
            "tier_32_policy": tier_32_policy,
            "n_bootstrap": n_bootstrap,
        },
        "results_by_tier": results,
    }

    with open(summary_path, "w") as f:
        json.dump(full_output, f, indent=2)

    print(f"Saved hardened evaluation metrics to: {summary_path}")
    print("=" * 105)

    return full_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SolSight Hardened Benchmark Evaluation")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=str(DEFAULT_THRESHOLD_CONFIG), help="Threshold config path")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_SYNTHETIC_DATA_DIR), help="Dataset root")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Outputs directory")
    parser.add_argument("--use-val-split", action="store_true", help="Evaluate on legacy validation split instead of test")
    parser.add_argument("--bootstrap-rounds", type=int, default=1500, help="Number of bootstrap iterations for CI")
    args = parser.parse_args()

    run_full_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_val_split=args.use_val_split,
        n_bootstrap=args.bootstrap_rounds,
    )
