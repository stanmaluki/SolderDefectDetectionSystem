"""Evaluation Script for SolSight (Implementation Plan Phase 5.2 & 5.3).

Evaluates the multi-scale trained model across all 4 resolution tiers:
- 16px (trained native tier)
- 32px (UNTRAINED zero-shot generalization tier)
- 64px (trained native tier)
- 128px (trained native tier)

Reports:
1. False Positive Rate (FPR) on held-out defect-free normal joints.
2. Recall (TPR) broken down per defect class: voids, bridging, cold_joints, solder_amount.
3. Area Under the ROC Curve (AUROC) per tier.
4. Generates ROC curves plot saved to outputs/roc_curves.png.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch

from src.model.cae import load_trained_model
from src.inference.predict import inspect_patch
from src.data.dataset import SolderPatchDataset


def evaluate_tier_dataset(
    model: torch.nn.Module,
    tier_size: int,
    data_base: Path,
    threshold: float,
    device: torch.device,
) -> dict:
    """Evaluate normal and defect datasets for a specific resolution tier."""
    # 1. Normal patches
    norm_dir = data_base / f"val_normal/{tier_size}px"
    norm_ds = SolderPatchDataset(norm_dir)
    norm_scores = []
    for i in range(len(norm_ds)):
        t, _ = norm_ds[i]
        res = inspect_patch(model, t, device=device)
        norm_scores.append(res["anomaly_score"])

    fpr = float(np.mean([s > threshold for s in norm_scores]))

    # 2. Defect classes
    defect_classes = ["voids", "bridging", "cold_joints", "solder_amount"]
    defect_scores_by_class: Dict[str, List[float]] = {}
    class_recalls: Dict[str, float] = {}
    all_defect_scores = []

    for dclass in defect_classes:
        d_dir = data_base / f"val_defects/{dclass}/{tier_size}px"
        d_ds = SolderPatchDataset(d_dir)
        d_scores = []
        for i in range(len(d_ds)):
            t, _ = d_ds[i]
            res = inspect_patch(model, t, device=device)
            d_scores.append(res["anomaly_score"])

        defect_scores_by_class[dclass] = d_scores
        all_defect_scores.extend(d_scores)
        recall = float(np.mean([s > threshold for s in d_scores])) if d_scores else 0.0
        class_recalls[dclass] = recall

    overall_recall = float(np.mean([s > threshold for s in all_defect_scores]))

    # 3. AUROC
    y_true = [0] * len(norm_scores) + [1] * len(all_defect_scores)
    y_scores = norm_scores + all_defect_scores
    try:
        auroc = float(roc_auc_score(y_true, y_scores))
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
    except Exception:
        auroc = 0.0
        fpr_curve, tpr_curve = np.array([0, 1]), np.array([0, 1])

    return {
        "tier": tier_size,
        "is_untrained": (tier_size == 32),
        "fpr": fpr,
        "overall_recall": overall_recall,
        "class_recalls": class_recalls,
        "auroc": auroc,
        "fpr_curve": fpr_curve.tolist(),
        "tpr_curve": tpr_curve.tolist(),
        "norm_scores": norm_scores,
        "defect_scores": all_defect_scores,
    }


def run_full_evaluation(
    checkpoint_path: str = "outputs/checkpoints/best_cae.pt",
    config_path: str = "outputs/threshold_config.json",
    data_dir: str = "data/synthetic",
    output_dir: str = "outputs",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("SolSight Comprehensive Defect Detection & Resolution Generalization Evaluation")
    print(f"Loading checkpoint: {checkpoint_path} | Device: {device}")
    print("=" * 95)

    # Load threshold configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    global_threshold = config["global_threshold"]
    print(f"Global Anomaly Threshold T: {global_threshold:.4f} (selected k={config.get('selected_k', 2.5)})")

    # Load model
    model, _ = load_trained_model(checkpoint_path, device=device)

    tiers = [16, 32, 64, 128]
    data_base = Path(data_dir)
    results = {}

    print("\n" + "-" * 95)
    print(f"{'Tier':<14} | {'Type':<12} | {'FPR':<8} | {'Overall Rec':<12} | {'Voids':<8} | {'Bridge':<8} | {'Cold J':<8} | {'Amount':<8} | {'AUROC':<8}")
    print("-" * 95)

    plt.figure(figsize=(8, 7))

    for tier in tiers:
        # Note: 32px ALWAYS uses global threshold T
        tier_res = evaluate_tier_dataset(model, tier, data_base, global_threshold, device)
        results[f"{tier}px"] = tier_res

        tier_label = "32px (UNTRAINED)" if tier == 32 else f"{tier}px"
        tier_type = "GENERALIZE" if tier == 32 else "TRAINED"
        cr = tier_res["class_recalls"]

        print(
            f"{tier_label:<14} | {tier_type:<12} | {tier_res['fpr']:<8.1%} | {tier_res['overall_recall']:<12.1%} | "
            f"{cr['voids']:<8.1%} | {cr['bridging']:<8.1%} | {cr['cold_joints']:<8.1%} | {cr['solder_amount']:<8.1%} | "
            f"{tier_res['auroc']:<8.3f}"
        )

        plt.plot(
            tier_res["fpr_curve"],
            tier_res["tpr_curve"],
            label=f"{tier_label} (AUC = {tier_res['auroc']:.3f})",
            linewidth=2.0 if tier == 32 else 1.5,
            linestyle="--" if tier == 32 else "-",
        )

    print("-" * 95)

    # Plot ROC curves
    plt.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random Guess")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("SolSight ROC Curves Across Trained and Untrained Generalization Tiers")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_p = Path(output_dir)
    roc_plot_path = out_p / "roc_curves.png"
    plt.savefig(roc_plot_path, dpi=150)
    plt.close()
    print(f"\nSaved ROC curves plot to: {roc_plot_path}")

    # Save metrics summary JSON
    summary_path = out_p / "evaluation_metrics.json"
    clean_results = {
        k: {
            "tier": v["tier"],
            "is_untrained": v["is_untrained"],
            "fpr": v["fpr"],
            "overall_recall": v["overall_recall"],
            "class_recalls": v["class_recalls"],
            "auroc": v["auroc"],
        }
        for k, v in results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"Saved evaluation metrics to: {summary_path}")
    print("=" * 95)

    return clean_results


if __name__ == "__main__":
    import argparse
    from src.config import (
        DEFAULT_CHECKPOINT_PATH,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SYNTHETIC_DATA_DIR,
        DEFAULT_THRESHOLD_CONFIG,
    )

    parser = argparse.ArgumentParser(description="Multi-Scale Benchmark Evaluation for SolSight")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=str(DEFAULT_THRESHOLD_CONFIG), help="Threshold config path")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_SYNTHETIC_DATA_DIR), help="Dataset root")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Outputs directory")
    args = parser.parse_args()

    run_full_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
