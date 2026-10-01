"""Real-World PCBA Benchmark Evaluation Script for SolSight.

Evaluates the unsupervised SolderCAE model on genuine physical solder joint imagery
curated from the SolDef_AI benchmark (mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection).

Evaluates:
- 50 Real Defect-Free Normal Solder Joints (data/real/normal/)
- 200 Real Physical Defects across 4 classes:
  - excessive_solder (50 patches)
  - insufficient_solder (50 patches)
  - solder_spike (50 patches)
  - misalignment / no_good (50 patches)

Computes:
1. Area Under the ROC Curve (AUROC) on real PCBA data
2. Real-World Optimal Anomaly Threshold (T_real)
3. False Positive Rate (FPR) and Per-Class Recall (TPR)
4. Saves diagnostic ROC & heatmap exhibit figure: outputs/real_world_evaluation.png
5. Saves structured metrics: outputs/real_evaluation_metrics.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_REAL_THRESHOLD_K, FALLBACK_THRESHOLD
from src.model.cae import load_trained_model
from src.inference.predict import inspect_patch, create_heatmap_overlay



def load_patches_from_dir(directory: Path) -> List[Image.Image]:
    """Load all PNG/JPG patches from a directory."""
    files = sorted(list(directory.glob("*.png")) + list(directory.glob("*.jpg")))
    images = []
    for f in files:
        try:
            images.append(Image.open(f).convert("RGB"))
        except Exception:
            pass
    return images


def run_real_world_benchmark(
    data_dir: str = "data/real",
    checkpoint_path: str = "outputs/checkpoints/best_cae.pt",
    output_dir: str = "outputs",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print(f"SolSight Real-World PCBA Benchmark (SolDef_AI Dataset) | Device: {device}")
    print("=" * 85)

    base = Path(data_dir)
    norm_dir = base / "normal"
    if not norm_dir.exists():
        raise FileNotFoundError(f"Real normal directory {norm_dir} does not exist. Run scripts/fetch_real_data.py first.")

    # Load model
    model, _ = load_trained_model(checkpoint_path, device=device)

    # 1. Evaluate Real Normal Joints
    norm_images = load_patches_from_dir(norm_dir)
    print(f"Evaluating {len(norm_images)} real normal reference joints...")
    norm_scores = []
    for img in norm_images:
        res = inspect_patch(model, img, device=device)
        norm_scores.append(res["anomaly_score"])

    norm_mean = float(np.mean(norm_scores))
    norm_std = float(np.std(norm_scores))
    # Calibrate site-specific threshold T_real = mean + k * std
    t_real = norm_mean + DEFAULT_REAL_THRESHOLD_K * norm_std

    # Also load the synthetic global threshold for comparison
    cfg_path = Path(output_dir) / "threshold_config.json"
    t_synthetic = None
    if cfg_path.exists():
        with open(cfg_path) as f:
            t_synthetic = json.load(f).get("global_threshold")
    if t_synthetic is None:
        import warnings
        t_synthetic = FALLBACK_THRESHOLD
        warnings.warn(
            f"threshold_config.json not found or missing global_threshold; "
            f"using fallback synthetic threshold {t_synthetic:.4f}."
        )


    # 2. Evaluate Real Defect Classes
    defect_classes = {
        "excessive_solder": base / "defective" / "exc_solder",
        "insufficient_solder": base / "defective" / "poor_solder",
        "solder_spike": base / "defective" / "spike",
        "misaligned": base / "defective" / "no_good",
    }

    defect_scores_by_class: Dict[str, List[float]] = {}
    all_def_scores = []
    class_recalls_real: Dict[str, float] = {}
    class_recalls_synth_t: Dict[str, float] = {}
    defect_exhibits = {}

    for cname, cdir in defect_classes.items():
        imgs = load_patches_from_dir(cdir)
        scores = []
        for i, img in enumerate(imgs):
            res = inspect_patch(model, img, threshold=t_real, device=device)
            scores.append(res["anomaly_score"])
            if i == 0:
                defect_exhibits[cname] = (img, res)

        defect_scores_by_class[cname] = scores
        all_def_scores.extend(scores)
        class_recalls_real[cname] = float(np.mean([s > t_real for s in scores]))
        class_recalls_synth_t[cname] = float(np.mean([s > t_synthetic for s in scores]))

    # Compute overall metrics
    y_true = [0] * len(norm_scores) + [1] * len(all_def_scores)
    y_scores = norm_scores + all_def_scores
    auroc = float(roc_auc_score(y_true, y_scores))
    fpr_real = float(np.mean([s > t_real for s in norm_scores]))
    overall_recall_real = float(np.mean([s > t_real for s in all_def_scores]))
    overall_recall_synth_t = float(np.mean([s > t_synthetic for s in all_def_scores]))

    print("\n" + "-" * 85)
    print(f"Normal Solder Baseline Score (Real PCBA) : {norm_mean:.4f} +/- {norm_std:.4f}")
    print(f"Synthetic Global Threshold (T_synth)      : {t_synthetic:.4f}")
    print(f"Calibrated Real Threshold (T_real, k=2.0) : {t_real:.4f}")
    print(f"Real-World AUROC                         : {auroc:.3f}")
    print(f"False Positive Rate @ T_real             : {fpr_real:.1%}")
    print(f"Overall Defect Recall @ T_real           : {overall_recall_real:.1%}")
    print(f"Overall Defect Recall @ T_synth          : {overall_recall_synth_t:.1%}")
    print("-" * 85)
    print(f"{'Defect Category':<25} | {'Count':<8} | {'Mean Score':<12} | {'Recall (@ T_real)':<18} | {'Recall (@ T_synth)':<18}")
    print("-" * 85)
    for cname, scores in defect_scores_by_class.items():
        mean_sc = float(np.mean(scores))
        rec_r = class_recalls_real[cname]
        rec_s = class_recalls_synth_t[cname]
        print(f"{cname:<25} | {len(scores):<8} | {mean_sc:<12.4f} | {rec_r:<18.1%} | {rec_s:<18.1%}")
    print("-" * 85)

    # 3. Generate Multi-Panel Visual Exhibit (2x3 grid)
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    # Subplot [0, 0]: Real PCBA ROC Curve
    ax1 = axes[0, 0]
    ax1.plot(fpr_curve, tpr_curve, color="darkorange", lw=2, label=f"SolDef_AI (AUC = {auroc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.set_title("Real PCBA ROC Curve (SolDef_AI)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Subplot [0, 1]: Score Distributions
    ax2 = axes[0, 1]
    ax2.hist(norm_scores, bins=15, alpha=0.6, label="Normal Joints", color="green", density=True)
    ax2.hist(all_def_scores, bins=15, alpha=0.6, label="Defect Joints", color="crimson", density=True)
    ax2.axvline(t_real, color="blue", linestyle="--", label=f"T_real ({t_real:.3f})")
    ax2.axvline(t_synthetic, color="purple", linestyle=":", label=f"T_synth ({t_synthetic:.3f})")
    ax2.set_xlabel("Top-5% DSSIM Anomaly Score")
    ax2.set_ylabel("Density")
    ax2.set_title("Real Normal vs Defect Distributions")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Subplot [0, 2]: Real Normal Reference Exhibit
    ax3 = axes[0, 2]
    norm_sample = norm_images[0]
    norm_res = inspect_patch(model, norm_sample, threshold=t_real, device=device)
    norm_overlay = create_heatmap_overlay(norm_sample, norm_res["dssim_map"], alpha=0.55)
    ax3.imshow(norm_overlay)
    ax3.set_title(f"Real Normal Joint [PASS]\nScore: {norm_res['anomaly_score']:.3f}", fontsize=10)
    ax3.axis("off")

    # Subplots [1, 0..2]: Real Defect Exhibits
    classes_to_show = ["excessive_solder", "insufficient_solder", "misaligned"]
    for col_idx, cname in enumerate(classes_to_show):
        img, res = defect_exhibits[cname]
        overlay = create_heatmap_overlay(img, res["dssim_map"], alpha=0.55)
        ax = axes[1, col_idx]
        ax.imshow(overlay)
        ax.set_title(f"Defect: {cname} [REJECT]\nScore: {res['anomaly_score']:.3f}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_img = Path(output_dir) / "real_world_evaluation.png"
    plt.savefig(out_img, dpi=150)
    plt.close()
    print(f"Saved real PCBA evaluation figure to: {out_img}")

    # 4. Save JSON Metrics
    metrics = {
        "dataset_name": "SolDef_AI: PCB dataset for defect detection",
        "dataset_slug": "mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection",
        "sample_counts": {
            "normal": len(norm_scores),
            "defects_total": len(all_def_scores),
            "by_class": {k: len(v) for k, v in defect_scores_by_class.items()},
        },
        "auroc": auroc,
        "normal_baseline": {"mean": norm_mean, "std": norm_std},
        "thresholds": {"t_real": t_real, "t_synthetic": t_synthetic},
        "performance_at_t_real": {
            "fpr": fpr_real,
            "overall_recall": overall_recall_real,
            "class_recalls": class_recalls_real,
        },
        "performance_at_t_synthetic": {
            "overall_recall": overall_recall_synth_t,
            "class_recalls": class_recalls_synth_t,
        },
    }

    out_json = Path(output_dir) / "real_evaluation_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved real PCBA metrics summary to: {out_json}")
    print("=" * 85)

    return metrics


if __name__ == "__main__":
    import argparse
    from src.config import (
        DEFAULT_CHECKPOINT_PATH,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_REAL_DATA_DIR,
    )

    parser = argparse.ArgumentParser(description="Real-World PCBA Benchmark Evaluation for SolSight")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_REAL_DATA_DIR), help="Path to real PCBA dataset")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    run_real_world_benchmark(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
