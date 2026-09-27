"""Empirical Model Stress Testing & Failure Boundary Analysis for SolSight.

Systematically probes the boundaries of the CAE + SSIM architecture to discover
exactly where and why the model breaks:

1. Resolution Breakdown Curve:
   Sweeps resolutions from 12px to 160px: [12, 14, 16, 20, 24, 32, 48, 64, 80, 96, 128, 160].
   Identifies the low-resolution collapse threshold (where SSIM window > patch).

2. Micro-Defect Size Limit (Pinhole Sensitivity):
   Sweeps defect area from <0.5% of patch area to 15% of patch area.
   Identifies the minimum detectable defect footprint before top-k DSSIM dilution occurs.

3. Sensor Noise Robustness:
   Injects sensor Gaussian noise sigma in [0.0, 0.02, 0.04, 0.06, 0.08, 0.12].
   Measures at what signal-to-noise ratio the model begins producing false rejects.

4. Saves diagnostic plots to outputs/stress_test/ and outputs JSON summary.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch

from src.model.cae import SolderCAE
from src.inference.predict import inspect_patch
from src.data.synthetic_generator import (
    render_normal_joint,
    render_defect_void,
    render_defect_bridging,
    render_defect_cold_joint,
    render_defect_solder_amount,
)


def run_resolution_stress_test(
    model: torch.nn.Module,
    global_threshold: float,
    device: torch.device,
    resolutions=(12, 14, 16, 20, 24, 32, 48, 64, 80, 96, 128, 160),
    samples_per_res: int = 40,
) -> dict:
    """Evaluate model performance across wide resolution spectrum."""
    print("\n" + "=" * 80)
    print("1. RESOLUTION BREAKDOWN STRESS TEST (12px to 160px)")
    print("=" * 80)
    print(f"{'Resolution':<12} | {'AUROC':<8} | {'FPR @ T':<10} | {'Recall @ T':<12} | {'Normal Score':<14} | {'Defect Score':<14}")
    print("-" * 80)

    res_results = {}
    rng = random.Random(101)

    for r in resolutions:
        normal_scores = []
        defect_scores = []

        # Generate test normals
        for _ in range(samples_per_res):
            img = render_normal_joint(r, rng)
            res = inspect_patch(model, img, device=device)
            normal_scores.append(res["anomaly_score"])

        # Generate test defects (mix of 4 classes)
        for i in range(samples_per_res):
            dtype = i % 4
            if dtype == 0:
                img = render_defect_void(r, rng)
            elif dtype == 1:
                img = render_defect_bridging(r, rng)
            elif dtype == 2:
                img = render_defect_cold_joint(r, rng)
            else:
                img = render_defect_solder_amount(r, rng)
            res = inspect_patch(model, img, device=device)
            defect_scores.append(res["anomaly_score"])

        y_true = [0] * len(normal_scores) + [1] * len(defect_scores)
        y_scores = normal_scores + defect_scores

        try:
            auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            auc = 0.5

        fpr = float(np.mean([s > global_threshold for s in normal_scores]))
        recall = float(np.mean([s > global_threshold for s in defect_scores]))
        mean_norm = float(np.mean(normal_scores))
        mean_def = float(np.mean(defect_scores))

        res_results[r] = {
            "resolution": r,
            "auroc": auc,
            "fpr": fpr,
            "recall": recall,
            "mean_norm": mean_norm,
            "mean_def": mean_def,
        }

        print(f"{r}x{r:<10} | {auc:<8.3f} | {fpr:<10.1%} | {recall:<12.1%} | {mean_norm:<14.4f} | {mean_def:<14.4f}")

    return res_results


def run_defect_size_stress_test(
    model: torch.nn.Module,
    global_threshold: float,
    device: torch.device,
    size: int = 64,
    scales=(0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0),
    samples_per_scale: int = 30,
) -> dict:
    """Find minimum detectable defect size."""
    print("\n" + "=" * 80)
    print("2. MICRO-DEFECT SIZE LIMIT TEST (Pinhole Sensitivity at 64px)")
    print("=" * 80)
    print(f"{'Scale Factor':<14} | {'Est Area %':<12} | {'Recall @ T':<12} | {'Mean Score':<12} | {'Detection Status'}")
    print("-" * 80)

    size_results = {}
    rng = random.Random(202)

    for sc in scales:
        scores = []
        for _ in range(samples_per_scale):
            img = render_defect_void(size, rng, size_scale=sc)
            res = inspect_patch(model, img, device=device)
            scores.append(res["anomaly_score"])

        recall = float(np.mean([s > global_threshold for s in scores]))
        mean_sc = float(np.mean(scores))
        # Estimate defect area as circle of radius ~ 0.10 * size * sc
        est_radius = 0.10 * size * sc
        est_area_pct = ((math.pi * est_radius ** 2) / (size ** 2)) * 100.0

        status = "DETECTED" if recall >= 0.70 else ("MARGINAL" if recall >= 0.30 else "MISSED (BREAKING POINT)")

        size_results[sc] = {
            "scale": sc,
            "est_area_pct": est_area_pct,
            "recall": recall,
            "mean_score": mean_sc,
            "status": status,
        }

        print(f"{sc:<14.2f} | {est_area_pct:<11.2f}% | {recall:<12.1%} | {mean_sc:<12.4f} | {status}")

    return size_results


def run_noise_stress_test(
    model: torch.nn.Module,
    global_threshold: float,
    device: torch.device,
    size: int = 64,
    noise_levels=(0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12),
    samples: int = 30,
) -> dict:
    """Measure false reject rate under sensor noise."""
    print("\n" + "=" * 80)
    print("3. SENSOR NOISE & SNR SENSITIVITY TEST (Normal Joints at 64px)")
    print("=" * 80)
    print(f"{'Noise Sigma':<14} | {'Normal Score':<14} | {'FPR @ T (False Rejects)':<25} | {'Robustness'}")
    print("-" * 80)

    noise_results = {}
    rng = random.Random(303)

    for sigma in noise_levels:
        scores = []
        for _ in range(samples):
            img = render_normal_joint(size, rng)
            arr = np.array(img, dtype=np.float32) / 255.0
            if sigma > 0:
                noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
                arr = np.clip(arr + noise, 0.0, 1.0).astype(np.float32)
            res = inspect_patch(model, arr, device=device)
            scores.append(res["anomaly_score"])

        fpr = float(np.mean([s > global_threshold for s in scores]))
        mean_sc = float(np.mean(scores))
        status = "STABLE" if fpr < 0.05 else ("DEGRADED" if fpr < 0.25 else "BROKEN")

        noise_results[sigma] = {
            "sigma": sigma,
            "mean_score": mean_sc,
            "fpr": fpr,
            "status": status,
        }

        print(f"{sigma:<14.2f} | {mean_sc:<14.4f} | {fpr:<25.1%} | {status}")

    return noise_results


def plot_stress_diagnostics(
    res_results: dict,
    size_results: dict,
    noise_results: dict,
    save_dir: Path,
) -> None:
    """Generate multi-panel diagnostic chart visualizing breaking points."""
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Resolution Breakdown
    res_keys = sorted(res_results.keys())
    aurocs = [res_results[k]["auroc"] for k in res_keys]
    recalls = [res_results[k]["recall"] for k in res_keys]
    axes[0].plot(res_keys, aurocs, "b-o", label="AUROC", linewidth=2.0)
    axes[0].plot(res_keys, recalls, "r--s", label="Recall @ T", linewidth=1.5)
    axes[0].axvline(15, color="gray", linestyle=":", label="15px Spec Floor")
    axes[0].set_xlabel("Patch Resolution (pixels)")
    axes[0].set_ylabel("Metric Score")
    axes[0].set_title("Resolution Breakdown Curve", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Defect Size Breaking Point
    sc_keys = sorted(size_results.keys())
    areas = [size_results[k]["est_area_pct"] for k in sc_keys]
    recs = [size_results[k]["recall"] for k in sc_keys]
    axes[1].plot(areas, recs, "g-^", linewidth=2.0)
    axes[1].axhline(0.5, color="red", linestyle="--", label="50% Detection Limit")
    axes[1].set_xlabel("Defect Footprint (% of Patch Area)")
    axes[1].set_ylabel("Recall Rate")
    axes[1].set_title("Micro-Defect Detection Limit", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Noise Robustness
    noise_keys = sorted(noise_results.keys())
    fprs = [noise_results[k]["fpr"] for k in noise_keys]
    axes[2].plot(noise_keys, fprs, "m-d", linewidth=2.0)
    axes[2].axhline(0.05, color="black", linestyle="--", label="5% Allowable FPR")
    axes[2].set_xlabel("Sensor Noise Sigma")
    axes[2].set_ylabel("False Positive Rate (FPR)")
    axes[2].set_title("Sensor Noise Sensitivity", fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("SolSight Failure Boundary & Stress-Test Diagnostics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = save_dir / "stress_diagnostics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved multi-panel stress diagnostics plot to: {plot_path}")


def run_full_stress_suite(
    checkpoint_path: str = "outputs/checkpoints/best_cae.pt",
    config_path: str = "outputs/threshold_config.json",
    output_dir: str = "outputs/stress_test",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"SolSight Model Hardening & Failure Boundary Stress Suite | Device: {device}")
    print("=" * 80)

    with open(config_path, "r") as f:
        config = json.load(f)
    global_t = config["global_threshold"]

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SolderCAE(in_channels=3, base_channels=16).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # Run tests
    res_results = run_resolution_stress_test(model, global_t, device)
    size_results = run_defect_size_stress_test(model, global_t, device)
    noise_results = run_noise_stress_test(model, global_t, device)

    # Plot
    plot_stress_diagnostics(res_results, size_results, noise_results, out_p)

    summary_data = {
        "global_threshold": global_t,
        "resolution_breakdown": {str(k): v for k, v in res_results.items()},
        "defect_size_limit": {str(k): v for k, v in size_results.items()},
        "noise_sensitivity": {str(k): v for k, v in noise_results.items()},
    }

    summary_file = out_p / "stress_test_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved stress test summary to: {summary_file}")
    print("=" * 80)

    return summary_data


if __name__ == "__main__":
    run_full_stress_suite()
