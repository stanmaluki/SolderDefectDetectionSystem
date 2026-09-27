"""Threshold Calibration Script (Implementation Plan Phase 4.3).

Calibrates anomaly thresholds on held-out defect-free validation patches across
the trained native tiers {16, 64, 128}.

Key deliverables:
1. Always computes and saves global threshold T = mean(all_scores) + k * std(all_scores).
2. Computes per-resolution thresholds T_r = mean_r + k * std_r.
3. Tests zero-shot generalization of global T on the untrained 32px validation normals.
4. Outputs JSON configuration to outputs/threshold_config.json.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Dict, List
import numpy as np
import torch

from src.model.cae import load_trained_model
from src.inference.predict import inspect_patch
from src.data.dataset import SolderPatchDataset


def calibrate(
    checkpoint_path: str = "outputs/checkpoints/best_cae.pt",
    val_normal_dir: str = "data/synthetic/val_normal",
    output_config: str = "outputs/threshold_config.json",
    k_values: List[float] = [2.0, 2.5, 3.0],
    selected_k: float = 2.5,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("SolSight Anomaly Threshold Calibration")
    print(f"Loading checkpoint: {checkpoint_path} | Device: {device}")
    print("=" * 80)

    # Load model
    model, _ = load_trained_model(checkpoint_path, device=device)

    val_base = Path(val_normal_dir)
    trained_tiers = [16, 64, 128]
    untrained_tier = 32

    # Collect scores per tier
    scores_by_tier: Dict[int, List[float]] = {}
    all_trained_scores: List[float] = []

    for tier in trained_tiers:
        tier_dir = val_base / f"{tier}px"
        ds = SolderPatchDataset(tier_dir)
        tier_scores = []
        for i in range(len(ds)):
            tensor, _ = ds[i]
            res = inspect_patch(model, tensor, device=device)
            tier_scores.append(res["anomaly_score"])

        scores_by_tier[tier] = tier_scores
        all_trained_scores.extend(tier_scores)
        print(f"Tier {tier}px ({len(tier_scores)} patches): mean={np.mean(tier_scores):.4f}, std={np.std(tier_scores):.4f}")

    # Untrained 32px tier scores
    untrained_dir = val_base / f"{untrained_tier}px"
    ds_32 = SolderPatchDataset(untrained_dir)
    scores_32 = []
    for i in range(len(ds_32)):
        tensor, _ = ds_32[i]
        res = inspect_patch(model, tensor, device=device)
        scores_32.append(res["anomaly_score"])
    print(f"Tier 32px (untrained test, {len(scores_32)} patches): mean={np.mean(scores_32):.4f}, std={np.std(scores_32):.4f}")

    all_arr = np.array(all_trained_scores)
    global_mean = float(all_arr.mean())
    global_std = float(all_arr.std())

    print("\n" + "-" * 80)
    print(f"{'k':<6} | {'Global Threshold T':<20} | {'FPR 16px':<10} | {'FPR 64px':<10} | {'FPR 128px':<10} | {'FPR 32px (Untrained)':<22}")
    print("-" * 80)

    calibration_results = {}

    for k in k_values:
        t_global = global_mean + k * global_std
        fpr_16 = float(np.mean([s > t_global for s in scores_by_tier[16]]))
        fpr_64 = float(np.mean([s > t_global for s in scores_by_tier[64]]))
        fpr_128 = float(np.mean([s > t_global for s in scores_by_tier[128]]))
        fpr_32 = float(np.mean([s > t_global for s in scores_32]))

        calibration_results[f"k_{k}"] = {
            "k": k,
            "global_threshold": t_global,
            "fpr_16px": fpr_16,
            "fpr_64px": fpr_64,
            "fpr_128px": fpr_128,
            "fpr_32px_untrained": fpr_32,
        }

        print(
            f"{k:<6.1f} | {t_global:<20.4f} | {fpr_16:<10.1%} | {fpr_64:<10.1%} | "
            f"{fpr_128:<10.1%} | {fpr_32:<22.1%}"
        )

    # Per-resolution thresholds at selected k
    per_res_thresholds = {}
    for tier in trained_tiers:
        m = float(np.mean(scores_by_tier[tier]))
        s = float(np.std(scores_by_tier[tier]))
        per_res_thresholds[str(tier)] = m + selected_k * s

    selected_global_t = global_mean + selected_k * global_std

    config_data = {
        "selected_k": selected_k,
        "global_threshold": selected_global_t,
        "global_mean": global_mean,
        "global_std": global_std,
        "per_resolution_thresholds": per_res_thresholds,
        "sweep_results": calibration_results,
    }

    out_file = Path(output_config)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(config_data, f, indent=2)

    print("-" * 80)
    print(f"Selected k={selected_k} | Global Threshold T={selected_global_t:.4f}")
    print(f"Saved threshold configuration to: {out_file}")
    print("=" * 80)

    return config_data


if __name__ == "__main__":
    calibrate()
