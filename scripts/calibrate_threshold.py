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

from src.config import (
    DEFAULT_ALL_TIERS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_SYNTHETIC_DATA_DIR,
    DEFAULT_THRESHOLD_CONFIG,
    DEFAULT_THRESHOLD_K,
    DEFAULT_TRAIN_TIERS,
    DEFAULT_VAL_CALIBRATION_DIR,
)
from src.model.cae import load_trained_model
from src.inference.predict import inspect_patch
from src.data.dataset import SolderPatchDataset


def calibrate(
    checkpoint_path: str = str(DEFAULT_CHECKPOINT_PATH),
    val_normal_dir: str = str(DEFAULT_VAL_CALIBRATION_DIR),
    output_config: str = str(DEFAULT_THRESHOLD_CONFIG),
    k_values: List[float] = [2.0, 2.5, 3.0],
    selected_k: float = DEFAULT_THRESHOLD_K,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("SolSight Anomaly Threshold Calibration (Independent Calibration Protocol)")
    print(f"Loading checkpoint : {checkpoint_path}")
    print(f"Compute Device     : {device}")
    print(f"Calibration Source : {val_normal_dir}")
    print("=" * 85)

    # Resolve calibration directory (with fallback to legacy val_normal if calibration dir not yet present)
    val_base = Path(val_normal_dir)
    if not val_base.exists() or len(list(val_base.glob("*"))) == 0:
        legacy_dir = DEFAULT_SYNTHETIC_DATA_DIR / "val_normal"
        if legacy_dir.exists():
            print(f"Notice: Calibration dir '{val_base}' not populated; falling back to legacy '{legacy_dir}'")
            val_base = legacy_dir

    # Load model
    model, _ = load_trained_model(checkpoint_path, device=device)

    trained_tiers = list(DEFAULT_TRAIN_TIERS)
    all_tiers = list(DEFAULT_ALL_TIERS)

    # Collect calibration normal joint scores per tier
    scores_by_tier: Dict[int, List[float]] = {}
    all_trained_scores: List[float] = []

    for tier in all_tiers:
        tier_dir = val_base / f"{tier}px"
        if not tier_dir.exists():
            continue
        ds = SolderPatchDataset(tier_dir)
        tier_scores = []
        for i in range(len(ds)):
            tensor, _ = ds[i]
            res = inspect_patch(model, tensor, device=device)
            tier_scores.append(res["anomaly_score"])

        scores_by_tier[tier] = tier_scores
        if tier in trained_tiers:
            all_trained_scores.extend(tier_scores)
        tier_type = "TRAINED" if tier in trained_tiers else "UNTRAINED_GENERALIZE"
        print(f"Tier {tier}px ({tier_type:<20}, {len(tier_scores)} patches): mean={np.mean(tier_scores):.4f}, std={np.std(tier_scores):.4f}")

    all_arr = np.array(all_trained_scores)
    global_mean = float(all_arr.mean())
    global_std = float(all_arr.std())

    print("\n" + "-" * 90)
    print(f"{'k':<6} | {'Global Threshold T':<20} | {'In-Sample 16px':<15} | {'In-Sample 64px':<15} | {'In-Sample 128px':<15} | {'In-Sample 32px':<15}")
    print("-" * 90)

    sweep_results = {}
    for k in k_values:
        t_global = global_mean + k * global_std
        in_sample_rates = {}
        for tier in all_tiers:
            if tier in scores_by_tier and len(scores_by_tier[tier]) > 0:
                in_sample_rates[tier] = float(np.mean([s > t_global for s in scores_by_tier[tier]]))
            else:
                in_sample_rates[tier] = 0.0

        sweep_results[f"k_{k}"] = {
            "k": k,
            "global_threshold": t_global,
            "in_sample_fpr_by_tier": {str(t): in_sample_rates[t] for t in all_tiers},
        }

        print(
            f"{k:<6.1f} | {t_global:<20.4f} | {in_sample_rates.get(16, 0.0):<15.1%} | {in_sample_rates.get(64, 0.0):<15.1%} | "
            f"{in_sample_rates.get(128, 0.0):<15.1%} | {in_sample_rates.get(32, 0.0):<15.1%}"
        )

    # Compute per-resolution thresholds at selected k
    per_res_thresholds: Dict[str, float] = {}
    per_res_stats: Dict[str, Dict[str, float]] = {}

    for tier in all_tiers:
        if tier in scores_by_tier and len(scores_by_tier[tier]) > 0:
            m = float(np.mean(scores_by_tier[tier]))
            s = float(np.std(scores_by_tier[tier]))
            per_res_thresholds[str(tier)] = float(m + selected_k * s)
            per_res_stats[str(tier)] = {"mean": m, "std": s}

    selected_global_t = float(global_mean + selected_k * global_std)

    # Explicit policy documentation for intermediate/untrained tier (32px)
    tier_32_calibrated = per_res_thresholds.get("32", selected_global_t)
    tier_32_policy = {
        "calibrated_operational": tier_32_calibrated,
        "global_prior": selected_global_t,
        "nearest_trained_16px": per_res_thresholds.get("16", selected_global_t),
        "selected_default": "calibrated_operational",
        "policy_rationale": (
            "Calibrated operational threshold derived from line validation normals accounts for "
            "spatial DSSIM receptive field differences without requiring model retraining."
        ),
    }

    config_data = {
        "selected_k": selected_k,
        "global_threshold": selected_global_t,
        "global_mean": global_mean,
        "global_std": global_std,
        "per_resolution_thresholds": per_res_thresholds,
        "per_resolution_statistics": per_res_stats,
        "tier_32_operational_policy": tier_32_policy,
        "sweep_results": sweep_results,
        "calibration_source": str(val_base),
        "note": "Operating point validation (FPR, recall, AUROC) must be performed strictly on held-out test splits.",
    }

    out_file = Path(output_config)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(config_data, f, indent=2)

    print("-" * 90)
    print(f"Selected k={selected_k} | Global Threshold T={selected_global_t:.4f}")
    for t_str, val in per_res_thresholds.items():
        print(f"  --> Per-Resolution T_{t_str}px : {val:.4f}")
    print(f"Saved threshold configuration to: {out_file}")
    print("=" * 90)

    return config_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate SolSight Anomaly Detection Thresholds")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Path to checkpoint")
    parser.add_argument("--val-dir", type=str, default=str(DEFAULT_VAL_CALIBRATION_DIR), help="Normal calibration directory")
    parser.add_argument("--output-config", type=str, default=str(DEFAULT_THRESHOLD_CONFIG), help="Output config path")
    parser.add_argument("--k", type=float, default=DEFAULT_THRESHOLD_K, help="Standard deviation multiplier k")
    args = parser.parse_args()

    calibrate(
        checkpoint_path=args.checkpoint,
        val_normal_dir=args.val_dir,
        output_config=args.output_config,
        selected_k=args.k,
    )

