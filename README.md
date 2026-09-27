# SolSight — Solder Defect Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Fully%20Convolutional%20CAE-green.svg)]()
[![Scoring](https://img.shields.io/badge/Metric-Top--5%25%20DSSIM-orange.svg)]()

Unsupervised, resolution-agnostic convolutional autoencoder (CAE) for automatic optical inspection (AOI) of solder-joint defects on printed circuit board assemblies (PCBAs).

- **Unsupervised:** Trained exclusively on defect-free solder joints; flags defects as structural reconstruction failures.
- **Resolution-Agnostic:** Fully convolutional architecture (no dense layers) running across $16\text{px}$ to $160\text{px}$ natively.
- **Localized DSSIM Scoring:** Top-5% spatial structural dissimilarity prevents small pinholes and voids from being diluted by whole-patch averaging.
- **Dataset Disclosure:** All training, validation, and benchmark datasets in this release are 100% procedurally synthesized from first-principles 3D optics. Commercial deployment on physical SMT lines is strictly gated on collecting site-specific PCBA calibration imagery.

---

## Evaluation Guide

Pretrained weights (`outputs/checkpoints/best_cae.pt`) and calibrated thresholds (`outputs/threshold_config.json`) are bundled in the repository.

### Setup

```bash
git clone https://github.com/stanmaluki/SolderDefectDetectionSystem.git
cd SolderDefectDetectionSystem
pip install -r requirements.txt
```

---

### Step 1: Verify Architecture & Resolution Parity

Verify that the model dynamically adapts to arbitrary patch dimensions and scales the SSIM window for small patches:

```bash
python scripts/shape_check.py
```

Expected output:
```
[PASS] Resolution  15x15  -> Latent: torch.Size([2, 64, 3, 3]), Output: torch.Size([2, 3, 15, 15])
[PASS] Resolution  16x16  -> Latent: torch.Size([2, 64, 4, 4]), Output: torch.Size([2, 3, 16, 16])
[PASS] Resolution  32x32  -> Latent: torch.Size([2, 64, 8, 8]), Output: torch.Size([2, 3, 32, 32])
[PASS] Resolution  64x64  -> Latent: torch.Size([2, 64, 16, 16]), Output: torch.Size([2, 3, 64, 64])
[PASS] Resolution 128x128 -> Latent: torch.Size([2, 64, 32, 32]), Output: torch.Size([2, 3, 128, 128])
ALL SHAPE AND CONVOLUTION CHECKS PASSED SUCCESSFULLY!
```

---

### Step 2: Run Benchmark Evaluation

Evaluate the model across all native trained tiers ($16\text{px}, 64\text{px}, 128\text{px}$) and the untrained generalization tier ($32\text{px}$):

```bash
python scripts/evaluate.py
```

Expected output:
```
-----------------------------------------------------------------------------------------------
Tier           | Type         | FPR      | Overall Rec  | Voids    | Bridge   | Cold J   | Amount   | AUROC   
-----------------------------------------------------------------------------------------------
16px           | TRAINED      | 10.0%    | 86.0%        | 64.0%    | 84.0%    | 96.0%    | 100.0%   | 0.940   
32px (UNTRAIN) | GENERALIZE   | 15.0%    | 53.5%        | 44.0%    | 44.0%    | 100.0%   | 26.0%    | 0.862   
64px           | TRAINED      | 0.0%     | 26.0%        | 2.0%     | 2.0%     | 100.0%   | 0.0%     | 0.963   
128px          | TRAINED      | 0.0%     | 25.0%        | 0.0%     | 0.0%     | 100.0%   | 0.0%     | 0.850   
-----------------------------------------------------------------------------------------------
Saved ROC curves plot to: outputs/roc_curves.png
Saved evaluation metrics to: outputs/evaluation_metrics.json
```

---

### Step 3: Inspect a Solder Joint Patch

Run anomaly inference on any patch to generate a side-by-side reconstruction and DSSIM heatmap overlay:

```bash
# Automatic sample run
python src/inference/predict.py

# Or inspect a specific defect patch
python src/inference/predict.py --input outputs/demo_visuals/02_defect_void_128px.png
```

Expected output:
```
============================================================
SOLSIGHT SOLDER DEFECT INSPECTION RESULT
============================================================
Patch Size     : 128 x 128 px
Anomaly Score  : 0.2612 (Top-5% DSSIM)
Threshold (T)  : 0.1300
Verdict        : DEFECT DETECTED [REJECT]
SSIM Mean      : 0.9114
============================================================
Saved inspection panel to: outputs/prediction_result.png
```

Inspection output is saved to `outputs/prediction_result.png`.

---

### Step 4: Run Failure Boundary & Stress Suite

Run the stress test suite to measure where the model empirically breaks:

```bash
python scripts/stress_test.py
```

Outputs are saved to `outputs/stress_test/stress_diagnostics.png` and `outputs/stress_test/stress_test_summary.json`.

---

### Step 5: Interactive Demo (Optional)

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Benchmark Results

Evaluated on held-out test sets across trained and untrained tiers:

| Tier | Type | AUROC | Cold Joint Recall | Bridging Recall* | Normal Joint FPR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$16 \times 16$** | Trained Native Tier | **0.940** | 96.0% | 84.0% | 10.0% |
| **$32 \times 32$** | **Untrained Zero-Shot** | **0.862** | 100.0% | 44.0% | 15.0% |
| **$64 \times 64$** | Trained Native Tier | **0.963** | 100.0% | 90.0%* | 0.0% |
| **$128 \times 128$**| Trained Native Tier | **0.850** | 100.0% | 88.0%* | 0.0% |

> **Threshold Trade-off Note on Higher Resolutions:**  
> When evaluating under a single global threshold ($T=0.1300$), recall on subtle localized defects drops at 64px and 128px ($25\%-26\%$) because $T$ is elevated by the higher normal variance of 16px joints ($\mu=0.083, \sigma=0.034$). Macro-defects (cold joints) maintain 100% recall. Under operational **resolution-aware thresholds** ($T_{64}=0.0773, T_{128}=0.0769$ in `outputs/threshold_config.json`), recall returns to **88%–90%+**. Full empirical analysis: [`docs/empirical_breaking_points.md`](docs/empirical_breaking_points.md).

---

## Empirical Failure Boundaries

| Dimension | Nominal Range | Failure Boundary | Root Cause |
| :--- | :--- | :--- | :--- |
| **Spatial Resolution** | $16\text{px}$ to $160\text{px}$ | $\le 14\text{px}$ (AUROC = 0.346) | SSIM window ($11\times11$) exceeds patch size; odd stride-2 downsampling ($14\to 7\to 3$). |
| **Defect Footprint** | $\ge 1.5\%$ patch area | $< 0.5\%$ (under global $T$) | Single multi-tier threshold ($T=0.130$) is elevated by 16px noise; resolved by per-tier thresholds ($T_{64}=0.077$). |
| **Sensor Noise** | $\sigma \le 0.01$ ($\text{SNR} \ge 40\text{ dB}$) | $\sigma \ge 0.02$ ($\text{SNR} \le 34\text{ dB}$) | High-frequency sensor noise triggers SSIM contrast penalty against smoothed CAE reconstruction. |

*Detailed write-up: [`docs/empirical_breaking_points.md`](docs/empirical_breaking_points.md).*

---

## Architecture

```
Patch (H x W x 3) ──► Encoder (3 Conv layers, s=2, s=2, s=1) ──► Latent (H/4 x W/4 x 64)
                   ──► Decoder (3 ConvTranspose, s=1, s=2, s=2) ──► Reconstruction (H x W x 3)
                   ──► Spatial DSSIM Map ──► Top-5% Spatial Dissimilarity Score
```

---

## Retraining From Scratch (Optional)

```bash
# 1. Generate synthetic dataset (5,700 patches)
python src/data/synthetic_generator.py

# 2. Train CAE multi-scale across discrete tiers {16, 64, 128}px
python src/train.py --epochs 30 --batch-size 32 --lr 1e-3

# 3. Calibrate thresholds
python scripts/calibrate_threshold.py --k 2.5
```

---

## Repository Structure

```
SolderDefectDetectionSystem/
├── README.md                            # Project documentation & evaluation guide
├── requirements.txt                     # Dependencies
├── src/
│   ├── model/cae.py                     # Fully convolutional autoencoder
│   ├── loss/ssim_loss.py                # SSIM loss and spatial map
│   ├── data/
│   │   ├── synthetic_generator.py       # Procedural solder patch generator
│   │   ├── augmentation.py              # AOI-safe augmentations
│   │   └── dataset.py                   # Multi-scale DataLoader
│   ├── inference/predict.py             # Anomaly scoring and heatmap CLI
│   └── train.py                         # Multi-scale training pipeline
├── scripts/
│   ├── shape_check.py                   # Shape and dynamic window check
│   ├── calibrate_threshold.py           # Multi-resolution threshold calibration
│   ├── evaluate.py                      # Multi-tier evaluation and ROC plot
│   └── stress_test.py                   # Failure boundary stress suite
├── notebooks/demo.ipynb                 # Interactive demo notebook
├── docs/                                # Technical specifications and reports
└── outputs/                             # Checkpoints, metrics, and visual artifacts
```

---

## License

MIT License. Developed for the SolSight PCBA Automatic Optical Inspection Hackathon.
