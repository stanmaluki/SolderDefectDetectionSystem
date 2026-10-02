# SolSight — Unsupervised Solder Defect Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Fully%20Convolutional%20CAE-green.svg)]()
[![Scoring](https://img.shields.io/badge/Metric-Top--5%25%20DSSIM-orange.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-AWS%20Graviton2%20%2B%20NVIDIA%20T4G-76B900.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

**SolSight** is an unsupervised, resolution-agnostic computer vision system engineered for automated optical inspection (AOI) of surface-mount technology (SMT) printed circuit board assemblies (PCBAs). 

Traditional SMT optical inspection lines suffer from two structural bottlenecks: lengthy manual programming of golden templates for every new component package, and high false-alarm rates (false rejects) that create severe manual review burdens. SolSight resolves this by modeling normal solder joint geometry in an unsupervised representation space:

- **Unsupervised Anomaly Detection:** Trained strictly on defect-free reference joints. Any structural defect—bridging, voids, cold joints, misalignments, solder burrs, or insufficient wetting—manifests as a reconstruction failure.
- **Resolution-Agnostic Fully Convolutional CAE:** Built without dense/flatten layers or rigid input dimensions. A single trained model checkpoint processes arbitrary patch resolutions from $16\text{px}$ to $160\text{px}$ natively.
- **Hybrid Domain Adaptation Prior:** Seamlessly blends synthetic multi-scale procedural patches with physical PCBA golden reference joints (`--include-real-normals`), closing surface finish domain gaps while preserving the strictly unsupervised guarantee.
- **Localized Top-5% DSSIM Scoring:** Evaluates high-frequency structural dissimilarity in the top 5% most anomalous spatial regions, ensuring localized defects (e.g. pinhole voids covering $<2\%$ of patch area) are never diluted by whole-patch averaging.
- **Hardened Split-Safe Evaluation:** Enforces strict 3-way disjoint data isolation (`train`, `val_calibration`, `test_heldout`) with independent threshold calibration and 1,500-iteration stratified bootstrap confidence intervals.
- **Sub-Millisecond Inference:** Optimized for production line speed, delivering **~1.03 ms per patch (~970 fps)** on NVIDIA T4G edge/cloud GPUs.

---

## Architecture & Scoring Pipeline

```
  Input Patch x           Encoder (Conv2d, s=1, s=2, s=1, s=2, s=1)
 [H x W x 3] ───────► ┌──────────────────────────────────────────────┐
                      │ 3 -> 16 -> 16 -> 32 -> 32 -> 64 channels     │
                      └──────────────────────┬───────────────────────┘
                                             │ Latent Bottleneck [H/4 x W/4 x 64]
                                             ▼
                      Decoder (Nearest Upsample 2x + Conv2d + Sigmoid)
                      ┌──────────────────────────────────────────────┐
                      │ 64 -> 32 -> 16 -> 3 channels                 │
                      └──────────────────────┬───────────────────────┘
                                             │ Reconstruction x_hat [H x W x 3]
                                             ▼
                               Dynamic Structural Dissimilarity Map
                           DSSIM(x, x_hat) = (1 - SSIM(x, x_hat)) / 2
                                             │
                                             ▼
                                Top-5% Spatial Percentile Pool
                     Score = Mean(Top 5% Highest DSSIM Pixels)
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             Score <= T_operational                      Score > T_operational
              [ PASS: NORMAL ]                            [ REJECT: DEFECT ]
```

### Key Architectural Safeguards
1. **No Dense Layers:** Zero dimension hardcoding; center-crop decoder logic cleanly absorbs odd-dimension rounding (e.g., 15px, 17px, 95px).
2. **Nearest-Neighbor Upsampling:** Eliminates checkerboard deconvolution artifacts common to transposed convolutions.
3. **Dynamic SSIM Window Sizing:** Dynamically clamps convolution filter size $K = \min(11, 2 \lfloor \min(H, W)/2 \rfloor - 1)$ for sub-11px regions, with strict ValueError guards for patches below $3\text{px}$.

---

## Validated Benchmark Results

### 1. Multi-Scale Held-Out Synthetic Benchmark
Evaluated on 1,200 independent held-out test patches across trained native and untrained zero-shot tiers. Metrics report point estimates and **stratified bootstrap 95% confidence intervals** (1,500 resamples):

| Resolution Tier | Model Role | AUROC [95% CI] | Global Threshold ($T=0.1344$)<br>FPR / Defect Recall / Review Burden* | Operational Tier Threshold ($T_{\text{tier}}$)<br>FPR / Defect Recall / Review Burden* |
| :--- | :---: | :---: | :---: | :---: |
| **$16 \times 16$** | Trained Native Tier | **0.935** [0.898–0.965] | 9.0% / 82.0% / 900 per 10k | **5.0% / 48.0% / 500 per 10k** ($T_{16}=0.1804$) |
| **$32 \times 32$** | **Untrained Zero-Shot** | **0.876** [0.833–0.917] | 8.0% / 42.5% / 800 per 10k | **4.0% / 34.0% / 400 per 10k** ($T_{32}=0.1455$) |
| **$64 \times 64$** | Trained Native Tier | **0.903** [0.866–0.934] | 0.0% / 25.5% / 0 per 10k | **3.0% / 68.0% / 300 per 10k** ($T_{64}=0.0872$) |
| **$128 \times 128$** | Trained Native Tier | **0.834** [0.785–0.879] | 1.0% / 25.0% / 100 per 10k | **5.0% / 63.5% / 500 per 10k** ($T_{128}=0.0729$) |

*\*Review Burden: Expected false alarms per 10,000 inspected solder joints requiring manual operator verification.*

#### Per-Defect Class Breakdown (Under Operational Per-Tier Thresholds)
| Resolution Tier | Operational $T$ | Voids Recall | Bridging Recall | Cold Joint Recall | Solder Amount Recall | Precision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **16px** | 0.1804 | 26.0% | 22.0% | 50.0% | 94.0% | **95.0%** |
| **32px** | 0.1455 | 26.0% | 6.0% | 100.0% | 4.0% | **94.4%** |
| **64px** | 0.0872 | **74.0%** | **94.0%** | **100.0%** | 4.0% | **97.8%** |
| **128px** | 0.0729 | 60.0% | 94.0% | 100.0% | 0.0% | **96.2%** |

---

### 2. Physical PCBA Industrial Transfer Benchmark (SolDef_AI Dataset)
Evaluated on 250 physical solder joint patches from genuine manufacturing assemblies (`data/real/` with full SHA-256 provenance in `data/real/provenance.json`: 50 normal golden references and 200 real defects across 4 failure classes):

| Metric / Failure Mode | Pure-Synthetic Baseline | **Hybrid Prior (Synthetic + Real Normals)** | Operational Impact / Delta |
| :--- | :---: | :---: | :--- |
| **AUROC (Overall Physical PCBA)** | 0.8521 | **0.8546** | Robust separation on physical hardware |
| **Normal Baseline Reconstruction Loss** | $0.1997 \pm 0.0238$ | **$0.1900 \pm 0.0233$** | $-0.0097$ lower reconstruction variance |
| **Recall at $T_{\text{real}}$ ($0.0\%$ False Rejects)** | **33.0%** | **42.5%** | **$+9.5\%$ absolute defect escape reduction** |
| — *Misaligned Components* | 62.0% | **74.0%** | $+12.0\%$ detection improvement |
| — *Insufficient Solder* | 26.0% | **36.0%** | $+10.0\%$ detection improvement |
| — *Excessive Solder* | 26.0% | **34.0%** | $+8.0\%$ detection improvement |
| — *Solder Spikes / Burrs* | 18.0% | **26.0%** | $+8.0\%$ detection improvement |
| **Recall at Line Sensitivity ($T_{\text{synthetic}}=0.1344$)** | 98.0% | **99.5%** | **0.5% defect escape rate** (199/200 defects detected) |

#### Operational Dual-Threshold Deployment Model
- **Zero-False-Alarm Line Gate ($T_{\text{real}} = 0.2367$)**: Zero false rejections ($\text{FPR} = 0.0\%$) on golden production joints while intercepting 74% of component misalignments and over a third of volume defects without spurious operator callouts.
- **High-Sensitivity Screen ($T_{\text{synthetic}} = 0.1344$)**: Catches **99.5% of all physical defects** (100% of excessive solder, 100% of insufficient solder, and 100% of misalignments).

---

### 3. Production Inference Latency (T4G GPU vs ARM64 CPU)
Measured across 100 single-patch inference iterations on an AWS Graviton2 (`g5g.2xlarge`) instance with an NVIDIA T4G 16GB GPU:

| Resolution Tier | NVIDIA T4G GPU Latency | T4G GPU Throughput | Graviton2 CPU Latency | Graviton2 CPU Throughput | GPU Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **16×16** | **1.07 ms** | **933.8 fps** | 2.27 ms | 440.7 fps | **2.1×** |
| **32×32** | **1.03 ms** | **974.2 fps** | 3.85 ms | 259.8 fps | **3.7×** |
| **64×64** | **1.03 ms** | **969.2 fps** | 5.33 ms | 187.8 fps | **5.2×** |
| **128×128** | **1.03 ms** | **969.5 fps** | 24.62 ms | 40.6 fps | **23.9×** |

*Inference latency remains deterministic at **~1.03 ms/patch (~970 fps)** across all resolutions on GPU.*

---

## Quickstart & User Guide

### 1. Installation

```bash
git clone https://github.com/stanmaluki/SolderDefectDetectionSystem.git
cd SolderDefectDetectionSystem

# Install Python dependencies
pip install -r requirements.txt

# Run full unit test suite (23 tests covering config, model, SSIM, evaluation, bootstrap)
python -m unittest discover -v tests
# or with pytest:
pytest -v tests
```

---

### 2. Verify Architecture & Mathematical Shape Parity
Ensure the FCN CAE handles arbitrary scales and odd sizes seamlessly without shape distortion:

```bash
python scripts/shape_check.py
```

Expected output:
```
================================================================================
SolSight — Solder Defect Detection System: Architectural Shape & SSIM Check
================================================================================
Resolution   | Input Shape      | Output Shape     | SSIM Map Shape   | Win Size  | Win Cov %  | Status
-----------------------------------------------------------------------------------------------
6x6         | (1, 3, 6, 6)     | (1, 3, 6, 6)     | (1, 1, 6, 6)     | 5         |  83.3%     | PASS
8x8         | (1, 3, 8, 8)     | (1, 3, 8, 8)     | (1, 1, 8, 8)     | 7         |  87.5%     | PASS
16x16        | (1, 3, 16, 16)   | (1, 3, 16, 16)   | (1, 1, 16, 16)   | 11        |  68.8%     | PASS
32x32        | (1, 3, 32, 32)   | (1, 3, 32, 32)   | (1, 1, 32, 32)   | 11        |  34.4%     | PASS
64x64        | (1, 3, 64, 64)   | (1, 3, 64, 64)   | (1, 1, 64, 64)   | 11        |  17.2%     | PASS
128x128       | (1, 3, 128, 128) | (1, 3, 128, 128) | (1, 1, 128, 128) | 11        |   8.6%     | PASS
-----------------------------------------------------------------------------------------------
ALL RESOLUTIONS AND SHAPE CHECKS PASSED SUCCESSFULLY.
```

---

### 3. Run Independent Threshold Calibration
Fit global ($T$) and per-tier thresholds on the dedicated calibration split (`data/synthetic/val_calibration`):

```bash
python scripts/calibrate_threshold.py --k 2.5
```

Outputs are saved to `outputs/threshold_config.json`.

---

### 4. Evaluate Held-Out Benchmark
Run the split-safe benchmark evaluation across 1,200 test patches with bootstrap confidence intervals:

```bash
python scripts/evaluate.py
```

Outputs are saved to `outputs/roc_curves.png`, `outputs/pr_curves.png`, and `outputs/evaluation_metrics.json`.

---

### 5. Inspect Single or Batch Solder Patches
Run inference on any patch to generate side-by-side reconstruction and DSSIM heatmap overlays:

```bash
# Automatic default sample
python src/inference/predict.py

# Inspect a specific synthetic patch with calibrated global threshold
python src/inference/predict.py --input outputs/demo_visuals/02_defect_void_128px.png --threshold 0.1344

# Inspect a physical PCBA joint with calibrated zero-false-alarm physical threshold
python src/inference/predict.py --input data/real/defective/exc_solder/real_exc_solder_0000.png --threshold 0.2367 --output outputs/prediction_result_real.png
```

Inspection output is saved to `outputs/prediction_result.png` (or custom `--output` path).

---

### 6. Physical PCBA Evaluation (SolDef_AI)
Run evaluation against real-world physical joints from the SolDef_AI dataset:

```bash
python scripts/evaluate_real.py
```

Outputs are saved to `outputs/real_world_evaluation.png` and `outputs/real_evaluation_metrics.json`.

---

### 7. Latency Benchmarking (GPU vs CPU)
Benchmark inference performance across resolutions:

```bash
python scripts/benchmark_latency.py
```

---

### 8. Failure Boundary Stress Suite
Run the stress diagnostic probing resolution limits, pinhole sensitivity, and sensor noise:

```bash
python scripts/stress_test.py
```

Outputs are saved to `outputs/stress_test/stress_diagnostics.png` and `outputs/stress_test/stress_test_summary.json`.

---

## Empirical Failure Boundaries

Detailed analysis in [`docs/empirical_breaking_points.md`](docs/empirical_breaking_points.md):

| Dimension | Nominal Operating Range | Failure Boundary | Root Cause & Mitigation |
| :--- | :--- | :--- | :--- |
| **Spatial Resolution** | $16\text{px}$ to $160\text{px}$ | $\le 14\text{px}$ (AUROC drops to 0.325) | Spatial window ($11\times11$) exceeds patch bounds; stride-2 convolutions reduce spatial feature maps to $1\times1$. Clamped via dynamic window sizing down to 16px minimum. |
| **Defect Footprint** | $\ge 1.5\%$ patch area | $< 0.5\%$ (under global $T$) | Single global threshold ($T=0.1344$) suffers from cross-resolution variance disparity. Resolved by **per-tier threshold calibration** ($T_{16}=0.1804, T_{64}=0.0872, T_{128}=0.0729$). |
| **Sensor Noise / SNR**| $\sigma \le 0.01$ ($\text{SNR} \ge 40\text{ dB}$) | $\sigma \ge 0.02$ ($\text{SNR} \le 34\text{ dB}$) | High-frequency sensor noise penalizes SSIM contrast term against smooth autoencoder reconstruction. Requires controlled AOI line illumination. |

---

## Retraining Pipeline (From Scratch)

To regenerate datasets and train a new model from scratch:

```bash
# 1. Generate 3-way disjoint synthetic dataset (train, calibration, held-out test)
python src/data/synthetic_generator.py --seed 42

# 2a. Train CAE multi-scale across discrete native tiers {16, 64, 128}px (Pure Synthetic)
python src/train.py --epochs 30 --batch-size 64 --lr 1e-3 --patience 7

# 2b. Train CAE multi-scale incorporating Kaggle real PCBA normal reference joints (Hybrid)
python src/train.py --epochs 30 --batch-size 64 --lr 1e-3 --patience 7 --include-real-normals

# 3. Calibrate operating thresholds on held-out calibration split
python scripts/calibrate_threshold.py --k 2.5
```

---

## Repository Structure

```
SolderDefectDetectionSystem/
├── README.md                            # Industrial documentation and system overview
├── requirements.txt                     # Pinned project dependencies
├── pyproject.toml                       # Build and test configurations
├── src/
│   ├── __init__.py                      # Package root
│   ├── config.py                        # Centralized paths, tiers, and threshold defaults
│   ├── utils.py                         # Shared utility functions and model loaders
│   ├── train.py                         # Multi-scale CAE training engine
│   ├── model/
│   │   ├── __init__.py                  # Model exports
│   │   └── cae.py                       # Fully convolutional autoencoder (SolderCAE)
│   ├── loss/
│   │   ├── __init__.py                  # Loss exports
│   │   └── ssim_loss.py                 # Dynamic SSIM loss and spatial dissimilarity maps
│   ├── data/
│   │   ├── __init__.py                  # Data module exports
│   │   ├── augmentation.py              # AOI-safe geometric and photometric augmentations
│   │   ├── dataset.py                   # Multi-scale DataLoader and batch sampler
│   │   └── synthetic_generator.py       # Procedural 3D optics generator with 3-way split logic
│   ├── evaluation/
│   │   ├── __init__.py                  # Evaluation exports
│   │   ├── bootstrap.py                 # Stratified & percentile bootstrap confidence intervals
│   │   └── confusion.py                 # Confusion matrices, PR curves, operational review burden
│   └── inference/
│       ├── __init__.py                  # Inference exports
│       └── predict.py                   # Patch inspection and DSSIM heatmap overlay CLI
├── scripts/
│   ├── benchmark_latency.py             # Inference throughput benchmark (GPU vs CPU)
│   ├── calibrate_threshold.py           # Independent threshold calibration on val_calibration
│   ├── demo.py                          # Demo visual generation across native and untrained tiers
│   ├── evaluate.py                      # Split-safe synthetic evaluation with bootstrap CIs
│   ├── evaluate_real.py                 # Physical PCBA evaluation on SolDef_AI benchmark
│   ├── fetch_real_data.py               # Real dataset fetcher & provenance tracker
│   ├── setup_dirs.py                    # Output directory scaffolding
│   ├── shape_check.py                   # Architectural shape & dynamic SSIM window validation
│   └── stress_test.py                   # Empirical failure boundary stress suite
├── tests/
│   ├── __init__.py                      # Test suite package
│   ├── test_config.py                   # Path and configuration unit tests
│   ├── test_crop.py                     # Centroid-driven cropping & border clamping tests
│   ├── test_dataset.py                  # Dataset loader and augmentation pipeline tests
│   ├── test_evaluation.py               # Bootstrap CI and operational review burden tests
│   ├── test_model.py                    # CAE shape invariance and config restoration tests
│   └── test_ssim.py                     # Dynamic window clamping and sub-3px guard tests
├── docs/                                # Technical specifications and empirical reports
└── outputs/                             # Checkpoints, evaluation figures, and metric summaries
```

---

## SolSight Phase 1 Roadmap

- [x] **Track A — Foundation & Code Hygiene:** Centralized [`src/config.py`](src/config.py), shared model loader [`src/utils.py`](src/utils.py), checkpoint architecture serialization, and automated unit test suite.
- [x] **Track C — Split-Safe Methodology & Uncertainty:** Independent 3-way split protocol with SHA-256 hash manifest verification, operating-point evaluation ($T_{\text{global}}$ vs $T_{\text{tier}}$), 1,500-iteration stratified bootstrap confidence intervals, and operational review burden metrics.
- [ ] **Track B — CAD/Placement-File-Driven Pipeline (In Progress):** Gerber RS-274X copper layer parsing, centroid (pick-and-place) file extraction, multi-fiducial homography registration with RANSAC, and design-driven joint patch extraction.
- [ ] **Track D — Real-World Multi-Source Validation (In Progress):** Expansion of physical PCBA validation across diverse board finishes (ENIG, HASL, OSP), multi-angle illumination, and line-specific calibration.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
