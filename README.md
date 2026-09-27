#  Solder Defect Detection System

Unsupervised, resolution-agnostic convolutional autoencoder (CAE) engine for automatic optical inspection (AOI) of solder-joint defects on printed circuit board assemblies (PCBAs).

## Key Features

- **Resolution-Agnostic Architecture:** Fully convolutional encoder-bottleneck-decoder design with no dense layers or hardcoded dimensions, capable of inspecting solder joints from 15px up to 128px+.
- **SSIM-Based Anomaly Scoring:** Uses custom structural dissimilarity (DSSIM) maps rather than blurry MSE pixel differences, capturing voids, bridging, cold joints, and shape irregularities directly.
- **Top-5% DSSIM Metric:** Concentrates defect signal on localized structural disruptions rather than diluting subtle defects in patch-wide averages.
- **Unsupervised Training:** Trained exclusively on defect-free ("golden reference") joints; detects anomalies as structural reconstruction failures.
- **Discrete Native Multi-Scale Training:** Trained across discrete native tiers `{16, 64, 128}` px and tested for zero-shot generalization on untrained scales (e.g., 32px).

## Project Structure

```
SolderDefectDetectionSystem/
├── solsight-hackathon-build-spec.md  # Event specification & technical requirements
├── requirements.txt                 # Project dependencies
├── src/
│   ├── model/
│   │   └── cae.py                   # Fully convolutional autoencoder
│   ├── loss/
│   │   └── ssim_loss.py             # Custom SSIM via conv2d (map + scalar)
│   ├── data/
│   │   ├── synthetic_generator.py   # Procedural normal & defect generator
│   │   ├── augmentation.py          # Data augmentation
│   │   └── dataset.py               # Discrete multi-scale DataLoader
│   ├── inference/
│   │   └── predict.py               # Heatmap generation & top-k scoring
│   └── train.py                     # Multi-scale training pipeline
├── scripts/
│   ├── shape_check.py               # Resolution & dynamic SSIM window validator
│   ├── calibrate_threshold.py       # Threshold calibration script
│   └── evaluate.py                  # FPR, recall, and cross-resolution metrics
├── notebooks/
│   └── demo.ipynb                   # Interactive demo notebook
├── data/                            # Synthetic & real validation datasets
├── outputs/                         # Checkpoints & visual assets
└── docs/                            # Technical write-up & demo script
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the shape-check validator:
   ```bash
   python scripts/shape_check.py
   ```
