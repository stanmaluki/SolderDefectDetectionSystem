"""Centralized configuration constants and defaults for SolSight."""

from pathlib import Path

# Core directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CHECKPOINT_PATH = DEFAULT_OUTPUT_DIR / "checkpoints" / "best_cae.pt"
DEFAULT_THRESHOLD_CONFIG = DEFAULT_OUTPUT_DIR / "threshold_config.json"
DEFAULT_SYNTHETIC_DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
DEFAULT_REAL_DATA_DIR = PROJECT_ROOT / "data" / "real"

# Model architecture defaults
DEFAULT_IN_CHANNELS = 3
DEFAULT_BASE_CHANNELS = 16

# Resolution tiers
DEFAULT_TRAIN_TIERS = (16, 64, 128)
DEFAULT_ALL_TIERS = (16, 32, 64, 128)
DEFAULT_CROP_SIZE = 64

# Scoring & Thresholding defaults
DEFAULT_TOP_K_PCT = 0.05
DEFAULT_THRESHOLD_K = 2.5
DEFAULT_SYNTHETIC_GLOBAL_THRESHOLD = 0.1300
DEFAULT_REAL_THRESHOLD_K = 2.0
