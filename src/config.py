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
DEFAULT_REAL_THRESHOLD_K = 2.0
# Deprecated fallback if threshold_config.json is missing (emit warning when used)
FALLBACK_THRESHOLD = 0.1300

# CAD pipeline directories (Track B)
DEFAULT_CAD_DIR = PROJECT_ROOT / "data" / "cad"
DEFAULT_GERBER_DIR = DEFAULT_CAD_DIR / "gerber"
DEFAULT_CENTROID_DIR = DEFAULT_CAD_DIR / "centroid"
DEFAULT_FIDUCIAL_DIR = DEFAULT_CAD_DIR / "fiducials"
DEFAULT_CAD_CROPS_DIR = DEFAULT_OUTPUT_DIR / "cad_crops"

# Registration
DEFAULT_FIDUCIAL_MIN_COUNT = 3  # Minimum fiducials for homography

# Held-out test & calibration splits (3-way independent partition)
DEFAULT_VAL_CALIBRATION_DIR = DEFAULT_SYNTHETIC_DATA_DIR / "val_calibration"
DEFAULT_TEST_NORMAL_DIR = DEFAULT_SYNTHETIC_DATA_DIR / "test_normal"
DEFAULT_TEST_DEFECTS_DIR = DEFAULT_SYNTHETIC_DATA_DIR / "test_defects"
DEFAULT_SPLIT_MANIFEST_PATH = DEFAULT_SYNTHETIC_DATA_DIR / "split_manifest.json"

# Multi-source real data manifest (Track D)
DEFAULT_MANIFEST_PATH = DEFAULT_REAL_DATA_DIR / "manifest.json"


