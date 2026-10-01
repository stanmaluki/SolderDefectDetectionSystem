"""Helper script to create all standard project directories for SolSight."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_ALL_TIERS,
    DEFAULT_CAD_CROPS_DIR,
    DEFAULT_CAD_DIR,
    DEFAULT_CENTROID_DIR,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FIDUCIAL_DIR,
    DEFAULT_GERBER_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REAL_DATA_DIR,
    DEFAULT_SYNTHETIC_DATA_DIR,
    DEFAULT_TEST_DEFECTS_DIR,
    DEFAULT_TEST_NORMAL_DIR,
    DEFAULT_TRAIN_TIERS,
)

DEFECT_TYPES = ("voids", "bridging", "cold_joints", "solder_amount")


def get_project_directories() -> list[Path]:
    """Derive full list of standard project directories from centralized configuration."""
    dirs: list[Path] = []

    # 1. Synthetic training tiers
    for tier in DEFAULT_TRAIN_TIERS:
        dirs.append(DEFAULT_SYNTHETIC_DATA_DIR / "train" / f"{tier}px")

    # 2. Synthetic validation (normal and defect classes)
    for tier in DEFAULT_ALL_TIERS:
        dirs.append(DEFAULT_SYNTHETIC_DATA_DIR / "val_normal" / f"{tier}px")
        for defect in DEFECT_TYPES:
            dirs.append(DEFAULT_SYNTHETIC_DATA_DIR / "val_defects" / defect / f"{tier}px")

    # 3. Synthetic held-out test splits (Track C)
    for tier in DEFAULT_ALL_TIERS:
        dirs.append(DEFAULT_TEST_NORMAL_DIR / f"{tier}px")
        for defect in DEFECT_TYPES:
            dirs.append(DEFAULT_TEST_DEFECTS_DIR / defect / f"{tier}px")

    # 4. CAD pipeline directories (Track B)
    dirs.extend([
        DEFAULT_CAD_DIR,
        DEFAULT_GERBER_DIR,
        DEFAULT_CENTROID_DIR,
        DEFAULT_FIDUCIAL_DIR,
        DEFAULT_CAD_CROPS_DIR,
    ])

    # 5. Real data directories
    dirs.extend([
        DEFAULT_REAL_DATA_DIR / "normal",
        DEFAULT_REAL_DATA_DIR / "defective" / "exc_solder",
        DEFAULT_REAL_DATA_DIR / "defective" / "poor_solder",
        DEFAULT_REAL_DATA_DIR / "defective" / "spike",
        DEFAULT_REAL_DATA_DIR / "defective" / "no_good",
    ])

    # 6. Outputs and Documentation
    dirs.extend([
        DEFAULT_CHECKPOINT_PATH.parent,
        DEFAULT_OUTPUT_DIR / "demo_visuals",
        DEFAULT_OUTPUT_DIR / "stress_test",
        DEFAULT_OUTPUT_DIR / "roi_debug",
        PROJECT_ROOT / "docs",
    ])

    return dirs


def create_all_directories() -> None:
    dirs = get_project_directories()
    for p in dirs:
        p.mkdir(parents=True, exist_ok=True)
        gitkeep = p / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    print(f"Created/verified {len(dirs)} project directories successfully.")


if __name__ == "__main__":
    create_all_directories()

