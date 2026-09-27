"""Real Data Integration Script for SolSight (Implementation Plan Phase 5.1).

Downloads and organizes real PCBA solder joint inspection imagery from open public datasets
(e.g. SolDef_AI / public AOI solder defect benchmarks).
Organizes into:
  data/real/normal/
  data/real/defective/

Provides attribution metadata in docs and README per responsible AI governance.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shutil
import urllib.request
from PIL import Image


def setup_real_validation_data(target_dir: str = "data/real") -> None:
    real_path = Path(target_dir)
    norm_path = real_path / "normal"
    def_path = real_path / "defective"
    norm_path.mkdir(parents=True, exist_ok=True)
    def_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SolSight Real Data Integration (SolDef_AI / Public PCBA Benchmark)")
    print("=" * 80)

    # Attempt download via kagglehub if available
    downloaded = False
    try:
        import kagglehub
        print("Attempting to fetch SolDef_AI via kagglehub...")
        dataset_path = kagglehub.dataset_download("subinium/soldef-ai")
        print(f"Downloaded SolDef_AI to: {dataset_path}")

        # Scan for images and sort into normal and defective
        p = Path(dataset_path)
        img_files = list(p.rglob("*.jpg")) + list(p.rglob("*.png"))
        print(f"Found {len(img_files)} images in SolDef_AI dataset.")

        normal_count = 0
        defect_count = 0
        for img_p in img_files:
            lower_name = str(img_p).lower()
            if "defect" in lower_name or "bad" in lower_name or "ng" in lower_name:
                if defect_count < 30:
                    shutil.copy(img_p, def_path / f"real_defect_{defect_count:03d}{img_p.suffix}")
                    defect_count += 1
            else:
                if normal_count < 30:
                    shutil.copy(img_p, norm_path / f"real_normal_{normal_count:03d}{img_p.suffix}")
                    normal_count += 1

        print(f"Curated {normal_count} real normal joints and {defect_count} real defective joints.")
        downloaded = (normal_count > 0 and defect_count > 0)
    except Exception as e:
        print(f"Error fetching real PCBA dataset: {e}")
        downloaded = False

    if not downloaded:
        # Honest failure: Never synthesize fake "real" data
        raise RuntimeError(
            "Kaggle download unavailable or unauthenticated.\n"
            "SolSight validates natively on procedural synthetic imagery. Real-world validation against "
            "external benchmarks (e.g. SolDef_AI) requires authenticated Kaggle credentials.\n"
            "To download real data:\n"
            "  1. Place your Kaggle API key at ~/.kaggle/kaggle.json\n"
            "  2. Re-run: python scripts/fetch_real_data.py\n"
            "Procedural synthetic fallback has been disabled to guarantee data provenance integrity."
        )

    print("=" * 80)
    print("Real-data curation complete.")
    print("Attribution: SolDef_AI Dataset (Public Domain / Research License, Kaggle).")
    print("=" * 80)


if __name__ == "__main__":
    setup_real_validation_data()
