"""Real Data Ingestion and Curation Pipeline for SolSight.

Downloads and extracts genuine physical PCBA solder joint inspection imagery from the
official SolDef_AI benchmark dataset on Kaggle:
Dataset: mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection
Authors: Maurizio Calabrese, Gianmauro Fontana, et al.

Processes polygon annotations from LabelMe JSON files into standardized solder joint crops:
  data/real/normal/                  - Genuine defect-free golden reference solder joints
  data/real/defective/exc_solder/    - Real excessive solder / bridging defects
  data/real/defective/poor_solder/   - Real insufficient / starved solder joints
  data/real/defective/spike/         - Real solder spikes / burrs / peaks
  data/real/defective/no_good/       - Real component misalignments and gross solder flaws

Saves full provenance record to data/real/provenance.json.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def extract_square_crop(img: Image.Image, pts: list, margin: float = 1.15) -> tuple[Image.Image, list]:
    """Extract a square patch centered on the annotated polygon bounding box."""
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    side = max(w, h) * margin

    box = (
        max(0, int(cx - side / 2.0)),
        max(0, int(cy - side / 2.0)),
        min(img.width, int(cx + side / 2.0)),
        min(img.height, int(cy + side / 2.0)),
    )
    crop = img.crop(box)
    return crop, list(box)


def setup_real_validation_data(
    target_dir: str = "data/real",
    samples_per_class: int = 50,
    crop_size: int = 64,
) -> dict:
    """Download SolDef_AI and curate genuine solder joint patches."""
    print("=" * 80)
    print("SolSight Real-World Data Integration (SolDef_AI Benchmark)")
    print("Dataset: mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection")
    print("=" * 80)

    import kagglehub

    # Check common cache locations first to bypass Windows file-lock on archive removal
    cache_candidates = [
        Path.home() / ".cache" / "kagglehub" / "datasets" / "mauriziocalabrese" / "soldef-ai-pcb-dataset-for-defect-detection" / "1" / "SolDef_AI",
        Path.home() / ".cache" / "kagglehub" / "datasets" / "mauriziocalabrese" / "soldef-ai-pcb-dataset-for-defect-detection" / "1",
    ]
    raw_path = None
    for cand in cache_candidates:
        if cand.exists():
            raw_path = str(cand)
            print(f"Found existing cached SolDef_AI at: {raw_path}")
            break

    if raw_path is None:
        try:
            print("Locating / downloading SolDef_AI via kagglehub...")
            raw_path = kagglehub.dataset_download("mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection")
            print(f"Dataset root: {raw_path}")
        except Exception as e:
            print(f"Error fetching dataset: {e}")
            raise RuntimeError(
                "Kaggle download failed. Ensure valid credentials in .env or ~/.kaggle/kaggle.json.\n"
                "Dataset URL: https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection"
            )

    base = Path(raw_path)
    labeled_dir = base / "SolDef_AI" / "Labeled"
    if not labeled_dir.exists():
        labeled_dir = next(base.rglob("Labeled"), None)

    if labeled_dir is None or not labeled_dir.exists():
        raise FileNotFoundError(f"Could not locate 'Labeled' directory inside {raw_path}")

    real_base = Path(target_dir)
    norm_dir = real_base / "normal"
    norm_dir.mkdir(parents=True, exist_ok=True)

    defect_dirs = {
        "exc_solder": real_base / "defective" / "exc_solder",
        "poor_solder": real_base / "defective" / "poor_solder",
        "spike": real_base / "defective" / "spike",
        "no_good": real_base / "defective" / "no_good",
    }
    for d in defect_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    counts = {"normal": 0, "exc_solder": 0, "poor_solder": 0, "spike": 0, "no_good": 0}
    provenance = {
        "dataset_name": "SolDef_AI: PCB dataset for defect detection",
        "authors": "Maurizio Calabrese, Gianmauro Fontana, et al.",
        "kaggle_slug": "mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection",
        "crop_size": crop_size,
        "samples": [],
    }

    print(f"Scanning annotations in: {labeled_dir}")
    json_files = sorted(labeled_dir.glob("*.json"))

    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue

        img_file = labeled_dir / (jf.stem + ".jpg")
        if not img_file.exists():
            continue

        img = None  # Lazy load image only if needed

        for shape in data.get("shapes", []):
            lbl = shape.get("label")
            pts = shape.get("points", [])
            if not pts or len(pts) < 3:
                continue

            # Case A: Normal 2-terminal component ('good')
            if lbl == "good" and counts["normal"] < samples_per_class:
                xs = [pt[0] for pt in pts]
                ys = [pt[1] for pt in pts]
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                if w > h * 1.2:
                    if img is None:
                        img = Image.open(img_file)
                    cy = (min(ys) + max(ys)) / 2.0
                    side = h * 1.15

                    # Left terminal fillet
                    b_l = (
                        max(0, int(min(xs))),
                        max(0, int(cy - side / 2.0)),
                        min(img.width, int(min(xs) + side)),
                        min(img.height, int(cy + side / 2.0)),
                    )
                    crop_l = img.crop(b_l).resize((crop_size, crop_size), Image.Resampling.LANCZOS)
                    fn_l = f"real_normal_{counts['normal']:04d}.png"
                    crop_l.save(norm_dir / fn_l)
                    provenance["samples"].append({
                        "file": fn_l,
                        "class": "normal",
                        "source_image": jf.stem + ".jpg",
                        "bbox": b_l,
                        "original_label": "good_left_terminal",
                    })
                    counts["normal"] += 1

                    # Right terminal fillet
                    if counts["normal"] < samples_per_class:
                        b_r = (
                            max(0, int(max(xs) - side)),
                            max(0, int(cy - side / 2.0)),
                            min(img.width, int(max(xs))),
                            min(img.height, int(cy + side / 2.0)),
                        )
                        crop_r = img.crop(b_r).resize((crop_size, crop_size), Image.Resampling.LANCZOS)
                        fn_r = f"real_normal_{counts['normal']:04d}.png"
                        crop_r.save(norm_dir / fn_r)
                        provenance["samples"].append({
                            "file": fn_r,
                            "class": "normal",
                            "source_image": jf.stem + ".jpg",
                            "bbox": b_r,
                            "original_label": "good_right_terminal",
                        })
                        counts["normal"] += 1

            # Case B: Defect classes
            elif lbl in defect_dirs and counts[lbl] < samples_per_class:
                if img is None:
                    img = Image.open(img_file)
                crop, box = extract_square_crop(img, pts, margin=1.15)
                crop_resized = crop.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
                fn = f"real_{lbl}_{counts[lbl]:04d}.png"
                crop_resized.save(defect_dirs[lbl] / fn)
                provenance["samples"].append({
                    "file": fn,
                    "class": lbl,
                    "source_image": jf.stem + ".jpg",
                    "bbox": box,
                    "original_label": lbl,
                })
                counts[lbl] += 1

    # Save provenance JSON
    prov_path = real_base / "provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)

    print("\n" + "=" * 80)
    print("Real Solder Joint Curation Completed Successfully!")
    print("=" * 80)
    print(f"Normal Reference Joints : {counts['normal']:>4} patches (saved to data/real/normal/)")
    print(f"Excessive Solder Defects : {counts['exc_solder']:>4} patches (saved to data/real/defective/exc_solder/)")
    print(f"Insufficient Solder      : {counts['poor_solder']:>4} patches (saved to data/real/defective/poor_solder/)")
    print(f"Solder Spikes / Burrs    : {counts['spike']:>4} patches (saved to data/real/defective/spike/)")
    print(f"Misaligned / Gross Flaws : {counts['no_good']:>4} patches (saved to data/real/defective/no_good/)")
    print(f"Saved Provenance Record  : {prov_path}")
    print("=" * 80)

    return counts


if __name__ == "__main__":
    setup_real_validation_data()
