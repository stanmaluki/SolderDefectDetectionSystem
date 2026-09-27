"""Helper script to create all standard project directories for SolSight."""

import os
from pathlib import Path

DIRS = [
    # Synthetic train tiers
    "data/synthetic/train/16px",
    "data/synthetic/train/64px",
    "data/synthetic/train/128px",
    # Synthetic validation normals (including 32px untrained test tier)
    "data/synthetic/val_normal/16px",
    "data/synthetic/val_normal/32px",
    "data/synthetic/val_normal/64px",
    "data/synthetic/val_normal/128px",
    # Synthetic validation defects
    "data/synthetic/val_defects/voids/16px",
    "data/synthetic/val_defects/voids/32px",
    "data/synthetic/val_defects/voids/64px",
    "data/synthetic/val_defects/voids/128px",
    "data/synthetic/val_defects/bridging/16px",
    "data/synthetic/val_defects/bridging/32px",
    "data/synthetic/val_defects/bridging/64px",
    "data/synthetic/val_defects/bridging/128px",
    "data/synthetic/val_defects/cold_joints/16px",
    "data/synthetic/val_defects/cold_joints/32px",
    "data/synthetic/val_defects/cold_joints/64px",
    "data/synthetic/val_defects/cold_joints/128px",
    "data/synthetic/val_defects/solder_amount/16px",
    "data/synthetic/val_defects/solder_amount/32px",
    "data/synthetic/val_defects/solder_amount/64px",
    "data/synthetic/val_defects/solder_amount/128px",
    # Real validation dataset
    "data/real/normal",
    "data/real/defective",
    # Outputs
    "outputs/checkpoints",
    "outputs/demo_visuals",
    # Docs
    "docs",
]

def create_all_directories(base_dir: str = ".") -> None:
    base = Path(base_dir)
    for rel_path in DIRS:
        p = base / rel_path
        p.mkdir(parents=True, exist_ok=True)
        # Create a .gitkeep so empty directories are preserved
        gitkeep = p / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    print(f"Created {len(DIRS)} project directories successfully.")

if __name__ == "__main__":
    create_all_directories()
