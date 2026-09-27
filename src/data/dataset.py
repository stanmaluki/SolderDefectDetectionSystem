"""Dataset and discrete multi-scale DataLoader for SolSight (Phase 2.4).

Supports:
1. DiscreteMultiScaleLoader: Each training batch randomly draws from one native tier
   in {16, 64, 128}. All samples in a batch share identical spatial dimensions.
   No resizing or interpolation is applied.
2. FixedTierLoader: Loads patches from a single specific native tier for validation
   and testing (e.g., 16, 32, 64, 128).
"""

from pathlib import Path
import random
from typing import Dict, Iterator, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from src.data.augmentation import SolderAugmentation


class SolderPatchDataset(Dataset):
    """Dataset for solder joint patches stored in a single folder."""

    def __init__(
        self,
        folder_path: Union[str, Path],
        transform: Optional[SolderAugmentation] = None,
        label: int = 0,
    ) -> None:
        self.folder_path = Path(folder_path)
        self.transform = transform if transform is not None else SolderAugmentation(is_train=False)
        self.label = label

        # Collect image files (.png, .jpg, .jpeg)
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.image_paths = sorted([
            p for p in self.folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        p = self.image_paths[idx]
        with Image.open(p) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor, self.label


class DiscreteMultiScaleTrainLoader:
    """Multi-scale batch loader iterating over discrete native tiers {16, 64, 128}.

    Each iteration selects one tier randomly and returns a batch of shape (B, 3, S, S).
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        tiers: Tuple[int, ...] = (16, 64, 128),
        batch_size: int = 64,
        steps_per_epoch: Optional[int] = None,
        transform: Optional[SolderAugmentation] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.tiers = tiers
        self.batch_size = batch_size
        self.transform = transform if transform is not None else SolderAugmentation(is_train=True)

        self.tier_datasets: Dict[int, SolderPatchDataset] = {}
        for tier in tiers:
            tier_dir = self.base_dir / f"{tier}px"
            if not tier_dir.exists():
                raise FileNotFoundError(f"Tier directory not found: {tier_dir}")
            ds = SolderPatchDataset(tier_dir, transform=self.transform)
            if len(ds) == 0:
                raise ValueError(f"Tier directory is empty: {tier_dir}")
            self.tier_datasets[tier] = ds

        # Steps per epoch: sum of lengths divided by batch size, or specified
        total_samples = sum(len(ds) for ds in self.tier_datasets.values())
        self.steps_per_epoch = steps_per_epoch or max(1, total_samples // batch_size)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        for _ in range(self.steps_per_epoch):
            # Select random native tier for this batch
            tier = random.choice(self.tiers)
            ds = self.tier_datasets[tier]

            # Sample random indices with replacement
            indices = [random.randint(0, len(ds) - 1) for _ in range(self.batch_size)]
            batch_tensors = [ds[i][0] for i in indices]
            batch_tensor = torch.stack(batch_tensors, dim=0)

            yield batch_tensor, tier


def get_fixed_tier_loader(
    folder_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = False,
    is_train: bool = False,
    label: int = 0,
) -> DataLoader:
    """Return standard DataLoader for a single fixed tier."""
    transform = SolderAugmentation(is_train=is_train)
    dataset = SolderPatchDataset(folder_path, transform=transform, label=label)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Cross-platform compatibility on Windows
        drop_last=False,
    )
    return loader
