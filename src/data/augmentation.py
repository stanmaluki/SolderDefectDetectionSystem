"""Data Augmentation Pipeline for SolSight (Implementation Plan Phase 2.3).

Applies:
- Random horizontal & vertical flip.
- Random 90-degree rotations and subtle continuous rotation.
- Brightness and contrast jitter.
- Normalization to [0, 1] float tensor.

Constraint:
- NO scale jitter / resizing! Resolution is determined strictly by the native tier
  to eliminate interpolation artifacts.
"""

import torch
import torchvision.transforms as T
from PIL import Image



class SolderAugmentation:
    """Augmentation pipeline for training defect-free solder patches."""

    def __init__(self, is_train: bool = True) -> None:
        self.is_train = is_train

        if is_train:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.ToTensor(),  # Converts PIL Image [0, 255] to torch.FloatTensor [0.0, 1.0]
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Apply transform to a PIL Image, returning [0, 1] tensor."""
        return self.transform(img)
