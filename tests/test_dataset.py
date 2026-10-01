"""Unit tests for SolderAugmentation and dataset utilities."""

import unittest
from PIL import Image
import torch
from src.data.augmentation import SolderAugmentation


class TestDatasetAndAugmentation(unittest.TestCase):
    """Test data transforms and shape/range preservation."""

    def test_augmentation_train_mode(self):
        aug = SolderAugmentation(is_train=True)
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        tensor = aug(img)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 64, 64))
        self.assertTrue((tensor >= 0.0).all())
        self.assertTrue((tensor <= 1.0).all())

    def test_augmentation_eval_mode(self):
        aug = SolderAugmentation(is_train=False)
        img = Image.new("RGB", (16, 16), color=(255, 0, 0))
        tensor = aug(img)

        self.assertEqual(tensor.shape, (3, 16, 16))
        # Pure red channel normalized
        self.assertAlmostEqual(tensor[0, 0, 0].item(), 1.0, places=3)
        self.assertAlmostEqual(tensor[1, 0, 0].item(), 0.0, places=3)
        self.assertAlmostEqual(tensor[2, 0, 0].item(), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
