"""Unit tests for dynamic SSIM loss and window computation."""

import unittest
import torch
from src.loss.ssim_loss import compute_win_size, ssim_map


class TestSSIMLoss(unittest.TestCase):
    """Test dynamic SSIM window scaling and structural dissimilarity map computation."""

    def test_sub_3px_guard_raises_value_error(self):
        """Inputs below 3px must raise ValueError to prevent degenerate window calculation."""
        with self.assertRaises(ValueError) as ctx:
            compute_win_size(2, 2)
        self.assertIn("below 3px are not supported", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            compute_win_size(1, 1)
        self.assertIn("below 3px are not supported", str(ctx.exception))

    def test_dynamic_window_sizing_odd_constraint(self):
        """Window sizes must always be odd integers clamped to spatial dimensions."""
        self.assertEqual(compute_win_size(6, 6), 5)
        self.assertEqual(compute_win_size(7, 7), 7)
        self.assertEqual(compute_win_size(8, 8), 7)
        self.assertEqual(compute_win_size(10, 10), 9)
        self.assertEqual(compute_win_size(11, 11), 11)
        self.assertEqual(compute_win_size(16, 16), 11)
        self.assertEqual(compute_win_size(64, 64), 11)
        self.assertEqual(compute_win_size(128, 128), 11)

    def test_ssim_map_shape_and_bounds(self):
        """SSIM map must match input spatial dimensions and lie strictly in [-1, 1]."""
        for size in [6, 8, 15, 16, 32, 64]:
            x = torch.rand(2, 3, size, size)
            y = torch.rand(2, 3, size, size)
            s_map, s_scalar = ssim_map(x, y, data_range=1.0)

            self.assertEqual(s_map.shape, (2, 1, size, size))
            self.assertTrue(torch.all(s_map >= -1.0))
            self.assertTrue(torch.all(s_map <= 1.0))
            self.assertFalse(torch.isnan(s_map).any())
            self.assertFalse(torch.isnan(s_scalar).any())

    def test_ssim_identity_and_dissimilarity(self):
        """Identical inputs must yield SSIM of ~1.0; inverted inputs must yield significantly lower SSIM."""
        x = torch.rand(1, 3, 32, 32)
        _, s_ident = ssim_map(x, x, data_range=1.0)
        self.assertAlmostEqual(s_ident.item(), 1.0, places=4)

        inverted = 1.0 - x
        _, s_inv = ssim_map(x, inverted, data_range=1.0)
        self.assertLess(s_inv.item(), 0.5)


if __name__ == "__main__":
    unittest.main()
