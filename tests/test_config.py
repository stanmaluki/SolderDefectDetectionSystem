"""Unit tests for centralized configuration and path resolution."""

import unittest
from src.config import (
    DEFAULT_ALL_TIERS,
    DEFAULT_CAD_DIR,
    DEFAULT_CAD_CROPS_DIR,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REAL_DATA_DIR,
    DEFAULT_SYNTHETIC_DATA_DIR,
    DEFAULT_TEST_DEFECTS_DIR,
    DEFAULT_TEST_NORMAL_DIR,
    DEFAULT_THRESHOLD_CONFIG,
    DEFAULT_TRAIN_TIERS,
    FALLBACK_THRESHOLD,
    PROJECT_ROOT,
)


class TestConfig(unittest.TestCase):
    """Test path hierarchy and hyperparameter defaults."""

    def test_project_root_exists(self):
        self.assertTrue(PROJECT_ROOT.exists())
        self.assertTrue((PROJECT_ROOT / "src").is_dir())

    def test_tier_tuples(self):
        self.assertEqual(DEFAULT_TRAIN_TIERS, (16, 64, 128))
        self.assertEqual(DEFAULT_ALL_TIERS, (16, 32, 64, 128))
        # Train tiers must be subset of all tiers
        for tier in DEFAULT_TRAIN_TIERS:
            self.assertIn(tier, DEFAULT_ALL_TIERS)

    def test_cad_paths_derivation(self):
        self.assertEqual(DEFAULT_CAD_DIR, PROJECT_ROOT / "data" / "cad")
        self.assertEqual(DEFAULT_CAD_CROPS_DIR, DEFAULT_OUTPUT_DIR / "cad_crops")

    def test_test_splits_derivation(self):
        self.assertEqual(DEFAULT_TEST_NORMAL_DIR, DEFAULT_SYNTHETIC_DATA_DIR / "test_normal")
        self.assertEqual(DEFAULT_TEST_DEFECTS_DIR, DEFAULT_SYNTHETIC_DATA_DIR / "test_defects")

    def test_default_paths_derivation(self):
        self.assertEqual(DEFAULT_CHECKPOINT_PATH, DEFAULT_OUTPUT_DIR / "checkpoints" / "best_cae.pt")
        self.assertEqual(DEFAULT_THRESHOLD_CONFIG, DEFAULT_OUTPUT_DIR / "threshold_config.json")
        self.assertEqual(DEFAULT_REAL_DATA_DIR, PROJECT_ROOT / "data" / "real")

    def test_threshold_fallback_bounds(self):
        self.assertGreater(FALLBACK_THRESHOLD, 0.0)
        self.assertLess(FALLBACK_THRESHOLD, 1.0)


if __name__ == "__main__":
    unittest.main()
