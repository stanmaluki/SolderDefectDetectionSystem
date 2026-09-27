"""Unit tests for SolderCAE architecture, resolution-invariance, and model loading."""

import unittest
from pathlib import Path
import tempfile
import torch
from src.model.cae import SolderCAE, load_trained_model


class TestSolderCAE(unittest.TestCase):
    """Test resolution invariance, sigmoid bounding, and checkpoint config restoration."""

    def setUp(self):
        self.model = SolderCAE(in_channels=3, base_channels=16)
        self.model.eval()

    def test_forward_shape_invariance_powers_of_two(self):
        """Reconstruction shape must exactly match input shape for standard power-of-two sizes."""
        for res in [16, 32, 64, 128]:
            x = torch.rand(2, 3, res, res)
            with torch.no_grad():
                out = self.model(x)
            self.assertEqual(out.shape, x.shape)
            self.assertTrue((out >= 0.0).all())
            self.assertTrue((out <= 1.0).all())

    def test_forward_shape_invariance_odd_and_arbitrary_sizes(self):
        """Center-cropping logic must preserve exact input dimensions for odd and non-power-of-two inputs."""
        for res in [15, 17, 25, 48, 55, 96]:
            x = torch.rand(1, 3, res, res)
            with torch.no_grad():
                out = self.model(x)
            self.assertEqual(out.shape, x.shape)

    def test_checkpoint_roundtrip_with_config(self):
        """Checkpoint saving and load_trained_model must preserve custom architecture hyperparameters."""
        custom_model = SolderCAE(in_channels=3, base_channels=8)
        state_dict = custom_model.state_dict()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            torch.save({
                "model_state_dict": state_dict,
                "config": {
                    "in_channels": 3,
                    "base_channels": 8,
                },
            }, tmp_path)

            loaded_model, ckpt = load_trained_model(tmp_path, device=torch.device("cpu"))
            self.assertIsInstance(loaded_model, SolderCAE)
            self.assertEqual(ckpt["config"]["base_channels"], 8)

            # Test forward pass with loaded model
            x = torch.rand(1, 3, 32, 32)
            with torch.no_grad():
                out = loaded_model(x)
            self.assertEqual(out.shape, (1, 3, 32, 32))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
