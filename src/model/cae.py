"""Resolution-agnostic Convolutional Autoencoder (CAE) for Solder Defect Detection.

Architectural highlights:
- Fully convolutional: No Dense/Flatten layers, no hardcoded H/W dimensions.
- Total downsampling factor of 4x (two stride-2 convolutions).
- No skip connections: forces anomalies to be compressed and reconstructed from
  normal prior representation.
- Sigmoid activation on output: bounds output to [0, 1] range for SSIM stability.
- Center-crop odd-size handling: crops decoder output back to exact input (H, W).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class SolderCAE(nn.Module):
    """Fully Convolutional Autoencoder for unsupervised PCBA solder anomaly detection."""

    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()

        # Encoder: 4x total downsampling (2x at conv2, 2x at conv4)
        c1 = base_channels       # 16
        c2 = base_channels * 2   # 32
        c3 = base_channels * 4   # 64

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1),  # Downsample x2
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1),  # Downsample x2 (x4 total)
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Bottleneck: preserves spatial correspondence
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder: Upsample + Conv to avoid checkerboard artifacts
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c3, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c2, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder, bottleneck, and decoder with odd-size center crop.

        Args:
            x: Input tensor of shape (B, C, H, W) normalized to [0, 1].

        Returns:
            Reconstruction tensor of exact same shape (B, C, H, W) in [0, 1].
        """
        orig_h, orig_w = x.shape[2], x.shape[3]

        latent = self.encoder(x)
        bottleneck_feat = self.bottleneck(latent)
        reconstruction = self.decoder(bottleneck_feat)

        # Center-crop decoder output back to input dimensions if upsampling rounded up
        out_h, out_w = reconstruction.shape[2], reconstruction.shape[3]
        if out_h != orig_h or out_w != orig_w:
            start_h = (out_h - orig_h) // 2
            start_w = (out_w - orig_w) // 2
            reconstruction = reconstruction[:, :, start_h : start_h + orig_h, start_w : start_w + orig_w]

        return reconstruction


def load_trained_model(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    eval_mode: bool = True,
) -> Tuple[SolderCAE, Dict[str, Any]]:
    """Load a trained SolderCAE model from a checkpoint.

    Reads architecture hyperparameters ('config') from the checkpoint dictionary
    if present, falling back to defaults (in_channels=3, base_channels=16)
    for backward compatibility.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to place the model on. Defaults to cuda if available, else cpu.
        eval_mode: Whether to set model.eval() upon returning.

    Returns:
        Tuple of (model, checkpoint_dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    in_channels = config.get("in_channels", 3)
    base_channels = config.get("base_channels", 16)

    model = SolderCAE(in_channels=in_channels, base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    if eval_mode:
        model.eval()
    return model, ckpt
