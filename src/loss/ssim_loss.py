"""Custom differentiable SSIM implementation with runtime dynamic window sizing and spatial dissimilarity maps.

Key requirements:
1. Returns both unreduced spatial map (B, 1, H, W) for localized heatmaps and scalar mean for loss.
2. Dynamically derives window size from runtime (H, W) to support inputs from 15px to 128px+.
3. Mandatory data_range=1.0 for [0, 1] normalized images.
4. Non-assert ValueError guard for sub-3px inputs (survives python -O).
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """1D Gaussian kernel, normalized to sum to 1.0."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    return g / g.sum()


def _gaussian_kernel_2d(size: int, sigma: float, channels: int) -> torch.Tensor:
    """2D Gaussian kernel for depthwise convolution, shape (channels, 1, size, size)."""
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.unsqueeze(1) * k1d.unsqueeze(0)  # Outer product
    kernel = k2d.expand(channels, 1, size, size).contiguous()
    return kernel


def compute_win_size(h: int, w: int) -> int:
    """Derive SSIM Gaussian window size at runtime from spatial dimensions.

    Logic:
    - Default window is 11x11 (standard for SSIM).
    - If either spatial dimension is smaller than 11, clamp to min(h, w).
    - Window size must be odd.
    - Floor at 3x3.
    - Raise ValueError if window size exceeds min(h, w) (guard for h, w < 3).

    Args:
        h: Height of patch in pixels.
        w: Width of patch in pixels.

    Returns:
        Odd integer window size.
    """
    win = min(11, h, w)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)

    if win > min(h, w):
        raise ValueError(
            f"Derived SSIM window size ({win}) exceeds spatial dimensions ({h}, {w}). "
            f"Inputs below 3px are not supported."
        )

    return win


def ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    win_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-pixel SSIM spatial map and scalar mean between x and y.

    Args:
        x: First image batch of shape (B, C, H, W) normalized to [0, 1].
        y: Second image batch of shape (B, C, H, W) normalized to [0, 1].
        data_range: Value range of inputs (must be 1.0 for sigmoid-bounded inputs).
        win_size: Optional fixed window size. If None, derived dynamically from (H, W).

    Returns:
        Tuple of:
            - spatial_map: Tensor of shape (B, 1, H, W) representing local structural similarity.
            - scalar_mean: Scalar Tensor mean SSIM for optimization (loss = 1 - scalar_mean).
    """
    if x.shape != y.shape:
        raise ValueError(f"Input tensors must have identical shape, got {x.shape} vs {y.shape}")

    b, c, h, w = x.shape

    if win_size is None:
        win_size = compute_win_size(h, w)

    sigma = 1.5  # Standard SSIM Gaussian sigma
    kernel = _gaussian_kernel_2d(win_size, sigma, c).to(device=x.device, dtype=x.dtype)
    pad = win_size // 2

    # Windowed statistics via depthwise convolution
    mu_x = F.conv2d(x, kernel, padding=pad, groups=c)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=c)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=pad, groups=c) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=pad, groups=c) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=c) - mu_xy

    # SSIM stability constants
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    smap = num / den                        # Shape: (B, C, H, W)
    spatial_map = smap.mean(dim=1, keepdim=True)  # Mean across channels -> (B, 1, H, W)
    scalar_mean = spatial_map.mean()

    return spatial_map, scalar_mean
