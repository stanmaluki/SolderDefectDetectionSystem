"""Inference Pipeline for SolSight (Implementation Plan Phase 4).

Provides:
1. inspect_patch: Computes CAE reconstruction, unreduced DSSIM heatmap, and top-5% anomaly score.
2. overlay_heatmap: Composites structural dissimilarity heatmap over original solder patch.
3. Batch and single-image inference supporting arbitrary input resolutions.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from src.loss.ssim_loss import ssim_map


def compute_top_k_dssim_score(dssim_map: torch.Tensor, top_pct: float = 0.05) -> float:
    """Compute anomaly score from top-k percentile regions of the DSSIM map.

    Formula: score = mean(top 5% highest-DSSIM regions in dssim_map)
    Ensures small, localized defects (e.g. pinhole void covering ~2% of patch)
    are not diluted by whole-patch averaging across normal regions.

    Args:
        dssim_map: Spatial DSSIM map of shape (H, W) or (1, 1, H, W) in [0, 1].
        top_pct: Fraction of highest pixels to average (default: 0.05 for top 5%).

    Returns:
        Scalar anomaly score in [0, 1].
    """
    flat = dssim_map.flatten()
    k = max(1, int(round(flat.numel() * top_pct)))
    top_values, _ = torch.topk(flat, k=k)
    return float(top_values.mean().item())


def inspect_patch(
    model: torch.nn.Module,
    patch: Union[torch.Tensor, Image.Image, np.ndarray],
    threshold: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Run full SolSight inspection pipeline on a single solder joint patch.

    Args:
        model: Trained SolderCAE model.
        patch: Image patch as PIL Image, NumPy array, or torch.Tensor.
        threshold: Optional anomaly score threshold. If provided, yields is_defect boolean.
        device: Torch compute device.

    Returns:
        Dictionary containing:
            - input_tensor: (1, 3, H, W)
            - recon_tensor: (1, 3, H, W)
            - dssim_map: (H, W) numpy array in [0, 1]
            - anomaly_score: float
            - is_anomaly: bool (if threshold provided, else None)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Convert patch to normalized tensor (1, 3, H, W) in [0, 1]
    if isinstance(patch, Image.Image):
        arr = np.array(patch.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    elif isinstance(patch, np.ndarray):
        if patch.max() > 1.0:
            patch = patch.astype(np.float32) / 255.0
        if patch.ndim == 3:
            tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)
        else:
            tensor = torch.from_numpy(patch)
    elif isinstance(patch, torch.Tensor):
        tensor = patch if patch.ndim == 4 else patch.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported patch type: {type(patch)}")

    tensor = tensor.to(device).float()

    with torch.no_grad():
        recon = model(tensor)
        # Compute spatial SSIM map (B, 1, H, W) and convert to dissimilarity: DSSIM = (1 - SSIM) / 2
        ssim_spatial, ssim_mean = ssim_map(tensor, recon, data_range=1.0)
        dssim_tensor = torch.clamp((1.0 - ssim_spatial) / 2.0, 0.0, 1.0)

        # Compute top-5% anomaly score
        score = compute_top_k_dssim_score(dssim_tensor[0, 0], top_pct=0.05)

    dssim_np = dssim_tensor[0, 0].cpu().numpy()
    is_anomaly = (score > threshold) if threshold is not None else None

    return {
        "input_tensor": tensor.cpu(),
        "recon_tensor": recon.cpu(),
        "dssim_map": dssim_np,
        "anomaly_score": score,
        "is_anomaly": is_anomaly,
        "ssim_mean": float(ssim_mean.item()),
    }


def create_heatmap_overlay(
    input_img: Union[Image.Image, np.ndarray, torch.Tensor],
    dssim_map: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "hot",
) -> Image.Image:
    """Composite the DSSIM heatmap over the original input patch."""
    if isinstance(input_img, torch.Tensor):
        if input_img.ndim == 4:
            input_img = input_img[0]
        input_np = input_img.permute(1, 2, 0).cpu().numpy()
        input_uint8 = np.clip(input_np * 255.0, 0, 255).astype(np.uint8)
        base_pil = Image.fromarray(input_uint8)
    elif isinstance(input_img, np.ndarray):
        if input_img.max() <= 1.0:
            input_img = input_img * 255.0
        base_pil = Image.fromarray(input_img.astype(np.uint8))
    else:
        base_pil = input_img.convert("RGB")

    # Colormap transformation on normalized DSSIM map
    norm_map = np.clip(dssim_map, 0.0, 1.0)
    try:
        cmap = plt.get_cmap(colormap)
    except Exception:
        import matplotlib
        cmap = matplotlib.colormaps[colormap]
    rgba = cmap(norm_map)  # (H, W, 4) in [0, 1]
    heat_rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    heat_pil = Image.fromarray(heat_rgb)

    # Blend
    blended = Image.blend(base_pil, heat_pil, alpha=alpha)
    return blended
