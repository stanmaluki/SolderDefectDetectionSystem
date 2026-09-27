"""Inference Pipeline for SolSight (Implementation Plan Phase 4).

Provides:
1. inspect_patch: Computes CAE reconstruction, unreduced DSSIM heatmap, and top-5% anomaly score.
2. overlay_heatmap: Composites structural dissimilarity heatmap over original solder patch.
3. Batch and single-image inference supporting arbitrary input resolutions.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def main():
    import argparse
    import json
    from src.model.cae import SolderCAE

    parser = argparse.ArgumentParser(description="SolSight Single-Patch Anomaly Inspection")
    parser.add_argument("--input", type=str, default=None, help="Path to patch image (png/jpg)")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_cae.pt", help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=None, help="Custom anomaly threshold (defaults to calibrated global T)")
    parser.add_argument("--output", type=str, default="outputs/prediction_result.png", help="Path to save output visual")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load threshold config
    thresh = args.threshold
    if thresh is None:
        cfg_path = Path("outputs/threshold_config.json")
        if cfg_path.exists():
            with open(cfg_path) as f:
                thresh = json.load(f).get("global_threshold", 0.1300)
        else:
            thresh = 0.1300

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint {ckpt_path} not found.")
        return

    model = SolderCAE(in_channels=3, base_channels=16).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Determine input image
    if args.input:
        img_path = Path(args.input)
        if not img_path.exists():
            print(f"Error: input file {img_path} not found.")
            return
        img = Image.open(img_path).convert("RGB")
    else:
        sample_candidates = [
            Path("data/synthetic/val_defect/voids/64px/defect_voids_64px_0000.png"),
            Path("data/synthetic/val_defect/bridging/64px/defect_bridging_64px_0000.png"),
            Path("data/synthetic/val_normal/64px/val_normal_64px_0000.png"),
        ]
        img_path = next((p for p in sample_candidates if p.exists()), None)
        if img_path:
            img = Image.open(img_path).convert("RGB")
            print(f"No --input specified. Using sample image: {img_path}")
        else:
            from src.data.synthetic_generator import render_defect_void
            img = render_defect_void(64)
            img_path = Path("synthetic_sample_64px.png")
            print("No sample found on disk. Rendered on-the-fly 64px void defect patch.")

    # Run inspection
    res = inspect_patch(model, img, threshold=thresh, device=device)
    overlay = create_heatmap_overlay(img, res["dssim_map"], alpha=0.55)

    print("\n" + "=" * 60)
    print("SOLSIGHT SOLDER DEFECT INSPECTION RESULT")
    print("=" * 60)
    print(f"Patch Size     : {img.size[0]} x {img.size[1]} px")
    print(f"Anomaly Score  : {res['anomaly_score']:.4f} (Top-5% DSSIM)")
    print(f"Threshold (T)  : {thresh:.4f}")
    verdict = "DEFECT DETECTED [REJECT]" if res['is_anomaly'] else "PASS (NORMAL) [ACCEPT]"
    print(f"Verdict        : {verdict}")
    print(f"SSIM Mean      : {res['ssim_mean']:.4f}")
    print("=" * 60)

    # Save 3-panel figure
    recon_np = np.clip(res["recon_tensor"][0].permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
    recon_img = Image.fromarray(recon_np)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(img)
    axes[0].set_title(f"Input ({img.size[0]}x{img.size[1]})")
    axes[0].axis("off")

    axes[1].imshow(recon_img)
    axes[1].set_title("CAE Reconstruction")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"DSSIM Heatmap ({verdict.split()[0]})")
    axes[2].axis("off")

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_p, dpi=150)
    plt.close()
    print(f"Saved inspection panel to: {out_p}\n")


if __name__ == "__main__":
    main()
