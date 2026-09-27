"""Multi-Scale Training Pipeline for SolSight (Implementation Plan Phase 3).

Key elements:
1. Loss: 1.0 - SSIM(batch, recon) with dynamic window and data_range=1.0.
2. Multi-scale training over discrete native tiers {16, 64, 128} (one checkpoint).
3. Pre-committed Epoch 10 hard convergence check across 16px, 64px, 128px val sets.
4. Early stopping (patience=5) on validation loss plateau.
5. Saves best checkpoint to outputs/checkpoints/best_cae.pt and plots training curves.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from src.model.cae import SolderCAE
from src.loss.ssim_loss import ssim_map
from src.data.dataset import DiscreteMultiScaleTrainLoader, get_fixed_tier_loader


def evaluate_tier(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """Evaluate 1 - SSIM loss on a fixed tier validation set."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device)
            recon = model(batch)
            _, ssim_scalar = ssim_map(batch, recon, data_range=1.0)
            loss = 1.0 - ssim_scalar.item()
            total_loss += loss * batch.size(0)
            count += batch.size(0)

    return total_loss / max(1, count)


def evaluate_all_tiers(
    model: torch.nn.Module,
    val_loaders: Dict[int, torch.utils.data.DataLoader],
    device: torch.device,
) -> Tuple[float, Dict[int, float]]:
    """Evaluate per-resolution validation loss across all tiers."""
    losses = {}
    for tier, loader in val_loaders.items():
        losses[tier] = evaluate_tier(model, loader, device)
    overall_mean = float(np.mean(list(losses.values())))
    return overall_mean, losses


def train(
    data_dir: str = "data/synthetic",
    output_dir: str = "outputs",
    max_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    in_channels: int = 3,
    base_channels: int = 16,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"SolSight Multi-Scale Training | Device: {device} | Max Epochs: {max_epochs}")
    print("=" * 80)

    # Output paths
    out_path = Path(output_dir)
    ckpt_dir = out_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_cae.pt"

    # Datasets and Loaders
    train_dir = Path(data_dir) / "train"
    train_loader = DiscreteMultiScaleTrainLoader(
        base_dir=train_dir,
        tiers=(16, 64, 128),
        batch_size=batch_size,
    )

    val_base = Path(data_dir) / "val_normal"
    val_loaders = {
        16: get_fixed_tier_loader(val_base / "16px", batch_size=batch_size, is_train=False),
        64: get_fixed_tier_loader(val_base / "64px", batch_size=batch_size, is_train=False),
        128: get_fixed_tier_loader(val_base / "128px", batch_size=batch_size, is_train=False),
    }

    # Model & Optimizer
    model = SolderCAE(in_channels=in_channels, base_channels=base_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)

    # History tracking
    history = {
        "train_loss": [],
        "val_loss_overall": [],
        "val_loss_16": [],
        "val_loss_64": [],
        "val_loss_128": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    epoch_10_losses = {}

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        steps = len(train_loader)

        for batch, tier in train_loader:
            batch = batch.to(device)
            recon = model(batch)

            _, ssim_scalar = ssim_map(batch, recon, data_range=1.0)
            loss = 1.0 - ssim_scalar

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / steps

        # Validation
        val_overall, val_per_tier = evaluate_all_tiers(model, val_loaders, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss_overall"].append(val_overall)
        history["val_loss_16"].append(val_per_tier[16])
        history["val_loss_64"].append(val_per_tier[64])
        history["val_loss_128"].append(val_per_tier[128])

        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss (1-SSIM): {avg_train_loss:.4f} | "
            f"Val Loss: {val_overall:.4f} "
            f"(16px: {val_per_tier[16]:.4f}, 64px: {val_per_tier[64]:.4f}, 128px: {val_per_tier[128]:.4f})"
        )

        scheduler.step(val_overall)

        # -------------------------------------------------------------
        # Hard Convergence Checkpoint at Epoch 10 (Rev 8 specification)
        # -------------------------------------------------------------
        if epoch == 10:
            print("\n" + "=" * 80)
            print("EPOCH 10 HARD CONVERGENCE CHECKPOINT")
            print("=" * 80)
            loss_init_16 = history["val_loss_16"][0]
            loss_init_64 = history["val_loss_64"][0]
            loss_init_128 = history["val_loss_128"][0]

            t16_down = val_per_tier[16] < loss_init_16
            t64_down = val_per_tier[64] < loss_init_64
            t128_down = val_per_tier[128] < loss_init_128

            down_count = sum([t16_down, t64_down, t128_down])
            print(f"16px  trend: initial={loss_init_16:.4f} -> ep10={val_per_tier[16]:.4f} [{'DOWN' if t16_down else 'FLAT/UP'}]")
            print(f"64px  trend: initial={loss_init_64:.4f} -> ep10={val_per_tier[64]:.4f} [{'DOWN' if t64_down else 'FLAT/UP'}]")
            print(f"128px trend: initial={loss_init_128:.4f} -> ep10={val_per_tier[128]:.4f} [{'DOWN' if t128_down else 'FLAT/UP'}]")

            if down_count == 3:
                print("DECISION: [GO] - All 3 tiers converging. Proceeding multi-scale training.")
            elif down_count == 2:
                print("DECISION: [CONDITIONAL GO] - 2 tiers converging. Monitoring for 5 more epochs.")
            else:
                print("DECISION: [STOP/FALLBACK] - Convergence threshold failed. Please investigate.")
            print("=" * 80 + "\n")

        # Save best model
        if val_overall < best_val_loss:
            best_val_loss = val_overall
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "val_per_tier": val_per_tier,
                "history": history,
                "config": {
                    "in_channels": in_channels,
                    "base_channels": base_channels,
                },
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= 15:
                print(f"\nEarly stopping triggered: Validation loss did not improve for {patience} epochs.")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # Plot training curves
    plot_curves(history, out_path / "training_curves.png")


def plot_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    """Plot and save training and per-resolution validation loss curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], "k--", label="Train Loss (Mean 1-SSIM)", linewidth=1.5)
    plt.plot(epochs, history["val_loss_overall"], "b-", label="Val Loss Overall", linewidth=2.0)
    plt.plot(epochs, history["val_loss_16"], "r-.", label="Val 16px", alpha=0.75)
    plt.plot(epochs, history["val_loss_64"], "g-.", label="Val 64px", alpha=0.75)
    plt.plot(epochs, history["val_loss_128"], "m-.", label="Val 128px", alpha=0.75)

    plt.xlabel("Epoch")
    plt.ylabel("1 - SSIM Loss")
    plt.title("SolSight CAE Multi-Scale Training Curves (SSIM Loss across Native Tiers)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves plot to {save_path}")


if __name__ == "__main__":
    import argparse
    from src.config import (
        DEFAULT_BASE_CHANNELS,
        DEFAULT_IN_CHANNELS,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SYNTHETIC_DATA_DIR,
    )

    parser = argparse.ArgumentParser(description="Multi-Scale Training Pipeline for SolSight")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_SYNTHETIC_DATA_DIR), help="Path to synthetic dataset")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--in-channels", type=int, default=DEFAULT_IN_CHANNELS, help="Input channels")
    parser.add_argument("--base-channels", type=int, default=DEFAULT_BASE_CHANNELS, help="Base feature channels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        seed=args.seed,
    )

