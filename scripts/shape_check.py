"""Shape and Dynamic SSIM Window Validation Script (Build Spec §2.4 & Implementation Plan Phase 1.5).

Validates:
1. SolderCAE forward pass preserves exact spatial dimensions across powers of 2 and odd sizes.
2. Odd-size center-crop handles 15px, 17px, etc. seamlessly.
3. Custom SSIM spatial map function produces valid (non-NaN, non-degenerate) maps matching input shapes.
4. Dynamic window logic properly clamps on sub-11px inputs (6px -> 5, 8px -> 7).
5. Guards against sub-3px inputs with a clean ValueError.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.config import DEFAULT_BASE_CHANNELS, DEFAULT_IN_CHANNELS
from src.model import SolderCAE
from src.loss.ssim_loss import compute_win_size, ssim_map


def run_shape_checks() -> bool:
    print("=" * 80)
    print("SolSight — Solder Defect Detection System: Architectural Shape & SSIM Check")
    print("=" * 80)

    resolutions = [
        # (size, expected_win_size, tier_name)
        (6, 5, "Window-logic check (round-to-odd: 6 -> 5)"),
        (8, 7, "Window-logic check (round-to-odd: 8 -> 7)"),
        (15, 11, "Operational (odd floor constraint)"),
        (16, 11, "Operational (trained native tier)"),
        (17, 11, "Operational (odd size)"),
        (32, 11, "Operational (untrained generalization tier)"),
        (64, 11, "Operational (trained native tier)"),
        (96, 11, "Operational (arbitrary scale)"),
        (128, 11, "Operational (trained native tier)"),
    ]

    model = SolderCAE(in_channels=DEFAULT_IN_CHANNELS, base_channels=DEFAULT_BASE_CHANNELS)
    model.eval()


    all_passed = True
    print(f"{'Resolution':<12} | {'Input Shape':<16} | {'Output Shape':<16} | {'SSIM Map Shape':<16} | {'Win Size':<9} | {'Win Cov %':<10} | {'Status'}")
    print("-" * 95)

    with torch.no_grad():
        for res, exp_win, description in resolutions:
            x = torch.rand(1, 3, res, res, dtype=torch.float32)

            try:
                # 1. Forward pass
                recon = model(x)
                shape_ok = (recon.shape == x.shape)
                bounds_ok = (recon.min() >= 0.0 and recon.max() <= 1.0)

                # 2. Window size computation
                derived_win = compute_win_size(res, res)
                win_ok = (derived_win == exp_win)

                # 3. SSIM map and scalar loss
                smap, ssim_scalar = ssim_map(x, recon, data_range=1.0)
                ssim_shape_ok = (smap.shape == (1, 1, res, res))
                no_nan = not torch.isnan(smap).any() and not torch.isnan(ssim_scalar).any()

                passed = shape_ok and bounds_ok and win_ok and ssim_shape_ok and no_nan
                if not passed:
                    all_passed = False

                status = "PASS" if passed else "FAIL"
                coverage_pct = (derived_win / res) * 100.0

                print(
                    f"{res}x{res:<9} | {str(tuple(x.shape)):<16} | {str(tuple(recon.shape)):<16} | "
                    f"{str(tuple(smap.shape)):<16} | {derived_win:<9} | {coverage_pct:5.1f}%     | {status}"
                )

            except Exception as e:
                all_passed = False
                print(f"{res}x{res:<9} | ERROR: {str(e)}")

    print("-" * 95)

    # Sub-3px guard check
    print("\nTesting sub-3px guard (expected: ValueError)...")
    guard_passed = False
    try:
        compute_win_size(2, 2)
        print("FAIL: Sub-3px input did not raise ValueError!")
    except ValueError as e:
        guard_passed = True
        print(f"PASS: Caught expected ValueError: {e}")

    if not guard_passed:
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL RESOLUTIONS AND SHAPE CHECKS PASSED SUCCESSFULLY.")
        print("CAE model and dynamic SSIM pipeline verified resolution-agnostic.")
    else:
        print("SOME CHECKS FAILED. Please review the output above.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_shape_checks()
    sys.exit(0 if success else 1)
