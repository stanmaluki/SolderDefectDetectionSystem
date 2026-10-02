"""Inference Latency Benchmark across Resolutions (GPU vs CPU)."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import torch
from src.model import load_trained_model
from src.config import DEFAULT_CHECKPOINT_PATH

def benchmark():
    print("=" * 80)
    print("SolSight Inference Latency Benchmark (T4G GPU vs ARM64 CPU)")
    print("=" * 80)

    device_gpu = "cuda" if torch.cuda.is_available() else None
    if not device_gpu:
        print("CUDA device not available! Exiting.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Device: {gpu_name}")
    print(f"Checkpoint: {DEFAULT_CHECKPOINT_PATH}\n")

    model_gpu, _ = load_trained_model(str(DEFAULT_CHECKPOINT_PATH), device="cuda")
    model_cpu, _ = load_trained_model(str(DEFAULT_CHECKPOINT_PATH), device="cpu")
    model_gpu.eval()
    model_cpu.eval()

    print(f"{'Resolution':<12} | {'T4G GPU (ms)':<14} | {'T4G GPU (fps)':<14} | {'CPU (ms)':<10} | {'CPU (fps)':<10} | {'Speedup':<8}")
    print("-" * 80)

    for res in [16, 32, 64, 128]:
        x_gpu = torch.randn(1, 3, res, res, device="cuda")
        x_cpu = torch.randn(1, 3, res, res, device="cpu")

        # Warmup
        with torch.no_grad():
            for _ in range(50):
                _ = model_gpu(x_gpu)
                _ = model_cpu(x_cpu)
            torch.cuda.synchronize()

            # GPU timing (batch of 100 single patches)
            t0 = time.perf_counter()
            for _ in range(100):
                _ = model_gpu(x_gpu)
            torch.cuda.synchronize()
            gpu_ms = ((time.perf_counter() - t0) / 100) * 1000

            # CPU timing (batch of 100 single patches)
            t0 = time.perf_counter()
            for _ in range(100):
                _ = model_cpu(x_cpu)
            cpu_ms = ((time.perf_counter() - t0) / 100) * 1000

        gpu_fps = 1000.0 / gpu_ms
        cpu_fps = 1000.0 / cpu_ms
        speedup = cpu_ms / gpu_ms
        print(f"{res}x{res:<9} | {gpu_ms:8.2f} ms     | {gpu_fps:8.1f} fps    | {cpu_ms:6.2f} ms | {cpu_fps:6.1f} fps | {speedup:6.1f}x")

    print("=" * 80)

if __name__ == "__main__":
    benchmark()
