"""
WaveSSM-X Inference Speed Profiler
==================================
Validates O(N) complexity claim by measuring latency at multiple resolutions.

Usage:
    python profile_speed.py --checkpoint ./checkpoints/wavessm_x_best.pth
    python profile_speed.py --resolutions 128 256 512 1024 --warmup 10 --runs 50
"""
import torch
import torch.nn as nn
import argparse
import time
import numpy as np
from typing import List, Dict
import json
from pathlib import Path

from wavessm_x.models.inpainting import Inpainting


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def profile_resolution(
    model: nn.Module,
    resolution: int,
    device: torch.device,
    warmup_runs: int = 10,
    timed_runs: int = 50,
    batch_size: int = 1
) -> Dict[str, float]:
    """Profile inference at a specific resolution."""

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, resolution, resolution).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # Sync CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    latencies = []

    with torch.no_grad():
        for _ in range(timed_runs):
            start = time.perf_counter()
            _ = model(dummy_input)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)

    # Calculate throughput
    pixels = resolution * resolution * batch_size
    throughput_mpx = pixels / (latencies.mean() / 1000) / 1e6  # Megapixels/sec

    return {
        "resolution": resolution,
        "pixels": resolution * resolution,
        "batch_size": batch_size,
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "throughput_mpx_sec": float(throughput_mpx),
        "fps": float(1000 / np.mean(latencies))
    }


def estimate_memory(model: nn.Module, resolution: int, device: torch.device) -> Dict[str, float]:
    """Estimate GPU memory usage."""
    if device.type != 'cuda':
        return {"peak_mb": -1, "allocated_mb": -1}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    dummy_input = torch.randn(1, 3, resolution, resolution).to(device)

    with torch.no_grad():
        _ = model(dummy_input)

    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024

    return {"peak_mb": float(peak_mb), "allocated_mb": float(allocated_mb)}


def analyze_complexity(results: List[Dict]) -> Dict:
    """Analyze if complexity is O(N), O(N log N), or O(N²)."""

    pixels = np.array([r["pixels"] for r in results])
    latencies = np.array([r["mean_ms"] for r in results])

    # Normalize
    base_pixels = pixels[0]
    base_latency = latencies[0]

    # Expected scaling for different complexities
    # O(N): latency ∝ N
    # O(N log N): latency ∝ N log N
    # O(N²): latency ∝ N²

    ratios = pixels / base_pixels
    actual_scaling = latencies / base_latency

    # Calculate fit error for each complexity model
    o_n = ratios
    o_n_log_n = ratios * np.log2(ratios + 1)
    o_n_sq = ratios ** 2

    error_o_n = np.mean((actual_scaling - o_n) ** 2)
    error_o_n_log_n = np.mean((actual_scaling - o_n_log_n) ** 2)
    error_o_n_sq = np.mean((actual_scaling - o_n_sq) ** 2)

    # Determine best fit
    errors = {
        "O(N)": error_o_n,
        "O(N log N)": error_o_n_log_n,
        "O(N²)": error_o_n_sq
    }

    best_fit = min(errors, key=errors.get)

    return {
        "best_fit": best_fit,
        "errors": errors,
        "actual_scaling": actual_scaling.tolist(),
        "expected_o_n": o_n.tolist(),
        "expected_o_n_log_n": o_n_log_n.tolist(),
        "expected_o_n_sq": o_n_sq.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="WaveSSM-X Speed Profiler")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (optional)")
    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=[128, 256, 384, 512, 768, 1024],
                        help="Resolutions to test")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup runs before timing")
    parser.add_argument("--runs", type=int, default=50,
                        help="Timed runs per resolution")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for profiling")
    parser.add_argument("--use_mamba", action="store_true", default=True,
                        help="Use Mamba architecture")
    parser.add_argument("--no_fass", action="store_true",
                        help="Disable DualStreamFASS")
    parser.add_argument("--no_ffc", action="store_true",
                        help="Disable MultiScaleWaveFFC")
    parser.add_argument("--output", type=str, default="speed_profile.json",
                        help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize model
    print("\nInitializing model...")
    model = Inpainting(
        use_mamba=args.use_mamba,
        use_fass=not args.no_fass,
        use_ffc=not args.no_ffc
    ).to(device)

    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {args.checkpoint}")

    model.eval()

    # Count parameters
    params = count_parameters(model)
    print(f"Parameters: {params['total']:,} total | {params['trainable']:,} trainable")

    # Profile each resolution
    print(f"\nProfiling {len(args.resolutions)} resolutions...")
    print(f"Warmup: {args.warmup} | Timed runs: {args.runs} | Batch: {args.batch_size}")
    print("-" * 70)

    results = []

    for res in args.resolutions:
        try:
            print(f"\n[{res}x{res}]", end=" ")

            # Speed profile
            speed = profile_resolution(
                model, res, device,
                warmup_runs=args.warmup,
                timed_runs=args.runs,
                batch_size=args.batch_size
            )

            # Memory profile
            memory = estimate_memory(model, res, device)

            result = {**speed, **memory}
            results.append(result)

            print(f"→ {speed['mean_ms']:.2f} ± {speed['std_ms']:.2f} ms | "
                  f"{speed['fps']:.1f} FPS | {memory['peak_mb']:.0f} MB")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"→ OOM (resolution too large)")
                torch.cuda.empty_cache()
            else:
                raise e

    # Analyze complexity
    print("\n" + "=" * 70)
    print("COMPLEXITY ANALYSIS")
    print("=" * 70)

    if len(results) >= 3:
        complexity = analyze_complexity(results)
        print(f"\nBest fit: {complexity['best_fit']}")
        print(f"Fit errors:")
        for name, err in complexity['errors'].items():
            print(f"  {name}: {err:.4f}")

        # Print scaling table
        print("\nScaling Analysis:")
        print(f"{'Resolution':>10} {'Pixels':>12} {'Actual':>10} {'O(N)':>10} {'O(N logN)':>10} {'O(N²)':>10}")
        print("-" * 70)
        for i, r in enumerate(results):
            print(f"{r['resolution']:>10} {r['pixels']:>12,} "
                  f"{complexity['actual_scaling'][i]:>10.2f} "
                  f"{complexity['expected_o_n'][i]:>10.2f} "
                  f"{complexity['expected_o_n_log_n'][i]:>10.2f} "
                  f"{complexity['expected_o_n_sq'][i]:>10.2f}")
    else:
        complexity = {}

    # Save results
    output = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
        "model_config": {
            "use_mamba": args.use_mamba,
            "use_fass": not args.no_fass,
            "use_ffc": not args.no_ffc
        },
        "parameters": params,
        "profiling_config": {
            "warmup_runs": args.warmup,
            "timed_runs": args.runs,
            "batch_size": args.batch_size
        },
        "results": results,
        "complexity_analysis": complexity
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Resolution':>10} {'Latency (ms)':>15} {'FPS':>10} {'Memory (MB)':>15}")
    print("-" * 70)
    for r in results:
        print(f"{r['resolution']:>10} {r['mean_ms']:>12.2f} ± {r['std_ms']:.1f} "
              f"{r['fps']:>10.1f} {r['peak_mb']:>15.0f}")


if __name__ == "__main__":
    main()
