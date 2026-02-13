"""
WaveSSM-X Ablation Study Runner
===============================
Systematically runs ablation experiments for research paper.

Usage:
    python run_ablations.py --data_path ./datasets/celeb --num_iter 50000
    python run_ablations.py --config frequency_modulation --seeds 42 43 44
    python run_ablations.py --config all --dry_run  # Preview commands
"""
import os
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    # ─── A. Architecture Ablations ───
    "full_model": {
        "desc": "Full WaveSSM-X (baseline)",
        "args": "--use_mamba"
    },
    "no_fass": {
        "desc": "Without DualStreamFASS bottleneck",
        "args": "--use_mamba --no_fass"
    },
    "no_ffc": {
        "desc": "Without MultiScaleWaveFFC refinement",
        "args": "--use_mamba --no_ffc"
    },
    "mamba_only": {
        "desc": "Mamba backbone only (no FASS, no FFC)",
        "args": "--use_mamba --no_fass --no_ffc"
    },
    "transformer": {
        "desc": "Transformer baseline (no Mamba)",
        "args": "--no_fass --no_ffc"
    },

    # ─── B. Frequency Modulation Isolation ───
    "fass_no_b": {
        "desc": "FASS without B (structure) modulation",
        "args": "--use_mamba --fass_no_b"
    },
    "fass_no_c": {
        "desc": "FASS without C (texture) modulation",
        "args": "--use_mamba --fass_no_c"
    },
    "fass_no_delta": {
        "desc": "FASS without Delta (edge) modulation",
        "args": "--use_mamba --fass_no_delta"
    },
    "fass_b_only": {
        "desc": "FASS with B modulation only",
        "args": "--use_mamba --fass_no_c --fass_no_delta"
    },
    "fass_c_only": {
        "desc": "FASS with C modulation only",
        "args": "--use_mamba --fass_no_b --fass_no_delta"
    },
    "fass_delta_only": {
        "desc": "FASS with Delta modulation only",
        "args": "--use_mamba --fass_no_b --fass_no_c"
    },

    # ─── C. Wavelet Type Sensitivity ───
    "wavelet_haar": {
        "desc": "Haar wavelet (simplest)",
        "args": "--use_mamba --wavelet haar"
    },
    "wavelet_db1": {
        "desc": "Daubechies-1 wavelet",
        "args": "--use_mamba --wavelet db1"
    },
    "wavelet_db3": {
        "desc": "Daubechies-3 wavelet (default)",
        "args": "--use_mamba --wavelet db3"
    },
    "wavelet_db5": {
        "desc": "Daubechies-5 wavelet",
        "args": "--use_mamba --wavelet db5"
    },
    "wavelet_sym4": {
        "desc": "Symlet-4 wavelet",
        "args": "--use_mamba --wavelet sym4"
    },
    "wavelet_coif2": {
        "desc": "Coiflet-2 wavelet",
        "args": "--use_mamba --wavelet coif2"
    },

    # ─── D. Loss Function Ablations ───
    "no_freq_loss": {
        "desc": "Without frequency loss",
        "args": "--use_mamba --loss_weights 1.0 1.0 1.0 0.5 0.0"
    },
    "no_perceptual_loss": {
        "desc": "Without perceptual loss",
        "args": "--use_mamba --loss_weights 1.0 0.0 1.0 0.5 0.3"
    },
    "no_ssim_loss": {
        "desc": "Without SSIM loss",
        "args": "--use_mamba --loss_weights 1.0 1.0 0.0 0.5 0.3"
    },
    "l1_only": {
        "desc": "L1 loss only",
        "args": "--use_mamba --loss_weights 1.0 0.0 0.0 0.0 0.0"
    },
}

# Experiment groups for selective running
EXPERIMENT_GROUPS = {
    "architecture": ["full_model", "no_fass", "no_ffc", "mamba_only", "transformer"],
    "frequency_modulation": ["full_model", "fass_no_b", "fass_no_c", "fass_no_delta",
                             "fass_b_only", "fass_c_only", "fass_delta_only"],
    "wavelet": ["wavelet_haar", "wavelet_db1", "wavelet_db3", "wavelet_db5",
                "wavelet_sym4", "wavelet_coif2"],
    "loss": ["full_model", "no_freq_loss", "no_perceptual_loss", "no_ssim_loss", "l1_only"],
    "all": list(ABLATION_CONFIGS.keys()),
    "quick": ["full_model", "no_fass", "fass_no_b", "fass_no_c", "fass_no_delta"],
}


def run_experiment(config_name: str, seed: int, args: argparse.Namespace, dry_run: bool = False):
    """Run a single experiment configuration."""
    config = ABLATION_CONFIGS[config_name]

    # Build checkpoint path
    ckpt_dir = Path(args.output_dir) / config_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"seed_{seed}.pth"

    # Build command
    cmd = [
        "python", "train.py",
        "--data_path", args.data_path,
        "--data_path_test", args.data_path_test or args.data_path,
        "--model_file", str(ckpt_path),
        "--num_iter", str(args.num_iter),
        "--batch_size", str(args.batch_size),
        "--seed", str(seed),
    ]

    # Add config-specific args
    cmd.extend(config["args"].split())

    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"[DRY RUN] {config_name} (seed={seed}):")
        print(f"  {cmd_str}")
        return None

    print(f"\n{'='*70}")
    print(f"Running: {config_name} | Seed: {seed}")
    print(f"Description: {config['desc']}")
    print(f"Command: {cmd_str}")
    print(f"{'='*70}\n")

    # Run experiment
    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    end_time = datetime.now()

    # Log result
    log_entry = {
        "config": config_name,
        "seed": seed,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_minutes": (end_time - start_time).total_seconds() / 60,
        "return_code": result.returncode,
        "checkpoint": str(ckpt_path),
    }

    # Append to log file
    log_file = Path(args.output_dir) / "ablation_log.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return log_entry


def main():
    parser = argparse.ArgumentParser(description="WaveSSM-X Ablation Runner")

    parser.add_argument("--data_path", type=str, default="./datasets/celeb",
                        help="Path to training data")
    parser.add_argument("--data_path_test", type=str, default=None,
                        help="Path to test data (default: same as data_path)")
    parser.add_argument("--output_dir", type=str, default="./ablation_results",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--num_iter", type=int, default=50000,
                        help="Training iterations per experiment")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                        help="Random seeds to run")
    parser.add_argument("--config", type=str, default="quick",
                        choices=list(EXPERIMENT_GROUPS.keys()) + list(ABLATION_CONFIGS.keys()),
                        help="Experiment group or single config to run")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--list", action="store_true",
                        help="List all available configurations")

    args = parser.parse_args()

    if args.list:
        print("\n=== Available Ablation Configurations ===\n")
        for group_name, configs in EXPERIMENT_GROUPS.items():
            print(f"\n[{group_name}]")
            for cfg in configs:
                desc = ABLATION_CONFIGS[cfg]["desc"]
                print(f"  {cfg:25s} - {desc}")
        return

    # Determine which configs to run
    if args.config in EXPERIMENT_GROUPS:
        configs_to_run = EXPERIMENT_GROUPS[args.config]
    else:
        configs_to_run = [args.config]

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Summary
    total_runs = len(configs_to_run) * len(args.seeds)
    print(f"\n{'='*70}")
    print(f"WaveSSM-X Ablation Study")
    print(f"{'='*70}")
    print(f"Configurations: {len(configs_to_run)}")
    print(f"Seeds: {args.seeds}")
    print(f"Total runs: {total_runs}")
    print(f"Iterations per run: {args.num_iter}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}\n")

    if args.dry_run:
        print("[DRY RUN MODE - No experiments will be executed]\n")

    # Run experiments
    results = []
    for config_name in configs_to_run:
        for seed in args.seeds:
            result = run_experiment(config_name, seed, args, dry_run=args.dry_run)
            if result:
                results.append(result)

    if not args.dry_run and results:
        print(f"\n{'='*70}")
        print(f"Ablation study complete!")
        print(f"Results logged to: {args.output_dir}/ablation_log.jsonl")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
