"""
Training visualization: loss curves, metric plots, sample grids.
All plots are saved to disk — no display required.
"""
import os
import csv
import torch
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend (headless-safe)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False


def plot_training_curves(csv_path: str, save_dir: str, title: str = 'WaveSSM-X Training'):
    """
    Read the CSV log and plot loss curves + metric curves.
    Generates: training_curves.png, loss_breakdown.png
    
    Args:
        csv_path: Path to the metrics CSV written by PerformanceMonitor
        save_dir: Directory to save plots
        title: Plot title
    """
    if not PLT_AVAILABLE:
        print("matplotlib not available, skipping plots")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Parse CSV
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                try:
                    data[k].append(float(v))
                except ValueError:
                    data[k].append(v)
    
    if 'iteration' not in data:
        print(f"No 'iteration' column in {csv_path}")
        return
    
    iters = np.array(data['iteration'])
    
    # ---- Plot 1: Main Training Curve ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Total loss
    if 'total_loss' in data:
        vals = np.array(data['total_loss'])
        # Apply moving average for smoother curve
        window = min(50, len(vals) // 5 + 1)
        if window > 1:
            smoothed = np.convolve(vals, np.ones(window)/window, mode='valid')
            smooth_iters = iters[:len(smoothed)]
            axes[0].plot(smooth_iters, smoothed, 'b-', linewidth=1.5, label='Total Loss (smoothed)')
            axes[0].plot(iters, vals, 'b-', alpha=0.15, linewidth=0.5)
        else:
            axes[0].plot(iters, vals, 'b-', linewidth=1.0, label='Total Loss')
        axes[0].set_ylabel('Total Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in data:
        axes[1].plot(iters, np.array(data['lr']), 'r-', linewidth=1.0)
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_xlabel('Iteration')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ---- Plot 2: Loss Breakdown ----
    loss_keys = [k for k in data.keys() if k.startswith('loss_') or k in ('l1', 'perceptual', 'ssim', 'edge', 'freq', 'mask')]
    
    if loss_keys:
        n_losses = len(loss_keys)
        fig, axes = plt.subplots(min(n_losses, 3), max(1, (n_losses + 2) // 3), figsize=(14, 4 * min(n_losses, 3)))
        if n_losses == 1:
            axes = [axes]
        elif n_losses > 1:
            axes = axes.flat
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_losses))
        for idx, key in enumerate(loss_keys):
            if idx >= len(axes):
                break
            vals = np.array(data[key])
            axes[idx].plot(iters, vals, color=colors[idx], alpha=0.4, linewidth=0.5)
            # Smoothed
            window = min(50, len(vals) // 5 + 1)
            if window > 1:
                smoothed = np.convolve(vals, np.ones(window)/window, mode='valid')
                axes[idx].plot(iters[:len(smoothed)], smoothed, color=colors[idx], linewidth=1.5)
            axes[idx].set_title(key, fontsize=10)
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Loss Component Breakdown', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_breakdown.png'), dpi=150, bbox_inches='tight')
        plt.close()


def plot_validation_metrics(val_history: list, save_dir: str):
    """
    Plot validation metrics over training.
    
    Args:
        val_history: List of dicts with keys like 'iteration', 'psnr', 'ssim', 'val_loss'
        save_dir: Directory to save
    """
    if not PLT_AVAILABLE or not val_history:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    iters = [v.get('iteration', i) for i, v in enumerate(val_history)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = [
        ('psnr', 'PSNR (dB)', 'tab:blue'),
        ('ssim', 'SSIM', 'tab:green'),
        ('val_loss', 'Val Loss', 'tab:red'),
    ]
    
    for ax, (key, label, color) in zip(axes, metrics):
        vals = [v.get(key, 0) for v in val_history]
        if any(v != 0 for v in vals):
            ax.plot(iters, vals, 'o-', color=color, markersize=3, linewidth=1.5)
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.grid(True, alpha=0.3)
            
            # Mark best
            best_idx = np.argmax(vals) if key != 'val_loss' else np.argmin(vals)
            ax.axvline(x=iters[best_idx], color=color, linestyle='--', alpha=0.5)
            ax.annotate(f'Best: {vals[best_idx]:.4f}', 
                       xy=(iters[best_idx], vals[best_idx]),
                       fontsize=8, fontweight='bold')
    
    plt.suptitle('Validation Metrics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'validation_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_sample_grid(
    corrupted: torch.Tensor,
    output: torch.Tensor,
    target: torch.Tensor, 
    save_path: str,
    n_samples: int = 4,
    title: str = None
):
    """
    Save a grid showing [Corrupted | Output | Ground Truth] for visual inspection.
    
    Args:
        corrupted: (B, 3, H, W) input images
        output: (B, 3, H, W) model output
        target: (B, 3, H, W) ground truth
        save_path: Where to save the image
        n_samples: Number of rows
        title: Optional title
    """
    if not PLT_AVAILABLE:
        return
    
    n = min(n_samples, corrupted.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        for j, (img, label) in enumerate([
            (corrupted[i], 'Corrupted'),
            (output[i], 'Output'),
            (target[i], 'Ground Truth')
        ]):
            img_np = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            axes[i, j].imshow(img_np)
            axes[i, j].set_title(label if i == 0 else '', fontsize=10)
            axes[i, j].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_gpu_memory(csv_path: str, save_dir: str):
    """Plot GPU memory usage over training from CSV log."""
    if not PLT_AVAILABLE:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                try:
                    data[k].append(float(v))
                except ValueError:
                    data[k].append(v)
    
    if 'gpu_mem_mb' not in data:
        return
    
    fig, ax = plt.subplots(figsize=(10, 4))
    iters = np.array(data.get('iteration', list(range(len(data['gpu_mem_mb'])))))
    ax.plot(iters, np.array(data['gpu_mem_mb']), 'g-', linewidth=1.0)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('GPU Memory (MB)')
    ax.set_title('GPU Memory Usage', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gpu_memory.png'), dpi=150, bbox_inches='tight')
    plt.close()
