"""
Robust checkpointing: save/restore full training state to resume mid-training.
Saves model, optimizer, scheduler, iteration, best metrics, loss history, and config.
"""
import os
import torch
import json
from datetime import datetime


def save_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    iteration: int = 0,
    epoch: int = 0,
    best_psnr: float = 0.0,
    best_ssim: float = 0.0,
    best_val_loss: float = float('inf'),
    loss_history: list = None,
    val_history: list = None,
    config: dict = None,
    extra: dict = None
):
    """
    Save a full training checkpoint.
    
    Args:
        filepath: Where to save (.pth)
        model: The model
        optimizer: The optimizer
        scheduler: LR scheduler (optional)
        iteration: Current training iteration
        epoch: Current epoch
        best_psnr: Best validation PSNR so far
        best_ssim: Best validation SSIM so far
        best_val_loss: Best validation loss so far
        loss_history: List of training loss values
        val_history: List of validation metric dicts
        config: Training config dict for reproducibility
        extra: Any additional state to save
    """
    
    # Safety Check: Scan for NaNs in model state
    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
             print(f"[FATAL] Checkpoint save aborted! Model corrupted at key: {k}")
             return # Abort save to protect previous checkpoint
             
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'epoch': epoch,
        'best_psnr': best_psnr,
        'best_ssim': best_ssim,
        'best_val_loss': best_val_loss,
        'loss_history': loss_history or [],
        'val_history': val_history or [],
        'timestamp': datetime.now().isoformat(),
    }
    
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    
    if config is not None:
        state['config'] = config
    
    if extra is not None:
        state['extra'] = extra
    
    # Save to a temp file first, then rename for atomicity
    tmp_path = filepath + '.tmp'
    torch.save(state, tmp_path)
    if os.path.exists(filepath):
        os.replace(tmp_path, filepath)
    else:
        os.rename(tmp_path, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    device: torch.device = None,
    strict: bool = True
) -> dict:
    """
    Load a training checkpoint and restore all state.
    
    Args:
        filepath: Path to checkpoint .pth
        model: Model to load weights into
        optimizer: Optimizer to restore state (optional)
        scheduler: LR scheduler to restore (optional)
        device: Device to map tensors to
        strict: Whether to strictly enforce state_dict key matching
    
    Returns:
        Dict with restored metadata: iteration, best_psnr, best_ssim, 
        best_val_loss, loss_history, val_history, config, extra
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    map_location = device or torch.device('cpu')
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    
    # Model
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Optimizer
    # Optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            print(f"  [WARN] Optimizer state mismatch (likely new param groups): {e}")
            print("  Skipping optimizer load. Starting with fresh optimizer state.")
    
    # Scheduler
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
             print(f"  [WARN] Scheduler state mismatch: {e}")
             print("  Skipping scheduler load. Starting with fresh scheduler.")
    
    meta = {
        'iteration': checkpoint.get('iteration', 0),
        'epoch': checkpoint.get('epoch', 0),
        'best_psnr': checkpoint.get('best_psnr', 0.0),
        'best_ssim': checkpoint.get('best_ssim', 0.0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'loss_history': checkpoint.get('loss_history', []),
        'val_history': checkpoint.get('val_history', []),
        'config': checkpoint.get('config', None),
        'extra': checkpoint.get('extra', None),
        'timestamp': checkpoint.get('timestamp', 'unknown'),
    }
    
    return meta


def find_latest_checkpoint(checkpoint_dir: str, prefix: str = 'wavessm_x') -> str:
    """
    Find the most recent checkpoint file in a directory.
    
    Returns:
        Path to latest checkpoint, or None if none found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    
    candidates = [
        os.path.join(checkpoint_dir, f) 
        for f in os.listdir(checkpoint_dir)
        if f.startswith(prefix) and f.endswith('.pth') and '_best' not in f
    ]
    
    if not candidates:
        return None
    
    # Return most recently modified
    return max(candidates, key=os.path.getmtime)
