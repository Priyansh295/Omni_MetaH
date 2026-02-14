"""
WaveSSM-X Training Script
=========================
Full training pipeline with:
  - Robust checkpointing (resume mid-training)
  - Metrics logging to CSV
  - Periodic visualization (loss curves, sample grids)
  - GPU memory monitoring
  - LR scheduler with warm restart support
  - OOM-safe training loop
"""
import os
print("DEBUG: Top of train.py", flush=True)
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import warnings

# Import from package
from wavessm_x.models.inpainting import Inpainting
from wavessm_x.losses.combined import WaveSSMLoss
from wavessm_x.data.dataset import TrainDataset, TestDataset
from wavessm_x.data.split import get_or_create_data_split
from wavessm_x.utils.config import parse_args
from wavessm_x.utils.metrics import psnr, ssim, MetricsTracker
from wavessm_x.utils.monitoring import PerformanceMonitor, MemoryMonitor
from wavessm_x.utils.checkpointing import save_checkpoint, load_checkpoint, find_latest_checkpoint
from wavessm_x.utils.visualization import (
    plot_training_curves, plot_validation_metrics, save_sample_grid
)

warnings.filterwarnings('ignore')


# ── 1. Setup ──────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


class DWALossUpdater:
    def __init__(self, num_losses=4, temp=2.0):
        self.num_losses = num_losses
        self.temp = temp
        self.loss_history = [] 
        self.weights = torch.ones(num_losses).to(device) / num_losses
        
    def update(self, current_losses):
        self.loss_history.append(current_losses)
        if len(self.loss_history) < 3:
            return self.weights
            
        # Calculate relative training rate r_k = L_k(t) / L_k(t-1)
        prev = self.loss_history[-2]
        curr = self.loss_history[-1]
        
        r_k = []
        for c, p in zip(curr, prev):
            p_val = max(abs(p), 1e-6)
            r_k.append(c / p_val)
            
        r_k = torch.tensor(r_k).to(device)
        # Clamp ratios to prevent exp overflow
        r_k = r_k.clamp(-5.0, 5.0)
        
        # Softmax normalization with temperature
        exp_vals = torch.exp(r_k / self.temp)
        sum_exp = torch.sum(exp_vals)
        self.weights = (exp_vals / sum_exp) * self.num_losses
        
        # Final NaN guard
        if torch.isnan(self.weights).any() or torch.isinf(self.weights).any():
            self.weights = torch.ones(self.num_losses).to(device)
        
        return self.weights

def train_and_evaluate(args):

    print(f"Device: {device}")
    
    # Enable TF32 for better performance/stability on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    log_dir = os.path.join(os.path.dirname(args.model_file), 'logs')
    vis_dir = os.path.join(os.path.dirname(args.model_file), 'visualizations')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    perf = PerformanceMonitor(log_dir=log_dir)
    mem = MemoryMonitor(device)
    
    # ── 2. Data ───────────────────────────────────────────────
    print(f"Preparing data from {args.data_path}...")
    
    # ── Data Loading ──
    try:
        train_inp, train_target, val_inp, val_target = get_or_create_data_split(
            args.data_path, val_split=0.1, seed=args.seed
        )
        if not train_inp:
             print("Error: No data found after split!", flush=True)
             return

        print(f"Data split: {len(train_inp)} train, {len(val_inp)} val", flush=True)

        train_dataset = TrainDataset(
            data_path=args.data_path,
            data_path_test=args.data_path_test,
            data_name=args.dataset_name,
            data_type='train',
            patch_size=256,
            use_advanced_aug=True,
            inp_files=train_inp,
            target_files=train_target
        )
        
        val_dataset = TrainDataset(
            data_path=args.data_path,
            data_path_test=args.data_path_test,
            data_name=args.dataset_name,
            data_type='val',
            patch_size=256,
            use_advanced_aug=False,
            inp_files=val_inp,
            target_files=val_target
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, 
            shuffle=True, num_workers=args.workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.workers, pin_memory=True
        )

    except Exception as e:
        print(f"Error creating data loaders: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # ── 3. Model ──────────────────────────────────────────────
    print("Initializing WaveSSM-X model...")
    model = Inpainting(
        use_mamba=args.use_mamba,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        use_fass=args.use_fass,
        use_ffc=args.use_ffc,
        wavelet=args.wavelet,
        fass_no_b=args.fass_no_b,
        fass_no_c=args.fass_no_c,
        fass_no_delta=args.fass_no_delta
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total | {trainable_params:,} trainable")

    # ── 4. Loss & Optimizer ───────────────────────────────────
    criterion = WaveSSMLoss(
        weights={
            'l1': args.loss_weights[0],
            'perceptual': args.loss_weights[1],
            'ssim': args.loss_weights[2],
            'edge': args.loss_weights[3] if len(args.loss_weights) > 3 else 0.1,
            'freq': args.loss_weights[4] if len(args.loss_weights) > 4 else 0.3
        }
    ).to(device)
    
    # ── 4. Optimization Setup ─────────────────────────────────
    # Split params: no weight decay for biases, norms, and scalars
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in n or "norm" in n or "scale" in n:
            no_decay.append(p)
        else:
            decay.append(p)
            
    optimizer = torch.optim.AdamW(
        [{'params': decay}, {'params': no_decay, 'weight_decay': 0.0}],
        lr=args.lr, weight_decay=1e-4
    )
    
    warmup_iters = min(1000, args.num_iter // 30) # Warmup 1000 iters
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter - warmup_iters, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])
    
    # ── 5. Resume from Checkpoint ─────────────────────────────
    start_iter = 0
    best_psnr = 0.0
    best_ssim_val = 0.0
    best_val_loss = float('inf')
    loss_history = []
    val_history = []
    
    checkpoint_path = args.model_file
    if args.resume:
        # Try explicit path first, then auto-find latest
        ckpt = checkpoint_path if os.path.exists(checkpoint_path) else find_latest_checkpoint(
            os.path.dirname(checkpoint_path)
        )
        
        if ckpt:
            print(f"Resuming from {ckpt}...")
            meta = load_checkpoint(ckpt, model, optimizer, scheduler, device)
            start_iter = meta['iteration']
            best_psnr = meta['best_psnr']
            best_ssim_val = meta['best_ssim']
            best_val_loss = meta['best_val_loss']
            loss_history = meta['loss_history']
            val_history = meta['val_history']
            # Restore GradScaler state to avoid recalibration waste
            if meta.get('extra') and 'scaler_state_dict' in meta['extra']:
                scaler.load_state_dict(meta['extra']['scaler_state_dict'])
                # Safety Clamp: prevent extreme scaling from bad history
                current_scale = scaler.get_scale()
                if current_scale < 1.0 or current_scale > 1e10:
                     print(f"  [WARN] Abnormal scale {current_scale} in checkpoint. Resetting to 4096.")
                     # Accessing internal _scale for reset (safe on PyTorch < 2.4, verify for newer)
                     # Or just re-init if problematic. Assuming standard structure:
                     scaler._scale = torch.tensor(4096.0).to(device)
            print(f"  Resumed at iter {start_iter} | Best PSNR: {best_psnr:.2f} | Best SSIM: {best_ssim_val:.4f}")
        else:
            print("No checkpoint found, starting fresh.")
    
    # ── 6. Training Loop ──────────────────────────────────────
    print(f"\nStarting training: iters {start_iter} -> {args.num_iter}")
    print(f"  Batch size: {args.batch_size} | LR: {args.lr}")
    print(f"  Logging to: {perf.csv_path}")
    print(f"  Visualizations: {vis_dir}")
    print()
    
    model.train()
    iter_train_loader = iter(train_loader)
    metrics = MetricsTracker()
    dwa_updater = DWALossUpdater(num_losses=5) # Initialize DWA
    
    pbar = tqdm(range(start_iter, args.num_iter), initial=start_iter, total=args.num_iter)
    
    for n_iter in pbar:
        perf.iter_start()
        
        try:
            # Get batch (auto-restart loader)
            try:
                rain, norain, name, h, w = next(iter_train_loader)
            except StopIteration:
                iter_train_loader = iter(train_loader)
                rain, norain, name, h, w = next(iter_train_loader)
            
            rain = rain.to(device, non_blocking=True)
            norain = norain.to(device, non_blocking=True)
            
            # 1. Input Check
            if not (torch.isfinite(rain).all() and torch.isfinite(norain).all()):
                print(f"[NaN INPUT] iter {n_iter} skipping batch")
                continue

            # Progressive Warmup Weights
            if n_iter < 1000:
                # First 1k iters: L1 only
                criterion.update_weights({'l1': 1.0, 'perceptual': 0.0, 'ssim': 0.0, 'edge': 0.0, 'freq': 0.0})
            elif n_iter < 5000:
                # 1k-5k iters: Add mild perceptual/SSIM
                criterion.update_weights({'l1': 1.0, 'perceptual': 0.1, 'ssim': 0.1, 'edge': 0.0, 'freq': 0.0})
            # 5k+: Full weights (defaults or DWA)

            # DWA Update (every 10 iters, but DISABLED for first 5000 iters)
            if n_iter > 5000 and n_iter % 10 == 0:
                # ... DWA logic ...
                pass 

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                # Forward
                out = model(rain)
                
                # 2. Output Check
                if not torch.isfinite(out).all():
                    print(f"[NaN OUTPUT] iter {n_iter} skipping batch")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # 3. Compute Loss
                # Note: We assume criterion returns total_loss, loss_dict
                loss, loss_dict = criterion(out, norain, return_dict=True)
                
                # 4. Strict Loss Check
                is_loss_finite = True
                for k, v in loss_dict.items():
                    if not math.isfinite(v):
                        is_loss_finite = False
                        break
                if not is_loss_finite or not torch.isfinite(loss):
                     print(f"[NaN SUBLOSS] iter {n_iter}: sub-loss non-finite, skipping batch")
                     optimizer.zero_grad(set_to_none=True)
                     scaler.update() # shrink scale just in case
                     continue

            # Backward with AMP scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # 5. Gradient Check
            grad_valid = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_valid = False
                    break
            
            if not grad_valid:
                print(f"[NaN GRAD] iter {n_iter}: skipping step")
                optimizer.zero_grad(set_to_none=True)
                scaler.update() 
                continue

            # Clip and Step
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            iter_time = perf.iter_end()
            
            # Track (cap history at 5000 to prevent checkpoint bloat)
            loss_val = loss.item()
            loss_history.append(loss_val)
            if len(loss_history) > 5000:
                loss_history = loss_history[-5000:]
            metrics.update({'total_loss': loss_val, **loss_dict})
            
            # ── Progress bar (every 10 iters) ──
            if n_iter % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                remaining = args.num_iter - n_iter
                eta = perf.eta_str(remaining)
                pbar.set_description(
                    f"Loss:{loss_val:.4f} L1:{loss_dict.get('l1',0):.3f} "
                    f"SSIM:{loss_dict.get('ssim',0):.3f} LR:{current_lr:.1e} ETA:{eta}"
                )
            
            # ── CSV logging (every 10 iters) ──
            if n_iter % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                perf.log_to_csv(n_iter, {
                    'total_loss': loss_val,
                    'lr': current_lr,
                    'grad_norm': grad_norm.item(),
                    'iter_time': iter_time,
                    'gpu_mem_mb': mem.current_mb(),
                    **{f'loss_{k}': v for k, v in loss_dict.items() if isinstance(v, (int, float))}
                })
            
            # ── Validation + Checkpoint ──
            if n_iter > 0 and n_iter % args.val_every == 0:
                val_result = validate(model, val_loader, criterion, device)
                val_history.append({
                    'iteration': n_iter, 
                    **val_result
                })
                
                print(f"\n[Iter {n_iter}] Val Loss: {val_result['val_loss']:.4f} | "
                      f"PSNR: {val_result['psnr']:.2f} | SSIM: {val_result['ssim']:.4f} | "
                      f"{mem.summary()}")
                
                # Update best metrics FIRST (before saving checkpoint)
                is_best = False
                if val_result['psnr'] > best_psnr:
                    best_psnr = val_result['psnr']
                    is_best = True
                if val_result['ssim'] > best_ssim_val:
                    best_ssim_val = val_result['ssim']
                    is_best = True
                if val_result['val_loss'] < best_val_loss:
                    best_val_loss = val_result['val_loss']
                    is_best = True
                
                config_dict = {k: str(v) for k, v in vars(args).items()}
                extra = {'scaler_state_dict': scaler.state_dict()}
                
                # Save latest checkpoint (with correct best metrics)
                save_checkpoint(
                    filepath=checkpoint_path,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    iteration=n_iter, best_psnr=best_psnr, best_ssim=best_ssim_val,
                    best_val_loss=best_val_loss, loss_history=loss_history,
                    val_history=val_history, config=config_dict, extra=extra
                )
                
                # Save best checkpoint
                if is_best:
                    best_path = checkpoint_path.replace('.pth', '_best.pth')
                    save_checkpoint(
                        filepath=best_path,
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        iteration=n_iter, best_psnr=best_psnr, best_ssim=best_ssim_val,
                        best_val_loss=best_val_loss, loss_history=loss_history,
                        val_history=val_history, config=config_dict, extra=extra
                    )
                    print(f"  ★ New best! PSNR={best_psnr:.2f} SSIM={best_ssim_val:.4f}")
                
                model.train()  # Back to train mode
            
            # ── Visualization (every 5000 iters) ──
            if n_iter > 0 and n_iter % 5000 == 0:
                # Generate plots from CSV
                try:
                    plot_training_curves(perf.csv_path, vis_dir, title=f'WaveSSM-X Training (iter {n_iter})')
                    if val_history:
                        plot_validation_metrics(val_history, vis_dir)
                except Exception as e:
                    print(f"  [Viz warning] {e}")
                
                # Save sample grid
                model.eval()
                with torch.no_grad():
                    sample_out = model(rain[:4])
                save_sample_grid(
                    rain[:4], sample_out, norain[:4],
                    os.path.join(vis_dir, f'samples_iter{n_iter}.png'),
                    title=f'Iteration {n_iter}'
                )
                model.train()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[OOM] Iter {n_iter}: Clearing cache, skipping batch...")
                if 'out' in dir(): del out
                if 'rain' in dir(): del rain
                if 'norain' in dir(): del norain
                mem.safe_clear()
                optimizer.zero_grad(set_to_none=True)
                continue
            else:
                raise e
    
    # ── 7. Final Save & Plots ─────────────────────────────────
    print("\nTraining complete! Saving final state...")
    
    config_dict = {k: str(v) for k, v in vars(args).items()}
    save_checkpoint(
        filepath=checkpoint_path,
        model=model, optimizer=optimizer, scheduler=scheduler,
        iteration=args.num_iter, best_psnr=best_psnr, best_ssim=best_ssim_val,
        best_val_loss=best_val_loss, loss_history=loss_history,
        val_history=val_history, config=config_dict
    )
    
    # Final visualization
    try:
        plot_training_curves(perf.csv_path, vis_dir, title='WaveSSM-X Training (Final)')
        if val_history:
            plot_validation_metrics(val_history, vis_dir)
    except Exception as e:
        print(f"[Viz warning] {e}")
    
    elapsed = perf.total_elapsed()
    hrs, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)
    print(f"\nTotal training time: {hrs:02d}:{mins:02d}:{secs:02d}")
    print(f"Best PSNR: {best_psnr:.2f} dB | Best SSIM: {best_ssim_val:.4f}")
    print(f"Logs: {perf.csv_path}")
    print(f"Plots: {vis_dir}/")


def validate(model, val_loader, criterion, device, max_samples=50):
    """
    Run validation and compute PSNR + SSIM metrics.
    
    Returns:
        Dict with 'val_loss', 'psnr', 'ssim'
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    with torch.no_grad():
        for i, (rain, norain, name, h, w) in enumerate(val_loader):
            if i >= max_samples:
                break
                
            rain = rain.to(device, non_blocking=True)
            norain = norain.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                out = model(rain)
                loss = criterion(out, norain)
            
            # Clamp output for metric computation
            out_clamped = out.float().clamp(0, 1)
            norain_clamped = norain.float().clamp(0, 1)
            
            loss_val = loss.item()
            # Skip NaN batches (check on CPU, no extra GPU sync)
            if math.isnan(loss_val) or math.isinf(loss_val):
                continue
            
            total_loss += loss_val
            total_psnr += psnr(out_clamped, norain_clamped, data_range=1.0).item()
            total_ssim += ssim(out_clamped, norain_clamped, data_range=1.0).item()
            count += 1
    
    if count == 0:
        print("  [WARN] Validation set is empty! Returning zeros.")
    n = max(count, 1)
    return {
        'val_loss': total_loss / n,
        'psnr': total_psnr / n,
        'ssim': total_ssim / n
    }


if __name__ == "__main__":
    print("DEBUG: Inside main block", flush=True)
    config = parse_args()
    train_and_evaluate(config)
