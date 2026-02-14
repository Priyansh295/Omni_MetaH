"""
WaveSSM-X Training Script - STABILITY FIXED VERSION
====================================================
Critical fixes applied:
1. Lower default LR (1e-4 instead of 2e-4)
2. Tighter gradient clipping (0.25 instead of 0.5)
3. Gradual perceptual loss ramp-up (prevents iter 1000 explosion)
4. Additional NaN detection and recovery
5. All v9 stability features included

Changes from original marked with: # FIXED:
"""
import os
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


# ═══════════════════════════════════════════════════════════
# v9 Stability Functions
# ═══════════════════════════════════════════════════════════

def sanitize_model_bn_stats(model, verbose=True):
    """Reset corrupted BatchNorm running statistics to safe defaults."""
    corrupted_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if hasattr(module, 'running_mean') and module.running_mean is not None:
                if not torch.isfinite(module.running_mean).all():
                    if verbose:
                        print(f"  [SANITIZE] Resetting corrupted running_mean: {name}")
                    module.running_mean.zero_()
                    corrupted_count += 1
            
            if hasattr(module, 'running_var') and module.running_var is not None:
                if not torch.isfinite(module.running_var).all():
                    if verbose:
                        print(f"  [SANITIZE] Resetting corrupted running_var: {name}")
                    module.running_var.fill_(1.0)
                    corrupted_count += 1
    
    return corrupted_count


def check_model_health(model, check_weights=False):
    """Check model for NaN/Inf in parameters and BN statistics."""
    issues = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if hasattr(module, 'running_mean') and module.running_mean is not None:
                if not torch.isfinite(module.running_mean).all():
                    issues.append(f"BN running_mean: {name}")
            
            if hasattr(module, 'running_var') and module.running_var is not None:
                if not torch.isfinite(module.running_var).all():
                    issues.append(f"BN running_var: {name}")
    
    if check_weights:
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                issues.append(f"Weight: {name}")
    
    is_healthy = len(issues) == 0
    return is_healthy, issues


def configure_bn_for_stability(model, momentum=0.01, eps=1e-3):
    """Configure all BatchNorm layers for better numerical stability."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.momentum = momentum
            module.eps = eps


class DWALossUpdater:
    """Dynamic Weight Averaging for multi-task loss balancing."""
    def __init__(self, num_losses=5, temp=2.0):
        self.num_losses = num_losses
        self.temp = temp
        self.loss_history = [] 
        self.weights = torch.ones(num_losses).to(device) / num_losses
        
    def update(self, current_losses):
        self.loss_history.append(current_losses)
        if len(self.loss_history) < 3:
            return self.weights
            
        prev = self.loss_history[-2]
        curr = self.loss_history[-1]
        
        r_k = []
        for c, p in zip(curr, prev):
            p_val = max(abs(p), 1e-6)
            r_k.append(c / p_val)
            
        r_k = torch.tensor(r_k).to(device)
        r_k = r_k.clamp(-5.0, 5.0)
        
        exp_vals = torch.exp(r_k / self.temp)
        sum_exp = torch.sum(exp_vals)
        self.weights = (exp_vals / sum_exp) * self.num_losses
        
        if torch.isnan(self.weights).any() or torch.isinf(self.weights).any():
            self.weights = torch.ones(self.num_losses).to(device)
        
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-50:]
        
        return self.weights


def train_and_evaluate(args):
    global scaler

    print(f"Device: {device}")
    
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
    
    # Full loss weights from config
    full_weights = {
        'l1': args.loss_weights[0],
        'perceptual': args.loss_weights[1],
        'ssim': args.loss_weights[2],
        'edge': args.loss_weights[3] if len(args.loss_weights) > 3 else 0.1,
        'freq': args.loss_weights[4] if len(args.loss_weights) > 4 else 0.3
    }
    
    # Data
    print(f"Preparing data from {args.data_path}...")
    
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

    # Model
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
    
    # Configure BN for stability
    print("  Configuring BatchNorm layers for numerical stability...")
    configure_bn_for_stability(model, momentum=0.01, eps=1e-3)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total | {trainable_params:,} trainable")

    # Loss & Optimizer
    criterion = WaveSSMLoss(weights=full_weights).to(device)
    
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
    
    warmup_iters = min(1000, args.num_iter // 30)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter - warmup_iters, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])
    
    # Resume from checkpoint
    start_iter = 0
    best_psnr = 0.0
    best_ssim_val = 0.0
    best_val_loss = float('inf')
    loss_history = []
    val_history = []
    
    checkpoint_path = args.model_file
    if args.resume:
        ckpt = checkpoint_path if os.path.exists(checkpoint_path) else find_latest_checkpoint(
            os.path.dirname(checkpoint_path)
        )
        
        if ckpt:
            print(f"Resuming from {ckpt}...")
            meta = load_checkpoint(ckpt, model, optimizer, device=device)
            start_iter = meta['iteration']
            best_psnr = meta['best_psnr']
            best_ssim_val = meta['best_ssim']
            best_val_loss = meta['best_val_loss']
            loss_history = meta['loss_history']
            val_history = meta['val_history']
            if meta.get('extra') and 'scaler_state_dict' in meta['extra']:
                scaler.load_state_dict(meta['extra']['scaler_state_dict'])
                current_scale = scaler.get_scale()
                if current_scale < 1.0 or current_scale > 1e10:
                     print(f"  [WARN] Abnormal scale {current_scale}. Creating fresh GradScaler.")
                     scaler = torch.cuda.amp.GradScaler(init_scale=4096.0, enabled=torch.cuda.is_available())
            
            # Post-load health check
            print("  Performing post-load model health check...")
            is_healthy, issues = check_model_health(model, check_weights=False)
            if not is_healthy:
                print(f"  [WARN] Found {len(issues)} corrupted stats after loading!")
                corrupted = sanitize_model_bn_stats(model, verbose=True)
                print(f"  Sanitized {corrupted} BatchNorm statistics.")
            else:
                print("  Model health: OK")
            
            # Rebuild scheduler
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter - warmup_iters, eta_min=1e-6)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])
            for _ in range(start_iter):
                scheduler.step()
            
            resumed_lr = optimizer.param_groups[0]['lr']
            print(f"  Resumed at iter {start_iter} | Best PSNR: {best_psnr:.2f} | Best SSIM: {best_ssim_val:.4f} | LR: {resumed_lr:.2e}")
        else:
            print("No checkpoint found, starting fresh.")
    
    # Training Loop
    print(f"\nStarting training: iters {start_iter} -> {args.num_iter}")
    print(f"  Batch size: {args.batch_size} | LR: {args.lr}")
    print(f"  Full loss weights: {full_weights}")
    print(f"  Logging to: {perf.csv_path}")
    print(f"  Visualizations: {vis_dir}")
    print()
    
    # Pre-training health check
    print("Pre-training model health check...")
    is_healthy, issues = check_model_health(model, check_weights=False)
    if is_healthy:
        print("  Model health: OK")
    else:
        print(f"  [WARN] Found {len(issues)} issues before training!")
        for issue in issues[:5]:
            print(f"    - {issue}")
        corrupted = sanitize_model_bn_stats(model, verbose=False)
        print(f"  Sanitized {corrupted} stats.")
    
    model.train()
    iter_train_loader = iter(train_loader)
    metrics = MetricsTracker()
    dwa_updater = DWALossUpdater(num_losses=5)
    
    last_rain = None
    last_norain = None
    
    pbar = tqdm(range(start_iter, args.num_iter), initial=start_iter, total=args.num_iter)
    
    for n_iter in pbar:
        perf.iter_start()
        
        try:
            # Get batch
            try:
                rain, norain, name, h, w = next(iter_train_loader)
            except StopIteration:
                iter_train_loader = iter(train_loader)
                rain, norain, name, h, w = next(iter_train_loader)
            
            rain = rain.to(device, non_blocking=True)
            norain = norain.to(device, non_blocking=True)
            
            model.train()
            
            # Input check
            if not (torch.isfinite(rain).all() and torch.isfinite(norain).all()):
                print(f"[NaN INPUT] iter {n_iter} skipping batch")
                continue

            # ══════════════════════════════════════════════════════════
            # FIXED: Gradual perceptual loss ramp-up (prevents explosion)
            # ══════════════════════════════════════════════════════════
            if n_iter < 1000:
                # L1 only warmup
                criterion.update_weights({'l1': 1.0, 'perceptual': 0.0, 'ssim': 0.0, 'edge': 0.0, 'freq': 0.0})
            elif n_iter < 3000:
                # GRADUAL ramp-up (0.01 → 0.1 over 2000 iters)
                t = (n_iter - 1000) / 2000.0
                criterion.update_weights({
                    'l1': 1.0, 
                    'perceptual': 0.01 + t * 0.09,  # 0.01 → 0.1
                    'ssim': 0.01 + t * 0.09,        # 0.01 → 0.1
                    'edge': 0.0, 
                    'freq': 0.0
                })
            elif n_iter < 5000:
                # Continue ramp to full weights
                t = (n_iter - 3000) / 2000.0
                criterion.update_weights({
                    'l1': full_weights['l1'],
                    'perceptual': 0.1 + t * (full_weights['perceptual'] - 0.1),
                    'ssim': 0.1 + t * (full_weights['ssim'] - 0.1),
                    'edge': t * full_weights['edge'],
                    'freq': t * full_weights['freq']
                })
            else:
                # Full weights
                criterion.update_weights(full_weights)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                out = model(rain)
                
                # Output check
                if not torch.isfinite(out).all():
                    print(f"[NaN OUTPUT] iter {n_iter} skipping batch")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss, loss_dict = criterion(out, norain, return_dict=True)
                
                # Strict loss check
                is_loss_finite = True
                for k, v in loss_dict.items():
                    if not math.isfinite(v):
                        is_loss_finite = False
                        break
                if not is_loss_finite or not torch.isfinite(loss):
                     print(f"[NaN SUBLOSS] iter {n_iter}: sub-loss non-finite, skipping batch")
                     optimizer.zero_grad(set_to_none=True)
                     scaler.update()
                     continue

            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # ══════════════════════════════════════════════════════════
            # FIXED: Tighter gradient clipping (0.25 instead of 0.5)
            # ══════════════════════════════════════════════════════════
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            
            if not torch.isfinite(grad_norm):
                print(f"[NaN GRAD] iter {n_iter}: grad_norm={grad_norm:.2f}, skipping step")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            iter_time = perf.iter_end()
            
            # Track
            loss_val = loss.item()
            loss_history.append(loss_val)
            if len(loss_history) > 5000:
                loss_history = loss_history[-5000:]
            metrics.update({'total_loss': loss_val, **loss_dict})
            
            last_rain = rain.detach()
            last_norain = norain.detach()
            
            # DWA update
            if n_iter > 5000 and n_iter % 50 == 0:
                dwa_losses = [
                    loss_dict.get('l1', 0),
                    loss_dict.get('perceptual', 0),
                    loss_dict.get('ssim', 0),
                    loss_dict.get('edge', 0),
                    loss_dict.get('freq', 0)
                ]
                dwa_weights = dwa_updater.update(dwa_losses)
                keys = ['l1', 'perceptual', 'ssim', 'edge', 'freq']
                new_w = {}
                for i, k in enumerate(keys):
                    new_w[k] = full_weights[k] * dwa_weights[i].item()
                criterion.update_weights(new_w)
            
            # Periodic health checks (every 100 iters)
            if n_iter > 0 and n_iter % 100 == 0:
                is_healthy, issues = check_model_health(model, check_weights=False)
                if not is_healthy:
                    print(f"\n[HEALTH CHECK] iter {n_iter}: Found {len(issues)} corrupted stats!")
                    for issue in issues[:5]:
                        print(f"  - {issue}")
                    corrupted = sanitize_model_bn_stats(model, verbose=False)
                    print(f"  Auto-sanitized {corrupted} BatchNorm statistics.\n")
            
            # Progress bar (every 10 iters)
            if n_iter % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                remaining = args.num_iter - n_iter
                eta = perf.eta_str(remaining)
                pbar.set_description(
                    f"Loss:{loss_val:.4f} L1:{loss_dict.get('l1',0):.3f} "
                    f"SSIM:{loss_dict.get('ssim',0):.3f} LR:{current_lr:.1e} ETA:{eta}"
                )
            
            # CSV logging (every 10 iters)
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
            
            # Validation + Checkpoint
            if n_iter > 0 and n_iter % args.val_every == 0:
                # Pre-validation sanitization
                print(f"\n[Pre-validation sanitization at iter {n_iter}]")
                corrupted = sanitize_model_bn_stats(model, verbose=True)
                if corrupted > 0:
                    print(f"  Sanitized {corrupted} corrupted stats before validation.")
                
                val_result = validate(model, val_loader, criterion, device)
                val_history.append({
                    'iteration': n_iter, 
                    **val_result
                })
                
                print(f"\n[Iter {n_iter}] Val Loss: {val_result['val_loss']:.4f} | "
                      f"PSNR: {val_result['psnr']:.2f} | SSIM: {val_result['ssim']:.4f} | "
                      f"{mem.summary()}")
                
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
                
                save_checkpoint(
                    filepath=checkpoint_path,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    iteration=n_iter, best_psnr=best_psnr, best_ssim=best_ssim_val,
                    best_val_loss=best_val_loss, loss_history=loss_history,
                    val_history=val_history, config=config_dict, extra=extra
                )
                
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
                
                model.train()
            
            # Visualization (every 5000 iters)
            if n_iter > 0 and n_iter % 5000 == 0:
                try:
                    plot_training_curves(perf.csv_path, vis_dir, title=f'WaveSSM-X Training (iter {n_iter})')
                    if val_history:
                        plot_validation_metrics(val_history, vis_dir)
                    
                    if last_rain is not None:
                        model.eval()
                        try:
                            with torch.no_grad():
                                sample_out = model(last_rain[:4])
                            save_sample_grid(
                                last_rain[:4], sample_out, last_norain[:4],
                                os.path.join(vis_dir, f'samples_iter{n_iter}.png'),
                                title=f'Iteration {n_iter}'
                            )
                        finally:
                            model.train()
                except Exception as e:
                    print(f"  [Viz warning] {e}")
                    model.train()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[OOM] Iter {n_iter}: Clearing cache, skipping batch...")
                mem.safe_clear()
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            else:
                raise e
    
    # Final save
    print("\nTraining complete! Saving final state...")
    
    config_dict = {k: str(v) for k, v in vars(args).items()}
    save_checkpoint(
        filepath=checkpoint_path,
        model=model, optimizer=optimizer, scheduler=scheduler,
        iteration=args.num_iter, best_psnr=best_psnr, best_ssim=best_ssim_val,
        best_val_loss=best_val_loss, loss_history=loss_history,
        val_history=val_history, config=config_dict
    )
    
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
    """Run validation and compute PSNR + SSIM metrics."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    failed_batches = 0
    
    with torch.no_grad():
        for i, (rain, norain, name, h, w) in enumerate(val_loader):
            if i >= max_samples:
                break
            
            try:
                rain = rain.to(device, non_blocking=True)
                norain = norain.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    out = model(rain)
                    loss = criterion(out, norain)
                
                out_clamped = out.float().clamp(0, 1)
                norain_clamped = norain.float().clamp(0, 1)
                
                loss_val = loss.item()
                if math.isnan(loss_val) or math.isinf(loss_val):
                    failed_batches += 1
                    continue
                
                total_loss += loss_val
                total_psnr += psnr(out_clamped, norain_clamped, data_range=1.0).item()
                total_ssim += ssim(out_clamped, norain_clamped, data_range=1.0).item()
                count += 1
                
            except Exception as e:
                print(f"  [VAL ERROR] Batch {i} failed: {e}")
                failed_batches += 1
                continue
    
    if count == 0:
        print(f"  [WARN] Validation produced 0 valid batches! ({failed_batches} failed)")
    elif failed_batches > 0:
        print(f"  [INFO] {failed_batches} validation batches failed, {count} succeeded")
    
    n = max(count, 1)
    return {
        'val_loss': total_loss / n,
        'psnr': total_psnr / n,
        'ssim': total_ssim / n
    }


if __name__ == "__main__":
    config = parse_args()
    train_and_evaluate(config)