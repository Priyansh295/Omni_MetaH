import os
import glob
import argparse
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
import nevergrad as ng
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import random
import numpy as np
try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None
    print("Warning: ptflops not found. Complexity penalty will be disabled.")

from model_directional_query_od import Inpainting
from utils_train import OptimizedConfig, parse_args, TrainDataset, psnr, ssim, VGGPerceptualLoss, rgb_to_y
import kornia

# Import frequency-aware loss
try:
    from frequency_loss import FrequencyAwareLoss
    FREQ_LOSS_AVAILABLE = True
except ImportError:
    FREQ_LOSS_AVAILABLE = False
    print("Warning: frequency_loss not found. Frequency loss will be disabled.")

# Optimized parameters from previous hyperparameter optimization
OPTIMIZED_PARAMS = {
    'lr': 0.0008801771034220976,
    'num_blocks': [2, 4, 4, 6],
    'num_heads': [2, 2, 4, 8], 
    'channels': [24, 48, 96, 192],
    'num_refinement': 4,
    'expansion_factor': 2.7582489201175653,
    'loss_weights': {
        'w_l1': 0.4434123210862072,
        'w_percep': 0.19067420661049642,
        'w_ssim': 0.33932717524436795,
        'w_edge': 0.6021258595296576,
        'w_freq': 0.3  # NEW: Frequency-aware loss weight
    },
    # Mamba-specific parameters
    # mamba-ssm now installed with CUDA kernels - WG-SSM is fast!
    'use_mamba': True,  # ENABLED - Using WaveletGuidedSSM with optimized Mamba
    'd_state': 16,       # SSM state dimension
    'd_conv': 4,         # Local convolution kernel
    'expand': 2          # Mamba expansion factor
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
perceptual_loss = VGGPerceptualLoss().to(device)

# Initialize frequency loss (NEW)
frequency_loss = None
if FREQ_LOSS_AVAILABLE:
    frequency_loss = FrequencyAwareLoss(
        wavelet='db3',
        levels=1,
        weights={'LL': 1.0, 'LH': 1.5, 'HL': 1.5, 'HH': 2.0},
        loss_type='l1'
    ).to(device)
    print("Frequency-aware loss initialized successfully.")

class DWALossUpdater:
    def __init__(self, num_losses=5, temp=2.0):
        self.num_losses = num_losses
        self.temp = temp
        self.loss_history = []  # List of lists [l1, percep, ssim, edge, freq]
        self.weights = torch.ones(num_losses).to(device) / num_losses
        
    def update(self, current_losses):
        # current_losses is a list/tensor of [l1, percep, ssim, edge, freq]
        self.loss_history.append(current_losses)
        if len(self.loss_history) < 3:
            return self.weights
            
        # Calculate relative training rate r_k
        # r_k = L_k(t) / L_k(t-1)
        prev_losses = self.loss_history[-2]
        curr_losses = self.loss_history[-1]
        
        r_k = []
        for c, p in zip(curr_losses, prev_losses):
            # Avoid division by zero and handle small values
            p_val = p if p > 1e-6 else 1e-6
            r_k.append(c / p_val)
            
        r_k = torch.tensor(r_k).to(device)
        
        # Calculate weights using Softmax(r_k / T)
        # We want higher r_k (slower learning) to have higher weight? 
        # DWA paper: w_k = K * softmax(r_k / T)
        # Here we just normalize to sum to num_losses or 1? 
        # Usually sum to num_losses so scale matches.
        
        exp_vals = torch.exp(r_k / self.temp)
        sum_exp = torch.sum(exp_vals)
        self.weights = (exp_vals / sum_exp) * self.num_losses
        
        return self.weights

def test_loop(net, data_loader, num_iter):
    net.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for rain, norain, name, h, w in test_bar:
            rain, norain = rain.to(device), norain.to(device)
            out = torch.clamp((torch.clamp(net(rain)[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            norain = torch.clamp(norain[:, :, :h, :w].mul(255), 0, 255).byte()
            # computer the metrics with Y channel and double precision
            y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
            current_psnr, current_ssim = psnr(y, gt), ssim(y, gt)
            total_psnr += current_psnr.item()
            total_ssim += current_ssim.item()
            count += 1
            save_path = '{}/{}/{}'.format(args.save_path, args.data_name, name[0])
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()).save(save_path)
            test_bar.set_description('Test Iter: [{}/{}] PSNR: {:.2f} SSIM: {:.3f}'
                                    .format(num_iter, 1 if args.model_file else args.num_iter,
                                            total_psnr / count, total_ssim / count))
    return total_psnr / count, total_ssim / count


def save_loop(net, data_loader, num_iter):
    global best_psnr, best_ssim
    val_psnr, val_ssim = test_loop(net, data_loader, num_iter)
    results['PSNR'].append('{:.2f}'.format(val_psnr))
    results['SSIM'].append('{:.3f}'.format(val_ssim))
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, (num_iter if args.model_file else num_iter // 1000) + 1))
    data_frame.to_csv('{}/{}.csv'.format(args.save_path, args.data_name), index_label='Iter', float_format='%.3f')
    if val_psnr > best_psnr and val_ssim > best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        with open('{}/{}.txt'.format(args.save_path, args.data_name), 'w') as f:
            f.write('Iter: {} PSNR:{:.2f} SSIM:{:.3f}'.format(num_iter, best_psnr, best_ssim))
        torch.save(model.state_dict(), '{}/{}.pth'.format(args.save_path, args.data_name))


def save_checkpoint(model, optimizer, lr_scheduler, n_iter, best_psnr, best_ssim, 
                    results, stage_i, save_path, data_name):
    """Save full training checkpoint for resuming later"""
    checkpoint = {
        'iteration': n_iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'best_psnr': best_psnr,
        'best_ssim': best_ssim,
        'results': results,
        'stage_i': stage_i,
    }
    ckpt_path = os.path.join(save_path, f'{data_name}_checkpoint.pth')
    torch.save(checkpoint, ckpt_path)
    print(f"[Checkpoint] Saved at iteration {n_iter} to {ckpt_path}")


def load_checkpoint(model, optimizer, lr_scheduler, save_path, data_name, device):
    """Load training checkpoint to resume training"""
    ckpt_path = os.path.join(save_path, f'{data_name}_checkpoint.pth')
    if os.path.exists(ckpt_path):
        print(f"[Checkpoint] Found checkpoint at {ckpt_path}, loading...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print(f"[Checkpoint] Resumed from iteration {checkpoint['iteration']}")
        return (checkpoint['iteration'], checkpoint['best_psnr'], 
                checkpoint['best_ssim'], checkpoint['results'], 
                checkpoint['stage_i'])
    print(f"[Checkpoint] No checkpoint found at {ckpt_path}")
    return None


def apply_optimized_params(args):
    """Apply optimized parameters to args object"""
    print("Applying optimized parameters from previous hyperparameter optimization...")
    
    # Apply optimized architectural parameters
    args.num_blocks = OPTIMIZED_PARAMS['num_blocks']
    args.num_heads = OPTIMIZED_PARAMS['num_heads']
    args.channels = OPTIMIZED_PARAMS['channels']
    args.num_refinement = OPTIMIZED_PARAMS['num_refinement']
    args.expansion_factor = OPTIMIZED_PARAMS['expansion_factor']
    
    # Apply optimized training parameters
    args.lr = OPTIMIZED_PARAMS['lr']
    
    # Store loss weights for later use
    args.loss_weights = (
        OPTIMIZED_PARAMS['loss_weights']['w_l1'],
        OPTIMIZED_PARAMS['loss_weights']['w_percep'],
        OPTIMIZED_PARAMS['loss_weights']['w_ssim'],
        OPTIMIZED_PARAMS['loss_weights']['w_edge']
    )
    
    print(f"Applied optimized parameters:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Num blocks: {args.num_blocks}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Channels: {args.channels}")
    print(f"  Refinement layers: {args.num_refinement}")
    print(f"  Expansion factor: {args.expansion_factor}")
    print(f"  Loss weights: L1={args.loss_weights[0]:.3f}, Perceptual={args.loss_weights[1]:.3f}, SSIM={args.loss_weights[2]:.3f}, Edge={args.loss_weights[3]:.3f}")


_DATA_SPLIT_CACHE = {}

def get_or_create_data_split(data_path, val_split=0.2, seed=42):
    """
    Get or create a reproducible train/val split for the given data path.
    This ensures the SAME split is used across all optimization trials.
    """
    cache_key = (data_path, val_split, seed)
    if cache_key in _DATA_SPLIT_CACHE:
        return _DATA_SPLIT_CACHE[cache_key]

    inp_files = sorted(
        glob.glob(f'{data_path}/inp/*.png') +
        glob.glob(f'{data_path}/inp/*.jpg') +
        glob.glob(f'{data_path}/inp/*.jpeg') +
        glob.glob(f'{data_path}/inp/*.PNG') +
        glob.glob(f'{data_path}/inp/*.JPG') +
        glob.glob(f'{data_path}/input/*.png') +
        glob.glob(f'{data_path}/input/*.jpg') +
        glob.glob(f'{data_path}/input/*.jpeg') +
        glob.glob(f'{data_path}/input/*.PNG') +
        glob.glob(f'{data_path}/input/*.JPG')
    )
    target_files = sorted(
        glob.glob(f'{data_path}/target/*.png') +
        glob.glob(f'{data_path}/target/*.jpg') +
        glob.glob(f'{data_path}/target/*.jpeg') +
        glob.glob(f'{data_path}/target/*.PNG') +
        glob.glob(f'{data_path}/target/*.JPG') +
        glob.glob(f'{data_path}/gt/*.png') +
        glob.glob(f'{data_path}/gt/*.jpg') +
        glob.glob(f'{data_path}/gt/*.jpeg') +
        glob.glob(f'{data_path}/gt/*.PNG') +
        glob.glob(f'{data_path}/gt/*.JPG')
    )

    if len(inp_files) == 0 or len(target_files) == 0:
        return None, None, None, None

    min_len = min(len(inp_files), len(target_files))
    inp_files = inp_files[:min_len]
    target_files = target_files[:min_len]

    rng = random.Random(seed)
    indices = list(range(min_len))
    rng.shuffle(indices)
    split_idx = int(min_len * (1 - val_split))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_inp = [inp_files[i] for i in train_indices]
    train_target = [target_files[i] for i in train_indices]
    val_inp = [inp_files[i] for i in val_indices]
    val_target = [target_files[i] for i in val_indices]

    _DATA_SPLIT_CACHE[cache_key] = (train_inp, train_target, val_inp, val_target)
    print(f"[Data Split] Created split: {len(train_inp)} train, {len(val_inp)} val (seed={seed})")

    return train_inp, train_target, val_inp, val_target


def train_and_evaluate(num_blocks, num_heads, channels, lr, batch_size, expansion_factor, num_refinement, loss_weights,
                       kernel_size=3, act_type='gelu', norm_type='layernorm',
                       use_dwa=False, dwa_temp=2.0, val_split=0.2,
                       num_iter=100, data_path='./Blind_Omni_Wav_Net/datasets/celeb', data_path_test='./Blind_Omni_Wav_Net/datasets/celeb',
                       pre_split_data=None):
    """Updated train_and_evaluate with longer training iterations for more stable evaluation"""
    torch.cuda.empty_cache()
    try:
        if pre_split_data is not None:
            train_inp, train_target, val_inp, val_target = pre_split_data
        else:
            split_result = get_or_create_data_split(data_path, val_split)
            if split_result[0] is None:
                print(f"WARNING: No files found at {data_path}")
                return float('inf')
            train_inp, train_target, val_inp, val_target = split_result

        length = len(train_inp)

        if length == 0:
            return float('inf')
            
        # batch_size = min(batch_size, 1)  # Removed restriction for GPU training
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        perceptual_loss = VGGPerceptualLoss().to(device)
        
        # Get Mamba parameters from OPTIMIZED_PARAMS
        use_mamba = OPTIMIZED_PARAMS.get('use_mamba', False)
        d_state = OPTIMIZED_PARAMS.get('d_state', 16)
        d_conv = OPTIMIZED_PARAMS.get('d_conv', 4)
        expand = OPTIMIZED_PARAMS.get('expand', 2)
        
        model = Inpainting(num_blocks, num_heads, channels, num_refinement, expansion_factor, 
                          kernel_size, act_type, norm_type,
                          use_mamba=use_mamba, d_state=d_state, d_conv=d_conv, expand=expand).to(device)
        
        # Calculate Model Complexity (FLOPs) - skip if memory is tight
        flops_penalty = 0.0
        if get_model_complexity_info is not None:
            try:
                # Clear cache before FLOPs calculation
                torch.cuda.empty_cache()
                # Use smaller 128x128 input to avoid OOM
                with torch.no_grad():
                    macs, params = get_model_complexity_info(
                        model, (3, 128, 128), 
                        as_strings=False, 
                        print_per_layer_stat=False, 
                        verbose=False
                    )
                if macs is not None and params is not None:
                    gflops = macs / 1e9
                    flops_penalty = gflops * 0.05
                    print(f"Model Complexity: {gflops:.2f} GMACs, {params/1e6:.2f} M Params")
                # Clear cache after FLOPs calculation
                torch.cuda.empty_cache()
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"Skipping FLOPs calculation (OOM) - continuing training...")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Complexity calculation skipped: {type(e).__name__}")
        
        # Create datasets with explicit splits
        train_dataset = TrainDataset(data_path, data_path_test, 'inpaint', 'train', 128, length, inp_files=train_inp, target_files=train_target)
        val_dataset = TrainDataset(data_path, data_path_test, 'inpaint', 'val', 128, len(val_inp), inp_files=val_inp, target_files=val_target)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        if len(train_loader) == 0:
            return float('inf')
            
        # Initialize DWA if enabled - now with 5 losses
        dwa_updater = DWALossUpdater(num_losses=5, temp=dwa_temp) if use_dwa else None
        
        # Convert loss_weights dict to list for proper handling
        if isinstance(loss_weights, dict):
            w_l1 = loss_weights.get('w_l1', 0.44)
            w_percep = loss_weights.get('w_percep', 0.19)
            w_ssim = loss_weights.get('w_ssim', 0.34)
            w_edge = loss_weights.get('w_edge', 0.60)
            w_freq = loss_weights.get('w_freq', 0.30)
            current_weights = torch.tensor([w_l1, w_percep, w_ssim, w_edge, w_freq]).to(device)
        else:
            # Handle legacy list/tuple format (add default freq weight)
            current_weights = torch.tensor(list(loss_weights) + [0.3]).to(device)
        
        # Initialize optimizer
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        best_val_score = -float('inf') # Initialize best validation score
        
        model.train()
        for n_iter, (rain, norain, name, h, w) in enumerate(train_loader):
            if n_iter >= num_iter:
                break
            rain, norain = rain.to(device), norain.to(device)
            out = model(rain)
            ssim_loss = 1 - ssim(out, norain)
            edge_out = kornia.filters.sobel(out, normalized=True, eps=1e-06)
            edge_gt = kornia.filters.sobel(norain, normalized=True, eps=1e-06)
            edge_loss = F.l1_loss(edge_out, edge_gt)  # Fixed: removed [0] indexing
            
            # Calculate frequency loss
            freq_loss_val = torch.tensor(0.0).to(device)
            if frequency_loss is not None:
                freq_loss_val = frequency_loss(out, norain)
            
            if use_dwa and n_iter > 0 and n_iter % 10 == 0: # Update weights every 10 iters
                # Collect current loss values for DWA (now includes frequency loss)
                current_loss_vals = [
                    F.l1_loss(out, norain).item(),
                    perceptual_loss(out, norain).item(),
                    ssim_loss.item(),
                    edge_loss.item(),
                    freq_loss_val.item()  # NEW: include frequency loss
                ]
                current_weights = dwa_updater.update(current_loss_vals)
                
            # Unpack weights (now 5 weights)
            w_l1, w_percep, w_ssim, w_edge, w_freq = current_weights
            
            # Compute total loss with all 5 components
            loss = (F.l1_loss(out, norain) * w_l1 + 
                    perceptual_loss(out, norain) * w_percep + 
                    ssim_loss * w_ssim + 
                    edge_loss * w_edge +
                    freq_loss_val * w_freq)
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability (from best practices)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

                    # Validation on unseen data
            if n_iter > 0 and n_iter % 50 == 0:
                model.eval()
                with torch.no_grad():
                    try:
                        # Get a batch from validation loader
                        val_iter = iter(val_loader)
                        rain, norain, name, h, w = next(val_iter)
                    except StopIteration:
                        # Restart iterator if needed
                        val_iter = iter(val_loader)
                        rain, norain, name, h, w = next(val_iter)
                    except Exception:
                        # Fallback to training data if val fails (shouldn't happen)
                        rain, norain, name, h, w = next(iter(train_loader))
                        
                    rain, norain = rain.to(device), norain.to(device)
                    
                    # Request attention maps every 500 iterations (or at specific intervals)
                    if n_iter % 500 == 0:
                        out, attn = model(rain, return_attn=True)
                        
                        # Visualize Attention Map
                        if attn is not None:
                            # attn shape: [B, Heads, H*W, H*W] or similar depending on implementation
                            # For visualization, we usually average over heads and reshape
                            # Here we assume attn is [B, Heads, H*W, H*W] from MDTA
                            
                            # Take the first image in batch
                            attn_map = attn[0].mean(dim=0) # Average over heads -> [H*W, H*W]
                            # We want to see attention for a specific query pixel or average attention
                            # Let's visualize the average attention received by each pixel
                            attn_map = attn_map.mean(dim=0) # [H*W]
                            
                            # Reshape to image dimensions (H, W)
                            # Note: MDTA operates on downsampled features usually
                            # We need to know the feature map size. 
                            # Assuming standard 256x256 input and 4 levels, features might be 32x32 or 64x64
                            feat_h = int(np.sqrt(attn_map.shape[0]))
                            attn_img = attn_map.reshape(feat_h, feat_h).cpu().numpy()
                            
                            # Save visualization
                            plt.figure(figsize=(10, 5))
                            plt.subplot(1, 3, 1)
                            plt.imshow(rain[0].permute(1, 2, 0).cpu().numpy())
                            plt.title("Input")
                            plt.axis('off')
                            
                            plt.subplot(1, 3, 2)
                            plt.imshow(out[0].permute(1, 2, 0).cpu().numpy())
                            plt.title("Output")
                            plt.axis('off')
                            
                            plt.subplot(1, 3, 3)
                            plt.imshow(attn_img, cmap='jet')
                            plt.title("Attention Map")
                            plt.axis('off')
                            
                            os.makedirs(f'{data_path}/../visualizations', exist_ok=True)
                            plt.savefig(f'{data_path}/../visualizations/attn_{n_iter}.png')
                            plt.close()
                    else:
                        out = model(rain)
                    
                    # Calculate validation metrics
                    val_l1 = F.l1_loss(out, norain).item()
                    val_psnr = psnr(out, norain).item()
                    val_ssim = ssim(out, norain).item()
                    
                    # Use composite score for optimization (higher is better)
                    # Score = PSNR * 0.1 + SSIM * 10 - L1 * 10 - Complexity Penalty
                    current_score = (val_psnr * 0.1 + val_ssim * 10 - val_l1 * 10) - flops_penalty
                    
                    if current_score > best_val_score:
                        best_val_score = current_score
                        
                model.train()
        return -best_val_score # Return negative best_val_score for minimization
    except IndexError:
        print("IndexError: Empty dataset encountered in train_and_evaluate. Skipping trial.")
        return float('inf')
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"CUDA Out of Memory: {e}")
            torch.cuda.empty_cache()
            return float('inf')
        else:
            raise e
    finally:
        # Aggressive cleanup
        if 'model' in locals(): del model
        if 'optimizer' in locals(): del optimizer
        if 'dwa_updater' in locals(): del dwa_updater
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def run_improved_optimization(args):
    """Improved metaheuristic optimization with better exploration"""
    print("Running improved metaheuristic optimization...")

    GA_BUDGET = 25
    PSO_BUDGET = 25
    DE_BUDGET = 25
    BO_TRIALS = 25
    TRAINING_ITERS = 100

    print("[Data Split] Creating reproducible train/val split ONCE before optimization...")
    pre_split_data = get_or_create_data_split(args.data_path, val_split=0.2, seed=42)
    if pre_split_data[0] is None:
        print("ERROR: Could not create data split. Check data_path.")
        return None

    def clamp(val, minval, maxval):
        return max(minval, min(val, maxval))

    def inject_parameter_noise(params, noise_scale=0.1):
        """Inject noise for diverse seeding"""
        noisy_params = {}
        # Exclude discrete parameters from noise injection
        exclude = ['batch_size', 'kernel_size', 'act_type', 'norm_type']
        
        for key, value in params.items():
            if key in exclude:
                noisy_params[key] = value
            elif isinstance(value, (int, float)):
                noise = np.random.normal(0, abs(value) * noise_scale)
                noisy_params[key] = value + noise
            else:
                noisy_params[key] = value
        return noisy_params
    
    def create_diverse_architectures():
        """Create diverse architectural configurations"""
        architectures = [
            # Original optimized
            {'num_blocks': [2, 4, 4, 6], 'num_heads': [2, 2, 4, 8], 'channels': [24, 48, 96, 192]},
            # Smaller variants
            {'num_blocks': [1, 2, 3, 4], 'num_heads': [1, 1, 2, 4], 'channels': [16, 32, 64, 128]},
            {'num_blocks': [1, 3, 3, 5], 'num_heads': [1, 2, 2, 4], 'channels': [20, 40, 80, 160]},
            # Larger variants
            {'num_blocks': [3, 5, 5, 7], 'num_heads': [2, 4, 4, 8], 'channels': [32, 64, 128, 256]},
            {'num_blocks': [4, 6, 6, 8], 'num_heads': [2, 2, 4, 8], 'channels': [24, 48, 96, 192]},
            # Different ratios
            {'num_blocks': [2, 2, 4, 8], 'num_heads': [1, 2, 4, 8], 'channels': [24, 48, 96, 192]},
        ]
        return architectures

    # SOLUTION 2: Wider parameter ranges for better exploration
    # Note: Arrays must be initialized with values within bounds
    # FIXED: Bounds now accommodate OPTIMIZED_PARAMS values [2,4,4,6]
    instrum_ga = ng.p.Instrumentation(
        num_blocks=ng.p.Array(init=[2,4,4,6]).set_bounds(lower=[1,2,2,4], upper=[4,6,6,8]).set_integer_casting(),
        num_heads=ng.p.Array(init=[2,2,4,8]).set_bounds(lower=[1,1,2,4], upper=[4,4,8,8]).set_integer_casting(),
        channels=ng.p.Array(init=[24,48,96,192]).set_bounds(lower=[16,32,64,128], upper=[32,64,128,256]).set_integer_casting(),
        num_refinement=ng.p.Scalar(init=4, lower=2, upper=6).set_integer_casting(),
        # Micro-NAS parameters
        act_type=ng.p.Choice(['relu', 'gelu', 'silu']),
        norm_type=ng.p.Choice(['layernorm', 'instancenorm']),
        kernel_size=ng.p.Choice([3, 5]),
        batch_size=ng.p.Choice([1, 2, 4])  # Allow more batch sizes
    )
    
    instrum_pso = ng.p.Instrumentation(
        w_l1=ng.p.Scalar(lower=0.1, upper=1.0),      # Wider range
        w_percep=ng.p.Scalar(lower=0.05, upper=0.8), # Wider range
        w_ssim=ng.p.Scalar(lower=0.1, upper=1.0),    # Wider range
        w_edge=ng.p.Scalar(lower=0.1, upper=1.2),    # Wider range
        dwa_temp=ng.p.Scalar(lower=1.0, upper=5.0)   # DWA Temperature
    )
    
    instrum_de = ng.p.Instrumentation(
        expansion_factor=ng.p.Scalar(lower=1.2, upper=4.5)  # Wider range
    )

    # SOLUTION 3: Use CMA-ES for better exploration of GA parameters
    # SOLUTION 3: Use TwoPointsDE for discrete GA parameters, CMA for others
    ga_optimizer = ng.optimizers.TwoPointsDE(parametrization=instrum_ga, budget=GA_BUDGET)
    pso_optimizer = ng.optimizers.PSO(parametrization=instrum_pso, budget=PSO_BUDGET)
    de_optimizer = ng.optimizers.DE(parametrization=instrum_de, budget=DE_BUDGET)

    # SOLUTION 4: Multiple diverse seeds for GA with different noise levels
    ga_base_seed = {
        'num_blocks': OPTIMIZED_PARAMS['num_blocks'],
        'num_heads': OPTIMIZED_PARAMS['num_heads'],
        'channels': OPTIMIZED_PARAMS['channels'],
        'num_refinement': OPTIMIZED_PARAMS['num_refinement'],
        'act_type': 'gelu',
        'norm_type': 'layernorm',
        'kernel_size': 3,
        'batch_size': 8 # FIXED: Use batch_size=8 for T4 GPU optimizing
    }
    
    # Create diverse architectural seeds
    diverse_architectures = create_diverse_architectures()
    ga_seeds = []
    
    # Add original optimized
    ga_seeds.append(ga_base_seed)
    
    # Add diverse architectures
    for arch in diverse_architectures[:3]:  # Add 3 diverse architectures
        seed = {**ga_base_seed, **arch}
        ga_seeds.append(seed)
    
    # Add noisy versions with different noise levels
    for noise in [0.05, 0.1, 0.15, 0.2]:
        noisy_seed = inject_parameter_noise(ga_base_seed, noise)
        # Clamp refinement to valid range
        if 'num_refinement' in noisy_seed:
            noisy_seed['num_refinement'] = max(1, min(8, int(noisy_seed['num_refinement'])))
        ga_seeds.append(noisy_seed)
    
    # Seed GA optimizer with diverse initial points
    for i, seed in enumerate(ga_seeds[:min(len(ga_seeds), GA_BUDGET//2)]):
        try:
            print(f"Seeding GA with configuration {i+1}: {seed}")
            ga_candidate = instrum_ga.spawn_child(((), seed))
            ga_value = train_and_evaluate(
                seed['num_blocks'], seed['num_heads'], seed['channels'],
                OPTIMIZED_PARAMS['lr'], 1, OPTIMIZED_PARAMS['expansion_factor'],
                seed['num_refinement'],
                (OPTIMIZED_PARAMS['loss_weights']['w_l1'],
                 OPTIMIZED_PARAMS['loss_weights']['w_percep'],
                 OPTIMIZED_PARAMS['loss_weights']['w_ssim'],
                 OPTIMIZED_PARAMS['loss_weights']['w_edge']),
                kernel_size=seed.get('kernel_size', 3),
                act_type=seed.get('act_type', 'gelu'),
                norm_type=seed.get('norm_type', 'layernorm'),
                use_dwa=True, dwa_temp=2.0,
                num_iter=TRAINING_ITERS,
                data_path=args.data_path,
                data_path_test=args.data_path_test,
                pre_split_data=pre_split_data
            )
            ga_optimizer.tell(ga_candidate, ga_value)
            print(f"GA seed {i+1} evaluation: {ga_value}")
        except Exception as e:
            print(f"Failed to seed GA with {seed}: {e}")

    # SOLUTION 5: Multiple PSO seeds with diverse loss weight combinations
    pso_seeds = []
    
    # Original optimized
    pso_base_seed = {
        'w_l1': clamp(OPTIMIZED_PARAMS['loss_weights']['w_l1'], 0.1, 1.0),
        'w_percep': clamp(OPTIMIZED_PARAMS['loss_weights']['w_percep'], 0.05, 0.8),
        'w_ssim': clamp(OPTIMIZED_PARAMS['loss_weights']['w_ssim'], 0.1, 1.0),
        'w_edge': clamp(OPTIMIZED_PARAMS['loss_weights']['w_edge'], 0.1, 1.2),
        'dwa_temp': 2.0
    }
    pso_seeds.append(pso_base_seed)
    
    # Diverse loss weight combinations
    diverse_loss_configs = [
        {'w_l1': 0.8, 'w_percep': 0.3, 'w_ssim': 0.6, 'w_edge': 0.4, 'dwa_temp': 1.5},
        {'w_l1': 0.6, 'w_percep': 0.4, 'w_ssim': 0.8, 'w_edge': 0.5, 'dwa_temp': 2.5},
        {'w_l1': 0.5, 'w_percep': 0.6, 'w_ssim': 0.4, 'w_edge': 0.7, 'dwa_temp': 3.0},
        {'w_l1': 0.7, 'w_percep': 0.2, 'w_ssim': 0.9, 'w_edge': 0.3, 'dwa_temp': 4.0},
    ]
    
    for config in diverse_loss_configs:
        clamped_config = {k: clamp(v, 0.05 if k == 'w_percep' else (1.0 if k == 'dwa_temp' else 0.1), 
                                  0.8 if k == 'w_percep' else (5.0 if k == 'dwa_temp' else (1.2 if k == 'w_edge' else 1.0))) 
                         for k, v in config.items()}
        pso_seeds.append(clamped_config)
    
    # Add noisy versions
    for noise in [0.1, 0.2]:
        noisy_seed = inject_parameter_noise(pso_base_seed, noise)
        clamped_seed = {k: clamp(v, 0.05 if k == 'w_percep' else (1.0 if k == 'dwa_temp' else 0.1), 
                               0.8 if k == 'w_percep' else (5.0 if k == 'dwa_temp' else (1.2 if k == 'w_edge' else 1.0))) 
                       for k, v in noisy_seed.items()}
        pso_seeds.append(clamped_seed)
    
    # Seed PSO optimizer
    for i, seed in enumerate(pso_seeds[:min(len(pso_seeds), PSO_BUDGET//2)]):
        try:
            print(f"Seeding PSO with loss weights {i+1}: {seed}")
            pso_candidate = instrum_pso.spawn_child(((), seed))
            pso_seed_value = train_and_evaluate(
                OPTIMIZED_PARAMS['num_blocks'],
                OPTIMIZED_PARAMS['num_heads'],
                OPTIMIZED_PARAMS['channels'],
                OPTIMIZED_PARAMS['lr'],
                1,
                OPTIMIZED_PARAMS['expansion_factor'],
                OPTIMIZED_PARAMS['num_refinement'],
                (seed['w_l1'], seed['w_percep'], seed['w_ssim'], seed['w_edge']),
                kernel_size=3, act_type='gelu', norm_type='layernorm',
                use_dwa=True, dwa_temp=seed.get('dwa_temp', 2.0),
                num_iter=TRAINING_ITERS,
                data_path=args.data_path,
                data_path_test=args.data_path_test,
                pre_split_data=pre_split_data
            )
            pso_optimizer.tell(pso_candidate, pso_seed_value)
            print(f"PSO seed {i+1} evaluation: {pso_seed_value}")
        except Exception as e:
            print(f"Failed to seed PSO with {seed}: {e}")

    # SOLUTION 6: Multiple DE seeds with diverse expansion factors
    de_seeds = [
        {'expansion_factor': clamp(OPTIMIZED_PARAMS['expansion_factor'], 1.2, 4.5)},
        {'expansion_factor': 2.0},
        {'expansion_factor': 3.0},
        {'expansion_factor': 1.5},
        {'expansion_factor': 3.5},
    ]
    
    # Add noisy versions
    for noise in [0.1, 0.2, 0.3]:
        base_factor = OPTIMIZED_PARAMS['expansion_factor']
        noisy_factor = base_factor + np.random.normal(0, abs(base_factor) * noise)
        de_seeds.append({'expansion_factor': clamp(noisy_factor, 1.2, 4.5)})
    
    # Seed DE optimizer
    for i, seed in enumerate(de_seeds[:min(len(de_seeds), DE_BUDGET//2)]):
        try:
            print(f"Seeding DE with expansion factor {i+1}: {seed}")
            de_candidate = instrum_de.spawn_child(((), seed))
            de_seed_value = train_and_evaluate(
                OPTIMIZED_PARAMS['num_blocks'],
                OPTIMIZED_PARAMS['num_heads'],
                OPTIMIZED_PARAMS['channels'],
                OPTIMIZED_PARAMS['lr'],
                1,
                seed['expansion_factor'],
                OPTIMIZED_PARAMS['num_refinement'],
                (OPTIMIZED_PARAMS['loss_weights']['w_l1'],
                 OPTIMIZED_PARAMS['loss_weights']['w_percep'],
                 OPTIMIZED_PARAMS['loss_weights']['w_ssim'],
                 OPTIMIZED_PARAMS['loss_weights']['w_edge']),
                kernel_size=3, act_type='gelu', norm_type='layernorm',
                use_dwa=True, dwa_temp=2.0,
                num_iter=TRAINING_ITERS,
                data_path=args.data_path,
                data_path_test=args.data_path_test,
                pre_split_data=pre_split_data
            )
            de_optimizer.tell(de_candidate, de_seed_value)
            print(f"DE seed {i+1} evaluation: {de_seed_value}")
        except Exception as e:
            print(f"Failed to seed DE with {seed}: {e}")

    # SOLUTION 7: Improved BO with more trials and better parameter exploration
    def bo_objective(trial):
        lr = trial.suggest_float('lr', 5e-5, 5e-3, log=True)  # Wider range
        batch_size = trial.suggest_categorical('batch_size', [1, 2, 4])
        num_iter = TRAINING_ITERS
        
        # Get current best recommendations from optimizers
        best_ga = ga_optimizer.provide_recommendation().value
        best_pso = pso_optimizer.provide_recommendation().value
        best_de = de_optimizer.provide_recommendation().value
        
        # Extract GA parameters
        if best_ga and isinstance(best_ga, tuple) and len(best_ga) > 1 and isinstance(best_ga[1], dict):
            params_ga = best_ga[1]
            num_blocks = params_ga['num_blocks']
            num_heads = params_ga['num_heads']
            channels = params_ga['channels']
            num_refinement = params_ga['num_refinement']
            act_type = params_ga.get('act_type', 'gelu')
            norm_type = params_ga.get('norm_type', 'layernorm')
            kernel_size = params_ga.get('kernel_size', 3)
        else:
            num_blocks = OPTIMIZED_PARAMS['num_blocks']
            num_heads = OPTIMIZED_PARAMS['num_heads']
            channels = OPTIMIZED_PARAMS['channels']
            num_refinement = OPTIMIZED_PARAMS['num_refinement']
            act_type = 'gelu'
            norm_type = 'layernorm'
            kernel_size = 3
        
        # Extract PSO parameters
        if best_pso and isinstance(best_pso, tuple) and len(best_pso) > 1 and isinstance(best_pso[1], dict):
            params_pso = best_pso[1]
            loss_weights = (
                params_pso['w_l1'],
                params_pso['w_percep'],
                params_pso['w_ssim'],
                params_pso['w_edge']
            )
            dwa_temp = params_pso.get('dwa_temp', 2.0)
        else:
            loss_weights = (
                OPTIMIZED_PARAMS['loss_weights']['w_l1'],
                OPTIMIZED_PARAMS['loss_weights']['w_percep'],
                OPTIMIZED_PARAMS['loss_weights']['w_ssim'],
                OPTIMIZED_PARAMS['loss_weights']['w_edge']
            )
            dwa_temp = 2.0
        
        # Extract DE parameters
        if best_de and isinstance(best_de, tuple) and len(best_de) > 1 and isinstance(best_de[1], dict):
            params_de = best_de[1]
            expansion_factor = params_de['expansion_factor']
        else:
            expansion_factor = OPTIMIZED_PARAMS['expansion_factor']
        
        return train_and_evaluate(num_blocks, num_heads, channels, lr, batch_size,
                                expansion_factor, num_refinement, loss_weights,
                                kernel_size=kernel_size, act_type=act_type, norm_type=norm_type,
                                use_dwa=True, dwa_temp=dwa_temp,
                                num_iter=num_iter, data_path=args.data_path,
                                data_path_test=args.data_path_test,
                                pre_split_data=pre_split_data)

    study = optuna.create_study(direction='minimize')
    study.optimize(bo_objective, n_trials=BO_TRIALS)
    print('Best BO trial:', study.best_trial.params)
    print('Best BO value:', study.best_trial.value)

    # Visualization of hyperparameter optimization
    trials_df = study.trials_dataframe()
    plot_dir = os.path.join(args.save_path, 'hyperparam_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    for col in trials_df.columns:
        if col.startswith('params_'):
            plt.figure(figsize=(8, 5))
            scatter = sns.scatterplot(
                data=trials_df, x=col, y='value', hue='value', palette='viridis', s=100, legend='auto'
            )
            plt.title(f'Objective vs {col.replace("params_", "").capitalize()}')
            plt.xlabel(col.replace('params_', ''))
            plt.ylabel('Objective Value')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{col}_vs_objective.png'))
            plt.close()
    
    param_cols = [c for c in trials_df.columns if c.startswith('params_')]
    if len(param_cols) > 1:
        sns.pairplot(trials_df, vars=param_cols, hue='value', palette='viridis')
        plt.suptitle('Pairplot of Hyperparameters (colored by objective)', y=1.02)
        plt.savefig(os.path.join(plot_dir, 'pairplot_hyperparams.png'))
        plt.close()
    
    print(f"Saved hyperparameter optimization plots to {plot_dir}")
    
    # Print final optimization summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"GA Budget Used: {GA_BUDGET}")
    print(f"PSO Budget Used: {PSO_BUDGET}")
    print(f"DE Budget Used: {DE_BUDGET}")
    print(f"BO Trials Used: {BO_TRIALS}")
    print(f"Training Iterations per Evaluation: {TRAINING_ITERS}")
    
    return study, ga_optimizer, pso_optimizer, de_optimizer


def run_training(args, use_optimized=False, resume=False):
    """Run the main training loop"""
    if use_optimized:
        print("Using optimized parameters for training...")
    else:
        print("Using default/CLI parameters for training...")
    
    test_dataset = TrainDataset(args.data_path_test, args.data_path_test, args.data_name, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    global results, best_psnr, best_ssim
    results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': [], 'Loss': []}, 0.0, 0.0
    
    # Create model with appropriate parameters
    global model
    act_type = getattr(args, 'act_type', 'gelu')
    norm_type = getattr(args, 'norm_type', 'layernorm')
    kernel_size = getattr(args, 'kernel_size', 3)
    
    # Get Mamba parameters
    use_mamba = OPTIMIZED_PARAMS.get('use_mamba', False)
    d_state = OPTIMIZED_PARAMS.get('d_state', 16)
    d_conv = OPTIMIZED_PARAMS.get('d_conv', 4)
    expand = OPTIMIZED_PARAMS.get('expand', 2)
    
    model = Inpainting(args.num_blocks, args.num_heads, args.channels, args.num_refinement, args.expansion_factor,
                      kernel_size=kernel_size, act_type=act_type, norm_type=norm_type,
                      use_mamba=use_mamba, d_state=d_state, d_conv=d_conv, expand=expand).to(device)
    print('Parameters of model are:', sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()))
    
    if args.model_file:
        model.load_state_dict(torch.load(args.model_file))
        save_loop(model, test_loader, 1)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)
        total_loss, total_num, results['Loss'], i = 0.0, 0, [], 0
        start_iter = 1
        
        # Resume from checkpoint if requested
        if resume:
            ckpt_data = load_checkpoint(model, optimizer, lr_scheduler, args.save_path, args.data_name, device)
            if ckpt_data is not None:
                start_iter, best_psnr, best_ssim, results, i = ckpt_data
                start_iter += 1  # Start from next iteration
                print(f"Resuming training from iteration {start_iter}")
            else:
                print("No checkpoint found, starting from scratch")
        
        train_bar = tqdm(range(start_iter, args.num_iter + 1), initial=start_iter, dynamic_ncols=True)
        
        # Get loss weights (either optimized or default)
        if hasattr(args, 'loss_weights'):
            loss_weights = args.loss_weights
            print(f"Using loss weights: L1={loss_weights[0]:.3f}, Perceptual={loss_weights[1]:.3f}, SSIM={loss_weights[2]:.3f}, Edge={loss_weights[3]:.3f}")
        else:
            loss_weights = (0.9, 0.5, 0.5, 0.4)  # Default weights
            print("Using default loss weights: L1=0.9, Perceptual=0.5, SSIM=0.5, Edge=0.4")
        
        for n_iter in train_bar:
            # progressive learning
            if n_iter == 1 or n_iter - 1 in args.milestone:
                end_iter = args.milestone[i] if i < len(args.milestone) else args.num_iter
                start_iter = args.milestone[i - 1] if i > 0 else 0
                # Use optimized batch size if available, otherwise default
                current_batch_size = getattr(args, 'batch_size', [2]*len(args.milestone))
                if isinstance(current_batch_size, list):
                    target_batch_size = current_batch_size[i]
                else:
                    target_batch_size = current_batch_size
                
                # Gradient accumulation settings
                # A40 (48GB) can handle batch_size=16 directly; T4 (16GB) needs accumulation at batch>4
                # Threshold: batch_size > 8 uses accumulation (conservative for stability)
                accum_iter = 1
                if target_batch_size > 8:
                    accum_iter = target_batch_size // 8
                    actual_batch_size = 8
                    print(f"Using gradient accumulation: actual_batch={actual_batch_size}, accum_steps={accum_iter}")
                else:
                    actual_batch_size = target_batch_size
                    print(f"Using direct batch size: {actual_batch_size}")
                    
                length = target_batch_size * (end_iter - start_iter)
                train_dataset = TrainDataset(args.data_path, args.data_path_test, args.data_name, 'train', args.patch_size[i], length)
                train_loader = iter(DataLoader(train_dataset, actual_batch_size, True, num_workers=args.workers))
                i += 1
            # train
            model.train()
            
            try:
                # Gradient Accumulation Loop
                total_acc_loss = 0
                optimizer.zero_grad()
                
                for _ in range(accum_iter):
                    try:
                        rain, norain, name, h, w = next(train_loader)
                    except StopIteration:
                        train_loader = iter(DataLoader(train_dataset, actual_batch_size, True, num_workers=args.workers))
                        rain, norain, name, h, w = next(train_loader)
                        
                    rain, norain = rain.to(device), norain.to(device)
                    out = model(rain)

                    ssim_loss = 1 - ssim(out, norain)
                    edge_out = kornia.filters.sobel(out,  normalized=True, eps=1e-06)
                    edge_gt = kornia.filters.sobel(norain, normalized=True, eps=1e-06)
                    edge_loss = F.l1_loss(edge_out, edge_gt) 

                    # Base loss
                    loss = (F.l1_loss(out, norain)*loss_weights[0] + perceptual_loss(out, norain)*loss_weights[1] + ssim_loss*loss_weights[2] + edge_loss*loss_weights[3])
                    
                    # Add frequency loss if available (NEW)
                    if frequency_loss is not None:
                        freq_loss = frequency_loss(out, norain)
                        w_freq = loss_weights[4] if len(loss_weights) > 4 else OPTIMIZED_PARAMS['loss_weights'].get('w_freq', 0.3)
                        loss = loss + freq_loss * w_freq
                    
                    loss = loss / accum_iter  # Scale for accumulation
                    
                    loss.backward()
                    total_acc_loss += loss.item() * accum_iter # Scale back for logging
                    
                optimizer.step()
                
                total_num += rain.size(0) * accum_iter
                total_loss += total_acc_loss * rain.size(0) * accum_iter
                train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'
                                          .format(n_iter, args.num_iter, total_loss / total_num))
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA OOM in training at iter {n_iter}. Attempting fallback to batch_size=1.")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    
                    # Fallback: Try with batch_size=1
                    try:
                        # Process one sample at a time
                        fallback_accum = target_batch_size # Accumulate everything to match target
                        total_acc_loss = 0
                        
                        for _ in range(fallback_accum):
                            try:
                                rain, norain, name, h, w = next(train_loader)
                            except StopIteration:
                                train_loader = iter(DataLoader(train_dataset, 1, True, num_workers=args.workers))
                                rain, norain, name, h, w = next(train_loader)
                                
                            # Ensure single batch dimension
                            if rain.size(0) > 1:
                                rain = rain[0:1]
                                norain = norain[0:1]
                                
                            rain, norain = rain.to(device), norain.to(device)
                            out = model(rain)
                            
                            ssim_loss = 1 - ssim(out, norain)
                            edge_out = kornia.filters.sobel(out,  normalized=True, eps=1e-06)
                            edge_gt = kornia.filters.sobel(norain, normalized=True, eps=1e-06)
                            edge_loss = F.l1_loss(edge_out, edge_gt) 
                            
                            loss = (F.l1_loss(out, norain)*loss_weights[0] + perceptual_loss(out, norain)*loss_weights[1] + ssim_loss*loss_weights[2] + edge_loss*loss_weights[3]) / fallback_accum
                            
                            loss.backward()
                            total_acc_loss += loss.item() * fallback_accum
                            
                        optimizer.step()
                        total_num += fallback_accum
                        total_loss += total_acc_loss * fallback_accum
                        train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f} (Fallback)'
                                                  .format(n_iter, args.num_iter, total_loss / total_num))
                                                  
                    except RuntimeError as e2:
                        if "out of memory" in str(e2):
                            print(f"CUDA OOM even with batch_size=1 at iter {n_iter}. Skipping.")
                            torch.cuda.empty_cache()
                            optimizer.zero_grad()
                            continue
                        else:
                            raise e2
                else:
                    raise e

            lr_scheduler.step()
            if n_iter % 1000 == 0:
                results['Loss'].append('{:.3f}'.format(total_loss / total_num))
                save_loop(model, test_loader, n_iter)
            
            # Save checkpoint every 500 iterations for resume capability
            if n_iter % 500 == 0:
                save_checkpoint(model, optimizer, lr_scheduler, n_iter, best_psnr, best_ssim, 
                               results, i, args.save_path, args.data_name)

            # Save inpainted image for each batch
            save_dir = os.path.join(args.save_path, args.data_name)
            os.makedirs(save_dir, exist_ok=True)
            if isinstance(name, (list, tuple)) or (hasattr(name, '__iter__') and not isinstance(name, str)):
                for img, img_name in zip(out, name):
                    img_np = img.detach().cpu().permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).clip(0, 255).astype('uint8') if img_np.max() <= 1.0 else img_np.astype('uint8')
                    out_path = os.path.join(save_dir, str(img_name))
                    print(f"Saving inpainted image to: {out_path}")
                    Image.fromarray(img_np).save(out_path)
            else:
                img = out[0]
                img_np = img.detach().cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).clip(0, 255).astype('uint8') if img_np.max() <= 1.0 else img_np.astype('uint8')
                out_path = os.path.join(save_dir, str(name))
                print(f"Saving inpainted image to: {out_path}")
                Image.fromarray(img_np).save(out_path)

        # === AUTO-EVALUATION AND VISUALIZATION AFTER TRAINING ===
        print("\n" + "="*60)
        print("TRAINING COMPLETE - Running Evaluation and Visualization")
        print("="*60)
        
        # Generate training visualizations
        try:
            from evaluation.visualize import generate_all_visualizations
            csv_path = os.path.join(args.save_path, f'{args.data_name}.csv')
            if os.path.exists(csv_path):
                figures_dir = os.path.join(args.save_path, 'figures')
                generate_all_visualizations(csv_path, figures_dir)
                print(f"Visualizations saved to {figures_dir}")
        except Exception as e:
            print(f"Visualization generation failed: {e}")
        
        # Run evaluation if test data available
        try:
            from evaluation.metrics import MetricsCalculator, print_results
            pred_dir = os.path.join(args.save_path, args.data_name)
            gt_dir = os.path.join(args.data_path_test, 'input')  # Ground truth
            
            if os.path.exists(pred_dir) and os.path.exists(gt_dir):
                calculator = MetricsCalculator()
                results = calculator.evaluate_directories(
                    pred_dir=pred_dir,
                    gt_dir=gt_dir,
                    save_csv=os.path.join(args.save_path, 'evaluation_results.csv')
                )
                print_results(results, "Wave-Mamba")
        except Exception as e:
            print(f"Evaluation failed: {e}")


if __name__ == '__main__':
    # Parse all arguments first, including the --optimize flag
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize', action='store_true', help='Run metaheuristic optimization instead of using optimized defaults')
    parser.add_argument('--use-defaults', action='store_true', help='Use original default parameters instead of optimized ones')
    parser.add_argument('--force_batch_size', type=int, default=None, help='Force batch size to this value (e.g., 1 for Kaggle). Overrides optimized batch size.')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    temp_args, remaining = parser.parse_known_args()
    
    # Remove custom flags from sys.argv to avoid conflicts with parse_args
    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()
    
    # Override batch_size if force_batch_size is specified
    if temp_args.force_batch_size is not None:
        args.batch_size = [temp_args.force_batch_size] * len(args.milestone)
        print(f"Batch size overridden to {temp_args.force_batch_size} for all stages (Kaggle mode)")
    
    if temp_args.optimize:
        print("=" * 60)
        print("RUNNING HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        # Use improved optimization
        study, ga_optimizer, pso_optimizer, de_optimizer = run_improved_optimization(args)
        
        # Extract best parameters from optimization
        best_params = study.best_trial.params
        best_ga = ga_optimizer.provide_recommendation().value
        best_pso = pso_optimizer.provide_recommendation().value
        best_de = de_optimizer.provide_recommendation().value
        
        print("Optimization results:")
        print("best_ga:", best_ga)
        print("best_pso:", best_pso)
        print("best_de:", best_de)
        
        # Apply optimized parameters to args
        if best_ga and isinstance(best_ga, tuple) and len(best_ga) > 1 and isinstance(best_ga[1], dict):
            params_ga = best_ga[1]
            args.num_blocks = params_ga['num_blocks']
            args.num_heads = params_ga['num_heads']
            args.channels = params_ga['channels']
            args.num_refinement = params_ga['num_refinement']
            args.act_type = params_ga.get('act_type', 'gelu')
            args.norm_type = params_ga.get('norm_type', 'layernorm')
            args.kernel_size = params_ga.get('kernel_size', 3)
        else:
            print("Warning: best_ga is not valid, using optimized defaults.")
            args.num_blocks = OPTIMIZED_PARAMS['num_blocks']
            args.num_heads = OPTIMIZED_PARAMS['num_heads']
            args.channels = OPTIMIZED_PARAMS['channels']
            args.num_refinement = OPTIMIZED_PARAMS['num_refinement']
        
        if best_pso and isinstance(best_pso, tuple) and len(best_pso) > 1 and isinstance(best_pso[1], dict):
            params_pso = best_pso[1]
            args.loss_weights = (
                params_pso['w_l1'],
                params_pso['w_percep'],
                params_pso['w_ssim'],
                params_pso['w_edge']
            )
            args.dwa_temp = params_pso.get('dwa_temp', 2.0)
        else:
            print("Warning: best_pso is not valid, using optimized defaults.")
            args.loss_weights = (
                OPTIMIZED_PARAMS['loss_weights']['w_l1'],
                OPTIMIZED_PARAMS['loss_weights']['w_percep'],
                OPTIMIZED_PARAMS['loss_weights']['w_ssim'],
                OPTIMIZED_PARAMS['loss_weights']['w_edge']
            )
            args.dwa_temp = 2.0
        
        if best_de and isinstance(best_de, tuple) and len(best_de) > 1 and isinstance(best_de[1], dict):
            params_de = best_de[1]
            args.expansion_factor = params_de['expansion_factor']
        else:
            print("Warning: best_de is not valid, using optimized default.")
            args.expansion_factor = OPTIMIZED_PARAMS['expansion_factor']
        
        args.lr = best_params.get('lr', OPTIMIZED_PARAMS['lr'])
        
        print("=" * 60)
        print("STARTING TRAINING WITH NEWLY OPTIMIZED PARAMETERS")
        print("=" * 60)
        
        run_training(args, use_optimized=True, resume=temp_args.resume)
        
    elif temp_args.use_defaults:
        print("=" * 60)
        print("USING ORIGINAL DEFAULT PARAMETERS")
        print("=" * 60)
        
        run_training(args, use_optimized=False, resume=temp_args.resume)
        
    else:
        print("=" * 60)
        print("USING PRE-OPTIMIZED PARAMETERS")
        print("=" * 60)
        
        # Apply pre-optimized parameters
        apply_optimized_params(args)
        run_training(args, use_optimized=True, resume=temp_args.resume)