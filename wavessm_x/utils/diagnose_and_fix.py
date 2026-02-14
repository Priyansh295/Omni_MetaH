"""
Checkpoint Diagnostic and Recovery Tool
========================================
Use this to inspect and repair corrupted model checkpoints.

Usage:
    python diagnose_and_fix.py --checkpoint ./checkpoints/wavessm_x.pth

This will:
1. Load and analyze the checkpoint
2. Identify corrupted BatchNorm statistics
3. Optionally create a fixed version
"""
import torch
import argparse
import os
from collections import defaultdict


def analyze_checkpoint(checkpoint_path):
    """
    Deep analysis of a checkpoint file.
    Returns detailed corruption report.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: File not found: {checkpoint_path}")
        return None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_state = checkpoint['model_state_dict']
    
    # Analysis results
    results = {
        'total_params': 0,
        'corrupted_params': [],
        'corrupted_bn_stats': [],
        'param_stats': defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'has_nan': False, 'has_inf': False})
    }
    
    print("📊 Parameter Analysis:")
    print("-" * 60)
    
    for key, tensor in model_state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        
        results['total_params'] += 1
        
        # Check for NaN/Inf
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        is_finite = torch.isfinite(tensor).all().item()
        
        # Statistics
        if tensor.numel() > 0 and is_finite:
            results['param_stats'][key]['min'] = tensor.min().item()
            results['param_stats'][key]['max'] = tensor.max().item()
        
        results['param_stats'][key]['has_nan'] = has_nan
        results['param_stats'][key]['has_inf'] = has_inf
        
        # Report corruption
        if not is_finite:
            param_type = "Unknown"
            if 'running_mean' in key:
                param_type = "BatchNorm running_mean"
                results['corrupted_bn_stats'].append(key)
            elif 'running_var' in key:
                param_type = "BatchNorm running_var"
                results['corrupted_bn_stats'].append(key)
            elif 'weight' in key or 'bias' in key:
                param_type = "Model weight/bias"
                results['corrupted_params'].append(key)
            
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            print(f"❌ CORRUPTED [{param_type}]: {key}")
            print(f"   Shape: {list(tensor.shape)}")
            print(f"   NaN count: {nan_count} / {tensor.numel()}")
            print(f"   Inf count: {inf_count} / {tensor.numel()}")
            print()
    
    # Summary
    print("\n" + "="*60)
    print("📈 Summary:")
    print("="*60)
    print(f"Total parameters/buffers: {results['total_params']}")
    print(f"Corrupted BatchNorm stats: {len(results['corrupted_bn_stats'])}")
    print(f"Corrupted weights/biases: {len(results['corrupted_params'])}")
    
    if results['corrupted_bn_stats']:
        print(f"\n⚠️  BatchNorm corruptions (fixable):")
        for key in results['corrupted_bn_stats'][:10]:
            print(f"   - {key}")
        if len(results['corrupted_bn_stats']) > 10:
            print(f"   ... and {len(results['corrupted_bn_stats'])-10} more")
    
    if results['corrupted_params']:
        print(f"\n🔥 Weight corruptions (SEVERE - training likely unrecoverable):")
        for key in results['corrupted_params'][:10]:
            print(f"   - {key}")
        if len(results['corrupted_params']) > 10:
            print(f"   ... and {len(results['corrupted_params'])-10} more")
    
    # Additional metadata
    print(f"\n📝 Checkpoint Metadata:")
    print(f"   Iteration: {checkpoint.get('iteration', 'N/A')}")
    print(f"   Best PSNR: {checkpoint.get('best_psnr', 'N/A')}")
    print(f"   Best SSIM: {checkpoint.get('best_ssim', 'N/A')}")
    print(f"   Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    
    return results, checkpoint


def fix_checkpoint(checkpoint_path, output_path=None):
    """
    Create a fixed version of the checkpoint by resetting corrupted BN stats.
    """
    results, checkpoint = analyze_checkpoint(checkpoint_path)
    
    if results is None:
        return False
    
    if not results['corrupted_bn_stats'] and not results['corrupted_params']:
        print("\n✅ Checkpoint is healthy! No fixes needed.")
        return True
    
    if results['corrupted_params']:
        print("\n⚠️  WARNING: This checkpoint has corrupted weights, not just BN stats.")
        print("    Fixing BN stats may help, but training may still be unstable.")
        response = input("    Continue with fix? (y/n): ")
        if response.lower() != 'y':
            print("Fix cancelled.")
            return False
    
    print(f"\n🔧 Fixing {len(results['corrupted_bn_stats'])} corrupted BatchNorm stats...")
    
    model_state = checkpoint['model_state_dict']
    fixed_count = 0
    
    for key in results['corrupted_bn_stats']:
        if 'running_mean' in key:
            print(f"   Resetting {key} to zeros")
            model_state[key] = torch.zeros_like(model_state[key])
            fixed_count += 1
        elif 'running_var' in key:
            print(f"   Resetting {key} to ones")
            model_state[key] = torch.ones_like(model_state[key])
            fixed_count += 1
    
    # Save fixed checkpoint
    if output_path is None:
        base, ext = os.path.splitext(checkpoint_path)
        output_path = f"{base}_FIXED{ext}"
    
    print(f"\n💾 Saving fixed checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    print(f"✅ Fixed {fixed_count} corrupted stats")
    
    # Verify the fix
    print("\n🔍 Verifying fixed checkpoint...")
    verify_results, _ = analyze_checkpoint(output_path)
    
    if verify_results and len(verify_results['corrupted_bn_stats']) == 0:
        print("\n✅ SUCCESS! All BatchNorm stats are now healthy.")
        return True
    else:
        print("\n⚠️  Some corruptions remain. Manual intervention may be needed.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Diagnose and fix corrupted checkpoints')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--fix', action='store_true',
                       help='Attempt to fix the checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for fixed checkpoint (default: <checkpoint>_FIXED.pth)')
    
    args = parser.parse_args()
    
    if args.fix:
        success = fix_checkpoint(args.checkpoint, args.output)
        if success:
            print("\n" + "="*60)
            print("Next steps:")
            print("="*60)
            print("1. Use the FIXED checkpoint for training")
            print("2. Use the new train_FIXED.py script which has:")
            print("   - Automatic BN sanitization before validation")
            print("   - Periodic model health checks")
            print("   - Better gradient clipping")
            print("   - More stable BatchNorm configuration")
            print("="*60)
    else:
        analyze_checkpoint(args.checkpoint)
        print("\n" + "="*60)
        print("To fix this checkpoint, run:")
        print(f"  python {os.path.basename(__file__)} --checkpoint {args.checkpoint} --fix")
        print("="*60)


if __name__ == '__main__':
    main()
