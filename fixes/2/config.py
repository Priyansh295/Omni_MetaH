"""
Configuration and argument parsing
FIXED: Lower default learning rate (1e-4 instead of 2e-4)
"""
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='WaveSSM-X Training')
    
    # ══════════════════════════════════════════════════════════
    # FIXED: Lower default LR for stability (1e-4 instead of 2e-4)
    # ══════════════════════════════════════════════════════════
    parser.add_argument('--lr', type=float, default=1e-4, 
                       help='Learning rate (default: 1e-4, more stable than 2e-4)')
    
    # Data
    parser.add_argument('--data_path', type=str, default='./DataSetFiles/Main_Dataset',
                       help='Path to training dataset')
    parser.add_argument('--data_path_test', type=str, default='./DataSetFiles/Test_Dataset',
                       help='Path to test dataset')
    parser.add_argument('--dataset_name', type=str, default='WaveSSM',
                       help='Dataset name')
    
    # Training
    parser.add_argument('--num_iter', type=int, default=33000,
                       help='Total training iterations')
    parser.add_argument('--batch_size', type=int, default=24,
                       help='Batch size')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Model
    parser.add_argument('--use_mamba', action='store_true', default=True,
                       help='Use Mamba SSM blocks')
    parser.add_argument('--d_state', type=int, default=16,
                       help='SSM state dimension')
    parser.add_argument('--d_conv', type=int, default=4,
                       help='SSM convolution kernel size')
    parser.add_argument('--expand', type=int, default=2,
                       help='SSM expansion factor')
    parser.add_argument('--use_fass', action='store_true', default=True,
                       help='Use FASS blocks')
    parser.add_argument('--use_ffc', action='store_true', default=True,
                       help='Use FFC blocks')
    parser.add_argument('--wavelet', type=str, default='db3',
                       help='Wavelet type for FFC')
    
    # FASS ablation
    parser.add_argument('--fass_no_b', action='store_true', default=False,
                       help='Disable B parameter in FASS')
    parser.add_argument('--fass_no_c', action='store_true', default=False,
                       help='Disable C parameter in FASS')
    parser.add_argument('--fass_no_delta', action='store_true', default=False,
                       help='Disable delta parameter in FASS')
    
    # Loss weights
    parser.add_argument('--loss_weights', type=float, nargs='+',
                       default=[1.0, 1.0, 1.0, 0.5, 0.3],
                       help='Loss weights: [l1, perceptual, ssim, edge, freq]')
    
    # Checkpointing
    parser.add_argument('--model_file', type=str, default='./checkpoints/wavessm_x.pth',
                       help='Path to save/load checkpoint')
    parser.add_argument('--resume', action='store_true', default=False,
                       help='Resume from checkpoint')
    parser.add_argument('--val_every', type=int, default=500,
                       help='Validation frequency (iterations)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {args.batch_size}")
    
    if args.lr <= 0:
        raise ValueError(f"Learning rate must be > 0, got {args.lr}")
    
    if args.num_iter < 1:
        raise ValueError(f"Number of iterations must be >= 1, got {args.num_iter}")
    
    return args
