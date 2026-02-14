import argparse
import random
import torch
import numpy as np
import os
from torch.backends import cudnn

def parse_args():
    desc = 'WaveSSM-X: Blind Omni-dimensional Wavelet-Guided State Space Model'
    parser = argparse.ArgumentParser(description=desc)
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./datasets/celeb', help='Path to training data')
    parser.add_argument('--data_path_test', type=str, default='./datasets/celeb', help='Path to test data')
    parser.add_argument('--task_name', type=str, default='inpaint', choices=['inpaint'])
    parser.add_argument('--dataset_name', type=str, default='celeb')
    
    # Model arguments
    parser.add_argument('--model_file', type=str, default='./checkpoints/wavessm_x.pth', help='Path to save/load model')
    parser.add_argument('--use_mamba', action='store_true', default=False, help='Use Mamba (O(N)) instead of Transformer')
    parser.add_argument('--d_state', type=int, default=16, help='SSM state dimension')
    parser.add_argument('--d_conv', type=int, default=4, help='SSM convolution kernel size')
    parser.add_argument('--expand', type=int, default=2, help='SSM expansion factor')
    parser.add_argument('--no_fass', action='store_true', help='Disable DualStreamFASS bottleneck')
    parser.add_argument('--no_ffc', action='store_true', help='Disable MultiScaleWaveFFC refinement')
    
    # Training arguments
    parser.add_argument('--num_iter', type=int, default=100000, help='Total training iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--val_every', type=int, default=500, help='Validation frequency (iterations)')
    
    # Loss weights
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[1.0, 1.0, 1.0, 0.5, 0.3], 
                        help='Weights for [L1, Perceptual, SSIM, Edge, Frequency]')
                        
    # Ablation arguments
    parser.add_argument('--fass_no_b', action='store_true', help='Disable B modulation in FASS')
    parser.add_argument('--fass_no_c', action='store_true', help='Disable C modulation in FASS')
    parser.add_argument('--fass_no_delta', action='store_true', help='Disable Delta modulation in FASS')
    parser.add_argument('--wavelet', type=str, default='db3', help='Wavelet type for frequency extraction')

    args = parser.parse_args()
    return init_args(args)


class Config(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data_path_test = args.data_path_test
        self.task_name = args.task_name
        self.dataset_name = args.dataset_name
        
        self.model_file = args.model_file
        self.use_mamba = args.use_mamba
        self.d_state = args.d_state
        self.d_conv = args.d_conv
        self.expand = args.expand
        self.use_fass = not args.no_fass
        self.use_ffc = not args.no_ffc
        
        self.num_iter = args.num_iter
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.workers = args.workers
        self.seed = args.seed
        self.resume = args.resume
        self.val_every = args.val_every
        self.loss_weights = args.loss_weights
        
        # Ablation settings
        self.fass_no_b = args.fass_no_b
        self.fass_no_c = args.fass_no_c
        self.fass_no_delta = args.fass_no_delta
        self.wavelet = args.wavelet


def init_args(args):
    """Initialize environment and directories"""
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    if not os.path.exists('./checkpoints/'):
        os.makedirs('./checkpoints/')

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            cudnn.deterministic = False
            cudnn.benchmark = True  # Fixed 256x256 patches → autotuning gives 20-30% speedup
            
    return Config(args)
