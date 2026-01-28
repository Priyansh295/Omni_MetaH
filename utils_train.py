import argparse
import glob
import os
import random
import time
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image, ImageOps
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
import torchvision
import cv2


class OptimizedConfig(object):
    """Enhanced configuration class with metaheuristic optimization support"""
    
    def __init__(self, args):
        # Data paths
        self.data_path = args.data_path
        self.data_path_test = args.data_path_test
        self.data_name = args.data_name
        self.save_path = args.save_path
        
        # Model architecture (optimized by metaheuristics)
        self.num_blocks = args.num_blocks
        self.num_heads = args.num_heads
        self.channels = args.channels
        self.expansion_factor = args.expansion_factor
        self.num_refinement = args.num_refinement
        
        # Training parameters
        self.num_iter = args.num_iter
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.lr = args.lr
        self.milestone = args.milestone
        self.workers = args.workers
        self.model_file = args.model_file
        self.finetune = args.finetune
        
        # Validation and early stopping
        self.val_every = getattr(args, 'val_every', 500)
        self.early_stop = getattr(args, 'early_stop', True)
        self.target_psnr = getattr(args, 'target_psnr', 30.0)
        self.target_ssim = getattr(args, 'target_ssim', 0.9)
        self.early_stop_patience = getattr(args, 'early_stop_patience', 3)
        self.min_delta = getattr(args, 'min_delta', 0.001)
        
        # Optimization parameters
        self.mixed_precision = getattr(args, 'mixed_precision', True)
        self.gradient_clip = getattr(args, 'gradient_clip', 1.0)
        self.memory_efficient = getattr(args, 'memory_efficient', True)
        self.prefetch_factor = getattr(args, 'prefetch_factor', 2)
        self.persistent_workers = getattr(args, 'persistent_workers', True)
        
        # Data augmentation parameters
        self.use_advanced_augmentation = getattr(args, 'use_advanced_augmentation', True)
        self.augmentation_prob = getattr(args, 'augmentation_prob', 0.5)
        
        # Performance monitoring
        self.enable_profiling = getattr(args, 'enable_profiling', False)
        self.memory_monitoring = getattr(args, 'memory_monitoring', True)


def parse_args():
    """Enhanced argument parser with metaheuristic optimization support"""
    desc = 'Pytorch Implementation of Enhanced Blind Image Inpainting with Metaheuristic Optimization'
    parser = argparse.ArgumentParser(description=desc)
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='/train/', 
                       help='Training data path')
    parser.add_argument('--data_path_test', type=str, default='/test/', 
                       help='Test data path')
    parser.add_argument('--data_name', type=str, default='inpaint', 
                       choices=['rain100L', 'rain100H', 'inpaint'],
                       help='Dataset name')
    parser.add_argument('--save_path', type=str, default='test_places',
                       help='Path to save results')
    
    # Model architecture parameters (optimized by GA)
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[4, 6, 6, 8],
                       help='Number of transformer blocks for each level')
    parser.add_argument('--num_heads', nargs='+', type=int, default=[1, 2, 4, 8],
                       help='Number of attention heads for each level')
    parser.add_argument('--channels', nargs='+', type=int, default=[48//3, 96//3, 192//3, 384//3],
                       help='Number of channels for each level')
    parser.add_argument('--expansion_factor', type=float, default=2.66, 
                       help='Factor of channel expansion for GDFN (optimized by DE)')
    parser.add_argument('--num_refinement', type=int, default=4, 
                       help='Number of channels for refinement stage (optimized by GA)')
    
    # Training parameters
    parser.add_argument('--num_iter', type=int, default=9057800, 
                       help='Iterations of training')
    parser.add_argument('--batch_size', nargs='+', type=int, default=[8, 8, 8, 8, 8, 8],
                       help='Batch size for progressive learning (default 8 for A40/48GB)')
    parser.add_argument('--patch_size', nargs='+', type=int, default=[128, 160, 192, 256, 320, 384],
                       help='Patch size for progressive learning')
    parser.add_argument('--lr', type=float, default=0.00003, 
                       help='Initial learning rate (optimized by BO)')
    parser.add_argument('--milestone', nargs='+', type=int, default=[1125000, 1800000, 840000, 2250000],
                       help='When to change patch size and batch size')
    parser.add_argument('--workers', type=int, default=4, 
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=-1, 
                       help='Random seed (-1 for no manual seed)')
    parser.add_argument('--model_file', type=str, default=None, 
                       help='Path of pre-trained model file')
    parser.add_argument('--finetune', default=True, 
                       help='Enable fine-tuning')

    # Validation cadence and early stopping
    parser.add_argument('--val_every', type=int, default=500,
                       help='Validate and checkpoint every N iterations')
    parser.add_argument('--early_stop', type=bool, default=True,
                       help='Enable metric-based early stopping')
    parser.add_argument('--target_psnr', type=float, default=30.0,
                       help='Stop when validation PSNR reaches this threshold')
    parser.add_argument('--target_ssim', type=float, default=0.9,
                       help='Stop when validation SSIM reaches this threshold')
    parser.add_argument('--early_stop_patience', type=int, default=3,
                       help='Stop after this many validations without improvement')
    parser.add_argument('--min_delta', type=float, default=0.001,
                       help='Minimum improvement to reset patience')
    
    # Optimization parameters
    parser.add_argument('--mixed_precision', type=bool, default=True,
                       help='Use mixed precision training')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--memory_efficient', type=bool, default=True,
                       help='Use memory efficient training')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                       help='DataLoader prefetch factor')
    parser.add_argument('--persistent_workers', type=bool, default=True,
                       help='Use persistent workers for DataLoader')
    
    # Data augmentation parameters
    parser.add_argument('--use_advanced_augmentation', type=bool, default=True,
                       help='Use advanced data augmentation')
    parser.add_argument('--augmentation_prob', type=float, default=0.5,
                       help='Probability for augmentation operations')
    
    # Performance monitoring
    parser.add_argument('--enable_profiling', type=bool, default=False,
                       help='Enable performance profiling')
    parser.add_argument('--memory_monitoring', type=bool, default=True,
                       help='Enable memory usage monitoring')
    
    return init_args(parser.parse_args())


def init_args(args):
    """Enhanced argument initialization with optimization support"""
    
    # Create save directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Create subdirectories for organization
    subdirs = ['checkpoints', 'logs', 'visualizations', 'metaheuristic_results']
    for subdir in subdirs:
        os.makedirs(os.path.join(args.save_path, subdir), exist_ok=True)
    
    # Set random seeds for reproducibility
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        print(f"Random seed set to: {args.seed}")
    else:
        # Enable cudnn benchmark for better performance
        cudnn.benchmark = True
        print("CuDNN benchmark enabled for performance")
    
    # Optimize worker count based on system capabilities
    if args.workers <= 0:
        args.workers = min(4, os.cpu_count() if os.cpu_count() else 2)
        print(f"Auto-set workers to: {args.workers}")
    
    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("CUDA optimization flags enabled")
    
    return OptimizedConfig(args)


def pad_image_needed(img: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Optimized image padding with memory efficiency"""
    _, height, width = img.shape
    target_height, target_width = size
    
    # Only pad if necessary
    pad_width = max(0, target_width - width)
    pad_height = max(0, target_height - height)
    
    if pad_width > 0 or pad_height > 0:
        # Use reflect padding for better visual quality
        img = T.pad(img, [0, 0, pad_width, pad_height], padding_mode='reflect')
    
    return img


class AdvancedAugmentation:
    """Advanced data augmentation techniques for improved generalization"""
    
    def __init__(self, probability: float = 0.5):
        self.prob = probability
    
    def apply_color_jitter(self, img: torch.Tensor) -> torch.Tensor:
        """Apply subtle color jittering"""
        if torch.rand(1) < self.prob:
            # Convert to PIL for color jittering
            pil_img = T.to_pil_image(img)
            jitter = torchvision.transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            )
            pil_img = jitter(pil_img)
            return T.to_tensor(pil_img)
        return img
    
    def apply_gaussian_noise(self, img: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        """Add subtle Gaussian noise"""
        if torch.rand(1) < self.prob:
            noise = torch.randn_like(img) * noise_std
            return torch.clamp(img + noise, 0, 1)
        return img
    
    def apply_random_erasing(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random erasing for robustness"""
        if torch.rand(1) < self.prob * 0.3:  # Lower probability for erasing
            erasing = torchvision.transforms.RandomErasing(
                p=1.0, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=0
            )
            return erasing(img)
        return img

class OptimizedTrainDataset(Dataset):
    """Enhanced training dataset with optimizations and advanced augmentation"""
    
    def __init__(self, data_path: str, data_path_test: str, data_name: str, 
                 data_type: str, patch_size: Optional[int] = None, 
                 length: Optional[int] = None, use_advanced_aug: bool = True,
                 inp_files: Optional[List[str]] = None, target_files: Optional[List[str]] = None):
        super().__init__()
        
        self.data_name = data_name
        self.data_type = data_type
        self.patch_size = patch_size
        self.use_advanced_aug = use_advanced_aug
        
        # Advanced augmentation
        self.augmentation = AdvancedAugmentation(probability=0.5) if use_advanced_aug else None
        
        # Enhanced file discovery with multiple extensions and error handling
        if inp_files is not None and target_files is not None:
            # Use provided file lists (e.g. from train/val split)
            # FIX: Always put provided files in primary dataset for both train and val
            self.corrupt_images = inp_files
            self.clear_images = target_files
            self.corrupt_images_test = []
            self.clear_images_test = []
        else:
            # Discover files from directories
            self.corrupt_images, self.clear_images = self._discover_images(data_path)
            self.corrupt_images_test, self.clear_images_test = self._discover_images(data_path_test)
        
        # Validate discovered images
        self._validate_image_pairs()
        
        self.num = len(self.corrupt_images)
        self.num_test = len(self.corrupt_images_test)
        self.sample_num = length if data_type == 'train' else self.num
        
        # Caching for frequently accessed small datasets
        self.cache_size = 100
        self.image_cache = {}
        
        print(f"Dataset initialized: {self.sample_num} samples, "
              f"train: {self.num}, test: {self.num_test}, "
              f"patch_size: {patch_size}, advanced_aug: {use_advanced_aug}")
    
    def _discover_images(self, data_path: str) -> Tuple[List[str], List[str]]:
        """Enhanced image discovery with multiple formats and fallback paths"""
        
        file_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.bmp', '*.tiff']
        
        # Initialize lists to avoid NameError
        corrupt_images = []
        clear_images = []
        
        # Custom logic for user's specific dataset structure:
        # User confirmed: 'input' folder contains GROUND TRUTH (clear)
        # User confirmed: 'target' folder contains MASKED IMAGE (corrupt)
        
        # 1. Find Corrupt Images (Source: 'target', 'masked', etc.)
        # Priority: target -> masked -> corrupted
        corrupt_folders = ['target', 'masked', 'corrupted', 'inp'] 
        for folder in corrupt_folders:
            folder_path = os.path.join(data_path, folder)
            if os.path.exists(folder_path):
                print(f"  [Dataset] Found corrupt/masked images in: {folder}")
                for ext in file_extensions:
                    corrupt_images.extend(glob.glob(os.path.join(folder_path, ext)))
                if corrupt_images:
                    break

        # 2. Find Clear Images (Source: 'input', 'gt', etc.)
        # Priority: input -> gt -> clean -> original
        clear_folders = ['input', 'gt', 'clean', 'original', 'target'] # Added target at end just in case but input is prio
        for folder in clear_folders:
            folder_path = os.path.join(data_path, folder)
            if os.path.exists(folder_path):
                print(f"  [Dataset] Found ground truth/clear images in: {folder}")
                for ext in file_extensions:
                    clear_images.extend(glob.glob(os.path.join(folder_path, ext)))
                if clear_images:
                    break
        
        return sorted(corrupt_images), sorted(clear_images)
    
    def _validate_image_pairs(self):
        """Validate that corrupt and clear images are properly paired"""
        
        if len(self.corrupt_images) == 0:
            print(f"Warning: No corrupt images found for {self.data_type} dataset")
        
        if len(self.clear_images) == 0:
            print(f"Warning: No clear images found for {self.data_type} dataset")
        
        if len(self.corrupt_images) != len(self.clear_images):
            min_len = min(len(self.corrupt_images), len(self.clear_images))
            print(f"Warning: Mismatch in image counts. Using {min_len} pairs.")
            self.corrupt_images = self.corrupt_images[:min_len]
            self.clear_images = self.clear_images[:min_len]
    
    def _load_image_with_cache(self, image_path: str) -> torch.Tensor:
        """Load image with caching for frequently accessed small datasets"""
        
        # Use caching for small datasets to improve performance
        if len(self.corrupt_images) <= self.cache_size:
            if image_path in self.image_cache:
                return self.image_cache[image_path].clone()
        
        try:
            # Enhanced image loading with error handling
            with Image.open(image_path) as img:
                # Convert to RGB to handle different formats consistently
                img = img.convert('RGB')
                
                # Optimize image loading for common issues
                img = ImageOps.exif_transpose(img)  # Handle EXIF rotation
                
                tensor_img = T.to_tensor(img)
                
                # Cache if appropriate
                if len(self.corrupt_images) <= self.cache_size:
                    self.image_cache[image_path] = tensor_img.clone()
                
                return tensor_img
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a default black image to avoid crashes
            return torch.zeros(3, 256, 256)
    
    def _apply_optimized_augmentation(self, corrupt: torch.Tensor, 
                                    clear: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply optimized augmentation with single random sampling"""
        
        # Generate all random values at once for efficiency
        flip_h = torch.rand(1) < 0.5
        flip_v = torch.rand(1) < 0.5
        
        # Apply geometric transformations to both images consistently
        if flip_h:
            corrupt = T.hflip(corrupt)
            clear = T.hflip(clear)
        
        if flip_v:
            corrupt = T.vflip(corrupt)
            clear = T.vflip(clear)
        
        # Apply advanced augmentation only during training
        if self.data_type == 'train' and self.augmentation is not None:
            # Apply augmentation to corrupt image only (preserves target)
            corrupt = self.augmentation.apply_color_jitter(corrupt)
            corrupt = self.augmentation.apply_gaussian_noise(corrupt)
            # Note: Random erasing not applied to maintain pairing
        
        return corrupt, clear
    
    def __len__(self) -> int:
        return self.sample_num
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int, int]:
        """Enhanced getitem with optimized loading and error handling"""
        
        try:
            # Determine which list to use
            # If we explicitly provided files (which populates corrupt_images but not corrupt_images_test),
            # or if we are in train mode, use the primary list.
            # Only use test list if we are NOT in train mode AND we have test images discovered.
            use_test_list = (self.data_type != 'train') and (self.num_test > 0)
            
            if not use_test_list:
                # Training data OR Validation data provided as explicit files
                if self.num == 0:
                    raise IndexError(f"Primary dataset is empty (data_type={self.data_type})")
                corrupt_image_path = self.corrupt_images[idx % self.num]
                clear_image_path = self.clear_images[idx % self.num]
            else:
                # Test data from separated test directory
                if self.num_test == 0:
                     raise IndexError(f"Test dataset is empty (data_type={self.data_type})")
                corrupt_image_path = self.corrupt_images_test[idx % self.num_test]
                clear_image_path = self.clear_images_test[idx % self.num_test]
            
            # Load images with caching
            corrupt = self._load_image_with_cache(corrupt_image_path)
            clear = self._load_image_with_cache(clear_image_path)
            
            # Get original dimensions
            h, w = corrupt.shape[1:]
            
            # Apply patch cropping for training
            if self.data_type == 'train' and self.patch_size:
                # Ensure images are large enough
                corrupt = pad_image_needed(corrupt, (self.patch_size, self.patch_size))
                clear = pad_image_needed(clear, (self.patch_size, self.patch_size))
                
                # Get crop parameters once for both images
                i, j, th, tw = RandomCrop.get_params(corrupt, (self.patch_size, self.patch_size))
                
                # Apply same crop to both images
                corrupt = T.crop(corrupt, i, j, th, tw)
                clear = T.crop(clear, i, j, th, tw)
                
                # Apply augmentation
                corrupt, clear = self._apply_optimized_augmentation(corrupt, clear)
            
            corrupt_image_name = os.path.basename(corrupt_image_path)
            
            return corrupt, clear, corrupt_image_name, h, w
            
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            # Return default tensors to avoid training interruption
            return (torch.zeros(3, 256, 256), torch.zeros(3, 256, 256), 
                   f"error_{idx}.png", 256, 256)


# Create alias for backward compatibility
TrainDataset = OptimizedTrainDataset


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """Optimized RGB to Y channel conversion with improved precision"""
    # Using ITU-R BT.601 luma coefficients for better accuracy
    rgb_to_grey = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True)


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 255.0) -> torch.Tensor:
    """Enhanced PSNR calculation with numerical stability"""
    x, y = x / data_range, y / data_range
    
    # Add small epsilon for numerical stability
    mse = torch.mean((x - y) ** 2) + 1e-8
    score = -10 * torch.log10(mse)
    
    # Clamp to reasonable range
    return torch.clamp(score, 0, 100)


def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5, 
         data_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    """Enhanced SSIM calculation with optimizations"""
    
    x, y = x / data_range, y / data_range
    
    # Dynamic downsampling for large images to improve speed
    original_size = min(x.size()[-2:])
    if original_size > 512:
        scale_factor = 512 / original_size
        x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        y = F.interpolate(y, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    
    # Generate Gaussian kernel more efficiently
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * kernel_sigma ** 2))
    g = g / g.sum()
    
    # Create 2D kernel
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    kernel = kernel.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    # SSIM computation with optimized convolutions
    c1, c2 = k1 ** 2, k2 ** 2
    n_channels = x.size(1)
    
    # Use groups for faster convolution
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=kernel_size//2, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=kernel_size//2, groups=n_channels)
    
    mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    
    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=kernel_size//2, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=kernel_size//2, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=kernel_size//2, groups=n_channels) - mu_xy
    
    # SSIM calculation
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    
    ssim_map = numerator / (denominator + 1e-8)  # Add epsilon for stability
    
    return ssim_map.mean()


class OptimizedVGGPerceptualLoss(torch.nn.Module):
    """Enhanced VGG perceptual loss with optimizations and caching"""
    
    def __init__(self, resize: bool = True, feature_layers: List[int] = [0, 1, 2, 3], 
                 style_layers: List[int] = [], weights: List[float] = None):
        super(OptimizedVGGPerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16 efficiently
        vgg = torchvision.models.vgg16(pretrained=True)
        
        # Extract feature blocks
        self.blocks = torch.nn.ModuleList([
            vgg.features[:4],   # relu1_2
            vgg.features[4:9],  # relu2_2
            vgg.features[9:16], # relu3_3
            vgg.features[16:23] # relu4_3
        ])
        
        # Freeze parameters for efficiency
        for block in self.blocks:
            block.eval()
            for param in block.parameters():
                param.requires_grad = False
        
        self.resize = resize
        self.feature_layers = feature_layers
        self.style_layers = style_layers
        self.weights = weights or [1.0] * len(feature_layers)
        
        # Normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Cache for intermediate features (optional optimization)
        self.cache_features = False
        self.feature_cache = {}
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized preprocessing with memory efficiency"""
        
        # Handle single channel inputs
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            # Take first 3 channels if more than 3
            x = x[:, :3]
        
        # Normalize
        x = (x - self.mean) / self.std
        
        # Resize if needed
        if self.resize and (x.shape[-1] != 224 or x.shape[-2] != 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from all layers efficiently"""
        
        features = []
        current = x
        
        for i, block in enumerate(self.blocks):
            current = block(current)
            if i in self.feature_layers or i in self.style_layers:
                features.append(current)
        
        return features
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, 
                feature_layers: Optional[List[int]] = None, 
                style_layers: Optional[List[int]] = None) -> torch.Tensor:
        """Enhanced forward pass with optimizations"""
        
        feature_layers = feature_layers or self.feature_layers
        style_layers = style_layers or self.style_layers
        
        # Preprocess inputs
        input = self.preprocess(input)
        target = self.preprocess(target)
        
        # Extract features
        input_features = self.extract_features(input)
        target_features = self.extract_features(target)
        
        loss = 0.0
        feature_idx = 0
        
        # Feature loss
        for i in range(len(self.blocks)):
            if i in feature_layers:
                weight = self.weights[feature_idx] if feature_idx < len(self.weights) else 1.0
                loss += weight * F.l1_loss(input_features[feature_idx], target_features[feature_idx])
                feature_idx += 1
            
            # Style loss (Gram matrix)
            if i in style_layers:
                input_feat = input_features[feature_idx - 1]
                target_feat = target_features[feature_idx - 1]
                
                # Reshape for Gram matrix computation
                b, c, h, w = input_feat.shape
                input_feat = input_feat.view(b, c, -1)
                target_feat = target_feat.view(b, c, -1)
                
                # Compute Gram matrices efficiently
                gram_input = torch.bmm(input_feat, input_feat.transpose(1, 2)) / (c * h * w)
                gram_target = torch.bmm(target_feat, target_feat.transpose(1, 2)) / (c * h * w)
                
                loss += F.l1_loss(gram_input, gram_target)
        
        return loss


# Create alias for backward compatibility
VGGPerceptualLoss = OptimizedVGGPerceptualLoss


class PerformanceMonitor:
    """Performance monitoring utility for optimization analysis"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        if self.enabled:
            self.start_times[name] = time.time()
    
    def end_timer(self, name: str):
        """End timing an operation"""
        if self.enabled and name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(elapsed)
            del self.start_times[name]
            return elapsed
        return 0
    
    def get_average_time(self, name: str) -> float:
        """Get average time for an operation"""
        if name in self.metrics and len(self.metrics[name]) > 0:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return 0
    
    def get_summary(self) -> dict:
        """Get summary of all metrics"""
        summary = {}
        for name, times in self.metrics.items():
            summary[name] = {
                'count': len(times),
                'total_time': sum(times),
                'average_time': sum(times) / len(times) if times else 0,
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0
            }
        return summary


class MemoryMonitor:
    """Memory usage monitoring for optimization"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.peak_memory = 0
        
    def get_memory_usage(self) -> Tuple[float, float, float]:
        """Get current memory usage (allocated, cached, total)"""
        if not self.enabled or not torch.cuda.is_available():
            return 0, 0, 0
        
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        self.peak_memory = max(self.peak_memory, allocated)
        
        return allocated, cached, total
    
    def get_memory_summary(self) -> dict:
        """Get memory usage summary"""
        allocated, cached, total = self.get_memory_usage()
        
        return {
            'current_allocated_gb': allocated,
            'current_cached_gb': cached,
            'total_gpu_memory_gb': total,
            'peak_allocated_gb': self.peak_memory,
            'memory_utilization': allocated / total if total > 0 else 0
        }
    
    def cleanup_memory(self):
        """Cleanup GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global instances for monitoring
performance_monitor = PerformanceMonitor()
memory_monitor = MemoryMonitor()


def get_optimization_summary() -> dict:
    """Get comprehensive optimization summary for analysis"""
    
    summary = {
        'performance_metrics': performance_monitor.get_summary(),
        'memory_metrics': memory_monitor.get_memory_summary(),
        'cuda_info': {}
    }
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        summary['cuda_info'] = {
            'device_name': props.name,
            'compute_capability': f"{props.major}.{props.minor}",
            'total_memory_gb': props.total_memory / 1e9,
            'multiprocessor_count': props.multi_processor_count
        }
    
    return summary


def create_optimized_dataloader(dataset: OptimizedTrainDataset, batch_size: int, 
                            shuffle: bool = True, num_workers: int = 4,
                            pin_memory: bool = True, prefetch_factor: int = 2,
                            persistent_workers: bool = True) -> torch.utils.data.DataLoader:
    """Create optimized DataLoader with best performance settings"""
    
    # Adjust workers based on dataset size and system capabilities
    if len(dataset) < 100:
        num_workers = min(2, num_workers)  # Reduce workers for small datasets
    
    # Disable persistent workers if no workers
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None
    
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=True if shuffle else False,  # Drop last for training consistency
        collate_fn=None,  # Use default collate function
        timeout=60 if num_workers > 0 else 0  # Timeout for worker processes
    )