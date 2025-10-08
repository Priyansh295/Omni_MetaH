import os
import time
import argparse
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model_directional_query_od import Inpainting
from utils_train import OptimizedTrainDataset, rgb_to_y, psnr, ssim, PerformanceMonitor, MemoryMonitor
from thop import profile
from ptflops import get_model_complexity_info
import json


class OptimizedTestConfig:
    """Enhanced test configuration with performance optimization"""
    
    def __init__(self, args):
        self.data_path_test = args.data_path_test
        self.task_name = args.task_name
        self.dataset_name = args.dataset_name
        self.num_iter = args.num_iter
        self.workers = args.workers
        self.model_file = args.model_file
        
        # Optimization parameters
        self.mixed_precision = getattr(args, 'mixed_precision', True)
        self.batch_processing = getattr(args, 'batch_processing', False)
        self.save_results = getattr(args, 'save_results', True)
        self.enable_profiling = getattr(args, 'enable_profiling', True)
        self.memory_monitoring = getattr(args, 'memory_monitoring', True)
        
        # Quality assessment
        self.compute_metrics = getattr(args, 'compute_metrics', True)
        self.save_comparison = getattr(args, 'save_comparison', False)
        
        # Metaheuristic analysis
        self.analyze_architecture = getattr(args, 'analyze_architecture', False)
        self.benchmark_performance = getattr(args, 'benchmark_performance', True)


def parse_test_args():
    """Enhanced argument parser for testing with optimization features"""
    
    desc = 'Enhanced Testing for Blind Image Inpainting with Performance Analysis'
    parser = argparse.ArgumentParser(description=desc)
    
    # Basic test parameters
    parser.add_argument('--data_path_test', type=str, default='./datasets/',
                       help='Test data path')
    parser.add_argument('--task_name', type=str, default='inpaint', 
                       choices=['inpaint'],
                       help='Task name')
    parser.add_argument('--dataset_name', type=str, default='places',
                       help='Dataset name for saving results')
    parser.add_argument('--model_file', type=str, default='./checkpoints/places.pth', 
                       help='Path of pre-trained model file')
    parser.add_argument('--num_iter', type=int, default=700000, 
                       help='Training iterations (for reference)')
    parser.add_argument('--workers', type=int, default=2, 
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=-1, 
                       help='Random seed (-1 for no manual seed)')
    
    # Model architecture parameters (for metaheuristic analysis)
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[4, 6, 6, 8],
                       help='Number of transformer blocks')
    parser.add_argument('--num_heads', nargs='+', type=int, default=[1, 2, 4, 8],
                       help='Number of attention heads')
    parser.add_argument('--channels', nargs='+', type=int, default=[48//3, 96//3, 192//3, 384//3],
                       help='Number of channels')
    parser.add_argument('--num_refinement', type=int, default=4,
                       help='Number of refinement layers')
    parser.add_argument('--expansion_factor', type=float, default=2.66,
                       help='Expansion factor')
    
    # Optimization parameters
    parser.add_argument('--mixed_precision', type=bool, default=True,
                       help='Use mixed precision inference')
    parser.add_argument('--batch_processing', type=bool, default=False,
                       help='Process multiple images in batch')
    parser.add_argument('--save_results', type=bool, default=True,
                       help='Save inference results')
    parser.add_argument('--enable_profiling', type=bool, default=True,
                       help='Enable performance profiling')
    parser.add_argument('--memory_monitoring', type=bool, default=True,
                       help='Enable memory monitoring')
    
    # Quality assessment
    parser.add_argument('--compute_metrics', type=bool, default=True,
                       help='Compute quality metrics (PSNR, SSIM)')
    parser.add_argument('--save_comparison', type=bool, default=False,
                       help='Save side-by-side comparisons')
    
    # Analysis parameters
    parser.add_argument('--analyze_architecture', type=bool, default=False,
                       help='Analyze model architecture complexity')
    parser.add_argument('--benchmark_performance', type=bool, default=True,
                       help='Benchmark inference performance')
    
    return init_test_args(parser.parse_args())


def init_test_args(args):
    """Initialize test arguments with optimizations"""
    
    # Create results directory
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    
    # Create subdirectories for organized output
    subdirs = ['inference_results', 'performance_analysis', 'quality_metrics', 'comparisons']
    for subdir in subdirs:
        os.makedirs(f'./results/{subdir}', exist_ok=True)
    
    # Set random seed for reproducible results
    if args.seed >= 0:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Test random seed set to: {args.seed}")
    else:
        # Enable benchmark for better performance
        torch.backends.cudnn.benchmark = True
    
    # CUDA optimizations for inference
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("CUDA optimization enabled for inference")
    
    return OptimizedTestConfig(args)


class TestDataset(OptimizedTrainDataset):
    """Optimized test dataset with enhanced loading and error handling"""
    
    def __init__(self, data_path: str, data_path_test: str, task_name: str, data_type: str, 
                 length: Optional[int] = None):
        # Initialize with test-specific optimizations
        super().__init__(data_path, data_path_test, task_name, data_type, 
                        patch_size=None, length=length, use_advanced_aug=False)
        
        # Override with test-specific settings
        self.data_type = data_type
        self.task_name = task_name
        
        # Use test images
        if data_type == 'test':
            self.corrupted_images = self.corrupt_images_test
            self.target_images = self.clear_images_test
            self.num = self.num_test
        else:
            self.corrupted_images = self.corrupt_images
            self.target_images = self.clear_images
        
        self.sample_num = length if length else self.num
        
        print(f"Test dataset: {self.sample_num} samples from {self.num} available")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int, int]:
        """Optimized getitem for testing with enhanced error handling"""
        
        try:
            corrupted_image_path = self.corrupted_images[idx % self.num]
            target_image_path = self.target_images[idx % self.num]
            
            corrupted_image_name = os.path.basename(corrupted_image_path)
            target_image_name = os.path.basename(target_image_path)
            
            # Load and process images for testing
            if self.task_name == 'inpaint':
                # Enhanced image loading with consistent sizing
                corrupted = self._load_and_resize_image(corrupted_image_path, (256, 256))
                target = self._load_and_resize_image(target_image_path, (256, 256))
            else:
                # Handle other tasks
                corrupted = self._load_and_resize_image(corrupted_image_path, (256, 256))
                target = self._load_and_resize_image(target_image_path, (256, 256))
            
            h, w = corrupted.shape[1:]
            
            return corrupted, target, corrupted_image_name, h, w
            
        except Exception as e:
            print(f"Error loading test sample {idx}: {e}")
            # Return default tensors
            return (torch.zeros(3, 256, 256), torch.zeros(3, 256, 256), 
                   f"error_{idx}.png", 256, 256)
    
    def _load_and_resize_image(self, image_path: str, target_size: Tuple[int, int]) -> torch.Tensor:
        """Load and resize image with optimization"""
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB and handle EXIF
                img = img.convert('RGB')
                img = ImageOps.exif_transpose(img)
                
                # Resize with high quality
                img = img.resize(target_size, Image.LANCZOS)
                
                return torch.clamp(torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0, 0, 1)
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros(3, target_size[0], target_size[1])


class EnhancedInferencePipeline:
    """Enhanced inference pipeline with performance optimization and analysis"""
    
    def __init__(self, config: OptimizedTestConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(enabled=config.enable_profiling)
        self.memory_monitor = MemoryMonitor(enabled=config.memory_monitoring)
        
        # Results storage
        self.results = {
            'image_names': [],
            'psnr_values': [],
            'ssim_values': [],
            'inference_times': [],
            'memory_usage': []
        }
        
        # Quality metrics
        self.quality_metrics = {
            'total_psnr': 0.0,
            'total_ssim': 0.0,
            'count': 0,
            'best_psnr': 0.0,
            'best_ssim': 0.0,
            'worst_psnr': float('inf'),
            'worst_ssim': 0.0
        }
    
    def load_model(self, model_file: str, num_blocks: List[int], num_heads: List[int], 
                  channels: List[int], num_refinement: int, expansion_factor: float) -> Inpainting:
        """Load model with enhanced error handling and optimization"""
        
        print("Loading model...")
        self.performance_monitor.start_timer('model_loading')
        
        try:
            # Create model with specified architecture
            model = Inpainting(
                num_blocks=num_blocks,
                num_heads=num_heads,
                channels=channels,
                num_refinement=num_refinement,
                expansion_factor=expansion_factor,
                use_checkpoint=False  # Disable checkpointing for inference
            ).to(self.device)
            
            # Load state dict with error handling
            if os.path.exists(model_file):
                try:
                    state_dict = torch.load(model_file, map_location=self.device)
                    
                    # Handle different checkpoint formats
                    if 'model_state_dict' in state_dict:
                        model.load_state_dict(state_dict['model_state_dict'])
                        print(f"Loaded model checkpoint from {model_file}")
                    else:
                        model.load_state_dict(state_dict)
                        print(f"Loaded model state dict from {model_file}")
                        
                except Exception as e:
                    print(f"Error loading model weights: {e}")
                    print("Using randomly initialized model")
            else:
                print(f"Model file {model_file} not found. Using randomly initialized model")
            
            # Set to evaluation mode
            model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Model loaded successfully:")
            print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
            print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
            
            loading_time = self.performance_monitor.end_timer('model_loading')
            print(f"  Loading time: {loading_time:.2f}s")
            
            return model
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def analyze_model_complexity(self, model: Inpainting, input_size: Tuple[int, int, int] = (3, 256, 256)):
        """Analyze model computational complexity for metaheuristic research"""
        
        if not self.config.analyze_architecture:
            return {}
        
        print("Analyzing model complexity...")
        self.performance_monitor.start_timer('complexity_analysis')
        
        try:
            # Create dummy input
            dummy_input = torch.randn(1, *input_size).to(self.device)
            
            # Analyze with thop
            try:
                flops, params = profile(model, inputs=(dummy_input,), verbose=False)
                flops_g = flops / 1e9
                params_m = params / 1e6
            except Exception as e:
                print(f"THOP analysis failed: {e}")
                flops_g, params_m = 0, 0
            
            # Analyze with ptflops
            try:
                ptflops_result = get_model_complexity_info(
                    model, input_size, print_per_layer_stat=False, as_strings=False
                )
                ptflops_gflops = ptflops_result[0] / 1e9
                ptflops_params_m = ptflops_result[1] / 1e6
            except Exception as e:
                print(f"ptflops analysis failed: {e}")
                ptflops_gflops, ptflops_params_m = 0, 0
            
            # Memory usage analysis
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
            
            complexity_info = {
                'flops_gflops': flops_g,
                'params_millions': params_m,
                'ptflops_gflops': ptflops_gflops,
                'ptflops_params_millions': ptflops_params_m,
                'model_size_mb': model_size_mb,
                'input_shape': input_size
            }
            
            print(f"Model Complexity Analysis:")
            print(f"  FLOPs: {flops_g:.2f} GFLOPs")
            print(f"  Parameters: {params_m:.2f}M")
            print(f"  Model size: {model_size_mb:.2f} MB")
            
            analysis_time = self.performance_monitor.end_timer('complexity_analysis')
            complexity_info['analysis_time'] = analysis_time
            
            return complexity_info
            
        except Exception as e:
            print(f"Complexity analysis failed: {e}")
            return {}
    
    def optimized_test_loop(self, model: Inpainting, data_loader: DataLoader, 
                          num_iter: int) -> Tuple[float, float]:
        """Enhanced test loop with comprehensive performance monitoring"""
        
        print("Starting optimized inference...")
        model.eval()
        
        total_psnr, total_ssim, count = 0.0, 0.0, 0
        inference_times = []
        
        # Reset quality metrics
        self.quality_metrics = {
            'total_psnr': 0.0,
            'total_ssim': 0.0,
            'count': 0,
            'best_psnr': 0.0,
            'best_ssim': 0.0,
            'worst_psnr': float('inf'),
            'worst_ssim': 1.0
        }
        
        with torch.no_grad():
            test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
            
            for batch_idx, (corrupted, target, name, h, w) in enumerate(test_bar):
                self.performance_monitor.start_timer('batch_processing')
                
                # Move to device with non-blocking transfer
                corrupted = corrupted.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # Monitor memory before inference
                mem_before = self.memory_monitor.get_memory_usage()
                
                # Timed inference with mixed precision
                inference_start = time.time()
                
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = model(corrupted)
                else:
                    output = model(corrupted)
                
                # Synchronize for accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # Post-process output
                output = torch.clamp(output[:, :, :h[0], :w[0]], 0, 1)
                target_resized = target[:, :, :h[0], :w[0]]
                
                # Convert to uint8 for saving
                output_uint8 = torch.clamp(output.mul(255), 0, 255).byte()
                target_uint8 = torch.clamp(target_resized.mul(255), 0, 255).byte()
                
                # Compute quality metrics
                if self.config.compute_metrics:
                    # Use Y channel for metrics as in training
                    y_output = rgb_to_y(output_uint8.double())
                    y_target = rgb_to_y(target_uint8.double())
                    
                    current_psnr = psnr(y_output, y_target)
                    current_ssim = ssim(y_output, y_target)
                    
                    # Update statistics
                    total_psnr += current_psnr.item()
                    total_ssim += current_ssim.item()
                    count += 1
                    
                    # Update quality metrics
                    self.quality_metrics['total_psnr'] += current_psnr.item()
                    self.quality_metrics['total_ssim'] += current_ssim.item()
                    self.quality_metrics['count'] += 1
                    self.quality_metrics['best_psnr'] = max(self.quality_metrics['best_psnr'], current_psnr.item())
                    self.quality_metrics['best_ssim'] = max(self.quality_metrics['best_ssim'], current_ssim.item())
                    self.quality_metrics['worst_psnr'] = min(self.quality_metrics['worst_psnr'], current_psnr.item())
                    self.quality_metrics['worst_ssim'] = min(self.quality_metrics['worst_ssim'], current_ssim.item())
                    
                    # Store per-image results
                    self.results['image_names'].append(name[0])
                    self.results['psnr_values'].append(current_psnr.item())
                    self.results['ssim_values'].append(current_ssim.item())
                    self.results['inference_times'].append(inference_time)
                    self.results['memory_usage'].append(mem_before[0])  # Allocated memory
                
                # Save results
                if self.config.save_results:
                    self._save_inference_result(output_uint8, name[0], batch_idx)
                
                # Save comparison if requested
                if self.config.save_comparison:
                    self._save_comparison(corrupted, output, target_resized, name[0])
                
                # Update progress bar
                if self.config.compute_metrics:
                    avg_psnr = total_psnr / count
                    avg_ssim = total_ssim / count
                    test_bar.set_description(
                        f'Test [{batch_idx+1}] PSNR: {avg_psnr:.2f} SSIM: {avg_ssim:.3f} '
                        f'Time: {inference_time*1000:.1f}ms'
                    )
                else:
                    test_bar.set_description(f'Test [{batch_idx+1}] Time: {inference_time*1000:.1f}ms')
                
                batch_time = self.performance_monitor.end_timer('batch_processing')
                
                # Memory cleanup periodically
                if batch_idx % 10 == 0:
                    self.memory_monitor.cleanup_memory()
        
        # Calculate final metrics
        if count > 0:
            avg_psnr = total_psnr / count
            avg_ssim = total_ssim / count
        else:
            avg_psnr = avg_ssim = 0.0
        
        # Performance summary
        if inference_times:
            avg_inference_time = sum(inference_times) / len(inference_times)
            fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
            
            print(f"\nInference Performance Summary:")
            print(f"  Average inference time: {avg_inference_time*1000:.2f} ms")
            print(f"  Throughput: {fps:.2f} FPS")
            print(f"  Total images processed: {len(inference_times)}")
            
            if self.config.compute_metrics:
                print(f"\nQuality Metrics Summary:")
                print(f"  Average PSNR: {avg_psnr:.2f} dB")
                print(f"  Average SSIM: {avg_ssim:.3f}")
                print(f"  Best PSNR: {self.quality_metrics['best_psnr']:.2f} dB")
                print(f"  Best SSIM: {self.quality_metrics['best_ssim']:.3f}")
        
        return avg_psnr, avg_ssim
    
    def _save_inference_result(self, output: torch.Tensor, filename: str, batch_idx: int):
        """Save inference result with optimization"""
        
        try:
            dataset_name = f'results/{self.config.dataset_name}/{filename}'
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(dataset_name), exist_ok=True)
            
            # Convert to numpy and save
            output_np = output.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()
            
            # Save with optimization
            Image.fromarray(output_np).save(dataset_name, optimize=True, quality=95)
            
        except Exception as e:
            print(f"Error saving result for {filename}: {e}")
    
    def _save_comparison(self, corrupted: torch.Tensor, output: torch.Tensor, 
                        target: torch.Tensor, filename: str):
        """Save side-by-side comparison"""
        
        try:
            # Create comparison image
            comparison_dir = f'results/comparisons/{self.config.dataset_name}'
            os.makedirs(comparison_dir, exist_ok=True)
            
            # Convert tensors to numpy
            corrupted_np = corrupted[0].permute(1, 2, 0).cpu().numpy()
            output_np = output[0].permute(1, 2, 0).cpu().numpy()
            target_np = target[0].permute(1, 2, 0).cpu().numpy()
            
            # Create side-by-side comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(corrupted_np)
            axes[0].set_title('Input (Corrupted)')
            axes[0].axis('off')
            
            axes[1].imshow(output_np)
            axes[1].set_title('Output (Inpainted)')
            axes[1].axis('off')
            
            axes[2].imshow(target_np)
            axes[2].set_title('Target (Ground Truth)')
            axes[2].axis('off')
            
            plt.tight_layout()
            comparison_path = os.path.join(comparison_dir, f'comparison_{filename}')
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error saving comparison for {filename}: {e}")
    
    def save_performance_analysis(self, complexity_info: Dict):
        """Save comprehensive performance analysis for research"""
        
        analysis_dir = 'results/performance_analysis'
        
        # Performance metrics summary
        performance_summary = {
            'inference_performance': {
                'average_inference_time_ms': np.mean(self.results['inference_times']) * 1000 if self.results['inference_times'] else 0,
                'std_inference_time_ms': np.std(self.results['inference_times']) * 1000 if self.results['inference_times'] else 0,
                'min_inference_time_ms': np.min(self.results['inference_times']) * 1000 if self.results['inference_times'] else 0,
                'max_inference_time_ms': np.max(self.results['inference_times']) * 1000 if self.results['inference_times'] else 0,
                'throughput_fps': 1.0 / np.mean(self.results['inference_times']) if self.results['inference_times'] else 0,
                'total_images': len(self.results['inference_times'])
            },
            'quality_metrics': {
                'average_psnr': self.quality_metrics['total_psnr'] / max(1, self.quality_metrics['count']),
                'average_ssim': self.quality_metrics['total_ssim'] / max(1, self.quality_metrics['count']),
                'best_psnr': self.quality_metrics['best_psnr'],
                'best_ssim': self.quality_metrics['best_ssim'],
                'worst_psnr': self.quality_metrics['worst_psnr'] if self.quality_metrics['worst_psnr'] != float('inf') else 0,
                'worst_ssim': self.quality_metrics['worst_ssim'],
                'psnr_std': np.std(self.results['psnr_values']) if self.results['psnr_values'] else 0,
                'ssim_std': np.std(self.results['ssim_values']) if self.results['ssim_values'] else 0
            },
            'memory_usage': self.memory_monitor.get_memory_summary(),
            'model_complexity': complexity_info,
            'system_info': {
                'cuda_available': torch.cuda.is_available(),
                'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                'mixed_precision_enabled': self.config.mixed_precision
            }
        }
        
        # Save JSON summary
        with open(os.path.join(analysis_dir, 'performance_summary.json'), 'w') as f:
            json.dump(performance_summary, f, indent=2)
        
        # Save detailed CSV results
        if self.results['image_names']:
            results_df = pd.DataFrame({
                'image_name': self.results['image_names'],
                'psnr': self.results['psnr_values'],
                'ssim': self.results['ssim_values'],
                'inference_time_ms': [t * 1000 for t in self.results['inference_times']],
                'memory_usage_gb': self.results['memory_usage']
            })
            results_df.to_csv(os.path.join(analysis_dir, 'detailed_results.csv'), index=False)
        
        # Create performance visualizations
        self._create_performance_plots(analysis_dir)
        
        print(f"Performance analysis saved to {analysis_dir}")
        return performance_summary
    
    def _create_performance_plots(self, save_dir: str):
        """Create performance visualization plots"""
        
        try:
            if not self.results['inference_times']:
                return
            
            # Performance distribution plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Inference time distribution
            ax1.hist(np.array(self.results['inference_times']) * 1000, bins=20, alpha=0.7, color='blue')
            ax1.set_xlabel('Inference Time (ms)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Inference Time Distribution')
            ax1.grid(True, alpha=0.3)
            
            # PSNR distribution
            if self.results['psnr_values']:
                ax2.hist(self.results['psnr_values'], bins=20, alpha=0.7, color='green')
                ax2.set_xlabel('PSNR (dB)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('PSNR Distribution')
                ax2.grid(True, alpha=0.3)
            
            # SSIM distribution
            if self.results['ssim_values']:
                ax3.hist(self.results['ssim_values'], bins=20, alpha=0.7, color='red')
                ax3.set_xlabel('SSIM')
                ax3.set_ylabel('Frequency')
                ax3.set_title('SSIM Distribution')
                ax3.grid(True, alpha=0.3)
            
            # Performance vs Quality scatter
            if self.results['psnr_values'] and self.results['inference_times']:
                ax4.scatter(np.array(self.results['inference_times']) * 1000, 
                           self.results['psnr_values'], alpha=0.6)
                ax4.set_xlabel('Inference Time (ms)')
                ax4.set_ylabel('PSNR (dB)')
                ax4.set_title('Performance vs Quality')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Time series plot if enough data
            if len(self.results['inference_times']) > 10:
                plt.figure(figsize=(12, 6))
                plt.plot(np.array(self.results['inference_times']) * 1000, 'b-', alpha=0.7, label='Inference Time')
                plt.xlabel('Image Index')
                plt.ylabel('Inference Time (ms)')
                plt.title('Inference Time Over Test Set')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'inference_time_series.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Error creating performance plots: {e}")


def run_optimized_test(args):
    """Main optimized testing function with comprehensive analysis"""
    
    print("="*80)
    print("ENHANCED BLIND INPAINTING TESTING WITH PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Initialize inference pipeline
    config = args
    pipeline = EnhancedInferencePipeline(config)
    
    # Create optimized test dataset
    test_dataset = TestDataset(
        config.data_path_test, 
        config.data_path_test, 
        config.task_name, 
        'test'
    )
    
    # Create optimized data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=config.workers,
        pin_memory=True and torch.cuda.is_available(),
        prefetch_factor=2 if config.workers > 0 else None
    )
    
    # Load model with architecture analysis
    model = pipeline.load_model(
        config.model_file,
        config.num_blocks,
        config.num_heads,
        config.channels,
        config.num_refinement,
        config.expansion_factor
    )
    
    # Analyze model complexity for metaheuristic research
    complexity_info = pipeline.analyze_model_complexity(model)
    
    # Run optimized inference
    avg_psnr, avg_ssim = pipeline.optimized_test_loop(model, test_loader, 1)
    
    # Save comprehensive analysis
    performance_summary = pipeline.save_performance_analysis(complexity_info)
    
    # Print final summary
    print("\n" + "="*80)
    print("TESTING COMPLETED - FINAL SUMMARY")
    print("="*80)
    
    if config.compute_metrics:
        print(f"Quality Metrics:")
        print(f"  Average PSNR: {avg_psnr:.2f} dB")
        print(f"  Average SSIM: {avg_ssim:.3f}")
    
    if pipeline.results['inference_times']:
        avg_time = np.mean(pipeline.results['inference_times']) * 1000
        fps = 1000 / avg_time if avg_time > 0 else 0
        print(f"Performance Metrics:")
        print(f"  Average inference time: {avg_time:.2f} ms")
        print(f"  Throughput: {fps:.2f} FPS")
    
    if complexity_info:
        print(f"Model Complexity:")
        print(f"  FLOPs: {complexity_info.get('flops_gflops', 0):.2f} GFLOPs")
        print(f"  Parameters: {complexity_info.get('params_millions', 0):.2f}M")
    
    print(f"Results saved to: ./results/")
    
    return performance_summary


if __name__ == '__main__':
    # Parse arguments and run optimized testing
    args = parse_test_args()
    
    print(f"Testing with configuration:")
    print(f"  Model file: {args.model_file}")
    print(f"  Test data: {args.data_path_test}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Workers: {args.workers}")
    print(f"  Compute metrics: {args.compute_metrics}")
    print(f"  Architecture analysis: {args.analyze_architecture}")
    
    # Run testing
    performance_summary = run_optimized_test(args)
    
    print("\nTesting completed successfully!")
    print("Check ./results/ directory for detailed analysis and visualizations.")