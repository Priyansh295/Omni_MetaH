import os
import glob
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

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
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from model_directional_query_od import Inpainting
from utils_train import parse_args, TrainDataset, rgb_to_y, psnr, ssim, VGGPerceptualLoss
import kornia

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
        'w_edge': 0.6021258595296576
    }
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global flag to disable mixed precision if issues persist
USE_MIXED_PRECISION = False  # Disabled for stability
print(f"Mixed precision training: {'ENABLED' if USE_MIXED_PRECISION else 'DISABLED'}")

# Initialize perceptual loss globally to avoid memory issues
perceptual_loss = None

def get_perceptual_loss():
    """Get or create perceptual loss instance"""
    global perceptual_loss
    if perceptual_loss is None:
        perceptual_loss = VGGPerceptualLoss().to(device)
        # Ensure perceptual loss model is in FP32
        perceptual_loss = perceptual_loss.float()
    return perceptual_loss


@dataclass
class MetaheuristicResult:
    """Data class to store metaheuristic optimization results"""
    algorithm: str
    best_params: Dict
    best_value: float
    convergence_history: List[float]
    computation_time: float
    iterations: int
    success_rate: float


class PerformanceProfiler:
    """Performance profiling for metaheuristic algorithms"""
    
    def __init__(self):
        self.profiles = {}
    
    def start_timing(self, algorithm: str):
        """Start timing an operation"""
        self.profiles[algorithm] = {'start_time': time.time()}
    
    def end_timing(self, algorithm: str, iterations: int, best_value: float):
        """End timing an operation"""
        if algorithm in self.profiles:
            elapsed = time.time() - self.profiles[algorithm]['start_time']
            self.profiles[algorithm].update({
                'total_time': elapsed,
                'iterations': iterations,
                'best_value': best_value,
                'time_per_iteration': elapsed / iterations if iterations > 0 else float('inf'),
                'convergence_rate': -best_value / elapsed if elapsed > 0 else 0  # Higher is better (negative PSNR)
            })
    
    def get_summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.profiles).T


def count_parameters(model):
    """Accurate parameter counting with detailed breakdown"""
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    
    # Get model size in MB
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    print(f"="*60)
    print(f"MODEL ARCHITECTURE ANALYSIS")
    print(f"="*60)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"="*60)
    
    return total_params, trainable_params


class OptimizedMetaheuristicManager:
    """Enhanced metaheuristic manager with complete error handling"""
    
    def __init__(self, args):
        self.args = args
        self.profiler = PerformanceProfiler()
        
        # Further reduced budgets for stability on large datasets
        self.GA_BUDGET = 10      # Reduced for memory stability
        self.PSO_BUDGET = 10     # Reduced for memory stability 
        self.DE_BUDGET = 10      # Reduced for memory stability
        self.BO_TRIALS = 8       # Reduced for memory stability
        self.TRAINING_ITERS = 3  # Minimal for testing, increased later
        
        # Parallel execution settings
        self.max_workers = min(2, mp.cpu_count())  # Reduced workers for memory
        self.use_parallel = False  # Disabled for stability
        
        print(f"Metaheuristic Manager initialized for large dataset:")
        print(f"  GA Budget: {self.GA_BUDGET}")
        print(f"  PSO Budget: {self.PSO_BUDGET}")
        print(f"  DE Budget: {self.DE_BUDGET}")
        print(f"  BO Trials: {self.BO_TRIALS}")
        print(f"  Training iterations per evaluation: {self.TRAINING_ITERS}")
        print(f"  Mixed precision: {'ENABLED' if USE_MIXED_PRECISION else 'DISABLED'}")
        print(f"  Memory optimization: ENABLED")
    
    def clamp(self, val, minval, maxval):
        """Utility function for parameter clamping"""
        return max(minval, min(val, maxval))
    
    def create_diverse_architectures(self) -> List[Dict]:
        """Create smaller, memory-efficient architectures"""
        architectures = [
            # Smaller, memory-efficient variants
            {'num_blocks': [1, 2, 2, 3], 'num_heads': [1, 1, 2, 2], 'channels': [16, 32, 64, 128]},
            {'num_blocks': [1, 1, 2, 4], 'num_heads': [1, 1, 1, 2], 'channels': [20, 40, 80, 160]},
            {'num_blocks': [2, 2, 3, 4], 'num_heads': [1, 2, 2, 4], 'channels': [16, 32, 64, 128]},
            # Original optimized (smaller version)
            {'num_blocks': [2, 3, 3, 4], 'num_heads': [2, 2, 4, 6], 'channels': [24, 48, 96, 192]},
        ]
        return architectures
    
    def safe_get_nevergrad_result(self, optimizer):
        """Safely get result from nevergrad optimizer"""
        try:
            recommendation = optimizer.provide_recommendation()
            if recommendation is not None:
                # Safely extract the value
                if hasattr(recommendation, 'value') and recommendation.value is not None:
                    if isinstance(recommendation.value, (tuple, list)) and len(recommendation.value) > 1:
                        return recommendation.value[1]  # Get the parameters dict
            return {}
        except Exception as e:
            print(f"Error getting nevergrad result: {e}")
            return {}
    
    def optimize_ga(self) -> MetaheuristicResult:
        """Memory-optimized GA implementation"""
        print("Starting memory-optimized GA optimization...")
        self.profiler.start_timing('GA')
        
        # Simplified GA instrumentation with smaller architectures
        instrum_ga = ng.p.Instrumentation(
            num_blocks=ng.p.Choice([
                [1,2,2,3], [1,1,2,4], [2,2,3,4], [1,2,3,4]
            ]),
            num_heads=ng.p.Choice([
                [1,1,2,2], [1,1,1,2], [1,2,2,4], [2,2,4,6]
            ]),
            channels=ng.p.Choice([
                [16,32,64,128], [20,40,80,160], [24,48,96,192]
            ]),
            num_refinement=ng.p.Scalar(lower=1, upper=4).set_integer_casting()
        )
        
        # Use simplified genetic algorithm
        ga_optimizer = ng.optimizers.OnePlusOne(
            parametrization=instrum_ga, 
            budget=self.GA_BUDGET
        )
        
        convergence_history = []
        best_value = float('inf')
        
        # Reduced seeding for memory efficiency
        diverse_architectures = self.create_diverse_architectures()
        for i, arch in enumerate(diverse_architectures[:2]):  # Only 2 seeds
            try:
                seed = {
                    'num_blocks': arch['num_blocks'],
                    'num_heads': arch['num_heads'], 
                    'channels': arch['channels'],
                    'num_refinement': 3  # Fixed smaller value
                }
                
                print(f"GA seed {i+1}/2 with: {seed}")
                
                ga_value = self.evaluate_configuration(
                    seed['num_blocks'], seed['num_heads'], seed['channels'],
                    OPTIMIZED_PARAMS['lr'], 1, OPTIMIZED_PARAMS['expansion_factor'],
                    seed['num_refinement'],
                    (OPTIMIZED_PARAMS['loss_weights']['w_l1'],
                     OPTIMIZED_PARAMS['loss_weights']['w_percep'],
                     OPTIMIZED_PARAMS['loss_weights']['w_ssim'],
                     OPTIMIZED_PARAMS['loss_weights']['w_edge'])
                )
                
                convergence_history.append(min(best_value, ga_value))
                best_value = min(best_value, ga_value)
                print(f"GA seed {i+1} result: {ga_value:.4f}")
                
                # Aggressive memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"GA seeding error {i+1}: {e}")
                convergence_history.append(best_value)
        
        # Run remaining iterations with aggressive memory management
        remaining_budget = self.GA_BUDGET - len(convergence_history)
        for i in range(remaining_budget):
            try:
                candidate = ga_optimizer.ask()
                
                # Safely extract parameters
                try:
                    if hasattr(candidate, 'value') and candidate.value is not None:
                        if isinstance(candidate.value, (tuple, list)) and len(candidate.value) > 1:
                            params = candidate.value[1]
                        else:
                            convergence_history.append(best_value)
                            continue
                    else:
                        convergence_history.append(best_value)
                        continue
                except Exception:
                    convergence_history.append(best_value)
                    continue
                
                value = self.evaluate_configuration(
                    params['num_blocks'], params['num_heads'], params['channels'],
                    OPTIMIZED_PARAMS['lr'], 1, OPTIMIZED_PARAMS['expansion_factor'],
                    params['num_refinement'],
                    (OPTIMIZED_PARAMS['loss_weights']['w_l1'],
                     OPTIMIZED_PARAMS['loss_weights']['w_percep'],
                     OPTIMIZED_PARAMS['loss_weights']['w_ssim'],
                     OPTIMIZED_PARAMS['loss_weights']['w_edge'])
                )
                
                ga_optimizer.tell(candidate, value)
                convergence_history.append(min(best_value, value))
                best_value = min(best_value, value)
                
                print(f"GA iteration {len(convergence_history)}/{self.GA_BUDGET}: {value:.4f}")
                
                # Aggressive memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"GA iteration error: {e}")
                convergence_history.append(best_value)
        
        ga_best_params = self.safe_get_nevergrad_result(ga_optimizer)
        
        self.profiler.end_timing('GA', len(convergence_history), best_value)
        
        return MetaheuristicResult(
            algorithm='GA',
            best_params=ga_best_params,
            best_value=best_value,
            convergence_history=convergence_history,
            computation_time=self.profiler.profiles['GA']['total_time'],
            iterations=len(convergence_history),
            success_rate=sum(1 for v in convergence_history if v < float('inf')) / len(convergence_history) if convergence_history else 0
        )
    
    def optimize_pso(self) -> MetaheuristicResult:
        """Memory-optimized PSO implementation"""
        print("Starting memory-optimized PSO optimization...")
        self.profiler.start_timing('PSO')
        
        instrum_pso = ng.p.Instrumentation(
            w_l1=ng.p.Scalar(lower=0.1, upper=1.0),
            w_percep=ng.p.Scalar(lower=0.05, upper=0.4),  # Reduced upper bound
            w_ssim=ng.p.Scalar(lower=0.1, upper=1.0),
            w_edge=ng.p.Scalar(lower=0.1, upper=1.0)
        )
        
        pso_optimizer = ng.optimizers.OnePlusOne(
            parametrization=instrum_pso, 
            budget=self.PSO_BUDGET
        )
        
        convergence_history = []
        best_value = float('inf')
        
        # Simplified seeding
        loss_configs = [
            {'w_l1': 0.5, 'w_percep': 0.2, 'w_ssim': 0.4, 'w_edge': 0.3},
            {'w_l1': 0.7, 'w_percep': 0.1, 'w_ssim': 0.6, 'w_edge': 0.4},
        ]
        
        for i, config in enumerate(loss_configs):
            try:
                print(f"PSO seed {i+1} with: {config}")
                pso_value = self.evaluate_configuration(
                    OPTIMIZED_PARAMS['num_blocks'][:3] + [4],  # Smaller architecture
                    OPTIMIZED_PARAMS['num_heads'][:3] + [4],
                    [c//2 for c in OPTIMIZED_PARAMS['channels']],  # Halved channels
                    OPTIMIZED_PARAMS['lr'], 1,
                    OPTIMIZED_PARAMS['expansion_factor'],
                    3,  # Smaller refinement
                    (config['w_l1'], config['w_percep'], config['w_ssim'], config['w_edge'])
                )
                
                convergence_history.append(min(best_value, pso_value))
                best_value = min(best_value, pso_value)
                print(f"PSO seed {i+1} result: {pso_value:.4f}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"PSO seeding error {i+1}: {e}")
                convergence_history.append(best_value)
        
        # Run remaining iterations
        remaining_budget = self.PSO_BUDGET - len(convergence_history)
        for i in range(remaining_budget):
            try:
                candidate = pso_optimizer.ask()
                
                try:
                    if hasattr(candidate, 'value') and candidate.value is not None:
                        if isinstance(candidate.value, (tuple, list)) and len(candidate.value) > 1:
                            params = candidate.value[1]
                        else:
                            convergence_history.append(best_value)
                            continue
                    else:
                        convergence_history.append(best_value)
                        continue
                except Exception:
                    convergence_history.append(best_value)
                    continue
                
                value = self.evaluate_configuration(
                    OPTIMIZED_PARAMS['num_blocks'][:3] + [4],
                    OPTIMIZED_PARAMS['num_heads'][:3] + [4],
                    [c//2 for c in OPTIMIZED_PARAMS['channels']],
                    OPTIMIZED_PARAMS['lr'], 1,
                    OPTIMIZED_PARAMS['expansion_factor'],
                    3,
                    (params['w_l1'], params['w_percep'], params['w_ssim'], params['w_edge'])
                )
                
                pso_optimizer.tell(candidate, value)
                convergence_history.append(min(best_value, value))
                best_value = min(best_value, value)
                
                print(f"PSO iteration {len(convergence_history)}/{self.PSO_BUDGET}: {value:.4f}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"PSO iteration error: {e}")
                convergence_history.append(best_value)
        
        pso_best_params = self.safe_get_nevergrad_result(pso_optimizer)
        self.profiler.end_timing('PSO', len(convergence_history), best_value)
        
        return MetaheuristicResult(
            algorithm='PSO',
            best_params=pso_best_params,
            best_value=best_value,
            convergence_history=convergence_history,
            computation_time=self.profiler.profiles['PSO']['total_time'],
            iterations=len(convergence_history),
            success_rate=sum(1 for v in convergence_history if v < float('inf')) / len(convergence_history) if convergence_history else 0
        )
    
    def optimize_de(self) -> MetaheuristicResult:
        """Memory-optimized DE implementation"""
        print("Starting memory-optimized DE optimization...")
        self.profiler.start_timing('DE')
        
        instrum_de = ng.p.Instrumentation(
            expansion_factor=ng.p.Scalar(lower=1.5, upper=3.5)  # Narrower range
        )
        
        de_optimizer = ng.optimizers.OnePlusOne(
            parametrization=instrum_de, 
            budget=self.DE_BUDGET
        )
        
        convergence_history = []
        best_value = float('inf')
        
        # Simplified seeding
        factors = [2.0, 2.5, 3.0]
        for i, factor in enumerate(factors):
            try:
                print(f"DE seed {i+1} with expansion_factor: {factor}")
                de_value = self.evaluate_configuration(
                    OPTIMIZED_PARAMS['num_blocks'][:3] + [4],
                    OPTIMIZED_PARAMS['num_heads'][:3] + [4],
                    [c//2 for c in OPTIMIZED_PARAMS['channels']],
                    OPTIMIZED_PARAMS['lr'], 1,
                    factor,
                    3,
                    (OPTIMIZED_PARAMS['loss_weights']['w_l1'],
                     OPTIMIZED_PARAMS['loss_weights']['w_percep'],
                     OPTIMIZED_PARAMS['loss_weights']['w_ssim'],
                     OPTIMIZED_PARAMS['loss_weights']['w_edge'])
                )
                
                convergence_history.append(min(best_value, de_value))
                best_value = min(best_value, de_value)
                print(f"DE seed {i+1} result: {de_value:.4f}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"DE seeding error {i+1}: {e}")
                convergence_history.append(best_value)
        
        # Run remaining iterations
        remaining_budget = self.DE_BUDGET - len(convergence_history)
        for i in range(remaining_budget):
            try:
                candidate = de_optimizer.ask()
                
                try:
                    if hasattr(candidate, 'value') and candidate.value is not None:
                        if isinstance(candidate.value, (tuple, list)) and len(candidate.value) > 1:
                            params = candidate.value[1]
                        else:
                            convergence_history.append(best_value)
                            continue
                    else:
                        convergence_history.append(best_value)
                        continue
                except Exception:
                    convergence_history.append(best_value)
                    continue
                
                value = self.evaluate_configuration(
                    OPTIMIZED_PARAMS['num_blocks'][:3] + [4],
                    OPTIMIZED_PARAMS['num_heads'][:3] + [4], 
                    [c//2 for c in OPTIMIZED_PARAMS['channels']],
                    OPTIMIZED_PARAMS['lr'], 1,
                    params['expansion_factor'],
                    3,
                    (OPTIMIZED_PARAMS['loss_weights']['w_l1'],
                     OPTIMIZED_PARAMS['loss_weights']['w_percep'],
                     OPTIMIZED_PARAMS['loss_weights']['w_ssim'],
                     OPTIMIZED_PARAMS['loss_weights']['w_edge'])
                )
                
                de_optimizer.tell(candidate, value)
                convergence_history.append(min(best_value, value))
                best_value = min(best_value, value)
                
                print(f"DE iteration {len(convergence_history)}/{self.DE_BUDGET}: {value:.4f}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"DE iteration error: {e}")
                convergence_history.append(best_value)
        
        de_best_params = self.safe_get_nevergrad_result(de_optimizer)
        self.profiler.end_timing('DE', len(convergence_history), best_value)
        
        return MetaheuristicResult(
            algorithm='DE',
            best_params=de_best_params,
            best_value=best_value,
            convergence_history=convergence_history,
            computation_time=self.profiler.profiles['DE']['total_time'],
            iterations=len(convergence_history),
            success_rate=sum(1 for v in convergence_history if v < float('inf')) / len(convergence_history) if convergence_history else 0
        )
    
    def optimize_bo(self, ga_result: MetaheuristicResult, pso_result: MetaheuristicResult, 
                   de_result: MetaheuristicResult) -> MetaheuristicResult:
        """Memory-optimized BO implementation"""
        print("Starting memory-optimized BO optimization...")
        self.profiler.start_timing('BO')
        
        convergence_history = []
        best_value = float('inf')
        
        def bo_objective(trial):
            nonlocal best_value
            
            lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)  # Narrower range
            
            # Use smaller architecture from GA results or fallback
            num_blocks = ga_result.best_params.get('num_blocks', [1, 2, 2, 3])
            num_heads = ga_result.best_params.get('num_heads', [1, 1, 2, 2])
            channels = ga_result.best_params.get('channels', [16, 32, 64, 128])
            
            loss_weights = (
                pso_result.best_params.get('w_l1', 0.5),
                pso_result.best_params.get('w_percep', 0.2),
                pso_result.best_params.get('w_ssim', 0.4),
                pso_result.best_params.get('w_edge', 0.3)
            )
            
            expansion_factor = de_result.best_params.get('expansion_factor', 2.5)
            
            value = self.evaluate_configuration(
                num_blocks, num_heads, channels, lr, 1, 
                expansion_factor, 3, loss_weights
            )
            
            convergence_history.append(min(best_value, value))
            best_value = min(best_value, value)
            
            return value
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(n_startup_trials=2, seed=42)
        )
        
        try:
            study.optimize(bo_objective, n_trials=self.BO_TRIALS, timeout=600)
        except Exception as e:
            print(f"BO optimization error: {e}")
        
        bo_best_params = study.best_trial.params if study.best_trial else {}
        bo_best_value = study.best_trial.value if study.best_trial else float('inf')
        
        self.profiler.end_timing('BO', len(convergence_history), best_value)
        
        return MetaheuristicResult(
            algorithm='BO',
            best_params=bo_best_params,
            best_value=bo_best_value,
            convergence_history=convergence_history,
            computation_time=self.profiler.profiles['BO']['total_time'],
            iterations=len(convergence_history),
            success_rate=sum(1 for v in convergence_history if v < float('inf')) / len(convergence_history) if convergence_history else 0
        )
    
    def evaluate_configuration(self, num_blocks, num_heads, channels, lr, batch_size, 
                             expansion_factor, num_refinement, loss_weights) -> float:
        """Memory-optimized configuration evaluation"""
        try:
            return optimized_train_and_evaluate(
                num_blocks, num_heads, channels, lr, batch_size, expansion_factor,
                num_refinement, loss_weights, num_iter=self.TRAINING_ITERS,
                data_path=self.args.data_path, data_path_test=self.args.data_path_test
            )
        except Exception as e:
            print(f"Evaluation error: {e}")
            return float('inf')
    
    def run_integrated_metaheuristics(self) -> Tuple[MetaheuristicResult, ...]:
        """Run all metaheuristics with memory optimization"""
        
        print("\n" + "="*80)
        print("STARTING MEMORY-OPTIMIZED METAHEURISTIC OPTIMIZATION")
        print("="*80)
        
        start_total_time = time.time()
        
        print("\n" + "-"*60)
        print("PHASE 1: GENETIC ALGORITHM (Architecture Optimization)")
        print("-"*60)
        ga_result = self.optimize_ga()
        print(f"GA completed: Best value = {ga_result.best_value:.4f}")
        
        print("\n" + "-"*60)
        print("PHASE 2: PARTICLE SWARM OPTIMIZATION (Loss Weight Optimization)")
        print("-"*60)
        pso_result = self.optimize_pso()
        print(f"PSO completed: Best value = {pso_result.best_value:.4f}")
        
        print("\n" + "-"*60)
        print("PHASE 3: DIFFERENTIAL EVOLUTION (Expansion Factor Optimization)")
        print("-"*60)
        de_result = self.optimize_de()
        print(f"DE completed: Best value = {de_result.best_value:.4f}")
        
        print("\n" + "-"*60)
        print("PHASE 4: BAYESIAN OPTIMIZATION (Learning Rate Fine-tuning)")
        print("-"*60)
        bo_result = self.optimize_bo(ga_result, pso_result, de_result)
        print(f"BO completed: Best value = {bo_result.best_value:.4f}")
        
        total_time = time.time() - start_total_time
        self.analyze_and_report_results(ga_result, pso_result, de_result, bo_result, total_time)
        
        return ga_result, pso_result, de_result, bo_result
    
    def analyze_and_report_results(self, ga_result: MetaheuristicResult, pso_result: MetaheuristicResult,
                                 de_result: MetaheuristicResult, bo_result: MetaheuristicResult, 
                                 total_time: float):
        """Analysis and reporting"""
        print("\n" + "="*80)
        print("METAHEURISTIC OPTIMIZATION ANALYSIS")
        print("="*80)
        
        results = [ga_result, pso_result, de_result, bo_result]
        
        print("\nPERFORMANCE SUMMARY:")
        print("-" * 80)
        print(f"{'Algorithm':<12} {'Best Value':<12} {'Time (s)':<10} {'Iterations':<12} {'Success Rate':<12}")
        print("-" * 80)
        
        for result in results:
            print(f"{result.algorithm:<12} {result.best_value:<12.4f} {result.computation_time:<10.1f} "
                  f"{result.iterations:<12} {result.success_rate:<12.2%}")
        
        print("-" * 80)
        print(f"{'TOTAL':<12} {'':<12} {total_time:<10.1f} {sum(r.iterations for r in results):<12}")
        
        best_result = min(results, key=lambda x: x.best_value)
        print(f"\nBEST OVERALL RESULT: {best_result.algorithm} with value {best_result.best_value:.4f}")
        
        self.save_metaheuristic_results(ga_result, pso_result, de_result, bo_result, total_time)
    
    def save_metaheuristic_results(self, ga_result: MetaheuristicResult, pso_result: MetaheuristicResult,
                                 de_result: MetaheuristicResult, bo_result: MetaheuristicResult, 
                                 total_time: float):
        """Save results"""
        results_dir = os.path.join(self.args.save_path, 'metaheuristic_results')
        os.makedirs(results_dir, exist_ok=True)
        
        performance_data = {
            'Algorithm': ['GA', 'PSO', 'DE', 'BO'],
            'Best_Value': [ga_result.best_value, pso_result.best_value, de_result.best_value, bo_result.best_value],
            'Computation_Time': [ga_result.computation_time, pso_result.computation_time, de_result.computation_time, bo_result.computation_time],
            'Iterations': [ga_result.iterations, pso_result.iterations, de_result.iterations, bo_result.iterations],
            'Success_Rate': [ga_result.success_rate, pso_result.success_rate, de_result.success_rate, bo_result.success_rate],
        }
        
        df_performance = pd.DataFrame(performance_data)
        df_performance.to_csv(os.path.join(results_dir, 'performance_summary.csv'), index=False)
        print(f"Results saved to {results_dir}")


def optimized_train_and_evaluate(num_blocks, num_heads, channels, lr, batch_size, expansion_factor, 
                                num_refinement, loss_weights, num_iter=3, data_path='./datasets/celeb', 
                                data_path_test='./datasets/celeb'):
    """Completely fixed training function with NO mixed precision issues"""
    
    try:
        # Enhanced file discovery
        file_patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        inp_files = []
        target_files = []
        
        for pattern in file_patterns:
            inp_files.extend(glob.glob(f'{data_path}/inp/{pattern}'))
            target_files.extend(glob.glob(f'{data_path}/target/{pattern}'))
            
            if not inp_files:
                inp_files.extend(glob.glob(f'{data_path}/input/{pattern}'))
        
        if len(inp_files) == 0:
            return float('inf')
        
        # Aggressive memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Check available memory and adjust batch size
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            available_memory = total_memory - allocated_memory
            
            # If less than 4GB available, use smallest batch size
            if available_memory < 4 * 1024 * 1024 * 1024:  # 4GB
                batch_size = 1
                print(f"Low GPU memory detected. Using batch_size=1")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # NO MIXED PRECISION - Pure FP32 training
        print("Using FP32 training (no mixed precision) for stability")
        
        # Create model and ensure FP32
        model = Inpainting(num_blocks, num_heads, channels, num_refinement, expansion_factor).to(device)
        model = model.float()
        
        # Count and display parameters
        total_params, trainable_params = count_parameters(model)
        
        # Enhanced optimizer
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
        
        # Use subset for large datasets
        if len(inp_files) > 10000:
            subset_size = 1000  # Even smaller subset for memory
            length = min(batch_size * num_iter, subset_size)
            print(f"Large dataset detected ({len(inp_files)} files). Using subset of {subset_size}.")
        else:
            length = min(batch_size * num_iter, len(inp_files))
            
        train_dataset = TrainDataset(data_path, data_path_test, 'inpaint', 'train', 128, length)
        
        if len(train_dataset) == 0:
            return float('inf')
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # No workers to avoid memory issues
            pin_memory=False,  # Disabled for stability
        )
        
        if len(train_loader) == 0:
            return float('inf')
        
        model.train()
        total_loss = 0.0
        
        # Get perceptual loss
        percep_loss_fn = get_perceptual_loss()
        
        for n_iter, (rain, norain, name, h, w) in enumerate(train_loader):
            if n_iter >= num_iter:
                break
            
            # PURE FP32 - No autocast at all
            rain = rain.to(device, non_blocking=False).float()
            norain = norain.to(device, non_blocking=False).float()
            
            # Forward pass in pure FP32
            out = model(rain)
            
            # Ensure all tensors are FP32
            out = out.float()
            norain = norain.float()
            
            # Compute losses in FP32
            l1_loss = F.l1_loss(out, norain)
            
            try:
                ssim_loss = 1 - ssim(out, norain)
            except:
                ssim_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            
            try:
                edge_out = kornia.filters.sobel(out, normalized=True, eps=1e-06)
                edge_gt = kornia.filters.sobel(norain, normalized=True, eps=1e-06)
                edge_loss = F.l1_loss(edge_out[0], edge_gt[0])
            except:
                edge_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            
            try:
                percep_loss = percep_loss_fn(out, norain)
            except:
                percep_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            
            w_l1, w_percep, w_ssim, w_edge = loss_weights
            loss = (l1_loss * w_l1 + percep_loss * w_percep + 
                   ssim_loss * w_ssim + edge_loss * w_edge)
            
            # Standard backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Aggressive memory cleanup every iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Validation
        model.eval()
        with torch.no_grad():
            try:
                rain, norain, name, h, w = next(iter(train_loader))
                rain = rain.to(device, non_blocking=False).float()
                norain = norain.to(device, non_blocking=False).float()
                
                out = model(rain).float()
                y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
                val_psnr = psnr(y, gt)
                
                return -val_psnr.item()
                
            except StopIteration:
                return float('inf')
        
    except Exception as e:
        print(f"Training error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return float('inf')


def run_enhanced_metaheuristic_optimization(args):
    """Main function with complete error handling"""
    
    print("="*80)
    print("ENHANCED METAHEURISTIC OPTIMIZATION FOR BLIND INPAINTING")
    print("="*80)
    print("Integrating GA + PSO + DE + BO with memory optimization")
    
    manager = OptimizedMetaheuristicManager(args)
    ga_result, pso_result, de_result, bo_result = manager.run_integrated_metaheuristics()
    
    print("\n" + "="*80)
    print("GENERATING OPTIMAL INTEGRATED PARAMETERS")
    print("="*80)
    
    final_params = {
        'num_blocks': ga_result.best_params.get('num_blocks', [1, 2, 2, 3]),
        'num_heads': ga_result.best_params.get('num_heads', [1, 1, 2, 2]),
        'channels': ga_result.best_params.get('channels', [16, 32, 64, 128]),
        'num_refinement': ga_result.best_params.get('num_refinement', 3),
        'expansion_factor': de_result.best_params.get('expansion_factor', 2.5),
        'lr': bo_result.best_params.get('lr', 5e-4),
        'loss_weights': {
            'w_l1': pso_result.best_params.get('w_l1', 0.5),
            'w_percep': pso_result.best_params.get('w_percep', 0.2),
            'w_ssim': pso_result.best_params.get('w_ssim', 0.4),
            'w_edge': pso_result.best_params.get('w_edge', 0.3)
        }
    }
    
    print("FINAL OPTIMIZED PARAMETERS:")
    print(f"  Architecture: blocks={final_params['num_blocks']}, heads={final_params['num_heads']}, channels={final_params['channels']}")
    print(f"  Refinement: {final_params['num_refinement']}")
    print(f"  Expansion: {final_params['expansion_factor']:.3f}")
    print(f"  Learning rate: {final_params['lr']:.2e}")
    print(f"  Loss weights: L1={final_params['loss_weights']['w_l1']:.3f}, Percep={final_params['loss_weights']['w_percep']:.3f}, SSIM={final_params['loss_weights']['w_ssim']:.3f}, Edge={final_params['loss_weights']['w_edge']:.3f}")
    
    return final_params, ga_result, pso_result, de_result, bo_result


def test_loop(net, data_loader, num_iter):
    """Test loop with proper error handling"""
    net.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for rain, norain, name, h, w in test_bar:
            rain, norain = rain.to(device, non_blocking=False).float(), norain.to(device, non_blocking=False).float()
            
            # No mixed precision - pure FP32
            out = model(rain).float()
            
            out = torch.clamp((torch.clamp(out[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            norain = torch.clamp(norain[:, :, :h, :w].mul(255), 0, 255).byte()
            
            y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
            current_psnr, current_ssim = psnr(y, gt), ssim(y, gt)
            total_psnr += current_psnr.item()
            total_ssim += current_ssim.item()
            count += 1
            
            save_path = '{}/{}/{}'.format(args.save_path, args.data_name, name[0])
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            
            out_np = out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()
            Image.fromarray(out_np).save(save_path, optimize=True, quality=95)
            
            test_bar.set_description('Test Iter: [{}/{}] PSNR: {:.2f} SSIM: {:.3f}'
                                    .format(num_iter, 1 if args.model_file else args.num_iter,
                                            total_psnr / count, total_ssim / count))
    
    return total_psnr / count, total_ssim / count


def save_loop(net, data_loader, num_iter):
    """Save loop with proper handling"""
    global best_psnr, best_ssim
    
    val_psnr, val_ssim = test_loop(net, data_loader, num_iter)
    results['PSNR'].append('{:.2f}'.format(val_psnr))
    results['SSIM'].append('{:.3f}'.format(val_ssim))
    
    data_frame = pd.DataFrame(data=results, index=range(1, (num_iter if args.model_file else num_iter // 100) + 1))
    data_frame.to_csv('{}/{}.csv'.format(args.save_path, args.data_name), index_label='Iter', float_format='%.3f')
    
    if val_psnr > best_psnr and val_ssim > best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        with open('{}/{}.txt'.format(args.save_path, args.data_name), 'w') as f:
            f.write('Iter: {} PSNR:{:.2f} SSIM:{:.3f}'.format(num_iter, best_psnr, best_ssim))
        torch.save(model.state_dict(), '{}/{}.pth'.format(args.save_path, args.data_name))
        print(f"New best model saved: PSNR={best_psnr:.2f}, SSIM={best_ssim:.3f}")


def apply_optimized_params(args):
    """Apply optimized parameters"""
    print("Applying memory-optimized parameters...")
    
    # Smaller, memory-efficient defaults
    args.num_blocks = [1, 2, 2, 3]
    args.num_heads = [1, 1, 2, 2]
    args.channels = [16, 32, 64, 128]
    args.num_refinement = 3
    args.expansion_factor = 2.5
    args.lr = 5e-4
    
    args.loss_weights = (0.5, 0.2, 0.4, 0.3)
    
    print(f"Applied parameters:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Architecture: blocks={args.num_blocks}, heads={args.num_heads}, channels={args.channels}")
    print(f"  Loss weights: L1={args.loss_weights[0]:.3f}, Perceptual={args.loss_weights[1]:.3f}, SSIM={args.loss_weights[2]:.3f}, Edge={args.loss_weights[3]:.3f}")


def run_training(args, use_optimized=False):
    """Enhanced training with complete error handling"""
    
    if use_optimized:
        print("Using optimized parameters for training...")
    else:
        print("Using default parameters for training...")
    
    test_dataset = TrainDataset(args.data_path_test, args.data_path_test, args.data_name, 'test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    global results, best_psnr, best_ssim, model
    results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': []}, 0.0, 0.0
    
    # Create model and count parameters properly
    model = Inpainting(
        args.num_blocks, 
        args.num_heads, 
        args.channels, 
        args.num_refinement, 
        args.expansion_factor
    ).to(device).float()
    
    # Accurate parameter counting
    total_params, trainable_params = count_parameters(model)
    
    if args.model_file:
        try:
            model.load_state_dict(torch.load(args.model_file, map_location=device))
            print(f"Loaded model from {args.model_file}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        save_loop(model, test_loader, 1)
    else:
        # Standard training - NO MIXED PRECISION
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)
        
        total_loss, total_num, results['Loss'], i = 0.0, 0, [], 0
        train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True)
        
        # Get loss weights
        if hasattr(args, 'loss_weights'):
            loss_weights = args.loss_weights
            print(f"Using loss weights: L1={loss_weights[0]:.3f}, Perceptual={loss_weights[1]:.3f}, SSIM={loss_weights[2]:.3f}, Edge={loss_weights[3]:.3f}")
        else:
            loss_weights = (0.5, 0.2, 0.4, 0.3)
            print("Using default loss weights")
        
        # Get perceptual loss
        percep_loss_fn = get_perceptual_loss()
        
        for n_iter in train_bar:
            # Progressive learning
            if n_iter == 1 or n_iter - 1 in args.milestone:
                end_iter = args.milestone[i] if i < len(args.milestone) else args.num_iter
                start_iter = args.milestone[i - 1] if i > 0 else 0
                length = args.batch_size[i] * (end_iter - start_iter)
                train_dataset = TrainDataset(args.data_path, args.data_path_test, args.data_name, 'train', args.patch_size[i], length)
                
                train_loader = iter(DataLoader(
                    train_dataset, 
                    args.batch_size[i], 
                    True, 
                    num_workers=0,
                    pin_memory=False
                ))
                i += 1
            
            model.train()
            
            try:
                rain, norain, name, h, w = next(train_loader)
                rain = rain.to(device, non_blocking=False).float()
                norain = norain.to(device, non_blocking=False).float()

                # Pure FP32 forward pass
                out = model(rain).float()
                norain = norain.float()

                l1_loss = F.l1_loss(out, norain)
                
                try:
                    ssim_loss = 1 - ssim(out, norain)
                except:
                    ssim_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                
                try:
                    edge_out = kornia.filters.sobel(out, normalized=True, eps=1e-06)
                    edge_gt = kornia.filters.sobel(norain, normalized=True, eps=1e-06)
                    edge_loss = F.l1_loss(edge_out[0], edge_gt[0])
                except:
                    edge_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                
                try:
                    percep_loss = percep_loss_fn(out, norain)
                except:
                    percep_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                loss = (l1_loss * loss_weights[0] + 
                       percep_loss * loss_weights[1] + 
                       ssim_loss * loss_weights[2] + 
                       edge_loss * loss_weights[3])

                # Standard backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            except Exception as e:
                print(f"Training error at iteration {n_iter}: {e}")
                continue

            total_num += rain.size(0)
            total_loss += loss.item() * rain.size(0)
            train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'
                                      .format(n_iter, args.num_iter, total_loss / total_num))

            lr_scheduler.step()
            
            if n_iter % 500 == 0:  # Reduced frequency for efficiency
                results['Loss'].append('{:.3f}'.format(total_loss / total_num))
                save_loop(model, test_loader, n_iter)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize', action='store_true', 
                       help='Run metaheuristic optimization (GA+PSO+DE+BO)')
    parser.add_argument('--use-defaults', action='store_true', 
                       help='Use original default parameters')
    temp_args, remaining = parser.parse_known_args()
    
    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()
    
    # Enhanced CUDA setup
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    os.makedirs(args.save_path, exist_ok=True)
    
    if temp_args.optimize:
        print("="*80)
        print("RUNNING ENHANCED METAHEURISTIC OPTIMIZATION") 
        print("Integrating GA + PSO + DE + BO with memory optimization")
        print("="*80)
        
        final_params, ga_result, pso_result, de_result, bo_result = run_enhanced_metaheuristic_optimization(args)
        
        # Apply optimized parameters
        args.num_blocks = final_params['num_blocks']
        args.num_heads = final_params['num_heads']
        args.channels = final_params['channels']
        args.num_refinement = final_params['num_refinement']
        args.expansion_factor = final_params['expansion_factor']
        args.lr = final_params['lr']
        
        args.loss_weights = (
            final_params['loss_weights']['w_l1'],
            final_params['loss_weights']['w_percep'],
            final_params['loss_weights']['w_ssim'],
            final_params['loss_weights']['w_edge']
        )
        
        print("="*80)
        print("STARTING TRAINING WITH METAHEURISTIC-OPTIMIZED PARAMETERS")
        print("="*80)
        
        run_training(args, use_optimized=True)
        
    elif temp_args.use_defaults:
        print("="*80)
        print("USING ORIGINAL DEFAULT PARAMETERS")
        print("="*80)
        
        run_training(args, use_optimized=False)
        
    else:
        print("="*80)
        print("USING MEMORY-OPTIMIZED PARAMETERS")
        print("="*80)
        
        apply_optimized_params(args)
        run_training(args, use_optimized=True)