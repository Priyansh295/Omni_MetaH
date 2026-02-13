"""
Performance and memory monitoring utilities.
"""
import time
import torch
import os
import csv
from datetime import datetime
from collections import deque


class PerformanceMonitor:
    """
    Tracks iteration time, throughput, and ETA.
    Logs to CSV for post-training analysis.
    """
    def __init__(self, log_dir: str = './logs', log_name: str = 'training'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f'{log_name}_{timestamp}.csv')
        self._csv_initialized = False
        
        self._iter_times = deque(maxlen=100)  # Rolling window
        self._start_time = None
        self._total_start = time.time()
    
    def iter_start(self):
        """Call at the start of each iteration."""
        self._start_time = time.time()
    
    def iter_end(self):
        """Call at the end of each iteration. Returns elapsed seconds."""
        if self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        self._iter_times.append(elapsed)
        return elapsed
    
    @property
    def avg_iter_time(self) -> float:
        if not self._iter_times:
            return 0.0
        return sum(self._iter_times) / len(self._iter_times)
    
    def eta_seconds(self, remaining_iters: int) -> float:
        return self.avg_iter_time * remaining_iters
    
    def eta_str(self, remaining_iters: int) -> str:
        secs = self.eta_seconds(remaining_iters)
        hrs, rem = divmod(int(secs), 3600)
        mins, secs = divmod(rem, 60)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    
    def total_elapsed(self) -> float:
        return time.time() - self._total_start
    
    def log_to_csv(self, iteration: int, metrics: dict):
        """Append a row of metrics to the CSV log."""
        row = {'iteration': iteration, 'timestamp': time.time(), **metrics}
        
        if not self._csv_initialized:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            self._csv_initialized = True
            self._csv_fields = list(row.keys())
        else:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fields, extrasaction='ignore')
                writer.writerow(row)


class MemoryMonitor:
    """
    Monitors GPU memory usage and provides OOM-safe utilities.
    """
    def __init__(self, device: torch.device = None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self._peak_allocated = 0
    
    def current_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(self.device) / (1024 ** 2)
    
    def peak_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
    
    def reserved_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_reserved(self.device) / (1024 ** 2)
    
    def reset_peak(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def utilization_pct(self) -> float:
        """GPU memory utilization as percentage of total."""
        if not torch.cuda.is_available():
            return 0.0
        total = torch.cuda.get_device_properties(self.device).total_mem
        used = torch.cuda.memory_allocated(self.device)
        return (used / total) * 100.0
    
    def summary(self) -> str:
        return (
            f"GPU Mem: {self.current_mb():.0f}MB allocated, "
            f"{self.peak_mb():.0f}MB peak, "
            f"{self.reserved_mb():.0f}MB reserved, "
            f"{self.utilization_pct():.1f}% util"
        )
    
    def safe_clear(self):
        """Clear GPU cache without crashing on CPU-only systems."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
