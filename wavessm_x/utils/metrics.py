"""
Image quality metrics for evaluation.
PSNR, SSIM (proper Gaussian), and optional LPIPS.
"""
import torch
import torch.nn.functional as F
import math

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

_lpips_model = None


def _get_gaussian_kernel(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
    g2d = g2d / g2d.sum()
    return g2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image tensor
        target: Ground truth tensor
        data_range: Value range (1.0 for [0,1], 255.0 for [0,255])
    
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(pred, target)
    if mse < 1e-10:
        return torch.tensor(100.0, device=pred.device)
    return 10.0 * torch.log10((data_range ** 2) / mse)


def ssim(
    pred: torch.Tensor, 
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
    per_channel: bool = False
) -> torch.Tensor:
    """
    Structural Similarity Index with proper Gaussian window.
    
    Args:
        pred: (B, C, H, W) predicted
        target: (B, C, H, W) ground truth
        window_size: Gaussian window size
        sigma: Gaussian standard deviation
        data_range: Value range
        k1, k2: SSIM stability constants
        per_channel: If True, return per-channel SSIM
    """
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    
    channels = pred.shape[1]
    kernel = _get_gaussian_kernel(window_size, sigma, channels).to(pred.device, pred.dtype)
    pad = window_size // 2
    
    mu1 = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu12
    
    # Clamp to avoid negative variance from numerical errors
    sigma1_sq = sigma1_sq.clamp(min=0)
    sigma2_sq = sigma2_sq.clamp(min=0)
    
    numerator = (2 * mu12 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    
    if per_channel:
        return ssim_map.mean(dim=[0, 2, 3])
    return ssim_map.mean()


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, net: str = 'alex') -> torch.Tensor:
    """
    Learned Perceptual Image Patch Similarity.
    Returns 0.0 if lpips package not available.
    
    Args:
        pred: (B, C, H, W) in [0, 1]
        target: (B, C, H, W) in [0, 1]
        net: 'alex' or 'vgg'
    """
    if not LPIPS_AVAILABLE:
        return torch.tensor(0.0, device=pred.device)
    
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net=net).to(pred.device).eval()
    
    # LPIPS expects [-1, 1]
    with torch.no_grad():
        return _lpips_model(pred * 2 - 1, target * 2 - 1).mean()


class MetricsTracker:
    """Tracks running averages of metrics across iterations."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._sums = {}
        self._counts = {}
    
    def update(self, metrics_dict: dict):
        for k, v in metrics_dict.items():
            val = v.item() if torch.is_tensor(v) else v
            self._sums[k] = self._sums.get(k, 0.0) + val
            self._counts[k] = self._counts.get(k, 0) + 1
    
    def averages(self) -> dict:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}
    
    def __repr__(self):
        avgs = self.averages()
        parts = [f"{k}: {v:.4f}" for k, v in avgs.items()]
        return " | ".join(parts)
