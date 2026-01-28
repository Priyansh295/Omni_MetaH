"""
Frequency-Aware Loss for Blind Image Inpainting
================================================

This module implements wavelet-based frequency losses that provide
separate supervision for different frequency bands, improving
generalization to unseen masks and data.

Key Features:
- DWT decomposition into LL (structure), LH/HL (edges), HH (texture) subbands
- Configurable weights per subband
- Multi-scale option for hierarchical frequency supervision
- Seamless integration with existing training pipeline

Reference Papers:
- Focal Frequency Loss (ICCV 2021)
- Wavelet-based Perceptual Loss (various)

Author: Research Enhancement Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import warnings

# Import wavelet transforms (already in your requirements)
try:
    from pytorch_wavelets import DWTForward, DWTInverse
    WAVELETS_AVAILABLE = True
except ImportError:
    WAVELETS_AVAILABLE = False
    warnings.warn("pytorch_wavelets not found. Install with: pip install pytorch_wavelets")


class FrequencyAwareLoss(nn.Module):
    """
    Frequency-Aware Loss using Discrete Wavelet Transform (DWT).
    
    Decomposes images into frequency subbands and computes separate losses:
    - LL (Low-Low): Structure/color loss
    - LH (Low-High): Horizontal edge loss  
    - HL (High-Low): Vertical edge loss
    - HH (High-High): Texture/detail loss
    
    Args:
        wavelet: Wavelet type ('db1', 'db2', 'db3', 'haar', 'sym2', etc.)
        levels: Number of DWT decomposition levels (1-3 recommended)
        weights: Dict or tuple of weights for (LL, LH, HL, HH) subbands
        loss_type: 'l1', 'l2', or 'charbonnier'
        reduction: 'mean', 'sum', or 'none'
        
    Example:
        >>> freq_loss = FrequencyAwareLoss(wavelet='db3', levels=1)
        >>> output = model(input)
        >>> loss = freq_loss(output, target)
    """
    
    def __init__(
        self,
        wavelet: str = 'db3',
        levels: int = 1,
        weights: Optional[Dict[str, float]] = None,
        loss_type: str = 'l1',
        reduction: str = 'mean',
        eps: float = 1e-6
    ):
        super(FrequencyAwareLoss, self).__init__()
        
        if not WAVELETS_AVAILABLE:
            raise ImportError("pytorch_wavelets is required for FrequencyAwareLoss")
        
        self.wavelet = wavelet
        self.levels = levels
        self.reduction = reduction
        self.eps = eps
        self.loss_type = loss_type.lower()
        
        # Initialize DWT
        self.dwt = DWTForward(J=levels, mode='zero', wave=wavelet)
        
        # Default weights: emphasize high-frequency for texture preservation
        # LL=1.0: structure is important but network learns this easily
        # LH/HL=1.5: edges need moderate emphasis
        # HH=2.0: textures need strongest emphasis (often undertrained)
        default_weights = {
            'LL': 1.0,   # Structure/approximation
            'LH': 1.5,   # Horizontal edges
            'HL': 1.5,   # Vertical edges  
            'HH': 2.0    # Textures/diagonal details
        }
        
        self.weights = weights if weights is not None else default_weights
        
        # Register weights as buffers for device handling
        self.register_buffer('w_ll', torch.tensor(self.weights['LL']))
        self.register_buffer('w_lh', torch.tensor(self.weights['LH']))
        self.register_buffer('w_hl', torch.tensor(self.weights['HL']))
        self.register_buffer('w_hh', torch.tensor(self.weights['HH']))
        
    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss based on loss_type."""
        if self.loss_type == 'l1':
            return F.l1_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == 'l2':
            return F.mse_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == 'charbonnier':
            # Charbonnier loss: sqrt((x-y)^2 + eps^2) - smoother than L1
            diff = pred - target
            loss = torch.sqrt(diff ** 2 + self.eps ** 2)
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply DWT decomposition.
        
        Returns:
            yl: Low-frequency approximation (LL)
            yh: List of high-frequency details per level
                Each element has shape [B, C, 3, H, W] for (LH, HL, HH)
        """
        yl, yh = self.dwt(x)
        return yl, yh
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute frequency-aware loss.
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            return_components: If True, also return dict of individual losses
            
        Returns:
            total_loss: Weighted sum of all frequency losses
            components (optional): Dict with individual subband losses
        """
        # Ensure same size
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
        # Decompose both images
        pred_yl, pred_yh = self._decompose(pred)
        target_yl, target_yh = self._decompose(target)
        
        # Initialize loss components
        components = {}
        
        # LL (approximation) loss - structure
        loss_ll = self._compute_loss(pred_yl, target_yl)
        components['LL'] = loss_ll.item() if isinstance(loss_ll, torch.Tensor) else loss_ll
        
        # High-frequency losses (summed across levels)
        loss_lh, loss_hl, loss_hh = 0.0, 0.0, 0.0
        
        for level_idx, (pred_h, target_h) in enumerate(zip(pred_yh, target_yh)):
            # pred_h shape: [B, C, 3, H, W] where dim=2 is (LH, HL, HH)
            
            # LH: Horizontal details (index 0)
            loss_lh += self._compute_loss(pred_h[:, :, 0], target_h[:, :, 0])
            
            # HL: Vertical details (index 1)
            loss_hl += self._compute_loss(pred_h[:, :, 1], target_h[:, :, 1])
            
            # HH: Diagonal details / textures (index 2)
            loss_hh += self._compute_loss(pred_h[:, :, 2], target_h[:, :, 2])
        
        components['LH'] = loss_lh.item() if isinstance(loss_lh, torch.Tensor) else loss_lh
        components['HL'] = loss_hl.item() if isinstance(loss_hl, torch.Tensor) else loss_hl
        components['HH'] = loss_hh.item() if isinstance(loss_hh, torch.Tensor) else loss_hh
        
        # Weighted total loss
        total_loss = (
            self.w_ll * loss_ll +
            self.w_lh * loss_lh +
            self.w_hl * loss_hl +
            self.w_hh * loss_hh
        )
        
        if return_components:
            return total_loss, components
        return total_loss


class MultiScaleFrequencyLoss(nn.Module):
    """
    Multi-Scale Frequency Loss with progressive weighting.
    
    Applies frequency loss at multiple image scales, with different
    weights per scale. Coarse scales focus on structure, fine scales
    on details.
    
    Args:
        scales: List of scale factors (1.0 = original, 0.5 = half, etc.)
        scale_weights: Weights for each scale
        **freq_kwargs: Arguments passed to FrequencyAwareLoss
        
    Example:
        >>> ms_freq_loss = MultiScaleFrequencyLoss(scales=[1.0, 0.5, 0.25])
        >>> loss = ms_freq_loss(output, target)
    """
    
    def __init__(
        self,
        scales: List[float] = [1.0, 0.5, 0.25],
        scale_weights: Optional[List[float]] = None,
        **freq_kwargs
    ):
        super(MultiScaleFrequencyLoss, self).__init__()
        
        self.scales = scales
        self.scale_weights = scale_weights or [1.0] * len(scales)
        
        # Create frequency loss for each scale
        self.freq_losses = nn.ModuleList([
            FrequencyAwareLoss(**freq_kwargs) for _ in scales
        ])
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        return_per_scale: bool = False
    ) -> torch.Tensor:
        """
        Compute multi-scale frequency loss.
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            return_per_scale: If True, also return losses per scale
            
        Returns:
            total_loss: Weighted sum across all scales
        """
        total_loss = 0.0
        per_scale_losses = []
        
        for scale, weight, freq_loss in zip(self.scales, self.scale_weights, self.freq_losses):
            if scale == 1.0:
                # Original scale
                scaled_pred = pred
                scaled_target = target
            else:
                # Downscale
                size = (int(pred.shape[2] * scale), int(pred.shape[3] * scale))
                scaled_pred = F.interpolate(pred, size=size, mode='bilinear', align_corners=False)
                scaled_target = F.interpolate(target, size=size, mode='bilinear', align_corners=False)
            
            scale_loss = freq_loss(scaled_pred, scaled_target)
            per_scale_losses.append(scale_loss.item())
            total_loss += weight * scale_loss
        
        if return_per_scale:
            return total_loss, per_scale_losses
        return total_loss


class AdaptiveFrequencyLoss(nn.Module):
    """
    Adaptive Frequency Loss with learnable subband weights.
    
    Instead of fixed weights, this version learns optimal weights
    during training. Useful when integrated with metaheuristic
    optimization or end-to-end training.
    
    Args:
        init_weights: Initial weights for (LL, LH, HL, HH)
        **freq_kwargs: Arguments passed to FrequencyAwareLoss
    """
    
    def __init__(
        self,
        init_weights: Tuple[float, float, float, float] = (1.0, 1.5, 1.5, 2.0),
        **freq_kwargs
    ):
        super(AdaptiveFrequencyLoss, self).__init__()
        
        # Create base frequency loss
        self.freq_loss = FrequencyAwareLoss(**freq_kwargs)
        
        # Learnable weights (in log space for stability)
        self.log_weights = nn.Parameter(torch.log(torch.tensor(init_weights)))
        
    def get_weights(self) -> Tuple[float, float, float, float]:
        """Get current weights (for logging/monitoring)."""
        weights = torch.exp(self.log_weights)
        return tuple(weights.detach().cpu().tolist())
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss with current learned weights."""
        # Get current weights
        weights = torch.exp(self.log_weights)
        
        # Update frequency loss weights
        self.freq_loss.w_ll = weights[0]
        self.freq_loss.w_lh = weights[1]
        self.freq_loss.w_hl = weights[2]
        self.freq_loss.w_hh = weights[3]
        
        return self.freq_loss(pred, target)


class CombinedLoss(nn.Module):
    """
    Combined Loss that integrates frequency loss with existing losses.
    
    This is a convenience wrapper that combines:
    - L1 loss (spatial)
    - Perceptual loss (VGG features)
    - SSIM loss (structural)
    - Edge loss (Sobel)
    - Frequency loss (wavelet subbands) [NEW]
    
    Args:
        perceptual_loss: VGGPerceptualLoss instance
        weights: Dict of loss weights
        use_frequency: Whether to include frequency loss
        freq_kwargs: Arguments for FrequencyAwareLoss
    """
    
    def __init__(
        self,
        perceptual_loss: Optional[nn.Module] = None,
        weights: Optional[Dict[str, float]] = None,
        use_frequency: bool = True,
        freq_kwargs: Optional[Dict] = None
    ):
        super(CombinedLoss, self).__init__()
        
        self.perceptual_loss = perceptual_loss
        self.use_frequency = use_frequency
        
        # Default weights (including frequency loss)
        default_weights = {
            'l1': 0.4,
            'perceptual': 0.2,
            'ssim': 0.3,
            'edge': 0.5,
            'frequency': 0.3  # New weight for frequency loss
        }
        self.weights = weights if weights is not None else default_weights
        
        # Initialize frequency loss
        if use_frequency:
            freq_kwargs = freq_kwargs or {}
            self.freq_loss = FrequencyAwareLoss(**freq_kwargs)
        else:
            self.freq_loss = None
        
        # We'll import these dynamically to avoid circular imports
        self._ssim = None
        self._sobel = None
    
    def _get_ssim(self):
        """Lazy import of SSIM function."""
        if self._ssim is None:
            from utils_train import ssim
            self._ssim = ssim
        return self._ssim
    
    def _get_sobel(self):
        """Lazy import of Sobel filter."""
        if self._sobel is None:
            import kornia
            self._sobel = kornia.filters.sobel
        return self._sobel
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            return_components: If True, return dict of individual losses
            
        Returns:
            total_loss: Weighted combination of all losses
        """
        components = {}
        total_loss = 0.0
        
        # L1 loss
        l1_loss = F.l1_loss(pred, target)
        components['l1'] = l1_loss.item()
        total_loss += self.weights['l1'] * l1_loss
        
        # Perceptual loss
        if self.perceptual_loss is not None and self.weights.get('perceptual', 0) > 0:
            perc_loss = self.perceptual_loss(pred, target)
            components['perceptual'] = perc_loss.item()
            total_loss += self.weights['perceptual'] * perc_loss
        
        # SSIM loss
        if self.weights.get('ssim', 0) > 0:
            ssim_val = self._get_ssim()(pred, target)
            ssim_loss = 1 - ssim_val
            components['ssim'] = ssim_loss.item()
            total_loss += self.weights['ssim'] * ssim_loss
        
        # Edge loss (Sobel)
        if self.weights.get('edge', 0) > 0:
            sobel = self._get_sobel()
            edge_pred = sobel(pred, normalized=True, eps=1e-6)
            edge_target = sobel(target, normalized=True, eps=1e-6)
            edge_loss = F.l1_loss(edge_pred, edge_target)
            components['edge'] = edge_loss.item()
            total_loss += self.weights['edge'] * edge_loss
        
        # Frequency loss [NEW]
        if self.use_frequency and self.freq_loss is not None:
            freq_loss, freq_components = self.freq_loss(pred, target, return_components=True)
            components['frequency'] = freq_loss.item()
            components['freq_LL'] = freq_components['LL']
            components['freq_LH'] = freq_components['LH']
            components['freq_HL'] = freq_components['HL']
            components['freq_HH'] = freq_components['HH']
            total_loss += self.weights['frequency'] * freq_loss
        
        if return_components:
            return total_loss, components
        return total_loss


# ============================================================================
# Utility Functions
# ============================================================================

def create_frequency_loss(
    config: str = 'default',
    **kwargs
) -> FrequencyAwareLoss:
    """
    Factory function to create frequency loss with preset configurations.
    
    Args:
        config: One of 'default', 'texture_focus', 'edge_focus', 'balanced'
        **kwargs: Override any configuration parameters
        
    Returns:
        FrequencyAwareLoss instance
    """
    configs = {
        'default': {
            'wavelet': 'db3',
            'levels': 1,
            'weights': {'LL': 1.0, 'LH': 1.5, 'HL': 1.5, 'HH': 2.0}
        },
        'texture_focus': {
            'wavelet': 'db3',
            'levels': 1,
            'weights': {'LL': 0.5, 'LH': 1.0, 'HL': 1.0, 'HH': 3.0}  # Heavy HH
        },
        'edge_focus': {
            'wavelet': 'db3',
            'levels': 1,
            'weights': {'LL': 0.5, 'LH': 2.5, 'HL': 2.5, 'HH': 1.0}  # Heavy LH/HL
        },
        'balanced': {
            'wavelet': 'db3',
            'levels': 1,
            'weights': {'LL': 1.0, 'LH': 1.0, 'HL': 1.0, 'HH': 1.0}  # Equal
        },
        'multi_level': {
            'wavelet': 'db3',
            'levels': 2,  # 2-level decomposition
            'weights': {'LL': 1.0, 'LH': 1.5, 'HL': 1.5, 'HH': 2.0}
        }
    }
    
    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Choose from {list(configs.keys())}")
    
    # Merge with user overrides
    final_config = {**configs[config], **kwargs}
    
    return FrequencyAwareLoss(**final_config)


def visualize_frequency_decomposition(
    image: torch.Tensor,
    wavelet: str = 'db3',
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the wavelet decomposition of an image.
    
    Useful for understanding what each subband captures and
    for debugging/paper figures.
    
    Args:
        image: Input image [1, C, H, W] or [C, H, W]
        wavelet: Wavelet type
        save_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    if not WAVELETS_AVAILABLE:
        raise ImportError("pytorch_wavelets required for visualization")
    
    # Ensure 4D
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Move to CPU for visualization
    image = image.cpu()
    
    # Decompose
    dwt = DWTForward(J=1, mode='zero', wave=wavelet)
    yl, yh = dwt(image)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    if image.shape[1] == 3:
        axes[0, 0].imshow(image[0].permute(1, 2, 0).clamp(0, 1))
    else:
        axes[0, 0].imshow(image[0, 0], cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # LL (approximation)
    ll = yl[0].mean(dim=0)  # Average across channels
    axes[0, 1].imshow(ll, cmap='viridis')
    axes[0, 1].set_title('LL (Low-Low)\nStructure/Color')
    axes[0, 1].axis('off')
    
    # LH (horizontal details)
    lh = yh[0][0, :, 0].mean(dim=0)
    axes[0, 2].imshow(lh.abs(), cmap='hot')
    axes[0, 2].set_title('LH (Low-High)\nHorizontal Edges')
    axes[0, 2].axis('off')
    
    # HL (vertical details)
    hl = yh[0][0, :, 1].mean(dim=0)
    axes[1, 0].imshow(hl.abs(), cmap='hot')
    axes[1, 0].set_title('HL (High-Low)\nVertical Edges')
    axes[1, 0].axis('off')
    
    # HH (diagonal details)
    hh = yh[0][0, :, 2].mean(dim=0)
    axes[1, 1].imshow(hh.abs(), cmap='hot')
    axes[1, 1].set_title('HH (High-High)\nTextures/Details')
    axes[1, 1].axis('off')
    
    # Combined high-freq
    combined_hf = (lh.abs() + hl.abs() + hh.abs())
    axes[1, 2].imshow(combined_hf, cmap='magma')
    axes[1, 2].set_title('Combined High-Freq\n(LH + HL + HH)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == '__main__':
    print("Testing FrequencyAwareLoss...")
    
    # Create test tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randn(2, 3, 256, 256).to(device)
    
    # Test basic frequency loss
    freq_loss = FrequencyAwareLoss().to(device)
    loss, components = freq_loss(pred, target, return_components=True)
    print(f"✓ FrequencyAwareLoss: {loss.item():.4f}")
    print(f"  Components: LL={components['LL']:.4f}, LH={components['LH']:.4f}, "
          f"HL={components['HL']:.4f}, HH={components['HH']:.4f}")
    
    # Test multi-scale loss
    ms_loss = MultiScaleFrequencyLoss(scales=[1.0, 0.5]).to(device)
    loss_ms = ms_loss(pred, target)
    print(f"✓ MultiScaleFrequencyLoss: {loss_ms.item():.4f}")
    
    # Test adaptive loss
    adapt_loss = AdaptiveFrequencyLoss().to(device)
    loss_adapt = adapt_loss(pred, target)
    print(f"✓ AdaptiveFrequencyLoss: {loss_adapt.item():.4f}")
    print(f"  Learned weights: {adapt_loss.get_weights()}")
    
    # Test factory function
    texture_loss = create_frequency_loss('texture_focus').to(device)
    loss_texture = texture_loss(pred, target)
    print(f"✓ texture_focus config: {loss_texture.item():.4f}")
    
    print("\n✅ All tests passed!")
