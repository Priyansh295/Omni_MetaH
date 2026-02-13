import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters

from .perceptual import VGGPerceptualLoss
from .frequency import FrequencyAwareLoss, AdaptiveFrequencyLoss
from .ssim import SSIMLoss


class MaskAwareLoss(nn.Module):
    """
    Loss specifically for mask prediction supervision.
    Combines Weighted BCE (focal-like) and Dice/IoU loss.
    """
    def __init__(self, use_iou=True, bce_weight=1.0, iou_weight=1.0):
        super().__init__()
        self.use_iou = use_iou
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight

    def forward(self, pred_mask, target_mask):
        # BCE Loss (pixel-wise)
        bce = F.binary_cross_entropy(pred_mask, target_mask)
        
        loss = bce * self.bce_weight
        
        if self.use_iou:
            # Soft IoU / Dice
            # Flatten: (B, 1, H, W) -> (B, H*W)
            pred_flat = pred_mask.view(pred_mask.size(0), -1)
            target_flat = target_mask.view(target_mask.size(0), -1)
            
            intersection = (pred_flat * target_flat).sum(dim=1)
            union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
            
            # Dice coeff: 2*I / (U + eps) - use float16-safe epsilon
            dice = (2. * intersection + 1e-4) / (union + 1e-4)
            loss += (1.0 - dice.mean()) * self.iou_weight
            
        return loss


class WaveSSMLoss(nn.Module):
    """
    Master loss module for WaveSSM-X.
    Combines:
    1. Reconstruction L1 (Spatial)
    2. Perceptual L1 (VGG features)
    3. Structural SSIM (Gaussian window)
    4. Edge Loss (Sobel filter)
    5. Frequency Loss (Wavelet domain)
    6. Mask Loss (Auxiliary supervision)
    """
    def __init__(self, 
                 weights={'l1': 1.0, 'perceptual': 1.0, 'ssim': 1.0, 'edge': 0.1, 'freq': 0.5, 'mask': 0.5},
                 use_adaptive_freq=False):
        super().__init__()
        self.weights = weights
        self.l1 = nn.L1Loss()
        
        # Initialize sub-losses
        self.perceptual = VGGPerceptualLoss(resize=True)
        
        self.ssim = SSIMLoss(window_size=11, sigma=1.5)
        
        if use_adaptive_freq:
            self.freq_loss = AdaptiveFrequencyLoss()
        else:
            self.freq_loss = FrequencyAwareLoss()
            
        self.mask_loss_fn = MaskAwareLoss()

    def forward(self, pred, target, pred_mask=None, target_mask=None, return_dict=False):
        loss_dict = {}
        total_loss = 0.0
        
        # 1. L1 Loss
        l1_val = self.l1(pred, target)
        total_loss += l1_val * self.weights.get('l1', 1.0)
        loss_dict['l1'] = l1_val.item()
        
        # 2. Perceptual Loss (Standard VGG)
        perc_val = self.perceptual(pred, target)
        total_loss += perc_val * self.weights.get('perceptual', 1.0)
        loss_dict['perceptual'] = perc_val.item()
        
        # 3. SSIM Loss
        ssim_val = self.ssim(pred, target)
        total_loss += ssim_val * self.weights.get('ssim', 1.0)
        loss_dict['ssim'] = ssim_val.item()
        
        # 4. Edge Loss (Sobel)
        if self.weights.get('edge', 0.0) > 0:
            pred_edge = kornia.filters.sobel(pred)
            target_edge = kornia.filters.sobel(target)
            edge_val = F.l1_loss(pred_edge, target_edge)
            total_loss += edge_val * self.weights.get('edge', 0.1)
            loss_dict['edge'] = edge_val.item()
            
        # 5. Frequency Loss
        if self.weights.get('freq', 0.0) > 0:
            freq_val = self.freq_loss(pred, target)
            total_loss += freq_val * self.weights.get('freq', 0.5)
            loss_dict['freq'] = freq_val.item()
            
        # 6. Mask Loss (Auxiliary)
        if pred_mask is not None and target_mask is not None:
             mask_val = self.mask_loss_fn(pred_mask, target_mask)
             total_loss += mask_val * self.weights.get('mask', 0.5)
             loss_dict['mask'] = mask_val.item()
             
        if return_dict:
            return total_loss, loss_dict
        return total_loss

    def update_weights(self, new_weights: dict):
        """Update loss weights dynamically (for DWA)."""
        for key, value in new_weights.items():
            if key in self.weights:
                self.weights[key] = float(value) if torch.is_tensor(value) else value
