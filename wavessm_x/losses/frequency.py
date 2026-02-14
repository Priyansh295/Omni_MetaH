import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from pytorch_wavelets import DWTForward


class FrequencyAwareLoss(nn.Module):
    """
    Frequency-Aware Loss.
    Decomposes image into LL and HF bands and supervises them separately.
    """
    def __init__(self, wavelet='db3', levels=1, weights=None, loss_type='l1'):
        super().__init__()
        self.dwt = DWTForward(J=levels, mode='zero', wave=wavelet)
        self.levels = levels
        self.weights = weights or {'LL': 1.0, 'HF': 1.5} # HF weighted more
        self.loss_fn = nn.L1Loss() if loss_type == 'l1' else nn.MSELoss()

    def forward(self, pred, target):
        loss = 0.0
        
        # DWT does not support float16 — force float32 under AMP
        # P1 Fix: strict float32 for both DWT and Loss computation to prevent overflow
        with torch.amp.autocast('cuda', enabled=False):
            pl, ph = self.dwt(pred.float())
            tl, th = self.dwt(target.float())
        
            # LL Loss
            loss += self.loss_fn(pl, tl) * self.weights.get('LL', 1.0)
            
            # HF Loss
            for i in range(self.levels):
                # ph[i] is (B, C, 3, H, W) for LH, HL, HH
                loss += self.loss_fn(ph[i], th[i]) * self.weights.get('HF', 1.5)
            
        return loss


class AdaptiveFrequencyLoss(FrequencyAwareLoss):
    """
    Adaptive Frequency Loss.
    Learnable weights for each subband to balance structure vs texture.
    """
    def __init__(self, init_weights=(1.0, 1.5, 1.5, 2.0)):
        super().__init__()
        # weights: LL, LH, HL, HH
        self.w_ll = nn.Parameter(torch.tensor(init_weights[0]))
        self.w_lh = nn.Parameter(torch.tensor(init_weights[1]))
        self.w_hl = nn.Parameter(torch.tensor(init_weights[2]))
        self.w_hh = nn.Parameter(torch.tensor(init_weights[3]))

    def forward(self, pred, target):
        # DWT does not support float16 — force float32 under AMP
        with torch.amp.autocast('cuda', enabled=False):
            pl, ph = self.dwt(pred.float())
            tl, th = self.dwt(target.float())
        
        loss_ll = self.loss_fn(pl, tl) * self.w_ll.abs()
        
        # Level 1 only for simplicity
        ph_coeffs, th_coeffs = ph[0], th[0]
        
        loss_lh = self.loss_fn(ph_coeffs[:,:,0], th_coeffs[:,:,0]) * self.w_lh.abs()
        loss_hl = self.loss_fn(ph_coeffs[:,:,1], th_coeffs[:,:,1]) * self.w_hl.abs()
        loss_hh = self.loss_fn(ph_coeffs[:,:,2], th_coeffs[:,:,2]) * self.w_hh.abs()
        
        return loss_ll + loss_lh + loss_hl + loss_hh


class MultiScaleFrequencyLoss(nn.Module):
    """
    Multi-Scale Frequency Loss with progressive weighting.
    Applies frequency loss at multiple image scales.
    Coarse scales focus on structure, fine scales on texture.
    """
    def __init__(self, scales=(1.0, 0.5, 0.25), scale_weights=None, **freq_kwargs):
        super().__init__()
        self.scales = scales
        self.scale_weights = scale_weights or [1.0 / (2 ** i) for i in range(len(scales))]
        self.freq_loss = FrequencyAwareLoss(**freq_kwargs)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=pred.device)
        
        for scale, weight in zip(self.scales, self.scale_weights):
            if scale == 1.0:
                p, t = pred, target
            else:
                size = (int(pred.shape[2] * scale), int(pred.shape[3] * scale))
                p = F.interpolate(pred, size=size, mode='bilinear', align_corners=False)
                t = F.interpolate(target, size=size, mode='bilinear', align_corners=False)
            
            total_loss = total_loss + self.freq_loss(p, t) * weight
        
        return total_loss
