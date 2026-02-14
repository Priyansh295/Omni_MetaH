import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from pytorch_wavelets import DWTForward


class SpectralTransform(nn.Module):
    """
    FFT-based global convolution from LaMa.
    Processes features in the frequency domain for image-wide receptive field.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels * 2, 1)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        orig_dtype = x.dtype

        # Force FP32 for FFT stability
        with torch.cuda.amp.autocast(enabled=False):
            x_f = x.float()
            x_fft = torch.fft.rfft2(x_f, norm='ortho')
            x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)

            # Process in FP32 or cast back for conv?
            # Usually Conv/BN/ReLU are fine in FP16, but FFT output is sensitive.
            # Let's cast back to orig_dtype for Conv block to save memory, 
            # then cast to FP32 for IFFT.
            x_fft = x_fft.to(orig_dtype)
            
        x_fft = self.relu(self.bn(self.conv(x_fft)))

        with torch.cuda.amp.autocast(enabled=False):
            x_fft = x_fft.float()
            real, imag = x_fft.chunk(2, dim=1)
            x_fft_complex = torch.complex(real, imag)
            x = torch.fft.irfft2(x_fft_complex, s=(H, W), norm='ortho')
            
        return x.to(orig_dtype)


class WaveFFC(nn.Module):
    """
    Wavelet-guided Fast Fourier Convolution.

    Combines:
    - Local path: Standard convolution for local features
    - Global path: FFT-based convolution for image-wide context
    - Wavelet guidance: DWT-based gating for frequency-aware fusion

    This is inspired by LaMa's FFC but adds wavelet-based modulation.
    """

    def __init__(self, channels: int, ratio: float = 0.5, wavelet: str = 'db3'):
        super().__init__()
        self.ratio = ratio
        local_ch = int(channels * (1 - ratio))
        global_ch = channels - local_ch

        self.local_ch = local_ch
        self.global_ch = global_ch

        self.local_conv = nn.Sequential(
            nn.Conv2d(local_ch, local_ch, 3, 1, 1),
            nn.BatchNorm2d(local_ch),
            nn.ReLU()
        )

        self.global_conv = SpectralTransform(global_ch, global_ch)

        self.dwt = DWTForward(J=1, mode='zero', wave=wavelet)
        self.freq_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        self.fusion = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_x = x[:, :self.local_ch, :, :]
        global_x = x[:, self.local_ch:, :, :]

        local_out = self.local_conv(local_x)
        global_out = self.global_conv(global_x)

        try:
            # DWT does not support float16 — force float32 under AMP
            with torch.amp.autocast('cuda', enabled=False):
                LL, _ = self.dwt(x.float())
            gate = self.freq_gate(F.interpolate(LL, size=x.shape[-2:], mode='bilinear', align_corners=False))
            gate = gate.to(x.dtype)  # Match input dtype
        except RuntimeError:
            gate = torch.ones(x.shape, device=x.device, dtype=x.dtype) * 0.5

        out = torch.cat([local_out, global_out], dim=1)
        out = self.fusion(out * gate) + x

        return out


class WaveFFCBlock(nn.Module):
    """
    Complete WaveFFC block with residual connection and normalization.
    """

    def __init__(self, channels: int, ratio: float = 0.5, wavelet: str = 'db3'):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.ffc = WaveFFC(channels, ratio, wavelet)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffc(self.norm(x))


class MultiScaleWaveFFC(nn.Module):
    """
    Multi-scale WaveFFC that processes features at multiple resolutions.
    """

    def __init__(self, channels: int, num_scales: int = 3):
        super().__init__()
        self.scales = nn.ModuleList([
            WaveFFCBlock(channels, ratio=max(0.1, 0.5 - i * 0.1))
            for i in range(num_scales)
        ])
        self.fusion = nn.Conv2d(channels * num_scales, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for scale_block in self.scales:
            outputs.append(scale_block(x))
        return self.fusion(torch.cat(outputs, dim=1))
