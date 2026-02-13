import torch
import torch.nn as nn
import torch.nn.functional as F
from .odconv import ODConv2d
from .fass_ssm import FrequencyAdaptiveSSM, DualStreamFASS
from .wave_ffc import MultiScaleWaveFFC


def get_activation(act_type='gelu'):
    """Get activation function based on name"""
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def get_normalization(norm_type, channels):
    """Get normalization layer based on name"""
    norm_type = norm_type.lower()
    if norm_type == 'batchnorm':
        return nn.BatchNorm2d(channels)
    elif norm_type == 'instancenorm':
        return nn.InstanceNorm2d(channels)
    elif norm_type == 'layernorm':
        return nn.LayerNorm(channels) # Note: LayerNorm usually requires [..., C, H, W] -> [..., C, H, W] adaptation or permute
    else:
        return nn.Identity()


class od_attention(nn.Module):
    def __init__(self, channels, kernel_size=3, act_type='gelu'):
        super(od_attention, self).__init__()
        padding = kernel_size // 2
        self.od_conv = ODConv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.act = get_activation(act_type)

    def forward(self, x):
        return self.act(self.od_conv(x) + self.conv(x))


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor, kernel_size=3, act_type='gelu'):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        padding = kernel_size // 2
        
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=kernel_size, stride=1, padding=padding,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.act = get_activation(act_type)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(self.act(x1) * x2)


class MambaAttention(nn.Module):
    """
    Frequency-Adaptive SSM for O(N) context modeling.
    ALWAYS uses FrequencyAdaptiveSSM - our core contribution.
    """
    def __init__(self, channels, d_state=16, d_conv=4, expand=2, kernel_size=3, wavelet='db3', ablation_flags=None):
        super(MambaAttention, self).__init__()
        
        self.mamba = FrequencyAdaptiveSSM(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_snake_scan=True,
            use_local_enhancement=True,
            wavelet=wavelet,
            ablation_flags=ablation_flags
        )

        # Optimization: Use GroupNorm(1, C) instead of LayerNorm
        # This allows 4D input (B,C,H,W) without expensive permute/flatten
        self.norm = nn.GroupNorm(1, channels) 
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x, return_attn=False):
        b, c, h, w = x.shape

        # Normalize 4D input directly (GroupNorm)
        x_norm = self.norm(x)

        # Flatten for Mamba: (B, C, H, W) -> (B, L, C)
        x_seq = x_norm.flatten(2).transpose(1, 2)
        
        # Apply Mamba SSM
        out_seq = self.mamba(x_seq, h, w)
        
        # Reshape back: (B, L, C) -> (B, C, H, W)
        out = out_seq.transpose(1, 2).view(b, c, h, w)
        out = self.project_out(out)

        if return_attn:
            return out, None
        return out


class MambaBlock(nn.Module):
    """
    Mamba Block = MambaAttention + GDFN
    Drop-in replacement for TransformerBlock with O(N) complexity.
    """
    def __init__(self, channels, expansion_factor, d_state=16, d_conv=4, expand=2,
                 kernel_size=3, act_type='gelu', norm_type='layernorm', 
                 wavelet='db3', ablation_flags=None):
        super(MambaBlock, self).__init__()
        
        self.norm_type = norm_type.lower()
        if self.norm_type == 'layernorm':
             # Optimization: GroupNorm(1, C) is mathematically equivalent to LayerNorm
             # but works on (B, C, H, W) without permuting
             self.norm1 = nn.GroupNorm(1, channels)
             self.norm2 = nn.GroupNorm(1, channels)
        else:
             self.norm1 = get_normalization(norm_type, channels)
             self.norm2 = get_normalization(norm_type, channels)
        
        self.attn = MambaAttention(channels, d_state, d_conv, expand, kernel_size, 
                                   wavelet=wavelet, ablation_flags=ablation_flags)
        self.ffn = GDFN(channels, expansion_factor, kernel_size, act_type)

    def forward(self, x, return_attn=False):
        # Attention (MambaAttention has internal Norm, so we skip norm1 here or use it?)
        # Standard implementation: x = x + attn(norm1(x))
        # But MambaAttention usually takes raw x and norms internally. 
        # Current code: x = x + x_in (residual). MambaAttention does x_seq = self.norm(x_seq).
        # So we don't need norm1 here if MambaAttention does it.
        # Let's keep it consistent with previous code structure but use GroupNorm logic if we were valid.
        
        x_in = x
        attn_out = self.attn(x, return_attn=return_attn) 
        if return_attn:
            x, attn_map = attn_out
        else:
            x = attn_out
            attn_map = None
        x = x + x_in

        # FFN with pre-norm
        x_in = x
        # Efficient 4D Norm
        x_normed = self.norm2(x) 
        x = self.ffn(x_normed) + x_in

        if return_attn:
            return x, attn_map
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor, kernel_size=3, act_type='gelu', norm_type='layernorm'):
        super(TransformerBlock, self).__init__()

        self.norm_type = norm_type.lower()
        if self.norm_type == 'layernorm':
            # Optimization: GroupNorm(1, C) is equivalent to LayerNorm but 4D-native
            self.norm1 = nn.GroupNorm(1, channels)
            self.norm2 = nn.GroupNorm(1, channels)
        else:
            self.norm1 = get_normalization(norm_type, channels)
            self.norm2 = get_normalization(norm_type, channels)

        self.attn = od_attention(channels, kernel_size=kernel_size)
        self.ffn = GDFN(channels, expansion_factor, kernel_size, act_type)

    def forward(self, x, return_attn=False):
        # Pre-norm architecture: x = x + block(norm(x))
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        if return_attn:
            return x, None
        return x


class DownSample(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(DownSample, self).__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=kernel_size, padding=padding, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(UpSample, self).__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=kernel_size, padding=padding, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Inpainting(nn.Module):
    """
    WaveSSM-X Inpainting Model (formerly MOBOWN).
    
    Args:
        use_mamba (bool): If True, use MambaBlock instead of TransformerBlock.
        d_state, d_conv, expand: Mamba parameters.
    """
    def __init__(self, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48//3, 96//3, 192//3, 384//3], num_refinement=4,
                 expansion_factor=2.66, kernel_size=3, act_type='gelu', norm_type='layernorm',
                 use_mamba=False, d_state=16, d_conv=4, expand=2, use_fass=True, use_ffc=True,
                 wavelet='db3', fass_no_b=False, fass_no_c=False, fass_no_delta=False):
        super(Inpainting, self).__init__()
        
        self.use_mamba = use_mamba
        self.use_fass = use_fass
        self.use_ffc = use_ffc
        self.num_blocks = num_blocks
        self.channels = channels
        
        # Ablation configuration
        self.ablation_flags = {
            'no_b': fass_no_b,
            'no_c': fass_no_c,
            'no_delta': fass_no_delta
        }
        self.wavelet = wavelet
        
        padding = kernel_size // 2

        self.input_conv = nn.Conv2d(3, channels[0], kernel_size=kernel_size, padding=padding, bias=False)

        self.encoder_layers = nn.ModuleList([])
        self.downsample_layers = nn.ModuleList([])
        
        Block = MambaBlock if use_mamba else TransformerBlock
        # Pass ablation flags to MambaBlocks
        block_kwargs = {
            'd_state': d_state, 
            'd_conv': d_conv, 
            'expand': expand,
            'wavelet': wavelet,
            'ablation_flags': self.ablation_flags
        } if use_mamba else {'num_heads': 0}

        for i in range(len(num_blocks)):
            layers = []
            for _ in range(num_blocks[i]):
                if use_mamba:
                     layers.append(Block(channels[i], expansion_factor, kernel_size=kernel_size, 
                                       act_type=act_type, norm_type=norm_type, **block_kwargs))
                else:
                     layers.append(Block(channels[i], num_heads[i], expansion_factor, kernel_size=kernel_size,
                                       act_type=act_type, norm_type=norm_type))
            self.encoder_layers.append(nn.Sequential(*layers))
            
            if i < len(num_blocks) - 1:
                self.downsample_layers.append(DownSample(channels[i], kernel_size=kernel_size))

        self.upsample_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        
        for i in range(len(num_blocks) - 2, -1, -1):
            self.upsample_layers.append(UpSample(channels[i+1], kernel_size=kernel_size))
            
            layers = []
            for _ in range(num_blocks[i]):
                if use_mamba:
                     layers.append(Block(channels[i], expansion_factor, kernel_size=kernel_size,
                                       act_type=act_type, norm_type=norm_type, **block_kwargs))
                else:
                     layers.append(Block(channels[i], num_heads[i], expansion_factor, kernel_size=kernel_size,
                                       act_type=act_type, norm_type=norm_type))
            self.decoder_layers.append(nn.Sequential(*layers))

        # ── Bottleneck: Dual-stream wavelet SSM (structure + texture separation) ──
        if use_mamba and use_fass:
            self.bottleneck_fass = DualStreamFASS(channels[-1], wavelet=wavelet, ablation_flags=self.ablation_flags)
        else:
            self.bottleneck_fass = None

        # ── Refinement: Multi-scale WaveFFC (global FFT context before output) ──
        if use_ffc:
            self.refinement_ffc = MultiScaleWaveFFC(channels[0], num_scales=3)
        else:
            self.refinement_ffc = None

        self.output = nn.Conv2d(channels[0], 3, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x, return_attn=False):
        return self._forward_impl(x, return_attn)

    def _forward_impl(self, x, return_attn=False):
        x_input = x  # Save original image for global residual
        x = self.input_conv(x)
        x_skip = []

        # Encoder
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            x_skip.append(x)
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)

        # Bottleneck: wavelet-domain dual-stream processing
        if self.bottleneck_fass is not None:
            x = self.bottleneck_fass(x) + x  # Residual for safe gradient flow

        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            x = self.upsample_layers[i](x)
            x = x + x_skip[len(x_skip) - 2 - i]
            x = layer(x)

        # Refinement: multi-scale FFT global context
        if self.refinement_ffc is not None:
             x = self.refinement_ffc(x) + x  # Residual for safe gradient flow

        x = self.output(x) + x_input  # Global residual to original image
        
        if return_attn:
             return x, None
        return x
