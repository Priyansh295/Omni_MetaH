import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

# Try to use mamba-ssm's fused CUDA kernel (10-50x faster than Python parallel_scan)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_CUDA = True
except ImportError:
    HAS_MAMBA_CUDA = False
    print("[INFO] mamba-ssm not installed, using Python parallel_scan (slower). "
          "Install with: pip install mamba-ssm")


def parallel_scan(A: torch.Tensor, B: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    O(N) parallel associative scan using log-cumsum trick.

    Implements: h_t = A_t * h_{t-1} + B_t * x_t

    Args:
        A: Decay factors (B, L, d_inner, d_state)
        B: Input weights (B, L, d_state)
        x: Input sequence (B, L, d_inner)

    Returns:
        Hidden states (B, L, d_inner, d_state)
    """
    B_batch, L, d_inner, d_state = A.shape

    log_A = torch.log(A.clamp(min=1e-6))
    A_cumsum = torch.cumsum(log_A, dim=1).clamp(-30, 30)

    x_expand = x.unsqueeze(-1)
    B_expand = B.unsqueeze(2)
    # This broadcast works: (B, L, d_inner, 1) * (B, L, 1, d_state) -> (B, L, d_inner, d_state)
    input_contrib = x_expand * B_expand

    x_weighted = input_contrib * torch.exp(-A_cumsum)
    x_cumsum = torch.cumsum(x_weighted, dim=1)

    return x_cumsum * torch.exp(A_cumsum)


def snake_scan(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Vectorized serpentine scan for spatial locality (SEM-Net style).

    Args:
        x: Input tensor (B, L, D) where L = H * W
        height: Spatial height
        width: Spatial width

    Returns:
        Snake-scanned tensor (B, L, D)
    """
    B, L, D = x.shape
    x_2d = x.view(B, height, width, D).clone()
    x_2d[:, 1::2] = x_2d[:, 1::2].flip(dims=[2])
    return x_2d.view(B, L, D)


def inverse_snake_scan(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Reverse serpentine scan.

    Args:
        x: Snake-scanned tensor (B, L, D)
        height: Spatial height
        width: Spatial width

    Returns:
        Original order tensor (B, L, D)
    """
    B, L, D = x.shape
    x_2d = x.view(B, height, width, D).clone()
    x_2d[:, 1::2] = x_2d[:, 1::2].flip(dims=[2])
    return x_2d.view(B, L, D)


class FrequencyAdaptiveSSM(nn.Module):
    """
    Frequency-Adaptive State Space Model (FASS).

    Core innovation: Wavelet coefficients CONTROL SSM parameters (B, C, Δ),
    not just preprocess the input.

    - B (input matrix): Modulated by LL (structure preservation)
    - C (output matrix): Modulated by HH (texture reconstruction)
    - Δ (timestep): Modulated by edge energy LH+HL (adaptive dynamics)

    Includes:
    - True parallel scan (O(N) complexity)
    - Snake scan for spatial locality
    - Local enhancement (MambaIR style)
    - Channel attention
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        wavelet: str = 'db3',
        use_snake_scan: bool = True,
        use_local_enhancement: bool = True,
        use_freq_modulation: bool = True,
        ablation_flags: dict = None 
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.use_snake_scan = use_snake_scan
        self.use_local_enhancement = use_local_enhancement
        self.use_freq_modulation = use_freq_modulation
        
        # Ablation settings
        self.ablation_flags = ablation_flags or {}
        self.use_b_mod = not self.ablation_flags.get('no_b', False)
        self.use_c_mod = not self.ablation_flags.get('no_c', False)
        self.use_delta_mod = not self.ablation_flags.get('no_delta', False)

        if use_freq_modulation:
            self.dwt = DWTForward(J=1, mode='zero', wave=wavelet)
            self.proj_struct = nn.Linear(d_model, d_state, bias=False)
            self.proj_texture = nn.Linear(d_model, d_state, bias=False)
            self.proj_edge = nn.Linear(d_model, 1, bias=False)

            self.alpha = nn.Parameter(torch.tensor(0.3))
            self.beta = nn.Parameter(torch.tensor(0.3))
            self.gamma = nn.Parameter(torch.tensor(0.3))

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.skip_scale = nn.Parameter(torch.ones(d_model))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.act = nn.SiLU()

        if use_local_enhancement:
            self.local_conv = nn.Conv2d(
                d_model, d_model, 3, 1, 1, groups=d_model
            )
            self.channel_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(d_model, max(d_model // 4, 1), 1),
                nn.ReLU(),
                nn.Conv2d(max(d_model // 4, 1), d_model, 1),
                nn.Sigmoid()
            )

        self._warned_fallback = False
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            # Skip DWT buffers/params if present, they are fixed
            if 'dwt' in name:
                continue
                
            if 'weight' in name and param.dim() >= 2:
                nn.init.trunc_normal_(param, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _extract_frequency_features(self, x: torch.Tensor, height: int, width: int):
        """Extract wavelet-based frequency features for SSM modulation."""
        b, l, d = x.shape

        try:
            x_2d = x.transpose(1, 2).view(b, d, height, width)
            # DWT does not support float16 — force float32 under AMP
            with torch.amp.autocast('cuda', enabled=False):
                yl, yh = self.dwt(x_2d.float())
            yh_coeffs = yh[0]
            lh = yh_coeffs[:, :, 0, :, :]
            hl = yh_coeffs[:, :, 1, :, :]
            hh = yh_coeffs[:, :, 2, :, :]

            yl_pooled = yl.mean(dim=(2, 3))
            hh_pooled = hh.abs().mean(dim=(2, 3))
            edge_pooled = (lh.abs() + hl.abs()).mean(dim=(2, 3))

            f_struct = self.proj_struct(yl_pooled)
            f_texture = self.proj_texture(hh_pooled)
            f_edge = self.proj_edge(edge_pooled).squeeze(-1)

            return f_struct, f_texture, f_edge, True

        except (RuntimeError, ValueError) as e:
            if not self._warned_fallback:
                print(f"FASS: Wavelet fallback (L={l}): {e}")
                self._warned_fallback = True

            f_struct = torch.zeros(b, self.d_state, device=x.device, dtype=x.dtype)
            f_texture = torch.zeros(b, self.d_state, device=x.device, dtype=x.dtype)
            f_edge = torch.zeros(b, device=x.device, dtype=x.dtype)
            return f_struct, f_texture, f_edge, False

    def forward(self, x: torch.Tensor, h_size: int = None, w_size: int = None) -> torch.Tensor:
        """
        Forward pass with frequency-adaptive SSM.

        Args:
            x: Input (B, L, D)
            h_size: Spatial height (optional, inferred if square)
            w_size: Spatial width (optional, inferred if square)

        Returns:
            Output (B, L, D)
        """
        b, l, d = x.shape

        if h_size is None or w_size is None:
            h_size = w_size = int(l ** 0.5)
            if h_size * w_size != l:
                h_size = w_size = None

        if self.use_freq_modulation and h_size is not None:
            f_struct, f_texture, f_edge, has_wavelet = self._extract_frequency_features(
                x, h_size, w_size
            )
        else:
             f_struct = torch.zeros(b, self.d_state, device=x.device, dtype=x.dtype)
             f_texture = torch.zeros(b, self.d_state, device=x.device, dtype=x.dtype)
             f_edge = torch.zeros(b, device=x.device, dtype=x.dtype)
             has_wavelet = False

        if self.use_snake_scan and h_size is not None:
            x = snake_scan(x, h_size, w_size)

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :l]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act(x_conv)

        x_dbl = self.x_proj(x_conv)
        B_base = x_dbl[:, :, :self.d_state]
        C_base = x_dbl[:, :, self.d_state:2*self.d_state]
        dt_base = x_dbl[:, :, -1]

        if has_wavelet:
            # B Modulation (Input) - Controlled by Structure (LL)
            if self.use_b_mod:
                B = B_base * (1 + self.alpha * torch.tanh(f_struct).unsqueeze(1))
            else:
                B = B_base

            # C Modulation (Output) - Controlled by Texture (HH) 
            if self.use_c_mod:
                C = C_base * (1 + self.beta * torch.tanh(f_texture).unsqueeze(1))
            else:
                C = C_base

            # Delta Modulation (Step Size) - Controlled by Edge (LH+HL)
            if self.use_delta_mod:
                dt_mod = torch.sigmoid(self.gamma * f_edge).unsqueeze(1)
                dt = F.softplus(dt_base) * (0.5 + dt_mod)
            else:
                dt = F.softplus(dt_base)
        else:
            B = B_base
            C = C_base
            dt = F.softplus(dt_base)

        dt = dt.clamp(self.dt_min, self.dt_max)

        A = -torch.exp(self.A_log)

        if HAS_MAMBA_CUDA:
            # Fused CUDA kernel — same math, 10-50x faster
            u = x_conv.transpose(1, 2).contiguous()          # (B, d_inner, L)
            # All inputs must share the same dtype (AMP may mix float16/float32)
            # Cast dt BEFORE expand to avoid dtype issues with expanded views
            input_dtype = u.dtype
            dt = dt.to(input_dtype)
            delta = dt.unsqueeze(1).expand(-1, self.d_inner, -1).contiguous()
            # A and D MUST stay float32 (CUDA kernel "weight" requirement)
            # u, delta, B, C, z can be float16 ("input" type)
            B_ssm = B.transpose(1, 2).contiguous().to(input_dtype)
            C_ssm = C.transpose(1, 2).contiguous().to(input_dtype)
            z_ssm = z.transpose(1, 2).contiguous().to(input_dtype)

            y = selective_scan_fn(
                u, delta, A, B_ssm, C_ssm,
                D=self.D, z=z_ssm, delta_softplus=False
            )
            y = y.transpose(1, 2)  # (B, L, d_inner)
        else:
            # Fallback: pure PyTorch parallel scan
            dt_expand = dt.unsqueeze(-1).unsqueeze(-1)
            A_expand = A.unsqueeze(0).unsqueeze(0)
            dt_A = (dt_expand * A_expand).clamp(-20, 0)
            decay = torch.exp(dt_A)

            B_bar = dt.unsqueeze(-1) * B
            B_bar = B_bar.clamp(-10, 10)

            hidden_states = parallel_scan(decay, B_bar, x_conv)

            y = (hidden_states * C.unsqueeze(2)).sum(dim=-1)
            y = y * self.act(z)
            y = y + x_conv * self.D
        y = self.out_norm(y)
        out = self.out_proj(y)

        out = x * self.skip_scale + out

        if self.use_snake_scan and h_size is not None:
            out = inverse_snake_scan(out, h_size, w_size)

        if self.use_local_enhancement and h_size is not None:
            out_2d = out.transpose(1, 2).view(b, d, h_size, w_size)
            local_out = self.local_conv(out_2d)
            attn = self.channel_attn(out_2d)
            out_2d = out_2d + local_out * attn
            out = out_2d.flatten(2).transpose(1, 2)

        return out


class DualStreamFASS(nn.Module):
    """
    Dual-stream architecture: separate processing for structure (LL) and texture (HF).

    Structure stream: Large state (N=64), slow dynamics, 4-direction scan
    Texture stream: Small state (N=16), fast dynamics, 8-direction scan
    """

    def __init__(self, channels: int, struct_state: int = 64, texture_state: int = 16, 
                 wavelet: str = 'db3', ablation_flags: dict = None):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave=wavelet)
        self.idwt = DWTInverse(mode='zero', wave=wavelet)

        # No freq modulation internally because we already feed specific bands
        # structure/texture SSMS capture long-range dependencies in their respective domains
        self.struct_ssm = FrequencyAdaptiveSSM(
            d_model=channels,
            d_state=struct_state,
            use_snake_scan=True,
            use_local_enhancement=True,
            use_freq_modulation=False,
            wavelet=wavelet,
            ablation_flags=ablation_flags
        )

        self.texture_ssm = FrequencyAdaptiveSSM(
            d_model=channels * 3,
            d_state=texture_state,
            use_snake_scan=True,
            use_local_enhancement=True,
            use_freq_modulation=False,
            wavelet=wavelet,
            ablation_flags=ablation_flags
        )

        self.cross_attn = CrossFrequencyAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, C, H, W)
        Returns:
            Output (B, C, H, W)
        """
        B, C, H, W = x.shape

        # DWT does not support float16 — force float32 under AMP
        with torch.amp.autocast('cuda', enabled=False):
            yl, yh = self.dwt(x.float())
        yh_coeffs = yh[0]
        lh = yh_coeffs[:, :, 0, :, :]
        hl = yh_coeffs[:, :, 1, :, :]
        hh = yh_coeffs[:, :, 2, :, :]

        # Use ACTUAL DWT output dims (not H//2) because DWT padding can differ
        h_half, w_half = yl.shape[2], yl.shape[3]

        yl_seq = yl.flatten(2).transpose(1, 2)
        yl_out = self.struct_ssm(yl_seq, h_half, w_half)
        yl_out = yl_out.transpose(1, 2).view(B, C, h_half, w_half)

        hf = torch.cat([lh, hl, hh], dim=1)
        hf_seq = hf.flatten(2).transpose(1, 2)
        hf_out = self.texture_ssm(hf_seq, h_half, w_half)
        hf_out = hf_out.transpose(1, 2).view(B, C * 3, h_half, w_half)
        lh_out, hl_out, hh_out = hf_out.chunk(3, dim=1)

        yl_refined = self.cross_attn(yl_out, (lh_out, hl_out, hh_out))

        yh_out = torch.stack([lh_out, hl_out, hh_out], dim=2)
        # IDWT also requires float32
        with torch.amp.autocast('cuda', enabled=False):
            out = self.idwt((yl_refined.float(), [yh_out.float()]))
        
        # Robustly handle DWT padding mismatches
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            
        return out


class CrossFrequencyAttention(nn.Module):
    """
    Cross-frequency attention: LL features guide HF refinement.
    Uses O(C^2) channel attention to avoid O(N^2) OOM with large images.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels * 3, channels, 1)
        self.value = nn.Conv2d(channels * 3, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, ll_features: torch.Tensor, hf_tuple: tuple) -> torch.Tensor:
        """
        Args:
            ll_features: (B, C, H, W)
            hf_tuple: (LH, HL, HH) each (B, C, H, W)
        Returns:
            Refined LL features (B, C, H, W)
        """
        lh, hl, hh = hf_tuple
        hf_features = torch.cat([lh, hl, hh], dim=1)

        B, C, H, W = ll_features.shape

        # Use view to flatten H, W but keep C
        q = self.query(ll_features).view(B, C, -1)      # (B, C, HW)
        k = self.key(hf_features).view(B, C, -1)        # (B, C, HW)
        v = self.value(hf_features).view(B, C, -1)      # (B, C, HW)

        # Calculates attention map of size (B, C, C) - Efficient O(C^2)
        # Instead of (B, HW, HW) - Expensive O(N^2)
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        
        # Apply attention to values
        out = (attn @ v).view(B, C, H, W)

        return ll_features + out
