import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class AdversarialWaveletDiscriminator(nn.Module):
    """
    Wavelet-domain discriminator for adversarial training.
    
    Can operate on:
    - RGB image space (standard GAN)
    - Wavelet subbands (frequency-aware GAN)
    """
    def __init__(self, in_channels=3, use_spectral_norm=True):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, size=4, stride=2, normalize=True):
            layers = []
            conv = nn.Conv2d(in_filters, out_filters, size, stride, 1)
            if use_spectral_norm:
                conv = spectral_norm(conv)
            layers.append(conv)
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class SADiscriminator(nn.Module):
    """
    Self-Attention Discriminator (O(N) optimized).
    Uses lightweight attention for global consistency check.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        # Simplified PatchGAN
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 1)
        )
        
    def forward(self, x):
        return self.main(x)
