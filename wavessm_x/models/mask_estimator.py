import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MaskEstimator(nn.Module):
    """
    Lightweight mask prediction network for true blind inpainting.

    Uses a pretrained ResNet18 encoder with a lightweight decoder
    to predict soft masks indicating corrupted regions.

    Architecture:
        Input (3, H, W) -> ResNet18 Encoder -> 512 features
        -> Decoder (5 upsampling blocks) -> Soft Mask (1, H, W)
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Robustly handle non-3-channel inputs
        if in_channels != 3:
            old_conv = self.encoder[0]
            # ResNet conv1 is 7x7 stride 2
            self.encoder[0] = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

        self._init_decoder()

    def _init_decoder(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (B, 3, H, W) in [0, 1] range

        Returns:
            Soft mask (B, 1, H, W) in [0, 1] range
            where 1 indicates corrupted regions
        """
        original_size = x.shape[-2:]

        features = self.encoder(x)

        mask = self.decoder(features)

        if mask.shape[-2:] != original_size:
            mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=False)

        return mask


class LightweightMaskEstimator(nn.Module):
    """
    Even lighter mask estimator using depthwise separable convolutions.
    ~4x fewer params than ResNet18-based MaskEstimator.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.encoder = nn.Sequential(
            self._conv_block(in_channels, 32, stride=2),
            self._conv_block(32, 64, stride=2),
            self._conv_block(64, 128, stride=2),
            self._conv_block(128, 256, stride=2),
            self._conv_block(256, 512, stride=2),
        )

        self.decoder = nn.Sequential(
            self._upconv_block(512, 256),
            self._upconv_block(256, 128),
            self._upconv_block(128, 64),
            self._upconv_block(64, 32),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch),  # depthwise
            nn.Conv2d(in_ch, out_ch, 1),                           # pointwise
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _upconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        features = self.encoder(x)
        mask = self.decoder(features)

        if mask.shape[-2:] != original_size:
            mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=False)

        return mask
