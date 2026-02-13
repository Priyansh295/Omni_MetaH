from .combined import WaveSSMLoss, MaskAwareLoss
from .perceptual import VGGPerceptualLoss, HighReceptiveFieldPerceptualLoss
from .frequency import FrequencyAwareLoss, AdaptiveFrequencyLoss, MultiScaleFrequencyLoss
from .ssim import SSIMLoss
from .adversarial import AdversarialWaveletDiscriminator, SADiscriminator
