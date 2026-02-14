import torch
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, sigma=1.5, size_average=True, window=None):
    # ── FIX: force float32 ──
    # img1*img1 squared then summed over 11x11 window overflows float16
    img1 = img1.float()
    img2 = img2.float()
    
    channel = img1.size(1)
    if window is None:
        window = create_window(window_size, channel, sigma).to(img1.device)
    window = window.float()

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    sigma1_sq = sigma1_sq.clamp(min=0)
    sigma2_sq = sigma2_sq.clamp(min=0)

    C1 = 0.01**2
    C2 = 0.03**2

    numerator = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    
    ssim_map = ssim_map.clamp(0, 1)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self._cached_window = None
        self._cached_channels = 0

    def forward(self, img1, img2):
        channel = img1.size(1)
        if self._cached_window is None or self._cached_channels != channel:
            self._cached_window = create_window(self.window_size, channel, self.sigma)
            self._cached_channels = channel
        window = self._cached_window.to(img1.device).float()
        return 1 - ssim(img1, img2, self.window_size, self.sigma, window=window)