import torch
import torch.nn as nn
import torchvision.transforms.functional as T
import random


class AdvancedAugmentation:
    """
    Advanced data augmentation techniques for improved generalization
    """
    def __init__(self, probability: float = 0.5):
        self.prob = probability
        
    def apply_color_jitter(self, img: torch.Tensor) -> torch.Tensor:
        """Apply subtle color jittering"""
        if random.random() < self.prob:
            img = T.adjust_brightness(img, 1.0 + random.uniform(-0.1, 0.1))
            img = T.adjust_contrast(img, 1.0 + random.uniform(-0.1, 0.1))
            img = T.adjust_saturation(img, 1.0 + random.uniform(-0.1, 0.1))
        return img
        
    def apply_gaussian_noise(self, img: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        """Add subtle Gaussian noise"""
        if random.random() < self.prob:
            noise = torch.randn_like(img) * noise_std
            img = torch.clamp(img + noise, 0, 1) # Ensure valid range
            
            # Additional safety: guard against zero std if normalization happens here
            # (Though dataset.py does standard ImageNet norm usually, 
            #  this fixes potential noise-induced issues)
        return img
        
    def apply_random_erasing(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random erasing for robustness"""
        if random.random() < self.prob:
            _, h, w = img.shape
            # Create a small hole
            x = random.randint(0, w - w//4)
            y = random.randint(0, h - h//4)
            h_erase = random.randint(h//10, h//5)
            w_erase = random.randint(w//10, w//5)
            
            img[:, y:y+h_erase, x:x+w_erase] = 0.0
        return img
