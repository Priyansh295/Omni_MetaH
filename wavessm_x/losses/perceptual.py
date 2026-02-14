import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    """
    Standard VGG Perceptual Loss (Johnson et al. 2016).
    Computes L1 distance between features extracted from VGG16.
    
    FIX: Entire forward runs in float32 (autocast disabled).
    VGG deeper layers (relu3_3, relu4_3) produce feature magnitudes
    that overflow float16 (~65504 max) after ~1-4k training iters.
    """
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg_features = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        blocks = []
        blocks.append(vgg_features[:4])
        blocks.append(vgg_features[4:9])
        blocks.append(vgg_features[9:16])
        blocks.append(vgg_features[16:23])
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
        self.blocks = nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), requires_grad=False)

    def forward(self, input, target):
        # ── FIX: disable autocast for entire VGG forward ──
        with torch.amp.autocast('cuda', enabled=False):
            input = input.float()
            target = target.float()
            
            if input.shape[1] != 3:
                input = input.repeat(1, 3, 1, 1)
                target = target.repeat(1, 3, 1, 1)

            input = input.clamp(0, 1)
            target = target.clamp(0, 1)

            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std
            
            if self.resize:
                input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
                target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
                
            loss = 0.0
            x = input
            y = target
            for i, block in enumerate(self.blocks):
                x = block(x)
                y = block(y)
                loss += F.l1_loss(x, y)
        return loss


class HighReceptiveFieldPerceptualLoss(nn.Module):
    """
    High Receptive Field Perceptual Loss (LaMa style).
    Uses deeper layers of VGG19 (relu5_4) to capture global structure.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg[x])
            
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, pred, target):
        # ── FIX: disable autocast for entire VGG19 forward ──
        with torch.amp.autocast('cuda', enabled=False):
            pred = pred.float()
            target = target.float()
            
            mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
            
            pred_norm = (pred - mean) / std
            target_norm = (target - mean) / std
            
            loss = 0.0
            x_pred = pred_norm
            x_target = target_norm
            
            for i, (slicer, weight) in enumerate(zip(
                [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5],
                [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
            )):
                x_pred = slicer(x_pred)
                x_target = slicer(x_target)
                loss += weight * F.l1_loss(x_pred, x_target)
                
        return loss