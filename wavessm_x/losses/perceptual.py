import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    """
    Standard VGG Perceptual Loss (Johnson et al. 2016).
    Computes L1 distance between features extracted from VGG16.
    """
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # Load VGG16 ONCE, then slice into feature blocks
        vgg_features = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        blocks = []
        # Features: relu1_2, relu2_2, relu3_3, relu4_3
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
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Clamp to valid range to prevent normalization overflow
        input = input.clamp(0, 1)
        target = target.clamp(0, 1)

        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


class HighReceptiveFieldPerceptualLoss(nn.Module):
    """
    High Receptive Field Perceptual Loss (LaMa style).
    Uses deeper layers of VGG19 (relu5_4) to capture global structure.
    Also includes standard VGG19 features for consistency.
    """
    def __init__(self):
        super().__init__()
        # VGG19 features: relu1_2, relu2_2, relu3_2, relu4_2, relu5_2
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential() # relu5_2 (deep features)
        
        # Build slices
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
            
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, pred, target):
        # Normalize: assumes input is [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        loss = 0.0
        x_pred = pred_norm
        x_target = target_norm
        
        # Sequential feeding through slices (correct)
        for i, (slicer, weight) in enumerate(zip(
            [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5],
            [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] # Weigh deep features more
        )):
            x_pred = slicer(x_pred)
            x_target = slicer(x_target)
            loss += weight * F.l1_loss(x_pred, x_target)
            
        return loss
