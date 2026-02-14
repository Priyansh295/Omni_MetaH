"""
Perceptual Loss using VGG19 features
FIXED: FP32 enforcement to prevent overflow in mixed precision training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    """
    VGG19-based perceptual loss.
    
    CRITICAL FIX: Entire forward pass runs in FP32 to prevent activation overflow.
    VGG was trained in FP32 and its intermediate activations can exceed float16 range.
    """
    def __init__(self, layers=None, weights=None):
        super().__init__()
        
        # Load pre-trained VGG19
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        
        # Default feature extraction layers
        if layers is None:
            # Use relu outputs from different depth levels
            layers = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']
        
        # Default weights for each layer
        if weights is None:
            weights = [1.0 / len(layers)] * len(layers)
        
        self.layers = layers
        self.weights = weights
        
        # Build feature extractor
        self.feature_extractor = self._build_feature_extractor(vgg)
        
        # Freeze all parameters (no training)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_extractor.eval()
        
        # VGG normalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _build_feature_extractor(self, vgg):
        """Build feature extractor from VGG19."""
        # Map layer names to VGG indices
        layer_map = {
            'relu1_1': 1,  'relu1_2': 3,
            'relu2_1': 6,  'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
            'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
            'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35,
        }
        
        # Get maximum index needed
        max_idx = max(layer_map[layer] for layer in self.layers)
        
        # Extract layers up to max_idx
        features = nn.Sequential(*list(vgg.features.children())[:max_idx + 1])
        
        return features
    
    def normalize(self, x):
        """Normalize input using ImageNet statistics."""
        return (x - self.mean) / self.std
    
    def extract_features(self, x):
        """Extract features at specified layers."""
        # Map layer names to VGG indices
        layer_map = {
            'relu1_1': 1,  'relu1_2': 3,
            'relu2_1': 6,  'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
            'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
            'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35,
        }
        
        features = []
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            # Check if this is one of our target layers
            for target_layer in self.layers:
                if layer_map[target_layer] == i:
                    features.append(x)
        
        return features
    
    def forward(self, pred, target):
        """
        Compute perceptual loss.
        
        CRITICAL: Entire computation in FP32 to prevent overflow.
        
        Args:
            pred: Predicted image [B, 3, H, W] in range [0, 1]
            target: Target image [B, 3, H, W] in range [0, 1]
        
        Returns:
            Perceptual loss (scalar)
        """
        # ═══════════════════════════════════════════════════════════
        # CRITICAL FIX: Force FP32 for entire VGG forward pass
        # ═══════════════════════════════════════════════════════════
        with torch.amp.autocast('cuda', enabled=False):
            # Convert to FP32
            pred = pred.float()
            target = target.float()
            
            # Normalize using ImageNet stats
            pred = self.normalize(pred)
            target = self.normalize(target)
            
            # Extract features
            pred_features = self.extract_features(pred)
            target_features = self.extract_features(target)
            
            # Compute weighted L1 loss across all feature layers
            loss = 0.0
            for weight, pred_feat, target_feat in zip(self.weights, pred_features, target_features):
                loss += weight * F.l1_loss(pred_feat, target_feat)
        
        return loss


class StyleLoss(nn.Module):
    """
    Style loss using Gram matrices.
    
    Can be combined with perceptual loss for style transfer tasks.
    Also uses FP32 for stability.
    """
    def __init__(self, layers=None):
        super().__init__()
        
        if layers is None:
            layers = ['relu2_2', 'relu3_4', 'relu4_4']
        
        self.perceptual_loss = PerceptualLoss(layers=layers)
    
    def gram_matrix(self, x):
        """Compute Gram matrix (feature correlations)."""
        B, C, H, W = x.shape
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (C * H * W)
    
    def forward(self, pred, target):
        """Compute style loss using Gram matrices."""
        with torch.amp.autocast('cuda', enabled=False):
            pred = pred.float()
            target = target.float()
            
            # Normalize
            pred = self.perceptual_loss.normalize(pred)
            target = self.perceptual_loss.normalize(target)
            
            # Extract features
            pred_features = self.perceptual_loss.extract_features(pred)
            target_features = self.perceptual_loss.extract_features(target)
            
            # Compute style loss using Gram matrices
            loss = 0.0
            for pred_feat, target_feat in zip(pred_features, target_features):
                pred_gram = self.gram_matrix(pred_feat)
                target_gram = self.gram_matrix(target_feat)
                loss += F.mse_loss(pred_gram, target_gram)
        
        return loss


class HighReceptiveFieldPerceptualLoss(nn.Module):
    """
    High Receptive Field Perceptual Loss (LaMa style).
    Uses deeper layers of VGG19 (relu5_4) to capture global structure.
    
    CRITICAL: Entire computation in FP32 to prevent overflow.
    """
    def __init__(self):
        super().__init__()
        # Reuse PerceptualLoss logic but with specific layers/weights
        self.impl = PerceptualLoss(
            layers=['relu1_2', 'relu2_2', 'relu3_2', 'relu4_2', 'relu5_2'],
            weights=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        )
            
    def forward(self, pred, target):
        return self.impl(pred, target)


# ═══════════════════════════════════════════════════════════
# Simpler alternative: Use LPIPS if available
# ═══════════════════════════════════════════════════════════

# Alias for backward compatibility
VGGPerceptualLoss = PerceptualLoss

try:
    import lpips
    
    class LPIPSLoss(nn.Module):
        """
        Learned Perceptual Image Patch Similarity.
        
        Often more stable than raw VGG features.
        """
        def __init__(self, net='vgg'):
            super().__init__()
            self.lpips = lpips.LPIPS(net=net, verbose=False)
            self.lpips.eval()
            for param in self.lpips.parameters():
                param.requires_grad = False
        
        def forward(self, pred, target):
            """
            Compute LPIPS loss.
            
            Args:
                pred, target: Images in range [0, 1]
            
            Returns:
                LPIPS distance (lower is better)
            """
            with torch.amp.autocast('cuda', enabled=False):
                # LPIPS expects range [-1, 1]
                pred = pred.float() * 2.0 - 1.0
                target = target.float() * 2.0 - 1.0
                return self.lpips(pred, target).mean()

except ImportError:
    # LPIPS not available, use VGG-based loss
    pass
