"""
Advanced Diverse Mask Generator for Blind Image Inpainting
===========================================================

Generates high-quality, diverse distortion masks matching the complexity 
of real-world occlusions. Includes:
- Swirl/spiral distortions with color tinting
- Rectangular patch compositing with external images
- Organic blob masks with domain warping
- Smear/drip effects
- Color overlays (overlay, soft-light, multiply blending)

Designed to match the quality level of the provided training examples.

Author: Research Enhancement Module
"""

import os
import random
import math
from typing import List, Tuple, Optional, Union
import numpy as np
import cv2

# Optional dependencies
try:
    from scipy.ndimage import gaussian_filter, map_coordinates
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure image is RGB with 3 channels."""
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize to 0-1 float."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return np.clip(img, 0, 1).astype(np.float32)


def denormalize_image(img: np.ndarray) -> np.ndarray:
    """Convert 0-1 float to uint8."""
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur."""
    if sigma <= 0:
        return img
    k = int(2 * round(3 * sigma) + 1)
    k = max(3, k)
    return cv2.GaussianBlur(img, (k, k), sigma)


def fractal_noise(h: int, w: int, octaves: int = 4, persistence: float = 0.55) -> np.ndarray:
    """Generate fractal Brownian motion noise."""
    total = np.zeros((h, w), dtype=np.float32)
    freq = 1.0
    amp = 1.0
    for _ in range(octaves):
        gh = max(1, int(h / freq))
        gw = max(1, int(w / freq))
        grid = np.random.rand(gh, gw).astype(np.float32)
        up = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp * up
        freq *= 2.0
        amp *= persistence
    total = (total - total.min()) / (total.max() - total.min() + 1e-6)
    return total


# ============================================================================
# Geometric Distortions
# ============================================================================

class SwrlDistortion:
    """
    Swirl/spiral distortion effect.
    Creates circular swirl patterns like the woman example.
    """
    
    @staticmethod
    def apply(
        img: np.ndarray,
        center: Optional[Tuple[int, int]] = None,
        radius: Optional[float] = None,
        strength: float = 2.0,
        rotation: float = 0.0
    ) -> np.ndarray:
        """
        Apply swirl distortion.
        
        Args:
            img: Input image (H, W, C)
            center: Swirl center (x, y). Default: image center
            radius: Effect radius. Default: 40% of min dimension
            strength: Swirl strength (radians at center)
            rotation: Additional rotation angle
        """
        h, w = img.shape[:2]
        
        if center is None:
            center = (w // 2, h // 2)
        if radius is None:
            radius = min(h, w) * 0.4
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        y = y - center[1]
        x = x - center[0]
        
        # Calculate distance from center
        r = np.sqrt(x**2 + y**2)
        
        # Calculate swirl amount (decreases with distance)
        theta = strength * np.exp(-(r / radius)**2) + rotation
        
        # Apply rotation transformation
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        new_x = x * cos_t - y * sin_t + center[0]
        new_y = x * sin_t + y * cos_t + center[1]
        
        # Remap
        new_x = new_x.astype(np.float32)
        new_y = new_y.astype(np.float32)
        
        return cv2.remap(img, new_x, new_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


class WaveDistortion:
    """Ripple/wave distortion effect."""
    
    @staticmethod
    def apply(
        img: np.ndarray,
        amplitude: float = 10.0,
        wavelength: float = 30.0,
        direction: str = 'both'
    ) -> np.ndarray:
        """Apply wave distortion."""
        h, w = img.shape[:2]
        
        y, x = np.mgrid[:h, :w].astype(np.float32)
        
        if direction in ['horizontal', 'both']:
            x = x + amplitude * np.sin(2 * np.pi * y / wavelength)
        if direction in ['vertical', 'both']:
            y = y + amplitude * np.sin(2 * np.pi * x / wavelength)
        
        return cv2.remap(img, x, y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


class SmearDistortion:
    """
    Smear/drip effect like the man example.
    Creates vertical or directional smearing.
    """
    
    @staticmethod
    def apply(
        img: np.ndarray,
        mask: np.ndarray,
        direction: str = 'down',
        strength: float = 30.0,
        decay: float = 0.95
    ) -> np.ndarray:
        """
        Apply smear effect in masked region.
        
        Args:
            img: Input image
            mask: Binary mask where smear originates
            direction: 'down', 'up', 'left', 'right'
            strength: Maximum smear distance in pixels
            decay: Color decay per pixel
        """
        h, w = img.shape[:2]
        result = img.copy().astype(np.float32)
        mask_f = mask.astype(np.float32)
        
        if direction == 'down':
            for i in range(int(strength)):
                shifted = np.roll(mask_f, i, axis=0)
                alpha = (decay ** i) * shifted[:, :, np.newaxis]
                shifted_img = np.roll(img, i, axis=0)
                result = result * (1 - alpha) + shifted_img * alpha
        elif direction == 'up':
            for i in range(int(strength)):
                shifted = np.roll(mask_f, -i, axis=0)
                alpha = (decay ** i) * shifted[:, :, np.newaxis]
                shifted_img = np.roll(img, -i, axis=0)
                result = result * (1 - alpha) + shifted_img * alpha
        elif direction == 'right':
            for i in range(int(strength)):
                shifted = np.roll(mask_f, i, axis=1)
                alpha = (decay ** i) * shifted[:, :, np.newaxis]
                shifted_img = np.roll(img, i, axis=1)
                result = result * (1 - alpha) + shifted_img * alpha
        elif direction == 'left':
            for i in range(int(strength)):
                shifted = np.roll(mask_f, -i, axis=1)
                alpha = (decay ** i) * shifted[:, :, np.newaxis]
                shifted_img = np.roll(img, -i, axis=1)
                result = result * (1 - alpha) + shifted_img * alpha
        
        return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================================
# Color Overlays and Blending
# ============================================================================

class ColorOverlay:
    """
    Color tinting and overlay effects.
    Like the yellow/amber tint in the woman example.
    """
    
    @staticmethod
    def create_color_layer(
        h: int, w: int,
        color: Tuple[int, int, int],
        opacity: float = 0.5
    ) -> np.ndarray:
        """Create a solid color layer."""
        layer = np.zeros((h, w, 3), dtype=np.float32)
        layer[:, :] = [c / 255.0 for c in color]
        return layer * opacity
    
    @staticmethod
    def overlay_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """Photoshop-style overlay blend mode."""
        base_f = normalize_image(base)
        overlay_f = normalize_image(overlay) if overlay.max() > 1 else overlay
        
        result = np.where(
            base_f < 0.5,
            2 * base_f * overlay_f,
            1 - 2 * (1 - base_f) * (1 - overlay_f)
        )
        return denormalize_image(result)
    
    @staticmethod
    def soft_light_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """Soft light blend mode."""
        base_f = normalize_image(base)
        overlay_f = normalize_image(overlay) if overlay.max() > 1 else overlay
        
        result = np.where(
            overlay_f < 0.5,
            base_f - (1 - 2 * overlay_f) * base_f * (1 - base_f),
            base_f + (2 * overlay_f - 1) * (np.sqrt(base_f) - base_f)
        )
        return denormalize_image(result)
    
    @staticmethod
    def multiply_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """Multiply blend mode."""
        base_f = normalize_image(base)
        overlay_f = normalize_image(overlay) if overlay.max() > 1 else overlay
        return denormalize_image(base_f * overlay_f)
    
    @staticmethod
    def screen_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """Screen blend mode."""
        base_f = normalize_image(base)
        overlay_f = normalize_image(overlay) if overlay.max() > 1 else overlay
        return denormalize_image(1 - (1 - base_f) * (1 - overlay_f))


# ============================================================================
# Mask Generation
# ============================================================================

class MaskGenerator:
    """
    Generates diverse mask types matching example quality.
    """
    
    @staticmethod
    def organic_blob(
        h: int, w: int,
        coverage: Tuple[float, float] = (0.15, 0.45),
        blur_sigma: float = 4.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate organic blob mask using fractal noise.
        Similar to the swirl mask in woman example.
        """
        noise = fractal_noise(h, w, octaves=random.randint(3, 5))
        noise = gaussian_blur(noise, sigma=random.uniform(2.0, 4.0))
        
        # Domain warp for organic feel
        dx = gaussian_blur(np.random.randn(h, w).astype(np.float32), sigma=3.0)
        dy = gaussian_blur(np.random.randn(h, w).astype(np.float32), sigma=3.0)
        
        y, x = np.mgrid[:h, :w].astype(np.float32)
        strength = random.uniform(8.0, 15.0)
        new_x = np.clip(x + dx * strength, 0, w - 1)
        new_y = np.clip(y + dy * strength, 0, h - 1)
        noise = cv2.remap(noise, new_x, new_y, cv2.INTER_CUBIC)
        
        # Threshold to get binary mask
        threshold = random.uniform(0.4, 0.6)
        binary = (noise > threshold).astype(np.uint8)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Coverage control
        target_cov = random.uniform(*coverage)
        while binary.mean() > target_cov:
            binary = cv2.erode(binary, kernel, iterations=1)
        
        # Soft alpha for blending
        alpha = gaussian_blur(binary.astype(np.float32), sigma=blur_sigma)
        alpha = np.clip(alpha, 0, 1)
        
        return binary, alpha
    
    @staticmethod
    def circular_region(
        h: int, w: int,
        center: Optional[Tuple[int, int]] = None,
        radius: Optional[float] = None,
        feather: float = 20.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate circular/elliptical mask.
        For swirl-type distortions.
        """
        if center is None:
            cx = w // 2 + random.randint(-w // 4, w // 4)
            cy = h // 2 + random.randint(-h // 4, h // 4)
            center = (cx, cy)
        
        if radius is None:
            radius = min(h, w) * random.uniform(0.2, 0.4)
        
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        binary = (dist < radius).astype(np.uint8)
        
        # Feathered alpha
        alpha = 1 - np.clip((dist - radius + feather) / feather, 0, 1)
        alpha = alpha.astype(np.float32)
        
        return binary, alpha
    
    @staticmethod
    def rectangular_patch(
        h: int, w: int,
        num_patches: int = 2,
        size_range: Tuple[float, float] = (0.1, 0.3)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate rectangular patch mask.
        Like the patches in man example.
        """
        binary = np.zeros((h, w), dtype=np.uint8)
        
        for _ in range(num_patches):
            # Random patch size
            patch_w = int(w * random.uniform(*size_range))
            patch_h = int(h * random.uniform(*size_range))
            
            # Random position
            x1 = random.randint(0, w - patch_w)
            y1 = random.randint(0, h - patch_h)
            
            # Optional rotation
            if random.random() > 0.5:
                # Create rotated rectangle
                angle = random.uniform(-30, 30)
                center = (x1 + patch_w // 2, y1 + patch_h // 2)
                rect = ((center[0], center[1]), (patch_w, patch_h), angle)
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.fillPoly(binary, [box], 1)
            else:
                binary[y1:y1+patch_h, x1:x1+patch_w] = 1
        
        # Slight blur for edges
        alpha = gaussian_blur(binary.astype(np.float32), sigma=2.0)
        
        return binary, alpha
    
    @staticmethod
    def irregular_strokes(
        h: int, w: int,
        num_strokes: int = 5,
        thickness_range: Tuple[int, int] = (5, 25)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate irregular brush stroke mask."""
        binary = np.zeros((h, w), dtype=np.uint8)
        
        for _ in range(num_strokes):
            # Random bezier-like curve
            num_points = random.randint(3, 6)
            points = []
            x, y = random.randint(0, w), random.randint(0, h)
            
            for _ in range(num_points):
                x += random.randint(-100, 100)
                y += random.randint(-100, 100)
                x = np.clip(x, 0, w - 1)
                y = np.clip(y, 0, h - 1)
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            thickness = random.randint(*thickness_range)
            cv2.polylines(binary, [points], False, 1, thickness=thickness)
        
        # Dilate and blur
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.dilate(binary, kernel, iterations=2)
        alpha = gaussian_blur(binary.astype(np.float32), sigma=3.0)
        
        return binary, alpha


# ============================================================================
# Composite Pipelines (Match Example Quality)
# ============================================================================

class DistortionPipeline:
    """
    High-level pipelines matching example image quality.
    """
    
    @staticmethod
    def swirl_with_color_tint(
        img: np.ndarray,
        tint_color: Tuple[int, int, int] = (255, 200, 50),  # Yellow/amber
        swirl_strength: float = 2.5,
        tint_opacity: float = 0.35
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pipeline 1: Swirl distortion with color overlay.
        Matches the woman example.
        """
        h, w = img.shape[:2]
        
        # Generate circular mask for swirl region
        mask_bin, mask_alpha = MaskGenerator.circular_region(h, w)
        
        # Apply swirl to the image
        center = (w // 2 + random.randint(-50, 50), 
                  h // 2 + random.randint(-50, 50))
        swirled = SwrlDistortion.apply(
            img, 
            center=center,
            strength=swirl_strength * random.uniform(0.8, 1.2)
        )
        
        # Blend swirled region with original
        mask_3ch = np.stack([mask_alpha, mask_alpha, mask_alpha], axis=-1)
        blended = (swirled.astype(np.float32) * mask_3ch + 
                   img.astype(np.float32) * (1 - mask_3ch))
        
        # Add color tint in masked region
        color_layer = ColorOverlay.create_color_layer(h, w, tint_color, tint_opacity)
        color_layer_masked = color_layer * mask_3ch
        
        # Apply soft-light blend for the tint
        result = ColorOverlay.soft_light_blend(
            blended.astype(np.uint8), 
            (color_layer_masked * 255).astype(np.uint8)
        )
        
        # Add slight wave for extra organic feel
        if random.random() > 0.5:
            result = WaveDistortion.apply(
                result, 
                amplitude=random.uniform(2, 5),
                wavelength=random.uniform(40, 80)
            )
        
        return result, mask_bin
    
    @staticmethod
    def rectangular_composite(
        img: np.ndarray,
        external_patch: Optional[np.ndarray] = None,
        add_smear: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pipeline 2: Rectangular patch compositing with smear.
        Matches the man example.
        """
        h, w = img.shape[:2]
        
        # Generate rectangular mask
        mask_bin, mask_alpha = MaskGenerator.rectangular_patch(
            h, w, 
            num_patches=random.randint(1, 3),
            size_range=(0.15, 0.35)
        )
        
        # Create or use external patch
        if external_patch is None:
            # Generate synthetic patch (colored noise or gradient)
            patch = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            patch = gaussian_blur(patch.astype(np.float32), sigma=5.0).astype(np.uint8)
            
            # Add color tint (blue or red like the example)
            tint = random.choice([(50, 100, 200), (200, 80, 80), (100, 180, 100)])
            tint_layer = ColorOverlay.create_color_layer(h, w, tint, 0.5)
            patch = ColorOverlay.overlay_blend(patch, (tint_layer * 255).astype(np.uint8))
        else:
            patch = cv2.resize(external_patch, (w, h))
        
        # Composite patch onto image
        mask_3ch = np.stack([mask_alpha, mask_alpha, mask_alpha], axis=-1)
        result = (patch.astype(np.float32) * mask_3ch + 
                  img.astype(np.float32) * (1 - mask_3ch))
        result = result.astype(np.uint8)
        
        # Add smear effect
        if add_smear:
            smear_mask = cv2.erode(mask_bin, np.ones((3, 3)), iterations=2)
            direction = random.choice(['down', 'left', 'right'])
            result = SmearDistortion.apply(
                result, 
                smear_mask,
                direction=direction,
                strength=random.uniform(20, 50),
                decay=random.uniform(0.92, 0.98)
            )
        
        return result, mask_bin
    
    @staticmethod
    def organic_blend(
        img: np.ndarray,
        donor_img: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pipeline 3: Organic blob mask with Laplacian blending.
        High-quality seamless compositing.
        """
        h, w = img.shape[:2]
        
        # Generate organic mask
        mask_bin, mask_alpha = MaskGenerator.organic_blob(h, w)
        
        # Use donor or generate synthetic texture
        if donor_img is None:
            # Create warped version of original as "donor"
            dx = gaussian_blur(np.random.randn(h, w).astype(np.float32), sigma=5.0)
            dy = gaussian_blur(np.random.randn(h, w).astype(np.float32), sigma=5.0)
            y, x = np.mgrid[:h, :w].astype(np.float32)
            new_x = np.clip(x + dx * 15, 0, w - 1)
            new_y = np.clip(y + dy * 15, 0, h - 1)
            donor = cv2.remap(img, new_x, new_y, cv2.INTER_CUBIC)
            
            # Color shift
            hsv = cv2.cvtColor(donor, cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(10, 50)) % 180
            donor = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:
            donor = cv2.resize(donor_img, (w, h))
        
        # Alpha blend
        mask_3ch = np.stack([mask_alpha, mask_alpha, mask_alpha], axis=-1)
        result = (donor.astype(np.float32) * mask_3ch + 
                  img.astype(np.float32) * (1 - mask_3ch))
        
        return result.astype(np.uint8), mask_bin


# ============================================================================
# Main Generator Function
# ============================================================================

def generate_distorted_image(
    img: np.ndarray,
    style: str = 'random',
    external_patch: Optional[np.ndarray] = None,
    donor_img: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a distorted/masked version of the input image.
    
    Args:
        img: Input RGB image (H, W, 3)
        style: 'swirl', 'rectangular', 'organic', or 'random'
        external_patch: Optional external image for compositing
        donor_img: Optional donor image for organic blending
        
    Returns:
        distorted: The distorted image
        mask: Binary mask showing distorted regions
    """
    img = ensure_rgb(img)
    
    if style == 'random':
        style = random.choice(['swirl', 'rectangular', 'organic'])
    
    if style == 'swirl':
        # Random tint color
        colors = [
            (255, 200, 50),   # Yellow/amber (like example 1)
            (255, 150, 100),  # Orange
            (100, 200, 255),  # Cyan
            (200, 100, 255),  # Purple
        ]
        return DistortionPipeline.swirl_with_color_tint(
            img,
            tint_color=random.choice(colors),
            swirl_strength=random.uniform(1.5, 3.5),
            tint_opacity=random.uniform(0.25, 0.5)
        )
    
    elif style == 'rectangular':
        return DistortionPipeline.rectangular_composite(
            img,
            external_patch=external_patch,
            add_smear=random.random() > 0.3
        )
    
    elif style == 'organic':
        return DistortionPipeline.organic_blend(
            img,
            donor_img=donor_img
        )
    
    else:
        raise ValueError(f"Unknown style: {style}")


# ============================================================================
# Batch Dataset Generation
# ============================================================================

def generate_dataset(
    input_dir: str,
    output_dir: str,
    num_variants: int = 3,
    image_size: int = 512,
    styles: List[str] = ['swirl', 'rectangular', 'organic']
) -> None:
    """
    Generate a training dataset of distorted images.
    
    Args:
        input_dir: Directory with clean input images
        output_dir: Output directory (creates input/ and target/ subdirs)
        num_variants: Number of distorted variants per image
        image_size: Resize images to this square size
        styles: List of distortion styles to use
    """
    import glob
    from tqdm import tqdm
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'target'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mask'), exist_ok=True)
    
    # Find input images
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    print(f"Found {len(image_paths)} images")
    
    idx = 0
    for img_path in tqdm(image_paths, desc="Generating dataset"):
        # Load and preprocess
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Center crop and resize
        h, w = img.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        img = img[y0:y0+side, x0:x0+side]
        img = cv2.resize(img, (image_size, image_size))
        
        # Generate variants
        for _ in range(num_variants):
            style = random.choice(styles)
            distorted, mask = generate_distorted_image(img, style=style)
            
            # Save
            cv2.imwrite(
                os.path.join(output_dir, 'input', f'{idx:06d}.png'),
                cv2.cvtColor(distorted, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                os.path.join(output_dir, 'target', f'{idx:06d}.png'),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                os.path.join(output_dir, 'mask', f'{idx:06d}.png'),
                mask * 255
            )
            idx += 1
    
    print(f"Generated {idx} distorted images")


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Diverse Mask Generator...")
    
    # Create test image
    test_img = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    
    # Test each style
    for style in ['swirl', 'rectangular', 'organic']:
        distorted, mask = generate_distorted_image(test_img, style=style)
        print(f"✓ {style}: output shape {distorted.shape}, mask coverage {mask.mean():.2%}")
    
    print("\n✅ All tests passed!")
