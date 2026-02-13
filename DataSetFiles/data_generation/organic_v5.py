"""
Organic Mask Generator v5 - Truly Organic Shapes
=================================================

Uses FRACTAL NOISE and DOMAIN WARPING to create
irregular, organic-looking masks - NOT geometric boxes.

Mask Types:
1. Fractal Blobs - Noise-based organic regions
2. Flowy Curves - Smooth curved regions
3. Organic Patches - Warped, irregular patches
4. Cloudy Regions - Cloud-like soft masks
5. Natural Edges - Masks with natural, irregular boundaries

Dependencies: pip install opencv-python numpy tqdm
"""

import cv2
import numpy as np
import os
import random
import math
from typing import Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable


class OrganicMaskGeneratorV5:
    """
    Creates truly ORGANIC, irregular masks using:
    - Fractal Brownian Motion (fBm) noise
    - Domain warping for distortion
    - Perlin-like noise patterns
    - NO geometric shapes like rectangles
    """
    
    def __init__(self, height: int = 512, width: int = 512, seed: int = None):
        self.h = height
        self.w = width
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    # =========================================
    # NOISE GENERATION
    # =========================================
    
    def fractal_noise(self, octaves: int = 5, persistence: float = 0.5) -> np.ndarray:
        """Generate fractal Brownian motion noise."""
        total = np.zeros((self.h, self.w), dtype=np.float32)
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            # Generate noise at current frequency
            noise_h = max(1, int(self.h / frequency))
            noise_w = max(1, int(self.w / frequency))
            noise = np.random.rand(noise_h, noise_w).astype(np.float32)
            
            # Upsample to full resolution
            noise = cv2.resize(noise, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
            
            total += noise * amplitude
            max_value += amplitude
            
            frequency *= 2.0
            amplitude *= persistence
        
        # Normalize to 0-1
        total = total / max_value
        return total
    
    def domain_warp(self, data: np.ndarray, strength: float = 30.0) -> np.ndarray:
        """Apply domain warping for organic distortion."""
        # Generate warp fields
        warp_x = self.fractal_noise(octaves=3, persistence=0.6)
        warp_y = self.fractal_noise(octaves=3, persistence=0.6)
        
        # Center around zero
        warp_x = (warp_x - 0.5) * 2 * strength
        warp_y = (warp_y - 0.5) * 2 * strength
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        
        # Apply warp
        map_x = np.clip(x_coords + warp_x, 0, self.w - 1)
        map_y = np.clip(y_coords + warp_y, 0, self.h - 1)
        
        return cv2.remap(data.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
    
    def ensure_coverage(self, mask: np.ndarray, min_cover: float = 0.15, max_cover: float = 0.50) -> np.ndarray:
        """
        Adjust mask to ensure coverage is within target range.
        Optimized version - uses percentile-based thresholding.
        """
        total_pixels = self.h * self.w
        target_coverage = random.uniform(min_cover, max_cover)
        
        # If mask is empty, create a basic blob
        mask_float = mask.astype(np.float32)
        current_coverage = np.sum(mask_float > 0) / total_pixels
        
        if current_coverage < 0.01:
            # Create fallback blob
            cx, cy = self.w // 2, self.h // 2
            y, x = np.ogrid[:self.h, :self.w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            radius = min(self.h, self.w) * 0.3
            mask_float = (dist < radius).astype(np.float32) * 255
        
        # Use percentile-based adjustment (FAST)
        non_zero = mask_float[mask_float > 0]
        
        if len(non_zero) > 0:
            # Calculate what threshold gives target coverage
            target_count = int(total_pixels * target_coverage)
            
            if np.sum(mask_float > 0) > target_count:
                # Too much coverage - increase threshold
                percentile = 100 * (1 - target_coverage / (np.sum(mask_float > 0) / total_pixels))
                threshold = np.percentile(mask_float[mask_float > 0], percentile)
                mask_float = (mask_float > threshold).astype(np.float32) * 255
            else:
                # Not enough - dilate (max 3 iterations)
                for _ in range(3):
                    if np.sum(mask_float > 0) >= target_count * 0.8:
                        break
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
                    mask_float = cv2.dilate(mask_float, kernel, iterations=1)
        
        return mask_float.astype(np.uint8)
    
    # =========================================
    # ORGANIC MASK STYLES
    # =========================================
    
    def generate_fractal_blob(self) -> np.ndarray:
        """
        Organic blob using fractal noise thresholding.
        Creates cloud-like, irregular shapes.
        """
        # Generate base noise
        noise = self.fractal_noise(
            octaves=random.randint(4, 6),
            persistence=random.uniform(0.45, 0.65)
        )
        
        # Apply domain warping for more organic feel
        noise = self.domain_warp(noise, strength=random.uniform(20, 50))
        
        # Smooth
        blur_size = random.choice([7, 11, 15, 21])
        noise = cv2.GaussianBlur(noise, (blur_size, blur_size), 0)
        
        # Threshold to create mask
        # Random threshold for varied coverage
        threshold = random.uniform(0.4, 0.65)
        mask = (noise > threshold).astype(np.uint8) * 255
        
        # Morphological operations for cleaner edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def generate_flowing_region(self) -> np.ndarray:
        """
        Smooth, flowing organic regions.
        Like water or smoke patterns.
        """
        # Create gradient-based flow
        noise1 = self.fractal_noise(octaves=4, persistence=0.5)
        noise2 = self.fractal_noise(octaves=4, persistence=0.5)
        
        # Combine noises for flow effect
        combined = (noise1 * 0.6 + noise2 * 0.4)
        
        # Multiple warps for organic flow
        for _ in range(2):
            combined = self.domain_warp(combined, strength=random.uniform(15, 35))
        
        # Smooth heavily for flow look
        blur_size = random.choice([15, 21, 31])
        combined = cv2.GaussianBlur(combined, (blur_size, blur_size), 0)
        
        # Select band of values for mask
        low = random.uniform(0.3, 0.5)
        high = low + random.uniform(0.15, 0.35)
        
        mask = ((combined > low) & (combined < high)).astype(np.uint8) * 255
        
        # Close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def generate_organic_patch(self) -> np.ndarray:
        """
        Multiple organic patches with natural edges.
        """
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
        num_patches = random.randint(2, 5)
        
        for _ in range(num_patches):
            # Create patch using noise
            patch = self.fractal_noise(
                octaves=random.randint(3, 5),
                persistence=random.uniform(0.5, 0.7)
            )
            
            # Random center bias
            cx = random.gauss(0.5, 0.25)
            cy = random.gauss(0.5, 0.25)
            cx = max(0.1, min(0.9, cx))
            cy = max(0.1, min(0.9, cy))
            
            # Create radial falloff
            y, x = np.ogrid[:self.h, :self.w]
            center_x = int(self.w * cx)
            center_y = int(self.h * cy)
            
            # Distance from center
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            radius = random.uniform(0.2, 0.4) * min(self.h, self.w)
            
            # Combine noise with radial mask
            radial = 1 - np.clip(dist / radius, 0, 1)
            radial = radial ** random.uniform(0.5, 1.5)  # Adjust falloff
            
            combined = patch * radial
            
            # Warp for organic edges
            combined = self.domain_warp(combined, strength=random.uniform(10, 25))
            
            # Threshold
            thresh = random.uniform(0.3, 0.5)
            patch_mask = (combined > thresh).astype(np.uint8) * 255
            
            mask = cv2.bitwise_or(mask, patch_mask)
        
        return mask
    
    def generate_cloudy_mask(self) -> np.ndarray:
        """
        Cloud-like soft masks with varying density.
        """
        # Multi-octave noise
        cloud = self.fractal_noise(
            octaves=random.randint(5, 7),
            persistence=random.uniform(0.55, 0.7)
        )
        
        # Heavy domain warping
        cloud = self.domain_warp(cloud, strength=random.uniform(30, 60))
        cloud = self.domain_warp(cloud, strength=random.uniform(15, 30))
        
        # Very smooth
        blur_size = random.choice([21, 31, 41])
        cloud = cv2.GaussianBlur(cloud, (blur_size, blur_size), 0)
        
        # Threshold
        thresh = random.uniform(0.45, 0.6)
        mask = (cloud > thresh).astype(np.uint8) * 255
        
        return mask
    
    def generate_natural_edge_mask(self) -> np.ndarray:
        """
        Mask with very natural, torn-paper-like edges.
        """
        # Start with multiple blob seeds
        mask = np.zeros((self.h, self.w), dtype=np.float32)
        
        num_seeds = random.randint(3, 8)
        
        for _ in range(num_seeds):
            # Random position
            cx = random.randint(self.w // 4, 3 * self.w // 4)
            cy = random.randint(self.h // 4, 3 * self.h // 4)
            
            # Create gradient blob
            y, x = np.ogrid[:self.h, :self.w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            blob_radius = random.randint(50, 150)
            blob = np.exp(-(dist**2) / (2 * blob_radius**2))
            
            mask = np.maximum(mask, blob)
        
        # Add noise to edges
        edge_noise = self.fractal_noise(octaves=4, persistence=0.6)
        mask = mask + edge_noise * 0.3
        
        # Domain warp for irregular edges
        mask = self.domain_warp(mask, strength=random.uniform(25, 45))
        
        # Smooth
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        
        # Threshold
        thresh = random.uniform(0.35, 0.55)
        result = (mask > thresh).astype(np.uint8) * 255
        
        return result
    
    def generate_composite_organic(self) -> np.ndarray:
        """
        Combination of organic styles.
        """
        styles = [
            self.generate_fractal_blob,
            self.generate_flowing_region,
            self.generate_organic_patch,
            self.generate_cloudy_mask,
            self.generate_natural_edge_mask
        ]
        
        # Pick 2 random styles
        selected = random.sample(styles, 2)
        
        combined = np.zeros((self.h, self.w), dtype=np.uint8)
        for func in selected:
            combined = cv2.bitwise_or(combined, func())
        
        return combined
    
    # =========================================
    # EDGE EFFECTS
    # =========================================
    
    def add_organic_edge(self, mask: np.ndarray) -> np.ndarray:
        """Add natural, organic edge distortion."""
        # Find edges
        edges = cv2.Canny(mask, 50, 150)
        
        # Dilate edges
        kernel = np.ones((random.randint(3, 7), random.randint(3, 7)), np.uint8)
        border = cv2.dilate(edges, kernel, iterations=1)
        
        # Warp the mask slightly
        warp_strength = random.uniform(3, 8)
        noise_x = self.fractal_noise(octaves=3, persistence=0.6)
        noise_y = self.fractal_noise(octaves=3, persistence=0.6)
        
        noise_x = (noise_x - 0.5) * 2 * warp_strength
        noise_y = (noise_y - 0.5) * 2 * warp_strength
        
        # Apply only where border exists
        border_mask = (border > 0).astype(np.float32)
        noise_x = noise_x * border_mask
        noise_y = noise_y * border_mask
        
        y_coords, x_coords = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        
        map_x = np.clip(x_coords + noise_x, 0, self.w - 1)
        map_y = np.clip(y_coords + noise_y, 0, self.h - 1)
        
        warped = cv2.remap(mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
        
        # Blur for soft edges
        blur = random.choice([5, 7, 9])
        warped = cv2.GaussianBlur(warped, (blur, blur), 0)
        
        return warped.astype(np.uint8)
    
    # =========================================
    # MAIN FUNCTION
    # =========================================
    
    def apply_corruption(
        self, 
        base_image: np.ndarray,
        overlay_image: np.ndarray,
        style: str = 'random'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply organic corruption."""
        self.h, self.w = base_image.shape[:2]
        
        styles = {
            'fractal': self.generate_fractal_blob,
            'flow': self.generate_flowing_region,
            'patch': self.generate_organic_patch,
            'cloud': self.generate_cloudy_mask,
            'natural': self.generate_natural_edge_mask,
            'composite': self.generate_composite_organic
        }
        
        if style == 'random':
            style = random.choice(list(styles.keys()))
        
        mask = styles[style]()
        
        # ENSURE COVERAGE IS 15-50% (not too small, not too large)
        mask = self.ensure_coverage(mask, min_cover=0.15, max_cover=0.50)
        
        # Resize overlay
        overlay = cv2.resize(overlay_image, (self.w, self.h))
        
        # Random transforms on overlay
        if random.random() > 0.5:
            overlay = cv2.flip(overlay, 1)
        if random.random() > 0.5:
            overlay = cv2.flip(overlay, 0)
        if random.random() > 0.4:
            angle = random.randint(-180, 180)
            center = (self.w // 2, self.h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            overlay = cv2.warpAffine(overlay, M, (self.w, self.h))
        
        # Add organic edge effect
        mask = self.add_organic_edge(mask)
        
        # Soft edge blur
        blur_size = random.choice([7, 11, 15])
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        
        # Composite
        alpha = mask.astype(np.float32) / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=-1)
        
        opacity = random.uniform(0.65, 1.0)
        alpha = alpha * opacity
        
        result = (overlay.astype(np.float32) * alpha + 
                  base_image.astype(np.float32) * (1 - alpha))
        
        return result.astype(np.uint8), mask


# =========================================
# WORKER FOR MULTIPROCESSING
# =========================================

def process_image(args):
    """Worker function."""
    idx, base_path, overlay_paths, output_dir, save_mask, seed = args
    
    random.seed(seed + idx)
    np.random.seed(seed + idx)
    
    try:
        base_img = cv2.imread(base_path)
        if base_img is None:
            return None
        
        overlay_path = random.choice(overlay_paths)
        overlay_img = cv2.imread(overlay_path)
        if overlay_img is None:
            overlay_img = cv2.flip(base_img, 1)
        
        generator = OrganicMaskGeneratorV5()
        corrupted, mask = generator.apply_corruption(base_img, overlay_img)
        
        out_name = f"{idx:06d}.png"
        cv2.imwrite(os.path.join(output_dir, 'input', out_name), corrupted)
        cv2.imwrite(os.path.join(output_dir, 'target', out_name), base_img)
        
        if save_mask:
            cv2.imwrite(os.path.join(output_dir, 'mask', out_name), mask)
        
        return idx
    except:
        return None


def generate_dataset(
    clean_folder: str,
    overlay_folder: str,
    output_folder: str,
    total_images: int = 10000,
    num_workers: int = None,
    save_masks: bool = True,
    seed: int = 42
) -> None:
    """Generate dataset with organic masks."""
    os.makedirs(os.path.join(output_folder, 'input'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'target'), exist_ok=True)
    if save_masks:
        os.makedirs(os.path.join(output_folder, 'mask'), exist_ok=True)
    
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    clean_files = [os.path.join(clean_folder, f) 
                   for f in os.listdir(clean_folder) 
                   if f.lower().endswith(valid_ext)]
    
    overlay_files = [os.path.join(overlay_folder, f)
                     for f in os.listdir(overlay_folder)
                     if f.lower().endswith(valid_ext)]
    
    print("=" * 60)
    print("ORGANIC MASK GENERATOR v5 - Truly Organic Shapes")
    print("=" * 60)
    print(f"Clean images:    {len(clean_files):,}")
    print(f"Overlay images:  {len(overlay_files):,}")
    print(f"Target output:   {total_images:,} images")
    print("=" * 60)
    print("Styles: fractal, flow, patch, cloud, natural, composite")
    print("All masks use fractal noise + domain warping (NO boxes!)")
    print("=" * 60)
    
    if len(clean_files) == 0:
        raise ValueError(f"No images in {clean_folder}")
    if len(overlay_files) == 0:
        overlay_files = clean_files
    
    tasks = [(i, clean_files[i % len(clean_files)], overlay_files, 
              output_folder, save_masks, seed) for i in range(total_images)]
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Workers: {num_workers}")
    print()
    
    completed = 0
    
    if num_workers == 1:
        for task in tqdm(tasks, desc="Generating", unit="img"):
            if process_image(task) is not None:
                completed += 1
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_image, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc="Generating", unit="img"):
                if future.result() is not None:
                    completed += 1
    
    print()
    print("=" * 60)
    print(f"COMPLETE! Generated {completed:,}/{total_images:,} images")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Organic Mask Generator v5')
    parser.add_argument('--clean', '-c', required=True, help='Clean images')
    parser.add_argument('--overlays', '-l', required=True, help='Overlay images')
    parser.add_argument('--output', '-o', required=True, help='Output folder')
    parser.add_argument('--num-images', '-n', type=int, default=10000)
    parser.add_argument('--workers', '-w', type=int, default=None)
    parser.add_argument('--no-masks', action='store_true')
    parser.add_argument('--seed', '-s', type=int, default=42)
    
    args = parser.parse_args()
    
    generate_dataset(
        clean_folder=args.clean,
        overlay_folder=args.overlays,
        output_folder=args.output,
        total_images=args.num_images,
        num_workers=args.workers,
        save_masks=not args.no_masks,
        seed=args.seed
    )
