"""
High-Quality Mask Generator v4 - Production Scale (100K+)
==========================================================

Improvements over v3:
- REMOVED thin line masks (too simple)
- Added high-quality mask styles matching reference datasets
- Better overlay blending with realistic corruptions
- Optimized for 100K+ images

Mask Styles:
1. Large Organic Blobs - Irregular patches covering significant areas
2. Brush Strokes - Thick, overlapping brush strokes
3. Rectangular Patches - Multiple overlapping rectangles
4. Splatter/Spray - Paint splatter effect
5. Scratch/Damage - Simulated physical damage
6. Mixed Composite - Combination of above

Usage:
    python high_quality_mask_generator.py -c ./clean -l ./overlays -o ./output -n 100000

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


class HighQualityMaskGenerator:
    """
    High-quality mask generator matching reference dataset quality.
    Focuses on realistic, substantial corruptions - NOT simple lines.
    """
    
    def __init__(self, height: int = 512, width: int = 512, seed: int = None):
        self.h = height
        self.w = width
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    # =========================================
    # HIGH-QUALITY MASK STYLES
    # =========================================
    
    def generate_large_organic_blob(self) -> np.ndarray:
        """
        Large organic blob covering 20-50% of image.
        Matches the chunky, irregular patches in reference datasets.
        """
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
        # Multiple overlapping blobs
        num_blobs = random.randint(2, 5)
        
        for _ in range(num_blobs):
            # Random center, biased towards center of image
            cx = int(self.w * random.gauss(0.5, 0.25))
            cy = int(self.h * random.gauss(0.5, 0.25))
            cx = max(0, min(self.w - 1, cx))
            cy = max(0, min(self.h - 1, cy))
            
            # Large irregular polygon
            num_vertices = random.randint(8, 20)
            base_radius = random.randint(int(min(self.h, self.w) * 0.15), 
                                         int(min(self.h, self.w) * 0.35))
            
            angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_vertices)])
            
            pts = []
            for angle in angles:
                # Vary radius for irregular shape
                r = base_radius * random.uniform(0.6, 1.4)
                x = int(cx + r * math.cos(angle))
                y = int(cy + r * math.sin(angle))
                pts.append([x, y])
            
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        
        # Morphological operations for organic edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def generate_thick_brush_strokes(self) -> np.ndarray:
        """
        Thick, overlapping brush strokes.
        Much thicker than thin lines - substantial coverage.
        """
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
        num_strokes = random.randint(3, 8)
        
        for _ in range(num_strokes):
            # Start point
            x = random.randint(0, self.w)
            y = random.randint(0, self.h)
            
            # THICK brush width (30-80 pixels)
            brush_width = random.randint(30, 80)
            
            # Draw curved stroke
            num_points = random.randint(5, 15)
            pts = [(x, y)]
            
            for _ in range(num_points):
                # Smooth curve movement
                angle = random.uniform(-math.pi/3, math.pi/3)  # Limit angle change
                if len(pts) > 1:
                    # Continue in similar direction
                    prev_angle = math.atan2(pts[-1][1] - pts[-2][1], 
                                           pts[-1][0] - pts[-2][0])
                    angle = prev_angle + random.uniform(-math.pi/4, math.pi/4)
                
                length = random.randint(30, 100)
                x = int(pts[-1][0] + length * math.cos(angle))
                y = int(pts[-1][1] + length * math.sin(angle))
                pts.append((x, y))
            
            # Draw thick polyline
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(mask, [pts_arr], False, 255, thickness=brush_width)
            
            # Add circles at joints for smooth connections
            for p in pts:
                cv2.circle(mask, p, brush_width // 2, 255, -1)
        
        return mask
    
    def generate_rectangular_patches(self) -> np.ndarray:
        """
        Multiple overlapping rectangles with rotation.
        Like the rectangular patch in the man example.
        """
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
        num_rects = random.randint(2, 6)
        
        for _ in range(num_rects):
            # Random rectangle size (substantial, not tiny)
            rect_w = random.randint(int(self.w * 0.15), int(self.w * 0.45))
            rect_h = random.randint(int(self.h * 0.15), int(self.h * 0.45))
            
            # Random position
            cx = random.randint(rect_w // 2, self.w - rect_w // 2)
            cy = random.randint(rect_h // 2, self.h - rect_h // 2)
            
            # Random rotation
            angle = random.randint(-45, 45)
            
            # Create rotated rectangle
            rect = ((cx, cy), (rect_w, rect_h), angle)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(mask, [box], 255)
        
        return mask
    
    def generate_splatter(self) -> np.ndarray:
        """
        Paint splatter effect - clusters of circles.
        Creates realistic spray/splatter corruption.
        """
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
        # Multiple splatter centers
        num_centers = random.randint(2, 5)
        
        for _ in range(num_centers):
            # Splatter center
            cx = random.randint(0, self.w)
            cy = random.randint(0, self.h)
            
            # Main blob
            main_radius = random.randint(40, 100)
            cv2.circle(mask, (cx, cy), main_radius, 255, -1)
            
            # Surrounding splatter dots
            num_dots = random.randint(30, 80)
            for _ in range(num_dots):
                # Distance from center (exponential decay)
                dist = abs(np.random.exponential(main_radius * 1.5))
                angle = random.uniform(0, 2 * math.pi)
                
                dx = int(cx + dist * math.cos(angle))
                dy = int(cy + dist * math.sin(angle))
                
                # Smaller radius for dots further away
                dot_radius = max(3, int(random.randint(10, 30) * (1 - dist / (main_radius * 3))))
                
                if 0 <= dx < self.w and 0 <= dy < self.h:
                    cv2.circle(mask, (dx, dy), dot_radius, 255, -1)
        
        # Connect some dots for drip effect
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask
    
    def generate_scratch_damage(self) -> np.ndarray:
        """
        Simulated physical damage - scratches, tears.
        Thicker than simple lines.
        """
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
        num_scratches = random.randint(3, 8)
        
        for _ in range(num_scratches):
            # Start point
            x1 = random.randint(0, self.w)
            y1 = random.randint(0, self.h)
            
            # Scratch direction (mostly diagonal)
            angle = random.uniform(-math.pi, math.pi)
            length = random.randint(100, max(self.h, self.w))
            
            x2 = int(x1 + length * math.cos(angle))
            y2 = int(y1 + length * math.sin(angle))
            
            # Variable thickness scratch (20-50 pixels)
            thickness = random.randint(20, 50)
            
            # Draw with some wobble
            pts = []
            steps = 20
            for i in range(steps + 1):
                t = i / steps
                x = int(x1 + (x2 - x1) * t + random.randint(-10, 10))
                y = int(y1 + (y2 - y1) * t + random.randint(-10, 10))
                pts.append([x, y])
            
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(mask, [pts_arr], False, 255, thickness=thickness)
        
        # Add some damage blobs along scratches
        for _ in range(random.randint(5, 15)):
            cx = random.randint(0, self.w)
            cy = random.randint(0, self.h)
            r = random.randint(15, 40)
            cv2.circle(mask, (cx, cy), r, 255, -1)
        
        return mask
    
    def generate_mixed_composite(self) -> np.ndarray:
        """
        Combination of multiple styles for maximum variety.
        """
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
        # Pick 2-3 random styles
        styles = [
            self.generate_large_organic_blob,
            self.generate_thick_brush_strokes,
            self.generate_rectangular_patches,
            self.generate_splatter,
            self.generate_scratch_damage
        ]
        
        selected = random.sample(styles, random.randint(2, 3))
        
        for style_func in selected:
            m = style_func()
            mask = cv2.bitwise_or(mask, m)
        
        return mask
    
    # =========================================
    # BORDER EFFECTS
    # =========================================
    
    def add_smudged_border(self, mask: np.ndarray) -> np.ndarray:
        """Add distorted smudged border around edges."""
        edges = cv2.Canny(mask, 50, 150)
        
        border_width = random.randint(3, 8)
        kernel = np.ones((border_width, border_width), np.uint8)
        border = cv2.dilate(edges, kernel, iterations=1)
        
        h, w = mask.shape
        displacement = random.uniform(3, 8)
        noise_x = np.random.randn(h, w).astype(np.float32) * displacement
        noise_y = np.random.randn(h, w).astype(np.float32) * displacement
        
        noise_x = cv2.GaussianBlur(noise_x, (7, 7), 0)
        noise_y = cv2.GaussianBlur(noise_y, (7, 7), 0)
        
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        map_x = x_coords + noise_x * (border > 0).astype(np.float32)
        map_y = y_coords + noise_y * (border > 0).astype(np.float32)
        
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        
        distorted = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR)
        
        blur_size = random.choice([5, 7, 9])
        smudged = cv2.GaussianBlur(border, (blur_size, blur_size), 0)
        
        opacity = random.uniform(0.4, 0.8)
        result = distorted.astype(np.float32) + smudged.astype(np.float32) * opacity
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    # =========================================
    # MAIN CORRUPTION
    # =========================================
    
    def apply_corruption(
        self, 
        base_image: np.ndarray,
        overlay_image: np.ndarray,
        style: str = 'random'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply high-quality corruption."""
        self.h, self.w = base_image.shape[:2]
        
        # Style selection - NO thin lines!
        styles = {
            'blob': self.generate_large_organic_blob,
            'brush': self.generate_thick_brush_strokes,
            'rect': self.generate_rectangular_patches,
            'splatter': self.generate_splatter,
            'scratch': self.generate_scratch_damage,
            'mixed': self.generate_mixed_composite
        }
        
        if style == 'random':
            style = random.choice(list(styles.keys()))
        
        mask = styles[style]()
        
        # Resize overlay
        overlay = cv2.resize(overlay_image, (self.w, self.h))
        
        # Random transforms
        if random.random() > 0.5:
            overlay = cv2.flip(overlay, 1)
        if random.random() > 0.5:
            overlay = cv2.flip(overlay, 0)
        if random.random() > 0.4:
            angle = random.choice([90, 180, 270, random.randint(-30, 30)])
            center = (self.w // 2, self.h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            overlay = cv2.warpAffine(overlay, M, (self.w, self.h))
        
        # Add smudged border
        mask = self.add_smudged_border(mask)
        
        # Soft edges
        blur_size = random.choice([5, 7, 11, 15])
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        
        # Composite
        alpha = mask.astype(np.float32) / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=-1)
        
        opacity = random.uniform(0.7, 1.0)
        alpha = alpha * opacity
        
        result = (overlay.astype(np.float32) * alpha + 
                  base_image.astype(np.float32) * (1 - alpha))
        
        return result.astype(np.uint8), mask


# =========================================
# MULTIPROCESSING WORKER
# =========================================

def process_image(args):
    """Worker function for parallel processing."""
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
            # Use transformed base as fallback
            overlay_img = cv2.flip(base_img, 1)
            hsv = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(20, 80)) % 180
            overlay_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        generator = HighQualityMaskGenerator()
        corrupted, mask = generator.apply_corruption(base_img, overlay_img)
        
        out_name = f"{idx:06d}.png"
        cv2.imwrite(os.path.join(output_dir, 'input', out_name), corrupted)
        cv2.imwrite(os.path.join(output_dir, 'target', out_name), base_img)
        
        if save_mask:
            cv2.imwrite(os.path.join(output_dir, 'mask', out_name), mask)
        
        return idx
    
    except Exception as e:
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
    """
    Generate large-scale dataset (100K+).
    
    Args:
        clean_folder: Folder with clean images
        overlay_folder: Folder with overlay/texture images
        output_folder: Output directory
        total_images: Total images to generate (can be > clean images)
        num_workers: Parallel workers (default: CPU - 1)
        save_masks: Save binary masks
        seed: Random seed
    """
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
    
    print("=" * 65)
    print("HIGH-QUALITY MASK GENERATOR v4 - Production Scale")
    print("=" * 65)
    print(f"Clean images:    {len(clean_files):,}")
    print(f"Overlay images:  {len(overlay_files):,}")
    print(f"Target output:   {total_images:,} images")
    print(f"Output folder:   {output_folder}")
    print("=" * 65)
    print("Mask styles: blob, brush, rect, splatter, scratch, mixed")
    print("(NO thin lines - high quality only)")
    print("=" * 65)
    
    if len(clean_files) == 0:
        raise ValueError(f"No images in {clean_folder}")
    if len(overlay_files) == 0:
        print("Warning: Using transformed base images as overlays")
        overlay_files = clean_files
    
    # Create tasks - cycle through clean images
    tasks = []
    for i in range(total_images):
        base_path = clean_files[i % len(clean_files)]
        tasks.append((i, base_path, overlay_files, output_folder, save_masks, seed))
    
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
    print("=" * 65)
    print(f"COMPLETE! Generated {completed:,}/{total_images:,} images")
    print(f"Output: {output_folder}")
    print("=" * 65)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='High-Quality Mask Generator v4 - Scale to 100K+ images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10,000 high-quality corrupted images
  python high_quality_mask_generator.py -c ./clean -l ./overlays -o ./output -n 10000

  # Generate 100,000 images with 8 workers
  python high_quality_mask_generator.py -c ./clean -l ./overlays -o ./output -n 100000 -w 8

Mask Styles (all high-quality, NO thin lines):
  - blob:     Large organic blobs (20-50% coverage)
  - brush:    Thick brush strokes (30-80px)
  - rect:     Overlapping rectangles
  - splatter: Paint splatter effect
  - scratch:  Simulated damage
  - mixed:    Combination of above
        """
    )
    
    parser.add_argument('--clean', '-c', required=True, help='Clean images folder')
    parser.add_argument('--overlays', '-l', required=True, help='Overlay images folder')
    parser.add_argument('--output', '-o', required=True, help='Output folder')
    parser.add_argument('--num-images', '-n', type=int, default=10000, help='Total images')
    parser.add_argument('--workers', '-w', type=int, default=None, help='Parallel workers')
    parser.add_argument('--no-masks', action='store_true', help='Skip saving masks')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
    
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
