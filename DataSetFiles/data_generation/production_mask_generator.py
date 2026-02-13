"""
Production Mask Generator v3 - Scalable to 10K+ Images
=======================================================

Features:
- Multiprocessing for fast generation
- Progress tracking with tqdm
- Separate overlay folder support
- Memory efficient batch processing
- Easy CLI usage

Usage:
    python production_mask_generator.py --help

Folder Structure:
    datasets/
    ├── clean/           # Clean target images (your base images)
    ├── overlays/        # Images to use as overlay content
    └── corrupted/       # Output (generated corrupted images)
        ├── input/       # Corrupted images
        ├── target/      # Clean images (copied)
        └── mask/        # Binary masks (optional)

Dependencies: pip install opencv-python numpy tqdm
"""

import cv2
import numpy as np
import os
import random
import math
import shutil
from typing import Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Try to import tqdm, fallback to simple progress
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable


class ProductionMaskGenerator:
    """
    Production-ready mask generator for large-scale dataset creation.
    """
    
    def __init__(self, height: int = 512, width: int = 512, seed: int = None):
        self.h = height
        self.w = width
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    # =========================================
    # MASK SHAPE GENERATORS
    # =========================================
    
    def generate_thin_lines(self) -> np.ndarray:
        """Thin scribble-like lines."""
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        num_lines = random.randint(3, 10)
        
        for _ in range(num_lines):
            pts = []
            x, y = random.randint(0, self.w), random.randint(0, self.h)
            num_points = random.randint(4, 12)
            
            for _ in range(num_points):
                x += random.randint(-80, 80)
                y += random.randint(-80, 80)
                x = max(0, min(self.w - 1, x))
                y = max(0, min(self.h - 1, y))
                pts.append([x, y])
            
            pts = np.array(pts, dtype=np.int32)
            thickness = random.randint(1, 5)
            cv2.polylines(mask, [pts], False, 255, thickness=thickness)
        
        return mask
    
    def generate_thick_strokes(self) -> np.ndarray:
        """Thick brush strokes."""
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        num_strokes = random.randint(2, 6)
        
        for _ in range(num_strokes):
            pts = []
            x, y = random.randint(0, self.w), random.randint(0, self.h)
            num_points = random.randint(3, 8)
            
            for _ in range(num_points):
                angle = random.uniform(0, 2 * math.pi)
                length = random.randint(30, 120)
                x = int(x + length * math.cos(angle))
                y = int(y + length * math.sin(angle))
                x = max(0, min(self.w - 1, x))
                y = max(0, min(self.h - 1, y))
                pts.append([x, y])
            
            pts = np.array(pts, dtype=np.int32)
            thickness = random.randint(15, 50)
            cv2.polylines(mask, [pts], False, 255, thickness=thickness)
            
            for p in pts:
                cv2.circle(mask, tuple(p), thickness // 2, 255, -1)
        
        return mask
    
    def generate_irregular_patches(self) -> np.ndarray:
        """Irregular blob/patch shapes."""
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        num_patches = random.randint(1, 5)
        
        for _ in range(num_patches):
            cx = random.randint(self.w // 4, 3 * self.w // 4)
            cy = random.randint(self.h // 4, 3 * self.h // 4)
            
            num_vertices = random.randint(5, 12)
            angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_vertices)])
            
            pts = []
            for angle in angles:
                r = random.randint(20, 100)
                x = int(cx + r * math.cos(angle))
                y = int(cy + r * math.sin(angle))
                pts.append([x, y])
            
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        
        return mask
    
    def generate_weird_shapes(self) -> np.ndarray:
        """Random weird/abstract shapes."""
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        shape_type = random.choice(['ellipses', 'triangles', 'splatter', 'cracks'])
        
        if shape_type == 'ellipses':
            for _ in range(random.randint(2, 8)):
                cx = random.randint(0, self.w)
                cy = random.randint(0, self.h)
                ax1 = random.randint(20, 100)
                ax2 = random.randint(10, 60)
                angle = random.randint(0, 180)
                cv2.ellipse(mask, (cx, cy), (ax1, ax2), angle, 0, 360, 255, -1)
        
        elif shape_type == 'triangles':
            for _ in range(random.randint(2, 6)):
                pts = np.array([
                    [random.randint(0, self.w), random.randint(0, self.h)],
                    [random.randint(0, self.w), random.randint(0, self.h)],
                    [random.randint(0, self.w), random.randint(0, self.h)]
                ], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
        
        elif shape_type == 'splatter':
            for _ in range(random.randint(20, 50)):
                cx = random.randint(0, self.w)
                cy = random.randint(0, self.h)
                r = random.randint(5, 30)
                cv2.circle(mask, (cx, cy), r, 255, -1)
        
        elif shape_type == 'cracks':
            for _ in range(random.randint(5, 15)):
                x1 = random.randint(0, self.w)
                y1 = random.randint(0, self.h)
                for _ in range(random.randint(3, 8)):
                    x2 = x1 + random.randint(-50, 50)
                    y2 = y1 + random.randint(-50, 50)
                    cv2.line(mask, (x1, y1), (x2, y2), 255, random.randint(1, 4))
                    x1, y1 = x2, y2
        
        return mask
    
    def generate_mixed_mask(self) -> np.ndarray:
        """Combines multiple mask types."""
        mask_funcs = [
            self.generate_thin_lines,
            self.generate_thick_strokes,
            self.generate_irregular_patches,
            self.generate_weird_shapes
        ]
        
        combined = np.zeros((self.h, self.w), dtype=np.uint8)
        num_types = random.randint(1, 3)
        selected = random.sample(mask_funcs, num_types)
        
        for func in selected:
            m = func()
            combined = cv2.bitwise_or(combined, m)
        
        return combined
    
    def _add_smudged_border(self, mask: np.ndarray) -> np.ndarray:
        """Add distorted smudged border around mask edges."""
        edges = cv2.Canny(mask, 50, 150)
        
        border_width = random.randint(2, 6)
        kernel = np.ones((border_width, border_width), np.uint8)
        border = cv2.dilate(edges, kernel, iterations=1)
        
        h, w = mask.shape
        displacement_strength = random.uniform(2, 5)
        noise_x = np.random.randn(h, w).astype(np.float32) * displacement_strength
        noise_y = np.random.randn(h, w).astype(np.float32) * displacement_strength
        
        noise_x = cv2.GaussianBlur(noise_x, (5, 5), 0)
        noise_y = cv2.GaussianBlur(noise_y, (5, 5), 0)
        
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        map_x = x_coords + noise_x * (border > 0).astype(np.float32)
        map_y = y_coords + noise_y * (border > 0).astype(np.float32)
        
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        
        distorted_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR)
        
        blur_strength = random.choice([3, 5, 7])
        smudged_border = cv2.GaussianBlur(border, (blur_strength, blur_strength), 0)
        
        border_opacity = random.uniform(0.3, 0.7)
        result = distorted_mask.astype(np.float32)
        result = result + smudged_border.astype(np.float32) * border_opacity
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def apply_corruption(
        self, 
        base_image: np.ndarray,
        overlay_image: np.ndarray,
        mask_type: str = 'random'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply corruption to base image using overlay."""
        self.h, self.w = base_image.shape[:2]
        
        if mask_type == 'random':
            mask_type = random.choice(['thin', 'thick', 'patch', 'weird', 'mixed'])
        
        mask_generators = {
            'thin': self.generate_thin_lines,
            'thick': self.generate_thick_strokes,
            'patch': self.generate_irregular_patches,
            'weird': self.generate_weird_shapes,
            'mixed': self.generate_mixed_mask
        }
        
        mask = mask_generators[mask_type]()
        
        # Resize overlay to match
        overlay = cv2.resize(overlay_image, (self.w, self.h))
        
        # Random transforms on overlay
        if random.random() > 0.5:
            overlay = cv2.flip(overlay, 1)
        if random.random() > 0.5:
            overlay = cv2.flip(overlay, 0)
        if random.random() > 0.3:
            angle = random.choice([90, 180, 270])
            center = (self.w // 2, self.h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            overlay = cv2.warpAffine(overlay, M, (self.w, self.h))
        
        # Add smudged border
        mask = self._add_smudged_border(mask)
        
        # Soft edge
        if random.random() > 0.3:
            kernel_size = random.choice([3, 5, 7])
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        # Composite
        alpha = mask.astype(np.float32) / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=-1)
        opacity = random.uniform(0.6, 1.0)
        alpha = alpha * opacity
        
        result = (overlay.astype(np.float32) * alpha + 
                  base_image.astype(np.float32) * (1 - alpha))
        
        return result.astype(np.uint8), mask


# =========================================
# WORKER FUNCTION FOR MULTIPROCESSING
# =========================================

def process_single_image(args):
    """Process a single image (for multiprocessing)."""
    idx, base_path, overlay_paths, output_dir, save_mask, seed = args
    
    # Set unique seed for this worker
    random.seed(seed + idx)
    np.random.seed(seed + idx)
    
    try:
        # Load base image
        base_img = cv2.imread(base_path)
        if base_img is None:
            return None
        
        # Pick random overlay
        overlay_path = random.choice(overlay_paths)
        overlay_img = cv2.imread(overlay_path)
        if overlay_img is None:
            overlay_img = base_img.copy()
        
        # Generate corruption
        generator = ProductionMaskGenerator()
        corrupted, mask = generator.apply_corruption(base_img, overlay_img)
        
        # Save
        out_name = f"{idx:06d}.png"
        cv2.imwrite(os.path.join(output_dir, 'input', out_name), corrupted)
        cv2.imwrite(os.path.join(output_dir, 'target', out_name), base_img)
        
        if save_mask:
            cv2.imwrite(os.path.join(output_dir, 'mask', out_name), mask)
        
        return idx
    
    except Exception as e:
        print(f"Error processing {base_path}: {e}")
        return None


def generate_large_dataset(
    clean_folder: str,
    overlay_folder: str,
    output_folder: str,
    total_images: int = 10000,
    num_workers: int = None,
    save_masks: bool = True,
    seed: int = 42
) -> None:
    """
    Generate large-scale corrupted dataset efficiently.
    
    Args:
        clean_folder: Folder with clean/target images
        overlay_folder: Folder with images to use as overlays
        output_folder: Output directory (will create input/, target/, mask/ subdirs)
        total_images: Total number of corrupted images to generate
        num_workers: Number of parallel workers (default: CPU count - 1)
        save_masks: Whether to save binary masks
        seed: Random seed for reproducibility
    """
    # Setup output directories
    os.makedirs(os.path.join(output_folder, 'input'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'target'), exist_ok=True)
    if save_masks:
        os.makedirs(os.path.join(output_folder, 'mask'), exist_ok=True)
    
    # Get image lists
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    clean_files = [os.path.join(clean_folder, f) 
                   for f in os.listdir(clean_folder) 
                   if f.lower().endswith(valid_ext)]
    
    overlay_files = [os.path.join(overlay_folder, f)
                     for f in os.listdir(overlay_folder)
                     if f.lower().endswith(valid_ext)]
    
    print(f"=" * 60)
    print(f"PRODUCTION MASK GENERATOR")
    print(f"=" * 60)
    print(f"Clean images:   {len(clean_files)}")
    print(f"Overlay images: {len(overlay_files)}")
    print(f"Target output:  {total_images} images")
    print(f"Output folder:  {output_folder}")
    print(f"=" * 60)
    
    if len(clean_files) == 0:
        raise ValueError(f"No images found in {clean_folder}")
    if len(overlay_files) == 0:
        print("Warning: No overlay images, will use transformed base images")
        overlay_files = clean_files
    
    # Create task list - cycle through clean images to reach total
    tasks = []
    for i in range(total_images):
        base_path = clean_files[i % len(clean_files)]
        tasks.append((i, base_path, overlay_files, output_folder, save_masks, seed))
    
    # Determine workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Using {num_workers} parallel workers")
    print(f"Generating {total_images} corrupted images...")
    print()
    
    # Process with progress bar
    completed = 0
    
    if num_workers == 1:
        # Single process mode (easier debugging)
        for task in tqdm(tasks, desc="Generating", unit="img"):
            result = process_single_image(task)
            if result is not None:
                completed += 1
    else:
        # Multiprocessing mode
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_image, task): task[0] for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating", unit="img"):
                result = future.result()
                if result is not None:
                    completed += 1
    
    print()
    print(f"=" * 60)
    print(f"COMPLETE!")
    print(f"Generated: {completed}/{total_images} images")
    print(f"Output:    {output_folder}")
    print(f"=" * 60)


# =========================================
# CLI INTERFACE
# =========================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Production Mask Generator - Scale to 10K+ images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 test images
  python production_mask_generator.py -c ./clean -l ./overlays -o ./output -n 100

  # Generate 10,000 images with 8 workers
  python production_mask_generator.py -c ./clean -l ./overlays -o ./output -n 10000 -w 8

  # Use same folder for clean and overlays
  python production_mask_generator.py -c ./images -l ./images -o ./corrupted -n 5000

Folder Structure:
  After running, output folder will contain:
    output/
    ├── input/    # Corrupted images (for model input)
    ├── target/   # Clean images (ground truth)
    └── mask/     # Binary masks (optional)
        """
    )
    
    parser.add_argument('--clean', '-c', required=True, 
                        help='Folder with clean/target images')
    parser.add_argument('--overlays', '-l', required=True,
                        help='Folder with overlay images (can be same as clean)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output folder')
    parser.add_argument('--num-images', '-n', type=int, default=1000,
                        help='Total number of images to generate (default: 1000)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--no-masks', action='store_true',
                        help='Do not save binary masks')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    generate_large_dataset(
        clean_folder=args.clean,
        overlay_folder=args.overlays,
        output_folder=args.output,
        total_images=args.num_images,
        num_workers=args.workers,
        save_masks=not args.no_masks,
        seed=args.seed
    )
