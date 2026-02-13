"""
Enhanced Organic Mask Generator v2
===================================

Key improvements based on feedback:
1. Uses REAL IMAGES as overlay content (not just noise)
2. More varied mask shapes: thin lines, thick patches, irregular shapes
3. Better matches original dataset quality

Dependencies: pip install opencv-python numpy
"""

import cv2
import numpy as np
import os
import random
import math
from typing import Optional, List, Tuple


class EnhancedMaskGenerator:
    """
    Generates masks that match the original dataset quality:
    - Real image overlays (another image composited on top)
    - Varied mask shapes (thin, thick, irregular, patches)
    """
    
    def __init__(self, height: int = 512, width: int = 512):
        self.h = height
        self.w = width
    
    # =========================================
    # MASK SHAPE GENERATORS (More Variety)
    # =========================================
    
    def generate_thin_lines(self) -> np.ndarray:
        """Thin scribble-like lines."""
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        num_lines = random.randint(3, 10)
        
        for _ in range(num_lines):
            # Random bezier-like curve with thin width
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
            thickness = random.randint(1, 5)  # THIN lines
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
            thickness = random.randint(15, 50)  # THICK strokes
            cv2.polylines(mask, [pts], False, 255, thickness=thickness)
            
            # Add circles at joints
            for p in pts:
                cv2.circle(mask, tuple(p), thickness // 2, 255, -1)
        
        return mask
    
    def generate_irregular_patches(self) -> np.ndarray:
        """Irregular blob/patch shapes."""
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        num_patches = random.randint(1, 5)
        
        for _ in range(num_patches):
            # Generate random polygon
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
        
        # Mix of different elements
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
            # Random splatter points
            for _ in range(random.randint(20, 50)):
                cx = random.randint(0, self.w)
                cy = random.randint(0, self.h)
                r = random.randint(5, 30)
                cv2.circle(mask, (cx, cy), r, 255, -1)
        
        elif shape_type == 'cracks':
            # Crack-like lines
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
        """Combines multiple mask types for variety."""
        # Randomly pick 1-3 mask types
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
    
    # =========================================
    # OVERLAY CONTENT (Real Images)
    # =========================================
    
    def get_overlay_image(self, overlay_images: List[str]) -> Optional[np.ndarray]:
        """
        Get a random REAL IMAGE to use as overlay content.
        This is the key difference - using actual images, not noise.
        """
        if not overlay_images:
            return None
        
        img_path = random.choice(overlay_images)
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Resize to match dimensions
        img = cv2.resize(img, (self.w, self.h))
        
        # Random transformations
        if random.random() > 0.5:
            img = cv2.flip(img, 1)  # Horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 0)  # Vertical flip
        if random.random() > 0.3:
            # Rotate
            angle = random.choice([90, 180, 270])
            center = (self.w // 2, self.h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (self.w, self.h))
        
        return img
    
    # =========================================
    # MAIN CORRUPTION FUNCTION
    # =========================================
    
    def apply_corruption(
        self, 
        base_image: np.ndarray,
        overlay_images: List[str],
        mask_type: str = 'random'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply image-based corruption to base image.
        
        Args:
            base_image: The clean image to corrupt
            overlay_images: List of image paths to use as overlay content
            mask_type: 'thin', 'thick', 'patch', 'weird', 'mixed', or 'random'
        
        Returns:
            (corrupted_image, mask)
        """
        self.h, self.w = base_image.shape[:2]
        
        # 1. Generate mask shape
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
        
        # 2. Get overlay content (REAL IMAGE)
        overlay = self.get_overlay_image(overlay_images)
        
        if overlay is None:
            # Fallback: use warped version of base image
            M = cv2.getRotationMatrix2D((self.w//2, self.h//2), random.randint(10, 350), 1.0)
            overlay = cv2.warpAffine(base_image, M, (self.w, self.h))
            # Shift colors
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2HSV)
            overlay[:, :, 0] = (overlay[:, :, 0] + random.randint(20, 80)) % 180
            overlay = cv2.cvtColor(overlay, cv2.COLOR_HSV2BGR)
        
        # 3. Add distorted smudged border (NEW!)
        mask = self._add_smudged_border(mask)
        
        # 4. Soft edge for mask (optional)
        if random.random() > 0.3:
            kernel_size = random.choice([3, 5, 7])
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        # 5. Composite
        alpha = mask.astype(np.float32) / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=-1)
        
        # Random opacity
        opacity = random.uniform(0.6, 1.0)
        alpha = alpha * opacity
        
        result = (overlay.astype(np.float32) * alpha + 
                  base_image.astype(np.float32) * (1 - alpha))
        
        return result.astype(np.uint8), mask
    
    def _add_smudged_border(self, mask: np.ndarray) -> np.ndarray:
        """
        Add a thin distorted/smudged border around mask edges.
        Creates organic, blurred edge transitions.
        """
        # Find edges of the mask
        edges = cv2.Canny(mask, 50, 150)
        
        # Dilate edges slightly to create thin border
        border_width = random.randint(2, 6)  # Thin border
        kernel = np.ones((border_width, border_width), np.uint8)
        border = cv2.dilate(edges, kernel, iterations=1)
        
        # Create distortion for the border using random displacement
        h, w = mask.shape
        
        # Generate displacement fields for distortion
        displacement_strength = random.uniform(2, 5)
        noise_x = np.random.randn(h, w).astype(np.float32) * displacement_strength
        noise_y = np.random.randn(h, w).astype(np.float32) * displacement_strength
        
        # Smooth the noise for organic feel
        noise_x = cv2.GaussianBlur(noise_x, (5, 5), 0)
        noise_y = cv2.GaussianBlur(noise_y, (5, 5), 0)
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Apply distortion only to border region
        map_x = x_coords + noise_x * (border > 0).astype(np.float32)
        map_y = y_coords + noise_y * (border > 0).astype(np.float32)
        
        # Clip coordinates
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        
        # Remap the mask for distorted edges
        distorted_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR)
        
        # Blur the border region for smudged effect
        blur_strength = random.choice([3, 5, 7])
        smudged_border = cv2.GaussianBlur(border, (blur_strength, blur_strength), 0)
        
        # Blend: add smudged border with varying opacity
        border_opacity = random.uniform(0.3, 0.7)
        result = distorted_mask.astype(np.float32)
        result = result + smudged_border.astype(np.float32) * border_opacity
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result


def generate_dataset(
    input_folder: str,
    output_folder: str,
    overlay_folder: str,
    num_variants: int = 3,
    mask_types: List[str] = ['thin', 'thick', 'patch', 'weird', 'mixed']
) -> None:
    """
    Generate corrupted dataset using real image overlays.
    
    Args:
        input_folder: Folder with clean/target images
        output_folder: Where to save corrupted images
        overlay_folder: Folder with images to use as overlays (can be same as input)
        num_variants: Corrupted versions per image
        mask_types: Types of masks to use
    """
    os.makedirs(output_folder, exist_ok=True)
    
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    # Get input images
    input_files = [os.path.join(input_folder, f) 
                   for f in os.listdir(input_folder) 
                   if f.lower().endswith(valid_ext)]
    
    # Get overlay images
    overlay_files = [os.path.join(overlay_folder, f)
                     for f in os.listdir(overlay_folder)
                     if f.lower().endswith(valid_ext)]
    
    print(f"Found {len(input_files)} input images")
    print(f"Found {len(overlay_files)} overlay images")
    
    generator = EnhancedMaskGenerator()
    total = 0
    
    for img_path in input_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        basename = os.path.splitext(os.path.basename(img_path))[0]
        
        for v in range(num_variants):
            mask_type = random.choice(mask_types)
            corrupted, mask = generator.apply_corruption(img, overlay_files, mask_type)
            
            # Save
            if num_variants > 1:
                out_name = f"{basename}_v{v}.png"
            else:
                out_name = f"{basename}.png"
            
            cv2.imwrite(os.path.join(output_folder, out_name), corrupted)
            total += 1
            print(f"Generated: {out_name} (type: {mask_type})")
    
    print(f"\nComplete! Generated {total} corrupted images.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Mask Generator v2')
    parser.add_argument('--input', '-i', required=True, help='Clean images folder')
    parser.add_argument('--output', '-o', required=True, help='Output folder')
    parser.add_argument('--overlays', '-l', required=True, help='Overlay images folder')
    parser.add_argument('--variants', '-v', type=int, default=3, help='Variants per image')
    
    args = parser.parse_args()
    
    generate_dataset(
        input_folder=args.input,
        output_folder=args.output,
        overlay_folder=args.overlays,
        num_variants=args.variants
    )
