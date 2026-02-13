"""
Flickr8k Fusion Dataset Generator for Google Colab
Generates paired dataset: target (clean base) and input (fused with mask)
"""

# ============================================================================
# STEP 1: Install Dependencies
# ============================================================================

print("Installing dependencies...")
!pip install -q kaggle opencv-python-headless
print("✓ Dependencies installed!\n")


# ============================================================================
# STEP 2: Download Flickr8k Dataset
# ============================================================================

import os
import zipfile
from pathlib import Path

def download_flickr8k():
    """
    Download Flickr8k dataset using Kaggle API.
    You need to upload your kaggle.json file first.
    """
    print("Setting up Kaggle API...")
    
    # Create kaggle directory
    os.makedirs('/root/.kaggle', exist_ok=True)
    
    # Upload kaggle.json instructions
    print("\n" + "="*70)
    print("IMPORTANT: Upload your kaggle.json file")
    print("="*70)
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Upload the downloaded kaggle.json file when prompted below")
    print("="*70 + "\n")
    
    from google.colab import files
    uploaded = files.upload()
    
    if 'kaggle.json' in uploaded:
        # Move kaggle.json to correct location
        !mv kaggle.json /root/.kaggle/
        !chmod 600 /root/.kaggle/kaggle.json
        print("✓ Kaggle credentials configured!\n")
    else:
        print("⚠ kaggle.json not found. Please upload it.")
        return False
    
    # Download dataset
    print("Downloading Flickr8k dataset (this may take a few minutes)...")
    !kaggle datasets download -d adityajn105/flickr8k
    
    # Extract
    print("Extracting dataset...")
    with zipfile.ZipFile('flickr8k.zip', 'r') as zip_ref:
        zip_ref.extractall('flickr8k_data')
    
    print("✓ Flickr8k dataset ready!\n")
    return True

# Alternative: Manual upload (if you already have the dataset)
def upload_flickr8k_manual():
    """
    If you already have Flickr8k images, upload them manually.
    """
    print("Please upload your Flickr8k images folder as a zip file...")
    from google.colab import files
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('flickr8k_data')
            print(f"✓ Extracted {filename}")
    
    return True


# ============================================================================
# STEP 3: Fusion Mask Generator (Compact Version for Colab)
# ============================================================================

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

class FusionMaskGenerator:
    """Organic fusion mask generator."""
    
    def __init__(self, height, width, seed=None):
        self.height = height
        self.width = width
        if seed is not None:
            np.random.seed(seed)
    
    def _generate_perlin_noise(self, scale=10.0):
        """Generate Perlin-like noise."""
        grid_h = int(self.height / scale) + 2
        grid_w = int(self.width / scale) + 2
        
        angles = np.random.rand(grid_h, grid_w) * 2 * np.pi
        gradients_x = np.cos(angles)
        gradients_y = np.sin(angles)
        
        y = np.linspace(0, grid_h - 1, self.height)
        x = np.linspace(0, grid_w - 1, self.width)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        
        y0 = yy.astype(int)
        x0 = xx.astype(int)
        y1 = np.minimum(y0 + 1, grid_h - 1)
        x1 = np.minimum(x0 + 1, grid_w - 1)
        
        fy = yy - y0
        fx = xx - x0
        fy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
        fx = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
        
        n00 = gradients_x[y0, x0] * (xx - x0) + gradients_y[y0, x0] * (yy - y0)
        n01 = gradients_x[y0, x1] * (xx - x1) + gradients_y[y0, x1] * (yy - y0)
        n10 = gradients_x[y1, x0] * (xx - x0) + gradients_y[y1, x0] * (yy - y1)
        n11 = gradients_x[y1, x1] * (xx - x1) + gradients_y[y1, x1] * (yy - y1)
        
        nx0 = n00 * (1 - fx) + n01 * fx
        nx1 = n10 * (1 - fx) + n11 * fx
        noise = nx0 * (1 - fy) + nx1 * fy
        
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return noise
    
    def _generate_worley_noise(self, n_points=20):
        """Generate Worley (cellular) noise."""
        points = np.random.rand(n_points, 2)
        points[:, 0] *= self.height
        points[:, 1] *= self.width
        
        y, x = np.mgrid[0:self.height, 0:self.width]
        min_dist = np.full((self.height, self.width), np.inf)
        
        for point in points:
            dist = np.sqrt((y - point[0])**2 + (x - point[1])**2)
            min_dist = np.minimum(min_dist, dist)
        
        min_dist = (min_dist - min_dist.min()) / (min_dist.max() - min_dist.min() + 1e-8)
        return min_dist
    
    def _generate_turbulence(self, octaves=4, persistence=0.5):
        """Generate fractal turbulence."""
        noise = np.zeros((self.height, self.width))
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            noise += amplitude * self._generate_perlin_noise(scale=30.0 / frequency)
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2.0
        
        noise /= max_value
        return noise
    
    def generate_mask(self, blend_ratio=0.5, smoothness=30.0, 
                     complexity=4, worley_strength=0.3):
        """Generate organic fusion mask."""
        # Generate base noise
        base_noise = self._generate_turbulence(octaves=complexity)
        
        # Add Worley noise
        if worley_strength > 0:
            worley = self._generate_worley_noise(n_points=int(15 + complexity * 5))
            base_noise = (1 - worley_strength) * base_noise + worley_strength * worley
        
        # Add fine variation
        fine_noise = np.random.rand(self.height, self.width) * 0.1
        base_noise = base_noise * 0.9 + fine_noise
        
        # Apply threshold
        threshold = np.random.uniform(0.3, 0.7)
        mask = np.clip((base_noise - threshold + 0.2) / 0.4, 0, 1)
        
        # Smooth transitions
        mask = gaussian_filter(mask, sigma=smoothness)
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask = cv2.bilateralFilter(mask_uint8, d=9, sigmaColor=75, sigmaSpace=75)
        mask = mask.astype(np.float32) / 255.0
        
        # Adjust blend ratio
        mask = mask * blend_ratio
        
        # Random invert
        if np.random.rand() < 0.5:
            mask = blend_ratio - mask
        
        # Add vignette
        y, x = np.mgrid[0:self.height, 0:self.width]
        cy, cx = self.height / 2, self.width / 2
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        vignette = 1 - (dist / (np.sqrt(cy**2 + cx**2))) * 0.15
        mask = mask * vignette
        
        mask = np.clip(mask, 0, 1).astype(np.float32)
        return mask
    
    def apply_mask(self, image1, image2, mask):
        """Apply mask to composite two images."""
        img1 = image1.astype(np.float32)
        img2 = image2.astype(np.float32)
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        result = img1 * (1 - mask_3ch) + img2 * mask_3ch
        return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================================
# STEP 4: Dataset Generation Pipeline
# ============================================================================

import json
from tqdm import tqdm
import random

def generate_fusion_dataset(
    images_dir,
    output_base_dir,
    num_samples=1000,
    image_size=512,
    train_split=0.9
):
    """
    Generate fusion dataset with target and input folders.
    
    Args:
        images_dir: Path to Flickr8k images
        output_base_dir: Base directory for output
        num_samples: Number of samples to generate
        image_size: Size to resize images to
        train_split: Train/val split ratio
    """
    print("="*70)
    print("FUSION DATASET GENERATION")
    print("="*70)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).rglob(f'*{ext}'))
    
    image_files = [str(f) for f in image_files]
    print(f"\n✓ Found {len(image_files)} images")
    
    if len(image_files) < 2:
        raise ValueError("Need at least 2 images!")
    
    # Create output directories
    train_dir = Path(output_base_dir) / 'train'
    val_dir = Path(output_base_dir) / 'val'
    
    for split_dir in [train_dir, val_dir]:
        (split_dir / 'target').mkdir(parents=True, exist_ok=True)
        (split_dir / 'input').mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created output directories")
    print(f"  - {train_dir}")
    print(f"  - {val_dir}")
    
    # Generate samples
    num_train = int(num_samples * train_split)
    num_val = num_samples - num_train
    
    dataset_info = {'train': [], 'val': []}
    
    print(f"\n📊 Generating {num_train} train + {num_val} val samples...")
    print("="*70)
    
    for split_name, split_dir, split_count in [
        ('train', train_dir, num_train),
        ('val', val_dir, num_val)
    ]:
        print(f"\n🔄 Processing {split_name} set...")
        
        for i in tqdm(range(split_count), desc=f"{split_name.upper()}"):
            # Randomly select two different images
            idx1, idx2 = random.sample(range(len(image_files)), 2)
            img_path1 = image_files[idx1]
            img_path2 = image_files[idx2]
            
            # Load images
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)
            
            if img1 is None or img2 is None:
                print(f"⚠ Skipping sample {i}: Could not load images")
                continue
            
            # Resize to same dimensions
            img1 = cv2.resize(img1, (image_size, image_size))
            img2 = cv2.resize(img2, (image_size, image_size))
            
            # Generate unique mask
            seed = hash(f"{split_name}_{i}") % (2**32)
            generator = FusionMaskGenerator(image_size, image_size, seed=seed)
            
            # Randomize parameters for diversity
            mask = generator.generate_mask(
                blend_ratio=np.random.uniform(0.45, 0.65),
                smoothness=np.random.uniform(30.0, 45.0),
                complexity=np.random.randint(4, 6),
                worley_strength=np.random.uniform(0.25, 0.45)
            )
            
            # Create fusion (input) and keep base (target)
            fused = generator.apply_mask(img1, img2, mask)
            
            # Generate filename (same for both)
            filename = f"{i:06d}.jpg"
            
            # Save target (base image - clean)
            target_path = split_dir / 'target' / filename
            cv2.imwrite(str(target_path), img1)
            
            # Save input (fused image - corrupted)
            input_path = split_dir / 'input' / filename
            cv2.imwrite(str(input_path), fused)
            
            # Store metadata
            dataset_info[split_name].append({
                'filename': filename,
                'source_image1': img_path1,
                'source_image2': img_path2,
                'blend_ratio': float(mask.mean()),
                'target_path': str(target_path),
                'input_path': str(input_path)
            })
    
    # Save metadata
    metadata_path = Path(output_base_dir) / 'dataset_info.json'
    with open(metadata_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ DATASET GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Output structure:")
    print(f"  {output_base_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── target/  ({num_train} clean base images)")
    print(f"  │   └── input/   ({num_train} fused images)")
    print(f"  ├── val/")
    print(f"  │   ├── target/  ({num_val} clean base images)")
    print(f"  │   └── input/   ({num_val} fused images)")
    print(f"  └── dataset_info.json")
    print(f"\n✓ All images have matching filenames between target and input!")
    print("="*70)
    
    return dataset_info


# ============================================================================
# STEP 5: Visualization
# ============================================================================

import matplotlib.pyplot as plt

def visualize_samples(output_base_dir, num_samples=6):
    """Visualize some samples from the generated dataset."""
    train_dir = Path(output_base_dir) / 'train'
    
    # Get sample files
    target_files = sorted(list((train_dir / 'target').glob('*.jpg')))[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    
    for i, target_file in enumerate(target_files):
        filename = target_file.name
        input_file = train_dir / 'input' / filename
        
        # Load images
        target = cv2.imread(str(target_file))
        input_img = cv2.imread(str(input_file))
        
        # Convert BGR to RGB
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Plot
        axes[i, 0].imshow(target)
        axes[i, 0].set_title(f'Target (Clean) - {filename}', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(input_img)
        axes[i, 1].set_title(f'Input (Fused) - {filename}', fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(output_base_dir) / 'sample_visualization.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Visualization saved to {output_base_dir}/sample_visualization.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FLICKR8K FUSION DATASET GENERATOR FOR GOOGLE COLAB")
    print("="*70 + "\n")
    
    # Step 1: Choose download method
    print("Choose download method:")
    print("1. Download via Kaggle API (recommended)")
    print("2. Manual upload (if you already have the dataset)")
    print("3. Use existing path in Colab")
    
    method = input("\nEnter choice (1/2/3): ").strip()
    
    if method == '1':
        success = download_flickr8k()
        images_dir = 'flickr8k_data/Images'
    elif method == '2':
        success = upload_flickr8k_manual()
        images_dir = 'flickr8k_data/Images'
    else:
        images_dir = input("Enter path to Flickr8k images folder: ").strip()
        success = True
    
    if not success:
        print("❌ Dataset setup failed!")
        exit()
    
    # Step 2: Generate dataset
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    
    # Configuration
    num_samples = int(input("Number of samples to generate (default: 1000): ") or "1000")
    image_size = int(input("Image size (default: 512): ") or "512")
    output_dir = input("Output directory (default: fusion_dataset): ").strip() or "fusion_dataset"
    
    print(f"\n📝 Configuration:")
    print(f"  - Images directory: {images_dir}")
    print(f"  - Number of samples: {num_samples}")
    print(f"  - Image size: {image_size}x{image_size}")
    print(f"  - Output directory: {output_dir}")
    
    # Generate dataset
    dataset_info = generate_fusion_dataset(
        images_dir=images_dir,
        output_base_dir=output_dir,
        num_samples=num_samples,
        image_size=image_size,
        train_split=0.9
    )
    
    # Step 3: Visualize samples
    print("\n" + "="*70)
    print("Visualizing sample results...")
    visualize_samples(output_dir, num_samples=6)
    
    # Step 4: Download results (optional)
    print("\n" + "="*70)
    download_choice = input("Download dataset as zip? (y/n): ").strip().lower()
    
    if download_choice == 'y':
        import shutil
        print("Creating zip file...")
        shutil.make_archive(output_dir, 'zip', output_dir)
        
        from google.colab import files
        print("Downloading...")
        files.download(f'{output_dir}.zip')
        print("✓ Download complete!")
    
    print("\n" + "="*70)
    print("🎉 ALL DONE!")
    print("="*70)