"""
Data Generation Package for Blind Image Inpainting
===================================================

This package contains tools for generating training datasets with
corrupted/masked images for blind inpainting models.

Main Components:
    - production_mask_generator.py : Scalable generation (10K+ images)
    - diverse_mask_generator.py    : Swirl, rectangular, organic masks
    - organic_mask_generator.py    : Enhanced masks with real image overlays

Usage:
    # Generate 10,000 corrupted images
    python production_mask_generator.py \\
        --clean ./clean_images \\
        --overlays ./overlay_images \\
        --output ./train_data \\
        --num-images 10000 \\
        --workers 4

    # Or import in code
    from data_generation.production_mask_generator import generate_large_dataset
    generate_large_dataset(
        clean_folder='./clean',
        overlay_folder='./overlays',
        output_folder='./output',
        total_images=10000
    )

Output Structure:
    output/
    ├── input/    # Corrupted images (model input)
    ├── target/   # Clean images (ground truth)
    └── mask/     # Binary masks (optional)
"""

from .production_mask_generator import (
    ProductionMaskGenerator,
    generate_large_dataset,
    process_single_image
)

__all__ = [
    'ProductionMaskGenerator',
    'generate_large_dataset',
    'process_single_image'
]
