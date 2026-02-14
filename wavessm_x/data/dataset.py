import torch
import torchvision.transforms.functional as T
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import os
import glob
import random
import numpy as np
from typing import Optional, List, Tuple


def pad_image_needed(img: torch.Tensor, size: Tuple[int, int]):
    """Optimized image padding with memory efficiency"""
    _, h, w = img.shape
    pad_h = max(0, size[0] - h)
    pad_w = max(0, size[1] - w)
    
    if pad_h > 0 or pad_w > 0:
        img = T.pad(img, (0, 0, pad_w, pad_h), padding_mode='reflect')
    return img


def rgb_to_y(x: torch.Tensor):
    """Optimized RGB to Y channel conversion"""
    return 0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3]


class TrainDataset(Dataset):
    """
    Enhanced training dataset with optimizations.
    """
    def __init__(self, data_path: str, data_path_test: str, data_name: str, 
                 data_type: str, patch_size: Optional[int] = None, 
                 length: Optional[int] = None, use_advanced_aug: bool = True,
                 inp_files: Optional[List[str]] = None, target_files: Optional[List[str]] = None):
        super().__init__()
        self.data_name, self.data_path, self.patch_size = data_name, data_path, patch_size
        self.data_type = data_type
        
        if inp_files and target_files:
             self.inp_files = inp_files
             self.target_files = target_files
        else:
             self._discover_images(data_path)
             
        self._validate_image_pairs()

        self.sample_num = len(self.inp_files)
        if length:
             self.sample_num = length
        
        # Cache for small datasets (under 5000 images)
        self.use_cache = self.sample_num < 5000 
        self.image_cache = {}

    def _discover_images(self, data_path: str):
        """Enhanced image discovery with multiple formats"""
        exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']
        self.inp_files = []
        self.target_files = []
        
        inp_dirs = ['inp', 'input']
        target_dirs = ['target', 'gt', 'ground_truth']
        
        found_inp = False
        for d in inp_dirs:
            p = os.path.join(data_path, d)
            if os.path.exists(p):
                for ext in exts:
                    self.inp_files.extend(glob.glob(os.path.join(p, ext)))
                found_inp = True
                break
                
        found_target = False
        for d in target_dirs:
            p = os.path.join(data_path, d)
            if os.path.exists(p):
                for ext in exts:
                    self.target_files.extend(glob.glob(os.path.join(p, ext)))
                found_target = True
                break

        if not found_inp or not found_target:
             print(f"Dataset warning: Could not find inputs/targets in {data_path}")
             
        self.inp_files.sort()
        self.target_files.sort()

    def _validate_image_pairs(self):
        """Validate that corrupt and clear images are properly paired"""
        min_len = min(len(self.inp_files), len(self.target_files))
        if len(self.inp_files) != len(self.target_files):
            print(f"Warning: Mismatched dataset. Inp: {len(self.inp_files)}, Target: {len(self.target_files)}. Truncating to {min_len}.")
        
        self.inp_files = self.inp_files[:min_len]
        self.target_files = self.target_files[:min_len]

    def _load_image_with_cache(self, image_path: str):
        """Load image with caching for frequently accessed small datasets"""
        if self.use_cache and image_path in self.image_cache:
            return self.image_cache[image_path]
            
        try:
            img = Image.open(image_path).convert('RGB')
            if self.use_cache:
                self.image_cache[image_path] = img
            return img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return Image.new('RGB', (256, 256))

    def _apply_optimized_augmentation(self, corrupt: torch.Tensor, clear: torch.Tensor):
        """Apply augmentation that maintains input-target consistency.
        
        FIX: Removed T.rotate and T.affine - these use bilinear interpolation
        which fills borders with zeros, creating artificial "corruption" regions
        that don't match the target and poison the training signal.
        Only lossless geometric transforms are safe for paired data.
        """
        # Random 90-degree rotation (lossless, no interpolation artifacts)
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            corrupt = torch.rot90(corrupt, k, dims=[1, 2])
            clear = torch.rot90(clear, k, dims=[1, 2])

        # Random horizontal flip
        if random.random() < 0.5:
            corrupt = T.hflip(corrupt)
            clear = T.hflip(clear)

        # Random vertical flip
        if random.random() < 0.5:
            corrupt = T.vflip(corrupt)
            clear = T.vflip(clear)

        return corrupt, clear

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx: int):
        idx = idx % len(self.inp_files)
        
        try:
            corrupt_img = self._load_image_with_cache(self.inp_files[idx])
            clear_img = self._load_image_with_cache(self.target_files[idx])
            
            w, h = corrupt_img.size

            if self.patch_size:
                if w < self.patch_size or h < self.patch_size:
                     corrupt = T.to_tensor(corrupt_img)
                     clear = T.to_tensor(clear_img)
                     corrupt = pad_image_needed(corrupt, (self.patch_size, self.patch_size))
                     clear = pad_image_needed(clear, (self.patch_size, self.patch_size))
                     
                     # ── FIX: manual random crop instead of T.RandomCrop.get_params ──
                     # T is torchvision.transforms.functional which has no RandomCrop class
                     _, ph, pw = corrupt.shape
                     i = random.randint(0, ph - self.patch_size)
                     j = random.randint(0, pw - self.patch_size)
                     corrupt = T.crop(corrupt, i, j, self.patch_size, self.patch_size)
                     clear = T.crop(clear, i, j, self.patch_size, self.patch_size)
                else:
                     i = random.randint(0, h - self.patch_size)
                     j = random.randint(0, w - self.patch_size)
                     corrupt_patch = corrupt_img.crop((j, i, j + self.patch_size, i + self.patch_size))
                     clear_patch = clear_img.crop((j, i, j + self.patch_size, i + self.patch_size))
                     corrupt = T.to_tensor(corrupt_patch)
                     clear = T.to_tensor(clear_patch)
            else:
                 corrupt = T.to_tensor(corrupt_img)
                 clear = T.to_tensor(clear_img)

            corrupt, clear = self._apply_optimized_augmentation(corrupt, clear)
            
            train_name = os.path.basename(self.inp_files[idx])
            h, w = corrupt.shape[1], corrupt.shape[2]
            
            return corrupt, clear, train_name, h, w
            
        except Exception as e:
            print(f"Error in __getitem__ at idx {idx}: {e}")
            ps = self.patch_size or 256
            dummy = torch.zeros((3, ps, ps))
            return dummy, dummy, "error.png", ps, ps


class TestDataset(Dataset):
    """Standard test dataset"""
    def __init__(self, data_path, data_path_test, task_name, data_type, length=None, resolution=None):
        super().__init__()
        self.task_name, self.data_type = task_name, data_type
        self.resolution = resolution

        p_inp = os.path.join(data_path_test, 'input')
        if not os.path.exists(p_inp):
             p_inp = os.path.join(data_path_test, 'inp')
             
        p_target = os.path.join(data_path_test, 'target')
        if not os.path.exists(p_target):
             p_target = os.path.join(data_path_test, 'gt')

        self.corrupted_images_test = sorted(glob.glob(f'{p_inp}/*'))
        self.target_images_test = sorted(glob.glob(f'{p_target}/*'))

        self.num = len(self.corrupted_images_test)
        self.sample_num = length if data_type == 'train' else self.num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        path_inp = self.corrupted_images_test[idx % self.num]
        path_target = self.target_images_test[idx % self.num]
        
        name = os.path.basename(path_inp)

        img_inp = Image.open(path_inp).convert('RGB')
        img_target = Image.open(path_target).convert('RGB')
        
        if self.resolution is not None:
            img_inp = img_inp.resize((self.resolution, self.resolution))
            img_target = img_target.resize((self.resolution, self.resolution))
        
        corrupt = T.to_tensor(img_inp)
        target = T.to_tensor(img_target)
        
        h, w = corrupt.shape[1:]
        return corrupt, target, name, h, w