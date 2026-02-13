import glob
import random
import os

_DATA_SPLIT_CACHE = {}

def get_or_create_data_split(data_path, val_split=0.2, seed=42):
    """
    Get or create a reproducible train/val split for the given data path.
    This ensures the SAME split is used across all optimization trials.
    """
    cache_key = (data_path, val_split, seed)
    if cache_key in _DATA_SPLIT_CACHE:
        return _DATA_SPLIT_CACHE[cache_key]

    inp_files = sorted(
        glob.glob(f'{data_path}/inp/*.png') +
        glob.glob(f'{data_path}/inp/*.jpg') +
        glob.glob(f'{data_path}/inp/*.jpeg') +
        glob.glob(f'{data_path}/inp/*.PNG') +
        glob.glob(f'{data_path}/inp/*.JPG') +
        glob.glob(f'{data_path}/input/*.png') +
        glob.glob(f'{data_path}/input/*.jpg') +
        glob.glob(f'{data_path}/input/*.jpeg') +
        glob.glob(f'{data_path}/input/*.PNG') +
        glob.glob(f'{data_path}/input/*.JPG')
    )
    target_files = sorted(
        glob.glob(f'{data_path}/target/*.png') +
        glob.glob(f'{data_path}/target/*.jpg') +
        glob.glob(f'{data_path}/target/*.jpeg') +
        glob.glob(f'{data_path}/target/*.PNG') +
        glob.glob(f'{data_path}/target/*.JPG') +
        glob.glob(f'{data_path}/gt/*.png') +
        glob.glob(f'{data_path}/gt/*.jpg') +
        glob.glob(f'{data_path}/gt/*.jpeg') +
        glob.glob(f'{data_path}/gt/*.PNG') +
        glob.glob(f'{data_path}/gt/*.JPG')
    )

    if len(inp_files) == 0 or len(target_files) == 0:
        return None, None, None, None

    min_len = min(len(inp_files), len(target_files))
    inp_files = inp_files[:min_len]
    target_files = target_files[:min_len]

    rng = random.Random(seed)
    indices = list(range(min_len))
    rng.shuffle(indices)
    split_idx = int(min_len * (1 - val_split))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_inp = [inp_files[i] for i in train_indices]
    train_target = [target_files[i] for i in train_indices]
    val_inp = [inp_files[i] for i in val_indices]
    val_target = [target_files[i] for i in val_indices]

    _DATA_SPLIT_CACHE[cache_key] = (train_inp, train_target, val_inp, val_target)
    print(f"[Data Split] Created split: {len(train_inp)} train, {len(val_inp)} val (seed={seed})")

    return train_inp, train_target, val_inp, val_target
