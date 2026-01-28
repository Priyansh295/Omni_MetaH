# Blind_Omni_Wav_Net

Blind_Omni_Wav_Net is a PyTorch-based image inpainting and restoration system with a transformer-style encoder-decoder backbone and dynamic convolution (ODConv2d). It includes metaheuristic optimization (GA, PSO, DE, BO) for architecture and loss weighting, progressive training, and robust validation with PSNR/SSIM.

## Overview

- Purpose: Restore occluded or corrupted regions using a learned inpainting model.
- Core: `Inpainting` network with multi-head attention, gated feed-forward, multi-scale feature aggregation.
- Optimization: Integrated GA/PSO/DE/BO to search architecture, hyperparameters, and loss weights.
- Validation: Periodic evaluation every `100` iterations in `main_py.py`, saving the best model on metric improvements.

## Architecture Diagram

![Model Architecture](architecture.jpg)

## System Architecture

- `model_directional_query_od.py`
  - `ODConv2d`: Dynamic convolution with attention across channel/filter/spatial/kernel dimensions.
  - Blocks: `MDTA` (attention), `GDFN` (feed-forward), `TransformerBlock` composed of MDTA+GDFN.
  - Pyramids: `DownSample` and `UpSample` for multi-scale processing.
  - `Inpainting`: U-shaped encoder-decoder using the above components; outputs restored image.

- `main_py.py` (memory-optimized training)
  - Training: Progressive learning stages using varying `patch_size` and `batch_size` across milestones.
  - Losses: L1, perceptual (VGG-based), SSIM, edge Sobel; weighted and combined.
  - Validation: `save_loop` runs every `100` iterations; records PSNR/SSIM, saves CSV, persists best weights.
  - Early stopping: Threshold-based (PSNR/SSIM) and patience-based with `min_delta`.
  - Metaheuristics: `OptimizedMetaheuristicManager` orchestrates GA, PSO, DE, and BO with reduced budgets for stability.

- `main.py` (simpler training/optimization script)
  - Similar training loop with `val_every` configurable via CLI (default parsed as `500`).
  - Fallback frequency uses `100` in-code but CLI parsing provides the value.

- `utils_train.py`
  - `parse_args` and `OptimizedConfig`: CLI parsing and default configuration values (data paths, iteration counts, milestones, early stopping).
  - Dataset: `OptimizedTrainDataset` (and `TrainDataset`) with augmentation and length control for progressive stages.
  - Metrics: `rgb_to_y`, `psnr`, `ssim` consistent with luminance evaluation.
  - Loss: `OptimizedVGGPerceptualLoss` wrapper for perceptual features.
  - Monitoring: `PerformanceMonitor`, `MemoryMonitor` utility classes.

- `test.py`
  - Inference: `optimized_test_loop` computes PSNR/SSIM on test sets; periodic memory cleanup every 10 batches.
  - Reporting: Saves restored images and comparison visuals.

## Data Flow & Execution Flow

1. CLI parse
   - `main_py.py` uses a small argument pre-parser for `--optimize` or `--use-defaults`, then defers to `utils_train.parse_args` for full configuration.

2. Model & optimizer
   - Build `Inpainting` with architecture from args (or optimized defaults).
   - Optimizer: `AdamW` with `CosineAnnealingLR` over `args.num_iter`.

3. Progressive training
   - On each milestone, rebuild `TrainDataset` with new `patch_size` and `batch_size`.
   - Training steps compute multi-loss (L1, perceptual, SSIM, edge), backpropagate in FP32, gradient clip, and step.

4. Validation & saving
   - Every `100` iterations in `main_py.py`, `save_loop` runs:
     - Calls `test_loop`, averages PSNR/SSIM over test loader.
     - Appends to `results` and writes `{save_path}/{data_name}.csv`.
     - If both PSNR and SSIM improve over best, writes best metrics to `{save_path}/{data_name}.txt` and saves `{save_path}/{data_name}.pth`.
   - Early stopping checks thresholds and patience.

5. Metaheuristic optimization (optional)
   - `run_enhanced_metaheuristic_optimization` integrates GA/PSO/DE/BO and returns final parameters.
   - Training then runs with the optimized parameters.

## Validation Process

- Frequency: Fixed at `100` iterations in `main_py.py`.
- Metrics: PSNR and SSIM computed on Y (luminance) channel.
- Best model: Saved only when both PSNR and SSIM improve vs. previous best.
- Early stopping:
  - Threshold mode: stop if PSNR ≥ `target_psnr` and SSIM ≥ `target_ssim`.
  - Patience mode: stop after `early_stop_patience` validations without improvement by `min_delta`.

## Dependencies

- Python `>=3.8`
- PyTorch `>=1.12`
- torchvision
- kornia
- numpy, pandas, tqdm, seaborn, matplotlib
- pillow, opencv-python
- optuna, nevergrad

Install all via:

```sh
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install core packages:

```sh
pip install torch torchvision kornia numpy pandas tqdm seaborn matplotlib pillow opencv-python optuna nevergrad
```

## Installation

```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/Blind_Omni_Wav_Net.git
cd Blind_Omni_Wav_Net
pip install -r requirements.txt
```

## Usage

### Train (memory-optimized)

```sh
python main_py.py --use-defaults --data_path ./datasets/celeb --data_path_test ./datasets/celeb --save_path ./results
```

### Train (with metaheuristic optimization)

```sh
python main_py.py --optimize --data_path ./datasets/celeb --data_path_test ./datasets/celeb --save_path ./results
```

### Inference / Testing

```sh
python test.py --data_path_test ./datasets/celeb --save_path ./results
```

### Key Arguments

- `--data_path`: Training images directory
- `--data_path_test`: Test images directory
- `--save_path`: Output directory for metrics, images, and weights
- `--num_iter`: Total training iterations
- `--milestone`: Iteration boundaries for progressive training
- `--patch_size`: Patch sizes per stage
- `--batch_size`: Batch sizes per stage
- `--early_stop`, `--target_psnr`, `--target_ssim`, `--early_stop_patience`, `--min_delta`: Validation and stopping controls
- `--val_every` (main.py): Validation cadence (default parsed as `500`). Note: `main_py.py` uses a fixed cadence of `100`.
- `--model_file`: Resume from a saved weight file for evaluation only

## Project Structure

- `main_py.py`: Memory-optimized training, validation every 100 iterations, integrated metaheuristics
- `main.py`: Alternate training/optimization script with configurable `val_every`
- `model_directional_query_od.py`: Full model architecture and building blocks
- `utils_train.py`: Arguments, datasets, metrics, loss, and monitoring utilities
- `datasets/`: Dataset utilities and mask generation logic
- `test.py`: Inference pipeline, metric reporting, and visualizations
- `results/`: Saved metrics (`.csv`), best stats (`.txt`), and weights (`.pth`)

## Troubleshooting

- CUDA OOM: Reduce `batch_size` or `patch_size` in `utils_train.py` args; consider CPU training.
- Mixed precision issues: `main_py.py` disables AMP for stability; leave as FP32.
- Missing data: Ensure `--data_path` and `--data_path_test` exist and contain images.
- Validation cadence: In `main_py.py` it is fixed at `100`; in `main.py` pass `--val_every 100` to match.
- Slow I/O: Preprocess datasets and use SSD; set `num_workers=0` if encountering Windows `DataLoader` issues.

## Notes for Developers

- Progressive training rebuilds the dataset at each milestone; expect cache misses on first load.
- Metrics are evaluated on the Y channel to reflect perceptual luminance quality.
- Best model saving requires simultaneous improvement in PSNR and SSIM.
- Metaheuristic budgets in `main_py.py` are reduced (e.g., GA/PSO/DE set to 10) for stability.

## License

MIT License. See `LICENSE` for details.
