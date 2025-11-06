# Blind Omni Wav Net

Deep learning framework for blind image inpainting and restoration with an integrated metaheuristic optimizer (GA, PSO, DE, BO) that tunes architecture and training hyperparameters. The model combines transformer-style attention, omni-dimensional convolutions, and a wavelet-based self-supervised pathway for robust completion.

![Architecture](architecture.jpg)

## Highlights

- Transformer-based inpainting with multi-scale U-Net-like hierarchy.
- Omni-Dimensional Convolution (`ODConv2d`) adapts kernels across channel/filter/spatial dimensions.
- Wavelet SSL path (`DWT/IDWT`) enables frequency-aware representation learning.
- Integrated metaheuristics: GA (architecture), PSO (loss weights), DE (expansion factor), BO (LR/batch size).
- Progressive training with multi-loss: L1, Perceptual (VGG16), SSIM, Edge (Sobel).
- Evaluation metrics and complexity analysis (PSNR/SSIM, THOP/PTFlops).

## Architecture Overview

- Core model: `Inpainting` in `model_directional_query_od.py`.
- Blocks
  - `MDTA`: multi-dilation transformer attention for long-range context.
  - `GDFN`: gated-dilation feed-forward network for efficient channel mixing.
  - `TransformerBlock`: residual attention + feed-forward fusion.
  - `DownSample`/`UpSample`: multi-scale feature pyramid with skip connections.
- Auxiliary modules
  - `ODConv2d` (in `odconv.py`): generates channel, filter, spatial, and kernel attention weights; dynamically composes convolution kernels.
  - Wavelet SSL (via `pytorch_wavelets`): `DWTForward`/`DWTInverse` for frequency-domain modeling.
  - Losses: `OptimizedVGGPerceptualLoss`, SSIM/PSNR (stable double precision), edge loss via `kornia.sobel`.

## Repository Structure

- Top-level
  - `main_py.py`: primary training + metaheuristics driver (robust, memory-optimized).
  - `main.py`: legacy training/optimization entry point (still functional).
  - `inference_inpaint.py`: batch inference with JSON-config loading.
  - `test.py`: inference and performance analysis (metrics, THOP/PTFlops).
  - `model_directional_query_od.py`: core `Inpainting` model and submodules.
  - `odconv.py`: omni-dimensional convolution and attention generator.
  - `utils_train.py`: args/config, dataset/augmentations, metrics, perceptual loss, performance monitors.
  - `mask_generator.py`, `x.py`: comprehensive mask/composite generators (basic/composite/enhanced/demo).
  - `datasets/`: dataset roots (e.g., `NewData`, `celeb`, `ffhq`, `paris`, `places`).
  - `results/`: checkpoints, logs, metaheuristic results, visualizations, hyperparam plots.

## Dependencies and Roles

- `torch`, `torchvision`: model, training, IO utilities.
- `numpy`, `Pillow`, `opencv-python` (via `cv2` used in `x.py`): image IO and processing.
- `scipy` (optional but recommended): Gaussian blur utilities and Voronoi ops used by mask/composite generators.
- `kornia`: edge filters (`sobel`) for edge loss.
- `pytorch_wavelets`: `DWT/IDWT` for frequency-aware SSL.
- `thop`, `ptflops`: complexity and FLOPs analysis.
- `pandas`, `tqdm`: reporting and progress bars.
- `matplotlib` (and optionally `seaborn`): result visualization; `seaborn` is imported in `main_py.py` (install as needed).
- `optuna`, `nevergrad`: metaheuristic optimization framework.

Install all with `pip install -r requirements.txt`. For GPU acceleration, install PyTorch with the correct CUDA build from the official instructions.

## Setup

- Requirements
  - Python 3.9+ (tested with modern PyTorch)
  - Optional CUDA GPU
  - OS: Windows and Linux supported
- Installation
  ```sh
  git clone https://github.com/YOUR_GITHUB_USERNAME/Blind_Omni_Wav_Net.git
  cd Blind_Omni_Wav_Net
  python -m venv .venv
  .venv\Scripts\activate  # Windows
  # Or: source .venv/bin/activate  # Linux/macOS
  pip install -r requirements.txt
  # Install PyTorch per CUDA/CPU from https://pytorch.org/get-started/locally/
  ```

## Dataset Preparation

- Expected layout (example: `NewData`)
  ```
  datasets/NewData/
    inp/      # corrupted/masked inputs
    target/   # ground-truth targets
  ```
- Other provided datasets: `celeb`, `ffhq`, `paris`, `places` use similar `input`/`target` directory naming.
- Mask generation (optional) via `x.py`:
  ```sh
  # Basic irregular masks
  python x.py --mode basic --input_dir datasets/NewData/inp --output_dir datasets/NewData/inp --mask_type irregular

  # Enhanced composite overlays (opaque), preserving faces
  python x.py --mode enhanced --input_dir datasets/NewData/inp --output_dir datasets/NewData/inp \
    --num_variations 2 --style high_coverage_splashes --preserve_face
  ```

## Usage

### Primary Training (memory-optimized defaults)

```sh
python main_py.py
```

- Optimization flow
  - `--optimize`: run GA+PSO+DE+BO to synthesize final params, then train.
  - `--use-defaults`: train with the original defaults without metaheuristics.
- Example
  ```sh
  # Run integrated metaheuristics then train
  python main_py.py --optimize

  # Use original defaults
  python main_py.py --use-defaults
  ```

### Legacy Training Driver

```sh
python main.py --optimize
# Or
python main.py
```

### Inference

```sh
python inference_inpaint.py --input_dir ./test --output_dir ./results/visualizations

# With a saved JSON config describing model params
python inference_inpaint.py --input_dir ./test --output_dir ./results/visualizations \
  --config_json ./results/metaheuristic_results/final_params.json
```

### Testing and Analysis

```sh
python test.py --compute_metrics --compute_thop --compute_ptflops
```

### Mask Generation (artistic/composite)

```sh
python x.py --mode composite --input_dir ./datasets/NewData/inp --output_dir ./datasets/NewData/inp \
  --style mixed_artistic --num_variations 1

python x.py --mode enhanced --input_dir ./datasets/NewData/inp --output_dir ./datasets/NewData/inp \
  --num_variations 2 --style high_coverage_splashes --preserve_face
```

### Organic Patch-Based Compositing (advanced)

Use `organic_patch_composite.py` to create highly realistic, organic patch masks that paste regions from a donor image into the base image, with soft, multi-scale blending. Outputs `input/` (composite) and `target/` (binary mask), plus a `donor/` note.

Pipelines:
- `fbm_poisson`: fractal noise (FBM) + domain warp → Poisson color harmonization.
- `voronoi_laplacian`: warped Voronoi regions → Laplacian pyramid blend.
- `flow_laplacian`: curl-noise streamlines (scribbles) → Laplacian blend.
- `vae_masks` (optional): tiny VAE samples mask shapes → Laplacian blend.

Examples:
```sh
# FBM + Poisson (recommended default)
python organic_patch_composite.py \
  --input_images_dir ./datasets/ffhq/input \
  --overlay_images_dir ./datasets/ffhq/input \
  --output_dir ./datasets/OrganicMaskOut \
  --pipeline fbm_poisson --per_image 4

# Voronoi + Laplacian
python organic_patch_composite.py --input_images_dir ./datasets/ffhq/input \
  --output_dir ./datasets/OrganicMaskOut --pipeline voronoi_laplacian

# Flow + Laplacian (scribble occlusions)
python organic_patch_composite.py --input_images_dir ./datasets/ffhq/input \
  --output_dir ./datasets/OrganicMaskOut --pipeline flow_laplacian

# VAE masks (optional; trains a tiny VAE quickly)
python organic_patch_composite.py --input_images_dir ./datasets/ffhq/input \
  --output_dir ./datasets/OrganicMaskOut --pipeline vae_masks --use_vae

# Force CPU (deterministic CPU path)
python organic_patch_composite.py --input_images_dir ./datasets/ffhq/input \
  --output_dir ./datasets/OrganicMaskOut --pipeline fbm_poisson --cpu_only
```

Colab tips:
- Enable GPU runtime; the script auto-detects CUDA and accelerates noise/blur + VAE.
- If `scipy` is missing: `pip install -q scipy`.
- Save to Drive paths for large runs.

## Configuration and CLI Notes

- Training args are defined in `utils_train.parse_args` and include:
  - Architecture: `num_blocks`, `num_heads`, `channels`, `num_refinement`, `expansion_factor`
  - Training: `num_iter`, `batch_size`, `lr`, `milestone`, `patch_size`
  - Paths: `data_path`, `data_path_test`, `save_path`
  - Augmentations and performance toggles
- Inference can load model configs from a JSON file via `--config_json`.
- Environment tuning (CUDA): set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.
- Reproducibility: seeds and directories initialized in `utils_train.init_args`.

## Metaheuristic Optimization

- Manager: `OptimizedMetaheuristicManager` in `main_py.py`.
- Algorithms
  - GA: selects `num_blocks`, `num_heads`, `channels`, `num_refinement` from compact, memory-safe sets.
  - PSO: tunes loss weights `(w_l1, w_percep, w_ssim, w_edge)`.
  - DE: tunes `expansion_factor`.
  - BO: tunes `lr` (and batch size in legacy flow).
- Budgets are reduced for stability on larger datasets (see constants in `main_py.py`).
- Results and plots are saved under `results/metaheuristic_results` and `results/hyperparam_plots`.

## Results and Outputs

- Checkpoints: `results/checkpoints` and `results/inpaint.txt`.
- Metaheuristic artifacts: `results/metaheuristic_results`.
- Logs: `results/logs`.
- Visualizations: `results/visualizations`.
- Hyperparameter plots: `results/hyperparam_plots/params_lr_vs_objective.png`.

## Troubleshooting

- Mixed precision is disabled by default (`USE_MIXED_PRECISION = False`) due to stability; training runs in FP32.
- If you encounter CUDA OOM, reduce `batch_size`, use smaller `num_blocks/channels`, or enable `PYTORCH_CUDA_ALLOC_CONF`.
- Ensure `seaborn` is installed if using `main_py.py` visualizations.
- `pytorch_wavelets` may require specific PyTorch/CUDA builds; consult upstream docs.

## Changelog

- 0.5.0
  - New advanced compositing script `organic_patch_composite.py` with pipelines: `fbm_poisson`, `voronoi_laplacian`, `flow_laplacian`, and optional `vae_masks`.
  - Blending utilities: soft alpha (`smoothstep`), Laplacian pyramid blending, Poisson color harmonization.
  - Dataset runner CLI for large-scale generation; saves `input/`, `target/`, and donor references.
  - GPU/CPU fallback using PyTorch for noise/blur operations; `--cpu_only` flag.
  - Updated `scribble_mask_flow.py`: added GPU acceleration and `collage_overlay` style with `--overlay_images_dir`.

- 0.4.1
  - Documentation: comprehensive README with architecture, setup, usage, dependencies, and changelog.
- 0.4.0
  - Integrated metaheuristics manager (`main_py.py`) with GA+PSO+DE+BO and memory-optimized training.
- 0.3.2
  - Inference pipeline improvements (`inference_inpaint.py`): robust IO, JSON config loading.
- 0.3.0
  - Enhanced mask generation (`x.py`): composite/enhanced modes, face preservation, opaque overlays.
- 0.2.0
  - ODConv2d integration (`odconv.py`) and transformer attention blocks; multi-loss training.
- 0.1.0
  - Initial training loop, dataset, and basic utilities.

## Contributing

- Pull requests welcome. Please include clear descriptions and focused changes.
- For larger features, open an issue to discuss design and integration.

## Citation

If you use this repository in academic work, please cite the project and relevant upstream papers (ODConv, transformer attention, wavelet SSL).

## License

No license file is included. Please add a `LICENSE` file (MIT recommended) to specify terms.
