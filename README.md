# Omni MetaH: Blind Inpainting with Omni-dimensional Gated Attention and Wavelet Queries

![Architecture](image.png)

## 📌 Overview

**Omni MetaH** (also referred to as *Blind Omni Wav Net*) is a state-of-the-art framework for **blind image inpainting**—the task of restoring corrupted images without a known mask. 

This project introduces a novel architecture that combines:
- **Wavelet-Guided Selective State Space Models (WG-SSM):** Leveraging the O(N) efficiency of Mamba/SSMs while injecting frequency-domain guidance (Wavelets) to modulate state transitions.
- **Omni-dimensional Gated Attention:** Using `ODConv` (Omni-dimensional Convolution) for dynamic, multi-dimensional feature attention.
- **Wavelet Queries:** Preserving high-frequency details akin to Transformers but with far greater efficiency.
- **Metaheuristic Hyperparameter Optimization:** Utilizing Genetic Algorithms (GA), Particle Swarm Optimization (PSO), and Differential Evolution (DE) to automatically find the optimal architecture and training hyperparameters.

## ✨ Key Features

- **Hybrid Backbone:** Support for both **Mamba (SSM)** and **Transformer** backbones.
- **Frequency-Aware:** Explicitly models Low (LL) and High (LH, HL, HH) frequency components using Discrete Wavelet Transforms (DWT).
- **Efficient:** Mamba-based blocks offer linear complexity $O(N)$ vs Transformer's $O(N^2)$.
- **Automated Tuning:** Built-in metaheuristic search (`optuna`, `nevergrad`) for architecture search (NAS) and hyperparameter tuning.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Priyansh295/Omni_MetaH.git
   cd Omni_MetaH
   ```

2. **Install dependencies:**
   It is recommended to use a Conda environment.
   ```bash
   pip install -r requirements.txt
   ```

   **Note on Mamba:**
   To enable the optimized Mamba backbone, you need `mamba-ssm`. This requires CUDA.
   ```bash
   pip install mamba-ssm
   ```
   *If `mamba-ssm` is not found, the model falls back to a pure PyTorch implementation (`PureSSM` or `WaveletGuidedSSM`), which is slower but compatible with all GPUs/CPUs.*

## 🚀 Usage

### 1. Inference (Inpainting)

To run the model on a folder of corrupted images:

```bash
python inference_inpaint.py \
  --model_path checkpoints/best_model.pth \
  --input_dir datasets/test/input \
  --output_dir results/output \
  --device cuda
```

**Common Arguments:**
- `--model_path`: Path to the `.pth` checkpoint.
- `--input_dir`: Folder containing images to process.
- `--image_size`: (Optional) Resize images before processing (e.g., `--image_size 256`).
- `--config`: (Optional) Path to a JSON config file if your model uses non-standard hyperparameters.

### 2. Training

To train the model from scratch or fine-tune:

```bash
python main_py.py \
  --data_path ./datasets/train \
  --data_path_test ./datasets/test \
  --batch_size 8 \
  --num_iter 100000 \
  --val_every 500
```

**Metaheuristic Optimization:**
The training script (`main_py.py`) can automatically optimize hyperparameters. The `run_improved_optimization` function triggers GA/PSO/DE search strategies to find the best:
- `num_blocks`, `num_heads`, `channels`
- `learning_rate`, `loss_weights`
- `expansion_factor`

### 3. Configuration

Key model parameters (can be adjusted in `main_py.py` or passed as args):

- `--num_blocks`: Number of blocks per stage (e.g., `2 4 4 6`).
- `--channels`: Channel dimensions (e.g., `24 48 96 192`).
- `--expansion_factor`: Expansion for GDFN/FeedForward network.

## 📂 Project Structure

- `model_directional_query_od.py`: Core model definition (Inpainting, MambaBlock, WaveletGuidedSSM, MDTA).
- `odconv.py`: Implementation of Omni-dimensional Convolution.
- `main_py.py`: Main training script with metaheuristic optimization loops.
- `inference_inpaint.py`: Script for running inference on images.
- `utils_train.py`: Training utilities, dataset loading, and loss functions.
- `frequency_loss.py`: Frequency-aware loss implementation.
- `requirements.txt`: Project dependencies.

## 📝 Citation

If you use this code or model in your research, please cite:

```bibtex
@inproceedings{OmniMetaH2026,
  title={Blind Inpainting via Omni-dimensional Gated Attention and Wavelet Queries},
  author={Priyansh et al.},
  year={2026}
}
```
