# Blind_Omni_Wav_Net

A PyTorch-based deep learning framework for image inpainting and restoration, featuring metaheuristic optimization (GA, PSO, DE, BO) for neural architecture and loss function search.

## Features

- **Image Inpainting** using a custom neural network architecture
- **Metaheuristic Optimization**: Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Differential Evolution (DE), and Bayesian Optimization (BO) for hyperparameters and architecture
- **Progressive Training** and multi-loss (L1, perceptual, SSIM, edge)
- **Evaluation Metrics**: PSNR, SSIM

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/YOUR_GITHUB_USERNAME/Blind_Omni_Wav_Net.git
   cd Blind_Omni_Wav_Net
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model with default settings:

```sh
python main.py --data_path ./datasets/celeb --data_path_test ./datasets/celeb --save_path ./results
```

### Metaheuristic Optimization

To run architecture and loss optimization:

```sh
python main.py --optimize
```

### Arguments

- `--data_path`: Path to training data
- `--data_path_test`: Path to test data
- `--save_path`: Directory to save results
- `--optimize`: Run metaheuristic optimization

## Project Structure

- `main.py`: Main training and optimization script
- `model_directional_query_od.py`: Model architecture
- `utils_train.py`: Utilities for training, metrics, and dataset
- `datasets/`: Place your training and test images here
- `results/`: Output and logs

## Optimization Details

- **GA**: Optimizes architecture (blocks, heads, channels, refinement)
- **PSO**: Optimizes loss weights (L1, perceptual, SSIM, edge)
- **DE**: Optimizes expansion factor
- **BO**: Optimizes learning rate and batch size

## Citing

If you use this code, please cite the repository and relevant papers.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)
