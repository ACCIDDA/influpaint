# Training Architecture

## Denoising Diffusion Probabilistic Models (DDPM)

DDPM consists of two transforms in diffusion time:

- **Forward**: Markov chain that gradually adds noise to the data until the signal is destroyed
- **Backward**: A trained neural network that denoises the image step by step

Training uses forward transform samples from the dataset. Sampling transforms random Gaussian noise with the backward transform.

## Neural Network Architecture

The neural network for the backward transform is a U-net with:

- Wide ResNet blocks
- Attention modules
- Group normalization
- Residual connections
- Sinusoidal time-step embedding

## Training Data

Data sources:

- Mechanistic influenza transmission models (US Flu Scenario Modeling Hub Round 1)
- Reported US influenza data (FluView, FluSurv)
- State-level data at various locations
- Dataset augmentation with random transforms

## Training Process

1. Edit experiment name in `train.run` (e.g., "paper-2025-06")
2. Submit training job: `sbatch train.run`
3. MLflow logs all metrics, parameters, and artifacts
4. Models saved with scenario ID for later retrieval
