# Training architecture

A denoising diffusion model learns a distribution over season images. Training adds Gaussian noise at a diffusion time and teaches a U-Net to predict it. Sampling reverses this process from noise to a generated season.

The selected paper model uses 500 diffusion steps with a cosine schedule and a ResNet U-Net with channel multipliers `(1,2,2,4)`. Each input has one channel and a 64 × 64 week/location grid. The model uses the July 17 `30S70M` dataset, square-root preprocessing, and no additional training enrichment.

Dataset preparation mixes surveillance and simulated sources and completes frames before training. Training-time preprocessing is defined by the selected configuration in `influpaint/batch/config.py`; the weights and their corresponding dataset must be retained together.

See [step 2](../workflows/build-training-datasets.md) for the image construction and [step 3](../workflows/training.md) for the paper candidates, Slurm command, and archived checkpoint.
