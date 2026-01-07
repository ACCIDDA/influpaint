# InfluPaint

Inpainting denoising diffusion probabilistic models for infectious disease (influenza) forecasting.

**Authors:** Joseph Lemaitre, Justin Lessler
**Affiliation:** The University of North Carolina at Chapel Hill

## Overview

InfluPaint uses Denoising Diffusion Probabilistic Models (DDPM) to generate epidemic forecasts by treating influenza epidemic curves as images. The model generates synthetic epidemic trajectories and conditions forecasts on observed data using inpainting techniques.

## Key Features

- **DDPM-based forecasting**: Generative modeling approach for epidemic forecasting
- **Inpainting**: Condition forecasts on observed data using CoPaint algorithm
- **Map-Reduce architecture**: Parallel training and inpainting on HPC clusters
- **MLflow integration**: Experiment tracking and model management
- **Dual model loading**: MLflow-first with filesystem fallback

## Quick Links

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quick-start.md)
- [Architecture Overview](architecture/overview.md)
- [Training Workflow](workflows/training.md)
- [Inpainting Workflow](workflows/inpainting.md)

## Research Context

This is a research project focused on correctness and simplicity over production features. Manual control is preferred over complex automation.
