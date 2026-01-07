# Architecture Overview

## Dual ID System

When running `sbatch --array=23 inpaint.run`, there are two different IDs:

1. **SLURM Array Task ID (23)**: Which scenario to run inpainting for
   - Scenario specification (unet, dataset, transform, etc.)
   - Same as used in training

2. **MLflow Run ID from Training**: Which trained model to load
   - Specific MLflow run that trained scenario 23
   - Format: `abc123def456` (MLflow UUID)
   - Found by searching for runs where `scenario_id = 23`

The `get_mlflow_run_id.py` script maps: Scenario ID → MLflow Run ID

## Map-Reduce Architecture

### Map Phase (SLURM cluster)
- Atomic jobs: Each runs 1 scenario + 1 date + 1 config
- Massive parallelization: 1,920 jobs run simultaneously
- Results: Individual forecasts stored in MLflow + filesystem

### Reduce Phase (Local computer)
- `aggregate_results.py` reads all MLflow results
- Computes WIS scores across scenarios/dates/configs
- Creates ensemble forecasts, summary tables, plots
- Generates final analysis ready for papers

## Key Components

### Training
- `train.py`: Single scenario training script
- `train.run`: SLURM script for training arrays
- Manual experiment naming

### Inpainting
- `inpaint.py`: Atomic inpainting script (single scenario + single date + single config)
- `inpaint.run`: SLURM template for manual jobs
- `generate_inpaint_jobs.py`: Creates parallel job arrays across dates/configs
- `get_mlflow_run_id.py`: Maps scenario ID to MLflow run ID
- Automatic model loading from MLflow

### Core Files
- `epiframework.py`: Dataclass-based scenario management
- `influpaint/batch/`: Batch processing utilities
- `influpaint/models/`: DDPM and inpainting modules
- `influpaint/datasets/`: Data loading and preprocessing
- `influpaint/utils/`: Helper functions
