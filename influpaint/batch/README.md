# Batch Processing

Parallel training and inpainting for InfluPaint experiments.

## Files

- `config.py` - Model, dataset, transform configurations
- `scenarios.py` - Scenario generation and object creation
- `training.py` - Train diffusion models
- `inpainting.py` - Generate forecasts
- `generate_inpainting_jobs.py` - Generate SLURM array jobs
- `mlflow_utils.py` - MLflow utilities

## Usage

### Experiment Naming Convention
Use different experiment names for training vs inpainting:
- **Training**: `"paper-2025-06_training"`  
- **Inpainting**: `"paper-2025-06_inpainting"`

This keeps related experiments grouped while separating training metrics from forecasting metrics in MLflow.

### Training
```bash
python -m influpaint.batch.training -s 5 -e "paper-2025-06_training"
sbatch train.run
```

### Inpainting
```bash
python -m influpaint.batch.inpainting -s 5 -r "mlflow_run_id" -e "paper-2025-06_inpainting" --forecast_date "2022-11-14" --config_name "celebahq_try1"
sbatch inpaint.run
```

### Array Jobs
```bash
python -m influpaint.batch.jobs -e "experiment_name" --scenarios "0-31" --start_date "2022-10-12" --end_date "2023-05-15"
sbatch inpaint_array_*.run
```

## Configuration

### Add Model
```python
# config.py
AVAILABLE_MODELS = ["MyUnet200", "MyUnet500", "NewModel"]

def model_library():
    unet_spec = {
        "NewModel": ddpm.DDPM(...)
    }
```

### Add Dataset
```python
# config.py
AVAILABLE_DATASETS = ["R1Fv", "R1", "NewDataset"]

def dataset_library():
    dataset_spec = {
        "NewDataset": training_datasets.FluDataset.from_source(...)
    }
```

### Modify Parameters
```python
# config.py
def copaint_config_library(timesteps):
    config_lib = {
        "celebahq_try1": config.Config(default_config_dict={
            "jump_length": 20,  # Change value
            "jump_n_sample": 4,  # Change value
        })
    }
```

## Scenarios

Training scenarios combine:
- Model: MyUnet200, MyUnet500
- Dataset: R1Fv, R1, SURV_ONLY, HYBRID_70S_30M, HYBRID_30S_70M, MOD_ONLY
- Transform: Lins, Sqrt
- Enrichment: No, PoisPadScale, PoisPadScaleSmall, Pois

Inpainting scenarios use:
- Config: celebahq_try1, celebahq_noTT, celebahq_noTT2, celebahq_try3, celebahq

List all scenarios:
```bash
python -m influpaint.batch.scenarios
```

## Workflow

1. Train models: `sbatch train.run`
2. Generate jobs: `python jobs.py`
3. Run inpainting: `sbatch inpaint_array_*.run`
4. Check results in MLflow and output directory
5. Score forecast CSVs using the [evaluation workflow](../../docs/workflows/evaluation.md).

### Evaluation

Run from the repository root after forecasts have been copied locally:

```bash
python prepare_dataset_for_scoringutils.py
Rscript score_with_scoringutils.R results/combined_forecast_truth_data.csv results/scoringutils_scores.csv
python plot_evaluation_results.py
```

The preparation script uses the saved paper job manifest, forecast CSVs, and
FluSight observations. The R script computes scores with `scoringutils`; the
plotting script writes evaluation plots and the model leaderboard.

## Output

- Training: Model checkpoints, MLflow artifacts
- Inpainting: Forecast files, plots, MLflow metrics
- Jobs: SLURM scripts, job lists
