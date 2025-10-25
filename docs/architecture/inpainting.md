# Inpainting Architecture

## CoPaint Algorithm

InfluPaint uses the CoPaint algorithm (replacing earlier REpaint) to condition forecasts on observed data. Inpainting alters reverse diffusion iterations by sampling unmasked regions using ground-truth.

## Atomic Inpainting Design

Each inpainting job is independent and atomic:

- **Atomic**: Single scenario + single date + single config per job
- **Parallel**: 20 dates × 3 configs = 60 parallel jobs
- **Independent**: Each job can run on different nodes/GPUs
- **Fault-tolerant**: Failed jobs don't affect others

**Example**: For scenario 5 with 20 dates and 3 configs:
- Old way: 1 job × 60 sequential runs = slow
- New way: 60 parallel jobs = fast

## Model Loading

Inpainting uses dual model loading:

1. **MLflow-first**: Load trained model from MLflow using run ID
2. **Filesystem fallback**: Load from filesystem if MLflow unavailable

The `get_mlflow_run_id.py` utility maps scenario IDs to MLflow run IDs.

## Mega-Array Submission

All scenarios submitted in one job array:

```bash
python generate_inpaint_jobs.py -e "paper-2025-06" --scenarios "0-31"
sbatch inpaint_array_paper-2025-06_all_scenarios.run
```

Example: 32 scenarios × 20 dates × 3 configs = 1,920 parallel jobs
