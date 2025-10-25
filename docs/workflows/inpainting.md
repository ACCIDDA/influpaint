# Inpainting Workflow

## Mega-Array Generation

Generate job array for all scenarios:

```bash
python generate_inpaint_jobs.py \
  -e "paper-2025-06" \
  --scenarios "0-31" \
  --start_date "2022-10-12" \
  --end_date "2023-05-15"
```

Creates:
- `inpaint_jobs_paper-2025-06_all_scenarios.txt`: Job list (scenario×date×config combinations)
- `inpaint_array_paper-2025-06_all_scenarios.run`: SLURM submission script

## Submission

Submit all scenarios at once:

```bash
sbatch inpaint_array_paper-2025-06_all_scenarios.run
```

## Job Execution

Each job:
1. Reads scenario ID from array
2. Uses `get_mlflow_run_id.py` to find trained model
3. Loads model from MLflow
4. Runs inpainting for specific date and config
5. Saves results to MLflow and filesystem

## Why get_mlflow_run_id.py is Needed

When running `sbatch --array=23 inpaint.run`:
- **Array ID 23**: Which scenario to run inpainting for
- **MLflow Run ID**: Which specific trained model to load for scenario 23

Example:
- Scenario 23 trained in experiment "paper-2025-06_training"
- Multiple training runs might exist for scenario 23
- `get_mlflow_run_id.py` finds the latest successful run ID: `abc123def456`
- `inpaint.py` loads that exact model using MLflow run ID

Without this script, you'd manually search MLflow for "which run trained scenario 23?" - tedious and error-prone.
