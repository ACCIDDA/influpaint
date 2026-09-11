# Quick Start

## Training

Edit experiment name in `train.run`:

```bash
sbatch train.run
```

## Inpainting

Generate mega-array for all scenarios:

```bash
python generate_inpaint_jobs.py \
  -e "paper-2025-06" \
  --scenarios "0-31" \
  --start_date "2022-10-12" \
  --end_date "2023-05-15"
```

This creates:
- Job list: `inpaint_jobs_paper-2025-06_all_scenarios.txt`
- SLURM script: `inpaint_array_paper-2025-06_all_scenarios.run`

Submit all scenarios:

```bash
sbatch inpaint_array_paper-2025-06_all_scenarios.run
```

## Evaluation

Run locally from the repository root after copying the forecast outputs:

```bash
python prepare_dataset_for_scoringutils.py
Rscript score_with_scoringutils.R results/combined_forecast_truth_data.csv results/scoringutils_scores.csv
python plot_evaluation_results.py
```

The preparation script is configured for the saved `paper-2025-07-22` job
manifest and the local FluSight data. See the [evaluation workflow](../workflows/evaluation.md)
for required inputs and outputs.
