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

Aggregate results (run locally):

```bash
python aggregate_results.py \
  -e "paper-2025-06_inpainting" \
  --compute_wis \
  --create_ensemble \
  --plot_results
```
