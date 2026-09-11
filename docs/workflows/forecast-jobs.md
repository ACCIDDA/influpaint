# 4. Prepare forecast jobs

You will use **finished MLflow training runs, reference dates, and CoPaint settings** to create **a job manifest and a Slurm array**, with one forecast configuration per row.

!!! tip "Skip this step with the reproduction archive"

    To skip job generation and sampling entirely, use `influpaint-paper/influpaint_paper_reproduction_data/forecasts/retrospective/` and continue to step 6 or step 8. To rerun the historical jobs, the preserved manifest and launcher live in `main_training/`, where the paper run files are kept: `main_training/paper_runs_2025-07-22/inpaint_jobs_paper-2025-07-22.txt` and `main_training/inpaint_array_paper-2025-07-22.run`. They require the original MLflow store and checkpoints.


## Expand models into forecast tasks

The preserved manifest contains **15 models × 29 dates × 3 CoPaint configurations = 1,305 rows**. Dates occur every other Saturday: 15 dates from October 14, 2023 to April 27, 2024, and 14 dates from November 23, 2024 to May 24, 2025. The configurations are `celebahq_noTTJ5`, `celebahq_try3`, and `celebahq`.

A row has `job_id,scenario_id,run_id,season,date,config`. The Slurm array index selects a row; `scenario_id` selects the formulation; `run_id` selects its trained weights. These are different identifiers.

## Generate jobs after new training

```bash
python -m influpaint.batch.generate_inpainting_jobs \
  -e paper-2025-07-22_training \
  --configs celebahq_noTTJ5,celebahq_try3,celebahq \
  --output_dir main_training/generated
```

The generator queries the active MLflow store and writes `inpaint_jobs_paper-2025-07-22.txt` and `inpaint_array_paper-2025-07-22.run` under `main_training/generated/`. New training produces new run IDs; keep these generated jobs separate from the preserved historical manifest.

Submit the generated launcher in step 5. The historical launcher in `main_training/` was reconstructed from the current generator template because the original generated `.run` file was not available locally.

[Previous: 3. Train candidate models](training.md) · [Next: 5. Generate forecasts](inpainting.md)
