# Batch processing

Reusable implementations for InfluPaint training and inpainting:

| Module | Purpose |
| --- | --- |
| `config.py`, `scenarios.py` | Define configurations and map numeric scenario IDs |
| `training.py` | Train one diffusion formulation |
| `inpainting.py` | Condition one model on one date with one CoPaint setting |
| `generate_inpainting_jobs.py` | Expand finished MLflow runs into forecast manifests and Slurm arrays |
| `mlflow_utils.py` | Model-loading and tracking utilities |

The paper-specific launchers and saved job definitions are in [`main_training/`](../../main_training/README.md). Run from the repository root:

```bash
sbatch main_training/train.run
```

After training completes, generate and submit forecast jobs:

```bash
python -m influpaint.batch.generate_inpainting_jobs -e paper-2025-07-22_training --configs celebahq_noTTJ5,celebahq_try3,celebahq --output_dir main_training/generated
sbatch main_training/generated/inpaint_array_paper-2025-07-22.run
```

New runs need their new MLflow run IDs. The historical manifest is preserved under `main_training/paper_runs_2025-07-22/`.

See [the walkthrough](../../docs/index.md) for source data, exact training files, archived checkpoints, scoring commands, and figure reproduction. Configuration ordering in `scenarios.py` and `config.py` determines numeric scenario IDs; preserve it for the paper.
