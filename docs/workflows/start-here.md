# Start here

Follow InfluPaint from source data to publication figures. Each step explains what you will use and produce, and links to the saved files for that stage.

!!! tip "Skip any step with the reproduction archive"

    **You can skip any step.** Use the saved inputs or outputs in the Zenodo reproduction archive and continue from the stage that interests you. Every chapter begins with a tip like this one naming the exact archive paths to use. You do not need to gather data, retrain models, or generate forecasts to explore the archived results or reproduce the paper figures.

## Contents

| Step | You will use | You will produce |
| --- | --- | --- |
| [1. Gather source datasets](compiling-data-sources.md) | FluView, FluSurv, and simulated trajectories | One standardized source table |
| [2. Create training data](build-training-datasets.md) | Source table and mixing rules | Four complete season-image datasets |
| [3. Train candidate models](training.md) | Training images and diffusion configurations | Model checkpoints and training losses |
| [4. Prepare forecast jobs](forecast-jobs.md) | Finished training runs, dates, and CoPaint settings | A forecast manifest and Slurm array |
| [5. Generate forecasts with inpainting](inpainting.md) | A checkpoint and observed history | Conditional trajectories and forecast quantiles |
| [6. Score forecasts and select the formulation](evaluation.md) | Forecast quantiles and observed truth | Scores and a model leaderboard |
| [7. Reconstruct missing observations](mask-experiments.md) | The selected model and observation masks | Reconstructed seasons |
| [8. Reproduce the paper figures](paper-figures.md) | Archived samples, observations, and analysis tables | The paper's eight data figures |

## Before you start

Follow [Installation](../getting-started/installation.md) for the research environment. All commands and repository paths in this walkthrough are relative to the **research repository root**, the directory containing `influpaint/`, `main_training/`, and `mkdocs.yml`.

Place the reproduction archive at `influpaint-paper/influpaint_paper_reproduction_data/`. Its `README.md` describes the archived files and the figure-generation command. The paper's exact training inputs are the July 17, 2025 NetCDF files; newer datasets are different training realizations. [Steps 1 and 2](compiling-data-sources.md) also tell the notebooks' story through captioned plots from saved data.

The paper's cluster launchers, saved scenario definitions, and job manifests live in `main_training/`; `main_training/runs.txt` collects the commands. Reusable model and batch code lives in `influpaint/`.

To reproduce figures immediately, go to [step 8](paper-figures.md). To understand the model itself, start with [step 1](compiling-data-sources.md) and follow the next-page links.
