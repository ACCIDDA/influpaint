# How InfluPaint works

InfluPaint learns what an influenza season can look like, then uses observed hospitalizations to condition new trajectories. It represents a season as an image: rows are weeks, columns are locations, and pixel values represent incidence. A diffusion model learns to remove noise from these images; CoPaint guides generation using the entries we have observed.

This walkthrough follows the paper from source data to publication figures. Each numbered step has its own page, explains what you will use and produce, and identifies saved files that let you skip ahead.

## Contents

| Step | You will use | You will produce |
| --- | --- | --- |
| [1. Gather source datasets](workflows/compiling-data-sources.md) | FluView, FluSurv, and simulated trajectories | One standardized source table |
| [2. Create training data](workflows/build-training-datasets.md) | Source table and mixing rules | Four complete season-image datasets |
| [3. Train candidate models](workflows/training.md) | Training images and diffusion configurations | Model checkpoints and training losses |
| [4. Prepare forecast jobs](workflows/forecast-jobs.md) | Finished training runs, dates, and CoPaint settings | A forecast manifest and Slurm array |
| [5. Generate forecasts with inpainting](workflows/inpainting.md) | A checkpoint and observed history | Conditional trajectories and forecast quantiles |
| [6. Score forecasts and select the formulation](workflows/evaluation.md) | Forecast quantiles and observed truth | Scores and a model leaderboard |
| [7. Reconstruct missing observations](workflows/mask-experiments.md) | The selected model and observation masks | Reconstructed seasons |
| [8. Reproduce the paper figures](workflows/paper-figures.md) | Archived samples, observations, and analysis tables | The paper's eight data figures |

## Before you start

Follow [Installation](getting-started/installation.md) for the research environment. All commands and repository paths in this walkthrough are relative to the **research repository root**, the directory containing `influpaint/`, `main_training/`, and `mkdocs.yml`.

Place the reproduction archive at `influpaint-paper/influpaint_paper_reproduction_data/`. Its `README.md` inventories every shortcut used here. The paper's exact training inputs are the July 17, 2025 NetCDF files; newer datasets are different training realizations. [Steps 1 and 2](workflows/compiling-data-sources.md) also tell the notebooks' story through captioned plots from saved data.

The paper's cluster launchers, saved scenario definitions, and job manifests live in `main_training/`; `main_training/runs.txt` collects the commands. Reusable model and batch code lives in `influpaint/`.

To reproduce figures immediately, go to [step 8](workflows/paper-figures.md). To understand the model itself, start with [step 1](workflows/compiling-data-sources.md) and follow the next-page links.
