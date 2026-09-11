# Start here

The goal of this walkthrough is to build an influenza forecasting model and understand the choices that shape its predictions. InfluPaint first learns what complete influenza seasons look like. It then uses the observed part of a season to generate possible continuations, producing an ensemble of trajectories across states and weeks.

You will start with historical surveillance and simulated epidemics, put them on a common calendar, and build training datasets with different mixtures of the two. After training candidate models, you will forecast the same reference dates with each candidate, compare their probabilistic accuracy, and use a selected model to fill missing observations or make an operational forecast.

## Reproducibility archive

!!! tip "Paper reproducibility"

    **Zenodo DOI: [10.5281/zenodo.22699980](https://doi.org/10.5281/zenodo.22699980).**

    To reproduce the paper, use the Zenodo archive. It contains the source table, four training datasets, the selected model checkpoint, saved forecasts, and score and loss tables. The [Zenodo archive README](../reproducibility/README.md) documents the exact reproduction inputs and their provenance. Place it at `influpaint-paper/influpaint_paper_reproduction_data/` to use the archive paths in this walkthrough. To plot the figures in the paper without rerunning training or forecasts, go to [Reproduce the paper figures](paper-figures.md). To start at another step, use the saved files listed near the top of that chapter.

## Follow the data through the workflow

Steps 1–2 create the training material: a standardized source table becomes a collection of complete season images. Steps 3–6 turn those images into a forecasting formulation: train the model, define the forecasting experiment, sample trajectories, and score the resulting quantiles. Step 7 explores other observation patterns, and step 8 brings the results together in figures.

Once you have a trained model, [step 9](operational-forecasts.md) is the recurring operational workflow: update surveillance data, choose the forecast date, inspect the observation mask, sample, and review the export. You can reuse the checkpoint each week.

## Contents

| Step | You will use | You will produce |
| --- | --- | --- |
| [0. Install the environment](../getting-started/installation.md) | Research and CoPaint repositories, Conda | A working research environment |
| [1. Gather source datasets](compiling-data-sources.md) | FluView, FluSurv, and simulated trajectories | One standardized source table |
| [2. Create training data](build-training-datasets.md) | Source table and mixing rules | Four complete season-image datasets |
| [3. Train candidate models](training.md) | Training images and diffusion configurations | Model checkpoints and training losses |
| [4. Prepare forecast jobs](forecast-jobs.md) | Finished training runs, dates, and CoPaint settings | A forecast manifest and Slurm array |
| [5. Generate forecasts with inpainting](inpainting.md) | A checkpoint and observed history | Conditional trajectories and forecast quantiles |
| [6. Score forecasts and select the formulation](evaluation.md) | Forecast quantiles and observed truth | Scores and a model leaderboard |
| [7. Reconstruct missing observations](mask-experiments.md) | The selected model and observation masks | Reconstructed seasons |
| [8. Reproduce the paper figures](paper-figures.md) | Archived samples, observations, and analysis tables | The paper's eight data figures |
| [9. Generate operational forecasts](operational-forecasts.md) | A trained checkpoint and the latest reported hospitalizations | FluSight CSVs, forecast PDFs, and sampled trajectories |

## Before you start

Complete [step 0: install the environment](../getting-started/installation.md). All commands and repository paths in this walkthrough are relative to the **research repository root**, the directory containing `influpaint/`, `main_training/`, and `mkdocs.yml`. Run the data notebooks from that directory too, so their relative input paths resolve correctly.

The two data-preparation notebooks live in `dataset_creation/`. Model and dataset choices are defined in `influpaint/batch/config.py`, and `influpaint/batch/scenarios.py` combines them into training scenarios. Cluster launchers live in `main_training/`; the scripts that train models and generate forecasts live in `influpaint/batch/`.

The examples use the paper's configuration. For your own experiment, set the dataset paths in step 3, use a new experiment name for training and forecasting, and choose the forecast dates in step 4.

[Previous: 0. Install the environment](../getting-started/installation.md) · [Next: 1. Gather source datasets](compiling-data-sources.md)
