# InfluPaint

## Abstract

Forecasting infectious disease incidence can provide important information to guide public health planning, yet is difficult because epidemic dynamics are complex. Current mechanistic and statistical approaches often struggle to capture multimodal uncertainty or emergent trends. Influpaint adapts denoising diffusion probabilistic models to epidemic forecasting. By encoding influenza seasons as spatiotemporal images in which pixel intensity represents incidence, Influpaint learns a rich distribution of disease dynamics from a hybrid dataset of surveillance and simulated trajectories. Forecasting is formulated as a conditional generation (inpainting) task from partial observations. We show that Influpaint generates realistic, diverse epidemic trajectories and achieves forecast accuracy that is competitive with leading ensemble methods in retrospective evaluation. In real-time evaluation during the 2023–2025 U.S. CDC FluSight challenges, performance improved substantially across seasons, with highly accurate but somewhat overconfident projections in 2024–2025. The best performance was achieved with a training dataset containing 30% surveillance and 70% simulated trajectories. These results show that diffusion models can capture important spatiotemporal structure in influenza dynamics and provide a flexible framework for probabilistic infectious disease forecasting.

[Read the paper on arXiv: *Generative diffusion models for spatiotemporal influenza forecasting*](https://arxiv.org/abs/2604.24913){ .md-button .md-button--primary }

Joseph Lemaitre and Justin Lessler

## Forecast influenza hospitalizations

[![Paper Figure 2: four-week influenza hospitalization forecasts for two seasons](assets/paper/figure-2-forecasts.png)](assets/paper/figure-2-forecasts.png)

**Figure 2 — Forecasts from observed history.** InfluPaint conditions on the observed part of a season to generate 512 possible trajectories for future hospitalizations. Colored fans show forecast uncertainty and colored lines show medians for North Carolina, New York, Texas, and Florida across the 2023–2024 and 2024–2025 seasons. Black lines show observed hospitalizations, dotted lines show the FluSight ensemble, and dashed vertical lines mark the last observed week for each forecast. These retrospective forecasts use finalized observations up to each forecast date.

## Adapt to different observation patterns without retraining

[![Paper Figure 4: reconstructions with missing states, missing weeks, and checkerboard observation masks](assets/paper/figure-4-reconstruction.png)](assets/paper/figure-4-reconstruction.png)

**Figure 4 — One trained model, many reconstruction tasks.** InfluPaint can adapt to different patterns of missing data by changing the observation mask at sampling time, without retraining the model. The same model reconstructs missing states, fills midseason gaps, infers early-season dynamics from later observations, and handles checkerboard patterns of missing weeks and locations. Black curves show observed hospitalizations; colored fans and lines show predictive quantiles and medians. The insets identify observed entries in green and hidden entries in red. This flexibility lets the model work with partial spatial coverage and interrupted time series.

## Learn seasons as images, then fill in what is missing

[![Paper Figure 5: encoding epidemic seasons as images, learning to denoise, and conditioning generation with an observation mask](assets/paper/figure-5-methods.png)](assets/paper/figure-5-methods.png)

**Figure 5 — How InfluPaint works.** **a.** An influenza season becomes an image whose axes represent weeks and locations and whose pixel intensity represents incidence. **b.** A diffusion model learns to reverse the gradual addition of noise, allowing it to generate new, plausible seasons. **c.** Inpainting combines observed values and a mask with the generation process to infer the missing parts of a season. The paper's forecasting implementation uses CoPaint to condition these generated trajectories on the available observations.

## Follow the workflow

The walkthrough follows the paper from source data to publication figures. Each numbered step explains what you will use and produce, and identifies files in the Zenodo reproduction archive that let you skip ahead.

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
