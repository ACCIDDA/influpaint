# Architecture overview

InfluPaint represents seasons as week-by-location images. Dataset preparation makes complete frames; a U-Net diffusion model learns to denoise them; CoPaint conditions generated seasons on observed entries. The [paper walkthrough](../index.md) explains the full sequence.

| Component | Source |
| --- | --- |
| Source readers and frame construction | `influpaint/datasets/read_datasources.py`, `mixer.py` |
| Array loading and preprocessing | `influpaint/datasets/loaders.py`, `transforms.py` |
| Calendar, location ordering, and observation masks | `influpaint/utils/season_axis.py`, `ground_truth.py` |
| Diffusion and U-Net | `influpaint/models/ddpm.py`, `nn_blocks.py` |
| Paper conditioning sampler | `CoPaint4influpaint/` (`O_DDIMSampler`) |
| Scenario and configuration definitions | `influpaint/batch/scenarios.py`, `config.py` |
| Reusable training and forecasting entry points | `influpaint/batch/training.py`, `inpainting.py` |
| Paper launchers and saved job manifests | `main_training/` |
| Scoring | `evaluation/prepare_dataset_for_scoringutils.py`, `evaluation/score_with_scoringutils.R` |
| Evaluation and figures | `evaluation/plot_evaluation_results.py`, `evaluation/benchmark_plotting.py`, `evaluation/choose_best_model.py`, `paper_figures/` |

A training scenario ID describes a formulation. An MLflow run ID identifies a particular training execution and its weights. A forecast Slurm array index selects one manifest row, containing the scenario, training run ID, reference date, and CoPaint configuration. The preserved paper manifest has 1,305 rows; each defines an independent forecast job. See [step 4](../workflows/forecast-jobs.md).
