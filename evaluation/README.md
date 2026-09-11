# Evaluation

Forecast scoring, model comparison, and analysis exports. Run from the **research repository root** with the InfluPaint environment active.

## Score and compare forecasts

```bash
python -m evaluation.prepare_dataset_for_scoringutils
Rscript evaluation/score_with_scoringutils.R model_candidate_evaluation/combined_forecast_truth_data.csv model_candidate_evaluation/scoringutils_scores.csv
python -m evaluation.plot_evaluation_results
```

The preparation configuration identifies the saved job manifest, candidate forecasts, official FluSight forecasts, and observations. Outputs remain in `model_candidate_evaluation/`. R requires `scoringutils` and `dplyr`.

The reproduction archive includes a complete `model_candidate_evaluation/` folder. Its README explains how to regenerate the diagnostics from the saved scores using `--csv-path`, `--save-dir`, and `--leaderboard-dir`, without rescoring the forecasts.

| File | Purpose |
| --- | --- |
| `prepare_dataset_for_scoringutils.py` | Join forecast quantiles and observed truth |
| `score_with_scoringutils.R` | Compute WIS and other probabilistic scores |
| `plot_evaluation_results.py` | Apply inclusion filters and produce plots and rankings |
| `benchmark_plotting.py` | Shared evaluation plotting and ranking utilities |
| `choose_best_model.py` | Compare formulations and generate supplementary calibration figures |
| `resave_flusight_csvs.py` | Re-export forecast quantiles from saved trajectory arrays |
| `extract_mlflow_losses.py` | Export training-loss records and sample artifacts from MLflow |

## Additional commands

```bash
python -m evaluation.choose_best_model
python -m evaluation.extract_mlflow_losses --experiment_name paper-2025-07-22_training
python -m evaluation.resave_flusight_csvs
```

These commands write analysis outputs or re-export saved results; review their configured input and output paths before running them. `paper_figures.final_figures` imports `evaluation.choose_best_model` when producing the supplementary plots.

See [Walkthrough: scoring](../docs/workflows/evaluation.md) for interpretation and the saved score/leaderboard files available in the Zenodo reproduction archive.
