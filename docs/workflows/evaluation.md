# Evaluation Workflow

Run the evaluation locally from the repository root after copying the cluster
forecast outputs. Scores are computed from forecast quantiles and observed
hospital admissions using R's `scoringutils` package.

## Inputs

`prepare_dataset_for_scoringutils.py` selects inputs through its `Config` class:

- `paper_runs_2025-07-22/inpaint_jobs_paper-2025-07-22.txt`: reference dates for the two evaluation seasons.
- `from_longleaf/influpaint_res/`: candidate forecast CSVs exported by the inpainting jobs.
- `Flusight/2023-2024/FluSight-forecast-hub-official/` and `Flusight/2024-2025/FluSight-forecast-hub-official/`: official model forecasts and target hospital admissions.

The paper evaluation uses the `wk inc flu hosp` target and CSV horizons
`0, 1, 2, 3`. Interpret predicted weeks using `target_end_date`.
Update the preparation configuration when evaluating a different experiment.
The R environment must have `scoringutils` and `dplyr` installed.

## Prepare, Score, and Plot

```bash
python prepare_dataset_for_scoringutils.py
Rscript score_with_scoringutils.R results/combined_forecast_truth_data.csv results/scoringutils_scores.csv
python plot_evaluation_results.py
```

The preparation script merges forecast quantiles with observed truth. The R
script scores the resulting forecast distributions. The plotting script applies
model inclusion filters, calculates relative WIS against `FluSight-baseline`,
and produces plots and rankings using `benchmark_plotting.py`.

## Outputs

- `results/combined_forecast_truth_data.csv`: forecast quantiles joined to observations.
- `results/scoringutils_scores.csv`: WIS and other scores from `scoringutils`.
- `results/leaderboards/leaderboard_full.csv`: model scores and ranks by season, metric, and aggregation.
- `results/simple_plots/`: evaluation plots.

The paper figure command, `python -m paper_figures.final_figures`, reads the
leaderboard together with `mlflow_losses.csv` and `mlflow_loss_timeseries.csv`
to generate Supplementary Figures 1–3 through `choose_best_model.py`.
