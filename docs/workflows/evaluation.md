# 6. Score forecasts and select the formulation

You will use **candidate and benchmark forecast quantiles, observed truth, and R's `scoringutils`** to obtain **probabilistic scores and model rankings**.

!!! tip "Skip this step with the reproduction archive"

    Use `influpaint-paper/influpaint_paper_reproduction_data/analysis/scoringutils_scores.csv` for the saved per-forecast scores, and `influpaint-paper/influpaint_paper_reproduction_data/analysis/leaderboard_full.csv` for the saved rankings. The selected-model forecast archive does not contain all candidate forecast CSVs required to rescore the complete comparison. To remake the paper comparison figures directly, use step 8.


## Join forecasts to their observations

`prepare_dataset_for_scoringutils.py` reads the dates from `main_training/paper_runs_2025-07-22/inpaint_jobs_paper-2025-07-22.txt`, candidate CSVs under `from_longleaf/influpaint_res/`, and official forecasts/truth under the two season-specific `Flusight/` hub directories. Its `Config` class specifies the input locations; update it deliberately for a fresh experiment.

```bash
python prepare_dataset_for_scoringutils.py
Rscript score_with_scoringutils.R results/combined_forecast_truth_data.csv results/scoringutils_scores.csv
python plot_evaluation_results.py
```

Run locally from the repository root after copying the cluster results. R requires `scoringutils` and `dplyr`. The first command aligns forecast distributions with observed values. The second computes weighted interval score (WIS) and other scores. The third uses `benchmark_plotting.py` to apply inclusion/completeness filters, calculate relative WIS against FluSight-baseline, and write `results/leaderboards/leaderboard_full.csv`.

## Interpret the model comparison

Lower WIS means more accurate probabilistic forecasts. The saved leaderboard contains 36 candidate formulations: 12 included training scenarios × 3 conditioning settings. The plotting pipeline excludes i808 and applies per-season submission-completeness filters to benchmarks.

The paper's i868 + `celebahq_noTTJ5` ranks second in both combined absolute and relative WIS in the saved leaderboard. Selection considers both rankings; the formulation is not the individual winner of either score. `choose_best_model.py` highlights it explicitly and supplies the supplementary comparison plots.

The local full forecast directory lacks the expected i932/i996 outputs. Those models have large logged training losses, but the available artifacts do not establish their final scheduler outcome. The saved scores and leaderboard document the comparison that was actually available.

[Previous: 5. Generate forecasts](inpainting.md) · [Next: 7. Reconstruct missing observations](mask-experiments.md)
