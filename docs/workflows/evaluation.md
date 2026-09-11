# 6. Score forecasts and select the formulation

You will use **candidate and benchmark forecast quantiles, observed truth, and R's `scoringutils`** to obtain **probabilistic scores and model rankings**.

!!! tip "Skip this step with the reproduction archive"

    Use `influpaint-paper/influpaint_paper_reproduction_data/analysis/scoringutils_scores.csv` for the saved per-forecast scores, and `influpaint-paper/influpaint_paper_reproduction_data/analysis/leaderboard_full.csv` for the saved rankings. The selected-model forecast archive does not contain all candidate forecast CSVs required to rescore the complete comparison. To remake the paper comparison figures directly, use step 8.


## Join forecasts to their observations

`evaluation/prepare_dataset_for_scoringutils.py` reads the dates from `main_training/paper_runs_2025-07-22/inpaint_jobs_paper-2025-07-22.txt`, candidate CSVs under `from_longleaf/influpaint_res/`, and official forecasts/truth under the two season-specific `Flusight/` hub directories. Its `Config` class specifies the input locations; update it deliberately for a fresh experiment.

```bash
python -m evaluation.prepare_dataset_for_scoringutils
Rscript evaluation/score_with_scoringutils.R results/combined_forecast_truth_data.csv results/scoringutils_scores.csv
python -m evaluation.plot_evaluation_results
```

Run locally from the repository root after copying the cluster results. R requires `scoringutils` and `dplyr`. The first command aligns forecast distributions with observed values. The second computes weighted interval score (WIS) and other scores. The third uses `evaluation/benchmark_plotting.py` to apply inclusion/completeness filters, calculate relative WIS against FluSight-baseline, and write `results/leaderboards/leaderboard_full.csv`.

## Interpret the model comparison

Lower WIS means more accurate probabilistic forecasts. The saved leaderboard contains 36 candidate formulations: 12 included training scenarios × 3 conditioning settings. The plotting pipeline excludes i808 and applies per-season submission-completeness filters to benchmarks.

The paper's i868 + `celebahq_noTTJ5` ranks second in both combined absolute and relative WIS in the saved leaderboard. Selection considers both rankings; the formulation is not the individual winner of either score. `evaluation/choose_best_model.py` highlights it explicitly and supplies the supplementary comparison plots.

The local full forecast directory lacks the expected i932/i996 outputs. Those models have large logged training losses, but the available artifacts do not establish their final scheduler outcome. The saved scores and leaderboard document the comparison that was actually available.

## See how model choices affect forecast scores

[![Supplementary Figure 1: changes in forecast performance when varying one model setting](../assets/scoring/sup_forest_effect.png)](../assets/scoring/sup_forest_effect.png)

**Supplementary Figure 1 — Compare one setting at a time.** Each row changes one setting relative to the baseline formulation: the diffusion process, U-Net, training-data mixture, transform, enrichment, or inpainting configuration. The horizontal axis shows percentage improvement in WIS over the baseline. Positive values indicate better forecast performance; negative values indicate worse performance. The dashed zero line marks the baseline. This comparison helps identify which modeling choices matter for forecast accuracy.

## Compare training loss with forecasting performance

[![Supplementary Figure 3: training loss versus relative and absolute weighted interval scores](../assets/scoring/sup_lossVSwis.png)](../assets/scoring/sup_lossVSwis.png)

**Supplementary Figure 3 — Select on forecast performance as well as training loss.** The horizontal axis shows average training loss over the last 100 logged steps. The vertical axes show relative WIS (left) and absolute WIS (right), with lower scores indicating better forecasts. Colors distinguish training-data mixtures, and marker shapes distinguish inpainting settings. The selected `i868_noTTJ5` formulation is labeled in red. A low denoising loss alone does not identify the strongest forecasting formulation; the held-out forecast scores provide the comparison used for selection.

These are the existing figures included in the paper's supplement. The underlying saved inputs are `analysis/leaderboard_full.csv` and `analysis/mlflow_losses.csv` in the reproduction archive. [Step 8](paper-figures.md) explains how to regenerate the supplementary figures.

[Previous: 5. Generate forecasts](inpainting.md) · [Next: 7. Reconstruct missing observations](mask-experiments.md)
