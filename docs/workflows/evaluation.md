# Evaluation Workflow

## Aggregate Results

Run on local computer after inpainting completes:

```bash
python aggregate_results.py \
  -e "paper-2025-06_inpainting" \
  --compute_wis \
  --create_ensemble \
  --plot_results
```

## Operations

- **Read MLflow results**: Collect all individual forecasts
- **Compute WIS scores**: Weighted Interval Score across scenarios/dates/configs
- **Create ensemble forecasts**: Combine predictions
- **Generate summary tables**: Statistical summaries
- **Create plots**: Visualizations for papers

## WIS Scoring

Uses Adrian Lison's interval scoring library to compute Weighted Interval Scores for forecast evaluation.

## Output

Results ready for:
- Paper figures
- Performance analysis
- Model comparison
