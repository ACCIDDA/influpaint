# Influpaint Paper Figures Module

This package generates the Influpaint paper and supplementary figures. Run `python -m paper_figures.final_figures` from the repository root.

## Module Structure

```
paper_figures/
├── __init__.py                   # Package initialization
├── README.md                     # This file
├── config.py                     # Configuration and constants
├── helpers.py                    # Utility/helper functions
├── data_utils.py                 # Data preprocessing utilities
├── unconditional_figures.py      # Unconditional generation figures
├── correlation_analysis.py       # Spatial correlation analysis
├── analyze_flusight_dropbox_tables.py # Archived operational leaderboard analysis
├── csv_forecasts.py              # CSV forecast quantile fans
├── npy_forecasts.py              # NPY full-horizon forecasts
├── mask_experiments.py           # Mask experiment visualizations
├── main.py                       # Main orchestration script
└── final_figures.py              # Final paneled figures for paper
```

## Module Descriptions

### config.py
Contains all configuration constants, paths, and global settings:
- State name mappings
- Model configuration (BEST_MODEL_ID, BEST_CONFIG)
- File paths (UNCOND_SAMPLES_PATH, INPAINTING_BASE, FIG_DIR)
- Season x-limits for consistent plotting
- Matplotlib global settings

### helpers.py
Utility functions used across multiple modules:
- `state_to_code()` - Convert state names/abbrevs to location codes
- `load_unconditional_samples()` - Load unconditional samples from .npy
- `list_influpaint_csvs()` - Find CSV forecast files
- `list_inpainting_dirs()` - Find NPY forecast directories
- `forecast_week_saturdays()` - Get Saturday dates for forecast weeks
- `format_date_axis()` - Apply consistent date formatting

### unconditional_figures.py
Functions for generating unconditional (baseline) figures:
- `plot_unconditional_states_quantiles_and_trajs()` - State-level quantiles and trajectories
- `fig_unconditional_3d_heat_ridges()` - 3D matplotlib visualization
- `fig_unconditional_3d_heat_ridges_plotly()` - Interactive Plotly 3D viz
- `plot_unconditional_states_with_history()` - Overlay with historical data
- `plot_unconditional_states_with_history_alt()` - Alternative history overlay
- `generate_unconditional_us_grid()` - US state grid
- `generate_unconditional_trajs_and_heatmap()` - Trajectory and heatmap combo
- `generate_mean_heatmap()` - Mean heatmap only

### csv_forecasts.py
Functions for plotting CSV (4-week hubverse) forecast quantile fans:
- `load_truth_for_season()` - Load ground truth data
- `plot_csv_quantile_fans_for_season()` - Single-season forecast fans
- `plot_csv_quantile_fans_multiseasons()` - Multi-season forecast fans

### npy_forecasts.py
Functions for plotting NPY (full-horizon) forecasts:
- `plot_npy_multi_date_two_seasons()` - Multiple states across two seasons
- `plot_npy_two_panel_national()` - National-level two-panel figure

### mask_experiments.py
Functions for visualizing mask experiment results:
- `recreate_mask()` - Recreate mask patterns from experiment names
- `plot_mask_experiments()` - Generate figures for all mask experiments

### main.py
Main orchestration script that:
- Sets up the environment and loads data
- Calls all figure generation functions in organized groups
- Handles errors gracefully
- Provides progress feedback

### final_figures.py
Generates final paneled figures for paper publication by composing existing plotting functions:
- `figure1_unconditional_with_correlation()` - Unconditional generation with correlation analysis
- `figure2_csv_forecasts_two_seasons()` - CSV forecasts for two seasons in 4x2 layout
- `figure3_npy_forecasts_two_seasons()` - NPY forecasts with A/B panel labels
- `figure4_mask_experiments()` - Multi-panel mask experiments figure
- `add_panel_label()` - Utility to add A, B, C labels to panels

### publication_style.py
Controls the final dimensions and typography of main Figures 1-4 in one place.
`PAPER_WIDTH_IN = 6.1` matches the manuscript's LaTeX text width;
`FIGURE1_LATEX_WIDTH_FRACTION = 0.75` matches Figure 1's LaTeX inclusion width.
Its text and decorations are enlarged before export to compensate for that reduction.
`PAPER_FONT_PT = 7` controls the base font size. Figures 2-4 use a wider,
shorter 6.8-by-3.8-inch canvas. `WIDE_FONT_SCALE` sets their font sizes
independently of canvas dimensions; main text prints at approximately 7 pt when
LaTeX fits these figures to 6.1 inches. Exports use
`PAPER_DPI = 600` and tight cropping with 0.02-inch padding to remove outer
whitespace. Y-axis labels are aligned across rows. Panel letters use the base font size plus 2 pt;
`FORECAST_DATE_FONT_PT = 5` controls the complete set of forecast-date annotations.
X- and y-tick labels use the base font size minus 0.5 pt. The full hospitalization
axis label uses 6.5 pt in all four figures, before export scaling. Tick lengths,
spines, gridlines, and line widths scale with the typography for consistent
decoration sizes in the manuscript.
Tick density, panel spacing, inset styling, and the historical-season
legend are handled by `save_paper_figure()`.

## Usage

### Generate the Paper and Supplement Figures

```bash
python -m paper_figures.final_figures
```

This generates the Figure 1 correlation summary and the images included in the manuscript and supplement:
- **Figure 1**: Unconditional generation (excluding NC) + correlation analysis
- **Figure 2**: CSV forecasts for 2023-2024 and 2024-2025 seasons (4 states × 2 seasons)
- **Figure 3**: NPY forecasts for two seasons with A/B labels (excluding NC)
- **Figure 4**: Mask experiments with multiple panels (CA/FL/MD + NC/IL)
- **Supplementary Figures 1–3**: Ablation effects, training losses, and loss versus WIS, using the plotting function in `evaluation/choose_best_model.py`
- **Supplementary Figure 4**: Submitted operational forecasts

To use the archived reproduction inputs with the same script:

```bash
python -m paper_figures.final_figures --data-root influpaint-paper/influpaint_paper_reproduction_data
```

This writes those eight data plots, the correlation JSON, and the FluSight leaderboard CSV and plot into `influpaint-paper/influpaint_paper_reproduction_data/regenerated_paper_figures/`. Set `--output-dir` to choose another destination. The default seed is 0 (`--seed` overrides it). The exploratory figures in `paper_figures.main` are not part of this command.

All unconditional and correlation figures use the complete
ensemble of 512 samples. Figure 1's insets use
zero-based indices `(0, 255, 510)` in the full i868 sample array.
Generated and time-permuted correlations use only the 51 locations defined by
`SeasonAxis` (the 50 states and DC), excluding spatial padding. Observed
correlations use the same locations for the 2023–2024 and 2024–2025 seasons.

### Generate Specific Figure Types

```python
from influpaint.utils import SeasonAxis
from paper_figures import config, helpers, unconditional_figures

# Setup
season_axis = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
uncond = helpers.load_unconditional_samples(config.UNCOND_SAMPLES_PATH)

# Generate specific figures
unconditional_figures.generate_unconditional_us_grid(uncond, season_axis, config.FIG_DIR, config._MODEL_NUM)
```

## Bugs Fixed

The refactoring process identified and fixed several bugs present in the original code:

1. **Function naming inconsistency**: Changed `_state_to_code()` to `state_to_code()` and `_format_date_axis()` to `format_date_axis()` for proper module imports
2. **Import issues**: Fixed references to use proper module imports instead of relying on global scope
3. **Better error handling**: Added try-except blocks around major figure generation sections
4. **Documentation**: Added comprehensive docstrings to all functions

## Advantages of Modular Structure

1. **Maintainability**: Each module has a clear, focused purpose
2. **Reusability**: Functions can be imported and used independently
3. **Testing**: Easier to write unit tests for individual modules
4. **Readability**: ~200-300 lines per module instead of 1857 in one file
5. **Collaboration**: Multiple developers can work on different modules
6. **Debugging**: Issues are easier to locate and fix

## Dependencies

Required libraries:
- numpy
- pandas
- matplotlib
- seaborn
- scipy (shared Influpaint utilities)
- plotly (optional, for interactive 3D figures)
- influpaint package and its dependencies

## Exploratory Figures

```bash
python -m paper_figures.main
```

This writes additional unconditional, correlation, forecast, and mask plots to
`figures/`. The paper figure entry point writes its outputs to
`influpaint-paper/figures/generated/` by default.

## Reproduce the Zenodo archive

Run from the Influpaint repository root in the Influpaint Python environment:

```bash
python -m paper_figures.final_figures --data-root "/absolute/path/to/this_zenodo_folder"
```

For the local archive, the data root is `influpaint-paper/influpaint_paper_reproduction_data`.
The command writes the eight paper/supplementary data plots plus
`868_figure1_correlation_summary.json`, `flusight_dropbox_analysis.csv`, and
`2024-2025_wis_pairgrid.png` to `DATA_ROOT/regenerated_paper_figures/`.
Use `--output-dir` to change that destination. The default random seed is `0`.
The correlation JSON contains the count, mean, and median of each distribution
plotted in Figure 1b (time-permuted null, generated seasons, and observed seasons).

The leaderboard input tables live in `DATA_ROOT/analysis/FlusightScores/`.
To run only their analysis:

```bash
python -m paper_figures.analyze_flusight_dropbox_tables --data-root "/absolute/path/to/this_zenodo_folder"
```

This excludes FluSight-prefixed models in all three seasons and applies a 70%
submission threshold in 2023–2024 and 2024–2025. It summarizes archived scores
without rescoring forecasts. The archive README documents the input snapshots,
filtering, and all generated outputs.
