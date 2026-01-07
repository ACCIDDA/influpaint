# Influpaint Paper Figures Module

This package contains modularized code for generating publication-quality figures for the Influpaint paper. The original monolithic `influpaint_paperfigures.py` (1857 lines) has been refactored into logical, maintainable modules.

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
├── peak_analysis.py              # Peak distribution analysis
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

### peak_analysis.py
Functions for analyzing peak timing and size distributions:
- `plot_peak_distributions_comparison()` - Compare generated vs historical peaks
- `plot_peak_distributions_by_location()` - Location-specific peak analysis
- `plot_peak_distributions_by_metric()` - Metric-grouped swarmplots (timing/size)

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

## Usage

### Generate All Figures

```python
from paper_figures.main import main
main()
```

### Generate Final Paneled Figures

```bash
python -m paper_figures.final_figures
```

This generates the final multi-panel figures for the paper:
- **Figure 1**: Unconditional generation (excluding NC) + correlation analysis
- **Figure 2**: CSV forecasts for 2023-2024 and 2024-2025 seasons (4 states × 2 seasons)
- **Figure 3**: NPY forecasts for two seasons with A/B labels (excluding NC)
- **Figure 4**: Mask experiments with multiple panels (CA/KY/MD + NC/IL)

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

### Use Individual Plotting Functions

```python
from paper_figures import peak_analysis

fig = peak_analysis.plot_peak_distributions_comparison(
    inv_samples=samples,
    season_axis=season_axis,
    save_path="my_peaks.png",
    prominence_threshold=50.0
)
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

Same as the original script:
- numpy
- pandas
- matplotlib
- seaborn
- scipy (for peak detection)
- plotly (optional, for interactive 3D figures)
- influpaint package and its dependencies

## Migration from Original Script

To switch from the old `influpaint_paperfigures.py` to the new modular version:

```python
# Old way:
# python influpaint_paperfigures.py

# New way:
python -m paper_figures.main

# Or in Python:
from paper_figures.main import main
main()
```

All figures are generated in the same `figures/` directory as before, with identical filenames.
