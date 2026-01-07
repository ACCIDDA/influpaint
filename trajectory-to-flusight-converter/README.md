# Trajectory to FluSight Forecast Converter

A Python module for converting trajectory-based forecasting model outputs (such as those from influpaint) into the standardized FluSight forecast submission format required by the CDC FluSight Forecast Hub.

## Overview

Modern influenza forecasting models often generate trajectory samples (also called ensemble members or Monte Carlo samples) representing possible future scenarios. The FluSight Forecast Hub requires submissions in specific formats including:
- **Quantile forecasts** for weekly hospital admissions
- **Probability mass functions (PMF)** for rate change categories
- **PMF** for peak week timing
- **Quantile forecasts** for peak intensity
- **Sample trajectories** (optional but recommended)

This converter bridges the gap between raw trajectory outputs and submission-ready forecasts.

## Features

✓ **Complete FluSight Coverage**: Generates all required forecast targets
✓ **Flexible Input**: Dictionary-based API accepts trajectories for any subset of targets
✓ **Rate Change Logic**: Implements horizon-specific thresholds and count-based rules
✓ **Peak Detection**: Automatic identification of peak week and intensity from trajectories
✓ **Multi-Location Support**: Process multiple geographic locations in batch
✓ **Sample Format**: Convert connected trajectories to FluSight sample format
✓ **Validation-Ready**: Outputs match FluSight schema for automated validation

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- numpy >= 1.20.0
- pandas >= 1.3.0

## Quick Start

```python
from trajectory_converter import convert_trajectories_to_flusight
import numpy as np

# Example: 100 trajectory samples for each forecast horizon
n_samples = 100
trajectories_dict = {
    -1: np.random.poisson(1000, n_samples),  # Lookback week
     0: np.random.poisson(1050, n_samples),  # Nowcast
     1: np.random.poisson(1100, n_samples),  # 1-week ahead
     2: np.random.poisson(1150, n_samples),  # 2-week ahead
     3: np.random.poisson(1200, n_samples),  # 3-week ahead
}

# Add seasonal trajectories for peak forecasts (n_weeks in season)
trajectories_dict['season'] = np.random.poisson(1100, (n_samples, 26))

# Convert to FluSight format
forecast_df = convert_trajectories_to_flusight(
    trajectories_dict=trajectories_dict,
    reference_date="2025-11-29",  # Saturday after submission
    location="US",
    population=331000000  # Required for rate change calculations
)

# Save to CSV for submission
forecast_df.to_csv("2025-11-29-TeamName-ModelName.csv", index=False)
```

## Key Functions

### `convert_trajectories_to_flusight()`

**Main conversion function** - Converts trajectory samples to complete FluSight forecast DataFrame.

**Parameters:**
- `trajectories_dict` (dict): Trajectories keyed by horizon (-1, 0, 1, 2, 3) or 'season'
  - Weekly horizons: 1D arrays of shape `(n_samples,)`
  - Season: 2D array of shape `(n_samples, n_weeks)`
- `reference_date` (str): Forecast reference date (Saturday, YYYY-MM-DD format)
- `location` (str): Location code (e.g., "US", "06" for California)
- `population` (float): Population for rate calculations (per 100k basis)
- `week_end_dates` (list, optional): List of Saturday dates for season trajectories
- `quantiles` (list, optional): Quantile levels (default: 23 required quantiles)
- `season_start` (str, optional): Season start date (default: "2025-11-22")
- `season_end` (str, optional): Season end date (default: "2026-05-23")

**Returns:** pandas DataFrame with FluSight-compliant forecast data

**Targets Generated:**
1. `wk inc flu hosp` - Weekly hospital admission quantiles
2. `wk flu hosp rate change` - Rate change category probabilities
3. `peak week inc flu hosp` - Peak week probabilities
4. `peak inc flu hosp` - Peak intensity quantiles

### `convert_samples_to_flusight()`

**Sample format conversion** - Converts connected trajectory samples to FluSight sample format.

**Parameters:**
- `samples` (np.ndarray): Array of shape `(n_samples, n_horizons)`
- `reference_date` (str): Forecast reference date (Saturday, YYYY-MM-DD)
- `location` (str): Location code
- `horizons` (list, optional): List of horizons (default: [-1, 0, 1, 2, 3])

**Returns:** pandas DataFrame with sample-formatted forecasts

**Use Case:** Submit coherent trajectories showing correlated forecasts across horizons.

### Utility Functions

#### `trajectories_to_quantiles(trajectories, quantiles=None, round_to_int=True)`
Convert trajectory samples to quantile forecasts.

#### `trajectories_to_rate_change_pmf(trajectories, baseline_trajectories, population, horizon)`
Calculate rate change probabilities with horizon-specific thresholds.

#### `trajectories_to_peak_week_pmf(trajectories, week_end_dates, season_weeks=None)`
Identify peak week distribution from seasonal trajectories.

#### `trajectories_to_peak_intensity_quantiles(trajectories, quantiles=None)`
Calculate peak intensity quantiles from seasonal trajectories.

#### `generate_season_weeks(start_date="2025-11-22", end_date="2026-05-23")`
Generate list of Saturday dates spanning the flu season.

## Rate Change Classification

The converter implements FluSight's horizon-specific rate change thresholds:

| Horizon | Description | Stable Threshold | Large Change Threshold |
|---------|-------------|------------------|------------------------|
| 0 | 1-week ahead | < 0.3 per 100k | ≥ 1.7 per 100k |
| 1 | 2-week ahead | < 0.5 per 100k | ≥ 3.0 per 100k |
| 2 | 3-week ahead | < 0.7 per 100k | ≥ 4.0 per 100k |
| 3 | 4-week ahead | < 1.0 per 100k | ≥ 5.0 per 100k |

**Special Rule:** Changes < 10 hospital admissions are always classified as "stable" regardless of rate.

**Categories:**
- `large_increase` - Rate change ≥ large threshold
- `increase` - Positive change, stable < rate < large threshold
- `stable` - |rate| < stable threshold OR |count change| < 10
- `decrease` - Negative change, stable < |rate| < large threshold
- `large_decrease` - Rate change ≤ -large threshold

## Integration with Influpaint

Influpaint generates daily trajectories over the full flu season. Convert to weekly FluSight format:

```python
import numpy as np
from datetime import datetime, timedelta

# Assume influpaint_samples has shape (n_samples, n_days, n_locations)
# where n_days = 448 (64 weeks × 7 days)

def daily_to_weekly(daily_samples):
    """Convert daily samples to weekly sums."""
    n_samples, n_days, n_locs = daily_samples.shape
    n_weeks = n_days // 7

    weekly = np.zeros((n_samples, n_weeks, n_locs))
    for week in range(n_weeks):
        start_day = week * 7
        end_day = start_day + 7
        weekly[:, week, :] = daily_samples[:, start_day:end_day, :].sum(axis=1)

    return weekly

# Convert to weekly
weekly_samples = daily_to_weekly(influpaint_samples)

# Extract horizons relative to reference week
reference_week_idx = 10  # Example: week 10 in season
location_idx = 0  # First location

trajectories_dict = {}
for horizon in [-1, 0, 1, 2, 3]:
    week_idx = reference_week_idx + horizon
    if 0 <= week_idx < weekly_samples.shape[1]:
        trajectories_dict[horizon] = weekly_samples[:, week_idx, location_idx]

# Use full season for peak forecasts
trajectories_dict['season'] = weekly_samples[:, :, location_idx]

# Convert to FluSight format
forecast_df = convert_trajectories_to_flusight(
    trajectories_dict=trajectories_dict,
    reference_date="2025-11-29",
    location="US",
    population=331000000
)
```

See `example_usage.py` for complete working examples.

## Examples

The `example_usage.py` script contains four detailed examples:

### Example 1: Basic Conversion
Single-location forecast with all targets (quantiles, rate change, peaks).

### Example 2: Sample Trajectories
Convert connected trajectories to FluSight sample format preserving correlations.

### Example 3: Multi-Location Forecasts
Batch processing for multiple states/locations with appropriate populations.

### Example 4: Influpaint Integration
Pseudo-code showing the complete workflow from influpaint output to submission.

Run examples:
```bash
python example_usage.py
```

**Note:** Requires numpy and pandas installed.

## Output Format

The converter produces DataFrames with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `reference_date` | Saturday of forecast submission | 2025-11-29 |
| `target` | Forecast target type | wk inc flu hosp |
| `horizon` | Weeks from reference date | 1 |
| `target_end_date` | Saturday of target week | 2025-12-06 |
| `location` | FIPS code or "US" | US |
| `output_type` | Forecast type | quantile |
| `output_type_id` | Quantile level or category | 0.500 |
| `value` | Forecast value | 1234 |

### Output Types by Target

- **wk inc flu hosp**: quantile (23 levels) or sample
- **wk flu hosp rate change**: pmf (5 categories)
- **peak week inc flu hosp**: pmf (one per week in season)
- **peak inc flu hosp**: quantile (23 levels)

## Validation

Before submitting to FluSight, validate your forecast file:

```r
# In R
library(hubValidations)
hubValidations::validate_submission(
    hub_path="path/to/FluSight-forecast-hub",
    file_path="model-output/TeamName-ModelName/2025-11-29-TeamName-ModelName.csv"
)
```

The converter ensures:
- ✓ Correct column names and order
- ✓ Integer values for hospital admission targets
- ✓ 23 required quantiles at specified levels
- ✓ PMF probabilities sum to 1.0
- ✓ Proper date formats (YYYY-MM-DD)
- ✓ Appropriate horizon values (-1 to 3 for weekly, blank for seasonal)

## Multi-Location Workflow

Process multiple locations efficiently:

```python
import pandas as pd
from trajectory_converter import convert_trajectories_to_flusight

locations_data = {
    "US": {"population": 331000000},
    "06": {"population": 39500000},   # California
    "36": {"population": 19450000},   # New York
    "48": {"population": 29000000},   # Texas
}

all_forecasts = []

for location, data in locations_data.items():
    # Generate or load trajectories for this location
    trajectories_dict = get_trajectories_for_location(location)

    forecast_df = convert_trajectories_to_flusight(
        trajectories_dict=trajectories_dict,
        reference_date="2025-11-29",
        location=location,
        population=data["population"]
    )

    all_forecasts.append(forecast_df)

# Combine all locations
combined_df = pd.concat(all_forecasts, ignore_index=True)
combined_df.to_csv("2025-11-29-TeamName-ModelName.csv", index=False)
```

## Common Issues and Solutions

### Issue: PMF doesn't sum to 1.0
**Cause:** Numerical precision in probability calculation
**Solution:** Converter automatically normalizes by dividing counts by n_samples

### Issue: Missing baseline for rate change
**Cause:** No horizon=-1 trajectories provided
**Solution:** Include lookback week (horizon -1) in trajectories_dict

### Issue: Peak week dates don't match season
**Cause:** Custom week_end_dates don't align with season_start/season_end
**Solution:** Use `generate_season_weeks()` to create valid dates

### Issue: Integer overflow in quantiles
**Cause:** Extremely large trajectory values
**Solution:** Check input data scaling and units (should be admission counts)

## FluSight Submission Checklist

- [ ] File name format: `YYYY-MM-DD-team-model.csv`
- [ ] Reference date is the Saturday following submission deadline
- [ ] Location codes match FluSight locations (see auxiliary-data/locations.csv)
- [ ] All 23 required quantiles included
- [ ] Hospital admission values are integers
- [ ] Rate change PMFs sum to 1.0 for each location/horizon
- [ ] Peak week dates are Saturdays in season (2025-11-22 to 2026-05-23)
- [ ] Submission file validates with hubValidations
- [ ] Team/model metadata file exists in model-metadata folder

## Technical Details

### Required Quantiles
```python
[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
 0.95, 0.975, 0.99]
```

### Forecast Horizons
- **-1**: Lookback week (baseline for rate change)
- **0**: Nowcast (week of submission)
- **1**: 1-week ahead forecast
- **2**: 2-week ahead forecast
- **3**: 3-week ahead forecast

### Date Calculations
```
target_end_date = reference_date + (horizon × 7 days)
```

All dates must be Saturdays (end of epidemiological week).

### Season Definition (2025-2026)
- Start: Saturday, November 22, 2025
- End: Saturday, May 23, 2026
- Duration: 26 weeks

## Limitations

1. **Population Data**: Must be provided manually (not auto-loaded)
2. **Emergency Department Visits**: Target `wk inc flu prop ed visits` not implemented
3. **Validation**: No built-in hubValidations integration (run separately)
4. **Performance**: Processes locations sequentially (not parallelized)

## Future Enhancements

- Auto-load population data from FluSight auxiliary files
- Add emergency department visit proportion target
- Built-in validation against hubValidations
- Parallel processing for multi-location forecasts
- Support for additional output types (cdf, mean, etc.)

## Contributing

To extend this converter:
1. Add new target types by creating conversion functions
2. Update `convert_trajectories_to_flusight()` to include new targets
3. Add examples to `example_usage.py`
4. Update this README and docstrings

## References

- [FluSight Forecast Hub](https://github.com/cdcepi/FluSight-forecast-hub)
- [FluSight Model Output Guidelines](https://github.com/cdcepi/FluSight-forecast-hub/tree/main/model-output)
- [Hubverse Documentation](https://hubverse.io/en/latest/)
- [Influpaint Model](https://github.com/ACCIDDA/influpaint)

## License

This code is provided as-is for use with FluSight forecasting. Follow FluSight Forecast Hub guidelines when submitting forecasts.

## Contact

For questions about this converter or FluSight submissions, refer to:
- FluSight Forecast Hub: flusight@cdc.gov
- Hubverse Documentation: https://hubverse.io
- FluSight GitHub Issues: https://github.com/cdcepi/FluSight-forecast-hub/issues
