# Trajectory to FluSight Converter - Development Notes

## Task Overview
Build a function to convert influpaint trajectory samples into all required FluSight forecast outputs:
- Quantile forecasts for weekly hospital admissions
- Rate change probability mass functions (pmf)
- Peak week probability mass functions
- Peak intensity quantile forecasts

## Understanding the Input Data

### Influpaint Trajectories
Based on the ground truth processing code analysis:
- Trajectories are 3D arrays: (channels, time_steps, locations)
- Each sample/trajectory represents a possible future scenario
- Multiple samples (typically 100) provide uncertainty quantification
- Data format: xarray with dimensions for time, location, and channel

### FluSight Requirements
From the documentation:
1. **Weekly Hospital Admissions (`wk inc flu hosp`)**
   - Quantile forecasts: 23 quantiles (0.01, 0.025, 0.05, ..., 0.975, 0.99)
   - Horizons: -1, 0, 1, 2, 3 (lookback, nowcast, 1-3 weeks ahead)
   - Integer values required

2. **Rate Change (`wk flu hosp rate change`)**
   - Categorical pmf: large_decrease, decrease, stable, increase, large_increase
   - Horizons: 0, 1, 2, 3
   - Different thresholds for each horizon
   - Requires population data for rate calculation

3. **Peak Week (`peak week inc flu hosp`)**
   - Pmf over all weeks in season (2025-11-22 to 2026-05-23)
   - No horizon (seasonal forecast)

4. **Peak Intensity (`peak inc flu hosp`)**
   - Quantile forecasts: same 23 quantiles
   - Integer values required
   - No horizon (seasonal forecast)

## Implementation Plan

### Step 1: Quantile Calculation
- Input: Array of trajectory samples for each horizon
- Output: Quantile values at specified probability levels
- Method: numpy.percentile or numpy.quantile
- Round to integers for hospital admission forecasts

### Step 2: Rate Change Classification
- Calculate baseline: admission count from week before reference date (horizon -1)
- For each sample trajectory at each horizon:
  - Calculate count change
  - Calculate rate change (requires population)
  - Classify into category based on horizon-specific thresholds
- Aggregate across samples to get probabilities
- Ensure probabilities sum to 1

### Step 3: Peak Detection
- For each trajectory sample:
  - Identify the week with maximum admissions
  - Record the week end date
- Aggregate to get probability distribution over weeks
- For peak intensity, calculate quantiles of the maximum values

## Key Considerations

1. **Data Alignment**
   - Reference date is Saturday following submission
   - Target end dates are Saturdays
   - Horizon calculation: target_end_date = reference_date + horizon * 7 days

2. **Population Data**
   - Needed for rate change calculations
   - Should be per 100k population
   - Need to load from auxiliary data

3. **Thresholds for Rate Change**
   - Horizon 0: stable < 0.3, increase < 1.7
   - Horizon 1: stable < 0.5, increase < 3.0
   - Horizon 2: stable < 0.7, increase < 4.0
   - Horizon 3: stable < 1.0, increase < 5.0
   - Always check if count change < 10 (automatically stable)

4. **Season Definition**
   - 2025-2026 season: 2025-11-22 to 2026-05-23
   - Need to generate all Saturday dates in this range

## Implementation Summary

### Files Created
1. **trajectory_converter.py** - Main conversion module with all functionality
2. **example_usage.py** - Comprehensive examples showing how to use the converter
3. **requirements.txt** - Python dependencies (numpy, pandas)

### Functions Implemented

#### Core Conversion Functions
1. **`trajectories_to_quantiles()`**
   - Converts trajectory samples to quantile forecasts
   - Handles rounding to integers for hospital admissions
   - Supports custom quantile levels (defaults to 23 required quantiles)

2. **`trajectories_to_rate_change_pmf()`**
   - Converts trajectories to rate change probabilities
   - Implements horizon-specific thresholds
   - Handles the count change < 10 rule (automatic stable classification)
   - Returns proper probability mass function (sums to 1)

3. **`trajectories_to_peak_week_pmf()`**
   - Identifies peak week for each trajectory sample
   - Aggregates to probability distribution over weeks
   - Supports custom season week ranges

4. **`trajectories_to_peak_intensity_quantiles()`**
   - Extracts maximum admission count from each trajectory
   - Calculates quantiles of peak intensities
   - Rounds to integers

#### Helper Functions
5. **`classify_rate_change()`**
   - Implements the 5-category classification logic
   - Horizon-specific thresholds (0.3-1.0 for stable, 1.7-5.0 for large)
   - Count change threshold check (< 10 admissions)

6. **`generate_season_weeks()`**
   - Generates all Saturday dates in the flu season
   - Default: 2025-11-22 to 2026-05-23
   - Configurable for different seasons

#### High-Level API
7. **`convert_trajectories_to_flusight()`**
   - Main function to convert all trajectory types to FluSight format
   - Handles all required targets: weekly admissions, rate change, peak week, peak intensity
   - Returns properly formatted DataFrame ready for CSV export

8. **`convert_samples_to_flusight()`**
   - Converts trajectory samples to FluSight sample format
   - Maintains trajectory coherence across horizons
   - Generates proper output_type_id values

### Key Design Decisions

1. **Input Format**
   - Flexible dictionary-based input for trajectories
   - Keys: horizon values (-1, 0, 1, 2, 3) and 'season'
   - Values: numpy arrays of samples
   - This allows partial submissions (not all targets required)

2. **Rate Change Implementation**
   - Requires both count and rate thresholds to be met
   - Count change < 10 always results in "stable"
   - Different thresholds per horizon as specified in FluSight docs
   - Per 100k population rate calculation

3. **Integer Rounding**
   - All hospital admission forecasts rounded to integers
   - Peak intensity also rounded to integers
   - Rate change probabilities kept as floats (sum to 1.0)

4. **Season Handling**
   - Flexible week end date specification
   - Automatic generation of Saturday dates
   - Supports partial seasons (useful for mid-season forecasts)

5. **Multi-Location Support**
   - Each location processed independently
   - Population parameter required for rate calculations
   - Can concatenate multiple location forecasts

### Validation Considerations

The output format matches FluSight requirements:
- ✓ Correct column names and order
- ✓ Proper date formats (YYYY-MM-DD)
- ✓ Integer values for hospital admissions
- ✓ 23 required quantiles
- ✓ PMF values sum to 1.0 for each target/location/horizon
- ✓ Blank/empty values for non-applicable fields (peak targets)
- ✓ Proper horizon calculation (target_end_date = reference_date + horizon * 7)

### Integration with Influpaint

The influpaint model outputs trajectories as:
- xarray DataArray or numpy array
- Dimensions: (samples, time, location, channel)
- Daily resolution (needs conversion to weekly)
- Full season coverage

Conversion steps:
1. Aggregate daily to weekly (sum over 7-day windows)
2. Extract specific weeks relative to reference date for horizons
3. Use full/remaining season for peak forecasts
4. Apply the converter functions

See example_usage.py Example 4 for detailed pseudo-code.

### Testing

Created example_usage.py with 4 examples:
1. Basic conversion (single location, all targets)
2. Sample trajectory format
3. Multi-location forecasts
4. Influpaint integration workflow

Examples generate synthetic data and demonstrate proper usage.
Note: Requires numpy and pandas to run (see requirements.txt)

### Known Limitations and Future Enhancements

1. **Population Data**
   - Currently requires manual specification
   - Could be enhanced to auto-load from FluSight auxiliary data
   - Need location.csv with FIPS codes and populations

2. **Emergency Department Visits**
   - Not implemented (target: wk inc flu prop ed visits)
   - Would need separate trajectory input
   - Similar to weekly admissions but values between 0-1

3. **Data Validation**
   - No built-in validation against hubValidations
   - User should run hubValidations separately
   - Could add optional validation step

4. **Performance**
   - Current implementation processes locations sequentially
   - Could be parallelized for large multi-location forecasts
   - Vectorization used where possible

## Next Steps
1. ✓ Create notes.md
2. ✓ Build conversion functions
3. ✓ Create example usage code
4. ✓ Test logic and structure
5. Create README documentation
