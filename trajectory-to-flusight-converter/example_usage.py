"""
Example Usage: Converting Influpaint Trajectories to FluSight Format

This script demonstrates how to use the trajectory_converter module to convert
trajectory samples (from influpaint or similar models) into the required
FluSight forecast submission formats.
"""

import numpy as np
import pandas as pd
from trajectory_converter import (
    convert_trajectories_to_flusight,
    convert_samples_to_flusight,
    generate_season_weeks,
    REQUIRED_QUANTILES,
    WEEKLY_HORIZONS
)


def example_1_basic_conversion():
    """
    Example 1: Basic conversion from trajectories to quantile and pmf forecasts.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Trajectory Conversion")
    print("=" * 70)

    # Simulate trajectory samples (100 samples for each horizon)
    np.random.seed(42)
    n_samples = 100

    trajectories_dict = {}

    # Generate trajectories for horizons -1 (lookback), 0 (nowcast), 1, 2, 3 (forecasts)
    # In practice, these would come from your influpaint model output
    for horizon in WEEKLY_HORIZONS:
        # Simulate increasing trend with some noise
        mean_admissions = 1000 + horizon * 100  # Trend
        std_admissions = 150  # Variability

        # Generate samples from a lognormal distribution (more realistic for counts)
        log_mean = np.log(mean_admissions**2 / np.sqrt(mean_admissions**2 + std_admissions**2))
        log_std = np.sqrt(np.log(1 + std_admissions**2 / mean_admissions**2))
        trajectories_dict[horizon] = np.random.lognormal(log_mean, log_std, n_samples)

    # Generate seasonal trajectories (for peak forecasts)
    # This would be the full season forecast from influpaint
    n_weeks = 26  # Half a year
    season_traj = np.zeros((n_samples, n_weeks))

    for i in range(n_samples):
        # Simulate a seasonal curve (peak in middle)
        week_indices = np.arange(n_weeks)
        peak_week = np.random.randint(8, 18)  # Random peak week
        peak_intensity = np.random.uniform(1200, 1800)

        # Gaussian-like seasonal curve
        curve = peak_intensity * np.exp(-((week_indices - peak_week) ** 2) / (2 * 25))
        noise = np.random.normal(0, 50, n_weeks)
        season_traj[i] = np.maximum(100, curve + noise)  # Ensure non-negative

    trajectories_dict["season"] = season_traj

    # Set up forecast parameters
    reference_date = "2025-11-29"  # Saturday after submission
    location = "US"
    population = 331000000  # US population for rate calculations

    # Generate week end dates for the season
    season_weeks = generate_season_weeks("2025-11-22", "2026-05-23")
    week_end_dates = season_weeks[:n_weeks]

    # Convert to FluSight format
    forecast_df = convert_trajectories_to_flusight(
        trajectories_dict=trajectories_dict,
        reference_date=reference_date,
        location=location,
        population=population,
        week_end_dates=week_end_dates
    )

    print(f"\nGenerated {len(forecast_df)} rows of forecast data")
    print(f"\nTargets included: {list(forecast_df['target'].unique())}")
    print(f"\nOutput types: {list(forecast_df['output_type'].unique())}")

    # Show examples of each target type
    for target in forecast_df['target'].unique():
        print(f"\n{target}:")
        print(forecast_df[forecast_df['target'] == target].head(3).to_string(index=False))

    # Save to CSV
    output_file = "2025-11-29-Example-Model.csv"
    forecast_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved forecast to {output_file}")

    return forecast_df


def example_2_sample_trajectories():
    """
    Example 2: Converting to sample/trajectory format (optional but recommended).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Sample Trajectory Format")
    print("=" * 70)

    np.random.seed(123)
    n_samples = 100

    # Generate connected trajectories (each row is one trajectory across all horizons)
    trajectories = np.zeros((n_samples, len(WEEKLY_HORIZONS)))

    for i in range(n_samples):
        # Start from a baseline
        baseline = np.random.uniform(800, 1200)

        # Generate correlated trajectory with trend
        trend = np.random.uniform(-20, 50)  # Random trend per week
        noise_scale = 50

        for j, horizon in enumerate(WEEKLY_HORIZONS):
            trajectories[i, j] = baseline + trend * horizon + np.random.normal(0, noise_scale)
            trajectories[i, j] = max(0, trajectories[i, j])  # Ensure non-negative

    # Convert to sample format
    sample_df = convert_samples_to_flusight(
        samples=trajectories,
        reference_date="2025-11-29",
        location="US",
        horizons=WEEKLY_HORIZONS
    )

    print(f"\nGenerated {len(sample_df)} rows of sample data")
    print(f"Number of unique samples: {sample_df['output_type_id'].nunique()}")
    print(f"\nFirst sample trajectory (US00):")
    print(sample_df[sample_df['output_type_id'] == 'US00'].to_string(index=False))

    # Save to CSV
    output_file = "2025-11-29-Example-Model-samples.csv"
    sample_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved samples to {output_file}")

    return sample_df


def example_3_multi_location():
    """
    Example 3: Converting trajectories for multiple locations.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multi-Location Forecasts")
    print("=" * 70)

    np.random.seed(456)

    # Define locations and populations
    locations_data = {
        "US": {"population": 331000000},
        "06": {"population": 39500000},  # California
        "36": {"population": 19450000},  # New York
        "48": {"population": 29000000},  # Texas
    }

    all_forecasts = []

    for location, loc_data in locations_data.items():
        print(f"\nProcessing location: {location}")

        # Generate trajectories for this location
        n_samples = 100
        trajectories_dict = {}

        for horizon in WEEKLY_HORIZONS:
            # Scale by population (roughly)
            pop_scale = loc_data["population"] / 1000000  # Per million
            mean = pop_scale * (10 + horizon * 2)
            std = pop_scale * 3

            log_mean = np.log(mean**2 / np.sqrt(mean**2 + std**2))
            log_std = np.sqrt(np.log(1 + std**2 / mean**2))
            trajectories_dict[horizon] = np.random.lognormal(log_mean, log_std, n_samples)

        # Add seasonal trajectories
        n_weeks = 26
        season_traj = np.random.lognormal(
            np.log(pop_scale * 15),
            0.5,
            (n_samples, n_weeks)
        )
        trajectories_dict["season"] = season_traj

        # Convert for this location
        forecast_df = convert_trajectories_to_flusight(
            trajectories_dict=trajectories_dict,
            reference_date="2025-11-29",
            location=location,
            population=loc_data["population"],
            week_end_dates=generate_season_weeks()[:n_weeks]
        )

        all_forecasts.append(forecast_df)
        print(f"  Generated {len(forecast_df)} rows")

    # Combine all locations
    combined_df = pd.concat(all_forecasts, ignore_index=True)

    print(f"\n✓ Combined forecast has {len(combined_df)} rows for {len(locations_data)} locations")

    # Save combined forecast
    output_file = "2025-11-29-Example-MultiLocation.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"✓ Saved to {output_file}")

    return combined_df


def example_4_from_influpaint_output():
    """
    Example 4: Pseudo-code showing how to integrate with actual influpaint output.

    This example shows the expected workflow when you have actual influpaint
    trajectory outputs (as xarray or numpy arrays).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Integration with Influpaint (Pseudo-code)")
    print("=" * 70)

    pseudo_code = """
    # After running influpaint model, you typically have:
    # - samples: xarray.DataArray with dims (sample, time, location, channel)
    #   OR
    # - samples: numpy array with shape (n_samples, n_timepoints, n_locations)

    import xarray as xr
    from trajectory_converter import convert_trajectories_to_flusight

    # Example: Load influpaint output
    # influpaint_output = xr.open_dataset('path/to/influpaint/output.nc')
    # samples = influpaint_output['incidHosp']  # Shape: (100, 448, 56)

    # For this example, we'll simulate the structure
    n_samples = 100
    n_days = 448  # 64 weeks × 7 days
    n_locations = 1  # Single location for simplicity

    # Simulate influpaint-like output
    samples_full_season = np.random.poisson(100, (n_samples, n_days, n_locations))

    # Convert daily data to weekly (influpaint uses daily, FluSight uses weekly)
    def daily_to_weekly(daily_samples):
        '''Convert daily samples to weekly sums.'''
        n_samples, n_days, n_locs = daily_samples.shape
        n_weeks = n_days // 7

        weekly = np.zeros((n_samples, n_weeks, n_locs))
        for week in range(n_weeks):
            start_day = week * 7
            end_day = start_day + 7
            weekly[:, week, :] = daily_samples[:, start_day:end_day, :].sum(axis=1)

        return weekly

    weekly_samples = daily_to_weekly(samples_full_season)  # (100, 64, 1)

    # Extract specific horizons relative to reference week
    reference_week_idx = 10  # Example: week 10 in the season

    trajectories_dict = {}
    for horizon in [-1, 0, 1, 2, 3]:
        week_idx = reference_week_idx + horizon
        if 0 <= week_idx < weekly_samples.shape[1]:
            # Extract samples for this horizon and location
            trajectories_dict[horizon] = weekly_samples[:, week_idx, 0]

    # For peak forecasts, use the entire season (or remaining season)
    trajectories_dict['season'] = weekly_samples[:, :, 0]  # (100, 64)

    # Generate corresponding week end dates
    from datetime import datetime, timedelta
    season_start = datetime(2025, 11, 22)  # First Saturday
    week_end_dates = [
        (season_start + timedelta(weeks=i)).strftime('%Y-%m-%d')
        for i in range(weekly_samples.shape[1])
    ]

    # Convert to FluSight format
    forecast_df = convert_trajectories_to_flusight(
        trajectories_dict=trajectories_dict,
        reference_date=(season_start + timedelta(weeks=reference_week_idx)).strftime('%Y-%m-%d'),
        location='US',
        population=331000000,
        week_end_dates=week_end_dates
    )

    # Save
    forecast_df.to_csv('YYYY-MM-DD-TeamName-ModelName.csv', index=False)
    """

    print(pseudo_code)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRAJECTORY TO FLUSIGHT CONVERTER - EXAMPLES")
    print("=" * 70)

    # Run examples
    df1 = example_1_basic_conversion()
    df2 = example_2_sample_trajectories()
    df3 = example_3_multi_location()
    example_4_from_influpaint_output()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - 2025-11-29-Example-Model.csv (quantiles + pmf)")
    print("  - 2025-11-29-Example-Model-samples.csv (trajectory samples)")
    print("  - 2025-11-29-Example-MultiLocation.csv (multiple locations)")
