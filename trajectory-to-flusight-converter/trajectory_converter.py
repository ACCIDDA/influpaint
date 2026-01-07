"""
Trajectory to FluSight Forecast Converter

This module converts influpaint trajectory samples into all required FluSight
forecast output formats including quantile forecasts, rate change probabilities,
and peak forecasts.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union


# Required quantiles for FluSight submissions
REQUIRED_QUANTILES = [
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
    0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
    0.95, 0.975, 0.99
]

# Horizons for weekly forecasts
WEEKLY_HORIZONS = [-1, 0, 1, 2, 3]
RATE_CHANGE_HORIZONS = [0, 1, 2, 3]

# Rate change categories
RATE_CATEGORIES = ["large_decrease", "decrease", "stable", "increase", "large_increase"]


def get_rate_change_thresholds(horizon: int) -> Dict[str, float]:
    """
    Get rate change thresholds (per 100k) for a given horizon.

    Args:
        horizon: Forecast horizon (0, 1, 2, or 3)

    Returns:
        Dictionary with 'stable_threshold' and 'large_threshold' in per 100k units
    """
    thresholds = {
        0: {"stable": 0.3, "large": 1.7},  # 1-week ahead
        1: {"stable": 0.5, "large": 3.0},  # 2-week ahead
        2: {"stable": 0.7, "large": 4.0},  # 3-week ahead
        3: {"stable": 1.0, "large": 5.0},  # 4-week ahead
    }
    return thresholds.get(horizon, thresholds[3])


def classify_rate_change(
    count_change: float,
    rate_change: float,
    horizon: int
) -> str:
    """
    Classify a rate change into one of five categories.

    Args:
        count_change: Change in hospital admission counts
        rate_change: Change in rate per 100k population
        horizon: Forecast horizon (0, 1, 2, or 3)

    Returns:
        Category name: large_decrease, decrease, stable, increase, or large_increase
    """
    thresholds = get_rate_change_thresholds(horizon)
    stable_thresh = thresholds["stable"]
    large_thresh = thresholds["large"]

    # Check if count change is small (< 10 admissions) -> stable
    if abs(count_change) < 10:
        return "stable"

    # Check if rate change magnitude is below stable threshold -> stable
    if abs(rate_change) < stable_thresh:
        return "stable"

    # Classify based on direction and magnitude
    if rate_change >= large_thresh:
        return "large_increase"
    elif rate_change > 0:
        return "increase"
    elif rate_change <= -large_thresh:
        return "large_decrease"
    else:  # rate_change < 0 but > -large_thresh
        return "decrease"


def trajectories_to_quantiles(
    trajectories: np.ndarray,
    quantiles: Optional[List[float]] = None,
    round_to_int: bool = True
) -> np.ndarray:
    """
    Convert trajectory samples to quantile forecasts.

    Args:
        trajectories: Array of shape (n_samples, n_horizons) or (n_samples,)
        quantiles: List of quantile levels (default: REQUIRED_QUANTILES)
        round_to_int: Whether to round to integers (required for hospital admissions)

    Returns:
        Array of quantile values, shape (n_quantiles,) or (n_quantiles, n_horizons)
    """
    if quantiles is None:
        quantiles = REQUIRED_QUANTILES

    # Calculate quantiles
    quantile_values = np.quantile(trajectories, quantiles, axis=0)

    # Round to integers if required
    if round_to_int:
        quantile_values = np.round(quantile_values).astype(int)

    return quantile_values


def trajectories_to_rate_change_pmf(
    trajectories: np.ndarray,
    baseline_trajectories: np.ndarray,
    population: float,
    horizon: int
) -> Dict[str, float]:
    """
    Convert trajectories to rate change probability mass function.

    Args:
        trajectories: Array of trajectory samples at target horizon, shape (n_samples,)
        baseline_trajectories: Array of baseline (horizon -1) samples, shape (n_samples,)
        population: Population size for the location
        horizon: Forecast horizon (0, 1, 2, or 3)

    Returns:
        Dictionary mapping category names to probabilities
    """
    n_samples = len(trajectories)

    # Calculate changes for each sample
    count_changes = trajectories - baseline_trajectories
    rate_changes = (count_changes / population) * 100000  # per 100k

    # Classify each sample
    categories = [
        classify_rate_change(count_changes[i], rate_changes[i], horizon)
        for i in range(n_samples)
    ]

    # Calculate probabilities
    pmf = {cat: 0.0 for cat in RATE_CATEGORIES}
    for cat in categories:
        pmf[cat] += 1.0 / n_samples

    return pmf


def trajectories_to_peak_week_pmf(
    trajectories: np.ndarray,
    week_end_dates: List[str],
    season_weeks: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Convert trajectories to peak week probability mass function.

    Args:
        trajectories: Array of shape (n_samples, n_weeks) containing admission counts
        week_end_dates: List of target_end_date strings (YYYY-MM-DD) for each week
        season_weeks: Optional list of all valid season week end dates

    Returns:
        Dictionary mapping week end dates to probabilities
    """
    n_samples, n_weeks = trajectories.shape

    # Find peak week for each trajectory
    peak_indices = np.argmax(trajectories, axis=1)
    peak_weeks = [week_end_dates[idx] for idx in peak_indices]

    # Initialize PMF with all possible weeks
    if season_weeks is None:
        season_weeks = week_end_dates

    pmf = {week: 0.0 for week in season_weeks}

    # Count occurrences
    for week in peak_weeks:
        if week in pmf:
            pmf[week] += 1.0 / n_samples

    return pmf


def trajectories_to_peak_intensity_quantiles(
    trajectories: np.ndarray,
    quantiles: Optional[List[float]] = None
) -> np.ndarray:
    """
    Convert trajectories to peak intensity quantile forecasts.

    Args:
        trajectories: Array of shape (n_samples, n_weeks) containing admission counts
        quantiles: List of quantile levels (default: REQUIRED_QUANTILES)

    Returns:
        Array of quantile values for peak intensities
    """
    if quantiles is None:
        quantiles = REQUIRED_QUANTILES

    # Find peak intensity for each trajectory
    peak_intensities = np.max(trajectories, axis=1)

    # Calculate quantiles and round to integers
    quantile_values = np.quantile(peak_intensities, quantiles)
    quantile_values = np.round(quantile_values).astype(int)

    return quantile_values


def generate_season_weeks(
    start_date: str = "2025-11-22",
    end_date: str = "2026-05-23"
) -> List[str]:
    """
    Generate list of all Saturday dates (week end dates) in the flu season.

    Args:
        start_date: Season start date (YYYY-MM-DD)
        end_date: Season end date (YYYY-MM-DD)

    Returns:
        List of Saturday dates in YYYY-MM-DD format
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Find first Saturday on or after start_date
    days_until_saturday = (5 - start.weekday()) % 7
    current = start + timedelta(days=days_until_saturday)

    saturdays = []
    while current <= end:
        saturdays.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=7)

    return saturdays


def convert_trajectories_to_flusight(
    trajectories_dict: Dict[str, np.ndarray],
    reference_date: str,
    location: str,
    population: float,
    week_end_dates: Optional[List[str]] = None,
    quantiles: Optional[List[float]] = None,
    season_start: str = "2025-11-22",
    season_end: str = "2026-05-23"
) -> pd.DataFrame:
    """
    Convert trajectory samples to complete FluSight forecast format.

    Args:
        trajectories_dict: Dictionary with keys as horizons (-1, 0, 1, 2, 3) and
                          values as arrays of shape (n_samples,). For peak forecasts,
                          also include 'season' key with shape (n_samples, n_weeks)
        reference_date: Reference date in YYYY-MM-DD format (Saturday)
        location: Location code (e.g., "US", "01" for state FIPS)
        population: Population for the location (for rate calculations)
        week_end_dates: Optional list of week end dates for season trajectories
        quantiles: Optional list of quantile levels (default: REQUIRED_QUANTILES)
        season_start: Season start date (YYYY-MM-DD)
        season_end: Season end date (YYYY-MM-DD)

    Returns:
        DataFrame in FluSight submission format
    """
    if quantiles is None:
        quantiles = REQUIRED_QUANTILES

    ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
    rows = []

    # 1. Weekly hospital admission quantiles
    for horizon in WEEKLY_HORIZONS:
        if horizon not in trajectories_dict:
            continue

        target_end_date = (ref_date + timedelta(days=horizon * 7)).strftime("%Y-%m-%d")
        traj = trajectories_dict[horizon]
        quantile_values = trajectories_to_quantiles(traj, quantiles, round_to_int=True)

        for q_level, q_value in zip(quantiles, quantile_values):
            rows.append({
                "reference_date": reference_date,
                "target": "wk inc flu hosp",
                "horizon": horizon,
                "target_end_date": target_end_date,
                "location": location,
                "output_type": "quantile",
                "output_type_id": f"{q_level:.3f}",
                "value": int(q_value)
            })

    # 2. Rate change probabilities
    if -1 in trajectories_dict:  # Need baseline for rate change
        baseline_traj = trajectories_dict[-1]

        for horizon in RATE_CHANGE_HORIZONS:
            if horizon not in trajectories_dict:
                continue

            target_end_date = (ref_date + timedelta(days=horizon * 7)).strftime("%Y-%m-%d")
            traj = trajectories_dict[horizon]

            pmf = trajectories_to_rate_change_pmf(
                traj, baseline_traj, population, horizon
            )

            for category, prob in pmf.items():
                rows.append({
                    "reference_date": reference_date,
                    "target": "wk flu hosp rate change",
                    "horizon": horizon,
                    "target_end_date": target_end_date,
                    "location": location,
                    "output_type": "pmf",
                    "output_type_id": category,
                    "value": prob
                })

    # 3. Peak week PMF
    if "season" in trajectories_dict:
        season_weeks = generate_season_weeks(season_start, season_end)

        if week_end_dates is None:
            week_end_dates = season_weeks

        peak_week_pmf = trajectories_to_peak_week_pmf(
            trajectories_dict["season"],
            week_end_dates,
            season_weeks
        )

        for week, prob in peak_week_pmf.items():
            rows.append({
                "reference_date": reference_date,
                "target": "peak week inc flu hosp",
                "horizon": "",  # Not applicable for seasonal targets
                "target_end_date": "",  # Not applicable for seasonal targets
                "location": location,
                "output_type": "pmf",
                "output_type_id": week,
                "value": prob
            })

    # 4. Peak intensity quantiles
    if "season" in trajectories_dict:
        peak_quantiles = trajectories_to_peak_intensity_quantiles(
            trajectories_dict["season"],
            quantiles
        )

        for q_level, q_value in zip(quantiles, peak_quantiles):
            rows.append({
                "reference_date": reference_date,
                "target": "peak inc flu hosp",
                "horizon": "",  # Not applicable for seasonal targets
                "target_end_date": "",  # Not applicable for seasonal targets
                "location": location,
                "output_type": "quantile",
                "output_type_id": f"{q_level:.3f}",
                "value": int(q_value)
            })

    # Create DataFrame with proper column order
    df = pd.DataFrame(rows)
    column_order = [
        "reference_date", "target", "horizon", "target_end_date",
        "location", "output_type", "output_type_id", "value"
    ]

    return df[column_order]


def convert_samples_to_flusight(
    samples: np.ndarray,
    reference_date: str,
    location: str,
    horizons: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Convert trajectory samples to FluSight sample format.

    Args:
        samples: Array of shape (n_samples, n_horizons) containing hospital admissions
        reference_date: Reference date in YYYY-MM-DD format (Saturday)
        location: Location code (e.g., "US", "01" for state FIPS)
        horizons: List of horizons corresponding to columns (default: [-1, 0, 1, 2, 3])

    Returns:
        DataFrame in FluSight sample submission format
    """
    if horizons is None:
        horizons = WEEKLY_HORIZONS

    ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
    n_samples, n_horizons = samples.shape

    rows = []

    for sample_idx in range(n_samples):
        output_type_id = f"{location}{sample_idx:02d}"

        for h_idx, horizon in enumerate(horizons):
            target_end_date = (ref_date + timedelta(days=horizon * 7)).strftime("%Y-%m-%d")
            value = int(np.round(samples[sample_idx, h_idx]))

            rows.append({
                "reference_date": reference_date,
                "target": "wk inc flu hosp",
                "horizon": horizon,
                "target_end_date": target_end_date,
                "location": location,
                "output_type": "sample",
                "output_type_id": output_type_id,
                "value": value
            })

    df = pd.DataFrame(rows)
    column_order = [
        "reference_date", "target", "horizon", "target_end_date",
        "location", "output_type", "output_type_id", "value"
    ]

    return df[column_order]


# Example usage and testing
if __name__ == "__main__":
    # Example: Generate synthetic trajectories
    np.random.seed(42)
    n_samples = 100

    # Weekly trajectories for horizons -1 to 3
    trajectories_dict = {}
    for horizon in [-1, 0, 1, 2, 3]:
        # Simulate some trend + noise
        mean = 1000 + horizon * 50
        trajectories_dict[horizon] = np.random.poisson(mean, n_samples).astype(float)

    # Season trajectories (26 weeks)
    n_weeks = 26
    season_traj = np.random.poisson(1000, (n_samples, n_weeks)).astype(float)
    trajectories_dict["season"] = season_traj

    # Generate week end dates for season
    season_weeks = generate_season_weeks()
    week_end_dates = season_weeks[:n_weeks]

    # Convert to FluSight format
    forecast_df = convert_trajectories_to_flusight(
        trajectories_dict=trajectories_dict,
        reference_date="2025-10-18",
        location="US",
        population=331000000,  # US population
        week_end_dates=week_end_dates
    )

    print("Forecast DataFrame shape:", forecast_df.shape)
    print("\nFirst few rows:")
    print(forecast_df.head(10))
    print("\nUnique targets:", forecast_df['target'].unique())
    print("\nUnique output types:", forecast_df['output_type'].unique())

    # Test sample format
    sample_array = np.column_stack([trajectories_dict[h] for h in WEEKLY_HORIZONS])
    sample_df = convert_samples_to_flusight(
        samples=sample_array,
        reference_date="2025-10-18",
        location="US"
    )

    print("\n\nSample DataFrame shape:", sample_df.shape)
    print("Sample output_type_ids:", sample_df['output_type_id'].unique()[:5])
