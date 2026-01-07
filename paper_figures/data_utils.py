"""
Data preprocessing and utilities for paper figures.

This module contains common data operations to eliminate code duplication.
"""

import os
from functools import lru_cache
import numpy as np
import pandas as pd
from typing import Optional

from influpaint.utils import SeasonAxis
from influpaint.utils.helpers import flusight_quantiles, flusight_quantile_pairs


def normalize_samples_shape(inv_samples: np.ndarray) -> np.ndarray:
    """Normalize samples to (N, C, W, P) shape.

    Handles both 3D (N, W, P) and 4D (N, C, W, P) inputs.

    Args:
        inv_samples: Array of shape (N, W, P) or (N, C, W, P)

    Returns:
        Array of shape (N, C, W, P)

    Raises:
        ValueError: If input has invalid dimensions
    """
    if inv_samples.ndim == 4:
        return inv_samples
    elif inv_samples.ndim == 3:
        return inv_samples[:, None, :, :]
    else:
        raise ValueError(
            f"inv_samples must be (N, W, P) or (N, C, W, P), got shape {inv_samples.shape}"
        )


def get_real_weeks(samples: np.ndarray, max_weeks: int = 53) -> int:
    """Get the effective number of weeks from samples.

    Args:
        samples: Samples array (already normalized to 4D)
        max_weeks: Maximum weeks to consider

    Returns:
        Minimum of max_weeks and actual weeks in samples
    """
    return min(max_weeks, samples.shape[2])


def get_location_index(state: str, season_axis: SeasonAxis) -> int:
    """Get location index for a state code.

    Args:
        state: State code, abbreviation, or 'US'
        season_axis: SeasonAxis object

    Returns:
        Index in season_axis.locations

    Raises:
        ValueError: If state cannot be found
    """
    from .helpers import state_to_code
    loc_code = state_to_code(state, season_axis)
    return season_axis.locations.index(loc_code)


@lru_cache(maxsize=10)
def load_ground_truth_cached(season: str) -> pd.DataFrame:
    """Load ground truth data for a season with caching.

    Args:
        season: Season string like '2023-2024'

    Returns:
        DataFrame with ground truth data
    """
    from prepare_dataset_for_scoringutils import ScoringutilsFullEvaluator
    ev = ScoringutilsFullEvaluator()
    gt = ev.load_ground_truth(season)
    gt["date"] = pd.to_datetime(gt["date"])
    return gt


def get_state_timeseries(samples: np.ndarray,
                          state: str,
                          season_axis: SeasonAxis) -> np.ndarray:
    """Extract timeseries for a specific state from samples.

    Args:
        samples: Normalized samples (N, C, W, P)
        state: State code/abbreviation
        season_axis: SeasonAxis object

    Returns:
        Array of shape (N, W) for the specified state
    """
    place_idx = get_location_index(state, season_axis)
    real_weeks = get_real_weeks(samples)
    return samples[:, 0, :real_weeks, place_idx]


def compute_national_aggregate(samples: np.ndarray,
                                season_axis: SeasonAxis) -> np.ndarray:
    """Compute national (US) aggregate from state-level samples.

    Args:
        samples: Normalized samples (N, C, W, P)
        season_axis: SeasonAxis object

    Returns:
        Array of shape (N, W) with national aggregate
    """
    num_locations = len(season_axis.locations)
    return samples[:, 0, :, :num_locations].sum(axis=-1)


def get_state_labels(indices: list[int], season_axis: SeasonAxis) -> list[str]:
    """Get state abbreviation labels for location indices.

    Args:
        indices: List of location indices
        season_axis: SeasonAxis object

    Returns:
        List of state abbreviations
    """
    locdf = season_axis.locations_df
    if 'abbreviation' not in locdf.columns:
        return [str(season_axis.locations[i]) for i in indices]

    abbr_map = locdf.set_index('location_code')['abbreviation']
    return [
        abbr_map.get(str(season_axis.locations[i]), str(season_axis.locations[i]))
        for i in indices
    ]


def compute_quantile_curves(timeseries: np.ndarray,
                             quantile_pairs: Optional[np.ndarray] = None) -> list[tuple]:
    """Compute quantile curves for timeseries.

    Args:
        timeseries: Array of shape (N, W)
        quantile_pairs: Array of (low, high) quantile pairs.
                       Defaults to flusight_quantile_pairs

    Returns:
        List of (lo_curve, hi_curve) tuples
    """
    if quantile_pairs is None:
        quantile_pairs = flusight_quantile_pairs

    curves = []
    for lo, hi in quantile_pairs:
        lo_curve = np.quantile(timeseries, lo, axis=0)
        hi_curve = np.quantile(timeseries, hi, axis=0)
        curves.append((lo_curve, hi_curve))
    return curves


def compute_median(timeseries: np.ndarray) -> np.ndarray:
    """Compute median across samples.

    Args:
        timeseries: Array of shape (N, W)

    Returns:
        Median array of shape (W,)
    """
    return np.quantile(timeseries, 0.5, axis=0)


def validate_samples_and_season_axis(samples: np.ndarray, season_axis: SeasonAxis):
    """Validate that samples are compatible with season_axis.

    Args:
        samples: Normalized samples (N, C, W, P)
        season_axis: SeasonAxis object

    Raises:
        ValueError: If samples and season_axis are incompatible
    """
    num_locations_in_samples = samples.shape[3]
    num_locations_in_axis = len(season_axis.locations)

    if num_locations_in_samples < num_locations_in_axis:
        raise ValueError(
            f"Samples have {num_locations_in_samples} locations but "
            f"season_axis has {num_locations_in_axis}"
        )


def compute_historical_peak_threshold(season_axis: SeasonAxis,
                                       seasons: list[int] = None,
                                       threshold_fraction: float = 0.2,
                                       data_path: str = 'influpaint/data/nhsn_flusight_past.csv') -> np.ndarray:
    """Compute peak threshold per location from historical seasons.

    For each location, finds the lowest peak across specified historical seasons
    and returns a fraction of that value as the threshold for that location.

    Args:
        season_axis: SeasonAxis object for location mapping
        seasons: List of season years (e.g., [2022, 2023, 2024]).
                If None, uses [2022, 2023, 2024].
        threshold_fraction: Fraction of lowest peak to use as threshold (default 0.2 = 20%)
        data_path: Path to historical data CSV file

    Returns:
        Array of threshold values per location (shape: num_locations,)
    """
    if seasons is None:
        seasons = [2022, 2023, 2024]

    gt_df = pd.read_csv(data_path)
    num_locations = len(season_axis.locations)
    thresholds = np.zeros(num_locations)

    # For each location, find the lowest peak across all seasons
    for loc_idx, loc_code in enumerate(season_axis.locations):
        location_peaks = []

        for season_year in seasons:
            season_data = gt_df[gt_df['fluseason'] == season_year]
            if season_data.empty:
                continue

            season_pivot = season_data.pivot(columns='location_code', values='value', index='season_week')

            # Get peak for this location in this season
            if loc_code in season_pivot.columns:
                series = season_pivot[loc_code].dropna().values
                if len(series) > 0:
                    peak = np.max(series)
                    location_peaks.append(peak)

        if location_peaks:
            # Find the lowest peak for this location across seasons
            lowest_peak = np.min(location_peaks)
            thresholds[loc_idx] = threshold_fraction * lowest_peak
        else:
            # If no data for this location, set a very low threshold
            thresholds[loc_idx] = 0.0

    print(f"Historical peak threshold calculation (per location):")
    print(f"  Seasons analyzed: {seasons}")
    print(f"  Threshold range: [{np.min(thresholds):.2f}, {np.max(thresholds):.2f}]")
    print(f"  Mean threshold: {np.mean(thresholds):.2f}")
    print(f"  Threshold fraction: {threshold_fraction*100}% of lowest peak per location")

    return thresholds


def filter_trajectories_by_peak(samples: np.ndarray,
                                  season_axis: SeasonAxis,
                                  peak_thresholds: np.ndarray,
                                  max_low_locations: int = 10) -> np.ndarray:
    """Filter trajectories to remove those with unrealistically low peaks.

    For each trajectory and location, finds the maximum value (peak) for that location.
    Removes trajectories where more than max_low_locations have peaks below threshold.

    Args:
        samples: Array of shape (N, C, W, P) or (N, W, P)
        season_axis: SeasonAxis object
        peak_thresholds: Array of minimum allowed peak values per location (shape: num_locations,)
        max_low_locations: Maximum number of locations allowed to have peaks below threshold
                          before removing trajectory (default: 10)

    Returns:
        Filtered array with same shape but potentially fewer samples (N_filtered, C, W, P)
    """
    arr = normalize_samples_shape(samples)
    n, c, w, p = arr.shape
    real_weeks = get_real_weeks(arr)
    num_locations = len(season_axis.locations)

    # Check each trajectory
    keep_mask = np.ones(n, dtype=bool)
    low_location_counts = []

    for sample_idx in range(n):
        # Count how many locations have peaks below threshold
        n_low_locations = 0
        for loc_idx in range(num_locations):
            location_data = arr[sample_idx, 0, :real_weeks, loc_idx]
            location_peak = np.max(location_data)

            if location_peak < peak_thresholds[loc_idx]:
                n_low_locations += 1

        low_location_counts.append(n_low_locations)

        # Remove trajectory if MORE than max_low_locations have low peaks
        if n_low_locations > max_low_locations:
            keep_mask[sample_idx] = False

    filtered_samples = arr[keep_mask]

    # Statistics
    n_removed = n - filtered_samples.shape[0]
    print(f"Trajectory filtering by peak (removing blank/low trajectories):")
    print(f"  Original trajectories: {n}")
    print(f"  Max locations allowed below threshold: {max_low_locations}")
    print(f"  Trajectories removed: {n_removed} ({100*n_removed/n:.1f}%)")
    print(f"  Trajectories kept: {filtered_samples.shape[0]} ({100*filtered_samples.shape[0]/n:.1f}%)")

    if filtered_samples.shape[0] > 0:
        # Show peak stats for kept trajectories
        kept_peaks = []
        kept_low_counts = []
        for idx, sample_idx in enumerate(np.where(keep_mask)[0]):
            max_peak = np.max(arr[sample_idx, 0, :real_weeks, :num_locations])
            kept_peaks.append(max_peak)
            kept_low_counts.append(low_location_counts[sample_idx])
        print(f"  Peak range in kept trajectories: [{np.min(kept_peaks):.2f}, {np.max(kept_peaks):.2f}]")
        print(f"  Low location count in kept trajectories: [{np.min(kept_low_counts)}, {np.max(kept_low_counts)}] (mean: {np.mean(kept_low_counts):.1f})")

    return filtered_samples


# Export commonly used constants
__all__ = [
    'normalize_samples_shape',
    'get_real_weeks',
    'get_location_index',
    'load_ground_truth_cached',
    'get_state_timeseries',
    'compute_national_aggregate',
    'get_state_labels',
    'compute_quantile_curves',
    'compute_median',
    'validate_samples_and_season_axis',
    'compute_historical_peak_threshold',
    'filter_trajectories_by_peak',
    'flusight_quantiles',
    'flusight_quantile_pairs',
]
