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
    'flusight_quantiles',
    'flusight_quantile_pairs',
]
