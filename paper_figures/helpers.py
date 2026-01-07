"""
Helper functions for paper figures generation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from influpaint.utils import SeasonAxis


def forecast_week_saturdays(season: str, season_axis: SeasonAxis, max_weeks: int) -> pd.DatetimeIndex:
    """Return Saturday dates for forecast weeks for a given season string 'YYYY-YYYY'.

    Uses SeasonAxis.get_season_calendar under the hood.
    """
    season_year = int(str(season).split('-')[0])
    cal = season_axis.get_season_calendar(season_year)
    saturdays = pd.to_datetime(cal['saturday'])
    eff_len = min(max_weeks, len(saturdays))
    return saturdays[:eff_len]


def format_date_axis(ax):
    """Apply Dec 2025 date format and tilt labels to avoid overlap."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')


def load_unconditional_samples(path: str) -> np.ndarray:
    """Load unconditional samples from a .npy file.

    Args:
        path: Path to the .npy file

    Returns:
        Array with shape (N, 1, weeks, places)
    """
    x = np.load(path)
    if x.ndim == 3:
        x = x[:, None, :, :]
    return x


def list_influpaint_csvs(base_dir: str, model_id: str, config: str):
    """List all influpaint CSV forecast files matching model_id and config.

    Args:
        base_dir: Base directory to search
        model_id: Model identifier
        config: Configuration name

    Returns:
        Sorted list of CSV file paths
    """
    out = []
    for root, _, files in os.walk(base_dir):
        if model_id in root and f"conf_{config}" in root:
            for f in files:
                if f.endswith(".csv") and not f.endswith("-copaint.csv"):
                    out.append(os.path.join(root, f))
    return sorted(out)


def parse_date_from_folder(folder_name: str):
    """Parse date from folder name.

    Args:
        folder_name: Name of the folder

    Returns:
        Parsed date or None if parsing fails
    """
    try:
        d = folder_name.split("::")[-1]
        return pd.to_datetime(d).date()
    except Exception:
        return None


def list_inpainting_dirs(base_dir: str, model_id: str, config: str):
    """List all inpainting directories matching model_id and config.

    Args:
        base_dir: Base directory to search
        model_id: Model identifier
        config: Configuration name

    Returns:
        Sorted list of directory paths containing fluforecasts_ti.npy
    """
    out = []
    for d in os.listdir(base_dir):
        p = os.path.join(base_dir, d)
        if not os.path.isdir(p):
            continue
        if (d.startswith(model_id) and f"conf_{config}" in d):
            if os.path.exists(os.path.join(p, "fluforecasts_ti.npy")):
                out.append(p)
    return sorted(out)


def state_to_code(state: str, season_axis: SeasonAxis) -> str:
    """Map 'US', FIPS code like '37', or abbrev like 'NC' to location_code string.

    Args:
        state: State identifier (US, FIPS code, or abbreviation)
        season_axis: SeasonAxis object for location mapping

    Returns:
        Location code string

    Raises:
        ValueError: If state cannot be mapped
    """
    if state.upper() == 'US':
        return 'US'
    if state in set(season_axis.locations_df["location_code"].astype(str)):
        return str(state)
    m = season_axis.locations_df[season_axis.locations_df['abbreviation'].str.upper() == state.upper()]
    if not m.empty:
        return str(m.iloc[0]['location_code'])
    raise ValueError(f"Unknown state '{state}'")
