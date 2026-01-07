"""
Functions for plotting CSV forecast quantile fans.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .data_utils import flusight_quantile_pairs
from .helpers import state_to_code, list_influpaint_csvs, format_date_axis
from .config import SEASON_XLIMS


FLUSIGHT_BASES = {
    "2023-2024": "Flusight/2023-2024/FluSight-forecast-hub-official",
    "2024-2025": "Flusight/2024-2025/FluSight-forecast-hub-official",
}


def load_truth_for_season(season: str) -> pd.DataFrame:
    """Load ground truth data for a given season (with caching).

    Args:
        season: Season string like '2023-2024'

    Returns:
        DataFrame with ground truth data
    """
    from .data_utils import load_ground_truth_cached
    return load_ground_truth_cached(season)


def load_flusight_ensemble_forecast(season: str, location: str, reference_date) -> pd.DataFrame:
    """Load FluSight-Ensemble median forecast for a specific reference date and location.

    Args:
        season: Season string like '2023-2024'
        location: Location code like 'US' or state FIPS code
        reference_date: Reference date (can be date object or datetime)

    Returns:
        DataFrame with target_end_date and value columns for median forecast
    """
    if season not in FLUSIGHT_BASES:
        raise ValueError(f"Season {season} not found in FLUSIGHT_BASES")

    base_dir = FLUSIGHT_BASES[season]
    ensemble_dir = os.path.join(base_dir, "model-output", "FluSight-ensemble")

    if not os.path.exists(ensemble_dir):
        raise FileNotFoundError(f"FluSight-ensemble directory not found: {ensemble_dir}")

    if hasattr(reference_date, 'date'):
        ref_date_str = str(reference_date.date())
    else:
        ref_date_str = str(reference_date)

    csv_path = os.path.join(ensemble_dir, f"{ref_date_str}-FluSight-ensemble.csv")

    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path, dtype={"location": str})
    df["target_end_date"] = pd.to_datetime(df["target_end_date"])
    df["output_type_id"] = pd.to_numeric(df.get("output_type_id"), errors="coerce")
    df["horizon"] = pd.to_numeric(df.get("horizon"), errors="coerce")

    df_median = df[
        (df["location"] == location) &
        (np.isclose(df["output_type_id"], 0.5)) &
        (df["target"] == "wk inc flu hosp") &
        (df["output_type"] == "quantile") &
        (df["horizon"] >= 0)
    ].copy()

    return df_median.sort_values("target_end_date")


def plot_csv_quantile_fans_for_season(season: str, base_dir: str, model_id: str, config: str,
                                      season_axis,
                                      pick_every: int = 2, state='US',
                                      start_date: str = '2023-10-07',
                                      save_path: str | None = None,
                                      plot_median: bool = True,
                                      plot_flusight_ensemble: bool = True):
    """Plot CSV forecast quantile fans for a single season.

    Args:
        season: Season string like '2023-2024'
        base_dir: Base directory containing forecasts
        model_id: Model identifier
        config: Configuration name
        season_axis: SeasonAxis object for location mapping
        pick_every: Step for picking forecast dates
        state: State code(s) or list of states
        start_date: Start date for x-axis
        save_path: Optional path to save figure
        plot_median: Whether to plot median
        plot_flusight_ensemble: Whether to plot FluSight-Ensemble median

    Returns:
        matplotlib Figure object or None
    """
    states = state if isinstance(state, (list, tuple)) else [state]
    n = len(states)
    # Layout: for readability, use 2 rows when many states
    if n >= 4:
        nrows = 2
        ncols = int(np.ceil(n / 2))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), dpi=200, sharey=False, sharex=True)
        axes_list = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n, figsize=(4*n, 3.5), dpi=200, sharey=False, sharex=True)
        axes_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    csvs = list_influpaint_csvs(base_dir, model_id, config)
    if not csvs:
        print("No CSV forecasts found.")
        return None
    # Load all forecast CSVs once
    df_list = []
    for p in csvs:
        try:
            dfi = pd.read_csv(p, dtype={"location": str})
            if "reference_date" in dfi.columns:
                dfi["ref"] = pd.to_datetime(dfi["reference_date"]).dt.date
            elif "forecast_date" in dfi.columns:
                dfi["ref"] = pd.to_datetime(dfi["forecast_date"]).dt.date
            else:
                continue
            dfi["target_end_date"] = pd.to_datetime(dfi["target_end_date"]).dt.date
            dfi["q"] = pd.to_numeric(dfi.get("output_type_id", dfi.get("quantile")), errors="coerce")
            dfi["target"] = dfi.get("target", "wk inc flu hosp")
            df_list.append(dfi)
        except Exception:
            continue
    if not df_list:
        print("No valid CSV content parsed.")
        return None
    df_all = pd.concat(df_list, ignore_index=True)

    # Use fixed bounds if provided for season
    left_bound = SEASON_XLIMS.get(season, (pd.to_datetime(start_date), None))[0]
    # Default right bound is end-of-season if available, else 365 days after start
    default_right = pd.to_datetime(start_date) + pd.Timedelta(days=365)
    right_bound = SEASON_XLIMS.get(season, (None, default_right))[1] or default_right
    for i_ax, (ax, st) in enumerate(zip(axes_list, states)):
        loc_code = state_to_code(st, season_axis)
        # Ground truth for state
        gt = load_truth_for_season(season)
        gt = gt[gt["location"].astype(str) == loc_code].sort_values('date')
        gt = gt[(gt['date'] >= left_bound) & (gt['date'] <= right_bound)]
        ax.plot(gt['date'], gt['value'], color='black', lw=2)

        # State forecasts
        df = df_all[(df_all["location"].astype(str) == loc_code) & (df_all["target"] == "wk inc flu hosp") & (df_all["output_type"] == "quantile")]
        refs = sorted(df["ref"].unique())
        refs = refs[::max(1, pick_every)]
        palette = sns.color_palette("Set2", n_colors=len(refs))
        for j, r in enumerate(refs):
            sub = df[df["ref"] == r]
            if sub.empty:
                continue
            for lo, hi in flusight_quantile_pairs:
                low = sub[np.isclose(sub["q"], lo)].sort_values("target_end_date")
                up = sub[np.isclose(sub["q"], hi)].sort_values("target_end_date")
                if len(low) and len(up):
                    x = pd.to_datetime(low["target_end_date"]).values
                    mask = (x >= np.datetime64(left_bound)) & (x <= np.datetime64(right_bound))
                    if np.any(mask):
                        ax.fill_between(x[mask], low["value"].values[mask], up["value"].values[mask],
                                        color=palette[j], alpha=0.08, lw=0)
            med = sub[np.isclose(sub["q"], 0.5)].sort_values("target_end_date")
            if plot_median and len(med):
                x = pd.to_datetime(med["target_end_date"]).values
                mask = (x >= np.datetime64(left_bound)) & (x <= np.datetime64(right_bound))
                if np.any(mask):
                    ax.plot(x[mask], med["value"].values[mask], color=palette[j], lw=2)
                rdt = pd.to_datetime(r)
                if left_bound <= rdt <= right_bound:
                    ax.axvline(rdt, color=palette[j], ls='--', lw=1)
                    # Add date label near the top like in multi-season plot
                    ymax = ax.get_ylim()[1]
                    ax.text(rdt, ymax*0.95, str(rdt.date()), color=palette[j], rotation=90,
                            ha='right', va='top', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            if plot_flusight_ensemble:
                ensemble = load_flusight_ensemble_forecast(season, loc_code, r)
                if not ensemble.empty:
                    x = ensemble["target_end_date"].values
                    mask = (x >= np.datetime64(left_bound)) & (x <= np.datetime64(right_bound))
                    if np.any(mask):
                        ax.plot(x[mask], ensemble["value"].values[mask], color='#333333', lw=2, ls=':', label='FluSight-ensemble' if j == 0 else '')
        # Use full location name
        full_name = "United States" if loc_code == 'US' else season_axis.get_location_name(loc_code)
        ax.text(0.02, 0.98, full_name, transform=ax.transAxes, va='top', ha='left',
                               fontsize=11, fontweight='bold',
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_ylim(bottom=0)
        if i_ax == 0:
            ax.set_ylabel('Incident flu hospitalizations')
        else:
            ax.set_ylabel('')
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)
        ax.set_xlim(left_bound, right_bound)
        format_date_axis(ax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_csv_quantile_fans_multiseasons(seasons: list, base_dir: str, model_id: str, config: str,
                                        season_axis,
                                        states: list, pick_every: int = 2,
                                        save_path: str | None = None,
                                        plot_median: bool = True,
                                        plot_flusight_ensemble: bool = True):
    """Plot CSV forecast fans over full multi-season ground truth for multiple states.

    Args:
        seasons: list like ['2023-2024','2024-2025']
        base_dir: Base directory containing forecasts
        model_id: Model identifier
        config: Configuration name
        season_axis: SeasonAxis object for location mapping
        states: list like ['US','NC','CA','NY','TX']
        pick_every: Step for picking forecast dates
        save_path: Optional path to save figure
        plot_median: Whether to plot median
        plot_flusight_ensemble: Whether to plot FluSight-Ensemble median

    Returns:
        matplotlib Figure object or None
    """
    import datetime as dt

    # Build GT across requested seasons
    gt_all_list = []
    for s in seasons:
        g = load_truth_for_season(s)
        g['date'] = pd.to_datetime(g['date'])
        gt_all_list.append(g)
    gt_all = pd.concat(gt_all_list, ignore_index=True)

    # Load all forecast CSVs once
    csvs = list_influpaint_csvs(base_dir, model_id, config)
    if not csvs:
        print("No CSV forecasts found.")
        return None
    df_list = []
    for p in csvs:
        try:
            dfi = pd.read_csv(p, dtype={"location": str})
            if "reference_date" in dfi.columns:
                dfi["ref"] = pd.to_datetime(dfi["reference_date"]).dt.date
            elif "forecast_date" in dfi.columns:
                dfi["ref"] = pd.to_datetime(dfi["forecast_date"]).dt.date
            else:
                continue
            dfi["target_end_date"] = pd.to_datetime(dfi["target_end_date"]).dt.date
            dfi["q"] = pd.to_numeric(dfi.get("output_type_id", dfi.get("quantile")), errors="coerce")
            dfi["target"] = dfi.get("target", "wk inc flu hosp")
            df_list.append(dfi)
        except Exception:
            continue
    if not df_list:
        print("No valid CSV content parsed.")
        return None
    df_all = pd.concat(df_list, ignore_index=True)

    # Layout: two rows for readability when many states
    n = len(states)
    if n >= 4:
        nrows = 2
        ncols = int(np.ceil(n / 2))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), dpi=200, sharey=False, sharex=True)
        axes_list = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5), dpi=200, sharey=False, sharex=True)
        axes_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    # Global x-lims spanning both seasons
    x_left = dt.datetime(2023, 10, 7)
    x_right = dt.datetime(2025, 5, 31)

    for ax, st in zip(axes_list, states):
        loc_code = state_to_code(st, season_axis)
        # GT
        if loc_code == 'US':
            gt_us = gt_all[gt_all['location'].astype(str) == 'US'].sort_values('date')
            ax.plot(gt_us['date'], gt_us['value'], color='black', lw=2)
        else:
            gt_st = gt_all[gt_all['location'].astype(str) == loc_code].sort_values('date')
            ax.plot(gt_st['date'], gt_st['value'], color='black', lw=2)

        # Forecasts for this state across both seasons
        df = df_all[(df_all["location"].astype(str) == loc_code) & (df_all["target"] == "wk inc flu hosp") & (df_all["output_type"] == "quantile")]
        refs = sorted(df["ref"].unique())
        refs = refs[::max(1, pick_every)]
        palette = sns.color_palette("Set2", n_colors=len(refs))
        for i, r in enumerate(refs):
            sub = df[df["ref"] == r]
            if sub.empty:
                continue
            # quantile bands
            for lo, hi in flusight_quantile_pairs:
                low = sub[np.isclose(sub["q"], lo)].sort_values("target_end_date")
                up = sub[np.isclose(sub["q"], hi)].sort_values("target_end_date")
                if len(low) and len(up):
                    x = pd.to_datetime(low["target_end_date"]).values
                    ax.fill_between(x, low["value"].values, up["value"].values, color=palette[i], alpha=0.08, lw=0)
            # median
            med = sub[np.isclose(sub["q"], 0.5)].sort_values("target_end_date")
            if plot_median and len(med):
                x = pd.to_datetime(med["target_end_date"]).values
                ax.plot(x, med["value"].values, color=palette[i], lw=2)
                rdt = pd.to_datetime(r)
                ax.axvline(rdt, color=palette[i], ls='--', lw=1)
                ax.text(rdt, ax.get_ylim()[1]*0.95, str(r), color=palette[i], rotation=90,
                        ha='right', va='top', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            if plot_flusight_ensemble:
                for season in seasons:
                    ensemble = load_flusight_ensemble_forecast(season, loc_code, r)
                    if not ensemble.empty:
                        x = ensemble["target_end_date"].values
                        ax.plot(x, ensemble["value"].values, color='#333333', lw=2, ls=':', label='FluSight-ensemble' if i == 0 else '')
                        break

        # Styling - use full location name
        full_name = "United States" if loc_code == 'US' else season_axis.get_location_name(loc_code)
        ax.text(0.02, 0.98, full_name, transform=ax.transAxes, va='top', ha='left', fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_ylim(bottom=0)
        ax.set_xlim(x_left, x_right)
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)
        format_date_axis(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
