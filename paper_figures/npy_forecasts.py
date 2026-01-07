"""
Functions for plotting NPY (full-horizon) forecasts.
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from influpaint.utils import SeasonAxis
from .data_utils import flusight_quantile_pairs, load_ground_truth_cached as load_truth_for_season
from .helpers import (state_to_code, list_inpainting_dirs, parse_date_from_folder,
                      forecast_week_saturdays, format_date_axis)
from .config import SEASON_XLIMS, SHOW_NPY_PAST


def plot_npy_multi_date_two_seasons(base_dir: str, model_id: str, config: str,
                                    season_axis,
                                    seasons=("2023-2024", "2024-2025"),
                                    per_season_pick=4,
                                    state=('US',),
                                    start_date: str = '2023-10-07',
                                    save_path: str | None = None,
                                    n_sample_trajs: int = 10,
                                    plot_median: bool = True):
    """Plot NPY forecasts for multiple dates across two seasons.

    Args:
        base_dir: Base directory containing forecasts
        model_id: Model identifier
        config: Configuration name
        season_axis: SeasonAxis object for location mapping
        seasons: Tuple of season strings
        per_season_pick: Number of forecast dates to pick per season
        state: State code(s) or list of states
        start_date: Start date for x-axis
        save_path: Optional path to save figure
        n_sample_trajs: Number of sample trajectories to plot
        plot_median: Whether to plot median

    Returns:
        matplotlib Figure object
    """
    states = state if isinstance(state, (list, tuple)) else [state]
    nrows, ncols = len(seasons), len(states)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), dpi=200, sharey=False, sharex=False)
    # Normalize axes to 2D array shape (nrows, ncols)
    if nrows == 1 and ncols == 1:
        axes2 = np.array([[axes]])
    elif nrows == 1:
        axes2 = np.array(axes).reshape(1, ncols)
    elif ncols == 1:
        axes2 = np.array(axes).reshape(nrows, 1)
    else:
        axes2 = np.array(axes)
    for iax, season in enumerate(seasons):
        dirs = list_inpainting_dirs(base_dir, model_id, config)
        dirs_with_dates = []
        for d in dirs:
            dd = parse_date_from_folder(os.path.basename(d))
            if dd is None:
                continue
            y = season.split('-')[0]
            if season_axis.get_fluseason_year(pd.to_datetime(dd)) == int(y):
                dirs_with_dates.append((dd, d))
        dirs_with_dates = sorted(dirs_with_dates)[1:]
        if not dirs_with_dates:
            for icol in range(ncols):
                axes2[iax][icol].text(0.5, 0.5, f"{season}: no forecasts", transform=axes2[iax][icol].transAxes, ha='center', va='center')
                axes2[iax][icol].set_axis_off()
            continue
        step = max(1, len(dirs_with_dates) // max(1, per_season_pick))
        picked = dirs_with_dates[::step][:per_season_pick]
        # Fixed bounds per season if configured
        default_left = pd.to_datetime(start_date)
        left_bound = SEASON_XLIMS.get(season, (default_left, None))[0]
        end_year = int(season.split('-')[1])
        default_right = dt.datetime(end_year, 5, 31)
        right_bound = SEASON_XLIMS.get(season, (None, default_right))[1] or default_right
        for icol, st in enumerate(states):
            ax = axes2[iax, icol]
            loc_code = state_to_code(st, season_axis)
            # GT
            gt_df = load_truth_for_season(season)
            if loc_code == 'US':
                # Use national GT directly (do not sum states)
                gt_us = gt_df[gt_df['location'].astype(str) == 'US'].sort_values('date')
                gt_us = gt_us[(gt_us['date'] >= left_bound) & (gt_us['date'] <= right_bound)]
                x_dates = pd.to_datetime(gt_us['date'].values)
                ax.plot(x_dates, gt_us['value'].values, color='k', lw=2)
            else:
                gt = gt_df[gt_df['location'].astype(str) == loc_code].sort_values('date')
                gt = gt[(gt['date'] >= left_bound) & (gt['date'] <= right_bound)]
                x_dates = gt['date'].values
                ax.plot(x_dates, gt['value'], color='k', lw=2)
            # Forecasts
            palette = sns.color_palette('Dark2', n_colors=len(picked))
            for i, (dref, dpath) in enumerate(picked):
                arr = np.load(os.path.join(dpath, 'fluforecasts_ti.npy'))
                if loc_code == 'US':
                    ts = arr[:, 0, :, :len(season_axis.locations)].sum(axis=-1)
                else:
                    place_idx = season_axis.locations.index(loc_code)
                    ts = arr[:, 0, :, place_idx]
                # Get forecast Saturdays via SeasonAxis mapping
                x_weeks = forecast_week_saturdays(season, season_axis, ts.shape[1])
                eff_len = min(len(x_weeks), ts.shape[1])
                dates_plot = pd.to_datetime(x_weeks[:eff_len])
                ts = ts[:, :eff_len]
                # Light sampled trajectories
                if n_sample_trajs and n_sample_trajs > 0:
                    ns = min(n_sample_trajs, ts.shape[0])
                    sample_idxs = np.linspace(0, ts.shape[0]-1, num=ns, dtype=int)
                    # Future trajectories
                    mask_fut = dates_plot >= pd.to_datetime(dref)
                    if np.any(mask_fut):
                        for si in sample_idxs:
                            ax.plot(dates_plot[mask_fut], ts[si, mask_fut], color=palette[i], alpha=0.25, lw=0.7, zorder=1)
                    # Past trajectories (optional)
                    if SHOW_NPY_PAST:
                        mask_past = (dates_plot >= left_bound) & (dates_plot < pd.to_datetime(dref))
                        if np.any(mask_past):
                            for si in sample_idxs:
                                ax.plot(dates_plot[mask_past], ts[si, mask_past], color=palette[i], alpha=0.15, lw=0.6, ls=':', zorder=1)
                for lo, hi in flusight_quantile_pairs:
                    # Future (from forecast start)
                    mask_fut = dates_plot >= pd.to_datetime(dref)
                    x_plot_fut = dates_plot[mask_fut]
                    if len(x_plot_fut) > 0:
                        ylo_f = np.quantile(ts, lo, axis=0)[mask_fut]
                        yhi_f = np.quantile(ts, hi, axis=0)[mask_fut]
                        ax.fill_between(x_plot_fut, ylo_f, yhi_f, color=palette[i], alpha=0.05, lw=0)
                    # Past (before forecast start), optional
                    if SHOW_NPY_PAST:
                        mask_past = (dates_plot >= left_bound) & (dates_plot < pd.to_datetime(dref))
                        x_plot_past = dates_plot[mask_past]
                        if len(x_plot_past) > 0:
                            ylo_p = np.quantile(ts, lo, axis=0)[mask_past]
                            yhi_p = np.quantile(ts, hi, axis=0)[mask_past]
                            ax.fill_between(x_plot_past, ylo_p, yhi_p, color=palette[i], alpha=0.03, lw=0)
                if plot_median:
                    med_all = np.quantile(ts, 0.5, axis=0)
                    # Future median
                    mask_med_f = dates_plot >= pd.to_datetime(dref)
                    x_plot_med_f = dates_plot[mask_med_f]
                    if len(x_plot_med_f) > 0:
                        ax.plot(x_plot_med_f, med_all[mask_med_f], color=palette[i], lw=1.6, zorder=2)
                    # Past median (optional)
                    if SHOW_NPY_PAST:
                        mask_med_p = (dates_plot >= left_bound) & (dates_plot < pd.to_datetime(dref))
                        x_plot_med_p = dates_plot[mask_med_p]
                        if len(x_plot_med_p) > 0:
                            ax.plot(x_plot_med_p, med_all[mask_med_p], color=palette[i], lw=1.0, alpha=0.7, ls=':', zorder=2)
                rdt = pd.to_datetime(dref)
                ax.axvline(rdt, color=palette[i], ls='--', lw=1)
                # annotate forecast date on the dashed line
                ax.text(rdt, ax.get_ylim()[1]*0.95, str(rdt.date()), color=palette[i], rotation=90,
                        ha='right', va='top', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            # Style - use full location name
            full_name = "United States" if loc_code == 'US' else season_axis.get_location_name(loc_code)
            ax.text(0.02, 0.98, full_name, transform=ax.transAxes, va='top', ha='left', fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax, trim=True)
            if icol == 0:
                ax.set_ylabel('Incident flu hospitalizations')
            else:
                ax.set_ylabel('')
            # Use fixed bounds per season
            ax.set_xlim(left_bound, right_bound)
            format_date_axis(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_npy_two_panel_national(base_dir: str, model_id: str, config: str,
                                season_axis,
                                seasons=("2023-2024", "2024-2025"),
                                per_season_pick=4,
                                start_date: str = '2023-10-07',
                                save_path: str | None = None,
                                n_sample_trajs: int = 10,
                                plot_median: bool = True):
    """Plot NPY forecasts for national (US) level across two seasons.

    Args:
        base_dir: Base directory containing forecasts
        model_id: Model identifier
        config: Configuration name
        season_axis: SeasonAxis object for location mapping
        seasons: Tuple of season strings
        per_season_pick: Number of forecast dates to pick per season
        start_date: Start date for x-axis
        save_path: Optional path to save figure
        n_sample_trajs: Number of sample trajectories to plot
        plot_median: Whether to plot median

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, len(seasons), figsize=(14, 5), dpi=200, sharey=False, sharex=False)
    if len(seasons) == 1:
        axes = [axes]
    for iax, season in enumerate(seasons):
        ax = axes[iax]
        dirs = list_inpainting_dirs(base_dir, model_id, config)
        dirs_with_dates = []
        for d in dirs:
            dd = parse_date_from_folder(os.path.basename(d))
            if dd is None:
                continue
            y = season.split('-')[0]
            if season_axis.get_fluseason_year(pd.to_datetime(dd)) == int(y):
                dirs_with_dates.append((dd, d))
        dirs_with_dates = sorted(dirs_with_dates)[1:]
        if not dirs_with_dates:
            ax.set_title(f"{season}: no forecasts found")
            continue
        step = max(1, len(dirs_with_dates) // max(1, per_season_pick))
        picked = dirs_with_dates[::step][:per_season_pick]
        # national GT: use US row directly (no state summing)
        gt_df = load_truth_for_season(season)
        gt_us = gt_df[gt_df['location'].astype(str) == 'US'].sort_values('date')
        left_bound = SEASON_XLIMS.get(season, (pd.to_datetime(start_date), None))[0]
        right_bound = SEASON_XLIMS.get(season, (None, dt.datetime(int(season.split('-')[1]),5,31)))[1]
        gt_us = gt_us[(gt_us['date'] >= left_bound) & (gt_us['date'] <= right_bound)]
        x_dates = pd.to_datetime(gt_us['date'].values)
        ax.plot(x_dates, gt_us['value'].values, color='k', lw=2)
        palette = sns.color_palette('Dark2', n_colors=len(picked))
        for i, (dref, dpath) in enumerate(picked):
            arr = np.load(os.path.join(dpath, 'fluforecasts_ti.npy'))
            nat = arr.sum(axis=-1)[:, 0, :]  # (n_samples, weeks)
            x_weeks = forecast_week_saturdays(season, season_axis, nat.shape[1])
            eff_len = min(len(x_weeks), nat.shape[1])
            dates_plot = pd.to_datetime(x_weeks[:eff_len])
            # Light sampled trajectories
            if n_sample_trajs and n_sample_trajs > 0:
                ns = min(n_sample_trajs, nat.shape[0])
                sample_idxs = np.linspace(0, nat.shape[0]-1, num=ns, dtype=int)
                mask_fut = dates_plot >= pd.to_datetime(dref)
                if np.any(mask_fut):
                    for si in sample_idxs:
                        ax.plot(dates_plot[mask_fut], nat[si, :eff_len][mask_fut], color=palette[i], alpha=0.25, lw=0.7, zorder=1)
                if SHOW_NPY_PAST:
                    mask_past = (dates_plot >= left_bound) & (dates_plot < pd.to_datetime(dref))
                    if np.any(mask_past):
                        for si in sample_idxs:
                            ax.plot(dates_plot[mask_past], nat[si, :eff_len][mask_past], color=palette[i], alpha=0.15, lw=0.6, ls=':', zorder=1)
            for lo, hi in flusight_quantile_pairs:
                # Future
                mask_fut = dates_plot >= pd.to_datetime(dref)
                if np.any(mask_fut):
                    lo_curve = np.quantile(nat[:, :eff_len], lo, axis=0)[mask_fut]
                    hi_curve = np.quantile(nat[:, :eff_len], hi, axis=0)[mask_fut]
                    ax.fill_between(dates_plot[mask_fut], lo_curve, hi_curve, color=palette[i], alpha=0.05, lw=0)
                # Past (optional)
                if SHOW_NPY_PAST:
                    mask_past = (dates_plot >= left_bound) & (dates_plot < pd.to_datetime(dref))
                    if np.any(mask_past):
                        lo_curve_p = np.quantile(nat[:, :eff_len], lo, axis=0)[mask_past]
                        hi_curve_p = np.quantile(nat[:, :eff_len], hi, axis=0)[mask_past]
                        ax.fill_between(dates_plot[mask_past], lo_curve_p, hi_curve_p, color=palette[i], alpha=0.03, lw=0)
            if plot_median:
                # Medians
                mask_med_f = dates_plot >= pd.to_datetime(dref)
                if np.any(mask_med_f):
                    med = np.quantile(nat[:, :eff_len], 0.5, axis=0)[mask_med_f]
                    ax.plot(dates_plot[mask_med_f], med, color=palette[i], lw=1.8, zorder=2)
                if SHOW_NPY_PAST:
                    mask_med_p = (dates_plot >= left_bound) & (dates_plot < pd.to_datetime(dref))
                    if np.any(mask_med_p):
                        med_p = np.quantile(nat[:, :eff_len], 0.5, axis=0)[mask_med_p]
                        ax.plot(dates_plot[mask_med_p], med_p, color=palette[i], lw=1.0, alpha=0.7, ls=':', zorder=2)
            rdt = pd.to_datetime(dref)
            ax.axvline(rdt, color=palette[i], ls='--', lw=1)
            ax.text(rdt, ax.get_ylim()[1]*0.95, str(rdt.date()), color=palette[i], rotation=90,
                    ha='right', va='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)
        if iax == 0:
            ax.set_ylabel('Incidence')
        ax.set_xlim(left_bound, right_bound)
        format_date_axis(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
