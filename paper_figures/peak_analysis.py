"""
Functions for analyzing peak distributions in influenza data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from influpaint.utils import SeasonAxis
from .data_utils import normalize_samples_shape, get_real_weeks, get_state_timeseries
from .helpers import state_to_code


def plot_peak_distributions_comparison(inv_samples: np.ndarray,
                                       season_axis: SeasonAxis,
                                       save_path: str | None = None,
                                       prominence_threshold: float = 50.0):
    """Compare peak timing and size distributions between generated and historical seasons.

    Creates 2-panel figure showing density curves for peak characteristics aggregated across all locations.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object
        save_path: optional path to save the figure
        prominence_threshold: minimum prominence for peak detection

    Returns:
        matplotlib Figure object
    """
    arr = normalize_samples_shape(inv_samples)
    n, c, w, p = arr.shape
    real_weeks = get_real_weeks(arr)

    gt_df = pd.read_csv('influpaint/data/nhsn_flusight_past.csv')

    historical_peaks = {'timing': [], 'size': [], 'season': []}

    for season_year in sorted(gt_df['fluseason'].unique()):
        if season_year == 2021:  # Skip 2021-2022 season (incomplete data)
            continue
        season_data = gt_df[gt_df['fluseason'] == season_year]
        season_pivot = season_data.pivot(columns='location_code', values='value', index='season_week')
        season_label = f"{season_year}-{season_year+1}"

        for loc in season_pivot.columns:
            series = season_pivot[loc].dropna().values
            if len(series) < 5:
                continue
            peaks, properties = find_peaks(series, prominence=prominence_threshold)
            for peak_idx in peaks:
                historical_peaks['timing'].append(season_pivot.index[peak_idx])
                historical_peaks['size'].append(series[peak_idx])
                historical_peaks['season'].append(season_label)

    generated_peaks = {'timing': [], 'size': []}

    for sample_idx in range(n):
        for loc_idx in range(len(season_axis.locations)):
            series = arr[sample_idx, 0, :real_weeks, loc_idx]
            if np.all(np.isnan(series)):
                continue
            peaks, properties = find_peaks(series, prominence=prominence_threshold)
            for peak_idx in peaks:
                generated_peaks['timing'].append(peak_idx + 1)
                generated_peaks['size'].append(series[peak_idx])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    seasons = sorted(set(historical_peaks['season']))
    colors = sns.color_palette('Set2', n_colors=len(seasons))
    line_styles = ['--', '-.', ':']

    if len(historical_peaks['timing']) > 0:
        for i, season in enumerate(seasons):
            mask = [s == season for s in historical_peaks['season']]
            timing_data = [historical_peaks['timing'][j] for j in range(len(mask)) if mask[j]]
            size_data = [historical_peaks['size'][j] for j in range(len(mask)) if mask[j]]

            if len(timing_data) > 5:
                kde_timing = gaussian_kde(timing_data, bw_method=0.5)
                x_timing = np.linspace(1, real_weeks, 200)
                axes[0].plot(x_timing, kde_timing(x_timing),
                           color=colors[i], ls=line_styles[i % len(line_styles)],
                           lw=1.5, alpha=0.8, label=season)

            if len(size_data) > 5:
                log_size_data = np.log10(size_data)
                kde_size = gaussian_kde(log_size_data, bw_method=0.3)
                x_size_log = np.linspace(min(log_size_data), max(log_size_data) * 1.1, 200)
                axes[1].plot(10**x_size_log, kde_size(x_size_log),
                           color=colors[i], ls=line_styles[i % len(line_styles)],
                           lw=1.5, alpha=0.8, label=season)

    if len(generated_peaks['timing']) > 5:
        kde_timing = gaussian_kde(generated_peaks['timing'], bw_method=0.5)
        x_timing = np.linspace(1, real_weeks, 200)
        axes[0].plot(x_timing, kde_timing(x_timing),
                   color='black', ls='-', lw=2.5, alpha=1.0, label='Generated')

    if len(generated_peaks['size']) > 5:
        log_gen_size = np.log10(generated_peaks['size'])
        kde_size = gaussian_kde(log_gen_size, bw_method=0.3)
        x_size_log = np.linspace(min(log_gen_size), max(log_gen_size) * 1.1, 200)
        axes[1].plot(10**x_size_log, kde_size(x_size_log),
                   color='black', ls='-', lw=2.5, alpha=1.0, label='Generated')

    axes[0].set_xlabel('Peak timing (season week)')
    axes[0].set_ylabel('Density')
    axes[0].set_xlim(1, real_weeks)
    axes[0].grid(True, alpha=0.3)
    sns.despine(ax=axes[0], trim=True)

    axes[1].set_xlabel('Peak size (log incidence)')
    axes[1].set_ylabel('Density')
    axes[1].set_xscale('log')
    axes[1].set_xlim(left=10)
    axes[1].grid(True, alpha=0.3)
    sns.despine(ax=axes[1], trim=True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_peak_distributions_by_location(inv_samples: np.ndarray,
                                         season_axis: SeasonAxis,
                                         states: list[str],
                                         save_path: str | None = None,
                                         prominence_threshold: float = 50.0):
    """Compare peak timing and size distributions for specific locations.

    Creates multi-panel figure showing density curves for peak characteristics per location.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object
        states: list of state codes/abbrevs
        save_path: optional path to save the figure
        prominence_threshold: minimum prominence for peak detection

    Returns:
        matplotlib Figure object
    """
    arr = normalize_samples_shape(inv_samples)
    n, c, w, p = arr.shape
    real_weeks = get_real_weeks(arr)

    gt_df = pd.read_csv('influpaint/data/nhsn_flusight_past.csv')

    n_states = len(states)
    fig, axes = plt.subplots(n_states, 2, figsize=(12, 5*n_states), dpi=200,
                            sharex='col')
    if n_states == 1:
        axes = axes.reshape(1, -1)

    for i_state, state in enumerate(states):
        loc_code = state_to_code(state, season_axis)
        place_idx = season_axis.locations.index(loc_code)

        historical_peaks = {'timing': [], 'size': [], 'season': []}

        for season_year in sorted(gt_df['fluseason'].unique()):
            if season_year == 2021:  # Skip 2021-2022 season (incomplete data)
                continue
            season_data = gt_df[gt_df['fluseason'] == season_year]
            season_pivot = season_data.pivot(columns='location_code', values='value', index='season_week')
            season_label = f"{season_year}-{season_year+1}"

            if loc_code in season_pivot.columns:
                series = season_pivot[loc_code].dropna().values
                if len(series) < 5:
                    continue
                peaks, properties = find_peaks(series, prominence=prominence_threshold)
                for peak_idx in peaks:
                    historical_peaks['timing'].append(season_pivot.index[peak_idx])
                    historical_peaks['size'].append(series[peak_idx])
                    historical_peaks['season'].append(season_label)

        generated_peaks = {'timing': [], 'size': []}

        for sample_idx in range(n):
            series = arr[sample_idx, 0, :real_weeks, place_idx]
            if np.all(np.isnan(series)):
                continue
            peaks, properties = find_peaks(series, prominence=prominence_threshold)
            for peak_idx in peaks:
                generated_peaks['timing'].append(peak_idx + 1)
                generated_peaks['size'].append(series[peak_idx])

        seasons = sorted(set(historical_peaks['season']))
        colors = sns.color_palette('Set2', n_colors=len(seasons))
        line_styles = ['--', '-.', ':']

        ax_timing = axes[i_state, 0]
        ax_size = axes[i_state, 1]

        if len(historical_peaks['timing']) > 0:
            for j, season in enumerate(seasons):
                mask = [s == season for s in historical_peaks['season']]
                timing_data = [historical_peaks['timing'][k] for k in range(len(mask)) if mask[k]]
                size_data = [historical_peaks['size'][k] for k in range(len(mask)) if mask[k]]

                for idx, timing_val in enumerate(timing_data):
                    ax_timing.axvline(timing_val, color=colors[j], ls=line_styles[j % len(line_styles)],
                                    lw=2.0, alpha=0.8, label=season if idx == 0 else None, zorder=10)

                for idx, size_val in enumerate(size_data):
                    ax_size.axvline(size_val, color=colors[j], ls=line_styles[j % len(line_styles)],
                                  lw=2.0, alpha=0.8, label=season if idx == 0 else None, zorder=10)

        if len(generated_peaks['timing']) > 5:
            kde_timing = gaussian_kde(generated_peaks['timing'], bw_method=0.5)
            x_timing = np.linspace(1, real_weeks, 200)
            ax_timing.plot(x_timing, kde_timing(x_timing),
                       color='black', ls='-', lw=2.5, alpha=1.0, label='Generated')

        if len(generated_peaks['size']) > 5:
            log_gen_size = np.log10(generated_peaks['size'])
            kde_size = gaussian_kde(log_gen_size, bw_method=0.3)
            x_size_log = np.linspace(min(log_gen_size), max(log_gen_size) * 1.1, 200)
            ax_size.plot(10**x_size_log, kde_size(x_size_log),
                       color='black', ls='-', lw=2.5, alpha=1.0, label='Generated')

        ax_timing.text(0.02, 0.98, state.upper(), transform=ax_timing.transAxes, va='top', ha='left',
                fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax_timing.set_xlabel('Peak timing (season week)')
        ax_timing.set_ylabel('Density')
        ax_timing.set_xlim(1, real_weeks)
        ax_timing.grid(True, alpha=0.3)
        sns.despine(ax=ax_timing, trim=True)

        ax_size.text(0.02, 0.98, state.upper(), transform=ax_size.transAxes, va='top', ha='left',
                fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax_size.set_xlabel('Peak size (log incidence)')
        ax_size.set_ylabel('Density')
        ax_size.set_xscale('log')
        ax_size.set_xlim(left=10)
        ax_size.grid(True, alpha=0.3)
        sns.despine(ax=ax_size, trim=True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_peak_distributions_by_metric(inv_samples: np.ndarray,
                                       season_axis: SeasonAxis,
                                       states: list[str],
                                       metric: str,
                                       save_path: str | None = None,
                                       prominence_threshold: float = 50.0):
    """Compare peak distributions for multiple states, grouped by metric (timing or size).

    Creates 1-row × N-column figure with vertical swarmplots for one metric across states.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object
        states: list of state codes/abbrevs
        metric: 'timing' or 'size'
        save_path: optional path to save the figure
        prominence_threshold: minimum prominence for peak detection

    Returns:
        matplotlib Figure object
    """
    arr = normalize_samples_shape(inv_samples)
    n, c, w, p = arr.shape
    real_weeks = get_real_weeks(arr)

    gt_df = pd.read_csv('influpaint/data/nhsn_flusight_past.csv')

    n_states = len(states)
    fig, axes = plt.subplots(1, n_states, figsize=(3.75*n_states, 5), dpi=200, sharey=True)
    if n_states == 1:
        axes = [axes]

    for state_idx, state in enumerate(states):
        ax = axes[state_idx]
        ts = get_state_timeseries(arr, state, season_axis)
        loc_code = state_to_code(state, season_axis)
        place_idx = season_axis.locations.index(loc_code)

        # Extract historical peaks for this state
        historical_peaks = {'timing': [], 'size': [], 'season': []}

        for season_year in sorted(gt_df['fluseason'].unique()):
            if season_year == 2021:  # Skip 2021-2022 season (incomplete data)
                continue
            season_data = gt_df[gt_df['fluseason'] == season_year]
            season_pivot = season_data.pivot(columns='location_code', values='value', index='season_week')
            season_label = f"{season_year}-{season_year+1}"

            if loc_code in season_pivot.columns:
                series = season_pivot[loc_code].dropna().values
                if len(series) < 5:
                    continue
                peaks, properties = find_peaks(series, prominence=prominence_threshold)
                for peak_idx in peaks:
                    historical_peaks['timing'].append(season_pivot.index[peak_idx])
                    historical_peaks['size'].append(series[peak_idx])
                    historical_peaks['season'].append(season_label)

        # Extract generated peaks for this state
        generated_peaks = {'timing': [], 'size': []}

        for sample_idx in range(n):
            series = arr[sample_idx, 0, :real_weeks, place_idx]
            if np.all(np.isnan(series)):
                continue
            peaks, properties = find_peaks(series, prominence=prominence_threshold)
            for peak_idx in peaks:
                generated_peaks['timing'].append(peak_idx + 1)
                generated_peaks['size'].append(series[peak_idx])

        # Plot historical peaks as horizontal lines with labels
        seasons = sorted(set(historical_peaks['season']))
        colors = sns.color_palette('Set2', n_colors=len(seasons))
        line_styles = ['--', '-.', ':']

        if len(historical_peaks['timing']) > 0:
            for j, season in enumerate(seasons):
                mask = [s == season for s in historical_peaks['season']]

                if metric == 'timing':
                    data = [historical_peaks['timing'][k] for k in range(len(mask)) if mask[k]]
                    for idx, val in enumerate(data):
                        ax.axhline(val, color=colors[j], ls=line_styles[j % len(line_styles)],
                                  lw=2.0, alpha=0.9, zorder=10)
                        if idx == 0:
                            ax.text(0.85, val + 1.5, season, va='bottom', ha='right',
                                   fontsize=9, color=colors[j],
                                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1.5))

                elif metric == 'size':
                    data = [historical_peaks['size'][k] for k in range(len(mask)) if mask[k]]
                    for idx, val in enumerate(data):
                        ax.axhline(val, color=colors[j], ls=line_styles[j % len(line_styles)],
                                  lw=2.0, alpha=0.9, zorder=10)
                        if idx == 0:
                            ax.text(0.85, val * 1.15, season, va='bottom', ha='right',
                                   fontsize=9, color=colors[j],
                                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1.5))

        # Plot generated peaks as swarmplot
        if metric == 'timing' and len(generated_peaks['timing']) > 5:
            timing_values = np.array(generated_peaks['timing'])
            timing_jittered = timing_values + np.random.uniform(-0.4, 0.4, size=len(timing_values))
            df = pd.DataFrame({'type': 'Generated', 'value': timing_jittered})
            sns.swarmplot(data=df, y='value', x='type', ax=ax,
                         color='black', alpha=0.8, size=2.5, zorder=1)
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.set_ylim(1, real_weeks)
            ax.set_ylabel('Season week' if state_idx == 0 else '')
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax, trim=True, bottom=True)

        elif metric == 'size' and len(generated_peaks['size']) > 5:
            df = pd.DataFrame({'type': 'Generated', 'value': generated_peaks['size']})
            sns.swarmplot(data=df, y='value', x='type', ax=ax,
                         color='black', alpha=0.8, size=2.5, zorder=1)
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.set_yscale('log')
            ax.set_ylim(bottom=10)
            ax.set_ylabel('Log incidence' if state_idx == 0 else '')
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax, trim=True, bottom=True)

        # Set title - use full location name from SeasonAxis
        state_name = season_axis.get_location_name(loc_code)
        if state_idx == 0:
            metric_title = 'Peak timing' if metric == 'timing' else 'Peak size'
            ax.set_title(f'{metric_title}\n{state_name}', fontsize=12, fontweight='bold')
        else:
            ax.set_title(state_name, fontsize=12, fontweight='bold')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
