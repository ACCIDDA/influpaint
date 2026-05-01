"""
Generate final paneled figures for paper publication.

This module creates multi-panel figures by composing existing plotting functions.
Each figure corresponds to a specific figure number in the paper.
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from prepare_dataset_for_scoringutils import Config as ScoringConfig

from influpaint.utils import SeasonAxis

from .config import (
    BEST_MODEL_ID, BEST_CONFIG, UNCOND_SAMPLES_PATH,
    INPAINTING_BASE, _MODEL_NUM, MAX_LOW_LOCATIONS
)

# Output directory for final figures
FIG_DIR = "influpaint-paper/figures/generated"
os.makedirs(FIG_DIR, exist_ok=True)

from .helpers import format_count_axis, load_unconditional_samples
from .data_utils import compute_historical_peak_threshold, filter_trajectories_by_peak

from . import unconditional_figures as uncond_figs
from . import correlation_analysis
from . import csv_forecasts
from . import npy_forecasts
from . import mask_experiments


def add_panel_label(ax, label, x=-0.15, y=1.05, fontsize=16, fontweight='bold'):
    """Add a panel label (A, B, C, etc.) to an axis.

    Args:
        ax: Matplotlib axis
        label: Label text (e.g., 'A', 'B', 'C')
        x: x position in axis coordinates
        y: y position in axis coordinates
        fontsize: Font size for label
        fontweight: Font weight for label
    """
    ax.text(x, y, str(label).lower(), transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight, va='top', ha='right')


def figure1_unconditional_with_correlation(season_axis, uncond_samples):
    """Figure 1: Unconditional generation with correlation analysis.

    Top panels (A): Unconditional states with history inlet (all states except NC)
    Bottom left (B): Weekly incidence correlation

    Args:
        season_axis: SeasonAxis object
        uncond_samples: Unconditional samples array

    Returns:
        Path to saved figure
    """
    print("Generating Figure 1: Unconditional with correlation...")

    # States excluding North Carolina
    states = ['CA', 'NY', 'TX', 'FL', 'MT']

    # Recreate the plot in our figure
    # Layout: 5 states + 1 correlation plot in a single row
    fig = plt.figure(figsize=(30, 5), dpi=200)
    gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.3, width_ratios=[1, 1, 1, 1, 1, 0.8])

    # Call the function again to get axes we can embed
    from .data_utils import (
        normalize_samples_shape, get_real_weeks, get_state_timeseries,
        compute_quantile_curves, compute_median
    )
    from .helpers import state_to_code
    import subprocess
    import pandas as pd
    import seaborn as sns
    from .unconditional_figures import add_trajectory_inset

    arr = normalize_samples_shape(uncond_samples)
    real_weeks = get_real_weeks(arr)
    weeks = np.arange(1, real_weeks + 1)

    # Load historical data
    gt_df = pd.read_csv('influpaint/data/nhsn_flusight_past.csv')
    gt_plot_data = {}
    for season in gt_df['fluseason'].unique():
        if season == 2021:
            continue
        season_data = gt_df[gt_df['fluseason'] == season]
        season_pivot = season_data.pivot(columns='location_code', values='value', index='season_week')
        gt_plot_data[season] = season_pivot

    sorted_seasons = sorted(gt_plot_data.keys())
    line_styles = ['-', '--', '-.', ':']
    month_labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    month_weeks = [1, 5, 9, 13, 17, 22, 26, 31, 35, 40, 44, 48]

    axes_top = []
    for i, st in enumerate(states):
        ax = fig.add_subplot(gs[i])
        axes_top.append(ax)

        ts = get_state_timeseries(arr, st, season_axis)
        loc_code = state_to_code(st, season_axis)
        color = sns.color_palette('Set2', n_colors=len(states))[i]

        # Quantile bands
        for lo_curve, hi_curve in compute_quantile_curves(ts):
            ax.fill_between(weeks, lo_curve, hi_curve, color=color, alpha=0.08, lw=0, zorder=0)

        # Historical data
        for j, season_key in enumerate(sorted_seasons):
            season_data = gt_plot_data[season_key]
            if loc_code in season_data.columns:
                gt_series = season_data[loc_code].dropna()
                if not gt_series.empty:
                    ls = line_styles[j % len(line_styles)]
                    season_label = f"{season_key}-{int(season_key)+1}" if i == 0 else None
                    ax.plot(gt_series.index, gt_series.values,
                           color='black', lw=2.0, alpha=0.9, ls=ls, zorder=10,
                           label=season_label)

        # Add trajectory inset
        n_trajs = min(3, ts.shape[0])
        traj_indices = np.linspace(0, ts.shape[0]-1, num=n_trajs, dtype=int)
        inset_trajectories = ts[traj_indices]
        add_trajectory_inset(ax, weeks, inset_trajectories, color)

        state_name = season_axis.get_location_name(loc_code)
        ax.text(0.02, 0.98, state_name, transform=ax.transAxes, va='top', ha='left',
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xlim(1, real_weeks)
        ax.set_ylim(bottom=0)
        ax.set_xticks([month_weeks[j] for j in range(0, len(month_weeks), 2)])
        ax.set_xticklabels([month_labels[j] for j in range(0, len(month_labels), 2)])
        if i == 0:
            ax.set_ylabel('Incident flu hospitalizations')
        format_count_axis(ax)
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)

    # Add panel label A to first top axis
    add_panel_label(axes_top[0], 'A', x=-0.15, y=1.05)

    # Right side: Correlation analysis (last column)
    ax_corr = fig.add_subplot(gs[5])

    # Generate correlation figure but extract the plot
    from .correlation_analysis import (
        compute_random_correlation, compute_weekly_incidence_correlation,
        compute_observed_correlation
    )

    print("Computing correlations for Figure 1...")
    random_corr = compute_random_correlation(uncond_samples, 100)
    influpaint_corr = compute_weekly_incidence_correlation(uncond_samples)
    observed_corr = compute_observed_correlation()

    data = []
    for corr in random_corr:
        data.append({'Category': 'Expected\nif random', 'Correlation': corr})
    for corr in influpaint_corr:
        data.append({'Category': 'Influpaint', 'Correlation': corr})
    for corr in observed_corr:
        data.append({'Category': 'Observed', 'Correlation': corr})

    df = pd.DataFrame(data)

    sns.boxplot(
        data=df,
        x='Category',
        y='Correlation',
        ax=ax_corr,
        order=['Expected\nif random', 'Influpaint', 'Observed'],
        palette=['lightgray', 'skyblue', 'salmon'],
        showfliers=False,
    )
    ax_corr.set_ylabel('Correlation across US states', fontsize=13)
    ax_corr.set_xlabel('')
    ax_corr.grid(True, alpha=0.3, axis='y')
    sns.despine(ax=ax_corr, trim=True)

    # Add panel label B
    add_panel_label(ax_corr, 'B', x=-0.25, y=1.05)

    # Save figure
    save_path = os.path.join(FIG_DIR, f"{_MODEL_NUM}_figure1_unconditional_correlation.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Figure 1 saved to {save_path}")
    return save_path


def figure2_csv_forecasts_two_seasons(season_axis):
    """Figure 2: CSV forecast fans for two seasons.

    Panel A (top): 2023-2024 season
    Panel B (bottom): 2024-2025 season
    Remove USA and California, 4x1 layout (4 states per season)

    Args:
        season_axis: SeasonAxis object

    Returns:
        Path to saved figure
    """
    print("Generating Figure 2: CSV forecasts two seasons...")

    # States: remove US and CA, keep NC, NY, TX, FL
    states = ['NC', 'NY', 'TX', 'FL']

    # Create figure with 2 rows (one per season), 4 columns (one per state)
    # sharex=False because different seasons have different date ranges
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=200, sharex=False, sharey=False)

    from .csv_forecasts import (
        load_truth_for_season, load_flusight_ensemble_forecast,
        list_influpaint_csvs, flusight_quantile_pairs
    )
    from .helpers import state_to_code, format_date_axis
    from .config import SEASON_XLIMS
    import seaborn as sns
    import pandas as pd
    import matplotlib.dates as mdates

    seasons = ['2023-2024', '2024-2025']

    # Load all CSV forecasts once
    csvs = list_influpaint_csvs(INPAINTING_BASE, BEST_MODEL_ID, BEST_CONFIG)
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

    df_all = pd.concat(df_list, ignore_index=True)
    season_reference_dates = {}
    for season in seasons:
        left_bound = SEASON_XLIMS.get(season, (pd.to_datetime('2023-10-07'), None))[0]
        import datetime as dt
        default_right = dt.datetime(int(season.split('-')[1]), 5, 31)
        right_bound = SEASON_XLIMS.get(season, (None, default_right))[1] or default_right
        refs = sorted(
            r for r in df_all["ref"].unique()
            if left_bound <= pd.to_datetime(r) <= right_bound
        )
        season_reference_dates[season] = refs[::2]

    for row_idx, season in enumerate(seasons):
        left_bound = SEASON_XLIMS.get(season, (pd.to_datetime('2023-10-07'), None))[0]
        import datetime as dt
        default_right = dt.datetime(int(season.split('-')[1]), 5, 31)
        right_bound = SEASON_XLIMS.get(season, (None, default_right))[1] or default_right

        for col_idx, st in enumerate(states):
            ax = axes[row_idx, col_idx]
            loc_code = state_to_code(st, season_axis)

            # Ground truth
            gt = load_truth_for_season(season)
            gt = gt[gt["location"].astype(str) == loc_code].sort_values('date')
            gt = gt[(gt['date'] >= left_bound) & (gt['date'] <= right_bound)]
            ax.plot(gt['date'], gt['value'], color='black', lw=2)

            # Forecasts
            df = df_all[(df_all["location"].astype(str) == loc_code) &
                       (df_all["target"] == "wk inc flu hosp") &
                       (df_all["output_type"] == "quantile")]
            refs = season_reference_dates[season]
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
                            ax.fill_between(x[mask], low["value"].values[mask],
                                          up["value"].values[mask],
                                          color=palette[j], alpha=0.08, lw=0)

                # Median
                med = sub[np.isclose(sub["q"], 0.5)].sort_values("target_end_date")
                if len(med):
                    x = pd.to_datetime(med["target_end_date"]).values
                    mask = (x >= np.datetime64(left_bound)) & (x <= np.datetime64(right_bound))
                    if np.any(mask):
                        ax.plot(x[mask], med["value"].values[mask], color=palette[j], lw=2)
                    rdt = pd.to_datetime(r)
                    if left_bound <= rdt <= right_bound:
                        ax.axvline(rdt, color=palette[j], ls='--', lw=1)
                        # Add date label near the top
                        ymax = ax.get_ylim()[1]
                        ax.text(rdt, ymax*0.95, rdt.strftime('%Y-%m-%d'), color=palette[j], rotation=90,
                                ha='right', va='top', fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

                # FluSight-ensemble
                ensemble = load_flusight_ensemble_forecast(season, loc_code, r)
                if not ensemble.empty:
                    x = ensemble["target_end_date"].values
                    mask = (x >= np.datetime64(left_bound)) & (x <= np.datetime64(right_bound))
                    if np.any(mask):
                        ax.plot(x[mask], ensemble["value"].values[mask], color='#333333',
                               lw=2, ls=':', label='FluSight-ensemble' if j == 0 and row_idx == 0 else '')

            # Styling
            full_name = season_axis.get_location_name(loc_code)
            ax.text(0.02, 0.98, full_name, transform=ax.transAxes, va='top', ha='left',
                   fontsize=11, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.set_ylim(bottom=0)
            if col_idx == 0:
                ax.set_ylabel('Incident flu hospitalizations')
            format_count_axis(ax)
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax, trim=True)
            ax.set_xlim(left_bound, right_bound)

            # Custom date formatting: "Dec 25" format with flat labels
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            for label in ax.get_xticklabels():
                label.set_rotation(0)
                label.set_horizontalalignment('center')

    # Add panel labels
    add_panel_label(axes[0, 0], 'A', x=-0.15, y=1.05)
    add_panel_label(axes[1, 0], 'B', x=-0.15, y=1.05)

    # Save figure
    save_path = os.path.join(FIG_DIR, f"{_MODEL_NUM}_figure2_csv_forecasts_two_seasons.png")
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Figure 2 saved to {save_path}")
    return save_path


def figure_relaizedforecast(season_axis):
    """Plot submitted FluSight InfluPaint quantile fans in the Figure 2 layout."""
    print("Generating Relaizedforecast from submitted FluSight InfluPaint forecasts...")

    states = ['NC', 'NY', 'TX', 'FL']
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=200, sharex=False, sharey=False)

    from .csv_forecasts import (
        INFLUPAINT_FLUSIGHT_MODEL,
        load_truth_for_season,
        load_flusight_ensemble_forecast,
        load_flusight_model_forecasts_for_season,
        list_influpaint_csvs,
        flusight_quantile_pairs,
    )
    from .helpers import state_to_code
    from .config import SEASON_XLIMS
    import seaborn as sns
    import pandas as pd
    import matplotlib.dates as mdates

    seasons = ['2023-2024', '2024-2025']
    submissions = {
        season: load_flusight_model_forecasts_for_season(season, INFLUPAINT_FLUSIGHT_MODEL)
        for season in seasons
    }
    csvs = list_influpaint_csvs(INPAINTING_BASE, BEST_MODEL_ID, BEST_CONFIG)
    reference_df_list = []
    for p in csvs:
        dfi = pd.read_csv(p, dtype={"location": str})
        if "reference_date" in dfi.columns:
            dfi["ref"] = pd.to_datetime(dfi["reference_date"]).dt.date
        elif "forecast_date" in dfi.columns:
            dfi["ref"] = pd.to_datetime(dfi["forecast_date"]).dt.date
        else:
            continue
        reference_df_list.append(dfi[["ref"]].copy())
    reference_dates_df = pd.concat(reference_df_list, ignore_index=True)
    season_reference_dates = {}
    for season in seasons:
        left_bound = SEASON_XLIMS[season][0]
        default_right = pd.Timestamp(int(season.split('-')[1]), 5, 31)
        right_bound = SEASON_XLIMS[season][1] or default_right
        refs = sorted(
            r for r in reference_dates_df["ref"].unique()
            if left_bound <= pd.to_datetime(r) <= right_bound
        )
        season_reference_dates[season] = refs[::2]

    for row_idx, season in enumerate(seasons):
        left_bound = SEASON_XLIMS[season][0]
        default_right = pd.Timestamp(int(season.split('-')[1]), 5, 31)
        right_bound = SEASON_XLIMS[season][1] or default_right
        df_season = submissions[season]

        for col_idx, st in enumerate(states):
            ax = axes[row_idx, col_idx]
            loc_code = state_to_code(st, season_axis)

            gt = load_truth_for_season(season)
            gt = gt[gt["location"].astype(str) == loc_code].sort_values('date')
            gt = gt[(gt['date'] >= left_bound) & (gt['date'] <= right_bound)]
            ax.plot(gt['date'], gt['value'], color='black', lw=2)

            df = df_season[
                (df_season["location"].astype(str) == loc_code) &
                (df_season["target"] == "wk inc flu hosp") &
                (df_season["output_type"] == "quantile")
            ]
            refs = season_reference_dates[season]
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
                            ax.fill_between(
                                x[mask],
                                low["value"].values[mask],
                                up["value"].values[mask],
                                color=palette[j],
                                alpha=0.08,
                                lw=0,
                            )

                med = sub[np.isclose(sub["q"], 0.5)].sort_values("target_end_date")
                if len(med):
                    x = pd.to_datetime(med["target_end_date"]).values
                    mask = (x >= np.datetime64(left_bound)) & (x <= np.datetime64(right_bound))
                    if np.any(mask):
                        ax.plot(x[mask], med["value"].values[mask], color=palette[j], lw=2)
                    rdt = pd.to_datetime(r)
                    if left_bound <= rdt <= right_bound:
                        ax.axvline(rdt, color=palette[j], ls='--', lw=1)
                        ymax = ax.get_ylim()[1]
                        ax.text(
                            rdt,
                            ymax * 0.95,
                            rdt.strftime('%Y-%m-%d'),
                            color=palette[j],
                            rotation=90,
                            ha='right',
                            va='top',
                            fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
                        )

                ensemble = load_flusight_ensemble_forecast(season, loc_code, r)
                if not ensemble.empty:
                    x = ensemble["target_end_date"].values
                    mask = (x >= np.datetime64(left_bound)) & (x <= np.datetime64(right_bound))
                    if np.any(mask):
                        ax.plot(
                            x[mask],
                            ensemble["value"].values[mask],
                            color='#333333',
                            lw=2,
                            ls=':',
                            label='FluSight-ensemble' if j == 0 and row_idx == 0 else '',
                        )

            full_name = season_axis.get_location_name(loc_code)
            ax.text(
                0.02,
                0.98,
                full_name,
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=11,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            )
            ax.set_ylim(bottom=0)
            if col_idx == 0:
                ax.set_ylabel('Incident flu hospitalizations')
            format_count_axis(ax)
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax, trim=True)
            ax.set_xlim(left_bound, right_bound)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            for label in ax.get_xticklabels():
                label.set_rotation(0)
                label.set_horizontalalignment('center')

    add_panel_label(axes[0, 0], 'A', x=-0.15, y=1.05)
    add_panel_label(axes[1, 0], 'B', x=-0.15, y=1.05)

    save_path = os.path.join(FIG_DIR, f"{_MODEL_NUM}_Relaizedforecast.png")
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Relaizedforecast saved to {save_path}")
    return save_path


def figure3_npy_forecasts_two_seasons(season_axis):
    """Figure 3: NPY forecasts for two seasons.

    Same as 868_forecast_npy_two_panel_states.png but remove North Carolina.
    Add A and B labels for each season panel.

    Args:
        season_axis: SeasonAxis object

    Returns:
        Path to saved figure
    """
    print("Generating Figure 3: NPY forecasts two seasons...")

    # States: remove NC, keep US, CA, NY, TX
    states = ['US', 'CA', 'NY', 'TX']

    fig = npy_forecasts.plot_npy_multi_date_two_seasons(
        base_dir=INPAINTING_BASE,
        model_id=BEST_MODEL_ID,
        config=BEST_CONFIG,
        season_axis=season_axis,
        seasons=("2023-2024", "2024-2025"),
        per_season_pick=4,
        state=states,
        save_path=None,
        plot_median=False,
    )

    # Add panel labels to the figure
    # The figure has 2 rows (seasons) and len(states) columns
    axes = fig.get_axes()

    # Find the first axis in each row
    ncols = len(states)

    # Top row (2023-2024) - first axis
    add_panel_label(axes[0], 'A', x=-0.15, y=1.05)

    # Bottom row (2024-2025) - first axis in second row
    add_panel_label(axes[ncols], 'B', x=-0.15, y=1.05)

    # Update date formatting for all axes: "Dec 25" format with flat labels
    import matplotlib.dates as mdates
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        for label in ax.get_xticklabels():
            label.set_rotation(0)
            label.set_horizontalalignment('center')

    # Save figure
    save_path = os.path.join(FIG_DIR, f"{_MODEL_NUM}_figure3_npy_forecasts_two_seasons.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Figure 3 saved to {save_path}")
    return save_path


def figure3_ratio_flusight_over_influpaint(season_axis):
    """Figure 3 companion analysis: WIS ratio (FluSight / Influpaint) for 1-4 week ahead.

    Uses the same two seasons, states, and reference-date selection logic as Figure 3.

    Args:
        season_axis: SeasonAxis object

    Returns:
        Path to saved ratio plot
    """
    print("Generating Figure 3 ratio analysis: FluSight vs Influpaint...")

    import datetime as dt
    import pandas as pd
    import seaborn as sns
    from .csv_forecasts import FLUSIGHT_BASES
    from .helpers import (
        list_inpainting_dirs,
        parse_date_from_folder,
        state_to_code,
        forecast_week_saturdays,
    )

    target_name = ScoringConfig.TARGET_NAME
    seasons = ["2023-2024", "2024-2025"]
    states = ["US", "CA", "NY", "TX"]
    per_season_pick = 4
    horizons = [0, 1, 2, 3]
    required_quantiles = np.array(ScoringConfig.REQUIRED_QUANTILES, dtype=float)
    required_quantiles_rounded = sorted(np.round(required_quantiles, 6).tolist())

    out_combined = os.path.join(
        FIG_DIR,
        f"{_MODEL_NUM}_figure3_ratio_rescore_combined_forecast_truth.csv",
    )
    out_scores = os.path.join(
        FIG_DIR,
        f"{_MODEL_NUM}_figure3_ratio_rescore_scoringutils_scores.csv",
    )
    out_plot = os.path.join(FIG_DIR, f"{_MODEL_NUM}_figure3_ratio_flusight_over_influpaint.png")
    out_table = os.path.join(FIG_DIR, f"{_MODEL_NUM}_figure3_ratio_flusight_over_influpaint.csv")
    out_panel = os.path.join(FIG_DIR, f"{_MODEL_NUM}_figure3_ratio_panel_summary.csv")
    out_overall = os.path.join(FIG_DIR, f"{_MODEL_NUM}_figure3_ratio_overall_summary.csv")

    dirs = list_inpainting_dirs(INPAINTING_BASE, BEST_MODEL_ID, BEST_CONFIG)
    picked_rows = []
    picked_paths = []
    for season in seasons:
        season_year = int(season.split("-")[0])
        dated_dirs = []
        for d in dirs:
            ref_date = parse_date_from_folder(os.path.basename(d))
            if ref_date is None:
                raise ValueError(f"Could not parse reference date from inpainting directory: {d}")
            if season_axis.get_fluseason_year(pd.to_datetime(ref_date)) == season_year:
                dated_dirs.append((ref_date, d))
        dated_dirs = sorted(dated_dirs)[1:]
        step = max(1, len(dated_dirs) // per_season_pick)
        picked = dated_dirs[::step][:per_season_pick]
        if len(picked) != per_season_pick:
            raise ValueError(f"Expected {per_season_pick} picked dates for {season}, found {len(picked)}")
        for ref_date, dpath in picked:
            ref_date_str = pd.to_datetime(ref_date).strftime("%Y-%m-%d")
            picked_rows.append((season, ref_date_str))
            picked_paths.append((season, ref_date, dpath))

    picked_dates = pd.DataFrame(picked_rows, columns=["season", "reference_date"])
    if len(picked_dates) != len(seasons) * per_season_pick:
        raise ValueError("Figure 3 reference-date selection did not return expected date count")

    loc_map = {state: state_to_code(state, season_axis) for state in states}
    inv_loc_map = {code: state for state, code in loc_map.items()}

    influpaint_model_names = []
    influpaint_rows = []
    for season, ref_date, dpath in picked_paths:
        base_name = os.path.basename(dpath)
        parts = base_name.split("::")
        if len(parts) < 3:
            raise ValueError(f"Unexpected inpainting directory format: {base_name}")
        run_model_name = "::".join(parts[:-1]).replace("::conf_", "::")
        influpaint_model_names.append(run_model_name)

        arr = np.load(os.path.join(dpath, "fluforecasts_ti.npy"))
        forecast_dates = pd.to_datetime(
            forecast_week_saturdays(season, season_axis, arr.shape[2])
        ).dt.date
        date_to_idx = {d: i for i, d in enumerate(forecast_dates)}

        for state in states:
            loc_code = loc_map[state]
            if loc_code == "US":
                ts = arr[:, 0, :, :len(season_axis.locations)].sum(axis=-1)
            else:
                state_idx = season_axis.locations.index(loc_code)
                ts = arr[:, 0, :, state_idx]

            for horizon in horizons:
                target_date = ref_date + dt.timedelta(days=7 * horizon)
                if target_date not in date_to_idx:
                    raise ValueError(
                        f"Target date {target_date} (season={season}, ref={ref_date}, horizon={horizon}) not in NPY week grid"
                    )
                target_idx = date_to_idx[target_date]
                quantile_values = np.quantile(ts[:, target_idx], required_quantiles)
                for q, pred in zip(required_quantiles, quantile_values):
                    influpaint_rows.append(
                        {
                            "model": run_model_name,
                            "group": "influpaint",
                            "season": season,
                            "reference_date": str(ref_date),
                            "forecast_date": str(ref_date),
                            "target_end_date": str(target_date),
                            "location": loc_code,
                            "horizon": int(horizon),
                            "quantile": float(q),
                            "predicted": float(pred),
                            "target": target_name,
                            "output_type": "quantile",
                        }
                    )

    influpaint_model_names = sorted(set(influpaint_model_names))
    if len(influpaint_model_names) != 1:
        raise ValueError(f"Expected one Influpaint model in picked directories, found {influpaint_model_names}")
    influpaint_model = influpaint_model_names[0]

    expected_per_model_rows = (
        len(seasons) * per_season_pick * len(states) * len(horizons) * len(required_quantiles)
    )
    if len(influpaint_rows) != expected_per_model_rows:
        raise ValueError(
            f"Influpaint row count mismatch: expected {expected_per_model_rows}, found {len(influpaint_rows)}"
        )
    influpaint_df = pd.DataFrame(influpaint_rows)

    flusight_rows = []
    for season, ref_date, _ in picked_paths:
        fs_path = os.path.join(
            FLUSIGHT_BASES[season],
            "model-output",
            "FluSight-ensemble",
            f"{ref_date}-FluSight-ensemble.csv",
        )
        if not os.path.exists(fs_path):
            raise FileNotFoundError(f"Missing FluSight ensemble file: {fs_path}")
        fs = pd.read_csv(fs_path, dtype={"location": str})
        fs["target_end_date"] = pd.to_datetime(fs["target_end_date"]).dt.date
        fs["horizon"] = pd.to_numeric(fs["horizon"], errors="coerce")
        fs["output_type_id"] = pd.to_numeric(fs["output_type_id"], errors="coerce")

        for state in states:
            loc_code = loc_map[state]
            sub = fs[
                (fs["location"] == loc_code)
                & (fs["target"] == target_name)
                & (fs["output_type"] == "quantile")
                & (fs["horizon"].isin(horizons))
            ].copy()
            if sub.empty:
                raise ValueError(
                    f"No FluSight quantile rows for season={season}, ref={ref_date}, location={loc_code}"
                )
            sub["quantile"] = sub["output_type_id"].astype(float)
            # Snap to canonical quantile grid to avoid floating-representation drift.
            snapped = []
            for q in sub["quantile"].to_numpy(dtype=float):
                nearest = float(required_quantiles[np.argmin(np.abs(required_quantiles - q))])
                if abs(nearest - q) > 1e-6:
                    raise ValueError(
                        f"FluSight quantile {q} is not on required grid (season={season}, ref={ref_date}, location={loc_code})"
                    )
                snapped.append(nearest)
            sub["quantile"] = snapped
            have = sorted(np.round(sub["quantile"].unique(), 6).tolist())
            if have != required_quantiles_rounded:
                raise ValueError(
                    f"Quantiles mismatch for season={season}, ref={ref_date}, location={loc_code}: have {have}"
                )
            for horizon in horizons:
                sub_h = sub[sub["horizon"] == horizon]
                if len(sub_h) != len(required_quantiles):
                    raise ValueError(
                        f"Expected {len(required_quantiles)} FluSight quantiles for season={season}, ref={ref_date}, location={loc_code}, horizon={horizon}; found {len(sub_h)}"
                    )
                expected_target_date = ref_date + dt.timedelta(days=7 * horizon)
                unique_target_dates = sorted(sub_h["target_end_date"].unique().tolist())
                if unique_target_dates != [expected_target_date]:
                    raise ValueError(
                        f"Unexpected FluSight target date for season={season}, ref={ref_date}, location={loc_code}, horizon={horizon}: {unique_target_dates}"
                    )
            for _, row in sub.iterrows():
                flusight_rows.append(
                    {
                        "model": "FluSight-ensemble",
                        "group": "flusight",
                        "season": season,
                        "reference_date": str(ref_date),
                        "forecast_date": str(ref_date),
                        "target_end_date": str(row["target_end_date"]),
                        "location": loc_code,
                        "horizon": int(row["horizon"]),
                        "quantile": float(row["quantile"]),
                        "predicted": float(row["value"]),
                        "target": target_name,
                        "output_type": "quantile",
                    }
                )

    if len(flusight_rows) != expected_per_model_rows:
        raise ValueError(
            f"FluSight row count mismatch: expected {expected_per_model_rows}, found {len(flusight_rows)}"
        )
    flusight_df = pd.DataFrame(flusight_rows)

    forecasts = pd.concat([influpaint_df, flusight_df], ignore_index=True)
    forecasts["season"] = forecasts["season"].astype(str)
    forecasts["location"] = forecasts["location"].astype(str)
    forecasts["target_end_date"] = pd.to_datetime(forecasts["target_end_date"]).dt.strftime("%Y-%m-%d")

    truth_rows = []
    for season in seasons:
        truth_path = os.path.join(
            FLUSIGHT_BASES[season], "target-data", "target-hospital-admissions.csv"
        )
        if not os.path.exists(truth_path):
            raise FileNotFoundError(f"Missing truth file: {truth_path}")
        truth = pd.read_csv(truth_path, dtype={"location": str})
        truth["target_end_date"] = pd.to_datetime(truth["date"]).dt.date
        truth["observed"] = truth["value"].astype(float)
        truth_rows.append(truth[["target_end_date", "location", "observed"]].assign(season=season))
    truth_all = pd.concat(truth_rows, ignore_index=True)
    truth_all["season"] = truth_all["season"].astype(str)
    truth_all["location"] = truth_all["location"].astype(str)
    truth_all["target_end_date"] = pd.to_datetime(truth_all["target_end_date"]).dt.strftime("%Y-%m-%d")

    combined = forecasts.merge(
        truth_all,
        on=["season", "target_end_date", "location"],
        how="inner",
    )
    if len(combined) != len(forecasts):
        raise ValueError(
            f"Merge lost forecast rows: forecasts={len(forecasts)}, merged={len(combined)}"
        )
    combined.to_csv(out_combined, index=False)

    score_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "score_with_scoringutils.R"))
    score_cmd = ["Rscript", score_script, out_combined, out_scores]
    proc = subprocess.run(score_cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Exact scoringutils scoring failed.\n"
            f"Command: {' '.join(score_cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())

    scores = pd.read_csv(out_scores, dtype={"location": str})
    required_score_cols = {"model", "group", "season", "reference_date", "location", "horizon", "wis"}
    missing_cols = required_score_cols.difference(scores.columns)
    if missing_cols:
        raise ValueError(
            "scoringutils output is missing required columns. "
            f"Missing: {sorted(missing_cols)}. "
            "This usually means scoring failed due malformed quantile groups."
        )
    scores["reference_date"] = pd.to_datetime(scores["reference_date"]).dt.strftime("%Y-%m-%d")
    scores["location"] = scores["location"].astype(str)

    influpaint = scores[
        (scores["model"] == influpaint_model) & (scores["horizon"].isin(horizons))
    ].rename(columns={"wis": "wis_influpaint"})
    influpaint = influpaint[["season", "reference_date", "location", "horizon", "wis_influpaint"]]

    flusight = scores[
        (scores["model"] == "FluSight-ensemble") & (scores["horizon"].isin(horizons))
    ].rename(columns={"wis": "wis_flusight"})
    flusight = flusight[["season", "reference_date", "location", "horizon", "wis_flusight"]]

    merged = influpaint.merge(
        flusight, on=["season", "reference_date", "location", "horizon"], how="inner"
    )
    merged = merged[merged["location"].isin(loc_map.values())].copy()
    merged["location_name"] = merged["location"].map(inv_loc_map)
    merged["week_ahead"] = merged["horizon"] + 1
    merged["ratio_flusight_over_influpaint"] = merged["wis_flusight"] / merged["wis_influpaint"]
    merged = merged.sort_values(["season", "location_name", "reference_date", "week_ahead"])

    expected_rows = len(seasons) * len(states) * per_season_pick * len(horizons)
    if len(merged) != expected_rows:
        raise ValueError(f"Expected {expected_rows} ratio rows, found {len(merged)}")

    merged.to_csv(out_table, index=False)
    panel_summary = merged.groupby(["season", "location_name"])["ratio_flusight_over_influpaint"].agg(
        ["mean", "median", "min", "max"]
    )
    panel_summary.reset_index().to_csv(out_panel, index=False)
    overall_summary = pd.DataFrame(
        [
            {
                "overall_mean_ratio": merged["ratio_flusight_over_influpaint"].mean(),
                "overall_median_ratio": merged["ratio_flusight_over_influpaint"].median(),
                "ratio_q1": merged["ratio_flusight_over_influpaint"].quantile(0.25),
                "ratio_q3": merged["ratio_flusight_over_influpaint"].quantile(0.75),
                "influpaint_better_count": int((merged["ratio_flusight_over_influpaint"] > 1).sum()),
                "flusight_better_count": int((merged["ratio_flusight_over_influpaint"] < 1).sum()),
                "n_total": int(len(merged)),
            }
        ]
    )
    overall_summary.to_csv(out_overall, index=False)

    panel_order = []
    for season in seasons:
        for state in states:
            panel_order.append(f"{season} | {state}")
    merged["panel"] = merged["season"].astype(str) + " | " + merged["location_name"].astype(str)
    merged["panel"] = pd.Categorical(merged["panel"], categories=panel_order, ordered=True)

    date_order = sorted(merged["reference_date"].unique())
    palette = sns.color_palette("Dark2", n_colors=len(date_order))
    color_map = {date: palette[i] for i, date in enumerate(date_order)}

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=250, sharex=True, sharey=True)
    flat_axes = axes.flatten()
    for idx, panel in enumerate(panel_order):
        ax = flat_axes[idx]
        panel_df = merged[merged["panel"] == panel]
        for ref_date in sorted(panel_df["reference_date"].unique()):
            series = panel_df[panel_df["reference_date"] == ref_date].sort_values("week_ahead")
            ax.plot(
                series["week_ahead"],
                series["ratio_flusight_over_influpaint"],
                marker="o",
                lw=1.5,
                color=color_map[ref_date],
            )
        ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.set_title(panel, fontsize=10)
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(True, alpha=0.25)

    for ax in flat_axes[4:]:
        ax.set_xlabel("Week ahead")
    flat_axes[0].set_ylabel("WIS ratio (FluSight / Influpaint)")
    flat_axes[4].set_ylabel("WIS ratio (FluSight / Influpaint)")

    legend_handles = []
    legend_labels = []
    for ref_date in date_order:
        line, = flat_axes[0].plot([], [], color=color_map[ref_date], marker="o", lw=1.5)
        legend_handles.append(line)
        legend_labels.append(ref_date)
    fig.legend(
        legend_handles,
        legend_labels,
        title="Reference date",
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_plot, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure 3 ratio plot saved to {out_plot}")
    print(f"Figure 3 ratio table saved to {out_table}")
    print(f"Figure 3 ratio panel summary saved to {out_panel}")
    print(f"Figure 3 ratio overall summary saved to {out_overall}")
    print(f"Figure 3 rescoring combined file saved to {out_combined}")
    print(f"Figure 3 rescoring score file saved to {out_scores}")
    return out_plot


def figure4_mask_experiments(season_axis, season_first_year='2024', output_suffix=''):
    """Figure 4: Mask experiments.

    Top row: CA/FL/MD from missing_half_subpop + NC from missing_nc
    Bottom row: IL from missing_il, FL from missing_past,
    MD from midseason_biggap, CA from checkerboard_4x4.

    Args:
        season_axis: SeasonAxis object
        season_first_year: Season first year as string (e.g. '2024', '2023')
        output_suffix: Optional suffix appended to output filename

    Returns:
        Path to saved figure
    """
    print("Generating Figure 4: Mask experiments...")

    MASK_RESULTS_DIR = "from_longleaf/mask_experiments_868_celebahq_noTTJ5/"

    if not os.path.isdir(MASK_RESULTS_DIR):
        print(f"Mask results directory not found: {MASK_RESULTS_DIR}")
        return None

    # Create figure with uniform layout and shared x-axis.
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=200, sharex=True, sharey=False)

    # We'll manually plot the mask experiments using the same logic as mask_experiments.py
    from .mask_experiments import add_mask_heatmap_inset
    from influpaint.utils import ground_truth
    from influpaint.utils.helpers import flusight_quantile_pairs
    from .helpers import state_to_code
    from .config import IMAGE_SIZE, CHANNELS
    import seaborn as sns
    import pandas as pd
    import datetime as dt

    # Helper to plot a single mask experiment state
    def plot_mask_state(ax, arr, mk, gt, dates, state_idx, state_name, color, show_ylabel=False):
        # Add mask heatmap inset
        p_len = len(gt.season_setup.locations)
        add_mask_heatmap_inset(ax, gt.gt_xarr.data[0], mk[0], state_idx, p_len)

        # Plot ground truth
        gt_series = gt.gt_xarr.data[0, :, state_idx]
        ax.plot(dates, gt_series, color='k', lw=1.5, label='Ground truth')

        ts = arr[:, 0, :, state_idx]

        # Sample trajectories
        n_sample_trajs = 10
        if n_sample_trajs > 0:
            ns = min(n_sample_trajs, ts.shape[0])
            sample_idxs = np.linspace(0, ts.shape[0]-1, num=ns, dtype=int)
            keep = mk[0, :ts.shape[1], state_idx]
            for si in sample_idxs:
                y = ts[si, :len(dates)].copy()
                y[keep == 1] = np.nan
                ax.plot(dates[:len(y)], y, color=color, alpha=0.25, lw=0.7)

        # Quantile fans
        for lo, hi in flusight_quantile_pairs:
            lo_curve = np.quantile(ts, lo, axis=0)
            hi_curve = np.quantile(ts, hi, axis=0)
            keepw = mk[0, :len(lo_curve), state_idx]
            lo_curve = lo_curve.copy()
            hi_curve = hi_curve.copy()
            lo_curve[keepw == 1] = np.nan
            hi_curve[keepw == 1] = np.nan
            ax.fill_between(dates[:len(lo_curve)], lo_curve, hi_curve,
                           color=color, alpha=0.06, lw=0)

        # Median
        med = np.quantile(ts, 0.5, axis=0)
        med_masked = med.copy()
        med_masked[mk[0, :len(med), state_idx] == 1] = np.nan
        ax.plot(dates[:len(med_masked)], med_masked, color=color, lw=1.8)

        # Styling
        ax.text(0.02, 0.98, state_name, transform=ax.transAxes, va='top', ha='left',
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_ylim(bottom=0)
        format_count_axis(ax)
        ax.grid(True, alpha=0.3)
        if show_ylabel:
            ax.set_ylabel('Incident flu hospitalizations')

        month_labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
        month_positions_weeks = [1, 5, 9, 13, 17, 22, 26, 31, 35, 40, 44, 48]
        tick_dates = []
        tick_labels_to_show = []
        for k in range(0, len(month_positions_weeks), 2):
            week_idx = month_positions_weeks[k]
            if week_idx < len(dates):
                tick_dates.append(dates[week_idx])
                tick_labels_to_show.append(month_labels[k])
        ax.set_xticks(tick_dates)
        ax.set_xticklabels(tick_labels_to_show, rotation=0, ha='center')
        sns.despine(ax=ax, trim=True)

    def _masked_series(ts, mk, state_idx, n_weeks):
        masked_idx = np.where(mk[0, :n_weeks, state_idx] == 0)[0]
        return ts[:, masked_idx], masked_idx

    def _bootstrap_median_ci(values, n_boot=2000, alpha=0.05):
        n = values.shape[0]
        boot = np.empty(n_boot)
        for b in range(n_boot):
            idx = np.random.randint(0, n, size=n)
            boot[b] = np.median(values[idx])
        lo = np.percentile(boot, 100 * (alpha / 2))
        hi = np.percentile(boot, 100 * (1 - alpha / 2))
        return lo, hi

    def compute_panel_metrics(arr, mk, gt, state_idx):
        ts = arr[:, 0, :, state_idx]
        n_weeks = min(ts.shape[1], gt.shape[0])
        gt_use = gt[:n_weeks]
        ts_use = ts[:, :n_weeks]
        ts_masked, masked_idx = _masked_series(ts_use, mk, state_idx, n_weeks)
        gt_masked = gt_use[masked_idx]

        q25 = np.quantile(ts_masked, 0.25, axis=0)
        q75 = np.quantile(ts_masked, 0.75, axis=0)
        cov50 = np.mean((gt_masked >= q25) & (gt_masked <= q75))

        actual_peak_local = int(np.argmax(gt_masked))
        actual_peak_week = int(masked_idx[actual_peak_local])
        actual_peak_size = float(gt_masked[actual_peak_local])

        pred_peak_local = np.argmax(ts_masked, axis=1)
        pred_peak_week = masked_idx[pred_peak_local]
        pred_peak_size = np.max(ts_masked, axis=1)

        timing_err = pred_peak_week - actual_peak_week
        size_err = pred_peak_size - actual_peak_size
        timing_err_pct = 100.0 * timing_err / max(actual_peak_week, 1)
        size_err_pct = 100.0 * size_err / max(actual_peak_size, 1e-8)

        timing_med = float(np.median(timing_err))
        timing_lo, timing_hi = _bootstrap_median_ci(timing_err)
        timing_pct_med = float(np.median(timing_err_pct))
        timing_pct_lo, timing_pct_hi = _bootstrap_median_ci(timing_err_pct)
        size_med = float(np.median(size_err))
        size_lo, size_hi = _bootstrap_median_ci(size_err)
        size_pct_med = float(np.median(size_err_pct))
        size_pct_lo, size_pct_hi = _bootstrap_median_ci(size_err_pct)
        pred_peak_week_med = float(np.median(pred_peak_week))
        pred_peak_size_med = float(np.median(pred_peak_size))

        return {
            "cov50": float(cov50),
            "actual_peak_week_idx": int(actual_peak_week),
            "pred_peak_week_idx_median": pred_peak_week_med,
            "timing_med": timing_med,
            "timing_ci": (float(timing_lo), float(timing_hi)),
            "timing_pct_med": timing_pct_med,
            "timing_pct_ci": (float(timing_pct_lo), float(timing_pct_hi)),
            "actual_peak_size": actual_peak_size,
            "pred_peak_size_median": pred_peak_size_med,
            "size_med": size_med,
            "size_ci": (float(size_lo), float(size_hi)),
            "size_pct_med": size_pct_med,
            "size_pct_ci": (float(size_pct_lo), float(size_pct_hi)),
        }

    gt = ground_truth.GroundTruth.for_flusight(
        season_first_year=str(season_first_year),
        data_date=dt.datetime.today(),
        mask_date=pd.to_datetime(f"{int(season_first_year) + 1}-05-14"),
        channels=CHANNELS,
        image_size=IMAGE_SIZE,
        nogit=True,
    )
    dates = pd.to_datetime(gt.gt_xarr['date'].values)

    def load_mask(mask_name):
        subdir = os.path.join(MASK_RESULTS_DIR, mask_name)
        arr = np.load(os.path.join(subdir, 'fluforecasts_ti.npy'))
        mk = np.load(os.path.join(subdir, 'mask.npy'))
        return arr, mk

    # Fixed one-color-per-state palette across the whole figure.
    state_colors = {
        'CA': '#1b9e77',
        'FL': '#d95f02',
        'MD': '#7570b3',
        'NC': '#e7298a',
        'IL': '#66a61e',
    }

    # Explicit panel mapping in exact narrative order:
    # A.1-A.3 half-map, B leave-one-state-out (NC), C leave-one-state-out (IL),
    # D midseason gap, E past-only, F checkerboard.
    panel_defs = [
        {'label': 'A.1', 'row': 0, 'col': 0, 'state': 'CA',
         'mask_name': f'missing_half_subpop_season{season_first_year}', 'color': state_colors['CA']},
        {'label': 'A.2', 'row': 0, 'col': 1, 'state': 'FL',
         'mask_name': f'missing_half_subpop_season{season_first_year}', 'color': state_colors['FL']},
        {'label': 'A.3', 'row': 0, 'col': 2, 'state': 'MD',
         'mask_name': f'missing_half_subpop_season{season_first_year}', 'color': state_colors['MD']},
        {'label': 'B', 'row': 0, 'col': 3, 'state': 'NC',
         'mask_name': f'missing_nc_season{season_first_year}', 'color': state_colors['NC']},
        {'label': 'C', 'row': 1, 'col': 0, 'state': 'IL',
         'mask_name': f'missing_il_season{season_first_year}', 'color': state_colors['IL']},
        {'label': 'D', 'row': 1, 'col': 1, 'state': 'MD',
         'mask_name': f'missing_midseason_biggap_season{season_first_year}', 'color': state_colors['MD']},
        {'label': 'E', 'row': 1, 'col': 2, 'state': 'FL',
         'mask_name': f'missing_past_season{season_first_year}', 'color': state_colors['FL']},
        {'label': 'F', 'row': 1, 'col': 3, 'state': 'CA',
         'mask_name': f'missing_checkerboard_4x4_season{season_first_year}', 'color': state_colors['CA']},
    ]

    loaded_masks = {}
    for panel in panel_defs:
        if panel['mask_name'] not in loaded_masks:
            loaded_masks[panel['mask_name']] = load_mask(panel['mask_name'])
        arr, mk = loaded_masks[panel['mask_name']]
        ax = axes[panel['row'], panel['col']]
        code = state_to_code(panel['state'], gt.season_setup)
        idx = gt.season_setup.locations.index(code)
        state_name = gt.season_setup.get_location_name(code)
        plot_mask_state(
            ax, arr, mk, gt, dates, idx, state_name, panel['color'],
            show_ylabel=(panel['row'] == 0 and panel['col'] == 0),
        )

    x_axis_year = int(season_first_year)
    for col in range(4):
        axes[1, col].set_xlabel(f'{x_axis_year}-{x_axis_year + 1}')

    # Panel labels in reading order:
    # top row: A.1, A.2, A.3, B
    # bottom row: C, D, E, F
    for panel in panel_defs:
        add_panel_label(axes[panel['row'], panel['col']], panel['label'], x=-0.15, y=1.05)

    # Quantitative results printed under Figure 4 generation.
    panel_metrics = {}
    for panel in panel_defs:
        arr, mk = loaded_masks[panel['mask_name']]
        code = state_to_code(panel['state'], gt.season_setup)
        idx = gt.season_setup.locations.index(code)
        gt_series = gt.gt_xarr.data[0, :, idx]
        panel_metrics[panel['label']] = compute_panel_metrics(arr, mk, gt_series, idx)

    print("\nFigure 4 results:")
    for lbl in ['A.1', 'A.2', 'A.3']:
        m = panel_metrics[lbl]
        print(
            f"  {lbl}: peak timing idx actual={m['actual_peak_week_idx']}, "
            f"pred_median={m['pred_peak_week_idx_median']:.2f}; "
            f"median peak timing error (weeks)={m['timing_med']:.2f} "
            f"[95% CI {m['timing_ci'][0]:.2f}, {m['timing_ci'][1]:.2f}], "
            f"median peak timing error (%)={m['timing_pct_med']:.2f} "
            f"[95% CI {m['timing_pct_ci'][0]:.2f}, {m['timing_pct_ci'][1]:.2f}], "
            f"peak size actual={m['actual_peak_size']:.2f}, pred_median={m['pred_peak_size_median']:.2f}; "
            f"median peak size error={m['size_med']:.2f} "
            f"[95% CI {m['size_ci'][0]:.2f}, {m['size_ci'][1]:.2f}], "
            f"median peak size error (%)={m['size_pct_med']:.2f} "
            f"[95% CI {m['size_pct_ci'][0]:.2f}, {m['size_pct_ci'][1]:.2f}], "
            f"50% coverage={m['cov50']:.3f}"
        )
    for lbl in ['B', 'C']:
        m = panel_metrics[lbl]
        print(f"  {lbl}: 50% coverage={m['cov50']:.3f}")
    for lbl in ['D', 'E']:
        m = panel_metrics[lbl]
        print(
            f"  {lbl}: peak timing idx actual={m['actual_peak_week_idx']}, "
            f"pred_median={m['pred_peak_week_idx_median']:.2f}; "
            f"median peak timing error (weeks)={m['timing_med']:.2f} "
            f"[95% CI {m['timing_ci'][0]:.2f}, {m['timing_ci'][1]:.2f}], "
            f"median peak timing error (%)={m['timing_pct_med']:.2f} "
            f"[95% CI {m['timing_pct_ci'][0]:.2f}, {m['timing_pct_ci'][1]:.2f}], "
            f"peak size actual={m['actual_peak_size']:.2f}, pred_median={m['pred_peak_size_median']:.2f}; "
            f"median peak size error={m['size_med']:.2f} "
            f"[95% CI {m['size_ci'][0]:.2f}, {m['size_ci'][1]:.2f}]"
            f", median peak size error (%)={m['size_pct_med']:.2f} "
            f"[95% CI {m['size_pct_ci'][0]:.2f}, {m['size_pct_ci'][1]:.2f}]"
        )

    # Save figure
    save_path = os.path.join(FIG_DIR, f"{_MODEL_NUM}_figure4_mask_experiments{output_suffix}.png")
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Figure 4 saved to {save_path}")
    return save_path


def main():
    """Generate all final paneled figures for the paper."""
    print("="*60)
    print("Influpaint Final Paneled Figures Generation")
    print("="*60)

    # Setup
    print("\nSetting up...")
    season_axis = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
    uncond_samples = load_unconditional_samples(UNCOND_SAMPLES_PATH)
    print(f"Loaded unconditional samples: {uncond_samples.shape}")

    # Filter unconditional samples
    print("\nFiltering unconditional samples...")
    peak_thresholds = compute_historical_peak_threshold(
        season_axis=season_axis,
        seasons=[2022, 2023, 2024],
        threshold_fraction=0.1,
    )
    uncond_samples_filtered = filter_trajectories_by_peak(
        uncond_samples,
        season_axis,
        peak_thresholds,
        max_low_locations=MAX_LOW_LOCATIONS
    )

    # Generate figures
    figure1_unconditional_with_correlation(season_axis, uncond_samples_filtered)
    figure2_csv_forecasts_two_seasons(season_axis)
    figure_relaizedforecast(season_axis)
    figure3_npy_forecasts_two_seasons(season_axis)
    figure3_ratio_flusight_over_influpaint(season_axis)
    figure4_mask_experiments(season_axis, season_first_year='2023')

    print("\n" + "="*60)
    print("Final figures generation complete!")
    print(f"All figures saved to: {os.path.abspath(FIG_DIR)}")
    print("="*60)


if __name__ == "__main__":
    main()
