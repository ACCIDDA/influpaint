"""
Generate final paneled figures for paper publication.

This module creates multi-panel figures by composing existing plotting functions.
Each figure corresponds to a specific figure number in the paper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from influpaint.utils import SeasonAxis

from .config import (
    BEST_MODEL_ID, BEST_CONFIG, UNCOND_SAMPLES_PATH,
    INPAINTING_BASE, _MODEL_NUM, MAX_LOW_LOCATIONS
)

# Output directory for final figures
FIG_DIR = "influpaint-paper/figures/generated"
os.makedirs(FIG_DIR, exist_ok=True)

from .helpers import load_unconditional_samples
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
    ax.text(x, y, label, transform=ax.transAxes,
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
            ax.set_ylabel('Incidence')
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
        data.append({'Category': 'Expected \n if random', 'Correlation': corr})
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
        order=['Expected \n if random', 'Influpaint', 'Observed'],
        palette=['lightgray', 'skyblue', 'salmon'],
        showfliers=False,
    )
    ax_corr.set_ylabel('Correlation across U.S. states', fontsize=13)
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
            refs = sorted(df["ref"].unique())
            refs = refs[::2]  # pick_every=2
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
                        ax.text(rdt, ymax*0.95, rdt.strftime('%b %Y'), color=palette[j], rotation=90,
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


def figure4_mask_experiments(season_axis):
    """Figure 4: Mask experiments.

    Panel A: CA, Kentucky, MD from missing_nc_season2023
    Panel B: CA, KY, MD from missing_half_subpop_season2024
    Panel C (top): missing_nc_season2023
    Panel C (bottom): missing_il_season2023

    Args:
        season_axis: SeasonAxis object

    Returns:
        Path to saved figure
    """
    print("Generating Figure 4: Mask experiments...")

    MASK_RESULTS_DIR = "from_longleaf/mask_experiments_868_celebahq_noTTJ5/"

    if not os.path.isdir(MASK_RESULTS_DIR):
        print(f"Mask results directory not found: {MASK_RESULTS_DIR}")
        return None

    # Create figure with custom layout
    # 2 rows, 5 columns
    # Top row: 3 states (A) | missing_nc (C top)
    # Bottom row: 3 states (B) | missing_il (C bottom)
    fig = plt.figure(figsize=(25, 10), dpi=200)
    gs = gridspec.GridSpec(2, 5, figure=fig, wspace=0.25, hspace=0.3)

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
    def plot_mask_state(ax, arr, mk, gt, dates, state_idx, state_name, color):
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
        ax.set_title(state_name, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
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
        ax.set_xlabel('Month')
        sns.despine(ax=ax, trim=True)

    # Panel A: missing_nc_season2023 - CA, KY, MD
    mask_name_a = 'missing_nc_season2023'
    subdir_a = os.path.join(MASK_RESULTS_DIR, mask_name_a)
    arr_a = np.load(os.path.join(subdir_a, 'fluforecasts_ti.npy'))
    mk_a = np.load(os.path.join(subdir_a, 'mask.npy'))

    gt_a = ground_truth.GroundTruth(
        season_first_year='2023',
        data_date=dt.datetime.today(),
        mask_date=pd.to_datetime('2025-05-14'),
        channels=CHANNELS,
        image_size=IMAGE_SIZE,
        nogit=True,
    )
    dates_a = pd.to_datetime(gt_a.gt_xarr['date'].values)

    states_a = ['CA', 'KY', 'MD']
    palette_a = sns.color_palette('Set1', n_colors=3)

    for i, st in enumerate(states_a):
        ax = fig.add_subplot(gs[0, i])
        code = state_to_code(st, gt_a.season_setup)
        idx = gt_a.season_setup.locations.index(code)
        state_name = gt_a.season_setup.get_location_name(code)
        plot_mask_state(ax, arr_a, mk_a, gt_a, dates_a, idx, state_name, palette_a[i])

    # Add panel label A
    ax_a0 = fig.axes[0]
    add_panel_label(ax_a0, 'A', x=-0.15, y=1.05)

    # Panel B: missing_half_subpop_season2024 - CA, KY, MD
    mask_name_b = 'missing_half_subpop_season2024'
    subdir_b = os.path.join(MASK_RESULTS_DIR, mask_name_b)
    arr_b = np.load(os.path.join(subdir_b, 'fluforecasts_ti.npy'))
    mk_b = np.load(os.path.join(subdir_b, 'mask.npy'))

    gt_b = ground_truth.GroundTruth(
        season_first_year='2024',
        data_date=dt.datetime.today(),
        mask_date=pd.to_datetime('2025-05-14'),
        channels=CHANNELS,
        image_size=IMAGE_SIZE,
        nogit=True,
    )
    dates_b = pd.to_datetime(gt_b.gt_xarr['date'].values)

    states_b = ['CA', 'KY', 'MD']
    palette_b = sns.color_palette('Set1', n_colors=3)

    for i, st in enumerate(states_b):
        ax = fig.add_subplot(gs[1, i])
        code = state_to_code(st, gt_b.season_setup)
        idx = gt_b.season_setup.locations.index(code)
        state_name = gt_b.season_setup.get_location_name(code)
        plot_mask_state(ax, arr_b, mk_b, gt_b, dates_b, idx, state_name, palette_b[i])

    # Add panel label B
    ax_b0 = fig.axes[3]
    add_panel_label(ax_b0, 'B', x=-0.15, y=1.05)

    # Panel C top: missing_nc_season2023 - NC
    ax_c_top = fig.add_subplot(gs[0, 3:])
    code_nc = state_to_code('NC', gt_a.season_setup)
    idx_nc = gt_a.season_setup.locations.index(code_nc)
    state_name_nc = gt_a.season_setup.get_location_name(code_nc)
    plot_mask_state(ax_c_top, arr_a, mk_a, gt_a, dates_a, idx_nc, state_name_nc, 'purple')

    # Panel C bottom: missing_il_season2023 - IL
    mask_name_c = 'missing_il_season2023'
    subdir_c = os.path.join(MASK_RESULTS_DIR, mask_name_c)
    arr_c = np.load(os.path.join(subdir_c, 'fluforecasts_ti.npy'))
    mk_c = np.load(os.path.join(subdir_c, 'mask.npy'))

    ax_c_bottom = fig.add_subplot(gs[1, 3:])
    code_il = state_to_code('IL', gt_a.season_setup)
    idx_il = gt_a.season_setup.locations.index(code_il)
    state_name_il = gt_a.season_setup.get_location_name(code_il)
    plot_mask_state(ax_c_bottom, arr_c, mk_c, gt_a, dates_a, idx_il, state_name_il, 'orange')

    # Add panel label C to top right
    add_panel_label(ax_c_top, 'C', x=-0.08, y=1.05)

    # Save figure
    save_path = os.path.join(FIG_DIR, f"{_MODEL_NUM}_figure4_mask_experiments.png")
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
    figure3_npy_forecasts_two_seasons(season_axis)
    figure4_mask_experiments(season_axis)

    print("\n" + "="*60)
    print("Final figures generation complete!")
    print(f"All figures saved to: {os.path.abspath(FIG_DIR)}")
    print("="*60)


if __name__ == "__main__":
    main()
