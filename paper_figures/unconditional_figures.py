"""
Functions for generating unconditional figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from influpaint.utils import SeasonAxis
import influpaint.utils.plotting as idplots
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .data_utils import (
    normalize_samples_shape, get_real_weeks, get_state_timeseries,
    compute_quantile_curves, compute_median, flusight_quantile_pairs,
    get_state_labels
)
from .helpers import state_to_code


def plot_unconditional_states_quantiles_and_trajs(inv_samples: np.ndarray,
                                                  season_axis: SeasonAxis,
                                                  states: list[str],
                                                  n_sample_trajs: int = 10,
                                                  plot_median: bool = True,
                                                  save_path: str | None = None):
    """Plot unconditional sample trajectories and quantile fans for given states.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object for location mapping
        states: list of state codes/abbrevs (expected ~5)
        n_sample_trajs: number of light sample lines to overlay per state
        plot_median: toggle median line overlay
        save_path: optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    # Normalize shape
    arr = normalize_samples_shape(inv_samples)
    real_weeks = get_real_weeks(arr)
    weeks = np.arange(1, real_weeks + 1)

    n_states = len(states)
    ncols = n_states if n_states <= 5 else int(np.ceil(n_states / 2))
    nrows = 1 if n_states <= 5 else 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), dpi=200, sharey=False, sharex=True)
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, st in enumerate(states):
        ax = axes_list[i]
        ts = get_state_timeseries(arr, st, season_axis)
        color = sns.color_palette('Set2', n_colors=n_states)[i % n_states]

        # Light sampled trajectories
        if n_sample_trajs and n_sample_trajs > 0:
            ns = min(n_sample_trajs, ts.shape[0])
            sample_idxs = np.linspace(0, ts.shape[0]-1, num=ns, dtype=int)
            for si in sample_idxs:
                ax.plot(weeks, ts[si], color=color, alpha=0.25, lw=0.8, zorder=1)

        # Quantile bands
        for lo_curve, hi_curve in compute_quantile_curves(ts):
            ax.fill_between(weeks, lo_curve, hi_curve, color=color, alpha=0.08, lw=0)

        # Median
        if plot_median:
            med = compute_median(ts)
            ax.plot(weeks, med, color=color, lw=1.8, zorder=2)

        # Styling - use full location name from SeasonAxis
        loc_code = state_to_code(st, season_axis)
        full_name = season_axis.get_location_name(loc_code)
        ax.text(0.02, 0.98, full_name, transform=ax.transAxes, va='top', ha='left',
                fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xlim(1, real_weeks)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Epiweek')
        if i % ncols == 0:
            ax.set_ylabel('Incidence')
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)

    # Hide any unused axes
    for j in range(len(axes_list)):
        if j >= n_states:
            axes_list[j].set_axis_off()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def fig_unconditional_3d_heat_ridges(inv_samples: np.ndarray,
                                     season_axis: SeasonAxis,
                                     states: list[str] | None = None,
                                     stat: str = 'median',
                                     cmap: str = 'Reds',
                                     elev: float = 35,
                                     azim: float = -60,
                                     heatmap_mode: str = 'mean',
                                     sample_idx: int = 0,
                                     surface_zoffset_ratio: float = 0.0,
                                     ridge_offset_ratio: float = 0.005,
                                     location_stride: int = 1,
                                     fill_ridges: bool = True,
                                     fill_alpha: float = 0.35,
                                     save_path: str | None = None):
    """3D figure with bottom heatmap (mean across samples) and 3D ridgelines.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object
        states: optional list of state abbrevs/codes to overlay as ridges
        stat: 'median' or 'mean' used for ridge z-values
        cmap: colormap for bottom heatmap
        elev: elevation angle for 3D view
        azim: azimuth angle for 3D view
        heatmap_mode: 'mean' or 'sample' for heatmap data
        sample_idx: which sample to use if heatmap_mode='sample'
        surface_zoffset_ratio: z-offset ratio for surface
        ridge_offset_ratio: offset ratio for ridges
        location_stride: stride for location selection when states is None
        fill_ridges: whether to fill under ridges
        fill_alpha: transparency of ridge fill
        save_path: optional path to save the figure

    Returns:
        Tuple of (figure, axis)
    """
    # Normalize shape
    arr = normalize_samples_shape(inv_samples)
    n, c, w, p = arr.shape
    P = len(season_axis.locations)
    real_weeks = get_real_weeks(arr)

    # Compute heatmap
    if heatmap_mode == 'sample':
        sample_idx = int(np.clip(sample_idx, 0, n-1))
        heat = arr[sample_idx, 0, :real_weeks, :P]
    else:
        heat = arr[:, 0, :real_weeks, :P].mean(axis=0)

    # Build grid for bottom surface
    x_vals = np.arange(1, real_weeks + 1)
    y_vals = np.arange(P)
    X, Y = np.meshgrid(x_vals, y_vals)
    zmax_global = float(np.nanmax(heat)) if np.isfinite(heat).all() else 1.0
    zmax_global = max(1.0, zmax_global)
    Z0 = np.zeros_like(X, dtype=float) - (surface_zoffset_ratio * zmax_global if surface_zoffset_ratio else 0.0)
    Cdata = heat.T

    # Normalize colors
    cmap_obj = plt.cm.get_cmap(cmap)
    norm = plt.Normalize(vmin=np.nanmin(Cdata), vmax=np.nanmax(Cdata) if np.nanmax(Cdata) > 0 else 1.0)
    facecolors = cmap_obj(norm(Cdata))

    # Figure and 3D axis
    fig = plt.figure(figsize=(12, 7), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    # Plot bottom colored surface
    surf = ax.plot_surface(X, Y, Z0, rstride=1, cstride=1,
                           facecolors=facecolors[:-1, :-1], shade=False,
                           linewidth=0, antialiased=False, alpha=1)
    surf.set_zsort('max')

    # Choose locations for ridges
    if states:
        place_idxs = [season_axis.locations.index(state_to_code(s, season_axis)) for s in states]
        labels = [s.upper() for s in states]
    else:
        stride = max(1, int(location_stride))
        place_idxs = list(range(0, P, stride))[1:-1]
        labels = get_state_labels(place_idxs, season_axis)

    # Palette for ridges
    ridge_colors = sns.color_palette('Set2', n_colors=len(place_idxs))

    # Plot ridges
    ridge_offset = ridge_offset_ratio * zmax_global
    for j, (pi, lab) in enumerate(zip(place_idxs, labels)):
        ts = arr[:, 0, :real_weeks, pi]
        z = np.nanmean(ts, axis=0) if stat == 'mean' else np.nanmedian(ts, axis=0)
        y_curve = np.full_like(x_vals, fill_value=pi)

        if fill_ridges:
            for k in range(len(x_vals)):
                ax.plot([x_vals[k], x_vals[k]], [pi, pi], [ridge_offset * 0.1, z[k] + ridge_offset],
                       color=ridge_colors[j], alpha=fill_alpha, lw=1.5, zorder=50-pi)

        ax.plot(x_vals, y_curve, z + ridge_offset, color=ridge_colors[j], lw=2.0,
               zorder=100-pi, marker='.', markersize=4)
        ax.text(x_vals[-1]+0.5, pi, (z[-1] + ridge_offset), lab, color=ridge_colors[j],
               fontsize=9, ha='left', va='center')

    # Aesthetics
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('Epiweek')
    ax.set_ylabel('Location index')
    ax.set_zlabel('Incidence')
    ax.set_xlim(1, real_weeks)
    ax.set_ylim(0, P-1)
    ax.set_zlim(bottom=0)
    ax.grid(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def fig_unconditional_3d_heat_ridges_plotly(inv_samples: np.ndarray,
                                            season_axis: SeasonAxis,
                                            states: list[str] | None = None,
                                            stat: str = 'median',
                                            heatmap_mode: str = 'mean',
                                            sample_idx: int = 0,
                                            surface_opacity: float = 0.6,
                                            ridge_lift: float = 0.0,
                                            location_stride: int = 1,
                                            camera_eye: tuple[float, float, float] | None = None,
                                            save_path_html: str | None = None):
    """Interactive Plotly version of 3D heatmap + ridgelines.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object
        states: optional list of state abbrevs/codes
        stat: 'median' or 'mean' for ridge z-values
        heatmap_mode: 'mean' or 'sample'
        sample_idx: which sample if heatmap_mode='sample'
        surface_opacity: transparency of surface
        ridge_lift: vertical offset for ridges
        location_stride: stride for location selection
        camera_eye: tuple of (x, y, z) for camera position
        save_path_html: path to save interactive HTML

    Returns:
        Plotly figure object
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as e:
        raise RuntimeError("Plotly is required for this function. Please install plotly.") from e

    # Normalize shape
    arr = normalize_samples_shape(inv_samples)
    n, c, w, p = arr.shape
    P = len(season_axis.locations)
    real_weeks = get_real_weeks(arr)

    # Heatmap data
    if heatmap_mode == 'sample':
        sample_idx = int(np.clip(sample_idx, 0, n-1))
        heat = arr[sample_idx, 0, :real_weeks, :P]
    else:
        heat = arr[:, 0, :real_weeks, :P].mean(axis=0)

    weeks = np.arange(1, real_weeks + 1)
    y_idx = np.arange(P)
    Z = heat.T

    # Build figure with surface
    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=weeks, y=y_idx,
                             colorscale='Reds', showscale=True,
                             opacity=surface_opacity))

    # Choose locations for ridges
    if states:
        place_idxs = [season_axis.locations.index(state_to_code(s, season_axis)) for s in states]
        labels = [s.upper() for s in states]
    else:
        stride = max(1, int(location_stride))
        place_idxs = list(range(0, P, stride))
        if len(place_idxs) > 2:
            place_idxs = place_idxs[1:-1]
        labels = [str(l).upper() for l in get_state_labels(place_idxs, season_axis)]

    # Add ridge lines
    for pi, lab in zip(place_idxs, labels):
        ts = arr[:, 0, :real_weeks, pi]
        z = np.nanmean(ts, axis=0) if stat == 'mean' else np.nanmedian(ts, axis=0)
        fig.add_trace(go.Scatter3d(x=weeks, y=np.full_like(weeks, pi), z=z + ridge_lift,
                                   mode='lines', name=lab, line=dict(width=4)))

    # Layout and camera
    if camera_eye is None:
        camera_eye = (1.0, -1.6, 0.6)

    fig.update_layout(
        scene=dict(
            xaxis_title='Epiweek',
            yaxis_title='Location index',
            zaxis_title='Incidence',
            camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]))
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title=None
    )

    if save_path_html:
        pio.write_html(fig, file=save_path_html, include_plotlyjs=True, full_html=True)
        print(f"Saved Plotly HTML to {save_path_html}" if os.path.exists(save_path_html)
              else f"Failed to save Plotly HTML to {save_path_html}")
    return fig


def plot_unconditional_states_with_history(inv_samples: np.ndarray,
                                           season_axis: SeasonAxis,
                                           states: list[str],
                                           n_sample_trajs: int = 10,
                                           plot_median: bool = True,
                                           save_path: str | None = None):
    """Plot unconditional samples with historical data overlay for selected states.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object
        states: list of state codes/abbrevs
        n_sample_trajs: number of sample trajectories to plot
        plot_median: whether to plot median
        save_path: optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    arr = normalize_samples_shape(inv_samples)
    real_weeks = get_real_weeks(arr)
    weeks = np.arange(1, real_weeks + 1)

    n_states = len(states)
    ncols = min(3, n_states)
    nrows = int(np.ceil(n_states / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), dpi=200, sharey=False, sharex=True)
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Load historical data (excluding 2021-2022 season due to incomplete data)
    gt_df = pd.read_csv('influpaint/data/nhsn_flusight_past.csv')
    gt_plot_data = {}
    for season in gt_df['fluseason'].unique():
        if season == 2021:  # Skip 2021-2022 season (incomplete data)
            continue
        season_data = gt_df[gt_df['fluseason'] == season]
        season_pivot = season_data.pivot(columns='location_code', values='value', index='season_week')
        gt_plot_data[season] = season_pivot

    sorted_seasons = sorted(gt_plot_data.keys())
    line_styles = ['-', '--', '-.', ':']
    month_labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    month_weeks = [1, 5, 9, 13, 17, 22, 26, 31, 35, 40, 44, 48]

    for i, st in enumerate(states):
        ax = axes_list[i]
        ts = get_state_timeseries(arr, st, season_axis)
        loc_code = state_to_code(st, season_axis)
        color = sns.color_palette('Set2', n_colors=n_states)[i % n_states]

        # Quantile bands
        for lo_curve, hi_curve in compute_quantile_curves(ts):
            ax.fill_between(weeks, lo_curve, hi_curve, color=color, alpha=0.08, lw=0, zorder=0)

        # Sample trajectories
        if n_sample_trajs and n_sample_trajs > 0:
            ns = min(n_sample_trajs, ts.shape[0])
            sample_idxs = np.linspace(0, ts.shape[0]-1, num=ns, dtype=int)
            for si in sample_idxs:
                ax.plot(weeks, ts[si], color=color, alpha=0.8, lw=1.8, zorder=1)

        # Median
        if plot_median:
            med = compute_median(ts)
            ax.plot(weeks, med, color=color, lw=2.5, zorder=2)

        # Historical data
        for j, season_key in enumerate(sorted_seasons):
            season_data = gt_plot_data[season_key]
            if loc_code in season_data.columns:
                gt_series = season_data[loc_code].dropna()
                if not gt_series.empty:
                    ls = line_styles[j % len(line_styles)]
                    ax.plot(gt_series.index, gt_series.values,
                           color='black', lw=2.0, alpha=0.9, ls=ls, zorder=10,
                           label=season_key if i == 0 else None)

        state_name = season_axis.get_location_name(loc_code)
        ax.text(0.02, 0.98, state_name, transform=ax.transAxes, va='top', ha='left',
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xlim(1, real_weeks)
        ax.set_ylim(bottom=0)
        ax.set_xticks([month_weeks[j] for j in range(0, len(month_weeks), 2)])
        ax.set_xticklabels([month_labels[j] for j in range(0, len(month_labels), 2)])
        if i % ncols == 0:
            ax.set_ylabel('Incidence')
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)


    for j in range(len(axes_list)):
        if j >= n_states:
            axes_list[j].set_axis_off()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def add_trajectory_inset(ax, weeks, trajectories, color):
    """Add a small inset showing multiple trajectories.

    Args:
        ax: Matplotlib axis to add inset to
        weeks: Array of week numbers
        trajectories: Array of trajectories to plot (n_trajectories, n_weeks)
        color: Color for the trajectories
    """
    # Create inset axis in upper right (50% bigger: 48.75% size)
    axins = inset_axes(ax, width="43%", height="43%", loc='upper right',
                      bbox_to_anchor=(0, 0, 0.98, 0.98), bbox_transform=ax.transAxes)

    # Plot all trajectories
    for trajectory in trajectories:
        axins.plot(weeks, trajectory, color=color, lw=1.5, alpha=0.7)

    # Calculate ylim based on all trajectories
    all_max = np.max(trajectories)
    axins.set_xlim(weeks[0], weeks[-1])
    axins.set_ylim(bottom=0, top=all_max * 1.1)

    # Add 3 ticks on x and y axes
    x_ticks = [weeks[0], weeks[len(weeks)//2], weeks[-1]]
    axins.set_xticks(x_ticks)
    axins.set_xticklabels([str(int(t)) for t in x_ticks], fontsize=8)

    y_max = all_max * 1.1
    y_ticks = [0, y_max/2, y_max]
    axins.set_yticks(y_ticks)
    axins.set_yticklabels([f'{int(t)}' for t in y_ticks], fontsize=8)

    axins.grid(True, alpha=0.3)

    # Keep spines visible for clarity with ticks
    for spine in axins.spines.values():
        spine.set_linewidth(0.5)


def plot_unconditional_states_with_history_inlet(inv_samples: np.ndarray,
                                                   season_axis: SeasonAxis,
                                                   states: list[str],
                                                   n_inset_trajs: int = 3,
                                                   plot_median: bool = False,
                                                   save_path: str | None = None):
    """Plot unconditional samples with historical data and sample trajectories in an inset.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object
        states: list of state codes/abbrevs
        n_inset_trajs: number of sample trajectories to show in inset
        plot_median: whether to plot median on main graph
        save_path: optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    arr = normalize_samples_shape(inv_samples)
    real_weeks = get_real_weeks(arr)
    weeks = np.arange(1, real_weeks + 1)

    n_states = len(states)
    ncols = min(3, n_states)
    nrows = int(np.ceil(n_states / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), dpi=200, sharey=False, sharex=True)
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Load historical data (excluding 2021-2022 season due to incomplete data)
    gt_df = pd.read_csv('influpaint/data/nhsn_flusight_past.csv')
    gt_plot_data = {}
    for season in gt_df['fluseason'].unique():
        if season == 2021:  # Skip 2021-2022 season (incomplete data)
            continue
        season_data = gt_df[gt_df['fluseason'] == season]
        season_pivot = season_data.pivot(columns='location_code', values='value', index='season_week')
        gt_plot_data[season] = season_pivot

    sorted_seasons = sorted(gt_plot_data.keys())
    line_styles = ['-', '--', '-.', ':']
    month_labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    month_weeks = [1, 5, 9, 13, 17, 22, 26, 31, 35, 40, 44, 48]

    for i, st in enumerate(states):
        ax = axes_list[i]
        ts = get_state_timeseries(arr, st, season_axis)
        loc_code = state_to_code(st, season_axis)
        color = sns.color_palette('Set2', n_colors=n_states)[i % n_states]

        # Quantile bands only
        for lo_curve, hi_curve in compute_quantile_curves(ts):
            ax.fill_between(weeks, lo_curve, hi_curve, color=color, alpha=0.08, lw=0, zorder=0)

        # Median (optional)
        if plot_median:
            med = compute_median(ts)
            ax.plot(weeks, med, color=color, lw=2.5, zorder=2)

        # Historical data
        for j, season_key in enumerate(sorted_seasons):
            season_data = gt_plot_data[season_key]
            if loc_code in season_data.columns:
                gt_series = season_data[loc_code].dropna()
                if not gt_series.empty:
                    ls = line_styles[j % len(line_styles)]
                    # Create full season label (e.g., "2021-2022")
                    season_label = f"{season_key}-{int(season_key)+1}" if i == 0 else None
                    ax.plot(gt_series.index, gt_series.values,
                           color='black', lw=2.0, alpha=0.9, ls=ls, zorder=10,
                           label=season_label)

        # Add trajectory inset with multiple trajectories
        n_trajs = min(n_inset_trajs, ts.shape[0])
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
        if i % ncols == 0:
            ax.set_ylabel('Incidence')
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)


    for j in range(len(axes_list)):
        if j >= n_states:
            axes_list[j].set_axis_off()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_unconditional_states_with_history_alt(inv_samples: np.ndarray,
                                                season_axis: SeasonAxis,
                                                states: list[str],
                                                save_path: str | None = None):
    """Plot unconditional samples with historical data showing all trajectories and envelope.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object
        states: list of state codes/abbrevs
        save_path: optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    arr = normalize_samples_shape(inv_samples)
    real_weeks = get_real_weeks(arr)
    weeks = np.arange(1, real_weeks + 1)

    n_states = len(states)
    ncols = min(3, n_states)
    nrows = int(np.ceil(n_states / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), dpi=200, sharey=False, sharex=True)
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Load historical data (excluding 2021-2022 season due to incomplete data)
    gt_df = pd.read_csv('influpaint/data/nhsn_flusight_past.csv')
    gt_plot_data = {}
    for season in gt_df['fluseason'].unique():
        if season == 2021:  # Skip 2021-2022 season (incomplete data)
            continue
        season_data = gt_df[gt_df['fluseason'] == season]
        season_pivot = season_data.pivot(columns='location_code', values='value', index='season_week')
        gt_plot_data[season] = season_pivot

    sorted_seasons = sorted(gt_plot_data.keys())
    line_styles = ['-', '--', '-.', ':']
    month_labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    month_weeks = [1, 5, 9, 13, 17, 22, 26, 31, 35, 40, 44, 48]

    for i, st in enumerate(states):
        ax = axes_list[i]
        ts = get_state_timeseries(arr, st, season_axis)
        loc_code = state_to_code(st, season_axis)
        color = sns.color_palette('Set2', n_colors=n_states)[i % n_states]

        # 95% envelope
        lower_envelope = np.quantile(ts, 0.025, axis=0)
        upper_envelope = np.quantile(ts, 0.975, axis=0)
        ax.fill_between(weeks, lower_envelope, upper_envelope, color='gray', alpha=0.3, lw=0, zorder=0)

        # All trajectories
        for si in range(ts.shape[0]):
            ax.plot(weeks, ts[si], color=color, alpha=0.05, lw=0.6, zorder=1)

        # Historical data
        for j, season_key in enumerate(sorted_seasons):
            season_data = gt_plot_data[season_key]
            if loc_code in season_data.columns:
                gt_series = season_data[loc_code].dropna()
                if not gt_series.empty:
                    ls = line_styles[j % len(line_styles)]
                    ax.plot(gt_series.index, gt_series.values,
                           color='black', lw=2.0, alpha=0.9, ls=ls, zorder=10,
                           label=season_key if i == 0 else None)

        state_name = season_axis.get_location_name(loc_code)
        ax.text(0.02, 0.98, state_name, transform=ax.transAxes, va='top', ha='left',
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xlim(1, real_weeks)
        ax.set_ylim(bottom=0)
        ax.set_xticks([month_weeks[j] for j in range(0, len(month_weeks), 2)])
        ax.set_xticklabels([month_labels[j] for j in range(0, len(month_labels), 2)])
        if i % ncols == 0:
            ax.set_ylabel('Incidence')
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)


    for j in range(len(axes_list)):
        if j >= n_states:
            axes_list[j].set_axis_off()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def generate_unconditional_us_grid(inv_samples: np.ndarray, season_axis: SeasonAxis,
                                    fig_dir: str, model_num: str):
    """Generate US grid of unconditional samples using influpaint's plotting library."""
    fig, _ = idplots.plot_unconditional_us_map(
        inv_samples=inv_samples,
        season_axis=season_axis,
        sample_idx=list(np.arange(2, min(500, inv_samples.shape[0]), step=15)),
        multi_line=True,
        sharey=False,
        past_ground_truth=True,
    )
    plt.savefig(os.path.join(fig_dir, "unconditional_us_grid.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_unconditional_trajs_and_heatmap(inv_samples: np.ndarray, season_axis: SeasonAxis,
                                              fig_dir: str):
    """Generate trajectories + mean heatmap using influpaint's plotting library."""
    fig, _ = idplots.fig_unconditional_trajectories_and_mean_heatmap(
        inv_samples=inv_samples,
        season_axis=season_axis,
        n_samples=12,
        save_path=os.path.join(fig_dir, "unconditional_trajs_and_mean_heatmap.png"),
    )
    plt.close(fig)


def generate_mean_heatmap(inv_samples: np.ndarray, season_axis: SeasonAxis,
                          fig_dir: str, model_num: str):
    """Generate mean heatmap figure."""
    mean_heatmap = inv_samples[:, 0, :53, :len(season_axis.locations)].mean(axis=0)
    fig_heat, ax_heat = plt.subplots(figsize=(8, 8), dpi=200)
    im = ax_heat.imshow(mean_heatmap.T, cmap='Reds', aspect='equal', origin='lower')
    ax_heat.set_xlabel('Week')
    ax_heat.set_ylabel('Location')
    ax_heat.set_xticks([0, 13, 26, 39, 52])
    ax_heat.set_xticklabels(['1', '14', '27', '40', '53'])
    ax_heat.set_yticks([0, 12, 25, 38, len(season_axis.locations)-1])
    ax_heat.set_yticklabels(['1', '13', '26', '39', str(len(season_axis.locations))])
    plt.colorbar(im, ax=ax_heat, label='Incidence', shrink=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{model_num}_uncond_mean_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_heat)
