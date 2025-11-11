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
from influpaint.utils.helpers import flusight_quantile_pairs

from .helpers import state_to_code
from .config import STATE_NAMES


def plot_unconditional_states_quantiles_and_trajs(inv_samples: np.ndarray,
                                                  season_axis: SeasonAxis,
                                                  states: list[str],
                                                  n_sample_trajs: int = 10,
                                                  plot_median: bool = True,
                                                  save_path: str | None = None):
    """Plot unconditional sample trajectories and quantile fans for given states.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        states: list of state codes/abbrevs (expected ~5)
        n_sample_trajs: number of light sample lines to overlay per state
        plot_median: toggle median line overlay
        save_path: optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    # Normalize shape to (N, 1, W, P)
    if inv_samples.ndim == 4:
        arr = inv_samples
    elif inv_samples.ndim == 3:
        arr = inv_samples[:, None, :, :]
    else:
        raise ValueError("inv_samples must be (sample, feature, week, place) or (sample, week, place)")

    n, c, w, p = arr.shape
    real_weeks = min(53, w)
    weeks = np.arange(1, real_weeks + 1)

    n_states = len(states)
    ncols = n_states
    nrows = 1
    if n_states > 5:
        nrows = 2
        ncols = int(np.ceil(n_states / 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), dpi=200, sharey=False)
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, st in enumerate(states):
        ax = axes_list[i]
        loc_code = state_to_code(st, season_axis)
        place_idx = season_axis.locations.index(loc_code)
        ts = arr[:, 0, :real_weeks, place_idx]  # (N, W)

        # Color
        color = sns.color_palette('Set2', n_colors=n_states)[i % n_states]

        # Light sampled trajectories
        if n_sample_trajs and n_sample_trajs > 0:
            ns = min(n_sample_trajs, ts.shape[0])
            sample_idxs = np.linspace(0, ts.shape[0]-1, num=ns, dtype=int)
            for si in sample_idxs:
                ax.plot(weeks, ts[si], color=color, alpha=0.25, lw=0.8, zorder=1)

        # Quantile bands
        for lo, hi in flusight_quantile_pairs:
            lo_curve = np.quantile(ts, lo, axis=0)
            hi_curve = np.quantile(ts, hi, axis=0)
            ax.fill_between(weeks, lo_curve, hi_curve, color=color, alpha=0.08, lw=0)

        # Median
        if plot_median:
            med = np.quantile(ts, 0.5, axis=0)
            ax.plot(weeks, med, color=color, lw=1.8, zorder=2)

        # Styling
        ax.text(0.02, 0.98, st.upper(), transform=ax.transAxes, va='top', ha='left',
                fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xlim(1, real_weeks)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Epiweek')
        if i % ncols == 0:
            ax.set_ylabel('Incidence')
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)

    # Hide any unused axes if states < grid size
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
                                     surface_alpha: float = 0.8,
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
        states: optional list of state abbrevs/codes to overlay as ridges; if None, picks evenly spaced locations
        stat: 'median' or 'mean' used for ridge z-values across samples
        cmap: colormap for bottom heatmap
        elev: elevation angle for 3D view
        azim: azimuth angle for 3D view
        heatmap_mode: 'mean' or 'sample' for heatmap data
        sample_idx: which sample to use if heatmap_mode='sample'
        surface_alpha: transparency of the surface (not used in current implementation)
        surface_zoffset_ratio: z-offset ratio for surface
        ridge_offset_ratio: offset ratio for ridges
        location_stride: stride for location selection when states is None
        fill_ridges: whether to fill under ridges
        fill_alpha: transparency of ridge fill
        save_path: optional path to save the figure

    Returns:
        Tuple of (figure, axis)
    """
    # Normalize samples shape
    if inv_samples.ndim == 4:
        arr = inv_samples
    elif inv_samples.ndim == 3:
        arr = inv_samples[:, None, :, :]
    else:
        raise ValueError("inv_samples must be (sample, feature, week, place) or (sample, week, place)")

    n, c, w, p = arr.shape
    P = len(season_axis.locations)
    real_weeks = min(53, w)

    # Compute heatmap (mean across samples or a single sample)
    if heatmap_mode == 'sample':
        sample_idx = int(np.clip(sample_idx, 0, n-1))
        heat = arr[sample_idx, 0, :real_weeks, :P]
    else:
        heat = arr[:, 0, :real_weeks, :P].mean(axis=0)  # (W, P)

    # Build grid for bottom surface
    x_vals = np.arange(1, real_weeks + 1)
    y_vals = np.arange(P)
    X, Y = np.meshgrid(x_vals, y_vals)  # shapes (P, W)
    # Optional tiny z-offset for surface (defaults to 0 to avoid visible shift)
    zmax_global = float(np.nanmax(heat)) if np.isfinite(heat).all() else 1.0
    zmax_global = max(1.0, zmax_global)
    Z0 = np.zeros_like(X, dtype=float) - (surface_zoffset_ratio * zmax_global if surface_zoffset_ratio else 0.0)
    Cdata = heat.T  # (P, W)

    # Normalize colors
    cmap_obj = plt.cm.get_cmap(cmap)
    norm = plt.Normalize(vmin=np.nanmin(Cdata), vmax=np.nanmax(Cdata) if np.nanmax(Cdata) > 0 else 1.0)
    facecolors = cmap_obj(norm(Cdata))

    # Figure and 3D axis
    fig = plt.figure(figsize=(12, 7), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    # Plot bottom colored surface (flat at z=0)
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
        place_idxs = list(range(0, P, stride))[1:-1]  # Remove first (AL) and last (WY)
        # readable labels using abbreviations if available
        locdf = season_axis.locations_df
        if 'abbreviation' in locdf.columns:
            abbr_map = locdf.set_index('location_code')['abbreviation']
            labels = [abbr_map.get(str(season_axis.locations[i]), str(season_axis.locations[i])) for i in place_idxs]
        else:
            labels = [str(season_axis.locations[i]) for i in place_idxs]

    # Palette for ridges
    ridge_colors = sns.color_palette('Set2', n_colors=len(place_idxs))

    # Plot ridges: x=weeks, y=place_idx, z=statistic over samples
    ridge_offset = ridge_offset_ratio * zmax_global
    for j, (pi, lab) in enumerate(zip(place_idxs, labels)):
        ts = arr[:, 0, :real_weeks, pi]  # (N, W)
        if stat == 'mean':
            z = np.nanmean(ts, axis=0)
        else:
            z = np.nanmedian(ts, axis=0)
        y_curve = np.full_like(x_vals, fill_value=pi)
        # Optional ribbon fill under the curve using vertical lines
        if fill_ridges:
            for k in range(len(x_vals)):
                ax.plot([x_vals[k], x_vals[k]], [pi, pi], [ridge_offset * 0.1, z[k] + ridge_offset],
                       color=ridge_colors[j], alpha=fill_alpha, lw=1.5, zorder=50-pi)
        ax.plot(x_vals, y_curve, z + ridge_offset, color=ridge_colors[j], lw=2.0, zorder=100-pi, marker='.', markersize=4)
        # Label near end of ridge
        ax.text(x_vals[-1]+0.5, pi, (z[-1] + ridge_offset), lab, color=ridge_colors[j], fontsize=9, ha='left', va='center')

    # Aesthetics
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('Epiweek')
    ax.set_ylabel('Location index')
    ax.set_zlabel('Incidence')
    ax.set_xlim(1, real_weeks)
    ax.set_ylim(0, P-1)
    ax.set_zlim(bottom=0)

    # Light grid styling
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
    if inv_samples.ndim == 4:
        arr = inv_samples
    elif inv_samples.ndim == 3:
        arr = inv_samples[:, None, :, :]
    else:
        raise ValueError("inv_samples must be (sample, feature, week, place) or (sample, week, place)")

    n, c, w, p = arr.shape
    P = len(season_axis.locations)
    real_weeks = min(53, w)

    # Heatmap data
    if heatmap_mode == 'sample':
        sample_idx = int(np.clip(sample_idx, 0, n-1))
        heat = arr[sample_idx, 0, :real_weeks, :P]
    else:
        heat = arr[:, 0, :real_weeks, :P].mean(axis=0)

    weeks = np.arange(1, real_weeks + 1)
    y_idx = np.arange(P)
    Z = heat.T  # (P, W)

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
        locdf = season_axis.locations_df
        if 'abbreviation' in locdf.columns:
            abbr_map = locdf.set_index('location_code')['abbreviation']
            labels = [abbr_map.get(str(season_axis.locations[i]), str(season_axis.locations[i])) for i in place_idxs]
        else:
            labels = [str(season_axis.locations[i]) for i in place_idxs]
        labels = [str(l).upper() for l in labels]

    # Add ridge lines
    for pi, lab in zip(place_idxs, labels):
        ts = arr[:, 0, :real_weeks, pi]
        if stat == 'mean':
            z = np.nanmean(ts, axis=0)
        else:
            z = np.nanmedian(ts, axis=0)
        fig.add_trace(go.Scatter3d(x=weeks, y=np.full_like(weeks, pi), z=z + ridge_lift,
                                   mode='lines', name=lab,
                                   line=dict(width=4)))

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
        # Embed plotly.js for offline viewing
        pio.write_html(fig, file=save_path_html, include_plotlyjs=True, full_html=True)
        if os.path.exists(save_path_html):
            print(f"Saved Plotly HTML to {save_path_html}")
        else:
            print(f"Failed to save Plotly HTML to {save_path_html}")
    return fig


def plot_unconditional_states_with_history(inv_samples: np.ndarray,
                                           season_axis: SeasonAxis,
                                           states: list[str],
                                           n_sample_trajs: int = 10,
                                           plot_median: bool = True,
                                           save_path: str | None = None):
    """Plot unconditional samples with historical data overlay for selected states.

    Combines historical data by season with quantiles and trajectories.

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
    if inv_samples.ndim == 4:
        arr = inv_samples
    elif inv_samples.ndim == 3:
        arr = inv_samples[:, None, :, :]
    else:
        raise ValueError("inv_samples must be (sample, feature, week, place) or (sample, week, place)")

    n, c, w, p = arr.shape
    real_weeks = min(53, w)
    weeks = np.arange(1, real_weeks + 1)

    n_states = len(states)
    ncols = min(3, n_states)
    nrows = int(np.ceil(n_states / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), dpi=200, sharey=False)
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    gt_df = pd.read_csv('influpaint/data/nhsn_flusight_past.csv')

    gt_plot_data = {}
    for season in gt_df['fluseason'].unique():
        season_data = gt_df[gt_df['fluseason'] == season]
        season_pivot = season_data.pivot(columns='location_code', values='value', index='season_week')
        gt_plot_data[season] = season_pivot

    sorted_seasons = sorted(gt_plot_data.keys())
    line_styles = ['-', '--', '-.', ':']

    month_labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    month_weeks = [1, 5, 9, 13, 17, 22, 26, 31, 35, 40, 44, 48]

    for i, st in enumerate(states):
        ax = axes_list[i]
        loc_code = state_to_code(st, season_axis)
        place_idx = season_axis.locations.index(loc_code)
        ts = arr[:, 0, :real_weeks, place_idx]

        color = sns.color_palette('Set2', n_colors=n_states)[i % n_states]

        for lo, hi in flusight_quantile_pairs:
            lo_curve = np.quantile(ts, lo, axis=0)
            hi_curve = np.quantile(ts, hi, axis=0)
            ax.fill_between(weeks, lo_curve, hi_curve, color=color, alpha=0.08, lw=0, zorder=0)

        if n_sample_trajs and n_sample_trajs > 0:
            ns = min(n_sample_trajs, ts.shape[0])
            sample_idxs = np.linspace(0, ts.shape[0]-1, num=ns, dtype=int)
            for si in sample_idxs:
                ax.plot(weeks, ts[si], color=color, alpha=0.8, lw=1.8, zorder=1)

        if plot_median:
            med = np.quantile(ts, 0.5, axis=0)
            ax.plot(weeks, med, color=color, lw=2.5, zorder=2)

        for j, season_key in enumerate(sorted_seasons):
            season_data = gt_plot_data[season_key]
            if loc_code in season_data.columns:
                gt_series = season_data[loc_code].dropna()
                if not gt_series.empty:
                    ls = line_styles[j % len(line_styles)]
                    ax.plot(gt_series.index, gt_series.values,
                           color='black', lw=2.0, alpha=0.9, ls=ls, zorder=10,
                           label=season_key if i == 0 else None)

        state_name = STATE_NAMES.get(st.upper(), st.upper())
        ax.text(0.02, 0.98, state_name, transform=ax.transAxes, va='top', ha='left',
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xlim(1, real_weeks)
        ax.set_ylim(bottom=0)
        ax.set_xticks([month_weeks[j] for j in range(0, len(month_weeks), 2)])
        ax.set_xticklabels([month_labels[j] for j in range(0, len(month_labels), 2)])
        ax.set_xlabel('Season month')
        if i % ncols == 0:
            ax.set_ylabel('Incidence')
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)

        if i == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

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
    """Plot unconditional samples with historical data overlay showing all trajectories and envelope.

    Shows all sample trajectories with alpha and 95% envelope with fill_between.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object
        states: list of state codes/abbrevs
        save_path: optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    if inv_samples.ndim == 4:
        arr = inv_samples
    elif inv_samples.ndim == 3:
        arr = inv_samples[:, None, :, :]
    else:
        raise ValueError("inv_samples must be (sample, feature, week, place) or (sample, week, place)")

    n, c, w, p = arr.shape
    real_weeks = min(53, w)
    weeks = np.arange(1, real_weeks + 1)

    n_states = len(states)
    ncols = min(3, n_states)
    nrows = int(np.ceil(n_states / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), dpi=200, sharey=False)
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    gt_df = pd.read_csv('influpaint/data/nhsn_flusight_past.csv')

    gt_plot_data = {}
    for season in gt_df['fluseason'].unique():
        season_data = gt_df[gt_df['fluseason'] == season]
        season_pivot = season_data.pivot(columns='location_code', values='value', index='season_week')
        gt_plot_data[season] = season_pivot

    sorted_seasons = sorted(gt_plot_data.keys())
    line_styles = ['-', '--', '-.', ':']

    month_labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    month_weeks = [1, 5, 9, 13, 17, 22, 26, 31, 35, 40, 44, 48]

    for i, st in enumerate(states):
        ax = axes_list[i]
        loc_code = state_to_code(st, season_axis)
        place_idx = season_axis.locations.index(loc_code)
        ts = arr[:, 0, :real_weeks, place_idx]

        color = sns.color_palette('Set2', n_colors=n_states)[i % n_states]

        lower_envelope = np.quantile(ts, 0.025, axis=0)
        upper_envelope = np.quantile(ts, 0.975, axis=0)
        ax.fill_between(weeks, lower_envelope, upper_envelope, color='gray', alpha=0.3, lw=0, zorder=0)

        for si in range(ts.shape[0]):
            ax.plot(weeks, ts[si], color=color, alpha=0.05, lw=0.6, zorder=1)

        for j, season_key in enumerate(sorted_seasons):
            season_data = gt_plot_data[season_key]
            if loc_code in season_data.columns:
                gt_series = season_data[loc_code].dropna()
                if not gt_series.empty:
                    ls = line_styles[j % len(line_styles)]
                    ax.plot(gt_series.index, gt_series.values,
                           color='black', lw=2.0, alpha=0.9, ls=ls, zorder=10,
                           label=season_key if i == 0 else None)

        state_name = STATE_NAMES.get(st.upper(), st.upper())
        ax.text(0.02, 0.98, state_name, transform=ax.transAxes, va='top', ha='left',
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xlim(1, real_weeks)
        ax.set_ylim(bottom=0)
        ax.set_xticks([month_weeks[j] for j in range(0, len(month_weeks), 2)])
        ax.set_xticklabels([month_labels[j] for j in range(0, len(month_labels), 2)])
        ax.set_xlabel('Season month')
        if i % ncols == 0:
            ax.set_ylabel('Incidence')
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)

        if i == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    for j in range(len(axes_list)):
        if j >= n_states:
            axes_list[j].set_axis_off()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def generate_unconditional_us_grid(inv_samples: np.ndarray, season_axis: SeasonAxis, fig_dir: str, model_num: str):
    """Generate US grid of unconditional samples.

    Args:
        inv_samples: Unconditional samples array
        season_axis: SeasonAxis object
        fig_dir: Directory to save figures
        model_num: Model number for filenames
    """
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


def generate_unconditional_trajs_and_heatmap(inv_samples: np.ndarray, season_axis: SeasonAxis, fig_dir: str):
    """Generate trajectories + mean heatmap figure.

    Args:
        inv_samples: Unconditional samples array
        season_axis: SeasonAxis object
        fig_dir: Directory to save figures
    """
    fig, _ = idplots.fig_unconditional_trajectories_and_mean_heatmap(
        inv_samples=inv_samples,
        season_axis=season_axis,
        n_samples=12,
        save_path=os.path.join(fig_dir, "unconditional_trajs_and_mean_heatmap.png"),
    )
    plt.close(fig)


def generate_mean_heatmap(inv_samples: np.ndarray, season_axis: SeasonAxis, fig_dir: str, model_num: str):
    """Generate mean heatmap figure.

    Args:
        inv_samples: Unconditional samples array
        season_axis: SeasonAxis object
        fig_dir: Directory to save figures
        model_num: Model number for filenames
    """
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
