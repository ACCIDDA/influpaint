"""
Functions for plotting mask experiment results.
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from influpaint.utils import ground_truth
from influpaint.utils.helpers import flusight_quantile_pairs
from .helpers import state_to_code
from .config import IMAGE_SIZE, CHANNELS


def recreate_mask(gt: ground_truth.GroundTruth, mask_name: str):
    """Recreate mask pattern based on mask name.

    Args:
        gt: GroundTruth object
        mask_name: Name of the mask experiment

    Returns:
        Mask array with shape (CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    """
    mask = np.ones((CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    ss = gt.season_setup
    if mask_name == 'missing_half_subpop':
        half = len(ss.locations)//2
        mask[:, :, :half] = 0
    elif mask_name == 'missing_midseason':
        start = pd.to_datetime(f"{gt.season_first_year}-12-07")
        end = pd.to_datetime(f"{int(gt.season_first_year)+1}-01-07")
        w0 = ss.get_season_week(start)
        w1 = ss.get_season_week(end)
        mask[:, w0-1:w1, :] = 0
    elif mask_name == 'missing_midseason_peak':
        start = pd.to_datetime(f"{int(gt.season_first_year)+1}-02-01")
        end = pd.to_datetime(f"{int(gt.season_first_year)+1}-02-15")
        w0 = ss.get_season_week(start)
        w1 = ss.get_season_week(end)
        mask[:, w0-1:w1, :] = 0
    elif mask_name == 'missing_nc':
        code = '37'
        idx = ss.locations.index(code)
        mask[:, :, idx] = 0
    return mask


def plot_mask_experiments(mask_dir: str, forecast_date: str,
                          states=('NC', 'CA'),
                          n_sample_trajs: int = 10,
                          plot_median: bool = True):
    """Plot mask experiment results for all masks in directory.

    Args:
        mask_dir: Directory containing mask experiment subdirectories
        forecast_date: Forecast reference date string
        states: Default states to plot if masked locations aren't obvious
        n_sample_trajs: Number of sample trajectories to plot
        plot_median: Whether to plot median

    Returns:
        List of output file paths
    """
    masks = [d for d in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, d))]
    if not masks:
        print("No mask experiment subfolders found.")
        return []

    outs = []
    for name in sorted(masks):
        # Determine season from folder name if present
        season_detect = None
        try:
            import re
            m = re.search(r"_season(\d{4})", name)
            if m:
                season_detect = m.group(1)
        except Exception:
            season_detect = None
        season_use = season_detect

        # Build GT for this season
        gt = ground_truth.GroundTruth(
            season_first_year=str(season_use),
            data_date=dt.datetime.today(),
            mask_date=pd.to_datetime(forecast_date),
            channels=CHANNELS,
            image_size=IMAGE_SIZE,
            nogit=True,
        )
        dates = pd.to_datetime(gt.gt_xarr['date'].values)

        # Load data
        subdir = os.path.join(mask_dir, name)
        f_path = os.path.join(subdir, 'fluforecasts_ti.npy')
        m_path = os.path.join(subdir, 'mask.npy')
        if not (os.path.exists(f_path) and os.path.exists(m_path)):
            continue
        arr = np.load(f_path)
        mk = np.load(m_path)

        # Choose locations: if exactly 5 masked locations -> plot those; else pick up to 5 masked
        p_len = len(gt.season_setup.locations)
        masked_any = (mk[0, :arr.shape[2], :p_len] == 0).any(axis=0)
        masked_idx = np.where(masked_any)[0].tolist()
        if len(masked_idx) == 5:
            plot_indices = masked_idx
        elif len(masked_idx) > 0:
            plot_indices = masked_idx[:5]
        else:
            # fallback to provided states
            plot_indices = []
            for st in (states if isinstance(states, (list, tuple)) else [states]):
                code = state_to_code(st, gt.season_setup)
                plot_indices.append(gt.season_setup.locations.index(code))
            plot_indices = plot_indices[:5]

        # Labels
        locdf = gt.season_setup.locations_df
        abbr_map = None
        if 'abbreviation' in locdf.columns:
            abbr_map = locdf.set_index('location_code')['abbreviation']
        labels = [abbr_map.get(str(gt.season_setup.locations[i]), str(gt.season_setup.locations[i])) if abbr_map is not None else str(gt.season_setup.locations[i]) for i in plot_indices]

        ncols = 1 + len(plot_indices)
        fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4.5), dpi=200)
        if ncols == 2:
            axes = [axes[0], axes[1]]
        # Mask overlay
        base_crop = gt.gt_xarr.data[0][:52, :52]
        mask_crop = mk[0][:52, :52]
        axes[0].imshow(base_crop.T, cmap='Greys', aspect='equal')
        axes[0].imshow(mask_crop.T, alpha=.3, cmap='rainbow', aspect='equal')
        axes[0].set_aspect('equal')
        axes[0].set_axis_off()

        palette = sns.color_palette('Set1', n_colors=len(plot_indices))
        for j, (idx, lab) in enumerate(zip(plot_indices, labels)):
            ax = axes[j+1]
            gt_series = gt.gt_xarr.data[0, :, idx]
            ax.plot(dates, gt_series, color='k', lw=1.5)
            ts = arr[:, 0, :, idx]
            # Sample trajectories (only where masked)
            if n_sample_trajs and n_sample_trajs > 0:
                ns = min(n_sample_trajs, ts.shape[0])
                sample_idxs = np.linspace(0, ts.shape[0]-1, num=ns, dtype=int)
                keep = mk[0, :ts.shape[1], idx]
                for si in sample_idxs:
                    y = ts[si, :len(dates)].copy()
                    y[keep == 1] = np.nan
                    ax.plot(dates[:len(y)], y, color=palette[j], alpha=0.25, lw=0.7)
            # Quantile fans and median (only where masked)
            for lo, hi in flusight_quantile_pairs:
                lo_curve = np.quantile(ts, lo, axis=0)
                hi_curve = np.quantile(ts, hi, axis=0)
                keepw = mk[0, :len(lo_curve), idx]
                lo_curve = lo_curve.copy(); hi_curve = hi_curve.copy()
                lo_curve[keepw == 1] = np.nan
                hi_curve[keepw == 1] = np.nan
                ax.fill_between(dates[:len(lo_curve)], lo_curve, hi_curve, color=palette[j], alpha=0.06, lw=0)
            if plot_median:
                med = np.quantile(ts, 0.5, axis=0)
                med_masked = med.copy()
                med_masked[mk[0, :len(med), idx] == 1] = np.nan
                ax.plot(dates[:len(med_masked)], med_masked, color=palette[j], lw=1.8)
            # Corner label
            ax.text(0.02, 0.98, str(lab).upper(), transform=ax.transAxes, va='top', ha='left',
                    fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel('Incident flu hospitalizations')
            ax.set_xlabel('Date')
            sns.despine(ax=ax, trim=True)
        fig.tight_layout()

        # Save figure
        from .config import FIG_DIR, BEST_MODEL_ID
        _MODEL_NUM = BEST_MODEL_ID.lstrip('i') if isinstance(BEST_MODEL_ID, str) else str(BEST_MODEL_ID)
        os.makedirs(os.path.join(FIG_DIR, "mask_figures"), exist_ok=True)
        out_path = os.path.join(FIG_DIR, "mask_figures", f"{_MODEL_NUM}_mask_{name}.png")
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        outs.append(out_path)
    return outs
