"""
Spatial correlation analysis utilities for Influpaint figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

from .data_utils import normalize_samples_shape, get_real_weeks


def compute_weekly_incidence_correlation(inv_samples: np.ndarray) -> list[float]:
    """Compute temporal correlation between each pair of states.

    For each sample, we measure how similar state-level trajectories are by computing
    the Pearson correlation between every pair of states over the real season weeks.

    Args:
        inv_samples: Array shaped as (N, 1, weeks, places) or (N, weeks, places).

    Returns:
        List of correlation coefficients pooled over all samples and state pairs.
    """
    arr = normalize_samples_shape(inv_samples)
    n, c, w, p = arr.shape
    real_weeks = get_real_weeks(arr)

    correlations = []

    # For each sample
    for sample_idx in range(n):
        sample_data = arr[sample_idx, 0, :real_weeks, :]  # (weeks, places)

        # Compute pairwise state correlations (temporal correlation between states)
        for state1 in range(p):
            for state2 in range(state1 + 1, p):
                ts1 = sample_data[:, state1]
                ts2 = sample_data[:, state2]

                # Remove NaN values
                valid = ~(np.isnan(ts1) | np.isnan(ts2))
                if np.sum(valid) < 3:  # Need at least 3 time points
                    continue

                t1 = ts1[valid]
                t2 = ts2[valid]

                # Check for sufficient variation
                if np.std(t1) < 1e-6 or np.std(t2) < 1e-6:
                    continue

                # Compute correlation
                corr = np.corrcoef(t1, t2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

    return correlations


def compute_random_correlation(inv_samples: np.ndarray,
                               n_permutations: int = 100) -> list[float]:
    """Compute null correlations by permuting each state's time series.

    Args:
        inv_samples: Array shaped as (N, 1, weeks, places) or (N, weeks, places).
        n_permutations: Number of random permutations to draw.

    Returns:
        List of correlation coefficients from permuted samples.
    """
    arr = normalize_samples_shape(inv_samples)
    n, c, w, p = arr.shape
    real_weeks = get_real_weeks(arr)

    correlations = []

    for perm_idx in range(n_permutations):
        # Randomly select a sample
        sample_idx = np.random.randint(0, n)
        sample_data = arr[sample_idx, 0, :real_weeks, :].copy()

        # Permute each state's time series independently
        for state_idx in range(p):
            perm = np.random.permutation(real_weeks)
            sample_data[:, state_idx] = sample_data[perm, state_idx]

        # Compute pairwise state correlations on permuted data
        for state1 in range(p):
            for state2 in range(state1 + 1, p):
                ts1 = sample_data[:, state1]
                ts2 = sample_data[:, state2]

                valid = ~(np.isnan(ts1) | np.isnan(ts2))
                if np.sum(valid) < 3:
                    continue

                t1 = ts1[valid]
                t2 = ts2[valid]

                if np.std(t1) < 1e-6 or np.std(t2) < 1e-6:
                    continue

                corr = np.corrcoef(t1, t2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

    return correlations


def compute_observed_correlation(n_seasons: int = 3) -> list[float]:
    """Compute state-pair correlations from historical seasons.

    Args:
        n_seasons: Number of most recent historical seasons to include.

    Returns:
        List of correlation coefficients pooled across seasons and state pairs.
    """
    gt_df = pd.read_csv('influpaint/data/nhsn_flusight_past.csv')

    correlations = []

    # Get the most recent n_seasons (excluding 2021-2022 season due to incomplete data)
    all_seasons = sorted(gt_df['fluseason'].unique())
    all_seasons = [s for s in all_seasons if s != 2021]  # Exclude 2021-2022 season
    seasons = all_seasons[-n_seasons:]

    for season_year in seasons:
        season_data = gt_df[gt_df['fluseason'] == season_year]
        season_pivot = season_data.pivot(columns='location_code', values='value', index='season_week')

        # Get list of locations
        locations = season_pivot.columns.tolist()

        # Compute pairwise state correlations
        for i, loc1 in enumerate(locations):
            for j in range(i + 1, len(locations)):
                loc2 = locations[j]

                ts1 = season_pivot[loc1].values
                ts2 = season_pivot[loc2].values

                valid = ~(np.isnan(ts1) | np.isnan(ts2))
                if np.sum(valid) < 3:
                    continue

                t1 = ts1[valid]
                t2 = ts2[valid]

                if np.std(t1) < 1e-6 or np.std(t2) < 1e-6:
                    continue

                corr = np.corrcoef(t1, t2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

    return correlations


def plot_weekly_incidence_correlation(inv_samples: np.ndarray,
                                      save_path: Optional[str] = None,
                                      n_permutations: int = 100) -> plt.Figure:
    """Plot pairwise state correlation comparison.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places).
        save_path: Optional path to save the figure.
        n_permutations: Number of random permutations for the null distribution.

    Returns:
        Matplotlib Figure object.
    """
    print("Computing random correlations...")
    random_corr = compute_random_correlation(inv_samples, n_permutations)

    print("Computing influpaint correlations...")
    influpaint_corr = compute_weekly_incidence_correlation(inv_samples)

    print("Computing observed correlations...")
    observed_corr = compute_observed_correlation()

    # Prepare data for box plot
    data = []
    for corr in random_corr:
        data.append({'Category': 'Expected \n if random', 'Correlation': corr})
    for corr in influpaint_corr:
        data.append({'Category': 'Influpaint', 'Correlation': corr})
    for corr in observed_corr:
        data.append({'Category': 'Observed', 'Correlation': corr})

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)

    sns.boxplot(
        data=df,
        x='Category',
        y='Correlation',
        ax=ax,
        order=['Expected \n if random', 'Influpaint', 'Observed'],
        palette=['lightgray', 'skyblue', 'salmon'],
        showfliers=False,
    )
    ax.set_ylabel('Correlation across U.S. states', fontsize=13)
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3, axis='y')
    sns.despine(ax=ax, trim=True)

    # Report summary statistics to stdout
    for i, category in enumerate(['Expected \n if random', 'Influpaint', 'Observed']):
        cat_data = df[df['Category'] == category]['Correlation']
        median = cat_data.median()
        mean = cat_data.mean()
        print(f"{category}: mean={mean:.3f}, median={median:.3f}, n={len(cat_data)}")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation figure to {save_path}")

    return fig
