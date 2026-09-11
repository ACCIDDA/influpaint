"""Render notebook 1–2's data checks from the reproduction archive.

Run: python -m main_training.render_notebook_stories
No source downloads, dataset mixing, or model training are performed.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from influpaint.utils import SeasonAxis
from influpaint.utils import plotting as idplots

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'influpaint-paper/influpaint_paper_reproduction_data'
OUTPUT = ROOT / 'docs/assets/notebook-stories'
SOURCES = ['fluview', 'flusurv', 'flepiR1', 'SMH_R4-R5']
COLORS = ['#3466a4', '#169b91', '#dc9634', '#ae557c']


def save(fig, name):
    fig.savefig(OUTPUT / name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    axis = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
    data = pd.read_parquet(ARCHIVE / 'datasets/sources/all_datasets.parquet')
    # Notebook 1: the same season-overlay utility, with six states for readable panels.
    states = ['06', '17', '36', '37', '48', '53']
    for source in SOURCES[:2]:
        subset = data[(data.datasetH1 == source) & data.location_code.isin(states)]
        fig, axes = idplots.plot_season_overlap_grid(subset, axis, line_color=COLORS[SOURCES.index(source)])
        fig.suptitle(f'{source}: observed seasons aligned to an August start', y=1.03)
        fig.supxlabel('Season week'); fig.supylabel('Source value (before training rescaling)')
        save(fig, f'01-{source}-seasons.png')
    # Modeling families contain multiple samples per season; show them separately.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout='constrained')
    for ax, source, color in zip(axes, SOURCES[2:], COLORS[2:]):
        subset = data[(data.datasetH1 == source) & (data.location_code == '37')]
        groups = list(subset.groupby(['datasetH2', 'fluseason', 'sample'], sort=True))[:20]
        for _, frame in groups:
            frame = frame.sort_values('season_week')
            ax.plot(frame.season_week, frame.value, color=color, lw=1, alpha=.5)
        ax.set(title=f'{source}: North Carolina, {len(groups)} trajectories', xlabel='Season week', ylabel='Weekly hospitalizations')
    save(fig, '01-simulation-trajectories.png')
    # Notebook 2: its exact grouping for distributions of summed-location season peaks.
    keys = ['datasetH1', 'datasetH2', 'fluseason', 'sample']
    sums = data.groupby(keys + ['season_week']).value.sum()
    peaks = sums.groupby(keys).max().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    for ax, source, color in zip(axes.flat, SOURCES, COLORS):
        ax.hist(peaks.loc[peaks.datasetH1 == source, 'value'], bins=40, color=color, alpha=.85)
        ax.set(title=source, xlabel='Season peak, summed available locations', ylabel='Number of source frames')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    save(fig, '02-source-peaks.png')
    datasets = {name: xr.open_dataarray(ARCHIVE / f'datasets/training/TS_{name}_2025-07-17.nc')
                for name in ['100S', '70S30M', '30S70M', '100M']}
    counts = {name: pd.Series([origin.split('/')[0] for origin in ds.attrs['main_origins']]).value_counts()
              for name, ds in datasets.items()}
    fig, ax = plt.subplots(figsize=(10, 3.8), layout='constrained')
    left = np.zeros(4)
    for source, color in zip(SOURCES, COLORS):
        values = np.array([int(counts[name].get(source, 0)) for name in datasets])
        ax.barh(list(datasets), values, left=left, label=source, color=color)
        left += values
    for index, total in enumerate(left):
        ax.text(total + 35, index, f'{int(total):,}', va='center')
    ax.set(xlim=(0, 3850), xlabel='Saved training frames, grouped by main origin', title='Four archived mixtures: realized dataset sizes')
    ax.legend(ncols=4, loc='upper center', bbox_to_anchor=(.5, -.18), frameon=False)
    save(fig, '02-mixture-sizes.png')
    ds = datasets['30S70M']
    # Notebook 2's final US-grid diagnostic uses samples 3000–3004.
    fig, _ = idplots.plot_us_grid(data=ds, season_axis=axis, sample_idx=list(range(3000, 3005)), multi_line=True, sharey=False)
    save(fig, '02-training-us-grid.png')
    frame = ds.isel(sample=3000, feature=0)
    fig, ax = plt.subplots(figsize=(8.5, 6), layout='constrained')
    im = ax.imshow(np.sqrt(frame.values), aspect='auto', cmap='magma', interpolation='nearest')
    ax.axhline(52.5, color='#46d5df', lw=1.5); ax.axvline(50.5, color='#46d5df', lw=1.5)
    ax.set(title='One archived training frame: week × location', xlabel='Location position (0–50: states and DC; 51–63: padding)', ylabel='Week index (0–52: season; 53–63: padding)')
    fig.colorbar(im, ax=ax, label='Square root of value (display only; no model normalization)')
    save(fig, '02-training-image.png')
    metadata = {'source_snapshot': 'datasets/sources/all_datasets.parquet',
                'source_rows': len(data), 'training_date': '2025-07-17',
                'training_shapes': {n: list(d.shape) for n,d in datasets.items()},
                'grid_sample_indices': list(range(3000, 3005)),
                'image_sample_index': 3000,
                'source_snapshot_is_proven_original_training_input': False}
    (OUTPUT / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for ds in datasets.values():
        ds.close()
    print(f'Rendered 7 notebook story images to {OUTPUT}')


if __name__ == '__main__':
    main()
