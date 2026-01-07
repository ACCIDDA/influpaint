# Build Training Datasets

This page mirrors the steps in `2-build_training_flu_datasets_ipynb.py` to turn a combined dataset into training NetCDF files. It starts with the array specification, then covers mixing, framing, conversion, and basic checks.

## Training Dataset Format (xarray DataArray)

The training files are xarray DataArrays with dimensions `(sample, feature, season_week, place)`. Each `sample` is one complete frame (a single (H1/H2/season/sample) after mixing). `season_week` uses 1–53 fixed 7‑day bins from the `SeasonAxis`, and `place` follows the ordered set of `location_code`.

Typical saves use a filename like `training_datasets/TS_<NAME>_<YYYY-MM-DD>.nc`.

## Inputs

Use the combined DataFrame described in “Compiling Data Sources” (columns: `datasetH1`, `datasetH2`, `sample`, `fluseason`, `location_code`, `season_week`, `value`, `week_enddate`).

```python
from influpaint.utils import SeasonAxis, converters
from influpaint.datasets import mixer as dataset_mixer

season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
```

## Create mixing configurations

Define how many frames to take from each dataset family and whether to scale intensities. Examples:

```python
# Surveillance‑only (with scaling for fluview)
config_surv = {
    "fluview": {"multiplier": 26, "to_scale": True},
    "flusurv": {"multiplier": 26},
}

# 70% surveillance / 30% modeling
config_70S30M = {
    "fluview": {"proportion": 0.37, "total": 3000, "to_scale": True},
    "flusurv": {"proportion": 0.33, "total": 3000},
    "flepiR1": {"proportion": 0.05, "total": 3000},
    "SMH_R4-R5": {"proportion": 0.25, "total": 3000},
}

# Modeling‑only
config_mod = {
    "flepiR1": {"multiplier": 1},
    "SMH_R4-R5": {"multiplier": 1},
}
```

If any dataset has `to_scale=True`, prepare a 1D `scaling_distribution` of target national peak values (the notebook shows how to compute this from SMH trajectories).

## Build frames

Create fully padded frames (complete weeks and locations). Missing locations can be filled intelligently (`fill_missing_locations="random"`), and the provenance is written into an `origin` column.

```python
frames = dataset_mixer.build_frames(
    all_datasets_df,
    config_surv,                # or config_70S30M, config_mod, ...
    season_axis=season_setup,
    fill_missing_locations="random",
    # scaling_distribution=scaling_distribution,
)
```

## Convert to array

Concatenate frames and convert to the array format expected by training.

```python
all_frames_df = pd.concat(frames).reset_index(drop=True)
array_list = converters.dataframe_to_arraylist(df=all_frames_df, season_setup=season_setup)
xarr = season_setup.add_axis_to_numpy_array(np.array(array_list), truncate=True)
```

## Save and annotate

Attach optional attributes, then write the NetCDF file.

```python
# Optional example: xarr = xarr.assign_attrs(main_origins=list_of_origins, mix_cfg=str(config_surv))
today = datetime.datetime.now().strftime("%Y-%m-%d")
xarr.to_netcdf(f"training_datasets/TS_SURV_ONLY_{today}.nc")
```

## Visual check

Plot a few samples on the US grid to confirm ranges and shapes.

```python
from influpaint.utils import plotting as idplots

idplots.plot_us_grid(
    data=xarr,
    season_axis=season_setup,
    sample_idx=list(range(12)),
    multi_line=True,
    sharey=False,
    past_ground_truth=True,
)
```

## Load for training

```python
from influpaint.datasets import loaders
dl = loaders.FluDataset.from_xarray("training_datasets/TS_SURV_ONLY_YYYY-MM-DD.nc")
```

## Notes

- `fill_missing_locations="random"` performs hierarchical, randomized filling and keeps the source trace in `origin`.
- When `to_scale=True`, intensities are rescaled per frame to a random target peak while preserving curve shape.
- `truncate=True` keeps weeks 1–53 and drops any padding locations.

For a full end‑to‑end example, see `2-build_training_flu_datasets_ipynb.py`.
