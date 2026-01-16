# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: diffusion_torch6
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Dataset format for Influpaint
# The dataset is a xarray object stored as netcdf on disk. It has dimensions `(sample, feature, date, place)` where date and place are padded to have dimension 64.
# - dates are Saturdays
# - places are location from Flusight data locations. The sum of all places (whole U.S) is not included at that stage
# - samples are integers
#
#
# Create first dataframes with:
# * frame identifier:
#   * datasetH1
#   * datasetH2
#   * sample
#   * fluseason
# * Frame axis and value
#   * location_code'
#   * 'season_week'
#   * 'value'
# * available is also week_enddate
#

# %% [markdown]
# ## Setup: Geography and Time Axis
#
# We start by defining the spatial and temporal structure that all datasets will conform to. The `SeasonAxis` object defines:
# - **Geography**: Which locations to include (50 US states + DC)
# - **Season definition**: When flu seasons start/end (MMWR week 40 = early October)
# - **Week numbering**: Maps calendar dates to season weeks (1–53)
#
# Using `SeasonAxis.for_flusight()` creates a standard configuration matching FluSight forecasting challenges. The `remove_us=True` and `remove_territories=True` flags exclude the US national aggregate and territories, leaving 51 locations.
#
# Every dataset will be processed using this `season_setup` object to add `fluseason` (season start year), `season_week` (1–53), and standardize location codes.

# %% [markdown]
# Let's store our location so metrocast does not re-order these.
# ```bash
# cp Flusight/flu-metrocast/auxiliary-data/locations.csv influpaint/metrocast_locations.csv
# ```

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
import importlib

# InfluPaint modular imports
from influpaint.utils import SeasonAxis
from influpaint.utils import plotting as idplots
from influpaint.datasets import read_datasources

# Create the season structure for Flusight geography
season_setup = SeasonAxis.for_metrocast(remove_us=True, remove_territories=True)

# %%
historical_data  = pd.read_csv("Flusight/metrocast/flu-metrocast/target-data/latest-data.csv")
historical_data = historical_data.rename(columns={"location": "location_code", "observation": "value", "target_end_date": "week_enddate"})
historical_data_df = read_datasources.clean_dataset(historical_data, season_setup=season_setup)
historical_data_df["week_enddate"] = pd.to_datetime(historical_data_df["week_enddate"])
historical_data_df["datasetH1"] = "metrocast"
historical_data_df["datasetH2"] = "latest-date.csv"
historical_data_df["sample"] = 1

historical_data_df = season_setup.add_season_columns(historical_data_df, do_fluseason_year=True)

fig, axes = idplots.plot_timeseries_grid(historical_data_df, season_setup)
fig, axes = idplots.plot_season_overlap_grid(historical_data_df, season_setup)

# %%
print(historical_data_df.fluseason.unique())
historical_data_df = historical_data_df[historical_data_df["fluseason"] != 2025]
print(historical_data_df.fluseason.unique())

# %% [markdown]
# ## C. Generate dataset for fitting
#
# ## C. Combine All Sources
#
# All surveillance and modeling datasets are concatenated into a single DataFrame. This combined dataset contains all required columns for frame construction:
# - Frame identifiers: `datasetH1`, `datasetH2`, `sample`, `fluseason`
# - Spatial/temporal axes: `location_code`, `season_week`, `week_enddate`
# - Values: `value`
#
# The combined DataFrame is saved as `all_datasets.parquet` for use in the next notebook (`2-build_training_flu_datasets_ipynb.py`), where mixing configurations determine how many frames to sample from each source.

# %%
all_datasets = {"metrocast_nssp": historical_data_df,
               # "csp_flusurv": csp_flusurv,
}
for source, df in all_datasets.items():
    print(f"Source: {source}, shape: {df.shape} > years: {len(df['fluseason'].unique())}, datasetH2: {len(df['datasetH2'].unique())}, sample: {len(df['sample'].unique())}")

all_datasets_df = pd.concat(all_datasets.values(), ignore_index=True)
print(f"All datasets combined shape: {all_datasets_df.shape}")
print(sorted(all_datasets_df.columns))

# %% [markdown]
# **Verify the combined dataset**
#
# Check the shape and contents of the combined DataFrame. All required columns should be present:
# - Frame identifiers: `datasetH1`, `datasetH2`, `sample`, `fluseason`
# - Spatial/temporal: `location_code`, `season_week`, `week_enddate`
# - Values: `value`

# %%
all_datasets_df

# %% [markdown]
# ### Save Combined Dataset
#
# The combined dataset is saved as a Parquet file for efficient loading in downstream workflows. Converting `sample` to string prevents issues with mixed integer/string sample IDs from different sources.

# %%
all_datasets_df['sample'] = all_datasets_df['sample'].astype(str)
all_datasets_df.to_parquet("Flusight/metrocast/metrocast-datasets/all_metrocast_datasets.parquet", index=False)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
import epiweeks
import warnings
import importlib
import tqdm
import datetime

# InfluPaint modular imports
from influpaint.utils import SeasonAxis
from influpaint.datasets import build_frames
from influpaint.utils import plotting as idplots
from influpaint.utils import converters
from influpaint.datasets import mixer as dataset_mixer
from influpaint.datasets import read_datasources

today = datetime.datetime.now().strftime("%Y-%m-%d")
# create the folder if exists:
Path("training_datasets").mkdir(parents=True, exist_ok=True)

def build_dataset_from_framelist(frame_list):
    main_origins = []
    for i, frame in enumerate(frame_list):
        df = frame_list[i]
        df["fluseason"] = i
        frame_list[i] = df
        assert df.season_week.max() == 53 and df.season_week.min() == 1, f"Frame {i} has invalid season_week range: {df.season_week.min()} to {df.season_week.max()}"
        assert set(df["location_code"].unique()) == set(season_setup.locations), f"Frame {i} has invalid locations: {set(df['location_code'].unique())} vs {set(season_setup.locations)}"
        # Add the most occurring origin for this frame
        main_origins.append(df["origin"].mode()[0] if not df["origin"].mode().empty else None)
    assert set(pd.concat(frame_list).fluseason.unique()) == set(range(len(frame_list)))

    all_frames_df = pd.concat(frame_list).reset_index(drop=True)
    array_list = converters.dataframe_to_arraylist(df=all_frames_df, season_setup=season_setup, array_dim=128)

    array = np.array(array_list)

    flu_payload_array = xr.DataArray(array, 
                    coords={'sample': np.arange(array.shape[0]),
                        'feature': np.arange(array.shape[1]),
                        'season_week': np.arange(1, array.shape[2]+1),
                        'place': season_setup.locations + [""]*(array.shape[3] - len(season_setup.locations))}, 
                    dims=["sample", "feature", "season_week", "place"])
    return flu_payload_array, main_origins


# %%
all_datasets_df = pd.read_parquet("Flusight/metrocast/metrocast-datasets/all_metrocast_datasets.parquet")
for dH1 in all_datasets_df['datasetH1'].unique():
    h1df= all_datasets_df[all_datasets_df['datasetH1'] == dH1]
    print(f"datasetH1: {dH1}, nH2= {len(h1df['datasetH2'].unique())}")
    for dH2 in h1df['datasetH2'].unique():
        h2df = h1df[h1df['datasetH2'] == dH2]
        print(f" -  datasetH2: {dH2}, shape: {h2df.shape}, years: {len(h2df['fluseason'].unique())}, samples: {len(h2df['sample'].unique())} ===> n_frames={len(h2df['fluseason'].unique())* len(h2df['sample'].unique())}")

# %%
DATASET_GRIDS = {
    "100M": {
        "metrocast":   {"multiplier": 29},
    },
}

# %%
for ds_name, mix_cfg in DATASET_GRIDS.items():
    frame_list = dataset_mixer.build_frames(all_datasets_df, mix_cfg, 
                    season_axis=season_setup, 
                    fill_missing_locations="random",
                    scaling_distribution=None)
    flu_payload_array, main_origins = build_dataset_from_framelist(frame_list)
    flu_payload_array = flu_payload_array.assign_attrs(
                main_origins=list(main_origins),
                mix_cfg=mix_cfg.__str__()
    )
    flu_payload_array.to_netcdf(f"training_datasets/MetrocastTS_{ds_name}_{today}.nc")
