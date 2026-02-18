import pandas as pd
import numpy as np
from delphi_epidata import Epidata
from ..utils.season_axis import SeasonAxis
import xarray as xr

def padto64x64(x: np.ndarray) -> np.ndarray:
    return np.pad(
        x,
        ((0, 64 - x.shape[0]), (0, 64 - x.shape[1])),
        mode="constant",
        constant_values=0,
    )

def dataframe_to_xarray(
    df: pd.DataFrame,
    season_setup: SeasonAxis = None,
    xarray_name="data",
    xarrax_features="value",
    date_column="week_enddate",
    value_column="value",
    pad=True,
) -> xr.DataArray:
    """
    Convert a long form dataframe to an xarray. Dataframe must have columns:
    - location_code
    - value
    The dataset is a xarray object stored as netcdf on disk. 
    It has dimensions `(feature, date, place)` 
    where date and place are padded to have dimension 64.
    - dates are Saturdays
    - places are location from Flusight data locations
    - samples are integers
    """

    df_piv = df.reset_index(drop=False).pivot(columns="location_code", values=value_column, index=date_column)

    if not isinstance(xarrax_features, list):
        xarrax_features = [xarrax_features]

    if season_setup is None:
        print(" ⚠️ No season_setup provided, using all locations in the dataframe.")
        places = df_piv.columns.to_list()
    else:
        df_piv = df_piv[
            season_setup.locations_df["location_code"]
        ]  # make sure order is right w.r.t flusight_locations
        places = season_setup.locations_df["location_code"]
        df_piv = df_piv.sort_index(axis=1)

    df_xarr = xr.DataArray(
        np.array([df_piv.to_numpy()]),
        name=xarray_name,
        coords={
            "feature": xarrax_features,
            "date": list(df_piv.index),
            "place": places,
        },
        dims=["feature", "date", "place"],
    )
    if pad:
        df_xarr = df_xarr.pad(
            {
                "date": (0, 64 - len(df_xarr.date)),
                "place": (0, 64 - len(df_xarr.place)),
            },
            mode="constant",
            constant_values=0,
        )

    return df_xarr

# --- PATCHED to accommodate 6 channels ---
def dataframe_to_arraylist(
    df: pd.DataFrame, 
    locations: set[str], 
    age_columns: list[str],
) -> list:
    """
    Standardizes data into a list of arrays shaped (6, 64, 64).
    Uses pivot_table to resolve duplicates and matches the original tower pivot logic.
    """
    samples = []
    sorted_locs = sorted(list(locations))
    unique_seasons = sorted(df["fluseason"].unique())

    towers = {}
    for age_group in age_columns:
        pivoted_tower = df.pivot_table(
            index=["fluseason", "season_week"],
            columns="location_code",
            values=age_group,
            aggfunc='mean'  
        )
        
        towers[age_group] = pivoted_tower.reindex(columns=sorted_locs)

    for season in unique_seasons:
        channel_slices = []
        
        for age_group in age_columns:
            try:
                season_slice = towers[age_group].loc[season].sort_index().to_numpy()
            except KeyError:
                season_slice = np.zeros((53, len(sorted_locs)))

            season_slice = np.nan_to_num(season_slice, nan=0.0)
            
            # Pad from (53, 52) to (64, 64)
            padded_array = padto64x64(season_slice)
            
            channel_slices.append(padded_array)
        # channel stacking
        # Stack 6 age groups to get (6, 64, 64) for a sample
        samples.append(np.stack(channel_slices))

    return samples