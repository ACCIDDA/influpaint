#!/usr/bin/env python3
"""
build_training_sets.py

Turn RSV_TRAIN.parquet into one NetCDF (.nc) training file per mix
configuration. A mix decides how much weight each data source gets in the
final training pool (only surveillance, only synthetic, balanced, etc.).

The output .nc files are consumed directly by rsv_training.py (and by the
InfluPaint FluDataset.from_xarray loader).

Usage (from repo root):
    python RSV/workflow/build_training_sets.py \
        --input RSV/data/RSV_TRAIN.parquet \
        --outdir training_datasets \
        --tag 2026-04-24
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from influpaint.utils import SeasonAxis, converters
from influpaint.datasets import mixer as dataset_mixer


# ---------------------------------------------------------------------------
# Mix configurations
# ---------------------------------------------------------------------------
# Each entry is a name -> "how to mix the sources" recipe. `multiplier` just
# oversamples each season N times. `proportion` + `total` target an exact
# composition. The mix name ends up in the output filename and is also the
# name you pass to the SLURM job.
#
#   100S  = 100% surveillance (NSSP + RSV-Net + NHSN), each x20
#   50S   = half-size surveillance pool (x10 instead of x20)
#   25S   = quarter-size surveillance pool (x5)
#   100M  = 100% synthetic (RSV Scenario Modelling Hub only)
#   100A  = balanced "all RSV data": 30% surveillance + 70% synthetic,
#           matches what the flu paper calls "30S70M"
# ---------------------------------------------------------------------------
RSV_CONFIG = {
    "100S": {
        "NSSP":    {"multiplier": 20},
        "RSV-Net": {"multiplier": 20},
        "NHSN":    {"multiplier": 20},
    },
    "50S": {
        "NSSP":    {"multiplier": 10},
        "RSV-Net": {"multiplier": 10},
        "NHSN":    {"multiplier": 10},
    },
    "25S": {
        "NSSP":    {"multiplier": 5},
        "RSV-Net": {"multiplier": 5},
        "NHSN":    {"multiplier": 5},
    },
    "100M": {
        "RSV_SMH": {"multiplier": 1},
    },
    "100A": {
        "NSSP":    {"proportion": 0.10, "total": 2229},
        "RSV-Net": {"proportion": 0.10, "total": 2229},
        "NHSN":    {"proportion": 0.10, "total": 2229},
        "RSV_SMH": {"proportion": 0.70, "total": 2229},
    },
}


def build_dataset_from_framelist(frame_list, season_setup):
    """Convert a list of per-season DataFrames into a 4D xarray DataArray."""
    main_origins = []
    for i, frame in enumerate(frame_list):
        df = frame_list[i]
        df["fluseason"] = i
        frame_list[i] = df
        assert df.season_week.max() == 53 and df.season_week.min() == 1
        main_origins.append(
            df["origin"].mode()[0] if not df["origin"].mode().empty else None
        )

    all_frames_df = pd.concat(frame_list).reset_index(drop=True)
    array_list = converters.dataframe_to_arraylist(
        df=all_frames_df, season_setup=season_setup
    )
    array = np.array(array_list)
    arr = xr.DataArray(
        array,
        coords={
            "sample": np.arange(array.shape[0]),
            "feature": np.arange(array.shape[1]),
            "season_week": np.arange(1, array.shape[2] + 1),
            "place": season_setup.locations
            + [""] * (array.shape[3] - len(season_setup.locations)),
        },
        dims=["sample", "feature", "season_week", "place"],
    )
    return arr, main_origins


def compute_scaling_distribution(df):
    """
    The scaling distribution is the set of per-season peak values we use to
    rescale training frames. We take the peaks from RSV_SMH (synthetic) when
    available, because they cover realistic intensity ranges well.
    """
    season_ids = ["datasetH1", "datasetH2", "fluseason", "sample"]
    location_sums = (
        df.groupby(season_ids + ["season_week"])["value"].sum().reset_index()
    )
    season_peaks = location_sums.groupby(season_ids)["value"].max().reset_index()
    if "RSV_SMH" in season_peaks["datasetH1"].unique():
        return season_peaks.loc[season_peaks["datasetH1"] == "RSV_SMH", "value"].values
    return season_peaks["value"].values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="RSV/data/RSV_TRAIN.parquet",
        help="Training-only RSV parquet (validation seasons already removed).",
    )
    p.add_argument(
        "--outdir",
        default="training_datasets",
        help="Folder where .nc training files are written.",
    )
    p.add_argument(
        "--tag",
        default=datetime.datetime.now().strftime("%Y-%m-%d"),
        help="Tag appended to output filenames (default: today's date).",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Build only these mix names (default: all mixes in RSV_CONFIG).",
    )
    args = p.parse_args()

    season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load the training-only parquet and make column names match what mixer
    # expects.
    df = pd.read_parquet(args.input)
    df = df.rename(columns={"fluseason_week": "season_week"})
    df["week_enddate"] = pd.NaT

    scaling_dist = compute_scaling_distribution(df)

    configs = RSV_CONFIG if args.only is None else {k: RSV_CONFIG[k] for k in args.only}

    for mix_name, mix_cfg in configs.items():
        print(f"\n=== Building RSV_{mix_name} ===")
        frame_list = dataset_mixer.build_frames(
            df,
            mix_cfg,
            season_axis=season_setup,
            fill_missing_locations="random",
            scaling_distribution=scaling_dist,
        )
        arr, origins = build_dataset_from_framelist(frame_list, season_setup)
        arr = arr.assign_attrs(main_origins=list(origins), mix_cfg=str(mix_cfg))

        outpath = Path(args.outdir) / f"RSV_{mix_name}_{args.tag}.nc"
        arr.to_netcdf(outpath)
        print(f"Saved {outpath}  (n_frames={arr.shape[0]})")


if __name__ == "__main__":
    main()
