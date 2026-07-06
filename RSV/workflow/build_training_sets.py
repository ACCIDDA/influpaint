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
        --tag 2026-06-23
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
# We build six training pools. The experimental variable is HOW MUCH and WHICH
# RSV data a model sees; the architecture, batch size and epochs are held fixed.
#
# Surveillance ladder (100S / 50S / 25S) -- 100%, 50%, 25% of the real RSV
# surveillance frames. After the validation season is removed there are 12
# surveillance frames (NHSN 2, NSSP 3, RSV-Net 7). The ladder keeps the source
# ratio as even as integers allow and never drops a source (NHSN must stay in:
# it is the signal we condition on / score against at inference):
#
#       mix    NHSN  NSSP  RSV-Net   frames
#       100S     2     3      7        12
#       50S      1     2      3         6
#       25S      1     1      1         3
#
# For 50S/25S we keep the most-RECENT seasons of each source (closest to the
# held-out season). Every frame is then replicated MULTIPLIER (=180) times, so
# the pools are 2160 / 1080 / 540 samples -- all >= the 512 DDPM batch size.
# Because the multiplier is identical for all three, the ONLY thing that
# changes down the ladder is the number of UNIQUE frames: 50S is literally 100S
# with half its frames removed. Missing states are filled from surveillance only,
# so the surveillance arms stay free of synthetic data.
#
# Reference pools:
#   100M = synthetic only (RSV Scenario Modelling Hub).
#   100A = "all RSV data": 30% surveillance + 70% synthetic (the flu paper's
#          "30S70M"), source ratio fixed by proportions.
# ---------------------------------------------------------------------------
MULTIPLIER = 180  # frames per surveillance season; clears the 512 batch size

# Surveillance ladder: how many of the most-recent seasons to keep per source.
SURVEILLANCE_LADDER = {
    "100S": {"NHSN": 2, "NSSP": 3, "RSV-Net": 7},   # 12 frames -> 2160 samples
    "50S":  {"NHSN": 1, "NSSP": 2, "RSV-Net": 3},   #  6 frames -> 1080 samples
    "25S":  {"NHSN": 1, "NSSP": 1, "RSV-Net": 1},   #  3 frames ->  540 samples
}

# Reference pools built directly from a mixer config (no frame subsampling).
REFERENCE_POOLS = {
    "100M": {"RSV_SMH": {"multiplier": 1}},
    "100A": {
        "NSSP":    {"proportion": 0.10, "total": 2160},
        "RSV-Net": {"proportion": 0.10, "total": 2160},
        "NHSN":    {"proportion": 0.10, "total": 2160},
        "RSV_SMH": {"proportion": 0.70, "total": 2160},
    },
}

ALL_MIXES = list(SURVEILLANCE_LADDER) + list(REFERENCE_POOLS)


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


def select_recent_seasons(df, per_source_counts):
    """
    Keep only the N most-recent seasons of each surveillance source.

    This is how the 50S / 25S rungs are built: 50S keeps fewer seasons than
    100S, so it is literally 100S with some of its unique frames removed.
    """
    kept = []
    for source, n in per_source_counts.items():
        sub = df[df["datasetH1"] == source]
        seasons = sorted(sub["fluseason"].dropna().unique())
        recent = seasons[-n:]
        print(f"  {source}: keep {n} most-recent season(s) {[int(s) for s in recent]}")
        kept.append(sub[sub["fluseason"].isin(recent)])
    return pd.concat(kept, ignore_index=True)


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
        help="Build only these mix names (default: all mixes in ALL_MIXES).",
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

    mixes = ALL_MIXES if args.only is None else args.only

    for mix_name in mixes:
        print(f"\n=== Building RSV_{mix_name} ===")
        if mix_name in SURVEILLANCE_LADDER:
            # Subsample frames to the chosen seasons, then replicate every
            # frame MULTIPLIER times. Same multiplier across 100S/50S/25S, so
            # the only difference is how many unique frames survive.
            counts = SURVEILLANCE_LADDER[mix_name]
            frame_df = select_recent_seasons(df, counts)
            mix_cfg = {src: {"multiplier": MULTIPLIER} for src in counts}
        else:
            frame_df = df
            mix_cfg = REFERENCE_POOLS[mix_name]

        frame_list = dataset_mixer.build_frames(
            frame_df,
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
