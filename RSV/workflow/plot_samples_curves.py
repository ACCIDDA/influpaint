#!/usr/bin/env python3
"""
plot_samples_curves.py

For each trained RSV checkpoint, load its saved sanity samples, inverse-
transform them back to incidence-count space (the same transform used at
training), and plot per-location curves with the held-out 2023-24 RSV
season overlaid as ground truth.

One PNG per checkpoint is written next to the source samples file.

Usage (from repo root, on Longleaf):
    python RSV/workflow/plot_samples_curves.py

Picks up every samples_*.npy under RESULTS_DIRS that matches a checkpoint
in CHECKPOINTS_DIR. Edit the constants at the top if your paths differ.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from influpaint.utils import SeasonAxis, converters
from influpaint.batch.scenarios import get_training_scenario
from influpaint.batch.config import transform_library
from influpaint.datasets import loaders as training_datasets


# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = "/proj/jlessler/projects/influpaint_general/influpaint_RSV/checkpoints/"
RESULTS_ROOT = "/proj/jlessler/projects/influpaint_general/influpaint_RSV/results/"
DATASET_DIR = "training_datasets"
DATE_TAG = "2026-06-23"
SCN_ID = 868
VALIDATION_PARQUET = "RSV/data/RSV_VALIDATION.parquet"
GT_SOURCE = "NSSP"     # which dataset in the validation file to plot as truth
N_LOCS_TO_PLOT = 10    # number of states to show per figure
MAX_SAMPLES_TO_DRAW = 50
# ---------------------------------------------------------------------------


def build_dataset_for_mix(mix_name):
    """Rebuild the training dataset wrapper for one mix - we only need it
    to recover the right `transform_inv`."""
    nc_path = f"{DATASET_DIR}/RSV_{mix_name}_{DATE_TAG}.nc"
    scenario_spec = get_training_scenario(SCN_ID)
    dataset = training_datasets.FluDataset.from_xarray(nc_path, channels=1)
    scaling = np.array(dataset.max_per_feature)
    tfs, enrichs = transform_library(
        scaling,
        data_mean=dataset.flu_dyn.mean(),
        data_std=dataset.flu_dyn.std(),
    )
    t = tfs[scenario_spec.transform_name]
    dataset.add_transform(
        transform=t["reg"],
        transform_inv=t["inv"],
        transform_enrich=enrichs[scenario_spec.enrich_name],
        bypass_test=False,
    )
    return dataset


def load_ground_truth(season_setup, source=GT_SOURCE):
    """Return the held-out RSV season (1, 1, 64, 64), in raw incidence units.
    Locations missing from `source` are filled with NaN so the line just stops."""
    df = pd.read_parquet(VALIDATION_PARQUET)
    df = df.rename(columns={"fluseason_week": "season_week"})
    df = df[df["datasetH2"] == source]
    df = (
        df.groupby(["fluseason", "season_week", "location_code"])["value"]
        .mean()
        .reset_index()
    )
    # Inject any missing locations as NaN so dataframe_to_arraylist sees them.
    missing = [loc for loc in season_setup.locations if loc not in df["location_code"].unique()]
    if missing:
        print(f"  note: {source} has no data for {len(missing)} locations: {missing}")
        seasons = df["fluseason"].unique()
        filler = pd.DataFrame(
            [(s, w, loc, np.nan) for s in seasons for w in range(1, 54) for loc in missing],
            columns=["fluseason", "season_week", "location_code", "value"],
        )
        df = pd.concat([df, filler], ignore_index=True)
    arrs = converters.dataframe_to_arraylist(df, season_setup=season_setup)
    return np.array(arrs)  # (n_seasons, 1, 64, 64)


def find_samples_npy(ckpt_filename):
    """Map scratch_100A_scratch_scn868_ep3000.pth -> samples_scratch_100A_scratch_scn868.npy.
    Searches every results/*/ folder; if multiple match (e.g. a run was redone),
    returns the most recently modified one."""
    stem = ckpt_filename.rsplit("_ep", 1)[0]
    target = f"samples_{stem}.npy"
    matches = list(Path(RESULTS_ROOT).glob(f"*/{target}"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def mix_from_filename(name):
    m = re.search(r"(100A|100M|100S|50S|25S)", name)
    return m.group(1) if m else None


def inverse_transform_batch(dataset, samples):
    """Apply dataset.apply_transform_inv per sample; return a numpy array."""
    out = []
    for i in range(samples.shape[0]):
        out.append(np.asarray(dataset.apply_transform_inv(samples[i])))
    return np.stack(out, axis=0)


def make_figure(ckpt_path, dataset, gt_array, season_setup):
    """Returns the matplotlib Figure, or None if samples can't be found."""
    samples_path = find_samples_npy(ckpt_path.name)
    if samples_path is None:
        print(f"  skip - no samples .npy found")
        return None, None
    samples = np.load(samples_path)        # (N, 1, 64, 64) in [-1, 1]-ish space
    samples_ti = inverse_transform_batch(dataset, samples)  # back to counts

    locs = season_setup.locations
    n = min(N_LOCS_TO_PLOT, len(locs))
    step = max(1, len(locs) // n)
    loc_idx = list(range(0, n * step, step))

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax, i in zip(axes.flat, loc_idx):
        for s in range(min(MAX_SAMPLES_TO_DRAW, samples_ti.shape[0])):
            ax.plot(samples_ti[s, 0, :53, i], color="black", alpha=0.12, lw=0.6)
        ax.plot(
            gt_array[0, 0, :53, i],
            color="red", lw=2, label=f"GT ({GT_SOURCE} 2023-24)",
        )
        ax.set_title(locs[i], fontsize=9)
        ax.set_xlabel("Season week")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(ckpt_path.stem, fontsize=11)
    fig.tight_layout()
    return fig, samples_path


def main():
    season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
    print(f"Loading ground truth from {VALIDATION_PARQUET} (source={GT_SOURCE})...")
    gt = load_ground_truth(season_setup)

    pdf_path = Path(CHECKPOINTS_DIR) / "rsv_samples_curves.pdf"
    with PdfPages(pdf_path) as pdf:
        for ckpt in sorted(Path(CHECKPOINTS_DIR).glob("*.pth")):
            mix = mix_from_filename(ckpt.name)
            if not mix:
                continue
            print(f"\n{ckpt.name}  (mix={mix})")
            dataset = build_dataset_for_mix(mix)
            fig, samples_path = make_figure(ckpt, dataset, gt, season_setup)
            if fig is None:
                continue
            out_png = samples_path.with_name(ckpt.stem + "_curves.png")
            fig.savefig(out_png, dpi=120)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  saved {out_png}")
    print(f"\nCombined PDF: {pdf_path}")


if __name__ == "__main__":
    main()
