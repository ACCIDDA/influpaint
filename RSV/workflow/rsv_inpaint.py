#!/usr/bin/env python3
"""
rsv_inpaint.py

Turn ONE trained RSV checkpoint into a forecast for the held-out season, and
write it as a Hubverse model-output CSV (exactly the shape you would submit to
the CDC RSV Forecast Hub).

What inpainting does here, in plain terms
-----------------------------------------
The model learned to draw a whole RSV season as a 64x64 image (rows = weeks,
cols = US states). "Inpainting" = we paint in the weeks we have already
observed, lock those pixels, and let the model fill in the rest of the season.
The filled-in part is the forecast. We draw `batch_size` independent fills, so
the spread across those draws gives us forecast quantiles.

Two things make this RSV-specific (and are why we do NOT just reuse the flu
notebook as-is):

1. Scaling must match training. A diffusion model works on normalised numbers.
   To read its output back as real RSV values you must undo that normalisation
   with the *same* scaling used in training. That scaling is derived from the
   training .nc file, so you MUST pass the same-mix .nc the checkpoint was
   trained on (e.g. finetune_100A_*.pth  ->  RSV_100A_<tag>.nc). Using a
   different mix's .nc silently rescales every forecast.

2. Ground truth comes from the held-out RSV parquet, not a flu hub. We build it
   with GroundTruth.from_rsv(...) using one surveillance signal (default NHSN,
   which reports all states) for the held-out season. Whatever signal you pick
   is used both to condition on (the observed weeks) and to score against, so
   the forecast comes out in that signal's units.

Smoke-test usage (run from the repo root on Longleaf, one GPU):

    export PYTHONPATH=$PWD
    python RSV/workflow/rsv_inpaint.py \
        --checkpoint /proj/.../checkpoints/finetune_100A_scn868_ep500.pth \
        --dataset-nc training_datasets/RSV_100A_2026-04-24.nc \
        --reference-date 2024-12-21 \
        --signal NHSN \
        --season 2024 \
        --outdir RSV/forecasts_validation
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# --- repo paths -----------------------------------------------------------
# This file lives at <repo>/RSV/workflow/rsv_inpaint.py
SCRIPT_DIR = Path(__file__).resolve().parent          # .../RSV/workflow
REPO_ROOT = SCRIPT_DIR.parents[1]                     # repo root
sys.path.insert(0, str(SCRIPT_DIR))                   # import sibling rsv_training
sys.path.insert(0, str(REPO_ROOT / "CoPaint4influpaint"))

from influpaint.utils import SeasonAxis, ground_truth
from influpaint.utils.helpers import flusight_quantiles, flusight_quantile_pairs
from influpaint.batch.scenarios import get_training_scenario
from influpaint.batch.config import copaint_config_library, create_folders

# Reuse the EXACT model + transform builders the training script used, so the
# inference-time scaling is guaranteed identical to training.
from rsv_training import build_ddpm, build_rsv_dataset

from guided_diffusion import O_DDIMSampler


# ==========================================================================
# Hubverse export
# ==========================================================================
def export_hubverse_rsv(
    fluforecasts_ti,
    gt1,
    season_setup,
    reference_date,
    target,
    directory,
    team,
    model,
    signal="NHSN",
    horizons=(0, 1, 2, 3),
    save_plot=True,
):
    """
    Write one Hubverse model-output CSV from inpainted forecast trajectories.

    fluforecasts_ti : (n_samples, 1, n_weeks, n_places) inverse-transformed.
    Columns written : reference_date, target, horizon, target_end_date,
                      location, output_type, output_type_id, value.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    ref = pd.to_datetime(reference_date).normalize()
    image_size = gt1.image_size
    n_locs = len(season_setup.locations)

    # Row i of the forecast trajectory corresponds to base_index[i] (a Saturday
    # counted from season week 1). This matches how gt_xarr rows are laid out.
    season_start = datetime.date(
        int(gt1.season_first_year),
        season_setup.season_start_month,
        season_setup.season_start_day,
    )
    base_index = pd.date_range(
        season_start,
        season_start + datetime.timedelta(days=image_size * 7),
        freq="W-SAT",
    )

    target_end_dates = [ref + datetime.timedelta(days=7 * h) for h in horizons]
    horizon_of_date = {pd.Timestamp(ref + datetime.timedelta(days=7 * h)): h for h in horizons}

    # National = sum across states (per draw, per week).
    forecasts_national = fluforecasts_ti[:, :, :, :n_locs].sum(axis=-1)  # (n,1,weeks)

    rows = []
    for qt in flusight_quantiles:
        # state quantiles: (weeks, places)
        q_states = np.quantile(fluforecasts_ti[:, 0, :, :n_locs], qt, axis=0)
        q_us = np.quantile(forecasts_national[:, 0, :], qt, axis=0)        # (weeks,)

        wide = pd.DataFrame(q_states, index=base_index, columns=list(season_setup.locations))
        wide["US"] = q_us
        wide = wide.loc[wide.index.isin(target_end_dates)]

        tidy = wide.reset_index().rename(columns={"index": "target_end_date"})
        tidy = tidy.melt(id_vars="target_end_date", var_name="location", value_name="value")
        tidy["output_type_id"] = "{:.3f}".format(qt).rstrip("0").rstrip(".")
        rows.append(tidy)

    df = pd.concat(rows, ignore_index=True)
    df["reference_date"] = ref.date().isoformat()
    df["target"] = target
    df["horizon"] = df["target_end_date"].map(lambda d: horizon_of_date[pd.Timestamp(d)])
    df["target_end_date"] = pd.to_datetime(df["target_end_date"]).dt.date.astype(str)
    df["output_type"] = "quantile"
    # Hub requires value >= 0 and non-decreasing quantiles; clip tiny negatives.
    df["value"] = df["value"].clip(lower=0.0)
    df = df[
        ["reference_date", "target", "horizon", "target_end_date",
         "location", "output_type", "output_type_id", "value"]
    ].sort_values(["location", "horizon", "output_type_id"]).reset_index(drop=True)

    out_csv = directory / f"{ref.date().isoformat()}-{team}-{model}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Hubverse CSV written: {out_csv}  ({len(df)} rows)")

    if save_plot:
        _plot_national(fluforecasts_ti, gt1, season_setup, base_index, ref, directory, team, model, signal)

    return out_csv, df


def _plot_national(fluforecasts_ti, gt1, season_setup, base_index, ref, directory, team, model, signal):
    """Two panels: the full season the model fills (context, NOT submitted) and
    a zoom on the -1..3 week window we actually submit, with the held-out truth
    overlaid so you can see forecast vs reality."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_locs = len(season_setup.locations)
    national = fluforecasts_ti[:, :, :, :n_locs].sum(axis=-1)  # (n,1,weeks)
    med = np.quantile(national, 0.5, axis=0)[0]
    nweeks = med.shape[0]
    x = base_index[:nweeks]
    idx = gt1.inpaintfrom_idx

    # Full-season national truth (the held-out parquet has all weeks, even the
    # ones we masked), so we can show what actually happened after `ref`.
    truth = np.nan_to_num(gt1.gt_xarr.data[0].sum(axis=1), nan=0.0)[:nweeks]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=120)
    for panel, ax in enumerate(axes):
        for iqt in range(flusight_quantile_pairs.shape[0]):
            lo = np.quantile(national, flusight_quantile_pairs[iqt, 0], axis=0)[0]
            hi = np.quantile(national, flusight_quantile_pairs[iqt, 1], axis=0)[0]
            ax.fill_between(x, lo[:nweeks], hi[:nweeks], alpha=0.1, color="darkred")
        ax.plot(x, med, color="r", lw=2, label="median forecast")
        ax.plot(x[:idx], truth[:idx], "k.", ms=7, label=f"observed ({signal})")
        ax.plot(x[idx:52], truth[idx:52], ".", color="tab:blue", ms=6, label="held-out truth")
        ax.axvline(ref, c="k", ls="--", lw=1.2, alpha=0.6)
        ax.set_ylim(bottom=0)
        if panel == 0:
            ax.set_title("Full season the model fills (context, not submitted)")
        else:
            lo_w, hi_w = max(0, idx - 6), min(nweeks, idx + 4)
            ax.set_xlim(ref - pd.Timedelta(weeks=6), ref + pd.Timedelta(weeks=4))
            band_hi = np.quantile(national, 0.95, axis=0)[0][lo_w:hi_w]
            ymax = max(band_hi.max(), truth[lo_w:hi_w].max(), 1.0)
            ax.set_ylim(0, ymax * 1.15)
            ax.set_title("Submitted window: horizons 0..3 (3 weeks ahead)")
        ax.legend(fontsize=8)
    fig.suptitle(f"National RSV ({signal}) — ref {ref.date()} — {model}")
    fig.tight_layout()
    out_png = directory / f"{ref.date().isoformat()}-{team}-{model}-national.png"
    fig.savefig(out_png)
    print(f"Sanity plot written: {out_png}")


# ==========================================================================
# Main
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to the trained RSV .pth checkpoint.")
    p.add_argument("--dataset-nc", required=True,
                   help="Same-mix training .nc the checkpoint was trained on (sets the scaling).")
    p.add_argument("--reference-date", required=True,
                   help="Hubverse reference_date, a Saturday (YYYY-MM-DD). Data is treated as "
                        "known up to (not including) this date; horizons -1..3 are forecast.")
    p.add_argument("--validation-parquet", default=str(REPO_ROOT / "RSV/data/RSV_VALIDATION.parquet"),
                   help="Held-out RSV parquet to use as ground truth.")
    p.add_argument("--signal", default="NHSN",
                   help="Surveillance signal used as truth (datasetH1 value): NHSN / NSSP / RSV-Net. "
                        "NHSN (hospital admission counts) covers all states.")
    p.add_argument("--season", default="2024", help="Held-out season start year (fluseason).")
    p.add_argument("--target", default="wk inc rsv hosp",
                   help="Hubverse target id. Default matches RSV-Net hospitalizations.")
    p.add_argument("--scn-id", type=int, default=868, help="Scenario id (architecture). Must match training.")
    p.add_argument("--config-name", default="celebahq_noTTJ5", help="CoPaint config name.")
    p.add_argument("--batch-size", type=int, default=512, help="Number of forecast draws.")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--team", default="UNC_IDD", help="Hubverse team id (filename prefix).")
    p.add_argument("--model", default="InfluPaint-RSV", help="Hubverse model id (filename).")
    p.add_argument("--outdir", default=str(REPO_ROOT / "RSV/forecasts_validation"),
                   help="Where to write the CSV + sanity plot.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RSV inpainting ===")
    print(f"Device       : {device}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Training .nc  : {args.dataset_nc}  (sets scaling)")
    print(f"Signal/season: {args.signal} / {args.season}")
    print(f"Reference    : {args.reference_date}  target={args.target!r}")

    season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
    scenario_spec = get_training_scenario(args.scn_id)
    print(f"Scenario {args.scn_id}: {scenario_spec.scenario_string}")

    # ---- Model + matching transform --------------------------------------
    ddpm = build_ddpm(
        scenario_spec,
        image_size=args.image_size,
        channels=args.channels,
        batch_size=args.batch_size,
        epochs=1,            # unused at inference
        device=device,
    )
    dataset, scaling_per_channel = build_rsv_dataset(
        args.dataset_nc, args.channels, scenario_spec
    )
    print(f"Scaling per channel: {scaling_per_channel}")

    print("Loading checkpoint...")
    ddpm.load_model_checkpoint(args.checkpoint)

    # ---- Ground truth from the held-out RSV parquet ----------------------
    gt1 = ground_truth.GroundTruth.from_rsv(
        validation_parquet=args.validation_parquet,
        signal=args.signal,
        season_first_year=args.season,
        mask_date=pd.to_datetime(args.reference_date),
        channels=args.channels,
        image_size=args.image_size,
        season_setup=season_setup,
    )
    print(f"Ground-truth image shape: {gt1.gt_xarr.shape}")
    print(f"Known weeks: 1..{gt1.inpaintfrom_idx}  (inpainting weeks {gt1.inpaintfrom_idx + 1}..{args.image_size})")

    # ---- CoPaint sampler (generic; copied from the flu pipeline) ---------
    available_configs = copaint_config_library(ddpm.timesteps)
    if args.config_name not in available_configs:
        raise ValueError(f"Config '{args.config_name}' not in {list(available_configs.keys())}")
    conf = available_configs[args.config_name]

    sampler = O_DDIMSampler(
        use_timesteps=np.arange(ddpm.timesteps),
        conf=conf,
        betas=ddpm.betas,
        model_mean_type=None,
        model_var_type=None,
        loss_type=None,
    )

    gt_transformed = dataset.apply_transform(np.nan_to_num(gt1.gt_xarr.data, nan=0.0))
    gt_keep_mask = torch.from_numpy(gt1.gt_keep_mask).type(torch.FloatTensor).to(device)
    gt_tensor = torch.from_numpy(gt_transformed).type(torch.FloatTensor).to(device)

    print(f"Running CoPaint with {args.batch_size} draws (this takes a few minutes)...")
    result = sampler.p_sample_loop(
        model_fn=ddpm.model,
        shape=(args.batch_size, args.channels, args.image_size, args.image_size),
        conf=conf,
        model_kwargs={
            "gt": gt_tensor.repeat(args.batch_size, 1, 1, 1),
            "gt_keep_mask": gt_keep_mask.repeat(args.batch_size, 1, 1, 1),
            "mymodel": True,
        },
    )

    fluforecasts = np.array(result["sample"].cpu())
    fluforecasts_ti = dataset.apply_transform_inv(fluforecasts)
    print(f"Generated {len(fluforecasts)} draws, array shape {fluforecasts_ti.shape}")

    # ---- Export ----------------------------------------------------------
    out_csv, _ = export_hubverse_rsv(
        fluforecasts_ti=fluforecasts_ti,
        gt1=gt1,
        season_setup=season_setup,
        reference_date=args.reference_date,
        target=args.target,
        directory=args.outdir,
        team=args.team,
        model=args.model,
        signal=args.signal,
    )

    # Raw draws, for re-scoring / debugging.
    np.save(Path(args.outdir) / f"{pd.to_datetime(args.reference_date).date()}_draws_ti.npy", fluforecasts_ti)
    print("Done.")


if __name__ == "__main__":
    main()
