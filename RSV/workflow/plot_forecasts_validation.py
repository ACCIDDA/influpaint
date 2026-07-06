#!/usr/bin/env python3
"""
plot_forecasts_validation.py

State-level summary of the retrospective RSV inpainting forecasts, one figure per
model. Each figure is a 2 x len(states) grid (default columns: North Carolina,
New York, Texas, Florida):

    Top row  — 4-week-ahead forecasts.
        For each of a handful of reference dates (colored), the Hubverse horizons
        0..3 window: a quantile fan + median from the 512 conditional
        trajectories, and a dashed vertical line at the reference date. The solid
        black line is the observed (final) NHSN values.

    Bottom row — full-season forecasts.
        For the SAME reference dates, the forecast extends over the rest of the
        season: a quantile fan + median from the 512 trajectories, plus a few
        example trajectories drawn thin, and a dashed vertical line at the
        reference date. The solid black line is the reported NHSN hospitalizations.

Both rows share the same reference dates and colors, so a single legend maps
color -> reference date across the whole figure.

Data (written by rsv_inpaint.py) lives in RSV/forecasts_validation/<MODEL>/:
    <YYYY-MM-DD>_draws_ti.npy   shape (n_samples=512, 1, 64 weeks, 64 places)
Reference dates are discovered from these filenames. Week i of a draw maps to the
Saturday season_start + i weeks; the ground truth (and that week->date calendar)
is rebuilt with GroundTruth.from_rsv on the same held-out NHSN parquet the
forecasts used, so truth and forecast are aligned by construction.

Usage (from the repo root):
    export PYTHONPATH=$PWD
    python RSV/workflow/plot_forecasts_validation.py                  # all 8 models
    python RSV/workflow/plot_forecasts_validation.py --model RSVpaint-100S
    python RSV/workflow/plot_forecasts_validation.py --n-refs 6 --states NC NY TX FL
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- repo paths -----------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # .../RSV/workflow
REPO_ROOT = SCRIPT_DIR.parents[1]                     # repo root
sys.path.insert(0, str(REPO_ROOT))

from influpaint.utils import SeasonAxis, ground_truth  # noqa: E402  (after sys.path setup)

IMAGE_SIZE = 64
N_LOCS = 51          # real FluSight locations; draw images are padded to 64 cols
HORIZONS = (0, 1, 2, 3)  # Hubverse horizons shown in the top "4-week-ahead" row

# Two nested prediction intervals per forecast (outer 95%, inner 50%). Overlaying
# the full 23-quantile FluSight fan for several reference dates turns into mud, so
# we draw just these two bands -> the colors stay distinguishable.
BANDS = ((0.025, 0.975, 0.08), (0.25, 0.75, 0.16))   # (q_lo, q_hi, alpha)


# ==========================================================================
# Ground truth + week->date calendar (model-independent; built once)
# ==========================================================================
def load_truth(parquet, signal, season, season_setup):
    """Rebuild the held-out ground truth exactly as rsv_inpaint.py conditions on
    it. Returns (dates, truth_arr) where dates is a length-64 DatetimeIndex of
    Saturdays and truth_arr is (64 weeks, 64 places) on the season_setup location
    order (NaN where a place has no report)."""
    gt = ground_truth.GroundTruth.from_rsv(
        validation_parquet=str(parquet),
        signal=signal,
        season_first_year=str(season),
        # gt_xarr.data carries the WHOLE season regardless of mask_date (only the
        # keep-mask depends on it), so any date works for pulling out the truth.
        mask_date=pd.to_datetime(f"{season}-12-07"),
        channels=1,
        image_size=IMAGE_SIZE,
        season_setup=season_setup,
    )
    dates = pd.to_datetime(gt.gt_xarr.coords["date"].values)
    truth_arr = np.asarray(gt.gt_xarr.data[0], dtype=float)   # (weeks, places)
    return dates, truth_arr


def state_to_code(state, season_setup):
    """'US', a FIPS code like '37', or an abbreviation like 'NC' -> location code."""
    if state.upper() == "US":
        return "US"
    codes = set(season_setup.locations_df["location_code"].astype(str))
    if state in codes:
        return str(state)
    m = season_setup.locations_df[
        season_setup.locations_df["abbreviation"].str.upper() == state.upper()
    ]
    if not m.empty:
        return str(m.iloc[0]["location_code"])
    raise ValueError(f"Unknown state '{state}'")


def series_for_location(arr2d, loc_code, locs):
    """Pick one location's column (or sum the 51 real states for 'US').
    arr2d trailing axis is places. Works for truth (weeks, places) and for a
    single draw's (weeks, places)."""
    if loc_code == "US":
        return arr2d[..., :N_LOCS].sum(axis=-1)
    return arr2d[..., locs.index(loc_code)]


# ==========================================================================
# Forecast discovery
# ==========================================================================
def discover_refs(model_dir):
    """Return sorted [(ref_date: Timestamp, npy_path: Path), ...] for a model."""
    out = []
    for p in sorted(Path(model_dir).glob("*_draws_ti.npy")):
        stamp = p.name[: -len("_draws_ti.npy")]
        try:
            out.append((pd.to_datetime(stamp), p))
        except (ValueError, TypeError):
            continue
    return sorted(out)


def pick_reference_dates(refs, xlim, n_refs):
    """Keep refs inside the x-window, then take n_refs of them evenly spaced."""
    lo, hi = xlim
    in_win = [(r, p) for (r, p) in refs if lo <= r <= hi]
    if not in_win:
        return []
    if n_refs >= len(in_win):
        return in_win
    idx = np.linspace(0, len(in_win) - 1, n_refs).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [in_win[i] for i in idx]


# ==========================================================================
# Plotting
# ==========================================================================
def plot_model(model_dir, model_id, dates, truth_arr, season_setup,
               states, xlim, n_refs, n_traj, signal, out_path):
    locs = list(season_setup.locations)
    refs = discover_refs(model_dir)
    picked = pick_reference_dates(refs, xlim, n_refs)
    if not picked:
        print(f"  [skip] {model_id}: no draws_ti.npy inside the window")
        return None

    # Load each selected draw file once; share across the row pair.
    draws = {ref: np.load(path) for ref, path in picked}
    n_samples = next(iter(draws.values())).shape[0]   # conditional trajectories per forecast

    # Color per reference date (shared by both rows -> one legend).
    cmap = plt.get_cmap("Dark2" if len(picked) <= 8 else "turbo")
    if len(picked) <= 8:
        colors = [cmap(i) for i in range(len(picked))]
    else:
        colors = [cmap(i / (len(picked) - 1)) for i in range(len(picked))]
    ref_color = {ref: colors[i] for i, (ref, _) in enumerate(picked)}

    win = (dates >= xlim[0]) & (dates <= xlim[1])
    end_row = int(np.where(win)[0].max())          # last week kept on the x-axis

    n = len(states)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7.2), dpi=150,
                             sharex=True, squeeze=False)

    for j, st in enumerate(states):
        loc = state_to_code(st, season_setup)
        name = "United States" if loc == "US" else season_setup.get_location_name(loc)
        truth = series_for_location(truth_arr, loc, locs)

        ax_top, ax_bot = axes[0, j], axes[1, j]
        for ax in (ax_top, ax_bot):
            ax.plot(dates[win], truth[win], color="black", lw=2, zorder=5)

        for ref, path in picked:
            c = ref_color[ref]
            ts = np.clip(series_for_location(draws[ref][:, 0], loc, locs), 0, None)  # (samples, weeks)
            r0 = int(np.argmin(np.abs(dates - ref)))   # week index of the reference date

            # --- Top: 4-week-ahead (horizons 0..3) ------------------------
            top_rows = [r0 + h for h in HORIZONS if r0 + h <= end_row]
            if top_rows:
                xt = dates[top_rows]
                seg = ts[:, top_rows]
                for lo_q, hi_q, a in BANDS:
                    ax_top.fill_between(xt, np.quantile(seg, lo_q, axis=0),
                                        np.quantile(seg, hi_q, axis=0),
                                        color=c, alpha=a, lw=0)
                ax_top.plot(xt, np.quantile(seg, 0.5, axis=0), color=c, lw=2.0, zorder=4)
            ax_top.axvline(ref, color=c, ls="--", lw=1.0, alpha=0.8)

            # --- Bottom: full-season forecast -----------------------------
            fut = list(range(r0, end_row + 1))
            xf = dates[fut]
            seg = ts[:, fut]
            for lo_q, hi_q, a in BANDS:
                ax_bot.fill_between(xf, np.quantile(seg, lo_q, axis=0),
                                    np.quantile(seg, hi_q, axis=0),
                                    color=c, alpha=a * 0.7, lw=0)
            if n_traj > 0:
                samp = np.linspace(0, seg.shape[0] - 1, min(n_traj, seg.shape[0])).astype(int)
                for si in samp:
                    ax_bot.plot(xf, seg[si], color=c, alpha=0.32, lw=0.6, zorder=2)
            ax_bot.plot(xf, np.quantile(seg, 0.5, axis=0), color=c, lw=2.0, zorder=4)
            ax_bot.axvline(ref, color=c, ls="--", lw=1.0, alpha=0.8)

        # Styling
        ax_top.set_title(name, fontsize=12, fontweight="bold")
        for ax in (ax_top, ax_bot):
            ax.set_ylim(bottom=0)
            ax.set_xlim(*xlim)
            ax.grid(True, alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            for lab in ax.get_xticklabels():
                lab.set_rotation(45)
                lab.set_horizontalalignment("right")
        if j == 0:
            ax_top.set_ylabel(f"4-week-ahead\nincident RSV hosp. ({signal})")
            ax_bot.set_ylabel(f"Full season\nincident RSV hosp. ({signal})")

    # Shared legend: reference-date colors + observed line + trajectory style.
    handles = [Line2D([0], [0], color="black", lw=2, label=f"Observed ({signal})")]
    handles += [Patch(facecolor=ref_color[ref], label=ref.date().isoformat())
                for ref, _ in picked]
    handles.append(Line2D([0], [0], color="0.4", lw=0.6, alpha=0.6,
                          label=f"{n_traj} example trajectories"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"RSV forecast validation — {model_id}  "
                 f"(season {season_setup_year(dates)}, {signal}, {n_samples} trajectories)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}  ({len(picked)} reference dates: "
          f"{', '.join(r.date().isoformat() for r, _ in picked)})")
    return out_path


def season_setup_year(dates):
    """Human label for the season (the Aug start year of the first week)."""
    return f"{dates[0].year}-{dates[0].year + 1}"


# ==========================================================================
# Main
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forecasts-dir", default=str(REPO_ROOT / "RSV/forecasts_validation"),
                   help="Folder holding one sub-folder of draws per model.")
    p.add_argument("--model", default=None,
                   help="Single model sub-folder to plot (default: every sub-folder).")
    p.add_argument("--validation-parquet",
                   default=str(REPO_ROOT / "RSV/data/RSV_VALIDATION.parquet"),
                   help="Held-out RSV parquet used as ground truth.")
    p.add_argument("--signal", default="NHSN",
                   help="Ground-truth signal the forecasts were built on (NHSN default).")
    p.add_argument("--season", default="2025", help="Held-out season start year (2025 = 2025-26 season).")
    p.add_argument("--states", nargs="+", default=["NC", "NY", "TX", "FL"],
                   help="Columns: abbreviations, FIPS codes, or US.")
    p.add_argument("--n-refs", type=int, default=5,
                   help="How many reference dates to show (same set in both rows).")
    p.add_argument("--n-traj", type=int, default=10,
                   help="Example trajectories per reference date (bottom row).")
    p.add_argument("--xlim-start", default="2025-10-01", help="x-axis left bound (YYYY-MM-DD).")
    p.add_argument("--xlim-end", default="2026-05-31", help="x-axis right bound (YYYY-MM-DD).")
    p.add_argument("--outdir", default=None,
                   help="Where to write PNGs (default: alongside each model's draws).")
    return p.parse_args()


def main():
    args = parse_args()
    season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
    xlim = (pd.to_datetime(args.xlim_start), pd.to_datetime(args.xlim_end))

    print(f"Loading ground truth ({args.signal}, season {args.season})...")
    dates, truth_arr = load_truth(args.validation_parquet, args.signal,
                                  args.season, season_setup)

    forecasts_dir = Path(args.forecasts_dir)
    if args.model:
        model_dirs = [forecasts_dir / args.model]
    else:
        model_dirs = sorted(d for d in forecasts_dir.iterdir() if d.is_dir())
    if not model_dirs:
        raise SystemExit(f"No model folders found under {forecasts_dir}")

    for model_dir in model_dirs:
        model_id = model_dir.name
        out_dir = Path(args.outdir) if args.outdir else model_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_id}_forecast_validation.png"
        print(f"[{model_id}]")
        plot_model(model_dir, model_id, dates, truth_arr, season_setup,
                   args.states, xlim, args.n_refs, args.n_traj, args.signal, out_path)


if __name__ == "__main__":
    main()
