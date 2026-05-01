import argparse
import os
import subprocess
import tempfile
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from influpaint.utils import SeasonAxis

from prepare_dataset_for_scoringutils import Config as ScoringConfig

from .config import _MODEL_NUM, SEASON_XLIMS, INPAINTING_BASE, BEST_MODEL_ID, BEST_CONFIG
from .helpers import forecast_week_saturdays, list_inpainting_dirs, parse_date_from_folder, state_to_code
from .csv_forecasts import FLUSIGHT_BASES


TARGET_NAME = ScoringConfig.TARGET_NAME
SEASONS = ("2023-2024", "2024-2025")
STATES = ("US", "CA", "NY", "TX")
QUANTILES = np.array(ScoringConfig.REQUIRED_QUANTILES, dtype=float)
PER_SEASON_PICK = 4


def pick_figure3_paths(season_axis: SeasonAxis) -> List[Tuple[str, pd.Timestamp, str]]:
    dirs = list_inpainting_dirs(INPAINTING_BASE, BEST_MODEL_ID, BEST_CONFIG)
    picked: List[Tuple[str, pd.Timestamp, str]] = []
    for season in SEASONS:
        season_year = int(season.split("-")[0])
        dated_dirs: List[Tuple[pd.Timestamp, str]] = []
        for d in dirs:
            ref_date = parse_date_from_folder(os.path.basename(d))
            if ref_date is None:
                raise ValueError(f"Could not parse reference date from inpainting directory: {d}")
            if season_axis.get_fluseason_year(pd.to_datetime(ref_date)) == season_year:
                dated_dirs.append((pd.to_datetime(ref_date), d))
        dated_dirs = sorted(dated_dirs)[1:]
        step = max(1, len(dated_dirs) // PER_SEASON_PICK)
        chosen = dated_dirs[::step][:PER_SEASON_PICK]
        if len(chosen) != PER_SEASON_PICK:
            raise ValueError(f"Expected {PER_SEASON_PICK} reference dates for {season}, found {len(chosen)}")
        for ref_date, d in chosen:
            picked.append((season, ref_date, d))
    expected = len(SEASONS) * PER_SEASON_PICK
    if len(picked) != expected:
        raise ValueError(f"Expected {expected} picked paths, found {len(picked)}")
    return picked


def run_model_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    parts = base.split("::")
    if len(parts) < 3:
        raise ValueError(f"Unexpected inpainting path format: {base}")
    return "::".join(parts[:-1]).replace("::conf_", "::")


def load_truth_for_seasons() -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        truth_path = os.path.join(
            FLUSIGHT_BASES[season], "target-data", "target-hospital-admissions.csv"
        )
        if not os.path.exists(truth_path):
            raise FileNotFoundError(f"Missing truth file: {truth_path}")
        truth = pd.read_csv(truth_path, dtype={"location": str})
        truth["season"] = season
        truth["target_end_date"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d")
        truth["observed"] = pd.to_numeric(truth["value"], errors="raise")
        rows.append(truth[["season", "target_end_date", "location", "observed"]])
    return pd.concat(rows, ignore_index=True)


def state_maps(season_axis: SeasonAxis) -> tuple[dict[str, str], dict[str, str]]:
    state_to_loc = {state: state_to_code(state, season_axis) for state in STATES}
    loc_to_state = {loc: state for state, loc in state_to_loc.items()}
    return state_to_loc, loc_to_state


def _build_long_forecasts(season_axis: SeasonAxis) -> pd.DataFrame:
    state_to_loc, _ = state_maps(season_axis)
    picked = pick_figure3_paths(season_axis)
    rows = []
    model_names = set()

    for season, ref_date_ts, dpath in picked:
        ref_date = ref_date_ts.date()
        season_end = pd.to_datetime(SEASON_XLIMS[season][1]).date()
        model_name = run_model_name_from_path(dpath)
        model_names.add(model_name)

        arr = np.load(os.path.join(dpath, "fluforecasts_ti.npy"))
        sat = pd.to_datetime(forecast_week_saturdays(season, season_axis, arr.shape[2])).to_list()
        week_dates = [pd.to_datetime(x).date() for x in sat]
        if ref_date not in set(week_dates):
            raise ValueError(f"Reference date {ref_date} not present in week grid for {dpath}")

        for state in STATES:
            loc_code = state_to_loc[state]
            if loc_code == "US":
                ts = arr[:, 0, :, :len(season_axis.locations)].sum(axis=-1)
            else:
                state_idx = season_axis.locations.index(loc_code)
                ts = arr[:, 0, :, state_idx]

            for week_idx, target_date in enumerate(week_dates):
                if target_date < ref_date:
                    continue
                if target_date > season_end:
                    continue
                delta_days = (target_date - ref_date).days
                if delta_days % 7 != 0:
                    raise ValueError(
                        f"Target date {target_date} is not whole-week aligned to reference date {ref_date}"
                    )
                horizon = delta_days // 7
                if horizon < 3:
                    continue
                qvals = np.quantile(ts[:, week_idx], QUANTILES)
                for q, pred in zip(QUANTILES, qvals):
                    rows.append(
                        {
                            "model": model_name,
                            "group": "influpaint",
                            "season": season,
                            "reference_date": str(ref_date),
                            "forecast_date": str(ref_date),
                            "target_end_date": str(target_date),
                            "location": loc_code,
                            "horizon": int(horizon),
                            "quantile": float(q),
                            "predicted": float(pred),
                            "target": TARGET_NAME,
                            "output_type": "quantile",
                        }
                    )

    if len(model_names) != 1:
        raise ValueError(f"Expected one model from picked NPY directories, found {sorted(model_names)}")

    forecasts = pd.DataFrame(rows)
    forecasts["location"] = forecasts["location"].astype(str)
    forecasts["target_end_date"] = pd.to_datetime(forecasts["target_end_date"]).dt.strftime("%Y-%m-%d")
    forecasts["quantile"] = pd.to_numeric(forecasts["quantile"], errors="raise").round(6)

    unit_cols = ["model", "season", "reference_date", "target_end_date", "location", "horizon"]
    dup = forecasts[forecasts.duplicated(unit_cols + ["quantile"], keep=False)]
    if not dup.empty:
        raise ValueError("Duplicate long-horizon quantile rows found")

    counts = forecasts.groupby(unit_cols).size()
    if not np.all(counts.values == len(QUANTILES)):
        raise ValueError("Each long-horizon forecast unit must contain the full quantile grid")

    return forecasts


def _build_flusight_h4_forecasts() -> pd.DataFrame:
    rows = []
    qset = sorted(np.round(QUANTILES, 6).tolist())
    wanted_locs = {"US", "06", "36", "48"}

    for season in SEASONS:
        model_dir = os.path.join(FLUSIGHT_BASES[season], "model-output", "FluSight-ensemble")
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Missing FluSight ensemble directory: {model_dir}")
        files = sorted([f for f in os.listdir(model_dir) if f.endswith("-FluSight-ensemble.csv")])

        for fname in files:
            ref = fname.split("-FluSight-ensemble.csv")[0]
            csv_path = os.path.join(model_dir, fname)
            df = pd.read_csv(csv_path, dtype={"location": str})
            df["target_end_date"] = pd.to_datetime(df["target_end_date"]).dt.strftime("%Y-%m-%d")

            sub = df[
                (df["target"] == TARGET_NAME)
                & (df["output_type"] == "quantile")
                & (df["horizon"] == 3)
                & (df["location"].isin(wanted_locs))
            ].copy()
            if sub.empty:
                continue
            sub["horizon"] = pd.to_numeric(sub["horizon"], errors="raise").astype(int)
            sub["output_type_id"] = pd.to_numeric(sub["output_type_id"], errors="raise")

            for loc in sorted(wanted_locs):
                per_loc = sub[sub["location"] == loc]
                if per_loc.empty:
                    continue
                ted = sorted(per_loc["target_end_date"].unique().tolist())
                if len(ted) != 1:
                    raise ValueError(
                        f"Expected one target_end_date for season={season}, ref={ref}, loc={loc}; found {ted}"
                    )
                have = sorted(np.round(per_loc["output_type_id"].astype(float).unique(), 6).tolist())
                if have != qset:
                    raise ValueError(
                        f"Quantile mismatch season={season}, ref={ref}, loc={loc}. Have={have}"
                    )

            sub = sub.rename(columns={"output_type_id": "quantile", "value": "predicted"})
            sub["model"] = "FluSight-ensemble"
            sub["group"] = "flusight"
            sub["season"] = season
            sub["reference_date"] = ref
            sub["forecast_date"] = ref
            rows.append(
                sub[
                    [
                        "model",
                        "group",
                        "season",
                        "reference_date",
                        "forecast_date",
                        "target_end_date",
                        "location",
                        "horizon",
                        "quantile",
                        "predicted",
                        "target",
                        "output_type",
                    ]
                ]
            )

    if not rows:
        raise ValueError("No FluSight horizon-3 quantile rows found")

    forecasts = pd.concat(rows, ignore_index=True)
    forecasts["quantile"] = pd.to_numeric(forecasts["quantile"], errors="raise").round(6)
    forecasts["predicted"] = pd.to_numeric(forecasts["predicted"], errors="raise")
    return forecasts


def _run_scoringutils(combined_csv: str, score_csv: str) -> None:
    rscript_bin = "/usr/local/bin/Rscript"
    if not os.path.exists(rscript_bin):
        raise FileNotFoundError(f"Required Rscript binary not found: {rscript_bin}")
    score_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "score_with_scoringutils.R"))
    cmd = [rscript_bin, score_script, combined_csv, score_csv]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "scoringutils run failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())


def _build_scoring_input(forecasts: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "group",
        "season",
        "reference_date",
        "forecast_date",
        "target_end_date",
        "location",
        "horizon",
        "quantile",
        "predicted",
        "target",
        "output_type",
    }
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Forecasts missing required scoring columns: {sorted(missing)}")

    truth = load_truth_for_seasons()
    truth["location"] = truth["location"].astype(str)
    truth["target_end_date"] = pd.to_datetime(truth["target_end_date"]).dt.strftime("%Y-%m-%d")
    combined = forecasts.merge(
        truth,
        on=["season", "target_end_date", "location"],
        how="inner",
    )
    if len(combined) != len(forecasts):
        raise ValueError(f"Merge lost rows: forecasts={len(forecasts)}, combined={len(combined)}")
    return combined


def _compute_horizon_summary(
    long_scores: pd.DataFrame,
    flusight_scores: pd.DataFrame,
    min_target_dates_per_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"model", "season", "reference_date", "location", "horizon", "wis", "target_end_date"}
    for name, df in [("long", long_scores), ("flusight", flusight_scores)]:
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{name} scores missing columns: {sorted(missing)}")
        df["reference_date"] = pd.to_datetime(df["reference_date"]).dt.strftime("%Y-%m-%d")
        df["location"] = df["location"].astype(str)
        df["horizon"] = pd.to_numeric(df["horizon"], errors="raise").astype(int)
        df["wis"] = pd.to_numeric(df["wis"], errors="raise")
        df["target_end_date"] = pd.to_datetime(df["target_end_date"]).dt.strftime("%Y-%m-%d")

    candidates = sorted(
        m for m in long_scores["model"].dropna().unique().tolist() if m != "FluSight-ensemble"
    )
    if len(candidates) != 1:
        raise ValueError(f"Could not infer unique influpaint model from long scores. Found: {candidates}")
    influpaint_model = candidates[0]

    wanted_locs = {"US", "06", "36", "48"}

    flusight_h4 = flusight_scores[
        (flusight_scores["model"] == "FluSight-ensemble")
        & (flusight_scores["location"].isin(wanted_locs))
        & (flusight_scores["horizon"] == 3)
    ][["season", "reference_date", "target_end_date", "location", "wis"]].rename(
        columns={"reference_date": "flusight_reference_date", "wis": "wis_flusight_h4"}
    )

    long_inf = long_scores[
        (long_scores["model"] == influpaint_model)
        & (long_scores["location"].isin(wanted_locs))
        & (long_scores["horizon"] >= 3)
    ][["season", "reference_date", "target_end_date", "location", "horizon", "wis"]].rename(
        columns={"reference_date": "long_reference_date", "wis": "wis_long_influpaint"}
    )

    merged = long_inf.merge(flusight_h4, on=["season", "target_end_date", "location"], how="inner")
    if merged.empty:
        raise ValueError("No overlap between long-horizon Influpaint and FluSight h=4 scores")

    target_counts = merged.groupby("season")["target_end_date"].nunique()
    low = target_counts[target_counts < min_target_dates_per_season]
    if not low.empty:
        raise ValueError(
            "Too few target dates after merge. "
            f"Counts={target_counts.to_dict()} and threshold={min_target_dates_per_season}"
        )

    merged["long_term_horizon"] = merged["horizon"] + 1
    merged["ratio_paint_over_flusightensembleh4"] = merged["wis_long_influpaint"] / merged["wis_flusight_h4"]
    merged["paint_beats_flusightensembleh4"] = merged["ratio_paint_over_flusightensembleh4"] < 1.0
    merged["target_end_date"] = pd.to_datetime(merged["target_end_date"])
    merged["location_name"] = merged["location"].map({"US": "US", "06": "CA", "36": "NY", "48": "TX"})
    merged = merged.sort_values(["season", "location_name", "long_reference_date", "long_term_horizon"])

    summary = (
        merged.groupby("long_term_horizon", as_index=False)
        .agg(
            avg_rel_wis_paint_over_flusightensembleh4=(
                "ratio_paint_over_flusightensembleh4",
                "mean",
            ),
            pct_realizations_beat_flusightensembleh4=(
                "paint_beats_flusightensembleh4",
                lambda x: 100.0 * float(np.mean(x)),
            ),
        )
        .sort_values("long_term_horizon")
    )
    return summary, merged


def _plot_initial_comparison(merged: pd.DataFrame, out_png: str, log_y: bool) -> None:
    panel_order = []
    for season in ["2023-2024", "2024-2025"]:
        for loc in ["US", "CA", "NY", "TX"]:
            panel_order.append(f"{season} | {loc}")
    merged = merged.copy()
    merged["panel"] = merged["season"].astype(str) + " | " + merged["location_name"].astype(str)
    merged["panel"] = pd.Categorical(merged["panel"], categories=panel_order, ordered=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=250, sharex=False, sharey=False)
    flat_axes = axes.flatten()
    for idx, panel in enumerate(panel_order):
        ax = flat_axes[idx]
        sub = merged[merged["panel"] == panel]
        season_dates = sorted(sub["long_reference_date"].unique())
        season_palette = sns.color_palette("Dark2", n_colors=max(1, len(season_dates)))
        season_color_map = {
            ref_date: season_palette[i] for i, ref_date in enumerate(season_dates)
        }
        for ref in sorted(sub["long_reference_date"].unique()):
            s = sub[sub["long_reference_date"] == ref].sort_values("long_term_horizon")
            ax.plot(
                s["target_end_date"],
                s["ratio_paint_over_flusightensembleh4"],
                marker="o",
                lw=1.4,
                color=season_color_map[ref],
            )
        ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.set_title(panel, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        for label in ax.get_xticklabels():
            label.set_rotation(0)
            label.set_horizontalalignment("center")
        if log_y:
            ratios = sub["ratio_paint_over_flusightensembleh4"].to_numpy(dtype=float)
            if np.any(ratios <= 0):
                raise ValueError("Cannot use --log-y because detailed ratio plot contains non-positive values.")
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        sns.despine(ax=ax, trim=True)

    for idx, ax in enumerate(flat_axes):
        row = idx // 4
        col = idx % 4
        if row == 1:
            ax.set_xlabel("Target date")
        else:
            ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel("WIS ratio (Influpaint long-term / FluSight 4 wk)")
        else:
            ax.set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot_horizon_summary(summary: pd.DataFrame, out_png: str, log_y: bool) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=220)

    x = summary["long_term_horizon"].to_numpy(dtype=int)
    y_ratio = summary["avg_rel_wis_paint_over_flusightensembleh4"].to_numpy(dtype=float)
    y_beat = summary["pct_realizations_beat_flusightensembleh4"].to_numpy(dtype=float)

    ax1.plot(x, y_ratio, marker="o", lw=1.8, color="#1f77b4")
    ax1.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
    if log_y:
        if np.any(y_ratio <= 0):
            raise ValueError("Cannot use --log-y because avg relative WIS contains non-positive values.")
        ax1.set_yscale("log")
    ax1.set_xlabel("Long-term horizon (+x weeks)")
    ax1.set_ylabel("Average relative WIS (Influpaint / FluSight 4 wk)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, y_beat, marker="s", lw=1.6, color="#d62728")
    ax2.set_ylabel("Realizations beating FluSight 4 wk (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(
    out_dir: str, prefix: str, min_target_dates_per_season: int, log_y: bool
) -> tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)

    season_axis = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
    long_forecasts = _build_long_forecasts(season_axis)
    flusight_h4_forecasts = _build_flusight_h4_forecasts()

    long_combined = _build_scoring_input(long_forecasts)
    flusight_combined = _build_scoring_input(flusight_h4_forecasts)

    with tempfile.TemporaryDirectory(prefix="figure3_ratio_analysis_") as tmpdir:
        long_combined_csv = os.path.join(tmpdir, "long_combined.csv")
        long_scores_csv = os.path.join(tmpdir, "long_scores.csv")
        flusight_combined_csv = os.path.join(tmpdir, "flusight_h4_combined.csv")
        flusight_scores_csv = os.path.join(tmpdir, "flusight_h4_scores.csv")

        long_combined.to_csv(long_combined_csv, index=False)
        flusight_combined.to_csv(flusight_combined_csv, index=False)

        _run_scoringutils(long_combined_csv, long_scores_csv)
        _run_scoringutils(flusight_combined_csv, flusight_scores_csv)

        long_scores = pd.read_csv(long_scores_csv, dtype={"location": str})
        flusight_scores = pd.read_csv(flusight_scores_csv, dtype={"location": str})

    summary, merged = _compute_horizon_summary(
        long_scores=long_scores,
        flusight_scores=flusight_scores,
        min_target_dates_per_season=min_target_dates_per_season,
    )

    out_csv = os.path.join(out_dir, f"{prefix}_long_vs_flusight_h4_horizon_summary.csv")
    out_png_initial = os.path.join(out_dir, f"{prefix}_long_vs_flusight_h4.png")
    out_png_summary = os.path.join(out_dir, f"{prefix}_long_vs_flusight_h4_horizon_summary.png")
    summary.to_csv(out_csv, index=False)
    _plot_initial_comparison(merged, out_png_initial, log_y=log_y)
    _plot_horizon_summary(summary, out_png_summary, log_y=log_y)

    print(f"Saved horizon summary CSV: {out_csv}")
    print(f"Saved detailed comparison plot: {out_png_initial}")
    print(f"Saved horizon summary plot: {out_png_summary}")
    return out_csv, out_png_initial, out_png_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Figure 3 long-horizon analysis: save detailed comparison plot, horizon-summary plot, "
            "and horizon-summary CSV."
        )
    )
    parser.add_argument(
        "--out-dir",
        default="influpaint-paper/figures/generated",
        help="Output directory.",
    )
    parser.add_argument(
        "--prefix",
        default=f"{_MODEL_NUM}_figure3",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--min-target-dates-per-season",
        type=int,
        default=20,
        help="Fail if merged comparison has fewer target dates than this per season.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use log scale for the avg relative WIS y-axis.",
    )
    args = parser.parse_args()

    run_pipeline(
        out_dir=args.out_dir,
        prefix=args.prefix,
        min_target_dates_per_season=args.min_target_dates_per_season,
        log_y=args.log_y,
    )


if __name__ == "__main__":
    main()
