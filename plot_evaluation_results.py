#!/usr/bin/env python3
"""
InfluPaint vs FluSight evaluation plotting from scoringutils CSV.
Uses benchmark_plotting utilities for visualization.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union
import warnings
warnings.filterwarnings('ignore')

# Import the existing SeasonAxis and benchmark plotting
from influpaint.utils.season_axis import SeasonAxis
from benchmark_plotting import plot_components, plot_timeseries, plot_wis_heatmap, plot_cumulative_timeseries, plot_multi_location_stacked, print_ladderboard, compute_missing_data, get_rankings, plot_interactive_model_selection


# %% Configuration
CSV_PATH = "results/scoringutils_scores.csv" 
SAVE_DIR = "results/simple_plots"
GROUP_COLORS = {'influpaint': 'green', 'flusight': 'blue'}
ALLOW_MISSING_DATES_PER_MODEL = 5  # Same threshold as evaluation_pipeline.py
LEADERBOARD_DIR = "results/leaderboards"
LEADERBOARD_CSV = os.path.join(LEADERBOARD_DIR, "leaderboard_full.csv")
DEFAULT_SAVE_DIR = SAVE_DIR
DEFAULT_LEADERBOARD_DIR = LEADERBOARD_DIR

def add_inclusion_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns indicating which plots each model should be included in based on per-season performance."""
    print(f"🔍 ANALYZING MODEL INCLUSION:")
    print(f"   Total models: {df['model'].nunique()}")
    
    # Determine date column for filtering
    date_col = 'forecast_date' if 'forecast_date' in df.columns else 'reference_date' if 'reference_date' in df.columns else 'target_end_date'
    print(f"   Using '{date_col}' for filtering")
    
    # Debug: Show data structure
    if 'season' in df.columns:
        unique_seasons = sorted(df['season'].unique()) 
        print(f"   🗓️ Seasons: {unique_seasons}")
        
        season_info = {}
        for season in unique_seasons:
            season_data = df[df['season'] == season]
            if date_col in season_data.columns:
                season_forecast_dates = season_data[date_col].nunique()
                season_info[season] = season_forecast_dates
                print(f"   🗓️ {season}: {season_forecast_dates} forecast dates")
        
        # Create inclusion columns
        df_with_flags = df.copy()
        
        # Per-season analysis
        for season in unique_seasons:
            max_dates_in_season = season_info[season]
            min_required_in_season = max_dates_in_season - ALLOW_MISSING_DATES_PER_MODEL
            
            season_data = df[df['season'] == season]
            model_counts_in_season = season_data.groupby('model')[date_col].nunique()
            
            # Models that meet criteria for this season
            valid_models_in_season = model_counts_in_season[model_counts_in_season >= min_required_in_season].index
            
            # Add inclusion column for this season
            include_col = f'include_{season.replace("-", "_")}'
            df_with_flags[include_col] = df_with_flags['model'].isin(valid_models_in_season)
            
            print(f"   🗓️ {season}: {len(valid_models_in_season)}/{df['model'].nunique()} models meet criteria (≥{min_required_in_season}/{max_dates_in_season} dates)")
        
        # Combined inclusion (must meet criteria in ALL seasons)
        if len(unique_seasons) > 1:
            include_cols = [f'include_{season.replace("-", "_")}' for season in unique_seasons]
            df_with_flags['include_combined'] = df_with_flags[include_cols].all(axis=1)
            combined_models = df_with_flags[df_with_flags['include_combined']]['model'].unique()
            print(f"   🔄 Combined: {len(combined_models)}/{df['model'].nunique()} models meet criteria in ALL seasons")
        else:
            df_with_flags['include_combined'] = df_with_flags[f'include_{unique_seasons[0].replace("-", "_")}']
            combined_models = df_with_flags[df_with_flags['include_combined']]['model'].unique()
        
        # Show model breakdown
        print(f"   📊 MODEL BREAKDOWN:")
        all_models = sorted(df['model'].unique())
        
        for model in all_models:
            model_info = []
            for season in unique_seasons:
                season_data = df[df['season'] == season]
                model_count = season_data[season_data['model'] == model][date_col].nunique() if model in season_data['model'].values else 0
                max_dates = season_info[season]
                missing = max_dates - model_count
                status = "✅" if missing <= ALLOW_MISSING_DATES_PER_MODEL else "❌"
                model_info.append(f"{season}:{status}{model_count:02d}/{max_dates:02d}")
            
            combined_status = "✅" if model in combined_models else "❌"
            print(f"{' | '.join(model_info)} | Combined:{combined_status} {model}")
        
        return df_with_flags
    
    else:
        # No seasons, just use overall filtering
        print("   No season column found, using overall filtering")
        model_date_counts = df.groupby('model')[date_col].nunique()
        max_dates = model_date_counts.max()
        min_required_dates = max_dates - ALLOW_MISSING_DATES_PER_MODEL
        
        successful_models = model_date_counts[model_date_counts >= min_required_dates].index
        df_with_flags = df.copy()
        df_with_flags['include_combined'] = df_with_flags['model'].isin(successful_models)
        
        return df_with_flags


def get_missing_data_for_plot(original_df: pd.DataFrame, models_in_plot: List[str], location_filter: str, season_filter: str = None) -> Dict[str, Dict[str, Union[str, bool]]]:
    """Get missing data info using generic compute_missing_data function."""
    
    # Get expected horizons
    expected_horizons = [0, 1, 2, 3]
    
    # Get expected dates from jobs file or data
    jobs_file = "paper_runs_2025-07-22/inpaint_jobs_paper-2025-07-22.txt"
    try:
        jobs_df = pd.read_csv(jobs_file)
        if season_filter and season_filter != "Combined":
            season_jobs = jobs_df[jobs_df["season"] == season_filter]
            expected_dates = season_jobs["date"].unique().tolist()
        else:
            seasons_in_data = original_df['season'].unique()
            expected_dates = []
            for season in seasons_in_data:
                season_jobs = jobs_df[jobs_df["season"] == season]
                expected_dates.extend(season_jobs["date"].unique())
    except Exception as e:
        print(f"Warning: Could not read jobs file ({e}), using dates from data")
        if season_filter and season_filter != "Combined":
            season_data = original_df[original_df['season'] == season_filter]
            expected_dates = season_data['reference_date'].unique().tolist()
        else:
            expected_dates = original_df['reference_date'].unique().tolist()
    
    # Get expected locations
    if location_filter == "US":
        expected_locations = ["US"]
    elif location_filter == "sum_all_states":
        season_axis = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
        expected_locations = season_axis.locations
    else:
        expected_locations = original_df['location'].unique().tolist()
    
    return compute_missing_data(original_df, models_in_plot, expected_locations, expected_horizons, expected_dates)

def plot_dual_metric_bars(agg_df: pd.DataFrame, left_metric: str, right_metric: str, title: str, save_path: str, bar_color: str, missing_info: Dict[str, Dict[str, Union[str, bool]]] = None) -> None:
    """Plot two aligned horizontal bar charts (left/right metrics) ordered by left_metric, labeling missing data if provided."""
    df = agg_df.copy()
    df = df.replace({np.inf: np.nan})
    df = df.dropna(subset=[left_metric, right_metric])
    if df.empty:
        print(f"No data to plot for {title}")
        return

    df = df.sort_values(left_metric, ascending=True).reset_index(drop=True)
    df['rank'] = df[left_metric].rank(method='min', ascending=True).astype(int)

    # Build labels with rank prefix and optional missing-info text
    labels = []
    colors = []
    for _, row in df.iterrows():
        model_name = row['model']
        rank_prefix = f"#{row['rank']} "
        if missing_info and model_name in missing_info:
            info = missing_info[model_name]
            miss_text = info.get("text", "") if isinstance(info, dict) else str(info)
            label = f"{rank_prefix}{model_name}"
            if miss_text:
                label = f"{label}\n{miss_text}"
            colors.append('red' if isinstance(info, dict) and info.get("critical", False) else bar_color)
        else:
            label = f"{rank_prefix}{model_name}"
            colors.append(bar_color)
        labels.append(label)

    fig_height = max(4, 0.45 * len(df))
    fig_width = max(18, 0.8 * fig_height + 12)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)
    y_pos = np.arange(len(df))

    # Left: relative metric (ranking order)
    axes[0].barh(y_pos, df[left_metric], color=bar_color, alpha=0.85, label=left_metric)
    axes[0].set_xlabel(left_metric.upper().replace('_', ' '))
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    for label, color in zip(axes[0].get_yticklabels(), colors):
        label.set_color(color)
    for y, val in zip(y_pos, df[left_metric]):
        axes[0].text(val, y, f"{val:.2f}", va='center', ha='left', fontsize=9, color='black')
    axes[0].grid(True, axis='x', alpha=0.3, linewidth=0.5)
    axes[0].legend(loc='lower right', framealpha=0.8)

    # Right: secondary metric, same order
    axes[1].barh(y_pos, df[right_metric], color=bar_color, alpha=0.55, label=right_metric)
    axes[1].set_xlabel(right_metric.upper().replace('_', ' '))
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])
    for y, val in zip(y_pos, df[right_metric]):
        axes[1].text(val, y, f"{val:.2f}", va='center', ha='left', fontsize=9, color='black')
    axes[1].grid(True, axis='x', alpha=0.3, linewidth=0.5)
    axes[1].legend(loc='lower right', framealpha=0.8)

    fig.suptitle(title, y=0.98, fontsize=12, fontweight='medium')
    plt.tight_layout()
    plt.subplots_adjust(left=0.6, wspace=0.32)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# %% Main Script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot evaluation results.")
    parser.add_argument("--csv-path", default=CSV_PATH, help="Path to scoringutils CSV.")
    parser.add_argument("--save-dir", default=SAVE_DIR, help="Directory to save plots.")
    parser.add_argument("--group-filter", default=None, help="Restrict plots to a single group label (e.g., 'flusight').")
    parser.add_argument("--annotate-ranks", action="store_true", help="Add value and rank labels for WIS and relative WIS bar charts.")
    args = parser.parse_args()

    CSV_PATH = args.csv_path
    SAVE_DIR = args.save_dir
    LEADERBOARD_DIR = DEFAULT_LEADERBOARD_DIR if args.save_dir == DEFAULT_SAVE_DIR else os.path.join(SAVE_DIR, "leaderboards")
    LEADERBOARD_CSV = os.path.join(LEADERBOARD_DIR, "leaderboard_full.csv")

    # Load data with proper location column handling
    df_raw = pd.read_csv(CSV_PATH, dtype={'location': str})
    df_raw['target_end_date'] = pd.to_datetime(df_raw['target_end_date']).dt.date
    df_raw['location'] = df_raw['location'].astype(str).str.strip()
    
    # Filter out problematic models from analysis
    # i808 models have issues, UGuelph-CompositeCurve makes plot scale badly
    df_raw = df_raw[~df_raw['model'].str.startswith('i808')]
    df_raw = df_raw[df_raw['model'] != 'UGuelph-CompositeCurve']
    df_raw = df_raw[df_raw['model'] != 'CADPH-FluCAT_Ensemble']
    
    # Calculate relative WIS against baseline for all data
    baseline_model = 'FluSight-baseline'
    if baseline_model in df_raw['model'].unique():
        baseline_data = df_raw[df_raw['model'] == baseline_model].set_index(['location', 'target_end_date', 'horizon'])['wis']
        df_raw['relative_wis'] = df_raw.apply(
            lambda row: row['wis'] / baseline_data.get((row['location'], row['target_end_date'], row['horizon']), np.nan) 
            if baseline_data.get((row['location'], row['target_end_date'], row['horizon']), 0) > 0 else np.nan, 
            axis=1
        )
    else:
        df_raw['relative_wis'] = np.nan
    
    # Optional group filtering
    if args.group_filter:
        df_raw = df_raw[df_raw['group'] == args.group_filter].copy()
        if df_raw.empty:
            raise SystemExit(f"No rows found for group '{args.group_filter}'")
    
    # Add inclusion columns based on per-season performance
    df_with_flags = add_inclusion_columns(df_raw)
    
    # Create output directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LEADERBOARD_DIR, exist_ok=True)
    
    # Plot order: Combined, then individual seasons
    available_seasons = [s for s in df_with_flags['season'].unique() if s != "Combined"]
    seasons_to_plot = ["Combined"] + sorted(available_seasons)
    
    # Collect full leaderboards for all seasons/metrics
    leaderboard_rows = []

    for season in seasons_to_plot:
        print(f"\n{'='*50}")
        print(f"PLOTTING: {season.upper()}")
        print('='*50)
        
        # Create season-specific folder
        season_dir = os.path.join(SAVE_DIR, season)
        os.makedirs(season_dir, exist_ok=True)
        
        # Get season data with appropriate filtering
        if season == "Combined":
            # For combined, only include models that meet criteria in ALL seasons
            season_df = df_with_flags[df_with_flags['include_combined']].copy()
            season_raw_df = df_with_flags.copy()  # Keep all for missing data counting
        else:
            # For individual seasons, include models that meet criteria in THIS season
            include_col = f'include_{season.replace("-", "_")}'
            season_data = df_with_flags[df_with_flags['season'] == season]
            season_df = season_data[season_data[include_col]].copy()
            season_raw_df = df_with_flags[df_with_flags['season'] == season].copy()
        
        if season_df.empty:
            print(f"No data for {season}")
            continue
        
        print(f"Models in {season}: {len(season_df['model'].unique())} (after per-season filtering)")

        if args.group_filter and args.annotate_ranks:
            filtered_group_df = season_df[season_df['group'] == args.group_filter]
            if not filtered_group_df.empty:
                agg_group = filtered_group_df.groupby('model', as_index=False).agg({
                    'relative_wis': 'mean',
                    'wis': 'sum'
                })
                group_color = GROUP_COLORS.get(args.group_filter, 'blue')
                group_missing_info = get_missing_data_for_plot(
                    season_raw_df,
                    filtered_group_df['model'].unique().tolist(),
                    "sum_all_states",
                    season
                )

                plot_dual_metric_bars(
                    agg_group,
                    left_metric='relative_wis',
                    right_metric='wis',
                    title=f"{season}: {args.group_filter.title()} Relative WIS (left) and WIS (right)",
                    save_path=os.path.join(season_dir, f"{args.group_filter}_relative_wis_wis_dual.png"),
                    bar_color=group_color,
                    missing_info=group_missing_info
                )
        
        # INFLUPAINT LEADERBOARDS
        influpaint_df = season_df[season_df['group'] == 'influpaint'].copy()
        if not influpaint_df.empty:
            # Total WIS across all locations (print and get rankings)
            wis_rankings = get_rankings('wis', 'sum', influpaint_df, print_top_n=10)
            for rank_idx, (model, score) in enumerate(wis_rankings.items(), start=1):
                leaderboard_rows.append({
                    'season': season,
                    'metric': 'wis',
                    'aggregation': 'sum',
                    'model': model,
                    'score': float(score),
                    'rank': rank_idx
                })
            
            # Relative WIS (mean across all locations)
            rel_rankings = get_rankings('relative_wis', 'mean', influpaint_df, print_top_n=10)
            for rank_idx, (model, score) in enumerate(rel_rankings.items(), start=1):
                leaderboard_rows.append({
                    'season': season,
                    'metric': 'relative_wis',
                    'aggregation': 'mean',
                    'model': model,
                    'score': float(score),
                    'rank': rank_idx
                })
        
        # 1. WIS Heatmaps
        season_axis = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
        fig, ax = plot_wis_heatmap(
            df=season_df,
            location_filter="US", 
            title=f"{season}: Absolute WIS Heatmap (US National)",
            relative=False, 
            original_df=season_raw_df, 
            missing_info_fn=lambda df, models, loc_filter: get_missing_data_for_plot(df, models, loc_filter, season), 
            group_colors=GROUP_COLORS
        )
        fig.savefig(os.path.join(season_dir, "absolute_wis_heatmap_US.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plot_wis_heatmap(
            df=season_df,
            location_filter="US",
            title=f"{season}: Relative WIS Heatmap (US National)", 
            relative=True,
            original_df=season_raw_df,
            missing_info_fn=lambda df, models, loc_filter: get_missing_data_for_plot(df, models, loc_filter, season),
            group_colors=GROUP_COLORS
        )
        fig.savefig(os.path.join(season_dir, "relative_wis_heatmap_US.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plot_wis_heatmap(
            df=season_df,
            location_filter="sum_all_states",
            title=f"{season}: Absolute WIS Heatmap (Sum Over Locations)",
            relative=False,
            original_df=season_raw_df,
            missing_info_fn=lambda df, models, loc_filter: get_missing_data_for_plot(df, models, loc_filter, season),
            group_colors=GROUP_COLORS,
            valid_locations=season_axis.locations
        )
        fig.savefig(os.path.join(season_dir, "absolute_wis_heatmap_sum_all_states.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plot_wis_heatmap(
            df=season_df,
            location_filter="sum_all_states",
            title=f"{season}: Relative WIS Heatmap (Sum Over Locations)",
            relative=True,
            original_df=season_raw_df,
            missing_info_fn=lambda df, models, loc_filter: get_missing_data_for_plot(df, models, loc_filter, season),
            group_colors=GROUP_COLORS,
            valid_locations=season_axis.locations
        )
        fig.savefig(os.path.join(season_dir, "relative_wis_heatmap_sum_all_states.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        # 2. Full Performance Plot
        
        # US National full performance plot
        us_data = season_df[season_df['location'] == 'US'].copy()
        if not us_data.empty:
            
            us_missing_info = get_missing_data_for_plot(season_raw_df, us_data['model'].unique().tolist(), "US", season)
            plot_components(
                df=us_data,
                group_by=['model', 'group'],
                value_cols=['wis', 'dispersion', 'overprediction', 'underprediction', 
                           'interval_coverage_50', 'interval_coverage_90', 'relative_wis'],
                agg_func={'wis': 'sum', 'dispersion': 'sum', 'overprediction': 'sum', 'underprediction': 'sum',
                         'interval_coverage_50': 'mean', 'interval_coverage_90': 'mean', 'relative_wis': 'mean'},
                sort_by='wis',
                title=f"{season}: Full Performance (US National)",
                save_path=os.path.join(season_dir, "full_plot_US.png"),
                missing_info=us_missing_info,
                group_colors=GROUP_COLORS,
                reference_lines={
                    'interval_coverage_50': {'value': 0.5, 'label': 'Target 50%', 'color': 'red'},
                    'interval_coverage_90': {'value': 0.9, 'label': 'Target 90%', 'color': 'red'},
                    'relative_wis': {'value': 1.0, 'label': 'Baseline', 'color': 'black', 'linestyle': ':'}
                }
            )
        
        # Sum over locations full performance plot
        season_axis = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
        valid_locs = season_axis.locations
        allsum_data = season_df[season_df['location'].isin(valid_locs)].copy()
        if not allsum_data.empty:
            allsum_missing_info = get_missing_data_for_plot(season_raw_df, allsum_data['model'].unique().tolist(), "sum_all_states", season)
            
            plot_components(
                df=allsum_data,
                group_by=['model', 'group'],
                value_cols=['wis', 'dispersion', 'overprediction', 'underprediction', 
                           'interval_coverage_50', 'interval_coverage_90', 'relative_wis'],
                agg_func={'wis': 'sum', 'dispersion': 'sum', 'overprediction': 'sum', 'underprediction': 'sum',
                         'interval_coverage_50': 'mean', 'interval_coverage_90': 'mean', 'relative_wis': 'mean'},
                sort_by='wis',
                title=f"{season}: Full Performance (Sum Over Locations)",
                save_path=os.path.join(season_dir, "full_plot_sum_all_states.png"),
                missing_info=allsum_missing_info,
                group_colors=GROUP_COLORS,
                reference_lines={
                    'interval_coverage_50': {'value': 0.5, 'label': 'Target 50%', 'color': 'red'},
                    'interval_coverage_90': {'value': 0.9, 'label': 'Target 90%', 'color': 'red'},
                    'relative_wis': {'value': 1.0, 'label': 'Baseline', 'color': 'black', 'linestyle': ':'}
                }
            )
        
        # 5. Time Series
        # US National Time Series (Absolute WIS)
        us_ts_data = season_df[season_df['location'] == 'US'].copy()
        if not us_ts_data.empty:
            plot_timeseries(
                df=us_ts_data,
                x_col='target_end_date',
                y_col='wis',
                group_col='model',
                facet_col='horizon',
                filter_top_n=3,
                title=f"{season}: Absolute WIS Time Series (US National - Top 3 per Group)",
                save_path=os.path.join(season_dir, "absolute_timeseries_US.png"),
                relative=False
            )
            
            # Absolute WIS Cumulative Time Series
            # Filter to top 10 models per group for better readability
            if 'group' in us_ts_data.columns:
                from benchmark_plotting import get_top_models_per_group
                top_models = get_top_models_per_group(us_ts_data, 'wis', top_n=10, relative=False)
                us_ts_filtered = us_ts_data[us_ts_data['model'].isin(top_models)]
            else:
                # Fallback: top 10 overall
                model_avg = us_ts_data.groupby('model')['wis'].mean().nsmallest(10)
                us_ts_filtered = us_ts_data[us_ts_data['model'].isin(model_avg.index)]
            
            fig, ax = plot_cumulative_timeseries(
                plot_data=us_ts_filtered,
                title=f"{season}: Cumulative WIS (US National - Top 10 per Group)",
                relative=False
            )
            fig.savefig(os.path.join(season_dir, "cumulative_wis_US.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            # Relative WIS Time Series
            plot_timeseries(
                df=us_ts_data,
                x_col='target_end_date',
                y_col='relative_wis',
                group_col='model',
                facet_col='horizon',
                filter_top_n=3,
                title=f"{season}: Relative WIS Time Series (US National - Top 3 per Group)",
                save_path=os.path.join(season_dir, "relative_timeseries_US.png"),
                relative=True
            )
            
            # Relative WIS Cumulative Time Series
            # Filter to top 10 models per group for better readability
            if 'group' in us_ts_data.columns:
                top_models_rel = get_top_models_per_group(us_ts_data, 'relative_wis', top_n=10, relative=True)
                us_ts_filtered_rel = us_ts_data[us_ts_data['model'].isin(top_models_rel)]
            else:
                # Fallback: top 10 overall (closest to 1.0 for relative)
                model_avg_rel = us_ts_data.groupby('model')['relative_wis'].mean()
                closest_to_one = model_avg_rel.iloc[(model_avg_rel - 1.0).abs().argsort()[:10]]
                us_ts_filtered_rel = us_ts_data[us_ts_data['model'].isin(closest_to_one.index)]
            
            fig, ax = plot_cumulative_timeseries(
                plot_data=us_ts_filtered_rel,
                title=f"{season}: Relative Cumulative WIS (US National - Top 10 per Group)",
                relative=True
            )
            fig.savefig(os.path.join(season_dir, "relative_cumulative_wis_US.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)
        
        # 6. States Stacked WIS Components
        
        # Create states sum data
        states_data = season_df[season_df['location'].isin(season_axis.locations)]
        states_sum = states_data.groupby(['model', 'group'], as_index=False).agg({
            'underprediction': 'sum', 'overprediction': 'sum', 'dispersion': 'sum'
        })
        states_sum['location'] = 'States_Sum'
        
        # Combine with original data
        states_plot_data = pd.concat([season_df, states_sum], ignore_index=True)
        
        
        all_locations = ['US', 'States_Sum'] + season_axis.locations
        fig, axes = plot_multi_location_stacked(
            df=states_plot_data,
            locations=all_locations,
            reference_location='US',
            value_cols=['underprediction', 'overprediction', 'dispersion'],
            component_colors={'underprediction': 'red', 'overprediction': 'green', 'dispersion': 'blue'},
            title=f"{season}: WIS Components by State (Sorted by US Total WIS)",
            group_colors=GROUP_COLORS,
            location_name_mapper=season_axis.get_location_name
        )
        fig.savefig(os.path.join(season_dir, "wis_components_states_stacked.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        # 7. Scatter Plots using plot_components
        
        # WIS vs Relative WIS scatter - US National
        us_data = season_df[season_df['location'] == 'US']
        if not us_data.empty:
            us_missing_info = get_missing_data_for_plot(season_raw_df, us_data['model'].unique().tolist(), "US", season)
            plot_components(
                df=us_data,
                group_by=['model', 'group'],
                value_cols=['wis', 'relative_wis'], 
                agg_func={'wis': 'mean', 'relative_wis': 'mean'},
                title=f"{season}: WIS vs Relative WIS (US National)",
                save_path=os.path.join(season_dir, "wis_scatter_US.png"),
                missing_info=us_missing_info,
                group_colors=GROUP_COLORS,
                stacked=False,
                reference_lines={
                    'relative_wis': {'value': 1.0, 'label': 'Baseline', 'color': 'black', 'linestyle': ':'}
                }
            )
        
        # WIS vs Relative WIS scatter - All Locations
        all_locations_missing_info = get_missing_data_for_plot(season_raw_df, season_df['model'].unique().tolist(), "sum_all_states", season)
        plot_components(
            df=season_df,
            group_by=['model', 'group'],
            value_cols=['wis', 'relative_wis'], 
            agg_func={'wis': 'sum', 'relative_wis': 'mean'},
            title=f"{season}: WIS vs Relative WIS (All Locations)",
            save_path=os.path.join(season_dir, "wis_scatter_all_locations.png"),
            missing_info=all_locations_missing_info,
            group_colors=GROUP_COLORS,
            stacked=False,
            reference_lines={
                'relative_wis': {'value': 1.0, 'label': 'Baseline', 'color': 'black', 'linestyle': ':'}
            }
        )
        
        # Coverage scatter - US National
        us_data = season_df[season_df['location'] == 'US']
        if not us_data.empty:
            us_missing_info = get_missing_data_for_plot(season_raw_df, us_data['model'].unique().tolist(), "US", season)
            plot_components(
                df=us_data,
                group_by=['model', 'group'],
                value_cols=['interval_coverage_50', 'interval_coverage_90'],
                agg_func={'interval_coverage_50': 'mean', 'interval_coverage_90': 'mean'},
                title=f"{season}: Coverage Comparison (US National)",
                save_path=os.path.join(season_dir, "coverage_scatter_US.png"),
                missing_info=us_missing_info,
                group_colors=GROUP_COLORS,
                stacked=False,
                reference_lines={
                    'interval_coverage_50': {'value': 0.5, 'label': 'Target 50%', 'color': 'red'},
                    'interval_coverage_90': {'value': 0.9, 'label': 'Target 90%', 'color': 'red'}
                }
            )
        
        # Coverage scatter - All Locations
        all_locations_missing_info = get_missing_data_for_plot(season_raw_df, season_df['model'].unique().tolist(), "sum_all_states", season)
        plot_components(
            df=season_df,
            group_by=['model', 'group'],
            value_cols=['interval_coverage_50', 'interval_coverage_90'],
            agg_func={'interval_coverage_50': 'mean', 'interval_coverage_90': 'mean'},
            title=f"{season}: Coverage Comparison (All Locations)",
            save_path=os.path.join(season_dir, "coverage_scatter_all_locations.png"),
            missing_info=all_locations_missing_info,
            group_colors=GROUP_COLORS,
            stacked=False,
            reference_lines={
                'interval_coverage_50': {'value': 0.5, 'label': 'Target 50%', 'color': 'red'},
                'interval_coverage_90': {'value': 0.9, 'label': 'Target 90%', 'color': 'red'}
            }
        )
        
        print(f"Completed plots for {season}")
    
    # Save full leaderboard CSV
    if leaderboard_rows:
        lb_df = pd.DataFrame(leaderboard_rows)
        # Optional: stable ordering
        sort_cols = ['season', 'metric', 'aggregation', 'rank']
        lb_df = lb_df.sort_values(sort_cols)
        lb_df.to_csv(LEADERBOARD_CSV, index=False)
        print(f"\nSaved full leaderboard to: {LEADERBOARD_CSV}")

        # Create interactive model selection plot for InfluPaint models only
        influpaint_leaderboard = lb_df[lb_df['model'].isin(
            df_with_flags[df_with_flags['group'] == 'influpaint']['model'].unique()
        )]

        # Filter for WIS metric with sum aggregation for the interactive plot
        wis_data = influpaint_leaderboard[
            (influpaint_leaderboard['metric'] == 'wis') &
            (influpaint_leaderboard['aggregation'] == 'sum')
        ][['season', 'model', 'score']].rename(columns={'score': 'wis'})

        # Get relative WIS data
        rel_wis_data = influpaint_leaderboard[
            (influpaint_leaderboard['metric'] == 'relative_wis') &
            (influpaint_leaderboard['aggregation'] == 'mean')
        ][['season', 'model', 'score']].rename(columns={'score': 'relative_wis'})

        # Merge WIS and relative WIS data
        interactive_data = wis_data.merge(rel_wis_data, on=['season', 'model'], how='inner')

        interactive_save_path = os.path.join(SAVE_DIR, "interactive_model_selection.html")
        plot_interactive_model_selection(
            leaderboard_df=interactive_data,
            title="InfluPaint Model Selection",
            save_path=interactive_save_path
        )

    run_paper_analysis = (not args.group_filter) or (args.group_filter == 'influpaint')
    if run_paper_analysis:
        # PAPER ANALYSIS: Best model and ensemble comparison
        print(f"\n{'='*60}")
        print("PAPER ANALYSIS: Model Performance Summary")
        print('='*60)

        # Define models for paper analysis
        best_model_full = f"i868::m_U500cRx1224::ds_30S70M::tr_Sqrt::ri_No::inpaint_CoPaint::celebahq_noTTJ5"
        ensemble_model = "FluSight-ensemble"
        submitted_model = "UNC_IDD-InfluPaint"

        # Verify models exist
        if best_model_full not in df_with_flags['model'].unique():
            print(f"ERROR: Best model '{best_model_full}' not found in data")
        if ensemble_model not in df_with_flags['model'].unique():
            print(f"ERROR: Ensemble model '{ensemble_model}' not found in data")
        if submitted_model not in df_with_flags['model'].unique():
            print(f"ERROR: Submitted model '{submitted_model}' not found in data")

        # Process each season
        paper_results = []
        for season in seasons_to_plot:
            if season == "Combined":
                season_data = df_with_flags[df_with_flags['include_combined']].copy()
            else:
                include_col = f'include_{season.replace("-", "_")}'
                season_data = df_with_flags[df_with_flags['season'] == season]
                season_data = season_data[season_data[include_col]].copy()

            if season_data.empty:
                continue

            # Compute for all locations and all horizons
            all_data = season_data.copy()
            flusight_all = all_data[all_data['group'] == 'flusight'].copy()

            for model_name, model_id in [(best_model_full, "i868_celebahq_noTTJ5"),
                                          (ensemble_model, "FluSight-ensemble"),
                                          (submitted_model, "UNC_IDD-InfluPaint")]:
                model_data = all_data[all_data['model'] == model_name]

                if model_data.empty:
                    print(f"WARNING: {model_id} not found in {season} (may not meet inclusion criteria)")
                    continue

                # Compute metrics (all locations, all horizons)
                total_wis = model_data['wis'].sum()
                coverage_50 = model_data['interval_coverage_50'].mean()
                coverage_95 = model_data['interval_coverage_90'].mean()

                # Rank against FluSight models for Total WIS
                flusight_wis = flusight_all.groupby('model')['wis'].sum().sort_values()
                wis_rank = (flusight_wis < total_wis).sum() + 1
                wis_total_models = len(flusight_wis)

                # Rank against FluSight models for Coverage 50%
                flusight_cov50 = flusight_all.groupby('model')['interval_coverage_50'].mean()
                flusight_cov50_diff = (flusight_cov50 - 0.5).abs()
                model_cov50_diff = abs(coverage_50 - 0.5)
                cov50_rank = (flusight_cov50_diff < model_cov50_diff).sum() + 1

                # Rank against FluSight models for Coverage 95%
                flusight_cov95 = flusight_all.groupby('model')['interval_coverage_90'].mean()
                flusight_cov95_diff = (flusight_cov95 - 0.9).abs()
                model_cov95_diff = abs(coverage_95 - 0.9)
                cov95_rank = (flusight_cov95_diff < model_cov95_diff).sum() + 1

                paper_results.append({
                    'Model': model_id,
                    'Season': season,
                    'Total WIS': f"{total_wis:.2f}",
                    'WIS Rank': f"{wis_rank}/{wis_total_models}",
                    'Coverage 50%': f"{coverage_50:.3f}",
                    'Cov50 Rank': f"{cov50_rank}/{wis_total_models}",
                    'Coverage 95%': f"{coverage_95:.3f}",
                    'Cov95 Rank': f"{cov95_rank}/{wis_total_models}"
                })

        # Display results table
        if paper_results:
            results_df = pd.DataFrame(paper_results)
            print("\n" + results_df.to_string(index=False))

            # Save to CSV
            paper_csv_path = os.path.join(SAVE_DIR, "paper_model_analysis.csv")
            results_df.to_csv(paper_csv_path, index=False)
            print(f"\nSaved paper analysis to: {paper_csv_path}")

    print(f"\n{'='*50}")
    print(f"ALL PLOTS SAVED TO: {SAVE_DIR}")
    print('='*50)
