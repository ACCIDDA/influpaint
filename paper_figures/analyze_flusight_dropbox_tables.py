#!/usr/bin/env python3
"""
Analyze FluSight Dropbox leaderboard tables for InfluPaint performance.
Excludes FluSight-* models; applies the 70% submission threshold to 2023–2025.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns


def parse_table(file_path):
    """Parse FluSight table file (handles both tab and semicolon formats)."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Detect format: tab-delimited or semicolon-delimited
    first_line = lines[0] if lines else ""
    use_semicolon = ';' in first_line and '\t' not in first_line

    if use_semicolon:
        # Parse semicolon format (2022-2023)
        # Header is first line
        header_parts = [h.strip() for h in lines[0].split(';')]

        # Expected columns
        expected_cols = ['Model', 'Absolute WIS', 'Relative WIS', 'MAE', '50% Coverage (%)', '95% Coverage (%)', '% Forecasts Submitted']

        data = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # Skip if line starts with digit (row number from first format)
            if line[0].isdigit() and ' ' in line[:3]:
                continue

            # Split by multiple spaces (model name then values)
            parts = re.split(r'\s+', line)

            if len(parts) < 7:
                continue

            # Model name is first part, rest are numeric values
            model = parts[0]
            values = parts[1:7]

            data.append([model] + values)

        df = pd.DataFrame(data, columns=expected_cols)

    else:
        # Parse tab format (2023-2024, 2024-2025)
        # Find header line
        header_line = None
        for i, line in enumerate(lines):
            if 'Model' in line and 'Absolute WIS' in line:
                header_line = i
                break

        if header_line is None:
            raise ValueError(f"Could not find header in {file_path}")

        # Parse header
        header = re.split(r'\t+', lines[header_line].strip())

        # Parse data rows
        data = []
        for line in lines[header_line + 1:]:
            line = line.strip()
            if not line:
                continue

            # Split by tabs
            row = re.split(r'\t+', line)

            # Skip if first element is just a number (row number)
            if len(row) > 0 and row[0].isdigit():
                row = row[1:]

            # Skip if not enough columns
            if len(row) < len(header):
                continue

            data.append(row[:len(header)])

        # Create DataFrame
        df = pd.DataFrame(data, columns=header)

    # Convert numeric columns
    numeric_cols = ['Absolute WIS', 'Relative WIS', 'MAE', '50% Coverage (%)',
                    '95% Coverage (%)', '% Forecasts Submitted']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fix coverage and forecast percentage if they're in decimal form (0-1 range)
    # Convert to percentage (0-100 range)
    for col in ['50% Coverage (%)', '95% Coverage (%)', '% Forecasts Submitted']:
        if col in df.columns and df[col].max() <= 1.0:
            df[col] = df[col] * 100

    return df


def filter_models(df, use_70_threshold=True):
    """Apply FluSight filtering rules used for analysis and plotting."""
    filtered_df = df[
        ~df['Model'].str.lower().str.startswith('flusight', na=False)
    ].copy()

    if use_70_threshold:
        filtered_df = filtered_df[filtered_df['% Forecasts Submitted'] >= 70.0].copy()

    return filtered_df


def analyze_season(df, season_name, use_70_threshold=True):
    """Analyze InfluPaint performance for one season."""
    print(f"\n{'='*60}")
    print(f"Season: {season_name}")
    print('='*60)

    filtered_df = filter_models(df, use_70_threshold=use_70_threshold)

    if use_70_threshold:
        print(f"Total models after filtering: {len(filtered_df)}")
        print(f"  (Excluded FluSight-* models and models with <70% forecast submission)")
    else:
        print(f"Total models after filtering: {len(filtered_df)}")
        print(f"  (Excluded FluSight-* models; using FluSight inclusion criteria)")

    # Find InfluPaint
    influpaint_rows = filtered_df[filtered_df['Model'].str.contains('InfluPaint', case=False, na=False)]

    if influpaint_rows.empty:
        print("\nWARNING: InfluPaint not found in filtered data")
        return None

    influpaint = influpaint_rows.iloc[0]
    model_name = influpaint['Model']

    print(f"\nInfluPaint Model: {model_name}")

    # Calculate ranks
    results = {
        'Season': season_name,
        'Model': model_name,
        'Total Models': len(filtered_df)
    }

    # WIS rank (lower is better)
    wis_sorted = filtered_df.sort_values('Absolute WIS')
    wis_rank = wis_sorted[wis_sorted['Model'] == model_name].index[0]
    wis_rank_num = list(wis_sorted.index).index(wis_rank) + 1
    results['WIS Rank'] = f"{wis_rank_num}/{len(filtered_df)}"
    results['WIS Value'] = influpaint['Absolute WIS']

    # Relative WIS rank (lower is better)
    rel_wis_sorted = filtered_df.sort_values('Relative WIS')
    rel_wis_rank = rel_wis_sorted[rel_wis_sorted['Model'] == model_name].index[0]
    rel_wis_rank_num = list(rel_wis_sorted.index).index(rel_wis_rank) + 1
    results['Rel WIS Rank'] = f"{rel_wis_rank_num}/{len(filtered_df)}"
    results['Rel WIS Value'] = influpaint['Relative WIS']

    # MAE rank (lower is better)
    mae_sorted = filtered_df.sort_values('MAE')
    mae_rank = mae_sorted[mae_sorted['Model'] == model_name].index[0]
    mae_rank_num = list(mae_sorted.index).index(mae_rank) + 1
    results['MAE Rank'] = f"{mae_rank_num}/{len(filtered_df)}"
    results['MAE Value'] = influpaint['MAE']

    # Coverage values
    results['Coverage 50%'] = influpaint['50% Coverage (%)']
    results['Coverage 95%'] = influpaint['95% Coverage (%)']

    # Print results
    print(f"\n  Absolute WIS:    {results['WIS Value']:.2f}  (Rank: {results['WIS Rank']})")
    print(f"  Relative WIS:    {results['Rel WIS Value']:.2f}  (Rank: {results['Rel WIS Rank']})")
    print(f"  MAE:             {results['MAE Value']:.2f}  (Rank: {results['MAE Rank']})")
    print(f"  Coverage 50%:    {results['Coverage 50%']:.2f}%")
    print(f"  Coverage 95%:    {results['Coverage 95%']:.2f}%")

    return results


def plot_2024_2025_pairgrid(df, output_path):
    """Create a two-panel PairGrid plot for Relative WIS and Absolute WIS."""
    filtered_df = filter_models(df, use_70_threshold=True)

    if filtered_df.empty:
        raise ValueError("No models available after filtering for plotting.")

    ordered_df = filtered_df.sort_values('Absolute WIS', ascending=True)
    ordered_df = ordered_df.assign(
        abs_rank=range(1, len(ordered_df) + 1),
        rel_rank=ordered_df['Relative WIS'].rank(method='min').astype(int),
    )

    model_order = ordered_df['Model']

    height = max(0.2 * len(model_order), 4)
    g = sns.PairGrid(
        ordered_df,
        y_vars=["Model"],
        x_vars=["Relative WIS", "Absolute WIS"],
        height=height,
        aspect=1.2,
    )

    g.map(sns.stripplot, orient="h", order=model_order, size=5, color="#0b7285")

    for ax, metric, rank_col in zip(
        g.axes.flat,
        ["Relative WIS", "Absolute WIS"],
        ["rel_rank", "abs_rank"],
    ):
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

        # Annotate each point with rank and metric value
        x_margin = 0.02 * (ordered_df[metric].max() - ordered_df[metric].min() + 1e-9)
        for y_pos, (_, row) in enumerate(ordered_df.iterrows()):
            pct_forecasts = row.get('% Forecasts Submitted', float('nan'))
            pct_label = f"{pct_forecasts:.1f}%" if pd.notna(pct_forecasts) else ""
            label = f"{row[rank_col]} | {row[metric]:.2f} | {pct_label}"
            ax.text(
                row[metric] + x_margin,
                y_pos,
                label,
                va="center",
                ha="left",
                fontsize=8,
            )

    # Mirror model labels on the right panel for easier reading
    if g.axes.size >= 2:
        right_ax = g.axes.flat[1]
        right_ax.yaxis.tick_right()
        right_ax.yaxis.set_label_position("right")

    g.fig.suptitle("2024-2025 FluSight Models (>=70% forecasts)", y=1.02, fontsize=14)
    g.fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(output_path, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved PairGrid plot to: {output_path}")


def run_analysis(data_root, output_dir):
    """Regenerate operational leaderboard summaries from archived score tables."""
    tables = Path(data_root) / "analysis/FlusightScores"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_2022_2023 = tables / "2022-2023_table_flusight_mathispaperSI.txt"
    file_2023_2024 = tables / "2023-2024_table_flusight_dropbox.txt"
    file_2024_2025 = tables / "2024-2025_table_flusight_dropbox.txt"

    # Parse tables
    df_2022_2023 = parse_table(file_2022_2023)
    df_2023_2024 = parse_table(file_2023_2024)
    df_2024_2025 = parse_table(file_2024_2025)

    # Analyze each season
    results_list = []

    result_2022_2023 = analyze_season(df_2022_2023, "2022-2023", use_70_threshold=False)
    if result_2022_2023:
        results_list.append(result_2022_2023)

    result_2023_2024 = analyze_season(df_2023_2024, "2023-2024", use_70_threshold=True)
    if result_2023_2024:
        results_list.append(result_2023_2024)

    result_2024_2025 = analyze_season(df_2024_2025, "2024-2025", use_70_threshold=True)
    if result_2024_2025:
        results_list.append(result_2024_2025)

    # Summary table
    if results_list:
        print(f"\n{'='*60}")
        print("SUMMARY TABLE")
        print('='*60)

        summary_df = pd.DataFrame(results_list)

        # Select columns for display
        display_cols = ['Season', 'WIS Rank', 'WIS Value', 'Rel WIS Rank', 'Rel WIS Value',
                       'MAE Rank', 'MAE Value', 'Coverage 50%', 'Coverage 95%', 'Total Models']

        print("\n" + summary_df[display_cols].to_string(index=False))

        # Save to CSV
        output_file = output_dir / "flusight_dropbox_analysis.csv"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")

    # PairGrid plot for 2024-2025
    plot_output = output_dir / "2024-2025_wis_pairgrid.png"
    plot_2024_2025_pairgrid(df_2024_2025, plot_output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Zenodo reproduction folder containing analysis/FlusightScores.")
    parser.add_argument("--output-dir", type=Path,
                        help="Destination; defaults to DATA_ROOT/regenerated_paper_figures.")
    args = parser.parse_args(argv)
    plt.switch_backend("Agg")
    run_analysis(args.data_root, args.output_dir or args.data_root / "regenerated_paper_figures")


if __name__ == "__main__":
    main()
