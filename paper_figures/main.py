"""
Main script to generate all paper figures.

This orchestrates the generation of all figures for the Influpaint paper:
1) Unconditional generation figures
2) CSV forecast quantile fans
3) NPY full-horizon forecasts
4) Mask experiments
5) Peak analysis figures
"""

import os
import matplotlib.pyplot as plt

from influpaint.utils import SeasonAxis

# Import configuration
from .config import (
    FIG_DIR, BEST_MODEL_ID, BEST_CONFIG, UNCOND_SAMPLES_PATH,
    INPAINTING_BASE, PLOT_MEDIAN, _MODEL_NUM, MAX_LOW_LOCATIONS
)

# Import helper functions
from .helpers import load_unconditional_samples

# Import data utilities
from .data_utils import compute_historical_peak_threshold, filter_trajectories_by_peak

# Import figure generation modules
from . import unconditional_figures as uncond_figs
from . import peak_analysis
from . import correlation_analysis
from . import csv_forecasts
from . import npy_forecasts
from . import mask_experiments


def generate_unconditional_figures(season_axis, uncond_samples):
    """Generate all unconditional figure types.

    Args:
        season_axis: SeasonAxis object
        uncond_samples: Unconditional samples array
    """
    print("Generating unconditional figures...")

    # US grid of several samples
    uncond_figs.generate_unconditional_us_grid(uncond_samples, season_axis, FIG_DIR, _MODEL_NUM)

    # Trajectories + mean heatmap
    uncond_figs.generate_unconditional_trajs_and_heatmap(uncond_samples, season_axis, FIG_DIR)

    # States quantiles + trajectories
    fig = uncond_figs.plot_unconditional_states_quantiles_and_trajs(
        inv_samples=uncond_samples,
        season_axis=season_axis,
        states=['NC', 'CA', 'NY', 'TX', 'FL'],
        n_sample_trajs=10,
        plot_median=PLOT_MEDIAN,
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_uncond_states_quantiles_trajs.png"),
    )
    plt.close(fig)

    # 3D ridge + heatmap illustration
    fig3d, _ = uncond_figs.fig_unconditional_3d_heat_ridges(
        inv_samples=uncond_samples,
        season_axis=season_axis,
        states=None,
        stat='median',
        location_stride=10,
        elev=20,
        azim=-110,
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_uncond_3d_heat_ridges.png"),
    )
    plt.close(fig3d)

    # Mean heatmap only
    uncond_figs.generate_mean_heatmap(uncond_samples, season_axis, FIG_DIR, _MODEL_NUM)

    # Plotly interactive 3D version (HTML)
    try:
        html_path = os.path.join(FIG_DIR, f"{_MODEL_NUM}_uncond_3d_heat_ridges_plotly.html")
        _ = uncond_figs.fig_unconditional_3d_heat_ridges_plotly(
            inv_samples=uncond_samples,
            season_axis=season_axis,
            states=None,
            stat='median',
            surface_opacity=0.6,
            ridge_lift=0.0,
            location_stride=3,
            camera_eye=(1.0, -1.6, 0.6),
            save_path_html=html_path,
        )
    except RuntimeError as _e:
        print("Plotly not available; skipping interactive 3D figure.")

    # Unconditional with historical overlay
    fig_hist = uncond_figs.plot_unconditional_states_with_history(
        inv_samples=uncond_samples,
        season_axis=season_axis,
        states=['NC', 'CA', 'NY', 'TX', 'FL', 'MT'],
        n_sample_trajs=8,
        plot_median=False,
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_uncond_states_with_history.png"),
    )
    plt.close(fig_hist)

    # Unconditional with historical overlay and trajectory inset
    fig_hist_inlet = uncond_figs.plot_unconditional_states_with_history_inlet(
        inv_samples=uncond_samples,
        season_axis=season_axis,
        states=['NC', 'CA', 'NY', 'TX', 'FL', 'MT'],
        n_inset_trajs=3,
        plot_median=False,
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_uncond_states_with_history_inlet.png"),
    )
    plt.close(fig_hist_inlet)

    fig_hist_alt = uncond_figs.plot_unconditional_states_with_history_alt(
        inv_samples=uncond_samples,
        season_axis=season_axis,
        states=['NC', 'CA', 'NY', 'TX', 'FL', 'MT'],
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_uncond_states_with_history_alt.png"),
    )
    plt.close(fig_hist_alt)

    print("Unconditional figures complete.")


def generate_peak_analysis_figures(season_axis, uncond_samples):
    """Generate peak analysis figures.

    Args:
        season_axis: SeasonAxis object
        uncond_samples: Unconditional samples array
    """
    print("Generating peak analysis figures...")

    # Peak distributions comparison
    fig_peaks = peak_analysis.plot_peak_distributions_comparison(
        inv_samples=uncond_samples,
        season_axis=season_axis,
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_peak_distributions.png"),
        prominence_threshold=50.0,
    )
    plt.close(fig_peaks)

    # Peak distributions by location
    fig_peaks_loc = peak_analysis.plot_peak_distributions_by_location(
        inv_samples=uncond_samples,
        season_axis=season_axis,
        states=['CA', 'FL'],
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_peak_distributions_by_location.png"),
        prominence_threshold=50.0,
    )
    plt.close(fig_peaks_loc)

    # Peak timing
    fig_peak_timing = peak_analysis.plot_peak_distributions_by_metric(
        inv_samples=uncond_samples,
        season_axis=season_axis,
        states=['CA', 'FL'],
        metric='timing',
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_peak_timing.png"),
        prominence_threshold=50.0,
    )
    plt.close(fig_peak_timing)

    # Peak size
    fig_peak_size = peak_analysis.plot_peak_distributions_by_metric(
        inv_samples=uncond_samples,
        season_axis=season_axis,
        states=['CA', 'FL'],
        metric='size',
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_peak_size.png"),
        prominence_threshold=50.0,
    )
    plt.close(fig_peak_size)

    print("Peak analysis figures complete.")


def generate_csv_forecast_figures(season_axis):
    """Generate CSV forecast figures.

    Args:
        season_axis: SeasonAxis object
    """
    print("Generating CSV forecast figures...")

    # Multi-season CSV fans
    fig = csv_forecasts.plot_csv_quantile_fans_multiseasons(
        seasons=["2023-2024", "2024-2025"],
        base_dir=INPAINTING_BASE,
        model_id=BEST_MODEL_ID,
        config=BEST_CONFIG,
        season_axis=season_axis,
        states=['US','NC','CA','NY','TX','FL'],
        pick_every=2,
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_forecast_csv_fans_states_2023_2025.png"),
        plot_median=PLOT_MEDIAN,
    )
    if fig is not None:
        plt.close(fig)

    # Per-season CSV fan plots
    _csv_states = ['US','NC','CA','NY','TX','FL']
    for _season in ["2023-2024", "2024-2025"]:
        _fig = csv_forecasts.plot_csv_quantile_fans_for_season(
            season=_season,
            base_dir=INPAINTING_BASE,
            model_id=BEST_MODEL_ID,
            config=BEST_CONFIG,
            season_axis=season_axis,
            pick_every=2,
            state=_csv_states,
            save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_forecast_csv_fans_states_{_season.replace('-', '_')}.png"),
            plot_median=PLOT_MEDIAN,
        )
        if _fig is not None:
            plt.close(_fig)

    print("CSV forecast figures complete.")


def generate_npy_forecast_figures(season_axis):
    """Generate NPY forecast figures.

    Args:
        season_axis: SeasonAxis object
    """
    print("Generating NPY forecast figures...")

    # Multi-date two seasons
    fig = npy_forecasts.plot_npy_multi_date_two_seasons(
        base_dir=INPAINTING_BASE,
        model_id=BEST_MODEL_ID,
        config=BEST_CONFIG,
        season_axis=season_axis,
        seasons=("2023-2024", "2024-2025"),
        per_season_pick=4,
        state=['US','NC','CA','NY','TX'],
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_forecast_npy_two_panel_states.png"),
        plot_median=False,
    )
    plt.close(fig)

    # Two-panel national
    fig = npy_forecasts.plot_npy_two_panel_national(
        base_dir=INPAINTING_BASE,
        model_id=BEST_MODEL_ID,
        config=BEST_CONFIG,
        season_axis=season_axis,
        seasons=("2023-2024", "2024-2025"),
        per_season_pick=4,
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_forecast_npy_two_panel_US.png"),
        plot_median=False,
    )
    plt.close(fig)

    # State-specific (California)
    fig = npy_forecasts.plot_npy_multi_date_two_seasons(
        base_dir=INPAINTING_BASE,
        model_id=BEST_MODEL_ID,
        config=BEST_CONFIG,
        season_axis=season_axis,
        seasons=("2023-2024", "2024-2025"),
        per_season_pick=4,
        state='CA',
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_forecast_npy_two_panel_state_CA.png"),
        plot_median=False,
    )
    plt.close(fig)

    print("NPY forecast figures complete.")


def generate_correlation_figures(season_axis, uncond_samples):
    """Generate correlation analysis figures.

    Args:
        season_axis: SeasonAxis object
        uncond_samples: Unconditional samples array
    """
    print("Generating correlation analysis figures...")

    # Weekly incidence correlation
    fig_correlation = correlation_analysis.plot_weekly_incidence_correlation(
        inv_samples=uncond_samples,
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_weekly_incidence_correlation.png"),
        n_permutations=100,
    )
    plt.close(fig_correlation)

    print("Correlation analysis figures complete.")


def generate_mask_experiment_figures():
    """Generate mask experiment figures."""
    print("Generating mask experiment figures...")

    MASK_RESULTS_DIR = "from_longleaf/mask_experiments_868_celebahq_noTTJ5/"
    MASK_FORECAST_DATE = "2025-05-14"

    if os.path.isdir(MASK_RESULTS_DIR):
        outputs = mask_experiments.plot_mask_experiments(
            mask_dir=MASK_RESULTS_DIR,
            forecast_date=MASK_FORECAST_DATE,
            states=('NC', 'CA'),
            plot_median=False,
        )
        print(f"Mask figures: {outputs}")
    else:
        print(f"Mask results directory not found: {MASK_RESULTS_DIR}")

    print("Mask experiment figures complete.")


def main():
    """Main function to orchestrate all figure generation."""
    print("="*60)
    print("Influpaint Paper Figures Generation")
    print("="*60)

    # Setup
    print("\nSetting up...")
    season_axis = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
    uncond_samples = load_unconditional_samples(UNCOND_SAMPLES_PATH)
    print(f"Loaded unconditional samples: {uncond_samples.shape}")

    # Filter unconditional samples based on peak threshold (per location)
    print("\nFiltering unconditional samples to remove blank/low trajectories...")
    peak_thresholds = compute_historical_peak_threshold(
        season_axis=season_axis,
        seasons=[2022, 2023, 2024],
        threshold_fraction=0.1,
    )
    uncond_samples_filtered = filter_trajectories_by_peak(
        uncond_samples,
        season_axis,
        peak_thresholds,
        max_low_locations=MAX_LOW_LOCATIONS
    )

    # Generate all figure types using filtered samples
    try:
        generate_unconditional_figures(season_axis, uncond_samples_filtered)
    except Exception as e:
        print(f"Error generating unconditional figures: {e}")
        raise ValueError("")

    try:
        generate_peak_analysis_figures(season_axis, uncond_samples_filtered)
    except Exception as e:
        print(f"Error generating peak analysis figures: {e}")
        raise ValueError("")
    try:
        generate_correlation_figures(season_axis, uncond_samples_filtered)
    except Exception as e:
        print(f"Error generating correlation figures: {e}")
        raise ValueError("")
    try:
        generate_csv_forecast_figures(season_axis)
    except Exception as e:
        print(f"Error generating CSV forecast figures: {e}")
        raise ValueError("")
    try:
        generate_npy_forecast_figures(season_axis)
    except Exception as e:
        print(f"Error generating NPY forecast figures: {e}")
        raise ValueError("")
    try:
        generate_mask_experiment_figures()
    except Exception as e:
        print(f"Error generating mask experiment figures: {e}")
        raise ValueError("")
    print("\n" + "="*60)
    print("Figure generation complete!")
    print(f"All figures saved to: {FIG_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
