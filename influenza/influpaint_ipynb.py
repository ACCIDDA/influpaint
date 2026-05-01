# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: diffusion_torch
#     language: python
#     name: python3
# ---

# %% [markdown]
# # InfluPaint Interactive Forecasting
#
# This notebook provides an interactive interface for generating flu forecasts using trained diffusion models.
# It uses the new modular structure from `influpaint/batch/` while maintaining the exploratory nature of notebooks.
#
# **Workflow:**
# 1. Select a training scenario and load the trained model
# 2. Configure inpainting parameters (date, config, batch size)
# 3. Prepare ground truth data with masking
# 4. Run CoPaint inpainting to generate forecasts
# 5. Visualize and export results
#
# **Key differences from the old notebook:**
# - Uses scenario-based model/dataset selection
# - Integrates with MLflow for experiment tracking
# - Cleaner separation between model loading and inference
# - Supports both MLflow and filesystem model loading

# %% [markdown]
# ## Setup: Imports and Configuration

# %%
import sys
sys.path.append('..')  # Add parent directory to path for imports
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch
import numpy as np
import pandas as pd
import datetime
import sys
from pathlib import Path

# InfluPaint modular imports
from influpaint.utils import SeasonAxis, plotting as idplots
from influpaint.batch.scenarios import get_training_scenario, create_scenario_objects
from influpaint.batch.config import copaint_config_library, create_folders, get_git_revision_short_hash
from influpaint.utils import ground_truth

# CoPaint imports
sys.path.append('CoPaint4influpaint')
from guided_diffusion import O_DDIMSampler

# Configure plotting
sns.set_style("whitegrid")
# %matplotlib inline

# %% [markdown]
# ## Configuration Parameters
#
# Set the key parameters for this forecasting run:
# - **scenario_id**: Which training scenario to use (see scenarios.py)
# - **forecast_date**: The date to forecast from (mask date)
# - **config_name**: CoPaint configuration (e.g., 'celebahq_noTT', 'celebahq_try1')
# - **model_source**: Either auto-find from experiment, MLflow run_id, or filesystem path
# - **device**: 'cuda' or 'cpu'

# %%
# === USER CONFIGURATION ===
scenario_id = 868  # Choose your training scenario
forecast_date = "2026-03-07"  # YYYY-MM-DD format
config_name = "celebahq_noTTJ5"  # CoPaint config name
batch_size = 512
image_size = 64
channels = 1

# Model source: Choose ONE of the following options
# Option 1: Auto-find model from MLflow experiment (recommended - same as mask_experiments)
experiment_name = "paper-2025-07-22_training"  # MLflow experiment name
run_id = None
model_path = None

# Option 2: Specify MLflow run_id directly (uncomment to use)
# experiment_name = None
# run_id = "abc123def456"  # Your MLflow run ID
# model_path = None

# Option 3: Load from filesystem (uncomment to use)
# experiment_name = None
# run_id = None
# model_path = "/path/to/model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    from influpaint.utils.helpers import cuda_mem_info
    print(cuda_mem_info())
    torch.cuda.empty_cache()
    print(cuda_mem_info())

# %% [markdown]
# ## Load Scenario and Create Model/Dataset
#
# The scenario system provides a unified way to specify:
# - Model architecture (DDPM + UNet)
# - Dataset source and composition
# - Transformations and data augmentation
#
# All of these are bundled into a single scenario_id.

# %%
# Get scenario specification
scenario_spec = get_training_scenario(scenario_id)
print(f"Scenario {scenario_id}: {scenario_spec.scenario_string}")
print(f"  DDPM: {scenario_spec.ddpm_name}")
print(f"  UNet: {scenario_spec.unet_name}")
print(f"  Dataset: {scenario_spec.dataset_name}")
print(f"  Transform: {scenario_spec.transform_name}")
print(f"  Enrich: {scenario_spec.enrich_name}")

# Create season setup for Flusight geography
season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)

# Create model, dataset, and transforms using scenario helper
print("\nCreating model, dataset, and transforms...")
ddpm, dataset, transform, enrich, scaling_per_channel, data_mean, data_sd = create_scenario_objects(
    scenario_spec,
    season_setup,
    image_size,
    channels,
    batch_size,
    epochs=1,  # Not used for inference
    device=device
)

print(f"Dataset size: {len(dataset)} samples")
print(f"Scaling per channel: {scaling_per_channel}")
print(f"Data mean: {data_mean:.2f}, std: {data_sd:.2f}")
print(f"Timesteps: {ddpm.timesteps}")

# %% [markdown]
# ## Load Trained Model
#
# Load the trained model weights using one of three methods:
# - **Auto-find from experiment** (recommended): Searches MLflow experiment for matching scenario_id
# - **MLflow run_id**: Specify run_id to load from MLflow tracking server directly
# - **Filesystem**: Specify model_path to load a .pth checkpoint directly

# %%
# Import model loading utilities (same as mask_experiments)
from influpaint.batch.inpainting import load_model
from influpaint.batch.generate_inpainting_jobs import get_finished_models

# Determine run_id if using experiment_name
if experiment_name:
    print(f"Finding run ID for scenario {scenario_id} in experiment '{experiment_name}'...")
    finished_models = get_finished_models(experiment_name)

    # Find the specific model for this scenario
    target_model = None
    for model in finished_models:
        if model['scenario_id'] == scenario_id:
            target_model = model
            break

    if target_model is None:
        raise ValueError(f"No finished model found for scenario {scenario_id} in experiment '{experiment_name}'")

    run_id = target_model['run_id']
    print(f"✓ Found run ID: {run_id}")
    print(f"  Model scenario: {target_model['scenario_string']}")
    model_source = f"mlflow_experiment:{experiment_name}/scenario:{scenario_id}"

elif run_id:
    print(f"Using specified run_id: {run_id}")
    model_source = f"mlflow_run:{run_id}"

elif model_path:
    print(f"Using model from filesystem: {model_path}")
    model_source = f"filesystem:{model_path}"

else:
    raise ValueError("Must provide either experiment_name, run_id, or model_path")

# Load the model using the unified load_model function
print("Loading model checkpoint...")
load_model(ddpm, run_id=run_id, model_path=model_path)
print(f"✓ Model loaded from: {model_source}")

# %% [markdown]
# ## Prepare Ground Truth for Inpainting
#
# Create the ground truth data and mask for the forecast date:
# - Automatically determines flu season year from forecast_date
# - Loads surveillance data up to the mask_date
# - Creates a binary mask (1 = known, 0 = to be inferred)

# %%
# Parse forecast date
forecast_dt = pd.to_datetime(forecast_date)
print(f"Forecast date: {forecast_dt.date()}")

# Determine flu season year dynamically
season_first_year = str(season_setup.get_fluseason_year(forecast_dt))
print(f"Detected flu season: {season_first_year}-{int(season_first_year)+1}")

# Create ground truth object
gt1 = ground_truth.GroundTruth.for_flusight(
    season_first_year=season_first_year,
    data_date=datetime.datetime.today(),
    mask_date=forecast_dt,
    channels=channels,
    image_size=image_size,
    nogit=True  # Skip git operations for interactive use
)
fig, ax = plt.subplots(figsize=(8, 4))
gt1.plot_mask()
plt.show()

print(f"Ground truth shape: {gt1.gt_xarr.shape}")
print(f"Inpainting from week: {gt1.inpaintfrom_idx}")
print(f"Known weeks: 1-{gt1.inpaintfrom_idx-1}, Forecast weeks: {gt1.inpaintfrom_idx}-52")

# %% [markdown]
# ### Visualize Ground Truth and Mask

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot mask
ax = axes[0]
gt1.plot_mask()
ax.set_title(f"Mask (known weeks: 1-{gt1.inpaintfrom_idx-1})")

# Plot ground truth
ax = axes[1]
gt1.plot()
ax.set_title("Ground Truth Data")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Configure CoPaint Sampler
#
# CoPaint provides several configuration presets that control the inpainting process:
# - Time travel (jump diffusion)
# - Optimization steps
# - Learning rates
#
# Common configs:
# - `celebahq_noTT`: No time travel, optimized for stable results
# - `celebahq_try1`: With time travel, more exploratory

# %%
# Get available configs
available_configs = copaint_config_library(ddpm.timesteps)
print(f"Available CoPaint configs: {list(available_configs.keys())}")

# Select config
if config_name not in available_configs:
    raise ValueError(f"Config '{config_name}' not found. Available: {list(available_configs.keys())}")

conf = available_configs[config_name]
print(f"\nUsing CoPaint config: {config_name}")

# Create sampler
sampler = O_DDIMSampler(
    use_timesteps=np.arange(ddpm.timesteps),
    conf=conf,
    betas=ddpm.betas,
    model_mean_type=None,
    model_var_type=None,
    loss_type=None
)

print("✓ Sampler created")

# %% [markdown]
# ## Run Inpainting
#
# Generate forecast samples by running the CoPaint inpainting algorithm.
# This will take several minutes depending on:
# - Number of timesteps (typically 200-500)
# - Batch size
# - GPU/CPU performance

# %%
# Prepare ground truth tensors
gt_transformed = dataset.apply_transform(np.nan_to_num(gt1.gt_xarr.data, nan=0.0))
gt_keep_mask = torch.from_numpy(gt1.gt_keep_mask).type(torch.FloatTensor).to(device)
gt_tensor = torch.from_numpy(gt_transformed).type(torch.FloatTensor).to(device)

print(f"Running CoPaint inpainting with {batch_size} samples...")
print(f"This may take several minutes...")

# Run sampling
result = sampler.p_sample_loop(
    model_fn=ddpm.model,
    shape=(batch_size, channels, image_size, image_size),
    conf=conf,
    model_kwargs={
        "gt": gt_tensor.repeat(batch_size, 1, 1, 1),
        "gt_keep_mask": gt_keep_mask.repeat(batch_size, 1, 1, 1),
        "mymodel": True,
    }
)

# Extract results
fluforecasts = np.array(result['sample'].cpu())
fluforecasts_ti = dataset.apply_transform_inv(fluforecasts)
forecasts_national = fluforecasts_ti.sum(axis=-1)

print(f"✓ Generated {len(fluforecasts)} forecast samples")
print(f"Forecast array shape: {fluforecasts_ti.shape}")

# %% [markdown]
# ## Visualize Results: National Forecast

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=100)

for iax in range(2):
    ax = axes[iax]

    # Plot quantile bands
    from influpaint.utils.helpers import flusight_quantile_pairs
    for iqt in range(11):
        ax.fill_between(
            np.arange(64),
            np.quantile(forecasts_national, flusight_quantile_pairs[iqt, 0], axis=0)[0],
            np.quantile(forecasts_national, flusight_quantile_pairs[iqt, 1], axis=0)[0],
            alpha=0.1, color='darkred'
        )

    # Plot median
    ax.plot(np.arange(64),
            np.quantile(forecasts_national, 0.5, axis=0)[0],
            color='r', lw=2, label='Median forecast')

    # Plot ground truth
    ax.plot(gt1.gt_xarr.data[0, :gt1.inpaintfrom_idx].sum(axis=1),
            color='k', marker='.', ls='', markersize=8, label='Observed data')

    # Mark forecast start
    ax.axvline(gt1.inpaintfrom_idx - 1, c='k', ls='--', lw=1.5, alpha=0.5)

    if iax == 0:
        # Full season view
        ax.set_xlim(0, 52)
        ax.set_ylim(bottom=0, auto=True)
        ax.set_title("National Forecast - Full Season")
    else:
        # Zoomed view around forecast start
        ax.set_xlim(gt1.inpaintfrom_idx - 4, gt1.inpaintfrom_idx + 4)
        ax.set_ylim(bottom=0, auto=True)
        ax.set_title("National Forecast - Forecast Window")

    ax.grid(visible=True, alpha=0.3)
    ax.set_xlabel("Season Week")
    ax.set_ylabel("Hospitalizations")
    ax.legend(loc='upper left')
    sns.despine(ax=ax)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize Results: State-Level Forecasts
#
# Plot forecasts for individual states to inspect spatial patterns.

# %%
def plot_state_forecasts(fluforecasts_ti, gt1, season_setup, states_to_plot=None, n_samples=50):
    """Plot forecasts for selected states"""

    if states_to_plot is None:
        # Default: plot first 6 states
        states_to_plot = list(range(6))

    n_states = len(states_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    for idx, place_idx in enumerate(states_to_plot):
        if idx >= 6:
            break

        ax = axes.flat[idx]
        location_name = season_setup.get_location_name(season_setup.locations[place_idx])

        # Plot sample trajectories
        for i in range(min(n_samples, batch_size)):
            ax.plot(fluforecasts_ti[i, 0, :, place_idx],
                   lw=0.3, alpha=0.1, color='lightcoral')

        # Plot median forecast
        median_forecast = np.median(fluforecasts_ti[:, 0, :, place_idx], axis=0)
        ax.plot(median_forecast, color='red', lw=2, label='Median')

        # Plot ground truth
        ax.plot(gt1.gt_xarr.data[0, :gt1.inpaintfrom_idx, place_idx],
               color='k', marker='.', ls='', markersize=6, label='Observed')

        # Mark forecast start
        ax.axvline(gt1.inpaintfrom_idx - 1, c='k', ls='--', lw=1, alpha=0.5)

        ax.set_xlim(0, 52)
        ax.set_ylim(bottom=0, auto=True)
        ax.set_title(location_name)
        ax.grid(visible=True, alpha=0.3)

        if idx >= 3:
            ax.set_xlabel('Season Week')
        if idx % 3 == 0:
            ax.set_ylabel('Hospitalizations')

        sns.despine(ax=ax)

    axes.flat[0].legend(loc='upper left')
    fig.tight_layout()
    plt.show()

# Plot forecasts for selected states
plot_state_forecasts(fluforecasts_ti, gt1, season_setup)

# %% [markdown]
# ## Summary Statistics

# %%
print("=== Forecast Summary Statistics ===")
print(f"Number of samples: {len(forecasts_national)}")
print(f"\nNational peak hospitalizations:")
print(f"  Median: {np.median(forecasts_national.max(axis=1)):.0f}")
print(f"  Mean: {np.mean(forecasts_national.max(axis=1)):.0f}")
print(f"  10th percentile: {np.percentile(forecasts_national.max(axis=1), 10):.0f}")
print(f"  90th percentile: {np.percentile(forecasts_national.max(axis=1), 90):.0f}")

print(f"\nForecast horizon: {image_size - gt1.inpaintfrom_idx + 1} weeks")
print(f"Known weeks: 1-{gt1.inpaintfrom_idx - 1}")
print(f"Forecast weeks: {gt1.inpaintfrom_idx}-{image_size}")

# %% [markdown]
# ## Export Results
#
# Save forecasts in FluSight-compatible format and create visualizations.
# This will create:
# - CSV files with quantile forecasts for each location
# - Summary plots
# - Optionally save to MLflow
#
# **IMPORTANT**: Before exporting, we recreate the ground truth object to pull the latest surveillance data.

# %%
# Determine next Saturday for submission
today = datetime.datetime.today()
days_until_saturday = (5 - today.weekday()) % 7
next_saturday = today + datetime.timedelta(days=days_until_saturday)
submission_date = next_saturday.date()

print(f"Next Saturday (submission date): {submission_date}")

# %% [markdown]
# ### Update Ground Truth with Latest Data
#
# This is critical! We need to recreate gt1 with today's date to fetch the latest surveillance data
# from the FluSight hub. This ensures our forecast CSV files have the most recent observed data.

# %%
print("Updating ground truth with latest surveillance data...")
print(f"Original gt1 created with mask_date: {forecast_date}")

# Recreate ground truth with current date to get latest data
from importlib import reload
ground_truth = reload(ground_truth)

# Determine season for submission
submission_dt = pd.to_datetime(submission_date)
season_first_year_submission = str(season_setup.get_fluseason_year(submission_dt))

gt1 = ground_truth.GroundTruth.for_flusight(
    season_first_year=season_first_year_submission,
    data_date=datetime.datetime.today(),
    mask_date=datetime.datetime.today(),  # Use today to get all available data
    channels=channels,
    image_size=image_size,
    nogit=True
)

print(f"✓ Ground truth updated for season {season_first_year_submission}-{int(season_first_year_submission)+1}")
print(f"  Data available through week: {gt1.inpaintfrom_idx - 1}")
print(f"  This will be included in the forecast CSV files")

# %%
# Create output directory
output_dir = Path("operational_output") / str(submission_date)
output_dir.mkdir(parents=True, exist_ok=True)

# Export using ground truth helper
team_abbrv = "UNC_IDD-InfluPaint"
gt1.export_forecasts_2023(
    fluforecasts_ti=fluforecasts_ti,
    forecasts_national=forecasts_national,
    directory=str(output_dir),
    prefix=f"{team_abbrv}",
    forecast_date=submission_date,
    save_plot=True,
    nochecks=True
)

print(f"✓ Forecasts exported to: {output_dir}")
print(f"  - CSV files: {len(list(output_dir.glob('*.csv')))} files")
print(f"  - Plots: {len(list(output_dir.glob('*.png')))} + {len(list(output_dir.glob('*.pdf')))} files")

# %%
len(np.zeros(len(gt1.season_setup.locations)+1))

# %%
gt1.season_setup.locations

# %% [markdown]
# ## Optional: Save Raw Arrays
#
# Save the raw forecast arrays for further analysis.

# %%
save_raw_arrays = True  # Set to True to save

if save_raw_arrays:
    np.save(output_dir / f"{submission_date}_fluforecasts_raw.npy", fluforecasts)
    np.save(output_dir / f"{submission_date}_fluforecasts_transformed_inv.npy", fluforecasts_ti)
    np.save(output_dir / f"{submission_date}_forecasts_national.npy", forecasts_national)
    print(f"✓ Saved raw arrays to {output_dir}")

# %% [markdown]
# ## Optional: Log to MLflow
#
# Track this forecasting run in MLflow for reproducibility.

# %%
log_to_mlflow = False  # Set to True to enable MLflow logging

if log_to_mlflow:
    import mlflow

    experiment_name = "influpaint_interactive_forecasts"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"forecast_{forecast_date}_{config_name}"):
        # Log parameters
        mlflow.log_params({
            "scenario_id": scenario_id,
            "scenario_string": scenario_spec.scenario_string,
            "forecast_date": forecast_date,
            "config_name": config_name,
            "batch_size": batch_size,
            "model_source": model_source,
            "timesteps": ddpm.timesteps,
            "device": device,
        })

        # Log metrics
        mlflow.log_metrics({
            "num_samples": len(forecasts_national),
            "forecast_horizon_weeks": image_size - gt1.inpaintfrom_idx + 1,
            "national_peak_median": float(np.median(forecasts_national.max(axis=1))),
            "national_peak_mean": float(np.mean(forecasts_national.max(axis=1))),
        })

        # Log artifacts
        mlflow.log_artifacts(str(output_dir), "forecasts")

        print(f"✓ Logged to MLflow experiment: {experiment_name}")

# %%
submission_date

# %% [markdown]
# ## Session Complete
#
# Forecasts have been generated and exported. You can now:
# - Review the plots in the output directory
# - Submit the CSV files to FluSight
# - Run additional analyses on the raw forecast arrays
# - Try different configs or forecast dates by modifying the configuration cell

# %%
print("=" * 60)
print("SESSION SUMMARY")
print("=" * 60)
print(f"Scenario: {scenario_spec.scenario_string}")
print(f"Forecast date: {forecast_date}")
print(f"Config: {config_name}")
print(f"Samples generated: {len(forecasts_national)}")
print(f"Output directory: {output_dir}")
print(f"Model source: {model_source}")
print("=" * 60)
