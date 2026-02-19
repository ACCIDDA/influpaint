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
from influpaint.batch.config import (
    copaint_config_library,
    create_folders,
    get_git_revision_short_hash,
)
from influpaint.utils import ground_truth


# CoPaint imports
sys.path.append("CoPaint4influpaint")
from guided_diffusion import O_DDIMSampler

# Configure plotting
sns.set_style("whitegrid")


# === USER CONFIGURATION ===
scenario_id = 868  # Choose your training scenario
forecast_date = "2026-01-24"  # YYYY-MM-DD format
config_name = "celebahq_noTTJ5"  # CoPaint config name
batch_size = 512
image_size = 64
channels = 1

# Model source: Choose ONE of the following options
# Option 1: Auto-find model from MLflow experiment (recommended - same as mask_experiments)
# experiment_name = "paper-2025-07-22_training"  # MLflow experiment name
# run_id = None
# model_path = None

# Option 2: Specify MLflow run_id directly (uncomment to use)
# experiment_name = None
# run_id = "abc123def456"  # Your MLflow run ID
# model_path = None

# Option 3: Load from filesystem (uncomment to use)
experiment_name = None
run_id = None
model_path = "rsv.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    from influpaint.utils.helpers import cuda_mem_info

    print(cuda_mem_info())
    torch.cuda.empty_cache()
    print(cuda_mem_info())


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
ddpm, dataset, transform, enrich, scaling_per_channel, data_mean, data_sd = (
    create_scenario_objects(
        scenario_spec,
        season_setup,
        image_size,
        channels,
        batch_size,
        epochs=1,  # Not used for inference
        device=device,
    )
)

print(f"Dataset size: {len(dataset)} samples")
print(f"Scaling per channel: {scaling_per_channel}")
print(f"Data mean: {data_mean:.2f}, std: {data_sd:.2f}")
print(f"Timesteps: {ddpm.timesteps}")


from influpaint.batch.inpainting import load_model
from influpaint.batch.generate_inpainting_jobs import get_finished_models

# Determine run_id if using experiment_name
if experiment_name:
    print(
        f"Finding run ID for scenario {scenario_id} in experiment '{experiment_name}'..."
    )
    finished_models = get_finished_models(experiment_name)

    # Find the specific model for this scenario
    target_model = None
    for model in finished_models:
        if model["scenario_id"] == scenario_id:
            target_model = model
            break

    if target_model is None:
        raise ValueError(
            f"No finished model found for scenario {scenario_id} in experiment '{experiment_name}'"
        )

    run_id = target_model["run_id"]
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


forecast_dt = pd.to_datetime(forecast_date)
print(f"Forecast date: {forecast_dt.date()}")

# Determine flu season year dynamically
season_first_year = str(season_setup.get_fluseason_year(forecast_dt))
print(f"Detected flu season: {season_first_year}-{int(season_first_year) + 1}")

# Create ground truth object
gt1 = ground_truth.GroundTruth.for_flusight(
    season_first_year=season_first_year,
    data_date=datetime.datetime.today(),
    mask_date=forecast_dt,
    channels=channels,
    image_size=image_size,
    nogit=True,  # Skip git operations for interactive use
)
fig, ax = plt.subplots(figsize=(8, 4))
gt1.plot_mask()
plt.show()

print(f"Ground truth shape: {gt1.gt_xarr.shape}")
print(f"Inpainting from week: {gt1.inpaintfrom_idx}")
print(
    f"Known weeks: 1-{gt1.inpaintfrom_idx - 1}, Forecast weeks: {gt1.inpaintfrom_idx}-52"
)


# === Configure CoPaint Sampler ===
available_configs = copaint_config_library(ddpm.timesteps)
print(f"Available CoPaint configs: {list(available_configs.keys())}")

if config_name not in available_configs:
    raise ValueError(
        f"Config '{config_name}' not found. Available: {list(available_configs.keys())}"
    )

conf = available_configs[config_name]
print(f"\nUsing CoPaint config: {config_name}")

sampler = O_DDIMSampler(
    use_timesteps=np.arange(ddpm.timesteps),
    conf=conf,
    betas=ddpm.betas,
    model_mean_type=None,
    model_var_type=None,
    loss_type=None,
)
print("Sampler created")


# === Run Inpainting ===
gt_transformed = dataset.apply_transform(np.nan_to_num(gt1.gt_xarr.data, nan=0.0))
gt_keep_mask = torch.from_numpy(gt1.gt_keep_mask).type(torch.FloatTensor).to(device)
gt_tensor = torch.from_numpy(gt_transformed).type(torch.FloatTensor).to(device)

print(f"Running CoPaint inpainting with {batch_size} samples...")

result = sampler.p_sample_loop(
    model_fn=ddpm.model,
    shape=(batch_size, channels, image_size, image_size),
    conf=conf,
    model_kwargs={
        "gt": gt_tensor.repeat(batch_size, 1, 1, 1),
        "gt_keep_mask": gt_keep_mask.repeat(batch_size, 1, 1, 1),
        "mymodel": True,
    },
)

fluforecasts = np.array(result["sample"].cpu())
fluforecasts_ti = dataset.apply_transform_inv(fluforecasts)
forecasts_national = fluforecasts_ti.sum(axis=-1)

print(f"Generated {len(fluforecasts)} forecast samples")
print(f"Forecast array shape: {fluforecasts_ti.shape}")


# === Visualize National Forecast ===
from influpaint.utils.helpers import flusight_quantile_pairs

fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=100)

for iax in range(2):
    ax = axes[iax]

    for iqt in range(11):
        ax.fill_between(
            np.arange(64),
            np.quantile(forecasts_national, flusight_quantile_pairs[iqt, 0], axis=0)[0],
            np.quantile(forecasts_national, flusight_quantile_pairs[iqt, 1], axis=0)[0],
            alpha=0.1,
            color="darkred",
        )

    ax.plot(
        np.arange(64),
        np.quantile(forecasts_national, 0.5, axis=0)[0],
        color="r",
        lw=2,
        label="Median forecast",
    )
    ax.plot(
        gt1.gt_xarr.data[0, : gt1.inpaintfrom_idx].sum(axis=1),
        color="k",
        marker=".",
        ls="",
        markersize=8,
        label="Observed data",
    )
    ax.axvline(gt1.inpaintfrom_idx - 1, c="k", ls="--", lw=1.5, alpha=0.5)

    if iax == 0:
        ax.set_xlim(0, 52)
        ax.set_ylim(bottom=0, auto=True)
        ax.set_title("National Forecast - Full Season")
    else:
        ax.set_xlim(gt1.inpaintfrom_idx - 4, gt1.inpaintfrom_idx + 4)
        ax.set_ylim(bottom=0, auto=True)
        ax.set_title("National Forecast - Forecast Window")

    ax.grid(visible=True, alpha=0.3)
    ax.set_xlabel("Season Week")
    ax.set_ylabel("Hospitalizations")
    ax.legend(loc="upper left")
    sns.despine(ax=ax)

fig.tight_layout()
plt.show()


# === Visualize State Forecasts ===
def plot_state_forecasts(
    fluforecasts_ti, gt1, season_setup, states_to_plot=None, n_samples=50
):
    if states_to_plot is None:
        states_to_plot = list(range(6))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    for idx, place_idx in enumerate(states_to_plot):
        if idx >= 6:
            break

        ax = axes.flat[idx]
        location_name = season_setup.get_location_name(
            season_setup.locations[place_idx]
        )

        for i in range(min(n_samples, batch_size)):
            ax.plot(
                fluforecasts_ti[i, 0, :, place_idx],
                lw=0.3,
                alpha=0.1,
                color="lightcoral",
            )

        median_forecast = np.median(fluforecasts_ti[:, 0, :, place_idx], axis=0)
        ax.plot(median_forecast, color="red", lw=2, label="Median")
        ax.plot(
            gt1.gt_xarr.data[0, : gt1.inpaintfrom_idx, place_idx],
            color="k",
            marker=".",
            ls="",
            markersize=6,
            label="Observed",
        )
        ax.axvline(gt1.inpaintfrom_idx - 1, c="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlim(0, 52)
        ax.set_ylim(bottom=0, auto=True)
        ax.set_title(location_name)
        ax.grid(visible=True, alpha=0.3)

        if idx >= 3:
            ax.set_xlabel("Season Week")
        if idx % 3 == 0:
            ax.set_ylabel("Hospitalizations")
        sns.despine(ax=ax)

    axes.flat[0].legend(loc="upper left")
    fig.tight_layout()
    plt.show()


plot_state_forecasts(fluforecasts_ti, gt1, season_setup)


# === Summary Statistics ===
print("=== Forecast Summary Statistics ===")
print(f"Number of samples: {len(forecasts_national)}")
print(f"\nNational peak hospitalizations:")
print(f"  Median: {np.median(forecasts_national.max(axis=1)):.0f}")
print(f"  Mean: {np.mean(forecasts_national.max(axis=1)):.0f}")
print(f"  10th percentile: {np.percentile(forecasts_national.max(axis=1), 10):.0f}")
print(f"  90th percentile: {np.percentile(forecasts_national.max(axis=1), 90):.0f}")
print(f"\nForecast horizon: {image_size - gt1.inpaintfrom_idx + 1} weeks")


# === Export Results ===
today = datetime.datetime.today()
days_until_saturday = (5 - today.weekday()) % 7
next_saturday = today + datetime.timedelta(days=days_until_saturday)
submission_date = next_saturday.date()
print(f"Next Saturday (submission date): {submission_date}")

# Update ground truth with latest surveillance data
print("Updating ground truth with latest surveillance data...")
from importlib import reload

ground_truth = reload(ground_truth)

submission_dt = pd.to_datetime(submission_date)
season_first_year_submission = str(season_setup.get_fluseason_year(submission_dt))

gt1 = ground_truth.GroundTruth.for_flusight(
    season_first_year=season_first_year_submission,
    data_date=datetime.datetime.today(),
    mask_date=datetime.datetime.today(),
    channels=channels,
    image_size=image_size,
    nogit=True,
)
print(
    f"Ground truth updated for season {season_first_year_submission}-{int(season_first_year_submission) + 1}"
)

output_dir = Path("operational_output") / str(submission_date)
output_dir.mkdir(parents=True, exist_ok=True)

team_abbrv = "UNC_IDD-InfluPaint"
gt1.export_forecasts_2023(
    fluforecasts_ti=fluforecasts_ti,
    forecasts_national=forecasts_national,
    directory=str(output_dir),
    prefix=f"{team_abbrv}",
    forecast_date=submission_date,
    save_plot=True,
    nochecks=True,
)
print(f"Forecasts exported to: {output_dir}")


# === Save Raw Arrays ===
np.save(output_dir / f"{submission_date}_fluforecasts_raw.npy", fluforecasts)
np.save(
    output_dir / f"{submission_date}_fluforecasts_transformed_inv.npy", fluforecasts_ti
)
np.save(output_dir / f"{submission_date}_forecasts_national.npy", forecasts_national)
print(f"Saved raw arrays to {output_dir}")


# === Session Summary ===
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
