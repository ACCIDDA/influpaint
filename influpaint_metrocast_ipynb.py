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
#     display_name: diffusion_torch6
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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch
from torch import nn
import numpy as np
import pandas as pd
import datetime
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam

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

# %% [markdown]
#

# %%
# === USER CONFIGURATION ===
scenario_id = 868  # Choose your training scenario
forecast_date = "2026-01-17"  # YYYY-MM-DD format
config_name = "celebahq_noTTJ5"  # CoPaint config name
batch_size = 256
image_size = 128
channels = 1
# Fine-tune controls: enable if you want to adapt weights before inpainting.
train_finetune = True
# finetune_mode options:
# - "adapters": init_conv + final_conv + GroupNorm affine only
# - "adapters_time": adapters + time_mlp
# - "adapters_time_ups2": adapters + time_mlp + last 2 up blocks
# - "full": full model fine-tune (not recommended with small data)
finetune_mode = "adapters_time_ups2"
finetune_epochs = 40
finetune_lr = 1e-5
finetune_output_dir = Path("output/metrocast_finetune")
metrocast_nc_path = Path("training_datasets/MetrocastTS_100M_2026-01-15.nc")
do_uncond_preview = True
uncond_batch_size = 8

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

# %%
# Parse forecast date
season_setup = SeasonAxis.for_metrocast()

forecast_dt = pd.to_datetime(forecast_date)
print(f"Forecast date: {forecast_dt.date()}")

# Determine flu season year dynamically
season_first_year = str(season_setup.get_fluseason_year(forecast_dt))
print(f"Detected flu season: {season_first_year}-{int(season_first_year)+1}")

# Create ground truth object
gt1 = ground_truth.GroundTruth.from_metrocast(
    season_first_year=season_first_year,
    data_date=datetime.datetime.today(),
    mask_date=forecast_dt,
    channels=channels,
    image_size=image_size,
    nogit=True  # Skip git operations for interactive use
)

gt1.plot_mask()
plt.show()
print(f"Ground truth shape: {gt1.gt_xarr.shape}")
print(f"Inpainting from week: {gt1.inpaintfrom_idx}")
print(f"Known weeks: 1-{gt1.inpaintfrom_idx-1}, Forecast weeks: {gt1.inpaintfrom_idx}-52")

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
# ## Load MetroCast Dataset (Explicit)
#
# Override the scenario dataset with the MetroCast .nc file for clarity.

# %%
from influpaint.datasets import loaders as training_datasets
from influpaint.batch.config import transform_library

dataset = training_datasets.FluDataset.from_xarray(
    str(metrocast_nc_path),
    channels=channels,
)
scaling_per_channel = np.array(dataset.max_per_feature)
data_mean = dataset.flu_dyn.mean()
data_sd = dataset.flu_dyn.std()
transforms_spec, transform_enrich = transform_library(
    scaling_per_channel,
    data_mean=data_mean,
    data_std=data_sd,
)
transform = transforms_spec[scenario_spec.transform_name]
enrich = transform_enrich[scenario_spec.enrich_name]
dataset.add_transform(
    transform=transform["reg"],
    transform_inv=transform["inv"],
    transform_enrich=enrich,
    bypass_test=False,
)
print(f"MetroCast dataset size: {len(dataset)} samples")
print(f"MetroCast max per channel: {scaling_per_channel}")

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
# ## Optional: Unconditional Preview Against MetroCast Distribution
#
# This uses the same model weights and transforms to generate unconditional samples.
# Keep the batch size small; this can be memory-heavy.

# %%
if do_uncond_preview:
    from influpaint.batch.training import plot_sample

    prev_batch_size = ddpm.batch_size
    ddpm.batch_size = uncond_batch_size
    print(f"Generating {uncond_batch_size} unconditional samples...")
    samples = ddpm.sample()
    ddpm.batch_size = prev_batch_size

    # Overlay: generated vs historical on the same axes for scale comparison.
    n_show = min(uncond_batch_size, 8)
    fig, axes = plt.subplots(1, n_show, figsize=(2.8 * n_show, 3.5), dpi=100, sharey=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes[:n_show]):
        gen_img = dataset.apply_transform_inv(samples[-1][i])
        hist_img = dataset.get_sample_raw(i)
        idplots.show_tensor_image(gen_img, ax=ax, place=None, multi=True)
        idplots.show_tensor_image(hist_img, ax=ax, place=None, multi=True)
        ax.set_title(f"Gen vs Hist {i}")
        ax.grid(visible=True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Fine-Tune and Save a New Checkpoint
#
# Workflow:
# 1) Load the same model checkpoint as above
# 2) Modify trainable parameters (adapters/norms/time-MLP)
# 3) Fine-tune on the MetroCast dataset
# 4) Save the new weights and reload them for inpainting

# %%
def configure_finetune(model, train_init_final=True, train_norm=True, train_time_mlp=False, unfreeze_ups=0, train_all=False):
    """Freeze all parameters, then selectively enable a small adaptation subset."""
    unet = model.module if isinstance(model, nn.DataParallel) else model

    for param in unet.parameters():
        param.requires_grad = train_all

    if train_all:
        return

    if train_init_final:
        for param in unet.init_conv.parameters():
            param.requires_grad = True
        for param in unet.final_conv.parameters():
            param.requires_grad = True

    if train_norm:
        for module in unet.modules():
            if isinstance(module, nn.GroupNorm):
                if module.weight is not None:
                    module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.requires_grad = True

    if train_time_mlp and unet.time_mlp is not None:
        for param in unet.time_mlp.parameters():
            param.requires_grad = True

    if unfreeze_ups > 0:
        for block in unet.ups[-unfreeze_ups:]:
            for param in block.parameters():
                param.requires_grad = True

# Stage the minimal parameter set for adaptation.
train_time_mlp = finetune_mode in {"adapters_time", "adapters_time_ups2"}
unfreeze_ups = 2 if finetune_mode == "adapters_time_ups2" else 0
train_all = finetune_mode == "full"

configure_finetune(
    ddpm.model,
    train_time_mlp=train_time_mlp,
    unfreeze_ups=unfreeze_ups,
    train_all=train_all,
)
finetune_snapshot = {
    name: param.detach().clone()
    for name, param in ddpm.model.named_parameters()
    if param.requires_grad
}
print(f"Trainable params: {len(finetune_snapshot)} tensors")

if train_finetune:
    finetune_output_dir.mkdir(parents=True, exist_ok=True)


    ddpm.optimizer = Adam(
        filter(lambda p: p.requires_grad, ddpm.model.parameters()),
        lr=finetune_lr,
    )
    ddpm.epochs = finetune_epochs

    finetune_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Fine-tuning for {finetune_epochs} epochs with lr={finetune_lr}")
    ddpm.train(finetune_loader, mlflow_logging=False)

    finetune_ckpt = finetune_output_dir / f"{scenario_spec.scenario_string}::finetune_{finetune_epochs}.pth"
    ddpm.write_train_checkpoint(save_path=str(finetune_ckpt))
    print(f"✓ Fine-tuned checkpoint saved: {finetune_ckpt}")


# %%
# Reload to ensure downstream inpainting uses the fine-tuned weights.
finetune_ckpt = "output/metrocast_finetune/i868::m_U500cRx1224::ds_30S70M::tr_Sqrt::ri_No::finetune_40.pth"
ckpt = torch.load(finetune_ckpt, map_location="cpu")
ddpm.model.load_state_dict(ckpt["model_state_dict"])
ddpm.model.eval()
ddpm.model.to(device)
print("✓ Reloaded fine-tuned checkpoint for inpainting")

# %% [markdown]
# ## Unconditional Preview (Fine-Tuned) vs History
#
# Compare fine-tuned unconditional samples against historical MetroCast frames.

# %%
if do_uncond_preview:
    prev_batch_size = ddpm.batch_size
    ddpm.batch_size = uncond_batch_size
    print(f"Generating {uncond_batch_size} unconditional samples (fine-tuned)...")
    ft_samples = ddpm.sample()
    ddpm.batch_size = prev_batch_size

    n_show = min(uncond_batch_size, 8)
    fig, axes = plt.subplots(1, n_show, figsize=(2.8 * n_show, 3.5), dpi=100, sharey=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes[:n_show]):
        gen_img = dataset.apply_transform_inv(ft_samples[-1][i])
        hist_img = dataset.get_sample_raw(i)
        idplots.show_tensor_image(gen_img, ax=ax, place=None, multi=True)
        idplots.show_tensor_image(hist_img, ax=ax, place=None, multi=True)
        ax.set_title(f"FT gen vs Hist {i}")
        ax.grid(visible=True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Sanity Check: Did Fine-Tuning Change the Weights?
# This compares the trainable tensors before/after fine-tuning.

# %%
if train_finetune:
    deltas = []
    unexpected = []
    for name, param in ddpm.model.named_parameters():
        if name in finetune_snapshot:
            delta = (param.detach() - finetune_snapshot[name]).pow(2).sum().sqrt().item()
            deltas.append((name, delta))
        elif param.requires_grad:
            unexpected.append(name)
    deltas.sort(key=lambda x: x[1], reverse=True)
    print("Top weight changes (L2 norm):")
    for name, delta in deltas[:10]:
        print(f"{name}: {delta:.6f}")
    if unexpected:
        print("Unexpected trainable parameters:")
        for name in unexpected:
            print(f"  {name}")
    else:
        print("No unexpected trainable parameters.")
else:
    print("Fine-tune disabled; no weight-change check.")

# %% [markdown]
# ## Prepare Ground Truth for Inpainting
#
# Create the ground truth data and mask for the forecast date:
# - Automatically determines flu season year from forecast_date
# - Loads surveillance data up to the mask_date
# - Creates a binary mask (1 = known, 0 = to be inferred)

# %%
# Parse forecast date
season_setup = SeasonAxis.for_metrocast()

forecast_dt = pd.to_datetime(forecast_date)
print(f"Forecast date: {forecast_dt.date()}")

# Determine flu season year dynamically
season_first_year = str(season_setup.get_fluseason_year(forecast_dt))
print(f"Detected flu season: {season_first_year}-{int(season_first_year)+1}")

# Create ground truth object
gt1 = ground_truth.GroundTruth.from_metrocast(
    season_first_year=season_first_year,
    data_date=datetime.datetime.today(),
    mask_date=forecast_dt,
    channels=channels,
    image_size=image_size,
    nogit=True  # Skip git operations for interactive use
)

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
inpaint_batch_size = 256

if device == "cuda":
    from influpaint.utils.helpers import cuda_mem_info
    print(cuda_mem_info())
    torch.cuda.empty_cache()
    print(cuda_mem_info())
# Run sampling
result = sampler.p_sample_loop(
    model_fn=ddpm.model,
    shape=(inpaint_batch_size, channels, image_size, image_size),
    conf=conf,
    clip_denoised=False,
    model_kwargs={
        "gt": gt_tensor.repeat(inpaint_batch_size, 1, 1, 1),
        "gt_keep_mask": gt_keep_mask.repeat(inpaint_batch_size, 1, 1, 1),
        "mymodel": True,
    }
)
if device == "cuda":
    from influpaint.utils.helpers import cuda_mem_info
    print(cuda_mem_info())
    torch.cuda.empty_cache()
    print(cuda_mem_info())

# Extract results
fluforecasts = np.array(result['sample'].cpu())
fluforecasts_ti = dataset.apply_transform_inv(fluforecasts)
forecasts_national = fluforecasts_ti.sum(axis=-1)

print(f"✓ Generated {len(fluforecasts)} forecast samples")
print(f"Forecast array shape: {fluforecasts_ti.shape}")

# %% [markdown]
# ## Sanity Checks: Forecast Scale and Variability
#
# Quick checks to ensure forecasts are not collapsed to the neutral value.

# %%
print("Transformed min/max/mean:", fluforecasts.min(), fluforecasts.max(), fluforecasts.mean())
print("Inverse min/max/mean:", fluforecasts_ti.min(), fluforecasts_ti.max(), fluforecasts_ti.mean())
gt_vals = gt1.gt_xarr.data
print("GT min/max/mean:", np.nanmin(gt_vals), np.nanmax(gt_vals), np.nanmean(gt_vals))
print("Sample std median:", np.median(np.std(fluforecasts_ti, axis=0)))
print("gt_keep_mask sum:", gt_keep_mask.sum().item())
print("gt_keep_mask unique:", np.unique(gt_keep_mask.cpu().numpy()))
plt.hist(fluforecasts.flatten())
plt.show()



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

gt1 = ground_truth.GroundTruth.from_metrocast(
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
output_dir = Path("operational_output_metrocast") / str(submission_date)
output_dir.mkdir(parents=True, exist_ok=True)

# Export using ground truth helper
team_abbrv = "UNC_IDD-InfluPaint"
gt1.export_forecasts_2023(
    fluforecasts_ti=fluforecasts_ti,
    directory=str(output_dir),
    prefix=f"{team_abbrv}",
    forecast_date=submission_date,
    save_plot=True,
    nochecks=True,
    mode="metrocast"
)

print(f"✓ Forecasts exported to: {output_dir}")
print(f"  - CSV files: {len(list(output_dir.glob('*.csv')))} files")

# Optional: Plot forecasts using the same path as the CLI script
forecasts_national = fluforecasts_ti.sum(axis=-1)
gt1.plot_forecasts(
    fluforecasts_ti=fluforecasts_ti,
    forecasts_national=forecasts_national,
    directory=str(output_dir),
    prefix=f"{team_abbrv}_metrocast",
    forecast_date=submission_date,
    mode="metrocast",
)
print(f"  - Plots: {len(list(output_dir.glob('*.pdf')))} files")

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
