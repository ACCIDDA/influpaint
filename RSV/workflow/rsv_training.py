#!/usr/bin/env python3
"""
rsv_training.py

Train one RSV model. Supports two modes:

  * scratch   -> initialise a fresh DDPM + U-Net and train on RSV data
                 ("RSVpaint baseline").
  * finetune  -> load a pre-trained InfluPaint checkpoint and keep training
                 on RSV data ("FluPaint -> RSV").

The model architecture is picked by --scn-id, which indexes into the same
scenario catalogue used for the flu models (see
influpaint/batch/scenarios.py). The training data is loaded directly from a
NetCDF file produced by build_training_sets.py.

Typical use is through the SLURM script RSV/workflow/train_rsv.run, but it
can also be called standalone:

    # from scratch
    python RSV/workflow/rsv_training.py \\
        --dataset-nc training_datasets/RSV_100A_2026-04-24.nc \\
        --scn-id 868 --experiment-name rsv-2026-04-24 \\
        --run-tag 100A_scratch --epochs 3000

    # fine-tune from an existing InfluPaint checkpoint
    python RSV/workflow/rsv_training.py \\
        --dataset-nc training_datasets/RSV_100A_2026-04-24.nc \\
        --scn-id 868 --experiment-name rsv-2026-04-24 \\
        --run-tag 100A_finetune --epochs 500 \\
        --finetune-from /path/to/flupaint.pth
"""

import argparse
import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch.utils.data import DataLoader

from influpaint.utils import SeasonAxis
from influpaint.batch.scenarios import get_training_scenario
from influpaint.batch.config import (
    unet_library,
    ddpm_library,
    transform_library,
    get_git_revision_short_hash,
    create_folders,
)
from influpaint.datasets import loaders as training_datasets


def build_ddpm(scenario_spec, image_size, channels, batch_size, epochs, device):
    """
    Build a fresh DDPM + U-Net purely from a scenario spec.

    We do NOT use influpaint.batch.scenarios.create_scenario_objects here on
    purpose: that helper also tries to load the flu training datasets from
    disk, which is wasteful (and will fail if those files are not present on
    the cluster). We only need the model part for RSV.
    """
    unet = unet_library(image_size, channels)[scenario_spec.unet_name]
    ddpm = ddpm_library(image_size, channels, epochs, device, batch_size, unet=unet)[
        scenario_spec.ddpm_name
    ]
    return ddpm


def build_rsv_dataset(dataset_nc, channels, scenario_spec):
    """
    Load an RSV .nc file and attach the scenario's transform + enrichment.

    The transform turns counts into the [-1, 1]-ish range the network expects.
    The enrichment is the on-the-fly data augmentation (e.g. Poisson noise,
    random time shifts) applied during training.
    """
    dataset = training_datasets.FluDataset.from_xarray(dataset_nc, channels=channels)
    scaling_per_channel = np.array(dataset.max_per_feature)
    transforms_spec, enrich_spec = transform_library(
        scaling_per_channel,
        data_mean=dataset.flu_dyn.mean(),
        data_std=dataset.flu_dyn.std(),
    )
    transform = transforms_spec[scenario_spec.transform_name]
    enrich = enrich_spec[scenario_spec.enrich_name]
    dataset.add_transform(
        transform=transform["reg"],
        transform_inv=transform["inv"],
        transform_enrich=enrich,
        bypass_test=False,
    )
    return dataset, scaling_per_channel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-nc", required=True,
                   help="Path to the RSV .nc training file.")
    p.add_argument("--scn-id", type=int, required=True,
                   help="Training scenario ID (selects UNet/DDPM/transform).")
    p.add_argument("--experiment-name", required=True,
                   help="MLflow experiment name.")
    p.add_argument("--run-tag", required=True,
                   help="Short tag for this run; appears in filenames and MLflow.")
    p.add_argument("--outdir", default="/users/c/h/chadi/influpaint_res/",
                   help="Where to write checkpoints and plots on Longleaf.")
    p.add_argument("--finetune-from", default=None,
                   help="Path to a pre-trained checkpoint. If given, load it "
                        "before training (fine-tune mode).")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3000)
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
    scenario_spec = get_training_scenario(args.scn_id)

    print(f"=== RSV training ===")
    print(f"Scenario {args.scn_id}: {scenario_spec.scenario_string}")
    print(f"Dataset  : {args.dataset_nc}")
    print(f"Device   : {device}")

    # ---- Output folder ----------------------------------------------------
    git_hash = get_git_revision_short_hash()
    today = datetime.date.today()
    model_folder = Path(args.outdir) / f"{git_hash}_{args.experiment_name}_{today}"
    create_folders(model_folder)

    # ---- Model ------------------------------------------------------------
    ddpm = build_ddpm(
        scenario_spec,
        image_size=args.image_size,
        channels=args.channels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
    )

    # ---- Dataset ----------------------------------------------------------
    dataset, scaling_per_channel = build_rsv_dataset(
        args.dataset_nc, args.channels, scenario_spec
    )
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Scaling per channel: {scaling_per_channel}")

    # ---- Optionally load pre-trained weights ------------------------------
    if args.finetune_from:
        print(f"Loading weights for fine-tuning: {args.finetune_from}")
        ddpm.load_model_checkpoint(args.finetune_from)
        mode = "finetune"
    else:
        mode = "scratch"

    # ---- Train with MLflow logging ----------------------------------------
    mlflow.set_experiment(args.experiment_name)
    run_name = f"{mode}_{args.run_tag}_scn{args.scn_id}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "scenario_id": args.scn_id,
                "scenario_string": scenario_spec.scenario_string,
                "dataset_nc": args.dataset_nc,
                "mode": mode,
                "finetune_from": args.finetune_from or "",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "channels": args.channels,
                "dataset_size": len(dataset),
                "scaling_per_channel": scaling_per_channel.tolist(),
                "device": device,
                "output_folder": str(model_folder),
            }
        )

        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        print(f"Batches per epoch: {len(dataloader)}")

        losses = ddpm.train(dataloader, mlflow_logging=True)

        # ---- Save checkpoint ---------------------------------------------
        ckpt_path = model_folder / f"{run_name}_ep{args.epochs}.pth"
        ddpm.write_train_checkpoint(save_path=str(ckpt_path))
        mlflow.log_param("checkpoint_path", str(ckpt_path))
        mlflow.log_artifact(str(ckpt_path), "checkpoints")

        # ---- Sanity samples ----------------------------------------------
        print("Generating sanity samples...")
        samples = ddpm.sample()
        samples_path = model_folder / f"samples_{run_name}.npy"
        np.save(samples_path, samples[-1])
        mlflow.log_artifact(str(samples_path), "samples")

        mlflow.log_metrics(
            {
                "final_loss": float(losses[-1]) if losses else 0.0,
                "avg_loss_last_100": (
                    float(np.mean(losses[-100:]))
                    if len(losses) >= 100
                    else float(np.mean(losses)) if losses else 0.0
                ),
                "training_completed": 1,
            }
        )

        print(f"Done. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
