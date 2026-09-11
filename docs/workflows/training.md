# 3. Train candidate models

You will use **the four training datasets, `influpaint.batch.training`, and `main_training/train.run`** to learn **diffusion-model checkpoints** and record their training losses in MLflow.

!!! tip "Skip this step with the reproduction archive"

    The selected checkpoint is `influpaint-paper/influpaint_paper_reproduction_data/i868::m_U500cRx1224::ds_30S70M::tr_Sqrt::ri_No::3000.pth`. Training summaries and curves are in `influpaint-paper/influpaint_paper_reproduction_data/analysis/mlflow_losses.csv` and `analysis/mlflow_loss_timeseries.csv` (both under the archive). This skips training for archived analyses. The forecast batch runner requires the original MLflow run/store as well; a standalone checkpoint is not a replacement for that metadata. To skip cluster setup too, continue with the archived forecasts in step 5.


## Learn to remove noise

Training transforms an incidence image, adds noise at a selected diffusion time, and trains the U-Net to predict the noise. Repeating this over many season images learns a distribution of plausible seasons. After training, reverse diffusion starts with Gaussian noise and produces an unconditional season.

The paper's selected formulation is `i868::m_U500cRx1224::ds_30S70M::tr_Sqrt::ri_No`: 500 diffusion steps, cosine noise schedule, a ResNet U-Net with channel multipliers `(1, 2, 2, 4)`, the `30S70M` dataset, square-root preprocessing, and no additional training enrichment.

## Submit the paper candidates

Run from the repository root on Longleaf, after restoring the July 17 data:

```bash
sbatch main_training/train.run
```

The launcher requests one GPU, 32 GB RAM, and 12 hours per task. It invokes:

```bash
python -u -m influpaint.batch.training -s "$SLURM_ARRAY_TASK_ID" -e paper-2025-07-22_training
```

Adapt the launcher’s environment path and Slurm resources to your installation. Configuration lives in `influpaint/batch/config.py`; `influpaint/batch/scenarios.py` maps numeric IDs to formulations. Preserve option ordering when reproducing scenario IDs.

The saved request contains 17 scenarios: `804,36,292,548,1060,868,932,996,772,788,820,800,808,812,805,806,807`. These are baseline i804 plus individual parameter variations. The exported summary contains 15 finished runs, each at 3,000 epochs and batch size 512; i1060 and i812 are absent from that summary.

## Keep weights and provenance together

Training saves checkpoints, samples, parameters, and loss metrics. The selected training run ID is `fabf8cf1fbf2464380b1747e4974a1d7`. The next step uses those run IDs to load the correct checkpoint for each candidate.

[Previous: 2. Create training data](build-training-datasets.md) · [Next: 4. Prepare forecast jobs](forecast-jobs.md)
