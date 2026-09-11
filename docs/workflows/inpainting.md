# 5. Generate forecasts with inpainting

The objective is to turn the trained models from step 3 into forecasts for the jobs defined in step 4. A trained diffusion model can generate a complete influenza season; **inpainting supplies the observed part of a particular season and asks the model to generate the missing part**. Repeating this produces 512 possible trajectories, which you can inspect as complete seasons or summarize as forecast quantiles.

The entry point is `influpaint.batch.inpainting`. Each job loads one checkpoint, prepares observations for one reference date, runs the selected CoPaint configuration, and saves the resulting ensemble.

!!! tip "Use saved results from the Zenodo archive"

    `forecasts/retrospective/` contains the selected i868 ensemble and CSV for each of 29 reference dates; `forecasts/unconditional/` contains generated seasons without conditioning. These support the selected-model analyses. The full candidate comparison requires the other candidates' forecasts as well. To submit the preserved historical manifest using its MLflow runs, use `sbatch main_training/inpaint_array_paper-2025-07-22.run`.

## 1. Turn observed history into a conditioning image

`GroundTruth.for_flusight` places observed hospitalizations on the same week/location grid as the training data. The observation mask marks entries supplied as evidence with 1 and hidden entries with 0. For forecasting, the observed history is available and future weeks are hidden.

The runner transforms hospitalizations using the candidate's training-data transformation, then passes both the transformed values and mask to CoPaint's `O_DDIMSampler` from `CoPaint4influpaint/`. CoPaint guides the generated season toward agreement with the observed entries while sampling plausible values for the hidden entries. Network weights remain fixed; the changing observations and conditioning settings determine the forecast.

These retrospective jobs read the locally available truth with a historical mask date (`nogit=True`). They evaluate forecasts conditioned on that history, using the local data revisions. To make a live forecast with refreshed surveillance, follow [step 9](operational-forecasts.md).

## 2. Submit the jobs you prepared

After reviewing the manifest and adapting the launcher in step 4, submit it from the research repository root:

```bash
export OCP_OUTDIR="$PWD/forecast_output/"
sbatch main_training/generated/inpaint_array_my-flu-experiment.run
```

Each task invokes `python -m influpaint.batch.inpainting` with the scenario, training run ID, reference date, and conditioning configuration from its manifest row. The model's training NetCDF must remain available because the runner reconstructs its transformations from that dataset.

To inspect one job before launching the whole experiment, copy a run ID from your manifest and run it directly on your GPU machine:

```bash
python -m influpaint.batch.inpainting \
  -s 868 -r YOUR_TRAINING_RUN_ID \
  -e my-flu-experiment_inpainting \
  --forecast_date 2023-10-14 --config_name celebahq_noTTJ5 \
  -d ./forecast_output/
```

Replace `YOUR_TRAINING_RUN_ID` with the actual run ID for that scenario. To use a standalone checkpoint, replace `-r YOUR_TRAINING_RUN_ID` with `-m /path/to/checkpoint.pth`; the scenario and matching training dataset are still required. Keep the trailing slash on the output root.

## 3. Follow a forecast through the sampler

The runner loads the weights, builds the date-specific observations and mask, and generates a batch of samples with the configured diffusion schedule. For the paper's selected `celebahq_noTTJ5` setting, time travel is disabled and the sampler uses two latent optimization iterations.

After sampling, it inverse-transforms the output to hospitalization counts, aggregates national trajectories, and saves arrays, forecast quantiles, and plots. Results live beneath the output root in an experiment directory containing the code revision and run date, with one subdirectory per scenario/configuration/reference-date combination. Parameters and artifacts are also logged to the inpainting MLflow experiment.

## 4. Inspect trajectories and quantiles

| Output | What it lets you examine |
| --- | --- |
| `fluforecasts.npy` | Generated samples on the transformed model scale |
| `fluforecasts_ti.npy` | Complete sampled seasons on the hospitalization-count scale |
| FluSight-format CSV | Marginal forecast quantiles for scoring and short-horizon plots |

The inverse-transformed ensemble has shape `(512, 1, 64, 64)`: sample, incidence channel, week, location. The first 53 week positions and 51 location columns hold the season; the remaining positions are padding. Each ensemble member is a whole trajectory across weeks and locations. The CSV summarizes each target separately, so use the arrays when studying temporal or spatial relationships within generated seasons.

Inspect the plots around the forecast boundary: do the generated trajectories follow the observed history, and how do they spread into the future? The CSV contains 23 marginal quantiles at horizons `0,1,2,3`. Check `reference_date` and `target_end_date` together; in the paper exports, horizon 0 targets the reference date itself.

Once the jobs finish, carry the forecast CSVs and manifest into step 6 for scoring. Keep the full arrays for trajectory analyses and figures.

[Previous: 4. Prepare forecast jobs](forecast-jobs.md) · [Next: 6. Score forecasts](evaluation.md)
