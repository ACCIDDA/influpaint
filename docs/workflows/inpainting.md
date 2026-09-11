# 5. Generate forecasts with inpainting

You will use **a trained diffusion model, observed hospitalizations, and CoPaint** to generate **512 trajectories conditioned on observed history**, then export marginal forecast quantiles.

!!! tip "Skip this step with the reproduction archive"

    Use `influpaint-paper/influpaint_paper_reproduction_data/forecasts/retrospective/`. Each of its 29 date directories contains `fluforecasts_ti.npy` and a FluSight-format CSV for the selected i868 model. For unconditional generated seasons, use `influpaint-paper/influpaint_paper_reproduction_data/forecasts/unconditional/inverse_transformed_samples_i868::m_U500cRx1224::ds_30S70M::tr_Sqrt::ri_No.npy`. These are sufficient for the selected-model figure analyses, not the full candidate-model recalibration.


## Turn observations into a mask

The batch runner places observed history on the same week/location grid as the training images. A mask identifies entries supplied as evidence. Future weeks remain hidden. CoPaint's `O_DDIMSampler`, imported from `CoPaint4influpaint/`, guides denoising so sampled seasons agree with the observed portion while proposing plausible values for the hidden portion.

For the selected paper formulation, `celebahq_noTTJ5` disables time travel and uses two latent optimization iterations. The model weights stay fixed during these forecast jobs; conditioning changes the generated trajectories.

## Submit and save forecasts

After generating new jobs in step 4:

```bash
sbatch main_training/generated/inpaint_array_paper-2025-07-22.run
```

To rerun the saved historical manifest with its original MLflow store:

```bash
sbatch main_training/inpaint_array_paper-2025-07-22.run
```

Each task invokes `python -m influpaint.batch.inpainting` with the scenario, run ID, forecast date, and conditioning configuration. It saves transformed samples (`fluforecasts.npy`), inverse-transformed samples (`fluforecasts_ti.npy`), and exported quantiles.

## Read the outputs correctly

Inverse-transformed arrays have shape `(512, 1, 64, 64)` and retain complete sampled seasons on the hospitalization-count scale. The first 51 location columns represent states and DC; remaining columns are padding. CSVs contain 23 marginal quantiles at horizons `0,1,2,3`. Use `target_end_date` to identify the target week: horizon 0 in the archive has `target_end_date == reference_date`.

The retrospective jobs use available truth with a historical observation mask (`nogit=True`). They do not reconstruct the exact surveillance-data vintage available at each historical issue date. Actual operational submissions are archived separately under `forecasts/operational/` and span historical model versions.

[Previous: 4. Prepare forecast jobs](forecast-jobs.md) · [Next: 6. Score forecasts](evaluation.md)
